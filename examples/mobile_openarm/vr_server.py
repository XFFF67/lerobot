#!/usr/bin/env python3
"""Local HTTPS/WebXR to Telegrip input bridge for simulation or real-arm control."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import ipaddress
import json
import os
import ssl
import subprocess
import time
from pathlib import Path

from vr_input import ensure_fifo


def make_tls(cert_dir: Path, hostname: str) -> ssl.SSLContext:
    cert_dir.mkdir(parents=True, exist_ok=True)
    cert, key = cert_dir / "cert.pem", cert_dir / "key.pem"
    if not cert.exists() or not key.exists():
        try:
            host = str(ipaddress.ip_address(hostname))
            san = f"IP:{host}"
        except ValueError:
            if not hostname or any(
                c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-" for c in hostname
            ):
                raise ValueError("--hostname must be an IP address or DNS hostname") from None
            san = f"DNS:{hostname}"
        old_umask = os.umask(0o077)
        try:
            subprocess.run(
                [
                    "openssl",
                    "req",
                    "-x509",
                    "-newkey",
                    "rsa:2048",
                    "-nodes",
                    "-keyout",
                    str(key),
                    "-out",
                    str(cert),
                    "-days",
                    "365",
                    "-subj",
                    "/CN=Mobile OpenArm VR",
                    "-addext",
                    f"subjectAltName={san},DNS:localhost,IP:127.0.0.1",
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        finally:
            os.umask(old_umask)
        print(f"[VR] Created local certificate in {cert_dir}", flush=True)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert, key)
    return context


class BrowserBridge:
    def __init__(self, publisher, mobile_fifo=None):
        self.publisher = publisher
        self.mobile_fifo = mobile_fifo
        self.mobile_fifo_fd = None
        self.mobile_fifo_writes = 0
        self.mobile_fifo_error = None
        self.active = None
        self.web_dir = Path(__file__).resolve().parent / "vr_web"
        self.packet_count = 0
        self.last_packet_at = None
        self.diagnostic = {"stage": "waiting_for_browser"}

    def status(self):
        return {
            "browser_connected": self.active is not None,
            "packets": self.packet_count,
            "last_packet_age_s": None
            if self.last_packet_at is None
            else round(time.monotonic() - self.last_packet_at, 2),
            "browser": self.diagnostic,
            "mobile_fifo": {
                "path": self.mobile_fifo,
                "writer_open": self.mobile_fifo_fd is not None,
                "writes": self.mobile_fifo_writes,
                "last_error": self.mobile_fifo_error,
            },
        }

    def publish(self, packet):
        message = "vr_data " + json.dumps(packet, allow_nan=False)
        self.publisher.send_string(message)
        if self.mobile_fifo is not None:
            try:
                if self.mobile_fifo_fd is None:
                    self.mobile_fifo_fd = os.open(self.mobile_fifo, os.O_WRONLY | os.O_NONBLOCK)
                os.write(self.mobile_fifo_fd, message.encode() + b"\n")
                self.mobile_fifo_writes += 1
                self.mobile_fifo_error = None
            except OSError as exc:
                self.mobile_fifo_error = f"{type(exc).__name__}: {exc}"
                if self.mobile_fifo_fd is not None:
                    os.close(self.mobile_fifo_fd)
                    self.mobile_fifo_fd = None

    def close(self):
        if self.mobile_fifo_fd is not None:
            os.close(self.mobile_fifo_fd)
            self.mobile_fifo_fd = None

    def process_request(self, connection, request):
        if request.path == "/api/status":
            response = connection.respond(200, json.dumps(self.status(), ensure_ascii=False))
            del response.headers["Content-Type"]
            response.headers["Content-Type"] = "application/json; charset=utf-8"
            response.headers["Cache-Control"] = "no-store"
            return response
        if request.path == "/ws" and request.headers.get("Upgrade", "").lower() == "websocket":
            return None
        routes = {
            "/": ("index.html", "text/html"),
            "/real": ("index.html", "text/html"),
            "/vr.js": ("vr.js", "text/javascript"),
            "/controller_models.js": ("controller_models.js", "text/javascript"),
            **{
                f"/models/touch-{model}-{side}.glb": (
                    f"models/touch-{model}-{side}.glb",
                    "model/gltf-binary",
                )
                for model in ("v3", "plus", "pro")
                for side in ("left", "right")
            },
        }
        if request.path not in routes:
            return connection.respond(404, "Not found\n")
        filename, content_type = routes[request.path]
        asset = self.web_dir / filename
        response = connection.respond(
            200, "" if filename.endswith(".glb") else asset.read_text(encoding="utf-8")
        )
        if filename.endswith(".glb"):
            response.body = asset.read_bytes()
            del response.headers["Content-Length"]
            response.headers["Content-Length"] = str(len(response.body))
        del response.headers["Content-Type"]
        response.headers["Content-Type"] = (
            content_type if filename.endswith(".glb") else content_type + "; charset=utf-8"
        )
        response.headers["Cache-Control"] = "no-store"
        return response

    async def handle(self, connection):
        if self.active is not None:
            await connection.close(1013, "Another headset is connected")
            return
        self.active = connection
        self.diagnostic = {"stage": "browser_connected_waiting_for_xr"}
        self.packet_count = 0
        self.last_packet_at = None
        print("[VR] Headset connected", flush=True)
        try:
            async for message in connection:
                try:
                    packet = json.loads(message)
                    if not isinstance(packet, dict):
                        continue
                    if packet.get("type") == "diagnostic":
                        # Status only: do not refresh controller data or retain poses.
                        diagnostic = {
                            key: str(packet[key])[:400]
                            for key in (
                                "stage",
                                "error",
                                "visibility",
                                "frames",
                                "left",
                                "right",
                                "ws_close",
                                "xr_end",
                            )
                            if key in packet
                        }
                        if diagnostic.get("stage") != self.diagnostic.get("stage") or "error" in diagnostic:
                            print(f"[VR browser] {json.dumps(diagnostic, ensure_ascii=False)}", flush=True)
                        self.diagnostic = diagnostic
                        continue
                    # Forward only full controller snapshots, never arbitrary commands.
                    if "leftController" not in packet or "rightController" not in packet:
                        continue
                    self.publish({key: packet[key] for key in ("leftController", "rightController")})
                    self.packet_count += 1
                    self.last_packet_at = time.monotonic()
                except (ValueError, TypeError):
                    continue
        finally:
            self.publish({"leftController": None, "rightController": None})
            self.active = None
            print("[VR] Headset disconnected; controls released", flush=True)


async def run(args):
    try:
        import zmq
        from websockets.asyncio.server import serve
    except ImportError as exc:
        raise SystemExit("Install pyzmq and websockets>=14 in hei-rebot-vr; see README_zh.md") from exc
    context = zmq.Context()
    publisher = context.socket(zmq.PUB)
    ensure_fifo(args.mobile_fifo)
    publisher.setsockopt(zmq.SNDHWM, 2)
    bridge = None
    try:
        publisher.bind(args.publish_endpoint)
        tls = make_tls(args.cert_dir, args.hostname)
        bridge = BrowserBridge(publisher, args.mobile_fifo)
        async with serve(
            bridge.handle,
            args.host,
            args.port,
            ssl=tls,
            process_request=bridge.process_request,
            max_size=16384,
            max_queue=2,
            compression=None,
            close_timeout=1,
        ):
            print(f"[VR] On the headset open https://{args.hostname}:{args.port}", flush=True)
            print(f"[VR] Publishing vr_data on {args.publish_endpoint}; Ctrl+C to stop", flush=True)
            await asyncio.Future()
    finally:
        if bridge is not None:
            bridge.close()
        publisher.close(0)
        context.term()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--hostname", required=True, help="Computer LAN IP used by the Quest browser")
    parser.add_argument("--port", type=int, default=8443)
    parser.add_argument("--publish-endpoint", default="tcp://127.0.0.1:5567")
    parser.add_argument("--mobile-fifo", default="/tmp/openarm-vr-mobile.fifo")
    parser.add_argument("--cert-dir", type=Path, default=Path(__file__).resolve().parent / ".vr-certs")
    args = parser.parse_args()
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(run(args))


if __name__ == "__main__":
    main()

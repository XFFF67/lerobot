"""Local HTTPS -> WSS -> ZMQ -> validated controller regression; no headset."""

import asyncio
import json
import ssl
import tempfile
from pathlib import Path

import zmq
from check_vr_sim import controller
from vr_input import ZMQReceiver
from vr_server import BrowserBridge, make_tls
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve


async def check():
    context = zmq.Context()
    publisher = context.socket(zmq.PUB)
    port = publisher.bind_to_random_port("tcp://127.0.0.1")
    receiver = ZMQReceiver(f"tcp://127.0.0.1:{port}", "vr_data", 0.3)
    try:
        with tempfile.TemporaryDirectory(prefix="openarm-vr-test-") as directory:
            server_tls = make_tls(Path(directory), "127.0.0.1")
            client_tls = ssl.create_default_context(cafile=str(Path(directory) / "cert.pem"))
            bridge = BrowserBridge(publisher)
            async with serve(
                bridge.handle,
                "127.0.0.1",
                0,
                ssl=server_tls,
                process_request=bridge.process_request,
                close_timeout=1,
            ) as server:
                https_port = server.sockets[0].getsockname()[1]
                for path, expected in (
                    ("/", b"Mobile OpenArm VR"),
                    ("/real", b'id="realHelp"'),
                    ("/vr.js", b"requestSession"),
                    ("/controller_models.js", b"decodeControllerModel"),
                    ("/models/touch-v3-left.glb", b"glTF"),
                    ("/models/touch-plus-right.glb", b"glTF"),
                    ("/models/touch-pro-left.glb", b"glTF"),
                    ("/api/status", b'"packets": 0'),
                    ("/../vr_server.py", b"404"),
                    ("/models/../vr_server.py", b"404"),
                ):
                    reader, writer = await asyncio.open_connection("127.0.0.1", https_port, ssl=client_tls)
                    writer.write(
                        f"GET {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n".encode()
                    )
                    await writer.drain()
                    response = await reader.read()
                    assert expected in response
                    assert response.count(b"Content-Type:") == 1
                    assert response.count(b"Content-Length:") == 1
                    if path.endswith(".glb"):
                        headers, body = response.split(b"\r\n\r\n", 1)
                        assert b"model/gltf-binary" in headers
                        assert body == (bridge.web_dir / path.lstrip("/")).read_bytes()
                    writer.close()
                    await writer.wait_closed()
                print("PASS HTTPS: trusted test certificate, HTML/JS serving, no directory traversal")
                async with connect(f"wss://127.0.0.1:{https_port}/ws", ssl=client_tls) as socket:
                    packet = {"leftController": controller(grip=True), "rightController": controller()}
                    # PUB/SUB subscription establishment is asynchronous. Bound the wait.
                    for _ in range(100):
                        await socket.send(json.dumps(packet))
                        await asyncio.sleep(0.01)
                        receiver.poll()
                        if receiver.snapshot()["left"].grip:
                            break
                    assert receiver.snapshot()["left"].grip
                    received_at = receiver.received_at.copy()
                    await socket.send(
                        json.dumps({"type": "diagnostic", "stage": "xr_error", "error": "test error"})
                    )
                    await asyncio.sleep(0.05)
                    receiver.poll()
                    assert receiver.received_at == received_at, "Diagnostics must not keep controls alive"
                    assert bridge.status()["browser"]["stage"] == "xr_error"
                    async with connect(f"wss://127.0.0.1:{https_port}/ws", ssl=client_tls) as second:
                        await second.wait_closed()
                        assert second.close_code == 1013
                    assert bridge.active is not None
                    print("PASS WSS -> Telegrip ZMQ -> simulator parser; second headset rejected")
                for _ in range(100):
                    await asyncio.sleep(0.01)
                    receiver.poll()
                    if not receiver.snapshot()["left"].valid:
                        break
                assert not receiver.controllers["left"].valid
                print("PASS disconnect publishes explicit release before timeout")
    finally:
        receiver.close()
        publisher.close(0)
        context.term()


if __name__ == "__main__":
    asyncio.run(asyncio.wait_for(check(), timeout=15))

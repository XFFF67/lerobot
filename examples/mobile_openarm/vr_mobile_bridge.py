#!/usr/bin/env python3
"""Map VR controls to the original chassis frame with appended lift fields."""

import argparse
import math
import signal
import time

from mobile_mapper import MobileMapper
from mobile_serial_port import SerialPort
from mobile_serial_protocol import Commands
from vr_input import FifoReceiver, ZMQReceiver


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", help="CH343 device, preferably /dev/serial/by-id/...")
    parser.add_argument("--execute", action="store_true", help="Send commands to the real C board")
    parser.add_argument("--vr-endpoint", default="tcp://127.0.0.1:5567")
    parser.add_argument("--vr-transport", choices=("fifo", "zmq"), default="fifo")
    parser.add_argument("--vr-fifo", default="/tmp/openarm-vr-mobile.fifo")
    parser.add_argument("--speed", type=float, default=0.05, help="Translation m/s; maximum 0.1")
    parser.add_argument("--spin", type=float, default=0.005, help="Original chassis spin scalar; maximum 0.01")
    parser.add_argument("--lift-speed", type=float, default=0.05, help="Lift motor rad/s; maximum 0.1")
    for name in ("drive", "steer", "spin", "lift"):
        parser.add_argument(
            f"--{name}-sign",
            type=int,
            choices=(-1, 1),
            default=-1 if name == "drive" else 1,
        )
    parser.add_argument("--run-seconds", type=float, default=0)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.execute and not args.port:
        parser.error("--execute requires --port")
    if not math.isfinite(args.run_seconds) or args.run_seconds < 0:
        parser.error("--run-seconds must be finite and >=0")
    try:
        mapper = MobileMapper(
            speed=args.speed,
            spin=args.spin,
            lift_speed=args.lift_speed,
            **{f"{key}_sign": getattr(args, f"{key}_sign") for key in ("drive", "steer", "spin", "lift")},
        )
    except ValueError as exc:
        parser.error(str(exc))
    stopped = False

    def stop(*_):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    receiver = (
        FifoReceiver(args.vr_fifo, 0.25)
        if args.vr_transport == "fifo"
        else ZMQReceiver(args.vr_endpoint, "vr_data", 0.25)
    )
    port = SerialPort(args.port) if args.execute else None
    started = previous = time.monotonic()
    report = started
    try:
        print("EXECUTE" if port else "PREVIEW (no serial output)", flush=True)
        while not stopped:
            now = time.monotonic()
            dt, previous = now - previous, now
            receiver.poll()
            # poll() timestamps newly received frames. Snapshot against a time
            # captured afterwards, otherwise fresh frames appear to come from
            # the future and are rejected by VRInput.snapshot().
            controllers = receiver.snapshot()
            commands = mapper.step(controllers, dt)
            if port:
                # C-board status is not consumed by this bridge yet, but it
                # must be drained to prevent CH343 receive back-pressure.
                port.read()
                payload = commands.encode()
                if port.write(payload) != len(payload):
                    raise RuntimeError("Incomplete serial write")
                port.read()
            if now >= report:
                left, right = controllers["left"], controllers["right"]
                print(
                    f"vr=left(valid={left.valid}, grip={left.grip}, stick={left.stick.tolist()}) "
                    f"right(valid={right.valid}, grip={right.grip}, stick={right.stick.tolist()}) "
                    f"commands={commands}",
                    flush=True,
                )
                report = now + 1
            if args.run_seconds and now - started >= args.run_seconds:
                break
            time.sleep(max(0, 0.02 - (time.monotonic() - now)))
    finally:
        if port:
            try:
                port.reset_output_buffer()
                port.write(Commands().encode())
                time.sleep(0.02)
            finally:
                port.close()
        receiver.close()


if __name__ == "__main__":
    main()

"""One-time Damiao arm zero setting at the manually positioned lowered pose.

Default is preview only. --execute disables J1-J7 and writes their current
physical positions as motor zero. Gripper zero is changed only with
--include-grippers, with the grippers manually closed. No motion is commanded.
"""

import argparse
import logging
import math
import signal
import threading
import time

from lerobot.scripts.openarm_prepare import DamiaoPreparationBackend, MotionLimits

logger = logging.getLogger(__name__)


def wait_stationary(backend, keys, limits, stop, *, verify_zero=False):
    """Require fresh, quiet feedback for 0.3 s, including after torque release."""
    deadline = time.monotonic() + 3.0
    quiet_since = reference = None
    while not stop.is_set():
        tick = time.monotonic()
        positions, velocities = backend.read()
        now = time.monotonic()
        if now - tick > limits.feedback_timeout_s:
            raise RuntimeError("Zero-setting feedback deadline exceeded")
        if any(
            key not in positions
            or key not in velocities
            or not math.isfinite(positions[key])
            or not math.isfinite(velocities[key])
            for key in keys
        ):
            raise RuntimeError("Missing or invalid feedback during zero setting")
        quiet = all(abs(velocities[key]) <= limits.arrival_speed_deg_s for key in keys)
        if reference is not None and any(abs(positions[key] - reference[key]) > 0.3 for key in keys):
            quiet = False
        if verify_zero:
            quiet = quiet and all(abs(positions[key]) <= 0.5 for key in keys)
        if not quiet:
            quiet_since = reference = None
        elif quiet_since is None:
            quiet_since, reference = now, dict(positions)
        elif now - quiet_since >= 0.3:
            return positions
        if now >= deadline:
            raise RuntimeError(
                "Motor zero readback did not settle within 0.5 degrees"
                if verify_zero
                else "Arms must be stationary in the intended zero pose before writing motor zero"
            )
        stop.wait(0.02)
    raise InterruptedError("Zero setting interrupted; some motor zeros may already have been written")


def run(args, stop, backend_factory=DamiaoPreparationBackend):
    limits = MotionLimits()
    backend = backend_factory(args, limits)
    selected = [f"joint_{i}" for i in range(1, 8)]
    if args.include_grippers:
        selected.append("gripper")
    try:
        backend.connect()
        keys = [f"{side}_{motor}.pos" for side in backend.arms for motor in selected]
        for arm in backend.arms.values():
            if not set(selected) <= arm.bus.motors.keys():
                raise RuntimeError("Missing motors in configured zero-setting joint set")
        wait_stationary(backend, keys, limits, stop)
        logger.info("ZERO SETTING: disabling selected motors at the manually positioned zero pose")
        for arm in backend.arms.values():
            if stop.is_set():
                raise InterruptedError("Zero setting cancelled before writing")
            arm.bus.disable_torque(motors=selected)
        wait_stationary(backend, keys, limits, stop)
        for side, arm in backend.arms.items():
            for motor in selected:
                if stop.is_set():
                    raise InterruptedError("Zero setting interrupted; some motors may already be zeroed")
                logger.info("Writing motor zero: %s_%s", side, motor)
                arm.bus.set_zero_position(motors=[motor])
        positions = wait_stationary(backend, keys, limits, stop, verify_zero=True)
        for key in keys:
            logger.info("ZERO VERIFIED: %s = %.4f deg", key, positions[key])
        logger.info("ZERO SET: selected motors remain disabled; normal VR startup never writes zero")
    except Exception:
        logger.exception("Zero setting incomplete; inspect feedback before using the normal startup command")
        raise
    finally:
        # Selected motors were already disabled. Never enable or change torque
        # on unselected grippers, including on a partial write or preflight fault.
        backend.close(disable_torque=False)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute", action="store_true", help="Write the manually positioned pose as motor zero"
    )
    parser.add_argument("--arm", choices=("left", "right", "both"), default="both")
    parser.add_argument("--left-port", default="can2")
    parser.add_argument("--right-port", default="can3")
    parser.add_argument("--id", default="my_bimanual_follower")
    parser.add_argument("--include-grippers", action="store_true", help="Also zero manually closed grippers")
    parser.set_defaults(support_gain_scale=1.0)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.arm == "both" and args.left_port == args.right_port:
        parser.error("Left and right CAN interfaces must differ")
    if not args.execute:
        print("PREVIEW ONLY: no CAN. Place J1-J7 at the intended lowered zero pose, then add --execute.")
        print("This writes motor zeros once; normal VR startup moves to these zeros without rewriting them.")
        print(
            "Grippers: " + ("also zeroed; must be manually closed" if args.include_grippers else "unchanged")
        )
        return
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    run(args, stop)


if __name__ == "__main__":
    main()

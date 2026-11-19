"""Recover calibrated OpenArm joints to fixed zero without VR or rewriting zero.

Use --execute after the previous controller has exited and the return path is
clear. J1-J7 move together to zero; grippers slowly close to calibrated zero.
"""

import argparse
import logging
import math
import signal
import threading

from lerobot.scripts.openarm_prepare import (
    DamiaoPreparationBackend,
    MotionLimits,
    home_goal,
    run_motion,
    support_gain_scale,
)
from lerobot.scripts.openarm_vr import stationary_start

logger = logging.getLogger(__name__)
# Recovery reads may start outside VR operating limits. Commands remain inside
# them and use the same following-error guard as the rest of the motion.
OUTWARD_DRIFT_MARGIN_DEG = 1.0


class RecoveryBackend(DamiaoPreparationBackend):
    """Permit bounded feedback overshoot while keeping every command in limits."""

    def __init__(self, args, limits):
        super().__init__(args, limits)
        self.recovery_ranges = None
        self.last_positions = None

    def begin_recovery(self):
        """Capture the quiet measured pose before changing retained motor targets."""
        if self.last_positions is None:
            raise RuntimeError("Recovery requires fresh stationary startup feedback")
        initial = self.check_recovery_feedback(self.last_positions)
        # Check the actual proposed first target, not a second arbitrary margin
        # around the operating limit. This is also enforced throughout motion.
        for key, target in initial.items():
            measured = self.last_positions[key]
            if abs(target - measured) > self.limits.tracking_error_deg:
                raise RuntimeError(
                    f"Recovery initial following error: {key}; measured={measured:.3f}, "
                    f"first target={target:.3f}, allowed={self.limits.tracking_error_deg:.1f} deg"
                )
        self.recovery_ranges = {
            f"{side}_{motor}.pos": (
                min(low, self.last_positions[f"{side}_{motor}.pos"]),
                max(high, self.last_positions[f"{side}_{motor}.pos"]),
            )
            for side, arm in self.arms.items()
            for motor, (low, high) in arm.config.joint_limits.items()
            if motor != "gripper"
        }
        logger.info(
            "RECOVERY feedback bounds anchored to measured pose; outward drift allowance %.1f deg",
            OUTWARD_DRIFT_MARGIN_DEG,
        )

    def check_recovery_feedback(self, positions):
        ranges = {
            f"{side}_{motor}.pos": bounds
            for side, arm in self.arms.items()
            for motor, bounds in arm.config.joint_limits.items()
        }
        if positions.keys() != ranges.keys() or not all(math.isfinite(v) for v in positions.values()):
            raise RuntimeError("Invalid or incomplete recovery feedback")
        checked = dict(positions)
        for key, (low, high) in ranges.items():
            value = positions[key]
            feedback_low, feedback_high = low, high
            if key.endswith("_gripper.pos"):
                tolerance = 0.5
            elif self.recovery_ranges is None:
                # Accept the real starting position even outside VR operating
                # limits. begin_recovery checks the first command before enable.
                checked[key] = min(high, max(low, value))
                continue
            else:
                feedback_low, feedback_high = self.recovery_ranges[key]
                tolerance = OUTWARD_DRIFT_MARGIN_DEG
            if not feedback_low - tolerance <= value <= feedback_high + tolerance:
                raise RuntimeError(
                    f"Recovery feedback outside allowed bounds: {key}={value:.3f}, "
                    f"feedback bounds=[{feedback_low - tolerance:.3f}, {feedback_high + tolerance:.3f}], "
                    f"command limits=[{low}, {high}]. "
                    "Outward drift or gripper endpoint tolerance exceeded; inspect before retrying"
                )
            checked[key] = min(high, max(low, value))
        if self.recovery_ranges is not None:
            # As a measured joint returns inside its operating range, tighten
            # its recovery envelope. Outward movement can never enlarge it.
            for key, (start_low, start_high) in self.recovery_ranges.items():
                low, high = ranges[key]
                self.recovery_ranges[key] = (
                    max(start_low, min(low, positions[key])),
                    min(start_high, max(high, positions[key])),
                )
        return checked

    def read(self):
        positions, velocities = super().read()
        self.check_recovery_feedback(positions)
        self.last_positions = dict(positions)
        return positions, velocities  # Tracking and settling always use raw feedback.

    def validate_positions(self, positions, *, measurement=False, allow_joint_excursions=False):
        if measurement:
            return self.check_recovery_feedback(positions)
        return super().validate_positions(
            positions,
            allow_joint_excursions=allow_joint_excursions,
        )


def run(args, stop, backend_factory=RecoveryBackend):
    limits = MotionLimits(args.duration_s, args.speed_deg_s, args.accel_deg_s2)
    backend = backend_factory(args, limits)
    completed = False
    try:
        backend.connect()
        initial = stationary_start(backend, limits, stop)
        if initial is None or stop.is_set():
            return
        goal = home_goal(initial)
        for key in goal:
            if key.endswith("_gripper.pos"):
                goal[key] = 0.0
        goal = backend.validate_positions(goal)
        logger.info(
            "RECOVERY: J1-J7 -> calibrated zero; grippers -> closed zero; no zero-writing or J4 preparation"
        )
        logger.info("Initial in-limit targets: %s", initial)
        backend.begin_recovery()
        backend.enable(initial)
        last = run_motion(backend, initial, goal, limits, stop, hold=False, phase="HOMING")
        if last is not None:
            logger.warning("Recovery interrupted: last command retained, torque preserved")
            return
        completed = True
    except Exception:
        logger.exception("Recovery failed: last command and existing torque preserved; no automatic retry")
        raise
    finally:
        backend.close(disable_torque=completed)
    logger.info("HOME COMPLETE: calibrated arm zero reached and settled; torque disabled")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Move real arm joints to calibrated zero")
    parser.add_argument("--arm", choices=("left", "right", "both"), default="both")
    parser.add_argument("--left-port", default="can2")
    parser.add_argument("--right-port", default="can3")
    parser.add_argument("--id", default="my_bimanual_follower")
    parser.add_argument("--duration-s", type=float, default=30.0)
    parser.add_argument("--speed-deg-s", type=float, default=10.0)
    parser.add_argument("--accel-deg-s2", type=float, default=5.0)
    parser.add_argument("--support-gain-scale", type=support_gain_scale, default=4.0)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    MotionLimits(args.duration_s, args.speed_deg_s, args.accel_deg_s2)
    if args.speed_deg_s > 10 or args.accel_deg_s2 > 5:
        parser.error("Recovery speed must be <=10 deg/s and acceleration <=5 deg/s^2")
    if args.arm == "both" and args.left_port == args.right_port:
        parser.error("Left and right CAN interfaces must differ")
    if not args.execute:
        print("PREVIEW ONLY: no CAN. Add --execute to return J1-J7 to the existing calibrated zero.")
        print(
            f"Duration >= {args.duration_s:g}s; speed <= {args.speed_deg_s:g} deg/s; grippers slowly close to 0 deg."
        )
        print("Arm startup feedback is not rejected for crossing VR operating limits.")
        print("First target and motion retain the 10-degree following-error guard.")
        print("Outward drift allowance 1 deg; all motor commands stay inside the original limits.")
        print(
            "No motor zero writing, IK, VR input, or subsequent J4=90 preparation. Disable only after arrival."
        )
        return
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    run(args, stop)


if __name__ == "__main__":
    main()

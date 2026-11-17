#!/usr/bin/env python
# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Prepare calibrated Damiao follower arms: raise J4 to 90 degrees, then hold.

No zero-writing or calibration commands are used. By default this only prints a
trajectory without opening CAN. --read-state reads feedback without enabling or
disabling motors; --execute prepares the real arms; --lower lowers J4 to zero.
"""

import argparse
import logging
import math
import signal
import threading
import time
from collections import deque
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Small measured endpoint offsets are distinct from commanded operating limits.
# The gripper may read slightly positive at its closed mechanical endpoint.
JOINT_FEEDBACK_TOLERANCE_DEG = 0.1
GRIPPER_FEEDBACK_TOLERANCE_DEG = 0.5
# Calibrated elbows can settle slightly below zero in the lowered pose.
# This allowance only applies to measured J4 feedback at the zero lower limit.
ELBOW_ZERO_FEEDBACK_TOLERANCE_DEG = 0.5


def support_gain_scale(value):
    scale = float(value)
    if not math.isfinite(scale) or not 1.0 <= scale <= 4.0:
        raise argparse.ArgumentTypeError("Support gain scale must be finite and between 1 and 4")
    return scale


def gripper_gain_scale(value):
    scale = float(value)
    if not math.isfinite(scale) or not 1.0 <= scale <= 2.0:
        raise argparse.ArgumentTypeError("Gripper gain scale must be finite and between 1 and 2")
    return scale


def support_gains(config, scale, gripper_scale=1.0):
    """Preparation-only overrides for joints with measured holding deflection."""
    scale = support_gain_scale(scale)
    kp, kd = {}, {}
    for index in (0, 3, 6):
        motor = f"joint_{index + 1}"
        base_kp = config.position_kp[index] if isinstance(config.position_kp, list) else config.position_kp
        base_kd = config.position_kd[index] if isinstance(config.position_kd, list) else config.position_kd
        kp[motor] = base_kp * scale
        kd[motor] = base_kd * math.sqrt(scale)
    gripper_scale = gripper_gain_scale(gripper_scale)
    if gripper_scale != 1.0:
        base_kp = config.position_kp[7] if isinstance(config.position_kp, list) else config.position_kp
        base_kd = config.position_kd[7] if isinstance(config.position_kd, list) else config.position_kd
        kp["gripper"] = base_kp * gripper_scale
        kd["gripper"] = base_kd * math.sqrt(gripper_scale)
    return kp, kd


@dataclass(frozen=True)
class MotionLimits:
    duration_s: float = 20.0
    speed_deg_s: float = 10.0
    acceleration_deg_s2: float = 5.0
    feedback_timeout_s: float = 0.2
    tracking_error_deg: float = 10.0
    arrival_tolerance_deg: float = 2.0
    arrival_speed_deg_s: float = 2.0
    settle_s: float = 1.0
    settle_timeout_s: float = 10.0
    fps: float = 50.0
    settle_position_span_deg: float = 0.3
    settle_peak_speed_deg_s: float = 5.0

    def __post_init__(self):
        for name, value in vars(self).items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.arrival_tolerance_deg >= self.tracking_error_deg:
            raise ValueError("Arrival tolerance must be smaller than the tracking error limit")
        if self.settle_peak_speed_deg_s < self.arrival_speed_deg_s:
            raise ValueError("Settling peak speed must not be below arrival speed")


class PreparationTrajectory:
    """Quintic joint trajectory with zero endpoint velocity and acceleration."""

    def __init__(self, start: dict[str, float], goal: dict[str, float], limits: MotionLimits):
        if not start or start.keys() != goal.keys():
            raise ValueError("Start and goal must contain the same nonempty joint set")
        if not all(math.isfinite(v) for v in [*start.values(), *goal.values()]):
            raise ValueError("Non-finite joint position")
        self.start = dict(start)
        self.goal = dict(goal)
        travel = max(abs(goal[key] - value) for key, value in start.items())
        # max ds/du = 1.875; max |d²s/du²| = 10/sqrt(3).
        self.duration_s = max(
            limits.duration_s,
            1.875 * travel / limits.speed_deg_s,
            math.sqrt((10 / math.sqrt(3)) * travel / limits.acceleration_deg_s2),
        )

    def sample(self, elapsed_s: float) -> dict[str, float]:
        if not math.isfinite(elapsed_s):
            raise ValueError("Non-finite trajectory time")
        u = min(1.0, max(0.0, elapsed_s / self.duration_s))
        if u == 0.0:
            return dict(self.start)
        if u == 1.0:
            return dict(self.goal)
        scale = min(1.0, max(0.0, u**3 * (10 + u * (-15 + 6 * u))))
        return {
            key: min(
                max(value, self.goal[key]),
                max(min(value, self.goal[key]), (1 - scale) * value + scale * self.goal[key]),
            )
            for key, value in self.start.items()
        }


def preparation_goal(positions: dict[str, float], elbow_deg: float = 90.0) -> dict[str, float]:
    """Hold all measured joints and grippers; only change the selected elbows."""
    if not math.isfinite(elbow_deg):
        raise ValueError("Non-finite elbow target")
    goal = dict(positions)
    elbows = [key for key in goal if key.endswith("_joint_4.pos")]
    if not elbows:
        raise ValueError("No elbow joints in feedback")
    for key in elbows:
        goal[key] = elbow_deg
    return goal


def home_goal(positions: dict[str, float]) -> dict[str, float]:
    """Return calibrated arm joints to fixed zero; preserve gripper positions."""
    goal = dict(positions)
    for key in positions:
        if any(key.endswith(f"_joint_{i}.pos") for i in range(1, 8)):
            goal[key] = 0.0
    return goal


class PreparationMonitor:
    """Readiness requires measured arrival and settling, including held joints."""

    def __init__(self, trajectory: PreparationTrajectory, limits: MotionLimits):
        self.trajectory = trajectory
        self.limits = limits
        self.samples = deque()
        self.settle_detail = "waiting for trajectory completion"
        self.has_reached_ready = False
        self.hold_deviation_since: float | None = None

    def update(self, elapsed_s, positions, velocities, previous_command) -> bool:
        expected = self.trajectory.goal.keys()
        if positions.keys() != expected or velocities.keys() != expected:
            raise RuntimeError("Incomplete joint feedback")
        if not all(math.isfinite(v) for v in [*positions.values(), *velocities.values()]):
            raise RuntimeError("Non-finite joint feedback")
        for key, measured in positions.items():
            if abs(measured - previous_command[key]) > self.limits.tracking_error_deg:
                raise RuntimeError(f"Following error exceeds {self.limits.tracking_error_deg} degrees: {key}")
        if elapsed_s < self.trajectory.duration_s:
            return False
        deviations = [
            f"{key}: error={positions[key] - target:.3f} deg, velocity={velocities[key]:.3f} deg/s"
            for key, target in self.trajectory.goal.items()
            if abs(positions[key] - target) > self.limits.arrival_tolerance_deg
            or abs(velocities[key]) > self.limits.arrival_speed_deg_s
        ]
        arrived = not deviations
        # Damiao wrist velocity feedback can jump between -0.700 and -2.099
        # deg/s while position is unchanged. Use a full fresh window of position
        # stability AND mean absolute speed; never average signed speeds, which
        # could hide an oscillation. Large spikes and position errors reset it.
        position_ok = all(
            abs(positions[key] - target) <= self.limits.arrival_tolerance_deg
            for key, target in self.trajectory.goal.items()
        )
        peak_ok = all(abs(v) <= self.limits.settle_peak_speed_deg_s for v in velocities.values())
        if self.samples and (
            elapsed_s <= self.samples[-1][0]
            or elapsed_s - self.samples[-1][0] > self.limits.feedback_timeout_s
        ):
            self.samples.clear()
        ready = False
        if not position_ok or not peak_ok:
            self.samples.clear()
            self.settle_detail = "; ".join(deviations)
        else:
            self.samples.append((elapsed_s, dict(positions), dict(velocities)))
            cutoff = elapsed_s - self.limits.settle_s
            # Retain the boundary sample so the window covers the full duration.
            while len(self.samples) > 1 and self.samples[1][0] <= cutoff:
                self.samples.popleft()
            span = {
                key: max(s[1][key] for s in self.samples) - min(s[1][key] for s in self.samples)
                for key in expected
            }
            mean_speed = {
                key: sum(abs(s[2][key]) for s in self.samples) / len(self.samples) for key in expected
            }
            window_s = elapsed_s - self.samples[0][0]
            worst_span = max(span, key=span.get)
            worst_speed = max(mean_speed, key=mean_speed.get)
            ready = (
                window_s >= self.limits.settle_s - 1e-9
                and span[worst_span] <= self.limits.settle_position_span_deg
                and mean_speed[worst_speed] <= self.limits.arrival_speed_deg_s
                and arrived
            )
            self.settle_detail = (
                f"window={window_s:.2f}/{self.limits.settle_s:.2f}s; "
                f"max position span {worst_span}={span[worst_span]:.3f} deg; "
                f"max mean absolute speed {worst_speed}={mean_speed[worst_speed]:.3f} deg/s"
            )
            if deviations:
                self.settle_detail += "; current: " + "; ".join(deviations)
        if self.has_reached_ready:
            # The initial arrival deadline must not remain active forever.
            # During holding, fault only on a sustained departure from the
            # arrival conditions; tracking/freshness checks remain immediate.
            if ready:
                self.hold_deviation_since = None
            elif self.hold_deviation_since is None:
                self.hold_deviation_since = elapsed_s
            elif elapsed_s - self.hold_deviation_since >= self.limits.settle_timeout_s:
                raise RuntimeError("Prepared pose remained outside holding tolerance: " + self.settle_detail)
        elif not ready and elapsed_s > self.trajectory.duration_s + self.limits.settle_timeout_s:
            raise RuntimeError("Preparation did not settle at the measured target: " + self.settle_detail)
        if ready:
            self.has_reached_ready = True
        return ready


class DamiaoPreparationBackend:
    """Use follower gains/mapping from follow.md, without connect-time zeroing.

    Opening a normal follower also runs an enabling handshake and may write zero.
    Here each bus is opened with handshake=False and validated before enable.
    This backend owns CAN for its entire lifetime, including the READY hold.
    """

    def __init__(self, args, limits: MotionLimits):
        # Optional hardware dependencies are deliberately lazy.
        from lerobot.robots.openarm_follower import OpenArmFollower, OpenArmFollowerConfig

        self.limits = limits
        self.arms = {}
        self.support_gain_scale = support_gain_scale(args.support_gain_scale)
        self.gripper_gain_scale = gripper_gain_scale(getattr(args, "gripper_gain_scale", 1.0))
        self.enabled = False
        self._reported_feedback_clips = set()
        sides = ("left", "right") if args.arm == "both" else (args.arm,)
        for side in sides:
            self.arms[side] = OpenArmFollower(
                OpenArmFollowerConfig(
                    port=getattr(args, f"{side}_port"),
                    id=f"{args.id}_{side}",
                    side=side,
                    use_can_fd=True,
                    cameras={},
                )
            )

    def connect(self):
        for arm in self.arms.values():
            if not arm.is_calibrated:
                raise RuntimeError(
                    f"Missing existing calibration: {arm.calibration_fpath}; no auto-calibration"
                )
        for arm in self.arms.values():
            arm.bus.connect(handshake=False)

    def read(self):
        positions, velocities = {}, {}
        for side, arm in self.arms.items():
            states = arm.bus.sync_read_all_states(max_age_s=self.limits.feedback_timeout_s)
            for motor in arm.bus.motors:
                key = f"{side}_{motor}.pos"
                positions[key] = float(states[motor]["position"])
                velocities[key] = float(states[motor]["velocity"])
        if not all(math.isfinite(v) for v in [*positions.values(), *velocities.values()]):
            raise RuntimeError("Non-finite motor feedback")
        return positions, velocities

    def validate_positions(self, positions, *, measurement=False):
        checked = dict(positions)
        errors = []
        reported_feedback_clips = vars(self).setdefault("_reported_feedback_clips", set())
        for side, arm in self.arms.items():
            for motor, (low, high) in arm.config.joint_limits.items():
                # Permit small measured endpoint offsets, never out-of-range commands.
                key = f"{side}_{motor}.pos"
                value = positions[key]
                tolerance = 0.0
                if measurement:
                    tolerance = (
                        GRIPPER_FEEDBACK_TOLERANCE_DEG if motor == "gripper" else JOINT_FEEDBACK_TOLERANCE_DEG
                    )
                    if motor == "joint_4" and low == 0.0 and value < low:
                        tolerance = ELBOW_ZERO_FEEDBACK_TOLERANCE_DEG
                if not math.isfinite(value) or not low - tolerance <= value <= high + tolerance:
                    errors.append(f"{key}={value} outside configured operating limits [{low}, {high}]")
                    continue
                checked[key] = min(high, max(low, value))
                if checked[key] != value and key not in reported_feedback_clips:
                    reported_feedback_clips.add(key)
                    logger.warning(
                        "%s feedback %.4f deg is within %.2f deg endpoint tolerance; "
                        "holding target clipped to %.4f deg (limits [%.1f, %.1f])",
                        key,
                        value,
                        tolerance,
                        checked[key],
                        low,
                        high,
                    )
        if errors:
            raise RuntimeError("Position validation failed:\n" + "\n".join(errors))
        return checked

    def send(self, positions):
        self.validate_positions(positions)
        for side, arm in self.arms.items():
            prefix = f"{side}_"
            action = {
                key.removeprefix(prefix): val for key, val in positions.items() if key.startswith(prefix)
            }
            kp, kd = support_gains(
                arm.config, self.support_gain_scale, getattr(self, "gripper_gain_scale", 1.0)
            )
            arm.send_action(action, custom_kp=kp, custom_kd=kd)

    def enable(self, initial):
        for side, arm in self.arms.items():
            kp, kd = support_gains(
                arm.config, self.support_gain_scale, getattr(self, "gripper_gain_scale", 1.0)
            )
            logger.info("%s preparation support gains: Kp=%s Kd=%s", side, kp, kd)
        # Seed current-position commands, then enable and immediately resend them.
        self.send(initial)
        self.enabled = True  # Also covers a partially successful enable sequence.
        for arm in self.arms.values():
            arm.bus.enable_torque()
        self.send(initial)

    def close(self, *, disable_torque):
        for arm in self.arms.values():
            try:
                if arm.bus.is_connected:
                    arm.bus.disconnect(disable_torque=disable_torque)
                elif arm.bus.canbus is not None:
                    arm.bus.canbus.shutdown()
            except Exception:
                logger.exception("Failed to close CAN interface %s", arm.config.port)


def run_motion(
    backend,
    start,
    goal,
    limits,
    stop,
    *,
    hold,
    phase=None,
    verify_arrival=True,
    tick_callback=None,
):
    trajectory = PreparationTrajectory(start, goal, limits)
    monitor = PreparationMonitor(trajectory, limits)
    phase = phase or ("PREPARING" if hold else "LOWERING")
    logger.info("%s: motion duration %.1fs", phase, trajectory.duration_s)
    previous_command = dict(start)
    started = previous_tick = time.monotonic()
    ready_announced = False
    next_settle_report = trajectory.duration_s + 2
    while not stop.is_set():
        tick = time.monotonic()
        if tick - previous_tick > limits.feedback_timeout_s:
            raise RuntimeError("Control loop stalled; preparation interrupted")
        positions, velocities = backend.read()
        now = time.monotonic()
        if now - tick > limits.feedback_timeout_s:
            raise RuntimeError("Feedback read exceeded control deadline")
        elapsed = now - started
        if verify_arrival:
            ready = monitor.update(elapsed, positions, velocities, previous_command)
        else:
            # Some hardware reports a small, repeatable loaded offset and
            # quantized nonzero velocity at a stationary pose. For a timed
            # command trajectory, retain only freshness and the existing hard
            # following-error check; do not add a second arrival/settle gate.
            for key, measured in positions.items():
                if abs(measured - previous_command[key]) > limits.tracking_error_deg:
                    raise RuntimeError(f"Following error exceeds {limits.tracking_error_deg} degrees: {key}")
            ready = elapsed >= trajectory.duration_s
        if not ready and elapsed >= next_settle_report:
            logger.info("%s settling: %s", phase, monitor.settle_detail)
            next_settle_report = elapsed + 2
        if ready and not ready_announced:
            if verify_arrival:
                messages = {
                    "PREPARING": "READY: holding prepared pose",
                    "HOMING": "HOMED: calibrated arm zero reached and settled",
                }
                logger.info("%s", messages.get(phase, "LOWERED: reached lowering target"))
            else:
                logger.info("%s COMMAND COMPLETE: timed target trajectory finished", phase)
            ready_announced = True
        previous_command = trajectory.sample(elapsed)
        backend.send(previous_command)
        if tick_callback is not None:
            tick_callback()
        if ready and not hold:
            return
        previous_tick = tick
        stop.wait(max(0.0, 1.0 / limits.fps - (time.monotonic() - tick)))
    return previous_command


def lower(args, limits, stop, backend_factory=DamiaoPreparationBackend):
    """Recover from an exited holder: lower calibrated elbows, then disable.

    Never cut existing torque on a failed preflight or an interrupted descent.
    Unlike normal preparation, this does not return to the captured raised pose.
    """
    backend = backend_factory(args, limits)
    completed = False
    try:
        backend.connect()
        positions, velocities = backend.read()
        if any(abs(v) > limits.arrival_speed_deg_s for v in velocities.values()):
            raise RuntimeError("Arms must be stationary before lowering; existing torque preserved")
        initial = backend.validate_positions(positions, measurement=True)
        goal = backend.validate_positions(preparation_goal(initial, 0.0))
        if stop.is_set():
            return
        backend.enable(initial)
        last_command = run_motion(backend, initial, goal, limits, stop, hold=False)
        if last_command is not None:
            logger.warning("Lowering interrupted: last target retained, torque left enabled")
            return
        completed = True
    except Exception:
        logger.exception("Lowering failed: torque preserved; inspect locally before retrying")
        raise
    finally:
        backend.close(disable_torque=completed)
    logger.info("LOWERED: torque disabled")


def execute(args, limits, stop, backend_factory=DamiaoPreparationBackend):
    backend = backend_factory(args, limits)
    disable_on_exit = True
    initial = None
    try:
        backend.connect()
        positions, velocities = backend.read()
        if any(abs(v) > limits.arrival_speed_deg_s for v in velocities.values()):
            raise RuntimeError("Arms must be stationary before preparing")
        initial = backend.validate_positions(positions, measurement=True)
        goal = backend.validate_positions(preparation_goal(initial, args.elbow_deg))
        if stop.is_set():
            return
        backend.enable(initial)
        last_command = run_motion(backend, initial, goal, limits, stop, hold=True)
        # Keep targets continuous: measured positions can lag under gravity.
        # Starting the return target at that lagging measurement causes a step.
        positions, _ = backend.read()
        backend.validate_positions(positions, measurement=True)
        run_motion(backend, last_command, initial, limits, threading.Event(), hold=False)
    except Exception:
        if backend.enabled:
            disable_on_exit = False
            try:
                positions, _ = backend.read()
                backend.send(backend.validate_positions(positions, measurement=True))
                logger.error("FAULT: requested hold at measured pose; torque left enabled")
            except Exception:
                logger.exception("FAULT: cannot confirm hold; last motor command may remain active")
            logger.error("Support arms and inspect locally; do not automatically restart after a fault")
        raise
    finally:
        backend.close(disable_torque=disable_on_exit)


def inspect_feedback(args, limits, backend_factory=DamiaoPreparationBackend):
    """Read all joint states; never enable, disable, move, or write motor zero."""
    backend = backend_factory(args, limits)
    try:
        backend.connect()
        positions, velocities = backend.read()
        print("READ ONLY: motor feedback in degrees; no enable/disable, motion or zero commands")
        for side, arm in backend.arms.items():
            for motor, (low, high) in arm.config.joint_limits.items():
                key = f"{side}_{motor}.pos"
                state = "in range" if low <= positions[key] <= high else "outside nominal limits"
                print(
                    f"{key:24s} pos={positions[key]:9.4f} deg  vel={velocities[key]:8.4f} deg/s "
                    f"limits=[{low:6.1f}, {high:6.1f}]  {state}"
                )
    finally:
        backend.close(disable_torque=False)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true", help="Connect CAN and move the real arms")
    mode.add_argument(
        "--lower", action="store_true", help="Move real J4 joints to zero, then disable; no raise/return"
    )
    mode.add_argument(
        "--read-state", action="store_true", help="Read feedback without enabling or moving motors"
    )
    parser.add_argument("--arm", choices=("left", "right", "both"), default="both")
    parser.add_argument("--left-port", default="can2")
    parser.add_argument("--right-port", default="can3")
    parser.add_argument("--id", default="my_bimanual_follower")
    parser.add_argument("--elbow-deg", type=float, default=90.0)
    parser.add_argument("--duration-s", type=float, default=20.0)
    parser.add_argument("--speed-deg-s", type=float, default=10.0)
    parser.add_argument("--accel-deg-s2", type=float, default=5.0)
    parser.add_argument(
        "--support-gain-scale",
        type=support_gain_scale,
        default=1.0,
        help="Preparation-only J1/J4/J7 Kp multiplier (1–4); Kd scales by its square root",
    )
    return parser


def main():
    args = build_parser().parse_args()
    if not math.isfinite(args.elbow_deg) or not 0 <= args.elbow_deg <= 135:
        raise ValueError("--elbow-deg must be between 0 and 135 degrees")
    if args.left_port == args.right_port and args.arm == "both":
        raise ValueError("Left and right arms require different CAN interfaces")
    limits = MotionLimits(args.duration_s, args.speed_deg_s, args.accel_deg_s2)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    if args.read_state:
        inspect_feedback(args, limits)
        return
    if not args.execute and not args.lower:
        sides = ("left", "right") if args.arm == "both" else (args.arm,)
        start = {f"{side}_joint_4.pos": 0.0 for side in sides}
        trajectory = PreparationTrajectory(start, preparation_goal(start, args.elbow_deg), limits)
        print("PREVIEW ONLY: assumed J4 starts at 0 degrees; no CAN connection or motor commands")
        print(f"Ports: left={args.left_port}, right={args.right_port}; selected={args.arm}")
        print(f"J1/J4/J7 support gain scale: {args.support_gain_scale:g} (other joints unchanged)")
        for fraction in (0, 0.25, 0.5, 0.75, 1):
            t = fraction * trajectory.duration_s
            print(f"t={t:5.1f}s  {trajectory.sample(t)}")
        return
    stop = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stop.set())
    if args.lower:
        lower(args, limits, stop)
    else:
        execute(args, limits, stop)


if __name__ == "__main__":
    main()

"""One-PC VR control: home to calibrated zero, prepare, teleoperate, home, disable.

Only this process owns CAN. The isolated IK worker uses the existing VR Conda
environment and communicates through a private local socket, without a host PC.
"""

import argparse
import importlib.util
import json
import logging
import math
import os
import signal
import socket
import subprocess
import threading
import time
from dataclasses import replace
from pathlib import Path

from lerobot.scripts.openarm_prepare import (
    DamiaoPreparationBackend,
    MotionLimits,
    PreparationMonitor,
    PreparationTrajectory,
    gripper_gain_scale,
    home_goal,
    preparation_goal,
    run_motion,
    support_gain_scale,
)

logger = logging.getLogger(__name__)
SIDES = ("left", "right")
# Degrees in the calibrated motor coordinates: increasing closes the gripper.
# Bound the PD position error when an object prevents further closing. This is
# not a force measurement or a bypass of the 10-degree following-error guard.
GRIPPER_CLOSE_LEAD_DEG = 5.0
JOINT_SLOWDOWN_DEG = 8.0
JOINT_LOOKAHEAD_S = 0.1
JOINT_RECOVERY_SPEED_DEG_S = 5.0
DEFAULT_J2_MOTION_WEIGHT = 8.0
MOBILE_FRAME_SIZE = 9
MOBILE_NEUTRAL_FRAME = bytes((0x55, 0, 0, 0, 0, 0, 0, 0, 0))
# Repeated while the arms prepare: straighten and enable the chassis, and let
# the lift's proven first-takeover path hold its measured position at zero speed.
MOBILE_STARTUP_FRAME = bytes((0x55, 0, 0, 0, 0, 0, 0, 1, 0))
KEYS = tuple(
    f"{side}_{motor}.pos" for side in SIDES for motor in [*[f"joint_{i}" for i in range(1, 8)], "gripper"]
)


def open_mobile_serial(path):
    """Load the example transport without making the core package depend on pyserial."""
    module_path = Path(__file__).resolve().parents[3] / "examples/mobile_openarm/mobile_serial_port.py"
    spec = importlib.util.spec_from_file_location("openarm_mobile_serial_port", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load mobile serial transport from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SerialPort(path)


class MobileOutput:
    """Validate local IK frames and exclusively own the C-board serial port."""

    def __init__(self, path, port_factory=open_mobile_serial):
        self.port = port_factory(path)
        self.closed = False
        self.hold_frame = MOBILE_NEUTRAL_FRAME
        try:
            self.write(MOBILE_NEUTRAL_FRAME)
        except Exception:
            self.closed = True
            self.port.close()
            raise

    @staticmethod
    def validate(frame):
        if isinstance(frame, bytes):
            payload = frame
        elif (
            isinstance(frame, list)
            and len(frame) == MOBILE_FRAME_SIZE
            and all(type(value) is int and 0 <= value <= 255 for value in frame)
        ):
            payload = bytes(frame)
        else:
            raise ValueError("Invalid mobile frame from IK worker")
        if len(payload) != MOBILE_FRAME_SIZE or payload[0] != 0x55:
            raise ValueError("Invalid mobile frame from IK worker")
        return payload

    def write(self, frame):
        payload = self.validate(frame)
        # The C board transmits a 12-byte status frame every 10 ms. Always
        # drain it before writing so the CH343/tty receive queue cannot fill
        # and back-pressure the board's blocking HAL_UART_Transmit call.
        self.port.read()
        if self.port.write(payload) != len(payload):
            raise RuntimeError("Incomplete mobile serial write")
        self.port.read()

    def write_startup(self):
        self.write(MOBILE_STARTUP_FRAME)

    def write_live(self, frame):
        """Send live motion and remember an equivalent stationary hold frame."""
        payload = self.validate(frame)
        held = bytearray(payload)
        held[3] = 0
        if held[6] == 1:
            held[1] = held[2] = 0
        else:
            held[4] = held[5] = 0
        held[7] = held[8] = 0
        self.hold_frame = bytes(held)
        self.write(payload)

    def stop_without_reconfiguring(self):
        self.write(self.hold_frame)

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            self.port.reset_output_buffer()
            self.port.write(MOBILE_NEUTRAL_FRAME)
            time.sleep(0.02)
        finally:
            self.port.close()


def operating_ranges(ranges, margin=3.0):
    """Inset arm targets from physical limits; keep calibrated gripper travel."""
    return {
        key: (low, high) if key.endswith("_gripper.pos") else (low + margin, high - margin)
        for key, (low, high) in ranges.items()
    }


class FeedbackEnvelope:
    """Latch inward-only recovery for small excursions; retain the hard guard."""

    def __init__(self, ranges):
        self.ranges = ranges
        self.recovery_directions = {}

    def update(self, positions, now):
        if positions.keys() != self.ranges.keys() or not all(math.isfinite(v) for v in positions.values()):
            raise RuntimeError("Invalid motor feedback")
        for key, (low, high) in self.ranges.items():
            value = positions[key]
            excess = max(low - value, value - high, 0.0)
            if key.endswith("_gripper.pos"):
                if excess > 0.5:
                    raise RuntimeError(f"Gripper feedback outside endpoint tolerance: {key}={value:.3f}")
                continue
            if excess > 1.0:
                raise RuntimeError(f"Large joint-limit violation: {key}={value:.3f}, limits=[{low}, {high}]")
            if excess > 0:
                self.recovery_directions[key] = -1 if value < low else 1
            elif low + 0.5 <= value <= high - 0.5:
                # Hysteresis prevents repeated entry/exit at a noisy endpoint.
                self.recovery_directions.pop(key, None)
        return list(self.recovery_directions)


def send_packet(sock, packet):
    try:
        sock.send(json.dumps(packet, allow_nan=False).encode())
        return True
    except BlockingIOError:
        return False


def latest_packet(sock):
    latest = None
    for _ in range(64):
        try:
            raw = sock.recv(65536)
        except BlockingIOError:
            break
        packet = json.loads(raw)
        if not isinstance(packet, dict):
            raise ValueError("Invalid local IK packet")
        latest = packet
    return latest


class IKWorker:
    def __init__(self, args):
        self.socket, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_DGRAM)
        self.socket.setblocking(False)
        root = Path(__file__).resolve().parents[3]
        worker = root / "examples/mobile_openarm/vr_real_worker.py"
        command = [
            str(args.ik_python),
            str(worker),
            "--fd",
            str(child.fileno()),
            "--vr-endpoint",
            args.vr_endpoint,
            "--vr-position-scale",
            str(args.vr_position_scale),
            "--joint-speed-deg-s",
            str(args.joint_speed_deg_s),
            "--arm-linear-speed-m-s",
            str(args.arm_linear_speed_m_s),
            "--arm-angular-speed-deg-s",
            str(args.arm_angular_speed_deg_s),
            "--joint-limit-margin-deg",
            str(args.joint_limit_margin_deg),
            "--ik-j2-motion-weight",
            str(args.ik_j2_motion_weight),
            "--vr-filter-scale",
            str(args.vr_filter_scale),
            "--mobile-speed",
            str(args.mobile_speed),
            "--mobile-spin",
            str(args.mobile_spin),
            "--lift-speed",
            str(args.lift_speed),
        ]
        for name in ("drive", "steer", "spin", "lift"):
            command.extend((f"--{name}-sign", str(getattr(args, f"{name}_sign"))))
        if args.viewer:
            command.append("--viewer")
        env = dict(os.environ, PYTHONPATH=str(root / "src"), OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        env.pop("LD_LIBRARY_PATH", None)
        try:
            self.process = subprocess.Popen(
                command, pass_fds=(child.fileno(),), env=env, start_new_session=True
            )
        except Exception:
            self.socket.close()
            raise
        finally:
            child.close()

    def wait_ready(self, stop, timeout=60):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and not stop.is_set():
            if self.process.poll() is not None:
                raise RuntimeError("IK worker exited before startup; motors were not enabled")
            packet = latest_packet(self.socket)
            if packet and packet.get("type") == "ready":
                return
            stop.wait(0.05)
        raise RuntimeError("IK startup cancelled or timed out; motors were not enabled")

    def close(self):
        try:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=3)
        finally:
            self.socket.close()


class CommandGate:
    """Motor-side freshness, schema, deadman and joint slew limits."""

    def __init__(self, initial, speed=30.0, acceleration=120.0, *, gripper_speed=35.0, ranges=None):
        self.command = dict(initial)
        self.velocity = dict.fromkeys(initial, 0.0)
        self.speed = speed
        self.acceleration = acceleration
        self.gripper_speed = gripper_speed
        self.ranges = ranges
        self.epoch = 0
        self.limited_joints = set()
        self.blocked_joints = set()

    def hold(self):
        self.velocity = dict.fromkeys(self.command, 0.0)
        return dict(self.command)

    def apply(self, packet, now, dt, *, measured=None, velocities=None, recovery_directions=None):
        self.limited_joints.clear()
        self.blocked_joints.clear()
        recovery_directions = recovery_directions or {}
        if measured is not None and (
            measured.keys() != self.command.keys() or not all(math.isfinite(v) for v in measured.values())
        ):
            raise ValueError("Invalid motor feedback for command control")
        if velocities is not None and (
            velocities.keys() != self.command.keys() or not all(math.isfinite(v) for v in velocities.values())
        ):
            raise ValueError("Invalid motor velocities for command control")
        if packet is None:
            return self.hold()
        if packet.get("epoch") != self.epoch:
            return self.hold()
        if not all(0 <= now - float(packet[key]) <= 0.2 for key in ("state_time", "sent_at")):
            return self.hold()
        target = packet["positions"]
        if target.keys() != self.command.keys() or not all(math.isfinite(v) for v in target.values()):
            raise ValueError("Invalid IK joint targets")
        dt = min(0.05, max(0.0, dt))
        for side in SIDES:
            active = packet["active"][side] is True and 0 <= now - float(packet["vr_time"][side]) <= 0.3
            for key in self.command:
                if not key.startswith(side + "_"):
                    continue
                if not active or dt == 0:
                    self.velocity[key] = 0.0
                    continue
                gripper = key.endswith("_gripper.pos")
                destination = target[key]
                if self.ranges is not None:
                    low, high = self.ranges[key]
                    # A captured startup pose may be outside the inset. Permit
                    # holding or moving inward, never farther toward the edge.
                    low, high = min(low, self.command[key]), max(high, self.command[key])
                    destination = min(high, max(low, destination))
                if gripper and measured is not None and destination > measured[key]:
                    # Keep a bounded closing preload instead of integrating a
                    # full-close target through an object. Opening remains free
                    # to reverse immediately, subject to the same slew limits.
                    destination = min(destination, measured[key] + GRIPPER_CLOSE_LEAD_DEG)
                error = destination - self.command[key]
                speed = self.gripper_speed if gripper else self.speed
                acceleration = 100.0 if gripper else self.acceleration
                if not gripper and self.ranges is not None and error:
                    direction = math.copysign(1.0, error)
                    low, high = self.ranges[key]
                    observed = self.command[key] if measured is None else measured[key]
                    observed_velocity = 0.0 if velocities is None else velocities[key]
                    # Predict outward travel during feedback/IK latency. Use
                    # whichever is closer to the edge: actual or commanded q.
                    predicted = observed + observed_velocity * JOINT_LOOKAHEAD_S
                    clearance = (
                        high - max(self.command[key], observed, predicted)
                        if direction > 0
                        else min(self.command[key], observed, predicted) - low
                    )
                    clearance = max(0.0, clearance)
                    speed = min(
                        speed, speed * clearance / JOINT_SLOWDOWN_DEG, math.sqrt(2 * acceleration * clearance)
                    )
                    outward = recovery_directions.get(key)
                    if outward is not None:
                        speed = 0.0 if direction == outward else min(speed, JOINT_RECOVERY_SPEED_DEG_S)
                    if outward is not None or speed < min(
                        self.speed, max(abs(error / dt), abs(self.velocity[key]))
                    ):
                        self.limited_joints.add(key)
                    if speed <= 1e-6:
                        self.blocked_joints.add(key)
                desired = max(-speed, min(speed, error / dt))
                dv = acceleration * dt
                velocity = max(self.velocity[key] - dv, min(self.velocity[key] + dv, desired))
                # A feedback limit has priority over the normal acceleration
                # ramp; never carry old outward speed into a blocked boundary.
                velocity = max(-speed, min(speed, velocity))
                step = velocity * dt
                # Do not carry a previous velocity past a target or away from it.
                if step * error <= 0:
                    step = 0.0
                step = math.copysign(min(abs(error), abs(step)), error)
                self.command[key] += step
                self.velocity[key] = step / dt
        return dict(self.command)


def control_loop(backend, worker, initial_command, limits, stop, args, mobile=None):
    packet = None
    active = None
    timed_out = True
    started = previous_tick = time.monotonic()
    next_report = started
    ranges = {
        f"{side}_{motor}.pos": bounds
        for side, arm in backend.arms.items()
        for motor, bounds in arm.config.joint_limits.items()
    }
    gate = CommandGate(
        initial_command,
        args.joint_speed_deg_s,
        acceleration=args.joint_acceleration_deg_s2,
        gripper_speed=args.gripper_speed_deg_s,
        ranges=operating_ranges(ranges, args.joint_limit_margin_deg),
    )
    envelope = FeedbackEnvelope(ranges)
    previous_excursions = []
    reset_motion = None
    last_reset_id = 0
    logger.info(
        "VR_READY: release grips, then grip to control; release both grips + X resets both arms to J4=90"
    )
    while not stop.is_set():
        tick = time.monotonic()
        if tick - previous_tick > limits.feedback_timeout_s:
            raise RuntimeError("Motor control loop stalled")
        positions, velocities = backend.read()
        now = time.monotonic()
        if now - tick > limits.feedback_timeout_s:
            raise RuntimeError("Motor feedback deadline exceeded")
        excursions = envelope.update(positions, now)
        if excursions != previous_excursions:
            logger.warning(
                "Joint-limit recovery: %s; affected joints inward only at <=%.1f deg/s; grips remain active",
                excursions or "cleared",
                JOINT_RECOVERY_SPEED_DEG_S,
            )
        previous_excursions = excursions
        for key, value in positions.items():
            if abs(value - gate.command[key]) > limits.tracking_error_deg:
                raise RuntimeError(
                    f"VR following error: {key}; measured={value:.3f} deg, "
                    f"command={gate.command[key]:.3f} deg, error={value - gate.command[key]:.3f} deg"
                )
        if worker.process.poll() is not None:
            logger.warning("IK/viewer exited: returning to calibrated arm zero")
            break
        incoming = latest_packet(worker.socket)
        if incoming and incoming.get("type") == "stop":
            break
        if incoming and incoming.get("type") == "target" and incoming.get("epoch") == gate.epoch:
            packet = incoming
        fresh = (
            packet is not None
            and 0 <= now - packet["state_time"] <= 0.2
            and 0 <= now - packet["sent_at"] <= 0.2
        )
        if not fresh:
            if not timed_out:
                gate.epoch += 1
                logger.warning("IK stream stale: holding; release and re-grip after recovery")
            timed_out = True
            packet = None
        else:
            timed_out = False
            backend.validate_positions(packet["positions"])
        reset_safe = bool(
            packet
            and packet.get("reset_safe") is True
            and not any(packet["active"].values())
            and all(0 <= now - packet["vr_time"][side] <= 0.3 for side in SIDES)
        )
        request_id = packet.get("reset_id", 0) if packet else 0
        if type(request_id) is int and request_id > last_reset_id:
            # Consume even a rejected request: stale/unsafe button events must
            # never turn into delayed automatic motion when tracking recovers.
            last_reset_id = request_id
            if reset_motion is None and reset_safe:
                goal = backend.validate_positions(preparation_goal(home_goal(gate.command)))
                reset_limits = replace(
                    limits, duration_s=args.prepare_duration_s, speed_deg_s=min(5.0, limits.speed_deg_s)
                )
                reset_motion = PreparationTrajectory(gate.command, goal, reset_limits)
                # Grippers may intentionally hold a bounded PD preload against
                # an object. Keep their targets and the global following guard,
                # but require arrival/settling only for the fourteen arm axes.
                reset_arm_keys = [key for key in goal if not key.endswith("_gripper.pos")]
                reset_monitor = PreparationMonitor(
                    PreparationTrajectory(
                        {key: gate.command[key] for key in reset_arm_keys},
                        {key: goal[key] for key in reset_arm_keys},
                        reset_limits,
                    ),
                    reset_limits,
                )
                reset_started = now
                gate.hold()
                logger.info(
                    "VR_RESET: both arms -> J1-J7 zero except J4=90; grippers held; duration %.1fs",
                    reset_motion.duration_s,
                )
        if reset_motion is not None:
            command = gate.hold()
            if reset_safe:
                elapsed = now - reset_started
                ready = reset_monitor.update(
                    elapsed,
                    {key: positions[key] for key in reset_arm_keys},
                    {key: velocities[key] for key in reset_arm_keys},
                    {key: gate.command[key] for key in reset_arm_keys},
                )
                command = backend.validate_positions(reset_motion.sample(elapsed))
                gate.command = dict(command)
                if ready:
                    reset_motion = None
                    logger.info("VR_RESET_READY: prepared pose settled; release grips, then grip to control")
            else:
                reset_motion = None
                logger.warning(
                    "VR_RESET_CANCELLED: input lost or grip pressed; holding; press X again to retry"
                )
            if reset_motion is None:
                gate.epoch += 1
                packet = None
                timed_out = True
        else:
            command = gate.apply(
                packet,
                now,
                tick - previous_tick,
                measured=positions,
                velocities=velocities,
                recovery_directions=envelope.recovery_directions,
            )
        backend.send(command)
        if mobile is not None:
            mobile.write(packet["mobile_frame"] if fresh and packet is not None else MOBILE_NEUTRAL_FRAME)
        moving = tuple(
            bool(packet and packet["active"][side] and 0 <= now - packet["vr_time"][side] <= 0.3)
            for side in SIDES
        )
        if moving != active:
            logger.info("VR tracking control: left=%s right=%s", *moving)
            if packet and "reason" in packet:
                logger.info("VR input status: %s", packet["reason"])
            active = moving
        if now >= next_report:
            logger.info(
                "VR feedback: J4 left=%.2f right=%.2f deg; max tracking error=%.2f deg",
                positions["left_joint_4.pos"],
                positions["right_joint_4.pos"],
                max(abs(positions[key] - command[key]) for key in positions),
            )
            next_report = now + 5
            logger.info(
                "VR grippers measured/command: left=%.2f/%.2f right=%.2f/%.2f deg; closing lead <=%.1f deg",
                positions["left_gripper.pos"],
                command["left_gripper.pos"],
                positions["right_gripper.pos"],
                command["right_gripper.pos"],
                GRIPPER_CLOSE_LEAD_DEG,
            )
        send_packet(
            worker.socket,
            {
                "type": "state",
                "epoch": gate.epoch,
                "state_time": time.monotonic(),
                "positions": positions,
                "command": command,
                "limits": ranges,
                "velocities": velocities,
                "limited_joints": sorted(gate.limited_joints),
                "blocked_joints": sorted(gate.blocked_joints),
                "recovery_directions": dict(envelope.recovery_directions),
                "resetting": reset_motion is not None,
            },
        )
        previous_tick = tick
        if args.run_seconds and now - started >= args.run_seconds:
            break
        stop.wait(max(0, 1 / limits.fps - (time.monotonic() - tick)))
    return dict(gate.command)


def stationary_start(backend, limits, stop, tick_callback=None):
    """Require 0.2 s of quiet feedback, allowing transient startup samples."""
    deadline = time.monotonic() + 3.0
    quiet_since = None
    reference = None
    while not stop.is_set():
        tick = time.monotonic()
        positions, velocities = backend.read()
        now = time.monotonic()
        if now - tick > limits.feedback_timeout_s:
            raise RuntimeError("Startup feedback deadline exceeded")
        checked = backend.validate_positions(positions, measurement=True)
        if tick_callback is not None:
            tick_callback()
        quiet = all(abs(v) <= limits.arrival_speed_deg_s for v in velocities.values())
        if reference is not None and max(abs(positions[k] - reference[k]) for k in positions) > 0.5:
            quiet = False
        if not quiet:
            quiet_since = reference = None
        elif quiet_since is None:
            quiet_since, reference = now, dict(positions)
        elif now - quiet_since >= 0.2:
            return checked
        if now >= deadline:
            key = max(velocities, key=lambda k: abs(velocities[k]))
            raise RuntimeError(
                f"Arms did not remain stationary before preparation: {key} velocity={velocities[key]:.3f} deg/s"
            )
        stop.wait(0.02)
    return None


def run(args, stop, backend_factory=DamiaoPreparationBackend, worker_factory=IKWorker):
    limits = MotionLimits(duration_s=args.prepare_duration_s)
    worker = worker_factory(args)
    backend = None
    mobile = None
    completed = False
    try:
        logger.info("STARTING: loading IK before opening CAN")
        worker.wait_ready(stop)
        if args.mobile_port:
            mobile = MobileOutput(args.mobile_port)
            logger.info(
                "Mobile C-board preflight passed on %s; neutral command sent",
                args.mobile_port,
            )
        backend = backend_factory(args, limits)
        backend.connect()
        startup_tick = (lambda: mobile.write(MOBILE_NEUTRAL_FRAME)) if mobile is not None else None
        initial = stationary_start(backend, limits, stop, startup_tick)
        if initial is None:
            return
        home = backend.validate_positions(home_goal(initial))
        goal = backend.validate_positions(preparation_goal(home))
        if stop.is_set():
            return
        backend.enable(initial)
        last = run_motion(
            backend,
            initial,
            home,
            MotionLimits(duration_s=args.home_duration_s),
            stop,
            hold=False,
            phase="HOMING",
            verify_arrival=False,
            tick_callback=startup_tick,
        )
        if last is not None:
            logger.warning("Homing interrupted: last target retained, torque preserved; preparation skipped")
            return
        last = run_motion(
            backend,
            home,
            goal,
            limits,
            stop,
            hold=False,
            phase="PREPARING",
            verify_arrival=False,
            tick_callback=startup_tick,
        )
        if last is None:
            last = control_loop(backend, worker, goal, limits, stop, args, mobile)
            if mobile is not None:
                mobile.write(MOBILE_NEUTRAL_FRAME)
        logger.info("STOPPING: returning to calibrated arm zero, then disabling torque")
        # Preserve command continuity and use a separate, slower return duration.
        run_motion(
            backend,
            last,
            home,
            MotionLimits(duration_s=args.lower_duration_s),
            threading.Event(),
            hold=False,
            tick_callback=startup_tick,
        )
        completed = True
    except Exception:
        logger.exception(
            "VR FAULT: last command retained; existing torque preserved. Inspect before recovery"
        )
        raise
    finally:
        try:
            if mobile is not None:
                mobile.close()
        finally:
            try:
                if backend is not None:
                    backend.close(disable_torque=completed)
            finally:
                worker.close()
    logger.info("STOPPED: calibrated arm zero reached, torque disabled")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute", action="store_true", help="Enable the real arms and run preparation + VR"
    )
    parser.add_argument("--left-port", default="can2")
    parser.add_argument("--right-port", default="can3")
    parser.add_argument("--id", default="my_bimanual_follower")
    parser.add_argument("--home-duration-s", type=float, default=10)
    parser.add_argument("--prepare-duration-s", type=float, default=20)
    parser.add_argument("--lower-duration-s", type=float, default=30)
    parser.add_argument("--support-gain-scale", type=support_gain_scale, default=4)
    parser.add_argument("--joint-speed-deg-s", type=float, default=30)
    parser.add_argument("--joint-acceleration-deg-s2", type=float, default=120)
    parser.add_argument("--gripper-speed-deg-s", type=float, default=35)
    parser.add_argument("--gripper-gain-scale", type=gripper_gain_scale, default=1.5)
    parser.add_argument("--arm-linear-speed-m-s", type=float, default=0.2)
    parser.add_argument("--arm-angular-speed-deg-s", type=float, default=60)
    parser.add_argument("--vr-filter-scale", type=float, default=0.65)
    parser.add_argument("--joint-limit-margin-deg", type=float, default=3)
    parser.add_argument(
        "--ik-j2-motion-weight",
        type=float,
        default=DEFAULT_J2_MOTION_WEIGHT,
        help="IK J2 movement penalty relative to other joints (1-32; 1 restores equal weights)",
    )
    parser.add_argument("--vr-position-scale", type=float, default=1.0)
    parser.add_argument("--vr-endpoint", default="tcp://127.0.0.1:5567")
    parser.add_argument(
        "--mobile-port",
        default=None,
        help="C-board CH343 serial path; when set, integrate chassis/lift with real-arm VR control",
    )
    parser.add_argument("--mobile-speed", type=float, default=0.05, help="Translation m/s; maximum 0.1")
    parser.add_argument("--mobile-spin", type=float, default=0.005, help="Chassis spin scalar; maximum 0.01")
    parser.add_argument("--lift-speed", type=float, default=0.05, help="Lift motor rad/s; maximum 0.1")
    for name in ("drive", "steer", "spin", "lift"):
        parser.add_argument(
            f"--{name}-sign",
            type=int,
            choices=(-1, 1),
            default=-1 if name == "drive" else 1,
        )
    parser.add_argument(
        "--ik-python", type=Path, default=Path.home() / "anaconda3/envs/hei-rebot-vr/bin/python"
    )
    parser.add_argument("--viewer", action="store_true", help="Show measured joints in a MuJoCo window")
    parser.add_argument(
        "--run-seconds",
        type=float,
        default=0,
        help="Return automatically after this many VR-stage seconds; 0 holds indefinitely",
    )
    parser.set_defaults(arm="both")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    for name in (
        "home_duration_s",
        "prepare_duration_s",
        "lower_duration_s",
        "joint_speed_deg_s",
        "joint_acceleration_deg_s2",
        "vr_filter_scale",
        "vr_position_scale",
        "gripper_speed_deg_s",
        "arm_linear_speed_m_s",
        "arm_angular_speed_deg_s",
        "joint_limit_margin_deg",
        "ik_j2_motion_weight",
        "mobile_speed",
        "mobile_spin",
        "lift_speed",
    ):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"{name} must be finite and positive")
    if not math.isfinite(args.run_seconds) or args.run_seconds < 0:
        parser.error("run-seconds must be finite and nonnegative")
    if args.joint_speed_deg_s > 30 or args.vr_position_scale > 1:
        parser.error("Initial real profile supports joint speed <=30 deg/s and VR scale <=1")
    if args.gripper_speed_deg_s > 60 or args.arm_linear_speed_m_s > 0.2 or args.arm_angular_speed_deg_s > 60:
        parser.error("Real profile supports gripper <=60 deg/s, TCP <=0.2 m/s and rotation <=60 deg/s")
    if not 2 <= args.joint_limit_margin_deg <= 5:
        parser.error("Joint limit margin must be between 2 and 5 degrees")
    if not 1 <= args.ik_j2_motion_weight <= 32:
        parser.error("IK J2 motion weight must be between 1 and 32")
    if args.joint_acceleration_deg_s2 > 180 or not 0.5 <= args.vr_filter_scale <= 2:
        parser.error("VR acceleration must be <=180 deg/s² and filter scale between 0.5 and 2")
    if args.mobile_speed > 0.1 or args.mobile_spin > 0.01 or args.lift_speed > 0.1:
        parser.error("Mobile translation/lift must be <=0.1 and spin scalar <=0.01")
    if args.left_port == args.right_port:
        parser.error("Left and right CAN interfaces must differ")
    if not args.execute:
        print("PREVIEW: no CAN or motor commands. Add --execute to prepare and control both real arms.")
        print(
            f"Home J1-J7 to calibrated zero {args.home_duration_s:g}s -> "
            f"prepare J4=90 deg {args.prepare_duration_s:g}s -> VR hold/control -> "
            f"return to arm zero {args.lower_duration_s:g}s -> disable. Motor zeros are never rewritten."
        )
        return
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    run(args, stop)


if __name__ == "__main__":
    main()

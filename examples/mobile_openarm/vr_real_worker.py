#!/usr/bin/env python3
"""Isolated local IK worker. This module never imports a motor driver."""

import argparse
import socket
import time

import mujoco
import numpy as np
from keyboard_sim import parse_args
from mobile_mapper import MobileMapper
from vr_input import ZMQReceiver
from vr_sim import VRSimulator, add_vr_args

from lerobot.scripts.openarm_vr import (
    DEFAULT_J2_MOTION_WEIGHT,
    KEYS,
    SIDES,
    latest_packet,
    operating_ranges,
    send_packet,
)

J2_INWARD_WEIGHT = 32.0


class RealArmTargets:
    def __init__(
        self,
        receiver,
        *,
        position_scale=1.0,
        joint_speed_deg_s=30,
        linear_speed=0.2,
        angular_speed=60,
        margin=3,
        j2_motion_weight=DEFAULT_J2_MOTION_WEIGHT,
        filter_scale=0.65,
        mobile_speed=0.05,
        mobile_spin=0.005,
        lift_speed=0.05,
        drive_sign=-1,
        steer_sign=1,
        spin_sign=1,
        lift_sign=1,
    ):
        if not np.isfinite(j2_motion_weight) or not 1 <= j2_motion_weight <= 32:
            raise ValueError("IK J2 motion weight must be finite and between 1 and 32")
        if not np.isfinite(filter_scale) or not 0.5 <= filter_scale <= 2:
            raise ValueError("VR filter scale must be finite and between 0.5 and 2")
        args = parse_args(
            [
                "--no-chassis-physics",
                "--plain-scene",
                "--vr-position-scale",
                str(position_scale),
                "--joint-speed-rad-s",
                str(np.deg2rad(joint_speed_deg_s)),
                "--arm-linear-speed-m-s",
                str(linear_speed),
                "--arm-angular-speed-deg-s",
                str(angular_speed),
                "--gripper-speed-m-s",
                "0.01",
            ],
            configure=add_vr_args,
        )
        self.sim = VRSimulator(args, receiver)
        self.sim.target_filter_scale = filter_scale
        for side, arm in self.sim.arms.items():
            # Prefer redundant solutions with less shoulder J2 travel, while
            # retaining all seven axes and the same TCP pose objective/bounds.
            arm.ik.motion_weights[1] = j2_motion_weight
            # Inward shoulder roll is toward the narrow side of the mirrored
            # physical limits: positive on the left, negative on the right.
            arm.ik.inward_motion_sign = 1.0 if side == "left" else -1.0
            arm.ik.inward_motion_weight = J2_INWARD_WEIGHT
        print(
            f"[VR] IK J2 movement penalty: {j2_motion_weight:g}x; inward: {J2_INWARD_WEIGHT:g}x",
            flush=True,
        )
        self.receiver = receiver
        self.epoch = None
        self.limits = None
        self.margin = margin
        self.inner_limits = None
        self.unreachable = set()
        self.reset_id = 0
        self.reset_button_armed = False
        self.display = mujoco.MjData(self.sim.model)
        self.mobile = MobileMapper(
            speed=mobile_speed,
            spin=mobile_spin,
            lift_speed=lift_speed,
            drive_sign=drive_sign,
            steer_sign=steer_sign,
            spin_sign=spin_sign,
            lift_sign=lift_sign,
        )

    def rebase_anchor(self, side):
        """Discard blocked travel, retaining the *new* hand delta on this frame."""
        anchor = self.sim.anchors.get(side)
        if anchor is None:
            return
        arm = self.sim.arms[side]
        root = arm.ik.root_body
        rotation = self.sim.data.xmat[root].reshape(3, 3)
        anchor.controller_position = anchor.previous_position.copy()
        anchor.controller_rotation = anchor.previous_rotation.copy()
        anchor.tcp_position = rotation.T @ (self.sim.data.xpos[arm.tcp] - self.sim.data.xpos[root])
        anchor.tcp_rotation = rotation.T @ self.sim.data.xmat[arm.tcp].reshape(3, 3)
        anchor.filtered_position = anchor.tcp_position.copy()
        anchor.filtered_rotation = anchor.tcp_rotation.copy()

    def set_positions(self, data, positions):
        if set(positions) != set(KEYS) or not all(np.isfinite(v) for v in positions.values()):
            raise ValueError("Incomplete or nonfinite real joint state")
        for side, arm in self.sim.arms.items():
            data.qpos[arm.qpos] = np.deg2rad([positions[f"{side}_joint_{i}.pos"] for i in range(1, 8)])
            # Finger travel is a visualization/IK conversion only. The driver
            # uses real gripper degrees [-65,0], never HEI's -4.5 rad endpoints.
            ratio = np.clip(-positions[f"{side}_gripper.pos"] / 65, 0, 1)
            bounds = self.sim.model.jnt_range[arm.finger_joints]
            data.qpos[arm.fingers] = bounds[:, 0] + ratio * (bounds[:, 1] - bounds[:, 0])
        mujoco.mj_forward(self.sim.model, data)

    def configure_limits(self, limits):
        self.inner_limits = operating_ranges(limits, self.margin)
        for side, arm in self.sim.arms.items():
            ranges = np.deg2rad([self.inner_limits[f"{side}_joint_{i}.pos"] for i in range(1, 8)])
            lower = np.maximum(arm.ik.lower, ranges[:, 0])
            upper = np.minimum(arm.ik.upper, ranges[:, 1])
            if np.any(lower >= upper):
                raise ValueError("Physical and IK joint limits do not intersect")
            arm.ik.lower[:] = lower
            arm.ik.upper[:] = upper
            for index, (lo, hi) in enumerate(zip(lower, upper, strict=True), 1):
                self.inner_limits[f"{side}_joint_{index}.pos"] = (
                    float(np.rad2deg(lo)),
                    float(np.rad2deg(hi)),
                )
        self.limits = limits

    def step(self, state, dt):
        if self.limits is None:
            self.configure_limits(state["limits"])
        if state["epoch"] != self.epoch or state.get("blocked", False):
            self.sim.clear_vr()
            self.unreachable.clear()
            self.epoch = state["epoch"]
            self.reset_button_armed = False
        if state.get("resetting", False):
            self.sim.clear_vr()
            self.unreachable.clear()
        # Seed IK from the last applied motor target, preserving the PD holding
        # offset. Replacing that target with lagging measured q would cause sag
        # at the handover. Actual measured joints drive the separate viewer.
        self.set_positions(self.sim.data, state["command"])
        self.display.qpos[:] = self.sim.data.qpos
        self.set_positions(self.display, state["positions"])
        controllers = self.receiver.snapshot()
        mobile_commands = self.mobile.step(controllers, dt)
        mobile_suppressed = self.mobile.suppressed_arms
        reset_safe = all(c.valid and not c.grip for c in controllers.values())
        if not all(c.valid for c in controllers.values()):
            self.reset_button_armed = False
        elif not controllers["left"].primary:
            self.reset_button_armed = True
        else:
            if (
                self.reset_button_armed
                and reset_safe
                and not state.get("resetting", False)
                and not state.get("blocked", False)
            ):
                self.reset_id += 1
            self.reset_button_armed = False
        active = {}
        reasons = {}
        for side in SIDES:
            arm = self.sim.arms[side]
            current = self.sim.data.qpos[arm.qpos].copy()
            bounds = np.deg2rad([self.inner_limits[f"{side}_joint_{i}.pos"] for i in range(1, 8)])
            # An already held startup angle may be outside the interior margin.
            # Hold it continuously, while allowing only inward progress.
            arm.ik.lower[:] = np.minimum(bounds[:, 0], current)
            arm.ik.upper[:] = np.maximum(bounds[:, 1], current)
            recovery = state.get("recovery_directions", {})
            for index in range(7):
                direction = recovery.get(f"{side}_joint_{index + 1}.pos")
                if direction == -1:
                    arm.ik.lower[index] = current[index]
                elif direction == 1:
                    arm.ik.upper[index] = current[index]
            # Slowdown can still reach the target: do not erase its remaining
            # travel. Rebase only when outward travel is blocked or recovering.
            blocked = set(state.get("blocked_joints", ())) | recovery.keys()
            if side in self.unreachable or any(key.startswith(side + "_") for key in blocked):
                self.rebase_anchor(side)
            if side in mobile_suppressed:
                # Joystick commands have priority over incidental hand-pose
                # motion. Hold and rebase so returning the stick to center
                # resumes arm control without a jump.
                self.rebase_anchor(side)
                result = None
            else:
                result = self.sim.step_arm(side, controllers[side], min(0.05, max(0, dt)))
            self.unreachable.discard(side)
            if result is not None:
                # An ordinary joint-rate limit must retain a reachable target
                # for catch-up. A failed/stalled constrained solve must not.
                step = self.sim.args.joint_speed_rad_s * min(0.05, max(0, dt))
                rate_limited = np.max(np.abs(result.q - current)) >= 0.9 * step
                if not result.accepted or (not rate_limited and result.residual > 0.0005):
                    self.unreachable.add(side)
            c = controllers[side]
            active[side] = bool(
                not state.get("blocked", False)
                and not state.get("resetting", False)
                and side not in mobile_suppressed
                and c.valid
                and c.grip
                and self.sim.armed[side]
            )
            reasons[side] = (
                "limit_hold"
                if state.get("blocked", False)
                else "mobile_control"
                if side in mobile_suppressed
                else "tracking_lost_or_stale"
                if not c.valid
                else "grip_released"
                if not c.grip
                else "following"
                if self.sim.armed[side]
                else "release_grip_required"
            )
        # Only the motor process executes the explicit dual-arm X reset.
        # Simulator homing, A/B/Y and mobile controls cannot write real targets.
        positions = dict(state["command"])
        for side, arm in self.sim.arms.items():
            if not active[side]:
                continue
            for index, value in enumerate(np.rad2deg(self.sim.data.qpos[arm.qpos]), 1):
                key = f"{side}_joint_{index}.pos"
                positions[key] = float(np.clip(value, *state["limits"][key]))
            # Send the full endpoint, independently of arm IK and finger slide
            # increments. Motor-side gripper slew controls the opening duration.
            positions[f"{side}_gripper.pos"] = -65.0 if controllers[side].trigger else 0.0
        return {
            "type": "target",
            "epoch": self.epoch,
            "state_time": state["state_time"],
            "sent_at": time.monotonic(),
            "positions": positions,
            "active": active,
            "reason": reasons,
            "reset_id": self.reset_id,
            "reset_safe": reset_safe,
            "mobile_frame": list(mobile_commands.encode()),
            "vr_time": {
                side: self.receiver.received_at[side] if np.isfinite(self.receiver.received_at[side]) else -1
                for side in SIDES
            },
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fd", type=int, required=True)
    parser.add_argument("--vr-endpoint", default="tcp://127.0.0.1:5567")
    parser.add_argument("--vr-position-scale", type=float, default=1.0)
    parser.add_argument("--joint-speed-deg-s", type=float, default=30)
    parser.add_argument("--arm-linear-speed-m-s", type=float, default=0.2)
    parser.add_argument("--arm-angular-speed-deg-s", type=float, default=60)
    parser.add_argument("--vr-filter-scale", type=float, default=0.65)
    parser.add_argument("--joint-limit-margin-deg", type=float, default=3)
    parser.add_argument("--ik-j2-motion-weight", type=float, default=DEFAULT_J2_MOTION_WEIGHT)
    parser.add_argument("--mobile-speed", type=float, default=0.05)
    parser.add_argument("--mobile-spin", type=float, default=0.005)
    parser.add_argument("--lift-speed", type=float, default=0.05)
    for name in ("drive", "steer", "spin", "lift"):
        parser.add_argument(
            f"--{name}-sign",
            type=int,
            choices=(-1, 1),
            default=-1 if name == "drive" else 1,
        )
    parser.add_argument("--viewer", action="store_true")
    args = parser.parse_args()
    sock = socket.socket(fileno=args.fd)
    sock.setblocking(False)
    receiver = ZMQReceiver(args.vr_endpoint, "vr_data", 0.3)
    viewer = None
    try:
        controller = RealArmTargets(
            receiver,
            position_scale=args.vr_position_scale,
            joint_speed_deg_s=args.joint_speed_deg_s,
            linear_speed=args.arm_linear_speed_m_s,
            angular_speed=args.arm_angular_speed_deg_s,
            margin=args.joint_limit_margin_deg,
            j2_motion_weight=args.ik_j2_motion_weight,
            filter_scale=args.vr_filter_scale,
            mobile_speed=args.mobile_speed,
            mobile_spin=args.mobile_spin,
            lift_speed=args.lift_speed,
            **{
                f"{name}_sign": getattr(args, f"{name}_sign")
                for name in ("drive", "steer", "spin", "lift")
            },
        )
        if args.viewer:
            import mujoco.viewer

            viewer = mujoco.viewer.launch_passive(controller.sim.model, controller.display)
        send_packet(sock, {"type": "ready"})
        last = time.monotonic()
        while viewer is None or viewer.is_running():
            receiver.poll()
            state = latest_packet(sock)
            now = time.monotonic()
            if state and state.get("type") == "state" and 0 <= now - state["state_time"] <= 0.2:
                packet = controller.step(state, now - last)
                send_packet(sock, packet)
                last = now
                if viewer is not None:
                    viewer.sync()
            elif now - last > 0.2:
                controller.sim.clear_vr()
            time.sleep(0.002)
        send_packet(sock, {"type": "stop"})
    finally:
        if viewer is not None:
            viewer.close()
        receiver.close()
        sock.close()


if __name__ == "__main__":
    main()

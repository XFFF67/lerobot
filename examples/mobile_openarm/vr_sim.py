#!/usr/bin/env python3
"""Quest / Telegrip control of the existing mobile OpenArm simulation only."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from keyboard_sim import KeyboardSimulator, parse_args, rotation_matrix, rotation_vector, run_viewer
from vr_input import Controller, VRInput, ZMQReceiver, deadzone


@dataclass
class Anchor:
    controller_position: np.ndarray
    controller_rotation: np.ndarray
    tcp_position: np.ndarray
    tcp_rotation: np.ndarray
    previous_position: np.ndarray
    previous_rotation: np.ndarray


def controller_rotation(controller: Controller) -> np.ndarray:
    matrix = np.empty(9)
    mujoco.mju_quat2Mat(matrix, controller.quaternion)
    return matrix.reshape(3, 3)


def add_vr_args(parser):
    parser.description = __doc__
    parser.set_defaults(chassis_physics=True)
    parser.add_argument("--no-chassis-physics", dest="chassis_physics", action="store_false")
    parser.add_argument("--vr-endpoint", default="tcp://127.0.0.1:5567")
    parser.add_argument("--vr-topic", default="vr_data", help="Empty string accepts raw JSON too.")
    parser.add_argument("--vr-timeout-s", type=float, default=0.3)
    parser.add_argument("--vr-position-scale", type=float, default=1.0)
    parser.add_argument("--vr-deadzone", type=float, default=0.15)
    parser.add_argument(
        "--vr-forward-yaw-deg",
        type=int,
        default=180,
        help="VR forward in base_link XY; 180 matches keyboard W (-X).",
    )
    parser.add_argument("--vr-jump-m", type=float, default=0.25)
    parser.add_argument("--vr-jump-deg", type=float, default=75)


HELP = """VR: release both grips once after connection, then hold a grip to control that arm.
Trigger while gripping: open; release trigger: close. Release grip: hold.
Right grip + right stick up/down: drive; right stick left/right is unused.
Right grip + left stick left/right (left grip released): steer wheels.
Left grip + left stick up/down: lift; disables wheel steering from that stick.
Right grip + B/Y: clockwise/counterclockwise spin (automatic wheel mode switch).
Right grip + right stick click: return to translation and straighten wheels.
Grip released + X/A: slowly home left/right arm.
Space: stop and require re-grip | Backspace: reset scene | Esc: quit | V/C: frames/contacts
Observe the robot in the desktop MuJoCo window; no camera stream is sent to the headset."""


class VRSimulator(KeyboardSimulator):
    help_text = HELP

    def __init__(self, args, receiver: VRInput):
        self.receiver = receiver
        self.anchors = {}
        self.armed = dict.fromkeys(("left", "right"), False)
        self.previous_buttons = dict.fromkeys(self.armed, (False, False))
        self.chassis_command = (0.0, 0.0, 0.0)
        self.chassis_braking = True
        self.last_tracking = None
        # WebXR: +X right, +Y up, -Z forward. Map to robot forward/left/up,
        # then rotate to this model's forward (-X); determinant remains +1.
        xr_to_forward = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], dtype=float)
        self.mapping = rotation_matrix(np.array([0, 0, np.deg2rad(args.vr_forward_yaw_deg)])) @ xr_to_forward
        super().__init__(args)
        print(f"[VR] Waiting for {args.vr_endpoint}; timeout {args.vr_timeout_s}s", flush=True)

    def clear_vr(self):
        self.anchors.clear()
        for side in self.armed:
            self.armed[side] = False
            self.previous_buttons[side] = (False, False)
        self.chassis_command = (0.0, 0.0, 0.0)
        self.chassis_braking = True

    def reset(self):
        super().reset()
        self.clear_vr()

    def stop(self):
        super().stop()
        self.clear_vr()

    def press(self, key: str):
        if key in {"space", "backspace", "escape", "v", "c"}:
            super().press(key)

    def advance_physics(self, dt, *, braking=False):
        previous = self.chassis.last_status
        linear, steering, yaw = self.chassis_command
        self.chassis.advance(
            self.data, dt, linear=linear, steering=steering, yaw=yaw, braking=braking or self.chassis_braking
        )
        if previous != self.chassis.last_status:
            print(f"[VR] {self.chassis.last_status}", flush=True)

    def step_arm(self, side: str, controller: Controller, dt: float):
        arm = self.arms[side]
        if not controller.valid:
            self.armed[side] = False
            arm.resetting = False
        if not controller.valid or not controller.grip:
            self.anchors.pop(side, None)
            if controller.valid:
                self.armed[side] = True
            return
        if not self.armed[side]:
            return
        arm.resetting = False
        rotation = controller_rotation(controller)
        root = arm.ik.root_body
        root_rotation = self.data.xmat[root].reshape(3, 3)
        root_position = self.data.xpos[root]
        # Keep arm targets in shoulder coordinates: moving the lift/chassis
        # carries the arms rather than driving them backwards to a world anchor.
        if side not in self.anchors:
            self.anchors[side] = Anchor(
                controller.position.copy(),
                rotation.copy(),
                root_rotation.T @ (self.data.xpos[arm.tcp] - root_position),
                root_rotation.T @ self.data.xmat[arm.tcp].reshape(3, 3),
                controller.position.copy(),
                rotation.copy(),
            )
        anchor = self.anchors[side]
        jump = np.linalg.norm(controller.position - anchor.previous_position)
        angle = np.linalg.norm(rotation_vector(rotation @ anchor.previous_rotation.T))
        if jump > self.args.vr_jump_m or angle > np.deg2rad(self.args.vr_jump_deg):
            self.anchors.pop(side)
            self.armed[side] = False
            print(f"[VR] {side} tracking jumped; release and re-grip", flush=True)
            return
        anchor.previous_position = controller.position.copy()
        anchor.previous_rotation = rotation.copy()
        # Convert base-frame motions to the current shoulder axes.
        mapping = root_rotation.T @ self.data.xmat[self.base].reshape(3, 3) @ self.mapping
        offset = mapping @ (controller.position - anchor.controller_position) * self.args.vr_position_scale
        position = root_position + root_rotation @ (anchor.tcp_position + offset)
        delta_rotation = mapping @ rotation @ anchor.controller_rotation.T @ mapping.T
        orientation = root_rotation @ delta_rotation @ anchor.tcp_rotation
        # Bound Cartesian and joint steps even for large but continuous moves.
        error = position - self.data.xpos[arm.tcp]
        limit = self.args.arm_linear_speed_m_s * dt
        position = self.data.xpos[arm.tcp] + error * min(1, limit / max(np.linalg.norm(error), 1e-12))
        current_rotation = self.data.xmat[arm.tcp].reshape(3, 3)
        rot_error = rotation_vector(orientation @ current_rotation.T)
        rot_limit = np.deg2rad(self.args.arm_angular_speed_deg_s) * dt
        orientation = (
            rotation_matrix(rot_error * min(1, rot_limit / max(np.linalg.norm(rot_error), 1e-12)))
            @ current_rotation
        )
        result = arm.ik.solve(self.data, position, orientation, self.args.joint_speed_rad_s * dt)
        if result.accepted:
            self.data.qpos[arm.qpos] = result.q
        limits = self.model.jnt_range[arm.finger_joints]
        target = limits[:, 1] if controller.trigger else limits[:, 0]
        step = self.args.gripper_speed_m_s * dt
        self.data.qpos[arm.fingers] += np.clip(target - self.data.qpos[arm.fingers], -step, step)
        self.forward()

    def update_chassis(self, controllers):
        right, left = controllers["right"], controllers["left"]
        self.chassis_braking = not (right.valid and right.grip and self.armed["right"])
        self.chassis_command = (0.0, 0.0, 0.0)
        if self.chassis is None or self.chassis_braking:
            return
        yaw = float((left.valid and left.secondary) - right.secondary)
        linear = deadzone(right.stick[1], self.args.vr_deadzone)
        # Put wheel heading and travel on separate hands. Left grip selects
        # lift control, so diagonal lift inputs cannot also steer the wheels.
        steering = (
            deadzone(left.stick[0], self.args.vr_deadzone)
            if left.valid and not left.grip and self.armed["left"]
            else 0.0
        )
        click = right.stick_pressed and not self.previous_buttons["right"][1]
        requested = "spin" if yaw else "translation"
        active = bool(yaw or linear or steering or click)
        if active and (self.chassis.target_mode != requested or self.chassis.cancelled or click):
            self.chassis.request_mode(requested, self.data)
        # In translation the left stick changes retained heading, as with A/D;
        # it does not pretend this passive chassis is a holonomic mecanum base.
        self.chassis_command = (0.0, 0.0, yaw) if yaw else (linear, steering, 0.0)

    def step(self, dt: float):
        dt = float(np.clip(dt, 0, 0.05))
        if hasattr(self.receiver, "poll"):
            self.receiver.poll()
        controllers = self.receiver.snapshot()
        tracking = tuple(c.valid for c in controllers.values())
        if tracking != self.last_tracking:
            print(f"[VR] tracking left={tracking[0]} right={tracking[1]}; release grip to arm", flush=True)
            self.last_tracking = tracking
        if dt == 0:
            return
        if "space" in self.keys:
            self.stop()
            self.keys.add("space")
            if self.chassis is not None:
                self.advance_physics(dt, braking=True)
            return
        if self.chassis is not None:
            actual = self.data.qpos.copy()
            self.data.qpos[self.chassis.upper_qpos] = self.chassis.upper_target
            self.forward()
        for side, controller in controllers.items():
            arm = self.arms[side]
            if (
                self.armed[side]
                and controller.valid
                and not controller.grip
                and controller.primary
                and not self.previous_buttons[side][0]
            ):
                arm.resetting = True
            self.step_arm(side, controller, dt)
            if arm.resetting:
                error = self.home[arm.qpos] - self.data.qpos[arm.qpos]
                step = self.args.reset_speed_rad_s * dt
                self.data.qpos[arm.qpos] += np.clip(error, -step, step)
                arm.resetting = np.max(np.abs(error)) > step
                self.forward()
        left = controllers["left"]
        lift_axis = -deadzone(left.stick[1], self.args.vr_deadzone)
        if left.valid and left.grip and self.armed["left"]:
            self.lift.step(self.data, lift_axis, self.args.lift_speed_m_s, dt)
            self.forward()
        self.update_chassis(controllers)
        for side, controller in controllers.items():
            self.previous_buttons[side] = (controller.primary, controller.stick_pressed)
        if self.chassis is not None:
            self.chassis.upper_target = self.data.qpos[self.chassis.upper_qpos].copy()
            self.data.qpos[:] = actual
            self.advance_physics(dt)
        else:
            self.data.time += dt
            self.forward()


def main():
    args = parse_args(configure=add_vr_args)
    if not 0 < args.vr_deadzone < 1:
        raise SystemExit("--vr-deadzone must be between 0 and 1")
    if args.headless_grasp_check:
        raise SystemExit("Use run_keyboard_sim.sh --headless-grasp-check for the physical grasp regression")
    if args.headless_check:
        from check_vr_sim import check_simulator

        check_simulator(VRSimulator(args, VRInput(args.vr_timeout_s)))
        return
    receiver = ZMQReceiver(args.vr_endpoint, args.vr_topic, args.vr_timeout_s)
    try:
        run_viewer(VRSimulator(args, receiver))
    finally:
        receiver.close()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Window-local keyboard control of the mjcf1 OpenArm mobile robot.

Default: arm/finger and torso kinematic preview. --chassis-physics enables
experimental wheel/contact dynamics and target servos for the upper body.
No robot communication is opened.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import mujoco
import numpy as np

if TYPE_CHECKING:
    from pinocchio_ik import PinocchioCasadiIK

MODES = {"1": "chassis", "2": "torso", "3": "left", "4": "right"}
# Mesh estimate: the circular gripper mounting face is centred at hand X=0.001 mm,
# Y=5 mm, with its axis along hand Z. Choose the point on that axis at hand Z=0.
# This is not a measured motor-shaft origin; override it with calibrated CAD data.
GRIPPER_MOTOR_OFFSET_M = (0.000001, 0.005, 0.0)


@dataclass
class Arm:
    joints: np.ndarray
    qpos: np.ndarray
    dofs: np.ndarray
    tcp: int
    fingers: np.ndarray
    finger_joints: np.ndarray
    resetting: bool = False
    ik: PinocchioCasadiIK | None = None


def rotation_vector(matrix: np.ndarray) -> np.ndarray:
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, np.ascontiguousarray(matrix).reshape(9))
    if quat[0] < 0:
        quat *= -1
    result = np.empty(3)
    mujoco.mju_quat2Vel(result, quat, 1.0)
    return result


def rotation_matrix(vector: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(vector))
    if angle < 1e-12:
        return np.eye(3)
    quat = np.empty(4)
    mujoco.mju_axisAngle2Quat(quat, vector / angle, angle)
    result = np.empty(9)
    mujoco.mju_quat2Mat(result, quat)
    return result.reshape(3, 3)


class KeyboardSimulator:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        from mujoco_scene import build_mujoco_model

        self.chassis = None
        chassis_config = None
        if args.chassis_physics:
            from chassis_physics import ChassisConfig

            chassis_config = ChassisConfig(
                tire_friction=args.tire_friction,
                motor_torque=args.wheel_torque_nm,
                magnet_force=args.magnet_force_n,
                magnet_distance=args.magnet_distance_m,
            )
        # Loading by absolute path preserves model-relative includes and meshes.
        self.scene_enabled = not args.plain_scene
        self.model = build_mujoco_model(
            args.mjcf, add_environment=self.scene_enabled, chassis_config=chassis_config
        )
        self.data = mujoco.MjData(self.model)
        self.arms = {}
        for side in ("left", "right"):
            joints = np.array([self.id("joint", f"openarm_{side}_joint{i}") for i in range(1, 8)])
            fingers = np.array([self.id("joint", f"openarm_{side}_finger_joint{i}") for i in (1, 2)])
            if np.any(self.model.jnt_type[joints] != mujoco.mjtJoint.mjJNT_HINGE):
                raise ValueError(f"{side} arm must contain seven hinge joints")
            if np.any(self.model.jnt_type[fingers] != mujoco.mjtJoint.mjJNT_SLIDE):
                raise ValueError(f"{side} fingers must be slide joints")
            tcp = self.id("body", f"openarm_{side}_hand_tcp")
            if args.tcp_reference == "gripper-motor":
                hand = self.id("body", f"openarm_{side}_hand")
                # Reposition only the existing massless, fixed TCP leaf in memory.
                # FK, IK Jacobians, and the viewer then share the same reference.
                if (
                    self.model.body_parentid[tcp] != hand
                    or self.model.body_jntnum[tcp] != 0
                    or self.model.body_mass[tcp] != 0
                    or self.model.body_geomnum[tcp] != 0
                    or np.any(self.model.body_parentid == tcp)
                ):
                    raise ValueError(f"{side} hand_tcp must be an empty fixed child of hand")
                self.model.body_pos[tcp] = args.gripper_motor_offset_m
                self.model.body_quat[tcp] = [1, 0, 0, 0]
            self.arms[side] = Arm(
                joints,
                self.model.jnt_qposadr[joints],
                self.model.jnt_dofadr[joints],
                tcp,
                self.model.jnt_qposadr[fingers],
                fingers,
            )
        self.base = self.id("body", "base_link")
        self.keys: set[str] = set()
        self.mode = "left"
        self.joint_mode = False
        self.selected_joint = 0
        self.show_frames = False
        self.show_contacts = False
        self.exit_requested = False
        self.message = "Kinematic preview; chassis reserved"
        self.reset()
        self.home = self.data.qpos.copy()
        from pinocchio_ik import PinocchioCasadiIK
        from torso_lift import TorsoLift

        self.lift = TorsoLift(self.model, self.data)
        if chassis_config is not None:
            from chassis_physics import ChassisPhysics

            self.chassis = ChassisPhysics(self.model, self.data, chassis_config)
            print("[Keyboard] Chassis physics: 1 | T: W/S drive, A/D steer (-90..90 deg) | G: A/D spin")
        print(
            f"[Keyboard] Lift: 2 then I/K up/down; travel {self.lift.max_height - self.lift.min_height:.3f} m; "
            f"simulation hinge alignment {self.lift.axis_correction_deg:.3f} deg",
            flush=True,
        )

        for arm in self.arms.values():
            arm.ik = PinocchioCasadiIK(
                self.model, self.data, arm.joints, arm.tcp, smooth_weight=args.ik_smooth_weight
            )
        print("[Keyboard] IK: Pinocchio + CasADi / IPOPT (7 joints per arm)", flush=True)
        self.tcp_description = (
            f"gripper motor axis (estimate), hand offset {tuple(args.gripper_motor_offset_m)} m"
            if args.tcp_reference == "gripper-motor"
            else "original model hand_tcp"
        )
        print(f"[Keyboard] TCP: {self.tcp_description}", flush=True)
        print(
            f"[Keyboard] Scene: {'HEI-style table and objects' if self.scene_enabled else 'robot only'}",
            flush=True,
        )
        if self.chassis is not None and self.scene_enabled:
            print("[Keyboard] Grasp physics: table/object/arm contacts | Z open, X close; release holds grip")

    def id(self, kind: str, name: str) -> int:
        object_type = getattr(mujoco.mjtObj, f"mjOBJ_{kind.upper()}")
        index = mujoco.mj_name2id(self.model, object_type, name)
        if index < 0:
            raise ValueError(f"Missing {kind} {name!r}; this profile expects the mjcf1 OpenArm model")
        return index

    def forward(self):
        if self.chassis is None:
            self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)

    def reset(self):
        key = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if key >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, key)
        else:
            mujoco.mj_resetData(self.model, self.data)
        self.keys.clear()
        for arm in self.arms.values():
            arm.resetting = False
        self.mode = "left"
        self.joint_mode = False
        self.selected_joint = 0
        self.message = "Home restored; kinematic preview"
        if self.chassis is not None:
            self.chassis.reset(self.data)
            self.message = "Home restored; chassis physics"
        self.forward()

    def stop(self):
        self.keys.clear()
        if self.chassis is not None:
            self.chassis.stop()
            self.chassis.upper_target = self.data.qpos[self.chassis.upper_qpos].copy()
        for arm in self.arms.values():
            arm.resetting = False

    def press(self, key: str):
        if key in self.keys:
            return
        # Preserve a held stop across mode/reset key transitions.
        stopping = "space" in self.keys
        if key in MODES:
            if self.chassis is not None:
                self.chassis.stop()
            self.keys.clear()
            self.mode = MODES[key]
            if self.mode in self.arms:
                self.joint_mode = False
            if self.mode == "chassis":
                self.message = (
                    "chassis: T translation (W/S drive, A/D steer) / G spin (A/D)"
                    if self.chassis is not None
                    else "chassis: enable --chassis-physics for wheel/contact dynamics"
                )
            elif self.mode == "torso":
                self.message = "lift: hold I to raise / K to lower; Shift for fine control"
            else:
                self.message = f"{self.mode} arm | TCP: hold W/S A/D R/F to move"
            print(f"[Keyboard] {self.message}", flush=True)
        elif key in ("t", "g") and self.mode == "chassis" and self.chassis is not None:
            if not stopping:
                self.keys.clear()
                self.chassis.request_mode("translation" if key == "t" else "spin", self.data)
                print(f"[Keyboard] {self.chassis.last_status}", flush=True)
        elif key in ("5", "6"):
            self.keys.clear()
            self.arms["left" if key == "5" else "right"].resetting = True
        elif key == "space":
            self.stop()
        elif key == "backspace":
            self.reset()
        elif key == "escape":
            self.exit_requested = True
        elif key == "v":
            self.show_frames = not self.show_frames
        elif key == "c":
            self.show_contacts = not self.show_contacts
        elif key == "tab":
            self.keys.clear()
            self.joint_mode = not self.joint_mode
            self.message = (
                "JOINT: select with [ / ], hold - / = to move"
                if self.joint_mode
                else "TCP: hold W/S A/D R/F to move"
            )
            print(f"[Keyboard] {self.mode} | {self.message}", flush=True)
        elif key in ("[", "]"):
            self.selected_joint = (self.selected_joint + (1 if key == "]" else -1)) % 7
        self.keys.add(key)
        if stopping:
            self.keys.add("space")

    def release(self, key: str):
        self.keys.discard(key)

    @staticmethod
    def axis(keys: set[str], positive: str, negative: str) -> float:
        return float((positive in keys) - (negative in keys))

    def axes(self, pairs: tuple[tuple[str, str], ...]) -> np.ndarray:
        vector = np.array([self.axis(self.keys, *pair) for pair in pairs])
        return vector / max(1.0, float(np.linalg.norm(vector)))

    def clip(self, values: np.ndarray, joints: np.ndarray) -> np.ndarray:
        bounds = self.model.jnt_range[joints]
        limited = self.model.jnt_limited[joints].astype(bool)
        return np.where(limited, np.clip(values, bounds[:, 0], bounds[:, 1]), values)

    def step_chassis(self, dt: float):
        """Wheel forces are integrated after upper-body targets have been updated."""

    def advance_physics(self, dt, *, braking=False):
        chassis = self.chassis
        previous_status = chassis.last_status
        selected = self.mode == "chassis"
        chassis.advance(
            self.data,
            dt,
            linear=self.axis(self.keys, "s", "w") if selected else 0.0,
            steering=self.axis(self.keys, "d", "a") if selected else 0.0,
            yaw=self.axis(self.keys, "a", "d") if selected else 0.0,
            scale=self.args.fine_scale if "shift" in self.keys else 1.0,
            braking=braking,
        )
        if chassis.last_status != previous_status:
            print(
                f"[Keyboard] {chassis.last_status}; encoders {chassis.encoder_degrees(self.data).round(2)} deg",
                flush=True,
            )

    def step_torso(self, dt: float):
        """HEI keyboard mapping with the mjcf1 closed-chain lift mechanism."""
        direction = self.axis(self.keys, "i", "k")
        scale = self.args.fine_scale if "shift" in self.keys else 1.0
        self.lift.step(self.data, direction, self.args.lift_speed_m_s * scale, dt)
        self.message = f"Lift: {self.lift.last_status}"

    def move_tcp(self, arm: Arm, linear: np.ndarray, angular: np.ndarray, dt: float):
        # Each target is relative to measured FK, so unreachable commands cannot
        # accumulate a distant target that keeps moving after key release.
        base_rotation = self.data.xmat[self.base].reshape(3, 3)
        position = self.data.xpos[arm.tcp] + base_rotation @ linear * dt
        orientation = rotation_matrix(base_rotation @ angular * dt) @ self.data.xmat[arm.tcp].reshape(3, 3)
        result = arm.ik.solve(self.data, position, orientation, self.args.joint_speed_rad_s * dt)
        if result.accepted:
            self.data.qpos[arm.qpos] = result.q
        self.message = f"Pinocchio/CasADi TCP residual {result.residual:.5f}; {result.status}"

    def step(self, dt: float):
        dt = float(np.clip(dt, 0, 0.05))
        if dt == 0:
            return
        if "space" in self.keys:
            for arm in self.arms.values():
                arm.resetting = False
            if self.chassis is not None:
                self.advance_physics(dt, braking=True)
            return
        if self.chassis is not None:
            # Existing IK/lift code edits a temporary target pose. Restore the
            # actual pose before mj_step; chassis coordinates are never assigned.
            actual_qpos = self.data.qpos.copy()
            self.data.qpos[self.chassis.upper_qpos] = self.chassis.upper_target
            self.forward()
        scale = self.args.fine_scale if "shift" in self.keys else 1.0
        if self.mode in self.arms:
            arm = self.arms[self.mode]
            linear = self.axes((("s", "w"), ("d", "a"), ("r", "f")))
            angular = self.axes((("u", "j"), ("i", "k"), ("o", "l")))
            jog = self.axis(self.keys, "=", "-")
            moving = bool(jog) if self.joint_mode else bool(np.any(linear) or np.any(angular))
            if moving:
                arm.resetting = False
                if self.joint_mode:
                    values = self.data.qpos[arm.qpos].copy()
                    values[self.selected_joint] += jog * self.args.joint_speed_rad_s * scale * dt
                    self.data.qpos[arm.qpos] = self.clip(values, arm.joints)
                else:
                    self.move_tcp(
                        arm,
                        linear * self.args.arm_linear_speed_m_s * scale,
                        angular * np.deg2rad(self.args.arm_angular_speed_deg_s) * scale,
                        dt,
                    )
            if "z" in self.keys or "x" in self.keys:
                limits = self.model.jnt_range[arm.finger_joints]
                target = limits[:, 1] if "z" in self.keys else limits[:, 0]
                current = self.data.qpos[arm.fingers]
                max_step = self.args.gripper_speed_m_s * scale * dt
                self.data.qpos[arm.fingers] = current + np.clip(target - current, -max_step, max_step)
        elif self.mode == "chassis":
            self.step_chassis(dt)
        else:
            self.step_torso(dt)
        for arm in self.arms.values():
            if arm.resetting:
                error = self.home[arm.qpos] - self.data.qpos[arm.qpos]
                max_step = self.args.reset_speed_rad_s * scale * dt
                self.data.qpos[arm.qpos] += np.clip(error, -max_step, max_step)
                if np.max(np.abs(error)) <= max_step:
                    arm.resetting = False
        if self.chassis is not None:
            self.chassis.upper_target = self.data.qpos[self.chassis.upper_qpos].copy()
            self.data.qpos[:] = actual_qpos
            self.advance_physics(dt)
        else:
            self.data.time += dt
            self.forward()


HELP = """1 chassis (requires --chassis-physics) | 2 lift (I/K up/down) | 3 left | 4 right
W/S -X/+X | A/D -Y/+Y | R/F +Z/-Z | U/J Rx | I/K Ry | O/L Rz
Z/X open/close (hold) | Shift fine | Space stop
5/6 reset left/right | Backspace full reset | Esc quit
Tab TCP/joint mode | [/] select joint | -/= jog joint
V body frames | C contacts | mouse: left orbit, right pan, wheel zoom"""


def draw_tcp_frames(sim: KeyboardSimulator, scene: mujoco.MjvScene):
    """Mark the controlled points and their local XYZ axes, independent of V."""
    for arm in sim.arms.values():
        if scene.ngeom + 4 > scene.maxgeom:
            return
        position = sim.data.xpos[arm.tcp]
        rotation = sim.data.xmat[arm.tcp].reshape(3, 3)
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.full(3, 0.004),
            position,
            np.eye(3).ravel(),
            np.array([1, 0.85, 0, 1], dtype=np.float32),
        )
        geom.label = ""
        scene.ngeom += 1
        for axis in range(3):
            geom = scene.geoms[scene.ngeom]
            color = np.zeros(4, dtype=np.float32)
            color[axis] = color[3] = 1
            mujoco.mjv_initGeom(
                geom,
                mujoco.mjtGeom.mjGEOM_ARROW,
                np.zeros(3),
                position,
                np.eye(3).ravel(),
                color,
            )
            mujoco.mjv_connector(
                geom,
                mujoco.mjtGeom.mjGEOM_ARROW,
                0.002,
                position,
                position + 0.04 * rotation[:, axis],
            )
            scene.ngeom += 1


def run_viewer(sim: KeyboardSimulator):
    # GLFW provides press AND release events scoped to this window. No global
    # pynput hook is needed; losing window focus clears held keys and resets.
    import glfw
    from mujoco_scene import configure_camera

    if not glfw.init():
        raise RuntimeError("Cannot initialize GLFW; use a graphical desktop or --headless-check")
    title = (
        "Mobile dual arm - chassis physics"
        if sim.chassis is not None
        else "Mobile dual arm - kinematic preview"
    )
    window = glfw.create_window(1280, 900, title, None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("Cannot create MuJoCo window")
    context = None
    try:
        glfw.make_context_current(window)
        glfw.swap_interval(1)
        context = mujoco.MjrContext(sim.model, mujoco.mjtFontScale.mjFONTSCALE_150)
        scene = mujoco.MjvScene(sim.model, maxgeom=10000)
        camera = mujoco.MjvCamera()
        configure_camera(sim.model, camera, scene_enabled=sim.scene_enabled)
        option = mujoco.MjvOption()
        # Hide collision proxies and closure markers initially.
        option.geomgroup[3] = 0
        special = {
            glfw.KEY_SPACE: "space",
            glfw.KEY_BACKSPACE: "backspace",
            glfw.KEY_ESCAPE: "escape",
            glfw.KEY_TAB: "tab",
            glfw.KEY_LEFT_SHIFT: "shift",
            glfw.KEY_RIGHT_SHIFT: "shift",
        }

        def key_callback(window, key, scancode, action, mods):
            # Use GLFW physical key tokens, independent of the active input layout.
            name = special.get(key)
            if name is None and 32 <= key <= 96:
                name = chr(key).lower()
            if not name:
                return
            if action == glfw.PRESS:
                sim.press(name.lower())
            elif action == glfw.RELEASE:
                if name == "shift" and any(
                    glfw.get_key(window, k) == glfw.PRESS for k in (glfw.KEY_LEFT_SHIFT, glfw.KEY_RIGHT_SHIFT)
                ):
                    return
                sim.release(name.lower())

        mouse = [0.0, 0.0]

        def cursor_callback(window, x, y):
            dx, dy = x - mouse[0], y - mouse[1]
            mouse[:] = [x, y]
            height = max(glfw.get_window_size(window)[1], 1)
            if glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT):
                action = mujoco.mjtMouse.mjMOUSE_MOVE_V
            elif glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT):
                action = mujoco.mjtMouse.mjMOUSE_ROTATE_V
            else:
                return
            mujoco.mjv_moveCamera(sim.model, action, dx / height, dy / height, scene, camera)

        glfw.set_key_callback(window, key_callback)
        glfw.set_window_focus_callback(window, lambda _w, focused: None if focused else sim.stop())
        glfw.set_cursor_pos_callback(window, cursor_callback)
        glfw.set_scroll_callback(
            window,
            lambda _w, _x, y: mujoco.mjv_moveCamera(
                sim.model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0, 0.05 * y, scene, camera
            ),
        )
        print(getattr(sim, "help_text", HELP), flush=True)
        if sim.chassis is not None:
            print("[Keyboard] Physics: 1 chassis | T: W/S drive, A/D steer | G: A/D spin")
        previous = time.monotonic()
        while not glfw.window_should_close(window) and not sim.exit_requested:
            glfw.poll_events()
            now = time.monotonic()
            sim.step(now - previous)
            previous = now
            option.frame = mujoco.mjtFrame.mjFRAME_BODY if sim.show_frames else mujoco.mjtFrame.mjFRAME_NONE
            option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = sim.show_contacts
            mujoco.mjv_updateScene(
                sim.model, sim.data, option, None, camera, mujoco.mjtCatBit.mjCAT_ALL, scene
            )
            draw_tcp_frames(sim, scene)
            width, height = glfw.get_framebuffer_size(window)
            if width > 0 and height > 0:
                viewport = mujoco.MjrRect(0, 0, width, height)
                mujoco.mjr_render(viewport, scene, context)
                glfw.swap_buffers(window)
            else:
                time.sleep(0.02)
    finally:
        if context is not None:
            context.free()
        glfw.destroy_window(window)
        glfw.terminate()


def parse_args(argv=None, *, configure=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mjcf",
        "--model",
        type=Path,
        default=Path(__file__).resolve().parent / "model" / "mjcf1" / "robot.xml",
        help="MJCF path (default: model/mjcf1/robot.xml next to this script).",
    )
    parser.add_argument(
        "--tcp-reference",
        choices=("gripper-motor", "model"),
        default="gripper-motor",
        help="Control the estimated gripper drive axis or the original model hand_tcp.",
    )
    parser.add_argument(
        "--gripper-motor-offset-m",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=GRIPPER_MOTOR_OFFSET_M,
        help="Motor reference point in the hand frame, metres (default: mesh estimate).",
    )
    parser.add_argument("--arm-linear-speed-m-s", type=float, default=0.12)
    parser.add_argument("--arm-angular-speed-deg-s", type=float, default=35)
    parser.add_argument("--joint-speed-rad-s", type=float, default=1.0)
    parser.add_argument("--reset-speed-rad-s", type=float, default=0.5)
    parser.add_argument("--gripper-speed-m-s", type=float, default=0.04)
    parser.add_argument(
        "--lift-speed-m-s",
        type=float,
        default=0.08,
        help="Torso shoulder-height speed in metres per second (I/K in mode 2).",
    )
    parser.add_argument("--fine-scale", type=float, default=0.25)
    parser.add_argument(
        "--ik-smooth-weight",
        type=float,
        default=1e-4,
        help="CasADi IK joint continuity cost relative to weighted TCP pose error.",
    )
    parser.add_argument("--headless-check", action="store_true")
    parser.add_argument(
        "--headless-grasp-check", action="store_true", help="Test physical grasping without a window."
    )
    parser.add_argument(
        "--chassis-physics", action="store_true", help="Enable experimental wheel/contact dynamics."
    )
    parser.add_argument("--tire-friction", type=float, default=1.0)
    parser.add_argument("--wheel-torque-nm", type=float, default=4.0)
    parser.add_argument("--magnet-force-n", type=float, default=16.0)
    parser.add_argument("--magnet-distance-m", type=float, default=0.01)
    parser.add_argument(
        "--plain-scene", action="store_true", help="Show only the robot without the HEI-style environment."
    )
    if configure is not None:
        configure(parser)
    args = parser.parse_args(argv)
    if args.headless_grasp_check:
        if args.plain_scene:
            parser.error("--headless-grasp-check requires the table scene")
        args.chassis_physics = True
    if not np.all(np.isfinite(args.gripper_motor_offset_m)):
        parser.error("--gripper-motor-offset-m must contain three finite coordinates")
    for name, value in vars(args).items():
        if isinstance(value, float) and (not np.isfinite(value) or value <= 0):
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if args.fine_scale > 1:
        parser.error("--fine-scale must be <= 1")
    return args


def main():
    args = parse_args()
    simulator = KeyboardSimulator(args)
    if args.headless_check or args.headless_grasp_check:
        if args.headless_grasp_check:
            from check_grasp_physics import check_simulator
        elif args.chassis_physics:
            from check_chassis_physics import check_simulator
        else:
            from check_keyboard_sim import check_simulator

        check_simulator(simulator)
    else:
        run_viewer(simulator)


if __name__ == "__main__":
    main()

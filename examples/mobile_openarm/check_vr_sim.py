"""Headless regression with synthetic Quest packets and the actual MJCF/IK."""

import json
import time

import mujoco
import numpy as np
from vr_input import VRInput, deadzone, parse_controller


def controller(*, grip=False, trigger=False):
    return {
        "position": {"x": 0.0, "y": 1.0, "z": 0.0},
        "quaternion": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        "gripActive": grip,
        "trigger": trigger,
        "thumbstick": {"x": 0.0, "y": 0.0, "pressed": False},
    }


def check_protocol():
    source = VRInput(0.3)
    packet = {"leftController": controller(grip=True), "rightController": controller()}
    assert source.ingest("vr_data " + json.dumps(packet), now=1)
    assert source.snapshot(1.1)["left"].grip
    assert not source.snapshot(1.31)["left"].valid
    # Partial updates, button events and junk must not refresh the other hand.
    source.ingest(json.dumps({"rightController": controller()}), now=1.2)
    source.ingest('{"type":"button_press","hand":"left","button":"X"}', now=1.3)
    assert not source.snapshot(1.31)["left"].valid
    assert source.snapshot(1.31)["right"].valid
    source.ingest(json.dumps(packet), now=2)
    source.ingest('{"hand":"left","gripReleased":true}', now=2.1)
    assert not source.snapshot(2.1)["left"].grip
    for raw in ("garbage", "vr_data []", "null", "vr_data NaN", '{"hand": []}'):
        assert not source.ingest(raw, now=3)
    for invalid in (
        None,
        {},
        {**controller(), "tracked": False},
        {**controller(), "position": {"x": float("nan"), "y": 1, "z": 0}},
        {**controller(), "quaternion": {"x": 0, "y": 0, "z": 0, "w": 0}},
        {**controller(), "thumbstick": {"x": float("inf")}},
    ):
        assert not parse_controller(invalid, "left").valid
    assert deadzone(0.1, 0.15) == 0 and deadzone(-1, 0.15) == -1
    print("PASS protocol: Telegrip, invalid data, per-hand timeout, release events, deadzone")


def target_qpos(sim):
    qpos = sim.data.qpos.copy()
    if sim.chassis is not None:
        qpos[sim.chassis.upper_qpos] = sim.chassis.upper_target
    return qpos


def target_pose(sim, side):
    data = mujoco.MjData(sim.model)
    data.qpos[:] = target_qpos(sim)
    mujoco.mj_forward(sim.model, data)
    arm = sim.arms[side]
    root = arm.ik.root_body
    rotation = data.xmat[root].reshape(3, 3)
    return (
        rotation.T @ (data.xpos[arm.tcp] - data.xpos[root]),
        rotation.T @ data.xmat[arm.tcp].reshape(3, 3),
    )


def check_simulator(sim):
    check_protocol()
    dt = 0.02
    packet = {}

    def tick(n=1):
        for _ in range(n):
            before = target_qpos(sim)
            sim.receiver.ingest("vr_data " + json.dumps(packet))
            sim.step(dt)
            after = target_qpos(sim)
            assert np.all(np.isfinite(after))
            for arm in sim.arms.values():
                assert (
                    np.max(np.abs(after[arm.qpos] - before[arm.qpos]))
                    <= sim.args.joint_speed_rad_s * dt + 1e-8
                )
                assert np.all(after[arm.qpos] >= arm.ik.lower - 1e-8)
                assert np.all(after[arm.qpos] <= arm.ik.upper + 1e-8)

    def start():
        sim.reset()
        packet.clear()
        packet.update({f"{s}Controller": controller() for s in sim.arms})
        tick()

    np.testing.assert_allclose(sim.mapping @ [0, 0, -1], [-1, 0, 0], atol=1e-12)
    np.testing.assert_allclose(sim.mapping @ [-1, 0, 0], [0, -1, 0], atol=1e-12)
    np.testing.assert_allclose(sim.mapping @ [0, 1, 0], [0, 0, 1], atol=1e-12)
    assert abs(np.linalg.det(sim.mapping) - 1) < 1e-12
    print("PASS coordinates: forward=-X, left=-Y, up=+Z; proper rotation")

    start()
    for side in sim.arms:
        packet[f"{side}Controller"]["gripActive"] = True
    tick()
    initial = {s: target_pose(sim, s)[0] for s in sim.arms}
    for i in range(40):
        for side in sim.arms:
            packet[f"{side}Controller"]["position"]["y"] = 1 + (i + 1) * 0.001
        tick()
    for side in sim.arms:
        movement = target_pose(sim, side)[0] - initial[side]
        assert np.linalg.norm(movement) > 0.015, (side, movement)
        packet[f"{side}Controller"]["gripActive"] = False
    tick()
    held = target_qpos(sim)
    for side in sim.arms:
        packet[f"{side}Controller"]["position"]["z"] = -2.0
    tick(5)
    for arm in sim.arms.values():
        np.testing.assert_allclose(target_qpos(sim)[arm.qpos], held[arm.qpos], atol=1e-12)
    for side in sim.arms:
        packet[f"{side}Controller"]["gripActive"] = True
    tick()
    for arm in sim.arms.values():
        np.testing.assert_allclose(target_qpos(sim)[arm.qpos], held[arm.qpos], atol=1e-6)
    print("PASS simultaneous dual-arm motion, release hold, clutch without jumps, joint bounds")

    # Quaternion changes must move the TCP orientation, independently of position.
    start()
    right = packet["rightController"]
    right["gripActive"] = True
    tick()
    initial_rotation = target_pose(sim, "right")[1]
    for i in range(40):
        angle = (i + 1) * 0.004
        right["quaternion"] = {"w": np.cos(angle / 2), "x": 0, "y": np.sin(angle / 2), "z": 0}
        tick()
    assert np.linalg.norm(target_pose(sim, "right")[1] - initial_rotation) > 0.05
    right["trigger"] = True
    tick(65)
    arm = sim.arms["right"]
    opened = target_qpos(sim)[arm.fingers].copy()
    np.testing.assert_allclose(opened, sim.model.jnt_range[arm.finger_joints, 1], atol=1e-8)
    right["gripActive"] = False
    right["trigger"] = False
    tick(5)
    np.testing.assert_allclose(target_qpos(sim)[arm.fingers], opened)
    right["gripActive"] = True
    tick(65)
    np.testing.assert_allclose(target_qpos(sim)[arm.fingers], sim.model.jnt_range[arm.finger_joints, 0])
    print("PASS orientation tracking, trigger opens/closes, release preserves gripper")

    start()
    left = packet["leftController"]
    left["gripActive"] = True
    tick()
    before = target_qpos(sim)
    left["thumbstick"]["y"] = 1
    tick(20)
    assert np.linalg.norm(target_qpos(sim)[sim.lift.qpos] - before[sim.lift.qpos]) > 0.01
    before = target_qpos(sim)
    left["gripActive"] = False
    tick(5)
    np.testing.assert_allclose(target_qpos(sim)[sim.lift.qpos], before[sim.lift.qpos])
    print("PASS lift motion and grip release")

    start()
    right = packet["rightController"]
    right["gripActive"] = True
    tick()
    right["position"]["x"] = 2.0
    tick()
    assert not sim.armed["right"]
    before = target_qpos(sim)
    tick(5)
    np.testing.assert_allclose(target_qpos(sim)[sim.arms["right"].qpos], before[sim.arms["right"].qpos])
    right["gripActive"] = False
    tick()
    assert sim.armed["right"]
    right["gripActive"] = True
    tick()
    sim.receiver.received_at = {side: time.monotonic() - 1 for side in sim.arms}
    sim.step(dt)
    assert not any(sim.armed.values()) and sim.chassis_braking
    tick()
    assert not sim.armed["right"]  # reconnect while still holding grip is inert
    right["gripActive"] = False
    tick()
    right["gripActive"] = True
    tick()
    sim.press("space")
    tick(2)
    sim.release("space")
    tick()
    assert not sim.armed["right"]
    right["gripActive"] = False
    tick()
    right["gripActive"] = True
    tick()
    sim.stop()  # viewer focus loss uses the same path
    tick()
    assert not sim.armed["right"]
    print("PASS tracking jump, timeout, reconnect, Space and focus-loss rearm")

    # Primary button reset and interruption by re-gripping.
    start()
    right = packet["rightController"]
    right["gripActive"] = True
    tick()
    for _ in range(20):
        right["position"]["y"] += 0.002
        tick()
    right["gripActive"] = False
    right["aButton"] = True
    tick(100)
    np.testing.assert_allclose(target_qpos(sim)[sim.arms["right"].qpos], sim.home[sim.arms["right"].qpos])
    print("PASS A button gradual arm reset")

    if sim.chassis is not None:
        start()
        right = packet["rightController"]
        left = packet["leftController"]
        right["gripActive"] = True
        tick()
        # A sideways right-stick deflection must never steer or change modes.
        angle = sim.chassis.parallel_angle
        right["thumbstick"]["x"] = -1
        tick(5)
        np.testing.assert_allclose(sim.chassis_command, [0, 0, 0])
        assert sim.chassis.parallel_angle == angle
        base = sim.data.xpos[sim.base].copy()
        right["thumbstick"]["y"] = -1
        tick(80)
        displacement = sim.data.xpos[sim.base] - base
        assert displacement[0] < -0.02, displacement
        np.testing.assert_allclose(sim.chassis_command, [-1, 0, 0])
        right["thumbstick"]["y"] = 0
        right["thumbstick"]["x"] = 0
        left["thumbstick"]["x"] = -0.5
        left["thumbstick"]["y"] = 1  # diagonal steering must not lower the lift
        lift_before = target_qpos(sim)[sim.lift.qpos].copy()
        tick(20)
        assert sim.chassis.parallel_angle < -0.04
        assert sim.chassis_command[0] == 0 and sim.chassis_command[1] < 0
        np.testing.assert_allclose(target_qpos(sim)[sim.lift.qpos], lift_before)
        left["gripActive"] = True  # same stick now selects lift, never steering
        angle = sim.chassis.parallel_angle
        tick(10)
        assert sim.chassis_command[1] == 0
        assert sim.chassis.parallel_angle == angle
        assert np.linalg.norm(target_qpos(sim)[sim.lift.qpos] - lift_before) > 0.005
        left["gripActive"] = False
        left["thumbstick"]["y"] = 0
        tick()
        assert sim.chassis_command[1] < 0
        # Losing the steering hand cannot replay an old steering input.
        packet["leftController"] = None
        tick()
        assert sim.chassis_command[1] == 0
        packet["leftController"] = left
        left["thumbstick"]["x"] = 0
        tick()
        print("PASS independent sticks: right drift ignored, left steering, lift isolation, tracking loss")
        right["bButton"] = True
        tick(5)
        assert sim.chassis.target_mode == "spin"
        right["gripActive"] = False
        tick()
        assert sim.chassis.cancelled and sim.chassis_braking
        right["gripActive"] = True
        tick()
        assert not sim.chassis.cancelled
        for _ in range(750):
            tick()
            if sim.chassis.phase == "ready":
                break
        assert sim.chassis.phase == "ready", sim.chassis.last_status
        assert sim.chassis.target_mode == "spin"
        rotation = sim.data.xmat[sim.base].reshape(3, 3).copy()
        tick(60)
        current = sim.data.xmat[sim.base].reshape(3, 3)
        relative = rotation.T @ current
        assert np.arctan2(relative[1, 0], relative[0, 0]) < -0.01
        right["bButton"] = False
        right["thumbstick"]["pressed"] = True
        tick()
        assert sim.chassis.target_mode == "translation"
        right["thumbstick"]["pressed"] = False
        for _ in range(750):
            tick()
            if sim.chassis.phase == "ready":
                break
        assert sim.chassis.phase == "ready", sim.chassis.last_status
        assert not sim.chassis.faulted
        print("PASS physical chassis: forward, steer, spin mode, release cancels, resume, yaw, return")
    sim.reset()
    print("PASS all VR simulator checks")

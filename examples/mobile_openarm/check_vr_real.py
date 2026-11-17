"""No-hardware checks using actual dual-arm IK and isolated worker transport."""

import json
import threading
import time
from types import SimpleNamespace

import numpy as np
from check_vr_sim import controller
from vr_input import VRInput
from vr_real_worker import RealArmTargets
from vr_sim import rotation_vector

from lerobot.scripts.openarm_vr import (
    KEYS,
    MOBILE_NEUTRAL_FRAME,
    SIDES,
    CommandGate,
    IKWorker,
    MotionLimits,
    build_parser,
    control_loop,
    operating_ranges,
)


def physical_ranges():
    common = {
        "joint_1": (-75, 75),
        "joint_3": (-85, 85),
        "joint_4": (0, 135),
        "joint_5": (-85, 85),
        "joint_6": (-40, 40),
        "joint_7": (-80, 80),
        "gripper": (-65, 0),
    }
    return {side: {**common, "joint_2": (-90, 9) if side == "left" else (-9, 90)} for side in SIDES}


def state_packet():
    goal = dict.fromkeys(KEYS, 0.0)
    for side in SIDES:
        goal[f"{side}_joint_4.pos"] = 90.0
    measured = dict(goal)
    for side in SIDES:
        measured[f"{side}_joint_4.pos"] = 88.14
    return {
        "epoch": 0,
        "state_time": time.monotonic(),
        "command": goal,
        "positions": measured,
        "limits": {
            f"{side}_{motor}.pos": limits
            for side, joints in physical_ranges().items()
            for motor, limits in joints.items()
        },
    }


def check_targets():
    receiver = VRInput(0.3)
    targeter = RealArmTargets(receiver)
    state = state_packet()
    gate = CommandGate(state["command"], ranges=operating_ranges(state["limits"]))
    packet = {f"{side}Controller": controller(grip=True) for side in SIDES}
    mobile_before = targeter.sim.data.qpos.copy()

    def tick():
        receiver.ingest(json.dumps(packet))
        state["state_time"] = time.monotonic()
        result = targeter.step(state, 0.02)
        output = gate.apply(result, time.monotonic(), 0.02, measured=state["positions"])
        for key in KEYS:
            low, high = state["limits"][key]
            assert low <= output[key] <= high
            speed = 35 if key.endswith("_gripper.pos") else 30
            assert abs(output[key] - state["command"][key]) <= speed * 0.02 + 1e-9
        state["command"] = output
        state["positions"] = dict(output)
        state["limited_joints"] = sorted(gate.limited_joints)
        state["blocked_joints"] = sorted(gate.blocked_joints)
        return result

    result = tick()
    assert not any(result["active"].values()), "Startup-held grips must stay locked"
    for side in SIDES:
        packet[f"{side}Controller"]["gripActive"] = False
    tick()
    initial = dict(state["command"])
    for side in SIDES:
        packet[f"{side}Controller"]["gripActive"] = True
    tick()
    np.testing.assert_allclose(list(state["command"].values()), list(initial.values()), atol=1e-7)
    for i in range(50):
        packet["leftController"]["position"]["x"] = 0.0005 * i
        # Unwired mobile and simulation-home inputs must have no effect.
        packet["leftController"]["thumbstick"] = {"x": 1, "y": 1, "pressed": True}
        packet["leftController"]["xButton"] = True
        packet["rightController"]["bButton"] = True
        tick()
    assert max(abs(state["command"][k] - initial[k]) for k in KEYS if k.startswith("left_joint")) > 0.05
    np.testing.assert_allclose(
        [state["command"][k] for k in KEYS if k.startswith("right")],
        [initial[k] for k in KEYS if k.startswith("right")],
        atol=1e-6,
    )
    arm_indices = np.concatenate([np.r_[a.qpos, a.fingers] for a in targeter.sim.arms.values()])
    other = np.setdiff1d(np.arange(len(mobile_before)), arm_indices)
    np.testing.assert_allclose(targeter.sim.data.qpos[other], mobile_before[other])
    packet["leftController"]["trigger"] = True
    for _ in range(120):
        assert tick()["positions"]["left_gripper.pos"] == -65
    assert state["command"]["left_gripper.pos"] == -65
    packet["leftController"]["trigger"] = False
    for _ in range(120):
        assert tick()["positions"]["left_gripper.pos"] == 0
    assert state["command"]["left_gripper.pos"] == 0
    state["blocked"] = True
    assert not any(tick()["active"].values()), "Explicit global block must disarm both hands"
    state["blocked"] = False
    assert not any(tick()["active"].values()), "Global block recovery requires releasing grips"
    for side in SIDES:
        packet[f"{side}Controller"]["gripActive"] = False
    tick()
    for side in SIDES:
        packet[f"{side}Controller"]["gripActive"] = True
    assert all(tick()["active"].values())
    held = dict(state["command"])
    for side in SIDES:
        receiver.received_at[side] -= 1
    result = targeter.step(state, 0.02)
    assert not any(result["active"].values())
    assert gate.apply(result, time.monotonic(), 0.02) == held
    assert not any(tick()["active"].values()), "Restored held grips cannot resume"
    state["epoch"] += 1
    gate.epoch += 1
    assert not any(tick()["active"].values()), "New motor epoch requires release"
    for side, arm in targeter.sim.arms.items():
        np.testing.assert_allclose(
            np.rad2deg(arm.ik.lower),
            [state["limits"][f"{side}_joint_{i}.pos"][0] + 3 for i in range(1, 8)],
        )
    print(
        "PASS real IK: continuity, deadman, per-hand loss/recovery, epoch, physical limits, gripper degrees, mobile inputs ignored"
    )


def check_unreachable_targets():
    """Use the actual constrained solver, including orientation and reversal."""
    for rotate in (False, True):
        receiver = VRInput(0.3)
        targeter = RealArmTargets(receiver)
        state = state_packet()
        packet = {f"{side}Controller": controller() for side in SIDES}

        def tick(receiver=receiver, packet=packet, state=state, targeter=targeter):
            receiver.ingest(json.dumps(packet))
            state["state_time"] = time.monotonic()
            result = targeter.step(state, 0.02)
            state["command"] = result["positions"]
            state["positions"] = dict(state["command"])
            return result

        tick()
        packet["leftController"]["gripActive"] = True
        tick()
        arm = targeter.sim.arms["left"]
        original = dict(targeter.inner_limits)
        # Constrain all seven axes at the accepted pose to exercise a fully
        # unreachable translation/rotation without relying on a chosen IK branch.
        for index in range(1, 8):
            key = f"left_joint_{index}.pos"
            q = state["command"][key]
            targeter.inner_limits[key] = (q, q)
        for index in range(1, 101):
            if rotate:
                angle = np.deg2rad(index * 0.5)
                packet["leftController"]["quaternion"] = {
                    "x": 0,
                    "y": np.sin(angle / 2),
                    "z": 0,
                    "w": np.cos(angle / 2),
                }
            else:
                packet["leftController"]["position"]["x"] = index * 0.001
            assert tick()["active"]["left"]
        anchor = targeter.sim.anchors["left"]
        root = arm.ik.root_body
        rotation = targeter.sim.data.xmat[root].reshape(3, 3)
        tcp = rotation.T @ (targeter.sim.data.xpos[arm.tcp] - targeter.sim.data.xpos[root])
        orientation = rotation.T @ targeter.sim.data.xmat[arm.tcp].reshape(3, 3)
        assert np.linalg.norm(anchor.filtered_position - tcp) < 0.005
        assert np.linalg.norm(rotation_vector(anchor.filtered_rotation @ orientation.T)) < np.deg2rad(2)
        before = arm.ik.forward_world(targeter.sim.data.qpos[arm.qpos], targeter.sim.data)
        before_position, before_rotation = before.translation.copy(), before.rotation.copy()
        targeter.inner_limits = original
        # A small reversal must take effect without unwinding 10 cm / 50 deg.
        if rotate:
            angle = np.deg2rad(48)
            packet["leftController"]["quaternion"] = {
                "x": 0,
                "y": np.sin(angle / 2),
                "z": 0,
                "w": np.cos(angle / 2),
            }
        else:
            packet["leftController"]["position"]["x"] -= 0.005
        for _ in range(5):
            tick()
        after = arm.ik.forward_world(targeter.sim.data.qpos[arm.qpos], targeter.sim.data)
        world_mapping = targeter.sim.data.xmat[targeter.sim.base].reshape(3, 3) @ targeter.sim.mapping
        if rotate:
            assert rotation_vector(after.rotation @ before_rotation.T) @ world_mapping[:, 1] < -0.001
        else:
            assert (after.translation - before_position) @ world_mapping[:, 0] < -0.0002
        # Feedback recovery also reaches the solver itself, without disarming.
        key = "left_joint_3.pos"
        state["recovery_directions"] = {key: -1}
        previous = state["command"][key]
        assert tick()["active"]["left"]
        assert state["command"][key] >= previous - 1e-8
        assert np.rad2deg(arm.ik.lower[2]) >= previous - 1e-8
    print(
        "PASS real IK: unreachable translation/rotation do not accumulate; small reversal and inward recovery"
    )


def check_worker():
    import zmq

    context = zmq.Context()
    publisher = context.socket(zmq.PUB)
    port = publisher.bind_to_random_port("tcp://127.0.0.1")
    args = build_parser().parse_args(["--run-seconds", "1"])
    args.vr_endpoint = f"tcp://127.0.0.1:{port}"
    worker = IKWorker(args)
    stop = threading.Event()
    command = dict(state_packet()["command"])

    mobile_frames = []

    class FakeMobile:
        def write(self, frame):
            mobile_frames.append(bytes(frame))
    class FakeBackend:
        arms = {
            side: SimpleNamespace(config=SimpleNamespace(joint_limits=limits))
            for side, limits in physical_ranges().items()
        }

        def read(self):
            return dict(command), dict.fromkeys(command, 0.0)

        def send(self, target):
            command.update(target)

        def validate_positions(self, positions, **_):
            assert positions.keys() == command.keys()
            assert all(np.isfinite(v) for v in positions.values())
            return positions

    try:
        worker.wait_ready(stop)
        initial = dict(command)
        result = control_loop(FakeBackend(), worker, initial, MotionLimits(), stop, args, FakeMobile())
        assert result == initial, "No VR input must hold the preparation target"
        assert mobile_frames and all(frame == MOBILE_NEUTRAL_FRAME for frame in mobile_frames)
        print(
            "PASS actual worker IPC and unified motor loop with fake hardware: "
            "no VR input holds arms and sends neutral mobile frames"
        )
    finally:
        worker.close()
        publisher.close(0)
        context.term()


def check_reset_button():
    receiver = VRInput(0.3)
    targeter = RealArmTargets(receiver)
    state = state_packet()
    packet = {f"{side}Controller": controller() for side in SIDES}
    packet["leftController"]["xButton"] = True

    def tick(*, stale=False):
        receiver.ingest(json.dumps(packet))
        if stale:
            receiver.received_at["right"] -= 1
        state["state_time"] = time.monotonic()
        return targeter.step(state, 0.02)

    assert tick()["reset_id"] == 0, "Startup-held X must not reset"
    packet["leftController"]["xButton"] = False
    tick()
    packet["leftController"]["xButton"] = True
    result = tick()
    assert result["reset_id"] == 1 and result["reset_safe"]
    assert tick()["reset_id"] == 1, "Held X must not repeat"
    state["resetting"] = True
    packet["leftController"]["xButton"] = False
    tick()
    packet["leftController"]["xButton"] = True
    assert tick()["reset_id"] == 1, "Do not queue resets during motion"
    state["resetting"] = False
    state["epoch"] += 1
    assert tick()["reset_id"] == 1, "Held X at completion must not restart"
    for unsafe in ("grip", "tracking"):
        packet["leftController"]["xButton"] = False
        packet["rightController"]["aButton"] = True
        assert tick()["reset_id"] == 1, "A is not the dual-arm reset"
        packet["leftController"]["xButton"] = True
        packet["rightController"]["gripActive"] = unsafe == "grip"
        result = tick(stale=unsafe == "tracking")
        assert result["reset_id"] == 1 and not result["reset_safe"]
        packet["rightController"]["gripActive"] = False
        assert tick()["reset_id"] == 1, "Restored tracking/released grip cannot replay held X"
    packet["leftController"]["xButton"] = False
    tick()
    packet["leftController"]["xButton"] = True
    assert tick()["reset_id"] == 2
    print("PASS X reset: both grips released, edge-triggered, no held/stale/queued reset; A ignored")


def check_j2_retraction():
    """Compare redundant solutions for the same 15 cm inward hand motion."""
    excursions = {}
    for weight in (1, 8):
        receiver = VRInput(0.3)
        targeter = RealArmTargets(receiver, j2_motion_weight=weight)
        state = state_packet()
        for side, sign in (("left", -1), ("right", 1)):
            state["command"][f"{side}_joint_2.pos"] = sign * 20
            state["command"][f"{side}_joint_3.pos"] = -sign * 30
        state["positions"] = dict(state["command"])
        initial = dict(state["command"])
        packet = {f"{side}Controller": controller() for side in SIDES}

        def tick(receiver=receiver, targeter=targeter, state=state, packet=packet):
            receiver.ingest(json.dumps(packet))
            state["state_time"] = time.monotonic()
            result = targeter.step(state, 0.02)
            for key in KEYS:
                assert state["limits"][key][0] <= result["positions"][key] <= state["limits"][key][1]
                assert abs(result["positions"][key] - state["command"][key]) <= 0.6 + 1e-8
            state["command"] = result["positions"]
            state["positions"] = dict(state["command"])

        tick()
        for c in packet.values():
            c["gripActive"] = True
        tick()
        starts = {
            side: (
                targeter.sim.data.xpos[arm.tcp].copy(),
                targeter.sim.data.xmat[arm.tcp].reshape(3, 3).copy(),
            )
            for side, arm in targeter.sim.arms.items()
        }
        mapping = targeter.sim.data.xmat[targeter.sim.base].reshape(3, 3) @ targeter.sim.mapping
        travel = np.zeros(2)
        for retract in (True, False):
            for index in range(1, 251):
                fraction = min(1, index / 150)
                distance = 0.15 * (fraction if retract else 1 - fraction)
                for c in packet.values():
                    c["position"]["z"] = distance
                tick()
                if retract:
                    travel = np.maximum(
                        travel,
                        [
                            abs(state["command"][f"{side}_joint_2.pos"] - initial[f"{side}_joint_2.pos"])
                            for side in SIDES
                        ],
                    )
            for side, arm in targeter.sim.arms.items():
                start_position, start_rotation = starts[side]
                expected = start_position + mapping @ np.array([0, 0, 0.15 if retract else 0])
                assert np.linalg.norm(targeter.sim.data.xpos[arm.tcp] - expected) < 0.001
                rotation = targeter.sim.data.xmat[arm.tcp].reshape(3, 3)
                assert np.linalg.norm(rotation_vector(rotation @ start_rotation.T)) < np.deg2rad(0.5)
        excursions[weight] = travel
    assert np.all(excursions[1] > 5), "Exercise an actual redundant shoulder movement"
    assert np.all(excursions[8] < 0.5 * excursions[1])
    print(f"PASS J2 retraction: baseline {excursions[1]} deg -> weighted {excursions[8]} deg; same TCP pose")


def check_response():
    """Exercise filter + actual IK + acceleration/soft-limit gate together."""
    traces = []
    args = build_parser().parse_args([])
    for fast in (False, True):
        receiver = VRInput(0.3)
        targeter = RealArmTargets(
            receiver,
            linear_speed=args.arm_linear_speed_m_s if fast else 0.16,
            angular_speed=args.arm_angular_speed_deg_s if fast else 50,
            filter_scale=args.vr_filter_scale if fast else 1,
        )
        state = state_packet()
        gate = CommandGate(
            state["command"],
            acceleration=args.joint_acceleration_deg_s2 if fast else 90,
            ranges=operating_ranges(state["limits"]),
        )
        packet = {f"{side}Controller": controller() for side in SIDES}

        def tick(receiver=receiver, targeter=targeter, state=state, packet=packet, gate=gate):
            receiver.ingest(json.dumps(packet))
            state["state_time"] = time.monotonic()
            result = targeter.step(state, 0.02)
            command = gate.apply(result, time.monotonic(), 0.02, measured=state["positions"])
            for key in KEYS:
                assert abs(command[key] - state["command"][key]) <= 0.6 + 1e-8
            state["command"] = command
            state["positions"] = dict(command)
            state["limited_joints"] = sorted(gate.limited_joints)
            state["blocked_joints"] = sorted(gate.blocked_joints)
            targeter.set_positions(targeter.display, command)

        tick()
        packet["leftController"]["gripActive"] = True
        tick()
        arm = targeter.sim.arms["left"]
        start = targeter.display.xpos[arm.tcp].copy()
        direction = (targeter.sim.data.xmat[targeter.sim.base].reshape(3, 3) @ targeter.sim.mapping)[:, 2]
        packet["leftController"]["position"]["z"] = 0.03
        trace = []
        for _ in range(100):
            tick()
            trace.append(float((targeter.display.xpos[arm.tcp] - start) @ direction))
        assert abs(trace[-1] - 0.03) < 0.0001, "Soft slowdown must retain a reachable target for catch-up"
        traces.append(trace)
    previous, current = traces
    assert current[4] > 1.2 * previous[4], "Faster onset at 100 ms"
    times = [next((i + 1) * 0.02 for i, value in enumerate(trace) if value >= 0.027) for trace in traces]
    assert times[1] < times[0]
    print(
        f"PASS response: 3 cm step reaches 90% in {times[0]:.2f}s -> {times[1]:.2f}s, final target retained"
    )


if __name__ == "__main__":
    check_targets()
    check_unreachable_targets()
    check_reset_button()
    check_j2_retraction()
    check_response()
    check_worker()

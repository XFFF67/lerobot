"""Motor-side VR gates and lifecycle, with no CAN sockets or motors."""
import math
import threading
from unittest.mock import Mock

import pytest

from lerobot.scripts import openarm_vr as vr


def positions():
    return dict.fromkeys(vr.KEYS, 0.0)


def packet(now=10, **extra):
    return {
        "epoch": 0,
        "state_time": now,
        "sent_at": now,
        "positions": positions(),
        "active": dict.fromkeys(vr.SIDES, True),
        "vr_time": dict.fromkeys(vr.SIDES, now),
        **extra,
    }


def test_motor_gate_slew_limit_grippers_and_released_side():
    gate = vr.CommandGate(positions())
    target = packet()
    target["positions"] = dict.fromkeys(vr.KEYS, 20.0)
    target["active"]["right"] = False
    for _ in range(10):
        before = dict(gate.command)
        after = gate.apply(target, 10, 0.02)
        for key in after:
            speed = 35 if key.endswith("_gripper.pos") else 30
            assert abs(after[key] - before[key]) <= speed * 0.02 + 1e-9
            if key.startswith("right_"):
                assert after[key] == 0
    target["active"]["left"] = False
    assert gate.apply(target, 10, 0.02) == after
    assert all(v == 0 for v in gate.velocity.values())


def test_gripper_full_endpoints_moderate_speed_and_deadman():
    gate = vr.CommandGate(positions())
    target = packet()
    key = "left_gripper.pos"
    target["positions"][key] = -65
    for _ in range(20):
        gate.apply(target, 10, 0.02)
    partial = gate.command[key]
    assert -20 < partial < 0
    target["active"]["left"] = False
    for _ in range(20):
        assert gate.apply(target, 10, 0.02)[key] == partial
    target["active"]["left"] = True
    for endpoint in (-65, 0):
        target["positions"][key] = endpoint
        for _ in range(120):
            before = gate.command[key]
            after = gate.apply(target, 10, 0.02)[key]
            assert abs(after - before) <= 35 * 0.02 + 1e-9
            assert -65 <= after <= 0
        assert gate.command[key] == endpoint
    assert all(v == 0 for k, v in gate.command.items() if k != key)


@pytest.mark.parametrize("obstacle", [-50.0, -30.0, -10.0])
def test_gripper_closing_on_object_bounds_preload_and_can_reopen(obstacle):
    key = "left_gripper.pos"
    initial = positions()
    initial[key] = -65
    gate = vr.CommandGate(initial)
    target = packet()
    target["positions"]["left_joint_1.pos"] = 20
    for _ in range(250):
        measured = dict(gate.command)
        measured[key] = min(measured[key], obstacle)
        before = gate.command[key]
        result = gate.apply(target, 10, 0.02, measured=measured)
        assert abs(result[key] - before) <= 35 * 0.02 + 1e-9
        assert result[key] - measured[key] <= vr.GRIPPER_CLOSE_LEAD_DEG + 1e-9
    assert result[key] == obstacle + vr.GRIPPER_CLOSE_LEAD_DEG
    assert result["left_joint_1.pos"] == 20
    target["active"]["left"] = False
    assert gate.apply(target, 10, 0.02, measured=measured) == result
    assert gate.apply(None, 10, 0.02, measured=measured) == result
    target["active"]["left"] = True
    target["positions"][key] = -65
    for _ in range(120):
        measured = dict(gate.command)
        measured[key] = min(measured[key], obstacle)
        result = gate.apply(target, 10, 0.02, measured=measured)
    assert result[key] == -65


def test_inset_targets_and_existing_pose_only_move_inward():
    key = "right_joint_6.pos"
    ranges = {key: (-40, 40), "right_gripper.pos": (-65, 0)}
    inner = vr.operating_ranges(ranges)
    assert inner == {key: (-37, 37), "right_gripper.pos": (-65, 0)}
    target = packet()
    target["positions"][key] = 40
    gate = vr.CommandGate(positions(), ranges={k: inner.get(k, (-90, 90)) for k in vr.KEYS})
    for _ in range(150):
        gate.apply(target, 10, 0.02)
    assert 36.5 < gate.command[key] <= 37
    gate.command[key] = 38
    assert gate.apply(target, 10, 0.02)[key] == 38
    target["positions"][key] = 36
    for _ in range(50):
        assert gate.apply(target, 10, 0.02)[key] <= 38
    assert gate.command[key] == 36


def test_feedback_small_overshoot_allows_recovery_without_timeout():
    key = "right_joint_6.pos"
    envelope = vr.FeedbackEnvelope({key: (-40, 40)})
    assert envelope.update({key: 40.27140983934323}, 10) == [key]
    assert envelope.update({key: 40.6}, 10.4) == [key]
    assert envelope.update({key: 39.9}, 10.45) == [key]
    assert envelope.update({key: 39.5}, 10.5) == []
    assert envelope.update({key: 40.2}, 11) == [key]
    assert envelope.update({key: 40.2}, 60) == [key]
    assert envelope.recovery_directions == {key: 1}


@pytest.mark.parametrize("direction", [-1, 1])
def test_predictive_slowdown_and_immediate_inward_escape(direction):
    key = "right_joint_3.pos"
    initial = positions()
    initial[key] = direction * 78
    ranges = dict.fromkeys(vr.KEYS, (-82, 82))
    target = packet()
    target["positions"][key] = direction * 82
    steps = []
    for outward_speed in (0, 20, 50):
        gate = vr.CommandGate(initial, ranges=ranges)
        gate.velocity[key] = direction * 30
        velocities = positions()
        velocities[key] = direction * outward_speed
        result = gate.apply(target, 10, 0.02, measured=initial, velocities=velocities)
        steps.append(abs(result[key] - initial[key]))
        assert key in gate.limited_joints
        assert (key in gate.blocked_joints) == (outward_speed == 50)
    assert steps[0] > steps[1] > steps[2] == 0
    # Even a previously outward-moving joint can accept reversal immediately.
    measured = dict(initial)
    measured[key] = direction * 85.559
    gate = vr.CommandGate(initial, ranges=ranges)
    recovery = {key: direction}
    assert gate.apply(target, 10, 0.02, measured=measured, recovery_directions=recovery)[key] == initial[key]
    target["positions"][key] = 0
    for _ in range(20):
        before = gate.command[key]
        after = gate.apply(target, 10, 0.02, measured=measured, recovery_directions=recovery)[key]
        assert 0 < direction * (before - after) <= 5 * 0.02 + 1e-9
    target["active"]["right"] = False
    assert gate.apply(target, 10, 0.02, measured=measured, recovery_directions=recovery)[key] == after


@pytest.mark.parametrize("value", [41.01, -41.01, math.nan, math.inf])
def test_feedback_large_or_invalid_overshoot_still_faults(value):
    with pytest.raises(RuntimeError):
        vr.FeedbackEnvelope({"right_joint_6.pos": (-40, 40)}).update({"right_joint_6.pos": value}, 10)


def test_gripper_feedback_tolerance_does_not_expand_commands():
    key = "left_gripper.pos"
    envelope = vr.FeedbackEnvelope({key: (-65, 0)})
    assert envelope.update({key: 0.2076}, 10) == []
    with pytest.raises(RuntimeError, match="Gripper"):
        envelope.update({key: 0.51}, 10)


@pytest.mark.parametrize("kind", ["state_time", "sent_at", "vr_time", "epoch", "future", "none"])
def test_stale_or_mismatched_input_cannot_move(kind):
    gate = vr.CommandGate(positions())
    target = packet()
    target["positions"] = dict.fromkeys(vr.KEYS, 20.0)
    if kind == "vr_time":
        target["vr_time"] = dict.fromkeys(vr.SIDES, 9.6)
    elif kind == "epoch":
        target["epoch"] = 1
    elif kind == "future":
        target["sent_at"] = 11
    elif kind == "none":
        target = None
    else:
        target[kind] = 9.7
    assert gate.apply(target, 10, 0.02) == positions()


@pytest.mark.parametrize("value", [math.nan, math.inf])
def test_nonfinite_ik_rejected(value):
    target = packet()
    target["positions"]["left_joint_4.pos"] = value
    with pytest.raises(ValueError, match="Invalid IK"):
        vr.CommandGate(positions()).apply(target, 10, 0.02)


def fake_runtime():
    args = vr.build_parser().parse_args([])
    backend = Mock(enabled=False)
    backend.read.return_value = (positions(), positions())
    backend.validate_positions.side_effect = lambda q, **_: dict(q)
    worker = Mock()
    return args, backend, worker


class FakeMobilePort:
    def __init__(self):
        self.out_waiting = 0
        self.writes = []
        self.reads = 0
        self.reset = False
        self.closed = False

    def read(self):
        self.reads += 1
        return b""

    def write(self, payload):
        self.writes.append(bytes(payload))
        return len(payload)

    def reset_output_buffer(self):
        self.reset = True

    def close(self):
        self.closed = True


def test_mobile_output_validates_worker_frames_and_stops_on_close(monkeypatch):
    port = FakeMobilePort()
    monkeypatch.setattr(vr.time, "sleep", lambda _: None)
    output = vr.MobileOutput("unused", lambda _: port)
    motion = [0x55, 0, 50, 0, 0, 0, 1, 1, 20]
    output.write(motion)
    with pytest.raises(ValueError, match="Invalid mobile frame"):
        output.write([0x55, 0])
    output.close()
    assert port.writes == [vr.MOBILE_NEUTRAL_FRAME, bytes(motion), vr.MOBILE_NEUTRAL_FRAME]
    assert port.reads == 4 and port.reset and port.closed


def test_arm_control_loop_writes_mobile_frame_from_same_target_packet(monkeypatch):
    args, backend, worker = fake_runtime()
    args.run_seconds = 0.04
    worker.process.poll.return_value = None
    backend.arms = {
        side: Mock(
            config=Mock(joint_limits={**{f"joint_{i}": (-40, 40) for i in range(1, 8)}, "gripper": (-65, 0)})
        )
        for side in vr.SIDES
    }
    now = [0.0]
    stop = threading.Event()
    monkeypatch.setattr(vr.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(stop, "wait", lambda dt: now.__setitem__(0, now[0] + dt))
    mobile = Mock()
    frame = [0x55, 0, 0, 0, 0, 25, 0, 1, 10]

    def incoming(_):
        return packet(now=now[0], type="target", mobile_frame=frame)

    monkeypatch.setattr(vr, "latest_packet", incoming)
    monkeypatch.setattr(vr, "send_packet", lambda *_: True)
    vr.control_loop(backend, worker, positions(), vr.MotionLimits(), stop, args, mobile)
    assert mobile.write.call_count >= 1
    assert all(call.args[0] == frame for call in mobile.write.call_args_list)


@pytest.mark.parametrize("scenario", ["grasp", "opening_jam", "arm_jam"])
def test_motor_loop_grasp_continues_but_opening_and_arm_jams_fault(monkeypatch, scenario):
    args, backend, worker = fake_runtime()
    args.run_seconds = 7
    worker.process.poll.return_value = None
    backend.arms = {
        side: Mock(
            config=Mock(joint_limits={**{f"joint_{i}": (-40, 40) for i in range(1, 8)}, "gripper": (-65, 0)})
        )
        for side in vr.SIDES
    }
    now = [0.0]
    stop = threading.Event()
    monkeypatch.setattr(vr.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(stop, "wait", lambda dt: now.__setitem__(0, now[0] + dt))
    initial = positions()
    key = "left_gripper.pos"
    initial[key] = -30 if scenario == "opening_jam" else -65
    command = dict(initial)
    commands = []

    def read():
        measured = dict(command)
        if scenario == "grasp":
            measured[key] = min(command[key], -30)
        elif scenario == "opening_jam":
            measured[key] = -30
        else:
            measured["left_joint_1.pos"] = 0
        return measured, positions()

    def send(q):
        command.update(q)
        commands.append((now[0], dict(q)))

    def incoming(_):
        target = packet(now=now[0], type="target")
        target["positions"]["left_joint_1.pos"] = 20
        if scenario == "opening_jam" or now[0] >= 4:
            target["positions"][key] = -65
        return target

    backend.read.side_effect = read
    backend.send.side_effect = send
    monkeypatch.setattr(vr, "latest_packet", incoming)
    monkeypatch.setattr(vr, "send_packet", lambda *_: True)
    if scenario == "grasp":
        result = vr.control_loop(backend, worker, initial, vr.MotionLimits(), stop, args)
        assert result[key] == -65
        assert result["left_joint_1.pos"] == 20
        assert all(q[key] == -25 for t, q in commands if 3 < t < 4)
    else:
        joint = key if scenario == "opening_jam" else "left_joint_1.pos"
        with pytest.raises(RuntimeError, match=f"following error: {joint}; measured=.*command="):
            vr.control_loop(backend, worker, initial, vr.MotionLimits(), stop, args)


def test_control_loop_persistent_small_excursion_allows_inward_and_other_arm(monkeypatch):
    args, backend, worker = fake_runtime()
    args.run_seconds = 1.0
    worker.process.poll.return_value = None
    backend.arms = {
        side: Mock(
            config=Mock(joint_limits={**{f"joint_{i}": (-40, 40) for i in range(1, 8)}, "gripper": (-65, 0)})
        )
        for side in vr.SIDES
    }
    now = [0.0]
    stop = threading.Event()
    monkeypatch.setattr(vr.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(stop, "wait", lambda dt: now.__setitem__(0, now[0] + dt))
    initial = positions()
    initial["right_joint_6.pos"] = 37
    command = dict(initial)
    backend.send.side_effect = lambda q: command.update(q)

    def read():
        measured = dict(command)
        if now[0] < 0.7:
            measured["right_joint_6.pos"] = 40.2714
        return measured, positions()

    backend.read.side_effect = read

    def incoming(_):
        target = packet(now=now[0], type="target")
        target["positions"]["left_joint_1.pos"] = 20
        return target

    monkeypatch.setattr(vr, "latest_packet", incoming)
    sent = []
    monkeypatch.setattr(vr, "send_packet", lambda _, p: sent.append(p))
    result = vr.control_loop(backend, worker, initial, vr.MotionLimits(), stop, args)
    assert result["left_joint_1.pos"] > 10
    assert result["right_joint_6.pos"] < 34
    assert all(p["epoch"] == 0 for p in sent)
    assert sent[0]["recovery_directions"] == {"right_joint_6.pos": 1}
    assert not sent[-1]["recovery_directions"]
    assert all(not p.get("blocked") for p in sent)


@pytest.mark.parametrize("cancel", [None, "grasp", "stream", "grip", "tracking", "jam"])
def test_x_reset_prepares_both_arms_and_cancels_without_replaying(monkeypatch, caplog, cancel):
    args, backend, worker = fake_runtime()
    args.run_seconds = 35
    worker.process.poll.return_value = None
    backend.arms = {
        side: Mock(
            config=Mock(
                joint_limits={
                    **{f"joint_{i}": (-90, 90) for i in range(1, 8)},
                    "joint_4": (0, 135),
                    "gripper": (-65, 0),
                }
            )
        )
        for side in vr.SIDES
    }
    now = [0.0]
    epoch = [0]
    stop = threading.Event()
    monkeypatch.setattr(vr.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(stop, "wait", lambda dt: now.__setitem__(0, now[0] + dt))
    initial = positions()
    for side in vr.SIDES:
        initial[f"{side}_joint_1.pos"] = 20
        initial[f"{side}_joint_3.pos"] = -40
        initial[f"{side}_joint_4.pos"] = 40
        initial[f"{side}_gripper.pos"] = -25 if side == "left" else -50
    command = dict(initial)
    commands, states = [], []

    def read():
        measured = dict(command)
        if cancel == "jam":
            measured["left_joint_4.pos"] = initial["left_joint_4.pos"]
        if cancel == "grasp":
            measured["left_gripper.pos"] -= 5
        return measured, positions()

    def send(q):
        commands.append((now[0], dict(q)))
        command.update(q)

    def incoming(_):
        if cancel == "stream" and 1 <= now[0] < 2:
            return None
        target = packet(
            now=now[0],
            type="target",
            epoch=epoch[0],
            positions=dict(command),
            active=dict.fromkeys(vr.SIDES, False),
            reset_id=1,
            reset_safe=True,
        )
        if 1 <= now[0] < 2:
            if cancel == "grip":
                target["reset_safe"] = False
            elif cancel == "tracking":
                target["vr_time"]["right"] = now[0] - 1
        return target

    def state(_, p):
        epoch[0] = p["epoch"]
        states.append(p)

    backend.read.side_effect = read
    backend.send.side_effect = send
    monkeypatch.setattr(vr, "latest_packet", incoming)
    monkeypatch.setattr(vr, "send_packet", state)
    with caplog.at_level("INFO"):
        if cancel == "jam":
            with pytest.raises(RuntimeError, match="following error"):
                vr.control_loop(backend, worker, initial, vr.MotionLimits(), stop, args)
            return
        result = vr.control_loop(backend, worker, initial, vr.MotionLimits(), stop, args)
    assert caplog.text.count("VR_RESET: both arms") == 1
    assert commands[0][1] == initial
    assert any(p["resetting"] for p in states)
    assert not states[-1]["resetting"]
    assert states[-1]["epoch"] > 0
    for (t0, q0), (t1, q1) in zip(commands[:-1], commands[1:], strict=True):
        assert max(abs(q1[k] - q0[k]) for k in q0) <= 5 * (t1 - t0) + 1e-8
        for side in vr.SIDES:
            assert q1[f"{side}_gripper.pos"] == initial[f"{side}_gripper.pos"]
    if cancel in (None, "grasp"):
        assert result == vr.preparation_goal(vr.home_goal(initial))
        assert "VR_RESET_READY" in caplog.text
    else:
        assert "VR_RESET_CANCELLED" in caplog.text
        assert 40 < result["left_joint_4.pos"] < 90
        assert all(q == result for t, q in commands if t > 2)
    backend.enable.assert_not_called()
    backend.close.assert_not_called()


@pytest.mark.parametrize("elbow_offset", [0.0, -0.16392704140845193, 87.0])
def test_single_backend_prepares_then_vr_then_continuous_slow_return(monkeypatch, elbow_offset):
    args, backend, worker = fake_runtime()
    backend.arms = {
        side: Mock(config=Mock(joint_limits={"joint_4": (0.0, 135.0), "gripper": (-65.0, 0.0)}))
        for side in vr.SIDES
    }
    backend.read.return_value[0]["left_joint_4.pos"] = elbow_offset
    backend.read.return_value[0]["right_joint_1.pos"] = 12.0
    backend.read.return_value[0]["left_gripper.pos"] = -20.0
    backend.validate_positions.side_effect = lambda q, **kwargs: (
        vr.DamiaoPreparationBackend.validate_positions(backend, q, **kwargs)
    )
    motions = []

    def motion(b, start, goal, limits, stop, **kwargs):
        assert b is backend
        motions.append((dict(start), dict(goal), limits.duration_s, kwargs))

    monkeypatch.setattr(vr, "run_motion", motion)
    final = positions()
    final["left_joint_4.pos"] = 80
    teleop = Mock(return_value=final)
    monkeypatch.setattr(vr, "control_loop", teleop)
    vr.run(args, threading.Event(), lambda *_: backend, lambda *_: worker)
    backend.connect.assert_called_once()
    initial = dict(backend.read.return_value[0])
    initial["left_joint_4.pos"] = max(0.0, elbow_offset)
    backend.enable.assert_called_once_with(initial)
    home = vr.home_goal(initial)
    assert home["right_joint_1.pos"] == 0.0
    assert home["left_gripper.pos"] == -20.0
    assert motions[0][:3] == (initial, home, 10)
    assert motions[0][3]["phase"] == "HOMING"
    assert motions[1][:3] == (home, vr.preparation_goal(home), 20)
    assert teleop.call_args.args[2] == motions[1][1]
    assert motions[2][:3] == (final, home, 30)
    for arm in backend.arms.values():
        arm.bus.set_zero_position.assert_not_called()
    backend.close.assert_called_once_with(disable_torque=True)
    worker.close.assert_called_once()


def test_startup_failure_preserves_existing_torque(monkeypatch):
    args, backend, worker = fake_runtime()
    backend.read.side_effect = ConnectionError("stale")
    with pytest.raises(ConnectionError):
        vr.run(args, threading.Event(), lambda *_: backend, lambda *_: worker)
    backend.enable.assert_not_called()
    backend.close.assert_called_once_with(disable_torque=False)


def test_interrupted_homing_preserves_torque_and_never_prepares(monkeypatch):
    args, backend, worker = fake_runtime()
    backend.read.return_value[0]["left_joint_4.pos"] = 87
    midway = vr.preparation_goal(positions(), 40)
    motion = Mock(return_value=midway)
    monkeypatch.setattr(vr, "run_motion", motion)
    teleop = Mock()
    monkeypatch.setattr(vr, "control_loop", teleop)
    vr.run(args, threading.Event(), lambda *_: backend, lambda *_: worker)
    motion.assert_called_once()
    assert motion.call_args.kwargs["phase"] == "HOMING"
    teleop.assert_not_called()
    backend.close.assert_called_once_with(disable_torque=False)


@pytest.mark.parametrize("completed_phases", [0, 1])
def test_fault_never_starts_auto_lower_or_rezeros(monkeypatch, completed_phases):
    args, backend, worker = fake_runtime()
    motion = Mock(side_effect=[*[None] * completed_phases, RuntimeError("feedback")])
    monkeypatch.setattr(vr, "run_motion", motion)
    with pytest.raises(RuntimeError):
        vr.run(args, threading.Event(), lambda *_: backend, lambda *_: worker)
    assert motion.call_count == completed_phases + 1
    backend.close.assert_called_once_with(disable_torque=False)
    backend.send.assert_not_called()


def test_real_motion_runs_timed_home_prepare_then_verified_return(monkeypatch, caplog):
    args, backend, worker = fake_runtime()
    now = [0.0]
    monkeypatch.setattr(vr.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(threading.Event, "wait", lambda self, dt: now.__setitem__(0, now[0] + dt))
    current = positions()
    current["left_joint_4.pos"] = 25.0
    current["right_joint_1.pos"] = 12.0
    current["left_gripper.pos"] = -20.0
    backend.read.side_effect = lambda: (dict(current), positions())
    sent = []

    def send(command):
        sent.append((now[0], dict(command)))
        current.update(command)

    backend.send.side_effect = send
    home = vr.home_goal(current)
    goal = vr.preparation_goal(home)

    def control(*args):
        assert current == goal
        assert now[0] >= 30.0  # Startup uses 10 s home + 20 s prepare command trajectories.
        assert "HOMING COMMAND COMPLETE" in caplog.text
        assert "PREPARING COMMAND COMPLETE" in caplog.text
        return dict(goal)

    monkeypatch.setattr(vr, "control_loop", control)
    with caplog.at_level("INFO"):
        vr.run(args, threading.Event(), lambda *_: backend, lambda *_: worker)
    assert current == home
    first_home = next(i for i, (_, q) in enumerate(sent) if q == home)
    assert first_home > 0
    assert all(q["left_joint_4.pos"] <= 25.0 for _, q in sent[:first_home])
    assert any(q == goal for _, q in sent[first_home:])
    for (t0, q0), (t1, q1) in zip(sent[:-1], sent[1:], strict=True):
        assert max(abs(q1[k] - q0[k]) for k in q0) <= 10.0 * (t1 - t0) + 1e-7
    backend.close.assert_called_once_with(disable_torque=True)


def test_cancel_during_prepare_returns_from_last_command(monkeypatch):
    args, backend, worker = fake_runtime()
    midway = vr.preparation_goal(positions(), 45)
    motion = Mock(side_effect=[None, midway, None])
    monkeypatch.setattr(vr, "run_motion", motion)
    teleop = Mock()
    monkeypatch.setattr(vr, "control_loop", teleop)
    vr.run(args, threading.Event(), lambda *_: backend, lambda *_: worker)
    assert motion.call_args.args[1] == midway
    assert motion.call_args.args[2] == positions()
    teleop.assert_not_called()
    backend.close.assert_called_once_with(disable_torque=True)


def test_preview_never_constructs_worker_or_motor(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["openarm_vr"])
    run = Mock()
    monkeypatch.setattr(vr, "run", run)
    vr.main()
    run.assert_not_called()
    assert "no CAN" in capsys.readouterr().out


@pytest.mark.parametrize("duration", ["0", "-1", "nan", "inf"])
def test_invalid_homing_duration_never_constructs_hardware(monkeypatch, duration):
    monkeypatch.setattr("sys.argv", ["openarm_vr", "--execute", "--home-duration-s", duration])
    run = Mock()
    monkeypatch.setattr(vr, "run", run)
    with pytest.raises(SystemExit):
        vr.main()
    run.assert_not_called()


@pytest.mark.parametrize("weight", ["0", "0.5", "33", "nan", "inf"])
def test_invalid_ik_j2_weight_never_constructs_hardware(monkeypatch, weight):
    monkeypatch.setattr("sys.argv", ["openarm_vr", "--execute", "--ik-j2-motion-weight", weight])
    run = Mock()
    monkeypatch.setattr(vr, "run", run)
    with pytest.raises(SystemExit):
        vr.main()
    run.assert_not_called()


@pytest.mark.parametrize(
    "flag,value",
    [
        ("--gripper-gain-scale", "2.1"),
        ("--gripper-gain-scale", "nan"),
        ("--vr-filter-scale", "0"),
        ("--vr-filter-scale", "0.4"),
        ("--vr-filter-scale", "inf"),
        ("--joint-acceleration-deg-s2", "181"),
        ("--joint-acceleration-deg-s2", "nan"),
    ],
)
def test_invalid_response_or_gripper_settings_never_construct_hardware(monkeypatch, flag, value):
    monkeypatch.setattr("sys.argv", ["openarm_vr", "--execute", flag, value])
    run = Mock()
    monkeypatch.setattr(vr, "run", run)
    with pytest.raises(SystemExit):
        vr.main()
    run.assert_not_called()


@pytest.mark.parametrize("recovers", [True, False])
def test_startup_requires_sustained_quiet_feedback(monkeypatch, recovers):
    now = [0.0]
    stop = threading.Event()
    monkeypatch.setattr(vr.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(stop, "wait", lambda dt: now.__setitem__(0, now[0] + dt))
    backend = Mock()
    backend.validate_positions.side_effect = lambda q, **_: dict(q)

    def read():
        velocities = positions()
        if not recovers or now[0] < 0.06:
            velocities["left_joint_6.pos"] = 2.1
        return positions(), velocities

    backend.read.side_effect = read
    if recovers:
        assert vr.stationary_start(backend, vr.MotionLimits(), stop) == positions()
        assert now[0] >= 0.26
    else:
        with pytest.raises(RuntimeError, match="left_joint_6.pos"):
            vr.stationary_start(backend, vr.MotionLimits(), stop)
    backend.enable.assert_not_called()

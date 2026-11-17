"""Recovery from arbitrary in-range poses and small overshoot, without hardware."""

import math
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from lerobot.scripts import openarm_home as home


@pytest.fixture
def runtime(monkeypatch):
    args = home.build_parser().parse_args(["--execute"])
    backend = home.RecoveryBackend.__new__(home.RecoveryBackend)
    backend.limits = home.MotionLimits()
    backend.recovery_ranges = None
    backend.last_positions = None
    ranges = {f"joint_{i}": (-85.0, 85.0) for i in range(1, 8)}
    ranges.update(joint_4=(0.0, 135.0), gripper=(-65.0, 0.0))
    backend.arms = {
        side: SimpleNamespace(config=SimpleNamespace(joint_limits=dict(ranges)), bus=Mock())
        for side in ("left", "right")
    }
    measured = {f"{side}_{motor}.pos": 0.0 for side in backend.arms for motor in ranges}
    measured.update(
        {
            "right_joint_3.pos": -85.559,
            "left_joint_4.pos": 90.0,
            "left_gripper.pos": -25.0,
            "right_gripper.pos": -65.0,
        }
    )
    now = [0.0]
    monkeypatch.setattr("lerobot.scripts.openarm_vr.time.monotonic", lambda: now[0])
    monkeypatch.setattr(threading.Event, "wait", lambda self, dt: now.__setitem__(0, now[0] + dt))
    monkeypatch.setattr(
        home.DamiaoPreparationBackend, "read", lambda self: (dict(measured), dict.fromkeys(measured, 0.0))
    )
    backend.connect = Mock()
    backend.enable = Mock()
    backend.close = Mock()
    sent = []

    def send(command):
        backend.validate_positions(command)
        sent.append((now[0], dict(command)))
        measured.update(command)

    backend.send = Mock(side_effect=send)
    return args, backend, measured, sent


def test_full_recovery_from_reported_j3_overshoot_returns_all_joints_and_disables(runtime):
    args, backend, measured, sent = runtime
    home.run(args, threading.Event(), lambda *_: backend)
    initial = backend.enable.call_args.args[0]
    assert initial["right_joint_3.pos"] == -85.0
    assert initial["left_joint_4.pos"] == 90.0
    assert all(value == 0 for key, value in measured.items() if "_joint_" in key)
    assert measured["left_gripper.pos"] == 0.0
    assert measured["right_gripper.pos"] == 0.0
    assert sent[-1][0] >= 31.0
    assert sent[0][1]["left_gripper.pos"] == -25.0
    assert sent[0][1]["right_gripper.pos"] == -65.0
    for (t0, q0), (t1, q1) in zip(sent[:-1], sent[1:], strict=True):
        assert max(abs(q1[k] - q0[k]) for k in q0) <= 10 * (t1 - t0) + 1e-7
        assert q1["right_joint_3.pos"] >= q0["right_joint_3.pos"]
        for side in ("left", "right"):
            key = f"{side}_gripper.pos"
            assert q0[key] <= q1[key] <= 0.0
    backend.close.assert_called_once_with(disable_torque=True)
    for arm in backend.arms.values():
        arm.bus.set_zero_position.assert_not_called()


def test_obstructed_gripper_stops_recovery_without_disabling(runtime):
    args, backend, measured, _ = runtime
    send = backend.send.side_effect

    def obstructed_send(command):
        send(command)
        measured["left_gripper.pos"] = -25.0

    backend.send.side_effect = obstructed_send
    with pytest.raises(RuntimeError, match="Following error.*left_gripper"):
        home.run(args, threading.Event(), lambda *_: backend)
    backend.close.assert_called_once_with(disable_torque=False)


@pytest.mark.parametrize("value", [-95.001, 95.001, math.nan, math.inf])
def test_large_or_invalid_feedback_never_enables_or_disables(runtime, value):
    args, backend, measured, _ = runtime
    measured["right_joint_3.pos"] = value
    with pytest.raises(RuntimeError):
        home.run(args, threading.Event(), lambda *_: backend)
    backend.enable.assert_not_called()
    backend.send.assert_not_called()
    backend.close.assert_called_once_with(disable_torque=False)


def test_feedback_allowance_does_not_relax_motor_commands_or_normal_vr(runtime):
    _, backend, measured, _ = runtime
    checked = backend.validate_positions(measured, measurement=True)
    assert checked["right_joint_3.pos"] == -85.0
    assert measured["right_joint_3.pos"] == -85.559
    with pytest.raises(RuntimeError):
        backend.validate_positions(measured)
    with pytest.raises(RuntimeError):
        home.DamiaoPreparationBackend.validate_positions(backend, measured, measurement=True)


@pytest.mark.parametrize("start", [-85.559, -86.433, -88.444])
def test_recovery_accepts_reported_takeover_deflection_then_returns_to_zero(runtime, start):
    args, backend, measured, _ = runtime
    measured["right_joint_3.pos"] = start
    # Replay the observed 0.874-degree additional outward deflection at takeover.
    backend.enable.side_effect = lambda _: measured.update({"right_joint_3.pos": start - 0.874})
    home.run(args, threading.Event(), lambda *_: backend)
    assert measured["right_joint_3.pos"] == 0.0
    backend.close.assert_called_once_with(disable_torque=True)


def test_outside_operating_limit_is_read_before_validating_initial_command(runtime):
    _, backend, measured, _ = runtime
    measured["right_joint_3.pos"] = -96.0
    raw, _ = backend.read()
    assert raw["right_joint_3.pos"] == -96.0
    assert backend.validate_positions(raw, measurement=True)["right_joint_3.pos"] == -85.0
    with pytest.raises(RuntimeError, match="initial following error"):
        backend.begin_recovery()
    backend.enable.assert_not_called()


def test_recovery_envelope_only_tightens_and_gripper_tolerance_is_unchanged(runtime):
    _, backend, measured, _ = runtime
    measured["right_joint_3.pos"] = -86.433
    backend.read()
    backend.begin_recovery()
    measured["right_joint_3.pos"] = -87.0
    backend.read()  # Does not expand the envelope after this outward deflection.
    assert backend.recovery_ranges["right_joint_3.pos"][0] == -86.433
    measured["right_joint_3.pos"] = -85.2
    backend.read()
    assert backend.recovery_ranges["right_joint_3.pos"][0] == -85.2
    measured["right_joint_3.pos"] = -86.3
    with pytest.raises(RuntimeError, match="Outward drift"):
        backend.read()
    measured["right_joint_3.pos"] = -85.0
    measured["left_gripper.pos"] = 0.501
    with pytest.raises(RuntimeError):
        backend.read()


@pytest.mark.parametrize("phase", ["cancel", "stale", "overshoot", "stuck"])
def test_recovery_fault_or_cancel_preserves_torque(runtime, phase):
    args, backend, measured, _ = runtime
    stop = threading.Event()
    calls = [0]

    def send(command):
        calls[0] += 1
        if phase == "cancel":
            stop.set()
        elif phase == "stale":
            raise TimeoutError("feedback lost")
        elif phase == "overshoot":
            measured["right_joint_3.pos"] = -87.0
        # 'stuck' keeps fresh feedback at the original pose, causing following error.

    backend.send.side_effect = send
    if phase == "cancel":
        home.run(args, stop, lambda *_: backend)
    else:
        with pytest.raises((RuntimeError, TimeoutError)):
            home.run(args, stop, lambda *_: backend)
    assert calls[0] > 0
    backend.close.assert_called_once_with(disable_torque=False)


def test_missing_feedback_rejected(runtime):
    args, backend, measured, _ = runtime
    del measured["left_joint_1.pos"]
    with pytest.raises(RuntimeError, match="incomplete"):
        home.run(args, threading.Event(), lambda *_: backend)
    backend.enable.assert_not_called()


def test_preview_never_constructs_hardware(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["openarm_home"])
    run = Mock()
    monkeypatch.setattr(home, "run", run)
    home.main()
    run.assert_not_called()
    assert "PREVIEW ONLY" in capsys.readouterr().out


@pytest.mark.parametrize(
    "option,value", [("--speed-deg-s", "11"), ("--accel-deg-s2", "6"), ("--duration-s", "nan")]
)
def test_invalid_motion_limits_never_construct_hardware(monkeypatch, option, value):
    monkeypatch.setattr("sys.argv", ["openarm_home", "--execute", option, value])
    run = Mock()
    monkeypatch.setattr(home, "run", run)
    with pytest.raises((SystemExit, ValueError)):
        home.main()
    run.assert_not_called()

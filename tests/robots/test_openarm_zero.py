"""One-time zero setting, entirely with fake buses and feedback."""

import math
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from lerobot.scripts import openarm_zero as zero


@pytest.fixture
def runtime(monkeypatch):
    now = [0.0]
    stop = threading.Event()
    monkeypatch.setattr(zero.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(stop, "wait", lambda dt: now.__setitem__(0, now[0] + dt))
    args = zero.build_parser().parse_args(["--execute"])
    motors = [*[f"joint_{i}" for i in range(1, 8)], "gripper"]
    backend = Mock()
    backend.arms = {
        side: SimpleNamespace(bus=Mock(motors=dict.fromkeys(motors))) for side in ("left", "right")
    }
    # Deliberately outside normal limits: setting zero must work before zero is correct.
    positions = {f"{side}_{motor}.pos": -8.0 for side in backend.arms for motor in motors}
    velocities = dict.fromkeys(positions, 0.0)
    backend.read.side_effect = lambda: (dict(positions), dict(velocities))
    events = []
    for side, arm in backend.arms.items():
        arm.bus.disable_torque.side_effect = lambda *, motors, side=side: events.append(("disable", side))

        def write(*, motors, side=side):
            events.append(("zero", side, motors[0]))
            positions[f"{side}_{motors[0]}.pos"] = -0.0109

        arm.bus.set_zero_position.side_effect = write
    return args, stop, backend, positions, velocities, events


@pytest.mark.parametrize("include_grippers", [False, True])
def test_zero_disables_both_arms_before_writing_once_and_verifies_without_motion(runtime, include_grippers):
    args, stop, backend, positions, _, events = runtime
    args.include_grippers = include_grippers
    zero.run(args, stop, lambda *_: backend)
    assert events[:2] == [("disable", "left"), ("disable", "right")]
    selected = [f"joint_{i}" for i in range(1, 8)] + (["gripper"] if include_grippers else [])
    for side, arm in backend.arms.items():
        arm.bus.disable_torque.assert_called_once_with(motors=selected)
        assert [call.kwargs["motors"] for call in arm.bus.set_zero_position.call_args_list] == [
            [m] for m in selected
        ]
        arm.bus.enable_torque.assert_not_called()
        assert positions[f"{side}_gripper.pos"] == (-0.0109 if include_grippers else -8.0)
    backend.enable.assert_not_called()
    backend.send.assert_not_called()
    backend.close.assert_called_once_with(disable_torque=False)


@pytest.mark.parametrize("failure", ["moving", "invalid", "missing", "stale", "cancel"])
def test_failed_preflight_never_disables_or_writes_zero(runtime, failure):
    args, stop, backend, positions, velocities, events = runtime
    if failure == "moving":
        velocities["left_joint_1.pos"] = 3.0
    elif failure == "invalid":
        positions["left_joint_1.pos"] = math.nan
    elif failure == "missing":
        del positions["left_joint_1.pos"]
    elif failure == "stale":
        backend.read.side_effect = TimeoutError("stale")
    else:
        stop.set()
    with pytest.raises((RuntimeError, OSError)):
        zero.run(args, stop, lambda *_: backend)
    assert not events
    backend.close.assert_called_once_with(disable_torque=False)


def test_motion_after_disabling_blocks_zero_write(runtime):
    args, stop, backend, _, velocities, _ = runtime
    backend.arms["right"].bus.disable_torque.side_effect = lambda **_: velocities.update(
        {"left_joint_1.pos": 3.0}
    )
    with pytest.raises(RuntimeError, match="stationary"):
        zero.run(args, stop, lambda *_: backend)
    for arm in backend.arms.values():
        arm.bus.set_zero_position.assert_not_called()


def test_ignored_zero_command_fails_readback_and_never_enables(runtime):
    args, stop, backend, _, _, _ = runtime
    backend.arms["left"].bus.set_zero_position.side_effect = None
    with pytest.raises(RuntimeError, match="readback"):
        zero.run(args, stop, lambda *_: backend)
    backend.enable.assert_not_called()
    backend.close.assert_called_once_with(disable_torque=False)


def test_partial_write_failure_stops_without_zeroing_other_arm(runtime):
    args, stop, backend, _, _, _ = runtime
    backend.arms["left"].bus.set_zero_position.side_effect = ConnectionError("lost CAN")
    with pytest.raises(ConnectionError):
        zero.run(args, stop, lambda *_: backend)
    backend.arms["right"].bus.set_zero_position.assert_not_called()
    backend.close.assert_called_once_with(disable_torque=False)


def test_preview_does_not_open_hardware(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["openarm_zero"])
    run = Mock()
    monkeypatch.setattr(zero, "run", run)
    zero.main()
    run.assert_not_called()
    assert "PREVIEW ONLY" in capsys.readouterr().out

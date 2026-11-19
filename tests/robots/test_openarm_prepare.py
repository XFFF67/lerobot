"""Hardware-free preparation tests; no CAN sockets or motor enables."""

import math
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from lerobot.scripts import openarm_prepare as prepare


def arm_positions():
    return {
        f"{side}_{joint}.pos": 0.0
        for side in ("left", "right")
        for joint in (*[f"joint_{i}" for i in range(1, 8)], "gripper")
    }


def feed_window(monitor, start, goal, speeds, *, measured=None):
    """Supply a full 50 Hz window, not two isolated feedback snapshots."""
    for i in range(51):
        ready = monitor.update(start + i * 0.02, measured or goal, speeds, goal)
    return ready


def test_support_gains_preserve_other_joints_and_original_config():
    config = SimpleNamespace(
        position_kp=[50, 50, 50, 50, 25, 8, 8, 8],
        position_kd=[1, 1, 1, 1, 0.5, 0.2, 0.2, 0.2],
    )
    kp, kd = prepare.support_gains(config, 4)
    assert kp == {"joint_1": 200, "joint_4": 200, "joint_7": 32}
    assert kd == {"joint_1": 2, "joint_4": 2, "joint_7": 0.4}
    assert config.position_kp == [50, 50, 50, 50, 25, 8, 8, 8]
    assert prepare.support_gains(SimpleNamespace(position_kp=50, position_kd=1), 1)[0]["joint_4"] == 50


@pytest.mark.parametrize("scale", ["0", "-1", "4.1", "nan", "inf"])
def test_invalid_support_gain_scale_rejected_before_hardware(scale):
    with pytest.raises(SystemExit):
        prepare.build_parser().parse_args(["--execute", "--support-gain-scale", scale])


def test_send_applies_preparation_gains_without_changing_targets():
    backend = prepare.DamiaoPreparationBackend.__new__(prepare.DamiaoPreparationBackend)
    arm = Mock()
    arm.config = SimpleNamespace(position_kp=50, position_kd=1)
    backend.arms = {"left": arm}
    backend.support_gain_scale = 2
    backend.validate_positions = Mock()
    backend.send({"left_joint_4.pos": 90})
    arm.send_action.assert_called_once_with(
        {"joint_4.pos": 90},
        custom_kp={"joint_1": 100, "joint_4": 100, "joint_7": 100},
        custom_kd=dict.fromkeys(("joint_1", "joint_4", "joint_7"), math.sqrt(2)),
    )


def test_stronger_vr_gripper_changes_only_gains_and_preserves_target():
    config = SimpleNamespace(
        position_kp=[50, 50, 50, 50, 25, 8, 8, 8], position_kd=[1, 1, 1, 1, 0.5, 0.2, 0.2, 0.2]
    )
    backend = prepare.DamiaoPreparationBackend.__new__(prepare.DamiaoPreparationBackend)
    backend.arms = {side: Mock(config=config) for side in ("left", "right")}
    backend.support_gain_scale = 4
    backend.gripper_gain_scale = 1.5
    backend.validate_positions = Mock()
    backend.send({"left_gripper.pos": -25, "right_gripper.pos": -50})
    for side, angle in (("left", -25), ("right", -50)):
        call = backend.arms[side].send_action.call_args
        assert call.args == ({"gripper.pos": angle},)
        assert call.kwargs["custom_kp"] == {"joint_1": 200, "joint_4": 200, "joint_7": 32, "gripper": 12}
        assert call.kwargs["custom_kd"]["gripper"] == pytest.approx(0.2 * math.sqrt(1.5))
    assert config.position_kp[-1] == 8 and config.position_kd[-1] == 0.2


def test_trajectory_holds_other_joints_and_obeys_speed_acceleration_limits():
    initial = arm_positions()
    initial["left_joint_2.pos"] = -20
    initial["right_gripper.pos"] = -30
    limits = prepare.MotionLimits(duration_s=0.1)
    goal = prepare.preparation_goal(initial)
    trajectory = prepare.PreparationTrajectory(initial, goal, limits)
    assert trajectory.sample(0) == initial
    assert trajectory.sample(trajectory.duration_s) == goal
    assert trajectory.duration_s >= 1.875 * 90 / limits.speed_deg_s
    dt = trajectory.duration_s / 1000
    samples = [trajectory.sample(i * dt) for i in range(1001)]
    for key in initial:
        values = [sample[key] for sample in samples]
        if not key.endswith("_joint_4.pos"):
            assert values == [initial[key]] * len(values)
        speeds = [(b - a) / dt for a, b in zip(values, values[1:], strict=False)]
        accelerations = [(b - a) / dt for a, b in zip(speeds, speeds[1:], strict=False)]
        assert max(map(abs, speeds)) <= limits.speed_deg_s + 1e-6
        assert max(map(abs, accelerations)) <= limits.acceleration_deg_s2 + 1e-6
        assert abs(speeds[0]) < 0.001
        assert abs(speeds[-1]) < 0.001


def test_ready_requires_measured_settled_arrival():
    start = arm_positions()
    goal = prepare.preparation_goal(start)
    limits = prepare.MotionLimits()
    trajectory = prepare.PreparationTrajectory(start, goal, limits)
    monitor = prepare.PreparationMonitor(trajectory, limits)
    speed = dict.fromkeys(start, 0.0)
    lagging = dict(goal, **{"left_joint_4.pos": 85.0})
    assert not monitor.update(20, lagging, speed, goal)
    assert feed_window(monitor, 21, goal, speed)
    speed["left_joint_4.pos"] = 3.0
    assert not monitor.update(23, goal, speed, goal)


def test_lower_trajectory_never_rounds_below_zero_at_endpoint():
    start = {"left_joint_4.pos": 86.9797, "right_joint_4.pos": 87.4824}
    goal = dict.fromkeys(start, 0.0)
    trajectory = prepare.PreparationTrajectory(start, goal, prepare.MotionLimits(duration_s=30))
    for elapsed in [0, 15, 29, *[30 - 10**-i for i in range(1, 16)], 30, 31]:
        sample = trajectory.sample(elapsed)
        assert all(0 <= sample[key] <= start[key] for key in start)
    assert trajectory.sample(30) == goal
    assert trajectory.sample(31) == goal


def test_feedback_failure_and_unreachable_target_never_report_ready():
    start = arm_positions()
    limits = prepare.MotionLimits()
    trajectory = prepare.PreparationTrajectory(start, prepare.preparation_goal(start), limits)
    monitor = prepare.PreparationMonitor(trajectory, limits)
    speeds = dict.fromkeys(start, 0.0)
    with pytest.raises(RuntimeError, match="Incomplete"):
        monitor.update(0, {}, speeds, start)
    with pytest.raises(RuntimeError, match="Non-finite"):
        monitor.update(0, dict(start, **{"left_joint_4.pos": math.nan}), speeds, start)
    with pytest.raises(RuntimeError, match="Following error"):
        monitor.update(20, start, speeds, trajectory.goal)
    lagging = dict(trajectory.goal, **{"left_joint_4.pos": 85.0})
    with pytest.raises(RuntimeError, match="did not settle"):
        monitor.update(31, lagging, speeds, trajectory.goal)


@pytest.mark.parametrize("position_error,velocity", [(2.1, 0.0), (0.0, 2.1)])
def test_holding_transient_after_initial_deadline_can_recover(position_error, velocity):
    start = arm_positions()
    goal = prepare.preparation_goal(start)
    limits = prepare.MotionLimits(duration_s=30)
    monitor = prepare.PreparationMonitor(prepare.PreparationTrajectory(start, goal, limits), limits)
    speeds = dict.fromkeys(start, 0.0)
    assert feed_window(monitor, 30, goal, speeds)
    disturbed = dict(goal, **{"right_joint_4.pos": 90 - position_error})
    disturbed_speeds = dict(speeds, **{"right_joint_4.pos": velocity})
    # Reproduce transient loss after 30 + 10 seconds. It is not an arrival failure.
    assert not monitor.update(42, disturbed, disturbed_speeds, goal)
    assert feed_window(monitor, 42.1, goal, speeds)
    # The holding deviation timer resets after recovery, even much later.
    assert not monitor.update(100, disturbed, disturbed_speeds, goal)
    assert feed_window(monitor, 100.1, goal, speeds)


def test_holding_sustained_departure_and_large_errors_still_fault():
    start = arm_positions()
    goal = prepare.preparation_goal(start)
    limits = prepare.MotionLimits()
    monitor = prepare.PreparationMonitor(prepare.PreparationTrajectory(start, goal, limits), limits)
    speeds = dict.fromkeys(start, 0.0)
    assert feed_window(monitor, 20, goal, speeds)
    drift = dict(goal, **{"right_joint_4.pos": 87.0})
    assert not monitor.update(42, drift, speeds, goal)
    assert not monitor.update(51.9, drift, speeds, goal)
    with pytest.raises(RuntimeError, match="holding tolerance.*right_joint_4.pos"):
        monitor.update(52, drift, speeds, goal)
    with pytest.raises(RuntimeError, match="Following error"):
        monitor.update(52.1, start, speeds, goal)


def test_stationary_damiao_velocity_spikes_do_not_reset_settling():
    start = arm_positions()
    goal = prepare.preparation_goal(start)
    limits = prepare.MotionLimits()
    monitor = prepare.PreparationMonitor(prepare.PreparationTrajectory(start, goal, limits), limits)
    measured = dict(goal, **{"left_joint_4.pos": 88.25, "right_joint_4.pos": 88.29})
    for i in range(51):
        speeds = dict.fromkeys(start, -0.6996)
        if i % 10 == 5:
            speeds["left_joint_6.pos"] = -2.0987
        ready = monitor.update(20 + i * 0.02, measured, speeds, goal)
        if i < 50:
            assert not ready
    assert ready


@pytest.mark.parametrize("kind", ["drift", "oscillation", "speed", "large_spike", "gap", "position_error"])
def test_settling_rejects_motion_or_inadequate_evidence(kind):
    start = arm_positions()
    goal = prepare.preparation_goal(start)
    limits = prepare.MotionLimits()
    monitor = prepare.PreparationMonitor(prepare.PreparationTrajectory(start, goal, limits), limits)
    key = "left_joint_6.pos"
    for i in range(51):
        measured = dict(goal)
        speeds = dict.fromkeys(start, 0.0)
        t = 20 + i * 0.02
        if kind == "drift":
            measured[key] = i * 0.01
        elif kind == "oscillation":
            measured[key] = 0.2 if i % 2 else -0.2
        elif kind == "speed":
            speeds[key] = 3.0 if i % 2 else -3.0
        elif kind == "large_spike" and i == 25:
            speeds[key] = 5.1
        elif kind == "gap" and i >= 25:
            t += 0.3
        elif kind == "position_error":
            measured[key] = 2.1
        assert not monitor.update(t, measured, speeds, goal)
    assert monitor.settle_detail


@pytest.mark.parametrize("value", [0, -1, math.nan, math.inf])
def test_invalid_motion_limits(value):
    with pytest.raises(ValueError):
        prepare.MotionLimits(duration_s=value)


def test_preview_does_not_construct_hardware(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["openarm_prepare"])
    factory = Mock(side_effect=AssertionError("Must not construct real hardware"))
    monkeypatch.setattr(prepare, "DamiaoPreparationBackend", factory)
    prepare.main()
    factory.assert_not_called()
    assert "PREVIEW ONLY" in capsys.readouterr().out


def test_backend_opens_without_enabling_handshake_or_zeroing():
    backend = prepare.DamiaoPreparationBackend.__new__(prepare.DamiaoPreparationBackend)
    backend.arms = {side: Mock(is_calibrated=True) for side in ("left", "right")}
    backend.connect()
    for arm in backend.arms.values():
        arm.bus.connect.assert_called_once_with(handshake=False)
        arm.connect.assert_not_called()
        arm.bus.enable_torque.assert_not_called()
        arm.bus.set_zero_position.assert_not_called()


def position_validation_backend():
    backend = prepare.DamiaoPreparationBackend.__new__(prepare.DamiaoPreparationBackend)
    backend.arms = {
        side: SimpleNamespace(
            config=SimpleNamespace(joint_limits={"joint_4": (0.0, 135.0), "gripper": (-65.0, 0.0)})
        )
        for side in ("left", "right")
    }
    return backend


@pytest.mark.parametrize("measured,target", [(0.20764091911737245, 0.0), (0.5, 0.0), (-65.2, -65.0)])
def test_gripper_endpoint_offset_is_clipped_without_changing_raw_feedback(measured, target, caplog):
    backend = position_validation_backend()
    positions = arm_positions()
    positions["left_gripper.pos"] = measured
    checked = backend.validate_positions(positions, measurement=True)
    assert checked["left_gripper.pos"] == target
    assert positions["left_gripper.pos"] == measured
    assert "holding target clipped" in caplog.text
    # The resulting preparation and return targets remain valid commands.
    assert backend.validate_positions(checked) == checked
    goal = prepare.preparation_goal(checked)
    assert backend.validate_positions(goal) == goal
    assert goal["left_joint_4.pos"] == 90.0
    assert goal["left_gripper.pos"] == target


@pytest.mark.parametrize("measured", [0.501, -65.501, math.nan, math.inf])
def test_gripper_large_or_invalid_offset_still_rejected(measured):
    backend = position_validation_backend()
    positions = arm_positions()
    positions["left_gripper.pos"] = measured
    with pytest.raises(RuntimeError, match="left_gripper.pos"):
        backend.validate_positions(positions, measurement=True)


def test_gripper_measurement_tolerance_does_not_relax_commands_or_arm_limits():
    backend = position_validation_backend()
    positions = arm_positions()
    positions["left_gripper.pos"] = 0.20764091911737245
    with pytest.raises(RuntimeError, match="left_gripper.pos"):
        backend.validate_positions(positions)
    positions["left_gripper.pos"] = 0.0
    positions["left_joint_4.pos"] = 135.20764091911737
    with pytest.raises(RuntimeError, match="left_joint_4.pos"):
        backend.validate_positions(positions, measurement=True)


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("measured", [-0.16392704140845193, -0.5])
def test_lowered_elbow_offset_produces_valid_start_prepare_and_return_targets(side, measured):
    backend = position_validation_backend()
    positions = arm_positions()
    key = f"{side}_joint_4.pos"
    positions[key] = measured
    checked = backend.validate_positions(positions, measurement=True)
    assert positions[key] == measured
    assert checked[key] == 0.0
    assert backend.validate_positions(checked) == checked
    goal = backend.validate_positions(prepare.preparation_goal(checked))
    assert goal[key] == 90.0
    with pytest.raises(RuntimeError, match=key):
        backend.validate_positions(positions)


@pytest.mark.parametrize("measured", [-0.501, 135.101, math.nan, math.inf, -math.inf])
def test_elbow_zero_tolerance_still_rejects_large_upper_or_invalid_offsets(measured):
    backend = position_validation_backend()
    positions = arm_positions()
    positions["left_joint_4.pos"] = measured
    with pytest.raises(RuntimeError, match="left_joint_4.pos"):
        backend.validate_positions(positions, measurement=True)


def test_elbow_zero_tolerance_does_not_expand_other_joint_feedback_limits():
    backend = position_validation_backend()
    backend.arms["left"].config.joint_limits["joint_6"] = (-40.0, 40.0)
    positions = arm_positions()
    positions["left_joint_6.pos"] = -40.164
    with pytest.raises(RuntimeError, match="left_joint_6.pos"):
        backend.validate_positions(positions, measurement=True)


def test_right_elbow_negative_offset_remains_blocked_and_all_errors_reported():
    backend = position_validation_backend()
    positions = arm_positions()
    positions["right_joint_4.pos"] = -7.354859924525877
    positions["left_joint_4.pos"] = -2.0
    with pytest.raises(RuntimeError) as exc:
        backend.validate_positions(positions, measurement=True)
    assert "right_joint_4.pos=-7.354859924525877" in str(exc.value)
    assert "left_joint_4.pos=-2.0" in str(exc.value)


def test_startup_homing_accepts_joint_excursion_but_only_allows_inward_commands():
    backend = position_validation_backend()
    positions = arm_positions()
    positions["left_joint_4.pos"] = -1.1474892898591635
    initial = backend.validate_positions(
        positions,
        measurement=True,
        allow_joint_excursions=True,
    )
    assert initial["left_joint_4.pos"] == positions["left_joint_4.pos"]
    home = backend.validate_positions(prepare.home_goal(initial))
    assert home["left_joint_4.pos"] == 0.0
    assert backend.validate_recovery_positions(initial, initial) == initial
    assert backend.validate_recovery_positions(home, initial) == home
    outward = dict(initial)
    outward["left_joint_4.pos"] -= 0.01
    with pytest.raises(RuntimeError, match="moves outward"):
        backend.validate_recovery_positions(outward, initial)


def test_read_only_inspection_does_not_enable_disable_move_or_clip_feedback(capsys):
    backend = fake_backend()
    backend.arms = position_validation_backend().arms
    backend.read.return_value[0]["right_joint_4.pos"] = -7.354859924525877
    prepare.inspect_feedback(SimpleNamespace(), prepare.MotionLimits(), lambda *_: backend)
    backend.enable.assert_not_called()
    backend.send.assert_not_called()
    backend.validate_positions.assert_not_called()
    backend.close.assert_called_once_with(disable_torque=False)
    assert "-7.3549" in capsys.readouterr().out


def test_read_only_cannot_be_combined_with_execution():
    with pytest.raises(SystemExit):
        prepare.build_parser().parse_args(["--read-state", "--execute"])


@pytest.mark.parametrize("other", ["--execute", "--read-state"])
def test_lower_is_exclusive(other):
    with pytest.raises(SystemExit):
        prepare.build_parser().parse_args(["--lower", other])


def test_lower_reaches_zero_before_disabling_without_returning_up(monkeypatch):
    backend = fake_backend()
    initial = backend.read.return_value[0]
    initial["left_joint_4.pos"] = 87
    initial["right_joint_4.pos"] = 88
    initial["left_joint_3.pos"] = -9
    motion = Mock(return_value=None)
    monkeypatch.setattr(prepare, "run_motion", motion)
    prepare.lower(SimpleNamespace(), prepare.MotionLimits(), threading.Event(), lambda *_: backend)
    motion.assert_called_once()
    start, goal = motion.call_args.args[1:3]
    assert start == initial
    assert goal == prepare.preparation_goal(initial, 0)
    assert motion.call_args.kwargs == {"hold": False}
    backend.close.assert_called_once_with(disable_torque=True)


@pytest.mark.parametrize("phase", ["read", "motion", "interrupt"])
def test_failed_or_interrupted_lower_never_disables_existing_torque(monkeypatch, phase):
    backend = fake_backend()
    if phase == "read":
        backend.read.side_effect = ConnectionError("feedback")
    motion = Mock(return_value=arm_positions())
    if phase == "motion":
        motion.side_effect = RuntimeError("tracking")
    monkeypatch.setattr(prepare, "run_motion", motion)
    if phase == "interrupt":
        prepare.lower(SimpleNamespace(), prepare.MotionLimits(), threading.Event(), lambda *_: backend)
    else:
        with pytest.raises((RuntimeError, ConnectionError)):
            prepare.lower(SimpleNamespace(), prepare.MotionLimits(), threading.Event(), lambda *_: backend)
    backend.close.assert_called_once_with(disable_torque=False)


def fake_backend():
    backend = Mock(enabled=False)
    initial = arm_positions()
    backend.read.return_value = (initial, dict.fromkeys(initial, 0.0))
    backend.validate_positions.side_effect = lambda positions, **_: dict(positions)

    def enable(_):
        backend.enabled = True

    backend.enable.side_effect = enable
    return backend


def test_failed_startup_feedback_does_not_enable(monkeypatch):
    backend = fake_backend()
    backend.read.side_effect = ConnectionError("missing joint")
    with pytest.raises(ConnectionError):
        prepare.execute(
            SimpleNamespace(elbow_deg=90), prepare.MotionLimits(), threading.Event(), lambda *_: backend
        )
    backend.enable.assert_not_called()
    backend.close.assert_called_once_with(disable_torque=True)


def test_normal_stop_lowers_to_start_before_disabling(monkeypatch):
    backend = fake_backend()
    initial, _ = backend.read.return_value
    motions = []

    def run(backend, start, goal, limits, stop, *, hold):
        motions.append((start, goal, hold))
        return dict(goal)

    monkeypatch.setattr(prepare, "run_motion", run)
    prepare.execute(
        SimpleNamespace(elbow_deg=90), prepare.MotionLimits(), threading.Event(), lambda *_: backend
    )
    assert motions[0][1] == prepare.preparation_goal(initial)
    assert motions[0][2] is True
    assert motions[1][1] == initial
    # Feedback still equals initial in this fake. Return must start at the last
    # target, not the lagging measured pose, to avoid a position command step.
    assert motions[1][0] == motions[0][1]
    assert motions[1][2] is False
    backend.close.assert_called_once_with(disable_torque=True)


def test_motion_fault_holds_measured_pose_without_cutting_torque(monkeypatch):
    backend = fake_backend()
    monkeypatch.setattr(prepare, "run_motion", Mock(side_effect=RuntimeError("tracking fault")))
    with pytest.raises(RuntimeError, match="tracking fault"):
        prepare.execute(
            SimpleNamespace(elbow_deg=90), prepare.MotionLimits(), threading.Event(), lambda *_: backend
        )
    backend.send.assert_called_once_with(backend.read.return_value[0])
    backend.close.assert_called_once_with(disable_torque=False)


def test_motion_loop_reaches_ready_then_can_be_stopped(monkeypatch):
    now = [0.0]
    stop = threading.Event()
    backend = fake_backend()
    initial = arm_positions()
    goal = prepare.preparation_goal(initial)
    commanded = [dict(initial)]
    backend.send.side_effect = lambda action: commanded.__setitem__(0, dict(action))
    backend.read.side_effect = lambda: (dict(commanded[0]), dict.fromkeys(initial, 0.0))
    monkeypatch.setattr(prepare.time, "monotonic", lambda: now[0])

    def wait(dt):
        now[0] += dt
        if now[0] > 22:
            stop.set()

    monkeypatch.setattr(stop, "wait", wait)
    last = prepare.run_motion(backend, initial, goal, prepare.MotionLimits(), stop, hold=True)
    assert last == goal
    assert commanded[0] == goal


def test_timed_motion_ignores_loaded_offset_and_quantized_velocity(monkeypatch):
    now = [0.0]
    backend = fake_backend()
    initial = arm_positions()
    goal = prepare.home_goal(initial)
    commanded = [dict(initial)]

    def read():
        measured = dict(commanded[0])
        measured["left_joint_6.pos"] += 2.065
        velocities = dict.fromkeys(initial, 0.0)
        velocities["left_joint_5.pos"] = -2.099
        return measured, velocities

    backend.read.side_effect = read
    backend.send.side_effect = lambda action: commanded.__setitem__(0, dict(action))
    monkeypatch.setattr(prepare.time, "monotonic", lambda: now[0])
    stop = threading.Event()
    monkeypatch.setattr(stop, "wait", lambda dt: now.__setitem__(0, now[0] + dt))
    prepare.run_motion(
        backend,
        initial,
        goal,
        prepare.MotionLimits(duration_s=0.1),
        stop,
        hold=False,
        phase="HOMING",
        verify_arrival=False,
    )
    assert commanded[0] == goal


def test_timed_motion_retains_hard_following_error(monkeypatch):
    backend = fake_backend()
    initial = arm_positions()
    measured = dict(initial)
    measured["left_joint_6.pos"] += 10.1
    backend.read.return_value = (measured, dict.fromkeys(initial, 0.0))
    monkeypatch.setattr(prepare.time, "monotonic", lambda: 0.0)
    with pytest.raises(RuntimeError, match="Following error exceeds 10.0 degrees"):
        prepare.run_motion(
            backend,
            initial,
            prepare.home_goal(initial),
            prepare.MotionLimits(),
            threading.Event(),
            hold=False,
            verify_arrival=False,
        )
    backend.send.assert_not_called()


def test_stalled_control_loop_faults_before_sending(monkeypatch):
    backend = fake_backend()
    ticks = iter([0.0, 0.5])
    monkeypatch.setattr(prepare.time, "monotonic", lambda: next(ticks))
    initial = arm_positions()
    with pytest.raises(RuntimeError, match="stalled"):
        prepare.run_motion(
            backend,
            initial,
            prepare.preparation_goal(initial),
            prepare.MotionLimits(),
            threading.Event(),
            hold=True,
        )
    backend.send.assert_not_called()

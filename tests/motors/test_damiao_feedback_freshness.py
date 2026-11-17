"""Reject uninitialized or stale cached feedback without accessing hardware."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from lerobot.motors.damiao.damiao import DamiaoMotorsBus


def make_bus():
    bus = DamiaoMotorsBus.__new__(DamiaoMotorsBus)
    bus.port = "test-only"
    bus._get_motors_list = Mock(return_value=["joint_1", "joint_4"])
    bus._batch_refresh = Mock()
    bus._last_known_states = {"joint_1": {"position": 0.0}, "joint_4": {"position": 45.0}}
    bus._last_state_times = {}
    return bus


def test_strict_read_rejects_initial_zero_cache():
    bus = make_bus()
    with pytest.raises(ConnectionError, match="joint_1, joint_4"):
        bus.sync_read_all_states(max_age_s=0.2)
    # Existing callers retain their original caching behavior.
    assert bus.sync_read_all_states()["joint_4"]["position"] == 45.0


def test_each_motor_must_have_fresh_feedback(monkeypatch):
    bus = make_bus()
    monkeypatch.setattr("lerobot.motors.damiao.damiao.time.monotonic", lambda: 10.0)
    bus._last_state_times = {"joint_1": 9.9, "joint_4": 9.0}
    with pytest.raises(ConnectionError, match="joint_4"):
        bus.sync_read_all_states(max_age_s=0.2)
    bus._last_state_times["joint_4"] = 9.9
    assert bus.sync_read_all_states(max_age_s=0.2)["joint_4"]["position"] == 45.0


def test_bad_packet_does_not_refresh_timestamp(monkeypatch):
    bus = make_bus()
    bus._motor_types = {"joint_4": object()}
    bus._decode_motor_state = Mock(side_effect=ValueError("invalid packet"))
    bus._process_response("joint_4", SimpleNamespace(data=b""))
    assert "joint_4" not in bus._last_state_times
    bus._decode_motor_state.side_effect = None
    bus._decode_motor_state.return_value = (45, 0, 0, 25, 25)
    monkeypatch.setattr("lerobot.motors.damiao.damiao.time.monotonic", lambda: 10.0)
    bus._process_response("joint_4", SimpleNamespace(data=b"valid"))
    assert bus._last_state_times["joint_4"] == 10.0

#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import time
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from lerobot.teleoperators.bi_hyopenarm_leader.bi_hyopenarm_leader import BiHyOpenArmLeader
from lerobot.teleoperators.hyopenarm_leader.hyopenarm_leader import HyOpenArmLeader


MOTOR_NAMES = [f"joint_{index}" for index in range(1, 8)] + ["gripper"]
JOINT_INDICES = {name: (index, index) for index, name in enumerate(MOTOR_NAMES[:7])}


def make_bimanual_controller() -> BiHyOpenArmLeader:
    controller = object.__new__(BiHyOpenArmLeader)
    controller._motor_names = MOTOR_NAMES
    controller.config = SimpleNamespace(gravity_compensation_scale=0.5)
    return controller


def test_motor_angles_are_mapped_to_urdf_coordinates() -> None:
    controller = make_bimanual_controller()
    q = np.zeros(7)
    states = {name: {"position": 20.0} for name in MOTOR_NAMES[:7]}

    controller._fill_pinocchio_q(
        q,
        states,
        JOINT_INDICES,
        angle_signs=[-1.0] + [1.0] * 6,
        zero_offsets_deg=[10.0] * 7,
    )

    assert q[0] == pytest.approx(math.radians(-10.0))
    assert q[1] == pytest.approx(math.radians(10.0))


def test_gravity_torque_includes_velocity_damping_and_coordinate_signs() -> None:
    controller = make_bimanual_controller()
    states = {name: {"velocity": 180.0} for name in MOTOR_NAMES[:7]}

    torques = controller._extract_pinocchio_torques(
        gravity=np.full(7, 2.0),
        indices=JOINT_INDICES,
        states=states,
        torque_signs=[-1.0] + [1.0] * 6,
        angle_signs=[1.0] * 7,
        damping=[0.1] * 7,
    )

    expected_model_torque = 1.0 - 0.1 * math.pi
    assert torques["joint_1"] == pytest.approx(-expected_model_torque)
    assert torques["joint_2"] == pytest.approx(expected_model_torque)
    assert "gripper" not in torques


def test_per_joint_gravity_scales_are_applied_independently() -> None:
    controller = make_bimanual_controller()
    states = {name: {"velocity": 0.0} for name in MOTOR_NAMES[:7]}

    torques = controller._extract_pinocchio_torques(
        gravity=np.full(7, 2.0),
        indices=JOINT_INDICES,
        states=states,
        torque_signs=[1.0] * 7,
        angle_signs=[1.0] * 7,
        damping=[0.0] * 7,
        scales=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    )

    assert torques["joint_1"] == pytest.approx(0.2)
    assert torques["joint_4"] == pytest.approx(0.8)
    assert torques["joint_7"] == pytest.approx(1.4)


def test_gravity_compensation_diagnostics_report_each_torque_term_and_limiting() -> None:
    controller = make_bimanual_controller()
    states = {name: {"velocity": 180.0} for name in MOTOR_NAMES[:7]}
    gravity = np.full(7, 2.0)
    damping = [0.1] * 7
    scales = [0.5] * 7
    torque_limits = [0.5] * 7
    torques = controller._extract_pinocchio_torques(
        gravity,
        JOINT_INDICES,
        states,
        torque_signs=[1.0] * 7,
        angle_signs=[1.0] * 7,
        damping=damping,
        scales=scales,
    )

    diagnostics = controller._build_gravity_compensation_diagnostics(
        gravity,
        JOINT_INDICES,
        states,
        torque_signs=[1.0] * 7,
        angle_signs=[1.0] * 7,
        damping=damping,
        scales=scales,
        torque_limits=torque_limits,
        torques=torques,
    )

    joint_1 = diagnostics["joint_1"]
    assert joint_1["raw_gravity_torque"] == pytest.approx(2.0)
    assert joint_1["scaled_gravity_torque"] == pytest.approx(1.0)
    assert joint_1["damping_torque"] == pytest.approx(-0.1 * math.pi)
    assert joint_1["unclamped_torque"] == pytest.approx(1.0 - 0.1 * math.pi)
    assert joint_1["limited_torque"] == pytest.approx(0.5)
    assert joint_1["is_limited"] is True


def test_per_joint_torque_limits_are_applied_independently() -> None:
    arm = object.__new__(HyOpenArmLeader)
    arm.config = SimpleNamespace(
        gravity_compensation_enabled=True,
        gravity_compensation_torque_limit=1.0,
    )
    arm.motor_to_bus_name = {name: "robstride" for name in MOTOR_NAMES[:2]}
    arm._send_mit_commands = Mock()

    arm.apply_feedforward_torques(
        states={"joint_1": {"position": 0.0}, "joint_2": {"position": 0.0}},
        torques={"joint_1": 2.0, "joint_2": -2.0},
        torque_limits={"joint_1": 0.2, "joint_2": 0.7},
    )

    commands = arm._send_mit_commands.call_args.args[0]["robstride"]
    assert commands["joint_1"][-1] == pytest.approx(0.2)
    assert commands["joint_2"][-1] == pytest.approx(-0.7)


def test_stale_feedback_is_rejected() -> None:
    arm = object.__new__(HyOpenArmLeader)
    arm._motor_names = MOTOR_NAMES
    arm.buses = {
        "robstride": SimpleNamespace(
            motors={"joint_1": object()}, last_feedback_time={"joint_1": time.time() - 1.0}
        ),
        "damiao": SimpleNamespace(motors={}, last_feedback_time={}),
    }

    with pytest.raises(RuntimeError, match="joint_1"):
        arm.assert_feedback_fresh(max_age_s=0.1)


def test_gripper_feedback_is_not_required_for_gravity_compensation() -> None:
    arm = object.__new__(HyOpenArmLeader)
    arm._motor_names = MOTOR_NAMES
    arm.buses = {
        "robstride": SimpleNamespace(motors={}, last_feedback_time={}),
        "damiao": SimpleNamespace(motors={"gripper": object()}, last_feedback_time={"gripper": None}),
    }

    arm.assert_feedback_fresh(max_age_s=0.1, motors=MOTOR_NAMES[:7])


def test_stopping_gravity_compensation_does_not_command_gripper() -> None:
    arm = object.__new__(HyOpenArmLeader)
    arm.config = SimpleNamespace(gravity_compensation_enabled=True)
    arm._motor_names = MOTOR_NAMES
    arm.motor_to_bus_name = {
        name: "robstride" if index < 4 else "damiao" for index, name in enumerate(MOTOR_NAMES)
    }
    arm._send_mit_commands = Mock()

    arm.stop_gravity_compensation()

    commands_by_bus = arm._send_mit_commands.call_args.args[0]
    assert "gripper" not in commands_by_bus["damiao"]
    assert set(commands_by_bus["robstride"] | commands_by_bus["damiao"]) == set(MOTOR_NAMES[:7])


def test_configure_keeps_gripper_torque_disabled() -> None:
    arm = object.__new__(HyOpenArmLeader)
    arm.config = SimpleNamespace(
        manual_control=True,
        gravity_compensation_enabled=True,
        motor_config=dict.fromkeys(MOTOR_NAMES),
        position_kp=[1.0] * 8,
        position_kd=[0.1] * 8,
    )
    arm.robstride_bus = SimpleNamespace(configure_motors=Mock(), write=Mock())
    arm.damiao_bus = SimpleNamespace(configure_motors=Mock(), write=Mock(), disable_torque=Mock())
    arm.buses = {"robstride": arm.robstride_bus, "damiao": arm.damiao_bus}
    arm.motor_to_bus_name = {
        name: "robstride" if index < 4 else "damiao" for index, name in enumerate(MOTOR_NAMES)
    }

    arm.configure()

    arm.damiao_bus.disable_torque.assert_called_once_with("gripper")


def test_non_finite_torque_is_rejected() -> None:
    arm = object.__new__(HyOpenArmLeader)
    arm.config = SimpleNamespace(gravity_compensation_torque_limit=1.0)

    with pytest.raises(ValueError, match="non-finite"):
        arm._clamp_compensation_torque(float("nan"))


def test_standalone_gravity_compensation_is_rejected() -> None:
    arm = object.__new__(HyOpenArmLeader)
    arm.config = SimpleNamespace(
        gravity_compensation_enabled=True,
        gravity_compensation_torque_limit=1.0,
    )
    arm._gravity_compensation_managed_externally = False

    with pytest.raises(ValueError, match="Standalone"):
        arm._validate_gravity_compensation_config()

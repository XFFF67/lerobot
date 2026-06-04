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

import logging
import math
from pathlib import Path
from functools import cached_property
from typing import Any

from lerobot.processor import RobotAction
from lerobot.teleoperators.hyopenarm_leader import HyOpenArmLeaderConfig
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..hyopenarm_leader import HyOpenArmLeader
from ..teleoperator import Teleoperator
from .config_bi_hyopenarm_leader import BiHyOpenArmLeaderConfig

logger = logging.getLogger(__name__)


class BiHyOpenArmLeader(Teleoperator):
     """
    Bimanual HyOpenArm Leader Arms
    """
     
     config_class = BiHyOpenArmLeaderConfig
     name = "bi_hyopenarm_leader"

     def __init__(self, config: BiHyOpenArmLeaderConfig):
        super().__init__(config)
        self.config = config

        left_arm_config = HyOpenArmLeaderConfig(
            id=f"{config.id}_left" if config.id else None,
            calibration_dir=config.calibration_dir,
            port=config.left_arm_config.port,
            can_interface=config.left_arm_config.can_interface,
            use_can_fd=config.left_arm_config.use_can_fd,
            can_bitrate=config.left_arm_config.can_bitrate,
            can_data_bitrate=config.left_arm_config.can_data_bitrate,
            motor_config=config.left_arm_config.motor_config,
            manual_control=config.left_arm_config.manual_control,
            position_kd=config.left_arm_config.position_kd,
            position_kp=config.left_arm_config.position_kp,
            gravity_compensation_enabled=config.gravity_compensation_enabled,
            gravity_compensation_torque_limit=config.gravity_compensation_torque_limit,
        )

        right_arm_config = HyOpenArmLeaderConfig(
            id=f"{config.id}_right" if config.id else None,
            calibration_dir=config.calibration_dir,
            port=config.right_arm_config.port,
            can_interface=config.right_arm_config.can_interface,
            use_can_fd=config.right_arm_config.use_can_fd,
            can_bitrate=config.right_arm_config.can_bitrate,
            can_data_bitrate=config.right_arm_config.can_data_bitrate,
            motor_config=config.right_arm_config.motor_config,
            manual_control=config.right_arm_config.manual_control,
            position_kd=config.right_arm_config.position_kd,
            position_kp=config.right_arm_config.position_kp,
            gravity_compensation_enabled=config.gravity_compensation_enabled,
            gravity_compensation_torque_limit=config.gravity_compensation_torque_limit,
        )

        self.left_arm = HyOpenArmLeader(left_arm_config)
        self.right_arm = HyOpenArmLeader(right_arm_config)
        self._motor_names = list(left_arm_config.motor_config)
        self._pin = None
        self._pin_model = None
        self._pin_data = None
        self._pin_q = None
        self._left_pin_indices: dict[str, tuple[int, int]] = {}
        self._right_pin_indices: dict[str, tuple[int, int]] = {}
        self._validate_gravity_compensation_config()
    
     @cached_property
     def action_features(self) -> dict[str, type]:
        left_arm_features = self.left_arm.action_features
        right_arm_features = self.right_arm.action_features

        return {
            **{f"left_{k}": v for k, v in left_arm_features.items()},
            **{f"right_{k}": v for k, v in right_arm_features.items()},
        }

     @cached_property
     def feedback_features(self) -> dict[str, type]:
        return {}

     @property
     def is_connected(self) -> bool:
        return self.left_arm.is_connected and self.right_arm.is_connected

     @check_if_already_connected
     def connect(self, calibrate: bool = True) -> None:
        self._setup_pinocchio_gravity_compensation()
        self.left_arm.connect(calibrate)
        self.right_arm.connect(calibrate)

     @property
     def is_calibrated(self) -> bool:
        return self.left_arm.is_calibrated and self.right_arm.is_calibrated

     def calibrate(self) -> None:
        self.left_arm.calibrate()
        self.right_arm.calibrate()

     def configure(self) -> None:
        self.left_arm.configure()
        self.right_arm.configure()

     def setup_motors(self) -> None:
        raise NotImplementedError(
            "Motor ID configuration is typically done via manufacturer tools for CAN motors."
        )

     @check_if_not_connected
     def get_action(self) -> RobotAction:
        action_dict = {}

        left_states = self.left_arm.read_all_states()
        right_states = self.right_arm.read_all_states()
        self._apply_pinocchio_gravity_compensation(left_states, right_states)

        action_dict.update(self._states_to_action("left", left_states))
        action_dict.update(self._states_to_action("right", right_states))

        return action_dict

     def _validate_gravity_compensation_config(self) -> None:
        if self.config.gravity_compensation_torque_limit < 0:
            raise ValueError("gravity_compensation_torque_limit must be >= 0.")

        for field_name in (
            "left_pinocchio_joint_names",
            "right_pinocchio_joint_names",
            "left_gravity_compensation_signs",
            "right_gravity_compensation_signs",
        ):
            values = getattr(self.config, field_name)
            if len(values) != 7:
                raise ValueError(f"{field_name} must contain 7 values for joint_1 ... joint_7.")

     def _setup_pinocchio_gravity_compensation(self) -> None:
        if not self.config.gravity_compensation_enabled:
            return

        urdf_path = Path(self.config.gravity_compensation_urdf_path).expanduser()
        if not urdf_path.exists():
            raise FileNotFoundError(f"Pinocchio gravity compensation URDF not found: {urdf_path}")

        try:
            import pinocchio as pin
        except ImportError as e:
            raise ImportError("Pinocchio gravity compensation requires `pinocchio` to be installed.") from e

        self._pin = pin
        self._pin_model = pin.buildModelFromUrdf(str(urdf_path))
        self._pin_data = self._pin_model.createData()
        self._pin_q = pin.neutral(self._pin_model)
        self._left_pin_indices = self._build_pinocchio_joint_indices(self.config.left_pinocchio_joint_names)
        self._right_pin_indices = self._build_pinocchio_joint_indices(self.config.right_pinocchio_joint_names)

        logger.info("Loaded Pinocchio gravity compensation model from %s", urdf_path)

     def _build_pinocchio_joint_indices(self, joint_names: list[str]) -> dict[str, tuple[int, int]]:
        if self._pin_model is None:
            raise RuntimeError("Pinocchio model is not initialized.")

        indices: dict[str, tuple[int, int]] = {}
        for motor_name, pin_joint_name in zip(self._motor_names[:7], joint_names, strict=True):
            joint_id = self._pin_model.getJointId(pin_joint_name)
            if joint_id >= len(self._pin_model.joints):
                raise ValueError(f"Joint '{pin_joint_name}' was not found in Pinocchio model.")

            joint = self._pin_model.joints[joint_id]
            if joint.nq != 1 or joint.nv != 1:
                raise ValueError(f"Joint '{pin_joint_name}' must have nq=1 and nv=1, got nq={joint.nq}, nv={joint.nv}.")
            indices[motor_name] = (joint.idx_q, joint.idx_v)
        return indices

     def _states_to_action(self, prefix: str, states: dict[str, dict[str, Any]]) -> RobotAction:
        action_dict: dict[str, Any] = {}
        for motor_name in self._motor_names:
            state = states.get(motor_name, {})
            action_dict[f"{prefix}_{motor_name}.pos"] = state.get("position")
            action_dict[f"{prefix}_{motor_name}.vel"] = state.get("velocity")
            action_dict[f"{prefix}_{motor_name}.torque"] = state.get("torque")
        return action_dict

     def _fill_pinocchio_q(
        self,
        q,
        states: dict[str, dict[str, Any]],
        indices: dict[str, tuple[int, int]],
     ) -> None:
        for motor_name, (idx_q, _) in indices.items():
            position = states.get(motor_name, {}).get("position")
            if position is None:
                continue
            q[idx_q] = math.radians(float(position))

     def _extract_pinocchio_torques(
        self,
        gravity,
        indices: dict[str, tuple[int, int]],
        signs: list[float],
     ) -> dict[str, float]:
        torques: dict[str, float] = {}
        for motor_name, sign in zip(self._motor_names[:7], signs, strict=True):
            _, idx_v = indices[motor_name]
            torques[motor_name] = float(sign) * self.config.gravity_compensation_scale * float(gravity[idx_v])

        torques["gripper"] = 0.0
        return torques

     def _apply_pinocchio_gravity_compensation(
        self,
        left_states: dict[str, dict[str, Any]],
        right_states: dict[str, dict[str, Any]],
     ) -> None:
        if not self.config.gravity_compensation_enabled:
            return
        if self._pin is None or self._pin_model is None or self._pin_data is None or self._pin_q is None:
            raise RuntimeError("Pinocchio gravity compensation is enabled but the model is not initialized.")

        q = self._pin_q.copy()
        self._fill_pinocchio_q(q, left_states, self._left_pin_indices)
        self._fill_pinocchio_q(q, right_states, self._right_pin_indices)
        gravity = self._pin.computeGeneralizedGravity(self._pin_model, self._pin_data, q)

        left_torques = self._extract_pinocchio_torques(
            gravity, self._left_pin_indices, self.config.left_gravity_compensation_signs
        )
        right_torques = self._extract_pinocchio_torques(
            gravity, self._right_pin_indices, self.config.right_gravity_compensation_signs
        )

        self.left_arm.apply_feedforward_torques(left_states, left_torques)
        self.right_arm.apply_feedforward_torques(right_states, right_torques)

     def send_feedback(self, feedback: dict[str, float]) -> None:
        # TODO: Implement force feedback
        raise NotImplementedError

     @check_if_not_connected
     def disconnect(self) -> None:
        self.left_arm.disconnect()
        self.right_arm.disconnect()












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

import contextlib
import logging
import math
from functools import cached_property
from pathlib import Path
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

        self.left_arm = HyOpenArmLeader(left_arm_config, gravity_compensation_managed_externally=True)
        self.right_arm = HyOpenArmLeader(right_arm_config, gravity_compensation_managed_externally=True)
        self._motor_names = list(left_arm_config.motor_config)
        self._pin = None
        self._pin_model = None
        self._pin_data = None
        self._pin_q = None
        self._left_pin_indices: dict[str, tuple[int, int]] = {}
        self._right_pin_indices: dict[str, tuple[int, int]] = {}
        self._gravity_compensation_diagnostics: dict[str, dict[str, dict[str, float | bool]]] = {}
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
        try:
            self.left_arm.connect(calibrate)
            self.right_arm.connect(calibrate)
            self._stop_gravity_compensation()
        except Exception:
            if self.left_arm.is_connected:
                self.left_arm.disconnect()
            if self.right_arm.is_connected:
                self.right_arm.disconnect()
            raise

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
        action_dict: dict[str, Any] = {}

        try:
            left_states = self.left_arm.read_all_states()
            right_states = self.right_arm.read_all_states()
            self._apply_pinocchio_gravity_compensation(left_states, right_states)
        except Exception:
            self._stop_gravity_compensation()
            raise

        action_dict.update(self._states_to_action("left", left_states))
        action_dict.update(self._states_to_action("right", right_states))

        return action_dict

     def _validate_gravity_compensation_config(self) -> None:
        scalar_fields = (
            "gravity_compensation_scale",
            "gravity_compensation_torque_limit",
            "gravity_compensation_max_feedback_age_s",
        )
        for field_name in scalar_fields:
            value = getattr(self.config, field_name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{field_name} must be finite and >= 0.")
        if self.config.gravity_compensation_max_feedback_age_s == 0:
            raise ValueError("gravity_compensation_max_feedback_age_s must be > 0.")

        list_fields = (
            "left_pinocchio_joint_names",
            "right_pinocchio_joint_names",
            "left_gravity_compensation_angle_signs",
            "right_gravity_compensation_angle_signs",
            "left_gravity_compensation_zero_offsets_deg",
            "right_gravity_compensation_zero_offsets_deg",
            "left_gravity_compensation_damping",
            "right_gravity_compensation_damping",
            "left_gravity_compensation_signs",
            "right_gravity_compensation_signs",
        )
        for field_name in list_fields:
            values = getattr(self.config, field_name)
            if len(values) != 7:
                raise ValueError(f"{field_name} must contain 7 values for joint_1 ... joint_7.")

        optional_nonnegative_list_fields = (
            "left_gravity_compensation_scales",
            "right_gravity_compensation_scales",
            "left_gravity_compensation_torque_limits",
            "right_gravity_compensation_torque_limits",
        )
        for field_name in optional_nonnegative_list_fields:
            values = getattr(self.config, field_name)
            if values is None:
                continue
            if len(values) != 7:
                raise ValueError(f"{field_name} must contain 7 values for joint_1 ... joint_7.")
            if any(not math.isfinite(value) or value < 0 for value in values):
                raise ValueError(f"Every value in {field_name} must be finite and >= 0.")

        sign_fields = (
            "left_gravity_compensation_angle_signs",
            "right_gravity_compensation_angle_signs",
            "left_gravity_compensation_signs",
            "right_gravity_compensation_signs",
        )
        for field_name in sign_fields:
            if any(
                not math.isfinite(value) or abs(value) != 1.0
                for value in getattr(self.config, field_name)
            ):
                raise ValueError(f"Every value in {field_name} must be either -1 or 1.")

        offset_fields = (
            "left_gravity_compensation_zero_offsets_deg",
            "right_gravity_compensation_zero_offsets_deg",
        )
        for field_name in offset_fields:
            if any(not math.isfinite(value) for value in getattr(self.config, field_name)):
                raise ValueError(f"Every value in {field_name} must be finite.")

        damping_fields = (
            "left_gravity_compensation_damping",
            "right_gravity_compensation_damping",
        )
        for field_name in damping_fields:
            if any(not math.isfinite(value) or value < 0 for value in getattr(self.config, field_name)):
                raise ValueError(f"Every value in {field_name} must be finite and >= 0.")

     def _setup_pinocchio_gravity_compensation(self) -> None:
        if not self.config.gravity_compensation_enabled:
            return

        if not self.config.gravity_compensation_urdf_path:
            raise ValueError(
                "gravity_compensation_urdf_path is required when gravity compensation is enabled."
            )
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
                raise ValueError(
                    f"Joint '{pin_joint_name}' must have nq=1 and nv=1, "
                    f"got nq={joint.nq}, nv={joint.nv}."
                )
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
         angle_signs: list[float],
         zero_offsets_deg: list[float],
     ) -> None:
        for motor_name, angle_sign, zero_offset_deg in zip(
            self._motor_names[:7], angle_signs, zero_offsets_deg, strict=True
        ):
            idx_q, _ = indices[motor_name]
            position = states.get(motor_name, {}).get("position")
            if position is None:
                raise RuntimeError(
                    f"Missing position feedback for gravity compensation motor '{motor_name}'."
                )
            position = float(position)
            if not math.isfinite(position):
                raise RuntimeError(
                    f"Non-finite position feedback for gravity compensation motor '{motor_name}'."
                )
            q[idx_q] = float(angle_sign) * math.radians(position - float(zero_offset_deg))

     def _extract_pinocchio_torques(
        self,
         gravity,
         indices: dict[str, tuple[int, int]],
         states: dict[str, dict[str, Any]],
         torque_signs: list[float],
         angle_signs: list[float],
         damping: list[float],
         scales: list[float] | None = None,
     ) -> dict[str, float]:
        torques: dict[str, float] = {}
        joint_scales = scales or [self.config.gravity_compensation_scale] * 7
        for motor_name, torque_sign, angle_sign, damping_coefficient, scale in zip(
            self._motor_names[:7], torque_signs, angle_signs, damping, joint_scales, strict=True
        ):
            _, idx_v = indices[motor_name]
            velocity = states.get(motor_name, {}).get("velocity")
            if velocity is None:
                raise RuntimeError(
                    f"Missing velocity feedback for gravity compensation motor '{motor_name}'."
                )
            velocity = float(velocity)
            if not math.isfinite(velocity):
                raise RuntimeError(
                    f"Non-finite velocity feedback for gravity compensation motor '{motor_name}'."
                )
            model_velocity = float(angle_sign) * math.radians(velocity)
            model_torque = (
                float(scale) * float(gravity[idx_v])
                - float(damping_coefficient) * model_velocity
            )
            torques[motor_name] = float(torque_sign) * model_torque

        return torques

     def _build_gravity_compensation_diagnostics(
        self,
        gravity,
        indices: dict[str, tuple[int, int]],
        states: dict[str, dict[str, Any]],
        torque_signs: list[float],
        angle_signs: list[float],
        damping: list[float],
        scales: list[float],
        torque_limits: list[float],
        torques: dict[str, float],
     ) -> dict[str, dict[str, float | bool]]:
        """Describe each gravity-compensation term in motor coordinates."""
        diagnostics: dict[str, dict[str, float | bool]] = {}
        for motor_name, torque_sign, angle_sign, damping_coefficient, scale, torque_limit in zip(
            self._motor_names[:7],
            torque_signs,
            angle_signs,
            damping,
            scales,
            torque_limits,
            strict=True,
        ):
            _, idx_v = indices[motor_name]
            velocity = float(states[motor_name]["velocity"])
            model_velocity = float(angle_sign) * math.radians(velocity)
            raw_gravity_torque = float(torque_sign) * float(gravity[idx_v])
            scaled_gravity_torque = float(scale) * raw_gravity_torque
            damping_torque = float(torque_sign) * (-float(damping_coefficient) * model_velocity)
            unclamped_torque = float(torques[motor_name])
            limited_torque = max(-float(torque_limit), min(float(torque_limit), unclamped_torque))

            diagnostics[motor_name] = {
                "raw_gravity_torque": raw_gravity_torque,
                "scaled_gravity_torque": scaled_gravity_torque,
                "damping_torque": damping_torque,
                "unclamped_torque": unclamped_torque,
                "limited_torque": limited_torque,
                "torque_limit": float(torque_limit),
                "is_limited": abs(unclamped_torque) > float(torque_limit),
            }

        return diagnostics

     @property
     def gravity_compensation_diagnostics(self) -> dict[str, dict[str, dict[str, float | bool]]]:
        """Latest torque terms, expressed in motor coordinates and N·m."""
        return self._gravity_compensation_diagnostics

     def _apply_pinocchio_gravity_compensation(
        self,
        left_states: dict[str, dict[str, Any]],
        right_states: dict[str, dict[str, Any]],
     ) -> None:
        if not self.config.gravity_compensation_enabled:
            return
        if self._pin is None or self._pin_model is None or self._pin_data is None or self._pin_q is None:
            raise RuntimeError("Pinocchio gravity compensation is enabled but the model is not initialized.")

        compensation_motors = self._motor_names[:7]
        self.left_arm.assert_feedback_fresh(
            self.config.gravity_compensation_max_feedback_age_s, compensation_motors, left_states
        )
        self.right_arm.assert_feedback_fresh(
            self.config.gravity_compensation_max_feedback_age_s, compensation_motors, right_states
        )

        q = self._pin_q.copy()
        self._fill_pinocchio_q(
            q,
            left_states,
            self._left_pin_indices,
            self.config.left_gravity_compensation_angle_signs,
            self.config.left_gravity_compensation_zero_offsets_deg,
        )
        self._fill_pinocchio_q(
            q,
            right_states,
            self._right_pin_indices,
            self.config.right_gravity_compensation_angle_signs,
            self.config.right_gravity_compensation_zero_offsets_deg,
        )
        gravity = self._pin.computeGeneralizedGravity(self._pin_model, self._pin_data, q)

        left_scales = self.config.left_gravity_compensation_scales or [
            self.config.gravity_compensation_scale
        ] * 7
        right_scales = self.config.right_gravity_compensation_scales or [
            self.config.gravity_compensation_scale
        ] * 7
        left_torque_limits = self.config.left_gravity_compensation_torque_limits or [
            self.config.gravity_compensation_torque_limit
        ] * 7
        right_torque_limits = self.config.right_gravity_compensation_torque_limits or [
            self.config.gravity_compensation_torque_limit
        ] * 7

        left_torques = self._extract_pinocchio_torques(
            gravity,
            self._left_pin_indices,
            left_states,
            self.config.left_gravity_compensation_signs,
            self.config.left_gravity_compensation_angle_signs,
            self.config.left_gravity_compensation_damping,
            left_scales,
        )
        right_torques = self._extract_pinocchio_torques(
            gravity,
            self._right_pin_indices,
            right_states,
            self.config.right_gravity_compensation_signs,
            self.config.right_gravity_compensation_angle_signs,
            self.config.right_gravity_compensation_damping,
            right_scales,
        )

        self._gravity_compensation_diagnostics = {
            "left": self._build_gravity_compensation_diagnostics(
                gravity,
                self._left_pin_indices,
                left_states,
                self.config.left_gravity_compensation_signs,
                self.config.left_gravity_compensation_angle_signs,
                self.config.left_gravity_compensation_damping,
                left_scales,
                left_torque_limits,
                left_torques,
            ),
            "right": self._build_gravity_compensation_diagnostics(
                gravity,
                self._right_pin_indices,
                right_states,
                self.config.right_gravity_compensation_signs,
                self.config.right_gravity_compensation_angle_signs,
                self.config.right_gravity_compensation_damping,
                right_scales,
                right_torque_limits,
                right_torques,
            ),
        }

        compensation_motors = self._motor_names[:7]
        self.left_arm.apply_feedforward_torques(
            left_states, left_torques, dict(zip(compensation_motors, left_torque_limits, strict=True))
        )
        self.right_arm.apply_feedforward_torques(
            right_states, right_torques, dict(zip(compensation_motors, right_torque_limits, strict=True))
        )

     def _stop_gravity_compensation(self) -> None:
        if not self.config.gravity_compensation_enabled:
            return
        for arm in (self.left_arm, self.right_arm):
            if arm.is_connected:
                with contextlib.suppress(Exception):
                    arm.stop_gravity_compensation()

     def send_feedback(self, feedback: dict[str, float]) -> None:
        # TODO: Implement force feedback
        raise NotImplementedError

     @check_if_not_connected
     def disconnect(self) -> None:
        self._stop_gravity_compensation()
        self.left_arm.disconnect()
        self.right_arm.disconnect()





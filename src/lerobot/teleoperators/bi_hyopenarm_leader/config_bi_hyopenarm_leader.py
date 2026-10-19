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

from dataclasses import dataclass, field
from pathlib import Path

from lerobot.teleoperators.hyopenarm_leader import HyOpenArmLeaderConfigBase

from ..config import TeleoperatorConfig


DEFAULT_GRAVITY_COMPENSATION_URDF_PATH = str(
    Path(__file__).resolve().parents[4] / "urdf" / "openarm_bimanual.urdf"
)


@TeleoperatorConfig.register_subclass("bi_hyopenarm_leader")
@dataclass
class BiHyOpenArmLeaderConfig(TeleoperatorConfig):
    """Configuration class for Bi HyOpenArm Leader robots."""

    left_arm_config: HyOpenArmLeaderConfigBase
    right_arm_config: HyOpenArmLeaderConfigBase

    gravity_compensation_enabled: bool = False
    gravity_compensation_urdf_path: str | None = DEFAULT_GRAVITY_COMPENSATION_URDF_PATH
    # Start conservatively; raise only after verifying every joint's direction on hardware.
    gravity_compensation_scale: float = 0.1
    gravity_compensation_torque_limit: float = 1.0
    gravity_compensation_max_feedback_age_s: float = 0.1
    # Optional per-joint overrides, ordered as joint_1 ... joint_7. When omitted,
    # the global scale/torque limit above is used for every joint on that side.
    left_gravity_compensation_scales: list[float] | None = None
    right_gravity_compensation_scales: list[float] | None = None
    left_gravity_compensation_torque_limits: list[float] | None = None
    right_gravity_compensation_torque_limits: list[float] | None = None
    left_pinocchio_joint_names: list[str] = field(
        default_factory=lambda: [f"openarm_left_joint{i}" for i in range(1, 8)]
    )
    right_pinocchio_joint_names: list[str] = field(
        default_factory=lambda: [f"openarm_right_joint{i}" for i in range(1, 8)]
    )
    # q_urdf = angle_sign * radians(q_motor_deg - zero_offset_deg)
    left_gravity_compensation_angle_signs: list[float] = field(default_factory=lambda: [1.0] * 7)
    right_gravity_compensation_angle_signs: list[float] = field(default_factory=lambda: [1.0] * 7)
    left_gravity_compensation_zero_offsets_deg: list[float] = field(default_factory=lambda: [0.0] * 7)
    right_gravity_compensation_zero_offsets_deg: list[float] = field(default_factory=lambda: [0.0] * 7)
    # Damping coefficients in N·m per rad/s, expressed in URDF joint coordinates.
    left_gravity_compensation_damping: list[float] = field(default_factory=lambda: [0.1] * 7)
    right_gravity_compensation_damping: list[float] = field(default_factory=lambda: [0.1] * 7)
    # Maps torque from URDF coordinates to motor coordinates.
    left_gravity_compensation_signs: list[float] = field(default_factory=lambda: [1.0] * 7)
    right_gravity_compensation_signs: list[float] = field(default_factory=lambda: [1.0] * 7)

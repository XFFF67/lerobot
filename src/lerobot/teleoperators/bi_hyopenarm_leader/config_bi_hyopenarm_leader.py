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

from lerobot.teleoperators.hyopenarm_leader import HyOpenArmLeaderConfigBase

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("bi_hyopenarm_leader")
@dataclass
class BiHyOpenArmLeaderConfig(TeleoperatorConfig):
    """Configuration class for Bi HyOpenArm Leader robots."""

    left_arm_config: HyOpenArmLeaderConfigBase
    right_arm_config: HyOpenArmLeaderConfigBase

    gravity_compensation_enabled: bool = False  # 总开关
    gravity_compensation_urdf_path: str = "/tmp/v10_bimanual.urdf"
    gravity_compensation_scale: float = 1.0 # 整体补偿比例
    gravity_compensation_torque_limit: float = 1.0 # 单个关节最终输出力矩限幅，防止模型力矩或方向错误导致力矩过大。
    left_pinocchio_joint_names: list[str] = field(
        default_factory=lambda: [f"openarm_left_joint{i}" for i in range(1, 8)]
    )
    right_pinocchio_joint_names: list[str] = field(
        default_factory=lambda: [f"openarm_right_joint{i}" for i in range(1, 8)]
    )
    left_gravity_compensation_signs: list[float] = field(default_factory=lambda: [1.0] * 7)
    right_gravity_compensation_signs: list[float] = field(default_factory=lambda: [1.0] * 7)

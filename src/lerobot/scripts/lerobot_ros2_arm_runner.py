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

"""ROS2-triggered arm policy runner.

This entry point keeps base navigation outside LeRobot.  A ROS2 task or
navigation node sends a start command when the mobile base reaches a waypoint;
this process then runs the normal LeRobot arm policy loop until it is stopped
or times out, and publishes status back to ROS2.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from lerobot.configs import parser
from lerobot.rollout.configs import RolloutConfig
from lerobot.rollout.context import RolloutContext, build_rollout_context
from lerobot.rollout.strategies import send_next_action
from lerobot.utils.action_interpolator import ActionInterpolator
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.process import ProcessSignalHandler
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging

logger = logging.getLogger(__name__)


@dataclass
class ROS2ArmRunnerConfig(RolloutConfig):
    """Rollout config plus ROS2 task-level command/status topics.

    Commands are ``std_msgs/msg/String`` payloads.  Plain strings are accepted
    (``start``, ``stop``, ``pause``, ``resume``), and JSON objects may include
    ``command``, ``task_id``, ``task``, and ``timeout_s`` fields.
    """

    ros_node_name: str = "lerobot_arm_policy_runner"
    command_topic: str = "/arm_policy/command"
    status_topic: str = "/arm_policy/status"
    default_task_timeout_s: float = 0.0


@dataclass
class ArmTaskState:
    active: bool = False
    paused: bool = False
    task_id: str = ""
    task: str = ""
    deadline_s: float | None = None
    stop_reason: str | None = None
    last_command: dict[str, Any] = field(default_factory=dict)


class ROS2ArmRunner:
    def __init__(self, cfg: ROS2ArmRunnerConfig, ctx: RolloutContext):
        try:
            import rclpy
            from std_msgs.msg import String
        except ImportError as e:
            raise ImportError(
                "ROS2 Python packages are required for lerobot-ros2-arm-runner. "
                "Source your ROS2 environment so 'rclpy' and 'std_msgs' are importable."
            ) from e

        self.cfg = cfg
        self.ctx = ctx
        self.state = ArmTaskState()
        self.interpolator = ActionInterpolator(multiplier=cfg.interpolation_multiplier)

        self.rclpy = rclpy
        self.String = String
        rclpy.init(args=None)
        self.node = rclpy.create_node(cfg.ros_node_name)
        self.publisher = self.node.create_publisher(String, cfg.status_topic, 10)
        self.subscription = self.node.create_subscription(
            String,
            cfg.command_topic,
            self._on_command,
            10,
        )

    def close(self) -> None:
        self.node.destroy_node()
        self.rclpy.shutdown()

    def run(self, shutdown_event) -> None:
        engine = self.ctx.policy.inference
        engine.reset()
        engine.start()
        engine.pause()
        self.publish_status("idle")

        control_interval = 1.0 / self.cfg.fps
        try:
            while not shutdown_event.is_set():
                loop_start = time.perf_counter()
                self.rclpy.spin_once(self.node, timeout_sec=0.0)

                if self.state.active and not self.state.paused:
                    self._run_control_tick()
                    if (
                        self.state.deadline_s is not None
                        and time.monotonic() >= self.state.deadline_s
                    ):
                        self._finish_task("succeeded", reason="timeout")

                if engine.failed:
                    self._finish_task("failed", reason="inference_failed")
                    shutdown_event.set()

                dt_s = time.perf_counter() - loop_start
                precise_sleep(max(control_interval - dt_s, 0.0))
        finally:
            engine.stop()

    def publish_status(self, status: str, **extra: Any) -> None:
        payload = {
            "status": status,
            "task_id": self.state.task_id,
            "task": self.state.task,
            **extra,
        }
        msg = self.String()
        msg.data = json.dumps(payload, ensure_ascii=True)
        self.publisher.publish(msg)
        logger.info("arm_policy status: %s", msg.data)

    def _run_control_tick(self) -> None:
        obs_raw = self.ctx.hardware.robot_wrapper.get_observation()
        obs_processed = self.ctx.processors.robot_observation_processor(obs_raw)
        self.ctx.policy.inference.notify_observation(obs_processed)
        action = send_next_action(
            obs_processed=obs_processed,
            obs_raw=obs_raw,
            ctx=self.ctx,
            interpolator=self.interpolator,
        )
        if action is not None:
            logger.debug("sent arm action with %d keys", len(action))

    def _on_command(self, msg) -> None:
        command = self._parse_command(msg.data)
        name = str(command.get("command", "")).lower()
        if name in ("start", "run"):
            self._start_task(command)
        elif name == "stop":
            self._finish_task("stopped", reason=str(command.get("reason", "stop_command")))
        elif name == "pause":
            self.state.paused = True
            self.ctx.policy.inference.pause()
            self.publish_status("paused")
        elif name == "resume":
            if self.state.active:
                self.state.paused = False
                self.ctx.policy.inference.resume()
                self.publish_status("running")
        elif name:
            logger.warning("Ignoring unknown arm policy command: %s", name)

    @staticmethod
    def _parse_command(data: str) -> dict[str, Any]:
        data = data.strip()
        if not data:
            return {}
        try:
            parsed = json.loads(data)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
        return {"command": data}

    def _start_task(self, command: dict[str, Any]) -> None:
        self.state.active = True
        self.state.paused = False
        self.state.task_id = str(command.get("task_id", ""))
        self.state.task = str(command.get("task", self.cfg.task))
        self.state.stop_reason = None
        self.state.last_command = command

        timeout_s = float(command.get("timeout_s", self.cfg.default_task_timeout_s))
        self.state.deadline_s = time.monotonic() + timeout_s if timeout_s > 0 else None

        engine = self.ctx.policy.inference
        if self.state.task and hasattr(engine, "_task"):
            engine._task = self.state.task
        engine.reset()
        engine.resume()
        self.interpolator.reset()
        self.publish_status("running", timeout_s=timeout_s)

    def _finish_task(self, status: str, reason: str) -> None:
        if not self.state.active and status != "failed":
            return
        self.state.active = False
        self.state.paused = False
        self.state.stop_reason = reason
        self.ctx.policy.inference.pause()
        self.publish_status(status, reason=reason)


def _teardown_hardware(ctx: RolloutContext) -> None:
    robot = ctx.hardware.robot_wrapper.inner
    if robot.is_connected:
        robot.disconnect()
    teleop = ctx.hardware.teleop
    if teleop is not None and teleop.is_connected:
        teleop.disconnect()


@parser.wrap()
def ros2_arm_runner(cfg: ROS2ArmRunnerConfig) -> None:
    init_logging()
    signal_handler = ProcessSignalHandler(use_threads=True, display_pid=False)
    ctx = build_rollout_context(cfg, signal_handler.shutdown_event)
    runner: ROS2ArmRunner | None = None
    try:
        runner = ROS2ArmRunner(cfg, ctx)
        runner.run(signal_handler.shutdown_event)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        if runner is not None:
            runner.close()
        _teardown_hardware(ctx)


def main() -> None:
    register_third_party_plugins()
    ros2_arm_runner()


if __name__ == "__main__":
    main()

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

"""Power on all OpenArm follower motors and hold their zero position."""

import argparse
import logging
import time

from lerobot.robots.bi_openarm_follower import BiOpenArmFollower, BiOpenArmFollowerConfig
from lerobot.robots.openarm_follower import OpenArmFollower, OpenArmFollowerConfig
from lerobot.robots.openarm_follower.config_openarm_follower import OpenArmFollowerConfigBase


logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Enable OpenArm follower motors and continuously hold 0 degree position."
    )
    parser.add_argument(
        "--port",
        default=None,
        help='Optional single-arm CAN interface, for example "can2". If set, only one arm is controlled.',
    )
    parser.add_argument(
        "--left-port",
        default="can2",
        help='Left arm CAN interface for bimanual OpenArm follower. Defaults to "can2".',
    )
    parser.add_argument(
        "--right-port",
        default="can3",
        help='Right arm CAN interface for bimanual OpenArm follower. Defaults to "can3".',
    )
    parser.add_argument(
        "--id",
        default="bi_openarm_follower",
        help="Robot id used to load/save calibration. Defaults to bi_openarm_follower.",
    )
    parser.add_argument(
        "--side",
        choices=("left", "right"),
        default=None,
        help="Use the built-in left/right joint limits. If omitted, narrow safety limits are used.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=50.0,
        help="Command rate used to hold zero position.",
    )
    parser.add_argument(
        "--duration-s",
        type=float,
        default=None,
        help="Optional run duration in seconds. If omitted, runs until Ctrl+C.",
    )
    parser.add_argument(
        "--status-interval-s",
        type=float,
        default=5.0,
        help="How often to log that zero-position hold is still running.",
    )
    parser.add_argument(
        "--no-calibrate",
        action="store_true",
        help="Do not run calibration if no matching calibration is found.",
    )
    parser.add_argument(
        "--keep-torque-on-exit",
        action="store_true",
        help="Keep motor torque enabled when the script exits.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        help="Python logging level.",
    )
    return parser


def make_zero_action(robot: OpenArmFollower | BiOpenArmFollower) -> dict[str, float]:
    if isinstance(robot, BiOpenArmFollower):
        return {
            **{f"left_{motor}.pos": 0.0 for motor in robot.left_arm.bus.motors},
            **{f"right_{motor}.pos": 0.0 for motor in robot.right_arm.bus.motors},
        }

    return {f"{motor}.pos": 0.0 for motor in robot.bus.motors}


def motor_names(robot: OpenArmFollower | BiOpenArmFollower) -> list[str]:
    if isinstance(robot, BiOpenArmFollower):
        return [
            *[f"left_{motor}" for motor in robot.left_arm.bus.motors],
            *[f"right_{motor}" for motor in robot.right_arm.bus.motors],
        ]

    return list(robot.bus.motors)


def hold_zero(
    robot: OpenArmFollower | BiOpenArmFollower,
    fps: float,
    duration_s: float | None,
    status_interval_s: float,
) -> None:
    if fps <= 0:
        raise ValueError("--fps must be greater than 0")

    zero_action = make_zero_action(robot)
    period_s = 1.0 / fps
    start_t = time.monotonic()
    next_t = start_t
    next_status_t = start_t + status_interval_s

    logger.info("Holding zero position for motors: %s", ", ".join(motor_names(robot)))
    logger.info("Press Ctrl+C to stop.")
    while duration_s is None or time.monotonic() - start_t < duration_s:
        robot.send_action(zero_action)
        now = time.monotonic()
        if status_interval_s > 0 and now >= next_status_t:
            logger.info("Still holding zero position. Elapsed: %.1fs", now - start_t)
            next_status_t = now + status_interval_s
        next_t += period_s
        sleep_s = next_t - now
        if sleep_s > 0:
            time.sleep(sleep_s)
        else:
            next_t = time.monotonic()


def disconnect_robot(robot: OpenArmFollower | BiOpenArmFollower) -> None:
    if isinstance(robot, BiOpenArmFollower):
        for arm in (robot.left_arm, robot.right_arm):
            if arm.bus.is_connected:
                arm.disconnect()
        return

    if robot.bus.is_connected:
        robot.disconnect()


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    disable_torque_on_disconnect = not args.keep_torque_on_exit
    if args.port is not None:
        config = OpenArmFollowerConfig(
            port=args.port,
            id=args.id,
            side=args.side,
            disable_torque_on_disconnect=disable_torque_on_disconnect,
        )
        robot = OpenArmFollower(config)
    else:
        config = BiOpenArmFollowerConfig(
            id=args.id,
            left_arm_config=OpenArmFollowerConfigBase(
                port=args.left_port,
                side="left",
                disable_torque_on_disconnect=disable_torque_on_disconnect,
            ),
            right_arm_config=OpenArmFollowerConfigBase(
                port=args.right_port,
                side="right",
                disable_torque_on_disconnect=disable_torque_on_disconnect,
            ),
        )
        robot = BiOpenArmFollower(config)

    try:
        robot.connect(calibrate=not args.no_calibrate)
        hold_zero(
            robot,
            fps=args.fps,
            duration_s=args.duration_s,
            status_interval_s=args.status_interval_s,
        )
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        disconnect_robot(robot)


if __name__ == "__main__":
    main()

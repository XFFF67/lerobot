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

"""Validate bimanual HyOpenArm leader gravity compensation without a follower robot."""

import csv
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from pprint import pformat
from typing import Any

import draccus

from lerobot.teleoperators.bi_hyopenarm_leader.bi_hyopenarm_leader import BiHyOpenArmLeader
from lerobot.teleoperators.bi_hyopenarm_leader.config_bi_hyopenarm_leader import (
    BiHyOpenArmLeaderConfig,
)
from lerobot.teleoperators.config import TeleoperatorConfig
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging


@dataclass
class GravityCompensationTestConfig:
    teleop: TeleoperatorConfig
    fps: int = 30
    duration_s: float = 10.0
    status_interval_s: float = 1.0
    calibrate: bool = False
    diagnostics_csv_path: str | None = None
    diagnostics_plot_path: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.teleop, BiHyOpenArmLeaderConfig):
            raise ValueError("teleop.type must be bi_hyopenarm_leader.")
        if self.fps <= 0:
            raise ValueError("fps must be > 0.")
        if self.duration_s <= 0:
            raise ValueError("duration_s must be > 0.")
        if self.status_interval_s <= 0:
            raise ValueError("status_interval_s must be > 0.")
        if not self.teleop.gravity_compensation_enabled:
            raise ValueError("teleop.gravity_compensation_enabled must be true.")


def _format_joint_positions(action: dict[str, float | None]) -> str:
    values: list[str] = []
    for side in ("left", "right"):
        positions = []
        for joint_index in range(1, 8):
            value = action.get(f"{side}_joint_{joint_index}.pos")
            positions.append("None" if value is None else f"{float(value):.1f}")
        values.append(f"{side}=[{', '.join(positions)}]")
    return " ".join(values)


def _format_torque_diagnostics(
    diagnostics: dict[str, dict[str, dict[str, float | bool]]],
) -> str:
    lines = ["Gravity compensation torque diagnostics (motor coordinates, N·m):"]
    for side in ("left", "right"):
        lines.append(f"  {side}:")
        side_diagnostics = diagnostics.get(side, {})
        for joint_index in range(1, 8):
            motor_name = f"joint_{joint_index}"
            values = side_diagnostics.get(motor_name)
            if values is None:
                lines.append(f"    {motor_name}: unavailable")
                continue
            limited = "YES" if values["is_limited"] else "no"
            lines.append(
                f"    {motor_name}: raw_g={values['raw_gravity_torque']:+.3f} "
                f"scaled_g={values['scaled_gravity_torque']:+.3f} "
                f"damping={values['damping_torque']:+.3f} "
                f"pre_limit={values['unclamped_torque']:+.3f} "
                f"command={values['limited_torque']:+.3f} "
                f"clipped={limited} limit={values['torque_limit']:.3f}"
            )
    return "\n".join(lines)


def _collect_diagnostic_rows(
    rows: list[dict[str, Any]],
    timestamp_s: float,
    action: dict[str, float | None],
    diagnostics: dict[str, dict[str, dict[str, float | bool]]],
) -> None:
    for side in ("left", "right"):
        side_diagnostics = diagnostics.get(side, {})
        for joint_index in range(1, 8):
            motor_name = f"joint_{joint_index}"
            values = side_diagnostics.get(motor_name)
            if values is None:
                continue
            rows.append(
                {
                    "timestamp_s": timestamp_s,
                    "side": side,
                    "joint": motor_name,
                    "position_deg": action.get(f"{side}_{motor_name}.pos"),
                    "velocity_deg_s": action.get(f"{side}_{motor_name}.vel"),
                    "raw_gravity_torque": values["raw_gravity_torque"],
                    "scaled_gravity_torque": values["scaled_gravity_torque"],
                    "damping_torque": values["damping_torque"],
                    "unclamped_torque": values["unclamped_torque"],
                    "limited_torque": values["limited_torque"],
                    "torque_limit": values["torque_limit"],
                    "is_limited": values["is_limited"],
                }
            )


def _write_diagnostic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "timestamp_s",
        "side",
        "joint",
        "position_deg",
        "velocity_deg_s",
        "raw_gravity_torque",
        "scaled_gravity_torque",
        "damping_torque",
        "unclamped_torque",
        "limited_torque",
        "torque_limit",
        "is_limited",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_diagnostic_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("No diagnostics were recorded, cannot plot.")

    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("Plotting gravity compensation diagnostics requires `matplotlib`.") from e

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(7, 2, figsize=(14, 18), sharex=True)
    metrics = ("scaled_gravity_torque", "damping_torque", "limited_torque")

    for row_index, joint_index in enumerate(range(1, 8)):
        joint = f"joint_{joint_index}"
        for col_index, side in enumerate(("left", "right")):
            ax = axes[row_index][col_index]
            side_joint_rows = [row for row in rows if row["side"] == side and row["joint"] == joint]
            if side_joint_rows:
                timestamps = [float(row["timestamp_s"]) for row in side_joint_rows]
                for metric in metrics:
                    ax.plot(timestamps, [float(row[metric]) for row in side_joint_rows], label=metric)
                limits = [float(row["torque_limit"]) for row in side_joint_rows]
                ax.plot(timestamps, limits, color="gray", linestyle="--", linewidth=0.8, label="+limit")
                ax.plot(timestamps, [-limit for limit in limits], color="gray", linestyle="--", linewidth=0.8)
            ax.axhline(0.0, color="black", linewidth=0.6)
            ax.set_title(f"{side} {joint}")
            ax.set_ylabel("N.m")
            ax.grid(True, alpha=0.25)

    axes[-1][0].set_xlabel("time (s)")
    axes[-1][1].set_xlabel("time (s)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(labels))
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _polyline(points: list[tuple[float, float]], color: str) -> str:
    if not points:
        return ""
    point_data = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return f'<polyline points="{point_data}" fill="none" stroke="{color}" stroke-width="1.4"/>'


def _plot_diagnostic_rows_svg(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("No diagnostics were recorded, cannot plot.")

    path.parent.mkdir(parents=True, exist_ok=True)
    metrics = (
        ("scaled_gravity_torque", "#1f77b4"),
        ("damping_torque", "#ff7f0e"),
        ("limited_torque", "#2ca02c"),
    )
    width = 1200
    height = 1680
    margin_left = 60
    margin_top = 72
    plot_width = 500
    plot_height = 180
    row_gap = 36
    col_gap = 80
    timestamps = [float(row["timestamp_s"]) for row in rows]
    min_t = min(timestamps)
    max_t = max(timestamps)
    if max_t <= min_t:
        max_t = min_t + 1.0

    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="60" y="32" font-family="monospace" font-size="20">Gravity compensation diagnostics</text>',
        '<text x="60" y="56" font-family="monospace" font-size="13" fill="#1f77b4">scaled</text>',
        '<text x="130" y="56" font-family="monospace" font-size="13" fill="#ff7f0e">damping</text>',
        '<text x="215" y="56" font-family="monospace" font-size="13" fill="#2ca02c">command</text>',
    ]

    for row_index, joint_index in enumerate(range(1, 8)):
        joint = f"joint_{joint_index}"
        for col_index, side in enumerate(("left", "right")):
            x0 = margin_left + col_index * (plot_width + col_gap)
            y0 = margin_top + row_index * (plot_height + row_gap)
            side_joint_rows = [row for row in rows if row["side"] == side and row["joint"] == joint]
            values = [
                float(row[metric])
                for row in side_joint_rows
                for metric, _ in metrics
            ]
            limits = [float(row["torque_limit"]) for row in side_joint_rows]
            max_abs = max([abs(value) for value in values] + limits + [0.1])

            def map_x(timestamp: float) -> float:
                return x0 + (timestamp - min_t) / (max_t - min_t) * plot_width

            def map_y(value: float) -> float:
                return y0 + plot_height / 2 - value / max_abs * (plot_height * 0.44)

            svg.extend(
                [
                    f'<text x="{x0}" y="{y0 - 10}" font-family="monospace" font-size="14">{side} {joint}</text>',
                    f'<rect x="{x0}" y="{y0}" width="{plot_width}" height="{plot_height}" fill="none" stroke="#d0d0d0"/>',
                    f'<line x1="{x0}" y1="{map_y(0.0):.1f}" x2="{x0 + plot_width}" y2="{map_y(0.0):.1f}" stroke="#333" stroke-width="0.8"/>',
                    f'<text x="{x0}" y="{y0 + plot_height + 16}" font-family="monospace" font-size="11">{min_t:.1f}s</text>',
                    f'<text x="{x0 + plot_width - 45}" y="{y0 + plot_height + 16}" font-family="monospace" font-size="11">{max_t:.1f}s</text>',
                    f'<text x="{x0 + 4}" y="{y0 + 14}" font-family="monospace" font-size="11" fill="#666">+/-{max_abs:.2f} N.m</text>',
                ]
            )
            if not side_joint_rows:
                continue
            limit_points = [(map_x(float(row["timestamp_s"])), map_y(float(row["torque_limit"]))) for row in side_joint_rows]
            neg_limit_points = [(map_x(float(row["timestamp_s"])), map_y(-float(row["torque_limit"]))) for row in side_joint_rows]
            svg.append(_polyline(limit_points, "#999"))
            svg.append(_polyline(neg_limit_points, "#999"))
            for metric, color in metrics:
                points = [
                    (map_x(float(row["timestamp_s"])), map_y(float(row[metric])))
                    for row in side_joint_rows
                ]
                svg.append(_polyline(points, color))

    svg.append("</svg>")
    path.write_text("\n".join(svg), encoding="utf-8")


@draccus.wrap()
def test_gravity_compensation(cfg: GravityCompensationTestConfig) -> None:
    init_logging()
    logging.info("Gravity compensation validation config:\n%s", pformat(cfg))

    leader = BiHyOpenArmLeader(cfg.teleop)
    loop_period_s = 1.0 / cfg.fps
    diagnostic_rows: list[dict[str, Any]] = []

    try:
        leader.connect(calibrate=cfg.calibrate)
        logging.info(
            "Gravity compensation active for %.1fs at %d Hz. Press Ctrl+C to stop.",
            cfg.duration_s,
            cfg.fps,
        )

        start_time = time.perf_counter()
        next_status_time = start_time
        while time.perf_counter() - start_time < cfg.duration_s:
            loop_start = time.perf_counter()
            action = leader.get_action()

            now = time.perf_counter()
            elapsed_s = now - start_time
            _collect_diagnostic_rows(
                diagnostic_rows,
                elapsed_s,
                action,
                leader.gravity_compensation_diagnostics,
            )
            if now >= next_status_time:
                logging.info("Joint positions (deg): %s", _format_joint_positions(action))
                logging.info(
                    "%s", _format_torque_diagnostics(leader.gravity_compensation_diagnostics)
                )
                next_status_time = now + cfg.status_interval_s

            precise_sleep(max(loop_period_s - (time.perf_counter() - loop_start), 0.0))
    except KeyboardInterrupt:
        logging.info("Gravity compensation validation interrupted by user.")
    finally:
        if leader.is_connected:
            leader.disconnect()
        if cfg.diagnostics_csv_path:
            csv_path = Path(cfg.diagnostics_csv_path).expanduser()
            _write_diagnostic_csv(csv_path, diagnostic_rows)
            logging.info("Wrote gravity compensation diagnostics CSV to %s", csv_path)
        if cfg.diagnostics_plot_path:
            plot_path = Path(cfg.diagnostics_plot_path).expanduser()
            try:
                _plot_diagnostic_rows(plot_path, diagnostic_rows)
            except ImportError as e:
                logging.warning("Could not write gravity compensation diagnostics plot: %s", e)
                fallback_path = plot_path if plot_path.suffix.lower() == ".svg" else plot_path.with_suffix(".svg")
                _plot_diagnostic_rows_svg(fallback_path, diagnostic_rows)
                logging.info("Wrote gravity compensation diagnostics SVG plot to %s", fallback_path)
            else:
                logging.info("Wrote gravity compensation diagnostics plot to %s", plot_path)
        logging.info("Gravity compensation stopped and leader arms disconnected.")


def main() -> None:
    test_gravity_compensation()


if __name__ == "__main__":
    main()

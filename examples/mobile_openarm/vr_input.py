"""Validated Telegrip snapshots; no robot or simulator dependencies."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Controller:
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    quaternion: np.ndarray = field(default_factory=lambda: np.array([1.0, 0, 0, 0]))
    grip: bool = False
    trigger: bool = False
    stick: np.ndarray = field(default_factory=lambda: np.zeros(2))
    primary: bool = False
    secondary: bool = False
    stick_pressed: bool = False
    valid: bool = False


def parse_controller(value, side: str) -> Controller:
    """Missing/tracking-lost/nonfinite poses disable the entire controller."""
    try:
        if not isinstance(value, dict) or value.get("tracked") is False:
            return Controller()
        position = np.array([value["position"][axis] for axis in "xyz"], dtype=float)
        quaternion = np.array([value["quaternion"][axis] for axis in "wxyz"], dtype=float)
        stick = value.get("thumbstick", {})
        axes = np.array([stick.get("x", 0), stick.get("y", 0)], dtype=float)
        norm = float(np.linalg.norm(quaternion))
        if not np.all(np.isfinite(np.r_[position, quaternion, axes])) or not 0.5 < norm < 1.5:
            return Controller()
        return Controller(
            position=position,
            quaternion=quaternion / norm,
            grip=bool(value.get("gripActive", False)),
            trigger=bool(value.get("trigger", False)),
            stick=np.clip(axes, -1, 1),
            primary=bool(value.get("xButton" if side == "left" else "aButton", False)),
            secondary=bool(value.get("yButton" if side == "left" else "bButton", False)),
            stick_pressed=bool(stick.get("pressed", False)),
            valid=True,
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        return Controller()


class VRInput:
    def __init__(self, timeout: float = 0.3):
        self.timeout = timeout
        self.controllers = {side: Controller() for side in ("left", "right")}
        self.received_at = {side: -float("inf") for side in self.controllers}
        self.packet_count = 0

    def ingest(self, message: str, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        try:
            raw = message.lstrip()
            packet = json.loads(raw if raw.startswith("{") else raw.split(" ", 1)[1])
        except (ValueError, IndexError):
            return False
        if not isinstance(packet, dict):
            return False
        updated = False
        for side in self.controllers:
            field_name = f"{side}Controller"
            if field_name in packet:
                self.controllers[side] = parse_controller(packet[field_name], side)
                self.received_at[side] = now
                updated = True
        # Telegrip also emits release events. They must not keep stale poses alive.
        side = packet.get("hand")
        if isinstance(side, str) and side in self.controllers:
            if packet.get("gripReleased"):
                self.controllers[side].grip = False
            if packet.get("triggerReleased"):
                self.controllers[side].trigger = False
        if updated:
            self.packet_count += 1
        return updated

    def snapshot(self, now: float | None = None) -> dict[str, Controller]:
        now = time.monotonic() if now is None else now
        return {
            side: controller if 0 <= now - self.received_at[side] <= self.timeout else Controller()
            for side, controller in self.controllers.items()
        }


class ZMQReceiver(VRInput):
    def __init__(self, endpoint: str, topic: str, timeout: float):
        super().__init__(timeout)
        try:
            import zmq
        except ImportError as exc:
            raise ImportError("VR input requires pyzmq; see README_zh.md") from exc
        self.zmq = zmq
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        try:
            # Telegrip publishes single-part 'vr_data {json}' messages.
            self.socket.setsockopt(zmq.CONFLATE, 1)
            self.socket.setsockopt(zmq.SUBSCRIBE, (topic + " ").encode() if topic else b"")
            self.socket.connect(endpoint)
        except Exception:
            self.close()
            raise

    def poll(self):
        try:
            message = self.socket.recv_string(self.zmq.NOBLOCK)
        except (self.zmq.Again, UnicodeDecodeError):
            return
        self.ingest(message)

    def close(self):
        self.socket.close(0)
        self.context.term()


def deadzone(value: float, threshold: float) -> float:
    return float(np.sign(value) * max(abs(value) - threshold, 0) / (1 - threshold))

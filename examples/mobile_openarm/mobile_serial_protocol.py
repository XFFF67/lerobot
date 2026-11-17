"""Original seven-byte chassis frame with two appended lift bytes."""

import struct
from dataclasses import dataclass

FRAME_SIZE = 9
CHASSIS_HEADER = 0x55


@dataclass(frozen=True)
class ChassisFrame:
    angle_deg: int = 0
    speed_milli: int = 0
    spin: bool = False

    def encode(self):
        if not -90 <= self.angle_deg <= 90:
            raise ValueError("Chassis angle must be within +/-90 degrees")
        limit = 10 if self.spin else 100
        if not -limit <= self.speed_milli <= limit:
            raise ValueError("Chassis speed is outside the configured limit")
        direction = 0x20 if self.speed_milli < 0 else 0
        magnitude = abs(self.speed_milli)
        if self.spin:
            return bytes((CHASSIS_HEADER, magnitude >> 8, magnitude & 0xFF, direction, 0, 0, 1))
        # The original firmware negates the received signed angle.
        angle = struct.pack(">h", -self.angle_deg)
        return bytes((CHASSIS_HEADER, angle[0], angle[1], direction, magnitude >> 8, magnitude & 0xFF, 0))


@dataclass(frozen=True)
class LiftFrame:
    enabled: bool = False
    speed_mrad_s: int = 0

    def encode(self):
        if not -100 <= self.speed_mrad_s <= 100:
            raise ValueError("Lift speed is outside the configured limit")
        if not self.enabled and self.speed_mrad_s:
            raise ValueError("Lift speed requires the left Grip")
        return bytes((int(self.enabled),)) + struct.pack("b", self.speed_mrad_s)


@dataclass(frozen=True)
class Commands:
    chassis: ChassisFrame = ChassisFrame()
    lift: LiftFrame = LiftFrame()

    def encode(self):
        return self.chassis.encode() + self.lift.encode()

"""Pure VR-to-mobile command mapping shared by standalone and arm-integrated control."""

import math

from mobile_serial_protocol import ChassisFrame, Commands, LiftFrame
from vr_input import deadzone


class MobileMapper:
    def __init__(
        self,
        *,
        speed=0.05,
        spin=0.005,
        lift_speed=0.05,
        drive_sign=-1,
        steer_sign=1,
        spin_sign=1,
        lift_sign=1,
    ):
        for value, maximum in ((speed, 0.1), (spin, 0.01), (lift_speed, 0.1)):
            if not math.isfinite(value) or not 0 < value <= maximum:
                raise ValueError(f"Speed must be finite, positive and <= {maximum}")
        if any(value not in (-1, 1) for value in (drive_sign, steer_sign, spin_sign, lift_sign)):
            raise ValueError("Signs must be -1 or 1")
        self.speed, self.spin, self.lift_speed = speed, spin, lift_speed
        self.drive_sign, self.steer_sign = drive_sign, steer_sign
        self.spin_sign, self.lift_sign = spin_sign, lift_sign
        self.heading = 0.0
        self.spin_mode = False
        self.previous_click = False

    def step(self, controllers, dt):
        left, right = controllers["left"], controllers["right"]
        if not 0 < dt <= 0.1:
            # A delayed input sample must stop motion without changing the
            # wheel target or trying to repair the current configuration.
            self.previous_click = right.valid and right.grip and right.stick_pressed
            return Commands(
                ChassisFrame(0 if self.spin_mode else round(self.heading), 0, self.spin_mode),
                LiftFrame(),
            )
        dt = min(dt, 0.05)
        angle = 0 if self.spin_mode else round(self.heading)
        speed = 0
        spin = self.spin_mode
        if right.valid and right.grip:
            click = right.stick_pressed and not self.previous_click
            if click:
                self.spin_mode = not self.spin_mode
                # Keep the translation heading while in spin mode. Switching
                # back restores that heading instead of forcing wheel-centre.
                angle = 0 if self.spin_mode else round(self.heading)
                spin = self.spin_mode
            if self.spin_mode:
                # Keep turn_flag asserted at zero speed so the C board can
                # finish changing wheel configuration before rotation starts.
                spin = True
                # Mode selection and rotation are deliberately separate:
                # B/Y do nothing here; only the right stick commands yaw.
                yaw = -deadzone(float(right.stick[0]), 0.15)
                speed = round(yaw * self.spin_sign * self.spin * 1000)
            else:
                if left.valid and not left.grip:
                    self.heading -= deadzone(float(left.stick[0]), 0.15) * self.steer_sign * 10 * dt
                self.heading = max(-90, min(90, self.heading))
                angle = round(self.heading)
                speed = round(deadzone(float(right.stick[1]), 0.15) * self.drive_sign * self.speed * 1000)
        # Grip release or tracking loss is only a speed deadman. The stored
        # wheel angle and translation/spin flag keep being sent, allowing an
        # in-progress C-board configuration change to finish at its last target.
        self.previous_click = right.valid and right.grip and right.stick_pressed
        lift_speed = 0
        lift_enabled = left.valid and left.grip
        if lift_enabled:
            lift_speed = round(
                -deadzone(float(left.stick[1]), 0.15) * self.lift_sign * self.lift_speed * 1000
            )
        return Commands(
            ChassisFrame(angle, speed, spin),
            LiftFrame(lift_enabled, lift_speed),
        )

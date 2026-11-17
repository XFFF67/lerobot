"""Experimental wheel/contact dynamics for the original two-diagonal-link chassis.

No guide, steering actuator, or kinematic chassis pose update is added. Magnets
are an equivalent two-well force on the existing slides, not a field simulation.
Unmeasured tire, motor, bearing and magnet parameters are simulation defaults.
"""

from dataclasses import dataclass

import mujoco
import numpy as np

WHEELS = (3, 6, 9, 12)
STEERS = (2, 5, 8, 11)
STEER_SIGNS = np.array([1.0, 1.0, -1.0, -1.0])
SLIDE_SIGNS = np.array([1.0, -1.0])
WHEEL_RADIUS = 0.0984
# Axial centres measured from the imported wheel mesh bounds, in wheel frames.
WHEEL_CENTRES_Z = (-0.026383, -0.028290, -0.028290, 0.0212101)


@dataclass
class ChassisConfig:
    tire_friction: float = 1.0
    motor_torque: float = 4.0
    magnet_force: float = 16.0  # equivalent peak generalized force per slide, N
    magnet_distance: float = 0.01  # exponential decay distance, m
    linear_speed: float = 0.10
    yaw_speed: float = 0.10
    steering_speed: float = np.deg2rad(30)


def add_chassis_physics(spec, config):
    """Modify an in-memory MjSpec; preserve the source CAD and linkage topology."""
    # The original keyframe has no free-root coordinates. qpos0 is the same home.
    for key in list(spec.keys):
        spec.delete(key)
    for geom in spec.geoms:
        geom.contype = geom.conaffinity = 0
    spec.body("base_link").add_freejoint(name="chassis_free")
    spec.option.gravity = [0, 0, -9.81]
    spec.option.timestep = 0.001
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    floor = spec.geom("scene_ground")
    if floor is None:
        floor = spec.worldbody.add_geom(
            name="chassis_ground", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05]
        )
    floor.pos = [0, 0, -0.264]
    floor.contype, floor.conaffinity = 1, 2
    floor.friction = [config.tire_friction, 0.005, 0.0001]

    # Replace all original servos in memory. Upper-body target servos are added
    # below because the imported gripper actuators use different transmissions.
    for actuator in spec.actuators:
        actuator.gainprm[:] = actuator.biasprm[:] = 0
    for number, centre_z in zip(WHEELS, WHEEL_CENTRES_Z, strict=True):
        name = f"link_{number:03d}_joint"
        joint = spec.joint(name)
        joint.limited = False
        joint.damping, joint.armature = 0.01, 0.01
        actuator = spec.actuator(f"{name}_position")
        actuator.gainprm[0] = 1
        actuator.ctrlrange = [-config.motor_torque, config.motor_torque]
        actuator.forcerange = [-config.motor_torque, config.motor_torque]
        actuator.forcelimited = True
        spec.body(f"link_{number:03d}").add_geom(
            name=f"chassis_tire_{number}",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            pos=[0, 0, centre_z],
            size=[WHEEL_RADIUS, 0.026],
            contype=2,
            conaffinity=1,
            friction=[config.tire_friction, 0.005, 0.0001],
            condim=6,
            group=3,
            mass=0,
        )
    for number in range(2, 16):
        joint = spec.joint(f"link_{number:03d}_joint")
        if joint is None:
            continue
        joint.actfrclimited = False
        joint.solref_limit = [0.004, 1]
        joint.solimp_limit = [0.99, 0.999, 0.0001, 0.5, 2]
        # Finite passive bearing resistance, including the exported coaxial
        # link_013 hinge. This is friction, not a lock or a steering motor.
        joint.frictionloss = 0.15 if number == 13 else 0.02
    for joint in list(spec.joints):
        if joint.name.startswith("openarm_") or joint.name in {
            f"link_{n:03d}_joint" for n in (16, 17, 18, 19)
        }:
            joint.armature = 0.01
            joint.actfrclimited = False
            actuator = spec.add_actuator(
                name=f"physics_hold_{joint.name}", target=joint.name, trntype=mujoco.mjtTrn.mjTRN_JOINT
            )
            actuator.set_to_position(kp=1000, kv=50)


class ChassisPhysics:
    """Wheel-only mode switching, endpoint capture and motor velocity control."""

    def __init__(self, model, data, config):
        self.model, self.config = model, config
        self.base = model.body("base_link").id
        wheel_joints = [model.joint(f"link_{n:03d}_joint").id for n in WHEELS]
        self.wheel_dofs = model.jnt_dofadr[wheel_joints]
        self.wheel_joints = np.array(wheel_joints)
        self.tires = np.array([model.geom(f"chassis_tire_{n}").id for n in WHEELS])
        steering = [model.joint(f"link_{n:03d}_joint").id for n in STEERS]
        self.steer_qpos = model.jnt_qposadr[steering]
        self.steer_dofs = model.jnt_dofadr[steering]
        rotation = data.xmat[self.base].reshape(3, 3)
        self.steering_levers = np.empty(4)
        for i, joint in enumerate(wheel_joints):
            rolling = np.cross(rotation.T @ data.xaxis[joint], [0, 0, 1])
            offset = rotation.T @ (data.geom_xpos[self.tires[i]] - data.xanchor[steering[i]])
            self.steering_levers[i] = rolling @ np.cross([0, 0, 1], offset) / WHEEL_RADIUS
            if i == 0:
                self.wheel_heading_offset = float(np.arctan2(rolling[1], rolling[0]))
        # Parallel steering also counter-rotates the two diagonal-bar hinges.
        limits = model.jnt_range[steering].copy()
        bar_limits = np.array([model.joint(f"link_{n:03d}_joint").range for n in (4, 10)])
        self.parallel_limits = (
            max(-np.pi / 2, np.max(limits[:, 0]), np.max(-bar_limits[:, 1])) + 0.001,
            min(np.pi / 2, np.min(limits[:, 1]), np.min(-bar_limits[:, 0])) - 0.001,
        )
        self.motors = np.array([model.actuator(f"link_{n:03d}_joint_position").id for n in WHEELS])
        slides = [model.joint(f"link_{n:03d}_joint").id for n in (14, 15)]
        self.slide_qpos = model.jnt_qposadr[slides]
        self.slide_dofs = model.jnt_dofadr[slides]
        ranges = model.jnt_range[slides]
        self.travel = np.array([ranges[0, 1], -ranges[1, 0]])
        if not np.allclose([ranges[0, 0], ranges[1, 1]], 0) or np.any(self.travel <= 0):
            raise ValueError("Expected the original opposed slides with a zero-angle front endpoint")
        self.hold_actuators = np.array(
            [i for i in range(model.nu) if model.actuator(i).name.startswith("physics_hold_")]
        )
        upper = model.actuator_trnid[self.hold_actuators, 0]
        self.upper_qpos, self.upper_dofs = model.jnt_qposadr[upper], model.jnt_dofadr[upper]
        self.reset(data)

    def reset(self, data):
        self.upper_target = data.qpos[self.upper_qpos].copy()
        self.target_mode = "translation"
        self.phase = "ready"
        self.switch_elapsed = self.phase_elapsed = self.stable_elapsed = self.accumulator = 0.0
        self.last_status = "translation: front magnetic endpoint"
        self.cancelled = False
        self.faulted = False
        self.drive_targets = np.zeros(4)
        self.parallel_angle = 0.0
        self.parallel_integral = np.zeros(4)
        self.translation_active = False
        self.translation_heading = None
        data.ctrl[:] = 0
        data.qfrc_applied[:] = 0

    def request_mode(self, mode, data):
        if self.faulted:
            return
        if mode == self.target_mode == "spin" and self.phase == "ready" and not self.cancelled:
            return
        # T explicitly recentres a parallel wheel group. G instead starts the
        # reconfiguration at the measured angles, without a zero-angle waypoint.
        front = self.target_mode == "translation" and np.max(np.abs(self.slide_positions(data))) < 0.003
        self.target_mode, self.phase = mode, "straighten" if front and mode == "translation" else "switch"
        self.switch_elapsed = self.phase_elapsed = self.stable_elapsed = 0.0
        self.parallel_integral[:] = 0
        self.translation_active = False
        self.cancelled = False
        self.last_status = f"switching to {mode}"

    def stop(self):
        self.parallel_integral[:] = 0
        self.parallel_angle = None
        self.translation_active = False
        if self.phase != "ready":
            self.cancelled = True
            self.phase = "stopped"
            self.last_status = "switch cancelled; press T/G to retry"

    def slide_positions(self, data):
        return data.qpos[self.slide_qpos] * SLIDE_SIGNS

    def encoder_degrees(self, data):
        return np.rad2deg(data.qpos[self.steer_qpos])

    def magnet_forces(self, data):
        q = self.slide_positions(data)
        length = self.config.magnet_distance
        # Integrable, bounded attraction toward BOTH endpoints at all times.
        # Changing mode does not turn either permanent magnet on/off.
        front = np.exp(-np.maximum(q, 0) / length)
        rear = np.exp(-np.maximum(self.travel - q, 0) / length)
        return self.config.magnet_force * (rear - front) * SLIDE_SIGNS

    def _rolling_targets(self, data, velocity, omega):
        rotation = data.xmat[self.base].reshape(3, 3)
        result = np.empty(4)
        for i, joint in enumerate(self.wheel_joints):
            rolling = np.cross(rotation.T @ data.xaxis[joint], [0, 0, 1])
            position = rotation.T @ (data.geom_xpos[self.tires[i]] - data.xpos[self.base])
            result[i] = rolling @ (velocity + np.cross([0, 0, omega], position)) / WHEEL_RADIUS
        return result

    def _translation_targets(self, data, linear, steering, scale, *, straighten=False):
        """Adjust a retained wheel angle; drive forward/back along that angle.

        The passive steering pivot is laterally offset from the tire centre.
        Rolling the tire around that pivot contributes lever * steering_rate
        to its motor speed. No force/position command goes to a steering joint.
        """
        dt = self.model.opt.timestep
        rotation = data.xmat[self.base].reshape(3, 3)
        heading = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
        angles = data.qpos[self.steer_qpos]
        active = bool(linear or steering or straighten)
        if self.parallel_angle is None:
            self.parallel_angle = float(np.clip(np.mean(angles), *self.parallel_limits))
        if active:
            if not self.translation_active:
                self.translation_heading = heading
            if straighten:
                self.parallel_angle = 0.0
            elif steering:
                angle = self.parallel_angle + steering * self.config.steering_speed * scale * dt
                self.parallel_angle = float(np.clip(angle, *self.parallel_limits))
        self.translation_active = active
        # W/S changes only rolling speed, never the retained steering target.
        # Roll along the measured wheel heading while the retained target turns.
        # Waiting for a small target error blocks travel throughout a swivel.
        direction = float(np.mean(angles)) + self.wheel_heading_offset
        velocity = (
            linear * self.config.linear_speed * scale * np.array([np.cos(direction), np.sin(direction), 0.0])
        )
        error = self.parallel_angle - angles
        rates = np.zeros(4)
        omega = 0.0
        if active:
            # Only integrate near the target; winding up during a 90° swivel
            # would keep pushing against the original joint stops.
            self.parallel_integral = np.where(
                np.abs(error) < 0.15, np.clip(self.parallel_integral + error * dt, -0.3, 0.3), 0
            )
            rates = np.clip(
                8 * error + 1.5 * self.parallel_integral + 10 * (np.mean(angles) - angles),
                -2 * scale,
                2 * scale,
            )
            yaw_error = np.arctan2(
                np.sin(self.translation_heading - heading), np.cos(self.translation_heading - heading)
            )
            omega = float(np.clip(4 * yaw_error, -0.15, 0.15))
        else:
            self.parallel_integral[:] = 0
            # Hold the selected wheel angle after releasing the keys.
            rates = np.clip(8 * error, -1, 1)
        yaw_targets = self._rolling_targets(data, np.zeros(3), omega)
        # Heading correction must not counteract the common swivel command.
        # Remove its component along the wheel-speed vector that steers all
        # four wheels together. Near straight-ahead the chassis is strongly
        # underactuated; lateral contact, rather than a hidden base servo,
        # provides the remaining heading restraint.
        yaw_targets -= self.steering_levers * (
            yaw_targets @ self.steering_levers / (self.steering_levers @ self.steering_levers)
        )
        result = self._rolling_targets(data, velocity, 0.0) + yaw_targets
        self.drive_targets += np.clip(result - self.drive_targets, -dt, dt)
        return self.drive_targets + self.steering_levers * rates, 2.0

    @staticmethod
    def _alignment_targets(error, scale):
        """Correct all four residuals with one shared wheel-speed limit."""
        velocity = -8 * error
        velocity *= min(1.0, scale / max(float(np.max(np.abs(velocity))), 1e-9))
        return velocity

    def _wheel_targets(self, data, linear, steering, yaw, scale):
        dt = self.model.opt.timestep
        desired = STEER_SIGNS * (np.deg2rad(53) if self.target_mode == "spin" else 0)
        error = desired - data.qpos[self.steer_qpos]
        gap = (
            self.travel - self.slide_positions(data)
            if self.target_mode == "spin"
            else self.slide_positions(data)
        )
        if self.phase in ("straighten", "switch", "capture", "align"):
            # Reconfiguration has its own wheel controller. Scaling its speed
            # commands also reduces the P velocity servo's breakaway torque:
            # the passive linkage can stay on the old magnetic endpoint even
            # while individual wheels turn. Fine control applies only to manual
            # driving/steering, never to this automatic endpoint transition.
            scale = 1.0
            self.switch_elapsed += dt
            self.phase_elapsed += dt
            if self.switch_elapsed > 15:
                self.stop()
                self.last_status = "switch timed out; check friction/magnets, press T/G to retry"
                return np.zeros(4), 2.0
            if self.phase == "straighten":
                targets = self._translation_targets(data, 0, 0, scale, straighten=True)
                settled = (
                    np.max(np.abs(data.qpos[self.steer_qpos])) < np.deg2rad(1.5)
                    and np.max(np.abs(data.qvel[self.steer_dofs])) < 0.04
                )
                self.stable_elapsed = self.stable_elapsed + dt if settled else 0.0
                if self.stable_elapsed > 0.2:
                    self.phase = "ready"
                    self.phase_elapsed = self.stable_elapsed = 0.0
                    self.parallel_integral[:] = 0
                    self.translation_active = False
                    self.last_status = "translation: front magnetic endpoint reached"
                return targets
            if self.phase == "switch":
                if np.max(gap) < 0.012 and self.phase_elapsed > 0.5:
                    self.phase, self.phase_elapsed = "capture", 0.0
                else:
                    # Bent arms shift wheel loading. Track the requested wheel
                    # speeds firmly enough to release the magnetic endpoint;
                    # advance() still enforces the same motor torque ceiling.
                    return np.clip(-8 * error, -2, 2) * scale, 4.0
            if self.phase == "capture":
                # Let the magnets move the linkage's common coordinates while
                # opposite wheel torques keep each linked pair aligned. Strong
                # absolute-angle braking here would prevent endpoint seating.
                if np.max(np.abs(gap)) < 0.001 and self.phase_elapsed > 0.5:
                    self.phase, self.phase_elapsed = "align", 0.0
                else:
                    angles = data.qpos[self.steer_qpos]
                    paired_error = np.repeat(angles.reshape(2, 2).mean(axis=1), 2) - angles
                    correction = np.clip(-8 * paired_error, -scale, scale)
                    # Equivalent to -0.05 * wheel velocity damping plus the
                    # pair correction; only bounded wheel motor torque acts.
                    return 0.975 * data.qvel[self.wheel_dofs] + correction, 2.0
            if self.phase == "align":
                # All four drives participate in endpoint alignment, including
                # the wheel on the export's additional passive link_013 hinge.
                velocity = self._alignment_targets(error, scale)
                settled = (
                    np.max(np.abs(error)) < np.deg2rad(1.5)
                    and np.max(np.abs(gap)) < 0.001
                    and np.max(np.abs(data.qvel[self.steer_dofs])) < 0.04
                )
                self.stable_elapsed = self.stable_elapsed + dt if settled else 0.0
                if self.stable_elapsed > 0.2:
                    self.phase = "ready"
                    self.parallel_angle = 0.0
                    self.last_status = f"{self.target_mode}: magnetic endpoint reached"
                return velocity, 4.0
        if self.phase != "ready":
            return np.zeros(4), 2.0
        # Only accept driving while the sliders are seated. A slip/detachment
        # must be observed, never silently converted into the requested mode.
        # The front latch permits a common wheel angle anywhere in its range.
        # Check parallelism there, rather than incorrectly requiring 0°.
        angles = data.qpos[self.steer_qpos]
        bad_angle = (
            np.max(np.abs(error)) > np.deg2rad(8)
            if self.target_mode == "spin"
            else np.ptp(angles) > np.deg2rad(12)
        )
        if np.max(np.abs(gap)) > 0.003 or bad_angle:
            self.phase = "stopped"
            self.cancelled = True
            self.last_status = "endpoint lost; press T/G to reseat"
            return np.zeros(4), 2.0
        if self.target_mode == "translation":
            return self._translation_targets(data, linear, steering, scale)
        omega = yaw * self.config.yaw_speed * scale
        result = self._rolling_targets(data, np.zeros(3), omega)
        # Ramp wheel speeds so starting a drive does not jerk the passive
        # mechanism out of its magnetic endpoint.
        self.drive_targets += np.clip(result - self.drive_targets, -dt, dt)
        targets = self.drive_targets.copy()
        if yaw:
            targets[3] += np.clip(-8 * error[3], -1, 1)
        return targets, 2.0

    def advance(self, data, dt, *, linear=0.0, steering=0.0, yaw=0.0, scale=1.0, braking=False):
        if self.faulted:
            return
        if braking:
            self.stop()
        self.accumulator += dt
        step = self.model.opt.timestep
        while self.accumulator + 1e-12 >= step:
            self.accumulator -= step
            mujoco.mj_forward(self.model, data)
            data.ctrl[:] = 0
            data.ctrl[self.hold_actuators] = self.upper_target
            data.qfrc_applied[:] = 0
            # Ideal upper-body servo support. No compensation forces are put on
            # the free base, passive chassis hinges, or wheel drives.
            data.qfrc_applied[self.upper_dofs] = data.qfrc_bias[self.upper_dofs]
            data.qfrc_applied[self.slide_dofs] = self.magnet_forces(data) - 2 * data.qvel[self.slide_dofs]
            targets, gain = (
                (np.zeros(4), 2.0) if braking else self._wheel_targets(data, linear, steering, yaw, scale)
            )
            if self.phase not in ("ready", "straighten") or braking:
                self.drive_targets[:] = 0
            data.ctrl[self.motors] = np.clip(
                gain * (targets - data.qvel[self.wheel_dofs]),
                -self.config.motor_torque,
                self.config.motor_torque,
            )
            previous_time = data.time
            mujoco.mj_step(self.model, data)
            if data.time < previous_time or not np.all(np.isfinite(data.qpos)):
                self.faulted = True
                self.phase = "fault"
                self.last_status = "physics failed; Backspace to reset"
                data.ctrl[:] = 0
                break
        mujoco.mj_forward(self.model, data)

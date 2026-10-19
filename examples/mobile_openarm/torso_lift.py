"""Kinematic lift control for the mjcf1 hinged mast and two-stage slide.

The exported lower hinge is tilted relative to the upper hinge, which locks the
nominally planar closure. Align it in memory for this simulation profile only.
This is a model correction, not a calibration of the physical robot.
"""

import mujoco
import numpy as np


class TorsoLift:
    CLOSURE_TOLERANCE_M = 1e-6

    def __init__(self, model, data):
        self.model = model
        self.base = self._id("body", "base_link")
        self.joints = np.array([self._id("joint", f"link_{n:03}_joint") for n in (16, 19, 17, 18)])
        self.qpos = model.jnt_qposadr[self.joints].copy()
        if list(model.jnt_type[self.joints]) != [
            mujoco.mjtJoint.mjJNT_HINGE,
            mujoco.mjtJoint.mjJNT_HINGE,
            mujoco.mjtJoint.mjJNT_SLIDE,
            mujoco.mjtJoint.mjJNT_SLIDE,
        ]:
            raise ValueError("Lift requires the mjcf1 two-hinge/two-slide torso")
        self.sites = [self._id("site", f"closure_011_{side}_site") for side in "ab"]
        self.height_body = self._id("body", "openarm_left_link0")
        actuator = self._id("actuator", "link_019_joint_position")
        if not model.actuator_ctrllimited[actuator]:
            raise ValueError("Lift requires finite drive limits")
        self.drive_limits = model.actuator_ctrlrange[actuator].copy()
        self.hinge_limits = model.jnt_range[self.joints[0]].copy()
        self.slide_limits = model.jnt_range[self.joints[2:]].copy()
        self.scratch = mujoco.MjData(model)
        self.scratch.qpos[:] = data.qpos
        if np.max(np.abs(data.qpos[self.qpos])) > 1e-10 or self.drive_limits[0] != 0:
            raise ValueError("Lift profile expects the original zero-angle home configuration")

        mujoco.mj_kinematics(model, self.scratch)
        lower, upper = self.joints[:2]
        lower_axis = self.scratch.xaxis[lower]
        upper_axis = self.scratch.xaxis[upper].copy()
        self.axis_correction_deg = float(np.rad2deg(np.arccos(np.clip(lower_axis @ upper_axis, -1, 1))))
        if self.axis_correction_deg > 2:
            raise ValueError("Lift hinge alignment differs by more than 2 degrees; check the model")
        # At home the two hinge rotations are identity, so this leaves the CAD
        # pose intact while allowing the connect constraint to remain closed.
        rotation = self.scratch.xmat[model.jnt_bodyid[lower]].reshape(3, 3)
        model.jnt_axis[lower] = rotation.T @ upper_axis
        mujoco.mj_forward(model, data)

        # Tabulate only the scalar height-to-drive relation. Each accepted motion
        # recomputes the closed pose; interpolated joint angles are never applied.
        self.drives = np.linspace(*self.drive_limits, 257)
        heights = []
        for drive in self.drives:
            self._closed_configuration(float(drive))
            heights.append(self.height(self.scratch))
        self.heights = np.asarray(heights)
        if not np.all(np.diff(self.heights) < 0):
            raise ValueError("Lift height must decrease monotonically along the configured drive range")
        self.min_height, self.max_height = float(self.heights[-1]), float(self.heights[0])
        self.last_status = "holding"

    def _id(self, kind, name):
        index = mujoco.mj_name2id(self.model, getattr(mujoco.mjtObj, f"mjOBJ_{kind.upper()}"), name)
        if index < 0:
            raise ValueError(f"Missing lift {kind}: {name}")
        return index

    def closure_error(self, data):
        return float(np.linalg.norm(data.site_xpos[self.sites[0]] - data.site_xpos[self.sites[1]]))

    def height(self, data):
        """Height in the chassis frame, also valid after the free base moves."""
        relative = data.xmat[self.base].reshape(3, 3).T @ (data.xpos[self.height_body] - data.xpos[self.base])
        return float(relative[2])

    def _closed_configuration(self, drive):
        """Solve lower hinge plus equal slide extension for a fixed upper hinge."""
        d, m = self.scratch, self.model
        d.qpos[self.qpos] = [0, drive, 0, 0]

        def evaluate(angle):
            d.qpos[self.qpos[0]] = angle
            mujoco.mj_kinematics(m, d)
            return d.xmat[self.base].reshape(3, 3).T @ (
                d.site_xpos[self.sites[0]] - d.site_xpos[self.sites[1]]
            )

        if drive == 0:
            evaluate(0)
        else:
            low, high = self.hinge_limits
            if evaluate(low)[0] < 0 or evaluate(high)[0] > self.CLOSURE_TOLERANCE_M:
                raise ValueError("No closed lift configuration within the lower hinge limits")
            for _ in range(32):
                mid = (low + high) / 2
                if evaluate(mid)[0] > 0:
                    low = mid
                else:
                    high = mid
            error = evaluate((low + high) / 2)
            # Both slides translate along base Z. Their sum closes the linkage;
            # equal sharing selects a continuous pose for the redundant stages.
            d.qpos[self.qpos[2:]] = error[2] / 2
            mujoco.mj_kinematics(m, d)
        values = d.qpos[self.qpos].copy()
        if (
            not np.all(np.isfinite(values))
            or np.any(values[2:] < self.slide_limits[:, 0] - 1e-10)
            or np.any(values[2:] > self.slide_limits[:, 1] + 1e-10)
            or self.closure_error(d) > self.CLOSURE_TOLERANCE_M
        ):
            raise ValueError("Lift pose violates a slide limit or the closure tolerance")
        return values

    def step(self, data, direction, speed, dt):
        if not direction or dt <= 0:
            return
        height = self.height(data)
        target = float(np.clip(height + direction * speed * dt, self.min_height, self.max_height))
        if abs(target - height) < 1e-8:
            self.last_status = "at height limit"
            return
        drive = float(np.interp(target, self.heights[::-1], self.drives[::-1]))
        self.scratch.qpos[:] = data.qpos
        try:
            candidate = self._closed_configuration(drive)
        except ValueError as exc:
            self.last_status = f"holding: {exc}"
            return
        data.qpos[self.qpos] = candidate
        self.last_status = "raising" if direction > 0 else "lowering"

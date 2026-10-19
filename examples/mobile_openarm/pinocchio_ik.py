"""Arm-only Pinocchio kinematics and CasADi/IPOPT IK for the loaded MJCF.

The full robot has closed-chain torso constraints. Extract each serial arm from
MuJoCo's compiled transforms instead of asking Pinocchio to parse that mechanism.
No meshes, inertias or dynamics are needed for this position-level IK model.
"""

from dataclasses import dataclass

import mujoco
import numpy as np

try:
    import casadi as ca
    import pinocchio as pin
    from pinocchio import casadi as cpin
except ImportError as exc:
    raise ImportError(
        "Pinocchio with CasADi bindings is required. Start with ./run_keyboard_sim.sh "
        "or activate hei-rebot-vr; see README_zh.md for environment setup."
    ) from exc


@dataclass
class IKResult:
    q: np.ndarray
    residual: float
    accepted: bool
    status: str


def body_pose(data: mujoco.MjData, body: int) -> pin.SE3:
    return pin.SE3(data.xmat[body].reshape(3, 3).copy(), data.xpos[body].copy())


class PinocchioCasadiIK:
    """One seven-joint chain, with targets expressed in the current world frame."""

    def __init__(self, mj_model, mj_data, joints, tcp, smooth_weight=1e-4):
        self.qpos = mj_model.jnt_qposadr[joints].copy()
        self.root_body = int(mj_model.body_parentid[mj_model.jnt_bodyid[joints[0]]])
        self.tcp = int(tcp)
        self.last_status = "not solved"

        # Reject unmapped moving joints between the root and TCP.
        chain = []
        body = self.tcp
        while body != self.root_body:
            if body == 0:
                raise ValueError("TCP is not a descendant of the arm root")
            start = mj_model.body_jntadr[body]
            chain[0:0] = list(range(start, start + mj_model.body_jntnum[body]))
            body = int(mj_model.body_parentid[body])
        if chain != list(joints) or any(mj_model.body_jntnum[mj_model.jnt_bodyid[j]] != 1 for j in joints):
            raise ValueError("Pinocchio IK requires a serial arm with one mapped hinge per moving body")

        zero = mujoco.MjData(mj_model)
        zero.qpos[:] = mj_data.qpos
        zero.qpos[self.qpos] = 0
        mujoco.mj_forward(mj_model, zero)
        root_inverse = body_pose(zero, self.root_body).inverse()
        previous = pin.SE3.Identity()
        self.model = pin.Model()
        parent = 0
        for joint in joints:
            if mj_model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_HINGE:
                raise ValueError("Only hinge joints are supported in the arm IK chain")
            body = int(mj_model.jnt_bodyid[joint])
            # A joint frame is at its anchor, with the body's zero-pose orientation.
            placement = root_inverse * pin.SE3(
                zero.xmat[body].reshape(3, 3).copy(), zero.xanchor[joint].copy()
            )
            parent = self.model.addJoint(
                parent,
                pin.JointModelRevoluteUnaligned(mj_model.jnt_axis[joint]),
                previous.inverse() * placement,
                mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint),
            )
            previous = placement
        tcp_placement = previous.inverse() * root_inverse * body_pose(zero, self.tcp)
        self.frame = self.model.addFrame(pin.Frame("tcp", parent, 0, tcp_placement, pin.FrameType.OP_FRAME))
        limited = mj_model.jnt_limited[joints].astype(bool)
        self.lower = np.where(limited, mj_model.jnt_range[joints, 0], -np.inf)
        self.upper = np.where(limited, mj_model.jnt_range[joints, 1], np.inf)
        self.model.lowerPositionLimit[:] = self.lower
        self.model.upperPositionLimit[:] = self.upper
        self.data = self.model.createData()

        # Pinocchio supplies symbolic FK; CasADi differentiates the objective;
        # IPOPT solves for joint angles subject to position and per-step limits.
        symbolic_model = cpin.Model(self.model)
        symbolic_data = symbolic_model.createData()
        q = ca.SX.sym("q", self.model.nq)
        target_position = ca.SX.sym("target_position", 3)
        target_rotation = ca.SX.sym("target_rotation", 3, 3)
        current_q = ca.SX.sym("current_q", self.model.nq)
        cpin.framesForwardKinematics(symbolic_model, symbolic_data, q)
        pose = symbolic_data.oMf[self.frame]
        error = ca.vertcat(
            pose.translation - target_position,
            # Smooth at identical orientations, including second derivatives.
            # log3's symbolic branches can produce NaN derivatives at zero.
            (0.2 / np.sqrt(2)) * ca.vec(pose.rotation - target_rotation),
        )
        params = ca.vertcat(target_position, ca.vec(target_rotation), current_q)
        self.error_function = ca.Function("tcp_error", [q, params], [error])
        # Scaling keeps small keyboard increments above IPOPT's stopping tolerance.
        objective = 1e4 * (ca.sumsqr(error) + smooth_weight * ca.sumsqr(q - current_q))
        self.solver = ca.nlpsol(
            "arm_ik",
            "ipopt",
            {"x": q, "p": params, "f": objective},
            {
                "print_time": False,
                "error_on_fail": False,
                "ipopt.print_level": 0,
                "ipopt.sb": "yes",
                "ipopt.max_iter": 50,
                "ipopt.tol": 1e-8,
                "ipopt.acceptable_tol": 1e-6,
                "ipopt.bound_relax_factor": 0.0,
            },
        )
        self.validate_fk(mj_data)

    def forward_world(self, q, mj_data):
        pin.framesForwardKinematics(self.model, self.data, np.asarray(q))
        return body_pose(mj_data, self.root_body) * self.data.oMf[self.frame]

    def validate_fk(self, mj_data):
        pose = self.forward_world(mj_data.qpos[self.qpos], mj_data)
        np.testing.assert_allclose(pose.translation, mj_data.xpos[self.tcp], atol=1e-9, rtol=0)
        np.testing.assert_allclose(pose.rotation, mj_data.xmat[self.tcp].reshape(3, 3), atol=1e-9, rtol=0)

    def solve(self, mj_data, position, orientation, max_step):
        current = mj_data.qpos[self.qpos].copy()
        target = body_pose(mj_data, self.root_body).inverse() * pin.SE3(orientation, position)
        params = np.concatenate((target.translation, target.rotation.ravel(order="F"), current))
        initial_error = float(np.linalg.norm(self.error_function(current, params)))
        lower = np.maximum(self.lower, current - max_step)
        upper = np.minimum(self.upper, current + max_step)
        try:
            solution = self.solver(x0=current, p=params, lbx=lower, ubx=upper)
            stats = self.solver.stats()
            self.last_status = stats["return_status"]
            candidate = np.asarray(solution["x"]).reshape(-1)
            if (
                stats["success"]
                and np.all(np.isfinite(candidate))
                and np.all(candidate >= lower - 1e-9)
                and np.all(candidate <= upper + 1e-9)
            ):
                candidate = np.clip(candidate, lower, upper)
                residual = float(np.linalg.norm(self.error_function(candidate, params)))
                if np.isfinite(residual) and residual <= initial_error + 1e-10:
                    return IKResult(candidate, residual, True, self.last_status)
                self.last_status = "rejected: pose error increased"
            elif stats["success"]:
                self.last_status = "rejected: non-finite or out-of-bounds solution"
        except RuntimeError as exc:
            self.last_status = f"solver failed: {str(exc).splitlines()[-1]}"
        # A failed or invalid solve must never send IPOPT's intermediate iterate.
        return IKResult(current, initial_error, False, self.last_status)

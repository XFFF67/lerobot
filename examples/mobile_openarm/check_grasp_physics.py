"""Headless contact regression: support, squeeze, lift, hold, release and fall.

The arm is initialized at a collision-free pregrasp pose using Pinocchio IK.
All subsequent motion uses the normal keyboard controller and MuJoCo stepping;
no object weld, attachment, or pose update is used to perform a grasp.
"""

import mujoco
import numpy as np
from mujoco_scene import CUBE_HALF_SIZE, GROUND_Z, OBJECT_NAMES, TABLE_TOP_Z


def check_simulator(sim):
    model, data, chassis = sim.model, sim.data, sim.chassis
    assert chassis is not None and sim.scene_enabled
    objects = [model.body(name).id for name in OBJECT_NAMES]
    object_joints = model.body_jntadr[objects]
    assert np.all(model.jnt_type[object_joints] == mujoco.mjtJoint.mjJNT_FREE)
    assert np.all(model.body_mass[objects] > 0)
    object_dofs = np.concatenate([np.arange(dof, dof + 6) for dof in model.jnt_dofadr[object_joints]])

    def tick(seconds):
        for _ in range(round(seconds / 0.02)):
            sim.step(0.02)
            assert not chassis.faulted and np.all(np.isfinite(data.qpos))
            np.testing.assert_array_equal(data.qfrc_applied[object_dofs], 0)

    def touching(body, other):
        return any(
            {model.geom_bodyid[contact.geom1], model.geom_bodyid[contact.geom2]} == {body, other}
            for contact in data.contact
        )

    sim.reset()
    tick(2)
    table = model.body("scene_table").id
    for body in objects:
        assert touching(body, table), model.body(body).name
    np.testing.assert_allclose(data.xpos[objects[:3], 2], TABLE_TOP_Z + CUBE_HALF_SIZE, atol=0.001)
    print("PASS four free objects supported by table contacts")

    for side, name, key in (("left", "scene_cube_blue", "3"), ("right", "scene_cube_red", "4")):
        sim.reset()
        tick(2)
        sim.press("1")
        sim.press("w")
        tick(2)
        sim.release("w")
        tick(1)
        arm = sim.arms[side]
        body = model.body(name).id
        rotation = np.diag([-1.0, 1.0, -1.0])
        # Measured mesh fingertip region, relative to the hand body. The user
        # TCP may be the motor axis or the original model tool reference.
        grasp_offset = np.array([0, 0.006, 0.075]) - model.body_pos[arm.tcp]
        position = data.xpos[body] + [0, 0, 0.14] - rotation @ grasp_offset
        scratch = mujoco.MjData(model)
        scratch.qpos[:] = data.qpos
        seed = np.array([-0.8, -0.9, 0.9, 1.5, -1.1, 0.5, 1.5])
        if side == "right":
            seed *= [-1, -1, -1, 1, -1, -1, -1]
        scratch.qpos[arm.qpos] = seed
        mujoco.mj_forward(model, scratch)
        for _ in range(4):
            result = arm.ik.solve(scratch, position, rotation, 6.0)
            scratch.qpos[arm.qpos] = result.q
            mujoco.mj_forward(model, scratch)
        assert result.accepted and result.residual < 0.002, result
        data.qpos[arm.qpos] = result.q
        data.qpos[arm.fingers] = model.jnt_range[arm.finger_joints, 1]
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        chassis.reset(data)
        tick(1)
        sim.press(key)
        sim.press("f")
        tick(1.17)
        sim.release("f")
        tick(1)
        sim.press("x")
        tick(1.5)
        sim.release("x")
        tick(1)
        fingers = [model.body(f"openarm_{side}_{finger}_finger").id for finger in ("left", "right")]
        assert all(touching(body, finger) for finger in fingers), "Both jaws must contact the object"
        assert np.min(data.qpos[arm.fingers]) > 0.02, "Object must stop the jaws closing through it"
        original_height = data.xpos[body, 2]
        sim.press("r")
        tick(1.2)
        sim.release("r")
        tick(1)
        held_height = data.xpos[body, 2]
        tick(8)
        assert data.xpos[body, 2] > original_height + 0.12
        assert abs(data.xpos[body, 2] - held_height) < 0.005, "Sustained grasp must not creep downward"
        assert all(touching(body, finger) for finger in fingers)
        assert not touching(body, table)
        for number in (1, 2):
            actuator = model.actuator(f"physics_hold_openarm_{side}_finger_joint{number}").id
            assert abs(data.actuator_force[actuator]) <= 15.000001
        sim.press("z")
        tick(1.5)
        sim.release("z")
        tick(1)
        assert touching(body, table)
        assert not any(touching(body, finger) for finger in fingers)
        assert abs(data.xpos[body, 2] - original_height) < 0.001
        print(f"PASS {side}: close, lift >12 cm, hold 8 s, open and drop back onto table")

    # Fixture outside the table: falling objects must collide with the floor.
    sim.reset()
    body = model.body("scene_cube_green").id
    qpos = model.jnt_qposadr[model.body_jntadr[body]]
    data.qpos[qpos : qpos + 3] = [0.4, 1.0, 0.8]
    mujoco.mj_forward(model, data)
    tick(2)
    assert touching(body, 0)
    assert abs(data.xpos[body, 2] - GROUND_Z - CUBE_HALF_SIZE) < 0.001
    sim.reset()
    print("PASS released objects also collide with the floor; reset restores the scene")

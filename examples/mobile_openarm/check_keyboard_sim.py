"""Behavior checks against the selected MJCF, without a window."""

import mujoco
import numpy as np


def check_scene(sim):
    from mujoco_scene import GROUND_Z, OBJECT_NAMES, TABLE_TOP_Z

    floor = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_GEOM, "scene_ground")
    if not sim.scene_enabled:
        assert floor == -1
        print("PASS plain scene: no environment geometry")
        return
    assert floor >= 0
    np.testing.assert_allclose(sim.model.geom_pos[floor, 2], GROUND_Z - 0.002)
    for name in OBJECT_NAMES:
        body = sim.id("body", name)
        assert sim.model.body_jntnum[body] == 0
        assert sim.data.xpos[body, 2] > TABLE_TOP_Z
    for index in range(sim.model.ngeom):
        name = mujoco.mj_id2name(sim.model, mujoco.mjtObj.mjOBJ_GEOM, index) or ""
        if name.startswith("scene_"):
            assert sim.model.geom_contype[index] == 0 and sim.model.geom_conaffinity[index] == 0
    assert sim.model.nlight >= 2

    # Scene construction preserves model coordinates and the mechanical home;
    # the simulator selects a separate working pose with both elbows at 90 deg.
    original = mujoco.MjModel.from_xml_path(str(sim.args.mjcf.expanduser().resolve()))
    original_data = mujoco.MjData(original)
    key = mujoco.mj_name2id(original, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key >= 0:
        mujoco.mj_resetDataKeyframe(original, original_data, key)
    mujoco.mj_forward(original, original_data)
    assert (sim.model.nq, sim.model.nv, sim.model.nu) == (original.nq, original.nv, original.nu)
    np.testing.assert_array_equal(sim.model.jnt_qposadr, original.jnt_qposadr)
    np.testing.assert_array_equal(sim.model.actuator_trnid, original.actuator_trnid)
    np.testing.assert_array_equal(sim.model.key_qpos, original.key_qpos)
    expected = original_data.qpos.copy()
    for arm in sim.arms.values():
        expected[arm.qpos[3]] = np.pi / 2
    np.testing.assert_allclose(sim.data.qpos, expected, atol=1e-12)
    for joint in range(original.njnt):
        assert mujoco.mj_id2name(sim.model, mujoco.mjtObj.mjOBJ_JOINT, joint) == mujoco.mj_id2name(
            original, mujoco.mjtObj.mjOBJ_JOINT, joint
        )
    print("PASS HEI-style scene: unchanged model joints/keyframes, 90-degree elbow working home")


def check_pinocchio_ik(sim):
    """Compare independent FK implementations and exercise constrained solves."""
    rng = np.random.default_rng(42)
    for side, arm in sim.arms.items():
        sim.reset()
        ik = arm.ik
        for _ in range(20):
            sim.data.qpos[arm.qpos] = rng.uniform(ik.lower, ik.upper)
            sim.forward()
            ik.validate_fk(sim.data)

        # Also check the world transform with an upstream torso joint rotated.
        torso = sim.id("joint", "link_019_joint")
        sim.data.qpos[sim.model.jnt_qposadr[torso]] += 0.1
        sim.forward()
        ik.validate_fk(sim.data)
        sim.reset()
        sim.data.qpos[arm.qpos] = [0.1, -0.4 if side == "left" else 0.4, 0.2, 0.8, 0.1, 0.1, 0.1]
        sim.forward()
        current = sim.data.qpos[arm.qpos].copy()
        pose = ik.forward_world(current, sim.data)
        result = ik.solve(sim.data, pose.translation, pose.rotation, 0.01)
        assert result.accepted, result.status
        np.testing.assert_allclose(result.q, current, atol=1e-6)

        target = ik.forward_world(current + np.array([0.02, -0.03, 0.02, 0.03, -0.02, 0.01, 0.02]), sim.data)
        result = ik.solve(sim.data, target.translation, target.rotation, 0.1)
        assert result.accepted, result.status
        solved = ik.forward_world(result.q, sim.data)
        assert np.linalg.norm(solved.translation - target.translation) < 5e-4
        assert np.linalg.norm(solved.rotation - target.rotation) < 5e-3
        assert np.max(np.abs(result.q - current)) <= 0.1 + 1e-9

        # An unreachable goal must still respect the step and position bounds.
        sim.data.qpos[arm.qpos] = ik.upper - 0.001
        sim.forward()
        near_limit = sim.data.qpos[arm.qpos].copy()
        result = ik.solve(sim.data, np.array([100.0, -100.0, 100.0]), np.eye(3), 0.01)
        assert np.all(np.isfinite(result.q))
        assert np.all(result.q >= ik.lower) and np.all(result.q <= ik.upper)
        assert np.max(np.abs(result.q - near_limit)) <= 0.01 + 1e-9

        # A solver exception must hold the actual pose, not an intermediate iterate.
        original_solver = ik.solver

        def fail(**kwargs):
            raise RuntimeError("intentional solver failure")

        try:
            ik.solver = fail
            result = ik.solve(sim.data, target.translation, target.rotation, 0.01)
            assert not result.accepted
            np.testing.assert_array_equal(result.q, near_limit)
        finally:
            ik.solver = original_solver
        print(f"PASS {side}: Pinocchio/MuJoCo FK, zero-error IK, reachable target, bounds, failure hold")
    sim.reset()


def check_lift(sim):
    sim.reset()
    lift = sim.lift
    original = sim.data.qpos.copy()
    untouched = np.setdiff1d(np.arange(sim.model.nq), lift.qpos)
    step_seconds = 0.02

    def height():
        return float(sim.data.xpos[lift.height_body, 2])

    def tick(count):
        for _ in range(count):
            previous_height = height()
            sim.step(step_seconds)
            assert lift.closure_error(sim.data) < lift.CLOSURE_TOLERANCE_M
            np.testing.assert_array_equal(sim.data.qpos[untouched], original[untouched])
            assert abs(height() - previous_height) <= sim.args.lift_speed_m_s * step_seconds + 1e-5
            values = sim.data.qpos[lift.qpos]
            assert lift.hinge_limits[0] <= values[0] <= lift.hinge_limits[1]
            assert lift.drive_limits[0] <= values[1] <= lift.drive_limits[1]
            assert np.all(values[2:] >= lift.slide_limits[:, 0] - 1e-10)
            assert np.all(values[2:] <= lift.slide_limits[:, 1] + 1e-10)
            np.testing.assert_allclose(values[2], values[3], atol=1e-12)

    sim.press("2")
    sim.press("i")
    tick(5)  # Home is already the upper end of the mechanism.
    np.testing.assert_array_equal(sim.data.qpos, original)
    sim.release("i")
    sim.press("k")
    tick(30)
    sim.release("k")
    lowered_height = height()
    assert lift.max_height - lowered_height > 0.03
    held = sim.data.qpos.copy()
    tick(10)
    np.testing.assert_array_equal(sim.data.qpos, held)
    for arm in sim.arms.values():
        arm.ik.validate_fk(sim.data)
    sim.press("i")
    tick(10)
    assert height() > lowered_height + 0.01
    sim.release("i")

    # Opposite keys cancel, and stop survives a mode change while held.
    sim.press("i")
    sim.press("k")
    held = sim.data.qpos.copy()
    tick(5)
    np.testing.assert_array_equal(sim.data.qpos, held)
    sim.press("space")
    sim.press("3")
    sim.press("2")
    sim.press("k")
    tick(10)
    np.testing.assert_array_equal(sim.data.qpos, held)
    sim.stop()
    tick(10)  # The viewer also uses stop() on focus loss.
    np.testing.assert_array_equal(sim.data.qpos, held)

    distances = []
    for fine in (False, True):
        sim.reset()
        sim.press("2")
        if fine:
            sim.press("shift")
        sim.press("k")
        tick(20)
        distances.append(lift.max_height - height())
    np.testing.assert_allclose(distances[1] / distances[0], sim.args.fine_scale, atol=0.01)

    # Sweep the entire travel, then return and hold at the upper stop.
    sim.reset()
    sim.press("2")
    travel_steps = (
        int(np.ceil((lift.max_height - lift.min_height) / (sim.args.lift_speed_m_s * step_seconds))) + 10
    )
    sim.press("k")
    tick(travel_steps)
    np.testing.assert_allclose(height(), lift.min_height, atol=1e-6)
    sim.release("k")
    sim.press("i")
    tick(travel_steps)
    np.testing.assert_allclose(height(), lift.max_height, atol=1e-6)
    sim.stop()

    # Return to arm control after lowering: IK must use the moved shoulder frame.
    sim.press("2")
    sim.press("k")
    tick(40)
    lift_pose = sim.data.qpos[lift.qpos].copy()
    sim.press("3")
    arm = sim.arms["left"]
    before = sim.data.xpos[arm.tcp].copy()
    sim.press("w")
    for _ in range(15):
        sim.step(step_seconds)
    sim.stop()
    assert sim.data.xpos[arm.tcp, 0] < before[0] - 0.005
    np.testing.assert_array_equal(sim.data.qpos[lift.qpos], lift_pose)
    arm.ik.validate_fk(sim.data)
    sim.press("backspace")
    np.testing.assert_array_equal(sim.data.qpos, original)
    print(
        "PASS lift: I/K direction, full travel, closure, limits, release/stop/focus hold, fine control, arm IK, reset"
    )


def check_simulator(sim):
    check_scene(sim)
    check_pinocchio_ik(sim)
    initial = sim.data.qpos.copy()
    controlled = np.concatenate([np.concatenate((a.qpos, a.fingers)) for a in sim.arms.values()])
    untouched = np.setdiff1d(np.arange(sim.model.nq), controlled)

    def unchanged_mechanisms():
        np.testing.assert_array_equal(sim.data.qpos[untouched], initial[untouched])
        assert np.all(np.isfinite(sim.data.qpos))

    def tick(count=20):
        for _ in range(count):
            sim.step(0.01)
            unchanged_mechanisms()

    for side, mode_key, reset_key in (("left", "3", "5"), ("right", "4", "6")):
        sim.reset()
        arm = sim.arms[side]
        other = sim.arms["right" if side == "left" else "left"]
        # Exercise 6D IK from an asymmetric pose distinct from the working home.
        sim.data.qpos[arm.qpos] = [0.1, -0.4 if side == "left" else 0.4, 0.2, 0.8, 0.1, 0.1, 0.1]
        sim.forward()
        bent = sim.data.qpos.copy()
        base_rotation = sim.data.xmat[sim.base].reshape(3, 3).copy()
        if sim.args.tcp_reference == "gripper-motor":
            hand = sim.id("body", f"openarm_{side}_hand")
            local = sim.data.xmat[hand].reshape(3, 3).T @ (sim.data.xpos[arm.tcp] - sim.data.xpos[hand])
            np.testing.assert_allclose(local, sim.args.gripper_motor_offset_m, atol=1e-12)

        # The Jacobian must describe the relocated point, not the old fingertip TCP.
        jac = np.zeros((3, sim.model.nv))
        mujoco.mj_jacBody(sim.model, sim.data, jac, None, arm.tcp)
        before = sim.data.xpos[arm.tcp].copy()
        for address, dof in zip(arm.qpos, arm.dofs, strict=True):
            sim.data.qpos[:] = bent
            sim.data.qpos[address] += 1e-6
            sim.forward()
            derivative = (sim.data.xpos[arm.tcp] - before) / 1e-6
            np.testing.assert_allclose(derivative, jac[:, dof], atol=1e-6)
        sim.data.qpos[:] = bent
        sim.forward()
        for key, axis, sign in (
            ("w", 0, -1),
            ("s", 0, 1),
            ("a", 1, -1),
            ("d", 1, 1),
            ("r", 2, 1),
            ("f", 2, -1),
        ):
            sim.data.qpos[:] = bent
            sim.forward()
            sim.press(mode_key)
            before = sim.data.xpos[arm.tcp].copy()
            sim.press(key)
            tick(10)
            sim.release(key)
            displacement = base_rotation.T @ (sim.data.xpos[arm.tcp] - before)
            assert displacement[axis] * sign > 0.002, (side, key, displacement)
            np.testing.assert_array_equal(sim.data.qpos[other.qpos], bent[other.qpos])
            held = sim.data.qpos.copy()
            tick(5)
            np.testing.assert_array_equal(sim.data.qpos, held)

        from keyboard_sim import rotation_vector

        for key, axis, sign in (
            ("u", 0, 1),
            ("j", 0, -1),
            ("i", 1, 1),
            ("k", 1, -1),
            ("o", 2, 1),
            ("l", 2, -1),
        ):
            sim.data.qpos[:] = bent
            sim.forward()
            before = sim.data.xmat[arm.tcp].reshape(3, 3).copy()
            sim.press(key)
            tick(10)
            sim.release(key)
            rotation = base_rotation.T @ rotation_vector(sim.data.xmat[arm.tcp].reshape(3, 3) @ before.T)
            assert rotation[axis] * sign > 0.01, (side, key, rotation)

        distances = []
        for fine in (False, True):
            sim.data.qpos[:] = bent
            sim.forward()
            before = sim.data.xpos[arm.tcp].copy()
            if fine:
                sim.press("shift")
            sim.press("w")
            tick(10)
            sim.stop()
            distances.append(np.linalg.norm(sim.data.xpos[arm.tcp] - before))
        assert 0 < distances[1] < distances[0] * 0.5, distances

        sim.press("z")
        gripper_tcp = sim.data.xpos[arm.tcp].copy()
        gripper_rotation = sim.data.xmat[arm.tcp].copy()
        tick(150)
        sim.release("z")
        np.testing.assert_allclose(sim.data.qpos[arm.fingers], sim.model.jnt_range[arm.finger_joints, 1])
        sim.press("x")
        tick(150)
        sim.release("x")
        np.testing.assert_allclose(sim.data.qpos[arm.fingers], sim.model.jnt_range[arm.finger_joints, 0])
        np.testing.assert_allclose(sim.data.xpos[arm.tcp], gripper_tcp, atol=1e-12)
        np.testing.assert_allclose(sim.data.xmat[arm.tcp], gripper_rotation, atol=1e-12)

        sim.press(reset_key)
        tick(5)
        assert arm.resetting
        sim.press("space")
        held = sim.data.qpos.copy()
        sim.press("1")
        sim.press(mode_key)
        sim.press("w")
        tick(10)
        np.testing.assert_array_equal(sim.data.qpos, held)
        assert not arm.resetting
        sim.stop()
        sim.press(reset_key)
        tick(5)
        sim.press("w")
        tick(1)
        assert not arm.resetting
        sim.stop()
        sim.press(reset_key)
        sim.press("1")  # Switching mode must not interrupt the other arm's reset.
        tick(400)
        np.testing.assert_allclose(sim.data.qpos[arm.qpos], sim.home[arm.qpos], atol=1e-10)

        sim.press(mode_key)
        sim.press("tab")
        sim.release("tab")
        sim.selected_joint = 3
        sim.press("=")
        tick(400)
        np.testing.assert_allclose(sim.data.qpos[arm.qpos[3]], sim.model.jnt_range[arm.joints[3], 1])
        sim.release("=")
        sim.press("tab")
        sim.release("tab")
        sim.stop()
        print(f"PASS {side}: TCP Jacobian, 6D direction, fine control, gripper, stop, reset, joint limit")

    # Regression: arm selection must restore TCP after using joint jogging.
    for side, mode_key in (("left", "3"), ("right", "4")):
        sim.reset()
        sim.press("tab")
        sim.release("tab")
        assert sim.joint_mode
        sim.press(mode_key)
        sim.release(mode_key)
        assert not sim.joint_mode
        arm = sim.arms[side]
        start_tcp = sim.data.xpos[arm.tcp].copy()
        sim.press("w")
        tick(30)
        sim.release("w")
        assert np.linalg.norm(sim.data.xpos[arm.tcp] - start_tcp) > 0.015
    print("PASS selecting 3/4 restores TCP and W moves from the actual home pose")

    sim.reset()
    from keyboard_sim import draw_tcp_frames

    scene = mujoco.MjvScene(sim.model, maxgeom=8)
    draw_tcp_frames(sim, scene)
    assert scene.ngeom == 8
    for index, arm in enumerate(sim.arms.values()):
        np.testing.assert_allclose(scene.geoms[index * 4].pos, sim.data.xpos[arm.tcp], atol=1e-7)
    print("PASS TCP markers match the controlled reference points")
    sim.press("3")
    sim.press("w")
    sim.press("4")
    tick(10)
    np.testing.assert_array_equal(sim.data.qpos, initial)
    for mode in ("1", "2"):
        sim.press(mode)
        sim.press("w")
        sim.press("i")
        tick(10)
        np.testing.assert_array_equal(sim.data.qpos, initial)
    sim.press("backspace")
    np.testing.assert_array_equal(sim.data.qpos, initial)
    assert sim.mode == "left"
    mujoco.mj_forward(sim.model, sim.data)
    print(f"PASS model: nq={sim.model.nq}, nv={sim.model.nv}, actuators={sim.model.nu}")
    print("PASS arm mode isolation, release hold, full reset; no viewer opened")
    check_lift(sim)

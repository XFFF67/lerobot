"""Behavioral checks for real wheel/contact integration, without opening a window."""

import mujoco
import numpy as np
from chassis_physics import SLIDE_SIGNS, STEER_SIGNS


def check_simulator(sim):
    model, data, chassis = sim.model, sim.data, sim.chassis
    assert chassis is not None
    original = mujoco.MjModel.from_xml_path(str(sim.args.mjcf.expanduser().resolve()))
    # Free base and (with the table scene) four free objects; no extra guide.
    assert model.njnt == original.njnt + 1 + (4 if sim.scene_enabled else 0)
    assert model.neq == original.neq
    for number in (14, 15):
        name = f"link_{number:03d}_joint"
        np.testing.assert_array_equal(model.joint(name).range, original.joint(name).range)
    np.testing.assert_allclose(model.opt.gravity, [0, 0, -9.81])
    assert not np.any(model.jnt_limited[chassis.wheel_joints])
    steer_joints = [model.joint(f"link_{n:03d}_joint").id for n in (2, 5, 8, 11)]
    assert not set(steer_joints).intersection(model.actuator_trnid[:, 0])

    def tick(seconds):
        for _ in range(round(seconds / 0.02)):
            sim.step(0.02)
            assert not chassis.faulted
            assert np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel))
            q = chassis.slide_positions(data)
            assert np.all(q >= -0.0005), q
            assert np.all(q <= chassis.travel + 0.0005), q
            assert np.max(np.abs(data.ctrl[chassis.motors])) <= chassis.config.motor_torque
            # No hidden steering/base forces. Only wheel motors and slide magnet
            # forces can drive the chassis; upper-body support is separate.
            np.testing.assert_array_equal(data.qfrc_applied[:6], 0)
            np.testing.assert_array_equal(data.qfrc_applied[chassis.steer_dofs], 0)

    def select(mode):
        initial_pose = data.qpos.copy()
        initially_steered = np.min(np.abs(chassis.encoder_degrees(data))) > 20
        sim.press("g" if mode == "spin" else "t")
        sim.release("g" if mode == "spin" else "t")
        np.testing.assert_array_equal(data.qpos, initial_pose)
        calculate_targets = chassis._wheel_targets
        correcting = np.zeros(4, dtype=bool)
        alignment_pair_errors = []

        def record_targets(*args):
            result = calculate_targets(*args)
            if chassis.phase == "align":
                correcting[:] |= np.abs(result[0]) > 1e-6
                if not alignment_pair_errors:
                    paired = chassis.encoder_degrees(data).reshape(2, 2)
                    alignment_pair_errors.append(float(np.max(np.abs(paired[:, 0] - paired[:, 1]))))
            return result

        # Observe the real control calls without advancing or modifying them.
        chassis._wheel_targets = record_targets
        try:
            for _ in range(700):
                tick(0.02)
                if mode == "spin":
                    assert chassis.phase != "straighten", "G must not request a preliminary recenter"
                    if initially_steered:
                        assert np.max(np.abs(chassis.encoder_degrees(data))) > 2, "No all-zero waypoint"
                if chassis.phase == "ready":
                    break
        finally:
            chassis._wheel_targets = calculate_targets
        assert chassis.phase == "ready", chassis.last_status
        assert chassis.target_mode == mode
        if mode == "spin":
            assert np.all(correcting), "All four wheel drives must participate in final correction"
            assert alignment_pair_errors[0] < 3, alignment_pair_errors
        expected = 53 * STEER_SIGNS if mode == "spin" else np.zeros(4)
        np.testing.assert_allclose(chassis.encoder_degrees(data), expected, atol=1.5)
        expected_slide = chassis.travel if mode == "spin" else np.zeros(2)
        np.testing.assert_allclose(chassis.slide_positions(data), expected_slide, atol=0.001)
        print(f"PASS {mode} capture: encoders {chassis.encoder_degrees(data).round(2)} deg")

    def heading():
        rotation = data.xmat[chassis.base].reshape(3, 3)
        return float(np.arctan2(rotation[1, 0], rotation[0, 0]))

    sim.reset()
    tick(0.8)
    assert data.ncon > 0
    sim.press("1")
    # Drive both directions from the front endpoint.
    for key, sign in (("w", -1), ("s", 1)):
        initial = data.xpos[chassis.base].copy()
        rotation = data.xmat[chassis.base].reshape(3, 3).copy()
        sim.press(key)
        tick(2)
        sim.release(key)
        tick(1.2)
        displacement = rotation.T @ (data.xpos[chassis.base] - initial)
        assert sign * displacement[0] > 0.12, displacement
        assert abs(displacement[1]) < 0.02, displacement
        assert chassis.phase == "ready"
    print("PASS forward/reverse translation and release braking")

    # A/D changes only the retained wheel angle. W/S then rolls along it.
    for key, sign in (("a", -1), ("d", 1)):
        sim.reset()
        tick(0.8)
        sim.press("1")
        initial = data.xpos[chassis.base].copy()
        start_heading = heading()
        peak_yaw = 0.0
        sim.press(key)
        for _ in range(400):
            tick(0.02)
            delta = np.arctan2(np.sin(heading() - start_heading), np.cos(heading() - start_heading))
            peak_yaw = max(peak_yaw, abs(delta))
            assert chassis.phase == "ready" and chassis.target_mode == "translation"
            assert np.max(np.abs(chassis.slide_positions(data))) < 0.001
            assert abs(chassis.parallel_angle) <= np.pi / 2
        sim.release(key)
        tick(1.5)
        # Passive steering may shift the base slightly as the tire traces an arc.
        assert np.linalg.norm((data.xpos[chassis.base] - initial)[:2]) < 0.065, (
            data.xpos[chassis.base] - initial,
            chassis.encoder_degrees(data),
        )
        np.testing.assert_allclose(chassis.encoder_degrees(data), sign * 90, atol=3)
        selected_angle = chassis.parallel_angle
        for drive, direction_sign in (("w", -sign), ("s", sign)):
            initial = data.xpos[chassis.base].copy()
            rotation = data.xmat[chassis.base].reshape(3, 3).copy()
            sim.press(drive)
            for _ in range(150):
                tick(0.02)
                assert chassis.parallel_angle == selected_angle  # W/S must never recenter.
                delta = np.arctan2(np.sin(heading() - start_heading), np.cos(heading() - start_heading))
                peak_yaw = max(peak_yaw, abs(delta))
            sim.release(drive)
            tick(1.2)
            displacement = rotation.T @ (data.xpos[chassis.base] - initial)
            assert direction_sign * displacement[1] > 0.2, displacement
            assert abs(displacement[0]) < 0.03, displacement
            np.testing.assert_allclose(chassis.encoder_degrees(data), sign * 90, atol=3)
            assert chassis.phase == "ready", chassis.last_status
            assert np.linalg.norm(data.qvel[:2]) < 0.01
        assert peak_yaw < np.deg2rad(3), np.rad2deg(peak_yaw)
        print(
            f"PASS {key.upper()} steering limit, release holds angle, W/S rolls sideways; "
            f"peak heading drift {np.rad2deg(peak_yaw):.2f} deg"
        )

    # Switch directly from the side-facing group, without a straight waypoint.
    select("spin")
    select("translation")
    print("PASS sideways wheel configuration -> spin -> front translation")

    # Use real steering/contact dynamics to reach starting angles across the
    # allowed range, including a G press while A/D is still steering the wheels.
    for angle, settle in ((-90, 4), (-60, 4), (-30, 4), (0, 4), (30, 4), (60, 4), (90, 4), (30, 0)):
        sim.reset()
        tick(0.8)
        sim.press("1")
        if angle:
            key = "d" if angle > 0 else "a"
            sim.press(key)
            tick(abs(angle) / 30)
            if settle:
                sim.release(key)
                tick(settle)
                np.testing.assert_allclose(chassis.encoder_degrees(data), angle, atol=1.5)
        start = chassis.encoder_degrees(data).round(1)
        select("spin")
        assert not {"w", "s", "a", "d"}.intersection(sim.keys)
        print(f"PASS direct spin switch from {start} deg; elapsed {chassis.switch_elapsed:.2f} s")

    sim.reset()
    tick(0.8)
    sim.press("1")
    initial = data.qpos.copy()
    sim.press("a")
    sim.press("d")
    tick(0.5)
    np.testing.assert_allclose(data.qpos[:3], initial[:3], atol=0.001)
    np.testing.assert_allclose(data.qpos[chassis.steer_qpos], initial[chassis.steer_qpos], atol=0.01)
    sim.release("d")
    tick(0.5)
    # Focus loss during a partial swivel must not finish the 90° turn later.
    sim.stop()
    tick(1)
    held_angles = data.qpos[chassis.steer_qpos].copy()
    tick(1)
    np.testing.assert_allclose(data.qpos[chassis.steer_qpos], held_angles, atol=np.deg2rad(1))
    assert np.max(np.abs(chassis.encoder_degrees(data))) < 45
    print("PASS opposite steering keys cancel; focus loss stops an unfinished parallel swivel")

    sim.reset()
    tick(0.8)
    sim.press("1")
    select("spin")
    tick(1)
    for key, sign in (("a", 1), ("d", -1)):
        start_heading = heading()
        initial = data.xpos[chassis.base].copy()
        sim.press(key)
        tick(3)
        sim.release(key)
        tick(1)
        delta = np.arctan2(np.sin(heading() - start_heading), np.cos(heading() - start_heading))
        assert sign * delta > 0.15, delta
        assert np.linalg.norm((data.xpos[chassis.base] - initial)[:2]) < 0.025
        assert chassis.phase == "ready", chassis.last_status
    print("PASS clockwise/counterclockwise spin with magnetic endpoints retained")
    select("translation")

    # Upper-body control still follows a moving/rotated chassis frame.
    sim.press("2")
    initial_height = sim.lift.height(data)
    sim.press("k")
    tick(0.8)
    sim.release("k")
    tick(0.5)
    assert sim.lift.height(data) < initial_height - 0.03
    assert sim.lift.closure_error(data) < 0.001
    for arm in sim.arms.values():
        arm.ik.validate_fk(data)
    sim.press("3")
    sim.press("tab")
    sim.release("tab")
    sim.selected_joint = 3
    arm = sim.arms["left"]
    initial_joint = data.qpos[arm.qpos[3]]
    sim.press("=")
    tick(0.2)
    sim.release("=")
    tick(0.4)
    assert data.qpos[arm.qpos[3]] > initial_joint + 0.05
    sim.press("tab")
    sim.release("tab")
    initial_tcp = data.xpos[arm.tcp].copy()
    sim.press("r")
    tick(0.2)
    sim.release("r")
    tick(0.4)
    assert np.linalg.norm(data.xpos[arm.tcp] - initial_tcp) > 0.002
    print("PASS lift, arm servo and Pinocchio/CasADi IK after base rotation")

    # Cancel automated switching through both Space and focus-loss handler.
    for focus_loss in (False, True):
        sim.reset()
        tick(0.5)
        sim.press("1")
        sim.press("g")
        tick(0.2)
        if focus_loss:
            sim.stop()
        else:
            sim.press("space")
            sim.press("3")
            sim.press("1")
            sim.press("g")  # Held Space must suppress a new request.
        tick(0.4)
        assert chassis.phase == "stopped"
        assert chassis.cancelled
    print("PASS Space, held-stop across modes, and window-focus cancellation")

    # Magnets act toward both ends regardless of requested mode.
    sim.reset()
    scratch = mujoco.MjData(model)
    for fraction, sign in ((0.02, -1), (0.98, 1)):
        scratch.qpos[chassis.slide_qpos] = chassis.travel * fraction * SLIDE_SIGNS
        force = chassis.magnet_forces(scratch)
        assert np.all(force * SLIDE_SIGNS * sign > 0)
        chassis.target_mode = "spin"
        np.testing.assert_array_equal(chassis.magnet_forces(scratch), force)
    print("PASS passive two-end magnet forces independent of mode request")

    # With contact disabled, rotating wheel motors must NOT perform a scripted
    # base/steering motion. Zero gravity isolates this test from falling.
    sim.reset()
    contact_mask = model.geom_contype.copy(), model.geom_conaffinity.copy()
    gravity = model.opt.gravity.copy()
    try:
        model.geom_contype[:] = model.geom_conaffinity[:] = 0
        model.opt.gravity[:] = 0
        sim.press("1")
        sim.press("g")
        tick(0.5)
        assert data.ncon == 0
        assert chassis.phase != "ready"
        assert np.max(np.abs(chassis.encoder_degrees(data))) < 2
        assert np.max(np.abs(data.qvel[chassis.wheel_dofs])) > 0.1
        # Exercise timeout without spending another 15 seconds integrating.
        chassis.switch_elapsed = 15.01
        tick(0.02)
        assert chassis.phase == "stopped" and "timed out" in chassis.last_status
    finally:
        model.geom_contype[:], model.geom_conaffinity[:] = contact_mask
        model.opt.gravity[:] = gravity
        sim.reset()
    print("PASS loss of ground traction prevents switching; timeout and reset")
    print("PASS chassis physics checks")

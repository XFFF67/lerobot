"""No-hardware checks for the original chassis frame with appended lift bytes."""

import os
import pty
import select
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples/mobile_openarm"
sys.path.insert(0, str(EXAMPLES))
from mobile_serial_protocol import ChassisFrame, Commands, LiftFrame  # noqa: E402
from vr_input import Controller  # noqa: E402
from vr_mobile_bridge import MobileMapper  # noqa: E402


def test_original_chassis_bytes_are_unchanged_and_lift_is_appended():
    translation = Commands(ChassisFrame(30, -50, False), LiftFrame(True, -25)).encode()
    assert translation[:7] == bytes((0x55, 0xFF, 0xE2, 0x20, 0, 50, 0))
    assert translation[7:] == bytes((1, 0xE7))
    spin = Commands(ChassisFrame(0, 5, True), LiftFrame()).encode()
    assert spin[:7] == bytes((0x55, 0, 5, 0, 0, 0, 1))
    assert len(spin) == 9


def test_frame_bounds():
    with pytest.raises(ValueError):
        ChassisFrame(91, 0).encode()
    with pytest.raises(ValueError):
        ChassisFrame(0, 11, True).encode()
    with pytest.raises(ValueError):
        LiftFrame(False, 1).encode()


def controllers():
    return {side: Controller(valid=True) for side in ("left", "right")}


def test_exact_vr_mobile_mapping_and_arm_reset_independence():
    mapper, state = MobileMapper(), controllers()
    state["right"].grip = True
    state["right"].stick[1] = 1
    command = mapper.step(state, 0.02)
    assert command.chassis.speed_milli == -50
    state["right"].primary = True
    assert mapper.step(state, 0.02).chassis.speed_milli == -50
    state["right"].stick[0] = 1
    assert mapper.step(state, 0.02).chassis.angle_deg == 0
    state["left"].stick[0] = -1
    mapper.step(state, 0.05)
    turn = mapper.step(state, 0.05)
    assert turn.chassis.angle_deg > 0
    state["left"].grip = True
    state["left"].stick[1] = -1
    lift = mapper.step(state, 0.02)
    assert lift.chassis.angle_deg == turn.chassis.angle_deg
    assert lift.lift.enabled and lift.lift.speed_mrad_s > 0
    state["right"].stick[:] = 0
    state["right"].stick_pressed = True
    enter_spin = mapper.step(state, 0.02)
    assert enter_spin.chassis.spin and enter_spin.chassis.speed_milli == 0
    # Holding the click does not repeatedly toggle; releasing it retains the
    # self-rotation configuration at zero speed.
    assert mapper.step(state, 0.02).chassis.spin
    state["right"].stick_pressed = False
    assert mapper.step(state, 0.02).chassis.spin
    state["right"].stick[0] = 1
    spin = mapper.step(state, 0.02)
    assert spin.chassis.spin and spin.chassis.speed_milli == -5
    state["right"].secondary = True
    assert mapper.step(state, 0.02).chassis.speed_milli == -5  # B is not a chassis command.
    state["right"].secondary = False
    state["right"].stick[:] = 0
    state["right"].stick_pressed = True
    leave_spin = mapper.step(state, 0.02)
    assert not leave_spin.chassis.spin and leave_spin.chassis.angle_deg == turn.chassis.angle_deg
    state["right"].grip = False
    stopped = mapper.step(state, 0.02).chassis
    assert stopped.speed_milli == 0 and stopped.angle_deg == turn.chassis.angle_deg


def test_steering_reaches_full_range_and_release_preserves_configuration():
    mapper, state = MobileMapper(), controllers()
    state["right"].grip = True
    state["left"].stick[0] = -1

    for _ in range(200):
        command = mapper.step(state, 0.05)
    assert command.chassis == ChassisFrame(90, 0, False)

    state["right"].grip = False
    assert mapper.step(state, 0.02).chassis == ChassisFrame(90, 0, False)
    assert mapper.step(state, 0.2).chassis == ChassisFrame(90, 0, False)

    state["right"].grip = True
    state["left"].stick[0] = 1
    for _ in range(400):
        command = mapper.step(state, 0.05)
    assert command.chassis == ChassisFrame(-90, 0, False)

    state["left"].stick[0] = 0
    state["right"].stick_pressed = True
    assert mapper.step(state, 0.02).chassis == ChassisFrame(0, 0, True)
    state["right"].grip = False
    assert mapper.step(state, 0.02).chassis == ChassisFrame(0, 0, True)

    state["right"].grip = True
    state["right"].stick_pressed = False
    mapper.step(state, 0.02)
    state["right"].stick_pressed = True
    assert mapper.step(state, 0.02).chassis == ChassisFrame(-90, 0, False)


def test_mobile_hands_fail_independently():
    mapper, state = MobileMapper(), controllers()

    state["left"].grip = True
    state["left"].stick[1] = -1
    state["right"].valid = False
    lift = mapper.step(state, 0.02)
    assert lift.lift.enabled and lift.lift.speed_mrad_s == 50
    assert lift.chassis == ChassisFrame()

    state["left"].valid = False
    state["right"].valid = True
    state["right"].grip = True
    state["right"].stick[1] = 1
    drive = mapper.step(state, 0.02)
    assert drive.chassis.speed_milli == -50
    assert drive.lift == LiftFrame()


def test_pty_execute_sends_nine_byte_neutral_frames():
    master, slave = pty.openpty()
    proc = subprocess.Popen(
        [sys.executable, str(EXAMPLES / "vr_mobile_bridge.py"), "--port", os.ttyname(slave),
         "--execute", "--run-seconds", "0.2"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    raw = bytearray()
    try:
        deadline = time.monotonic() + 5
        while proc.poll() is None and time.monotonic() < deadline:
            if select.select([master], [], [], 0.05)[0]:
                raw.extend(os.read(master, 4096))
        out, err = proc.communicate(timeout=2)
        assert proc.returncode == 0, out + err
        expected = Commands().encode()
        assert len(raw) >= 9 and all(raw[i:i + 9] == expected for i in range(0, len(raw) - 8, 9))
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        os.close(master)
        os.close(slave)


def test_native_lift_control(tmp_path):
    package = EXAMPLES / "stm32_serial"
    sources = package / "firmware/bsp/boards"
    binary = tmp_path / "native-control"
    subprocess.run(
        ["gcc", "-std=c99", "-Wall", "-Wextra", "-Werror", f"-I{package / 'tests'}", f"-I{sources}",
         str(package / "tests/native_control.c"), str(sources / "LkTest.c"), "-o", str(binary)],
        check=True,
    )
    subprocess.run([str(binary)], check=True)

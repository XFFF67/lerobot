#!/usr/bin/env python3
"""Validate the merged OpenArm control + IK runtime without touching hardware."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import os
import subprocess
import sys
from pathlib import Path

from packaging.utils import canonicalize_name


EXPECTED = {
    "numpy": "2.2.6",
    "scipy": "1.18.0",
    "opencv-python-headless": "4.13.0.92",
    "Pillow": "12.2.0",
    "einops": "0.8.2",
    "draccus": "0.10.0",
    "huggingface-hub": "1.21.0",
    "requests": "2.34.2",
    "gymnasium": "1.3.0",
    "safetensors": "0.8.0",
    "packaging": "25.0",
    "termcolor": "3.3.0",
    "tqdm": "4.68.3",
    "cmake": "4.1.3",
    "setuptools": "80.10.2",
    "python-can": "4.6.1",
    "pyserial": "3.5",
    "casadi": "3.7.0",
    "mujoco": "3.5.0",
    "glfw": "2.11.0",
    "pyzmq": "27.3.0",
    "websockets": "16.1.1",
    "typeguard": "4.6.0",
}


def fail(message: str) -> None:
    raise RuntimeError(message)


def distribution_versions(name: str) -> list[str]:
    expected_name = canonicalize_name(name)
    return [
        dist.version
        for dist in importlib.metadata.distributions()
        if canonicalize_name(dist.metadata.get("Name") or "") == expected_name
    ]


def check_versions() -> None:
    for name, expected in EXPECTED.items():
        versions = distribution_versions(name)
        if versions != [expected]:
            fail(f"{name}: expected one {expected} distribution, found {versions or 'none'}")


def check_imports() -> None:
    for module in (
        "numpy",
        "scipy",
        "torch",
        "torchvision",
        "mujoco",
        "glfw",
        "casadi",
        "pinocchio",
        "can",
        "serial",
        "zmq",
    ):
        importlib.import_module(module)
    import torch
    import torchvision
    import pinocchio

    if torch.__version__ != "2.11.0+cpu":
        fail(f"torch: expected 2.11.0+cpu, found {torch.__version__}")
    if torchvision.__version__ != "0.26.0+cpu":
        fail(f"torchvision: expected 0.26.0+cpu, found {torchvision.__version__}")
    if pinocchio.__version__ != "3.9.0":
        fail(f"pinocchio: expected 3.9.0, found {pinocchio.__version__}")
    from pinocchio import casadi as _  # noqa: F401
    from lerobot.motors.damiao.damiao import DamiaoMotorsBus  # noqa: F401


def check_model(repo_root: Path) -> None:
    import mujoco

    model_path = repo_root / "examples/mobile_openarm/model/mjcf1/robot.xml"
    model = mujoco.MjModel.from_xml_path(str(model_path))
    if model.nq <= 0 or model.nv <= 0:
        fail("MuJoCo model has no degrees of freedom")


def run_full_check(repo_root: Path) -> None:
    command = [sys.executable, "-u", str(repo_root / "examples/mobile_openarm/check_vr_real.py")]
    env = dict(
        os.environ,
        PYTHONPATH=str(repo_root / "src"),
        LD_LIBRARY_PATH=str(Path(sys.prefix) / "lib"),
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
    )
    subprocess.run(command, cwd=repo_root, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="Also run the no-hardware IK/IPC integration check")
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    if sys.version_info[:2] != (3, 12):
        fail(f"expected Python 3.12.x, found {sys.version.split()[0]}")
    check_versions()
    check_imports()
    check_model(repo_root)
    if args.full:
        run_full_check(repo_root)
    print(f"PASS openarm-runtime: {sys.executable}")


if __name__ == "__main__":
    main()

"""HEI-style scene adapted to the mobile OpenArm coordinate frame.

Reference layout and asset provenance: model/scene_assets/README.md.
Environment contacts and free objects are enabled with chassis dynamics.
"""

from pathlib import Path

import mujoco
import numpy as np

ASSET_DIR = Path(__file__).resolve().parent / "model" / "scene_assets"
# The original robot frame is 264 mm above the wheel contact plane.
GROUND_Z = -0.264
TABLE_SCALE = 0.75
TABLE_HALF_SIZE = tuple(size * TABLE_SCALE for size in (0.48, 0.62, 0.035))
# Keep the near edge at X=-0.44 m so shrinking does not move it out of reach.
TABLE_CENTER = (-0.44 - TABLE_HALF_SIZE[0], 0.0)
TABLE_HEIGHT = 0.70
TABLE_TOP_Z = GROUND_Z + TABLE_HEIGHT
CUBE_HALF_SIZE = 0.03
# The robot is on the +X side of the table.
OBJECT_ROBOT_OFFSET_X = 0.05
OBJECT_NAMES = ("scene_cube_red", "scene_cube_green", "scene_cube_blue", "scene_banana")


def _geom(body, name, kind, *, pos=None, size=None, rgba=None, **kwargs):
    return body.add_geom(
        name=name,
        type=kind,
        **({"pos": pos} if pos is not None else {}),
        **({"size": size} if size is not None else {}),
        **({"rgba": rgba} if rgba is not None else {}),
        contype=0,
        conaffinity=0,
        group=2,
        **kwargs,
    )


def _environment(spec):
    spec.add_texture(
        name="scene_sky",
        type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
        rgb1=[0.32, 0.43, 0.55],
        rgb2=[0.72, 0.78, 0.84],
        width=512,
        height=512,
    )
    spec.add_texture(
        name="scene_grid",
        type=mujoco.mjtTexture.mjTEXTURE_2D,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
        mark=mujoco.mjtMark.mjMARK_EDGE,
        rgb1=[0.59, 0.62, 0.66],
        rgb2=[0.69, 0.72, 0.76],
        markrgb=[0.30, 0.32, 0.35],
        width=512,
        height=512,
    )
    material = spec.add_material(
        name="scene_floor_material",
        texrepeat=[12, 12],
        texuniform=True,
        reflectance=0.05,
        specular=0.15,
        shininess=0.2,
    )
    material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "scene_grid"
    _geom(
        spec.worldbody,
        "scene_ground",
        mujoco.mjtGeom.mjGEOM_PLANE,
        pos=[0, 0, GROUND_Z - 0.002],
        size=[0, 0, 0.05],
        material=material.name,
    )
    for name, pos, direction, diffuse, specular, shadow in (
        ("key", [0, 0.6, GROUND_Z + 1.5], [0, 0, -1], [0.52] * 3, [0.2] * 3, True),
        (
            "fill",
            [-2.5, 2.5, GROUND_Z + 2.4],
            [0.6, -0.5, -0.65],
            [0.08, 0.09, 0.12],
            [0.02, 0.02, 0.03],
            False,
        ),
    ):
        spec.worldbody.add_light(
            name=f"scene_{name}_light",
            pos=pos,
            dir=direction,
            type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
            castshadow=shadow,
            diffuse=diffuse,
            ambient=[0, 0, 0],
            specular=specular,
        )

    for name, half, width, z, color in (
        ("safety", 2.0, 0.015, GROUND_Z + 0.004, [0.95, 0.62, 0.08, 0.90]),
        ("start", 0.35, 0.008, GROUND_Z + 0.005, [0.05, 0.72, 0.70, 0.95]),
    ):
        for axis in (0, 1):
            for sign in (-1, 1):
                pos = [0, 0, z]
                pos[axis] = sign * half
                size = [half, half, 0.003]
                size[axis] = width
                _geom(
                    spec.worldbody,
                    f"scene_{name}_{axis}_{sign}",
                    mujoco.mjtGeom.mjGEOM_BOX,
                    pos=pos,
                    size=size,
                    rgba=color,
                )

    # World-aligned XYZ markers at the ground projection of the base origin.
    origin = np.array([0.0, 0.0, GROUND_Z + 0.025])
    for axis, name, color in zip(
        range(3), "xyz", ([0.9, 0.12, 0.1, 1], [0.1, 0.72, 0.18, 1], [0.12, 0.3, 0.95, 1]), strict=True
    ):
        tip = origin.copy()
        tip[axis] += 0.65 if axis < 2 else 0.625
        _geom(
            spec.worldbody,
            f"scene_axis_{name}",
            mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[0.012],
            fromto=[*origin, *tip],
            rgba=color,
        )
        _geom(
            spec.worldbody,
            f"scene_axis_{name}_tip",
            mujoco.mjtGeom.mjGEOM_SPHERE,
            pos=tip,
            size=[0.028],
            rgba=color,
        )


def _table_and_objects(spec):
    table = spec.worldbody.add_body(name="scene_table", pos=[*TABLE_CENTER, GROUND_Z])
    top_center_z = TABLE_HEIGHT - TABLE_HALF_SIZE[2]
    leg_half_height = (TABLE_HEIGHT - 2 * TABLE_HALF_SIZE[2]) / 2
    _geom(
        table,
        "scene_table_top",
        mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0, 0, top_center_z],
        size=TABLE_HALF_SIZE,
        rgba=[0.34, 0.25, 0.18, 1],
    )
    for x in (-0.40 * TABLE_SCALE, 0.40 * TABLE_SCALE):
        for y in (-0.54 * TABLE_SCALE, 0.54 * TABLE_SCALE):
            _geom(
                table,
                f"scene_table_leg_{x}_{y}",
                mujoco.mjtGeom.mjGEOM_BOX,
                pos=[x, y, leg_half_height],
                size=[0.035 * TABLE_SCALE, 0.035 * TABLE_SCALE, leg_half_height],
                rgba=[0.16, 0.17, 0.18, 1],
            )
    for name, y, color in zip(
        OBJECT_NAMES[:3],
        (0.28, 0.0, -0.28),
        ([0.82, 0.16, 0.12, 1], [0.16, 0.66, 0.25, 1], [0.12, 0.35, 0.86, 1]),
        strict=True,
    ):
        body = spec.worldbody.add_body(
            name=name,
            pos=[
                TABLE_CENTER[0] + 0.22 * TABLE_SCALE + OBJECT_ROBOT_OFFSET_X,
                y * TABLE_SCALE,
                TABLE_TOP_Z + CUBE_HALF_SIZE,
            ],
        )
        _geom(body, f"{name}_geom", mujoco.mjtGeom.mjGEOM_BOX, size=[CUBE_HALF_SIZE] * 3, rgba=color)

    banana_dir = ASSET_DIR / "ycb_011_banana"
    mesh_name, texture_name = "scene_banana.obj", "scene_banana.png"
    # In-memory assets avoid changing the robot's mesh/texture path settings.
    mesh_bytes = b"\n".join(
        line
        for line in (banana_dir / "textured.obj").read_bytes().splitlines()
        if not line.startswith(b"mtllib ")
    )
    spec.assets[str(Path(spec.compiler.meshdir) / mesh_name)] = mesh_bytes
    spec.assets[str(Path(spec.compiler.texturedir) / texture_name)] = (
        banana_dir / "texture_map.png"
    ).read_bytes()
    spec.add_mesh(name="scene_banana_mesh", file=mesh_name, smoothnormal=True)
    spec.add_texture(name="scene_banana_texture", type=mujoco.mjtTexture.mjTEXTURE_2D, file=texture_name)
    material = spec.add_material(name="scene_banana_material", specular=0.08, shininess=0.1)
    material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "scene_banana_texture"
    # This mesh's origin is almost at its bottom; seat it on the resized table.
    banana_min_z = min(float(line.split()[3]) for line in mesh_bytes.splitlines() if line.startswith(b"v "))
    # HEI placement rotated 180 degrees about Z to face the OpenArm's -X side.
    body = spec.worldbody.add_body(
        name="scene_banana",
        pos=[TABLE_CENTER[0] - 0.11 * TABLE_SCALE + OBJECT_ROBOT_OFFSET_X, 0, TABLE_TOP_Z - banana_min_z],
        quat=[-0.1736, 0, 0, 0.9848],
    )
    _geom(
        body,
        "scene_banana_geom",
        mujoco.mjtGeom.mjGEOM_MESH,
        meshname="scene_banana_mesh",
        material=material.name,
    )


def build_mujoco_model(path, *, add_environment=True, chassis_config=None):
    path = Path(path).expanduser().resolve()
    if not add_environment and chassis_config is None:
        return mujoco.MjModel.from_xml_path(str(path))
    spec = mujoco.MjSpec.from_file(str(path))
    if add_environment:
        _environment(spec)
        _table_and_objects(spec)
    if chassis_config is not None:
        from chassis_physics import add_chassis_physics

        add_chassis_physics(spec, chassis_config)
        if add_environment:
            from grasp_physics import add_grasp_physics

            add_grasp_physics(spec)
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 960
    return spec.compile()


def configure_camera(model, camera, *, scene_enabled):
    mujoco.mjv_defaultFreeCamera(model, camera)
    if scene_enabled:
        camera.lookat[:] = [-0.40, 0, GROUND_Z + 0.65]
        camera.distance = 3.6
        camera.azimuth, camera.elevation = 135, -25
    else:
        camera.distance = max(1.5, float(model.stat.extent) * 1.5)
        camera.azimuth, camera.elevation = 135, -20

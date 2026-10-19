"""Tabletop contacts and free objects; grasping uses finger friction only.

Contact bits: fixed environment=1, wheels=2, objects=4, arm collision meshes=8.
Robot self-contact remains disabled; meshes for cameras/decorations stay visual.
"""

from mujoco_scene import OBJECT_NAMES


def add_grasp_physics(spec):
    """Call after chassis setup, which clears the source CAD contact masks."""
    for geom in spec.geoms:
        if geom.name == "scene_ground" or geom.name.startswith("scene_table_"):
            geom.contype, geom.conaffinity = 1, 2 | 4 | 8
            # Preserve the configured tire friction on the ground.
            if geom.name != "scene_ground":
                geom.friction = [0.8, 0.005, 0.0001]
        elif geom.name.startswith("openarm_") and geom.name.endswith("_collision"):
            geom.contype, geom.conaffinity = 8, 1 | 4
            geom.condim = 4
            geom.friction = [1.2, 0.005, 0.0001]
            geom.solref = [0.005, 1]
            geom.solimp = [0.95, 0.99, 0.001, 0.5, 2]
            if "_finger_collision" in geom.name:
                # Reduce soft-contact creep under a sustained squeeze without
                # changing the global solver (and the wheel/ground dynamics).
                geom.solimp = [0.999, 0.9999, 0.001, 0.5, 2]

    for name in OBJECT_NAMES:
        body = spec.body(name)
        body.add_freejoint(name=f"{name}_free")
        geom = spec.geom(f"{name}_geom")
        geom.contype, geom.conaffinity = 4, 1 | 4 | 8
        geom.mass = 0.12 if name == "scene_banana" else 0.10
        geom.condim = 4
        geom.friction = [1.0, 0.005, 0.0001]
        geom.solref = [0.005, 1]
        geom.solimp = [0.95, 0.99, 0.001, 0.5, 2]
        # Banana mesh contact uses MuJoCo's convex hull of the visible mesh.

    actuators = {actuator.name: actuator for actuator in spec.actuators}
    for side in ("left", "right"):
        for number in (1, 2):
            actuator = actuators[f"physics_hold_openarm_{side}_finger_joint{number}"]
            # Finite squeeze force; releasing X retains the closing target.
            actuator.forcelimited = True
            actuator.forcerange = [-15, 15]

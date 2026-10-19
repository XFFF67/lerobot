# HEI-style scene assets

The scene in `../../mujoco_scene.py` follows the colors, dimensions and layout of
[HEI ReBot Lift's scene](https://github.com/lipengdong/hei-rebot-lift/blob/main/software/lerobot-hei-rebot-lift/examples/hei_rebot_lift/VR_mujoco_ik/mujoco_ik/hei_robot_mujoco_scene.py),
retrieved on 2026-10-15. The implementation is adapted to this robot's MJCF.

- Gray checker floor, gradient sky, key and fill lights.
- Orange 4 × 4 m perimeter and cyan 0.7 × 0.7 m start outline.
- World-aligned red/green/blue XYZ markers at the ground projection of the base origin.
- A 0.72 × 0.93 m table, 0.70 m above the floor, three 6 cm cubes, and a textured banana.
  The tabletop dimensions and leg cross sections are scaled to 75% of HEI's originals;
  object spacing follows the scale. Cubes have 6 cm edges; the banana retains its original size.
  All four objects are shifted 5 cm along world +X toward the robot from the scaled layout.
- The environment floor is at world Z = −0.264 m to match the original robot's wheels.
  The original robot model and base frame are unchanged.
- The table and objects are rotated 180° around world Z relative to HEI's layout,
  placing the table at X = −0.80 m, along the current keyboard W direction.
  The near edge stays at X = −0.44 m after resizing.
- The default kinematic preview uses visual fixtures. With --chassis-physics, grasp_physics.py enables table/arm contacts and free objects; the banana uses convex-hull mesh contact. Grasping relies on friction, with no object attachment.

The banana mesh and texture are copied from HEI's local YCB assets. They are
unchanged on disk; the loader ignores the OBJ material-library directive in memory
and assigns a MuJoCo material. See [YCB source and CC BY 4.0 attribution](ycb_011_banana/SOURCE.md).
All other scene geometry and textures are generated locally. No downloads occur at startup.

# Touch controller assets

These are the controller models selected by `oculus-touch-controls` in the
A-Frame bundle used by HEI ReBot Lift's `telegrip/web-ui/index.html`.
They are served locally so headset rendering does not depend on CDN access.

| Local files | A-Frame source |
| --- | --- |
| `touch-v3-{left,right}.glb` | `https://cdn.aframe.io/controllers/oculus/oculus-touch-controller-v3-{left,right}.glb` |
| `touch-plus-{left,right}.glb` | `https://cdn.aframe.io/controllers/meta/quest-touch-plus-{left,right}.glb` |
| `touch-pro-{left,right}.glb` | `https://cdn.aframe.io/controllers/meta/quest-touch-pro-{left,right}.glb` |

Retrieved 2026-10-20. Original GLB bytes and embedded base-color textures are
preserved. Models originate from the
[WebXR Input Profiles assets](https://github.com/immersive-web/webxr-input-profiles/tree/main/packages/assets);
its MIT license is included in `LICENSE.md`.

The local loader supports the static meshes in these six files. It applies the
complete node hierarchy and scale, displays base-color textures with simple
directional shading, and retains the existing grip indicator and axis arrows.
Pressed buttons (including trigger and squeeze) use HEI/A-Frame's `#22DDFF`
highlight on that mesh only; releasing restores the original color.
It does not animate individual model button transforms or implement full glTF PBR materials.
Unknown controller profiles use the Quest 2 model as a visual fallback.

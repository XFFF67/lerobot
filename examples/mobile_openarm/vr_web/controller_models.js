"use strict";

// The same Quest 2 / Touch Plus / Touch Pro assets used by HEI's
// A-Frame oculus-touch-controls. This loader handles only our bundled, static
// GLB meshes (float attributes, uint16 indices, embedded base-color textures).
// It is deliberately not a general glTF loader.
function controllerModelName(profiles = []) {
  if (profiles.includes("meta-quest-touch-plus")) return "touch-plus";
  if (profiles.includes("meta-quest-touch-pro")) return "touch-pro";
  return "touch-v3";
}

// HEI's oculus-touch-controls uses buttonHighlightColor "#2DF" on press,
// and restores the original material color on release.
const controllerButtonHighlight = [0x22 / 255, 0xdd / 255, 1, 1];
function controllerPartColor(part, controller) {
  const active = {
    trigger: controller.trigger,
    squeeze: controller.gripActive,
    thumbstick: controller.thumbstick?.pressed,
    x_button: controller.xButton,
    y_button: controller.yButton,
    a_button: controller.aButton,
    b_button: controller.bButton
  };
  return active[part.name] ? controllerButtonHighlight : part.color;
}

function multiplyMatrices(a, b) {
  const result = new Float32Array(16);
  for (let col = 0; col < 4; col++) {
    for (let row = 0; row < 4; row++) {
      for (let k = 0; k < 4; k++) result[col * 4 + row] += a[k * 4 + row] * b[col * 4 + k];
    }
  }
  return result;
}

function decodeControllerModel(data) {
  const header = new DataView(data);
  if (header.getUint32(0, true) !== 0x46546c67 || header.getUint32(4, true) !== 2) {
    throw new Error("Invalid controller GLB");
  }
  const jsonLength = header.getUint32(12, true);
  const json = JSON.parse(new TextDecoder().decode(new Uint8Array(data, 20, jsonLength)));
  const binaryStart = 20 + jsonLength + 8;
  const binary = new DataView(data, binaryStart);
  const readAccessor = (id) => {
    const a = json.accessors[id], v = json.bufferViews[a.bufferView];
    const width = { SCALAR: 1, VEC2: 2, VEC3: 3 }[a.type];
    const bytes = a.componentType === 5126 ? 4 : 2;
    if (!width || ![5126, 5123].includes(a.componentType) || a.sparse || a.normalized) {
      throw new Error("Unsupported controller accessor");
    }
    const values = [];
    for (let i = 0; i < a.count; i++) {
      for (let k = 0; k < width; k++) {
        const offset = (v.byteOffset || 0) + (a.byteOffset || 0) + i * (v.byteStride || width * bytes) + k * bytes;
        values.push(bytes === 4 ? binary.getFloat32(offset, true) : binary.getUint16(offset, true));
      }
    }
    return values;
  };
  const parts = [];
  const visit = (id, parent) => {
    const node = json.nodes[id];
    const [x, y, z, w] = node.rotation || [0, 0, 0, 1];
    const [px, py, pz] = node.translation || [0, 0, 0];
    const local = node.matrix ? new Float32Array(node.matrix) : controllerMatrix({
      quaternion: { x, y, z, w }, position: { x: px, y: py, z: pz }
    });
    if (!node.matrix) {
      for (let col = 0; col < 3; col++) {
        for (let row = 0; row < 3; row++) local[col * 4 + row] *= (node.scale || [1, 1, 1])[col];
      }
    }
    const world = multiplyMatrices(parent, local);
    if (node.mesh !== undefined) {
      for (const primitive of json.meshes[node.mesh].primitives) {
        if ((primitive.mode ?? 4) !== 4) throw new Error("Controller mesh must use triangles");
        const positions = readAccessor(primitive.attributes.POSITION);
        const normals = readAccessor(primitive.attributes.NORMAL);
        const uv = readAccessor(primitive.attributes.TEXCOORD_0);
        const indices = primitive.indices === undefined
          ? Array.from({ length: positions.length / 3 }, (_, i) => i) : readAccessor(primitive.indices);
        const vertices = new Float32Array(indices.length * 8);
        indices.forEach((index, i) => {
          const p = positions.slice(index * 3, index * 3 + 3);
          for (let row = 0; row < 3; row++) {
            vertices[i * 8 + row] = world[12 + row] +
              world[row] * p[0] + world[4 + row] * p[1] + world[8 + row] * p[2];
            // The bundled meshes have uniform scale, so rotation + normalization
            // gives the same normal direction as the inverse-transpose matrix.
            vertices[i * 8 + 5 + row] = world[row] * normals[index * 3] +
              world[4 + row] * normals[index * 3 + 1] + world[8 + row] * normals[index * 3 + 2];
          }
          vertices[i * 8 + 3] = uv[index * 2];
          vertices[i * 8 + 4] = uv[index * 2 + 1];
        });
        const material = json.materials[primitive.material].pbrMetallicRoughness;
        parts.push({ vertices, name: node.name, color: material.baseColorFactor || [1, 1, 1, 1],
          image: json.textures[material.baseColorTexture.index].source });
      }
    }
    for (const child of node.children || []) visit(child, world);
  };
  for (const node of json.scenes[json.scene || 0].nodes) visit(node, identity);
  const images = json.images.map(image => {
    const view = json.bufferViews[image.bufferView], start = binaryStart + (view.byteOffset || 0);
    return new Blob([data.slice(start, start + view.byteLength)], { type: image.mimeType });
  });
  return { parts, images };
}

class ControllerModels {
  constructor(gl) {
    this.gl = gl;
    this.models = new Map();
    this.disposed = false;
  }

  get(side, profiles) {
    const name = `${controllerModelName(profiles)}-${side}`;
    if (!this.models.has(name)) {
      const entry = { parts: [], textures: [] };
      this.models.set(name, entry);
      entry.loading = this.load(name, entry).catch(error => {
        if (!this.disposed) {
          console.error(error);
          diagnostic("controller_model_error", `${name}: ${error.message}`);
        }
      });
    }
    return this.models.get(name);
  }

  async load(name, entry) {
    const response = await fetch(`/models/${name}.glb`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const decoded = decodeControllerModel(await response.arrayBuffer());
    // Decode only images actually used as base color, not unused PBR maps.
    for (const index of new Set(decoded.parts.map(part => part.image))) {
      const image = await createImageBitmap(decoded.images[index], {
        imageOrientation: "none", premultiplyAlpha: "none"
      });
      if (this.disposed) { image.close(); return; }
      const gl = this.gl, texture = gl.createTexture();
      entry.textures[index] = texture;
      gl.bindTexture(gl.TEXTURE_2D, texture);
      gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, image);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      image.close();
    }
    if (this.disposed) return;
    for (const part of decoded.parts) {
      const gl = this.gl, buffer = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
      gl.bufferData(gl.ARRAY_BUFFER, part.vertices, gl.STATIC_DRAW);
      entry.parts.push({ ...part, buffer, count: part.vertices.length / 8 });
    }
  }

  draw(side, controller, program) {
    const gl = this.gl, entry = this.get(side, controller.profiles);
    gl.uniformMatrix4fv(gl.getUniformLocation(program, "model"), false, controllerMatrix(controller));
    gl.uniform1i(gl.getUniformLocation(program, "textured"), 1);
    gl.uniform1i(gl.getUniformLocation(program, "image"), 0);
    gl.activeTexture(gl.TEXTURE0);
    for (const part of entry.parts) {
      gl.bindBuffer(gl.ARRAY_BUFFER, part.buffer);
      for (const [name, size, offset] of [["p", 3, 0], ["uv", 2, 12], ["normal", 3, 20]]) {
        const location = gl.getAttribLocation(program, name);
        gl.enableVertexAttribArray(location);
        gl.vertexAttribPointer(location, size, gl.FLOAT, false, 32, offset);
      }
      gl.bindTexture(gl.TEXTURE_2D, entry.textures[part.image]);
      gl.uniform4fv(gl.getUniformLocation(program, "color"), controllerPartColor(part, controller));
      gl.drawArrays(gl.TRIANGLES, 0, part.count);
    }
  }

  dispose() {
    this.disposed = true;
    for (const entry of this.models.values()) {
      for (const part of entry.parts) this.gl.deleteBuffer(part.buffer);
      for (const texture of entry.textures) if (texture) this.gl.deleteTexture(texture);
    }
  }
}

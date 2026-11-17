// Browser lifecycle/packet regression with WebXR stubs; run with Node.js.
// This checks our adapter, not headset/browser support or GPU rendering.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

async function check(supportsAR, probeUnavailable = false, requestDenied = false) {
  const elements = {};
  const calls = [];
  const paintedText = [];
  const timers = [];
  let transparentClears = 0;
  const canvasContext = {
    clearRect() { transparentClears++; },
    fillRect() { assert.fail("HEI head text must not have an opaque backing panel"); },
    fillText(text) { paintedText.push(text); }
  };
  const gl = new Proxy({}, { get: (_target, name) => {
    if (name === "getShaderParameter" || name === "getProgramParameter") return () => true;
    if (name === "makeXRCompatible") return async () => {};
    if (name === "getAttribLocation") return (_program, attr) => ({ p: 0, uv: 1, normal: 2 })[attr];
    if (/^[A-Z_0-9]+$/.test(name)) return name;
    return (...args) => { calls.push([name, ...args]); return {}; };
  } });
  const xr = {
    inputSources: [], visibilityState: "visible", environmentBlendMode: "opaque",
    listeners: {}, referenceListeners: {},
    addEventListener(name, fn) { this.listeners[name] = fn; },
    requestAnimationFrame(fn) { this.frame = fn; },
    updateRenderState(state) { this.renderState = state; },
    async requestReferenceSpace() {
      return { addEventListener: (name, fn) => { this.referenceListeners[name] = fn; } };
    },
    async end() { this.listeners.end(); }
  };
  class Socket {
    static OPEN = 1;
    constructor() { this.readyState = 1; this.bufferedAmount = 0; this.sent = []; Socket.instance = this; }
    send(message) { this.sent.push(JSON.parse(message)); }
    close() { this.readyState = 3; this.onclose(); }
  }
  let requests = 0, requestedMode;
  const context = vm.createContext({
    document: {
      getElementById: (id) => elements[id] ||= { disabled: true, getContext: () => gl },
      createElement: () => ({ getContext: () => canvasContext })
    },
    navigator: { xr: {
      isSessionSupported: async mode => probeUnavailable ? false : mode === "immersive-ar" ? supportsAR : true,
      requestSession: async mode => {
        requests++; requestedMode = mode;
        if (requestDenied) { const error = new Error("No immersive device available"); error.name = "NotSupportedError"; throw error; }
        return xr;
      }
    } },
    location: { host: "localhost:8443", origin: "https://localhost:8443", pathname: supportsAR ? "/real" : "/" },
    window: { addEventListener() {} }, WebSocket: Socket,
    XRWebGLLayer: class { getViewport() { return { x: 0, y: 0, width: 1, height: 1 }; } },
    Float32Array, Blob, TextDecoder, performance: { now: () => 0 },
    setTimeout: fn => { timers.push(fn); return timers.length; }, clearTimeout() {}, console,
    fetch: async url => {
      const data = fs.readFileSync(path.join(__dirname, "vr_web", url));
      return { ok: true, arrayBuffer: async () => data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength) };
    },
    createImageBitmap: async () => ({ close() {} })
  });
  vm.runInContext(fs.readFileSync(path.join(__dirname, "vr_web/controller_models.js"), "utf8"), context);
  vm.runInContext(fs.readFileSync(path.join(__dirname, "vr_web/vr.js"), "utf8"), context);
  if (supportsAR) {
    assert.equal(elements.simHelp.hidden, true);
    assert.equal(elements.realHelp.hidden, false);
    assert.equal(elements.pageTitle.textContent, "双臂真机 VR");
  }
  // Decode every vendored asset: catches corrupt files, unsupported formats and
  // missing node transforms (Touch Plus geometry is authored in centimeters).
  for (const file of fs.readdirSync(path.join(__dirname, "vr_web/models")).filter(f => f.endsWith(".glb"))) {
    const bytes = fs.readFileSync(path.join(__dirname, "vr_web/models", file));
    context.modelData = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
    const decoded = vm.runInContext("decodeControllerModel(modelData)", context);
    assert.equal(decoded.parts.length, 6, file);
    assert.ok(decoded.parts.some(part => part.name === "trigger"), `${file}: trigger must be independently tintable`);
    for (const part of decoded.parts) {
      assert.equal(part.vertices.length % 24, 0);
      assert.ok(part.vertices.every(Number.isFinite), file);
      const coords = part.vertices.filter((_, i) => i % 8 < 3);
      assert.ok(Math.max(...coords.map(Math.abs)) < 0.3, `${file}: meter-scale geometry`);
      assert.ok(decoded.images[part.image].size > 0);
    }
  }
  assert.equal(vm.runInContext('controllerModelName(["meta-quest-touch-plus", "oculus-touch-v3"])', context), "touch-plus");
  assert.equal(vm.runInContext('controllerModelName(["meta-quest-touch-pro"])', context), "touch-pro");
  assert.equal(vm.runInContext('headRotation({x: 0, y: Math.SQRT1_2, z: 0, w: Math.SQRT1_2})', context), "0  90  0");
  await new Promise(resolve => setImmediate(resolve));
  const socket = Socket.instance;
  const packets = () => socket.sent.filter(p => "leftController" in p);
  socket.onopen();
  assert.equal(elements.enter.disabled, false);
  if (probeUnavailable) assert.match(elements.status.textContent, /点击 Start VR 重试/);
  await Promise.all([elements.enter.onclick(), elements.enter.onclick()]);
  assert.equal(requests, 1, "double clicks must not create two XR sessions");
  if (requestDenied) {
    assert.match(elements.status.textContent, /NotSupportedError: No immersive device available/);
    assert.equal(elements.enter.disabled, false, "failed request must allow an explicit retry");
    assert.ok(socket.sent.some(p => p.stage === "xr_error" && p.error.includes("NotSupportedError")));
    console.log("PASS unsupported probe: clickable explicit retry reports real XR error");
    return;
  }
  assert.equal(requestedMode, supportsAR ? "immersive-ar" : "immersive-vr");
  assert.equal(elements.enter.disabled, false);
  assert.equal(elements.enter.textContent, "退出控制");
  const buttons = Array.from({ length: 6 }, () => ({ pressed: false, value: 0 }));
  buttons[0].value = 1; buttons[1].pressed = true; buttons[4].pressed = true;
  const source = { handedness: "left", gripSpace: {}, gamepad: {
    mapping: "xr-standard", buttons, axes: [0, 0, 0.5, -0.75]
  } };
  xr.inputSources = [source];
  const frame = { session: xr,
    getPose: () => ({ transform: {
      position: { x: 0.1, y: 1.3, z: -0.4 }, orientation: { x: 0, y: 0, z: 0, w: 1 }
    } }),
    getViewerPose: () => ({ transform: {
      position: { x: 0.25, y: 1.65, z: -0.5 }, orientation: { x: 0, y: 0, z: 0, w: 1 },
      matrix: new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, 0.25,1.65,-0.5,1])
    }, views: Array.from({ length: 2 }, () => ({
      projectionMatrix: new Float32Array(16),
      transform: { inverse: { matrix: new Float32Array(16) } }
    })) })
  };
  xr.frame(100, frame);
  await vm.runInContext("Promise.all([...controllerModels.models.values()].map(entry => entry.loading))", context);
  assert.equal(vm.runInContext("controllerModels.models.get('touch-v3-left').parts.length", context), 6);
  assert.match(elements.poseStatus.textContent, /Head Pos:  X 0.250  Y 1.650  Z -0.500/);
  assert.ok(paintedText.some(text => text.startsWith("Head Rot: 0  0  0")));
  assert.deepEqual(paintedText, ["Head Pos: 0.25 1.65 -0.50", "Head Rot: 0  0  0"]);
  assert.equal(transparentClears, 1);
  assert.ok(calls.some(call => call[0] === "blendFuncSeparate" && call[1] === "SRC_ALPHA"));
  assert.equal(calls.filter(call => call[0] === "depthMask").at(-1)[1], true);
  assert.equal(calls.filter(call => call[0] === "drawArrays" && call[3] === 6).length, 2,
    "draw the HUD in both eyes");
  const packet = packets().at(-1);
  assert.equal(packet.leftController.gripActive, true);
  assert.equal(packet.leftController.trigger, true);
  assert.equal(packet.leftController.xButton, true);
  assert.equal(packet.leftController.thumbstick.y, -0.75);
  assert.equal(packet.rightController, null);
  // Both eyes must tint only the pressed trigger and restore it immediately on release.
  const isBlue = call => call[0] === "uniform4fv" &&
    JSON.stringify(call[2]) === JSON.stringify([0x22 / 255, 0xdd / 255, 1, 1]);
  buttons[1].pressed = buttons[4].pressed = false;
  let beforeColors = calls.length;
  xr.frame(120, frame);
  assert.equal(calls.slice(beforeColors).filter(isBlue).length, 2, "one blue trigger per eye");
  buttons[0].value = 0;
  beforeColors = calls.length;
  xr.frame(140, frame);
  assert.equal(calls.slice(beforeColors).filter(isBlue).length, 0, "release restores original model colors");
  buttons[0].value = 1;
  buttons[1].pressed = buttons[4].pressed = true;
  // 72/90 Hz headsets must not drop alternate pose packets through a 60 Hz gate.
  for (const hz of [72, 90, 120]) {
    const before = packets().length;
    for (let i = 1; i <= 10; i++) xr.frame(100 + i * 1000 / hz, frame);
    assert.equal(packets().length - before, 10, `${hz} Hz pose cadence`);
  }
  const beforeBackpressure = packets().length;
  socket.bufferedAmount = 4096;
  xr.frame(250, frame);
  assert.equal(packets().length, beforeBackpressure, "do not queue poses behind a slow network");
  socket.bufferedAmount = 0;
  xr.frame(270, frame);
  assert.equal(packets().length, beforeBackpressure + 1);
  const rotated = vm.runInContext(`controllerMatrix({
    position: {x: 1, y: 2, z: 3},
    quaternion: {x: 0, y: 0, z: Math.SQRT1_2, w: Math.SQRT1_2}
  })`, context);
  assert.ok(Math.abs(rotated[0]) < 1e-6 && Math.abs(rotated[1] - 1) < 1e-6,
    "local +X arrow must rotate to +Y with the controller");
  assert.deepEqual(Array.from(rotated.slice(12, 15)), [1, 2, 3]);
  xr.visibilityState = "hidden";
  xr.listeners.visibilitychange();
  assert.equal(packets().at(-1).leftController, null);
  xr.visibilityState = "visible";
  const getPose = frame.getPose;
  frame.getPose = () => null;
  xr.frame(130, frame);
  assert.equal(packets().at(-1).leftController, null);
  xr.referenceListeners.reset();
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(elements.enter.textContent, "退出控制", "recenter must keep the XR session open");
  assert.equal(requests, 1);
  assert.equal(packets().at(-1).leftController, null);
  frame.getPose = getPose;
  xr.frame(300, frame);
  assert.equal(packets().at(-1).leftController, null, "recenter pause");
  const beforeResume = calls.length;
  xr.frame(800, frame);
  assert.equal(packets().at(-1).leftController, null, "held grip must not resume after recenter");
  assert.match(elements.poseStatus.textContent, /Left:  X 0.100/);
  assert.ok(calls.slice(beforeResume).some(call => call[0] === "drawArrays" && call[3] > 1000),
    "controller model remains visible while commands wait for grip release");
  buttons[1].pressed = false;
  xr.frame(820, frame);
  assert.equal(packets().at(-1).leftController.gripActive, false);
  buttons[1].pressed = true;
  xr.frame(840, frame);
  assert.equal(packets().at(-1).leftController.gripActive, true);
  frame.getViewerPose = () => null;
  xr.frame(1000, frame);
  assert.match(elements.poseStatus.textContent, /Head Pos:  未追踪/);
  assert.equal(socket.sent.some(p => p.stage === "frame_error"), false);
  await elements.enter.onclick();
  assert.equal(elements.enter.disabled, false);
  assert.equal(elements.enter.textContent, "Start VR");
  assert.match(elements.poseStatus.textContent, /未进入 XR/);
  assert.ok(calls.some(call => call[0] === "deleteTexture"));
  await elements.enter.onclick();
  socket.close();
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(elements.enter.disabled, false, "manual XR exit remains available offline");
  assert.equal(elements.enter.textContent, "退出控制", "socket loss must not end XR");
  assert.match(elements.status.textContent, /VR 保持开启/);
  buttons[1].pressed = false;
  xr.frame(1500, frame); // An offline release must not count after reconnection.
  buttons[1].pressed = true;
  timers.shift()();
  const reconnected = Socket.instance;
  reconnected.onopen();
  xr.frame(1520, frame);
  const recoveredPackets = () => reconnected.sent.filter(p => "leftController" in p);
  assert.equal(recoveredPackets().at(-1).leftController, null, "held grip cannot resume after reconnect");
  assert.equal(requests, 2, "reconnect must reuse the existing XR session");
  socket.onclose({ code: 1006 });
  assert.match(elements.status.textContent, /连接已恢复/, "old socket cannot overwrite new connection");
  buttons[1].pressed = false;
  xr.frame(1540, frame);
  assert.equal(recoveredPackets().at(-1).leftController.gripActive, false);
  buttons[1].pressed = true;
  xr.frame(1560, frame);
  assert.equal(recoveredPackets().at(-1).leftController.gripActive, true);
  await elements.enter.onclick();
  assert.equal(elements.enter.textContent, "Start VR");
  assert.ok(reconnected.sent.some(p => p.xr_end === "点击退出控制"));
  console.log(`PASS WebXR ${supportsAR ? "automatic AR" : "VR fallback"}: lifecycle, mapping, cadence, backpressure, HEI models, stereo HUD, tracking loss, recenter, disconnect`);
}
check(true).then(() => check(false)).then(() => check(false, true))
  .then(() => check(false, true, true))
  .catch(error => { console.error(error); process.exitCode = 1; });

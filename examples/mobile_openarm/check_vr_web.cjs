// Browser lifecycle/packet regression with WebXR stubs; run with Node.js.
// This checks our adapter, not headset/browser support or GPU rendering.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

async function check() {
  const elements = {};
  const gl = new Proxy({}, { get: (_target, name) => {
    if (name === "getShaderParameter" || name === "getProgramParameter") return () => true;
    if (name === "makeXRCompatible") return async () => {};
    return () => ({});
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
  let requests = 0;
  const context = vm.createContext({
    document: { getElementById: (id) => elements[id] ||= { disabled: true, getContext: () => gl } },
    navigator: { xr: {
      isSessionSupported: async () => true,
      requestSession: async () => { requests++; return xr; }
    } },
    location: { host: "localhost:8443" },
    window: { addEventListener() {} }, WebSocket: Socket,
    XRWebGLLayer: class { getViewport() { return { x: 0, y: 0, width: 1, height: 1 }; } },
    Float32Array, performance: { now: () => 0 }, setTimeout: () => 1, clearTimeout() {}, console
  });
  vm.runInContext(fs.readFileSync(path.join(__dirname, "vr_web/vr.js"), "utf8"), context);
  await new Promise(resolve => setImmediate(resolve));
  const socket = Socket.instance;
  const packets = () => socket.sent.filter(p => "leftController" in p);
  socket.onopen();
  assert.equal(elements.enter.disabled, false);
  await Promise.all([elements.enter.onclick(), elements.enter.onclick()]);
  assert.equal(requests, 1, "double clicks must not create two XR sessions");
  assert.equal(elements.enter.disabled, true);
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
    getViewerPose: () => ({ views: [] })
  };
  xr.frame(100, frame);
  const packet = packets().at(-1);
  assert.equal(packet.leftController.gripActive, true);
  assert.equal(packet.leftController.trigger, true);
  assert.equal(packet.leftController.xButton, true);
  assert.equal(packet.leftController.thumbstick.y, -0.75);
  assert.equal(packet.rightController, null);
  xr.frame(101, frame);
  assert.equal(packets().length, 1, "rate limit");
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
  assert.equal(elements.enter.disabled, true, "recenter must keep the XR session open");
  assert.equal(packets().at(-1).leftController, null);
  frame.getPose = getPose;
  xr.frame(300, frame);
  assert.equal(packets().at(-1).leftController, null, "recenter pause");
  xr.frame(800, frame);
  assert.equal(packets().at(-1).leftController, null, "held grip must not resume after recenter");
  buttons[1].pressed = false;
  xr.frame(820, frame);
  assert.equal(packets().at(-1).leftController.gripActive, false);
  buttons[1].pressed = true;
  xr.frame(840, frame);
  assert.equal(packets().at(-1).leftController.gripActive, true);
  await elements.exit.onclick();
  assert.equal(elements.enter.disabled, false);
  await elements.passthrough.onclick();
  socket.close();
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(elements.enter.disabled, true);
  console.log("PASS WebXR adapter: session lifecycle, button/pose mapping, rate limit, tracking loss, recenter, disconnect");
}
check().catch(error => { console.error(error); process.exitCode = 1; });

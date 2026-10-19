"use strict";

// Native WebXR, no CDN or controller SDK. Wire format matches Telegrip.
const status = document.getElementById("status");
const enter = document.getElementById("enter");
const passthrough = document.getElementById("passthrough");
const exit = document.getElementById("exit");
let socket, session, referenceSpace, gl, program, buffer, lastSent = -Infinity;
let reconnectTimer, closing = false;
let starting = false, vrSupported = false, arSupported = false;
let frames = 0, lastDiagnostic = -Infinity, pauseUntil = -Infinity;
const needsRelease = new Set();
let stage = "loading", lastError = "";
const inputStatus = { left: "not_seen", right: "not_seen" };
const empty = () => ({ leftController: null, rightController: null });

function diagnostic(nextStage, error = "") {
  stage = nextStage;
  lastError = error;
  send({ type: "diagnostic", stage, ...(error ? { error } : {}),
    frames, visibility: session?.visibilityState || "not_in_xr", ...inputStatus });
  document.getElementById("diagnostics").textContent =
    `阶段：${stage} | XR 帧：${frames}\n左：${inputStatus.left} | 右：${inputStatus.right}${error ? `\n错误：${error}` : ""}`;
}

function updateButtons() {
  const ready = socket?.readyState === WebSocket.OPEN && !session && !starting;
  enter.disabled = !ready || !vrSupported;
  passthrough.disabled = !ready || !arSupported;
  exit.disabled = !session;
}

function send(packet) {
  if (socket?.readyState === WebSocket.OPEN && socket.bufferedAmount < 4096) {
    socket.send(JSON.stringify(packet));
  }
}

function connect() {
  socket = new WebSocket(`wss://${location.host}/ws`);
  socket.onopen = () => {
    status.textContent = "已连接电脑。选择 VR 或透视模式，连接后先松开握把。";
    updateButtons();
    diagnostic("page_ready_click_enter");
  };
  socket.onerror = () => { status.textContent = "连接失败，请检查电脑上的 VR 服务。"; };
  socket.onclose = () => {
    status.textContent = "连接已断开，控制已停止。正在重连…";
    // End the XR session: reconnect must be an explicit user gesture.
    session?.end().catch(() => {});
    updateButtons();
    if (!closing) reconnectTimer = setTimeout(connect, 1500);
  };
}

function controller(frame, source) {
  const pose = source.gripSpace && frame.getPose(source.gripSpace, referenceSpace);
  const gamepad = source.gamepad;
  let reason = "tracked";
  if (!pose) reason = "no_grip_pose";
  else if (pose.emulatedPosition) reason = "position_not_tracked";
  else if (!gamepad) reason = "no_gamepad_use_controllers";
  else if (gamepad.mapping !== "xr-standard") reason = `unsupported_mapping:${gamepad.mapping}`;
  inputStatus[source.handedness] = reason;
  if (reason !== "tracked") return null;
  const pressed = (i) => Boolean(gamepad.buttons[i]?.pressed || gamepad.buttons[i]?.value > 0.5);
  const p = pose.transform.position, q = pose.transform.orientation;
  return {
    tracked: true,
    position: { x: p.x, y: p.y, z: p.z },
    quaternion: { x: q.x, y: q.y, z: q.z, w: q.w },
    gripActive: pressed(1), trigger: pressed(0),
    thumbstick: { x: gamepad.axes[2] || 0, y: gamepad.axes[3] || 0, pressed: pressed(3) },
    ...(source.handedness === "left" ? { xButton: pressed(4), yButton: pressed(5) }
      : { aButton: pressed(4), bButton: pressed(5) })
  };
}

function shader(type, source) {
  const s = gl.createShader(type);
  gl.shaderSource(s, source); gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
  return s;
}

function initGraphics() {
  program = gl.createProgram();
  gl.attachShader(program, shader(gl.VERTEX_SHADER,
    "attribute vec3 p; uniform mat4 projection, view; void main(){gl_Position=projection*view*vec4(p,1.0);}"));
  gl.attachShader(program, shader(gl.FRAGMENT_SHADER,
    "precision mediump float; uniform vec4 color; void main(){gl_FragColor=color;}"));
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program));
  buffer = gl.createBuffer();
}

function lines(vertices, color) {
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertices), gl.DYNAMIC_DRAW);
  const p = gl.getAttribLocation(program, "p");
  gl.enableVertexAttribArray(p); gl.vertexAttribPointer(p, 3, gl.FLOAT, false, 0, 0);
  gl.uniform4fv(gl.getUniformLocation(program, "color"), color);
  gl.drawArrays(gl.LINES, 0, vertices.length / 3);
}

function render(frame, packet) {
  const pose = frame.getViewerPose(referenceSpace);
  if (!pose) return;
  const layer = session.renderState.baseLayer;
  gl.bindFramebuffer(gl.FRAMEBUFFER, layer.framebuffer);
  // Transparent background permits the headset's own passthrough in immersive-ar.
  gl.clearColor(0.025, 0.045, 0.07, session.environmentBlendMode === "opaque" ? 1 : 0);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  gl.useProgram(program);
  for (const view of pose.views) {
    const viewport = layer.getViewport(view);
    gl.viewport(viewport.x, viewport.y, viewport.width, viewport.height);
    gl.uniformMatrix4fv(gl.getUniformLocation(program, "projection"), false, view.projectionMatrix);
    gl.uniformMatrix4fv(gl.getUniformLocation(program, "view"), false, view.transform.inverse.matrix);
    if (session.environmentBlendMode === "opaque") {
      const grid = [];
      for (let i = -5; i <= 5; i++) grid.push(i, 0, -5, i, 0, 5, -5, 0, i, 5, 0, i);
      lines(grid, [0.15, 0.25, 0.32, 1]);
    }
    for (const side of ["left", "right"]) {
      const c = packet[`${side}Controller`];
      if (!c) continue;
      const { x, y, z } = c.position, d = 0.05;
      lines([x-d,y,z,x+d,y,z, x,y-d,z,x,y+d,z, x,y,z-d,x,y,z+d],
        c.gripActive ? [1, 0.65, 0.1, 1] : [0.25, 0.95, 0.8, 1]);
    }
  }
}

function onFrame(time, frame) {
  if (!session || frame.session !== session) return;
  session.requestAnimationFrame(onFrame);
  frames++;
  inputStatus.left = inputStatus.right = "not_seen";
  const packet = empty();
  try {
    if (session.visibilityState === "visible" && time >= pauseUntil) {
      for (const source of session.inputSources) {
        if (!["left", "right"].includes(source.handedness)) continue;
        let value = controller(frame, source);
        if (needsRelease.has(source.handedness)) {
          if (value && !value.gripActive) needsRelease.delete(source.handedness);
          else { value = null; inputStatus[source.handedness] = "release_grip_to_resume"; }
        }
        packet[`${source.handedness}Controller`] = value;
      }
    }
    if (time - lastSent >= 1000 / 60) { send(packet); lastSent = time; }
    render(frame, packet);
    if (time - lastDiagnostic >= 1000) {
      diagnostic(session.visibilityState === "visible" ? "xr_running" : "xr_hidden");
      lastDiagnostic = time;
    }
  } catch (error) {
    send(empty());
    if (time - lastDiagnostic >= 1000) {
      diagnostic("frame_error", `${error.name}: ${error.message}`); lastDiagnostic = time;
    }
  }
}

async function start(mode) {
  if (session || starting || socket?.readyState !== WebSocket.OPEN) return;
  starting = true; updateButtons();
  try {
    diagnostic("requesting_xr_permission");
    status.textContent = "正在请求进入 XR，请在头显中确认浏览器的权限提示…";
    session = await navigator.xr.requestSession(mode, { requiredFeatures: ["local-floor"] });
    diagnostic("xr_session_created");
    session.addEventListener("end", () => {
      send(empty()); session = null; updateButtons();
      if (gl) { gl.deleteProgram(program); gl.deleteBuffer(buffer); }
      diagnostic(lastError ? "xr_error" : "xr_ended_click_enter", lastError);
    }, { once: true });
    gl = document.getElementById("view").getContext("webgl", { xrCompatible: true, alpha: true });
    if (!gl) throw new Error("无法创建 WebGL 上下文");
    diagnostic("initializing_graphics");
    await gl.makeXRCompatible();
    initGraphics();
    session.updateRenderState({ baseLayer: new XRWebGLLayer(session, gl) });
    diagnostic("requesting_reference_space");
    referenceSpace = await session.requestReferenceSpace("local-floor");
    // Recentring invalidates anchors, but must not unexpectedly exit VR.
    referenceSpace.addEventListener("reset", () => {
      send(empty()); pauseUntil = performance.now() + 500;
      needsRelease.add("left"); needsRelease.add("right");
      diagnostic("recentered_release_grips");
    });
    session.addEventListener("visibilitychange", () => {
      if (session?.visibilityState !== "visible") send(empty());
    });
    lastSent = lastDiagnostic = -Infinity; frames = 0; pauseUntil = -Infinity; needsRelease.clear();
    diagnostic("waiting_for_first_xr_frame");
    status.textContent = "XR 已启动。先松开侧面握把，再握住并移动手柄。";
    session.requestAnimationFrame(onFrame);
  } catch (error) {
    const message = `${error.name}: ${error.message}`;
    status.textContent = `无法进入 XR：${message}`;
    diagnostic("xr_error", message);
    await session?.end().catch(() => {});
  } finally {
    starting = false; updateButtons();
  }
}

enter.onclick = () => start("immersive-vr");
passthrough.onclick = () => start("immersive-ar");
exit.onclick = () => session?.end();
window.addEventListener("pagehide", () => {
  closing = true; clearTimeout(reconnectTimer); send(empty()); socket?.close();
});
window.addEventListener("error", event => diagnostic("javascript_error", event.message));
window.addEventListener("unhandledrejection", event => diagnostic("javascript_error", String(event.reason)));

(async () => {
  if (!navigator.xr) { status.textContent = "未检测到 WebXR。请在 Quest 头显浏览器通过 HTTPS 打开此页面。"; return; }
  vrSupported = await navigator.xr.isSessionSupported("immersive-vr");
  arSupported = await navigator.xr.isSessionSupported("immersive-ar");
  connect();
})().catch(error => { status.textContent = error.message; });

"use strict";

// Native WebXR with locally served HEI/A-Frame controller assets.
const status = document.getElementById("status");
const enter = document.getElementById("enter");
let socket, session, referenceSpace, gl, program, buffer;
let hudProgram, hudTexture, hudCanvas, hudContext, lastHudUpdate = -Infinity;
let controllerModels;
let reconnectTimer, closing = false;
let starting = false, vrSupported = false, arSupported = false;
let frames = 0, lastDiagnostic = -Infinity, pauseUntil = -Infinity;
const needsRelease = new Set();
let stage = "loading", lastError = "";
let lastSocketClose = "", xrEndCause = "";
const inputStatus = { left: "not_seen", right: "not_seen" };
const empty = () => ({ leftController: null, rightController: null });
const identity = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
document.getElementById("serverUrl").textContent = location.origin;
if (location.pathname === "/real") {
  document.title = "OpenArm 双臂真机 VR";
  document.getElementById("pageTitle").textContent = "双臂真机 VR";
  document.getElementById("modeSubtitle").textContent = "电脑直连双臂 · 准备到位后握持控制";
  document.getElementById("simHelp").hidden = true;
  document.getElementById("realHelp").hidden = false;
}

// HEI-style solid arrows: local +X/+Y/+Z, rotating with the grip pose.
function arrowVertices(axis) {
  const vertices = [];
  const point = (along, radius, angle) => {
    const p = [0, 0, 0];
    p[axis] = along;
    p[(axis + 1) % 3] = radius * Math.cos(angle);
    p[(axis + 2) % 3] = radius * Math.sin(angle);
    return p;
  };
  for (let i = 0; i < 12; i++) {
    const a = i * Math.PI / 6, b = (i + 1) * Math.PI / 6;
    const baseA = point(0, 0.004, a), baseB = point(0, 0.004, b);
    const topA = point(0.085, 0.004, a), topB = point(0.085, 0.004, b);
    vertices.push(...baseA, ...baseB, ...topA, ...topA, ...baseB, ...topB);
    vertices.push(...point(0.085, 0.011, a), ...point(0.085, 0.011, b), ...point(0.11, 0, 0));
  }
  return vertices;
}
const arrows = [0, 1, 2].map(arrowVertices);
const axisColors = [[1, 0.23, 0.19, 1], [0.20, 0.78, 0.35, 1], [0.04, 0.52, 1, 1]];
const gripRing = [];
for (let i = 0; i < 24; i++) {
  for (const angle of [i * Math.PI / 12, (i + 1) * Math.PI / 12]) {
    gripRing.push(0.012 * Math.cos(angle), 0.012 * Math.sin(angle), 0);
  }
}

const hudQuad = new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]);

function headRotation(q) {
  const { x, y, z, w } = q;
  const m13 = 2 * (x * z + y * w);
  const angles = Math.abs(m13) < 0.9999999
    ? [Math.atan2(2 * (x * w - y * z), 1 - 2 * (x * x + y * y)),
      Math.asin(Math.max(-1, Math.min(1, m13))),
      Math.atan2(2 * (z * w - x * y), 1 - 2 * (y * y + z * z))]
    : [Math.atan2(2 * (y * z + x * w), 1 - 2 * (x * x + z * z)),
      Math.asin(Math.max(-1, Math.min(1, m13))), 0];
  return angles.map(angle => (angle * 180 / Math.PI).toFixed(0)).join("  ");
}

function poseText(label, position) {
  return position
    ? `${label}  X ${position.x.toFixed(3)}  Y ${position.y.toFixed(3)}  Z ${position.z.toFixed(3)}`
    : `${label}  未追踪`;
}

function updatePoseDisplay(pose, packet, time) {
  const lines = [
    "Position (m) / Rotation XYZ (deg) · local-floor",
    poseText("Head Pos:", pose && !pose.emulatedPosition ? pose.transform.position : null),
    `Head Rot: ${pose ? headRotation(pose.transform.orientation) : "未追踪"}`,
    poseText("Left:", packet.leftController?.position),
    poseText("Right:", packet.rightController?.position)
  ];
  // Refresh text at 10 Hz; all poses and controller graphics still run every XR frame.
  if (time - lastHudUpdate < 100) return;
  lastHudUpdate = time;
  document.getElementById("poseStatus").textContent = lines.join("\n");
  // Match HEI's camera-attached a-text: two centered lines, no backing panel.
  hudContext.clearRect(0, 0, hudCanvas.width, hudCanvas.height);
  hudContext.font = "42px monospace";
  hudContext.fillStyle = "#EAF4FF";
  hudContext.textAlign = "center";
  hudContext.textBaseline = "middle";
  const headPosition = pose && !pose.emulatedPosition ? pose.transform.position : null;
  hudContext.fillText(headPosition
    ? `Head Pos: ${["x", "y", "z"].map(axis => headPosition[axis].toFixed(2)).join(" ")}`
    : "Head Pos: 未追踪", hudCanvas.width / 2, 102);
  hudContext.fillText(lines[2], hudCanvas.width / 2, 154);
  if (socket?.readyState !== WebSocket.OPEN || needsRelease.size) {
    hudContext.font = "28px sans-serif";
    hudContext.fillText(socket?.readyState !== WebSocket.OPEN
      ? "连接中断 · 控制暂停" : "请松开握把，再握住恢复控制", hudCanvas.width / 2, 208);
  }
  gl.bindTexture(gl.TEXTURE_2D, hudTexture);
  gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, hudCanvas);
}

function controllerMatrix(controller) {
  const { x, y, z, w } = controller.quaternion;
  const p = controller.position;
  return new Float32Array([
    1-2*(y*y+z*z), 2*(x*y+z*w), 2*(x*z-y*w), 0,
    2*(x*y-z*w), 1-2*(x*x+z*z), 2*(y*z+x*w), 0,
    2*(x*z+y*w), 2*(y*z-x*w), 1-2*(x*x+y*y), 0,
    p.x, p.y, p.z, 1
  ]);
}

function diagnostic(nextStage, error = "") {
  stage = nextStage;
  lastError = error;
  send({ type: "diagnostic", stage, ...(error ? { error } : {}),
    ws_close: lastSocketClose, xr_end: xrEndCause,
    frames, visibility: session?.visibilityState || "not_in_xr", ...inputStatus });
  document.getElementById("diagnostics").textContent =
    `阶段：${stage} | XR 帧：${frames}\n左：${inputStatus.left} | 右：${inputStatus.right}${error ? `\n错误：${error}` : ""}` +
    (lastSocketClose ? `\n最近断线：${lastSocketClose}` : "") + (xrEndCause ? `\nXR 结束：${xrEndCause}` : "");
}

function updateButtons() {
  // The capability probe can report false while the headset runtime is not
  // ready. Let an explicit click request a session and surface its actual
  // error, instead of leaving an unexplained permanently disabled button.
  const ready = socket?.readyState === WebSocket.OPEN && typeof navigator.xr?.requestSession === "function";
  enter.disabled = starting || (!session && !ready);
  enter.textContent = starting ? "Starting…" : session ? "退出控制" : "Start VR";
}

function send(packet) {
  if (socket?.readyState === WebSocket.OPEN && socket.bufferedAmount < 4096) {
    socket.send(JSON.stringify(packet));
  }
}

function connect() {
  const connection = new WebSocket(`wss://${location.host}/ws`);
  socket = connection;
  connection.onopen = () => {
    if (socket !== connection) return;
    updateButtons();
    if (session) {
      needsRelease.add("left"); needsRelease.add("right");
      send(empty());
      status.textContent = "连接已恢复，VR 保持开启。请松开握把，再握住恢复控制。";
      diagnostic("socket_reconnected_release_grips");
    } else if (vrSupported || arSupported) {
      status.textContent = "已连接电脑。点击 Start VR，进入后先松开握把。";
      diagnostic("page_ready_click_enter");
    } else {
      status.textContent = "已连接电脑，但头显尚未报告可用 XR。点击 Start VR 重试，若失败会显示具体原因。";
      diagnostic("xr_probe_unavailable_click_retry", "WebXR 已提供，但 VR/AR 支持预检均未通过；等待点击获取实际结果。");
    }
  };
  connection.onerror = () => {
    if (socket === connection) status.textContent = "连接失败，请检查电脑上的 VR 服务。";
  };
  connection.onclose = event => {
    if (socket !== connection) return;
    lastSocketClose = `code=${event?.code ?? "unknown"}, reason=${event?.reason || "未提供"}, clean=${Boolean(event?.wasClean)}`;
    const occupied = event?.code === 1013;
    status.textContent = occupied
      ? "另一个 VR 页面已连接。请关闭旧的 VR 标签页（包括电脑上的），然后刷新本页。"
      : "连接已断开，控制暂停，VR 保持开启。正在重连…";
    // Keep rendering through a transport interruption. Neither held grips nor
    // a release while offline may silently re-arm control on reconnection.
    needsRelease.add("left"); needsRelease.add("right");
    diagnostic("socket_disconnected", lastSocketClose);
    updateButtons();
    if (!closing && !occupied) reconnectTimer = setTimeout(connect, 1500);
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
  const axis = (i) => Number.isFinite(gamepad.axes[i]) ? gamepad.axes[i] : 0;
  // xr-standard reserves two 2-D axis pairs. Quest/browser versions have
  // exposed the Touch thumbstick as either pair, so select whichever pair is
  // actually moving instead of permanently assuming axes 2/3.
  const primaryAxes = [axis(0), axis(1)];
  const secondaryAxes = [axis(2), axis(3)];
  const thumbAxes = Math.max(...secondaryAxes.map(Math.abs)) >= Math.max(...primaryAxes.map(Math.abs))
    ? secondaryAxes : primaryAxes;
  const p = pose.transform.position, q = pose.transform.orientation;
  return {
    tracked: true,
    position: { x: p.x, y: p.y, z: p.z },
    quaternion: { x: q.x, y: q.y, z: q.z, w: q.w },
    gripActive: pressed(1), trigger: pressed(0),
    thumbstick: { x: thumbAxes[0], y: thumbAxes[1], pressed: pressed(3) },
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
    "attribute vec3 p, normal; attribute vec2 uv; varying vec2 texcoord; varying float shade;" +
    "uniform mat4 projection, view, model; void main(){texcoord=uv;" +
    "vec3 n=mat3(model)*normal;n=n/max(length(n),0.000001);" +
    "shade=0.55+0.45*max(dot(n,normalize(vec3(0.3,0.7,1.0))),0.0);" +
    "gl_Position=projection*view*model*vec4(p,1.0);}"));
  gl.attachShader(program, shader(gl.FRAGMENT_SHADER,
    "precision mediump float; varying vec2 texcoord; varying float shade; uniform vec4 color; uniform bool textured;" +
    "uniform sampler2D image; void main(){gl_FragColor=textured?texture2D(image,texcoord)*color*vec4(vec3(shade),1.0):color;}"));
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program));
  buffer = gl.createBuffer();
  controllerModels = new ControllerModels(gl);
  hudProgram = gl.createProgram();
  gl.attachShader(hudProgram, shader(gl.VERTEX_SHADER,
    "attribute vec2 p; varying vec2 uv; uniform mat4 projection, view, model;" +
    "void main(){uv=(p+1.0)*0.5;gl_Position=projection*view*model*vec4(p.x*0.45,p.y*0.1125-0.38,-0.75,1.0);}"));
  gl.attachShader(hudProgram, shader(gl.FRAGMENT_SHADER,
    "precision mediump float; varying vec2 uv; uniform sampler2D panel;" +
    "void main(){gl_FragColor=texture2D(panel,uv);}"));
  gl.linkProgram(hudProgram);
  if (!gl.getProgramParameter(hudProgram, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(hudProgram));
  hudCanvas = document.createElement("canvas");
  hudCanvas.width = 1024; hudCanvas.height = 256;
  hudContext = hudCanvas.getContext("2d");
  if (!hudContext) throw new Error("无法创建坐标面板");
  hudTexture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, hudTexture);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  lastHudUpdate = -Infinity;
}

function draw(vertices, color, mode, model = identity) {
  gl.uniform1i(gl.getUniformLocation(program, "textured"), 0);
  gl.disableVertexAttribArray(gl.getAttribLocation(program, "uv"));
  gl.disableVertexAttribArray(gl.getAttribLocation(program, "normal"));
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertices), gl.DYNAMIC_DRAW);
  const p = gl.getAttribLocation(program, "p");
  gl.enableVertexAttribArray(p); gl.vertexAttribPointer(p, 3, gl.FLOAT, false, 0, 0);
  gl.uniform4fv(gl.getUniformLocation(program, "color"), color);
  gl.uniformMatrix4fv(gl.getUniformLocation(program, "model"), false, model);
  gl.drawArrays(mode, 0, vertices.length / 3);
}

function drawPosePanel(view, pose) {
  gl.useProgram(hudProgram);
  // Closer controllers remain visible in front of the head-attached panel.
  gl.enable(gl.DEPTH_TEST);
  gl.enable(gl.BLEND);
  gl.blendFuncSeparate(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
  gl.depthMask(false);
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.bufferData(gl.ARRAY_BUFFER, hudQuad, gl.DYNAMIC_DRAW);
  const p = gl.getAttribLocation(hudProgram, "p");
  gl.enableVertexAttribArray(p); gl.vertexAttribPointer(p, 2, gl.FLOAT, false, 0, 0);
  gl.uniformMatrix4fv(gl.getUniformLocation(hudProgram, "projection"), false, view.projectionMatrix);
  gl.uniformMatrix4fv(gl.getUniformLocation(hudProgram, "view"), false, view.transform.inverse.matrix);
  gl.uniformMatrix4fv(gl.getUniformLocation(hudProgram, "model"), false, pose.transform.matrix);
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, hudTexture);
  gl.uniform1i(gl.getUniformLocation(hudProgram, "panel"), 0);
  gl.drawArrays(gl.TRIANGLES, 0, 6);
  gl.depthMask(true);
  gl.disable(gl.BLEND);
}

function render(pose, packet) {
  const layer = session.renderState.baseLayer;
  gl.bindFramebuffer(gl.FRAMEBUFFER, layer.framebuffer);
  // Transparent background permits the headset's own passthrough in immersive-ar.
  gl.clearColor(0.02, 0.027, 0.039, session.environmentBlendMode === "opaque" ? 1 : 0);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  if (!pose) return;
  for (const view of pose.views) {
    gl.enable(gl.DEPTH_TEST);
    gl.useProgram(program);
    const viewport = layer.getViewport(view);
    gl.viewport(viewport.x, viewport.y, viewport.width, viewport.height);
    gl.uniformMatrix4fv(gl.getUniformLocation(program, "projection"), false, view.projectionMatrix);
    gl.uniformMatrix4fv(gl.getUniformLocation(program, "view"), false, view.transform.inverse.matrix);
    for (const side of ["left", "right"]) {
      const c = packet[`${side}Controller`];
      if (!c) continue;
      const model = controllerMatrix(c);
      controllerModels.draw(side, c, program);
      for (let axis = 0; axis < 3; axis++) {
        draw(arrows[axis], axisColors[axis], gl.TRIANGLES, model);
      }
      // A small grip marker leaves the axis colors unambiguous.
      draw(gripRing,
        c.gripActive ? [1, 0.65, 0.1, 1] : [1, 1, 1, 1], gl.LINES, model);
    }
    drawPosePanel(view, pose);
  }
}

function onFrame(time, frame) {
  if (!session || frame.session !== session) return;
  session.requestAnimationFrame(onFrame);
  frames++;
  inputStatus.left = inputStatus.right = "not_seen";
  const packet = empty();
  const displayPacket = empty();
  try {
    const pose = session.visibilityState === "visible" ? frame.getViewerPose(referenceSpace) : null;
    if (session.visibilityState === "visible") {
      for (const source of session.inputSources) {
        if (!["left", "right"].includes(source.handedness)) continue;
        let value = controller(frame, source);
        displayPacket[`${source.handedness}Controller`] = value
          ? { ...value, profiles: source.profiles } : null;
        // Keep tracking feedback visible while recentering pauses robot commands.
        if (time < pauseUntil) continue;
        if (needsRelease.has(source.handedness)) {
          if (value && !value.gripActive) needsRelease.delete(source.handedness);
          else { value = null; inputStatus[source.handedness] = "release_grip_to_resume"; }
        }
        packet[`${source.handedness}Controller`] = value;
      }
    }
    // Send the latest pose on every XR frame, as HEI does. A 60 Hz elapsed-time
    // gate drops every second frame on 72/90 Hz headsets (to 36/45 Hz).
    send(packet);
    updatePoseDisplay(pose, displayPacket, time);
    render(pose, displayPacket);
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
    xrEndCause = "";
    diagnostic("xr_session_created");
    session.addEventListener("end", () => {
      xrEndCause ||= "头显或浏览器结束 XR 会话（浏览器未提供具体原因）";
      send(empty()); session = null; updateButtons();
      document.getElementById("poseStatus").textContent = "未进入 XR，等待头显和手柄位置…";
      if (gl) {
        controllerModels?.dispose();
        gl.deleteProgram(program); gl.deleteProgram(hudProgram);
        gl.deleteTexture(hudTexture); gl.deleteBuffer(buffer);
      }
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
    lastDiagnostic = -Infinity; frames = 0; pauseUntil = -Infinity; needsRelease.clear();
    diagnostic("waiting_for_first_xr_frame");
    status.textContent = "XR 已启动。先松开侧面握把，再握住并移动手柄。";
    session.requestAnimationFrame(onFrame);
  } catch (error) {
    const message = `${error.name}: ${error.message}`;
    status.textContent = `无法进入 XR：${message}`;
    diagnostic("xr_error", message);
    xrEndCause = `初始化失败：${message}`;
    await session?.end().catch(() => {});
  } finally {
    starting = false; updateButtons();
  }
}

// HEI's single Start VR button requests AR (A-Frame enterVR(true)).
// Choose the supported mode before the click to preserve user activation.
enter.onclick = () => {
  if (session && !starting) {
    xrEndCause = "点击退出控制";
    return session.end();
  }
  return start(arSupported ? "immersive-ar" : "immersive-vr");
};
window.addEventListener("pagehide", () => {
  closing = true; clearTimeout(reconnectTimer); send(empty()); socket?.close();
});
window.addEventListener("error", event => diagnostic("javascript_error", event.message));
window.addEventListener("unhandledrejection", event => diagnostic("javascript_error", String(event.reason)));

(async () => {
  if (!navigator.xr) { status.textContent = "未检测到 WebXR。请在 Quest 头显浏览器通过 HTTPS 打开此页面。"; return; }
  // A rejected optional AR probe must not prevent VR or the network connection.
  [vrSupported, arSupported] = await Promise.all(["immersive-vr", "immersive-ar"].map(async mode => {
    try { return await navigator.xr.isSessionSupported(mode); }
    catch { return false; }
  }));
  connect();
})().catch(error => { status.textContent = error.message; });

#!/usr/bin/env python3
"""Front-camera + ego-motion eval recorder.

A trimmed variation of record_cameras.py: records ONLY the front camera
(front.mp4) and the iPhone ARKit ego-motion stream (ego.jsonl). No
front-left/right, back cams, GPS, PS5/cart state, segmentation maps, or
RealSense.

Run folders are grouped under a single parent directory instead of being
dropped loose in $HOME:

    ~/pi-eval-and-finetuning-data/CADDY-FRONT-EVAL-<timestamp>/
        front.mp4
        ego.jsonl
        timestamps.json

Override the parent with --data-root.

Usage:
    python record_front_eval.py
    python record_front_eval.py --front 16        # explicit /dev/video index
    python record_front_eval.py --data-root /mnt/ssd/eval

Open http://<jetson-ip>:8080 in a browser, hit Record.

The Camera and EgoRecorder implementations are imported from
record_cameras.py so this file stays a thin variation that tracks any
fixes made to the shared capture/ego logic.
"""
import argparse
import datetime as dt
import json
import threading
import time
import webbrowser
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template_string

import record_cameras
from record_cameras import (
    Camera,
    EgoRecorder,
    EGO_STATE_FILE,
    discover_video_devices,
)

# Parent folder that collects every run, instead of dropping timestamped
# folders straight into $HOME. Overridable with --data-root.
DEFAULT_DATA_ROOT = Path.home() / "pi-eval-and-finetuning-data"

# Bump the front camera brighter. The control is a v4l2 int in
# [-64, 64] (default 0); +32 lifts the image while leaving headroom.
# auto_exposure stays on the cam firmware (Aperture Priority). Tunable
# via --brightness. Applied through record_cameras.CAMERA_CONTROLS so it
# is re-issued on every _open() — including the stall-reopen path.
DEFAULT_BRIGHTNESS = 32

app = Flask(__name__)
camera: Camera | None = None
ego: EgoRecorder | None = None
data_root: Path = DEFAULT_DATA_ROOT
state = {"recording": False, "folder": None, "started_at": None}
state_lock = threading.Lock()


PAGE = """
<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<title>Caddy Front Eval Recorder</title>
<style>
  body { background:#111; color:#eee; font-family: system-ui, sans-serif; margin:0; padding:16px; }
  h1 { margin: 0 0 12px 0; font-size: 20px; }
  .layout { display:grid; grid-template-columns: 1fr 260px; gap:16px; align-items:start; }
  .grid { background:#000; border:1px solid #333; }
  .grid img { width:100%; display:block; }
  .controls { margin-top:14px; display:flex; align-items:center; gap:12px; }
  button { font-size:18px; padding:10px 24px; border:0; border-radius:4px; cursor:pointer; }
  .rec { background:#c0392b; color:#fff; }
  .stop { background:#444; color:#fff; }
  .status { font-family: monospace; }
  .dot { display:inline-block; width:10px; height:10px; border-radius:50%; background:#666; margin-right:6px; vertical-align:middle; }
  .dot.on { background:#e74c3c; animation: pulse 1s infinite; }
  @keyframes pulse { 50% { opacity:0.3; } }

  .ego-panel {
    background: #1a1c20; border: 1px solid #2a2d33; border-radius: 12px;
    padding: 10px 12px 8px 12px; font-size: 10px; color: #aab; letter-spacing: 0.05em;
  }
  .ego-title {
    display:flex; align-items:center; justify-content:space-between;
    font-weight:600; font-size:9.5px; text-transform:uppercase; color:#7e8896; margin-bottom:6px;
  }
  .ego-conn-dot { width:6px; height:6px; border-radius:50%; background:#d84a4a; transition:background-color 180ms ease, box-shadow 180ms ease; }
  .ego-conn-dot.connected { background:#7ed488; box-shadow:0 0 0 1.5px rgba(126,212,136,0.25); }
  .ego-canvas { display:block; width:100%; height:180px; background:#0e1013; border-radius:10px; }
  .ego-readout { display:grid; grid-template-columns:repeat(2, 1fr); gap:2px 10px; margin-top:6px; font-variant-numeric:tabular-nums; font-size:11px; }
  .ego-readout > div { display:flex; align-items:baseline; gap:4px; }
  .ego-k { color:#7e8896; font-weight:500; min-width:32px; }
  .ego-v { color:#e6e9ef; font-weight:500; margin-left:auto; }
  .ego-u { color:#7e8896; font-size:9.5px; }
</style>
</head>
<body>
  <h1>Caddy Front Eval Recorder</h1>
  <div class="layout">
    <div>
      <div class="grid"><img src="/grid_feed" /></div>
      <div class="controls">
        <button id="btn" class="rec" onclick="toggle()">Record</button>
        <span class="status"><span id="dot" class="dot"></span><span id="status">idle</span></span>
      </div>
    </div>
    <div class="ego-panel" id="ego-panel">
      <div class="ego-title">
        <span>EGO MOTION</span>
        <span class="ego-conn-dot" id="ego-conn-dot"></span>
      </div>
      <canvas class="ego-canvas" id="ego-canvas" width="220" height="180"></canvas>
      <div class="ego-readout">
        <div><span class="ego-k">x</span><span class="ego-v" id="ego-x">—</span><span class="ego-u">m</span></div>
        <div><span class="ego-k">y</span><span class="ego-v" id="ego-y">—</span><span class="ego-u">m</span></div>
        <div><span class="ego-k">z</span><span class="ego-v" id="ego-z">—</span><span class="ego-u">m</span></div>
        <div><span class="ego-k">speed</span><span class="ego-v" id="ego-speed">—</span><span class="ego-u">m/s</span></div>
        <div><span class="ego-k">yaw</span><span class="ego-v" id="ego-yaw">—</span><span class="ego-u">°</span></div>
        <div><span class="ego-k">curv</span><span class="ego-v" id="ego-curv">—</span><span class="ego-u">/m</span></div>
      </div>
    </div>
  </div>
<script>
async function refresh() {
  const r = await fetch('/status'); const j = await r.json();
  const btn = document.getElementById('btn');
  const dot = document.getElementById('dot');
  const status = document.getElementById('status');
  if (j.recording) {
    btn.textContent = 'Stop'; btn.className = 'stop';
    dot.classList.add('on');
    status.textContent = 'REC  ' + j.folder + '   elapsed ' + j.elapsed + 's';
  } else {
    btn.textContent = 'Record'; btn.className = 'rec';
    dot.classList.remove('on');
    status.textContent = j.last_folder ? ('saved: ' + j.last_folder) : 'idle';
  }
}
async function toggle() {
  const r = await fetch('/status'); const j = await r.json();
  await fetch(j.recording ? '/stop' : '/start', { method:'POST' });
  refresh();
}
setInterval(refresh, 500); refresh();

// ===== Ego-motion 3D mini-cube (ported from record_cameras.py) =====
const egoCanvas = document.getElementById('ego-canvas');
const egoCtx = egoCanvas.getContext('2d');
const egoConnDot = document.getElementById('ego-conn-dot');
const egoX = document.getElementById('ego-x');
const egoY = document.getElementById('ego-y');
const egoZ = document.getElementById('ego-z');
const egoSpeed = document.getElementById('ego-speed');
const egoYaw = document.getElementById('ego-yaw');
const egoCurv = document.getElementById('ego-curv');

(function fitCanvas() {
  const dpr = window.devicePixelRatio || 1;
  const cssW = egoCanvas.clientWidth || 220;
  const cssH = egoCanvas.clientHeight || 180;
  egoCanvas.width = cssW * dpr; egoCanvas.height = cssH * dpr;
  egoCtx.scale(dpr, dpr);
})();
const EGO_W = egoCanvas.clientWidth || 220;
const EGO_H = egoCanvas.clientHeight || 180;
const egoTrail = [];
const EGO_TRAIL_SECONDS = 6.0;
const EGO_TRAIL_MAX = 80;
const EGO_HALF_MIN = 1.0;
let egoLastTs = null;
const EGO_CAM_AZ = 32 * Math.PI / 180;
const EGO_CAM_EL = 18 * Math.PI / 180;
const EGO_COS_AZ = Math.cos(EGO_CAM_AZ), EGO_SIN_AZ = Math.sin(EGO_CAM_AZ);
const EGO_COS_EL = Math.cos(EGO_CAM_EL), EGO_SIN_EL = Math.sin(EGO_CAM_EL);

function isoProject(x, y, z, half) {
  const nx = x / half, ny = y / half, nz = -z / half;
  const x1 = nx * EGO_COS_AZ + nz * EGO_SIN_AZ;
  const y1 = ny;
  const z1 = -nx * EGO_SIN_AZ + nz * EGO_COS_AZ;
  const x2 = x1;
  const y2 = y1 * EGO_COS_EL - z1 * EGO_SIN_EL;
  const cx = EGO_W / 2, cy = EGO_H / 2 + 4;
  const scale = Math.min(EGO_W, EGO_H) * 0.34;
  return [cx + x2 * scale, cy - y2 * scale];
}
function drawEgoCube(half) {
  const C = [
    [-1,-1,-1],[+1,-1,-1],[+1,+1,-1],[-1,+1,-1],
    [-1,-1,+1],[+1,-1,+1],[+1,+1,+1],[-1,+1,+1],
  ].map(([a,b,c]) => isoProject(a*half,b*half,c*half,half));
  const E = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];
  egoCtx.lineWidth = 1;
  egoCtx.strokeStyle = 'rgba(150, 165, 195, 0.55)';
  egoCtx.beginPath();
  for (const [a,b] of E) { egoCtx.moveTo(C[a][0],C[a][1]); egoCtx.lineTo(C[b][0],C[b][1]); }
  egoCtx.stroke();
  egoCtx.strokeStyle = 'rgba(150, 165, 195, 0.18)';
  egoCtx.beginPath();
  const N = 4;
  for (let i = 1; i < N; i++) {
    const t = -half + (2*half)*(i/N);
    let p1 = isoProject(t,-half,-half,half), p2 = isoProject(t,-half,+half,half);
    egoCtx.moveTo(p1[0],p1[1]); egoCtx.lineTo(p2[0],p2[1]);
    p1 = isoProject(-half,-half,t,half); p2 = isoProject(+half,-half,t,half);
    egoCtx.moveTo(p1[0],p1[1]); egoCtx.lineTo(p2[0],p2[1]);
  }
  egoCtx.stroke();
  const o = isoProject(0,0,0,half);
  egoCtx.fillStyle = 'rgba(150, 165, 195, 0.45)';
  egoCtx.beginPath(); egoCtx.arc(o[0],o[1],1.6,0,Math.PI*2); egoCtx.fill();
}
function renderEgoScene() {
  egoCtx.clearRect(0, 0, EGO_W, EGO_H);
  const last = egoTrail.length > 0 ? egoTrail[egoTrail.length-1] : null;
  const cx0 = last ? last.x : 0, cy0 = last ? last.y : 0, cz0 = last ? last.z : 0;
  let half = EGO_HALF_MIN;
  for (const p of egoTrail) {
    half = Math.max(half, Math.abs(p.x-cx0), Math.abs(p.y-cy0), Math.abs(p.z-cz0));
  }
  half *= 1.15;
  drawEgoCube(half);
  if (egoTrail.length >= 2) {
    egoCtx.lineWidth = 1.6;
    egoCtx.strokeStyle = 'rgba(232, 96, 96, 0.9)';
    egoCtx.beginPath();
    const p0 = isoProject(egoTrail[0].x-cx0, egoTrail[0].y-cy0, egoTrail[0].z-cz0, half);
    egoCtx.moveTo(p0[0], p0[1]);
    for (let i = 1; i < egoTrail.length; i++) {
      const p = isoProject(egoTrail[i].x-cx0, egoTrail[i].y-cy0, egoTrail[i].z-cz0, half);
      egoCtx.lineTo(p[0], p[1]);
    }
    egoCtx.stroke();
  }
  const p = isoProject(0, 0, 0, half);
  egoCtx.fillStyle = '#f0f3f8';
  egoCtx.beginPath(); egoCtx.arc(p[0], p[1], 3.4, 0, Math.PI*2); egoCtx.fill();
}
async function pollEgo() {
  try {
    const r = await fetch('/ego'); const s = await r.json();
    const connected = !!s.connected;
    egoConnDot.classList.toggle('connected', connected);
    const sample = s.sample;
    if (sample && typeof sample.t_s === 'number' && sample.t_s !== egoLastTs) {
      egoLastTs = sample.t_s;
      egoTrail.push({ x: +sample.x_m||0, y: +sample.y_m||0, z: +sample.z_m||0, t_s: sample.t_s });
      const cutoff = sample.t_s - EGO_TRAIL_SECONDS;
      while (egoTrail.length > 0 && egoTrail[0].t_s < cutoff) egoTrail.shift();
      while (egoTrail.length > EGO_TRAIL_MAX) egoTrail.shift();
    }
    if (sample) {
      egoX.textContent = (+sample.x_m||0).toFixed(2);
      egoY.textContent = (+sample.y_m||0).toFixed(2);
      egoZ.textContent = (+sample.z_m||0).toFixed(2);
      egoSpeed.textContent = (+sample.speed_mps||0).toFixed(2);
      egoYaw.textContent = ((+sample.yaw_rad||0) * 180 / Math.PI).toFixed(1);
      egoCurv.textContent = (+sample.curvature_inv_m||0).toFixed(3);
    } else if (!connected) {
      egoX.textContent = egoY.textContent = egoZ.textContent = '—';
      egoSpeed.textContent = egoYaw.textContent = egoCurv.textContent = '—';
    }
    renderEgoScene();
  } catch (e) { /* keep polling */ }
}
setInterval(pollEgo, 100); pollEgo();
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(PAGE)


def _front_tile() -> np.ndarray:
    if camera is None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, "NO DEVICE", (18, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 60, 255), 2, cv2.LINE_AA)
        return frame
    frame = camera.get_frame()
    if frame is None:
        frame = np.zeros((camera.height, camera.width, 3), dtype=np.uint8)
        cv2.putText(frame, "NO FRAME", (18, camera.height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 60, 255), 2, cv2.LINE_AA)
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(frame, f"front  /dev/video{camera.device}", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


@app.route("/grid_feed")
def grid_feed():
    def gen():
        while True:
            tile = _front_tile()
            ok, buf = cv2.imencode(".jpg", tile, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if ok:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                       + buf.tobytes() + b"\r\n")
            time.sleep(1 / 15)

    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/status")
def status():
    with state_lock:
        elapsed = (
            int(time.time() - state["started_at"]) if state["started_at"] else 0
        )
        return jsonify(
            recording=state["recording"],
            folder=state["folder"],
            elapsed=elapsed,
            last_folder=state.get("last_folder"),
        )


@app.route("/ego")
def ego_state():
    return jsonify(ego.latest() if ego is not None else {"connected": False})


@app.route("/start", methods=["POST"])
def start():
    with state_lock:
        if state["recording"]:
            return jsonify(ok=True)
        ts = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        folder = data_root / f"CADDY-FRONT-EVAL-{ts}"
        folder.mkdir(parents=True, exist_ok=True)
        if camera is not None:
            camera.start_recording(folder / "front.mp4")
        if ego is not None:
            ego.start_recording(folder / "ego.jsonl")
        # Wall-clock start time captured AFTER the writers/log threads are
        # armed so it bounds the actual recording window from below.
        start_t = time.time()
        with open(folder / "timestamps.json", "w") as f:
            json.dump({
                "start_ms": int(round(start_t * 1000)),
                "start_iso": dt.datetime.fromtimestamp(start_t).isoformat(timespec="milliseconds"),
            }, f, indent=2)
        state["recording"] = True
        state["folder"] = str(folder)
        state["started_at"] = start_t
        print(f"[REC] started -> {folder}")
        return jsonify(ok=True, folder=str(folder))


@app.route("/stop", methods=["POST"])
def stop():
    with state_lock:
        if not state["recording"]:
            return jsonify(ok=True)
        # Capture stop BEFORE we tear down writers so it bounds the actual
        # recording window from above.
        stop_t = time.time()
        if camera is not None:
            camera.stop_recording()
        if ego is not None:
            ego.stop_recording()
        folder = state["folder"]
        start_t = state["started_at"]
        if folder is not None:
            with open(Path(folder) / "timestamps.json", "w") as f:
                json.dump({
                    "start_ms": int(round(start_t * 1000)),
                    "stop_ms":  int(round(stop_t  * 1000)),
                    "duration_ms": int(round((stop_t - start_t) * 1000)),
                    "start_iso": dt.datetime.fromtimestamp(start_t).isoformat(timespec="milliseconds"),
                    "stop_iso":  dt.datetime.fromtimestamp(stop_t ).isoformat(timespec="milliseconds"),
                }, f, indent=2)
        state["recording"] = False
        state["last_folder"] = folder
        state["folder"] = None
        state["started_at"] = None
        print(f"[REC] stopped -> {folder}")
        return jsonify(ok=True, folder=folder)


def parse_args():
    p = argparse.ArgumentParser()
    # -1 = auto-discover the front camera via USB path; pass an explicit
    # /dev/video index to override.
    p.add_argument("--front", type=int, default=-1,
                   help="Explicit /dev/videoN index for the front camera "
                        "(default: auto-discover via USB topology).")
    p.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT),
                   help="Parent folder that collects every run "
                        f"(default: {DEFAULT_DATA_ROOT}).")
    p.add_argument("--brightness", type=int, default=DEFAULT_BRIGHTNESS,
                   help="Front camera v4l2 brightness in [-64, 64] "
                        f"(default: {DEFAULT_BRIGHTNESS}).")
    # 640x480 matches production inference (alpamayo_infer / autoware_infer).
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--no-fourcc", action="store_true",
                   help="Do not force MJPG; let OpenCV/V4L2 negotiate format.")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--ego-host", default="127.0.0.1")
    p.add_argument("--ego-port", type=int, default=5005)
    p.add_argument("--no-browser", action="store_true",
                   help="Don't auto-open the UI in a browser at startup")
    return p.parse_args()


def main():
    global camera, ego, data_root
    args = parse_args()
    data_root = Path(args.data_root).expanduser()
    data_root.mkdir(parents=True, exist_ok=True)
    print(f"[REC] run folders -> {data_root}")

    if args.front >= 0:
        front_dev = args.front
    else:
        discovered = discover_video_devices()
        if discovered:
            print(f"[CAM] discovered: {discovered}")
        front_dev = discovered.get("front")

    if front_dev is None:
        print("[CAM] front camera not discovered and no --front override; "
              "running ego-only (no video).")
    else:
        # Register the brightness bump so Camera._open() re-applies it via
        # v4l2-ctl on open AND on every stall-triggered reopen.
        if args.brightness:
            record_cameras.CAMERA_CONTROLS["front"] = (
                ("brightness", args.brightness),
            )
        camera = Camera(
            "front", front_dev, args.width, args.height, args.fps,
            force_mjpg=not args.no_fourcc,
        )
        print(f"[CAM] front -> /dev/video{front_dev} (brightness={args.brightness})")

    ego = EgoRecorder(host=args.ego_host, port=args.ego_port)
    print(f"[EGO] reading {EGO_STATE_FILE} "
          f"(writer feeds it from {args.ego_host}:{args.ego_port})")

    # Auto-open the UI in the local browser. Connect to localhost regardless
    # of bind host so it works when --host is 0.0.0.0.
    if not args.no_browser:
        url = f"http://localhost:{args.port}"

        def _open():
            time.sleep(1.2)  # let Flask bind the socket first
            try:
                webbrowser.open(url)
            except Exception as e:
                print(f"[WARN] could not auto-open browser: {e!r}  ({url})")

        threading.Thread(target=_open, daemon=True).start()

    try:
        app.run(host=args.host, port=args.port, threaded=True, debug=False)
    finally:
        if camera is not None:
            camera.release()
        if ego is not None:
            ego.shutdown()


if __name__ == "__main__":
    main()

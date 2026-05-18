import json
import os
import signal
import subprocess
import time

from flask import Flask, Response, abort, jsonify, render_template, request

STATE_FILE = os.environ.get("CART_STATE_FILE", "/tmp/cart_state.json")
AUTOWARE_STATE_FILE = os.environ.get(
    "AUTOWARE_STATE_FILE", "/tmp/autoware_state.json"
)
EGO_STATE_FILE = os.environ.get("EGO_STATE_FILE", "/tmp/ego_state.json")
FRAMES_DIR = os.environ.get("CART_FRAMES_DIR", "/tmp/cart_frames")
TELEOP_CMD_FILE = "/tmp/teleop_cmd.json"
TELEOP_CMD_FRESH_S = 0.50
TELEOP_TUNNEL_URL = "https://caddy.ethandgoodhart.com"
STATE_FRESH_S = 1.0
AUTOWARE_FRESH_S = 0.5  # autoware_infer writes at ~15 Hz; >500 ms = stale
EGO_FRESH_S = 1.0       # ego_state_writer writes at 10 Hz; >1 s = writer died

# Must match scripts/autoware_infer.py::ALL_STREAM_SLUGS. A request for
# any other slug returns 404 — never stream arbitrary paths off the
# filesystem. Order: 4 raw cameras, then 4 model-output viz tiles, then
# any auxiliary streams. ``lanes_solo`` is consumed by scene.js (not as
# a UI tile) to project the predicted lanes into the 3D cart scene.
CAM_SLUGS = (
    "front_wide", "front_narrow", "left", "right",
    "lanes", "depth", "seg", "objects",
    "lanes_solo",
    # Alpamayo-only viz: top-down trajectory tile written by alpamayo_infer.py.
    "bev",
)
MJPEG_FPS = 15           # per-client frame rate
MJPEG_STALE_S = 2.0      # stop streaming if frame file hasn't updated

app = Flask(__name__)


def _load_json(path: str, fresh_s: float) -> tuple[dict, bool]:
    """Return (data, stale). Missing/corrupt file → empty dict + stale=True."""
    try:
        with open(path) as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}, True
    stale = time.time() - float(data.get("ts", 0)) > fresh_s
    return data, stale


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/state")
def state():
    data, stale = _load_json(STATE_FILE, STATE_FRESH_S)
    if not data:
        data = {
            "mph": 0, "gas": 0, "brake": 0,
            "controller_connected": False,
            "arduino_connected": False,
            "motor_connected": False,
        }
    if stale:
        # ps5_drive.py only writes this file while it's running, so stale
        # state means the driver exited — usually a controller drop. Treat
        # everything as disconnected so the UI dots turn gray.
        data["mph"] = 0
        data["stale"] = True
        data["controller_connected"] = False
        data["arduino_connected"] = False
        data["motor_connected"] = False

    # Merge Autoware state into a sub-key so the UI can render the active
    # camera indicator + predicted angle + viz tiles without racing two
    # fetches. Field names mirror autoware_infer.py's state JSON 1:1
    # (running/stale are server-derived; everything else is pass-through).
    auto, auto_stale = _load_json(AUTOWARE_STATE_FILE, AUTOWARE_FRESH_S)
    # ``model`` (default "autoware" for backwards compat with older state
    # files) tells the UI which brain wrote this state — drives the badge
    # text on each camera tile + the status pill verb.
    data["autoware"] = {
        "running": bool(auto) and not auto_stale,
        "inference": bool(auto.get("inference", False)),
        "viz": bool(auto.get("viz", False)),
        "viz_streams": list(auto.get("viz_streams", [])),
        "object_count": int(auto.get("object_count", 0)),
        "steer_deg": float(auto.get("steer_deg", 0.0)) if auto else 0.0,
        "steer_deg_raw": float(auto.get("steer_deg_raw", 0.0)) if auto else 0.0,
        "active_cam": auto.get("active_cam"),
        "fps": float(auto.get("fps", 0.0)) if auto else 0.0,
        "cams": auto.get("cams", []),
        "model": str(auto.get("model", "autoware")) if auto else "autoware",
        # Alpamayo-only fields (autoware doesn't write these — defaults
        # to 0 / [] so the UI can blindly read them either way).
        "target_speed_mph": float(auto.get("target_speed_mph", 0.0)) if auto else 0.0,
        "ego_speed_mph": float(auto.get("ego_speed_mph", 0.0)) if auto else 0.0,
        "ego_speed_ok": bool(auto.get("ego_speed_ok", False)) if auto else False,
        "steer_deg_base": float(auto.get("steer_deg_base", 0.0)) if auto else 0.0,
        "steer_gain": float(auto.get("steer_gain", 1.0)) if auto else 1.0,
        "steer_lookahead_m": float(auto.get("steer_lookahead_m", 0.0)) if auto else 0.0,
        "target_gas": float(auto.get("target_gas", 0.0)) if auto else 0.0,
        "target_brake": float(auto.get("target_brake", 0.0)) if auto else 0.0,
        "predicted_path": list(auto.get("predicted_path", [])),
        # Pass the alpamayo diagnostics block through verbatim so the UI
        # can show the per-prediction latency breakdown + live MB/s.
        # Empty dict for autoware (it never writes one), so the JS can
        # safely read .alpamayo?.latency_ms?.gpu etc.
        "alpamayo": dict(auto.get("alpamayo", {})) if auto else {},
        "segmentation": dict(auto.get("segmentation", {})) if auto else {},
        "stale": auto_stale,
    }

    # iPhone ego-motion publisher status. ego_state_writer.py mirrors the
    # EgoSensor's connected flag + latest sample at ~10 Hz; if the writer
    # dies or the iOS app stops streaming, the file goes stale and the UI
    # dot turns red.
    ego, ego_stale = _load_json(EGO_STATE_FILE, EGO_FRESH_S)
    data["ego_connected"] = bool(ego.get("connected", False)) and not ego_stale
    data["ego"] = {
        "connected": data["ego_connected"],
        "host": ego.get("host"),
        "sample": ego.get("sample"),
        "history_len": int(ego.get("history_len", 0)),
        "stale": ego_stale,
    }

    if data.get("teleop_active"):
        data["teleop_url"] = TELEOP_TUNNEL_URL + "/teleop"

    return jsonify(data)


def _frame_path(slug: str) -> str | None:
    if slug not in CAM_SLUGS:
        return None
    return os.path.join(FRAMES_DIR, f"{slug}.jpg")


@app.route("/cam/<slug>.jpg")
def cam_snapshot(slug: str):
    """Single most-recent frame as a plain JPEG. Useful for polling clients
    (e.g. `<img>` with ?t=timestamp). MJPEG endpoint below is preferred for
    continuous playback."""
    path = _frame_path(slug)
    if path is None:
        abort(404)
    try:
        with open(path, "rb") as f:
            buf = f.read()
    except FileNotFoundError:
        abort(404)
    return Response(buf, mimetype="image/jpeg", headers={
        "Cache-Control": "no-store, must-revalidate",
    })


def _mjpeg_stream(path: str):
    """Generator that yields multipart frames at MJPEG_FPS. Reads the file's
    mtime to skip re-sending the same JPEG; exits quietly if the file stops
    updating (autoware_infer died) so the browser reconnects cleanly.
    """
    boundary = b"--caddyframe"
    period = 1.0 / MJPEG_FPS
    last_mtime = 0.0
    last_yield = 0.0
    idle_start: float | None = None
    while True:
        try:
            mtime = os.path.getmtime(path)
        except FileNotFoundError:
            # No frames yet — back off and try again. autoware_infer might
            # still be loading models on startup.
            time.sleep(0.2)
            continue
        now = time.monotonic()
        if mtime != last_mtime and now - last_yield >= period:
            try:
                with open(path, "rb") as f:
                    buf = f.read()
            except FileNotFoundError:
                time.sleep(0.05)
                continue
            last_mtime = mtime
            last_yield = now
            idle_start = None
            yield (boundary + b"\r\n"
                   b"Content-Type: image/jpeg\r\n"
                   b"Content-Length: " + str(len(buf)).encode() + b"\r\n\r\n"
                   + buf + b"\r\n")
        else:
            # Detect the producer having died — don't hold the socket open
            # forever rebroadcasting the last frame.
            if idle_start is None:
                idle_start = now
            elif now - idle_start > MJPEG_STALE_S:
                return
            time.sleep(period / 2)


@app.route("/cam/<slug>.mjpg")
def cam_mjpeg(slug: str):
    path = _frame_path(slug)
    if path is None:
        abort(404)
    return Response(_mjpeg_stream(path),
                     mimetype="multipart/x-mixed-replace; boundary=caddyframe",
                     headers={"Cache-Control": "no-store, must-revalidate"})


@app.route("/teleop")
def teleop():
    return render_template("teleop.html")


@app.route("/teleop/command", methods=["POST"])
def teleop_command():
    data = request.get_json(silent=True)
    if not data:
        abort(400)
    cmd = {
        "steer": float(data.get("steer", 0.0)),
        "gas": float(data.get("gas", 0.0)),
        "brake": float(data.get("brake", 0.0)),
        "ts": time.time(),
    }
    tmp = TELEOP_CMD_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cmd, f)
    os.replace(tmp, TELEOP_CMD_FILE)
    return "", 204


@app.route("/quit", methods=["POST"])
def quit_app():
    subprocess.Popen(["pkill", "-f", "firefox"])
    return "", 204


if __name__ == "__main__":
    # threaded=True so multiple MJPEG clients (4 cams × N browsers) don't
    # serialize on the dev server. Still a dev server — put it behind nginx
    # for anything other than local use.
    app.run(host="127.0.0.1", port=5050, debug=False, threaded=True)

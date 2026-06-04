import json
import os
import signal
import subprocess
import glob
import threading
import time
from pathlib import Path

from flask import Flask, Response, abort, jsonify, render_template, request

ROOT_DIR = Path(__file__).resolve().parents[1]
STATE_FILE = os.environ.get("CART_STATE_FILE", "/tmp/cart_state.json")
AUTOWARE_STATE_FILE = os.environ.get(
    "AUTOWARE_STATE_FILE", "/tmp/autoware_state.json"
)
EGO_STATE_FILE = os.environ.get("EGO_STATE_FILE", "/tmp/ego_state.json")
FRAMES_DIR = os.environ.get("CART_FRAMES_DIR", "/tmp/cart_frames")
TELEOP_CMD_FILE = "/tmp/teleop_cmd.json"
NAV_ROUTE_FILE = os.environ.get("NAV_ROUTE_FILE", "/tmp/nav_route.json")
GPS_STATE_FILE = os.environ.get("GPS_STATE_FILE", "/tmp/gps_state.json")
TELEOP_CMD_FRESH_S = 0.50
TELEOP_TUNNEL_URL = "https://caddy.ethandgoodhart.com"
STATE_FRESH_S = 1.0
AUTOWARE_FRESH_S = 0.5  # autoware_infer writes at ~15 Hz; >500 ms = stale
EGO_FRESH_S = 1.0       # ego_state_writer writes at 10 Hz; >1 s = writer died
ADVANCED_SETTINGS_FILE = Path(os.environ.get(
    "ADVANCED_DRIVE_BY_SEGMENTATION_CONFIG",
    str(ROOT_DIR / "config" / "advanced_drive_by_segmentation.json"),
))
ADVANCED_SETTINGS_DEFAULTS = {
    "start_usbmuxd": True,
    "open_browser": True,
    "mph": 4.0,
    "actor_detector_hz": 4.0,
    "actor_detector_imgsz": 512,
    "with_clrnet": False,
    "env_brake_horizon_s": 2.3,
    "env_brake_hard_ttc_s": 1.0,
    "env_brake_near_stop_m": 1.8,
    "env_brake_corridor_half_m": 0.60,
    "env_brake_object_radius_m": 0.22,
    "env_brake_gas_cut_frac": 0.22,
    "env_brake_stop_frac": 0.65,
}
ADVANCED_SETTINGS_SCHEMA = {
    "start_usbmuxd": ("bool", None, None),
    "open_browser": ("bool", None, None),
    "with_clrnet": ("bool", None, None),
    "mph": ("float", 1.0, 8.0),
    "actor_detector_hz": ("float", 0.0, 10.0),
    "actor_detector_imgsz": ("int", 256, 1024),
    "env_brake_horizon_s": ("float", 0.5, 6.0),
    "env_brake_hard_ttc_s": ("float", 0.2, 3.0),
    "env_brake_near_stop_m": ("float", 0.5, 5.0),
    "env_brake_corridor_half_m": ("float", 0.25, 1.5),
    "env_brake_object_radius_m": ("float", 0.05, 0.75),
    "env_brake_gas_cut_frac": ("float", 0.0, 1.0),
    "env_brake_stop_frac": ("float", 0.0, 1.0),
}

# Must match scripts/autoware_infer.py::ALL_STREAM_SLUGS. A request for
# any other slug returns 404 — never stream arbitrary paths off the
# filesystem. Order: 4 raw cameras, then 4 model-output viz tiles, then
# any auxiliary streams. ``lanes_solo`` is consumed by scene.js (not as
# a UI tile) to project the predicted lanes into the 3D cart scene.
CAM_SLUGS = (
    "live_1", "live_2", "live_3", "live_4", "live_5", "live_6",
    "front", "front_left", "front_right",
    "front_wide", "front_narrow", "left", "right",
    "lanes", "depth", "seg", "objects",
    "lanes_solo",
    # Alpamayo-only viz: top-down trajectory tile written by alpamayo_infer.py.
    "bev",
)
MJPEG_FPS = 15           # per-client frame rate
MJPEG_STALE_S = 2.0      # stop streaming if frame file hasn't updated

app = Flask(__name__)


try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None


LIVE_CAMERA_COUNT = int(os.environ.get("LIVE_CAMERA_COUNT", "6"))
LIVE_CAMERA_WIDTH = int(os.environ.get("LIVE_CAMERA_WIDTH", "640"))
LIVE_CAMERA_HEIGHT = int(os.environ.get("LIVE_CAMERA_HEIGHT", "480"))
LIVE_CAMERA_FPS = float(os.environ.get("LIVE_CAMERA_FPS", "12"))
LIVE_CAMERA_REFRESH_S = 2.0


class _LiveCameraReader:
    def __init__(self, index: int, width: int, height: int, fps: float):
        self.index = index
        self.width = width
        self.height = height
        self.fps = fps
        self._lock = threading.Lock()
        self._jpeg: bytes | None = None
        self._last_frame_wall = 0.0
        self._running = True
        self._thread = threading.Thread(
            target=self._loop, name=f"LiveCameraReader-{index}", daemon=True
        )
        self._thread.start()

    def _open(self):
        if cv2 is None:
            return None
        cap = cv2.VideoCapture(self.index, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            return None
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _loop(self):
        period = 1.0 / max(self.fps, 1.0)
        while self._running:
            cap = self._open()
            if cap is None:
                time.sleep(0.75)
                continue
            try:
                while self._running:
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        break
                    ok, encoded = cv2.imencode(".jpg", frame)
                    if ok:
                        with self._lock:
                            self._jpeg = encoded.tobytes()
                            self._last_frame_wall = time.time()
                    time.sleep(period)
            finally:
                cap.release()
            time.sleep(0.25)

    def snapshot(self) -> bytes | None:
        with self._lock:
            return self._jpeg

    def status(self) -> dict:
        with self._lock:
            age = time.time() - self._last_frame_wall if self._last_frame_wall else None
            return {
                "index": self.index,
                "has_frame": self._jpeg is not None,
                "age_s": age,
            }


class _LiveCameraHub:
    def __init__(self):
        self._lock = threading.Lock()
        self._capture_lock = threading.Lock()
        self._readers: dict[int, _LiveCameraReader] = {}
        self._last_status: dict[int, dict] = {}
        self._indices: list[int] = []
        self._last_refresh = 0.0

    def _capture_indices(self) -> list[int]:
        env = os.environ.get("LIVE_CAMERA_INDICES")
        if env:
            return [int(x) for x in env.replace(",", " ").split() if x.strip()]

        indices: list[int] = []
        for path in sorted(glob.glob("/dev/video*"), key=lambda p: int(p[10:])):
            idx = int(path[10:])
            try:
                info = subprocess.run(
                    ["v4l2-ctl", "-d", path, "--all"],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=0.6,
                ).stdout
            except (OSError, subprocess.TimeoutExpired):
                continue
            # UVC cameras expose one capture node and one metadata node.
            caps = info.split("Device Caps", 1)[-1]
            if "Video Capture" in caps:
                indices.append(idx)
        return indices[:LIVE_CAMERA_COUNT]

    def refresh(self):
        if cv2 is None or np is None:
            return
        now = time.time()
        with self._lock:
            if now - self._last_refresh < LIVE_CAMERA_REFRESH_S:
                return
            self._last_refresh = now
        indices = self._capture_indices()
        with self._lock:
            self._indices = indices

    def snapshot(self, slot: int) -> bytes:
        self.refresh()
        with self._lock:
            idx = self._indices[slot - 1] if 0 <= slot - 1 < len(self._indices) else None
        jpeg = self._capture_once(slot, idx) if idx is not None else None
        if jpeg is not None:
            return jpeg
        label = f"live {slot}: no {'frame' if idx is not None else 'device'}"
        return self._placeholder(label)

    def status(self) -> list[dict]:
        self.refresh()
        out = []
        with self._lock:
            indices = list(self._indices)
            last_status = dict(self._last_status)
        for slot in range(1, LIVE_CAMERA_COUNT + 1):
            idx = indices[slot - 1] if slot - 1 < len(indices) else None
            if idx is None:
                out.append({"slot": slot, "index": None, "has_frame": False})
            else:
                st = dict(last_status.get(slot, {
                    "index": idx,
                    "has_frame": False,
                    "age_s": None,
                }))
                st["slot"] = slot
                out.append(st)
        return out

    def _capture_once(self, slot: int, idx: int) -> bytes | None:
        if cv2 is None:
            return None
        with self._capture_lock:
            cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
            try:
                if not cap.isOpened():
                    ok = False
                    frame = None
                else:
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, LIVE_CAMERA_WIDTH)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, LIVE_CAMERA_HEIGHT)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    ok = False
                    frame = None
                    for _ in range(3):
                        ok, frame = cap.read()
                        if ok and frame is not None:
                            break
                        time.sleep(0.03)
            finally:
                cap.release()

        now = time.time()
        with self._lock:
            self._last_status[slot] = {
                "index": idx,
                "has_frame": bool(ok and frame is not None),
                "age_s": 0.0 if ok and frame is not None else None,
                "checked_wall": now,
            }
        if not ok or frame is None:
            return None
        ok, encoded = cv2.imencode(".jpg", frame)
        return encoded.tobytes() if ok else None

    def _placeholder(self, text: str) -> bytes:
        if cv2 is None or np is None:
            return b""
        frame = np.zeros((LIVE_CAMERA_HEIGHT, LIVE_CAMERA_WIDTH, 3), dtype=np.uint8)
        cv2.putText(
            frame, text, (28, LIVE_CAMERA_HEIGHT // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (230, 230, 230), 2, cv2.LINE_AA,
        )
        ok, encoded = cv2.imencode(".jpg", frame)
        return encoded.tobytes() if ok else b""


_live_cameras = _LiveCameraHub()


def _load_json(path: str, fresh_s: float) -> tuple[dict, bool]:
    """Return (data, stale). Missing/corrupt file → empty dict + stale=True."""
    try:
        with open(path) as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}, True
    stale = time.time() - float(data.get("ts", 0)) > fresh_s
    return data, stale


def _atomic_json_write(path: str, payload: dict) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    os.replace(tmp, path)


def _coerce_setting(key: str, value):
    kind, lo, hi = ADVANCED_SETTINGS_SCHEMA[key]
    if kind == "bool":
        return bool(value)
    if kind == "int":
        out = int(round(float(value)))
        return max(int(lo), min(int(hi), out))
    out = float(value)
    return max(float(lo), min(float(hi), out))


def _load_advanced_settings() -> dict:
    try:
        with ADVANCED_SETTINGS_FILE.open() as f:
            loaded = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        loaded = {}
    out = dict(ADVANCED_SETTINGS_DEFAULTS)
    if isinstance(loaded, dict):
        for key in ADVANCED_SETTINGS_DEFAULTS:
            if key not in loaded:
                continue
            try:
                out[key] = _coerce_setting(key, loaded[key])
            except (TypeError, ValueError):
                pass
    return out


def _save_advanced_settings(settings: dict) -> None:
    ADVANCED_SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = ADVANCED_SETTINGS_FILE.with_suffix(ADVANCED_SETTINGS_FILE.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(settings, f, indent=2)
        f.write("\n")
    os.replace(tmp, ADVANCED_SETTINGS_FILE)


MPH_PER_MPS = 2.2369362920544


def _finite_float(value, default: float | None = None) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if out == out and out not in (float("inf"), float("-inf")) else default


def _iphone_ego_speed_mph(ego: dict, stale: bool) -> float | None:
    if stale or not ego.get("connected", False):
        return None
    sample = ego.get("sample")
    if not isinstance(sample, dict):
        return None
    speed_mps = _finite_float(sample.get("speed_mps"))
    if speed_mps is None:
        return None
    return abs(speed_mps) * MPH_PER_MPS


def _iphone_gps_speed_mph() -> float | None:
    reader = globals().get("_gps_reader")
    if reader is None:
        return None
    try:
        snap = reader.latest()
    except Exception:
        return None
    if not snap.get("connected", False) or snap.get("stale", True):
        return None
    fix = snap.get("fix")
    if not isinstance(fix, dict):
        return None
    speed_mps = _finite_float(fix.get("speed_mps"))
    # CoreLocation reports negative speed when speed is unavailable.
    if speed_mps is None or speed_mps < 0.0:
        return None
    return speed_mps * MPH_PER_MPS


def _override_display_speed_from_iphone(data: dict, ego: dict, ego_stale: bool) -> None:
    drive_estimate = _finite_float(data.get("mph"), 0.0) or 0.0
    data["drive_mph_estimate"] = drive_estimate

    ego_mph = _iphone_ego_speed_mph(ego, ego_stale)
    if ego_mph is not None:
        data["mph"] = ego_mph
        data["mph_source"] = "iphone_arkit"
        data["iphone_speed_mph"] = ego_mph
        data["iphone_speed_ok"] = True
        return

    gps_mph = _iphone_gps_speed_mph()
    if gps_mph is not None:
        data["mph"] = gps_mph
        data["mph_source"] = "iphone_gps"
        data["iphone_speed_mph"] = gps_mph
        data["iphone_speed_ok"] = True
        return

    data["mph_source"] = "drive_estimate"
    data["iphone_speed_mph"] = None
    data["iphone_speed_ok"] = False


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
    _override_display_speed_from_iphone(data, ego, ego_stale)

    if data.get("teleop_active"):
        data["teleop_url"] = TELEOP_TUNNEL_URL + "/teleop"

    return jsonify(data)


@app.route("/settings", methods=["GET", "POST"])
def advanced_settings():
    if request.method == "GET":
        return jsonify({
            "settings": _load_advanced_settings(),
            "path": str(ADVANCED_SETTINGS_FILE),
        })

    incoming = request.get_json(silent=True)
    if not isinstance(incoming, dict):
        abort(400)
    if "settings" in incoming and isinstance(incoming["settings"], dict):
        incoming = incoming["settings"]

    current = _load_advanced_settings()
    for key in ADVANCED_SETTINGS_DEFAULTS:
        if key not in incoming:
            continue
        try:
            current[key] = _coerce_setting(key, incoming[key])
        except (TypeError, ValueError):
            abort(400)
    _save_advanced_settings(current)
    return jsonify({
        "settings": current,
        "path": str(ADVANCED_SETTINGS_FILE),
    })


def _frame_path(slug: str) -> str | None:
    if slug not in CAM_SLUGS:
        return None
    return os.path.join(FRAMES_DIR, f"{slug}.jpg")


@app.route("/cam/<slug>.jpg")
def cam_snapshot(slug: str):
    """Single most-recent frame as a plain JPEG. Useful for polling clients
    (e.g. `<img>` with ?t=timestamp). MJPEG endpoint below is preferred for
    continuous playback."""
    if slug.startswith("live_"):
        try:
            slot = int(slug.removeprefix("live_"))
        except ValueError:
            abort(404)
        if slot < 1 or slot > LIVE_CAMERA_COUNT:
            abort(404)
        buf = _live_cameras.snapshot(slot)
        if not buf:
            abort(503)
        return Response(buf, mimetype="image/jpeg", headers={
            "Cache-Control": "no-store, must-revalidate",
        })

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


@app.route("/live_cameras")
def live_cameras():
    return jsonify({"cameras": _live_cameras.status()})


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


# Mapbox public token + style. pk.* tokens are client-side by design (shipped
# to the browser), but GitHub push-protection flags them, so the literal is not
# committed. Provide it via the MAPBOX_TOKEN env var, or drop it in the
# gitignored web/.mapbox_token file. Swap MAPBOX_STYLE to a custom Studio style
# (mapbox://styles/<user>/<id> → here as "<user>/<id>") for the custom map.
def _load_mapbox_token() -> str:
    env = os.environ.get("MAPBOX_TOKEN")
    if env:
        return env
    try:
        with open(os.path.join(os.path.dirname(__file__), ".mapbox_token")) as f:
            return f.read().strip()
    except OSError:
        return ""


MAPBOX_TOKEN = _load_mapbox_token()
MAPBOX_STYLE = os.environ.get("MAPBOX_STYLE", "mapbox/satellite-streets-v12")


@app.route("/mbtile/<int:z>/<int:x>/<int:y>")
def mapbox_tile(z: int, x: int, y: int):
    """Proxy Mapbox raster tiles through the Jetson. The kiosk browser can't
    reach api.mapbox.com directly (same outbound restriction that forced the
    /route proxy), so we relay tiles server-side and cache them briefly.
    """
    import urllib.request
    import urllib.error
    url = (
        f"https://api.mapbox.com/styles/v1/{MAPBOX_STYLE}/tiles/512/"
        f"{z}/{x}/{y}@2x?access_token={MAPBOX_TOKEN}"
    )
    try:
        with urllib.request.urlopen(url, timeout=5.0) as resp:
            body = resp.read()
            ctype = resp.headers.get("Content-Type", "image/png")
    except (urllib.error.URLError, TimeoutError):
        abort(502)
    return Response(body, mimetype=ctype, headers={
        "Cache-Control": "public, max-age=86400",
    })


@app.route("/map")
def campus_map():
    # Disable caching so iterative dev (and kiosk reload) always picks up
    # the latest template + JS; the file is tiny so caching gains nothing.
    resp = Response(
        render_template(
            "map.html",
            mapbox_token=MAPBOX_TOKEN,
            mapbox_style=MAPBOX_STYLE,
        ),
        mimetype="text/html",
    )
    resp.headers["Cache-Control"] = "no-store, must-revalidate"
    return resp


# iPhone CoreLocation publisher (Georg-Stanford-GC-iOS-Sensor GPSServer)
# exposes JSONL fixes on a sibling TCP port to the ego-motion stream.
# Wire format mirrors scripts/record_cameras.py::GpsRecorder.
GPS_HOST = os.environ.get("GPS_HOST", "127.0.0.1")
GPS_PORT = int(os.environ.get("GPS_PORT", "5006"))
GPS_FRESH_S = 5.0  # fixes arrive 1-5 Hz depending on phone state


class _GpsReader:
    """Background thread that holds the latest iPhone GPS fix.

    Same protocol as scripts/record_cameras.py::GpsRecorder but read-only —
    we never write the jsonl log here. Auto-reconnects on socket drops.
    """

    def __init__(self, host: str, port: int):
        import socket as _socket
        import threading as _threading
        self._socket = _socket
        self.host = host
        self.port = port
        self._lock = _threading.Lock()
        self._latest: dict | None = None
        self._latest_wall: float = 0.0
        self._tcp_connected = False
        self._stop = _threading.Event()
        t = _threading.Thread(target=self._loop, name="GpsReader", daemon=True)
        t.start()

    def _loop(self):
        while not self._stop.is_set():
            try:
                with self._socket.create_connection(
                    (self.host, self.port), timeout=2.0
                ) as s:
                    s.settimeout(1.0)
                    self._tcp_connected = True
                    buf = b""
                    while not self._stop.is_set():
                        try:
                            chunk = s.recv(4096)
                        except self._socket.timeout:
                            continue
                        if not chunk:
                            break
                        buf += chunk
                        while b"\n" in buf:
                            line, buf = buf.split(b"\n", 1)
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                fix = json.loads(line)
                            except json.JSONDecodeError:
                                continue
                            now = time.time()
                            with self._lock:
                                self._latest = fix
                                self._latest_wall = now
                            try:
                                _atomic_json_write(GPS_STATE_FILE, {
                                    "ts": now,
                                    "connected": True,
                                    "host": f"{self.host}:{self.port}",
                                    "fix": fix,
                                })
                            except OSError:
                                pass
            except (ConnectionRefusedError, OSError):
                pass
            finally:
                self._tcp_connected = False
            if not self._stop.is_set():
                time.sleep(0.5)

    def latest(self) -> dict:
        with self._lock:
            fix = dict(self._latest) if self._latest is not None else None
            wall = self._latest_wall
        stale = (time.time() - wall) > GPS_FRESH_S if fix is not None else True
        return {
            "tcp_connected": self._tcp_connected,
            "connected": self._tcp_connected and not stale and fix is not None,
            "stale": stale,
            "host": f"{self.host}:{self.port}",
            "fix": fix,
        }


_gps_reader = _GpsReader(GPS_HOST, GPS_PORT)


@app.route("/route")
def route():
    """Proxy to the public OSRM demo router. Browser-side fetches sometimes
    fail (CORS, captive portal, kiosk network policy), so we relay from the
    Jetson which already has outbound HTTPS.
    """
    try:
        points_arg = request.args.get("points")
        if points_arg:
            points: list[tuple[float, float]] = []
            for part in points_arg.split(";"):
                lat_s, lon_s = part.split(",", 1)
                lat = float(lat_s)
                lon = float(lon_s)
                if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
                    abort(400)
                points.append((lat, lon))
            if len(points) < 2:
                abort(400)
        else:
            from_lat = float(request.args["from_lat"])
            from_lon = float(request.args["from_lon"])
            to_lat = float(request.args["to_lat"])
            to_lon = float(request.args["to_lon"])
            points = [(from_lat, from_lon), (to_lat, to_lon)]
    except (KeyError, ValueError):
        abort(400)
    profile = request.args.get("profile", "foot")
    if profile not in ("foot", "driving", "bike"):
        abort(400)

    import urllib.request
    import urllib.error
    # overview=simplified cuts geometry size ~5-10x for long routes — the
    # demo router's payload was the slow part on Stanford-scale distances.
    coords = ";".join(f"{lon},{lat}" for lat, lon in points)
    url = (
        f"https://router.project-osrm.org/route/v1/{profile}/{coords}"
        f"?overview=simplified&geometries=geojson"
    )
    try:
        with urllib.request.urlopen(url, timeout=5.0) as resp:
            body = resp.read()
            return Response(body, mimetype="application/json")
    except (urllib.error.URLError, TimeoutError) as e:
        return jsonify({"code": "ProxyError", "message": repr(e)}), 502


def _validated_route_payload(data: dict) -> dict:
    coords = data.get("geometry") or data.get("coords")
    if not isinstance(coords, list) or len(coords) < 2:
        raise ValueError("route geometry must contain at least two [lat, lon] points")
    out_coords: list[list[float]] = []
    for p in coords:
        if not isinstance(p, (list, tuple)) or len(p) < 2:
            raise ValueError("route point must be [lat, lon]")
        lat = float(p[0])
        lon = float(p[1])
        if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
            raise ValueError("route point outside lat/lon bounds")
        out_coords.append([lat, lon])
    destination = data.get("destination") or out_coords[-1]
    if not isinstance(destination, (list, tuple)) or len(destination) < 2:
        destination = out_coords[-1]
    start = data.get("start") or out_coords[0]
    if not isinstance(start, (list, tuple)) or len(start) < 2:
        start = out_coords[0]
    waypoints = data.get("waypoints") or data.get("shape_points") or []
    out_waypoints: list[list[float]] = []
    if isinstance(waypoints, list):
        for p in waypoints:
            if not isinstance(p, (list, tuple)) or len(p) < 2:
                continue
            lat = float(p[0])
            lon = float(p[1])
            if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
                raise ValueError("waypoint outside lat/lon bounds")
            out_waypoints.append([lat, lon])
    start_lat = float(start[0])
    start_lon = float(start[1])
    dest_lat = float(destination[0])
    dest_lon = float(destination[1])
    if not (-90.0 <= start_lat <= 90.0 and -180.0 <= start_lon <= 180.0):
        raise ValueError("start point outside lat/lon bounds")
    if not (-90.0 <= dest_lat <= 90.0 and -180.0 <= dest_lon <= 180.0):
        raise ValueError("destination point outside lat/lon bounds")
    return {
        "ts": time.time(),
        "active": True,
        "source": "map",
        "geometry": out_coords,
        "start": [start_lat, start_lon],
        "waypoints": out_waypoints,
        "shape_points": out_waypoints,
        "destination": [dest_lat, dest_lon],
        "distance_m": float(data.get("distance_m", 0.0) or 0.0),
        "duration_s": float(data.get("duration_s", 0.0) or 0.0),
    }


@app.route("/nav_route", methods=["GET", "POST", "DELETE"])
def nav_route():
    if request.method == "GET":
        route, stale = _load_json(NAV_ROUTE_FILE, fresh_s=365 * 24 * 3600)
        if not route:
            return jsonify({"active": False, "stale": True})
        route["stale"] = stale
        return jsonify(route)
    if request.method == "DELETE":
        payload = {"ts": time.time(), "active": False, "geometry": []}
        try:
            _atomic_json_write(NAV_ROUTE_FILE, payload)
        except OSError as e:
            return jsonify({"ok": False, "error": repr(e)}), 500
        return jsonify({"ok": True, **payload})

    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        abort(400)
    try:
        payload = _validated_route_payload(data)
    except (TypeError, ValueError) as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    try:
        _atomic_json_write(NAV_ROUTE_FILE, payload)
    except OSError as e:
        return jsonify({"ok": False, "error": repr(e)}), 500
    return jsonify({"ok": True, **payload})


@app.route("/gps")
def gps():
    snap = _gps_reader.latest()
    fix = snap.get("fix") or {}
    has_fix = bool(fix) and "lat_deg" in fix and "lon_deg" in fix
    # Pull the next-turn announcement the segmentation brain computes against
    # the active route, so the map UI can render a turn banner from one poll.
    turn_dir = turn_dist_m = None
    turn_text = ""
    auto, auto_stale = _load_json(AUTOWARE_STATE_FILE, AUTOWARE_FRESH_S)
    if auto and not auto_stale:
        gr = (auto.get("segmentation") or {}).get("gps_route") or {}
        turn_dir = gr.get("turn_dir")
        turn_dist_m = gr.get("turn_dist_m")
        turn_text = gr.get("turn_text") or ""
    return jsonify({
        "connected": snap["connected"],
        "tcp_connected": snap["tcp_connected"],
        "stale": snap["stale"],
        "host": snap["host"],
        "has_fix": has_fix,
        "lat": fix.get("lat_deg"),
        "lon": fix.get("lon_deg"),
        "alt_m": fix.get("alt_m"),
        "h_acc_m": fix.get("h_acc_m"),
        "v_acc_m": fix.get("v_acc_m"),
        "speed_mps": fix.get("speed_mps"),
        "course_deg": fix.get("course_deg"),
        "t_unix": fix.get("t_unix"),
        "turn_dir": turn_dir,
        "turn_dist_m": turn_dist_m,
        "turn_text": turn_text,
    })


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


@app.route("/browser-log", methods=["POST"])
def browser_log():
    data = request.get_json(silent=True) or {}
    event = str(data.get("event", "browser"))
    loaded = data.get("loaded", {})
    changed = data.get("changed", {})
    visible = data.get("visible", [])
    print(
        f"[browser] {event} visible={visible} loaded={loaded} changed={changed}",
        flush=True,
    )
    return "", 204


if __name__ == "__main__":
    # threaded=True so multiple MJPEG clients (4 cams × N browsers) don't
    # serialize on the dev server. Still a dev server — put it behind nginx
    # for anything other than local use.
    app.run(host="127.0.0.1", port=5050, debug=False, threaded=True)

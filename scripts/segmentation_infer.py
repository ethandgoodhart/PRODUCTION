#!/usr/bin/env python3
"""
segmentation_infer.py — drive-by-segmentation live planner sidecar.

Uses the cloned https://github.com/ethandgoodhart/drive-by-segmentation
implementation for SegFormer semantic segmentation, BEV projection,
lane-aware trajectory planning, and steering estimation. Also runs the local
CLRerNet lane detector; any current lane detection with score >= 0.40
overrides the segmentation steering centerline for that frame. Publishes the
same /tmp contract as the other model sidecars:

  /tmp/cart_frames/{front_narrow,left,front_wide,right}.jpg  raw cameras
  /tmp/cart_frames/seg.jpg                                  segmentation overlay
  /tmp/cart_frames/bev.jpg                                  BEV trajectory
  /tmp/autoware_state.json                                  steer + pedal targets

start.sh launches this with --model segmentation. ps5_drive.py --autosteer
then reads steer_deg plus target_gas/target_brake and applies the usual
operator trigger overrides.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import seg_fast  # noqa: E402
import seg_occupancy  # noqa: E402


SEG_REPO_DEFAULT = Path("/home/caddy/drive-by-segmentation")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from limits import BRAKE_POT_MAX  # noqa: E402

E2E_CALIB_DEFAULT = PROJECT_ROOT / "calibration/cameras/sparsedrive_REAL_pvc_calibration.json"
FRAMES_DIR_DEFAULT = Path("/tmp/cart_frames")
STATE_FILE_DEFAULT = Path("/tmp/autoware_state.json")
SEGMENTATION_MAP_FILE_DEFAULT = Path(
    os.environ.get("SEGMENTATION_MAP_FILE", str(FRAMES_DIR_DEFAULT / "segmentation_map.json"))
)
EGO_STATE_FILE_DEFAULT = Path(os.environ.get("EGO_STATE_FILE", "/tmp/ego_state.json"))
GPS_STATE_FILE_DEFAULT = Path(os.environ.get("GPS_STATE_FILE", "/tmp/gps_state.json"))
NAV_ROUTE_FILE_DEFAULT = Path(os.environ.get("NAV_ROUTE_FILE", "/tmp/nav_route.json"))

# Segmentation currently drives from the center-front E2E-calibrated camera.
SLUGS = ("front",)
ACTIVE_SLUG = "front"
CAMERA_ORIENTATION_FIX = {"front_narrow": -1}

CAM_W, CAM_H = 640, 480
JPEG_QUALITY = 72
PUBLISH_HZ_DEFAULT = 15.0
INFER_HZ_DEFAULT = 0.0
MODEL_NAME = "segmentation"
FRONT_CAMERA_BRIGHTNESS = 32
STEERING_SIGN = -1.0
STEERING_COLUMN_RATIO = 15.0
STEERING_EMA = 0.35
CLRNET_CONF_THRESHOLD = 0.40
LOOKAHEAD_MIN_M = 2.5
LOOKAHEAD_MAX_M = 10.0
LOOKAHEAD_TIME_S = 0.9
# Scalar gain on the pure-pursuit road-wheel angle. PP is geometrically
# correct but tends to feel under-geared on a short-wheelbase low-speed
# kart; bump above 1.0 for more aggressive turn-in.
STEER_GAIN = 1.6

# Pure-pursuit on an arc-length-parameterized BEV centerline.
# WHEELBASE_M : kart rear-to-front axle. ~0.8 m placeholder; tune live.
# CENTERLINE_SMOOTH_WIN : window (in lane_local samples) for rolling-mean
#   smoothing of the centerline before arc-length resampling. Tames per-
#   frame planner jitter without lagging genuine path bends.
# LOOKAHEAD_POINT_EMA : EMA on the (xL, yL) lookahead point itself. Smooths
#   point jitter without adding lag to the steering response — preferred
#   over cranking STEERING_EMA, which would delay sharp turns.
WHEELBASE_M = 0.8
CENTERLINE_SMOOTH_WIN = 5
LOOKAHEAD_POINT_EMA = 0.3

# Limit pure-pursuit input to a near window — the far end of the planner's
# polyline is too noisy to inform the lookahead point reliably.
STEER_FIT_MIN_M = 1.0
STEER_FIT_MAX_M = 10.0

# Constant-speed target. ps5_drive consumes absolute pedal pot targets, so
# keep this conservative and publish the intended speed separately for UI/logs.
TARGET_MPH = 8.0
GAS_CONSTANT = 0.24
GAS_PER_MPH = GAS_CONSTANT / TARGET_MPH
BRAKE_CONSTANT = 0.0

# Closed-loop speed control: PI on (target_mph − ego_mph) when ARKit is
# fresh. Output is added to the open-loop feed-forward gas. Conservative
# gains — we'd rather creep up than overshoot into the cart's natural
# spring-back. Reset integrator when ARKit drops to avoid wind-up.
SPEED_KP = 0.018  # gas per mph error
SPEED_KI = 0.006  # gas per (mph.s) error
SPEED_I_CLAMP = 0.12  # safety cap on |integral term| (gas units)
# Trim clamp scales with the feed-forward but stays small at low speeds. This
# prevents the PI loop from adding a large gas jump while the cart is still
# accelerating toward a 2-4 mph target.
GAS_TRIM_FLOOR = 0.06
GAS_TRIM_SCALE = 0.75
# Rolling-resistance floor: below this gas the kart barely moves. The pot↔mph
# mapping (GAS_PER_MPH) is calibrated against the linear 5–8 mph regime; at
# low mph the kart needs more gas than the linear extrapolation suggests just
# to overcome rolling friction. Force feed-forward to at least this when the
# user actually wants to move.
ROLLING_GAS_FLOOR = 0.18

# Brake-on-overshoot: when ego > target AND gas has already hit 0 (cannot
# trim further), engage brake proportional to overshoot. Brake and gas are
# mutually exclusive — coast-then-brake, never both — to avoid pad wear
# under normal cruise. Only activates after BRAKE_DEADBAND_MPH of overshoot
# so small ARKit jitter doesn't pulse the brake.
BRAKE_KP = 0.035          # brake per mph overshoot
BRAKE_MAX = 0.25          # absolute brake-pot ceiling under autosteer
BRAKE_DEADBAND_MPH = 0.8  # mph overshoot before brake engages

# Launch ramp: start at LAUNCH_GAS_MIN and ease up to the commanded gas
# over LAUNCH_RAMP_S seconds so the kart doesn't jerk off the line. The
# ramp is applied to the *final* commanded gas (ff + PI trim), so the PI
# loop still corrects within the ramp ceiling instead of fighting it.
LAUNCH_RAMP_S = 3.75
LAUNCH_GAS_MIN = 0.02
SPEED_SETPOINT_RAMP_MPH_S = 0.65
GAS_RISE_RATE_PER_S = 0.055
GAS_FALL_RATE_PER_S = 0.16
# Below this gas the kart may not break static friction. Do not force this
# during the initial ramp because it can launch low-speed targets too hard;
# only ease toward it after the ramp if the phone says we are still parked.
STICTION_GAS_BREAK = 0.22
STICTION_EGO_MPH = 0.3
STICTION_STUCK_S = 1.0

# GPS route bias: route following is deliberately a bias, not a takeover. The
# BEV segmentation centerline must still exist; GPS only nudges the chosen
# lookahead direction toward the active route drawn in the map UI.
GPS_ROUTE_FRESH_S = 2.0
GPS_ROUTE_MAX_ACC_M = 8.0
GPS_ROUTE_MIN_SPEED_MPS = 0.35
GPS_ROUTE_LOOKAHEAD_M = 7.0
GPS_ROUTE_GAIN = 0.35
GPS_ROUTE_MAX_BIAS_DEG = 35.0
GPS_ROUTE_DONE_M = 2.0

# Turn announcements: surface the next significant route turn (direction +
# distance) so the UI can show "RIGHT TURN NOW" / "left turn in 18 m".
TURN_MIN_DEG = 25.0          # heading change at a route vertex to count as a turn
TURN_NOW_M = 5.0             # within this distance the call becomes "NOW"
TURN_ANNOUNCE_MAX_M = 40.0   # don't announce turns farther out than this

# Environment brake. Independent of steering: moving objects (pedestrians,
# riders, bikes, vehicles) are detected from the SegFormer BEV class map,
# tracked to estimate velocity, and extrapolated into a future-occupancy risk
# mask (scripts/seg_occupancy.py, ported from drive-by-segmentation's
# unified-planner live.py). The planner's trajectory is braked in proportion to
# how much of it conflicts with that risk mask, with an immediate semantic
# override for a vulnerable road user sitting directly on the planned path.
#
# Smoothed brake fraction (0..1) maps onto the pedal pot:
#   * gas is cut to 0 once the brake fraction clears ENV_BRAKE_GAS_CUT_FRAC
#     (coast-then-brake), and
#   * the UI "protective stop" state / STOP badge trips at ENV_BRAKE_STOP_FRAC.
ENV_BRAKE_GAS_CUT_FRAC = float(os.environ.get("ENV_BRAKE_GAS_CUT_FRAC", "0.15"))
ENV_BRAKE_STOP_FRAC = float(os.environ.get("ENV_BRAKE_STOP_FRAC", "0.5"))
# Braking is a per-object SPACE-TIME (time-to-collision) test, not a space-only
# corridor overlap: for each tracked obstacle we ask whether the cart and the
# object will be at the same place at the same TIME (seg_occupancy.
# evaluate_collision_brake). A person standing off to the side never enters the
# corridor; a person crossing fast clears it before the cart arrives; a person
# stopped or stepping into the path does not -> brake. Brake is graded by TTC,
# with a hard-stop floor for anything close and currently dead-ahead. Object
# forward distance comes from the blob's nearest (feet) point so the no-depth
# BEV smear doesn't push obstacles artificially far away.
ENV_BRAKE_CORRIDOR_HALF_M = float(os.environ.get("ENV_BRAKE_CORRIDOR_HALF_M", "0.9"))   # lateral half-corridor the cart sweeps (m)
ENV_BRAKE_OBJECT_RADIUS_M = float(os.environ.get("ENV_BRAKE_OBJECT_RADIUS_M", "0.3"))   # added body radius of the obstacle (m)
ENV_BRAKE_NEAR_STOP_M = float(os.environ.get("ENV_BRAKE_NEAR_STOP_M", "3.0"))  # hard-stop floor: dead-ahead obstacle within this (m)
ENV_BRAKE_HORIZON_S = float(os.environ.get("ENV_BRAKE_HORIZON_S", "4.0"))      # ignore collisions predicted beyond this TTC (s)
ENV_BRAKE_HARD_TTC_S = float(os.environ.get("ENV_BRAKE_HARD_TTC_S", "2.0"))    # TTC at/below this -> full brake; ramps to 0 by horizon
# When a hard protective stop is active, freeze steering (hold the last command)
# instead of letting the road-mask centerline swerve around the obstacle.
ENV_BRAKE_FREEZE_STEER = True

# Dynamic actor policy. Pedestrians/riders/bicycles should not carve holes into
# the static road mask that the centerline planner chases; they are handled by
# the actor tracker/brake policy instead.
DYNAMIC_ACTOR_CLASS_IDS = (11, 12, 18)  # person, rider, bicycle
DYNAMIC_ACTOR_CLASS_NAMES = {
    11: "person",
    12: "rider",
    18: "bicycle",
}
DYNAMIC_ACTOR_DILATE_PX = 5
DYNAMIC_ACTOR_ROAD_CONTEXT_PX = 23
DYNAMIC_ACTOR_MIN_ROAD_FRACTION = 0.03

# Optional detector classes from COCO-style YOLO models.
YOLO_ACTOR_CLASS_NAMES = frozenset({"person", "bicycle"})


def discover_v4l2_indices(count: int = 4, max_scan: int = 16) -> list[int]:
    found: list[int] = []
    for idx in range(max_scan):
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
        ok, _ = cap.read()
        cap.release()
        if ok:
            found.append(idx)
            if len(found) >= count:
                break
    return found


def open_camera(index: int) -> cv2.VideoCapture | None:
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def apply_front_camera_controls(index: int, slug: str) -> None:
    if not slug.startswith("front"):
        return
    for ctrl, value in (("auto_exposure", 3), ("brightness", FRONT_CAMERA_BRIGHTNESS)):
        try:
            subprocess.run(
                ["v4l2-ctl", "-d", f"/dev/video{index}", "-c", f"{ctrl}={value}"],
                check=False, timeout=2,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return


def camera_index_from_source(source: str | None) -> int | None:
    if not source:
        return None
    name = str(source)
    if name.startswith("/dev/video"):
        name = name.removeprefix("/dev/video")
    try:
        return int(name)
    except ValueError:
        return None


class RealSenseReader(threading.Thread):
    """RealSense color + aligned depth source.

    Mirrors CameraReader's `.latest()` / `.stop()` API so it slots into the
    existing slug_to_reader map. `latest_color_depth()` additionally returns
    the matched depth frame and pinhole intrinsics needed for the BEV cloud.
    """

    def __init__(self, slug: str, width: int = 640, height: int = 480,
                 fps: int = 30, enable_depth: bool = True):
        super().__init__(daemon=True, name=f"cam-{slug}")
        import pyrealsense2 as rs
        self.rs = rs
        self.slug = slug
        self.enable_depth = enable_depth
        self.flip_code = CAMERA_ORIENTATION_FIX.get(slug)
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        if enable_depth:
            cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        profile = self.pipeline.start(cfg)
        if enable_depth:
            depth_sensor = profile.get_device().first_depth_sensor()
            self.depth_scale = float(depth_sensor.get_depth_scale())
            self.align = rs.align(rs.stream.color)
        else:
            self.depth_scale = 0.0
            self.align = None
        color_prof = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = color_prof.get_intrinsics()
        self.intrinsics = (float(intr.fx), float(intr.fy),
                           float(intr.ppx), float(intr.ppy))
        self.lock = threading.Lock()
        self.frame: np.ndarray | None = None
        self.depth: np.ndarray | None = None
        self.frame_count = 0
        self.last_ok_s = 0.0
        self._stop = threading.Event()

    def run(self) -> None:
        while not self._stop.is_set():
            try:
                frames = self.pipeline.wait_for_frames(2000)
            except Exception:
                time.sleep(0.05)
                continue
            if self.enable_depth:
                aligned = self.align.process(frames)
                c = aligned.get_color_frame()
                d = aligned.get_depth_frame()
                if not c or not d:
                    continue
                color = np.asanyarray(c.get_data())
                depth = np.asanyarray(d.get_data()).astype(np.float32) * self.depth_scale
                if self.flip_code is not None:
                    color = cv2.flip(color, self.flip_code)
                    depth = cv2.flip(depth, self.flip_code)
            else:
                c = frames.get_color_frame()
                if not c:
                    continue
                color = np.asanyarray(c.get_data())
                depth = None
                if self.flip_code is not None:
                    color = cv2.flip(color, self.flip_code)
            with self.lock:
                self.frame = color
                self.depth = depth
                self.frame_count += 1
                self.last_ok_s = time.monotonic()

    def latest(self) -> np.ndarray | None:
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def latest_with_meta(self) -> tuple[np.ndarray | None, int, float]:
        with self.lock:
            frame = None if self.frame is None else self.frame.copy()
            return frame, int(self.frame_count), float(self.last_ok_s)

    def latest_color_depth(self) -> tuple[np.ndarray, np.ndarray | None] | None:
        with self.lock:
            if self.frame is None:
                return None
            if not self.enable_depth:
                return self.frame.copy(), None
            if self.depth is None:
                return None
            return self.frame.copy(), self.depth.copy()

    def stop(self) -> None:
        self._stop.set()
        try:
            self.pipeline.stop()
        except Exception:
            pass


class CameraReader(threading.Thread):
    def __init__(self, cap: cv2.VideoCapture, slug: str):
        super().__init__(daemon=True, name=f"cam-{slug}")
        self.cap = cap
        self.slug = slug
        self.flip_code = CAMERA_ORIENTATION_FIX.get(slug)
        self.lock = threading.Lock()
        self.frame: np.ndarray | None = None
        self.frame_count = 0
        self.last_ok_s = 0.0
        self._stop = threading.Event()

    def run(self) -> None:
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            if ok and frame is not None:
                if self.flip_code is not None:
                    frame = cv2.flip(frame, self.flip_code)
                with self.lock:
                    self.frame = frame
                    self.frame_count += 1
                    self.last_ok_s = time.monotonic()
            else:
                time.sleep(0.01)

    def latest(self) -> np.ndarray | None:
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def latest_with_meta(self) -> tuple[np.ndarray | None, int, float]:
        with self.lock:
            frame = None if self.frame is None else self.frame.copy()
            return frame, int(self.frame_count), float(self.last_ok_s)

    def stop(self) -> None:
        self._stop.set()
        self.cap.release()


class VideoReader(threading.Thread):
    def __init__(self, video_path: str, loop: bool = True):
        super().__init__(daemon=True, name="video-reader")
        self.video_path = video_path
        self.loop = loop
        self.lock = threading.Lock()
        self.frame: np.ndarray | None = None
        self.frame_count = 0
        self.last_ok_s = 0.0
        self._stop = threading.Event()

    def run(self) -> None:
        while not self._stop.is_set():
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                print(f"[video] failed to open {self.video_path}", flush=True)
                return
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            period = 1.0 / fps
            next_t = time.monotonic()
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                frame = cv2.resize(frame, (CAM_W, CAM_H), interpolation=cv2.INTER_AREA)
                with self.lock:
                    self.frame = frame
                    self.frame_count += 1
                    self.last_ok_s = time.monotonic()
                next_t += period
                wait = next_t - time.monotonic()
                if wait > 0:
                    time.sleep(wait)
            cap.release()
            if not self.loop:
                return

    def latest(self) -> np.ndarray | None:
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def latest_with_meta(self) -> tuple[np.ndarray | None, int, float]:
        with self.lock:
            frame = None if self.frame is None else self.frame.copy()
            return frame, int(self.frame_count), float(self.last_ok_s)

    def stop(self) -> None:
        self._stop.set()


def write_jpeg_atomic(path: Path, frame_bgr: np.ndarray, quality: int = JPEG_QUALITY) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(buf.tobytes())
    os.replace(tmp, path)


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(payload, f, separators=(",", ":"))
    os.replace(tmp, path)


def encode_segmentation_rle(seg_map: np.ndarray) -> dict:
    """Run-length encode a uint8 label map in row-major order for JSON."""
    flat = np.ascontiguousarray(seg_map, dtype=np.uint8).ravel()
    if flat.size == 0:
        return {"values": [], "counts": []}
    changes = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    starts = np.concatenate(([0], changes))
    ends = np.concatenate((changes, [flat.size]))
    return {
        "values": flat[starts].astype(int).tolist(),
        "counts": (ends - starts).astype(int).tolist(),
    }


def segmentation_map_payload(seg_map: np.ndarray, colors: list,
                             active_cam: str, infer_count: int,
                             model_name: str) -> dict:
    h, w = seg_map.shape[:2]
    return {
        "_schema": "caddy.segmentation_map.v1",
        "ts": time.time(),
        "infer_count": int(infer_count),
        "active_cam": active_cam,
        "model": MODEL_NAME,
        "model_full": model_name,
        "shape": [int(h), int(w)],
        "dtype": "uint8",
        "encoding": "rle_flat_c_order",
        "rle": encode_segmentation_rle(seg_map),
        "palette_rgb": [[int(c) for c in color] for color in colors],
        "note": (
            "Decode with np.repeat(rle.values, rle.counts).astype(np.uint8)"
            ".reshape(shape). Labels use the drive-by-segmentation Cityscapes"
            " palette order."
        ),
    }


def sync_cuda_if_needed(device: str) -> None:
    if device != "cuda":
        return
    import torch
    torch.cuda.synchronize()


def timed_segment_frame(frame_rgb: np.ndarray, proc, model, device: str) -> tuple[np.ndarray, dict[str, float]]:
    import torch

    timings: dict[str, float] = {}
    t = time.perf_counter()
    pil = Image.fromarray(frame_rgb)
    timings["seg_pil_ms"] = (time.perf_counter() - t) * 1000.0

    t = time.perf_counter()
    inputs = proc(images=pil, return_tensors="pt")
    timings["seg_preprocess_ms"] = (time.perf_counter() - t) * 1000.0

    t = time.perf_counter()
    inputs = inputs.to(device)
    dtype = next(model.parameters()).dtype
    if inputs["pixel_values"].dtype != dtype:
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype)
    sync_cuda_if_needed(device)
    timings["seg_transfer_ms"] = (time.perf_counter() - t) * 1000.0

    t = time.perf_counter()
    with torch.no_grad():
        out = model(**inputs)
    sync_cuda_if_needed(device)
    timings["seg_forward_ms"] = (time.perf_counter() - t) * 1000.0

    t = time.perf_counter()
    seg = proc.post_process_semantic_segmentation(
        out, target_sizes=[frame_rgb.shape[:2]]
    )[0].cpu().numpy().astype(np.uint8)
    sync_cuda_if_needed(device)
    timings["seg_postprocess_ms"] = (time.perf_counter() - t) * 1000.0
    return seg, timings


def configure_segformer_processor(proc, input_size: int) -> int | None:
    """Optionally lower SegFormer resize resolution for faster CPU inference."""
    if input_size <= 0:
        return None
    size = int(np.clip(input_size, 256, 1024))
    proc.size = {"height": size, "width": size}
    return size


def read_ego_speed_mph(path: Path, fresh_s: float = 1.0) -> tuple[float, bool]:
    try:
        with path.open() as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError, TypeError, ValueError):
        return 0.0, False
    if time.time() - float(data.get("ts", 0.0)) > fresh_s:
        return 0.0, False
    if not data.get("connected", False):
        return 0.0, False
    sample = data.get("sample") or {}
    try:
        speed_mps = abs(float(sample.get("speed_mps", 0.0)))
    except (TypeError, ValueError):
        return 0.0, False
    return speed_mps * 2.23694, True


def adaptive_lookahead_m(speed_mph: float, speed_ok: bool) -> float:
    if not speed_ok:
        speed_mph = 2.0
    speed_mps = speed_mph * 0.44704
    return float(np.clip(speed_mps * LOOKAHEAD_TIME_S, LOOKAHEAD_MIN_M, LOOKAHEAD_MAX_M))


def constant_gas_for_mph(target_mph: float) -> float:
    # Linear pot↔mph map, but raised to ROLLING_GAS_FLOOR whenever the user
    # asks for any forward motion. Without the floor, low-mph targets sit
    # under the kart's actual rolling-resistance gas and the loop tops out
    # short of the setpoint.
    if target_mph <= 0.0:
        return 0.0
    return float(np.clip(max(target_mph * GAS_PER_MPH, ROLLING_GAS_FLOOR), 0.0, 1.0))


class SpeedController:
    """PI on (target_mph - ego_mph), output added to the open-loop gas."""

    def __init__(self) -> None:
        self.integral = 0.0
        self.last_t = None
        self.last_gas = 0.0

    def reset(self) -> None:
        self.integral = 0.0
        self.last_t = None
        self.last_gas = 0.0

    def step(self, target_mph: float, ego_mph: float, ego_ok: bool,
             feed_forward_gas: float, gas_ceiling: float = 1.0) -> tuple[float, float, float]:
        """Return (gas, trim, brake). Gas/brake are mutually exclusive.

        `gas_ceiling` lets the launch ramp cap the output without winding
        the integrator: when the ramp is clipping us low, we suppress
        integration in the direction that would wind further into the
        clip — classic conditional-integration anti-windup.
        """
        now = time.monotonic()
        dt = 0.1 if self.last_t is None else max(0.0, now - self.last_t)
        self.last_t = now
        if not ego_ok:
            self.integral = 0.0
            gas_raw = float(np.clip(feed_forward_gas, 0.0, gas_ceiling))
            gas = self.slew_gas(gas_raw, dt)
            return gas, 0.0, 0.0
        err = target_mph - ego_mph

        # Trim clamp scales with feed-forward → consistent authority at any
        # target speed. At 2 mph: ±0.10 ; at 8 mph: ±0.36.
        trim_clamp = max(GAS_TRIM_FLOOR, GAS_TRIM_SCALE * feed_forward_gas)

        # Provisional integral, with conditional anti-windup:
        # don't integrate further into the direction we're already saturated.
        trial_integral = self.integral + err * dt
        trim_unclipped = SPEED_KP * err + SPEED_KI * trial_integral
        gas_unclipped = feed_forward_gas + trim_unclipped
        saturated_high = gas_unclipped > gas_ceiling
        saturated_low = gas_unclipped < 0.0
        winding_into_clip = (
            (saturated_high and err > 0) or (saturated_low and err < 0)
        )
        if not winding_into_clip:
            self.integral = float(np.clip(
                trial_integral,
                -SPEED_I_CLAMP / max(SPEED_KI, 1e-6),
                 SPEED_I_CLAMP / max(SPEED_KI, 1e-6),
            ))

        trim = SPEED_KP * err + SPEED_KI * self.integral
        trim = float(np.clip(trim, -trim_clamp, trim_clamp))
        gas_raw = float(np.clip(feed_forward_gas + trim, 0.0, gas_ceiling))
        gas = self.slew_gas(gas_raw, dt)

        # Engine braking via gas-off comes "free" — only blend in pad brake
        # when we're still over target *with gas already at zero*. This keeps
        # cruise running on gas alone and only touches the pedal pad when
        # gravity (downhill) wants to push us past target.
        brake = 0.0
        overshoot = ego_mph - target_mph - BRAKE_DEADBAND_MPH
        if gas <= 1e-3 and overshoot > 0.0:
            brake = float(np.clip(BRAKE_KP * overshoot, 0.0, BRAKE_MAX))
        return gas, trim, brake

    def slew_gas(self, gas: float, dt: float) -> float:
        dt = max(0.0, min(0.25, dt))
        rise = GAS_RISE_RATE_PER_S * dt
        fall = GAS_FALL_RATE_PER_S * dt
        lo = self.last_gas - fall
        hi = self.last_gas + rise
        out = float(np.clip(gas, lo, hi))
        self.last_gas = out
        return out

    def sync_gas(self, gas: float) -> None:
        self.last_gas = float(np.clip(gas, 0.0, 1.0))


def bev_class_map_cached(seg_map: np.ndarray, remap: seg_fast.BevRemap) -> np.ndarray:
    """Project segmentation labels into BEV using the cached homography."""
    cls_ids = seg_map[remap.map_v, remap.map_u]
    out = np.full((remap.bev_size, remap.bev_size), 255, dtype=np.uint8)
    np.copyto(out, np.clip(cls_ids, 0, 254).astype(np.uint8), where=remap.valid)
    return out


def neutralize_dynamic_actors_for_planning(
    road_mask: np.ndarray,
    class_map: np.ndarray | None,
    *,
    enabled: bool = True,
) -> tuple[np.ndarray, dict]:
    """Fill VRU footprints back into the planning road mask.

    Segmentation still sees the actor, and occupancy still tracks/brakes for it.
    This only affects the static centerline planner so a person/bike blob does
    not become an artificial curb that makes the cart swerve.
    """
    raw = np.asarray(road_mask, dtype=bool)
    info = {
        "enabled": bool(enabled),
        "mode": "dynamic_actor_mask_to_road_context",
        "class_ids": list(DYNAMIC_ACTOR_CLASS_IDS),
        "actor_px": 0,
        "expanded_actor_px": 0,
        "recovered_px": 0,
        "actor_px_on_raw_road": 0,
        "actor_px_on_planning_road": 0,
        "actor_px_remaining_nonroad": 0,
        "road_px_raw": int(raw.sum()),
        "road_px_planning": int(raw.sum()),
    }
    if not enabled or class_map is None:
        return raw, info

    cls = np.asarray(class_map)
    if cls.shape[:2] != raw.shape[:2]:
        info["enabled"] = False
        info["reason"] = "shape_mismatch"
        return raw, info

    actor_raw = np.isin(cls, DYNAMIC_ACTOR_CLASS_IDS)
    info["actor_px"] = int(actor_raw.sum())
    info["actor_px_on_raw_road"] = int((actor_raw & raw).sum())
    if not actor_raw.any():
        return raw, info

    actor_u8 = actor_raw.astype(np.uint8)
    if DYNAMIC_ACTOR_DILATE_PX > 0:
        k = DYNAMIC_ACTOR_DILATE_PX * 2 + 1
        actor_u8 = cv2.dilate(actor_u8, np.ones((k, k), np.uint8), iterations=1)
    actor = actor_u8.astype(bool)

    road_u8 = raw.astype(np.uint8)
    ctx_k = max(3, int(DYNAMIC_ACTOR_ROAD_CONTEXT_PX) | 1)
    road_nearby = cv2.dilate(road_u8, np.ones((ctx_k, ctx_k), np.uint8), iterations=1).astype(bool)
    road_fraction = cv2.blur(road_u8.astype(np.float32), (ctx_k, ctx_k))
    valid = cls != 255
    recover = actor & valid & (road_nearby | (road_fraction >= DYNAMIC_ACTOR_MIN_ROAD_FRACTION))

    planning = raw.copy()
    planning[recover] = True
    info["expanded_actor_px"] = int(actor.sum())
    info["recovered_px"] = int(recover.sum())
    info["actor_px_on_planning_road"] = int((actor_raw & planning).sum())
    info["actor_px_remaining_nonroad"] = int((actor_raw & ~planning).sum())
    info["road_px_planning"] = int(planning.sum())
    return planning, info


class BevImageProjector:
    """Nearest-neighbour inverse lookup from image pixels to BEV local metres."""

    def __init__(self, remap: seg_fast.BevRemap, geom: seg_occupancy.BevGeometry):
        by, bx = np.nonzero(remap.valid)
        self.bx = bx.astype(np.int32)
        self.by = by.astype(np.int32)
        self.u = remap.map_u[by, bx].astype(np.float32)
        self.v = remap.map_v[by, bx].astype(np.float32)
        self.geom = geom

    def image_point_to_local(
        self,
        u: float,
        v: float,
        *,
        max_pixel_error: float = 48.0,
    ) -> dict | None:
        if self.u.size == 0:
            return None
        du = self.u - float(u)
        dv = self.v - float(v)
        d2 = du * du + dv * dv
        idx = int(np.argmin(d2))
        err = float(math.sqrt(float(d2[idx])))
        if err > max_pixel_error:
            return None
        bx = int(self.bx[idx])
        by = int(self.by[idx])
        fwd, left = self.geom.bev_to_local(bx, by)
        return {
            "fwd_m": float(fwd),
            "left_m": float(left),
            "bev_px": [bx, by],
            "projection_error_px": round(err, 2),
        }


class YoloActorDetector:
    """Optional pedestrian/bicycle detector.

    The segmentation model remains the primary source. YOLO adds box-level
    detections when the package/model are present, but this class degrades to a
    no-op so the cart can still run from semantic VRU blobs alone.
    """

    def __init__(self, model_name: str, conf: float, imgsz: int, enabled: bool = True):
        self.model_name = model_name
        self.conf = float(np.clip(conf, 0.01, 0.99))
        self.imgsz = int(max(128, imgsz))
        self.enabled = bool(enabled)
        self.available = False
        self.error: str | None = None
        self.model = None
        self.names: dict[int, str] = {}
        self.actor_class_ids: set[int] = set()
        if not self.enabled:
            return
        try:
            cfg_dir = Path(os.environ.get("YOLO_CONFIG_DIR", "/tmp"))
            cfg_dir.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("YOLO_CONFIG_DIR", str(cfg_dir))
            from ultralytics import YOLO

            self.model = YOLO(model_name)
            raw_names = getattr(self.model, "names", {}) or {}
            self.names = {
                int(k): str(v).lower()
                for k, v in (
                    raw_names.items()
                    if isinstance(raw_names, dict)
                    else enumerate(raw_names)
                )
            }
            self.actor_class_ids = {
                cls_id for cls_id, name in self.names.items()
                if name in YOLO_ACTOR_CLASS_NAMES
            }
            if not self.actor_class_ids:
                self.error = "model_has_no_person_or_bicycle_classes"
                return
            self.available = True
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"

    def status(self) -> dict:
        return {
            "enabled": bool(self.enabled),
            "available": bool(self.available),
            "model": self.model_name,
            "conf": self.conf,
            "imgsz": self.imgsz,
            "classes": sorted(YOLO_ACTOR_CLASS_NAMES),
            "error": self.error,
        }

    def detect(self, frame_bgr: np.ndarray) -> tuple[list[dict], dict]:
        state = self.status()
        if not self.available or self.model is None:
            state["count"] = 0
            return [], state
        t0 = time.perf_counter()
        try:
            results = self.model.predict(
                frame_bgr,
                imgsz=self.imgsz,
                conf=self.conf,
                verbose=False,
            )
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            state = self.status()
            state["count"] = 0
            state["latency_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
            return [], state

        actors: list[dict] = []
        result = results[0] if results else None
        boxes = getattr(result, "boxes", None) if result is not None else None
        if boxes is not None:
            for box in boxes:
                try:
                    cls_id = int(box.cls[0].detach().cpu().item())
                    if cls_id not in self.actor_class_ids:
                        continue
                    conf = float(box.conf[0].detach().cpu().item())
                    xyxy = box.xyxy[0].detach().cpu().numpy().astype(float)
                except Exception:
                    continue
                label = self.names.get(cls_id, str(cls_id))
                actors.append({
                    "source": "yolo",
                    "class": label,
                    "class_id": cls_id,
                    "confidence": round(conf, 3),
                    "bbox_xyxy": [round(float(v), 1) for v in xyxy.tolist()],
                })
        state = self.status()
        state["count"] = len(actors)
        state["latency_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
        return actors, state


def project_detector_actors_to_bev(
    actors: list[dict],
    projector: BevImageProjector | None,
) -> tuple[list[dict], list[seg_occupancy.MotionDetection]]:
    if projector is None:
        return actors, []
    projected: list[dict] = []
    detections: list[seg_occupancy.MotionDetection] = []
    for actor in actors:
        out = dict(actor)
        box = actor.get("bbox_xyxy") or []
        if len(box) == 4:
            x1, y1, x2, y2 = [float(v) for v in box]
            h = max(1.0, y2 - y1)
            # Feet/contact point: slightly above the bottom of the box to avoid
            # projecting a shadow/crop edge.
            u = 0.5 * (x1 + x2)
            v = y2 - 0.04 * h
            local = projector.image_point_to_local(u, v)
            if local is not None:
                out.update(local)
                conf = float(actor.get("confidence", 1.0))
                detections.append(
                    seg_occupancy.MotionDetection(
                        fwd_m=float(local["fwd_m"]),
                        left_m=float(local["left_m"]),
                        confidence=conf,
                    )
                )
        projected.append(out)
    return projected, detections


def summarize_actor_tracks(
    occ: "seg_occupancy.PredictedOccupancy | None",
    *,
    limit: int = 16,
) -> list[dict]:
    if occ is None:
        return []
    tracks = sorted(
        occ.all_tracks,
        key=lambda t: (float(t.pos_m[0]), -float(t.confidence)),
    )
    out: list[dict] = []
    for tk in tracks[:limit]:
        out.append({
            "id": int(tk.track_id),
            "fwd_m": round(float(tk.pos_m[0]), 3),
            "left_m": round(float(tk.pos_m[1]), 3),
            "vx_mps": round(float(tk.vel_mps[0]), 3),
            "vy_mps": round(float(tk.vel_mps[1]), 3),
            "speed_mps": round(float(tk.speed_mps), 3),
            "hits": int(tk.hits),
            "confidence": round(float(tk.confidence), 3),
        })
    return out


def build_actor_policy_state(
    neutralization: dict,
    detector_state: dict,
    detector_actors: list[dict],
    occ: "seg_occupancy.PredictedOccupancy | None",
    collision: dict | None,
    brake_01: float,
) -> dict:
    tracks = summarize_actor_tracks(occ)
    return {
        "mode": "single_bev_path_actor_speed_policy",
        "steering_source": "actor_neutralized_bev_centerline",
        "speed_source": "space_time_actor_brake",
        "neutralization": neutralization,
        "detector": detector_state,
        "detector_actors": detector_actors[:16],
        "tracks": tracks,
        "track_count": len(tracks),
        "brake_01": round(float(brake_01), 3),
        "collision": collision or {"brake_01": 0.0, "objects": 0, "threat": None},
    }


def build_environment_threat(
    collision: dict | None,
    image: dict | None,
    brake_01: float,
    enabled: bool,
) -> dict:
    """Summarize the brake decision (space-time actor collision) for state/UI.

    Mirrors the shape the web UI expects from the old protective-stop block
    (``active`` / ``objects`` / ``threat.{label,x_m}``) and adds the graded
    fields (``brake_target`` / ``ttc_s`` / ``image_coverage``). The threat
    label/distance/TTC come from ``evaluate_collision_brake``'s nearest
    collision."""
    collision = collision or {}
    image = image or {"brake_01": 0.0, "coverage": 0.0, "enabled": False}
    active = bool(enabled and brake_01 >= ENV_BRAKE_STOP_FRAC)
    if active:
        reason = str(collision.get("reason", ""))
    else:
        reason = ""
    return {
        "enabled": bool(enabled),
        "active": active,
        "source": "collision" if enabled else None,
        "reason": reason,
        "brake_target": round(float(brake_01), 3),
        "ttc_s": collision.get("ttc_s"),
        "image_backstop_enabled": bool(image.get("enabled", False)),
        "image_coverage": image.get("coverage", 0.0),
        "objects": int(collision.get("objects", 0)),
        "threat": collision.get("threat"),
    }


def forward_stop_corridor_bev(
    bev_geom: "seg_occupancy.BevGeometry", near_m: float
) -> np.ndarray:
    """Straight near-forward corridor centerline in BEV pixels (Nx2 ``[bx, by]``).

    The strip of ground directly ahead the cart will roll over next. Braking is
    judged against this corridor — independent of the reactive steering
    centerline — so an obstacle directly ahead stops the cart rather than being
    steered around."""
    fwds = np.linspace(0.3, float(near_m), 32)
    bx, by = bev_geom.local_to_bev(fwds, np.zeros_like(fwds))
    return np.stack([np.asarray(bx), np.asarray(by)], axis=1).astype(np.int32)


_lookahead_point_filt: tuple[float, float] | None = None


def reset_lookahead_heading_filter() -> None:
    global _lookahead_point_filt
    _lookahead_point_filt = None


def _smooth_centerline(pts: np.ndarray, win: int) -> np.ndarray:
    if win <= 1 or len(pts) < win:
        return pts
    k = np.ones(win, dtype=np.float64) / win
    xs = np.convolve(pts[:, 0], k, mode="same")
    ys = np.convolve(pts[:, 1], k, mode="same")
    return np.stack([xs, ys], axis=1)


def _resample_lookahead_point(pts: np.ndarray, lookahead_m: float) -> tuple[float, float] | None:
    """Interpolate the lane point at arc length ≈ lookahead_m.

    Works for arbitrarily sharp turns because arc length grows monotonically
    even when the path curls past 90° (where y(x) breaks down)."""
    if len(pts) < 2:
        return None
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] <= 0.0:
        return None
    s_target = min(float(lookahead_m), float(s[-1]))
    xL = float(np.interp(s_target, s, pts[:, 0]))
    yL = float(np.interp(s_target, s, pts[:, 1]))
    return xL, yL


def lookahead_heading_steering_deg(lane_local: np.ndarray, lookahead_m: float) -> float | None:
    """Pure-pursuit steering from a BEV lane centerline.

    1. Smooth lane_local with a rolling mean (kills planner jitter).
    2. Resample by arc length to get a stable lookahead point P=(xL, yL) —
       this works at any turn sharpness, unlike a y(x) polynomial fit.
    3. EMA the lookahead point (not the angle) so jitter is removed without
       lagging real steering response.
    4. Apply the bicycle pure-pursuit law:
           alpha = atan2(yL, xL)                          # bearing, ±π
           delta = atan2(2 L sin(alpha), Ld)              # road-wheel rad
       alpha ranges across the full ±π, so delta * STEERING_COLUMN_RATIO
       can ride all the way to ±270°.
    """
    global _lookahead_point_filt

    if lane_local is None or len(lane_local) < 4:
        return None

    pts = np.asarray(lane_local, dtype=np.float64)
    pts = pts[np.isfinite(pts).all(axis=1)]
    pts = pts[(pts[:, 0] > STEER_FIT_MIN_M) & (pts[:, 0] < STEER_FIT_MAX_M)]
    if len(pts) < 4:
        return None

    pts = _smooth_centerline(pts, CENTERLINE_SMOOTH_WIN)
    p = _resample_lookahead_point(pts, lookahead_m)
    if p is None:
        return None
    xL, yL = p

    if _lookahead_point_filt is None:
        _lookahead_point_filt = (xL, yL)
    else:
        a = LOOKAHEAD_POINT_EMA
        _lookahead_point_filt = (
            a * xL + (1.0 - a) * _lookahead_point_filt[0],
            a * yL + (1.0 - a) * _lookahead_point_filt[1],
        )
    xLf, yLf = _lookahead_point_filt

    Ld = math.hypot(xLf, yLf)
    if Ld < 1e-3:
        return None
    alpha = math.atan2(yLf, xLf)
    delta = math.atan2(2.0 * WHEELBASE_M * math.sin(alpha), Ld)
    column_deg = math.degrees(delta) * STEERING_COLUMN_RATIO * STEER_GAIN
    return float(np.clip(column_deg, -270.0, 270.0))


def make_sources(args) -> list[CameraReader | VideoReader | RealSenseReader]:
    active_slug = args.active_slug
    if args.video:
        reader = VideoReader(args.video, loop=not args.no_loop)
        reader.slug = active_slug
        return [reader]

    if args.source == "realsense":
        rs_reader = RealSenseReader(
            slug=active_slug,
            width=args.rs_width, height=args.rs_height, fps=args.rs_fps,
            enable_depth=not args.no_depth,
        )
        return [rs_reader]

    if args.camera_index is not None:
        cap = open_camera(args.camera_index)
        if cap is None:
            raise RuntimeError(f"failed to open camera {args.camera_index} for {active_slug}")
        apply_front_camera_controls(args.camera_index, active_slug)
        return [CameraReader(cap, active_slug)]

    indices = discover_v4l2_indices(count=len(SLUGS), max_scan=args.max_scan)
    if len(indices) < len(SLUGS):
        raise RuntimeError(f"expected {len(SLUGS)} cameras, found {indices}")

    readers: list[CameraReader] = []
    for slug, idx in zip(SLUGS, indices):
        cap = open_camera(idx)
        if cap is None:
            raise RuntimeError(f"failed to open camera {idx} for {slug}")
        apply_front_camera_controls(idx, slug)
        readers.append(CameraReader(cap, slug))
    return readers


def import_segmentation_stack(repo_dir: Path):
    if not (repo_dir / "live.py").exists():
        raise RuntimeError(f"drive-by-segmentation repo not found at {repo_dir}")
    sys.path.insert(0, str(repo_dir))
    import live as seg_live
    import render_trajectories as rt
    from path_planning import lane_aware_centerline_path
    from render import CITYSCAPES_COLORS, create_bev, create_overlay

    return seg_live, rt, lane_aware_centerline_path, create_bev, create_overlay, CITYSCAPES_COLORS


def import_clrnet_stack():
    import clrnet_infer as clrnet

    return clrnet


def load_json_file(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def find_e2e_slot(calib: dict, slot_name: str) -> dict | None:
    for slot in calib.get("slots", []):
        if str(slot.get("slot", "")).upper() == slot_name.upper():
            return slot
    return None


def drive_by_height_fallback(seg_repo: Path) -> float:
    path = seg_repo / "camera_calibration.json"
    try:
        old = load_json_file(path)
        return float(old.get("extrinsics", {}).get("height_m", 0.63094))
    except Exception:
        return 0.63094


def e2e_slot_to_bev_calib(e2e_calib: dict, slot: dict,
                          height_m: float) -> dict:
    intr = slot["intrinsics"]
    dist = slot.get("distortion_coeffs") or []
    fx = float(intr[0][0])
    fy = float(intr[1][1])
    cx = float(intr[0][2])
    cy = float(intr[1][2])
    image_size = slot.get("image_size") or e2e_calib.get("calibrated_image_size") or [CAM_W, CAM_H]
    return {
        "intrinsics": {
            "model": "pinhole",
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "k1": float(dist[0]) if len(dist) > 0 else 0.0,
            "k2": float(dist[1]) if len(dist) > 1 else 0.0,
            "resolution": [int(image_size[0]), int(image_size[1])],
        },
        "extrinsics": {
            "height_m": float(height_m),
            "pitch_deg": float(slot.get("camera_pitch_deg", 0.0)),
            "roll_deg": float(slot.get("camera_roll_deg", 0.0)),
            "yaw_deg": float(slot.get("camera_yaw_deg", 0.0)),
            "ego_to_camera": slot.get("ego_to_camera"),
            "camera_center_ego_m": slot.get("camera_center_ego_m"),
        },
        "road_width_ft": 20,
        "bev_range": {
            "forward_ft": 100,
            "side_ft": 50,
        },
        "source_calibration": {
            "type": e2e_calib.get("type", "e2e"),
            "slot": slot.get("slot"),
            "source": slot.get("source"),
            "physical_position": slot.get("physical_position"),
        },
    }


def load_bev_calibration(args) -> tuple[dict, dict | None]:
    path = args.calib or (args.seg_repo / "camera_calibration.json")
    raw = load_json_file(path)
    if "slots" not in raw:
        return raw, None

    slot = find_e2e_slot(raw, args.camera_slot)
    if slot is None:
        raise RuntimeError(f"E2E calibration {path} has no slot {args.camera_slot}")
    height_m = (
        args.camera_height_m
        if args.camera_height_m is not None
        else drive_by_height_fallback(args.seg_repo)
    )
    return e2e_slot_to_bev_calib(raw, slot, height_m), slot


def read_json_if_fresh(path: Path, max_age_s: float | None = None) -> dict | None:
    try:
        data = load_json_file(path)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    if not isinstance(data, dict):
        return None
    if max_age_s is not None:
        try:
            if time.time() - float(data.get("ts", 0.0)) > max_age_s:
                return None
        except (TypeError, ValueError):
            return None
    return data


def latlon_to_enu_m(lat: float, lon: float, ref_lat: float,
                    ref_lon: float) -> tuple[float, float]:
    radius_m = 6371000.0
    d_lat = math.radians(lat - ref_lat)
    d_lon = math.radians(lon - ref_lon)
    ref = math.radians(ref_lat)
    east = d_lon * math.cos(ref) * radius_m
    north = d_lat * radius_m
    return east, north


def route_target_enu(route_ll: list[list[float]], lat: float, lon: float,
                     lookahead_m: float) -> tuple[float, float, float] | None:
    route_pts: list[tuple[float, float]] = []
    for p in route_ll:
        if not isinstance(p, (list, tuple)) or len(p) < 2:
            continue
        try:
            p_lat = float(p[0])
            p_lon = float(p[1])
        except (TypeError, ValueError):
            continue
        if math.isfinite(p_lat) and math.isfinite(p_lon):
            route_pts.append(latlon_to_enu_m(p_lat, p_lon, lat, lon))
    pts = np.array(route_pts, dtype=np.float64)
    if pts.shape[0] < 2:
        return None
    seg = pts[1:] - pts[:-1]
    seg_len = np.linalg.norm(seg, axis=1)
    good = seg_len > 1e-3
    if not np.any(good):
        return None
    total_len = float(np.sum(seg_len[good]))
    if total_len < GPS_ROUTE_DONE_M:
        return None
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])

    best_dist = float("inf")
    best_s = 0.0
    for i, ok in enumerate(good):
        if not ok:
            continue
        a = pts[i]
        v = seg[i]
        t = float(np.clip(-np.dot(a, v) / (seg_len[i] * seg_len[i]), 0.0, 1.0))
        proj = a + t * v
        dist = float(np.linalg.norm(proj))
        if dist < best_dist:
            best_dist = dist
            best_s = float(cum[i] + t * seg_len[i])

    target_s = min(best_s + lookahead_m, float(cum[-1]))
    for i, ok in enumerate(good):
        if not ok:
            continue
        if cum[i] <= target_s <= cum[i + 1]:
            t = (target_s - cum[i]) / max(1e-6, seg_len[i])
            target = pts[i] + t * seg[i]
            remaining = max(0.0, float(cum[-1] - best_s))
            return float(target[0]), float(target[1]), remaining
    target = pts[-1]
    remaining = max(0.0, float(cum[-1] - best_s))
    return float(target[0]), float(target[1]), remaining


def route_steering_deg_from_gps(fix: dict, route: dict,
                                lookahead_m: float) -> tuple[float, dict] | None:
    if not route.get("active"):
        return None
    route_ll = route.get("geometry")
    if not isinstance(route_ll, list) or len(route_ll) < 2:
        return None
    try:
        lat = float(fix["lat_deg"])
        lon = float(fix["lon_deg"])
        course_deg = float(fix["course_deg"])
        speed_mps = float(fix.get("speed_mps", 0.0) or 0.0)
    except (KeyError, TypeError, ValueError):
        return None
    if not math.isfinite(course_deg) or course_deg < 0.0:
        return None
    try:
        h_acc = float(fix.get("h_acc_m", GPS_ROUTE_MAX_ACC_M))
    except (TypeError, ValueError):
        h_acc = GPS_ROUTE_MAX_ACC_M
    if h_acc > GPS_ROUTE_MAX_ACC_M or speed_mps < GPS_ROUTE_MIN_SPEED_MPS:
        return None

    target = route_target_enu(route_ll, lat, lon, lookahead_m)
    if target is None:
        return None
    east, north, remaining_m = target
    if remaining_m < GPS_ROUTE_DONE_M:
        return None

    course = math.radians(course_deg)
    # GPS course is degrees clockwise from north. Rotate the ENU target into
    # the cart frame: x forward, y lateral.
    #
    # CRITICAL sign convention: lateral is positive to the RIGHT, to match
    # seg_fast's lane_local. That array's column is named "local_left" but is
    # actually right-positive — it's (bx/bev_size - 0.5)*2*range_side, and bx
    # is the BEV image column, so the cart's right (image-right) is > 0.
    # Segmentation steering and route steering share the exact same
    # atan2(lateral, forward) -> RATIO*STEER_GAIN*STEERING_SIGN chain, so the
    # route MUST use the same lateral sign or it biases toward the opposite
    # turn (this was the wrong-direction bug).
    x_fwd = east * math.sin(course) + north * math.cos(course)
    y_right = east * math.cos(course) - north * math.sin(course)
    if x_fwd < -1.0:
        return None
    ld = math.hypot(x_fwd, y_right)
    if ld < 1e-3:
        return None
    alpha = math.atan2(y_right, x_fwd)
    delta = math.atan2(2.0 * WHEELBASE_M * math.sin(alpha), ld)
    column_deg = math.degrees(delta) * STEERING_COLUMN_RATIO * STEER_GAIN
    diag = {
        "target_x_m": round(float(x_fwd), 3),
        "target_y_m": round(float(y_right), 3),
        "remaining_m": round(float(remaining_m), 2),
        "course_deg": round(float(course_deg), 1),
        "h_acc_m": round(float(h_acc), 2),
    }
    return float(np.clip(column_deg, -270.0, 270.0)), diag


def _wrap_deg(d: float) -> float:
    return (d + 180.0) % 360.0 - 180.0


def next_turn_on_route(route_ll: list, lat: float, lon: float) -> dict | None:
    """First significant turn ahead of the current GPS position on the route.

    Walks the route polyline (in a local ENU frame centred on the cart),
    finds where the cart currently projects onto it, then returns the next
    vertex whose heading change exceeds TURN_MIN_DEG. Direction follows the
    GPS convention (bearing clockwise from north): a positive heading change
    is a right turn.
    """
    pts_ll: list[tuple[float, float]] = []
    for p in route_ll:
        if not isinstance(p, (list, tuple)) or len(p) < 2:
            continue
        try:
            pts_ll.append((float(p[0]), float(p[1])))
        except (TypeError, ValueError):
            continue
    if len(pts_ll) < 3:
        return None
    pts = np.array([latlon_to_enu_m(a, b, lat, lon) for a, b in pts_ll],
                   dtype=np.float64)
    seg = pts[1:] - pts[:-1]
    seg_len = np.linalg.norm(seg, axis=1)
    if not np.any(seg_len > 1e-3):
        return None
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])

    # Current arc-length position = projection of ego (origin) onto the route.
    best_dist = float("inf")
    best_s = 0.0
    for i in range(len(seg)):
        if seg_len[i] <= 1e-3:
            continue
        a = pts[i]
        v = seg[i]
        t = float(np.clip(-np.dot(a, v) / (seg_len[i] * seg_len[i]), 0.0, 1.0))
        proj = a + t * v
        d = float(np.linalg.norm(proj))
        if d < best_dist:
            best_dist = d
            best_s = float(cum[i] + t * seg_len[i])

    seg_head = np.degrees(np.arctan2(seg[:, 0], seg[:, 1]))  # cw from north
    for i in range(1, len(seg)):
        if seg_len[i - 1] <= 1e-3 or seg_len[i] <= 1e-3:
            continue
        turn = _wrap_deg(float(seg_head[i] - seg_head[i - 1]))
        if abs(turn) < TURN_MIN_DEG:
            continue
        dist = float(cum[i] - best_s)  # vertex i sits at arc-length cum[i]
        if dist <= 0.0:
            continue
        return {
            "turn_dir": "right" if turn > 0 else "left",
            "turn_dist_m": round(dist, 1),
            "turn_angle_deg": round(turn, 1),
        }
    return None


def turn_announce_text(turn: dict | None) -> str:
    if not turn:
        return ""
    d = float(turn["turn_dist_m"])
    if d > TURN_ANNOUNCE_MAX_M:
        return ""
    direction = turn["turn_dir"]
    if d <= TURN_NOW_M:
        return f"{direction.upper()} TURN NOW"
    return f"{direction} turn in {d:.0f} m"


def gps_route_bias_deg(seg_steer_deg: float, args) -> tuple[float, dict]:
    route = read_json_if_fresh(args.route_file, None)
    if route is None or not route.get("active"):
        return 0.0, {"active": False}
    gps = read_json_if_fresh(args.gps_state_file, GPS_ROUTE_FRESH_S)
    if gps is None or not gps.get("connected", False):
        return 0.0, {"active": True, "gps_ok": False}
    fix = gps.get("fix")
    if not isinstance(fix, dict):
        return 0.0, {"active": True, "gps_ok": False}

    # Turn announcement is independent of the speed/course gates below, so the
    # UI can warn about an upcoming turn even while the cart is crawling.
    turn = None
    try:
        turn = next_turn_on_route(
            route.get("geometry") or [],
            float(fix["lat_deg"]), float(fix["lon_deg"]),
        )
    except (KeyError, TypeError, ValueError):
        turn = None
    turn_diag = {
        "turn_dir": turn["turn_dir"] if turn else None,
        "turn_dist_m": turn["turn_dist_m"] if turn else None,
        "turn_text": turn_announce_text(turn),
    }

    gps_steer = route_steering_deg_from_gps(fix, route, args.gps_route_lookahead_m)
    if gps_steer is None:
        return 0.0, {"active": True, "gps_ok": False, **turn_diag}
    heading_deg, diag = gps_steer
    desired_deg = heading_deg * STEERING_SIGN
    bias = float(np.clip(
        (desired_deg - seg_steer_deg) * args.gps_route_gain,
        -args.gps_route_max_bias_deg,
        args.gps_route_max_bias_deg,
    ))
    diag.update({
        "active": True,
        "gps_ok": True,
        "route_steer_deg": round(float(desired_deg), 3),
        "bias_deg": round(float(bias), 3),
        **turn_diag,
    })
    return bias, diag


def clrnet_device_from_seg_device(device: str) -> str:
    if device == "cuda":
        return "cuda:0"
    return device


def fill_bev_holes(bev_rgb: np.ndarray, mode: str = "fast-inpaint",
                   radius: int = 3, iterations: int = 3,
                   scale: int = 4) -> np.ndarray:
    """Fill empty BEV pixels.

    The old cv2.INPAINT_TELEA path is visually smooth but costs ~150 ms on
    Thor for a 500x500 BEV and can create blended colors that no longer match
    Cityscapes classes exactly. The default dilation path is intentionally
    cheaper: it copies nearby class-colored pixels into small gaps and keeps
    exact palette colors for the planner's road mask.
    """
    if bev_rgb.size == 0:
        return bev_rgb
    empty = np.all(bev_rgb == 0, axis=-1).astype(np.uint8) * 255
    if empty.max() == 0:
        return bev_rgb
    if mode == "none":
        return bev_rgb
    if mode == "inpaint":
        return cv2.inpaint(bev_rgb, empty, radius, cv2.INPAINT_TELEA)
    if mode == "fast-inpaint":
        scale = max(1, int(scale))
        if scale == 1:
            return cv2.inpaint(bev_rgb, empty, radius, cv2.INPAINT_TELEA)
        h, w = bev_rgb.shape[:2]
        small_w = max(1, w // scale)
        small_h = max(1, h // scale)
        small = cv2.resize(bev_rgb, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
        small_empty = np.all(small == 0, axis=-1).astype(np.uint8) * 255
        if small_empty.max() != 0:
            small = cv2.inpaint(small, small_empty, radius, cv2.INPAINT_TELEA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    if mode == "nearest":
        filled = np.any(bev_rgb != 0, axis=-1)
        if not np.any(filled):
            return bev_rgb
        dt_src = (~filled).astype(np.uint8) * 255
        _, labels = cv2.distanceTransformWithLabels(
            dt_src, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL
        )
        colors = bev_rgb[filled]
        label_colors = np.zeros((int(labels.max()) + 1, 3), dtype=np.uint8)
        if len(colors) >= labels.max():
            label_colors[1:] = colors[: labels.max()]
        else:
            label_colors[labels[filled]] = colors
        return label_colors[labels]

    out = bev_rgb.copy()
    kernel = np.ones((3, 3), dtype=np.uint8)
    remaining = empty.astype(bool)
    for _ in range(max(0, int(iterations))):
        dilated = cv2.dilate(out, kernel, iterations=1)
        filled_now = remaining & np.any(dilated != 0, axis=-1)
        if not np.any(filled_now):
            break
        out[filled_now] = dilated[filled_now]
        remaining[filled_now] = False
        if not np.any(remaining):
            break
    return out


def cloud_bev_rgb(depth_m: np.ndarray, seg_map: np.ndarray,
                  intrinsics: tuple[float, float, float, float],
                  bev_size: int, range_fwd_m: float, range_side_m: float,
                  palette: np.ndarray, min_depth_m: float = 0.3,
                  max_depth_m: float = 80.0, stride: int = 2,
                  splat: int = 1, fill_holes: bool = True,
                  fill_mode: str = "fast-inpaint", inpaint_radius: int = 3,
                  fill_iterations: int = 3, fill_scale: int = 4) -> np.ndarray:
    """Render a top-down BEV by unprojecting depth pixels via pinhole intrinsics
    and splatting each onto a `bev_size x bev_size` canvas, colored by seg class.

    Output is RGB, ego at bottom-center, forward up, half-width = range_side_m.
    Compatible with the homography BEV's coordinate convention so plan_path /
    SteeringEstimator / lookahead all keep working.
    """
    fx, fy, cx, cy = intrinsics
    stride = max(1, int(stride))
    h, w = depth_m.shape
    bev = np.zeros((bev_size, bev_size, 3), dtype=np.uint8)
    if seg_map.shape != (h, w):
        seg_map = cv2.resize(seg_map.astype(np.int32), (w, h),
                             interpolation=cv2.INTER_NEAREST)

    if stride > 1:
        depth_sample = depth_m[::stride, ::stride]
        mask = (
            np.isfinite(depth_sample)
            & (depth_sample >= min_depth_m)
            & (depth_sample <= max_depth_m)
        )
        sv, su = np.nonzero(mask)
        if sv.size == 0:
            return bev
        v = sv * stride
        u = su * stride
        z = depth_sample[sv, su]
    else:
        mask = (
            np.isfinite(depth_m)
            & (depth_m >= min_depth_m)
            & (depth_m <= max_depth_m)
        )
        v, u = np.nonzero(mask)
        if v.size == 0:
            return bev
        z = depth_m[v, u]

    x = (u.astype(np.float32) - cx) * z / fx
    cls = np.clip(seg_map[v, u], 0, len(palette) - 1)
    colors = palette[cls]

    keep = (z > 0) & (z <= range_fwd_m) & (np.abs(x) <= range_side_m)
    if not np.any(keep):
        return bev
    x = x[keep]; z = z[keep]; colors = colors[keep]
    px = ((x / range_side_m * 0.5 + 0.5) * (bev_size - 1)).astype(np.int32)
    py = ((1.0 - z / range_fwd_m) * (bev_size - 1)).astype(np.int32)

    order = np.argsort(z)[::-1]
    px = px[order]; py = py[order]; colors = colors[order]

    if splat <= 0:
        bev[py, px] = colors
    else:
        for dy in range(-splat, splat + 1):
            for dx in range(-splat, splat + 1):
                yy = np.clip(py + dy, 0, bev_size - 1)
                xx = np.clip(px + dx, 0, bev_size - 1)
                bev[yy, xx] = colors

    if fill_holes:
        bev = fill_bev_holes(
            bev,
            mode=fill_mode,
            radius=inpaint_radius,
            iterations=fill_iterations,
            scale=fill_scale,
        )
    return bev


def draw_bev_viz(bev_rgb: np.ndarray, lane_traj: np.ndarray | None,
                 lane_local: np.ndarray | None, rt,
                 occ: "seg_occupancy.PredictedOccupancy | None" = None,
                 bev_geom: "seg_occupancy.BevGeometry | None" = None,
                 brake_corridor: np.ndarray | None = None,
                 brake_01: float = 0.0,
                 stop_active: bool = False,
                 actor_markers: list[dict] | None = None) -> np.ndarray:
    out = bev_rgb.copy()
    # Foot-labelled distance grid. RANGE_FWD covers 0..forward_ft (ego at
    # bottom); RANGE_SIDE covers ±side_ft (ego column at horizontal
    # center). Minor lines every 5 ft, major + label every 10 ft.
    FT_PER_M = 3.28084
    fwd_ft = rt.RANGE_FWD * FT_PER_M
    side_ft = rt.RANGE_SIDE * FT_PER_M
    h, w = out.shape[:2]
    ego_bx, ego_by = w // 2, h - 1
    minor_col = (90, 90, 90)
    major_col = (180, 180, 180)
    label_col = (255, 255, 255)
    label_shadow = (0, 0, 0)

    def put_label(img, text, org):
        cv2.putText(img, text, (org[0] + 1, org[1] + 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_shadow, 2, cv2.LINE_AA)
        cv2.putText(img, text, org,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_col, 1, cv2.LINE_AA)

    # Forward ticks (horizontal lines), every 5 ft up to fwd_ft.
    for ft in range(5, int(fwd_ft) + 1, 5):
        x_m = ft / FT_PER_M
        by = int(round((1.0 - x_m / rt.RANGE_FWD) * h))
        if not (0 <= by < h):
            continue
        is_major = (ft % 10 == 0)
        cv2.line(out, (0, by), (w - 1, by),
                 major_col if is_major else minor_col, 1, cv2.LINE_AA)
        if is_major:
            put_label(out, f"{ft}ft", (ego_bx + 6, by - 3))

    # Lateral ticks (vertical lines), every 5 ft from -side_ft..+side_ft.
    side_max = int(side_ft)
    for ft in range(-side_max, side_max + 1, 5):
        if ft == 0:
            continue
        y_m = ft / FT_PER_M
        bx = int(round((y_m / rt.RANGE_SIDE * 0.5 + 0.5) * w))
        if not (0 <= bx < w):
            continue
        is_major = (ft % 10 == 0)
        cv2.line(out, (bx, 0), (bx, h - 1),
                 major_col if is_major else minor_col, 1, cv2.LINE_AA)
        if is_major:
            put_label(out, f"{ft:+d}ft", (bx + 3, ego_by - 6))

    # Axes through the ego.
    cv2.line(out, (ego_bx, 0), (ego_bx, h - 1), (220, 220, 220), 1, cv2.LINE_AA)
    cv2.line(out, (0, ego_by), (w - 1, ego_by), (220, 220, 220), 1, cv2.LINE_AA)
    put_label(out, "x=forward (ft)", (ego_bx + 6, 16))
    put_label(out, "y=lateral (ft)", (8, ego_by - 6))

    # Predicted obstacle occupancy + detected obstacles (matches live.py's BEV).
    # Currently-detected obstacle cells -> orange so a stationary / just-appeared
    # object is always visible; predicted future-occupancy risk -> red; each
    # moving track's extrapolated path -> light-red polyline.
    if occ is not None:
        cur = occ.current_mask
        if cur is not None and cur.shape[:2] == out.shape[:2] and cur.any():
            out[cur] = (0.40 * out[cur] + 0.60 * np.array([255, 140, 0])).astype(np.uint8)
        risk = occ.risk_mask
        if risk is not None and risk.shape[:2] == out.shape[:2] and risk.any():
            out[risk] = (0.45 * out[risk] + 0.55 * np.array([255, 40, 40])).astype(np.uint8)
        if bev_geom is not None:
            for track in occ.tracks:
                pts = []
                for fwd_m, left_m in track.future_m:
                    bx, by = bev_geom.local_to_bev(fwd_m, left_m)
                    bxi, byi = int(round(float(bx))), int(round(float(by)))
                    if 0 <= bxi < w and 0 <= byi < h:
                        pts.append((bxi, byi))
                if len(pts) > 1:
                    cv2.polylines(out, [np.array(pts, dtype=np.int32)], False,
                                  (255, 80, 80), 2, cv2.LINE_AA)

    if actor_markers and bev_geom is not None:
        for actor in actor_markers[:12]:
            if "fwd_m" not in actor or "left_m" not in actor:
                continue
            bx, by = bev_geom.local_to_bev(actor["fwd_m"], actor["left_m"])
            bxi, byi = int(round(float(bx))), int(round(float(by)))
            if not (0 <= bxi < w and 0 <= byi < h):
                continue
            cv2.circle(out, (bxi, byi), 6, (80, 190, 255), 2, cv2.LINE_AA)
            label = str(actor.get("class", "actor"))[:10]
            cv2.putText(out, label, (bxi + 7, byi - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(out, label, (bxi + 7, byi - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 190, 255), 1, cv2.LINE_AA)

    # Near forward stop corridor the brake is actually judged against.
    if brake_corridor is not None and len(brake_corridor) > 1 and bev_geom is not None:
        half_px = max(3, int(round(ENV_BRAKE_CORRIDOR_HALF_M * bev_geom.px_per_meter_side)))
        cc = np.asarray(brake_corridor, dtype=np.int32)
        corr_col = (255, 60, 60) if stop_active else (90, 220, 120)
        left = cc.copy(); left[:, 0] -= half_px
        right = cc.copy(); right[:, 0] += half_px
        cv2.polylines(out, [left], False, corr_col, 1, cv2.LINE_AA)
        cv2.polylines(out, [right], False, corr_col, 1, cv2.LINE_AA)

    if lane_traj is not None:
        rt.draw_trajectory(out, lane_traj, (255, 255, 0), 3)
    la_point, _ = rt.lookahead_point(lane_local)
    if la_point is not None:
        la_bx = int((la_point[1] / rt.RANGE_SIDE * 0.5 + 0.5) * rt.BEV_SIZE)
        la_by = int((1 - la_point[0] / rt.RANGE_FWD) * rt.BEV_SIZE)
        if 0 <= la_bx < rt.BEV_SIZE and 0 <= la_by < rt.BEV_SIZE:
            cv2.line(out, (ego_bx, ego_by), (la_bx, la_by),
                     (0, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(out, (la_bx, la_by), 6, (0, 255, 255), -1, cv2.LINE_AA)

    # Brake banner so the environment-brake state is obvious even for a
    # stationary obstacle (no future polyline to show).
    if stop_active:
        cv2.putText(out, "STOP", (w // 2 - 60, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 6, cv2.LINE_AA)
        cv2.putText(out, "STOP", (w // 2 - 60, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 40, 40), 3, cv2.LINE_AA)
    elif brake_01 > 0.02:
        txt = f"BRAKE {brake_01:.2f}"
        cv2.putText(out, txt, (w // 2 - 70, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, txt, (w // 2 - 70, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 210, 0), 2, cv2.LINE_AA)
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)


def main() -> None:
    p = argparse.ArgumentParser(description="drive-by-segmentation WebUI sidecar")
    p.add_argument("--frames-dir", type=Path, default=FRAMES_DIR_DEFAULT)
    p.add_argument("--state-file", type=Path, default=STATE_FILE_DEFAULT)
    p.add_argument("--segmentation-map-file", type=Path, default=None,
                   help="Path for the latest raw semantic label map JSON. "
                        "Defaults to <frames-dir>/segmentation_map.json, or "
                        "SEGMENTATION_MAP_FILE when that environment variable is set.")
    p.add_argument("--seg-repo", type=Path, default=SEG_REPO_DEFAULT)
    p.add_argument("--calib", type=Path, default=E2E_CALIB_DEFAULT,
                   help="BEV calibration JSON. Supports the drive-by format or E2E slots.")
    p.add_argument("--camera-slot", default="CAM_FRONT",
                   help="E2E calibration slot to use when --calib has slots.")
    p.add_argument("--camera-index", type=int, default=None,
                   help="Open only this /dev/videoN camera for UVC input.")
    p.add_argument("--active-slug", default=ACTIVE_SLUG,
                   help="Logical name for the active camera frame stream.")
    p.add_argument("--camera-height-m", type=float, default=None,
                   help="Ground height for E2E BEV projection when the E2E origin is camera-center.")
    p.add_argument("--model", default="b0", choices=("b0", "b2", "b5"))
    p.add_argument("--seg-input-size", type=int,
                   default=int(os.environ.get("SEGMENTATION_INPUT_SIZE", "0")),
                   help="Resize SegFormer input to NxN before inference. "
                        "0 keeps the model processor default.")
    p.add_argument("--device", default=None)
    p.add_argument("--publish-hz", type=float, default=PUBLISH_HZ_DEFAULT)
    p.add_argument("--infer-hz", type=float, default=INFER_HZ_DEFAULT)
    p.add_argument("--video", default=None)
    p.add_argument("--no-loop", action="store_true")
    p.add_argument("--max-scan", type=int, default=16)
    p.add_argument("--source", default="uvc", choices=("uvc", "realsense"),
                   help="Active camera source.")
    p.add_argument("--bev-mode", default="homography", choices=("homography", "depth"),
                   help="BEV projection mode. 'homography' is the original "
                        "drive-by-segmentation calibrated ground-plane projection; "
                        "'depth' uses RealSense depth unprojection.")
    p.add_argument("--rs-width", type=int, default=640)
    p.add_argument("--rs-height", type=int, default=480)
    p.add_argument("--rs-fps", type=int, default=30)
    p.add_argument("--no-depth", action="store_true",
                   help="With --source realsense, skip the depth stream "
                        "(color only). Forces --bev-mode homography.")
    p.add_argument("--bev-splat", type=int, default=1,
                   help="Half-width (px) of the box used to splat each cloud point in the BEV.")
    p.add_argument("--bev-depth-stride", type=int, default=2,
                   help="Use every Nth RealSense depth pixel for BEV projection. "
                        "Higher values are faster but less dense.")
    p.add_argument("--no-bev-fill", action="store_true",
                   help="Disable inpaint-based hole filling on the cloud BEV.")
    p.add_argument("--bev-fill-mode", default="fast-inpaint",
                   choices=("fast-inpaint", "dilate", "nearest", "inpaint", "none"),
                   help="How to fill sparse cloud BEV holes. 'dilate' is fast; "
                        "'inpaint' is the old smooth but slow path.")
    p.add_argument("--bev-fill-iterations", type=int, default=3,
                   help="Number of 3x3 dilation passes for --bev-fill-mode dilate.")
    p.add_argument("--bev-fill-scale", type=int, default=4,
                   help="Downsample factor for --bev-fill-mode fast-inpaint.")
    p.add_argument("--bev-inpaint-radius", type=int, default=3,
                   help="cv2.inpaint radius used to fill empty BEV pixels.")
    p.add_argument("--profile-every", type=int, default=0,
                   help="Print one latency breakdown every N inference frames. "
                        "0 keeps profiling in state only.")
    p.add_argument("--ego-state-file", type=Path, default=EGO_STATE_FILE_DEFAULT)
    p.add_argument("--constant-speed", action="store_true",
                   help="Disable adaptive speed control (PI loop, launch ramp, "
                        "stiction punch, brake). Hold a flat open-loop pedal pot "
                        "at constant_gas_for_mph(--target-mph) the whole time.")
    p.add_argument("--gps-state-file", type=Path, default=GPS_STATE_FILE_DEFAULT)
    p.add_argument("--route-file", type=Path, default=NAV_ROUTE_FILE_DEFAULT)
    p.add_argument("--gps-route-lookahead-m", type=float, default=GPS_ROUTE_LOOKAHEAD_M)
    p.add_argument("--gps-route-gain", type=float, default=GPS_ROUTE_GAIN)
    p.add_argument("--gps-route-max-bias-deg", type=float, default=GPS_ROUTE_MAX_BIAS_DEG)
    p.add_argument("--no-actor-neutralization", action="store_true",
                   help="Do not fill pedestrian/rider/bicycle footprints back "
                        "into the BEV road mask before path planning.")
    p.add_argument("--no-actor-detector", action="store_true",
                   help="Disable the optional YOLO person/bicycle detector. "
                        "Semantic actor tracking still runs.")
    p.add_argument("--actor-detector-model",
                   default=os.environ.get("CADDY_ACTOR_DETECTOR_MODEL", "yolo11n.pt"),
                   help="Ultralytics YOLO model for optional person/bicycle detections.")
    p.add_argument("--actor-detector-conf", type=float, default=0.25)
    p.add_argument("--actor-detector-imgsz", type=int, default=640)
    p.add_argument("--actor-detector-hz", type=float, default=5.0)
    p.add_argument("--no-protective-stop", action="store_true",
                   help="Disable predicted-occupancy environment braking. "
                        "Intended only for bench tests.")
    p.add_argument(
        "--target-mph",
        type=float,
        default=TARGET_MPH,
        help=f"Open-loop constant speed target in MPH (default {TARGET_MPH:g}).",
    )
    p.add_argument(
        "--no-fast", action="store_true",
        help="Use the original scipy-heavy create_bev + lane_aware_centerline_path "
             "instead of the cached / cv2-accelerated versions in seg_fast.",
    )
    p.add_argument(
        "--no-clrnet", action="store_true",
        help="Disable the CLRerNet lane override and use segmentation only.",
    )
    p.add_argument("--clrnet-config", default=None)
    p.add_argument("--clrnet-ckpt", default=None)
    p.add_argument("--clrnet-device", default=None)
    args = p.parse_args()
    if args.segmentation_map_file is None:
        args.segmentation_map_file = SEGMENTATION_MAP_FILE_DEFAULT
        if "SEGMENTATION_MAP_FILE" not in os.environ:
            args.segmentation_map_file = args.frames_dir / "segmentation_map.json"
    args.target_mph = max(0.0, float(args.target_mph))
    args.gps_route_lookahead_m = float(np.clip(args.gps_route_lookahead_m, 2.0, 20.0))
    args.gps_route_gain = float(np.clip(args.gps_route_gain, 0.0, 1.0))
    # Ceiling is the steering-column travel limit, not 90°, so a strong route
    # authority can actually pull the cart through a turn instead of saturating.
    args.gps_route_max_bias_deg = float(np.clip(args.gps_route_max_bias_deg, 0.0, 270.0))
    args.actor_detector_hz = max(0.0, float(args.actor_detector_hz))
    target_gas_ff = constant_gas_for_mph(args.target_mph)
    speed_ctrl = SpeedController()
    # Predicted future-occupancy obstacle tracker + brake smoother (replaces the
    # old corridor protective-stop + UniAD merge). bev_geom is built once the BEV
    # ranges are resolved from the calibration below.
    occupancy_tracker = seg_occupancy.PredictedOccupancyTracker()
    brake_smoother = seg_occupancy.PedalCommandSmoother()
    bev_geom: seg_occupancy.BevGeometry | None = None
    last_occ_update_s = time.monotonic()
    launch_start_t: float | None = None
    stuck_since: float | None = None
    actor_detector = YoloActorDetector(
        args.actor_detector_model,
        args.actor_detector_conf,
        args.actor_detector_imgsz,
        enabled=not args.no_actor_detector,
    )
    if actor_detector.enabled:
        status = actor_detector.status()
        if actor_detector.available:
            print(
                f"[actors] detector ready model={status['model']} "
                f"classes={','.join(status['classes'])}",
                flush=True,
            )
        else:
            print(
                f"[actors] detector unavailable, using semantic actor tracks only: "
                f"{status.get('error')}",
                flush=True,
            )

    seg_live, rt, plan_path, create_bev, create_overlay, colors = (
        import_segmentation_stack(args.seg_repo)
    )

    import torch

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    calib, e2e_slot = load_bev_calibration(args)
    if e2e_slot is not None:
        args.active_slug = str(e2e_slot.get("physical_position") or args.active_slug)
        if args.camera_index is None:
            args.camera_index = camera_index_from_source(e2e_slot.get("source"))
        print(
            f"[calib] E2E slot={e2e_slot.get('slot')} "
            f"source={e2e_slot.get('source')} active={args.active_slug} "
            f"height_m={calib['extrinsics']['height_m']:.3f}",
            flush=True,
        )

    bev_range = calib.get("bev_range", {})
    rt.RANGE_FWD = bev_range.get("forward_ft", 50) * rt.FT_TO_M
    rt.RANGE_SIDE = bev_range.get("side_ft", 25) * rt.FT_TO_M
    road_width_ft = float(calib.get("road_width_ft", 20.0))
    # Occupancy geometry matches the BEV the planner draws into, so the risk
    # mask lines up 1:1 with the planned-trajectory polyline.
    bev_geom = seg_occupancy.BevGeometry.from_ranges(
        rt.BEV_SIZE, rt.RANGE_FWD, rt.RANGE_SIDE
    )

    readers = make_sources(args)
    for r in readers:
        r.start()

    active_slug = args.active_slug
    slug_to_reader = {getattr(r, "slug", active_slug): r for r in readers}
    active_reader = slug_to_reader.get(active_slug)
    if active_reader is None:
        raise RuntimeError(f"active camera {active_slug} not available")

    proc, model = seg_live.load_segformer(args.model, device)
    seg_input_size = configure_segformer_processor(proc, int(args.seg_input_size))
    if seg_input_size is not None:
        print(
            f"[seg] processor input resized to {seg_input_size}x{seg_input_size} "
            f"for faster CPU inference.",
            flush=True,
        )
    steer_est = rt.SteeringEstimator()
    clrnet = None
    clrnet_runner = None
    if not args.no_clrnet:
        try:
            clrnet = import_clrnet_stack()
            clrnet_config = args.clrnet_config or clrnet.CLRNET_CONFIG
            clrnet_ckpt = args.clrnet_ckpt or clrnet.CLRNET_CKPT
            clrnet_device = args.clrnet_device or clrnet_device_from_seg_device(device)
            clrnet_runner = clrnet.CLRerNetRunner(clrnet_config, clrnet_ckpt, device=clrnet_device)
        except Exception as e:
            print(
                f"[clrnet] unavailable, continuing with segmentation only: "
                f"{type(e).__name__}: {e}",
                flush=True,
            )
            clrnet = None
            clrnet_runner = None

    palette = np.array(colors, dtype=np.uint8)
    road_color = np.array(colors[0], dtype=np.uint8)
    grid_color = np.clip(road_color.astype(np.int16) + 35, 0, 255).astype(np.uint8)
    grid2_color = np.clip(road_color.astype(np.int16) + 70, 0, 255).astype(np.uint8)

    publish_period = 1.0 / max(args.publish_hz, 1e-3)
    infer_period = 1.0 / args.infer_hz if args.infer_hz > 0 else 0.0
    next_publish_t = 0.0
    next_infer_t = 0.0
    last_log_t = 0.0
    infer_times: list[float] = []

    bev_remap: seg_fast.BevRemap | None = None
    bev_projector: BevImageProjector | None = None
    use_fast = not args.no_fast and args.bev_mode == "homography"

    latest_overlay_bgr: np.ndarray | None = None
    latest_bev_bgr: np.ndarray | None = None
    latest_seg_map: np.ndarray | None = None
    latest_path: list[list[float]] = []
    latest_steer_raw = 0.0
    latest_steer_base = 0.0
    latest_steer_filtered = 0.0
    latest_lookahead_m = 0.0
    latest_ego_speed_mph = 0.0
    latest_ego_speed_ok = False
    latest_speed_setpoint_mph = args.target_mph
    latest_target_gas = target_gas_ff
    latest_gas_trim = 0.0
    latest_target_brake = 0.0
    latest_steer_source = "segmentation"
    latest_clrnet_lanes: list = []
    latest_clrnet_steer_state: dict = {
        "centerline": [],
        "lookahead": None,
        "lateral_err": 0.0,
        "steering_deg": 0.0,
    }
    latest_clrnet_override = False
    latest_clrnet_steer_raw = 0.0
    latest_clrnet_steer_filtered = 0.0
    latest_clrnet_fresh_count = 0
    latest_clrnet_confidences: list[float] = []
    latest_clrnet_overlay_bgr: np.ndarray | None = None
    latest_gps_route: dict = {"active": False}
    latest_env_brake_01 = 0.0
    latest_occupancy: "seg_occupancy.PredictedOccupancy | None" = None
    latest_collision: dict | None = None
    latest_image_brake: dict | None = None
    latest_brake_corridor: np.ndarray | None = None
    latest_actor_neutralization: dict = {
        "enabled": not args.no_actor_neutralization,
        "mode": "dynamic_actor_mask_to_road_context",
        "class_ids": list(DYNAMIC_ACTOR_CLASS_IDS),
        "actor_px": 0,
        "expanded_actor_px": 0,
        "recovered_px": 0,
        "road_px_raw": 0,
        "road_px_planning": 0,
        "actor_px_on_raw_road": 0,
        "actor_px_on_planning_road": 0,
        "actor_px_remaining_nonroad": 0,
    }
    latest_detector_state: dict = actor_detector.status()
    latest_detector_actors: list[dict] = []
    latest_actor_policy: dict = build_actor_policy_state(
        latest_actor_neutralization,
        latest_detector_state,
        latest_detector_actors,
        None,
        None,
        0.0,
    )
    next_detector_t = 0.0
    latest_protective_stop: dict = build_environment_threat(
        None, None, 0.0, enabled=not args.no_protective_stop
    )
    inference_ok = False
    latest_latency_ms: dict[str, float] = {}
    infer_count = 0
    last_processed_frame_count = -1
    latest_camera_frame_count = 0
    latest_camera_last_ok_s = 0.0
    last_camera_fps_sample_t = time.monotonic()
    last_camera_fps_sample_count = 0
    latest_camera_fps = 0.0
    latest_infer_ok_s = 0.0

    try:
        while True:
            now = time.monotonic()

            if now >= next_infer_t:
                stage_start = time.perf_counter()
                stage_last = stage_start
                stage_ms: dict[str, float] = {}

                def mark(name: str) -> None:
                    nonlocal stage_last
                    current = time.perf_counter()
                    stage_ms[name] = (current - stage_last) * 1000.0
                    stage_last = current

                if args.source == "realsense":
                    rs_pair = active_reader.latest_color_depth()
                    frame_bgr = rs_pair[0] if rs_pair is not None else None
                    depth_m = rs_pair[1] if rs_pair is not None else None
                    latest_camera_frame_count = int(active_reader.frame_count)
                    latest_camera_last_ok_s = float(active_reader.last_ok_s)
                else:
                    frame_bgr, latest_camera_frame_count, latest_camera_last_ok_s = (
                        active_reader.latest_with_meta()
                    )
                    depth_m = None
                mark("source_latest_ms")
                have_new_camera_frame = (
                    frame_bgr is not None
                    and latest_camera_frame_count != last_processed_frame_count
                )
                if have_new_camera_frame:
                    last_processed_frame_count = latest_camera_frame_count
                    t0 = time.perf_counter()
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    mark("bgr_to_rgb_ms")
                    seg_map, seg_stage_ms = timed_segment_frame(frame_rgb, proc, model, device)
                    latest_seg_map = seg_map
                    stage_ms.update(seg_stage_ms)
                    stage_last = time.perf_counter()

                    bev_cls_map = None
                    if args.bev_mode == "depth":
                        if args.source != "realsense":
                            raise RuntimeError("--bev-mode depth requires --source realsense")
                        bev_rgb = cloud_bev_rgb(
                            depth_m, seg_map, active_reader.intrinsics,
                            bev_size=rt.BEV_SIZE,
                            range_fwd_m=rt.RANGE_FWD,
                            range_side_m=rt.RANGE_SIDE,
                            palette=palette,
                            stride=args.bev_depth_stride,
                            splat=args.bev_splat,
                            fill_holes=not args.no_bev_fill,
                            fill_mode="none" if args.no_bev_fill else args.bev_fill_mode,
                            inpaint_radius=args.bev_inpaint_radius,
                            fill_iterations=args.bev_fill_iterations,
                            fill_scale=args.bev_fill_scale,
                        )
                    else:
                        if use_fast:
                            if bev_remap is None:
                                img_h, img_w = frame_rgb.shape[:2]
                                bev_remap = seg_fast.build_bev_remap(
                                    calib, img_h, img_w, rt.BEV_SIZE
                                )
                                print(
                                    f"[seg] built BEV remap ({img_w}x{img_h} -> "
                                    f"{rt.BEV_SIZE}x{rt.BEV_SIZE}); cv2-accelerated "
                                    f"planner active.",
                                    flush=True,
                                )
                                if bev_geom is not None:
                                    bev_projector = BevImageProjector(bev_remap, bev_geom)
                            bev_rgb = seg_fast.create_bev_cached(seg_map, palette, bev_remap)
                            bev_cls_map = bev_class_map_cached(seg_map, bev_remap)
                        else:
                            bev_out = create_bev(
                                seg_map, calib, rt.BEV_SIZE, return_class_map=True
                            )
                            bev_rgb, bev_cls_map = bev_out
                    mark("bev_ms")
                    road_mask_raw = (
                        np.all(bev_rgb == road_color, axis=-1)
                        | np.all(bev_rgb == grid_color, axis=-1)
                        | np.all(bev_rgb == grid2_color, axis=-1)
                    )
                    road_mask, latest_actor_neutralization = neutralize_dynamic_actors_for_planning(
                        road_mask_raw,
                        bev_cls_map,
                        enabled=not args.no_actor_neutralization,
                    )
                    mark("road_mask_ms")
                    if use_fast:
                        lane_traj, lane_local = seg_fast.lane_aware_centerline_path_fast(
                            road_mask,
                            bev_size=rt.BEV_SIZE,
                            range_fwd=rt.RANGE_FWD,
                            range_side=rt.RANGE_SIDE,
                            road_width_ft=road_width_ft,
                        )
                    else:
                        lane_traj, lane_local = plan_path(
                            road_mask,
                            bev_size=rt.BEV_SIZE,
                            range_fwd=rt.RANGE_FWD,
                            range_side=rt.RANGE_SIDE,
                            road_mask=road_mask,
                            road_width_ft=road_width_ft,
                        )
                    mark("path_plan_ms")

                    path_ok = lane_local is not None and len(lane_local) >= 4
                    if path_ok:
                        steer_est.update_bev(lane_local)
                        latest_path = [
                            [float(x), float(y)] for x, y in lane_local[:: max(1, len(lane_local) // 40)]
                        ]
                    else:
                        latest_path = []
                    # Keep autosteer "fresh" even when the planner fails — we
                    # hold the last commanded steering so the wheel doesn't
                    # toggle stale on momentary centerline dropouts.
                    inference_ok = True
                    mark("path_state_ms")

                    latest_ego_speed_mph, latest_ego_speed_ok = read_ego_speed_mph(
                        args.ego_state_file
                    )
                    latest_lookahead_m = adaptive_lookahead_m(latest_ego_speed_mph, latest_ego_speed_ok)
                    detector_detections_m: list[seg_occupancy.MotionDetection] = []
                    if (
                        actor_detector.available
                        and args.actor_detector_hz > 0.0
                        and now >= next_detector_t
                    ):
                        next_detector_t = now + 1.0 / args.actor_detector_hz
                        raw_actors, latest_detector_state = actor_detector.detect(frame_bgr)
                        latest_detector_actors, detector_detections_m = project_detector_actors_to_bev(
                            raw_actors, bev_projector
                        )
                        latest_detector_state["projected_count"] = sum(
                            1 for a in latest_detector_actors if "fwd_m" in a
                        )
                    else:
                        latest_detector_state = actor_detector.status()
                        latest_detector_state["count"] = len(latest_detector_actors)
                        latest_detector_state["projected_count"] = sum(
                            1 for a in latest_detector_actors if "fwd_m" in a
                        )
                    mark("actor_detector_ms")
                    # Per-object SPACE-TIME collision braking. Obstacles in the
                    # BEV class map are tracked (position + velocity); we brake
                    # only if the cart and an object are predicted to occupy the
                    # same place at the same time — a person off to the side or a
                    # fast crosser that clears the path does NOT stop the cart.
                    # Graded by time-to-collision, with a hard-stop floor for
                    # anything close and dead-ahead. (seg_occupancy.py, ported
                    # from drive-by-segmentation's unified-planner live.py.)
                    env_brake_target = 0.0
                    latest_collision = None
                    latest_brake_corridor = None
                    protective_enabled = not args.no_protective_stop
                    if protective_enabled and bev_cls_map is not None and bev_geom is not None:
                        occ_now = time.monotonic()
                        occ_dt = float(np.clip(occ_now - last_occ_update_s, 0.02, 0.5))
                        last_occ_update_s = occ_now
                        ego_mps = (
                            latest_ego_speed_mph * 0.44704 if latest_ego_speed_ok else 0.0
                        )
                        latest_occupancy = occupancy_tracker.update(
                            bev_cls_map,
                            bev_geom,
                            dt_s=occ_dt,
                            ego_speed_mps=ego_mps,
                            extra_detections_m=detector_detections_m,
                            use_segmentation_obstacles=True,
                        )
                        # Assume the cart will reach its target speed even if
                        # currently stopped, so it won't launch into a predicted
                        # collision while parked.
                        plan_mps = max(ego_mps, args.target_mph * 0.44704)
                        latest_collision = seg_occupancy.evaluate_collision_brake(
                            latest_occupancy.all_tracks,
                            plan_mps,
                            horizon_s=ENV_BRAKE_HORIZON_S,
                            hard_ttc_s=ENV_BRAKE_HARD_TTC_S,
                            near_stop_m=ENV_BRAKE_NEAR_STOP_M,
                            corridor_half_m=ENV_BRAKE_CORRIDOR_HALF_M,
                            object_radius_m=ENV_BRAKE_OBJECT_RADIUS_M,
                        )
                        latest_image_brake = {
                            "enabled": False,
                            "brake_01": 0.0,
                            "coverage": 0.0,
                        }
                        env_brake_target = float(latest_collision["brake_01"])
                        # Drawn on the BEV tile to show the hard-stop near zone.
                        latest_brake_corridor = forward_stop_corridor_bev(
                            bev_geom, ENV_BRAKE_NEAR_STOP_M
                        )
                    else:
                        latest_occupancy = None
                        latest_collision = None
                        latest_image_brake = None
                        if not protective_enabled:
                            occupancy_tracker.reset()
                    brake_smoother.step(env_brake_target)
                    _, latest_env_brake_01 = brake_smoother.snapshot()
                    latest_protective_stop = build_environment_threat(
                        latest_collision,
                        latest_image_brake,
                        latest_env_brake_01,
                        enabled=protective_enabled,
                    )
                    latest_actor_policy = build_actor_policy_state(
                        latest_actor_neutralization,
                        latest_detector_state,
                        latest_detector_actors,
                        latest_occupancy,
                        latest_collision,
                        latest_env_brake_01,
                    )
                    mark("protective_stop_ms")
                    if args.constant_speed:
                        # Adaptive speed disabled: hold a fixed open-loop pedal
                        # pot at the target-mph feed-forward gas. No launch ramp,
                        # no PI trim, no stiction punch, no brake — whatever
                        # constant_gas_for_mph(target) maps to, applied flat.
                        latest_speed_setpoint_mph = args.target_mph
                        latest_target_gas = target_gas_ff
                        latest_gas_trim = 0.0
                        latest_target_brake = 0.0
                    else:
                        if launch_start_t is None:
                            launch_start_t = time.monotonic()
                        ramp_age = time.monotonic() - launch_start_t
                        if args.target_mph > 1e-3 and SPEED_SETPOINT_RAMP_MPH_S > 0.0:
                            latest_speed_setpoint_mph = min(
                                args.target_mph,
                                ramp_age * SPEED_SETPOINT_RAMP_MPH_S,
                            )
                        else:
                            latest_speed_setpoint_mph = args.target_mph
                        speed_setpoint_gas_ff = constant_gas_for_mph(
                            latest_speed_setpoint_mph
                        )
                        if LAUNCH_RAMP_S > 0.0 and ramp_age < LAUNCH_RAMP_S:
                            frac = max(0.0, ramp_age / LAUNCH_RAMP_S)
                            gas_ceiling = LAUNCH_GAS_MIN + frac * max(
                                0.0, target_gas_ff - LAUNCH_GAS_MIN
                            )
                        else:
                            gas_ceiling = 1.0
                        latest_target_gas, latest_gas_trim, latest_target_brake = speed_ctrl.step(
                            latest_speed_setpoint_mph,
                            latest_ego_speed_mph,
                            latest_ego_speed_ok,
                            speed_setpoint_gas_ff,
                            gas_ceiling=gas_ceiling,
                        )
                        # Stuck-detector: if we're still essentially parked after
                        # the launch ramp ends, override the controller with
                        # stiction-break gas to get the wheels rolling. Once ego
                        # > STICTION_EGO_MPH the PI takes over and pulls back.
                        now_mono = time.monotonic()
                        if (latest_ego_speed_ok
                                and ramp_age > LAUNCH_RAMP_S
                                and args.target_mph > STICTION_EGO_MPH
                                and latest_ego_speed_mph < STICTION_EGO_MPH):
                            if stuck_since is None:
                                stuck_since = now_mono
                            elif now_mono - stuck_since > STICTION_STUCK_S:
                                latest_target_gas = speed_ctrl.slew_gas(
                                    max(latest_target_gas, STICTION_GAS_BREAK),
                                    1.0 / max(1.0, args.publish_hz),
                                )
                                latest_target_brake = 0.0
                        else:
                            stuck_since = None
                    # Environment brake overrides cruise: blend the smoothed
                    # occupancy/VRU brake fraction into the pedal pot (max with
                    # any overspeed brake) and cut gas once it's meaningfully
                    # engaged (coast-then-brake).
                    if latest_env_brake_01 > 1e-3:
                        latest_target_brake = max(
                            float(latest_target_brake),
                            latest_env_brake_01 * float(BRAKE_POT_MAX),
                        )
                        if latest_env_brake_01 >= ENV_BRAKE_GAS_CUT_FRAC:
                            latest_target_gas = 0.0
                            latest_gas_trim = 0.0
                            # Do not let the PI speed controller accumulate
                            # throttle demand while an obstacle is holding the
                            # cart back; otherwise release can surge.
                            speed_ctrl.reset()
                    # Hard protective stop: hold the last steering command — don't
                    # let the road-mask centerline swerve around the obstacle while
                    # we brake to a stop.
                    hold_steering = bool(
                        ENV_BRAKE_FREEZE_STEER and latest_protective_stop.get("active")
                    )
                    latest_steer_source = "segmentation"
                    latest_clrnet_override = False
                    if hold_steering:
                        latest_steer_source = "protective_hold"
                        latest_gps_route = {"active": False}
                    elif path_ok:
                        heading_steer = lookahead_heading_steering_deg(lane_local, latest_lookahead_m)
                        if heading_steer is not None:
                            seg_steer_base = heading_steer * STEERING_SIGN
                        else:
                            seg_steer_base = float(steer_est.steering_deg) * STEERING_SIGN
                        route_bias, latest_gps_route = gps_route_bias_deg(
                            seg_steer_base, args
                        )
                        latest_steer_base = float(np.clip(
                            seg_steer_base + route_bias,
                            -270.0,
                            270.0,
                        ))
                        if latest_gps_route.get("gps_ok") and abs(route_bias) > 1e-3:
                            latest_steer_source = "segmentation+gps"
                        latest_steer_filtered = (
                            STEERING_EMA * latest_steer_base
                            + (1.0 - STEERING_EMA) * latest_steer_filtered
                        )
                        latest_steer_raw = float(np.clip(latest_steer_filtered, -270.0, 270.0))
                    else:
                        latest_gps_route = {"active": False}
                    # else: no path — hold latest_steer_* at their previous values.

                    if clrnet_runner is not None:
                        try:
                            lanes = clrnet_runner.infer(frame_bgr)
                        except Exception as e:
                            print(
                                f"[clrnet] infer failed: {type(e).__name__}: {e}",
                                flush=True,
                            )
                            lanes = []
                        latest_clrnet_lanes = lanes
                        latest_clrnet_confidences = sorted(
                            [float(l.get("score", 0.0)) for l in lanes],
                            reverse=True,
                        )
                        latest_clrnet_steer_state = {
                            "centerline": [],
                            "lookahead": None,
                            "lateral_err": 0.0,
                            "steering_deg": 0.0,
                        }
                        fresh_lanes = [
                            l for l in lanes
                            if float(l.get("score", 0.0)) >= CLRNET_CONF_THRESHOLD
                        ]
                        latest_clrnet_fresh_count = len(fresh_lanes)
                        if fresh_lanes:
                            chosen_left = clrnet.best_on_side(fresh_lanes, "left")
                            chosen_right = clrnet.best_on_side(fresh_lanes, "right")
                            if chosen_left is None and chosen_right is not None:
                                chosen_left = clrnet._mirror_lane(chosen_right)
                            if chosen_right is None and chosen_left is not None:
                                chosen_right = clrnet._mirror_lane(chosen_left)
                            steer_state = clrnet.compute_steering(chosen_left, chosen_right)
                            if steer_state.get("centerline") and not hold_steering:
                                latest_clrnet_steer_state = steer_state
                                geom_deg = -float(steer_state.get("steering_deg", 0.0))
                                latest_clrnet_steer_raw = float(np.clip(
                                    geom_deg * clrnet.STEER_AMP,
                                    -clrnet.STEER_CLAMP_DEG,
                                    clrnet.STEER_CLAMP_DEG,
                                ))
                                latest_clrnet_steer_filtered = (
                                    clrnet.STEER_ALPHA * latest_clrnet_steer_filtered
                                    + (1.0 - clrnet.STEER_ALPHA) * latest_clrnet_steer_raw
                                )
                                latest_steer_raw = latest_clrnet_steer_filtered
                                latest_steer_base = latest_clrnet_steer_filtered
                                latest_steer_source = "clrnet"
                                latest_clrnet_override = True
                        mark("clrnet_ms")
                    mark("steer_ms")
                    overlay_rgb = create_overlay(frame_rgb, seg_map)
                    latest_overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
                    mark("overlay_ms")
                    latest_bev_bgr = draw_bev_viz(
                        bev_rgb, lane_traj, lane_local, rt,
                        occ=latest_occupancy, bev_geom=bev_geom,
                        brake_corridor=latest_brake_corridor,
                        brake_01=latest_env_brake_01,
                        stop_active=bool(latest_protective_stop.get("active")),
                        actor_markers=latest_detector_actors,
                    )
                    if clrnet is not None and latest_clrnet_lanes:
                        latest_clrnet_overlay_bgr = clrnet.render_overlay(
                            frame_bgr,
                            latest_clrnet_lanes,
                            latest_clrnet_steer_state,
                            fresh=latest_clrnet_override,
                            mph_target=args.target_mph,
                            mph_actual=latest_ego_speed_mph if latest_ego_speed_ok else None,
                            column_deg=latest_steer_raw,
                        )
                    elif clrnet is not None:
                        latest_clrnet_overlay_bgr = frame_bgr.copy()
                    mark("bev_viz_ms")

                    infer_times.append(time.perf_counter() - t0)
                    if len(infer_times) > 20:
                        infer_times.pop(0)
                    infer_count += 1
                    stage_ms["infer_total_ms"] = (time.perf_counter() - stage_start) * 1000.0
                    latest_latency_ms = {k: round(float(v), 3) for k, v in stage_ms.items()}
                    latest_infer_ok_s = time.monotonic()
                    if args.profile_every > 0 and infer_count % args.profile_every == 0:
                        ordered = " ".join(
                            f"{k}={latest_latency_ms[k]:.1f}"
                            for k in sorted(latest_latency_ms)
                        )
                        print(f"[profile] frame={infer_count} {ordered}", flush=True)
                next_infer_t = now + infer_period

            if now >= next_publish_t:
                publish_start = time.perf_counter()
                jpeg_ms: dict[str, float] = {}
                for r in readers:
                    f = r.latest()
                    if f is not None:
                        t = time.perf_counter()
                        write_jpeg_atomic(args.frames_dir / f"{r.slug}.jpg", f)
                        jpeg_ms[f"jpeg_{r.slug}_ms"] = (time.perf_counter() - t) * 1000.0
                if latest_overlay_bgr is not None:
                    t = time.perf_counter()
                    write_jpeg_atomic(args.frames_dir / "seg.jpg", latest_overlay_bgr)
                    jpeg_ms["jpeg_seg_ms"] = (time.perf_counter() - t) * 1000.0
                if latest_bev_bgr is not None:
                    t = time.perf_counter()
                    write_jpeg_atomic(args.frames_dir / "bev.jpg", latest_bev_bgr, quality=85)
                    jpeg_ms["jpeg_bev_ms"] = (time.perf_counter() - t) * 1000.0
                if latest_clrnet_overlay_bgr is not None:
                    t = time.perf_counter()
                    write_jpeg_atomic(args.frames_dir / "lanes.jpg", latest_clrnet_overlay_bgr, quality=80)
                    jpeg_ms["jpeg_lanes_ms"] = (time.perf_counter() - t) * 1000.0
                if latest_seg_map is not None:
                    t = time.perf_counter()
                    write_json_atomic(
                        args.segmentation_map_file,
                        segmentation_map_payload(
                            latest_seg_map,
                            colors,
                            active_slug,
                            infer_count,
                            f"drive-by-segmentation-segformer-{args.model}",
                        ),
                    )
                    jpeg_ms["json_segmentation_map_ms"] = (time.perf_counter() - t) * 1000.0
                publish_jpeg_total_ms = (time.perf_counter() - publish_start) * 1000.0

                cam_now = time.monotonic()
                cam_dt = cam_now - last_camera_fps_sample_t
                if cam_dt >= 1.0:
                    frame_delta = latest_camera_frame_count - last_camera_fps_sample_count
                    latest_camera_fps = max(0.0, float(frame_delta) / max(cam_dt, 1e-6))
                    last_camera_fps_sample_t = cam_now
                    last_camera_fps_sample_count = latest_camera_frame_count
                camera_age_s = (
                    cam_now - latest_camera_last_ok_s
                    if latest_camera_last_ok_s > 0.0 else float("inf")
                )
                camera_stale = camera_age_s > 0.5
                inference_stale = (cam_now - latest_infer_ok_s) > 0.5 if latest_infer_ok_s else True

                mean_dt = sum(infer_times) / len(infer_times) if infer_times else 0.0
                fps = 0.0 if camera_stale or inference_stale else (1.0 / mean_dt if mean_dt > 0 else 0.0)
                state = {
                    "steer_deg": latest_steer_raw,
                    "steer_deg_raw": latest_steer_raw,
                    "active_cam": active_slug,
                    "inference": inference_ok and not camera_stale and not inference_stale,
                    "viz": (
                        latest_overlay_bgr is not None
                        or latest_bev_bgr is not None
                        or latest_clrnet_overlay_bgr is not None
                    ),
                    "viz_streams": [
                        slug for slug, img in (
                            ("seg", latest_overlay_bgr),
                            ("bev", latest_bev_bgr),
                            ("lanes", latest_clrnet_overlay_bgr),
                        )
                        if img is not None
                    ],
                    "object_count": int(latest_protective_stop.get("objects", 0)),
                    "fps": float(fps),
                    "camera_fps": float(latest_camera_fps),
                    "camera_frame_count": int(latest_camera_frame_count),
                    "camera_age_s": (
                        float(camera_age_s) if math.isfinite(camera_age_s) else None
                    ),
                    "camera_stale": bool(camera_stale),
                    "cams": [r.slug for r in readers],
                    "source": "video" if args.video else "camera",
                    "model": MODEL_NAME,
                    "model_full": f"drive-by-segmentation-segformer-{args.model}",
                    "target_speed_mph": args.target_mph if inference_ok else 0.0,
                    "speed_setpoint_mph": latest_speed_setpoint_mph if inference_ok else 0.0,
                    "ego_speed_mph": latest_ego_speed_mph,
                    "ego_speed_ok": latest_ego_speed_ok,
                    "steer_deg_base": latest_steer_base,
                    "steer_source": latest_steer_source,
                    "steer_lookahead_m": latest_lookahead_m,
                    "target_gas": latest_target_gas if inference_ok else 0.0,
                    "target_gas_ff": target_gas_ff,
                    "target_gas_trim": latest_gas_trim,
                    "speed_mode": (
                        "protective_stop" if latest_protective_stop.get("active")
                        else "constant" if args.constant_speed
                        else ("arkit_pi" if latest_ego_speed_ok else "feedforward_pot")
                    ),
                    "target_brake": latest_target_brake if inference_ok else 0.0,
                    "predicted_path": (
                        []
                        if latest_clrnet_override
                        else (latest_path if inference_ok else [])
                    ),
                    "segmentation": {
                        "bev_mode": args.bev_mode,
                        "latency_ms": latest_latency_ms,
                        "jpeg_ms": {k: round(float(v), 3) for k, v in jpeg_ms.items()},
                        "publish_jpeg_total_ms": round(float(publish_jpeg_total_ms), 3),
                        "infer_count": int(infer_count),
                        "seg_input_size": seg_input_size,
                        "camera_fps": round(float(latest_camera_fps), 3),
                        "camera_frame_count": int(latest_camera_frame_count),
                        "camera_age_s": (
                            round(float(camera_age_s), 3)
                            if math.isfinite(camera_age_s) else None
                        ),
                        "camera_stale": bool(camera_stale),
                        "inference_stale": bool(inference_stale),
                        "clrnet_enabled": bool(clrnet_runner is not None),
                        "clrnet_override": bool(latest_clrnet_override),
                        "clrnet_conf_threshold": float(CLRNET_CONF_THRESHOLD),
                        "clrnet_lanes": len(latest_clrnet_lanes),
                        "clrnet_lanes_above_threshold": int(latest_clrnet_fresh_count),
                        "clrnet_confidences": [
                            round(float(s), 3) for s in latest_clrnet_confidences
                        ],
                        "clrnet_steer_deg": float(latest_clrnet_steer_filtered),
                        "clrnet_steer_deg_raw": float(latest_clrnet_steer_raw),
                        "protective_stop": latest_protective_stop,
                        "actor_policy": latest_actor_policy,
                        "gps_route": latest_gps_route,
                        "map_file": str(args.segmentation_map_file),
                        "map_schema": "caddy.segmentation_map.v1",
                    },
                    "ts": time.time(),
                }
                t = time.perf_counter()
                write_json_atomic(args.state_file, state)
                latest_latency_ms["json_state_ms"] = round((time.perf_counter() - t) * 1000.0, 3)
                next_publish_t = now + publish_period

            if now - last_log_t >= 1.0:
                last_log_t = now
                confs = " ".join(f"{s:.2f}" for s in latest_clrnet_confidences[:6])
                if not confs:
                    confs = "none"
                print(
                    f"[run] frame={infer_count} steer_source={latest_steer_source} "
                    f"clrnet_override={'Y' if latest_clrnet_override else 'N'} "
                    f"lanes={len(latest_clrnet_lanes)} "
                    f"above_{CLRNET_CONF_THRESHOLD:.2f}={latest_clrnet_fresh_count} "
                    f"conf=[{confs}] "
                    f"protective_stop={'Y' if latest_protective_stop.get('active') else 'N'} "
                    f"brake={latest_env_brake_01:.2f} "
                    f"ttc={latest_protective_stop.get('ttc_s')} "
                    f"cov={latest_protective_stop.get('image_coverage', 0.0)} "
                    f"objects={int(latest_protective_stop.get('objects', 0))} "
                    f"actors={latest_actor_policy.get('track_count', 0)} "
                    f"neutralized={latest_actor_neutralization.get('recovered_px', 0)} "
                    f"gps_bias={float(latest_gps_route.get('bias_deg', 0.0)):+5.1f} "
                    f"turn={latest_gps_route.get('turn_text') or '-'} "
                    f"steer={latest_steer_raw:+6.1f}",
                    flush=True,
                )

            time.sleep(0.005)
    finally:
        for r in readers:
            r.stop()


if __name__ == "__main__":
    main()

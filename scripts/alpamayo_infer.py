#!/usr/bin/env python3
"""
alpamayo_infer.py — camera capture + Alpamayo-R1 (remote Modal) steering.

Drop-in alternative to ``autoware_infer.py``. Same physical cameras, same
``/tmp/cart_frames/`` JPEG layout, same ``/tmp/autoware_state.json`` schema —
so ``ps5_drive.py --autosteer`` works unchanged whether the operator
chose Autoware or Alpamayo as the steering brain.

Transport: Modal forward tunnels (raw TCP), not WebSockets. The cart
spawns the Modal `LiveInference.call(request_id)` method, which posts
the container's tunnel address into a partitioned Modal Queue. We pop
that address and open a plain asyncio TCP connection. Wire framing is a
4-byte big-endian length prefix + msgpack body, both directions. Skipping
the WebSocket layer trims ~20-40 ms off each round trip vs. wss://.

Region: us-west (Modal's West-Coast region; closest to NorCal). Set on
the server side; the client only logs whatever the server reports in its
"hello".

Differences from autoware_infer.py:
  * Inference is OFF-DEVICE — frames go through a Modal tunnel to an
    Alpamayo-R1 H100 box and we receive a predicted ego trajectory back.
  * No torch / GPU on the Jetson — this script is pure-Python I/O.
  * Trajectory → steering: real-time tangent replay (see
    trajectory_to_steer_deg). ps5_drive.py consumes the column angle
    directly (AUTOSTEER_GAIN=1.0).
  * No viz tiles published (lanes/depth/seg/objects). The web UI keeps
    the camera tiles + a BEV tile.

Usage:
    /home/caddy/mayo/.venv-client/bin/python scripts/alpamayo_infer.py \
        --app-name alpamayo-live-demo

start.sh launches this when invoked with ``--model alpamayo`` (it picks
the mayo venv-client interpreter automatically — that's the only Python
on this box with the ``modal`` SDK installed; override via
``$ALPAMAYO_PY``).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import struct
import sys
import threading
import time
import uuid
from collections import deque
from pathlib import Path

import cv2
import msgpack
import numpy as np

try:
    import modal  # type: ignore
except ImportError as _e:  # pragma: no cover - surfaced at runtime
    modal = None  # type: ignore[assignment]
    _MODAL_IMPORT_ERROR = _e
else:
    _MODAL_IMPORT_ERROR = None

# EgoSensor lives outside this repo (~/ego_sensor/ego_sensor.py) — it's
# stdlib-only and consumed by both the live UI and now the alpamayo
# websocket client. Add to sys.path rather than vendoring a copy.
_EGO_DIR = os.path.expanduser("~/ego_sensor")
if _EGO_DIR not in sys.path:
    sys.path.insert(0, _EGO_DIR)
try:
    from ego_sensor import EgoSensor, HISTORY_LEN as EGO_HISTORY_LEN  # noqa: E402
except ImportError:
    EgoSensor = None  # type: ignore[assignment]
    EGO_HISTORY_LEN = 16


def build_ego_tensors(history) -> "tuple[np.ndarray, np.ndarray] | None":
    """Convert an EgoSensor history list into the (xyz, rot) pair Alpamayo-R1
    consumes as ``ego_history_xyz`` / ``ego_history_rot``.

    iPhone publisher frame: +X right, +Y up, -Z forward, distances in metres,
    ``yaw_rad`` is rotation about vertical with the vehicle convention used
    in the iOS app. Alpamayo training frame (PhysicalAI-AV): x=forward,
    y=left, z=up. Mapping per-axis::

        Alp.x =  -iPhone.z   (forward = phone -Z)
        Alp.y =  -iPhone.x   (left    = -phone +X)
        Alp.z =  +iPhone.y   (up      =  phone +Y)

    All trajectories are recentered to t0 (the most recent sample) so the
    last position is (0,0,0) and the last rotation is identity, matching
    how ``load_physical_aiavdataset.py`` builds the training tensors.

    Output shapes (no batch dims — the caller adds them):
        xyz: (HISTORY_LEN, 3) float32
        rot: (HISTORY_LEN, 3, 3) float32

    Returns None if history is empty. Pads by replicating the oldest
    sample if fewer than HISTORY_LEN points are available — better to
    feed the model a slightly stale history than zero out half the
    window.
    """
    if not history:
        return None
    if len(history) < EGO_HISTORY_LEN:
        history = [history[0]] * (EGO_HISTORY_LEN - len(history)) + list(history)
    history = history[-EGO_HISTORY_LEN:]

    n = len(history)
    xyz = np.empty((n, 3), dtype=np.float32)
    yaw = np.empty(n, dtype=np.float32)
    for i, s in enumerate(history):
        xyz[i, 0] = -s.z_m   # forward
        xyz[i, 1] = -s.x_m   # left
        xyz[i, 2] =  s.y_m   # up
        yaw[i] = s.yaw_rad

    t0_xyz = xyz[-1].copy()
    t0_yaw = float(yaw[-1])
    c0, s0 = math.cos(-t0_yaw), math.sin(-t0_yaw)
    R0_inv = np.array(
        [[c0, -s0, 0.0], [s0, c0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    delta = xyz - t0_xyz
    xyz_local = delta @ R0_inv.T

    rot_local = np.zeros((n, 3, 3), dtype=np.float32)
    for i in range(n):
        d = float(yaw[i] - t0_yaw)
        c, s = math.cos(d), math.sin(d)
        rot_local[i] = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    return xyz_local, rot_local


FRAMES_DIR_DEFAULT = Path("/tmp/cart_frames")
STATE_FILE_DEFAULT = Path("/tmp/autoware_state.json")

CALIBRATION_DIR = Path("calibration/cameras")
CAMERA_MAPPING_DEFAULT = CALIBRATION_DIR / "camera_mapping.json"
MANUAL_EXTRINSICS_DEFAULT = (
    CALIBRATION_DIR / "REAL_manual_extrinsics_front_reference.json"
)

# Secured PVC-pipe rig, measured 2026-05-20:
#   /dev/video0 = front_right
#   /dev/video2 = front_left
#   /dev/video4 = front
#
# Alpamayo-R1 still expects 4 channel slots:
#   0 front_wide, 1 front_tele, 2 cross_left, 3 cross_right.
# The real rig has one center front camera, so we feed it twice: raw into
# front_wide and center-cropped to ~30° into front_tele.
SLUGS = ("front_right", "front_left", "front")
ALPAMAYO_CHANNEL_SPECS = (
    {"channel": 0, "label": "front_wide", "slug": "front", "target_fov_deg": None},
    {"channel": 1, "label": "front_tele", "slug": "front", "target_fov_deg": 30.0},
    {"channel": 2, "label": "cross_left", "slug": "front_left", "target_fov_deg": None},
    {"channel": 3, "label": "cross_right", "slug": "front_right", "target_fov_deg": None},
)
# Cameras with non-standard install orientation. cv2.flip codes:
#   0 = vertical flip (across x-axis), 1 = horizontal, -1 = both.
CAMERA_ORIENTATION_FIX = {}

# Per-camera FOV correction is no longer applied to the raw PVC-rig
# streams. The lenses are measured as non-fisheye ~74° horizontal at
# 640x480, already narrower than Alpamayo's 120° wide/cross slots. The
# only crop is per-channel: front -> front_tele (~30°) in main().
CAMERA_FOV_CROP_RATIO = {}

CAM_W, CAM_H = 640, 480
JPEG_QUALITY = 72
# Quality for the JPEGs sent to Modal. Was 80; dropped to 60 because
# the cart's effective uplink to Modal us-west is closer to 3-4 Mbps
# (despite a 10 Mbps speedtest), making upload time the dominant
# component of round-trip latency. q60 cuts payload size ~35% with no
# perceptible degradation on the model's training distribution; q40
# would cut another ~30% if needed.
JPEG_QUALITY_SEND = 60
# Tunnel frame cadence. The 10B model is still slower than this, but the
# transport should run independently so the Modal server always has a
# fresh temporal window ready when inference finishes. The server's
# drop-stale reader overwrites old frame batches by design.
SEND_FPS_DEFAULT = 5.0
PUBLISH_HZ_DEFAULT = 15.0        # frames-dir JPEG cadence for the UI
# Steering policy: REPLAY THE TRAJECTORY'S TANGENT ANGLE IN REAL TIME.
# When a fresh prediction lands (last_recv_t resets), t_elapsed = 0 →
# sample the tangent at trajectory[0]. As wall-clock time advances, we
# walk forward along the curve at the same timescale Alpamayo predicted
# (dt = horizon / (T-1)), commanding the wheel to match the curve's
# tangent angle at that instant. A new prediction overwrites the curve
# and the t-clock restarts. No "lookahead distance" — the model's own
# time axis IS the lookahead.
#
# REPLAY_LEAD_S adds a tiny constant offset so the wheel is set for
# *where the cart is heading next*, not where it is right now (purely
# tracking t=0 would mean the wheel reacts a full cycle late).
REPLAY_LEAD_S = 0.2
# REPLAY_MAX_S is set right after TRAJECTORY_HORIZON_S below.
# Map "heading angle to lookahead point" → "steering wheel command".
# A car's steering wheel rotates ~15× more than its road wheels, so a
# 6° geometric heading needs a ~90° wheel rotation to actually make
# the car turn that much. AMP packages that ratio plus our cart-
# specific empirical fudge factor. Combined with AUTOSTEER_GAIN=3.0
# in ps5_drive, a 30° atan2 reading → 90° model out → 270° at the
# wheel (full lock).
STEER_AMP = 15.0
# alpamayo emits the *real* desired column angle directly: trajectory
# tangent (deg) × STEER_AMP (~15:1 steering ratio). STEER_CLAMP_DEG is
# the column's full mechanical range, so ps5_drive can pass this value
# straight through (AUTOSTEER_GAIN=1.0). No more double amplification.
STEER_CLAMP_DEG = 270.0
# EMA smoothing was masking the model's intent and adding lag between
# the on-screen wheel and the trajectory's tangent. Disable: each
# prediction lands as the new target, full strength.
STEER_SMOOTHING = 0.0
BEV_TILE_W, BEV_TILE_H = 640, 480
BEV_RANGE_M = 12.0               # half-extent shown around the ego in the BEV
MODEL_NAME = "alpamayo"          # surfaced to the UI's badges + status pill

# Trajectory horizon — Alpamayo-R1's pred_xy spans this many seconds
# end-to-end (matches the server's hello{"horizon_s": 6.4}). dt_per_point
# = HORIZON_S / (T-1). Used both for speed targeting and for the
# time-based steering replay (see trajectory_angle_at_time).
TRAJECTORY_HORIZON_S = 6.4
# Hard upper bound on how far past the prediction we'll keep replaying
# the same curve when a new one is delayed. After this, we hold the
# trajectory's last tangent — better than extrapolating off the end.
REPLAY_MAX_S = TRAJECTORY_HORIZON_S - 0.2
# Cap on how aggressive the auto-pedal can be while alpamayo drives.
# Both are well below the human-driving caps (GAS_POT_MAX=0.68,
# BRAKE_POT_MAX=0.45 in PRODUCTION/limits.py) so a runaway model can't
# floor the cart — operator takes over with the trigger if more is
# needed.
AUTO_GAS_MAX = 0.20
AUTO_BRAKE_MAX = 0.18
# Speed targeting: cart creeps when the model wants to move, brakes
# proportionally when the model wants to stop. Conservative for the
# first wheel-out — tune on real driving data.
#
# Alpamayo was trained on real-car trajectories so its speed predictions
# easily reach 30+ mph. The cart can't go that fast and we don't want
# to display unrealistic numbers. Clamp at AUTO_TARGET_MAX_MPH and
# scale the gas mapping against AUTO_TARGET_CRUISE_MPH (the speed at
# which we'd want to apply AUTO_GAS_MAX). Both are well below the
# global speed limit; operator can pull R2 to override and accelerate
# faster.
AUTO_TARGET_MAX_MPH = 8.0        # hard ceiling for the displayed target
AUTO_TARGET_CRUISE_MPH = 4.0     # speed we'd cruise at if the model said "go fast"
AUTO_GAS_PER_MPH = AUTO_GAS_MAX / AUTO_TARGET_CRUISE_MPH
AUTO_BRAKE_TARGET_MPH = 0.5      # below this target speed we start applying brake
# Down-sample size for the predicted_path that scene.js renders in 3D.
# 12 points is enough for a smooth Catmull-Rom curve without sending
# excess JSON in /state every poll.
PATH_DOWNSAMPLE_N = 12


# ═══════════════════════════════════════════════════════════════════════════
# Camera capture (mirrors autoware_infer.py to stay binary-compatible)
# ═══════════════════════════════════════════════════════════════════════════

def discover_v4l2_indices(count: int = 4, max_scan: int = 16) -> list[int]:
    """Return the first ``count`` v4l2 indices that produce a frame.

    Same algorithm as autoware_infer.py.discover_v4l2_indices — kept
    duplicated rather than imported so this script stays standalone.
    """
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


def center_crop_zoom(frame: np.ndarray, ratio: float) -> np.ndarray:
    """Center-crop ``frame`` by ``ratio`` (0,1] then resize back to its
    original WxH so the apparent FOV shrinks but the consumer-side
    resolution is unchanged. ``ratio >= 1`` is a no-op (no crop).

    Used to narrow the cart's ~170° fisheye USB cameras down to the
    ~120° FOV the model expects (see CAMERA_FOV_CROP_RATIO).
    """
    if ratio >= 1.0 or ratio <= 0.0:
        return frame
    h, w = frame.shape[:2]
    nh = max(2, int(round(h * ratio)))
    nw = max(2, int(round(w * ratio)))
    y0 = (h - nh) // 2
    x0 = (w - nw) // 2
    cropped = frame[y0:y0 + nh, x0:x0 + nw]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)


def load_camera_mapping(path: Path) -> "dict[str, dict]":
    """Load the REAL camera mapping written during calibration."""
    with path.open() as f:
        data = json.load(f)
    out: dict[str, dict] = {}
    for item in data.get("mappings", []):
        name = item["logical_name"]
        entry = dict(item)
        intr_path = path.parent / item["intrinsics"]
        with intr_path.open() as f:
            intr = json.load(f)
        entry["intrinsics_path"] = str(intr_path)
        entry["intrinsics"] = intr
        out[name] = entry
    return out


def calibrated_hfov_deg(intrinsics: dict) -> float:
    """Horizontal pinhole FOV from calibrated K at the calibration size."""
    w = float(intrinsics["image_width"])
    fx = float(intrinsics["camera_matrix"][0][0])
    return math.degrees(2.0 * math.atan(w / (2.0 * fx)))


def pinhole_crop_ratio_for_fov(source_fov_deg: float, target_fov_deg: float) -> float:
    """Center-crop ratio for pinhole cameras: tan(target/2) / tan(source/2)."""
    if target_fov_deg >= source_fov_deg:
        return 1.0
    return (
        math.tan(math.radians(target_fov_deg / 2.0))
        / math.tan(math.radians(source_fov_deg / 2.0))
    )


def open_camera(index: int) -> "cv2.VideoCapture | None":
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


class CameraReader(threading.Thread):
    """Background grabber, single-slot last-write-wins.

    Identical contract to autoware_infer.CameraReader so this file can be
    diffed against it. Applies per-camera orientation fix at grab time
    so downstream code (Modal payload, JPEG writer) sees the "intended"
    pose.
    """

    def __init__(self, cap: "cv2.VideoCapture", slug: str):
        super().__init__(daemon=True, name=f"cam-{slug}")
        self.cap = cap
        self.slug = slug
        self.flip_code = CAMERA_ORIENTATION_FIX.get(slug)
        # FOV crop ratio: 1.0 (or missing) = no crop. Applied AFTER the
        # orientation flip so the crop stays centered on the cart's
        # forward axis even on upside-down-mounted lenses.
        self.crop_ratio = CAMERA_FOV_CROP_RATIO.get(slug, 1.0)
        self.lock = threading.Lock()
        self.frame: "np.ndarray | None" = None
        self.frame_count = 0
        self.last_ok_s = 0.0
        self._stop = threading.Event()

    def run(self) -> None:
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            if ok and frame is not None:
                if self.flip_code is not None:
                    frame = cv2.flip(frame, self.flip_code)
                if self.crop_ratio < 1.0:
                    frame = center_crop_zoom(frame, self.crop_ratio)
                with self.lock:
                    self.frame = frame
                    self.frame_count += 1
                    self.last_ok_s = time.monotonic()
            else:
                time.sleep(0.01)

    def latest(self) -> "np.ndarray | None":
        with self.lock:
            return None if self.frame is None else self.frame

    def stop(self) -> None:
        self._stop.set()
        self.cap.release()


class VideoReader(threading.Thread):
    """File-backed frame source. Plays at the clip's native fps and loops
    on EOF unless ``loop=False``. No per-slug flip / FOV crop — eval
    clips aren't fisheye-mounted USB cams. Pair with ``_BroadcastView``
    to expose a single source under all four cam slugs."""

    def __init__(self, video_path: str, loop: bool = True):
        super().__init__(daemon=True, name="video-reader")
        self.video_path = video_path
        self.loop = loop
        self.lock = threading.Lock()
        self.frame: "np.ndarray | None" = None
        self.frame_count = 0
        self.last_ok_s = 0.0
        self._stop = threading.Event()
        self.eof = False

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
                self.eof = True
                return

    def latest(self) -> "np.ndarray | None":
        with self.lock:
            return self.frame

    def stop(self) -> None:
        self._stop.set()


class _BroadcastView:
    """Slug-aware wrapper that delegates ``latest`` to a shared source.
    Lets one VideoReader fill all four ALPAMAYO channel slots."""

    def __init__(self, source: VideoReader, slug: str):
        self.source = source
        self.slug = slug

    def latest(self):
        return self.source.latest()

    @property
    def frame_count(self):
        return self.source.frame_count

    @property
    def last_ok_s(self):
        return self.source.last_ok_s


# ═══════════════════════════════════════════════════════════════════════════
# Atomic file writers (rename is atomic on same filesystem)
# ═══════════════════════════════════════════════════════════════════════════

def write_jpeg_atomic(path: Path, frame: np.ndarray,
                       quality: int = JPEG_QUALITY) -> None:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return
    tmp = path.with_suffix(".jpg.tmp")
    tmp.write_bytes(buf.tobytes())
    os.replace(tmp, path)


def write_state_atomic(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w") as f:
        json.dump(data, f)
    os.replace(tmp, path)


# ═══════════════════════════════════════════════════════════════════════════
# BEV (top-down trajectory) renderer for the web UI viz tile
# ═══════════════════════════════════════════════════════════════════════════

def render_bev(pred_xy: "np.ndarray | None",
               w: int = BEV_TILE_W, h: int = BEV_TILE_H,
               half_range_m: float = BEV_RANGE_M) -> np.ndarray:
    """Render a top-down trajectory tile as a BGR image for /tmp/cart_frames.

    pred_xy is in ego-frame meters: x=forward (up in BEV), y=left (left in
    BEV). Pure cv2 — no matplotlib — to keep render time well under 10 ms
    per frame and avoid pulling another heavy import.
    """
    img = np.full((h, w, 3), 18, dtype=np.uint8)  # near-black canvas

    cx, cy = w // 2, h // 2
    # Pixels per meter — same scale on both axes so the trajectory keeps
    # its true geometry (no anamorphic squish).
    pxm = min(w, h) / (2.0 * half_range_m)

    def to_px(xf: float, yl: float) -> tuple[int, int]:
        # Forward (xf) maps to "up" → smaller pixel y. Left (yl) maps to
        # "left" → smaller pixel x.
        u = int(round(cx - yl * pxm))
        v = int(round(cy - xf * pxm))
        return u, v

    # Grid lines on a feet ruler — minor every 5 ft, major (brighter
    # + labelled) every 10 ft. half_range_m=12 m ≈ 39.4 ft, so we
    # walk ±35 ft in 5 ft steps. Labels are placed along the +x axis
    # (vertical center column, marking forward distance) and along the
    # +y axis (horizontal center row, marking lateral distance).
    FT_PER_M = 3.28084
    half_range_ft = half_range_m * FT_PER_M
    ft_step = 5
    ft_major = 10
    for ft in range(-int(half_range_ft // ft_step) * ft_step,
                    int(half_range_ft // ft_step) * ft_step + 1, ft_step):
        if ft == 0:
            continue
        m = ft / FT_PER_M
        is_major = (ft % ft_major == 0)
        col = (95, 95, 95) if is_major else (55, 55, 55)
        # vertical line at lateral offset m (constant y)
        u_top, _ = to_px(half_range_m, m)
        u_bot, _ = to_px(-half_range_m, m)
        cv2.line(img, (u_top, 0), (u_bot, h - 1), col, 1, cv2.LINE_AA)
        # horizontal line at forward distance m (constant x)
        _, v = to_px(m, half_range_m)
        cv2.line(img, (0, v), (w - 1, v), col, 1, cv2.LINE_AA)
        # Labels on majors only — keep the ruler readable.
        if is_major:
            label = f"{ft:+d}ft"
            # horizontal axis label (lateral) below the center row
            ux, _ = to_px(0.0, m)
            cv2.putText(img, label, (ux + 3, cy + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                        (150, 150, 150), 1, cv2.LINE_AA)
            # vertical axis label (forward) right of the center column
            _, vy = to_px(m, 0.0)
            cv2.putText(img, label, (cx + 4, vy - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                        (150, 150, 150), 1, cv2.LINE_AA)
    # axes
    cv2.line(img, (cx, 0), (cx, h - 1), (170, 170, 170), 1, cv2.LINE_AA)
    cv2.line(img, (0, cy), (w - 1, cy), (170, 170, 170), 1, cv2.LINE_AA)

    # Ego marker: filled triangle pointing up.
    pts = np.array([
        [cx, cy - 12],
        [cx - 8, cy + 8],
        [cx + 8, cy + 8],
    ], dtype=np.int32)
    cv2.fillPoly(img, [pts], (255, 255, 255))

    # Trajectory polyline + endpoint dot.
    if pred_xy is not None and pred_xy.shape[0] >= 2:
        path_px = []
        for i in range(pred_xy.shape[0]):
            xf, yl = float(pred_xy[i, 0]), float(pred_xy[i, 1])
            path_px.append(to_px(xf, yl))
        # Color: alpamayo blue (BGR).
        cv2.polylines(img, [np.array(path_px, dtype=np.int32)], False,
                       (255, 196, 80), 3, cv2.LINE_AA)
        cv2.circle(img, path_px[-1], 6, (255, 220, 100), -1, cv2.LINE_AA)

    # Title strip
    title = "Alpamayo BEV  (x=forward, y=left, ft)"
    cv2.putText(img, title, (12, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)
    return img


# ═══════════════════════════════════════════════════════════════════════════
# Trajectory → steering
# ═══════════════════════════════════════════════════════════════════════════

def trajectory_to_speed_targets(pred_xy: np.ndarray) -> tuple[float, float, float]:
    """Derive (target_speed_mph, target_gas, target_brake) from the trajectory.

    Uses the average velocity over the first 3 trajectory segments as the
    near-term target speed (in mph), then maps:
      • target > AUTO_BRAKE_TARGET_MPH → gas proportional to target,
        capped at AUTO_GAS_MAX; no brake.
      • target ≤ AUTO_BRAKE_TARGET_MPH → no gas; brake proportional to
        how slow the model wants to go, capped at AUTO_BRAKE_MAX.

    Conservative on purpose — this is the first time alpamayo touches the
    pedals. Operator pulls the R2 trigger to override and accelerate
    harder; pulls L2 to brake harder.
    """
    if pred_xy is None or pred_xy.shape[0] < 2:
        return 0.0, 0.0, 0.0
    T = pred_xy.shape[0]
    dt = TRAJECTORY_HORIZON_S / max(T - 1, 1)
    # Distance covered over the first ``k`` segments (smoothing out
    # single-frame jitter), divided by elapsed time.
    k = min(3, T - 1)
    seg = pred_xy[:k + 1]
    dx = float(seg[-1, 0] - seg[0, 0])
    dy = float(seg[-1, 1] - seg[0, 1])
    dist_m = math.hypot(dx, dy)
    speed_mps = dist_m / max(k * dt, 1e-3)
    # If the trajectory bends backward (dx <= 0) the model is asking us
    # to reverse — the cart's pedal API has no reverse, so treat it as
    # "stop" and let the operator handle the rest.
    if dx <= 0.05:
        speed_mps = 0.0
    speed_mph = speed_mps * 2.23694
    # Clamp so a "highway-speed" prediction doesn't show 39 mph on a
    # cart that physically tops out around 8 mph. The gas mapping uses
    # the clamped value so the cart applies a smooth, capped pedal.
    speed_mph = min(speed_mph, AUTO_TARGET_MAX_MPH)
    if speed_mph > AUTO_BRAKE_TARGET_MPH:
        gas = min(speed_mph * AUTO_GAS_PER_MPH, AUTO_GAS_MAX)
        brake = 0.0
    else:
        gas = 0.0
        # Closer to 0 mph target → more brake.
        deficit = max(0.0, AUTO_BRAKE_TARGET_MPH - speed_mph)
        brake = min(deficit * (AUTO_BRAKE_MAX / AUTO_BRAKE_TARGET_MPH),
                     AUTO_BRAKE_MAX)
    return float(speed_mph), float(gas), float(brake)


def downsample_path(pred_xy: np.ndarray, n: int = PATH_DOWNSAMPLE_N) -> list:
    """Pick ``n`` evenly-spaced points along the trajectory for the JSON
    state file. Returns a list of [x_forward_m, y_left_m] floats."""
    if pred_xy is None or pred_xy.shape[0] == 0:
        return []
    T = pred_xy.shape[0]
    if T <= n:
        idx = list(range(T))
    else:
        # np.linspace gives us evenly spaced indices including endpoints.
        idx = np.linspace(0, T - 1, n).round().astype(int).tolist()
    return [[float(pred_xy[i, 0]), float(pred_xy[i, 1])] for i in idx]


def trajectory_angle_at_time(pred_xy: np.ndarray, t_elapsed: float) -> float:
    """Tangent angle of the predicted trajectory at time ``t_elapsed`` (deg).

    Alpamayo's pred_xy is in ego-frame meters: x=forward, y=left, with
    sample i corresponding to wall-clock time ``i * dt`` where
    ``dt = TRAJECTORY_HORIZON_S / (T - 1)``.

    Returns the heading of the curve's tangent at that future moment —
    i.e. the direction the cart will be pointing if it follows the
    prediction faithfully. Positive = left of straight-ahead, matching
    autoware sign convention. Linear-interpolated between samples and
    angle-unwrapped so we don't jump 359°→0°.

    Caller is responsible for clamping ``t_elapsed`` to the trajectory
    range and applying STEER_AMP / clamp.
    """
    if pred_xy is None or pred_xy.shape[0] < 2:
        return 0.0
    T = pred_xy.shape[0]
    dt = TRAJECTORY_HORIZON_S / max(T - 1, 1)
    f_idx = max(0.0, min(t_elapsed / dt, float(T - 1)))
    i_lo = int(math.floor(f_idx))
    i_hi = min(i_lo + 1, T - 1)
    frac = f_idx - i_lo

    def tangent_deg(i: int) -> float:
        # Central difference where possible, forward/backward at edges.
        i_prev = max(0, i - 1)
        i_next = min(T - 1, i + 1)
        dx = float(pred_xy[i_next, 0] - pred_xy[i_prev, 0])
        dy = float(pred_xy[i_next, 1] - pred_xy[i_prev, 1])
        if dx * dx + dy * dy < 1e-6:
            return 0.0
        return math.degrees(math.atan2(dy, dx))

    a_lo = tangent_deg(i_lo)
    a_hi = tangent_deg(i_hi)
    # Unwrap before interpolating.
    if a_hi - a_lo > 180.0:
        a_hi -= 360.0
    elif a_hi - a_lo < -180.0:
        a_hi += 360.0
    return a_lo * (1.0 - frac) + a_hi * frac


def trajectory_to_steer_deg(pred_xy: np.ndarray, t_elapsed: float) -> float:
    """Real-time-replay steering: tangent of the trajectory at t_elapsed.

    See ``trajectory_angle_at_time`` for the kernel. This wrapper applies
    STEER_AMP (geometric heading → wheel-ratio amplification) and clamps
    to ±STEER_CLAMP_DEG before ps5_drive.py multiplies by AUTOSTEER_GAIN.
    """
    deg = trajectory_angle_at_time(pred_xy, t_elapsed) * STEER_AMP
    return float(np.clip(deg, -STEER_CLAMP_DEG, STEER_CLAMP_DEG))


# ═══════════════════════════════════════════════════════════════════════════
# Modal client (asyncio in a worker thread)
# ═══════════════════════════════════════════════════════════════════════════

# Active per-cycle quality. AlpamayoClient sets this from --quality so the
# encoder closures don't need rebinding each frame.
_ACTIVE_QUALITY = JPEG_QUALITY_SEND


def encode_jpeg(frame_bgr: np.ndarray, quality: int = JPEG_QUALITY_SEND) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame_bgr,
                            [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return buf.tobytes()


def encode_webp(frame_bgr: np.ndarray, quality: int = JPEG_QUALITY_SEND) -> bytes:
    """libwebp via cv2. Roughly 2× the compression ratio of JPEG at the
    same quality level, at ~25× the CPU encode cost. Worth it only when
    upload bandwidth is the bottleneck and encode can be parallelized
    across cores. Server-side PIL.Image.open auto-detects WebP from the
    RIFF header — no decoder change needed."""
    ok, buf = cv2.imencode(".webp", frame_bgr,
                            [int(cv2.IMWRITE_WEBP_QUALITY), quality])
    if not ok:
        raise RuntimeError("cv2.imencode webp failed")
    return buf.tobytes()


def _encode_jpeg_q(frame: np.ndarray) -> bytes:
    return encode_jpeg(frame, _ACTIVE_QUALITY)


def _encode_webp_q(frame: np.ndarray) -> bytes:
    return encode_webp(frame, _ACTIVE_QUALITY)


# ── Stateful video stream encoder (H.264 / HEVC) ────────────────────────
# One VideoStreamEncoder per camera channel; maintained across the
# session so P-frames can reference earlier I/P-frames. Each call to
# .encode(bgr) returns 0 or more bytes-packets — usually exactly one
# with bframes=0 + tune=zerolatency. Server keeps a matching decoder
# context per channel and feeds packets to recover frames.
class VideoStreamEncoder:
    """Wraps PyAV's libx264/libx265 software encoder configured for low
    latency streaming. CRF defaults to 28 which is "near-lossless" on
    photo content and gives ~10× compression over JPEG q60 in driving
    scenes (verified on this cart's eval clips)."""

    def __init__(self, codec: str = "libx265", width: int = 1920,
                 height: int = 1080, crf: int = 28, gop: int = 0):
        import av  # local: keep alpamayo_infer importable without PyAV
        from fractions import Fraction
        self._av = av
        self.ctx = av.CodecContext.create(codec, "w")
        self.ctx.width = width
        self.ctx.height = height
        self.ctx.pix_fmt = "yuv420p"
        self.ctx.time_base = Fraction(1, 90000)
        self.ctx.framerate = Fraction(5, 1)
        # gop=0 → single keyframe at session start, all P-frames after.
        # Safe on TCP (no packet loss → no need for periodic recovery
        # I-frames) and removes the every-12s upload spikes that were
        # bumping the slack budget. Decoder still works fine because
        # PyAV preserves SPS/PPS state per-channel for the session.
        gop_arg = str(gop) if gop > 0 else "999999"
        self.ctx.options = {
            "preset": "ultrafast",
            "tune": "zerolatency",
            "crf": str(crf),
            "g": gop_arg,
            "keyint_min": gop_arg,
            "bf": "0",  # no B-frames → 1 packet per frame, low latency
            "log-level": "error",
        }
        self._pts = 0

    def encode(self, bgr: np.ndarray) -> "list[bytes]":
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        vf = self._av.VideoFrame.from_ndarray(rgb, format="rgb24")
        vf.pts = self._pts
        self._pts += 1
        out = []
        for pkt in self.ctx.encode(vf):
            out.append(bytes(pkt))
        return out

    def close(self) -> None:
        try:
            for _ in self.ctx.encode(None):
                pass
        except Exception:
            pass


# Resolved at construction time from the --codec arg.
def _encoder_for(codec: str):
    codec = codec.lower()
    if codec in ("jpeg", "jpg"):
        return _encode_jpeg_q
    if codec == "webp":
        return _encode_webp_q
    if codec in ("h264", "hevc"):
        # Stateful video codecs use the per-camera encoder pool inside
        # AlpamayoClient — no per-frame helper here.
        return None
    raise ValueError(f"unknown codec: {codec!r}")


class AlpamayoClient:
    """Owns the Modal tunnel TCP connection and a background asyncio loop.

    Lifecycle: spawn the Modal `LiveInference.call(request_id)` method,
    pop the container's tunnel address from the partitioned Modal Queue,
    open a plain asyncio TCP connection, and stream length-prefixed
    msgpack frames forever. Reconnects (with exponential backoff) on
    drop. ``stop()`` writes a `bye` sentinel and cancels the Modal call
    so the container can scale down.

    Public API is thread-safe: ``submit(frames_per_channel)`` is called
    from the main loop's thread, and the latest prediction is read via
    ``latest_steer_deg`` / ``latest_pred_xy``.
    """

    def __init__(self, app_name: str, send_fps: float,
                 server_hw_default: tuple[int, int],
                 cls_name: str = "LiveInference",
                 use_ego: bool = True,
                 codec: str = "jpeg",
                 encode_workers: int = 4,
                 tunnel_addr: "str | None" = None):
        if modal is None and not tunnel_addr:
            raise RuntimeError(
                "modal SDK not importable on this Python interpreter "
                f"(needed for tunnel transport): {_MODAL_IMPORT_ERROR!r}. "
                "Run: /usr/bin/python3 -m pip install modal"
            )
        self.app_name = app_name
        self.cls_name = cls_name
        self.send_fps = send_fps
        self.codec = codec
        self.direct_tunnel_addr = tunnel_addr
        self._encode = _encoder_for(codec)
        # ThreadPoolExecutor for per-camera encoding. cv2's encoders
        # release the GIL during the C call, so a thread pool gives a
        # real ~Nx speedup for CPU-bound codecs (WebP / future HEIC).
        # JPEG is already so fast that the pool overhead is wash; we
        # still use it for code uniformity.
        from concurrent.futures import ThreadPoolExecutor
        self._encode_pool = ThreadPoolExecutor(
            max_workers=max(1, encode_workers), thread_name_prefix="enc"
        )
        # For video-stream codecs: 4 stateful encoders, one per camera.
        # Created lazily on first send() once we know server's H/W.
        self._video_encoders: "list | None" = None
        self.server_hw: "tuple[int, int] | None" = None
        self.server_hw_default = server_hw_default

        self._loop: "asyncio.AbstractEventLoop | None" = None
        self._thread = threading.Thread(target=self._thread_main,
                                        daemon=True, name="alpamayo-tunnel")
        self._stop = threading.Event()
        # Region surfaced by the server in its hello message; logged for
        # operator awareness and exposed via the state file.
        self.server_region: "str | None" = None
        # Set by submit() each time a fresh frame batch is staged.
        # Outgoing seq counter; used to match send→receive timestamps for
        # a real client-measured RTT (see receiver()).
        self._next_seq = 0
        self._inflight: "dict[int, float]" = {}
        self._inflight_client_ms: "dict[int, dict]" = {}
        self.last_rtt_ms = 0.0
        self.last_oneway_recv_ms = 0.0  # server_send → client_recv (clock-skewed)
        self.tcp_address: "str | None" = None
        # Pending frames: 4-channel BGR tuple, last-write-wins (latest only).
        self._lock = threading.Lock()
        self._pending: "tuple[np.ndarray, ...] | None" = None
        self._pending_frame_times: "tuple[float, ...] | None" = None
        self._pending_t = 0.0
        # Latest prediction state (read by main loop):
        self.latest_pred_xy: "np.ndarray | None" = None
        self.latest_steer_deg = 0.0
        self.latest_steer_smoothed = 0.0
        self.last_recv_t = 0.0
        self.recv_count = 0
        self.send_count = 0
        self.gpu_ms = 0.0
        self.srv_total_ms = 0.0
        self.srv_recv_wait_ms = 0.0
        self.client_send_ms: dict = {}
        self.client_recv_ms: dict = {}
        self.server_stage_ms: dict = {}
        self.last_payload_bytes = 0
        self.last_frame_age_ms = 0.0
        self.warming = True
        self.connected = False
        # Latest chain-of-thought string from the VLM rollout (when the
        # server is configured with ALPAMAYO_MAX_GEN_LEN > 1). Empty
        # string until the first prediction lands or if the model
        # returns no usable text.
        self.reasoning = ""
        # Recent recv timestamps for hz estimate.
        self._recv_times: deque = deque(maxlen=20)
        # Bandwidth: cumulative bytes plus per-second sample windows for
        # rolling Mbps. Window = (timestamp_s, byte_total) snapshots,
        # rolled forward by the consumer.
        self.bytes_sent = 0
        self.bytes_recv = 0
        self._bw_window_s = 2.0
        self._sent_samples: deque = deque()  # (t, cumulative_bytes_sent)
        self._recv_samples: deque = deque()  # (t, cumulative_bytes_recv)

        # iPhone ARKit ego-motion publisher. EgoSensor reconnects on its
        # own and quietly returns "no history" if the iOS app isn't
        # streaming — in that case we just don't attach ego to the
        # payload and the server falls back to its reference clip.
        self._ego: "EgoSensor | None" = None
        if use_ego and EgoSensor is not None:
            try:
                self._ego = EgoSensor(verbose=False)
                self._ego.start()
            except Exception as e:
                print(f"[tunnel] could not start EgoSensor: {e!r}")
                self._ego = None
        elif not use_ego:
            print("[tunnel] ego disabled (--no-ego); server will use ref clip")
        self.ego_send_count = 0  # how many payloads included ego data

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._ego is not None:
            try:
                self._ego.stop()
            except Exception:
                pass

    def submit(
        self,
        frames_per_channel: tuple[np.ndarray, ...],
        frame_times: "tuple[float, ...] | None" = None,
    ) -> None:
        """Stage 4-channel frames for the next send. Last writer wins."""
        with self._lock:
            self._pending = frames_per_channel
            self._pending_frame_times = frame_times
            self._pending_t = time.monotonic()

    def _thread_main(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._run())
        finally:
            self._loop.close()

    async def _run(self) -> None:
        backoff = 1.0
        while not self._stop.is_set():
            try:
                await self._session()
                backoff = 1.0
            except Exception as e:
                self.connected = False
                print(f"[tunnel] session failed: {type(e).__name__}: {e}; "
                      f"reconnecting in {backoff:.1f}s")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)

    async def _session(self) -> None:
        """Spawn the Modal call, pop the tunnel address, open TCP, stream."""
        request_id = uuid.uuid4().hex
        fc = None
        if self.direct_tunnel_addr:
            tcp_addr = self.direct_tunnel_addr
            print(f"[tunnel] using direct tunnel address {tcp_addr}")
        else:
            service = modal.Cls.from_name(self.app_name, self.cls_name)
            addr_queue = modal.Queue.from_name(
                f"{self.app_name}-tunnel-addrs", create_if_missing=True
            )
            print(f"[tunnel] spawning {self.app_name}.{self.cls_name}.call "
                  f"(req={request_id[:8]}…)")
            fc = await service().call.spawn.aio(request_id)
        try:
            if not self.direct_tunnel_addr:
                tcp_addr = await asyncio.wait_for(
                    addr_queue.get.aio(partition=request_id), timeout=180.0
                )
            host, port_text = tcp_addr.split(":", maxsplit=1)
            port = int(port_text)
            self.tcp_address = tcp_addr
            print(f"[tunnel] container chose {tcp_addr}; opening TCP …")
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port), timeout=30.0
            )
            # TCP_NODELAY: disable Nagle's algorithm so each msgpack
            # write goes on the wire immediately. Without this, small
            # writes (the trajectory reply, ack-only TCP segments) can
            # sit in the kernel for up to 40 ms waiting for more data.
            try:
                import socket as _sock
                s = writer.get_extra_info("socket")
                if s is not None:
                    s.setsockopt(_sock.IPPROTO_TCP, _sock.TCP_NODELAY, 1)
            except Exception as e:
                print(f"[tunnel] could not set TCP_NODELAY: {e!r}")
            self.connected = True
            try:
                await self._stream(reader, writer)
            finally:
                # Best-effort graceful bye so the server's call() returns.
                try:
                    bye = msgpack.packb({"bye": True})
                    writer.write(struct.pack(">I", len(bye)) + bye)
                    await writer.drain()
                except Exception:
                    pass
                try:
                    writer.close()
                    await writer.wait_closed()
                except Exception:
                    pass
        finally:
            try:
                if fc is not None:
                    await fc.cancel.aio()
            except Exception:
                pass
            self.connected = False

    async def _stream(self, reader: asyncio.StreamReader,
                      writer: asyncio.StreamWriter) -> None:
        """Length-prefixed (4-byte BE) msgpack frames, both directions.

        Frames are sent at ``send_fps`` independently of prediction
        replies. The server has a drop-stale reader, so queuing multiple
        frame batches there is intentional: it keeps the latest temporal
        window fresh while the 10B model works on the previous one. TCP
        ``drain()`` is still the backpressure boundary if the uplink
        cannot sustain the requested frame rate.
        """

        async def read_frame() -> bytes:
            hdr = await reader.readexactly(4)
            n = struct.unpack(">I", hdr)[0]
            return await reader.readexactly(n)

        async def write_frame(payload: bytes) -> None:
            writer.write(struct.pack(">I", len(payload)) + payload)
            await writer.drain()

        async def receiver() -> None:
            while not self._stop.is_set():
                t_read0 = time.perf_counter()
                raw = await read_frame()
                t_read1 = time.perf_counter()
                self.bytes_recv += len(raw)
                self._recv_samples.append((time.monotonic(), self.bytes_recv))
                t_unpack0 = time.perf_counter()
                msg = msgpack.unpackb(raw, raw=False)
                t_unpack1 = time.perf_counter()
                if msg.get("hello"):
                    self.server_hw = (int(msg["H"]), int(msg["W"]))
                    self.server_region = str(msg.get("region", "") or "") or None
                    print(f"[tunnel] hello: res {msg['W']}x{msg['H']} "
                          f"region={self.server_region} "
                          f"transport={msg.get('transport', '?')}")
                    continue
                if msg.get("bye_ack"):
                    return
                if msg.get("warming"):
                    seq = msg.get("seq")
                    if isinstance(seq, int):
                        self._inflight.pop(seq, None)
                        self._inflight_client_ms.pop(seq, None)
                    continue
                self.warming = False
                t_parse0 = time.perf_counter()
                shape = tuple(msg["pred_shape"])
                pred = np.frombuffer(
                    msg["pred_xy"], dtype=np.float32
                ).reshape(shape)
                t_parse1 = time.perf_counter()
                now = time.monotonic()
                self.latest_pred_xy = pred
                self.last_recv_t = now
                self._recv_times.append(now)
                self.recv_count += 1
                self.gpu_ms = float(msg.get("gpu_ms", 0.0))
                self.srv_total_ms = float(msg.get("total_ms", 0.0))
                self.srv_recv_wait_ms = float(msg.get("recv_ms", 0.0))
                self.server_stage_ms = dict(msg.get("stage_ms", {}) or {})
                self.reasoning = str(msg.get("reasoning", "") or "")
                # Per-message RTT: matched send timestamp via seq.
                seq = msg.get("seq")
                if isinstance(seq, int):
                    sent_t = self._inflight.pop(seq, None)
                    client_ms = self._inflight_client_ms.pop(seq, None)
                    if isinstance(client_ms, dict):
                        self.client_send_ms = client_ms
                    if sent_t is not None:
                        self.last_rtt_ms = (now - sent_t) * 1000.0
                self.client_recv_ms = {
                    "reply_read": (t_read1 - t_read0) * 1000.0,
                    "reply_unpack": (t_unpack1 - t_unpack0) * 1000.0,
                    "reply_parse": (t_parse1 - t_parse0) * 1000.0,
                }
                # One-way recv leg (server_send wallclock → client_recv
                # wallclock). Uses time.time() so it's clock-skew biased,
                # but useful as a relative trend signal.
                ssm = msg.get("server_send_ms")
                if isinstance(ssm, (int, float)):
                    self.last_oneway_recv_ms = max(
                        0.0, time.time() * 1000.0 - float(ssm)
                    )

        async def sender() -> None:
            send_period = 1.0 / max(self.send_fps, 0.1)
            next_send_t = time.monotonic()
            while not self._stop.is_set():
                now_send = time.monotonic()
                if now_send < next_send_t:
                    await asyncio.sleep(next_send_t - now_send)
                    continue
                next_send_t = max(next_send_t + send_period,
                                  time.monotonic())
                # Sample the freshest frame available at send time.
                with self._lock:
                    frames = self._pending
                    frame_times = self._pending_frame_times
                    pending_t = self._pending_t
                if frames is None:
                    # Camera not ready yet; try again on the next tick.
                    continue
                # Encode the camera frames at their native capture
                # resolution. The server decodes and resizes to the
                # Alpamayo model's H/W, which preserves the same model
                # input while avoiding a 640x480 -> 1920x1080 client-side
                # upscale before compression. That upscale was pure
                # tunnel bloat.
                t_send0 = time.perf_counter()
                now_mono = time.monotonic()
                pending_age_ms = (
                    (now_mono - pending_t) * 1000.0 if pending_t else 0.0
                )
                frame_age_ms = 0.0
                if frame_times:
                    frame_age_ms = max(
                        0.0,
                        max(now_mono - float(t) for t in frame_times) * 1000.0,
                    )
                send_frames = list(frames)
                send_h, send_w = send_frames[0].shape[:2]
                t_resize0 = time.perf_counter()
                # Kept as a timed no-op so the telemetry schema stays
                # stable; any future downscale/crop for transport can
                # fill this bucket.
                t_resize1 = time.perf_counter()
                loop = asyncio.get_event_loop()
                seq = self._next_seq
                self._next_seq += 1
                t_encode0 = time.perf_counter()
                if self.codec in ("h264", "hevc"):
                    # Stateful video stream — encoders are per-channel
                    # and persist across the whole session. First call
                    # initializes them with the native transport H/W.
                    if self._video_encoders is None:
                        codec_name = "libx264" if self.codec == "h264" else "libx265"
                        crf = _ACTIVE_QUALITY  # repurposed: lower = higher quality
                        self._video_encoders = [
                            VideoStreamEncoder(codec=codec_name, width=send_w,
                                                height=send_h, crf=crf)
                            for _ in range(len(send_frames))
                        ]
                        print(f"[tunnel] video stream init: codec={codec_name} "
                              f"crf={crf} {send_w}x{send_h}")
                    pkt_lists = list(await asyncio.gather(*[
                        loop.run_in_executor(self._encode_pool, enc.encode, r)
                        for enc, r in zip(self._video_encoders, send_frames)
                    ]))
                    msg_out: dict = {
                        "video": {
                            "codec": self.codec,
                            "packets": pkt_lists,  # list[list[bytes]]
                            "shape": [send_h, send_w],
                        },
                        "seq": seq,
                    }
                else:
                    jpegs = list(await asyncio.gather(*[
                        loop.run_in_executor(self._encode_pool, self._encode, r)
                        for r in send_frames
                    ]))
                    msg_out = {"jpegs": jpegs, "seq": seq}
                t_encode1 = time.perf_counter()
                t_ego0 = time.perf_counter()
                if self._ego is not None:
                    ego_hist = self._ego.history()
                    tensors = build_ego_tensors(ego_hist)
                    if tensors is not None:
                        xyz_local, rot_local = tensors
                        msg_out["ego_history_xyz"] = xyz_local.tobytes()
                        msg_out["ego_history_rot"] = rot_local.tobytes()
                        msg_out["ego_history_shape"] = list(xyz_local.shape)
                        self.ego_send_count += 1
                t_ego1 = time.perf_counter()
                t_pack0 = time.perf_counter()
                payload = msgpack.packb(msg_out)
                t_pack1 = time.perf_counter()
                self._inflight[seq] = time.monotonic()
                self.last_payload_bytes = len(payload) + 4
                self.last_frame_age_ms = frame_age_ms
                self._inflight_client_ms[seq] = {
                    "pending_age": pending_age_ms,
                    "frame_age": frame_age_ms,
                    "resize": (t_resize1 - t_resize0) * 1000.0,
                    "encode": (t_encode1 - t_encode0) * 1000.0,
                    "ego": (t_ego1 - t_ego0) * 1000.0,
                    "pack": (t_pack1 - t_pack0) * 1000.0,
                    "payload_bytes": len(payload) + 4,
                }
                # Bound the inflight map so a long stall can't leak it.
                if len(self._inflight) > 64:
                    oldest = sorted(self._inflight.keys())[:32]
                    for k in oldest:
                        self._inflight.pop(k, None)
                        self._inflight_client_ms.pop(k, None)
                t_write0 = time.perf_counter()
                await write_frame(payload)
                t_write1 = time.perf_counter()
                self._inflight_client_ms[seq]["write_drain"] = (
                    t_write1 - t_write0
                ) * 1000.0
                self._inflight_client_ms[seq]["send_local_total"] = (
                    t_write1 - t_send0
                ) * 1000.0
                self.send_count += 1
                self.bytes_sent += len(payload) + 4
                self._sent_samples.append(
                    (time.monotonic(), self.bytes_sent)
                )

        await asyncio.gather(receiver(), sender())

    def hz(self) -> float:
        if len(self._recv_times) < 2:
            return 0.0
        span = self._recv_times[-1] - self._recv_times[0]
        if span <= 0.0:
            return 0.0
        return (len(self._recv_times) - 1) / span

    @staticmethod
    def _mbps_from_window(samples: deque, window_s: float, now: float) -> float:
        """Pop samples older than ``window_s`` and return MB/s over what's left.

        ``samples`` is a deque of (t_seconds, cumulative_bytes). The
        first kept sample is the oldest within the window; the rate is
        ``(latest_bytes - oldest_bytes) / dt`` converted to MB/s.
        """
        cutoff = now - window_s
        while len(samples) > 1 and samples[0][0] < cutoff:
            samples.popleft()
        if len(samples) < 2:
            return 0.0
        t0, b0 = samples[0]
        t1, b1 = samples[-1]
        dt = t1 - t0
        if dt <= 0.0:
            return 0.0
        return (b1 - b0) / dt / 1_000_000.0  # bytes/s -> MB/s

    def up_mbps(self) -> float:
        return self._mbps_from_window(
            self._sent_samples, self._bw_window_s, time.monotonic()
        )

    def down_mbps(self) -> float:
        return self._mbps_from_window(
            self._recv_samples, self._bw_window_s, time.monotonic()
        )

    def update_smoothed(self, raw_deg: float) -> float:
        # First-order EMA matching autoware's STEER_SMOOTHING semantics
        # (the *new* sample gets weight (1 - alpha)).
        self.latest_steer_deg = raw_deg
        self.latest_steer_smoothed = (
            STEER_SMOOTHING * self.latest_steer_smoothed
            + (1.0 - STEER_SMOOTHING) * raw_deg
        )
        return self.latest_steer_smoothed


# ═══════════════════════════════════════════════════════════════════════════
# Main loop
# ═══════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--app-name", default=os.environ.get(
                       "ALPAMAYO_APP_NAME", "alpamayo-live-demo"),
                   help="Modal App name to spawn for the tunnel session "
                        "(default 'alpamayo-live-demo'). Override via "
                        "$ALPAMAYO_APP_NAME.")
    p.add_argument("--tunnel-addr", default=os.environ.get("ALPAMAYO_TUNNEL_ADDR"),
                   help="Connect directly to an already-published Modal tunnel "
                        "address host:port. Useful for latency benchmarking "
                        "when queue delivery is not the thing being measured.")
    p.add_argument("--cls-name", default=os.environ.get(
                       "ALPAMAYO_CLS_NAME", "LiveInference"),
                   help="Modal class name on the app (default 'LiveInference').")
    p.add_argument("--frames-dir", default=str(FRAMES_DIR_DEFAULT),
                   help="Per-camera JPEG output dir (atomic replace).")
    p.add_argument("--state-file", default=str(STATE_FILE_DEFAULT),
                   help="JSON output for steer + cam status (autoware-compatible).")
    p.add_argument("--send-fps", type=float, default=SEND_FPS_DEFAULT,
                   help="Frames per second sent to Modal. Match server "
                        f"capacity (default {SEND_FPS_DEFAULT}).")
    p.add_argument("--jpeg-hz", type=float, default=PUBLISH_HZ_DEFAULT,
                   help="Per-camera JPEG publish rate to /tmp/cart_frames "
                        f"for the web UI (default {PUBLISH_HZ_DEFAULT}).")
    p.add_argument("--replay-lead-s", type=float, default=REPLAY_LEAD_S,
                   help="Constant offset added to the replay clock so "
                        "the wheel commands the curve's tangent slightly "
                        "ahead of where the cart currently is on the "
                        f"trajectory. Default {REPLAY_LEAD_S}.")
    p.add_argument("--indices", type=int, nargs="*", default=None,
                   help="Override v4l2 indices (e.g. --indices 0 2 4).")
    p.add_argument("--camera-mapping", type=Path,
                   default=CAMERA_MAPPING_DEFAULT,
                   help="REAL camera mapping JSON from calibration.")
    p.add_argument("--extrinsics", type=Path,
                   default=MANUAL_EXTRINSICS_DEFAULT,
                   help="Manual REAL extrinsics JSON for logging/state.")
    p.add_argument("--video", default=None,
                   help="Replay an MP4 instead of opening cameras. The clip "
                        "is broadcast into all 4 alpamayo channel slots.")
    p.add_argument("--no-loop", action="store_true",
                   help="Stop at video EOF instead of looping (only with --video).")
    p.add_argument("--no-ego", action="store_true",
                   help="Skip iPhone EgoSensor entirely. The Modal server will "
                        "fall back to its canned reference-clip ego history. "
                        "Useful for fake-mode latency benches where the cart "
                        "hardware (PS5 / IMU) isn't attached.")
    p.add_argument("--codec", default=os.environ.get("ALPAMAYO_CODEC", "hevc"),
                   choices=["jpeg", "webp", "h264", "hevc"],
                   help="Image codec for the cart→Modal payload. "
                        "jpeg/webp = independent frames. "
                        "h264/hevc = stateful video stream (PyAV + libx264/265, "
                        "exploits inter-frame redundancy for ~5–15× compression).")
    p.add_argument("--quality", type=int, default=None,
                   help="Quality for image codecs (0-100). For h264/hevc this "
                        "is mapped to a CRF (lower CRF = higher quality). "
                        "Defaults to JPEG_QUALITY_SEND (60).")
    p.add_argument("--server-hw", type=int, nargs=2, default=(1080, 1920),
                   metavar=("H", "W"),
                   help="Default server resolution before hello (H W). "
                        "Overridden by server's hello message.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if not args.app_name:
        print("ERROR: --app-name required (or set $ALPAMAYO_APP_NAME).",
              file=sys.stderr)
        return 2
    # Resolve quality default per-codec. For h264/hevc the field doubles
    # as CRF (0-51, lower = better quality). 28 is a sweet spot on
    # photo content. For jpeg/webp it stays in the 0-100 range.
    if args.quality is None:
        args.quality = 28 if args.codec in ("h264", "hevc") else JPEG_QUALITY_SEND
    global _ACTIVE_QUALITY
    _ACTIVE_QUALITY = args.quality
    print(f"[run] codec={args.codec} quality={args.quality}")
    if modal is None:
        print(f"ERROR: modal SDK not importable: {_MODAL_IMPORT_ERROR!r}\n"
              "This script must run on a Python with `modal` installed — "
              "use /home/caddy/mayo/.venv-client/bin/python (already has "
              "modal+cv2+msgpack), or set $ALPAMAYO_PY in start.sh to "
              "another interpreter that does.",
              file=sys.stderr)
        return 2

    frames_dir = Path(args.frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    state_path = Path(args.state_file)
    state_path.parent.mkdir(parents=True, exist_ok=True)

    camera_mapping: dict[str, dict] = {}
    manual_extrinsics: dict | None = None
    if args.camera_mapping.exists():
        camera_mapping = load_camera_mapping(args.camera_mapping)
        print(f"[calib] camera mapping: {args.camera_mapping}")
        for slug in SLUGS:
            item = camera_mapping.get(slug)
            if item is None:
                print(f"[calib] WARN: no mapping for {slug}")
                continue
            hfov = calibrated_hfov_deg(item["intrinsics"])
            print(
                f"[calib] {slug:<12} idx={item['camera_index']} "
                f"hfov={hfov:.1f}° intr={Path(item['intrinsics']).name}"
            )
    else:
        print(f"[calib] WARN: camera mapping not found: {args.camera_mapping}")
    if args.extrinsics.exists():
        with args.extrinsics.open() as f:
            manual_extrinsics = json.load(f)
        print(f"[calib] extrinsics: {args.extrinsics}")
    else:
        print(f"[calib] WARN: extrinsics not found: {args.extrinsics}")

    channel_specs = [dict(spec) for spec in ALPAMAYO_CHANNEL_SPECS]
    for spec in channel_specs:
        target_fov = spec.get("target_fov_deg")
        slug = spec["slug"]
        spec["crop_ratio"] = 1.0
        if target_fov is not None and slug in camera_mapping:
            source_fov = calibrated_hfov_deg(camera_mapping[slug]["intrinsics"])
            spec["source_fov_deg"] = source_fov
            spec["crop_ratio"] = pinhole_crop_ratio_for_fov(source_fov, float(target_fov))
            print(
                f"[calib] channel {spec['channel']} {spec['label']} <- {slug}: "
                f"{source_fov:.1f}°→{target_fov:.1f}° crop={spec['crop_ratio']:.3f}"
            )

    readers: list = []
    if args.video:
        if not Path(args.video).exists():
            print(f"ERROR: video not found: {args.video}", file=sys.stderr)
            return 1
        print(f"[video] replaying {args.video} (loop={'no' if args.no_loop else 'yes'})")
        video_src = VideoReader(args.video, loop=not args.no_loop)
        video_src.start()
        readers = [_BroadcastView(video_src, slug) for slug in SLUGS]
        for slug in SLUGS:
            print(f"[video] {slug:<12} <- {Path(args.video).name}")
    else:
        if args.indices:
            print("[cams] using CLI v4l2 indices...")
            indices = args.indices
        elif all(slug in camera_mapping for slug in SLUGS):
            print("[cams] using calibrated v4l2 mapping...")
            indices = [int(camera_mapping[slug]["camera_index"]) for slug in SLUGS]
        else:
            print("[cams] discovering v4l2 devices...")
            indices = discover_v4l2_indices(count=len(SLUGS))
        print(f"[cams] indices: {indices}")
        if not indices:
            print("ERROR: no cameras found. Try: v4l2-ctl --list-devices")
            return 1
        if len(indices) < len(SLUGS):
            print(f"ERROR: expected {len(SLUGS)} camera indices for {SLUGS}, got {indices}")
            return 1

        for idx, slug in zip(indices, SLUGS):
            cap = open_camera(idx)
            if cap is None:
                print(f"[cams] WARN: idx {idx} ({slug}) failed to open")
                continue
            r = CameraReader(cap, slug)
            r.start()
            readers.append(r)
            flip = CAMERA_ORIENTATION_FIX.get(slug)
            crop = CAMERA_FOV_CROP_RATIO.get(slug)
            bits = []
            if flip is not None:
                bits.append(f"flip={flip}")
            if crop is not None and crop < 1.0:
                bits.append(f"raw crop_ratio={crop:.3f}")
            tail = ("  (" + ", ".join(bits) + ")") if bits else ""
            print(f"[cams] {slug:<12} -> /dev/video{idx}{tail}")

    if not readers:
        print("ERROR: opened 0 cameras.")
        return 1

    slug_map = {r.slug: r for r in readers}
    # Order readers into Alpamayo channel slots (0..3). Duplicate source
    # slugs are allowed: the center `front` camera feeds both front_wide
    # and front_tele (the latter with a calibrated center crop).
    channel_slugs = [None] * 4
    channel_labels = [None] * 4
    channel_crop_ratios = [1.0] * 4
    for spec in channel_specs:
        ch = int(spec["channel"])
        slug = str(spec["slug"])
        if slug in slug_map:
            channel_slugs[ch] = slug
            channel_labels[ch] = str(spec["label"])
            channel_crop_ratios[ch] = float(spec.get("crop_ratio", 1.0))
    missing = [i for i, s in enumerate(channel_slugs) if s is None]
    if missing:
        print(f"ERROR: missing alpamayo channel slots: {missing}")
        return 1
    print(
        "[cams] alpamayo channels:",
        [
            f"{i}:{channel_labels[i]}<-{channel_slugs[i]} crop={channel_crop_ratios[i]:.3f}"
            for i in range(4)
        ],
    )

    time.sleep(0.4)  # let cameras prime

    # Connect to Modal via tunnel.
    client = AlpamayoClient(
        args.app_name, args.send_fps,
        server_hw_default=tuple(args.server_hw),
        cls_name=args.cls_name,
        use_ego=not args.no_ego,
        codec=args.codec,
        tunnel_addr=args.tunnel_addr,
    )
    client.start()
    print(f"[tunnel] dispatcher started; app={args.app_name} "
          f"cls={args.cls_name} send_fps={args.send_fps}")

    jpeg_period = 1.0 / max(args.jpeg_hz, 1.0)
    next_jpeg_t = time.monotonic()
    last_log_t = 0.0

    print("[run] publishing to", frames_dir, "and", state_path)
    try:
        while True:
            loop_t = time.monotonic()

            # ── 1) Stage the latest 4-channel frames for the WS sender.
            channel_frames = []
            channel_frame_times = []
            ready = True
            for slug, crop_ratio in zip(channel_slugs, channel_crop_ratios):
                r = slug_map[slug]
                f = r.latest()
                if f is None:
                    ready = False
                    break
                if crop_ratio < 1.0:
                    f = center_crop_zoom(f, crop_ratio)
                channel_frames.append(f)
                channel_frame_times.append(float(getattr(r, "last_ok_s", 0.0)))
            if ready:
                client.submit(tuple(channel_frames), tuple(channel_frame_times))

            # ── 2) Convert latest prediction to steer_deg + speed targets.
            pred = client.latest_pred_xy
            target_speed_mph = 0.0
            target_gas = 0.0
            target_brake = 0.0
            predicted_path: list = []
            # Time-based replay: how long has it been since this
            # prediction landed? That's our position on the curve. Add
            # a small constant lead so the wheel commands "next moment"
            # rather than "exactly now". Cap at REPLAY_MAX_S so a
            # delayed next prediction doesn't push us off the end of
            # the curve.
            t_elapsed = 0.0
            if pred is not None and client.last_recv_t > 0.0:
                age = time.monotonic() - client.last_recv_t
                t_elapsed = min(age + args.replay_lead_s, REPLAY_MAX_S)
            if pred is not None:
                raw = trajectory_to_steer_deg(pred, t_elapsed)
                client.update_smoothed(raw)
                target_speed_mph, target_gas, target_brake = (
                    trajectory_to_speed_targets(pred)
                )
                predicted_path = downsample_path(pred)

            # ── 3) Publish per-camera JPEGs + BEV viz to the UI dir.
            if loop_t >= next_jpeg_t:
                for r in readers:
                    f = r.latest()
                    if f is None:
                        continue
                    write_jpeg_atomic(frames_dir / f"{r.slug}.jpg", f)
                # BEV tile — only once we have a prediction; before that
                # the UI's "model loading…" placeholder shows up.
                if pred is not None:
                    bev = render_bev(pred)
                    write_jpeg_atomic(frames_dir / "bev.jpg", bev,
                                       quality=85)
                next_jpeg_t = loop_t + jpeg_period

            # ── 4) Publish state.
            recv_age = (time.monotonic() - client.last_recv_t
                        if client.last_recv_t else float("inf"))
            # Keep the prediction "live" for the full trajectory horizon
            # so the cart commits to the predicted curve for its entire
            # duration even if the next prediction is delayed (Modal
            # cold start, network hiccup, etc). When a new prediction
            # arrives, last_recv_t resets and the replay clock restarts
            # on the new curve. After REPLAY_MAX_S without a refresh we
            # treat the trajectory as exhausted and disengage.
            inference = (client.connected and not client.warming
                         and recv_age < REPLAY_MAX_S)
            bev_ready = pred is not None
            state_payload = {
                "steer_deg": float(client.latest_steer_smoothed),
                "steer_deg_raw": float(client.latest_steer_deg),
                "active_cam": "front",
                "inference": inference,
                "viz": bev_ready,
                "viz_streams": ["bev"] if bev_ready else [],
                "object_count": 0,
                "fps": float(client.hz()),
                "cams": [r.slug for r in readers],
                "source": "camera",
                "model": MODEL_NAME,
                "model_full": "alpamayo-r1-10b",
                # Auto-pedal targets consumed by ps5_drive.py when
                # --autosteer is on and the operator's R2/L2 triggers
                # are at rest. Always include even when not inferring
                # so a freshly-stale state file reads as "0 pedal".
                "target_speed_mph": float(target_speed_mph) if inference else 0.0,
                "target_gas": float(target_gas) if inference else 0.0,
                "target_brake": float(target_brake) if inference else 0.0,
                # Down-sampled trajectory for scene.js's 3D path. List of
                # [x_forward_m, y_left_m]. Empty if no fresh prediction.
                "predicted_path": predicted_path if inference else [],
                "ts": time.time(),
                "alpamayo": {
                    "transport": "modal-tunnel-tcp",
                    "app_name": args.app_name,
                    "tcp_address": client.tcp_address,
                    "region": client.server_region,
                    "connected": bool(client.connected),
                    "warming": bool(client.warming),
                    "send_count": int(client.send_count),
                    "recv_count": int(client.recv_count),
                    "gpu_ms": float(client.gpu_ms),
                    "replay_t_s": float(t_elapsed),
                    "replay_lead_s": float(args.replay_lead_s),
                    "recv_age_s": float(recv_age) if recv_age != float("inf") else None,
                    # Per-prediction latency breakdown for the HUD.
                    # rtt = real client-measured round trip (send→recv,
                    # matched on a per-message seq counter). With tunnels
                    # this is the headline number — was missing under
                    # WebSockets, where we only knew recv_age.
                    # net = rtt - srv_total: time spent over the wire
                    # (cart→Modal and back), once we've subtracted what
                    # the server self-reported.
                    "latency_ms": {
                        "gpu": float(client.gpu_ms),
                        "srv_total": float(client.srv_total_ms),
                        "srv_recv_wait": float(client.srv_recv_wait_ms),
                        "srv_pre": max(0.0, float(client.srv_total_ms)
                                       - float(client.gpu_ms)
                                       - float(client.srv_recv_wait_ms)),
                        "rtt": float(client.last_rtt_ms),
                        "net": max(0.0, float(client.last_rtt_ms)
                                   - float(client.srv_total_ms)),
                        "oneway_recv": float(client.last_oneway_recv_ms),
                    },
                    "latency_detail_ms": {
                        "client_send": dict(client.client_send_ms),
                        "client_recv": dict(client.client_recv_ms),
                        "server": dict(client.server_stage_ms),
                    },
                    "payload": {
                        "codec": args.codec,
                        "quality": int(args.quality),
                        "last_bytes": int(client.last_payload_bytes),
                        "last_frame_age_ms": float(client.last_frame_age_ms),
                    },
                    # Live bandwidth (rolling 2 s window). MB/s.
                    "bandwidth_mbps": {
                        "up": float(client.up_mbps()),
                        "down": float(client.down_mbps()),
                    },
                    "bytes_total": {
                        "sent": int(client.bytes_sent),
                        "recv": int(client.bytes_recv),
                    },
                    "camera_rig": {
                        "mapping": str(args.camera_mapping),
                        "extrinsics": str(args.extrinsics),
                        "slugs": list(SLUGS),
                        "channels": [
                            {
                                "channel": i,
                                "label": channel_labels[i],
                                "slug": channel_slugs[i],
                                "crop_ratio": channel_crop_ratios[i],
                            }
                            for i in range(4)
                        ],
                        "manual_extrinsics_loaded": manual_extrinsics is not None,
                    },
                    # Chain-of-thought string from the VLM rollout (only
                    # populated when the server has ALPAMAYO_MAX_GEN_LEN
                    # > 1). Surfaced in the kiosk top-left under the
                    # CADDY brand mark.
                    "reasoning": str(client.reasoning),
                },
            }
            write_state_atomic(state_path, state_payload)

            # ── 5) Periodic log.
            if loop_t - last_log_t > 1.0:
                last_log_t = loop_t
                print(f"[run] sent={client.send_count} recv={client.recv_count} "
                      f"hz={client.hz():4.2f} "
                      f"rtt={client.last_rtt_ms:5.1f}ms "
                      f"gpu={client.gpu_ms:5.1f}ms "
                      f"enc={float(client.client_send_ms.get('encode', 0.0)):5.1f}ms "
                      f"srv_in={float(client.srv_recv_wait_ms):5.1f}ms "
                      f"net={max(0.0, client.last_rtt_ms - client.srv_total_ms):5.1f}ms "
                      f"bytes={client.last_payload_bytes/1000.0:6.1f}k "
                      f"steer_raw={client.latest_steer_deg:+6.2f}° "
                      f"steer={client.latest_steer_smoothed:+6.2f}° "
                      f"tgt_mph={target_speed_mph:4.1f} "
                      f"gas={target_gas:.3f} brake={target_brake:.3f} "
                      f"infer={'Y' if inference else 'N'}",
                      flush=True)

            time.sleep(0.02)
    except KeyboardInterrupt:
        print("\n[run] interrupted.")
    finally:
        client.stop()
        for r in readers:
            r.stop()
        print("[run] done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

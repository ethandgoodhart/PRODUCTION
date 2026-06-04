#!/usr/bin/env python3
"""
clrnet_infer.py — local CLRerNet inference + steering on Thor.

Same /tmp/cart_frames/ + /tmp/autoware_state.json contract as
``alpamayo_infer.py`` so ``ps5_drive.py --autosteer`` consumes it
unchanged. Differences:
  * Inference is ON-DEVICE (Jetson Thor, /usr/bin/python3 + torch
    2.10/cu130). No Modal, no WebSocket.
  * Only the ``front_wide`` camera is fed to the model — CLRerNet is
    a single-image lane detector. The other 3 cams still get
    captured + published as JPEGs for the UI tiles.
  * Steering is the centerline+lookahead controller from
    ``lane-detection/visualize.py`` (per-side ego classification +
    mirror fallback + IIR smoothing). Output is in column degrees so
    ``AUTOSTEER_GAIN=1.0`` in ps5_drive passes it through.
  * Pedal targets implement a constant 7 mph closed loop using the
    cart's reported MPH from ``/tmp/cart_state.json``.
  * Publishes ``/tmp/cart_frames/lanes.jpg`` (annotated front_wide)
    and sets ``viz_streams: ["lanes"]`` so the kiosk shows it.

start.sh launches this when invoked with ``--model clrnet``.
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


FRAMES_DIR_DEFAULT = Path("/tmp/cart_frames")
STATE_FILE_DEFAULT = Path("/tmp/autoware_state.json")
CART_STATE_FILE_DEFAULT = Path("/tmp/cart_state.json")

CLRNET_REPO_DIR = "/home/caddy/CLRerNet"
CLRNET_CONFIG = (
    f"{CLRNET_REPO_DIR}/configs/clrernet/culane/clrernet_culane_dla34_ema.py"
)
CLRNET_CKPT = "/home/caddy/clrnet_weights/clrernet_culane_dla34_ema.pth"

# Physical layout on this cart (verified by `v4l2-ctl --list-devices` +
# autoware_infer.py:65 comment): discover_v4l2_indices() finds working
# nodes [0, 2, 6, 10] in that order, which on this USB topology are:
#   /dev/video0  = HD varifocal (narrow lens for inference)
#   /dev/video2  = H264 fisheye, LEFT-side
#   /dev/video6  = H264 fisheye, FRONT-WIDE (mounted upside-down)
#   /dev/video10 = H264 fisheye, RIGHT-side
# The other scripts (autoware/alpamayo) still use the legacy
# (narrow, wide, left, right) tuple — that's a known bug we're choosing
# not to chase here, but it's why the wide/left tiles look swapped when
# you run them. clrnet uses the corrected order so steering reads from
# the actual front-facing camera.
SLUGS = ("front_narrow", "left", "front_wide", "right")
ACTIVE_SLUG = "front_wide"

CAMERA_ORIENTATION_FIX = {
    # Only the narrow varifocal is mounted upside-down. front_wide and the
    # side fisheyes are all right-side up natively (verified live: adding
    # a flip on front_wide turned the picture upside-down, so don't).
    "front_narrow": -1,
}

# Per-camera FOV crop — copied from alpamayo_infer.py. The cart's USB
# fisheyes are ~170° but CLRerNet (CULane-trained) and the alpamayo UI
# both expect a ~120° forward view. Without this crop the inference
# frame is a barrel-distorted fisheye and the "front_wide" tile looks
# nothing like alpamayo's. Equidistant-fisheye approx: ratio = target/source.
SOURCE_FOV_DEG = {"front_wide": 170.0, "left": 170.0, "right": 170.0}
TARGET_FOV_DEG = {"front_wide": 120.0, "left": 120.0, "right": 120.0}
CAMERA_FOV_CROP_RATIO = {
    slug: TARGET_FOV_DEG[slug] / SOURCE_FOV_DEG[slug]
    for slug in SOURCE_FOV_DEG
}

CAM_W, CAM_H = 640, 480
JPEG_QUALITY = 72
PUBLISH_HZ_DEFAULT = 15.0
INFER_HZ_DEFAULT = 8.0    # CLRerNet on Thor: ~10-15 ms/frame, leave headroom
MODEL_NAME = "clrnet"
FRONT_CAMERA_BRIGHTNESS = 32

# CLRerNet was trained on CULane (1640x590, cropped to y=270:590 -> 1640x320,
# then resized to 800x320). The test pipeline does the crop+resize for us as
# long as we feed it 1640x590. Resize the cart's 640x480 source up to that
# expected shape; output lane points are returned in normalized [0,1] x
# (orig_w, orig_h) of the *fed* image.
CLRNET_INPUT_W, CLRNET_INPUT_H = 1640, 590
CLRNET_CONF_THRESHOLD = 0.40

# Steering controller — copied from lane-detection/visualize.py with the
# K_LATERAL_DEG / STEER_AMP tweaks needed to drive a real cart.
K_LATERAL_DEG = 60.0
MAX_GEOMETRIC_DEG = 35.0   # lane-frame "road wheel" angle cap
STEER_ALPHA = 0.75         # IIR smoothing on the column-degrees output
                           # (higher = heavier smoothing; tamped down from
                           # 0.45 because the wheel was twitchier than we
                           # wanted on the cart's first live runs)
LOOKAHEAD_Y = 0.55          # bottom-half lookahead (closer than 0.25 because
                            # 640x480 fisheye sees less road than CULane)
HEADING_Y_NEAR = 0.85
HEADING_Y_FAR = 0.55
K_HEADING_DEG = 0.4
# Steering ratio: geometric road-wheel deg -> column-deg. alpamayo uses
# STEER_AMP=15. Keep parity here so the wheel feel matches across models.
STEER_AMP = 15.0
STEER_CLAMP_DEG = 270.0     # column mechanical range

# Constant-speed target (mph). Open-loop gas: cart_state's mph reading is
# noisy (flips 0↔13 when stopped), so closing the loop on it makes gas/brake
# chatter. We just hold a fixed gas level tuned to ~6 mph on flat ground and
# let the operator override with R2. Reject implausibly high mph reads as
# sensor glitches when deciding whether to brake.
TARGET_MPH = 8.0
GAS_CONSTANT = 0.24          # fixed pot value while autosteering (~8 mph)
GAS_MAX = 0.30               # safety cap (also enforced downstream)
MPH_PLAUSIBLE_MAX = 10.0     # readings above this are treated as glitches
BRAKE_OVER_MPH = 2.0         # only brake if we're really overspeed
BRAKE_MAX = 0.15

# Confidence threshold for lane drawing; predictions with score below
# this are still considered for the "fallback" lane-pick logic but not
# tagged "fresh".
DRAW_CONF = 0.40


# ═══════════════════════════════════════════════════════════════════════════
# Camera capture (mirrors alpamayo_infer.CameraReader)
# ═══════════════════════════════════════════════════════════════════════════

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


def center_crop_zoom(frame: np.ndarray, ratio: float) -> np.ndarray:
    """Center-crop ``frame`` by ``ratio`` (0,1] then resize back to its
    original WxH so the apparent FOV shrinks but the resolution is
    unchanged. ``ratio >= 1`` is a no-op."""
    if ratio >= 1.0 or ratio <= 0.0:
        return frame
    h, w = frame.shape[:2]
    nh = max(2, int(round(h * ratio)))
    nw = max(2, int(round(w * ratio)))
    y0 = (h - nh) // 2
    x0 = (w - nw) // 2
    cropped = frame[y0:y0 + nh, x0:x0 + nw]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)


class CameraReader(threading.Thread):
    def __init__(self, cap: "cv2.VideoCapture", slug: str):
        super().__init__(daemon=True, name=f"cam-{slug}")
        self.cap = cap
        self.slug = slug
        self.flip_code = CAMERA_ORIENTATION_FIX.get(slug)
        # FOV crop ratio: 1.0 = no crop. Applied AFTER orientation flip
        # so the crop stays centered on the cart's forward axis even on
        # upside-down-mounted lenses. Mirrors alpamayo_infer.CameraReader.
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
    def __init__(self, video_path: str, loop: bool = True):
        super().__init__(daemon=True, name="video-reader")
        self.video_path = video_path
        self.loop = loop
        self.lock = threading.Lock()
        self.frame: "np.ndarray | None" = None
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

    def latest(self) -> "np.ndarray | None":
        with self.lock:
            return self.frame

    def stop(self) -> None:
        self._stop.set()


class _BroadcastView:
    """Slug-aware wrapper so one VideoReader fills all cam slots."""

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
# Atomic file writers
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


def read_cart_mph(path: Path) -> float | None:
    """Read current cart speed (mph) from ps5_drive's state file."""
    try:
        with path.open() as f:
            data = json.load(f)
        ts = float(data.get("ts", 0.0))
        if time.time() - ts > 1.0:
            return None
        return float(data.get("mph", 0.0))
    except (FileNotFoundError, ValueError, json.JSONDecodeError):
        return None


# ═══════════════════════════════════════════════════════════════════════════
# Lane → steering (port of lane-detection/visualize.py logic)
# ═══════════════════════════════════════════════════════════════════════════

def _side_of(lane: dict) -> str:
    pts = lane["points"]
    ys = pts[:, 1]
    bottom = pts[ys >= 0.5, 0] if (ys >= 0.5).sum() >= 2 else pts[:, 0]
    return "left" if float(np.mean(bottom)) < 0.5 else "right"


def _mirror_lane(lane: dict) -> dict:
    pts = lane["points"].copy()
    pts[:, 0] = 1.0 - pts[:, 0]
    return {"points": pts, "score": float(lane.get("score", 0.0))}


def _lane_x_at_y(lane: dict, y_norm: float) -> float | None:
    pts = lane["points"]
    if pts.shape[0] < 2:
        return None
    order = np.argsort(pts[:, 1])
    ys = pts[order, 1]
    xs = pts[order, 0]
    if y_norm < ys[0] or y_norm > ys[-1]:
        return None
    return float(np.interp(y_norm, ys, xs))


def _centerline_x_at_y(left, right, y_norm):
    xl = _lane_x_at_y(left, y_norm)
    xr = _lane_x_at_y(right, y_norm)
    if xl is None or xr is None:
        return None
    return 0.5 * (xl + xr)


def best_on_side(lane_list, side):
    cands = [l for l in lane_list if _side_of(l) == side]
    if not cands:
        return None
    def _bottom_x(lane):
        pts = lane["points"]
        ys = pts[:, 1]
        bottom = pts[ys >= 0.5, 0] if (ys >= 0.5).sum() >= 2 else pts[:, 0]
        return float(np.mean(bottom))
    return min(cands, key=lambda l: abs(_bottom_x(l) - 0.5))


def compute_steering(left, right):
    """Return (steering_deg_geometric, lateral_err, centerline, lookahead).

    steering_deg_geometric is in lane-frame degrees; caller multiplies by
    STEER_AMP to get column degrees.
    """
    out = {
        "steering_deg": 0.0,
        "lateral_err": 0.0,
        "lookahead": None,
        "centerline": [],
    }
    if left is None or right is None:
        return out

    left_ys = left["points"][:, 1]
    right_ys = right["points"][:, 1]
    y_min = float(max(left_ys.min(), right_ys.min()))
    y_max = float(min(left_ys.max(), right_ys.max()))
    if y_max - y_min < 0.05:
        return out

    ys = np.linspace(y_min, y_max, 24)
    centerline = []
    for y in ys:
        cx = _centerline_x_at_y(left, right, float(y))
        if cx is not None:
            centerline.append((cx, float(y)))
    if not centerline:
        return out
    out["centerline"] = centerline

    cl_la = _centerline_x_at_y(left, right, LOOKAHEAD_Y)
    if cl_la is None:
        cl_la = centerline[len(centerline) // 2][0]
        y_la = centerline[len(centerline) // 2][1]
    else:
        y_la = LOOKAHEAD_Y
    lateral_err = cl_la - 0.5
    out["lookahead"] = (cl_la, y_la)
    out["lateral_err"] = lateral_err

    heading_deg = 0.0
    cl_near = _centerline_x_at_y(left, right, HEADING_Y_NEAR)
    cl_far = _centerline_x_at_y(left, right, HEADING_Y_FAR)
    if cl_near is not None and cl_far is not None:
        dx = cl_far - cl_near
        dy = HEADING_Y_NEAR - HEADING_Y_FAR
        heading_deg = math.degrees(math.atan2(dx, dy))
    out["heading_deg"] = heading_deg

    raw = K_LATERAL_DEG * lateral_err + K_HEADING_DEG * heading_deg
    out["steering_deg"] = float(np.clip(raw, -MAX_GEOMETRIC_DEG, MAX_GEOMETRIC_DEG))
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Lane overlay renderer (for the UI's lanes.jpg tile)
# ═══════════════════════════════════════════════════════════════════════════

EGO_COLOR = (60, 240, 240)
OFF_EGO_COLOR = (220, 180, 80)
DRIVABLE_FILL = (60, 220, 60)
CENTERLINE_COLOR = (255, 220, 0)
LOOKAHEAD_COLOR = (200, 80, 255)


def _denorm(pts, w, h):
    out = pts.copy().astype(np.float32)
    out[:, 0] *= w
    out[:, 1] *= h
    return out.astype(np.int32)


def _draw_lane_confidence(canvas: np.ndarray, lane: dict, color) -> None:
    pts = lane["points"]
    if pts.shape[0] == 0:
        return
    h, w = canvas.shape[:2]
    bottom_idx = int(np.argmax(pts[:, 1]))
    x = int(np.clip(round(pts[bottom_idx, 0] * w), 0, w - 1))
    y = int(np.clip(round(pts[bottom_idx, 1] * h), 16, h - 8))
    text = f"{float(lane.get('score', 0.0)):.2f}"
    cv2.putText(canvas, text, (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(canvas, text, (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)


def render_overlay(frame, lanes_norm, steer_state,
                    fresh: bool, mph_target: float,
                    mph_actual: float | None,
                    column_deg: float) -> np.ndarray:
    canvas = frame.copy()
    h, w = canvas.shape[:2]

    if lanes_norm:
        left = best_on_side(lanes_norm, "left")
        right = best_on_side(lanes_norm, "right")
        if left is None and right is not None:
            left = _mirror_lane(right)
        if right is None and left is not None:
            right = _mirror_lane(left)

        ego_pts = {}
        if left is not None:
            ego_pts["left"] = _denorm(left["points"], w, h)
        if right is not None:
            ego_pts["right"] = _denorm(right["points"], w, h)

        if "left" in ego_pts and "right" in ego_pts:
            poly = np.concatenate(
                [ego_pts["left"], ego_pts["right"][::-1]], axis=0
            )
            overlay = canvas.copy()
            cv2.fillPoly(overlay, [poly], DRIVABLE_FILL)
            cv2.addWeighted(overlay, 0.20, canvas, 0.80, 0, dst=canvas)

        for ego_lane in lanes_norm:
            side = _side_of(ego_lane)
            if (side == "left" and left is not None
                    and np.array_equal(ego_lane["points"], left["points"])):
                continue
            if (side == "right" and right is not None
                    and np.array_equal(ego_lane["points"], right["points"])):
                continue
            pts = _denorm(ego_lane["points"], w, h).reshape(-1, 1, 2)
            cv2.polylines(canvas, [pts], False, OFF_EGO_COLOR, 2, cv2.LINE_AA)
            _draw_lane_confidence(canvas, ego_lane, OFF_EGO_COLOR)

        for side, pts in ego_pts.items():
            cv2.polylines(canvas, [pts.reshape(-1, 1, 2)], False,
                          EGO_COLOR, 4, cv2.LINE_AA)
        if left is not None:
            _draw_lane_confidence(canvas, left, EGO_COLOR)
        if right is not None and (
                left is None or not np.array_equal(right["points"], left["points"])):
            _draw_lane_confidence(canvas, right, EGO_COLOR)

    cl = steer_state.get("centerline", [])
    if len(cl) >= 2:
        cl_pts = np.array(
            [(int(round(x * w)), int(round(y * h))) for x, y in cl],
            dtype=np.int32,
        ).reshape(-1, 1, 2)
        cv2.polylines(canvas, [cl_pts], False, CENTERLINE_COLOR, 2, cv2.LINE_AA)

    la = steer_state.get("lookahead")
    if la is not None:
        cx = int(round(la[0] * w))
        cy = int(round(la[1] * h))
        cv2.circle(canvas, (cx, cy), 9, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy), 7, LOOKAHEAD_COLOR, -1, cv2.LINE_AA)

    # HUD
    pad = 8
    scores = sorted(
        [float(l.get("score", 0.0)) for l in lanes_norm],
        reverse=True,
    )
    score_summary = "conf: " + (
        " ".join(f"{s:.2f}" for s in scores[:4]) if scores else "none"
    )
    box_w, box_h = 270, 112
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (box_w, box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0, dst=canvas)
    lines = [
        "CLRerNet  (clrnet)",
        f"steer: {column_deg:+6.1f} deg",
        f"target: {mph_target:.1f} mph"
        + (f"  cur: {mph_actual:.1f}" if mph_actual is not None else "  cur: —"),
        score_summary,
        "FRESH" if fresh else "FALLBACK",
    ]
    for i, txt in enumerate(lines):
        y = pad + (i + 1) * 18 - 4
        col = (60, 220, 60) if (i == 4 and fresh) else (240, 240, 240)
        cv2.putText(canvas, txt, (pad, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
    return canvas


# ═══════════════════════════════════════════════════════════════════════════
# CLRerNet wrapper
# ═══════════════════════════════════════════════════════════════════════════

class CLRerNetRunner:
    """Loads CLRerNet once, exposes ``infer(frame)`` returning lanes
    in normalized [0,1] coords with per-lane scores.
    """

    def __init__(self, config_path: str, ckpt_path: str, device: str = "cuda:0"):
        if CLRNET_REPO_DIR not in sys.path:
            sys.path.insert(0, CLRNET_REPO_DIR)
        prev_cwd = os.getcwd()
        os.chdir(CLRNET_REPO_DIR)
        try:
            # Stub the test split list the CULane dataloader expects.
            stub = Path(CLRNET_REPO_DIR) / "dataset/culane/list/test.txt"
            stub.parent.mkdir(parents=True, exist_ok=True)
            if not stub.exists():
                stub.touch()

            # Patch alaug.py once: albumentations 1.x rejects bboxes=None.
            alaug_path = Path(CLRNET_REPO_DIR) / "libs/datasets/pipelines/alaug.py"
            if alaug_path.exists():
                src = alaug_path.read_text()
                old_call = (
                    "        aug = self.__augmentor(\n"
                    "            image=img,\n"
                    "            keypoints=keypoints_val,\n"
                    "            bboxes=bboxes,\n"
                    "            mask=masks,\n"
                    "            bbox_labels=bbox_labels,\n"
                    "        )"
                )
                new_call = (
                    "        kwargs = dict(image=img)\n"
                    "        if keypoints_val is not None: kwargs['keypoints'] = keypoints_val\n"
                    "        if bboxes is not None:\n"
                    "            kwargs['bboxes'] = bboxes\n"
                    "            kwargs['bbox_labels'] = bbox_labels\n"
                    "        if masks is not None: kwargs['mask'] = masks\n"
                    "        aug = self.__augmentor(**kwargs)"
                )
                if old_call in src:
                    alaug_path.write_text(src.replace(old_call, new_call))
                    print("[clrnet] patched alaug.py for albumentations 1.x")

            # Disable ImageNet DLA34 pretrained download (URL unreachable
            # and overwritten by checkpoint anyway).
            base_cfg = Path(CLRNET_REPO_DIR) / "configs/clrernet/base_clrernet.py"
            if base_cfg.exists():
                src = base_cfg.read_text()
                if "pretrained=True" in src:
                    base_cfg.write_text(src.replace("pretrained=True", "pretrained=False"))
                    print("[clrnet] disabled DLA34 ImageNet pretrained download")

            import torch
            from mmdet.apis import init_detector

            from libs.datasets.metrics.culane_metric import interp
            from libs.datasets.pipelines import Compose

            self.torch = torch
            self.interp = interp

            print(f"[clrnet] init_detector(config={config_path})")
            self.model = init_detector(config_path, ckpt_path, device=device)
            self.model.bbox_head.test_cfg.as_lanes = False
            # Drop conf threshold so we get per-lane scores; we filter
            # client-side at DRAW_CONF.
            self.model.bbox_head.test_cfg.conf_threshold = 0.05

            cfg = self.model.cfg
            self.test_pipeline = Compose(cfg.test_dataloader.dataset.pipeline)
            self.device = device
            print("[clrnet] ready")
        finally:
            os.chdir(prev_cwd)

    def infer(self, frame_bgr: np.ndarray):
        """Return list[{"points": (N,2) normalized [0,1], "score": float}]."""
        ori_shape = frame_bgr.shape  # (H, W, 3)
        pipe_in = cv2.resize(frame_bgr, (CLRNET_INPUT_W, CLRNET_INPUT_H),
                             interpolation=cv2.INTER_LINEAR)
        data = dict(
            filename="frame.jpg",
            sub_img_name=None,
            img=pipe_in,
            gt_points=[],
            id_classes=[],
            id_instances=[],
            img_shape=pipe_in.shape,
            ori_shape=ori_shape,
        )
        data = self.test_pipeline(data)
        data_ = dict(
            inputs=[data["inputs"]],
            data_samples=[data["data_samples"]],
        )
        with self.torch.no_grad():
            results = self.model.test_step(data_)
        lanes = results[0]["lanes"]
        scores_t = results[0].get("scores")
        if hasattr(scores_t, "detach"):
            scores = scores_t.detach().cpu().numpy().tolist()
        elif scores_t is not None:
            scores = list(scores_t)
        else:
            scores = [1.0] * len(lanes)

        out = []
        ori_h, ori_w = ori_shape[0], ori_shape[1]
        for lane, sc in zip(lanes, scores):
            arr = lane.detach().cpu().numpy() if hasattr(lane, "detach") else np.asarray(lane)
            xs = arr[:, 0]
            ys = arr[:, 1]
            valid = (xs >= 0) & (xs < 1)
            if valid.sum() < 2:
                continue
            lx = xs[valid] * ori_w
            ly = ys[valid] * ori_h
            lx, ly = lx[::-1], ly[::-1]
            pred_pts = list(zip(lx, ly))
            try:
                interp_pts = self.interp(pred_pts, n=5)
            except Exception:
                interp_pts = np.asarray(pred_pts, dtype=np.float32)
            interp_pts = np.asarray(interp_pts, dtype=np.float32)
            if interp_pts.size == 0 or interp_pts.ndim != 2:
                continue
            interp_pts[:, 0] /= max(ori_w, 1)
            interp_pts[:, 1] /= max(ori_h, 1)
            out.append({"points": interp_pts, "score": float(sc)})
        return out


# ═══════════════════════════════════════════════════════════════════════════
# Speed controller
# ═══════════════════════════════════════════════════════════════════════════

def speed_targets(actual_mph: float | None, target_mph: float) -> tuple[float, float]:
    """Pure open-loop: constant gas, never brake. cart_state's mph readout
    is unreliable (bounces 0↔~9 when stopped), so closing any loop on it
    just makes the pedals chatter. Operator can always override with R2."""
    del actual_mph, target_mph  # intentionally ignored
    return float(min(GAS_CONSTANT, GAS_MAX)), 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Main loop
# ═══════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--frames-dir", default=str(FRAMES_DIR_DEFAULT))
    p.add_argument("--state-file", default=str(STATE_FILE_DEFAULT))
    p.add_argument("--cart-state-file", default=str(CART_STATE_FILE_DEFAULT))
    p.add_argument("--config", default=CLRNET_CONFIG)
    p.add_argument("--ckpt", default=CLRNET_CKPT)
    p.add_argument("--target-mph", type=float, default=TARGET_MPH)
    p.add_argument("--infer-hz", type=float, default=INFER_HZ_DEFAULT)
    p.add_argument("--jpeg-hz", type=float, default=PUBLISH_HZ_DEFAULT)
    p.add_argument("--indices", type=int, nargs="*", default=None)
    p.add_argument("--video", default=None,
                   help="Replay an MP4 instead of opening cameras.")
    p.add_argument("--no-loop", action="store_true")
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    frames_dir = Path(args.frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    state_path = Path(args.state_file)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    cart_state_path = Path(args.cart_state_file)

    # ── Cameras ────────────────────────────────────────────────────────────
    readers: list = []
    if args.video:
        if not Path(args.video).exists():
            print(f"ERROR: video not found: {args.video}", file=sys.stderr)
            return 1
        print(f"[video] replaying {args.video} (loop={'no' if args.no_loop else 'yes'})")
        video_src = VideoReader(args.video, loop=not args.no_loop)
        video_src.start()
        readers = [_BroadcastView(video_src, slug) for slug in SLUGS]
    else:
        print("[cams] discovering v4l2 devices...")
        indices = args.indices or discover_v4l2_indices(count=len(SLUGS))
        print(f"[cams] indices: {indices}")
        if not indices:
            print("ERROR: no cameras found")
            return 1
        for idx, slug in zip(indices, SLUGS):
            cap = open_camera(idx)
            if cap is None:
                print(f"[cams] WARN: idx {idx} ({slug}) failed")
                continue
            apply_front_camera_controls(idx, slug)
            r = CameraReader(cap, slug)
            r.start()
            readers.append(r)
            print(f"[cams] {slug:<12} -> /dev/video{idx}")

    if not readers:
        print("ERROR: opened 0 cameras")
        return 1

    slug_map = {r.slug: r for r in readers}
    if ACTIVE_SLUG not in slug_map:
        print(f"ERROR: required cam slug '{ACTIVE_SLUG}' not opened")
        return 1

    time.sleep(0.4)  # let cameras prime

    # ── Model ──────────────────────────────────────────────────────────────
    runner = CLRerNetRunner(args.config, args.ckpt, device=args.device)

    # ── Loop ───────────────────────────────────────────────────────────────
    jpeg_period = 1.0 / max(args.jpeg_hz, 1.0)
    infer_period = 1.0 / max(args.infer_hz, 0.1)
    next_jpeg_t = time.monotonic()
    next_infer_t = time.monotonic()
    last_log_t = 0.0

    smoothed_steer_deg = 0.0      # column deg, IIR-smoothed
    last_lanes: list = []         # last successful inference's lanes (norm)
    last_steer_state: dict = {"centerline": [], "lookahead": None,
                               "lateral_err": 0.0, "steering_deg": 0.0}
    last_fresh = False
    infer_count = 0
    infer_ms_ema = 0.0
    last_infer_t = 0.0

    print("[run] publishing to", frames_dir, "and", state_path)
    print(f"[run] target_mph={args.target_mph:.1f}  infer_hz={args.infer_hz:.1f}")
    try:
        while True:
            loop_t = time.monotonic()

            # ── 1) Inference on front_wide (rate-limited) ──
            if loop_t >= next_infer_t:
                next_infer_t = loop_t + infer_period
                frame = slug_map[ACTIVE_SLUG].latest()
                if frame is not None:
                    t0 = time.monotonic()
                    try:
                        lanes = runner.infer(frame)
                    except Exception as e:
                        print(f"[infer] failed: {type(e).__name__}: {e}",
                              flush=True)
                        lanes = []
                    dt_ms = (time.monotonic() - t0) * 1000.0
                    infer_ms_ema = 0.7 * infer_ms_ema + 0.3 * dt_ms
                    infer_count += 1
                    last_infer_t = time.monotonic()

                    # Persist last good detection when CLRerNet returns 0
                    # lanes this frame — same idea as visualize_polarrcnn.py's
                    # PERSIST. Without it the steering snaps to 0 on every
                    # dropout and the cart wobbles. We only "see" empty by
                    # using the previous frame's lanes; freshness reflects
                    # the *current* frame so the HUD still tells the truth.
                    if not lanes and last_lanes:
                        effective_lanes = last_lanes
                    else:
                        effective_lanes = lanes
                        if lanes:
                            last_lanes = lanes

                    # Filter for "fresh" (above DRAW_CONF). If the fresh
                    # set is bilateral, use it; else fall back to all
                    # lanes per-side.
                    fresh_lanes = [l for l in effective_lanes
                                   if float(l.get("score", 0.0)) >= DRAW_CONF]
                    fresh_left = best_on_side(fresh_lanes, "left")
                    fresh_right = best_on_side(fresh_lanes, "right")
                    chosen_left = fresh_left or best_on_side(effective_lanes, "left")
                    chosen_right = fresh_right or best_on_side(effective_lanes, "right")
                    if chosen_left is None and chosen_right is not None:
                        chosen_left = _mirror_lane(chosen_right)
                    if chosen_right is None and chosen_left is not None:
                        chosen_right = _mirror_lane(chosen_left)

                    last_fresh = (fresh_left is not None
                                  and fresh_right is not None
                                  and bool(lanes))
                    last_steer_state = compute_steering(chosen_left, chosen_right)
                    # Sign flip: live test on the cart showed the wheel
                    # turning the opposite way the model wanted (left lane
                    # error → cart steered right). The viz overlay was
                    # correct, only the column command was inverted —
                    # negate here so steer_deg matches cart yaw convention.
                    geom_deg = -last_steer_state["steering_deg"]
                    raw_col_deg = float(np.clip(geom_deg * STEER_AMP,
                                                 -STEER_CLAMP_DEG,
                                                  STEER_CLAMP_DEG))
                    smoothed_steer_deg = (
                        STEER_ALPHA * smoothed_steer_deg
                        + (1.0 - STEER_ALPHA) * raw_col_deg
                    )

            # ── 2) Speed controller (independent of infer rate) ──
            actual_mph = read_cart_mph(cart_state_path)
            target_gas, target_brake = speed_targets(actual_mph, args.target_mph)

            # ── 3) Publish per-camera JPEGs + lanes overlay ──
            if loop_t >= next_jpeg_t:
                next_jpeg_t = loop_t + jpeg_period
                for r in readers:
                    f = r.latest()
                    if f is None:
                        continue
                    write_jpeg_atomic(frames_dir / f"{r.slug}.jpg", f)
                fw_frame = slug_map[ACTIVE_SLUG].latest()
                if fw_frame is not None and infer_count > 0:
                    overlay = render_overlay(
                        fw_frame, last_lanes, last_steer_state,
                        fresh=last_fresh,
                        mph_target=args.target_mph,
                        mph_actual=actual_mph,
                        column_deg=smoothed_steer_deg,
                    )
                    write_jpeg_atomic(frames_dir / "lanes.jpg", overlay,
                                       quality=80)

            # ── 4) Publish state ──
            infer_age = (time.monotonic() - last_infer_t) if last_infer_t else float("inf")
            inference = infer_count > 0 and infer_age < 1.0 and (
                last_steer_state.get("centerline") or last_lanes
            )
            state_payload = {
                "steer_deg": float(smoothed_steer_deg),
                "steer_deg_raw": float(last_steer_state.get("steering_deg", 0.0)
                                        * STEER_AMP),
                "active_cam": ACTIVE_SLUG,
                "inference": bool(inference),
                "viz": bool(inference),
                "viz_streams": ["lanes"] if inference else [],
                "object_count": 0,
                "fps": float(1000.0 / max(infer_ms_ema, 1.0))
                       if infer_ms_ema > 0 else 0.0,
                "cams": [r.slug for r in readers],
                "source": "video" if args.video else "camera",
                "model": MODEL_NAME,
                "model_full": "clrernet-ema-dla34-culane",
                "target_speed_mph": float(args.target_mph) if inference else 0.0,
                "target_gas": float(target_gas) if inference else 0.0,
                "target_brake": float(target_brake) if inference else 0.0,
                "predicted_path": [],
                "ts": time.time(),
                "clrnet": {
                    "infer_count": int(infer_count),
                    "infer_ms": float(infer_ms_ema),
                    "infer_age_s": float(infer_age) if infer_age != float("inf") else None,
                    "actual_mph": float(actual_mph) if actual_mph is not None else None,
                    "n_lanes": len(last_lanes),
                    "fresh": bool(last_fresh),
                    "lateral_err": float(last_steer_state.get("lateral_err", 0.0)),
                },
            }
            write_state_atomic(state_path, state_payload)

            # ── 5) Periodic log ──
            if loop_t - last_log_t > 1.0:
                last_log_t = loop_t
                mph_str = f"{actual_mph:4.1f}" if actual_mph is not None else " — "
                print(
                    f"[run] infer={infer_count} ms={infer_ms_ema:5.1f} "
                    f"lanes={len(last_lanes)} fresh={'Y' if last_fresh else 'N'} "
                    f"steer={smoothed_steer_deg:+6.1f}° "
                    f"mph={mph_str}/{args.target_mph:.1f} "
                    f"gas={target_gas:.3f} brake={target_brake:.3f}",
                    flush=True,
                )

            time.sleep(0.02)
    except KeyboardInterrupt:
        print("\n[run] interrupted.")
    finally:
        for r in readers:
            r.stop()
        print("[run] done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

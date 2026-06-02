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
import bisect
import json
import math
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import zlib
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import seg_fast  # noqa: E402
import seg_occupancy  # noqa: E402
import bev_fusion  # noqa: E402


SEG_REPO_DEFAULT = Path(
    os.environ.get(
        "SEGMENTATION_HOMOGRAPHY_REPO",
        str(Path.home() / "Programming/drive-by-segmentation"),
    )
)
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
VIDEO_CONTROL_FILE_DEFAULT = Path(os.environ.get("VIDEO_CONTROL_FILE", "/tmp/video_control.json"))

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
CLRNET_CONF_THRESHOLD = 0.60
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

# Autospeed controller: unified path-aware obstacle speed management. Replaces
# the old TTC-brake + PI speed-control stack with a single controller that
# outputs a commanded_speed in m/s from path geometry and obstacle predictions.
AUTOSPEED_PATH_WIDTH = 2.0        # half-width (m) of the corridor around the path
AUTOSPEED_COMFORT_DECEL = 1.5     # max deceleration for normal smooth stops (m/s^2)
AUTOSPEED_EMERGENCY_DECEL = 3.5   # max deceleration for emergency only (m/s^2)
AUTOSPEED_MAX_JERK = 0.9          # max rate of change of acceleration (m/s^3)
AUTOSPEED_LOOKAHEAD_TIME = 6.0    # how far ahead in time to care about obstacles (s)
AUTOSPEED_MIN_GAP = 2.0           # absolute minimum distance to maintain (m)
AUTOSPEED_REACTION_BUFFER = 0.5   # extra time margin for safety (s)
AUTOSPEED_CREEP_SPEED = 0.5       # speed for inching past uncertain situations (m/s)
AUTOSPEED_DT = 0.1                # control cycle period (s)
# When a hard protective stop is active, freeze steering (hold the last command)
# instead of letting the road-mask centerline swerve around the obstacle.
ENV_BRAKE_FREEZE_STEER = True
# Legacy constants kept for BEV viz corridor drawing
ENV_BRAKE_CORRIDOR_HALF_M = AUTOSPEED_PATH_WIDTH
ENV_BRAKE_NEAR_STOP_M = AUTOSPEED_MIN_GAP

# Stop sign controller: detects stop signs via YOLO, estimates distance from
# BEV projection, and manages a smooth stop→wait→depart cycle.
STOP_SIGN_APPROACH_M = 18.0
STOP_SIGN_STOP_BUFFER_M = 4.0
STOP_SIGN_WAIT_S = 1.5
STOP_SIGN_DEPART_RAMP_S = 2.5
STOP_SIGN_MIN_CONF = 0.28
STOP_SIGN_MIN_BBOX_AREA = 60
STOP_SIGN_LOST_FRAMES = 90

CITYSCAPES_COLORS = [
    (128, 64, 128),   # road
    (244, 35, 232),   # sidewalk
    (70, 70, 70),     # building
    (102, 102, 156),  # wall
    (190, 153, 153),  # fence
    (153, 153, 153),  # pole
    (250, 170, 30),   # traffic light
    (220, 220, 0),    # traffic sign
    (107, 142, 35),   # vegetation
    (152, 251, 152),  # terrain
    (70, 130, 180),   # sky
    (220, 20, 60),    # person
    (255, 0, 0),      # rider
    (0, 0, 142),      # car
    (0, 0, 70),       # truck
    (0, 60, 100),     # bus
    (0, 80, 100),     # train
    (0, 0, 230),      # motorcycle
    (119, 11, 32),    # bicycle
]

MODEL_VARIANTS = {
    "b0": "nvidia/segformer-b0-finetuned-cityscapes-1024-1024",
    "b2": "nvidia/segformer-b2-finetuned-cityscapes-1024-1024",
    "b5": "nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
}


def load_segformer(variant: str, device: str):
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

    name = MODEL_VARIANTS[variant]
    print(f"Loading {name} on {device}...", flush=True)
    t0 = time.time()
    proc = SegformerImageProcessor.from_pretrained(name)
    model = SegformerForSemanticSegmentation.from_pretrained(name).to(device).eval()
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Loaded in {time.time() - t0:.1f}s ({params:.1f}M params)", flush=True)
    return proc, model


def create_overlay(frame_rgb: np.ndarray, seg_map: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    palette = np.asarray(CITYSCAPES_COLORS, dtype=np.uint8)
    color_mask = palette[np.clip(seg_map, 0, len(palette) - 1)]
    return cv2.addWeighted(frame_rgb, 1.0 - alpha, color_mask, alpha, 0.0)


def create_bev(seg_map: np.ndarray, calib: dict, bev_size: int = 500,
               return_class_map: bool = False):
    remap = seg_fast.build_bev_remap(calib, seg_map.shape[0], seg_map.shape[1], bev_size)
    palette = np.array(CITYSCAPES_COLORS, dtype=np.uint8)
    bev = seg_fast.create_bev_cached(seg_map, palette, remap)
    if return_class_map:
        return bev, bev_class_map_cached(seg_map, remap)
    return bev


def object_viz_color_rgb(class_name: object) -> tuple[int, int, int]:
    name = str(class_name or "").lower()
    if "stop sign" in name or "sign" in name or "light" in name:
        return (216, 164, 192)
    if "person" in name or "rider" in name:
        return (245, 137, 55)
    if "motorcycle" in name or "bicycle" in name or "bike" in name:
        return (45, 172, 155)
    if any(v in name for v in ("car", "truck", "bus", "train", "vehicle")):
        return (105, 166, 232)
    return (126, 151, 180)


def object_dimensions_3d_m(class_name: object) -> tuple[float, float, float]:
    name = str(class_name or "").lower()
    if "bus" in name:
        return 11.5, 2.6, 3.2
    if "truck" in name:
        return 7.0, 2.5, 3.0
    if "car" in name or "vehicle" in name:
        return 4.4, 2.0, 1.6
    if "motorcycle" in name or "bicycle" in name or "bike" in name:
        return 1.9, 0.7, 1.6
    if "person" in name or "rider" in name:
        return 0.8, 0.6, 1.75
    if "stop sign" in name or "sign" in name or "light" in name:
        return 0.55, 0.55, 1.9
    return 1.2, 0.9, 1.5


def object_heading_rad(obj: dict) -> float:
    vx = float(obj.get("vx_mps", 0.0) or 0.0)
    vy = float(obj.get("vy_mps", 0.0) or 0.0)
    if math.hypot(vx, vy) > 0.05:
        return math.atan2(vy, vx)
    future = obj.get("future_m") if isinstance(obj.get("future_m"), list) else []
    if future:
        try:
            fx, fy = float(future[-1][0]), float(future[-1][1])
            x = float(obj["x_m"])
            y = float(obj["y_m"])
            if math.hypot(fx - x, fy - y) > 0.05:
                return math.atan2(fy - y, fx - x)
        except (KeyError, TypeError, ValueError, IndexError):
            pass
    return 0.0


def object_cuboid_local_m(obj: dict) -> dict | None:
    try:
        fwd = float(obj["x_m"])
        lat = float(obj["y_m"])
    except (KeyError, TypeError, ValueError):
        return None
    length_m, width_m, height_m = object_dimensions_3d_m(obj.get("class_name"))
    theta = object_heading_rad(obj)
    forward = np.array([math.cos(theta), math.sin(theta)], dtype=np.float32)
    left = np.array([-math.sin(theta), math.cos(theta)], dtype=np.float32)
    center = np.array([fwd, lat], dtype=np.float32)
    bottom: list[tuple[float, float, float]] = []
    top: list[tuple[float, float, float]] = []
    for lf, wl in ((1, 1), (1, -1), (-1, -1), (-1, 1)):
        p = center + forward * (lf * length_m * 0.5) + left * (wl * width_m * 0.5)
        bottom.append((float(p[0]), float(p[1]), 0.0))
        top.append((float(p[0]), float(p[1]), float(height_m)))
    return {
        "center_m": [round(fwd, 3), round(lat, 3), round(height_m * 0.5, 3)],
        "size_m": [round(length_m, 3), round(width_m, 3), round(height_m, 3)],
        "yaw_rad": round(float(theta), 4),
        "corners_m": bottom + top,
        "source": "yolo_ground_plane_class_priors",
    }


def mono3d_objects_to_tracks(objects: list[dict] | None) -> list[dict]:
    tracks: list[dict] = []
    for i, obj in enumerate(objects or []):
        if not isinstance(obj, dict):
            continue
        box = obj.get("camera_box3d")
        if not isinstance(box, list) or len(box) < 7:
            continue
        try:
            cam_x = float(box[0])
            cam_z = float(box[2])
            length_m = abs(float(box[3]))
            height_m = abs(float(box[4]))
            width_m = abs(float(box[5]))
            yaw_rad = float(box[6])
        except (TypeError, ValueError):
            continue
        if not all(math.isfinite(v) for v in (cam_x, cam_z, length_m, width_m, height_m, yaw_rad)):
            continue
        fwd_m = cam_z
        left_m = cam_x
        if fwd_m <= 0.2 or fwd_m > 90.0 or abs(left_m) > 50.0:
            continue

        vx_mps = 0.0
        vy_mps = 0.0
        if len(box) >= 9:
            try:
                vx_mps = float(box[8])
                vy_mps = float(box[7])
            except (TypeError, ValueError):
                vx_mps = 0.0
                vy_mps = 0.0
        speed_mps = math.hypot(vx_mps, vy_mps)
        future: list[list[float]] = []
        if speed_mps > 0.05:
            for step_s in (0.5, 1.0, 1.5, 2.0):
                future.append([
                    round(float(fwd_m + vx_mps * step_s), 3),
                    round(float(left_m + vy_mps * step_s), 3),
                ])

        confidence = float(obj.get("confidence", 0.0) or 0.0)
        class_name = str(obj.get("class_name") or "object")
        track = {
            "track_id": int(300000 + i),
            "class_id": int(obj.get("class_id", -1) or -1),
            "class_name": class_name,
            "confidence": confidence,
            "x_m": round(float(fwd_m), 3),
            "y_m": round(float(left_m), 3),
            "distance_m": round(float(math.hypot(fwd_m, left_m)), 3),
            "vx_mps": round(float(vx_mps), 3),
            "vy_mps": round(float(vy_mps), 3),
            "speed_mps": round(float(speed_mps), 3),
            "future_m": future,
            "future_modes": [{"prob": 1.0, "future_m": future, "source": "mono3d_velocity"}],
            "future_source": "mono3d_velocity",
            "length_m": round(float(length_m), 3),
            "width_m": round(float(width_m), 3),
            "height_m": round(float(height_m), 3),
            "yaw_rad": round(float(yaw_rad), 4),
            "camera_box3d": [float(x) for x in box],
            "box3d": {
                "center_m": [round(float(fwd_m), 3), round(float(left_m), 3), round(float(height_m * 0.5), 3)],
                "size_m": [round(float(length_m), 3), round(float(width_m), 3), round(float(height_m), 3)],
                "yaw_rad": round(float(yaw_rad), 4),
                "source": "mmdet3d_fcos3d_camera_box",
            },
            "provider": obj.get("provider", "mmdet3d_fcos3d_nuscenes"),
        }
        tracks.append(track)
    return tracks


def draw_mono3d_overlay(
    frame_bgr: np.ndarray,
    objects: list[dict] | None,
    calib: dict,
) -> np.ndarray:
    out = frame_bgr.copy()
    h, w = out.shape[:2]
    intr = calib.get("intrinsics", {})
    try:
        fx = float(intr.get("fx", intr.get("focal_length")))
        fy = float(intr.get("fy", intr.get("focal_length", fx)))
        cx = float(intr["cx"])
        cy = float(intr["cy"])
    except (KeyError, TypeError, ValueError):
        return out

    def project(pt: tuple[float, float, float]) -> tuple[int, int] | None:
        x, y, z = pt
        if z <= 0.15 or not all(math.isfinite(v) for v in (x, y, z)):
            return None
        u = fx * x / z + cx
        v = fy * y / z + cy
        if not (math.isfinite(u) and math.isfinite(v)):
            return None
        return int(round(u)), int(round(v))

    def draw_line(a: tuple[int, int] | None, b: tuple[int, int] | None, color: tuple[int, int, int], thick: int) -> None:
        if a is None or b is None:
            return
        ax, ay = a
        bx, by = b
        if (
            max(ax, bx) < -w or min(ax, bx) > 2 * w
            or max(ay, by) < -h or min(ay, by) > 2 * h
        ):
            return
        cv2.line(out, a, b, color, thick, cv2.LINE_AA)

    drawn = 0
    sorted_objects = sorted(
        [o for o in (objects or []) if isinstance(o, dict)],
        key=lambda o: float(o.get("confidence", 0.0) or 0.0),
        reverse=True,
    )
    for obj in sorted_objects[:80]:
        box = obj.get("camera_box3d")
        if not isinstance(box, list) or len(box) < 7:
            continue
        try:
            x = float(box[0])
            y_bottom = float(box[1])
            z = float(box[2])
            size_x = max(0.05, abs(float(box[3])))
            size_y = max(0.05, abs(float(box[4])))
            size_z = max(0.05, abs(float(box[5])))
            yaw = float(box[6])
            conf = float(obj.get("confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        if z <= 0.15 or not all(math.isfinite(v) for v in (x, y_bottom, z, size_x, size_y, size_z, yaw)):
            continue

        c = math.cos(yaw)
        s = math.sin(yaw)
        bottom_pts: list[tuple[float, float, float]] = []
        top_pts: list[tuple[float, float, float]] = []
        for x_sign, z_sign in ((1, 1), (1, -1), (-1, -1), (-1, 1)):
            lx = x_sign * size_x * 0.5
            lz = z_sign * size_z * 0.5
            # Camera coordinates: x right, y down, z forward. FCOS3D yaw is
            # around the camera vertical axis; this is enough for a clear
            # overlay even when exact dataset conventions vary slightly.
            px = x + c * lx + s * lz
            pz = z - s * lx + c * lz
            bottom_pts.append((px, y_bottom, pz))
            top_pts.append((px, y_bottom - size_y, pz))

        bottom = [project(p) for p in bottom_pts]
        top = [project(p) for p in top_pts]
        visible = [p for p in bottom + top if p is not None and -w <= p[0] <= 2 * w and -h <= p[1] <= 2 * h]
        if not visible:
            center = project((x, y_bottom - size_y * 0.5, z))
            if center is None:
                continue
            visible = [center]

        color_rgb = object_viz_color_rgb(obj.get("class_name"))
        color = tuple(int(v) for v in reversed(color_rgb))
        shadow = (0, 0, 0)
        for ring in (bottom, top):
            for a, b in zip(ring, ring[1:] + ring[:1]):
                draw_line(a, b, shadow, 5)
                draw_line(a, b, color, 2)
        for a, b in zip(bottom, top):
            draw_line(a, b, shadow, 5)
            draw_line(a, b, color, 2)

        anchor = min(visible, key=lambda p: p[1])
        label = f"{obj.get('class_name', 'object')} {conf:.2f}"
        tx = max(2, min(w - 2, anchor[0]))
        ty = max(14, min(h - 4, anchor[1] - 6))
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        cv2.rectangle(out, (tx, ty - th - 5), (min(w - 1, tx + tw + 6), ty + 3), shadow, -1)
        cv2.putText(out, label, (tx + 3, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
        drawn += 1

    summary = f"FCOS3D {drawn}/{len(objects or [])}"
    cv2.rectangle(out, (8, 8), (150, 30), (0, 0, 0), -1)
    cv2.putText(out, summary, (14, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def draw_yolo_overlay(
    frame_bgr: np.ndarray,
    objects: list[dict] | None,
    projector: "GroundPlaneProjector | None" = None,
) -> np.ndarray:
    out = frame_bgr.copy()
    for obj in objects or []:
        xyxy = obj.get("xyxy") if isinstance(obj, dict) else None
        if not isinstance(xyxy, list) or len(xyxy) != 4:
            continue
        try:
            x1, y1, x2, y2 = [int(round(float(v))) for v in xyxy]
        except (TypeError, ValueError):
            continue
        h, w = out.shape[:2]
        x1, x2 = sorted((max(0, min(w - 1, x1)), max(0, min(w - 1, x2))))
        y1, y2 = sorted((max(0, min(h - 1, y1)), max(0, min(h - 1, y2))))
        if x2 <= x1 or y2 <= y1:
            continue
        color_rgb = object_viz_color_rgb(obj.get("class_name"))
        color_bgr = tuple(reversed(color_rgb))
        box3d = obj.get("box3d")
        if projector is not None and isinstance(box3d, dict):
            corners = box3d.get("corners_m")
            pts = []
            if isinstance(corners, list) and len(corners) == 8:
                for corner in corners:
                    if not isinstance(corner, (list, tuple)) or len(corner) < 3:
                        pts = []
                        break
                    px = projector.local_to_image(float(corner[0]), float(corner[1]), float(corner[2]))
                    if px is None:
                        pts = []
                        break
                    pts.append((int(round(px[0])), int(round(px[1]))))
            if len(pts) == 8:
                edges = (
                    (0, 1), (1, 2), (2, 3), (3, 0),
                    (4, 5), (5, 6), (6, 7), (7, 4),
                    (0, 4), (1, 5), (2, 6), (3, 7),
                )
                for a, b in edges:
                    cv2.line(out, pts[a], pts[b], color_bgr, 2, cv2.LINE_AA)
            else:
                cv2.rectangle(out, (x1, y1), (x2, y2), color_bgr, 2, cv2.LINE_AA)
        else:
            cv2.rectangle(out, (x1, y1), (x2, y2), color_bgr, 2, cv2.LINE_AA)
        label = str(obj.get("class_name") or "object")
        conf = obj.get("confidence")
        try:
            label = f"3D {label} {float(conf):.2f}"
        except (TypeError, ValueError):
            label = f"3D {label}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        label_y1 = max(0, y1 - th - baseline - 5)
        label_y2 = label_y1 + th + baseline + 5
        label_x2 = min(w - 1, x1 + tw + 8)
        cv2.rectangle(out, (x1, label_y1), (label_x2, label_y2), color_bgr, -1)
        cv2.putText(
            out,
            label,
            (x1 + 4, label_y2 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return out


class SteeringEstimator:
    BEV_EMA = 0.30
    W_BEV = 142.0
    INTERCEPT = 9.0

    def __init__(self) -> None:
        self._ema_heading = 0.0

    @property
    def steering_deg(self) -> float:
        raw = self.W_BEV * self._ema_heading + self.INTERCEPT
        return max(-270.0, min(270.0, raw))

    def update_bev(self, lane_local: np.ndarray | None) -> None:
        if lane_local is None or len(lane_local) < 4:
            return
        fwd = lane_local[:, 0]
        left = lane_local[:, 1]
        mask = (fwd > 0.5) & (fwd < 3.0)
        if mask.sum() >= 2:
            raw_heading = math.atan2(
                left[mask][-1] - left[mask][0],
                fwd[mask][-1] - fwd[mask][0],
            )
            self._ema_heading = (
                self.BEV_EMA * raw_heading + (1.0 - self.BEV_EMA) * self._ema_heading
            )


class SegRuntime(SimpleNamespace):
    FT_TO_M = 0.3048
    RANGE_FWD = 50 * FT_TO_M
    RANGE_SIDE = 25 * FT_TO_M
    BEV_SIZE = 500
    LOOKAHEAD_FT = 25.0
    LOOKAHEAD_M = LOOKAHEAD_FT * FT_TO_M
    SteeringEstimator = SteeringEstimator

    def lookahead_point(self, traj_local: np.ndarray | None):
        if traj_local is None or len(traj_local) < 4:
            return None, 0.0
        fwd = traj_local[1:, 0]
        left = traj_local[1:, 1]
        if len(fwd) < 5:
            return None, 0.0
        coeffs = np.polyfit(fwd, left, 2)
        la_fwd_val = self.LOOKAHEAD_M
        la_left_val = float(np.polyval(coeffs, la_fwd_val))
        return (la_fwd_val, la_left_val), self.LOOKAHEAD_FT

    def draw_trajectory(self, bev_img: np.ndarray, traj_bev: np.ndarray | None,
                        color, thickness: int = 3, label: str | None = None) -> None:
        if traj_bev is None or len(traj_bev) < 2:
            return
        pts = traj_bev.copy()
        valid = (
            (pts[:, 0] >= 0)
            & (pts[:, 0] < self.BEV_SIZE)
            & (pts[:, 1] >= 0)
            & (pts[:, 1] < self.BEV_SIZE)
        )
        if valid.sum() < 2:
            return
        for i in range(len(pts) - 1):
            if valid[i] and valid[i + 1]:
                cv2.line(
                    bev_img,
                    (int(pts[i, 0]), int(pts[i, 1])),
                    (int(pts[i + 1, 0]), int(pts[i + 1, 1])),
                    color,
                    thickness,
                    cv2.LINE_AA,
                )
        for i in range(0, len(pts), 5):
            if valid[i]:
                cv2.circle(bev_img, (int(pts[i, 0]), int(pts[i, 1])), 3, color, -1, cv2.LINE_AA)
        if label:
            for i in range(len(pts)):
                if valid[i]:
                    cv2.putText(
                        bev_img,
                        label,
                        (int(pts[i, 0]) + 6, int(pts[i, 1]) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.35,
                        color,
                        1,
                        cv2.LINE_AA,
                    )
                    break


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
    def __init__(self, video_path: str, loop: bool = True,
                 control_file: Path | None = None):
        super().__init__(daemon=True, name="video-reader")
        self.video_path = video_path
        self.loop = loop
        self.control_file = control_file
        self.lock = threading.Lock()
        self.frame: np.ndarray | None = None
        self.frame_count = 0
        self.last_ok_s = 0.0
        self.fps = 30.0
        self.duration_s = 0.0
        self.total_frames = 0
        self.position_s = 0.0
        self.source_frame_index = 0
        self.paused = False
        self.last_control_seq = None
        self._stop = threading.Event()

    def _read_control(self) -> dict | None:
        if self.control_file is None:
            return None
        try:
            with self.control_file.open() as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, OSError, TypeError):
            return None
        return data if isinstance(data, dict) else None

    def _apply_control(self, cap: cv2.VideoCapture) -> bool:
        data = self._read_control()
        if not data:
            return False
        seq = data.get("seq")
        pause_changed = False
        with self.lock:
            old_paused = self.paused
            self.paused = bool(data.get("pause", False))
            pause_changed = old_paused != self.paused
        if seq == self.last_control_seq:
            return pause_changed
        self.last_control_seq = seq
        seek_s = data.get("seek_s")
        try:
            seek_s = float(seek_s)
        except (TypeError, ValueError):
            return pause_changed
        if self.duration_s > 0.0:
            seek_s = float(np.clip(seek_s, 0.0, max(0.0, self.duration_s - 1e-3)))
        else:
            seek_s = max(0.0, seek_s)
        cap.set(cv2.CAP_PROP_POS_MSEC, seek_s * 1000.0)
        with self.lock:
            self.position_s = seek_s
            self.source_frame_index = int(round(seek_s * max(self.fps, 1e-6)))
        return True

    def status(self) -> dict:
        with self.lock:
            return {
                "path": self.video_path,
                "duration_s": float(self.duration_s),
                "position_s": float(self.position_s),
                "fps": float(self.fps),
                "frame_index": int(self.source_frame_index),
                "frame_count": int(self.total_frames),
                "reader_frame_count": int(self.frame_count),
                "paused": bool(self.paused),
                "loop": bool(self.loop),
                "scrubbable": self.control_file is not None,
            }

    def run(self) -> None:
        while not self._stop.is_set():
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                print(f"[video] failed to open {self.video_path}", flush=True)
                return
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            duration_s = (total_frames / fps) if total_frames > 0 and fps > 0 else 0.0
            with self.lock:
                self.fps = float(fps)
                self.total_frames = int(total_frames)
                self.duration_s = float(duration_s)
            period = 1.0 / fps
            next_t = time.monotonic()
            while not self._stop.is_set():
                control_changed = self._apply_control(cap)
                with self.lock:
                    paused = self.paused
                if paused:
                    if control_changed:
                        ok, frame = cap.read()
                        if ok and frame is not None:
                            frame = cv2.resize(frame, (CAM_W, CAM_H), interpolation=cv2.INTER_AREA)
                            pos_msec = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
                            src_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES) or 0)
                            with self.lock:
                                self.frame = frame
                                self.frame_count += 1
                                self.last_ok_s = time.monotonic()
                                self.position_s = pos_msec / 1000.0
                                self.source_frame_index = src_frame
                    next_t = time.monotonic() + period
                    time.sleep(0.05)
                    continue
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                frame = cv2.resize(frame, (CAM_W, CAM_H), interpolation=cv2.INTER_AREA)
                pos_msec = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
                src_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES) or 0)
                with self.lock:
                    self.frame = frame
                    self.frame_count += 1
                    self.last_ok_s = time.monotonic()
                    self.position_s = pos_msec / 1000.0
                    self.source_frame_index = src_frame
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


def write_png_atomic(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", img)
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


class ModalSegmentationClient:
    def __init__(self, app_name: str, function_name: str, variant: str):
        import modal

        self.variant = variant
        self.fn = modal.Function.from_name(app_name, function_name)

    def segment(self, frame_rgb: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        timings: dict[str, float] = {}

        t = time.perf_counter()
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        ok, enc = cv2.imencode(
            ".jpg",
            frame_bgr,
            [int(cv2.IMWRITE_JPEG_QUALITY), 90],
        )
        if not ok:
            raise RuntimeError("failed to JPEG-encode frame for Modal segmentation")
        jpeg_bytes = enc.tobytes()
        timings["modal_jpeg_encode_ms"] = (time.perf_counter() - t) * 1000.0
        timings["modal_jpeg_bytes"] = float(len(jpeg_bytes))

        t = time.perf_counter()
        result = self.fn.remote(jpeg_bytes, self.variant)
        timings["modal_roundtrip_ms"] = (time.perf_counter() - t) * 1000.0

        t = time.perf_counter()
        if result.get("dtype") != "uint8":
            raise RuntimeError(f"Modal returned unsupported dtype: {result.get('dtype')}")
        shape = tuple(int(x) for x in result["shape"])
        raw = zlib.decompress(result["zlib"])
        seg = np.frombuffer(raw, dtype=np.uint8).reshape(shape).copy()
        timings["modal_decode_ms"] = (time.perf_counter() - t) * 1000.0

        for k, v in (result.get("timings_ms") or {}).items():
            try:
                timings[k] = float(v)
            except (TypeError, ValueError):
                pass
        return seg, timings


def timed_segment_frame_modal(
    frame_rgb: np.ndarray,
    client: ModalSegmentationClient,
) -> tuple[np.ndarray, dict[str, float]]:
    return client.segment(frame_rgb)


class ModalMonocular3DClient:
    def __init__(self, app_name: str, function_name: str, calib: dict, score_thr: float):
        import modal

        intr = calib["intrinsics"]
        self.fx = float(intr.get("fx", intr.get("focal_length")))
        self.fy = float(intr.get("fy", intr.get("focal_length", self.fx)))
        self.cx = float(intr["cx"])
        self.cy = float(intr["cy"])
        self.score_thr = float(score_thr)
        self.fn = modal.Function.from_name(app_name, function_name)
        self.provider = f"{app_name}/{function_name}"
        self.last_ok = False
        self.last_error = ""
        self.last_latency_ms = 0.0

    def detect(self, frame_bgr: np.ndarray) -> tuple[np.ndarray | None, list[dict], dict[str, float]]:
        import base64

        timings: dict[str, float] = {}
        t = time.perf_counter()
        ok, enc = cv2.imencode(
            ".jpg",
            frame_bgr,
            [int(cv2.IMWRITE_JPEG_QUALITY), 90],
        )
        if not ok:
            raise RuntimeError("failed to JPEG-encode frame for Modal 3D detection")
        jpeg_bytes = enc.tobytes()
        timings["mono3d_jpeg_encode_ms"] = (time.perf_counter() - t) * 1000.0
        timings["mono3d_jpeg_bytes"] = float(len(jpeg_bytes))

        t = time.perf_counter()
        try:
            result = self.fn.remote(
                jpeg_bytes,
                self.fx,
                self.fy,
                self.cx,
                self.cy,
                self.score_thr,
            )
        except Exception as exc:
            self.last_ok = False
            self.last_error = f"{type(exc).__name__}: {exc}"
            self.last_latency_ms = (time.perf_counter() - t) * 1000.0
            timings["mono3d_roundtrip_ms"] = self.last_latency_ms
            return None, [], timings
        self.last_latency_ms = (time.perf_counter() - t) * 1000.0
        timings["mono3d_roundtrip_ms"] = self.last_latency_ms

        for k, v in (result.get("timings_ms") or {}).items():
            try:
                timings[f"mono3d_{k}"] = float(v)
            except (TypeError, ValueError):
                pass

        viz_bgr = None
        jpeg_b64 = result.get("viz_jpeg_b64")
        if isinstance(jpeg_b64, str) and jpeg_b64:
            raw = base64.b64decode(jpeg_b64)
            arr = np.frombuffer(raw, dtype=np.uint8)
            viz_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        objects = result.get("objects") if isinstance(result.get("objects"), list) else []
        provider = str(result.get("provider") or self.provider)
        for obj in objects:
            if isinstance(obj, dict):
                obj["provider"] = provider
        self.provider = provider
        self.last_ok = viz_bgr is not None
        self.last_error = "" if self.last_ok else "no_visualization_returned"
        return viz_bgr, objects, timings

    def status(self) -> dict:
        return {
            "type": "modal_mmdet3d_fcos3d",
            "provider": self.provider,
            "ok": bool(self.last_ok),
            "error": self.last_error,
            "latency_ms": round(float(self.last_latency_ms), 3),
            "score_threshold": float(self.score_thr),
        }


class SegmentationMapCache:
    def __init__(self, meta_path: Path):
        with meta_path.open() as f:
            meta = json.load(f)
        self.meta_path = meta_path
        self.meta = meta
        self.frame_count = int(meta["frame_count"])
        self.height = int(meta["height"])
        self.width = int(meta["width"])
        self.model = str(meta.get("model", ""))
        data_path = Path(meta["data_path"])
        if not data_path.is_absolute():
            data_path = meta_path.parent / data_path
        self.data_path = data_path
        self.maps = np.memmap(
            self.data_path,
            dtype=np.uint8,
            mode="r",
            shape=(self.frame_count, self.height, self.width),
        )

    def get(self, frame_index: int) -> np.ndarray:
        idx = int(np.clip(frame_index, 0, self.frame_count - 1))
        return np.asarray(self.maps[idx], dtype=np.uint8).copy()


class GroundPlaneProjector:
    def __init__(self, calib: dict):
        intr = calib["intrinsics"]
        self.model = (intr.get("model") or "equidistant_fisheye").lower()
        self.fx = float(intr.get("fx", intr.get("focal_length")))
        self.fy = float(intr.get("fy", intr.get("focal_length", self.fx)))
        self.cx = float(intr["cx"])
        self.cy = float(intr["cy"])
        self.k1 = float(intr.get("k1", 0.0))
        self.k2 = float(intr.get("k2", 0.0))
        self.height_m = float(calib["extrinsics"]["height_m"])
        pitch = math.radians(calib["extrinsics"].get("pitch_deg", 0.0))
        roll = math.radians(calib["extrinsics"].get("roll_deg", 0.0))
        yaw = math.radians(calib["extrinsics"].get("yaw_deg", 0.0))
        cp, sp = math.cos(pitch), math.sin(pitch)
        cr, sr = math.cos(roll), math.sin(roll)
        cyw, syw = math.cos(yaw), math.sin(yaw)
        ryaw = np.array([[cyw, -syw, 0], [syw, cyw, 0], [0, 0, 1]], dtype=np.float64)
        rbase = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
        rpitch = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], dtype=np.float64)
        rroll = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]], dtype=np.float64)
        self.r_cam_from_ground = rroll @ rpitch @ rbase @ ryaw
        self.r_ground_from_cam = np.linalg.inv(self.r_cam_from_ground)
        bev_range = calib.get("bev_range", {})
        self.range_fwd_m = float(bev_range.get("forward_ft", 100.0)) * 0.3048
        self.range_side_m = float(bev_range.get("side_ft", 50.0)) * 0.3048

    def _ray_cam(self, u: float, v: float) -> np.ndarray:
        dx = float(u) - self.cx
        dy = float(v) - self.cy
        if self.model in ("pinhole", "brown_conrady", "inverse_brown_conrady",
                          "modified_brown_conrady", "rectilinear"):
            ray = np.array([dx / self.fx, dy / self.fy, 1.0], dtype=np.float64)
            return ray / max(np.linalg.norm(ray), 1e-9)
        r = math.hypot(dx, dy)
        if r < 1e-9:
            return np.array([0.0, 0.0, 1.0], dtype=np.float64)
        theta = r / max(self.fx, 1e-9)
        # k1/k2 are zero for the drive-by calibration, but keep one Newton
        # correction for fisheye files that include distortion.
        for _ in range(5):
            t2 = theta * theta
            f = self.fx * theta * (1.0 + self.k1 * t2 + self.k2 * t2 * t2) - r
            df = self.fx * (1.0 + 3.0 * self.k1 * t2 + 5.0 * self.k2 * t2 * t2)
            if abs(df) < 1e-9:
                break
            theta -= f / df
        s = math.sin(theta)
        return np.array([s * dx / r, s * dy / r, math.cos(theta)], dtype=np.float64)

    def image_to_local(self, u: float, v: float) -> tuple[float, float] | None:
        ray_ground = self.r_ground_from_cam @ self._ray_cam(u, v)
        if ray_ground[2] >= -1e-6:
            return None
        scale = -self.height_m / ray_ground[2]
        point = ray_ground * scale
        lateral = float(point[0])
        forward = float(point[1])
        if forward <= 0.0 or forward > self.range_fwd_m or abs(lateral) > self.range_side_m:
            return None
        return forward, lateral

    def local_to_image(self, forward_m: float, lateral_m: float, up_m: float = 0.0) -> tuple[float, float] | None:
        point_ground = np.array(
            [float(lateral_m), float(forward_m), -self.height_m + float(up_m)],
            dtype=np.float64,
        )
        point_cam = self.r_cam_from_ground @ point_ground
        if point_cam[2] <= 1e-6:
            return None
        x = point_cam[0] / point_cam[2]
        y = point_cam[1] / point_cam[2]
        if self.model in ("pinhole", "brown_conrady", "inverse_brown_conrady",
                          "modified_brown_conrady", "rectilinear"):
            return self.fx * x + self.cx, self.fy * y + self.cy
        r = math.hypot(x, y)
        if r < 1e-9:
            return self.cx, self.cy
        theta = math.atan(r)
        t2 = theta * theta
        radius = self.fx * theta * (1.0 + self.k1 * t2 + self.k2 * t2 * t2)
        return self.cx + radius * x / r, self.cy + radius * y / r


class GpsTrace:
    """Recorded GPS trace for offline video replay.

    Loads the caddy.gps.v1 JSON and interpolates lat/lon at a given
    video time offset, writing the result to gps_state.json so the
    Flask /gps endpoint serves it.
    """

    def __init__(self, path: Path):
        with path.open() as f:
            data = json.load(f)
        self.path = path
        samples = data.get("samples", [])
        if not samples:
            self.times: np.ndarray = np.array([])
            self.lats: np.ndarray = np.array([])
            self.lons: np.ndarray = np.array([])
            self.t0 = 0.0
            return
        self.times = np.array([s["t_s"] for s in samples])
        self.lats = np.array([s["lat"] for s in samples])
        self.lons = np.array([s["lon"] for s in samples])
        self.t0 = float(self.times[0])

    def sample_at(self, video_t_s: float) -> dict | None:
        if len(self.times) == 0:
            return None
        abs_t = self.t0 + video_t_s
        lat = float(np.interp(abs_t, self.times, self.lats))
        lon = float(np.interp(abs_t, self.times, self.lons))
        return {"lat_deg": lat, "lon_deg": lon}

    def write_state(self, video_t_s: float, state_path: Path) -> None:
        pos = self.sample_at(video_t_s)
        if pos is None:
            return
        write_json_atomic(state_path, {
            "ts": time.time(),
            "connected": True,
            "host": "gps_trace",
            "fix": {
                "lat_deg": pos["lat_deg"],
                "lon_deg": pos["lon_deg"],
                "speed_mps": 0.0,
                "course_deg": 0.0,
                "h_acc_m": 1.0,
                "t_unix": time.time(),
            },
        })


class CLRNetLaneCache:
    """Cached CLRNet lane detections for offline replay.

    Loads the JSON output of precompute_clrnet_modal.py and returns
    per-frame lane lists in the same format as CLRerNetRunner.infer().
    """

    def __init__(self, path: Path):
        with path.open() as f:
            data = json.load(f)
        self.path = path
        self.frame_count = int(data.get("frame_count") or 0)
        self.fps = float(data.get("fps") or 30.0)
        self.model = str(data.get("model") or "clrnet")
        self._frames = data.get("frames", [])
        self._index: dict[int, list] = {}
        for entry in self._frames:
            fi = int(entry.get("frame_index", -1))
            if fi >= 0:
                self._index[fi] = entry.get("lanes", [])

    def lanes_for_frame(self, frame_index: int) -> list[dict]:
        idx = int(np.clip(frame_index, 0, max(0, self.frame_count - 1)))
        raw = self._index.get(idx, [])
        out = []
        for lane in raw:
            pts = lane.get("points")
            if pts is None or len(pts) < 2:
                continue
            out.append({
                "points": np.array(pts, dtype=np.float32),
                "score": float(lane.get("score", 0.0)),
            })
        return out


class YoloDetectionCache:
    def __init__(self, path: Path):
        with path.open() as f:
            data = json.load(f)
        self.path = path
        self.data = data
        self.frames = data.get("frames", [])
        self.frame_count = int(data.get("frame_count") or len(self.frames))
        self.fps = float(data.get("fps") or 30.0)
        self.model = str(data.get("model") or "")
        self.conf = float(data.get("conf") or 0.0)

    def raw_detections(self, frame_index: int) -> list[dict]:
        if not self.frames:
            return []
        idx = int(np.clip(frame_index, 0, len(self.frames) - 1))
        return list(self.frames[idx].get("detections", []))

    def detection_local(
        self,
        det: dict,
        projector: GroundPlaneProjector,
    ) -> tuple[float, float, tuple[float, float]] | None:
        xyxy = det.get("xyxy") or []
        if len(xyxy) != 4:
            return None
        x1, _, x2, y2 = [float(v) for v in xyxy]
        foot = ((x1 + x2) * 0.5, y2)
        local = projector.image_to_local(*foot)
        if local is None:
            return None
        return float(local[0]), float(local[1]), foot

    def track_history(
        self,
        frame_index: int,
        track_id: int | None,
        projector: GroundPlaneProjector,
        history_s: float = 3.2,
    ) -> list[list[float]]:
        if track_id is None:
            return []
        idx = int(np.clip(frame_index, 0, max(0, len(self.frames) - 1)))
        start = max(0, idx - int(math.ceil(float(history_s) * self.fps)))
        hist = []
        for prev_idx in range(start, idx + 1):
            match = None
            for prev in self.raw_detections(prev_idx):
                if prev.get("track_id") == track_id:
                    match = prev
                    break
            if match is None:
                continue
            projected = self.detection_local(match, projector)
            if projected is None:
                continue
            fwd, lat, _ = projected
            t_s = (prev_idx - idx) / max(self.fps, 1e-6)
            hist.append([round(float(t_s), 3), round(float(fwd), 3), round(float(lat), 3)])
        return hist

    @staticmethod
    def constant_velocity_future(
        fwd: float,
        lat: float,
        vx: float,
        vy: float,
        horizon_s: float,
        step_s: float,
    ) -> list[list[float]]:
        future = []
        t = step_s
        while t <= horizon_s + 1e-6:
            future.append([round(fwd + vx * t, 3), round(lat + vy * t, 3)])
            t += step_s
        return future

    def objects_for_frame(
        self,
        frame_index: int,
        projector: GroundPlaneProjector,
        horizon_s: float = 4.0,
        step_s: float = 0.5,
        lookback_frames: int = 20,
        history_s: float = 2.0,
    ) -> list[dict]:
        idx = int(np.clip(frame_index, 0, max(0, len(self.frames) - 1)))
        objects = []
        for det in self.raw_detections(idx):
            xyxy = det.get("xyxy") or []
            if len(xyxy) != 4:
                continue
            x1, y1, x2, y2 = [float(v) for v in xyxy]
            projected = self.detection_local(det, projector)
            if projected is None:
                continue
            fwd, lat, foot = projected
            track_id = det.get("track_id")
            vx = 0.0
            vy = 0.0
            if track_id is not None:
                for prev_idx in range(idx - 1, max(-1, idx - lookback_frames - 1), -1):
                    prev_match = None
                    for prev in self.raw_detections(prev_idx):
                        if prev.get("track_id") == track_id:
                            prev_match = prev
                            break
                    if prev_match is None:
                        continue
                    pxy = prev_match.get("xyxy") or []
                    if len(pxy) != 4:
                        continue
                    px1, py1, px2, py2 = [float(v) for v in pxy]
                    prev_local = projector.image_to_local((px1 + px2) * 0.5, py2)
                    if prev_local is None:
                        continue
                    dt = (idx - prev_idx) / max(self.fps, 1e-6)
                    if dt > 1e-6:
                        vx = (fwd - prev_local[0]) / dt
                        vy = (lat - prev_local[1]) / dt
                    break
            speed = float(math.hypot(vx, vy))
            future = self.constant_velocity_future(fwd, lat, vx, vy, horizon_s, step_s)
            objects.append({
                "track_id": int(track_id) if track_id is not None else None,
                "class_id": int(det.get("class_id", -1)),
                "class_name": str(det.get("class_name", "object")),
                "confidence": float(det.get("confidence", 0.0)),
                "xyxy": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                "footpoint": [round(foot[0], 2), round(foot[1], 2)],
                "x_m": round(float(fwd), 3),
                "y_m": round(float(lat), 3),
                "distance_m": round(float(math.hypot(fwd, lat)), 3),
                "vx_mps": round(float(vx), 3),
                "vy_mps": round(float(vy), 3),
                "speed_mps": round(speed, 3),
                "future_m": future,
                "future_modes": [{"prob": 1.0, "future_m": future, "source": "constant_velocity"}],
                "future_source": "constant_velocity",
                "history_m": self.track_history(idx, int(track_id) if track_id is not None else None, projector, history_s),
            })
        objects.sort(key=lambda o: o["distance_m"])
        return objects


class Mono3DDetectionCache:
    def __init__(self, path: Path):
        with path.open() as f:
            data = json.load(f)
        self.path = path
        self.data = data
        self.frame_count = int(data.get("frame_count") or 0)
        self.fps = float(data.get("fps") or 30.0)
        self.provider = str(data.get("provider") or data.get("model") or "mmdet3d_fcos3d_nuscenes")
        self.score_threshold = float(data.get("score_threshold") or 0.0)
        viz_dir_raw = str(data.get("viz_dir") or "")
        self.viz_dir = (path.parent / viz_dir_raw) if viz_dir_raw else path.with_suffix(".viz")
        self._index: dict[int, dict] = {}
        for entry in data.get("frames", []):
            try:
                frame_index = int(entry.get("frame_index", -1))
            except (TypeError, ValueError):
                continue
            if frame_index >= 0:
                self._index[frame_index] = entry

    def entry_for_frame(self, frame_index: int) -> dict | None:
        if not self._index:
            return None
        idx = int(np.clip(frame_index, 0, max(0, self.frame_count - 1)))
        return self._index.get(idx)

    def objects_for_frame(self, frame_index: int) -> list[dict]:
        entry = self.entry_for_frame(frame_index)
        raw = entry.get("objects", []) if isinstance(entry, dict) else []
        objects = []
        for obj in raw:
            if not isinstance(obj, dict):
                continue
            clean = dict(obj)
            clean["provider"] = self.provider
            objects.append(clean)
        return objects

    def viz_for_frame(self, frame_index: int) -> np.ndarray | None:
        entry = self.entry_for_frame(frame_index)
        if not isinstance(entry, dict):
            return None
        rel = entry.get("viz")
        if not rel:
            return None
        path = self.viz_dir / str(rel)
        if not path.exists():
            return None
        return cv2.imread(str(path), cv2.IMREAD_COLOR)

    def status(self) -> dict:
        return {
            "type": "cache_mmdet3d_fcos3d",
            "provider": self.provider,
            "ok": bool(self._index),
            "error": "",
            "latency_ms": 0.0,
            "score_threshold": self.score_threshold,
            "cache_file": str(self.path),
            "viz_dir": str(self.viz_dir),
        }


class ObjectTrajectoryPredictorClient:
    def __init__(self, url: str | None, timeout_s: float = 0.25):
        self.url = (url or "").strip()
        self.timeout_s = float(timeout_s)
        self.last_ok = False
        self.last_error = ""
        self.last_latency_ms = 0.0
        self.provider = "constant_velocity"

    @property
    def enabled(self) -> bool:
        return bool(self.url)

    def _request_payload(
        self,
        frame_index: int,
        fps: float,
        objects: list[dict],
        ego_speed_mps: float,
        ego_path_m: list[list[float]],
        horizon_s: float,
        step_s: float,
    ) -> dict:
        agents = []
        for obj in objects:
            agents.append({
                "track_id": obj.get("track_id"),
                "class_id": obj.get("class_id"),
                "class_name": obj.get("class_name"),
                "confidence": obj.get("confidence"),
                "state": {
                    "x_m": obj.get("x_m"),
                    "y_m": obj.get("y_m"),
                    "vx_mps": obj.get("vx_mps"),
                    "vy_mps": obj.get("vy_mps"),
                    "speed_mps": obj.get("speed_mps"),
                },
                "history_m": obj.get("history_m", []),
            })
        return {
            "schema": "caddy.object_prediction.v1",
            "frame_index": int(frame_index),
            "fps": float(fps),
            "horizon_s": float(horizon_s),
            "step_s": float(step_s),
            "ego": {
                "speed_mps": float(ego_speed_mps),
                "planned_path_m": ego_path_m,
            },
            "agents": agents,
        }

    @staticmethod
    def _clean_future(points: object) -> list[list[float]]:
        if not isinstance(points, list):
            return []
        out = []
        for pt in points:
            if not isinstance(pt, (list, tuple)) or len(pt) < 2:
                continue
            try:
                x = float(pt[0])
                y = float(pt[1])
            except (TypeError, ValueError):
                continue
            if math.isfinite(x) and math.isfinite(y):
                out.append([round(x, 3), round(y, 3)])
        return out

    def apply(
        self,
        frame_index: int,
        fps: float,
        objects: list[dict],
        ego_speed_mps: float,
        ego_path_m: list[list[float]],
        horizon_s: float,
        step_s: float,
    ) -> list[dict]:
        if not self.enabled or not objects:
            self.last_ok = False
            self.last_error = "" if not self.enabled else "no_objects"
            self.provider = "constant_velocity"
            return objects
        payload = self._request_payload(
            frame_index, fps, objects, ego_speed_mps, ego_path_m, horizon_s, step_s
        )
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                body = resp.read(8 * 1024 * 1024)
            result = json.loads(body.decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError, ValueError) as exc:
            self.last_ok = False
            self.last_error = f"{type(exc).__name__}: {exc}"
            self.last_latency_ms = (time.perf_counter() - t0) * 1000.0
            self.provider = "constant_velocity"
            return objects
        self.last_latency_ms = (time.perf_counter() - t0) * 1000.0
        by_id: dict[int, dict] = {}
        for item in result.get("agents", []):
            if not isinstance(item, dict):
                continue
            tid = item.get("track_id")
            if tid is None:
                continue
            try:
                by_id[int(tid)] = item
            except (TypeError, ValueError):
                continue
        used = 0
        provider = str(result.get("provider") or result.get("model") or "external")
        for obj in objects:
            tid = obj.get("track_id")
            if tid is None:
                continue
            pred = by_id.get(int(tid))
            if pred is None:
                continue
            modes = []
            raw_modes = pred.get("modes")
            if isinstance(raw_modes, list):
                for mode in raw_modes:
                    if not isinstance(mode, dict):
                        continue
                    future = self._clean_future(mode.get("future_m"))
                    if not future:
                        continue
                    try:
                        prob = float(mode.get("prob", 1.0))
                    except (TypeError, ValueError):
                        prob = 1.0
                    modes.append({
                        "prob": float(np.clip(prob, 0.0, 1.0)),
                        "future_m": future,
                        "source": provider,
                    })
            else:
                future = self._clean_future(pred.get("future_m"))
                if future:
                    modes.append({"prob": 1.0, "future_m": future, "source": provider})
            if not modes:
                continue
            modes.sort(key=lambda m: float(m.get("prob", 0.0)), reverse=True)
            obj["future_modes"] = modes
            obj["future_m"] = modes[0]["future_m"]
            obj["future_source"] = provider
            used += 1
        self.last_ok = used > 0
        self.last_error = "" if self.last_ok else "response_had_no_matching_tracks"
        self.provider = provider if self.last_ok else "constant_velocity"
        return objects

    def status(self) -> dict:
        return {
            "type": "http_json" if self.enabled else "constant_velocity",
            "url": self.url or None,
            "provider": self.provider,
            "ok": bool(self.last_ok),
            "error": self.last_error,
            "latency_ms": round(float(self.last_latency_ms), 3),
        }


class ControlTrace:
    def __init__(self, path: Path):
        self.path = path
        self.times: list[float] = []
        self.samples: list[dict] = []
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, dict) or "rel_t" not in row:
                    continue
                try:
                    rel_t = float(row["rel_t"])
                except (TypeError, ValueError):
                    continue
                self.times.append(rel_t)
                self.samples.append(row)
        order = sorted(range(len(self.times)), key=self.times.__getitem__)
        self.times = [self.times[i] for i in order]
        self.samples = [self.samples[i] for i in order]

    def sample_at(self, rel_t: float | None) -> dict | None:
        if rel_t is None or not self.samples:
            return None
        idx = bisect.bisect_right(self.times, max(0.0, float(rel_t))) - 1
        idx = max(0, min(idx, len(self.samples) - 1))
        row = self.samples[idx]

        def f(key: str, default: float = 0.0) -> float:
            try:
                return float(row.get(key, default))
            except (TypeError, ValueError):
                return default

        steer = f("column_deg_actual", f("steer_deg", 0.0))
        gas_frac = f("gas_frac", 0.0)
        brake_frac = f("brake_frac", 0.0)
        if "gas_frac" not in row:
            gas_frac = f("gas", 0.0) / max(f("gas_cap", 1.0), 1e-6)
        if "brake_frac" not in row:
            brake_frac = f("brake", 0.0) / max(f("brake_max", 1.0), 1e-6)
        return {
            "source": str(self.path),
            "rel_t": round(f("rel_t"), 3),
            "steer_deg": round(steer, 3),
            "gas": round(f("gas", 0.0), 4),
            "brake": round(f("brake", 0.0), 4),
            "gas_frac": round(float(np.clip(gas_frac, 0.0, 1.0)), 4),
            "brake_frac": round(float(np.clip(brake_frac, 0.0, 1.0)), 4),
            "mph": round(f("mph", 0.0), 3),
            "autosteer": bool(row.get("autosteer", False)),
            "mode": row.get("mode"),
        }


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


class AutoSpeedController:
    """Unified path-aware obstacle speed controller.

    Replaces SpeedController + evaluate_yolo_collision + PedalCommandSmoother.
    Computes a commanded_speed (m/s) from path geometry and obstacle predictions
    with jerk-limited smoothing and emergency override.
    """

    def __init__(self) -> None:
        self.previous_accel = 0.0
        self.last_gas = 0.0
        self.speed_limits: list[dict] = []
        self.emergency_active = False
        self.desired_accel = 0.0

    def reset(self) -> None:
        self.previous_accel = 0.0
        self.last_gas = 0.0
        self.speed_limits = []
        self.emergency_active = False
        self.desired_accel = 0.0

    @staticmethod
    def _find_closest_point_on_path(
        path: np.ndarray, point: tuple[float, float]
    ) -> tuple[tuple[float, float], float]:
        """Find the closest point on a polyline path to a given point.

        Returns (closest_point, path_distance) where path_distance is how far
        along the path the closest point is (i.e. how far ahead of the cart).
        """
        px, py = float(point[0]), float(point[1])
        min_dist_sq = float("inf")
        best_point = (px, py)
        best_path_distance = 0.0
        accumulated = 0.0

        for i in range(len(path) - 1):
            ax, ay = float(path[i, 0]), float(path[i, 1])
            bx, by = float(path[i + 1, 0]), float(path[i + 1, 1])
            dx, dy = bx - ax, by - ay
            seg_len_sq = dx * dx + dy * dy
            if seg_len_sq < 1e-12:
                accumulated += 0.0
                continue
            t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / seg_len_sq))
            proj_x = ax + t * dx
            proj_y = ay + t * dy
            d_sq = (px - proj_x) ** 2 + (py - proj_y) ** 2
            if d_sq < min_dist_sq:
                min_dist_sq = d_sq
                best_point = (proj_x, proj_y)
                seg_len = math.sqrt(seg_len_sq)
                best_path_distance = accumulated + t * seg_len
            accumulated += math.sqrt(seg_len_sq)

        return best_point, best_path_distance

    @staticmethod
    def _get_path_direction_at(path: np.ndarray, point: tuple[float, float]) -> tuple[float, float]:
        """Unit vector tangent to the path at the given point."""
        px, py = float(point[0]), float(point[1])
        best_i = 0
        min_dist_sq = float("inf")
        for i in range(len(path) - 1):
            ax, ay = float(path[i, 0]), float(path[i, 1])
            bx, by = float(path[i + 1, 0]), float(path[i + 1, 1])
            mx, my = (ax + bx) * 0.5, (ay + by) * 0.5
            d_sq = (px - mx) ** 2 + (py - my) ** 2
            if d_sq < min_dist_sq:
                min_dist_sq = d_sq
                best_i = i
        ax, ay = float(path[best_i, 0]), float(path[best_i, 1])
        bx, by = float(path[best_i + 1, 0]), float(path[best_i + 1, 1])
        dx, dy = bx - ax, by - ay
        length = math.hypot(dx, dy)
        if length < 1e-9:
            return (1.0, 0.0)
        return (dx / length, dy / length)

    def compute(
        self,
        path: np.ndarray | None,
        obstacles: list[dict],
        current_speed: float,
        max_speed: float,
    ) -> float:
        """Compute commanded speed (m/s) from path, obstacles, and current state."""
        dt = AUTOSPEED_DT
        self.speed_limits = []
        self.emergency_active = False

        if path is None or len(path) < 2:
            path_ok = False
        else:
            path_ok = True

        speed_limits_raw: list[float] = []
        speed_limit_details: list[dict] = []

        for obj in obstacles:
            x_m = float(obj.get("x_m", 0.0))
            y_m = float(obj.get("y_m", 0.0))
            vx = float(obj.get("vx_mps", 0.0))
            vy = float(obj.get("vy_mps", 0.0))
            cls = str(obj.get("class_name", "unknown"))
            conf = float(obj.get("confidence", 0.0))
            if conf < 0.20:
                continue
            if x_m <= 0.0:
                continue

            if not path_ok:
                # No path — use simple forward distance
                lateral_offset = abs(y_m)
                conflict_path_distance = x_m
            else:
                closest_pt, path_distance = self._find_closest_point_on_path(
                    path, (x_m, y_m)
                )
                lateral_offset = math.hypot(
                    x_m - closest_pt[0], y_m - closest_pt[1]
                )

                min_future_lateral = lateral_offset
                conflict_path_distance = path_distance

                for t_step in (0.5, 1.0, 1.5, 2.0):
                    fx = x_m + vx * t_step
                    fy = y_m + vy * t_step
                    f_closest, f_path_dist = self._find_closest_point_on_path(
                        path, (fx, fy)
                    )
                    f_lateral = math.hypot(
                        fx - f_closest[0], fy - f_closest[1]
                    )
                    if f_lateral < min_future_lateral:
                        min_future_lateral = f_lateral
                        conflict_path_distance = f_path_dist

                lateral_offset = min_future_lateral

            if lateral_offset > AUTOSPEED_PATH_WIDTH:
                continue
            if conflict_path_distance < 0:
                continue

            lateral_factor = 1.0 - (lateral_offset / AUTOSPEED_PATH_WIDTH)
            lateral_factor = max(0.0, min(1.0, lateral_factor))

            if path_ok and len(path) >= 2:
                closest_pt_for_dir, _ = self._find_closest_point_on_path(
                    path, (x_m, y_m)
                )
                path_dir = self._get_path_direction_at(path, closest_pt_for_dir)
                obstacle_speed_along_path = vx * path_dir[0] + vy * path_dir[1]
            else:
                obstacle_speed_along_path = vx

            closing_speed = current_speed - obstacle_speed_along_path

            stopping_distance_available = max(
                0.0, conflict_path_distance - AUTOSPEED_MIN_GAP
            )

            is_vehicle = cls in ("car", "truck", "bus", "train")

            if closing_speed <= 0 and lateral_factor < 0.8:
                safe_speed = max_speed
            elif is_vehicle and closing_speed <= 0:
                safe_speed = max_speed
            else:
                physics_limit = math.sqrt(
                    max(0.0, 2.0 * AUTOSPEED_COMFORT_DECEL * stopping_distance_available)
                )
                if is_vehicle and obstacle_speed_along_path > 0:
                    safe_speed = obstacle_speed_along_path + physics_limit * 0.5
                else:
                    safe_speed = physics_limit
                safe_speed = safe_speed + (max_speed - safe_speed) * (1.0 - lateral_factor)

            speed_limits_raw.append(safe_speed)
            speed_limit_details.append({
                "speed_mps": round(float(safe_speed), 3),
                "speed_mph": round(float(safe_speed) * 2.23694, 1),
                "obstacle_class": cls,
                "distance_m": round(float(conflict_path_distance), 2),
                "lateral_offset_m": round(float(lateral_offset), 2),
                "lateral_factor": round(float(lateral_factor), 3),
                "closing_speed_mps": round(float(closing_speed), 2),
                "track_id": obj.get("track_id"),
            })

        if not speed_limits_raw:
            desired_speed = max_speed
        else:
            desired_speed = min(speed_limits_raw)
        desired_speed = max(0.0, min(max_speed, desired_speed))

        # Jerk-limited smoothing
        desired_accel = (desired_speed - current_speed) / max(dt, 1e-6)
        if desired_accel > 0:
            desired_accel = min(desired_accel, AUTOSPEED_COMFORT_DECEL * 0.8)
        else:
            desired_accel = max(desired_accel, -AUTOSPEED_COMFORT_DECEL)

        accel_change = desired_accel - self.previous_accel
        max_accel_change = AUTOSPEED_MAX_JERK * dt
        if abs(accel_change) > max_accel_change:
            desired_accel = self.previous_accel + math.copysign(
                max_accel_change, accel_change
            )

        commanded_speed = current_speed + desired_accel * dt
        commanded_speed = max(0.0, min(max_speed, commanded_speed))
        self.previous_accel = desired_accel
        self.desired_accel = desired_accel

        # Emergency override
        for obj in obstacles:
            x_m = float(obj.get("x_m", 0.0))
            y_m = float(obj.get("y_m", 0.0))
            conf = float(obj.get("confidence", 0.0))
            if conf < 0.20 or x_m <= 0.0:
                continue
            if x_m < AUTOSPEED_MIN_GAP and abs(y_m) < AUTOSPEED_PATH_WIDTH:
                commanded_speed = 0.0
                self.previous_accel = -AUTOSPEED_EMERGENCY_DECEL
                self.desired_accel = -AUTOSPEED_EMERGENCY_DECEL
                self.emergency_active = True
                break

        self.speed_limits = speed_limit_details
        return commanded_speed

    def gas_brake_from_speed(
        self,
        commanded_speed_mps: float,
        current_speed_mps: float,
        max_speed_mps: float,
    ) -> tuple[float, float]:
        """Convert commanded speed to gas/brake pot fractions."""
        commanded_mph = commanded_speed_mps * 2.23694
        current_mph = current_speed_mps * 2.23694

        if self.emergency_active or commanded_speed_mps < 0.01:
            self.last_gas = 0.0
            brake = min(1.0, BRAKE_KP * max(current_mph, 1.0)) if self.emergency_active else 0.0
            return 0.0, float(np.clip(brake, 0.0, BRAKE_MAX))

        gas = constant_gas_for_mph(commanded_mph)
        # Slew-rate limit gas changes
        dt = AUTOSPEED_DT
        rise = GAS_RISE_RATE_PER_S * dt
        fall = GAS_FALL_RATE_PER_S * dt
        gas = float(np.clip(gas, self.last_gas - fall, self.last_gas + rise))
        gas = float(np.clip(gas, 0.0, 1.0))
        self.last_gas = gas

        brake = 0.0
        overshoot = current_mph - commanded_mph - BRAKE_DEADBAND_MPH
        if gas <= 1e-3 and overshoot > 0.0:
            brake = float(np.clip(BRAKE_KP * overshoot, 0.0, BRAKE_MAX))

        return gas, brake

    def status(self) -> dict:
        """Status dict for the state JSON."""
        most_limiting = None
        if self.speed_limits:
            most_limiting = min(self.speed_limits, key=lambda s: s["speed_mps"])
        return {
            "speed_limits": self.speed_limits[:8],
            "desired_accel": round(float(self.desired_accel), 3),
            "emergency_active": bool(self.emergency_active),
            "corridor_width_m": float(AUTOSPEED_PATH_WIDTH),
            "most_limiting": most_limiting,
        }


class StopSignController:
    """State machine for stop sign detection and smooth stop/start behavior.

    Uses YOLO 'stop sign' detections projected to BEV local coordinates to
    estimate distance.  When a sign is within APPROACH_M, the controller
    decelerates to a smooth stop at STOP_BUFFER_M before the sign, waits
    WAIT_S, then ramps back to cruise over DEPART_RAMP_S.
    """

    CLEAR = 0
    APPROACHING = 1
    STOPPED = 2
    DEPARTING = 3

    def __init__(self) -> None:
        self.state = self.CLEAR
        self.stop_target_m = 0.0
        self.stopped_at: float | None = None
        self.depart_start: float | None = None
        self.frames_without_sign = 0
        self.last_sign: dict | None = None

    def update(
        self,
        yolo_objects: list[dict],
        current_speed_mps: float,
        max_speed_mps: float,
        raw_stop_signs: list[dict] | None = None,
    ) -> float:
        """Return speed limit (m/s) based on current stop-sign state.

        raw_stop_signs: YOLO detections with class 'stop sign' that may not
        have BEV projection (x_m).  Distance is estimated from bbox height
        using a calibrated pinhole model.
        """
        signs = []
        # First try BEV-projected stop signs from yolo_objects
        for obj in yolo_objects:
            if obj.get("class_name") != "stop sign":
                continue
            if float(obj.get("confidence", 0)) < STOP_SIGN_MIN_CONF:
                continue
            x_m = float(obj.get("x_m", 0))
            if x_m > 0:
                signs.append({"x_m": x_m, "y_m": float(obj.get("y_m", 0)),
                              "confidence": float(obj.get("confidence", 0)),
                              "source": "bev"})
        # Fall back to raw detections if no BEV-projected signs
        for det in (raw_stop_signs or []):
            if float(det.get("confidence", 0)) < STOP_SIGN_MIN_CONF:
                continue
            xyxy = det.get("xyxy", [])
            if len(xyxy) != 4:
                continue
            bbox_h = xyxy[3] - xyxy[1]
            bbox_w = xyxy[2] - xyxy[0]
            area = bbox_h * bbox_w
            if area < STOP_SIGN_MIN_BBOX_AREA:
                continue
            # Estimate distance from bbox height: real sign ~0.75m tall,
            # focal length ~320px for 640-wide image.  d = (real_h * fy) / bbox_h
            est_dist = (0.75 * 320.0) / max(bbox_h, 1.0)
            cx = (xyxy[0] + xyxy[2]) * 0.5
            # Lateral from image center (assume 320px center)
            est_lat = (cx - 320.0) / 320.0 * est_dist * 0.5
            signs.append({"x_m": est_dist, "y_m": est_lat,
                          "confidence": float(det.get("confidence", 0)),
                          "source": "bbox"})

        signs.sort(key=lambda o: o.get("x_m", 999))
        nearest = signs[0] if signs else None

        if nearest:
            self.frames_without_sign = 0
            self.last_sign = nearest
        else:
            self.frames_without_sign += 1

        now = time.monotonic()

        if self.state == self.CLEAR:
            if nearest and nearest["x_m"] < STOP_SIGN_APPROACH_M:
                self.state = self.APPROACHING
                self.stop_target_m = max(0.0, nearest["x_m"] - STOP_SIGN_STOP_BUFFER_M)
            return max_speed_mps

        if self.state == self.APPROACHING:
            if self.frames_without_sign > STOP_SIGN_LOST_FRAMES:
                self.state = self.CLEAR
                self.last_sign = None
                return max_speed_mps
            dist = nearest["x_m"] if nearest else (
                self.last_sign["x_m"] if self.last_sign else 10.0
            )
            self.stop_target_m = max(0.0, dist - STOP_SIGN_STOP_BUFFER_M)
            # Transition to STOPPED when remaining distance is small.
            # The bbox distance estimate bottoms out at ~4m, so with the
            # buffer subtracted we may only reach ~0-1m target.  Use a
            # generous threshold to ensure we actually stop.
            if self.stop_target_m < 2.5:
                self.state = self.STOPPED
                self.stopped_at = now
                return 0.0
            # Smooth kinematic speed ramp: v = sqrt(2 * a * d)
            gentle_decel = min(AUTOSPEED_COMFORT_DECEL, 0.8)
            safe = math.sqrt(
                max(0.0, 2.0 * gentle_decel * self.stop_target_m)
            )
            linear_ramp = max_speed_mps * min(1.0, self.stop_target_m / 8.0)
            speed_limit = min(safe, linear_ramp, max_speed_mps)
            return speed_limit

        if self.state == self.STOPPED:
            elapsed = now - self.stopped_at if self.stopped_at else 0.0
            if elapsed >= STOP_SIGN_WAIT_S:
                self.state = self.DEPARTING
                self.depart_start = now
            return 0.0

        if self.state == self.DEPARTING:
            elapsed = now - self.depart_start if self.depart_start else 0.0
            frac = min(1.0, elapsed / STOP_SIGN_DEPART_RAMP_S)
            if frac >= 1.0:
                self.state = self.CLEAR
                self.last_sign = None
            return max_speed_mps * frac

        return max_speed_mps

    def status(self) -> dict:
        state_names = {0: "clear", 1: "approaching", 2: "stopped", 3: "departing"}
        sign_info = None
        if self.last_sign:
            sign_info = {
                "x_m": round(float(self.last_sign.get("x_m", 0)), 2),
                "y_m": round(float(self.last_sign.get("y_m", 0)), 2),
                "confidence": round(float(self.last_sign.get("confidence", 0)), 3),
            }
        wait_remaining = None
        if self.state == self.STOPPED and self.stopped_at:
            wait_remaining = round(
                max(0.0, STOP_SIGN_WAIT_S - (time.monotonic() - self.stopped_at)), 1
            )
        return {
            "state": state_names.get(self.state, "unknown"),
            "stop_target_m": round(self.stop_target_m, 2),
            "sign": sign_info,
            "wait_remaining_s": wait_remaining,
            "min_confidence": STOP_SIGN_MIN_CONF,
            "curr_confidence": round(float(self.last_sign.get("confidence", 0)), 3) if self.last_sign else None,
        }


def bev_class_map_cached(seg_map: np.ndarray, remap: seg_fast.BevRemap) -> np.ndarray:
    """Project segmentation labels into BEV using the cached homography."""
    cls_ids = seg_map[remap.map_v, remap.map_u]
    out = np.full((remap.bev_size, remap.bev_size), 255, dtype=np.uint8)
    np.copyto(out, np.clip(cls_ids, 0, 254).astype(np.uint8), where=remap.valid)
    return out


def build_environment_threat_from_autospeed(
    autospeed: AutoSpeedController,
    commanded_speed_mps: float,
    num_objects: int,
    enabled: bool,
) -> dict:
    """Build protective-stop state from the autospeed controller for UI compatibility."""
    active = bool(enabled and autospeed.emergency_active)
    most_limiting = autospeed.status().get("most_limiting")
    threat = None
    reason = ""
    if active and most_limiting:
        reason = f"emergency stop — {most_limiting['obstacle_class']} at {most_limiting['distance_m']:.1f}m"
        threat = {
            "label": most_limiting["obstacle_class"],
            "track_id": most_limiting.get("track_id"),
            "x_m": most_limiting["distance_m"],
        }
    elif most_limiting and commanded_speed_mps < 0.5:
        active = True
        reason = f"{most_limiting['obstacle_class']} blocking path at {most_limiting['distance_m']:.1f}m"
        threat = {
            "label": most_limiting["obstacle_class"],
            "track_id": most_limiting.get("track_id"),
            "x_m": most_limiting["distance_m"],
        }
    return {
        "enabled": bool(enabled),
        "active": active,
        "source": "autospeed" if enabled else None,
        "reason": reason,
        "brake_target": 1.0 if active else 0.0,
        "ttc_s": None,
        "image_coverage": 0.0,
        "objects": int(num_objects),
        "threat": threat,
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
        reader = VideoReader(args.video, loop=not args.no_loop,
                             control_file=args.video_control_file)
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
    # Keep this branch self-contained: the external drive-by-segmentation repo
    # supplies only camera_calibration.json homography values, never runtime code.
    seg_live = SimpleNamespace(load_segformer=load_segformer)
    return (
        seg_live,
        SegRuntime(),
        seg_fast.lane_aware_centerline_path_fast,
        create_bev,
        create_overlay,
        CITYSCAPES_COLORS,
    )


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


def gps_route_bearing_rad(args) -> float | None:
    """Return the GPS route target bearing in BEV coords (0 = forward, + = right).

    Returns None if GPS or route data is unavailable / stale.
    """
    route = read_json_if_fresh(args.route_file, None)
    if route is None or not route.get("active"):
        return None
    gps = read_json_if_fresh(args.gps_state_file, GPS_ROUTE_FRESH_S)
    if gps is None or not gps.get("connected", False):
        return None
    fix = gps.get("fix")
    if not isinstance(fix, dict):
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
    if speed_mps < GPS_ROUTE_MIN_SPEED_MPS:
        return None
    route_ll = route.get("geometry")
    if not isinstance(route_ll, list) or len(route_ll) < 2:
        return None
    target = route_target_enu(route_ll, lat, lon, args.gps_route_lookahead_m)
    if target is None:
        return None
    east, north, remaining_m = target
    if remaining_m < GPS_ROUTE_DONE_M:
        return None
    course = math.radians(course_deg)
    x_fwd = east * math.sin(course) + north * math.cos(course)
    y_right = east * math.cos(course) - north * math.sin(course)
    if x_fwd < 0.5:
        return None
    return math.atan2(y_right, x_fwd)


def clrnet_device_from_seg_device(device: str) -> str:
    if device == "cuda":
        return "cuda:0"
    if device == "modal":
        return "cpu"
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
                 stop_active: bool = False) -> np.ndarray:
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


def draw_object_viz(
    bev_rgb: np.ndarray,
    bev_geom: "seg_occupancy.BevGeometry | None",
    collision: dict | None,
    yolo_objects: list[dict] | None = None,
    autospeed_status: dict | None = None,
    stop_sign_status: dict | None = None,
    bev_cls_map: np.ndarray | None = None,
    clrnet_lanes: list[dict] | None = None,
    ground_projector: "GroundPlaneProjector | None" = None,
    clrnet_conf_threshold: float = CLRNET_CONF_THRESHOLD,
) -> np.ndarray:
    h, w = bev_rgb.shape[:2]
    out = np.full((h, w, 3), 255, dtype=np.uint8)
    if bev_cls_map is not None and bev_cls_map.shape[:2] == (h, w):
        road_mask = bev_cls_map == 0
    else:
        road_color = np.array(CITYSCAPES_COLORS[0], dtype=np.uint8)
        road_mask = np.all(bev_rgb == road_color, axis=-1)
    if road_mask.any():
        out[road_mask] = (205, 205, 205)

    h, w = out.shape[:2]
    if bev_geom is None:
        return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

    def lane_side(lane: dict) -> str:
        pts = lane.get("points")
        if pts is None or len(pts) < 2:
            return "right"
        arr = np.asarray(pts, dtype=np.float32)
        bottom = arr[arr[:, 1] >= 0.5, 0] if np.count_nonzero(arr[:, 1] >= 0.5) >= 2 else arr[:, 0]
        return "left" if float(np.mean(bottom)) < 0.5 else "right"

    def draw_clrnet_lanes() -> None:
        if ground_projector is None or not clrnet_lanes:
            return
        lane_colors = {
            "left": (38, 135, 210),
            "right": (45, 172, 155),
        }
        frame_w = float(CAM_W)
        frame_h = float(CAM_H)
        for lane in sorted(clrnet_lanes, key=lambda l: float(l.get("score", 0.0))):
            try:
                score = float(lane.get("score", 0.0))
            except (TypeError, ValueError):
                continue
            if score < float(clrnet_conf_threshold):
                continue
            pts_norm = np.asarray(lane.get("points"), dtype=np.float32)
            if pts_norm.ndim != 2 or pts_norm.shape[0] < 2 or pts_norm.shape[1] < 2:
                continue
            side = lane_side(lane)
            line_color = lane_colors.get(side, (38, 135, 210))
            segments: list[list[tuple[int, int]]] = []
            current: list[tuple[int, int]] = []
            for x_norm, y_norm in pts_norm:
                u = float(x_norm) * frame_w
                v = float(y_norm) * frame_h
                local = ground_projector.image_to_local(u, v)
                if local is None:
                    if len(current) >= 2:
                        segments.append(current)
                    current = []
                    continue
                bx, by = bev_geom.local_to_bev(local[0], local[1])
                bxi, byi = int(round(float(bx))), int(round(float(by)))
                if 0 <= bxi < w and 0 <= byi < h:
                    current.append((bxi, byi))
                elif len(current) >= 2:
                    segments.append(current)
                    current = []
            if len(current) >= 2:
                segments.append(current)
            for seg in segments:
                arr = np.array(seg, dtype=np.int32)
                cv2.polylines(out, [arr], False, line_color, 3, cv2.LINE_AA)

    draw_clrnet_lanes()

    def dashed_polyline(img: np.ndarray, pts: list[tuple[int, int]], color: tuple[int, int, int],
                        thickness: int = 2, dash_px: float = 9.0, gap_px: float = 7.0) -> None:
        if len(pts) < 2:
            return
        for p0, p1 in zip(pts[:-1], pts[1:]):
            x0, y0 = p0
            x1, y1 = p1
            dx = float(x1 - x0)
            dy = float(y1 - y0)
            dist = float(math.hypot(dx, dy))
            if dist <= 1e-3:
                continue
            ux = dx / dist
            uy = dy / dist
            pos = 0.0
            while pos < dist:
                end = min(pos + dash_px, dist)
                a = (int(round(x0 + ux * pos)), int(round(y0 + uy * pos)))
                b = (int(round(x0 + ux * end)), int(round(y0 + uy * end)))
                cv2.line(img, a, b, color, thickness, cv2.LINE_AA)
                pos += dash_px + gap_px

    def object_dimensions_m(class_name: object) -> tuple[float, float]:
        name = str(class_name or "").lower()
        if "bus" in name or "truck" in name:
            return 7.0, 2.5
        if "car" in name or "vehicle" in name:
            return 4.4, 2.0
        if "motorcycle" in name or "bicycle" in name or "bike" in name:
            return 1.9, 0.7
        if "person" in name or "rider" in name:
            return 0.8, 0.6
        if "stop sign" in name or "sign" in name or "light" in name:
            return 0.55, 0.55
        return 1.2, 0.9

    def heading_from_track(tk: dict) -> float:
        try:
            yaw = float(tk.get("yaw_rad"))
            if math.isfinite(yaw):
                return yaw
        except (TypeError, ValueError):
            pass
        vx = float(tk.get("vx_mps", 0.0) or 0.0)
        vy = float(tk.get("vy_mps", 0.0) or 0.0)
        if math.hypot(vx, vy) > 0.05:
            return math.atan2(vy, vx)
        future = tk.get("future_m") if isinstance(tk.get("future_m"), list) else []
        if future:
            try:
                fx, fy = float(future[-1][0]), float(future[-1][1])
                x = float(tk["x_m"])
                y = float(tk["y_m"])
                if math.hypot(fx - x, fy - y) > 0.05:
                    return math.atan2(fy - y, fx - x)
            except (KeyError, TypeError, ValueError, IndexError):
                pass
        return 0.0

    def box_points(tk: dict) -> np.ndarray | None:
        try:
            x = float(tk["x_m"])
            y = float(tk["y_m"])
        except (KeyError, TypeError, ValueError):
            return None
        try:
            length_m = float(tk.get("length_m"))
            width_m = float(tk.get("width_m"))
            if not (math.isfinite(length_m) and math.isfinite(width_m) and length_m > 0 and width_m > 0):
                raise ValueError
        except (TypeError, ValueError):
            length_m, width_m = object_dimensions_m(tk.get("class_name"))
        theta = heading_from_track(tk)
        forward = np.array([math.cos(theta), math.sin(theta)], dtype=np.float32)
        left = np.array([-math.sin(theta), math.cos(theta)], dtype=np.float32)
        center = np.array([x, y], dtype=np.float32)
        corners = []
        for lf, wl in ((1, 1), (1, -1), (-1, -1), (-1, 1)):
            local = center + forward * (lf * length_m * 0.5) + left * (wl * width_m * 0.5)
            bx, by = bev_geom.local_to_bev(float(local[0]), float(local[1]))
            corners.append((int(round(float(bx))), int(round(float(by)))))
        return np.array(corners, dtype=np.int32)

    def draw_object_box(tk: dict, fill: tuple[int, int, int]) -> None:
        poly = box_points(tk)
        if poly is None:
            return
        edge = tuple(int(max(0, c * 0.55)) for c in fill)
        cv2.fillPoly(out, [poly], fill, cv2.LINE_AA)
        cv2.polylines(out, [poly], True, edge, 2, cv2.LINE_AA)
        p0 = tuple(poly[2])
        p1 = tuple(poly[3])
        cv2.line(out, p0, p1, edge, 2, cv2.LINE_AA)

    def draw_short_object_trajectory(tk: dict, start_px: tuple[int, int], modes: list[dict]) -> None:
        name = str(tk.get("class_name", "")).lower()
        if "stop sign" in name or "sign" in name or "light" in name:
            return
        try:
            x0 = float(tk["x_m"])
            y0 = float(tk["y_m"])
        except (KeyError, TypeError, ValueError):
            return
        future = []
        if modes:
            future = modes[0].get("future_m", [])
        if not future:
            future = tk.get("future_m", [])
        pts = [start_px]
        max_len_m = 2.25
        for fwd_m, left_m in future:
            try:
                fx_m = float(fwd_m)
                fy_m = float(left_m)
            except (TypeError, ValueError):
                continue
            if math.hypot(fx_m - x0, fy_m - y0) > max_len_m and len(pts) >= 2:
                break
            fx, fy = bev_geom.local_to_bev(fx_m, fy_m)
            fxi, fyi = int(round(float(fx))), int(round(float(fy)))
            if 0 <= fxi < w and 0 <= fyi < h:
                pts.append((fxi, fyi))
            if len(pts) >= 5:
                break
        if len(pts) < 2:
            return
        arr = np.array(pts, dtype=np.int32)
        cv2.polylines(out, [arr], False, (46, 174, 92), 3, cv2.LINE_AA)

    tracks = yolo_objects or []
    for i, tk in enumerate(tracks[:12]):
        try:
            x = float(tk["x_m"])
            y = float(tk["y_m"])
            bx, by = bev_geom.local_to_bev(x, y)
            bxi, byi = int(round(float(bx))), int(round(float(by)))
        except (KeyError, TypeError, ValueError):
            continue
        if not (-w <= bxi <= 2 * w and -h <= byi <= 2 * h):
            continue
        modes = tk.get("future_modes") if isinstance(tk.get("future_modes"), list) else []
        if not modes:
            modes = [{"prob": 1.0, "future_m": tk.get("future_m", [])}]
        draw_short_object_trajectory(tk, (bxi, byi), modes)

    for tk in reversed(tracks[:12]):
        draw_object_box(tk, object_viz_color_rgb(tk.get("class_name")))

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
    p.add_argument("--device", default=None)
    p.add_argument("--remote-segmentation-modal", action="store_true",
                   help="Run SegFormer inference on Modal and keep BEV/planning local.")
    p.add_argument("--modal-app-name", default="caddy-segformer-remote")
    p.add_argument("--modal-function-name", default="segment_jpeg")
    p.add_argument("--mono3d-remote-modal", action="store_true",
                   help="Run learned monocular 3D detection on Modal and publish mono3d.jpg.")
    p.add_argument("--mono3d-cache-file", type=Path, default=None,
                   help="Precomputed FCOS3D cache JSON produced by precompute_mono3d_modal.py.")
    p.add_argument("--mono3d-modal-app-name", default="caddy-monocular3d-fcos3d")
    p.add_argument("--mono3d-modal-function-name", default="detect_jpeg")
    p.add_argument("--mono3d-score-threshold", type=float, default=0.05)
    p.add_argument("--segmentation-cache-meta", type=Path, default=None,
                   help="Precomputed uint8 segmentation cache metadata for offline video.")
    p.add_argument("--segmentation-cache-meta-left", type=Path, default=None,
                   help="Precomputed segmentation cache for the front-left camera.")
    p.add_argument("--segmentation-cache-meta-right", type=Path, default=None,
                   help="Precomputed segmentation cache for the front-right camera.")
    p.add_argument("--camera-slot-left", default="CAM_FRONT_LEFT",
                   help="E2E calibration slot for the front-left camera.")
    p.add_argument("--camera-slot-right", default="CAM_FRONT_RIGHT",
                   help="E2E calibration slot for the front-right camera.")
    p.add_argument("--yolo-cache-file", type=Path, default=None,
                   help="Precomputed YOLO11+ByteTrack detection JSON for offline video.")
    p.add_argument("--yolo-min-conf", type=float, default=0.20)
    p.add_argument("--object-predictor-url", default=None,
                   help="Optional HTTP endpoint for model-based object futures. "
                        "Receives caddy.object_prediction.v1 JSON and returns "
                        "per-track multimodal futures. Falls back to constant "
                        "velocity on errors.")
    p.add_argument("--object-predictor-timeout-ms", type=float, default=250.0)
    p.add_argument("--publish-hz", type=float, default=PUBLISH_HZ_DEFAULT)
    p.add_argument("--infer-hz", type=float, default=INFER_HZ_DEFAULT)
    p.add_argument("--video", default=None)
    p.add_argument("--no-loop", action="store_true")
    p.add_argument("--video-control-file", type=Path, default=VIDEO_CONTROL_FILE_DEFAULT,
                   help="JSON control file for offline video seek/pause.")
    p.add_argument("--control-log-file", type=Path, default=None,
                   help="Recorded control.jsonl to synchronize ground-truth steering/gas/brake in offline mode.")
    p.add_argument("--max-scan", type=int, default=16)
    p.add_argument("--source", default="uvc", choices=("uvc", "realsense"),
                   help="Active camera source.")
    p.add_argument("--bev-mode", default="homography", choices=("homography", "depth"),
                   help="BEV projection mode. 'homography' is the original "
                        "drive-by-segmentation calibrated ground-plane projection; "
                        "'depth' uses RealSense depth unprojection.")
    p.add_argument("--bev-size", type=int, default=500,
                   help="Square BEV raster size in pixels.")
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
    p.add_argument("--no-protective-stop", action="store_true",
                   help="Disable YOLO object time-to-collision braking. "
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
    p.add_argument("--clrnet-cache-file", type=Path, default=None,
                   help="Precomputed CLRNet lane cache JSON for offline video.")
    args = p.parse_args()
    if args.segmentation_map_file is None:
        args.segmentation_map_file = SEGMENTATION_MAP_FILE_DEFAULT
        if "SEGMENTATION_MAP_FILE" not in os.environ:
            args.segmentation_map_file = args.frames_dir / "segmentation_map.json"
    args.target_mph = max(0.0, float(args.target_mph))
    args.gps_route_lookahead_m = float(np.clip(args.gps_route_lookahead_m, 2.0, 20.0))
    args.gps_route_gain = float(np.clip(args.gps_route_gain, 0.0, 1.0))
    args.bev_size = int(np.clip(args.bev_size, 256, 800))
    # Ceiling is the steering-column travel limit, not 90°, so a strong route
    # authority can actually pull the cart through a turn instead of saturating.
    args.gps_route_max_bias_deg = float(np.clip(args.gps_route_max_bias_deg, 0.0, 270.0))
    target_gas_ff = constant_gas_for_mph(args.target_mph)
    autospeed_ctrl = AutoSpeedController()
    stop_sign_ctrl = StopSignController()
    bev_geom: seg_occupancy.BevGeometry | None = None
    launch_start_t: float | None = None

    seg_live, rt, plan_path, create_bev, create_overlay, colors = (
        import_segmentation_stack(args.seg_repo)
    )
    rt.BEV_SIZE = args.bev_size

    remote_segmentation = bool(args.remote_segmentation_modal)
    if remote_segmentation:
        device = "modal"
    else:
        import torch

        if args.device:
            device = args.device
        elif torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            device = "mps"
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
    # Metric geometry matches the BEV the planner draws into, so YOLO-projected
    # object positions line up with the planned-trajectory polyline.
    bev_geom = seg_occupancy.BevGeometry.from_ranges(
        rt.BEV_SIZE, rt.RANGE_FWD, rt.RANGE_SIDE
    )
    yolo_cache = YoloDetectionCache(args.yolo_cache_file) if args.yolo_cache_file else None
    ground_projector = GroundPlaneProjector(calib)
    mono3d_cache = Mono3DDetectionCache(args.mono3d_cache_file) if args.mono3d_cache_file else None
    mono3d_client: ModalMonocular3DClient | None = None
    latest_mono3d_status: dict = {"type": "disabled", "ok": False}
    if mono3d_cache is not None:
        latest_mono3d_status = mono3d_cache.status()
        print(
            f"[mono3d] using cache {mono3d_cache.path} "
            f"provider={mono3d_cache.provider} frames={mono3d_cache.frame_count} "
            f"score_thr={mono3d_cache.score_threshold:.2f}",
            flush=True,
        )
    elif args.mono3d_remote_modal:
        print(
            f"[mono3d] using Modal learned 3D detector "
            f"app={args.mono3d_modal_app_name} function={args.mono3d_modal_function_name} "
            f"score_thr={args.mono3d_score_threshold:.2f}",
            flush=True,
        )
        mono3d_client = ModalMonocular3DClient(
            args.mono3d_modal_app_name,
            args.mono3d_modal_function_name,
            calib,
            args.mono3d_score_threshold,
        )
        latest_mono3d_status = mono3d_client.status()
    object_predictor = ObjectTrajectoryPredictorClient(
        args.object_predictor_url,
        timeout_s=max(1.0, float(args.object_predictor_timeout_ms)) / 1000.0,
    )
    if yolo_cache is not None:
        print(
            f"[yolo] using cache {yolo_cache.path} model={yolo_cache.model} "
            f"frames={yolo_cache.frame_count} conf={yolo_cache.conf}",
            flush=True,
        )
        if object_predictor.enabled:
            print(
                f"[predictor] object futures via {object_predictor.url} "
                f"timeout_ms={args.object_predictor_timeout_ms:.0f}",
                flush=True,
            )
    control_log_file = args.control_log_file
    if control_log_file is None and args.video:
        candidate = Path(args.video).resolve().parent / "control.jsonl"
        if candidate.exists():
            control_log_file = candidate
    control_trace = None
    if control_log_file is not None and control_log_file.exists():
        control_trace = ControlTrace(control_log_file)
        print(
            f"[controls] using recorded controls {control_trace.path} "
            f"samples={len(control_trace.samples)}",
            flush=True,
        )

    gps_trace: GpsTrace | None = None
    if args.video:
        gps_candidate = Path(args.video).resolve().parent / "gps.json"
        if gps_candidate.exists():
            gps_trace = GpsTrace(gps_candidate)
            print(
                f"[gps] using recorded trace {gps_trace.path} "
                f"samples={len(gps_trace.times)}",
                flush=True,
            )

    readers = make_sources(args)
    for r in readers:
        r.start()

    active_slug = args.active_slug
    slug_to_reader = {getattr(r, "slug", active_slug): r for r in readers}
    active_reader = slug_to_reader.get(active_slug)
    if active_reader is None:
        raise RuntimeError(f"active camera {active_slug} not available")

    proc = None
    model = None
    modal_client: ModalSegmentationClient | None = None
    seg_cache: SegmentationMapCache | None = None
    seg_model_full = f"drive-by-segmentation-segformer-{args.model}"
    if args.segmentation_cache_meta is not None:
        seg_cache = SegmentationMapCache(args.segmentation_cache_meta)
        seg_model_full = f"{seg_model_full}-cache:{seg_cache.meta_path.name}"
        print(
            f"[seg] using segmentation cache {seg_cache.data_path} "
            f"frames={seg_cache.frame_count} shape={seg_cache.height}x{seg_cache.width}",
            flush=True,
        )
    seg_cache_left: SegmentationMapCache | None = None
    seg_cache_right: SegmentationMapCache | None = None
    multi_cam_remaps: list[seg_fast.BevRemap] | None = None
    if args.segmentation_cache_meta_left is not None:
        seg_cache_left = SegmentationMapCache(args.segmentation_cache_meta_left)
        print(f"[seg] left cache {seg_cache_left.data_path} frames={seg_cache_left.frame_count}", flush=True)
    if args.segmentation_cache_meta_right is not None:
        seg_cache_right = SegmentationMapCache(args.segmentation_cache_meta_right)
        print(f"[seg] right cache {seg_cache_right.data_path} frames={seg_cache_right.frame_count}", flush=True)
    multi_cam_enabled = seg_cache is not None and seg_cache_left is not None and seg_cache_right is not None
    if multi_cam_enabled:
        raw_calib = load_json_file(args.calib or (args.seg_repo / "camera_calibration.json"))
        multi_cam_height_m = calib["extrinsics"]["height_m"]
        multi_cam_fwd_ft = bev_range.get("forward_ft", 100)
        multi_cam_side_ft = bev_range.get("side_ft", 50)
        rt.RANGE_SIDE = multi_cam_side_ft * rt.FT_TO_M
        bev_geom = seg_occupancy.BevGeometry.from_ranges(
            rt.BEV_SIZE, rt.RANGE_FWD, rt.RANGE_SIDE
        )
        print(
            f"[seg] multi-cam fusion enabled: "
            f"slots={args.camera_slot},{args.camera_slot_left},{args.camera_slot_right} "
            f"range_side_ft={multi_cam_side_ft}",
            flush=True,
        )

    if remote_segmentation:
        if seg_cache is None:
            print(
                f"[seg] using Modal segmentation app={args.modal_app_name} "
                f"function={args.modal_function_name} variant={args.model}",
                flush=True,
            )
            modal_client = ModalSegmentationClient(
                args.modal_app_name,
                args.modal_function_name,
                args.model,
            )
            seg_model_full = f"{seg_model_full}-modal:{args.modal_app_name}/{args.modal_function_name}"
    else:
        if seg_cache is None:
            proc, model = seg_live.load_segformer(args.model, device)
    steer_est = rt.SteeringEstimator()
    clrnet = None
    clrnet_runner = None
    clrnet_lane_cache: CLRNetLaneCache | None = None
    if args.clrnet_cache_file and args.clrnet_cache_file.exists():
        clrnet_lane_cache = CLRNetLaneCache(args.clrnet_cache_file)
        try:
            clrnet = import_clrnet_stack()
        except Exception:
            clrnet = None
        print(
            f"[clrnet] using lane cache {clrnet_lane_cache.path} "
            f"frames={clrnet_lane_cache.frame_count}",
            flush=True,
        )
    if not args.no_clrnet and clrnet_lane_cache is None:
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
    use_fast = not args.no_fast and args.bev_mode == "homography"

    latest_overlay_bgr: np.ndarray | None = None
    latest_bev_bgr: np.ndarray | None = None
    latest_bev_cls_map: np.ndarray | None = None
    latest_objects_bgr: np.ndarray | None = None
    latest_yolo_bgr: np.ndarray | None = None
    latest_mono3d_bgr: np.ndarray | None = None
    latest_seg_map: np.ndarray | None = None
    latest_path: list[list[float]] = []
    latest_steer_raw = 0.0
    latest_steer_base = 0.0
    latest_steer_filtered = 0.0
    lane_traj = None
    lane_local = None
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
    latest_protective_stop: dict = build_environment_threat_from_autospeed(
        autospeed_ctrl, 0.0, 0, enabled=not args.no_protective_stop
    )
    latest_yolo_objects: list[dict] = []
    latest_mono3d_objects: list[dict] = []
    latest_mono3d_tracks: list[dict] = []
    latest_object_tracks: list[dict] = []
    latest_commanded_speed_mps = 0.0
    latest_autospeed_status: dict = autospeed_ctrl.status()
    latest_stop_sign_status: dict = stop_sign_ctrl.status()
    protective_enabled = not args.no_protective_stop
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
                    source_frame_index = max(0, latest_camera_frame_count - 1)
                    if args.video and hasattr(active_reader, "status"):
                        try:
                            source_frame_index = max(
                                0,
                                int(active_reader.status().get("frame_index", source_frame_index)),
                            )
                        except (TypeError, ValueError):
                            pass
                    if seg_cache is not None and args.video:
                        t_cache = time.perf_counter()
                        seg_map = seg_cache.get(source_frame_index)
                        seg_stage_ms = {
                            "seg_cache_lookup_ms": (time.perf_counter() - t_cache) * 1000.0,
                            "seg_cache_frame_index": float(source_frame_index),
                        }
                    elif remote_segmentation:
                        if modal_client is None:
                            raise RuntimeError("Modal segmentation client was not initialized")
                        seg_map, seg_stage_ms = timed_segment_frame_modal(frame_rgb, modal_client)
                    else:
                        seg_map, seg_stage_ms = timed_segment_frame(frame_rgb, proc, model, device)
                    latest_seg_map = seg_map
                    stage_ms.update(seg_stage_ms)
                    stage_last = time.perf_counter()

                    bev_cls_map = None
                    if multi_cam_enabled and args.video:
                        seg_map_left = seg_cache_left.get(source_frame_index)
                        seg_map_right = seg_cache_right.get(source_frame_index)
                        if multi_cam_remaps is None:
                            img_h, img_w = seg_map.shape[:2]
                            raw_calib_local = load_json_file(args.calib or (args.seg_repo / "camera_calibration.json"))
                            multi_cam_remaps = bev_fusion.build_multi_cam_remaps(
                                raw_calib_local,
                                [args.camera_slot, args.camera_slot_left, args.camera_slot_right],
                                img_h, img_w, rt.BEV_SIZE,
                                calib["extrinsics"]["height_m"],
                                range_fwd_ft=bev_range.get("forward_ft", 100),
                                range_side_ft=bev_range.get("side_ft", 50),
                            )
                            print(
                                f"[seg] built multi-cam BEV remaps "
                                f"({img_w}x{img_h} -> {rt.BEV_SIZE}x{rt.BEV_SIZE}) "
                                f"for {len(multi_cam_remaps)} cameras",
                                flush=True,
                            )
                        bev_rgb, bev_cls_map = bev_fusion.fused_bev_colored(
                            [seg_map, seg_map_left, seg_map_right],
                            multi_cam_remaps,
                            palette,
                        )
                    elif args.bev_mode == "depth":
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
                            bev_rgb = seg_fast.create_bev_cached(seg_map, palette, bev_remap)
                            bev_cls_map = bev_class_map_cached(seg_map, bev_remap)
                        else:
                            bev_out = create_bev(
                                seg_map, calib, rt.BEV_SIZE, return_class_map=True
                            )
                            bev_rgb, bev_cls_map = bev_out
                    mark("bev_ms")
                    # Run the expensive path planner every 3rd frame; reuse
                    # lane_traj/lane_local on the frames in between.
                    run_planner = (infer_count % 3 == 0) or lane_local is None
                    if run_planner:
                        if bev_cls_map is not None:
                            road_mask = bev_cls_map == 0
                        else:
                            road_mask = (
                                np.all(bev_rgb == road_color, axis=-1)
                                | np.all(bev_rgb == grid_color, axis=-1)
                                | np.all(bev_rgb == grid2_color, axis=-1)
                            )
                        if use_fast:
                            _gps_bearing = gps_route_bearing_rad(args)
                            lane_traj, lane_local = seg_fast.lane_aware_centerline_path_fast(
                                road_mask,
                                bev_size=rt.BEV_SIZE,
                                range_fwd=rt.RANGE_FWD,
                                range_side=rt.RANGE_SIDE,
                                road_width_ft=road_width_ft,
                                gps_bearing_rad=_gps_bearing,
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
                    # Autospeed: unified path-aware obstacle speed control.
                    # When the Modal 3D detector is enabled, learned 3D boxes
                    # are the object source. YOLO remains only as a fallback for
                    # launches that do not enable mono3d.
                    latest_collision = None
                    latest_brake_corridor = None
                    protective_enabled = not args.no_protective_stop
                    latest_yolo_objects = []
                    latest_mono3d_tracks = []
                    latest_object_tracks = []
                    mono3d_enabled = mono3d_cache is not None or mono3d_client is not None
                    if not mono3d_enabled and yolo_cache is not None and ground_projector is not None:
                        latest_yolo_objects = yolo_cache.objects_for_frame(
                            source_frame_index,
                            ground_projector,
                            horizon_s=AUTOSPEED_LOOKAHEAD_TIME,
                        )
                        latest_yolo_objects = object_predictor.apply(
                            source_frame_index,
                            yolo_cache.fps,
                            latest_yolo_objects,
                            latest_ego_speed_mph * 0.44704 if latest_ego_speed_ok else args.target_mph * 0.44704,
                            latest_path,
                            horizon_s=AUTOSPEED_LOOKAHEAD_TIME,
                            step_s=0.5,
                        )
                    latest_yolo_bgr = (
                        draw_yolo_overlay(frame_bgr, latest_yolo_objects, None)
                        if not mono3d_enabled and latest_yolo_objects
                        else None
                    )
                    if mono3d_cache is not None:
                        latest_mono3d_objects = mono3d_cache.objects_for_frame(source_frame_index)
                        latest_mono3d_tracks = mono3d_objects_to_tracks(latest_mono3d_objects)
                        latest_mono3d_bgr = draw_mono3d_overlay(frame_bgr, latest_mono3d_objects, calib)
                        latest_mono3d_status = mono3d_cache.status()
                    elif mono3d_client is not None:
                        mono3d_bgr, mono3d_objects, mono3d_ms = mono3d_client.detect(frame_bgr)
                        stage_ms.update(mono3d_ms)
                        latest_mono3d_objects = mono3d_objects
                        latest_mono3d_tracks = mono3d_objects_to_tracks(mono3d_objects)
                        latest_mono3d_bgr = draw_mono3d_overlay(
                            frame_bgr,
                            latest_mono3d_objects,
                            calib,
                        ) if mono3d_objects else mono3d_bgr
                        latest_mono3d_status = mono3d_client.status()
                        mark("mono3d_ms")
                    latest_object_tracks = (
                        latest_mono3d_tracks if mono3d_enabled else latest_yolo_objects
                    )
                    latest_occupancy = None
                    latest_image_brake = None
                    ego_mps = (
                        latest_ego_speed_mph * 0.44704
                        if latest_ego_speed_ok
                        else args.target_mph * 0.44704
                    )
                    max_speed_mps = args.target_mph * 0.44704
                    if protective_enabled:
                        latest_commanded_speed_mps = autospeed_ctrl.compute(
                            path=lane_local if path_ok else None,
                            obstacles=latest_object_tracks,
                            current_speed=ego_mps,
                            max_speed=max_speed_mps,
                        )
                        latest_brake_corridor = forward_stop_corridor_bev(
                            bev_geom, AUTOSPEED_MIN_GAP
                        )
                    else:
                        latest_commanded_speed_mps = max_speed_mps
                    raw_stop_signs = []
                    if not mono3d_enabled and yolo_cache is not None:
                        for det in yolo_cache.raw_detections(source_frame_index):
                            if str(det.get("class_name", "")) == "stop sign":
                                raw_stop_signs.append(det)
                    stop_sign_limit = stop_sign_ctrl.update(
                        latest_object_tracks, ego_mps, max_speed_mps,
                        raw_stop_signs=raw_stop_signs,
                    )
                    pre_stop_speed = latest_commanded_speed_mps
                    latest_commanded_speed_mps = min(
                        latest_commanded_speed_mps, stop_sign_limit,
                    )
                    latest_stop_sign_status = stop_sign_ctrl.status()
                    # Update desired_accel to reflect stop sign deceleration
                    if latest_commanded_speed_mps < pre_stop_speed - 0.01:
                        effective_accel = (latest_commanded_speed_mps - ego_mps) / max(AUTOSPEED_DT, 1e-6)
                        autospeed_ctrl.desired_accel = max(effective_accel, -AUTOSPEED_EMERGENCY_DECEL)
                    latest_autospeed_status = autospeed_ctrl.status()
                    latest_env_brake_01 = (
                        1.0 if autospeed_ctrl.emergency_active
                        else max(0.0, 1.0 - latest_commanded_speed_mps / max(max_speed_mps, 1e-6))
                    )
                    latest_protective_stop = build_environment_threat_from_autospeed(
                        autospeed_ctrl,
                        latest_commanded_speed_mps,
                        len(latest_object_tracks),
                        enabled=protective_enabled,
                    )
                    mark("protective_stop_ms")
                    # Map commanded speed to gas/brake pot values
                    latest_speed_setpoint_mph = latest_commanded_speed_mps * 2.23694
                    if args.constant_speed:
                        latest_target_gas = target_gas_ff
                        latest_gas_trim = 0.0
                        latest_target_brake = 0.0
                    else:
                        if launch_start_t is None:
                            launch_start_t = time.monotonic()
                        ramp_age = time.monotonic() - launch_start_t
                        if LAUNCH_RAMP_S > 0.0 and ramp_age < LAUNCH_RAMP_S:
                            frac = max(0.0, ramp_age / LAUNCH_RAMP_S)
                            ramp_max_speed = max_speed_mps * frac
                            clamped_cmd = min(latest_commanded_speed_mps, ramp_max_speed)
                        else:
                            clamped_cmd = latest_commanded_speed_mps
                        latest_target_gas, latest_target_brake = autospeed_ctrl.gas_brake_from_speed(
                            clamped_cmd, ego_mps, max_speed_mps,
                        )
                        latest_gas_trim = 0.0
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

                    if clrnet_lane_cache is not None:
                        lanes = clrnet_lane_cache.lanes_for_frame(source_frame_index)
                    elif clrnet_runner is not None:
                        try:
                            lanes = clrnet_runner.infer(frame_bgr)
                        except Exception as e:
                            print(
                                f"[clrnet] infer failed: {type(e).__name__}: {e}",
                                flush=True,
                            )
                            lanes = []
                    else:
                        lanes = None
                    if lanes is not None:
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
                    render_viz = (infer_count % 2 == 0)
                    if render_viz:
                        overlay_rgb = create_overlay(frame_rgb, seg_map)
                        latest_overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
                    mark("overlay_ms")
                    if render_viz:
                        if bev_cls_map is not None:
                            latest_bev_cls_map = bev_cls_map
                        latest_bev_bgr = draw_bev_viz(
                            bev_rgb, lane_traj, lane_local, rt,
                            occ=latest_occupancy, bev_geom=bev_geom,
                            brake_corridor=latest_brake_corridor,
                            brake_01=latest_env_brake_01,
                            stop_active=bool(latest_protective_stop.get("active")),
                        )
                        latest_objects_bgr = draw_object_viz(
                            bev_rgb,
                            bev_geom,
                            None,
                            yolo_objects=latest_object_tracks,
                            autospeed_status=latest_autospeed_status,
                            stop_sign_status=latest_stop_sign_status,
                            bev_cls_map=bev_cls_map,
                            clrnet_lanes=latest_clrnet_lanes,
                            ground_projector=ground_projector,
                            clrnet_conf_threshold=CLRNET_CONF_THRESHOLD,
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
                if latest_bev_cls_map is not None:
                    t = time.perf_counter()
                    write_png_atomic(args.frames_dir / "bev_classmap.png", latest_bev_cls_map)
                    jpeg_ms["png_bev_classmap_ms"] = (time.perf_counter() - t) * 1000.0
                if latest_objects_bgr is not None:
                    t = time.perf_counter()
                    write_jpeg_atomic(args.frames_dir / "objects.jpg", latest_objects_bgr, quality=85)
                    jpeg_ms["jpeg_objects_ms"] = (time.perf_counter() - t) * 1000.0
                if latest_yolo_bgr is not None:
                    t = time.perf_counter()
                    write_jpeg_atomic(args.frames_dir / "yolo.jpg", latest_yolo_bgr, quality=82)
                    jpeg_ms["jpeg_yolo_ms"] = (time.perf_counter() - t) * 1000.0
                if latest_mono3d_bgr is not None:
                    t = time.perf_counter()
                    write_jpeg_atomic(args.frames_dir / "mono3d.jpg", latest_mono3d_bgr, quality=82)
                    jpeg_ms["jpeg_mono3d_ms"] = (time.perf_counter() - t) * 1000.0
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
                            seg_model_full,
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
                video_status = (
                    active_reader.status()
                    if args.video and hasattr(active_reader, "status")
                    else None
                )
                video_paused = bool(video_status and video_status.get("paused"))
                video_position_s = (
                    float(video_status.get("position_s"))
                    if isinstance(video_status, dict) and video_status.get("position_s") is not None
                    else None
                )
                ground_truth_control = (
                    control_trace.sample_at(video_position_s)
                    if control_trace is not None else None
                )
                camera_stale = (camera_age_s > 0.5) and not video_paused
                inference_stale = (
                    ((cam_now - latest_infer_ok_s) > 0.5) and not video_paused
                    if latest_infer_ok_s else True
                )

                mean_dt = sum(infer_times) / len(infer_times) if infer_times else 0.0
                fps = 0.0 if camera_stale or inference_stale else (1.0 / mean_dt if mean_dt > 0 else 0.0)
                collision_speed_mph = float(latest_commanded_speed_mps * 2.23694) if inference_ok else 0.0
                state = {
                    "steer_deg": latest_steer_raw,
                    "steer_deg_raw": latest_steer_raw,
                    "active_cam": active_slug,
                    "inference": inference_ok and not camera_stale and not inference_stale,
                    "viz": (
                        latest_overlay_bgr is not None
                        or latest_bev_bgr is not None
                        or latest_objects_bgr is not None
                        or latest_yolo_bgr is not None
                        or latest_mono3d_bgr is not None
                        or latest_clrnet_overlay_bgr is not None
                    ),
                    "viz_streams": [
                        slug for slug, img in (
                            ("seg", latest_overlay_bgr),
                            ("bev", latest_bev_bgr),
                            ("objects", latest_objects_bgr),
                            ("yolo", latest_yolo_bgr),
                            ("mono3d", latest_mono3d_bgr),
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
                    "video": video_status,
                    "model": MODEL_NAME,
                    "model_full": seg_model_full,
                    "target_speed_mph": args.target_mph if inference_ok else 0.0,
                    "speed_setpoint_mph": latest_speed_setpoint_mph if inference_ok else 0.0,
                    "collision_speed_mph": collision_speed_mph if inference_ok else 0.0,
                    "ego_speed_mph": latest_ego_speed_mph,
                    "ego_speed_ok": latest_ego_speed_ok,
                    "steer_deg_base": latest_steer_base,
                    "steer_source": latest_steer_source,
                    "steer_lookahead_m": latest_lookahead_m,
                    "target_gas": latest_target_gas if inference_ok else 0.0,
                    "target_gas_ff": target_gas_ff,
                    "target_gas_trim": latest_gas_trim,
                    "speed_mode": (
                        "emergency_stop" if autospeed_ctrl.emergency_active
                        else "autospeed" if protective_enabled
                        else "constant" if args.constant_speed
                        else "feedforward_pot"
                    ),
                    "target_brake": latest_target_brake if inference_ok else 0.0,
                    "ground_truth_control": ground_truth_control,
                    "predicted_path": (
                        []
                        if latest_clrnet_override
                        else (latest_path if inference_ok else [])
                    ),
                    "bev": {
                        "bev_size": rt.BEV_SIZE,
                        "range_fwd_ft": round(rt.RANGE_FWD / rt.FT_TO_M, 1),
                        "range_side_ft": round(rt.RANGE_SIDE / rt.FT_TO_M, 1),
                        "multi_cam": bool(multi_cam_enabled),
                    },
                    "segmentation": {
                        "bev_mode": args.bev_mode,
                        "remote_modal": bool(remote_segmentation),
                        "modal_app_name": args.modal_app_name if remote_segmentation else None,
                        "modal_function_name": (
                            args.modal_function_name if remote_segmentation else None
                        ),
                        "latency_ms": latest_latency_ms,
                        "jpeg_ms": {k: round(float(v), 3) for k, v in jpeg_ms.items()},
                        "publish_jpeg_total_ms": round(float(publish_jpeg_total_ms), 3),
                        "infer_count": int(infer_count),
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
                        "object_detector": {
                            "type": (
                                latest_mono3d_status.get("type", "modal_mmdet3d_fcos3d")
                                if (mono3d_cache is not None or mono3d_client is not None)
                                else "yolo11+bytetrack" if yolo_cache is not None
                                else "none"
                            ),
                            "model": (
                                latest_mono3d_status.get("provider")
                                if (mono3d_cache is not None or mono3d_client is not None)
                                else yolo_cache.model if yolo_cache is not None
                                else None
                            ),
                            "cache_file": (
                                str(mono3d_cache.path) if mono3d_cache is not None
                                else None if mono3d_client is not None
                                else str(yolo_cache.path) if yolo_cache is not None
                                else None
                            ),
                            "min_confidence": (
                                float(latest_mono3d_status.get("score_threshold", args.mono3d_score_threshold))
                                if (mono3d_cache is not None or mono3d_client is not None)
                                else float(args.yolo_min_conf)
                            ),
                            "trajectory_predictor": (
                                {"type": "mono3d_velocity", "provider": "mmdet3d_fcos3d_nuscenes"}
                                if (mono3d_cache is not None or mono3d_client is not None)
                                else object_predictor.status()
                            ),
                        },
                        "object_tracks": latest_object_tracks,
                        "learned_3d_detector": latest_mono3d_status,
                        "learned_3d_objects": latest_mono3d_objects,
                        "gps_route": latest_gps_route,
                        "map_file": str(args.segmentation_map_file),
                        "map_schema": "caddy.segmentation_map.v1",
                    },
                    "autospeed": {
                        "commanded_speed_mph": round(float(latest_commanded_speed_mps * 2.23694), 2),
                        "commanded_speed_mps": round(float(latest_commanded_speed_mps), 3),
                        "max_speed_mph": round(float(args.target_mph), 1),
                        **latest_autospeed_status,
                    },
                    "stop_sign": latest_stop_sign_status,
                    "ts": time.time(),
                }
                t = time.perf_counter()
                write_json_atomic(args.state_file, state)
                latest_latency_ms["json_state_ms"] = round((time.perf_counter() - t) * 1000.0, 3)
                if gps_trace is not None and video_position_s is not None:
                    gps_trace.write_state(video_position_s, args.gps_state_file)
                next_publish_t = now + publish_period

            if now - last_log_t >= 1.0:
                log_fps = infer_count - getattr(main, '_last_log_count', 0)
                main._last_log_count = infer_count
                last_log_t = now
                confs = " ".join(f"{s:.2f}" for s in latest_clrnet_confidences[:6])
                if not confs:
                    confs = "none"
                cmd_mph = latest_commanded_speed_mps * 2.23694
                n_limits = len(autospeed_ctrl.speed_limits)
                lat = latest_latency_ms
                print(
                    f"[run] {log_fps}fps "
                    f"total={lat.get('infer_total_ms',0):.0f}ms "
                    f"plan={lat.get('path_plan_ms',0):.0f}ms "
                    f"bev={lat.get('bev_ms',0):.0f}ms "
                    f"viz={lat.get('bev_viz_ms',0):.0f}ms "
                    f"ovl={lat.get('overlay_ms',0):.0f}ms "
                    f"frame={infer_count} "
                    f"src={latest_steer_source or 'seg'} "
                    f"clr={'Y' if latest_clrnet_override else 'N'} "
                    f"conf=[{confs}] "
                    f"autospeed={cmd_mph:.1f}mph "
                    f"accel={autospeed_ctrl.desired_accel:+.2f} "
                    f"emergency={'Y' if autospeed_ctrl.emergency_active else 'N'} "
                    f"limits={n_limits} "
                    f"objects={len(latest_object_tracks)} "
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

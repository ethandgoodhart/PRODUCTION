#!/usr/bin/env python3
"""
segmentation_infer.py — drive-by-segmentation live planner sidecar.

Uses the cloned https://github.com/ethandgoodhart/drive-by-segmentation
implementation for SegFormer semantic segmentation, BEV projection,
lane-aware trajectory planning, and steering estimation. Publishes the
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
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import seg_fast  # noqa: E402


SEG_REPO_DEFAULT = Path("/home/caddy/drive-by-segmentation")
FRAMES_DIR_DEFAULT = Path("/tmp/cart_frames")
STATE_FILE_DEFAULT = Path("/tmp/autoware_state.json")
EGO_STATE_FILE_DEFAULT = Path(os.environ.get("EGO_STATE_FILE", "/tmp/ego_state.json"))

# Match the usual live-model camera labels on this cart. The second and
# third discovered cameras are front_wide / left respectively; if these are
# reversed, segmentation will infer from the side-facing left camera.
SLUGS = ("front_narrow", "front_wide", "left", "right")
ACTIVE_SLUG = "front_wide"
CAMERA_ORIENTATION_FIX = {"front_narrow": -1}

CAM_W, CAM_H = 640, 480
JPEG_QUALITY = 72
PUBLISH_HZ_DEFAULT = 15.0
INFER_HZ_DEFAULT = 0.0
MODEL_NAME = "segmentation"
STEERING_SIGN = -1.0
STEERING_COLUMN_RATIO = 15.0
STEERING_EMA = 1.0
LOOKAHEAD_MIN_M = 2.5
LOOKAHEAD_MAX_M = 10.0
LOOKAHEAD_TIME_S = 0.9

LOOKAHEAD_HEADING_EMA = 0.4

# Polynomial-decomposition steering gains.
# Evaluated against recorded driving on 2026-05-03. The planner's lane_local
# starts ~2 m ahead of the cart, so we evaluate the polynomial at the
# lookahead distance (not at x=0) — the offset/heading terms then have
# their conventional pure-pursuit meaning.
#   STEER_K_CROSSTRACK : per meter of lateral offset at the lookahead
#   STEER_K_HEADING    : per radian of path tangent at the lookahead
#   STEER_K_CURVATURE  : per (1/m) of path curvature × lookahead_m
STEER_K_CROSSTRACK = 0.04
STEER_K_HEADING = 0.15
STEER_K_CURVATURE = 0.08

# Limit polynomial fit to a near-only window — the far end of the planner's
# polyline is too noisy to inform our slope/curvature.
STEER_FIT_MIN_M = 1.0
STEER_FIT_MAX_M = 6.0

# Constant-speed target. ps5_drive consumes absolute pedal pot targets, so
# keep this conservative and publish the intended speed separately for UI/logs.
TARGET_MPH = 8.0
GAS_CONSTANT = 0.24
GAS_PER_MPH = GAS_CONSTANT / TARGET_MPH
BRAKE_CONSTANT = 0.0


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


class RealSenseReader(threading.Thread):
    """RealSense color + aligned depth source.

    Mirrors CameraReader's `.latest()` / `.stop()` API so it slots into the
    existing slug_to_reader map. `latest_color_depth()` additionally returns
    the matched depth frame and pinhole intrinsics needed for the BEV cloud.
    """

    def __init__(self, slug: str, width: int = 640, height: int = 480,
                 fps: int = 30):
        super().__init__(daemon=True, name=f"cam-{slug}")
        import pyrealsense2 as rs
        self.rs = rs
        self.slug = slug
        self.flip_code = CAMERA_ORIENTATION_FIX.get(slug)
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        profile = self.pipeline.start(cfg)
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = float(depth_sensor.get_depth_scale())
        color_prof = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = color_prof.get_intrinsics()
        self.intrinsics = (float(intr.fx), float(intr.fy),
                           float(intr.ppx), float(intr.ppy))
        self.align = rs.align(rs.stream.color)
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
            with self.lock:
                self.frame = color
                self.depth = depth
                self.frame_count += 1
                self.last_ok_s = time.monotonic()

    def latest(self) -> np.ndarray | None:
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def latest_color_depth(self) -> tuple[np.ndarray, np.ndarray] | None:
        with self.lock:
            if self.frame is None or self.depth is None:
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
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(payload, f, separators=(",", ":"))
    os.replace(tmp, path)


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
    return float(np.clip(target_mph * GAS_PER_MPH, 0.0, 1.0))


_lookahead_heading_filt_deg: float | None = None


def reset_lookahead_heading_filter() -> None:
    global _lookahead_heading_filt_deg
    _lookahead_heading_filt_deg = None


def lookahead_heading_steering_deg(lane_local: np.ndarray, lookahead_m: float) -> float | None:
    """Polynomial-decomposition steering from a BEV lane centerline.

    Fits y(x) = a0 + a1 x + a2 x^2 to the lane points (x forward, y lateral)
    inside a near-only window, then evaluates at the lookahead distance:
        y_lh = y(lookahead_m)         (cross-track at the lookahead)
        y'_lh = a1 + 2 a2 lookahead_m (heading at the lookahead)
        kappa = 2 a2                  (path curvature)
    Commanded heading = Ky·y_lh + Kh·y'_lh + Kκ·kappa·lookahead_m.
    """
    global _lookahead_heading_filt_deg

    if lane_local is None or len(lane_local) < 6:
        return None

    pts = np.asarray(lane_local, dtype=np.float64)
    pts = pts[np.isfinite(pts).all(axis=1)]
    pts = pts[(pts[:, 0] > STEER_FIT_MIN_M) & (pts[:, 0] < STEER_FIT_MAX_M)]
    if len(pts) < 6:
        return None

    x = pts[:, 0]
    y = pts[:, 1]

    # Sharper near-weighting than 1/(1+x) so the far end of the window doesn't
    # dominate sheer-numerically when the planner emits many close-spaced points.
    weights = np.exp(-x / 2.5)
    try:
        a2, a1, a0 = np.polyfit(x, y, deg=2, w=weights)
    except (np.linalg.LinAlgError, ValueError):
        return None

    y_lh = float(a0 + a1 * lookahead_m + a2 * lookahead_m * lookahead_m)
    yp_lh = float(a1 + 2.0 * a2 * lookahead_m)
    kappa = 2.0 * float(a2)

    crosstrack_rad = STEER_K_CROSSTRACK * y_lh
    heading_rad = STEER_K_HEADING * yp_lh
    curvature_rad = STEER_K_CURVATURE * kappa * float(lookahead_m)

    raw_heading_deg = math.degrees(crosstrack_rad + heading_rad + curvature_rad)

    if _lookahead_heading_filt_deg is None:
        _lookahead_heading_filt_deg = raw_heading_deg
    else:
        _lookahead_heading_filt_deg = (
            LOOKAHEAD_HEADING_EMA * raw_heading_deg
            + (1.0 - LOOKAHEAD_HEADING_EMA) * _lookahead_heading_filt_deg
        )

    return float(np.clip(_lookahead_heading_filt_deg * STEERING_COLUMN_RATIO, -270.0, 270.0))


def make_sources(args) -> list[CameraReader | VideoReader | RealSenseReader]:
    if args.video:
        reader = VideoReader(args.video, loop=not args.no_loop)
        reader.slug = ACTIVE_SLUG
        return [reader]

    if args.source == "realsense":
        rs_reader = RealSenseReader(
            slug=ACTIVE_SLUG,
            width=args.rs_width, height=args.rs_height, fps=args.rs_fps,
        )
        return [rs_reader]

    indices = discover_v4l2_indices(count=len(SLUGS), max_scan=args.max_scan)
    if len(indices) < len(SLUGS):
        raise RuntimeError(f"expected {len(SLUGS)} cameras, found {indices}")

    readers: list[CameraReader] = []
    for slug, idx in zip(SLUGS, indices):
        cap = open_camera(idx)
        if cap is None:
            raise RuntimeError(f"failed to open camera {idx} for {slug}")
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
                 lane_local: np.ndarray | None, rt) -> np.ndarray:
    out = bev_rgb.copy()
    if lane_traj is not None:
        rt.draw_trajectory(out, lane_traj, (255, 255, 0), 3)
    la_point, _ = rt.lookahead_point(lane_local)
    if la_point is not None:
        la_bx = int((la_point[1] / rt.RANGE_SIDE * 0.5 + 0.5) * rt.BEV_SIZE)
        la_by = int((1 - la_point[0] / rt.RANGE_FWD) * rt.BEV_SIZE)
        ego_bx, ego_by = rt.BEV_SIZE // 2, rt.BEV_SIZE - 1
        if 0 <= la_bx < rt.BEV_SIZE and 0 <= la_by < rt.BEV_SIZE:
            cv2.line(out, (ego_bx, ego_by), (la_bx, la_by),
                     (0, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(out, (la_bx, la_by), 6, (0, 255, 255), -1, cv2.LINE_AA)
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)


def main() -> None:
    p = argparse.ArgumentParser(description="drive-by-segmentation WebUI sidecar")
    p.add_argument("--frames-dir", type=Path, default=FRAMES_DIR_DEFAULT)
    p.add_argument("--state-file", type=Path, default=STATE_FILE_DEFAULT)
    p.add_argument("--seg-repo", type=Path, default=SEG_REPO_DEFAULT)
    p.add_argument("--model", default="b0", choices=("b0", "b2", "b5"))
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
    args = p.parse_args()
    args.target_mph = max(0.0, float(args.target_mph))
    target_gas = constant_gas_for_mph(args.target_mph)

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

    with (args.seg_repo / "camera_calibration.json").open() as f:
        calib = json.load(f)

    bev_range = calib.get("bev_range", {})
    rt.RANGE_FWD = bev_range.get("forward_ft", 50) * rt.FT_TO_M
    rt.RANGE_SIDE = bev_range.get("side_ft", 25) * rt.FT_TO_M
    road_width_ft = float(calib.get("road_width_ft", 20.0))

    proc, model = seg_live.load_segformer(args.model, device)
    steer_est = rt.SteeringEstimator()

    readers = make_sources(args)
    for r in readers:
        r.start()

    slug_to_reader = {getattr(r, "slug", ACTIVE_SLUG): r for r in readers}
    active_reader = slug_to_reader.get(ACTIVE_SLUG)
    if active_reader is None:
        raise RuntimeError(f"active camera {ACTIVE_SLUG} not available")

    palette = np.array(colors, dtype=np.uint8)
    road_color = np.array(colors[0], dtype=np.uint8)
    grid_color = np.clip(road_color.astype(np.int16) + 35, 0, 255).astype(np.uint8)
    grid2_color = np.clip(road_color.astype(np.int16) + 70, 0, 255).astype(np.uint8)

    publish_period = 1.0 / max(args.publish_hz, 1e-3)
    infer_period = 1.0 / args.infer_hz if args.infer_hz > 0 else 0.0
    next_publish_t = 0.0
    next_infer_t = 0.0
    infer_times: list[float] = []

    bev_remap: seg_fast.BevRemap | None = None
    use_fast = not args.no_fast and args.bev_mode == "homography"

    latest_overlay_bgr: np.ndarray | None = None
    latest_bev_bgr: np.ndarray | None = None
    latest_path: list[list[float]] = []
    latest_steer_raw = 0.0
    latest_steer_base = 0.0
    latest_steer_filtered = 0.0
    latest_lookahead_m = 0.0
    latest_ego_speed_mph = 0.0
    latest_ego_speed_ok = False
    inference_ok = False
    latest_latency_ms: dict[str, float] = {}
    infer_count = 0

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
                else:
                    frame_bgr = active_reader.latest()
                    depth_m = None
                mark("source_latest_ms")
                if frame_bgr is not None:
                    t0 = time.perf_counter()
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    mark("bgr_to_rgb_ms")
                    seg_map, seg_stage_ms = timed_segment_frame(frame_rgb, proc, model, device)
                    stage_ms.update(seg_stage_ms)
                    stage_last = time.perf_counter()

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
                            bev_rgb = seg_fast.create_bev_cached(seg_map, palette, bev_remap)
                        else:
                            bev_rgb = create_bev(seg_map, calib, rt.BEV_SIZE)
                    mark("bev_ms")
                    road_mask = (
                        np.all(bev_rgb == road_color, axis=-1)
                        | np.all(bev_rgb == grid_color, axis=-1)
                        | np.all(bev_rgb == grid2_color, axis=-1)
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

                    if lane_local is not None and len(lane_local) >= 4:
                        steer_est.update_bev(lane_local)
                        latest_path = [
                            [float(x), float(y)] for x, y in lane_local[:: max(1, len(lane_local) // 40)]
                        ]
                        inference_ok = True
                    else:
                        latest_path = []
                        inference_ok = False
                    mark("path_state_ms")

                    latest_ego_speed_mph, latest_ego_speed_ok = read_ego_speed_mph(
                        args.ego_state_file
                    )
                    latest_lookahead_m = adaptive_lookahead_m(latest_ego_speed_mph, latest_ego_speed_ok)
                    heading_steer = lookahead_heading_steering_deg(lane_local, latest_lookahead_m)
                    if heading_steer is not None:
                        latest_steer_base = heading_steer * STEERING_SIGN
                    else:
                        latest_steer_base = float(steer_est.steering_deg) * STEERING_SIGN
                    latest_steer_filtered = (
                        STEERING_EMA * latest_steer_base
                        + (1.0 - STEERING_EMA) * latest_steer_filtered
                    )
                    latest_steer_raw = float(np.clip(latest_steer_filtered, -270.0, 270.0))
                    mark("steer_ms")
                    overlay_rgb = create_overlay(frame_rgb, seg_map)
                    latest_overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
                    mark("overlay_ms")
                    latest_bev_bgr = draw_bev_viz(bev_rgb, lane_traj, lane_local, rt)
                    mark("bev_viz_ms")

                    infer_times.append(time.perf_counter() - t0)
                    if len(infer_times) > 20:
                        infer_times.pop(0)
                    infer_count += 1
                    stage_ms["infer_total_ms"] = (time.perf_counter() - stage_start) * 1000.0
                    latest_latency_ms = {k: round(float(v), 3) for k, v in stage_ms.items()}
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
                publish_jpeg_total_ms = (time.perf_counter() - publish_start) * 1000.0

                mean_dt = sum(infer_times) / len(infer_times) if infer_times else 0.0
                fps = 1.0 / mean_dt if mean_dt > 0 else 0.0
                state = {
                    "steer_deg": latest_steer_raw,
                    "steer_deg_raw": latest_steer_raw,
                    "active_cam": ACTIVE_SLUG,
                    "inference": inference_ok,
                    "viz": latest_overlay_bgr is not None or latest_bev_bgr is not None,
                    "viz_streams": [
                        slug for slug, img in (("seg", latest_overlay_bgr), ("bev", latest_bev_bgr))
                        if img is not None
                    ],
                    "object_count": 0,
                    "fps": float(fps),
                    "cams": [r.slug for r in readers],
                    "source": "video" if args.video else "camera",
                    "model": MODEL_NAME,
                    "model_full": f"drive-by-segmentation-segformer-{args.model}",
                    "target_speed_mph": args.target_mph if inference_ok else 0.0,
                    "ego_speed_mph": latest_ego_speed_mph,
                    "ego_speed_ok": latest_ego_speed_ok,
                    "steer_deg_base": latest_steer_base,
                    "steer_lookahead_m": latest_lookahead_m,
                    "target_gas": target_gas if inference_ok else 0.0,
                    "target_brake": BRAKE_CONSTANT,
                    "predicted_path": latest_path if inference_ok else [],
                    "segmentation": {
                        "bev_mode": args.bev_mode,
                        "latency_ms": latest_latency_ms,
                        "jpeg_ms": {k: round(float(v), 3) for k, v in jpeg_ms.items()},
                        "publish_jpeg_total_ms": round(float(publish_jpeg_total_ms), 3),
                        "infer_count": int(infer_count),
                    },
                    "ts": time.time(),
                }
                t = time.perf_counter()
                write_json_atomic(args.state_file, state)
                latest_latency_ms["json_state_ms"] = round((time.perf_counter() - t) * 1000.0, 3)
                next_publish_t = now + publish_period

            time.sleep(0.005)
    finally:
        for r in readers:
            r.stop()


if __name__ == "__main__":
    main()

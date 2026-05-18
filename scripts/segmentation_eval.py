#!/usr/bin/env python3
"""segmentation_eval.py — offline replay of a recorded front-wide video
through the segmentation pipeline + the polynomial-decomposition steering
controller in scripts/segmentation_infer.py.

Produces an MP4 with:
  - segmentation overlay on the camera image
  - BEV with road mask, lane center polyline (planner), and the
    polynomial fit reconstructed back into BEV pixels
  - steering wheel + numerical breakdown of the cross-track / heading /
    curvature terms

Usage:
  /usr/bin/python3 scripts/segmentation_eval.py \\
      /path/to/Caddy-Training-Data-... \\
      --output /tmp/seg_eval.mp4 \\
      --duration 30 \\
      --infer-hz 10
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, "/home/caddy/drive-by-segmentation")

import segmentation_infer as si  # noqa: E402
import live as seg_live  # noqa: E402
from path_planning import lane_aware_centerline_path  # noqa: E402
from render import CITYSCAPES_COLORS, create_bev, create_overlay  # noqa: E402
import render_trajectories as rt  # noqa: E402
import seg_fast  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("eval_dir", help="Path to a Caddy-Training-Data-* directory")
    p.add_argument("--video", default="front-wide.mp4",
                   help="Which video inside eval_dir to use")
    p.add_argument("--output", default="/tmp/seg_eval.mp4")
    p.add_argument("--model", default="b0", choices=("b0", "b2", "b5"))
    p.add_argument("--seg-repo", default="/home/caddy/drive-by-segmentation")
    p.add_argument("--device", default=None)
    p.add_argument("--duration", type=float, default=30.0,
                   help="Seconds of video to process (real-time clock, not output fps).")
    p.add_argument("--infer-hz", type=float, default=10.0,
                   help="Inference rate; mimics the live pipeline.")
    p.add_argument("--output-fps", type=float, default=10.0)
    p.add_argument("--speed-mph", type=float, default=8.0,
                   help="Constant ego speed assumed for the adaptive lookahead.")
    p.add_argument("--max-frames", type=int, default=0,
                   help="Hard cap on frames processed (0 = no cap).")
    p.add_argument("--slow", action="store_true",
                   help="Use the original create_bev + lane_aware_centerline_path "
                        "(scipy-heavy reference path) instead of seg_fast.")
    return p.parse_args()


def pick_device(arg):
    if arg:
        return arg
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def draw_steering_wheel(canvas, cx, cy, r, deg, label, color):
    cv2.circle(canvas, (cx, cy), r, color, 2, cv2.LINE_AA)
    cv2.circle(canvas, (cx, cy), 4, color, -1, cv2.LINE_AA)
    rad = math.radians(deg)
    tip = (int(cx + r * math.sin(rad)), int(cy - r * math.cos(rad)))
    cv2.line(canvas, (cx, cy), tip, color, 3, cv2.LINE_AA)
    cv2.putText(canvas, f"{label} {deg:+.1f}°", (cx - r, cy + r + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)


def local_to_bev_px(local_fwd, local_left, bev_size, range_fwd, range_side):
    bx = (np.asarray(local_left) / range_side * 0.5 + 0.5) * bev_size
    by = (1 - np.asarray(local_fwd) / range_fwd) * bev_size
    return bx.astype(np.int32), by.astype(np.int32)


def compute_decomposition(lane_local):
    """Re-run the windowed polynomial fit so we can show coefficients.

    Mirrors `lookahead_heading_steering_deg` exactly.
    """
    if lane_local is None or len(lane_local) < 6:
        return None
    pts = np.asarray(lane_local, dtype=np.float64)
    pts = pts[np.isfinite(pts).all(axis=1)]
    pts = pts[(pts[:, 0] > si.STEER_FIT_MIN_M) & (pts[:, 0] < si.STEER_FIT_MAX_M)]
    if len(pts) < 6:
        return None
    x, y = pts[:, 0], pts[:, 1]
    weights = np.exp(-x / 2.5)
    try:
        a2, a1, a0 = np.polyfit(x, y, deg=2, w=weights)
    except (np.linalg.LinAlgError, ValueError):
        return None
    return float(a0), float(a1), float(a2)


def reconstruct_poly_xy(a0, a1, a2, max_fwd):
    xs = np.linspace(0.0, max_fwd, 40)
    ys = a0 + a1 * xs + a2 * xs * xs
    return xs, ys


def load_control_log(path: Path):
    """Return (rel_t array, column_deg_actual array, steer_cmd_deg array).

    Falls back to (None, None, None) if the file is missing or unreadable.
    """
    if not path.exists():
        return None, None, None
    import json as _json
    rel_ts: list[float] = []
    actual: list[float] = []
    cmd: list[float] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = _json.loads(line)
            except _json.JSONDecodeError:
                continue
            if "rel_t" not in row:
                continue
            rel_ts.append(float(row["rel_t"]))
            actual.append(float(row.get("column_deg_actual") or 0.0))
            cmd.append(float(row.get("steer_deg") or 0.0))
    if not rel_ts:
        return None, None, None
    return np.array(rel_ts), np.array(actual), np.array(cmd)


def lookup_at(rel_ts, values, t):
    """Nearest-neighbor lookup in a sorted-by-time array."""
    if rel_ts is None or len(rel_ts) == 0:
        return None
    i = int(np.searchsorted(rel_ts, t))
    if i <= 0:
        return float(values[0])
    if i >= len(rel_ts):
        return float(values[-1])
    # nearest of i-1, i
    if abs(rel_ts[i] - t) < abs(rel_ts[i - 1] - t):
        return float(values[i])
    return float(values[i - 1])


def main():
    args = parse_args()
    eval_dir = Path(args.eval_dir).expanduser()
    video_path = eval_dir / args.video
    if not video_path.exists():
        sys.exit(f"video not found: {video_path}")

    seg_repo = Path(args.seg_repo)
    import json as _json
    with (seg_repo / "camera_calibration.json").open() as f:
        calib = _json.load(f)

    bev_range = calib.get("bev_range", {})
    rt.RANGE_FWD = bev_range.get("forward_ft", 50) * rt.FT_TO_M
    rt.RANGE_SIDE = bev_range.get("side_ft", 25) * rt.FT_TO_M
    road_width_ft = float(calib.get("road_width_ft", 20.0))

    device = pick_device(args.device)
    proc, model = seg_live.load_segformer(args.model, device)

    ctrl_rel_t, ctrl_actual, ctrl_cmd = load_control_log(eval_dir / "control.jsonl")
    if ctrl_rel_t is None:
        print("[eval] no control.jsonl found — ground-truth wheel disabled")
    else:
        print(f"[eval] control.jsonl: {len(ctrl_rel_t)} samples over "
              f"{ctrl_rel_t[-1] - ctrl_rel_t[0]:.1f}s")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        sys.exit(f"failed to open {video_path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    # Per-frame stride to mimic the live infer rate.
    stride = max(1, int(round(src_fps / args.infer_hz)))
    max_frames = int(args.duration * args.infer_hz)
    if args.max_frames > 0:
        max_frames = min(max_frames, args.max_frames)

    palette = np.array(CITYSCAPES_COLORS, dtype=np.uint8)
    road_color = np.array(CITYSCAPES_COLORS[0], dtype=np.uint8)
    grid_color = np.clip(road_color.astype(np.int16) + 35, 0, 255).astype(np.uint8)
    grid2_color = np.clip(road_color.astype(np.int16) + 70, 0, 255).astype(np.uint8)

    # Probe a frame to learn the source image size for the BEV precompute.
    probe_ok, probe_bgr = cap.read()
    if not probe_ok:
        sys.exit("could not read first frame")
    img_h, img_w = probe_bgr.shape[:2]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    bev_remap = None
    if not args.slow:
        t0 = time.perf_counter()
        bev_remap = seg_fast.build_bev_remap(calib, img_h, img_w, rt.BEV_SIZE)
        print(f"[eval] built BEV remap in {(time.perf_counter() - t0)*1000:.1f} ms "
              f"({img_w}x{img_h} -> {rt.BEV_SIZE}x{rt.BEV_SIZE})")

    si.reset_lookahead_heading_filter()

    out_writer = None
    canvas_size = None

    src_frame_idx = 0
    out_frame_idx = 0
    steer_filtered = 0.0
    t_start = time.perf_counter()
    contrib_log = {"ct": [], "hd": [], "cv": [], "cmd": [], "actual": []}

    print(
        f"[eval] {video_path.name}: src_fps={src_fps:.1f} stride={stride} "
        f"target_frames={max_frames}"
    )
    print(
        f"[eval] gains: K_y={si.STEER_K_CROSSTRACK} K_h={si.STEER_K_HEADING} "
        f"K_kappa={si.STEER_K_CURVATURE} | heading_EMA={si.LOOKAHEAD_HEADING_EMA} "
        f"steer_EMA={si.STEERING_EMA}"
    )

    while out_frame_idx < max_frames:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if src_frame_idx % stride != 0:
            src_frame_idx += 1
            continue
        src_frame_idx += 1

        t_frame = time.perf_counter()
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        seg_map, _ = si.timed_segment_frame(frame_rgb, proc, model, device)
        if bev_remap is not None:
            bev_rgb = seg_fast.create_bev_cached(seg_map, palette, bev_remap)
        else:
            bev_rgb = create_bev(seg_map, calib, rt.BEV_SIZE)

        road_mask = (
            np.all(bev_rgb == road_color, axis=-1)
            | np.all(bev_rgb == grid_color, axis=-1)
            | np.all(bev_rgb == grid2_color, axis=-1)
        )

        try:
            if args.slow:
                lane_traj, lane_local = lane_aware_centerline_path(
                    road_mask,
                    bev_size=rt.BEV_SIZE,
                    range_fwd=rt.RANGE_FWD,
                    range_side=rt.RANGE_SIDE,
                    road_mask=road_mask,
                    road_width_ft=road_width_ft,
                )
            else:
                lane_traj, lane_local = seg_fast.lane_aware_centerline_path_fast(
                    road_mask,
                    bev_size=rt.BEV_SIZE,
                    range_fwd=rt.RANGE_FWD,
                    range_side=rt.RANGE_SIDE,
                )
        except Exception as exc:  # noqa: BLE001
            lane_traj, lane_local = None, None
            print(f"[eval] frame {out_frame_idx}: planner failed ({exc!r})")

        lookahead_m = si.adaptive_lookahead_m(args.speed_mph, True)
        steer_base_raw = si.lookahead_heading_steering_deg(lane_local, lookahead_m)
        if steer_base_raw is None:
            steer_base = steer_filtered  # hold previous
            valid = False
        else:
            # ps5_drive applies AUTOSTEER_SIGN = -1 to the segmentation
            # output, and segmentation_infer's STEERING_SIGN is also -1; the
            # two cancel, so what the column motor ultimately receives is
            # just steer_base_raw. Comparable to `column_deg_actual` directly.
            steer_base = steer_base_raw
            valid = True
        steer_filtered = (
            si.STEERING_EMA * steer_base + (1.0 - si.STEERING_EMA) * steer_filtered
        )
        steer_cmd = float(np.clip(steer_filtered, -270.0, 270.0))

        decomp = compute_decomposition(lane_local)

        overlay_rgb = create_overlay(frame_rgb, seg_map)
        overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
        bev_bgr = cv2.cvtColor(bev_rgb, cv2.COLOR_RGB2BGR)

        # Draw planner lane points
        if lane_traj is not None and len(lane_traj) > 1:
            pts = np.asarray(lane_traj, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(bev_bgr, [pts], False, (0, 255, 255), 2, cv2.LINE_AA)

        # Draw the polynomial reconstruction in BEV pixels (magenta).
        if decomp is not None and lane_local is not None and len(lane_local) >= 6:
            a0, a1, a2 = decomp
            max_fwd = float(np.nanmax(lane_local[:, 0]))
            xs, ys = reconstruct_poly_xy(a0, a1, a2, max_fwd)
            bx, by = local_to_bev_px(xs, ys, rt.BEV_SIZE, rt.RANGE_FWD, rt.RANGE_SIDE)
            valid_px = (bx >= 0) & (bx < rt.BEV_SIZE) & (by >= 0) & (by < rt.BEV_SIZE)
            if np.any(valid_px):
                poly_pts = np.stack([bx[valid_px], by[valid_px]], axis=-1).reshape(-1, 1, 2)
                cv2.polylines(bev_bgr, [poly_pts], False, (255, 0, 255), 2, cv2.LINE_AA)

            # Lookahead-distance ring on the polynomial.
            xs_la = np.array([lookahead_m])
            ys_la = a0 + a1 * xs_la + a2 * xs_la * xs_la
            bx_la, by_la = local_to_bev_px(xs_la, ys_la, rt.BEV_SIZE, rt.RANGE_FWD, rt.RANGE_SIDE)
            if 0 <= bx_la[0] < rt.BEV_SIZE and 0 <= by_la[0] < rt.BEV_SIZE:
                cv2.circle(bev_bgr, (int(bx_la[0]), int(by_la[0])), 6,
                           (255, 0, 255), 2, cv2.LINE_AA)

        # Compose canvas: overlay (640) | BEV (480) above; panel below with wheel + text.
        ov_h = 480
        ov_w = 640
        bv_h = 480
        bv_w = 480
        panel_h = 180
        if canvas_size is None:
            canvas_size = (ov_w + bv_w, ov_h + panel_h)
            out_writer = cv2.VideoWriter(
                args.output,
                cv2.VideoWriter_fourcc(*"mp4v"),
                args.output_fps,
                canvas_size,
            )
        ov = cv2.resize(overlay_bgr, (ov_w, ov_h))
        bv = cv2.resize(bev_bgr, (bv_w, bv_h))
        canvas = np.zeros((canvas_size[1], canvas_size[0], 3), dtype=np.uint8)
        canvas[:ov_h, :ov_w] = ov
        canvas[:bv_h, ov_w:ov_w + bv_w] = bv

        cv2.putText(canvas, "Seg overlay", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, "BEV  (yellow=planner, magenta=poly fit)", (ov_w + 10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Source frame's relative time for control-log lookup.
        rel_t = (src_frame_idx - 1) / src_fps if src_fps > 0 else out_frame_idx / args.output_fps
        gt_actual = lookup_at(ctrl_rel_t, ctrl_actual, rel_t) if ctrl_rel_t is not None else None
        gt_cmd = lookup_at(ctrl_rel_t, ctrl_cmd, rel_t) if ctrl_rel_t is not None else None

        # Bottom panel — two wheels: predicted (left of center) and ground truth (right of center).
        wheel_r = 50
        wheel_y = ov_h + panel_h // 2
        cx_pred = ov_w // 2 - 90
        cx_gt = ov_w // 2 + 90
        draw_steering_wheel(canvas, cx_pred, wheel_y, wheel_r, steer_cmd,
                            "pred" if valid else "pred(hold)",
                            (90, 200, 255) if valid else (90, 120, 200))
        if gt_actual is not None:
            draw_steering_wheel(canvas, cx_gt, wheel_y, wheel_r, gt_actual,
                                "actual", (120, 220, 140))
            delta = steer_cmd - gt_actual
            cv2.putText(canvas, f"Δ {delta:+.1f}°",
                        (ov_w // 2 - 30, ov_h + panel_h - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (220, 220, 220), 1, cv2.LINE_AA)
        else:
            cv2.putText(canvas, "(no GT)",
                        (cx_gt - 30, wheel_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (130, 130, 130), 1, cv2.LINE_AA)

        text_x = ov_w + 20
        text_y = ov_h + 30

        def line(text, dy=22, color=(220, 220, 220)):
            nonlocal text_y
            cv2.putText(canvas, text, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            text_y += dy

        line(f"frame {out_frame_idx + 1}/{max_frames}    t={rel_t:.2f}s    v={args.speed_mph:.1f} mph")
        line(f"lookahead {lookahead_m:.2f} m    pred {steer_cmd:+.1f}° col")
        if gt_actual is not None:
            line(f"actual {gt_actual:+.1f}°   gt_cmd {gt_cmd:+.1f}°", color=(120, 220, 140))
        if decomp is not None:
            a0, a1, a2 = decomp
            y_lh = a0 + a1 * lookahead_m + a2 * lookahead_m * lookahead_m
            yp_lh = a1 + 2 * a2 * lookahead_m
            kappa = 2 * a2
            line(f"y(L) = {y_lh:+.3f} m   (offset at lookahead)")
            line(f"y'(L) = {yp_lh:+.3f}  ({math.degrees(math.atan(yp_lh)):+.1f}° heading)")
            line(f"κ   = {kappa:+.4f} 1/m")
            ct = math.degrees(si.STEER_K_CROSSTRACK * y_lh) * si.STEERING_COLUMN_RATIO
            hd = math.degrees(si.STEER_K_HEADING * yp_lh) * si.STEERING_COLUMN_RATIO
            cv = math.degrees(si.STEER_K_CURVATURE * kappa * lookahead_m) * si.STEERING_COLUMN_RATIO
            line(f"contrib (col°):  Ky·y={ct:+.1f}", color=(200, 220, 255))
            line(f"                Kh·y'={hd:+.1f}", color=(200, 220, 255))
            line(f"                Kκ·κ·L={cv:+.1f}", color=(200, 220, 255))
            contrib_log["ct"].append(ct)
            contrib_log["hd"].append(hd)
            contrib_log["cv"].append(cv)
            contrib_log["cmd"].append(steer_cmd)
            if gt_actual is not None:
                contrib_log["actual"].append(gt_actual)
        else:
            line("(no lane fit)", color=(180, 120, 120))

        out_writer.write(canvas)
        out_frame_idx += 1

        if out_frame_idx % 20 == 0:
            print(f"[eval] processed {out_frame_idx}/{max_frames} frames "
                  f"({(time.perf_counter() - t_frame)*1000:.0f} ms last)")

    cap.release()
    if out_writer is not None:
        out_writer.release()
    elapsed = time.perf_counter() - t_start
    print(f"[eval] done: {out_frame_idx} frames in {elapsed:.1f}s -> {args.output}")

    def _stats(name, arr):
        if not arr:
            print(f"  {name}: (no samples)")
            return
        a = np.asarray(arr)
        print(f"  {name:8s}  mean={a.mean():+7.1f}  std={a.std():6.1f}  "
              f"|p50|={np.median(np.abs(a)):6.1f}  |p95|={np.percentile(np.abs(a), 95):6.1f}  "
              f"max|x|={np.max(np.abs(a)):6.1f}")

    print("[eval] per-term contribution stats (column degrees):")
    _stats("Ky·a0", contrib_log["ct"])
    _stats("Kh·a1", contrib_log["hd"])
    _stats("Kκ·κ·L", contrib_log["cv"])
    _stats("pred", contrib_log["cmd"])
    _stats("actual", contrib_log["actual"])


if __name__ == "__main__":
    main()

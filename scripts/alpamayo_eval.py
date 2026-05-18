#!/usr/bin/env python3
"""
alpamayo_eval.py — offline replay of recorded eval data through the live
Modal Alpamayo-R1 tunnel.

Reads the four-camera + ego.jsonl bundle written by
``scripts/record_cameras.py``, paces frames into the deployed
alpamayo-live-demo Modal app, collects predictions, and renders a
single-tile MP4 of the front-wide camera with steering + trajectory
overlays.

Important: recordings made before the camera-orientation fix used the
old device-name layout. To make the right physical view land in each
Alpamayo channel, the eval script remaps:

  Alpamayo channel 0 (front_wide)   ← cross-left.mp4   (physical wide)
  Alpamayo channel 1 (front_tele)   ← front-narrow.mp4 (physical narrow)
  Alpamayo channel 2 (cross_left)   ← front-wide.mp4   (physical left)
  Alpamayo channel 3 (cross_right)  ← cross-right.mp4  (physical right)

The narrow lens was already flipped 180° at record time (record_cameras
applied flip=-1 to "front-narrow"), so no extra flip is needed here.
The three 170° fisheye lenses are center-cropped 0.706× to ~120° to
match Alpamayo's training distribution, identical to the live cart
pipeline.

Usage:
  /home/caddy/mayo/.venv-client/bin/python scripts/alpamayo_eval.py \
      ego_evals/Caddy-Training-Data-2026-05-03_16-15-24 \
      --output /tmp/alpamayo_eval.mp4 \
      --fps 5 \
      --duration 30
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import struct
import sys
import time
import uuid
from pathlib import Path

import cv2
import msgpack
import numpy as np

try:
    import modal
except ImportError as e:
    print(f"ERROR: modal SDK required: {e}", file=sys.stderr)
    sys.exit(2)


APP_NAME = "alpamayo-live-demo"
CLS_NAME = "LiveInference"
HORIZON_S = 6.4
EGO_HISTORY_LEN = 16
JPEG_QUALITY = 60

# Channel index → (filename, do_fov_crop). The recordings are named
# per the physical view they hold (record_cameras.py writes one .mp4
# per camera slug), so the mapping is the obvious 1:1.
FILE_FOR_CHANNEL = [
    ("front-wide.mp4",   True),   # 0: front_wide
    ("front-narrow.mp4", False),  # 1: front_tele (already flipped at record time)
    ("cross-left.mp4",   True),   # 2: cross_left
    ("cross-right.mp4",  True),   # 3: cross_right
]
FOV_CROP_RATIO = 0.706   # 170° → 120°, same as live cart


def center_crop_zoom(frame: np.ndarray, ratio: float) -> np.ndarray:
    if ratio >= 1.0 or ratio <= 0.0:
        return frame
    h, w = frame.shape[:2]
    nh = max(2, int(round(h * ratio)))
    nw = max(2, int(round(w * ratio)))
    y0 = (h - nh) // 2
    x0 = (w - nw) // 2
    return cv2.resize(frame[y0:y0+nh, x0:x0+nw], (w, h),
                      interpolation=cv2.INTER_LINEAR)


# ── ego history builder (mirrors alpamayo_infer.build_ego_tensors but
#    sources samples from ego.jsonl instead of the live EgoSensor) ──────

def load_control_samples(jsonl_path: Path) -> list:
    """Returns sorted list of (rel_t, steer_deg, mph). Empty if file missing
    or only header rows."""
    rows = []
    if not jsonl_path.exists():
        return rows
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("_schema"):
                continue
            t = d.get("rel_t")
            if t is None:
                continue
            rows.append((float(t),
                         float(d.get("steer_deg", 0.0)),
                         float(d.get("mph", 0.0))))
    rows.sort(key=lambda r: r[0])
    return rows


def lookup_control_at(rows: list, t: float) -> "tuple[float, float] | None":
    """Linear scan (good enough for ~12k samples) to find the sample
    closest in time to ``t``. Returns (steer_deg, mph) or None."""
    if not rows:
        return None
    # Binary search would be cleaner; the per-frame cost here is trivial.
    best_dt = float("inf")
    best = None
    for rt, sd, mph in rows:
        dt = abs(rt - t)
        if dt < best_dt:
            best_dt = dt
            best = (sd, mph)
        elif rt > t and best_dt < 0.5:
            break  # rows are sorted; we're moving away from optimum
    return best


def load_ego_samples(jsonl_path: Path) -> list:
    """Returns a list of (rel_t, alpamayo_xyz [3], alpamayo_yaw_rad)."""
    samples = []
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            d = json.loads(line)
            if d.get("_schema"):
                continue  # header row
            alp = d.get("alpamayo")
            if not alp:
                continue
            xyz = alp.get("xyz_m")
            yaw = alp.get("yaw_rad")
            if xyz is None or yaw is None:
                continue
            samples.append((float(d.get("rel_t", 0.0)),
                            np.array(xyz, dtype=np.float32),
                            float(yaw)))
    return samples


def build_ego_for_t(samples: list, t_now: float) -> "tuple[np.ndarray, np.ndarray] | None":
    """Build (xyz_local [16,3], rot_local [16,3,3]) recentred to t_now,
    matching alpamayo_infer.build_ego_tensors output."""
    if not samples:
        return None
    # Take all samples up to and including t_now.
    past = [s for s in samples if s[0] <= t_now]
    if not past:
        past = [samples[0]]
    if len(past) < EGO_HISTORY_LEN:
        past = [past[0]] * (EGO_HISTORY_LEN - len(past)) + past
    past = past[-EGO_HISTORY_LEN:]
    xyz = np.stack([s[1] for s in past])  # [16, 3]
    yaw = np.array([s[2] for s in past], dtype=np.float32)
    t0_xyz = xyz[-1].copy()
    t0_yaw = float(yaw[-1])
    c0, s0 = math.cos(-t0_yaw), math.sin(-t0_yaw)
    R0_inv = np.array([[c0, -s0, 0.0], [s0, c0, 0.0], [0.0, 0.0, 1.0]],
                      dtype=np.float32)
    delta = xyz - t0_xyz
    xyz_local = (delta @ R0_inv.T).astype(np.float32)
    rot_local = np.zeros((EGO_HISTORY_LEN, 3, 3), dtype=np.float32)
    for i in range(EGO_HISTORY_LEN):
        d = float(yaw[i] - t0_yaw)
        c, s = math.cos(d), math.sin(d)
        rot_local[i] = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    return xyz_local, rot_local


# ── Tunnel framing ─────────────────────────────────────────────────────

async def read_frame(reader: asyncio.StreamReader) -> bytes:
    hdr = await reader.readexactly(4)
    n = struct.unpack(">I", hdr)[0]
    return await reader.readexactly(n)


async def write_frame(writer: asyncio.StreamWriter, payload: bytes) -> None:
    writer.write(struct.pack(">I", len(payload)) + payload)
    await writer.drain()


# ── Render helpers ─────────────────────────────────────────────────────

def render_bev_inset(pred_xy: np.ndarray, w: int = 320, h: int = 320,
                     half_range_m: float = 12.0) -> np.ndarray:
    img = np.full((h, w, 3), 18, dtype=np.uint8)
    cx, cy = w // 2, h // 2
    pxm = min(w, h) / (2.0 * half_range_m)

    def to_px(xf, yl):
        return int(round(cx - yl * pxm)), int(round(cy - xf * pxm))

    for m in range(-int(half_range_m), int(half_range_m) + 1):
        col = (60, 60, 60) if m % 5 else (90, 90, 90)
        u_left, _ = to_px(half_range_m, m)
        u_right, _ = to_px(-half_range_m, m)
        cv2.line(img, (u_left, 0), (u_right, h-1), col, 1, cv2.LINE_AA)
        _, v = to_px(m, half_range_m)
        cv2.line(img, (0, v), (w-1, v), col, 1, cv2.LINE_AA)
    cv2.line(img, (cx, 0), (cx, h-1), (140, 140, 140), 1, cv2.LINE_AA)
    cv2.line(img, (0, cy), (w-1, cy), (140, 140, 140), 1, cv2.LINE_AA)

    # Ego triangle
    pts = np.array([[cx, cy-10], [cx-7, cy+7], [cx+7, cy+7]], dtype=np.int32)
    cv2.fillPoly(img, [pts], (255, 255, 255))

    if pred_xy is not None and pred_xy.shape[0] >= 2:
        path = [to_px(float(p[0]), float(p[1])) for p in pred_xy]
        cv2.polylines(img, [np.array(path, dtype=np.int32)], False,
                      (255, 196, 80), 3, cv2.LINE_AA)
        cv2.circle(img, path[-1], 5, (255, 220, 100), -1, cv2.LINE_AA)
    cv2.putText(img, "BEV  x=fwd  y=left  m", (8, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1, cv2.LINE_AA)
    return img


def trajectory_steer_deg(pred_xy: np.ndarray) -> float:
    """Tangent angle of trajectory at t=0 (first segment)."""
    if pred_xy is None or pred_xy.shape[0] < 2:
        return 0.0
    dx = float(pred_xy[1, 0] - pred_xy[0, 0])
    dy = float(pred_xy[1, 1] - pred_xy[0, 1])
    if dx*dx + dy*dy < 1e-6:
        return 0.0
    return math.degrees(math.atan2(dy, dx))


def trajectory_speed_mph(pred_xy: np.ndarray) -> float:
    if pred_xy is None or pred_xy.shape[0] < 2:
        return 0.0
    T = pred_xy.shape[0]
    dt = HORIZON_S / max(T - 1, 1)
    k = min(3, T - 1)
    seg = pred_xy[:k+1]
    dx = float(seg[-1, 0] - seg[0, 0])
    dy = float(seg[-1, 1] - seg[0, 1])
    speed_mps = math.hypot(dx, dy) / max(k * dt, 1e-3)
    if dx <= 0.05:
        speed_mps = 0.0
    return speed_mps * 2.23694


def render_output_frame(wide_frame: np.ndarray, pred: "np.ndarray | None",
                        meta: dict, out_w: int, out_h: int) -> np.ndarray:
    """Compose the final output: scaled wide camera + side panel."""
    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    # Main camera fills left ~75%
    main_w = int(out_w * 0.75)
    main_h = out_h
    main = cv2.resize(wide_frame, (main_w, main_h))
    canvas[:, :main_w] = main

    panel_x = main_w
    panel_w = out_w - main_w
    canvas[:, panel_x:panel_x+8] = (40, 40, 40)  # divider

    # BEV inset top of panel
    bev = render_bev_inset(pred, w=panel_w-16, h=panel_w-16)
    canvas[16:16+bev.shape[0], panel_x+8:panel_x+8+bev.shape[1]] = bev

    # Stats below BEV
    y = 16 + bev.shape[0] + 28
    def put(line, color=(220, 220, 220), size=0.55):
        nonlocal y
        cv2.putText(canvas, line, (panel_x + 14, y),
                    cv2.FONT_HERSHEY_SIMPLEX, size, color, 1, cv2.LINE_AA)
        y += int(28 * (size / 0.55))

    put("ALPAMAYO-R1", (255, 196, 80), 0.7)
    put(f"region: {meta.get('region', '?')}", (170, 170, 170), 0.5)
    put(f"src_t:  {meta.get('rel_t', 0):6.2f} s", (170, 170, 170), 0.5)
    put("", size=0.3)
    pred_steer = trajectory_steer_deg(pred) if pred is not None else None
    pred_speed = trajectory_speed_mph(pred) if pred is not None else None
    gt_steer = meta.get("gt_steer_deg")
    gt_speed = meta.get("gt_mph")
    if pred_steer is not None:
        put(f"pred steer: {pred_steer:+6.1f}°", (255, 196, 80), 0.6)
        put(f"pred speed: {pred_speed:5.1f} mph", (255, 196, 80), 0.55)
    else:
        put("pred steer: —", (170, 170, 170), 0.55)
    if gt_steer is not None:
        put(f"gt   steer: {gt_steer:+6.1f}°", (120, 220, 120), 0.6)
        put(f"gt   speed: {gt_speed:5.1f} mph", (120, 220, 120), 0.55)
    else:
        put("gt   steer: —", (170, 170, 170), 0.55)
    if pred_steer is not None and gt_steer is not None:
        put(f"Δ steer:    {pred_steer - gt_steer:+6.1f}°", (200, 200, 255), 0.55)
    put("", size=0.3)
    gpu = meta.get("gpu_ms", 0)
    rtt = meta.get("rtt_ms", 0)
    put(f"gpu:    {gpu:5.0f} ms", (170, 170, 170), 0.5)
    put(f"rtt:    {rtt:5.0f} ms", (170, 170, 170), 0.5)
    put(f"hz:     {meta.get('hz', 0):5.2f}", (170, 170, 170), 0.5)
    put(f"recv:   {meta.get('recv_count', 0):5d}", (170, 170, 170), 0.5)

    if meta.get("warming"):
        cv2.putText(canvas, f"WARMING ({meta.get('buffer', 0)}/4)",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (80, 200, 255), 2, cv2.LINE_AA)
    return canvas


# ── Main ───────────────────────────────────────────────────────────────

async def run(args) -> int:
    eval_dir = Path(args.eval_dir)
    if not eval_dir.exists():
        print(f"ERROR: eval dir not found: {eval_dir}", file=sys.stderr)
        return 1

    # Open the four cv2.VideoCapture in the order Alpamayo expects.
    caps = []
    for fn, _ in FILE_FOR_CHANNEL:
        p = eval_dir / fn
        c = cv2.VideoCapture(str(p))
        if not c.isOpened():
            print(f"ERROR: failed to open {p}", file=sys.stderr)
            return 1
        caps.append(c)
    src_fps = caps[0].get(cv2.CAP_PROP_FPS) or 30.0
    n_total = int(caps[0].get(cv2.CAP_PROP_FRAME_COUNT))
    src_duration = n_total / src_fps
    print(f"[eval] source: {n_total} frames @ {src_fps:.1f} fps "
          f"({src_duration:.1f}s)")

    # Frame stride = source_fps / target_fps
    stride = max(1, int(round(src_fps / args.fps)))
    duration_s = min(args.duration, src_duration) if args.duration > 0 else src_duration
    last_frame_idx = min(n_total - 1, int(duration_s * src_fps))
    frame_indices = list(range(0, last_frame_idx + 1, stride))
    print(f"[eval] target {args.fps} Hz × {duration_s:.1f}s "
          f"= {len(frame_indices)} inference cycles")

    # Load ego samples.
    ego_path = eval_dir / "ego.jsonl"
    ego_samples = load_ego_samples(ego_path) if ego_path.exists() else []
    print(f"[eval] ego samples: {len(ego_samples)}")

    # Load control (ground-truth steering + speed).
    control_path = eval_dir / "control.jsonl"
    control_samples = load_control_samples(control_path)
    print(f"[eval] control samples: {len(control_samples)}")

    # Output writer (created after we know server H/W).
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Spawn Modal call & open tunnel ─────────────────────────────────
    service = modal.Cls.from_name(APP_NAME, CLS_NAME)
    addr_q = modal.Queue.from_name(f"{APP_NAME}-tunnel-addrs",
                                   create_if_missing=True)
    rid = uuid.uuid4().hex
    print(f"[tunnel] spawning {APP_NAME}.{CLS_NAME} (req={rid[:8]}…)")
    fc = await service().call.spawn.aio(rid)
    try:
        addr = await asyncio.wait_for(addr_q.get.aio(partition=rid),
                                       timeout=600.0)
        host, port_text = addr.split(":", maxsplit=1)
        port = int(port_text)
        print(f"[tunnel] container at {addr}; opening TCP …")
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port), timeout=30.0)
        try:
            return await _stream_eval(reader, writer, caps, ego_samples,
                                       control_samples, frame_indices,
                                       src_fps, out_path, args)
        finally:
            try:
                bye = msgpack.packb({"bye": True})
                await write_frame(writer, bye)
                await asyncio.wait_for(read_frame(reader), timeout=5.0)
            except Exception:
                pass
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass
    finally:
        try:
            await fc.cancel.aio()
        except Exception:
            pass
        for c in caps:
            c.release()


async def _stream_eval(reader, writer, caps, ego_samples, control_samples,
                       frame_indices, src_fps, out_path, args) -> int:
    # Read hello first (server always sends it on connect).
    hello_raw = await asyncio.wait_for(read_frame(reader), timeout=30.0)
    hello = msgpack.unpackb(hello_raw, raw=False)
    if not hello.get("hello"):
        print(f"[tunnel] WARN: expected hello, got {hello}", file=sys.stderr)
    H, W = int(hello.get("H", 1080)), int(hello.get("W", 1920))
    region = hello.get("region", "?")
    print(f"[tunnel] hello: server expects {W}x{H}, region={region}")

    # Output writer setup.
    out_w, out_h = 1280, 720
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(out_path), fourcc, args.fps, (out_w, out_h))
    if not out.isOpened():
        print(f"ERROR: VideoWriter failed for {out_path}", file=sys.stderr)
        return 1

    seq = 0
    last_pred = None
    last_meta = {"region": region, "warming": True, "buffer": 0,
                 "gpu_ms": 0, "rtt_ms": 0, "hz": 0, "recv_count": 0}
    recv_times = []
    t_run0 = time.monotonic()

    for i, frame_idx in enumerate(frame_indices):
        # Pull source frames at this index, apply per-channel crop, resize.
        channel_frames = []
        wide_for_render = None
        for ch, ((_, do_crop), cap) in enumerate(zip(FILE_FOR_CHANNEL, caps)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                print(f"[eval] EOF at frame {frame_idx}", file=sys.stderr)
                break
            if do_crop:
                frame = center_crop_zoom(frame, FOV_CROP_RATIO)
            if ch == 0:  # front_wide is what we render in the output panel
                wide_for_render = frame.copy()
            if frame.shape[:2] != (H, W):
                frame = cv2.resize(frame, (W, H))
            channel_frames.append(frame)
        if len(channel_frames) != 4 or wide_for_render is None:
            break

        # Encode + msgpack + send.
        jpegs = []
        for f in channel_frames:
            ok, buf = cv2.imencode(".jpg", f,
                                    [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            if not ok:
                raise RuntimeError("imencode failed")
            jpegs.append(buf.tobytes())
        msg_out = {"jpegs": jpegs, "seq": seq}

        rel_t = frame_idx / src_fps
        gt = lookup_control_at(control_samples, rel_t)
        gt_steer_deg = gt[0] if gt else None
        gt_mph = gt[1] if gt else None
        ego = build_ego_for_t(ego_samples, rel_t)
        if ego is not None:
            xyz, rot = ego
            msg_out["ego_history_xyz"] = xyz.tobytes()
            msg_out["ego_history_rot"] = rot.tobytes()
            msg_out["ego_history_shape"] = list(xyz.shape)

        t_send = time.monotonic()
        await write_frame(writer, msgpack.packb(msg_out))
        seq_sent = seq
        seq += 1

        # Drain replies until we get a non-warming, non-hello message
        # for our seq. Server might burst warming replies for buffer fill.
        got_pred = False
        while not got_pred:
            raw = await read_frame(reader)
            reply = msgpack.unpackb(raw, raw=False)
            if reply.get("warming"):
                last_meta["warming"] = True
                last_meta["buffer"] = int(reply.get("buffer", 0))
                # Render a "warming" frame so the output video shows
                # what's happening during the buffer-fill phase.
                meta = dict(last_meta)
                meta["rel_t"] = rel_t
                meta["recv_count"] = 0
                meta["gt_steer_deg"] = gt_steer_deg
                meta["gt_mph"] = gt_mph
                out.write(render_output_frame(wide_for_render, None,
                                               meta, out_w, out_h))
                # The server sends one warming reply per send during fill;
                # break to send the next frame.
                break
            if reply.get("hello") or reply.get("bye_ack"):
                continue
            # Real prediction.
            shape = tuple(reply["pred_shape"])
            pred = np.frombuffer(reply["pred_xy"], dtype=np.float32).reshape(shape)
            now = time.monotonic()
            recv_times.append(now)
            if len(recv_times) > 20:
                recv_times = recv_times[-20:]
            hz = 0.0
            if len(recv_times) >= 2 and recv_times[-1] > recv_times[0]:
                hz = (len(recv_times) - 1) / (recv_times[-1] - recv_times[0])
            last_pred = pred
            last_meta.update({
                "warming": False,
                "rel_t": rel_t,
                "gpu_ms": float(reply.get("gpu_ms", 0)),
                "rtt_ms": (now - t_send) * 1000.0,
                "hz": hz,
                "recv_count": len(recv_times),
                "region": region,
                "gt_steer_deg": gt_steer_deg,
                "gt_mph": gt_mph,
            })
            out.write(render_output_frame(wide_for_render, last_pred,
                                           last_meta, out_w, out_h))
            got_pred = True

        # Compact periodic log so we can see progress.
        if i % 5 == 0 or i == len(frame_indices) - 1:
            elapsed = time.monotonic() - t_run0
            done = i + 1
            eta = (len(frame_indices) - done) * (elapsed / max(done, 1))
            print(f"[eval] {done}/{len(frame_indices)}  "
                  f"src_t={rel_t:6.2f}s  "
                  f"gpu={last_meta['gpu_ms']:5.0f}ms  "
                  f"rtt={last_meta['rtt_ms']:5.0f}ms  "
                  f"hz={last_meta['hz']:4.2f}  "
                  f"warm={'Y' if last_meta['warming'] else 'N'}  "
                  f"elapsed={elapsed:5.0f}s  eta={eta:5.0f}s",
                  flush=True)

    out.release()
    print(f"[eval] wrote {out_path}")
    return 0


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("eval_dir", help="Path to a Caddy-Training-Data-* dir")
    p.add_argument("--output", default="/tmp/alpamayo_eval.mp4",
                   help="Output MP4 path (default /tmp/alpamayo_eval.mp4)")
    p.add_argument("--fps", type=float, default=5.0,
                   help="Target inference / output framerate (default 5)")
    p.add_argument("--duration", type=float, default=30.0,
                   help="Seconds of source video to process. 0 = full clip "
                        "(careful: at ~5s RTT, full 240s clip would take "
                        "~100 min wall-clock).")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sys.exit(asyncio.run(run(args)))

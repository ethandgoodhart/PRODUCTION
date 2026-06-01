#!/usr/bin/env python3
"""Precompute CLRNet lane detections for a video via Modal.

Produces a JSON cache file consumed by segmentation_infer.py's CLRNetLaneCache.
Each frame entry contains a list of detected lanes with normalized [0,1] point
coordinates and confidence scores.

Usage:
    python3 scripts/precompute_clrnet_modal.py \
        --video path/to/front-wide.mp4 \
        --checkpoint /home/caddy/clrnet_weights/clrernet_culane_dla34_ema.pth \
        --cache-dir .cache/caddy_clrnet_cache
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import zlib
from pathlib import Path

import cv2
import modal


def video_meta(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open {path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    finally:
        cap.release()
    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_s": frame_count / fps if fps > 0 else 0.0,
    }


def cache_path(video: Path, cache_dir: Path, width: int, height: int) -> Path:
    st = video.stat()
    key_src = f"{video.resolve()}:{st.st_size}:{int(st.st_mtime_ns)}:clrnet_dla34:{width}x{height}"
    key = hashlib.sha1(key_src.encode("utf-8")).hexdigest()[:16]
    stem = f"{video.stem}.clrnet_dla34.{width}x{height}.{key}"
    return cache_dir / f"{stem}.json"


def main() -> None:
    p = argparse.ArgumentParser(description="Precompute CLRNet lanes on Modal into a local JSON cache.")
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="Path to CLRNet checkpoint .pth file (only needed on first run)")
    p.add_argument("--app-name", default="caddy-clrnet-remote")
    p.add_argument("--cache-dir", type=Path, default=Path(".cache/caddy_clrnet_cache"))
    p.add_argument("--chunk-frames", type=int, default=150)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    meta = video_meta(args.video)
    frame_count = int(meta["frame_count"])
    if frame_count <= 0:
        raise RuntimeError("video has no frames")

    out_path = cache_path(args.video, args.cache_dir, args.width, args.height)
    if not args.force and out_path.exists():
        with out_path.open() as f:
            existing = json.load(f)
        if existing.get("complete") and int(existing.get("frame_count", 0)) == frame_count:
            print(json.dumps({"cache": str(out_path), "complete": True}))
            return

    upload_video = modal.Function.from_name(args.app_name, "upload_video")
    upload_checkpoint_fn = modal.Function.from_name(args.app_name, "upload_checkpoint")
    detect_chunk = modal.Function.from_name(args.app_name, "detect_lanes_chunk")

    # Upload checkpoint if provided
    if args.checkpoint and args.checkpoint.exists():
        print(f"[precompute] uploading checkpoint {args.checkpoint} ({args.checkpoint.stat().st_size / 1e6:.1f} MB)...", flush=True)
        upload_checkpoint_fn.remote(args.checkpoint.read_bytes())
        print("[precompute] checkpoint uploaded", flush=True)

    # Upload video
    remote_name = f"{hashlib.sha1(str(args.video.resolve()).encode()).hexdigest()[:16]}-{args.video.name}"
    print(f"[precompute] uploading {args.video} ({args.video.stat().st_size / 1e6:.1f} MB) to Modal...", flush=True)
    upload_video.remote(remote_name, args.video.read_bytes())

    all_frames = []
    started = time.perf_counter()
    chunks_done = 0
    total_chunks = (frame_count + args.chunk_frames - 1) // args.chunk_frames

    for start in range(0, frame_count, max(1, args.chunk_frames)):
        want = min(args.chunk_frames, frame_count - start)
        t0 = time.perf_counter()
        result = detect_chunk.remote(
            remote_name,
            start,
            want,
            args.width,
            args.height,
        )
        got = int(result["num_frames"])
        raw = zlib.decompress(result["zlib"])
        frame_results = json.loads(raw)
        all_frames.extend(frame_results)
        chunks_done += 1
        elapsed = time.perf_counter() - t0
        timings = result.get("timings_ms", {})
        print(
            f"[precompute] chunk {chunks_done}/{total_chunks} "
            f"frames {start}..{start+got-1} "
            f"elapsed={elapsed:.1f}s "
            f"remote_load={timings.get('remote_load_ms', 0):.0f}ms "
            f"remote_infer={timings.get('remote_infer_ms', 0):.0f}ms",
            flush=True,
        )

    total_elapsed = time.perf_counter() - started
    cache_data = {
        "schema": "caddy.clrnet_cache.v1",
        "model": "clrernet_culane_dla34_ema",
        "frame_count": frame_count,
        "fps": meta["fps"],
        "duration_s": meta["duration_s"],
        "width": args.width,
        "height": args.height,
        "total_elapsed_s": round(total_elapsed, 3),
        "complete": True,
        "frames": all_frames,
    }

    with out_path.open("w") as f:
        json.dump(cache_data, f)
    size_mb = out_path.stat().st_size / 1e6
    print(
        f"[precompute] done: {out_path} ({size_mb:.1f} MB, "
        f"{frame_count} frames, {total_elapsed:.1f}s total)",
        flush=True,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
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
import numpy as np


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


def cache_paths(video: Path, cache_dir: Path, model: str, width: int, height: int) -> tuple[Path, Path]:
    st = video.stat()
    key_src = f"{video.resolve()}:{st.st_size}:{int(st.st_mtime_ns)}:{model}:{width}x{height}"
    key = hashlib.sha1(key_src.encode("utf-8")).hexdigest()[:16]
    stem = f"{video.stem}.{model}.{width}x{height}.{key}"
    return cache_dir / f"{stem}.uint8", cache_dir / f"{stem}.json"


def main() -> None:
    p = argparse.ArgumentParser(description="Precompute SegFormer maps on Modal into a local memmap cache.")
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--model", default="b0", choices=("b0", "b2", "b5"))
    p.add_argument("--app-name", default="caddy-segformer-remote")
    p.add_argument("--cache-dir", type=Path, default=Path("/tmp/caddy_segmentation_cache"))
    p.add_argument("--chunk-frames", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    meta = video_meta(args.video)
    frame_count = int(meta["frame_count"])
    if frame_count <= 0:
        raise RuntimeError("video has no frames")

    data_path, meta_path = cache_paths(args.video, args.cache_dir, args.model, args.width, args.height)
    if not args.force and data_path.exists() and meta_path.exists():
        with meta_path.open() as f:
            existing = json.load(f)
        if existing.get("complete") and int(existing.get("frame_count", 0)) == frame_count:
            print(json.dumps({"cache": str(data_path), "meta": str(meta_path), "complete": True}))
            return

    upload = modal.Function.from_name(args.app_name, "upload_video")
    segment_chunk = modal.Function.from_name(args.app_name, "segment_video_chunk")

    remote_video_name = f"{hashlib.sha1(str(args.video.resolve()).encode()).hexdigest()[:16]}-{args.video.name}"
    print(f"[precompute] uploading {args.video} ({args.video.stat().st_size / 1e6:.1f} MB) to Modal...", flush=True)
    upload.remote(remote_video_name, args.video.read_bytes())

    mmap = np.memmap(data_path, dtype=np.uint8, mode="w+", shape=(frame_count, args.height, args.width))
    complete_chunks: list[dict] = []
    started = time.perf_counter()
    try:
        for start in range(0, frame_count, max(1, args.chunk_frames)):
            want = min(args.chunk_frames, frame_count - start)
            t0 = time.perf_counter()
            result = segment_chunk.remote(
                remote_video_name,
                start,
                want,
                args.model,
                args.width,
                args.height,
                args.batch_size,
            )
            got = int(result["num_frames"])
            raw = zlib.decompress(result["zlib"])
            arr = np.frombuffer(raw, dtype=np.uint8).reshape((got, args.height, args.width))
            mmap[start:start + got] = arr
            mmap.flush()
            chunk = {
                "start_frame": start,
                "num_frames": got,
                "elapsed_s": round(time.perf_counter() - t0, 3),
                "timings_ms": result.get("timings_ms", {}),
            }
            complete_chunks.append(chunk)
            done = start + got
            print(
                f"[precompute] {done}/{frame_count} frames "
                f"({done / frame_count * 100:.1f}%) chunk_s={chunk['elapsed_s']:.1f}",
                flush=True,
            )
            if got < want:
                break
    finally:
        del mmap

    complete = sum(c["num_frames"] for c in complete_chunks) >= frame_count
    payload = {
        "schema": "caddy.segmentation_cache.v1",
        "complete": complete,
        "video": str(args.video.resolve()),
        "video_size": args.video.stat().st_size,
        "video_mtime_ns": args.video.stat().st_mtime_ns,
        "model": args.model,
        "app_name": args.app_name,
        "data_path": str(data_path),
        "frame_count": frame_count,
        "fps": meta["fps"],
        "duration_s": meta["duration_s"],
        "width": args.width,
        "height": args.height,
        "source_width": meta["width"],
        "source_height": meta["height"],
        "dtype": "uint8",
        "created_ts": time.time(),
        "elapsed_s": round(time.perf_counter() - started, 3),
        "chunks": complete_chunks,
    }
    tmp = meta_path.with_suffix(meta_path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, separators=(",", ":")))
    os.replace(tmp, meta_path)
    print(json.dumps({"cache": str(data_path), "meta": str(meta_path), "complete": complete}))


if __name__ == "__main__":
    main()

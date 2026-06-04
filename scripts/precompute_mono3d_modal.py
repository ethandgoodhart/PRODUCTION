#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
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


def load_intrinsics(calib_path: Path) -> dict:
    data = json.loads(calib_path.read_text())
    intr = data.get("intrinsics") or {}
    fx = float(intr.get("fx", intr.get("focal_length")))
    fy = float(intr.get("fy", intr.get("focal_length", fx)))
    return {
        "fx": fx,
        "fy": fy,
        "cx": float(intr["cx"]),
        "cy": float(intr["cy"]),
    }


def cache_path(video: Path, cache_dir: Path, score_thr: float, width: int, height: int) -> Path:
    st = video.stat()
    key_src = (
        f"{video.resolve()}:{st.st_size}:{int(st.st_mtime_ns)}:"
        f"fcos3d_nuscenes:{score_thr:g}:{width}x{height}"
    )
    key = hashlib.sha1(key_src.encode("utf-8")).hexdigest()[:16]
    return cache_dir / f"{video.stem}.fcos3d.{width}x{height}.conf{score_thr:g}.{key}.json"


def write_json_atomic(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, separators=(",", ":")))
    os.replace(tmp, path)


def main() -> None:
    p = argparse.ArgumentParser(description="Precompute Modal FCOS3D detections into a local cache.")
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--calib", type=Path, required=True)
    p.add_argument("--app-name", default="caddy-monocular3d-fcos3d")
    p.add_argument("--cache-dir", type=Path, default=Path(".cache/caddy_mono3d_cache"))
    p.add_argument("--chunk-frames", type=int, default=60)
    p.add_argument("--score-threshold", type=float, default=0.05)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    meta = video_meta(args.video)
    frame_count = int(meta["frame_count"])
    if frame_count <= 0:
        raise RuntimeError("video has no frames")
    intr = load_intrinsics(args.calib)

    out_path = cache_path(args.video, args.cache_dir, args.score_threshold, args.width, args.height)
    viz_dir = out_path.with_suffix("").with_name(out_path.with_suffix("").name + ".viz")
    if out_path.exists() and not args.force:
        existing = json.loads(out_path.read_text())
        if existing.get("complete") and int(existing.get("frame_count", 0)) == frame_count:
            print(json.dumps({"cache": str(out_path), "viz_dir": str(viz_dir), "complete": True}))
            return

    if args.force and out_path.exists():
        out_path.unlink()
    viz_dir.mkdir(parents=True, exist_ok=True)

    upload_video = modal.Function.from_name(args.app_name, "upload_video")
    detect_chunk = modal.Function.from_name(args.app_name, "detect_video_chunk")

    remote_name = f"{hashlib.sha1(str(args.video.resolve()).encode()).hexdigest()[:16]}-{args.video.name}"
    print(f"[mono3d] uploading {args.video} ({args.video.stat().st_size / 1e6:.1f} MB) to Modal...", flush=True)
    upload_video.remote(remote_name, args.video.read_bytes())

    frames: list[dict] = []
    started = time.perf_counter()
    total_chunks = (frame_count + max(1, args.chunk_frames) - 1) // max(1, args.chunk_frames)
    provider = "mmdet3d_fcos3d_nuscenes"

    for chunk_i, start in enumerate(range(0, frame_count, max(1, args.chunk_frames)), start=1):
        want = min(args.chunk_frames, frame_count - start)
        t0 = time.perf_counter()
        result = detect_chunk.remote(
            remote_name,
            start,
            want,
            intr["fx"],
            intr["fy"],
            intr["cx"],
            intr["cy"],
            args.score_threshold,
            args.width,
            args.height,
        )
        provider = str(result.get("provider") or provider)
        chunk_frames = json.loads(zlib.decompress(result["json_zlib"]).decode("utf-8"))
        for entry in chunk_frames:
            frame_index = int(entry["frame_index"])
            rel_viz = f"{frame_index:06d}.jpg"
            viz_bytes = base64.b64decode(entry.pop("viz_jpeg_b64"))
            (viz_dir / rel_viz).write_bytes(viz_bytes)
            frames.append({
                "frame_index": frame_index,
                "objects": entry.get("objects", []),
                "viz": rel_viz,
            })

        elapsed = time.perf_counter() - t0
        timings = result.get("timings_ms") or {}
        print(
            f"[mono3d] chunk {chunk_i}/{total_chunks} "
            f"frames {start}..{start + len(chunk_frames) - 1} "
            f"elapsed={elapsed:.1f}s remote_infer={timings.get('remote_infer_ms', 0):.0f}ms "
            f"objects={sum(len(f.get('objects', [])) for f in chunk_frames)}",
            flush=True,
        )

        partial = {
            "schema": "caddy.mono3d_cache.v1",
            "model": provider,
            "provider": provider,
            "score_threshold": float(args.score_threshold),
            "frame_count": frame_count,
            "fps": meta["fps"],
            "duration_s": meta["duration_s"],
            "width": int(args.width),
            "height": int(args.height),
            "video": str(args.video.resolve()),
            "video_size": args.video.stat().st_size,
            "video_mtime_ns": args.video.stat().st_mtime_ns,
            "viz_dir": viz_dir.name,
            "complete": False,
            "frames": frames,
        }
        write_json_atomic(out_path, partial)

    total_elapsed = time.perf_counter() - started
    payload = {
        "schema": "caddy.mono3d_cache.v1",
        "model": provider,
        "provider": provider,
        "score_threshold": float(args.score_threshold),
        "frame_count": frame_count,
        "fps": meta["fps"],
        "duration_s": meta["duration_s"],
        "width": int(args.width),
        "height": int(args.height),
        "video": str(args.video.resolve()),
        "video_size": args.video.stat().st_size,
        "video_mtime_ns": args.video.stat().st_mtime_ns,
        "created_ts": time.time(),
        "client_elapsed_s": round(total_elapsed, 3),
        "viz_dir": viz_dir.name,
        "complete": True,
        "frames": frames,
    }
    write_json_atomic(out_path, payload)
    print(json.dumps({
        "cache": str(out_path),
        "viz_dir": str(viz_dir),
        "complete": True,
        "frames": len(frames),
        "elapsed_s": round(total_elapsed, 3),
    }))


if __name__ == "__main__":
    main()

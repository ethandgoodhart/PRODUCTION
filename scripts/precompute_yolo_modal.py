#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import zlib
from pathlib import Path

import modal


def cache_path(video: Path, cache_dir: Path, model: str, imgsz: int, conf: float) -> Path:
    st = video.stat()
    key_src = f"{video.resolve()}:{st.st_size}:{st.st_mtime_ns}:{model}:{imgsz}:{conf}"
    key = hashlib.sha1(key_src.encode("utf-8")).hexdigest()[:16]
    return cache_dir / f"{video.stem}.{model.replace('.', '-')}.{imgsz}.conf{conf:g}.{key}.json"


def main() -> None:
    p = argparse.ArgumentParser(description="Precompute YOLO11+ByteTrack detections on Modal.")
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--app-name", default="caddy-segformer-remote")
    p.add_argument("--model", default="yolo11x.pt")
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--conf", type=float, default=0.20)
    p.add_argument("--iou", type=float, default=0.50)
    p.add_argument("--tracker", default="bytetrack.yaml")
    p.add_argument("--cache-dir", type=Path, default=Path("/tmp/caddy_yolo_cache"))
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_path(args.video, args.cache_dir, args.model, args.imgsz, args.conf)
    if out_path.exists() and not args.force:
        print(json.dumps({"cache": str(out_path), "complete": True}))
        return

    upload = modal.Function.from_name(args.app_name, "upload_video")
    detect = modal.Function.from_name(args.app_name, "detect_video_yolo")

    remote_video_name = f"{hashlib.sha1(str(args.video.resolve()).encode()).hexdigest()[:16]}-{args.video.name}"
    print(f"[yolo] uploading {args.video} ({args.video.stat().st_size / 1e6:.1f} MB) to Modal...", flush=True)
    upload.remote(remote_video_name, args.video.read_bytes())
    print(f"[yolo] running {args.model} imgsz={args.imgsz} conf={args.conf} tracker={args.tracker}...", flush=True)
    t0 = time.perf_counter()
    result = detect.remote(
        remote_video_name,
        args.model,
        args.imgsz,
        args.conf,
        args.iou,
        args.tracker,
        None,
    )
    payload = json.loads(zlib.decompress(result["json_zlib"]).decode("utf-8"))
    payload.update({
        "video": str(args.video.resolve()),
        "video_size": args.video.stat().st_size,
        "video_mtime_ns": args.video.stat().st_mtime_ns,
        "created_ts": time.time(),
        "client_elapsed_s": round(time.perf_counter() - t0, 3),
    })
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, separators=(",", ":")))
    os.replace(tmp, out_path)
    print(json.dumps({
        "cache": str(out_path),
        "complete": True,
        "frames": result["frame_count"],
        "elapsed_s": result["elapsed_s"],
    }))


if __name__ == "__main__":
    main()

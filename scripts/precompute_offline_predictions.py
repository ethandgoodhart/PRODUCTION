#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEG_CACHE_DIR = PROJECT_ROOT / ".cache" / "caddy_segmentation_cache"
CLR_CACHE_DIR = PROJECT_ROOT / ".cache" / "caddy_clrnet_cache"
MANIFEST_PATH = PROJECT_ROOT / ".cache" / "offline_predictions" / "manifest.json"
SELECTION_PATH = Path("/tmp/offline_prediction_selection.json")
CLRNET_CKPT = PROJECT_ROOT / ".cache" / "clrnet_weights" / "clrernet_culane_dla34_ema.pth"


def atomic_json_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(tmp, path)


def run_command(cmd: list[str]) -> None:
    print("[offline] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def video_meta(path: Path) -> dict:
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open {path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        cap.release()
    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_s": frame_count / fps if fps > 0 else 0.0,
    }


def segmentation_paths(video: Path, cache_dir: Path, model: str, width: int, height: int) -> tuple[Path, Path]:
    st = video.stat()
    key_src = f"{video.resolve()}:{st.st_size}:{int(st.st_mtime_ns)}:{model}:{width}x{height}"
    key = hashlib.sha1(key_src.encode("utf-8")).hexdigest()[:16]
    stem = f"{video.stem}.{model}.{width}x{height}.{key}"
    return cache_dir / f"{stem}.uint8", cache_dir / f"{stem}.json"


def clrnet_path(video: Path, cache_dir: Path, width: int, height: int) -> Path:
    st = video.stat()
    key_src = f"{video.resolve()}:{st.st_size}:{int(st.st_mtime_ns)}:clrnet_dla34:{width}x{height}"
    key = hashlib.sha1(key_src.encode("utf-8")).hexdigest()[:16]
    stem = f"{video.stem}.clrnet_dla34.{width}x{height}.{key}"
    return cache_dir / f"{stem}.json"


def load_manifest(path: Path) -> dict:
    try:
        with path.open() as f:
            manifest = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        manifest = {"schema": "caddy.offline_predictions.v1", "runs": []}
    if not isinstance(manifest.get("runs"), list):
        manifest["runs"] = []
    return manifest


def upsert_run(manifest: dict, run: dict) -> dict:
    runs = [r for r in manifest.get("runs", []) if isinstance(r, dict)]
    runs = [r for r in runs if r.get("id") != run["id"]]
    runs.append(run)
    runs.sort(key=lambda r: float(r.get("created_ts", 0.0)), reverse=True)
    manifest["runs"] = runs
    manifest["updated_ts"] = time.time()
    return manifest


def main() -> None:
    p = argparse.ArgumentParser(description="Precompute and register offline Modal predictions for one clip.")
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--label", default=None)
    p.add_argument("--run-id", default=None)
    p.add_argument("--seg-model", default="b0", choices=("b0", "b2", "b5"))
    p.add_argument("--seg-app-name", default="caddy-segformer-remote")
    p.add_argument("--clrnet-app-name", default="caddy-clrnet-remote")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--seg-chunk-frames", type=int, default=300)
    p.add_argument("--clrnet-chunk-frames", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--checkpoint", type=Path, default=CLRNET_CKPT)
    p.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    p.add_argument("--selection-file", type=Path, default=SELECTION_PATH)
    p.add_argument("--force", action="store_true")
    p.add_argument("--no-activate", action="store_true")
    args = p.parse_args()

    video = args.video.expanduser().resolve()
    if not video.exists():
        raise FileNotFoundError(video)

    SEG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    CLR_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    seg_cmd = [
        sys.executable, "scripts/precompute_segmentation_modal.py",
        "--video", str(video),
        "--model", args.seg_model,
        "--app-name", args.seg_app_name,
        "--cache-dir", str(SEG_CACHE_DIR),
        "--chunk-frames", str(args.seg_chunk_frames),
        "--batch-size", str(args.batch_size),
        "--width", str(args.width),
        "--height", str(args.height),
    ]
    clr_cmd = [
        sys.executable, "scripts/precompute_clrnet_modal.py",
        "--video", str(video),
        "--app-name", args.clrnet_app_name,
        "--cache-dir", str(CLR_CACHE_DIR),
        "--chunk-frames", str(args.clrnet_chunk_frames),
        "--width", str(args.width),
        "--height", str(args.height),
    ]
    if args.checkpoint.exists():
        clr_cmd.extend(["--checkpoint", str(args.checkpoint)])
    if args.force:
        seg_cmd.append("--force")
        clr_cmd.append("--force")

    started = time.perf_counter()
    run_command(seg_cmd)
    run_command(clr_cmd)

    meta = video_meta(video)
    seg_cache, seg_meta = segmentation_paths(video, SEG_CACHE_DIR, args.seg_model, args.width, args.height)
    clr_cache = clrnet_path(video, CLR_CACHE_DIR, args.width, args.height)
    missing = [str(p) for p in (seg_cache, seg_meta, clr_cache) if not p.exists()]
    if missing:
        raise FileNotFoundError("missing generated files: " + ", ".join(missing))

    st = video.stat()
    run_key = hashlib.sha1(
        f"{video}:{st.st_size}:{st.st_mtime_ns}:{args.seg_model}:{args.width}x{args.height}".encode("utf-8")
    ).hexdigest()[:12]
    run_id = args.run_id or f"{video.stem}-{args.seg_model}-clrnet-{run_key}"
    run = {
        "id": run_id,
        "label": args.label or f"{video.name} · SegFormer {args.seg_model.upper()} + CLRNet",
        "video": str(video),
        "video_size": st.st_size,
        "video_mtime_ns": st.st_mtime_ns,
        "frame_count": int(meta["frame_count"]),
        "fps": float(meta["fps"]),
        "duration_s": float(meta["duration_s"]),
        "source_width": int(meta["width"]),
        "source_height": int(meta["height"]),
        "width": int(args.width),
        "height": int(args.height),
        "segmentation_model": args.seg_model,
        "segmentation_meta": str(seg_meta),
        "segmentation_cache": str(seg_cache),
        "clrnet_cache": str(clr_cache),
        "created_ts": time.time(),
        "elapsed_s": round(time.perf_counter() - started, 3),
    }

    manifest = upsert_run(load_manifest(args.manifest), run)
    atomic_json_write(args.manifest, manifest)
    if not args.no_activate:
        atomic_json_write(args.selection_file, {
            "seq": time.time_ns(),
            "ts": time.time(),
            "id": run["id"],
            "label": run["label"],
            "video": run["video"],
            "segmentation_meta": run["segmentation_meta"],
            "clrnet_cache": run["clrnet_cache"],
        })

    print(json.dumps({
        "manifest": str(args.manifest),
        "selection_file": str(args.selection_file),
        "run": run,
        "active": not args.no_activate,
    }, indent=2))


if __name__ == "__main__":
    main()

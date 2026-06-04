"""Modal-hosted SegFormer inference for the local segmentation planner.

The local sidecar sends one JPEG frame and receives a zlib-compressed uint8
Cityscapes class-ID map. BEV rendering, trajectory planning, and protective
braking stay local.
"""
from __future__ import annotations

import modal


APP_NAME = "caddy-segformer-remote"
CACHE_DIR = "/cache"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name("caddy-segmentation-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglx-mesa0", "libglib2.0-0", "libsm6", "libxext6")
    .pip_install(
        "torch==2.4.1",
        "torchvision==0.19.1",
        "transformers==4.44.2",
        "Pillow",
        "numpy",
        "ultralytics>=8.3.0",
    )
    .run_commands(
        "pip uninstall -y opencv-python opencv-contrib-python opencv-contrib-python-headless || true",
        "pip install --no-cache-dir opencv-python-headless",
        "python -c 'import cv2; from ultralytics import YOLO; print(\"cv2\", cv2.__version__, \"ultralytics ok\")'",
    )
)

MODEL_VARIANTS = {
    "b0": "nvidia/segformer-b0-finetuned-cityscapes-1024-1024",
    "b2": "nvidia/segformer-b2-finetuned-cityscapes-1024-1024",
    "b5": "nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
}

_processor = None
_model = None
_model_id = None


def _load_model(variant: str):
    import time

    import torch
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

    global _processor, _model, _model_id

    model_id = MODEL_VARIANTS[variant]
    timings: dict[str, float] = {}
    if _model is None or _model_id != model_id:
        t0 = time.perf_counter()
        _processor = SegformerImageProcessor.from_pretrained(model_id)
        _model = SegformerForSemanticSegmentation.from_pretrained(model_id).cuda().eval()
        _model_id = model_id
        timings["remote_load_ms"] = (time.perf_counter() - t0) * 1000.0
    return _processor, _model, model_id, timings


@app.function(
    image=image,
    volumes={CACHE_DIR: volume},
    timeout=600,
    memory=2048,
)
def upload_video(video_name: str, video_bytes: bytes) -> dict:
    import os
    from pathlib import Path

    safe_name = Path(video_name).name
    path = Path(CACHE_DIR) / safe_name
    path.write_bytes(video_bytes)
    volume.commit()
    return {"path": str(path), "bytes": os.path.getsize(path)}


@app.function(
    image=image,
    gpu="A10G",
    timeout=120,
    memory=16384,
    scaledown_window=300,
)
def segment_jpeg(jpeg_bytes: bytes, variant: str = "b0") -> dict:
    import time
    import zlib

    import cv2
    import numpy as np
    import torch
    from PIL import Image

    processor, model, model_id, timings = _load_model(variant)

    t0 = time.perf_counter()
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    frame_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame_bgr is None:
        raise ValueError("failed to decode input JPEG")
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(frame_rgb)
    timings["remote_decode_ms"] = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    inputs = processor(images=pil, return_tensors="pt").to("cuda")
    torch.cuda.synchronize()
    timings["remote_preprocess_ms"] = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(**inputs)
    torch.cuda.synchronize()
    timings["remote_forward_ms"] = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    seg = processor.post_process_semantic_segmentation(
        out, target_sizes=[frame_rgb.shape[:2]]
    )[0].cpu().numpy().astype(np.uint8)
    timings["remote_postprocess_ms"] = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    compressed = zlib.compress(seg.tobytes(), level=1)
    timings["remote_compress_ms"] = (time.perf_counter() - t0) * 1000.0

    return {
        "shape": list(seg.shape),
        "dtype": "uint8",
        "zlib": compressed,
        "timings_ms": {k: round(float(v), 3) for k, v in timings.items()},
        "model": model_id,
    }


@app.function(
    image=image,
    gpu="A10G",
    volumes={CACHE_DIR: volume},
    timeout=900,
    memory=24576,
    scaledown_window=300,
)
def segment_video_chunk(
    video_name: str,
    start_frame: int,
    num_frames: int,
    variant: str = "b0",
    width: int = 640,
    height: int = 480,
    batch_size: int = 16,
) -> dict:
    import time
    import zlib
    from pathlib import Path

    import cv2
    import numpy as np
    import torch
    from PIL import Image

    volume.reload()
    video_path = Path(CACHE_DIR) / Path(video_name).name
    if not video_path.exists():
        raise FileNotFoundError(str(video_path))

    processor, model, model_id, timings = _load_model(variant)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_frame))

    maps: list[np.ndarray] = []
    frames: list[Image.Image] = []
    target_sizes: list[tuple[int, int]] = []
    read_count = 0
    t_total = time.perf_counter()
    t_read = 0.0
    t_infer = 0.0

    def flush_batch() -> None:
        nonlocal frames, target_sizes, t_infer
        if not frames:
            return
        t0 = time.perf_counter()
        inputs = processor(images=frames, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model(**inputs)
        torch.cuda.synchronize()
        segs = processor.post_process_semantic_segmentation(out, target_sizes=target_sizes)
        for seg in segs:
            maps.append(seg.cpu().numpy().astype(np.uint8))
        t_infer += time.perf_counter() - t0
        frames = []
        target_sizes = []

    try:
        while read_count < int(num_frames):
            t0 = time.perf_counter()
            ok, frame_bgr = cap.read()
            t_read += time.perf_counter() - t0
            if not ok or frame_bgr is None:
                break
            if frame_bgr.shape[1] != width or frame_bgr.shape[0] != height:
                frame_bgr = cv2.resize(frame_bgr, (width, height), interpolation=cv2.INTER_AREA)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
            target_sizes.append((height, width))
            read_count += 1
            if len(frames) >= int(batch_size):
                flush_batch()
        flush_batch()
    finally:
        cap.release()

    if maps:
        stack = np.stack(maps, axis=0).astype(np.uint8, copy=False)
    else:
        stack = np.empty((0, height, width), dtype=np.uint8)

    t0 = time.perf_counter()
    compressed = zlib.compress(stack.tobytes(order="C"), level=1)
    timings.update({
        "remote_read_ms": round(t_read * 1000.0, 3),
        "remote_infer_ms": round(t_infer * 1000.0, 3),
        "remote_total_ms": round((time.perf_counter() - t_total) * 1000.0, 3),
        "remote_compress_ms": round((time.perf_counter() - t0) * 1000.0, 3),
    })
    return {
        "start_frame": int(start_frame),
        "num_frames": int(stack.shape[0]),
        "shape": list(stack.shape),
        "dtype": "uint8",
        "zlib": compressed,
        "timings_ms": {k: round(float(v), 3) for k, v in timings.items()},
        "model": model_id,
    }


@app.function(
    image=image,
    gpu="A10G",
    volumes={CACHE_DIR: volume},
    timeout=3600,
    memory=24576,
    scaledown_window=300,
)
def detect_video_yolo(
    video_name: str,
    model_name: str = "yolo11x.pt",
    imgsz: int = 960,
    conf: float = 0.20,
    iou: float = 0.50,
    tracker: str = "bytetrack.yaml",
    classes: list[int] | None = None,
) -> dict:
    import json
    import time
    import zlib
    from pathlib import Path

    import cv2
    from ultralytics import YOLO

    volume.reload()
    video_path = Path(CACHE_DIR) / Path(video_name).name
    if not video_path.exists():
        raise FileNotFoundError(str(video_path))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()

    t0 = time.perf_counter()
    model = YOLO(model_name)
    names = getattr(model, "names", {}) or {}
    selected_classes = classes if classes is not None else [0, 1, 2, 3, 5, 7, 9, 11]
    frames: list[dict] = []
    result_iter = model.track(
        source=str(video_path),
        stream=True,
        persist=True,
        tracker=tracker,
        imgsz=int(imgsz),
        conf=float(conf),
        iou=float(iou),
        classes=selected_classes,
        verbose=False,
        device=0,
    )
    for frame_idx, result in enumerate(result_iter):
        detections = []
        boxes = getattr(result, "boxes", None)
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.detach().cpu().numpy()
            confs = boxes.conf.detach().cpu().numpy() if boxes.conf is not None else []
            clss = boxes.cls.detach().cpu().numpy() if boxes.cls is not None else []
            ids = (
                boxes.id.detach().cpu().numpy()
                if getattr(boxes, "id", None) is not None else [None] * len(xyxy)
            )
            for i, box in enumerate(xyxy):
                cls_id = int(clss[i]) if i < len(clss) else -1
                track_id = None if ids[i] is None else int(ids[i])
                detections.append({
                    "track_id": track_id,
                    "class_id": cls_id,
                    "class_name": str(names.get(cls_id, cls_id)),
                    "confidence": round(float(confs[i]) if i < len(confs) else 0.0, 4),
                    "xyxy": [round(float(v), 2) for v in box.tolist()],
                })
        frames.append({"frame_index": frame_idx, "detections": detections})

    payload = {
        "schema": "caddy.yolo_tracks.v1",
        "model": model_name,
        "tracker": tracker,
        "imgsz": int(imgsz),
        "conf": float(conf),
        "iou": float(iou),
        "classes": selected_classes,
        "names": {str(k): str(v) for k, v in names.items()},
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "elapsed_s": round(time.perf_counter() - t0, 3),
        "frames": frames,
    }
    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    compressed = zlib.compress(raw, level=6)
    return {
        "json_zlib": compressed,
        "bytes": len(raw),
        "zlib_bytes": len(compressed),
        "frame_count": len(frames),
        "elapsed_s": payload["elapsed_s"],
    }

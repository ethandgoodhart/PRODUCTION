"""Modal-hosted CLRerNet lane detection for offline precomputation.

Runs the CLRerNet DLA34-EMA model on Modal GPUs. Each frame returns a list of
detected lanes with normalized [0,1] point coordinates and confidence scores.
The local sidecar caches these and uses them to override segmentation steering.
"""
from __future__ import annotations

import modal

APP_NAME = "caddy-clrnet-remote"
CACHE_DIR = "/cache"
CLRNET_DIR = "/clrnet"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name("caddy-clrnet-cache", create_if_missing=True)

image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("libgl1", "libglx-mesa0", "libglib2.0-0", "libsm6", "libxext6", "git", "build-essential", "clang")
    .pip_install(
        "torch==2.1.2",
        "torchvision==0.16.2",
        "numpy<2",
        "Pillow",
        "scipy",
        "opencv-python-headless",
        "albumentations>=1.3.0",
        "mmengine>=0.10.0",
        "p_tqdm",
    )
    .run_commands(
        "pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html",
        "pip install mmdet==3.3.0",
        f"git clone --depth 1 https://github.com/hirotomusiker/CLRerNet.git {CLRNET_DIR}",
        f"mkdir -p {CLRNET_DIR}/dataset/culane/list && touch {CLRNET_DIR}/dataset/culane/list/test.txt",
        f"cd {CLRNET_DIR}/libs/models/layers/nms && TORCH_CUDA_ARCH_LIST='8.6' python setup.py install",
        f"cd {CLRNET_DIR} && python -c 'import libs.models; import libs.datasets; print(\"clrernet libs ok\")'",
        "python -c 'import cv2, torch, mmdet, mmcv; from mmcv.ops import nms; print(\"deps ok, mmcv._ext works\")'",
    )
)

CLRNET_INPUT_W, CLRNET_INPUT_H = 1640, 590

_model = None


def _load_model(ckpt_bytes: bytes | None = None):
    import os
    import sys
    import time

    import torch

    global _model
    if _model is not None:
        return _model, {}

    timings = {}
    t0 = time.perf_counter()

    if CLRNET_DIR not in sys.path:
        sys.path.insert(0, CLRNET_DIR)

    prev_cwd = os.getcwd()
    os.chdir(CLRNET_DIR)
    try:
        base_cfg_path = os.path.join(CLRNET_DIR, "configs/clrernet/base_clrernet.py")
        if os.path.exists(base_cfg_path):
            src = open(base_cfg_path).read()
            if "pretrained=True" in src:
                open(base_cfg_path, "w").write(src.replace("pretrained=True", "pretrained=False"))

        alaug_path = os.path.join(CLRNET_DIR, "libs/datasets/pipelines/alaug.py")
        if os.path.exists(alaug_path):
            src = open(alaug_path).read()
            old_call = (
                "        aug = self.__augmentor(\n"
                "            image=img,\n"
                "            keypoints=keypoints_val,\n"
                "            bboxes=bboxes,\n"
                "            mask=masks,\n"
                "            bbox_labels=bbox_labels,\n"
                "        )"
            )
            new_call = (
                "        kwargs = dict(image=img)\n"
                "        if keypoints_val is not None: kwargs['keypoints'] = keypoints_val\n"
                "        if bboxes is not None:\n"
                "            kwargs['bboxes'] = bboxes\n"
                "            kwargs['bbox_labels'] = bbox_labels\n"
                "        if masks is not None: kwargs['mask'] = masks\n"
                "        aug = self.__augmentor(**kwargs)"
            )
            if old_call in src:
                open(alaug_path, "w").write(src.replace(old_call, new_call))

        ckpt_path = os.path.join(CACHE_DIR, "clrernet_culane_dla34_ema.pth")
        if ckpt_bytes is not None:
            os.makedirs(CACHE_DIR, exist_ok=True)
            open(ckpt_path, "wb").write(ckpt_bytes)
            volume.commit()

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"CLRNet checkpoint not found at {ckpt_path}. Upload it first.")

        config_path = os.path.join(CLRNET_DIR, "configs/clrernet/culane/clrernet_culane_dla34_ema.py")
        from mmdet.apis import init_detector
        from libs.datasets.metrics.culane_metric import interp
        from libs.datasets.pipelines import Compose

        model_obj = init_detector(config_path, ckpt_path, device="cuda:0")
        model_obj.bbox_head.test_cfg.as_lanes = False
        model_obj.bbox_head.test_cfg.conf_threshold = 0.05

        cfg = model_obj.cfg
        test_pipeline = Compose(cfg.test_dataloader.dataset.pipeline)

        _model = {
            "model": model_obj,
            "pipeline": test_pipeline,
            "interp": interp,
            "torch": torch,
        }
        timings["remote_load_ms"] = (time.perf_counter() - t0) * 1000.0
    finally:
        os.chdir(prev_cwd)

    return _model, timings


def _infer_frame(frame_bgr, m):
    import numpy as np
    import cv2

    ori_shape = frame_bgr.shape
    pipe_in = cv2.resize(frame_bgr, (CLRNET_INPUT_W, CLRNET_INPUT_H),
                         interpolation=cv2.INTER_LINEAR)
    data = dict(
        filename="frame.jpg",
        sub_img_name=None,
        img=pipe_in,
        gt_points=[],
        id_classes=[],
        id_instances=[],
        img_shape=pipe_in.shape,
        ori_shape=ori_shape,
    )
    data = m["pipeline"](data)
    data_ = dict(
        inputs=[data["inputs"]],
        data_samples=[data["data_samples"]],
    )
    with m["torch"].no_grad():
        results = m["model"].test_step(data_)
    lanes = results[0]["lanes"]
    scores_t = results[0].get("scores")
    if hasattr(scores_t, "detach"):
        scores = scores_t.detach().cpu().numpy().tolist()
    elif scores_t is not None:
        scores = list(scores_t)
    else:
        scores = [1.0] * len(lanes)

    out = []
    ori_h, ori_w = ori_shape[0], ori_shape[1]
    for lane, sc in zip(lanes, scores):
        arr = lane.detach().cpu().numpy() if hasattr(lane, "detach") else np.asarray(lane)
        xs = arr[:, 0]
        ys = arr[:, 1]
        valid = (xs >= 0) & (xs < 1)
        if valid.sum() < 2:
            continue
        lx = xs[valid] * ori_w
        ly = ys[valid] * ori_h
        lx, ly = lx[::-1], ly[::-1]
        pred_pts = list(zip(lx.tolist(), ly.tolist()))
        try:
            interp_pts = m["interp"](pred_pts, n=5)
        except Exception:
            interp_pts = np.asarray(pred_pts, dtype=np.float32)
        interp_pts = np.asarray(interp_pts, dtype=np.float32)
        if interp_pts.size == 0 or interp_pts.ndim != 2:
            continue
        interp_pts[:, 0] /= max(ori_w, 1)
        interp_pts[:, 1] /= max(ori_h, 1)
        out.append({
            "points": interp_pts.tolist(),
            "score": round(float(sc), 4),
        })
    return out


@app.function(
    image=image,
    volumes={CACHE_DIR: volume},
    timeout=600,
    memory=2048,
)
def upload_checkpoint(ckpt_bytes: bytes) -> dict:
    import os
    path = os.path.join(CACHE_DIR, "clrernet_culane_dla34_ema.pth")
    os.makedirs(CACHE_DIR, exist_ok=True)
    open(path, "wb").write(ckpt_bytes)
    volume.commit()
    return {"path": path, "bytes": len(ckpt_bytes)}


@app.function(
    image=image,
    volumes={CACHE_DIR: volume},
    timeout=600,
    memory=2048,
)
def upload_video(video_name: str, video_bytes: bytes) -> dict:
    import os
    path = os.path.join(CACHE_DIR, video_name)
    os.makedirs(CACHE_DIR, exist_ok=True)
    open(path, "wb").write(video_bytes)
    volume.commit()
    return {"path": path, "bytes": len(video_bytes)}


@app.function(
    image=image,
    gpu="A10G",
    volumes={CACHE_DIR: volume},
    timeout=600,
    memory=8192,
)
def detect_lanes_chunk(
    video_name: str,
    start_frame: int,
    num_frames: int,
    width: int = 640,
    height: int = 480,
) -> dict:
    import json
    import time
    import zlib

    import cv2
    import numpy as np

    t0 = time.perf_counter()
    m, timings = _load_model()

    video_path = f"{CACHE_DIR}/{video_name}"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    t_read_start = time.perf_counter()

    results = []
    for i in range(num_frames):
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
        lanes = _infer_frame(frame, m)
        results.append({
            "frame_index": start_frame + i,
            "lanes": lanes,
        })
    cap.release()

    timings["remote_infer_ms"] = (time.perf_counter() - t_read_start) * 1000.0

    payload = json.dumps(results).encode("utf-8")
    compressed = zlib.compress(payload, level=6)
    timings["total_ms"] = (time.perf_counter() - t0) * 1000.0

    return {
        "num_frames": len(results),
        "start_frame": start_frame,
        "zlib": compressed,
        "timings_ms": timings,
    }

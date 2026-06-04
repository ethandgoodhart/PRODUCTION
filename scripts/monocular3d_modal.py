"""Modal-hosted learned monocular 3D object detection.

Runs an OpenMMLab FCOS3D monocular detector on Modal GPUs. The local planner
sends one JPEG frame plus camera intrinsics and receives learned 3D detections
and a rendered JPEG visualization. No 2D detector is used for this stream.
"""
from __future__ import annotations

import modal


APP_NAME = "caddy-monocular3d-fcos3d"
CACHE_DIR = "/cache"
MODEL_DIR = "/models/mono3d"
MMDET3D_REPO = "/opt/mmdetection3d"
CONFIG_PATH = (
    f"{MMDET3D_REPO}/configs/fcos3d/"
    "fcos3d_r101-caffe-dcn_fpn_head-gn_8xb2-1x_nus-mono3d_finetune.py"
)
CHECKPOINT_URL = (
    "https://download.openmmlab.com/mmdetection3d/v0.1.0_models/fcos3d/"
    "fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune/"
    "fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune_20210717_095645-8d806dc2.pth"
)

app = modal.App(APP_NAME)
volume = modal.Volume.from_name("caddy-monocular3d-cache", create_if_missing=True)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.1-devel-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install(
        "git",
        "wget",
        "curl",
        "ffmpeg",
        "libgl1",
        "libglib2.0-0",
        "libsm6",
        "libxext6",
        "libxrender-dev",
    )
    .pip_install(
        "numpy<2",
        "Pillow",
        "opencv-python-headless",
        "openmim",
    )
    .run_commands(
        "pip install --no-cache-dir torch==2.1.2 torchvision==0.16.2 "
        "--index-url https://download.pytorch.org/whl/cu121",
        "mim install 'mmengine>=0.8.0' 'mmcv>=2.0.0,<2.2.0' 'mmdet>=3.0.0,<3.4.0'",
        "pip install --no-cache-dir mmdet3d==1.4.0",
        f"mkdir -p {MODEL_DIR}",
        f"git clone --depth 1 https://github.com/open-mmlab/mmdetection3d.git {MMDET3D_REPO}",
        f"curl -fsSL {CHECKPOINT_URL} -o {MODEL_DIR}/fcos3d.pth",
        "python - <<'PY'\n"
        "from mmdet3d.apis import MonoDet3DInferencer\n"
        "print('mmdet3d MonoDet3DInferencer import ok')\n"
        "PY",
    )
)

_inferencer = None


def _load_inferencer():
    global _inferencer
    if _inferencer is None:
        from mmdet3d.apis import MonoDet3DInferencer

        _inferencer = MonoDet3DInferencer(
            model=CONFIG_PATH,
            weights=f"{MODEL_DIR}/fcos3d.pth",
            device="cuda:0",
        )
        _inferencer.show_progress = False
    return _inferencer


def _camera_info(img_path: str, fx: float, fy: float, cx: float, cy: float) -> dict:
    k = [[float(fx), 0.0, float(cx)], [0.0, float(fy), float(cy)], [0.0, 0.0, 1.0]]
    ident4 = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    camera_payload = {
        "img_path": img_path,
        "cam2img": k,
        "lidar2cam": ident4,
        "lidar2img": [
            [float(fx), 0.0, float(cx), 0.0],
            [0.0, float(fy), float(cy), 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
    }
    return {
        "metainfo": {
            "categories": {
                "car": 0,
                "truck": 1,
                "trailer": 2,
                "bus": 3,
                "construction_vehicle": 4,
                "bicycle": 5,
                "motorcycle": 6,
                "pedestrian": 7,
                "traffic_cone": 8,
                "barrier": 9,
            },
            "dataset": "nuscenes",
            "version": "caddy-live",
            "info_version": "1.1",
        },
        "data_list": [{
            "sample_idx": 0,
            "token": "caddy-live-frame",
            "timestamp": 0,
            "images": {
                "CAM_FRONT": camera_payload,
                # MMDetection3D's MonoDet3DInferencer has a KITTI default
                # camera key in one preprocessing path. Alias the same
                # calibrated frame so FCOS3D still receives the requested image.
                "CAM2": camera_payload,
            },
            "lidar_points": {
                "num_pts_feats": 5,
                "lidar_path": "",
                "lidar2ego": ident4,
            },
            "instances": [],
            "cam_instances": {"CAM_FRONT": [], "CAM2": []},
        }],
    }


@app.function(
    image=image,
    gpu="A10G",
    timeout=180,
    memory=24576,
    scaledown_window=300,
)
def detect_jpeg(
    jpeg_bytes: bytes,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    score_thr: float = 0.25,
) -> dict:
    import base64
    import tempfile
    import time
    from pathlib import Path

    import cv2
    import mmengine
    import numpy as np

    timings: dict[str, float] = {}
    inferencer = _load_inferencer()

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        img_path = tmp / "frame.jpg"
        info_path = tmp / "infos.pkl"
        img_path.write_bytes(jpeg_bytes)
        mmengine.dump(_camera_info(str(img_path), fx, fy, cx, cy), info_path)

        t0 = time.perf_counter()
        result = inferencer(
            inputs={"img": str(img_path), "infos": str(info_path)},
            cam_type="CAM_FRONT",
            pred_score_thr=float(score_thr),
            return_vis=True,
            show=False,
            no_save_vis=True,
            no_save_pred=True,
        )
        timings["remote_infer_ms"] = (time.perf_counter() - t0) * 1000.0

    predictions = (result or {}).get("predictions") or []
    pred = predictions[0] if predictions else {}
    labels = pred.get("labels_3d") or []
    scores = pred.get("scores_3d") or []
    boxes = pred.get("bboxes_3d") or []
    names = {
        0: "car",
        1: "truck",
        2: "trailer",
        3: "bus",
        4: "construction_vehicle",
        5: "bicycle",
        6: "motorcycle",
        7: "pedestrian",
        8: "traffic_cone",
        9: "barrier",
    }
    objects = []
    for i, box in enumerate(boxes):
        score = float(scores[i]) if i < len(scores) else 0.0
        if score < float(score_thr):
            continue
        label_id = int(labels[i]) if i < len(labels) else -1
        objects.append({
            "class_id": label_id,
            "class_name": names.get(label_id, f"class_{label_id}"),
            "confidence": score,
            "camera_box3d": [float(x) for x in box],
        })

    vis_list = (result or {}).get("visualization") or []
    if vis_list:
        vis = vis_list[0]
        if vis.dtype != np.uint8:
            vis = np.clip(vis, 0, 255).astype(np.uint8)
        # OpenMMLab visualizer returns RGB; encode JPEG with OpenCV BGR.
        vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
    else:
        arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        vis_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    ok, enc = cv2.imencode(".jpg", vis_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    if not ok:
        raise RuntimeError("failed to encode monocular 3D visualization")

    return {
        "provider": "mmdet3d_fcos3d_nuscenes",
        "objects": objects,
        "viz_jpeg_b64": base64.b64encode(enc.tobytes()).decode("ascii"),
        "timings_ms": {k: round(float(v), 3) for k, v in timings.items()},
    }


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
    volumes={CACHE_DIR: volume},
    timeout=1200,
    memory=24576,
    scaledown_window=300,
)
def detect_video_chunk(
    video_name: str,
    start_frame: int,
    num_frames: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    score_thr: float = 0.05,
    width: int = 640,
    height: int = 480,
) -> dict:
    import base64
    import json
    import tempfile
    import time
    import zlib
    from pathlib import Path

    import cv2
    import mmengine
    import numpy as np

    volume.reload()
    video_path = Path(CACHE_DIR) / Path(video_name).name
    if not video_path.exists():
        raise FileNotFoundError(str(video_path))

    inferencer = _load_inferencer()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    source_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or width)
    source_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or height)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_frame))

    names = {
        0: "car",
        1: "truck",
        2: "trailer",
        3: "bus",
        4: "construction_vehicle",
        5: "bicycle",
        6: "motorcycle",
        7: "pedestrian",
        8: "traffic_cone",
        9: "barrier",
    }
    frames = []
    timings: dict[str, float] = {}
    t_total = time.perf_counter()
    t_infer = 0.0
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for i in range(int(num_frames)):
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frame_idx = int(start_frame) + i
            if frame_bgr.shape[1] != int(width) or frame_bgr.shape[0] != int(height):
                frame_bgr = cv2.resize(frame_bgr, (int(width), int(height)), interpolation=cv2.INTER_LINEAR)
            img_path = tmp / f"frame_{frame_idx:06d}.jpg"
            info_path = tmp / f"infos_{frame_idx:06d}.pkl"
            cv2.imwrite(str(img_path), frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            mmengine.dump(_camera_info(str(img_path), fx, fy, cx, cy), info_path)

            t0 = time.perf_counter()
            result = inferencer(
                inputs={"img": str(img_path), "infos": str(info_path)},
                cam_type="CAM_FRONT",
                pred_score_thr=float(score_thr),
                return_vis=True,
                show=False,
                no_save_vis=True,
                no_save_pred=True,
            )
            t_infer += time.perf_counter() - t0

            predictions = (result or {}).get("predictions") or []
            pred = predictions[0] if predictions else {}
            labels = pred.get("labels_3d") or []
            scores = pred.get("scores_3d") or []
            boxes = pred.get("bboxes_3d") or []
            objects = []
            for j, box in enumerate(boxes):
                score = float(scores[j]) if j < len(scores) else 0.0
                if score < float(score_thr):
                    continue
                label_id = int(labels[j]) if j < len(labels) else -1
                objects.append({
                    "class_id": label_id,
                    "class_name": names.get(label_id, f"class_{label_id}"),
                    "confidence": score,
                    "camera_box3d": [float(x) for x in box],
                })

            vis_list = (result or {}).get("visualization") or []
            if vis_list:
                vis = vis_list[0]
                if vis.dtype != np.uint8:
                    vis = np.clip(vis, 0, 255).astype(np.uint8)
                vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
            else:
                vis_bgr = frame_bgr
            ok, enc = cv2.imencode(".jpg", vis_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
            if not ok:
                raise RuntimeError(f"failed to encode visualization for frame {frame_idx}")
            frames.append({
                "frame_index": frame_idx,
                "objects": objects,
                "viz_jpeg_b64": base64.b64encode(enc.tobytes()).decode("ascii"),
            })
    cap.release()

    timings["remote_infer_ms"] = t_infer * 1000.0
    timings["remote_total_ms"] = (time.perf_counter() - t_total) * 1000.0
    payload = json.dumps(frames, separators=(",", ":")).encode("utf-8")
    compressed = zlib.compress(payload, level=6)
    return {
        "provider": "mmdet3d_fcos3d_nuscenes",
        "start_frame": int(start_frame),
        "num_frames": len(frames),
        "fps": fps,
        "frame_count": frame_count,
        "width": int(width),
        "height": int(height),
        "source_width": source_w,
        "source_height": source_h,
        "json_zlib": compressed,
        "bytes": len(payload),
        "zlib_bytes": len(compressed),
        "timings_ms": {k: round(float(v), 3) for k, v in timings.items()},
    }

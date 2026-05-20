#!/usr/bin/env python3
"""
calibrate_extrinsics_charuco.py - relative camera extrinsics from ChArUco.

This estimates camera-to-camera transforms for a static rig. Put the printed
ChArUco board where all selected cameras can see it, move it through several
poses, and save samples. The script computes transforms relative to a reference
camera, defaulting to `front`.

The output is relative rig geometry, not full vehicle/ego geometry. To get
SparseDrive ego-frame extrinsics, measure where the reference camera sits on
the vehicle and compose that measured ego->front transform with these relative
transforms.

Examples:
    uv run python scripts/calibrate_extrinsics_charuco.py --autosave
    uv run python scripts/calibrate_extrinsics_charuco.py --cameras front front_left front_right

Keys:
    s       save current board pose if every camera detects it
    a       toggle autosave
    c       compute/write extrinsics
    q/Esc   quit
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import time
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
BOARD_META_DEFAULT = (
    PROJECT_ROOT
    / "calibration/boards/charuco_US_LETTER_7x10_25mm_18mm_DICT5X5_1000.json"
)
MAPPING_DEFAULT = PROJECT_ROOT / "calibration/cameras/camera_mapping.json"
OUT_DEFAULT = PROJECT_ROOT / "calibration/cameras/REAL_relative_extrinsics_front_reference.json"
DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480
PREFERRED_FOURCC = "MJPG"


def capture_backend() -> int:
    system = platform.system()
    if system == "Darwin":
        return cv2.CAP_AVFOUNDATION
    if system == "Linux":
        return cv2.CAP_V4L2
    return cv2.CAP_ANY


def open_camera(index: int, width: int, height: int, force_mjpg: bool) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index, capture_backend())
    if not cap.isOpened():
        raise RuntimeError(f"failed to open camera index {index}")
    if force_mjpg:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*PREFERRED_FOURCC))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        raise RuntimeError(f"camera index {index} opened but returned no frame")
    return cap


def load_board(meta_path: Path):
    meta = json.loads(meta_path.read_text())
    dictionary_id = getattr(cv2.aruco, meta["dictionary"])
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
    board = cv2.aruco.CharucoBoard(
        (int(meta["squares_x"]), int(meta["squares_y"])),
        float(meta["square_size_mm"]) / 1000.0,
        float(meta["marker_size_mm"]) / 1000.0,
        dictionary,
    )
    detector = cv2.aruco.CharucoDetector(board)
    return meta, board, detector


def load_camera_configs(mapping_path: Path, names: list[str]) -> list[dict]:
    mapping = json.loads(mapping_path.read_text())
    by_name = {m["logical_name"]: m for m in mapping["mappings"]}
    configs = []
    for name in names:
        if name not in by_name:
            raise KeyError(f"{name!r} not in {mapping_path}")
        item = dict(by_name[name])
        intr_path = mapping_path.parent / item["intrinsics"]
        intr = json.loads(intr_path.read_text())
        item["intrinsics_path"] = str(intr_path)
        item["intrinsic_width"] = int(intr["image_width"])
        item["intrinsic_height"] = int(intr["image_height"])
        item["K"] = np.array(intr["camera_matrix"], dtype=np.float64)
        item["D"] = np.array(intr["dist_coeffs"], dtype=np.float64).reshape(-1, 1)
        configs.append(item)
    return configs


def make_T(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return T


def inv_T(T: np.ndarray) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    R = T[:3, :3]
    t = T[:3, 3]
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def solve_board_pose(board, detector, frame: np.ndarray, K: np.ndarray, D: np.ndarray, min_corners: int):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray)
    corners_seen = 0 if charuco_ids is None else int(len(charuco_ids))
    viz = frame.copy()
    if marker_ids is not None and len(marker_ids) > 0:
        cv2.aruco.drawDetectedMarkers(viz, marker_corners, marker_ids)
    if charuco_ids is not None and corners_seen > 0:
        cv2.aruco.drawDetectedCornersCharuco(viz, charuco_corners, charuco_ids)
    if charuco_ids is None or corners_seen < min_corners:
        return None, viz, corners_seen

    obj_points, img_points = board.matchImagePoints(charuco_corners, charuco_ids)
    ok, rvec, tvec = cv2.solvePnP(obj_points, img_points, K, D, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None, viz, corners_seen
    try:
        cv2.drawFrameAxes(viz, K, D, rvec, tvec, 0.08)
    except Exception:
        pass
    return make_T(rvec, tvec), viz, corners_seen


def grid(frames: list[np.ndarray]) -> np.ndarray:
    h = min(f.shape[0] for f in frames)
    w = min(f.shape[1] for f in frames)
    resized = [cv2.resize(f, (w, h)) for f in frames]
    return np.hstack(resized)


def annotate(frame: np.ndarray, label: str, corners: int, ok: bool) -> np.ndarray:
    out = frame.copy()
    text = f"{label} corners={corners} {'OK' if ok else 'NO'}"
    color = (80, 230, 120) if ok else (80, 80, 255)
    cv2.putText(out, text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(out, text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return out


def rotation_median(rotations: list[np.ndarray]) -> np.ndarray:
    # Average rotation via SVD projection. Good enough for small sample spread.
    M = np.mean(np.stack(rotations), axis=0)
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def summarize(samples: list[dict], camera_names: list[str], reference: str) -> dict:
    out = {
        "reference_camera": reference,
        "coordinate_convention": "OpenCV camera frame: +x right, +y down, +z forward",
        "transform_definition": (
            "T_camera_reference maps a point expressed in the reference camera "
            "frame into the named camera frame."
        ),
        "sample_count": len(samples),
        "cameras": {},
    }
    for name in camera_names:
        Ts = [s["T_camera_reference"][name] for s in samples]
        translations = np.stack([T[:3, 3] for T in Ts], axis=0)
        rotations = [T[:3, :3] for T in Ts]
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = rotation_median(rotations)
        T[:3, 3] = np.median(translations, axis=0)
        rvec, _ = cv2.Rodrigues(T[:3, :3])
        out["cameras"][name] = {
            "T_camera_reference": T.tolist(),
            "T_reference_camera": inv_T(T).tolist(),
            "translation_m": T[:3, 3].tolist(),
            "rotation_vector_rad": rvec.reshape(3).tolist(),
            "translation_std_m": np.std(translations, axis=0).tolist(),
        }
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mapping", type=Path, default=MAPPING_DEFAULT)
    p.add_argument("--board", type=Path, default=BOARD_META_DEFAULT)
    p.add_argument("--out", type=Path, default=OUT_DEFAULT)
    p.add_argument("--cameras", nargs="+", default=["front", "front_left", "front_right"])
    p.add_argument("--reference", default="front")
    p.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    p.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    p.add_argument("--samples", type=int, default=25)
    p.add_argument("--min-corners", type=int, default=16)
    p.add_argument("--autosave", action="store_true")
    p.add_argument(
        "--autosave-interval-s", type=float, default=1.0,
        help="Seconds between autosaved shared detections.",
    )
    p.add_argument("--no-fourcc", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.reference not in args.cameras:
        raise ValueError("--reference must be included in --cameras")
    board_meta, board, detector = load_board(args.board)
    configs = load_camera_configs(args.mapping, args.cameras)
    caps = []
    try:
        for cfg in configs:
            cap = open_camera(cfg["camera_index"], args.width, args.height, force_mjpg=not args.no_fourcc)
            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or args.width
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or args.height
            sx = actual_w / float(cfg["intrinsic_width"])
            sy = actual_h / float(cfg["intrinsic_height"])
            cfg["K"] = cfg["K"].copy()
            cfg["K"][0, 0] *= sx
            cfg["K"][1, 1] *= sy
            cfg["K"][0, 2] *= sx
            cfg["K"][1, 2] *= sy
            caps.append(cap)
            print(
                f"[extrinsics] opened {cfg['logical_name']} "
                f"index={cfg['camera_index']} actual={actual_w}x{actual_h}"
            )

        samples: list[dict] = []
        autosave = bool(args.autosave)
        last_save_s = 0.0
        message = ""
        window = "charuco extrinsics"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        ref_idx = args.cameras.index(args.reference)

        while True:
            poses = {}
            views = []
            all_ok = True
            for cfg, cap in zip(configs, caps):
                ok, frame = cap.read()
                if not ok or frame is None:
                    frame = np.zeros((args.height, args.width, 3), dtype=np.uint8)
                    pose, viz, corners = None, frame, 0
                else:
                    pose, viz, corners = solve_board_pose(
                        board, detector, frame, cfg["K"], cfg["D"], args.min_corners
                    )
                if pose is None:
                    all_ok = False
                else:
                    poses[cfg["logical_name"]] = pose
                views.append(annotate(viz, cfg["logical_name"], corners, pose is not None))

            if all_ok and autosave and time.monotonic() - last_save_s > args.autosave_interval_s:
                ref_name = args.cameras[ref_idx]
                T_ref_board = poses[ref_name]
                rel = {}
                for name in args.cameras:
                    rel[name] = poses[name] @ inv_T(T_ref_board)
                samples.append({"T_camera_reference": rel})
                last_save_s = time.monotonic()
                message = f"autosaved {len(samples)}"

            shown = grid(views)
            status = (
                f"samples {len(samples)}/{args.samples}  "
                f"a=autosave {'on' if autosave else 'off'}  s=save  c=write  q=quit  {message}"
            )
            cv2.putText(shown, status, (14, shown.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(shown, status, (14, shown.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow(window, shown)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                print("[extrinsics] quit")
                return 1
            if key == ord("a"):
                autosave = not autosave
                message = f"autosave {'on' if autosave else 'off'}"
            if key == ord("s"):
                if not all_ok:
                    message = "need board detected in all cameras"
                else:
                    ref_name = args.cameras[ref_idx]
                    T_ref_board = poses[ref_name]
                    rel = {}
                    for name in args.cameras:
                        rel[name] = poses[name] @ inv_T(T_ref_board)
                    samples.append({"T_camera_reference": rel})
                    message = f"saved {len(samples)}"
            if key == ord("c") or (autosave and len(samples) >= args.samples):
                if len(samples) < 3:
                    message = "need at least 3 samples"
                    continue
                result = summarize(samples, args.cameras, args.reference)
                result["board"] = board_meta
                result["mapping_file"] = str(args.mapping.relative_to(PROJECT_ROOT))
                args.out.parent.mkdir(parents=True, exist_ok=True)
                args.out.write_text(json.dumps(result, indent=2) + "\n")
                print(f"[extrinsics] wrote {args.out}")
                for name, item in result["cameras"].items():
                    t = item["translation_m"]
                    s = item["translation_std_m"]
                    print(
                        f"[extrinsics] {name}: t=({t[0]:+.3f},{t[1]:+.3f},{t[2]:+.3f})m "
                        f"std=({s[0]:.4f},{s[1]:.4f},{s[2]:.4f})"
                    )
                return 0
    finally:
        for cap in caps:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())

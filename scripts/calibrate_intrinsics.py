#!/usr/bin/env python3
"""
calibrate_intrinsics.py - interactive ChArUco intrinsics calibration.

Use the US Letter board in calibration/boards. The script opens one camera,
draws detected ChArUco corners, and saves samples when you press `s`.
Press `c` to calibrate once enough samples have been collected.

Examples:
    uv run python scripts/calibrate_intrinsics.py --name front_narrow --index 0
    uv run python scripts/calibrate_intrinsics.py --name left --index 2 --samples 80
    uv run python scripts/calibrate_intrinsics.py --name front_narrow --index 0 --target-fov-deg 30 --source-fov-deg 85

Keys:
    s       save current detected board as a calibration sample
    a       toggle autosave
    c       calibrate and write JSON
    q/Esc   quit without calibrating
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
OUT_DIR_DEFAULT = PROJECT_ROOT / "calibration/cameras"
SAMPLES_DIR_DEFAULT = PROJECT_ROOT / "calibration/samples"

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
        raise RuntimeError(f"camera index {index} opened but did not return a frame")
    return cap


def load_board(meta_path: Path):
    meta = json.loads(meta_path.read_text())
    if meta.get("board_type") != "charuco":
        raise ValueError(f"unsupported board_type in {meta_path}: {meta.get('board_type')}")
    dictionary_name = meta["dictionary"]
    try:
        dictionary_id = getattr(cv2.aruco, dictionary_name)
    except AttributeError as e:
        raise ValueError(f"OpenCV has no aruco dictionary {dictionary_name}") from e
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
    board = cv2.aruco.CharucoBoard(
        (int(meta["squares_x"]), int(meta["squares_y"])),
        float(meta["square_size_mm"]) / 1000.0,
        float(meta["marker_size_mm"]) / 1000.0,
        dictionary,
    )
    detector = cv2.aruco.CharucoDetector(board)
    return meta, board, detector


def draw_status(
    frame: np.ndarray,
    name: str,
    sample_count: int,
    target_samples: int,
    corners_seen: int,
    autosave: bool,
    message: str,
) -> np.ndarray:
    out = frame.copy()
    h = out.shape[0]
    lines = [
        f"{name}  samples {sample_count}/{target_samples}  corners {corners_seen}",
        f"s=save  a=autosave {'on' if autosave else 'off'}  c=calibrate  q=quit",
    ]
    if message:
        lines.append(message)
    y = 28
    for line in lines:
        cv2.putText(out, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        y += 24
    cv2.putText(
        out,
        "Move board through center/corners/edges, close/far, tilted.",
        (14, h - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        out,
        "Move board through center/corners/edges, close/far, tilted.",
        (14, h - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )
    return out


def view_signature(corners: np.ndarray, image_size: tuple[int, int]) -> tuple[float, float, float]:
    pts = corners.reshape(-1, 2)
    w, h = image_size
    cx = float(np.mean(pts[:, 0]) / max(w, 1))
    cy = float(np.mean(pts[:, 1]) / max(h, 1))
    spread = float((np.ptp(pts[:, 0]) / max(w, 1)) * (np.ptp(pts[:, 1]) / max(h, 1)))
    return cx, cy, spread


def different_enough(sig: tuple[float, float, float], prev: tuple[float, float, float] | None) -> bool:
    if prev is None:
        return True
    return (
        abs(sig[0] - prev[0]) > 0.10
        or abs(sig[1] - prev[1]) > 0.10
        or abs(sig[2] - prev[2]) > 0.08
    )


def collect_sample(board, charuco_corners, charuco_ids):
    obj_points, img_points = board.matchImagePoints(charuco_corners, charuco_ids)
    return np.asarray(obj_points, dtype=np.float32), np.asarray(img_points, dtype=np.float32)


def per_view_errors(
    object_points: list[np.ndarray],
    image_points: list[np.ndarray],
    rvecs,
    tvecs,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> list[float]:
    errors: list[float] = []
    for obj, img, rvec, tvec in zip(object_points, image_points, rvecs, tvecs):
        projected, _ = cv2.projectPoints(obj, rvec, tvec, camera_matrix, dist_coeffs)
        e = cv2.norm(img, projected, cv2.NORM_L2) / math.sqrt(len(projected))
        errors.append(float(e))
    return errors


def derive_center_crop_profile(
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    width: int,
    height: int,
    ratio: float,
) -> dict:
    ratio = max(0.01, min(1.0, float(ratio)))
    crop_w = width * ratio
    crop_h = height * ratio
    x0 = (width - crop_w) / 2.0
    y0 = (height - crop_h) / 2.0
    sx = width / crop_w
    sy = height / crop_h

    k = camera_matrix.copy()
    k[0, 0] *= sx
    k[1, 1] *= sy
    k[0, 2] = (k[0, 2] - x0) * sx
    k[1, 2] = (k[1, 2] - y0) * sy

    return {
        "operation": "center_crop_then_resize_to_original",
        "crop_ratio": ratio,
        "crop_rect_px": {
            "x": x0,
            "y": y0,
            "width": crop_w,
            "height": crop_h,
        },
        "output_width": width,
        "output_height": height,
        "camera_matrix": k.tolist(),
        "dist_coeffs": dist_coeffs.reshape(-1).tolist(),
    }


def calibrate(
    object_points: list[np.ndarray],
    image_points: list[np.ndarray],
    image_size: tuple[int, int],
):
    flags = 0
    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None,
        flags=flags,
    )
    errors = per_view_errors(object_points, image_points, rvecs, tvecs, camera_matrix, dist_coeffs)
    return float(rms), camera_matrix, dist_coeffs, errors


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True, help="Logical camera name, e.g. front_narrow.")
    p.add_argument("--index", type=int, required=True, help="OpenCV/V4L2 camera index.")
    p.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    p.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    p.add_argument("--samples", type=int, default=60, help="Target number of samples before calibration.")
    p.add_argument("--min-corners", type=int, default=18, help="Minimum detected ChArUco corners for a sample.")
    p.add_argument("--board", type=Path, default=BOARD_META_DEFAULT)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR_DEFAULT)
    p.add_argument("--samples-dir", type=Path, default=SAMPLES_DIR_DEFAULT)
    p.add_argument("--no-fourcc", action="store_true", help="Do not force MJPG.")
    p.add_argument("--autosave", action="store_true", help="Start with autosave enabled.")
    p.add_argument("--crop-ratio", type=float, default=None, help="Optional center-crop ratio profile to save.")
    p.add_argument("--target-fov-deg", type=float, default=None, help="Optional desired cropped horizontal FOV.")
    p.add_argument("--source-fov-deg", type=float, default=85.0, help="Source horizontal FOV for --target-fov-deg.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sample_dir = args.samples_dir / args.name
    sample_dir.mkdir(parents=True, exist_ok=True)

    board_meta, board, detector = load_board(args.board)
    cap = open_camera(args.index, args.width, args.height, force_mjpg=not args.no_fourcc)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or args.width
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or args.height
    image_size = (actual_w, actual_h)

    print(f"[calib] camera={args.name} index={args.index} actual={actual_w}x{actual_h}")
    print(f"[calib] board={args.board}")
    print("[calib] keys: s save, a autosave, c calibrate, q quit")

    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    sample_meta: list[dict] = []
    autosave = bool(args.autosave)
    last_auto_s = 0.0
    last_sig: tuple[float, float, float] | None = None
    message = ""

    window = f"calibrate {args.name}"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                message = "no frame"
                time.sleep(0.02)
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray)
            corners_seen = 0 if charuco_ids is None else int(len(charuco_ids))
            viz = frame.copy()
            if marker_ids is not None and len(marker_ids) > 0:
                cv2.aruco.drawDetectedMarkers(viz, marker_corners, marker_ids)
            if charuco_ids is not None and corners_seen > 0:
                cv2.aruco.drawDetectedCornersCharuco(viz, charuco_corners, charuco_ids)

            now = time.monotonic()
            can_save = charuco_ids is not None and corners_seen >= args.min_corners
            if can_save:
                sig = view_signature(charuco_corners, image_size)
                if autosave and now - last_auto_s > 0.7 and different_enough(sig, last_sig):
                    obj, img = collect_sample(board, charuco_corners, charuco_ids)
                    object_points.append(obj)
                    image_points.append(img)
                    last_sig = sig
                    last_auto_s = now
                    sample_idx = len(object_points)
                    img_path = sample_dir / f"{args.name}_{sample_idx:03d}.jpg"
                    cv2.imwrite(str(img_path), frame)
                    sample_meta.append({
                        "sample": sample_idx,
                        "path": str(img_path.relative_to(PROJECT_ROOT)),
                        "corners": corners_seen,
                        "signature": sig,
                    })
                    message = f"autosaved sample {sample_idx}"

            shown = draw_status(viz, args.name, len(object_points), args.samples, corners_seen, autosave, message)
            cv2.imshow(window, shown)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                print("[calib] quit without calibration")
                return 1
            if key == ord("a"):
                autosave = not autosave
                message = f"autosave {'on' if autosave else 'off'}"
            if key == ord("s"):
                if not can_save:
                    message = f"need >= {args.min_corners} corners; saw {corners_seen}"
                    continue
                obj, img = collect_sample(board, charuco_corners, charuco_ids)
                object_points.append(obj)
                image_points.append(img)
                sig = view_signature(charuco_corners, image_size)
                last_sig = sig
                sample_idx = len(object_points)
                img_path = sample_dir / f"{args.name}_{sample_idx:03d}.jpg"
                cv2.imwrite(str(img_path), frame)
                sample_meta.append({
                    "sample": sample_idx,
                    "path": str(img_path.relative_to(PROJECT_ROOT)),
                    "corners": corners_seen,
                    "signature": sig,
                })
                message = f"saved sample {sample_idx}"
            if key == ord("c"):
                if len(object_points) < 15:
                    message = "need at least 15 samples before calibrating"
                    continue
                break
            if len(object_points) >= args.samples and autosave:
                message = "target reached; press c to calibrate"

        rms, camera_matrix, dist_coeffs, errors = calibrate(object_points, image_points, image_size)
        crop_ratio = args.crop_ratio
        fov_profile = None
        if args.target_fov_deg is not None:
            target = max(1.0, min(170.0, float(args.target_fov_deg)))
            source = max(1.0, min(170.0, float(args.source_fov_deg)))
            if target < source:
                crop_ratio = math.tan(math.radians(target / 2.0)) / math.tan(math.radians(source / 2.0))
                fov_profile = {"source_fov_deg": source, "target_fov_deg": target}
            else:
                print(f"[calib] target FOV {target:.1f} >= source {source:.1f}; no crop profile")

        result = {
            "camera_name": args.name,
            "camera_index": args.index,
            "model": "opencv_pinhole_brown",
            "image_width": actual_w,
            "image_height": actual_h,
            "board": board_meta,
            "sample_count": len(object_points),
            "rms_reprojection_error_px": rms,
            "per_view_error_px": {
                "mean": float(np.mean(errors)),
                "median": float(np.median(errors)),
                "max": float(np.max(errors)),
                "values": errors,
            },
            "camera_matrix": camera_matrix.tolist(),
            "dist_coeffs": dist_coeffs.reshape(-1).tolist(),
            "calibration_flags": 0,
            "samples": sample_meta,
        }
        if crop_ratio is not None and crop_ratio < 1.0:
            crop_profile = derive_center_crop_profile(camera_matrix, dist_coeffs, actual_w, actual_h, crop_ratio)
            if fov_profile is not None:
                crop_profile.update(fov_profile)
            result["center_crop_profile"] = crop_profile

        out_path = args.out_dir / f"{args.name}_intrinsics.json"
        out_path.write_text(json.dumps(result, indent=2) + "\n")
        print(f"[calib] wrote {out_path}")
        print(f"[calib] RMS={rms:.4f}px samples={len(object_points)}")
        if "center_crop_profile" in result:
            print(f"[calib] crop ratio={result['center_crop_profile']['crop_ratio']:.4f}")
        return 0
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())

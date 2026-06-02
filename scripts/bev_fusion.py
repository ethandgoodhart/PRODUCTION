"""Multi-camera BEV segmentation fusion.

Builds per-camera BevRemaps from an E2E calibration file and fuses
multiple cameras' segmentation maps into a single wider-FOV BEV.
"""
from __future__ import annotations

import cv2
import numpy as np

import seg_fast


def _slot_to_calib(e2e_calib: dict, slot_name: str, height_m: float,
                   range_fwd_ft: float, range_side_ft: float) -> dict:
    slot = None
    for s in e2e_calib.get("slots", []):
        if str(s.get("slot", "")).upper() == slot_name.upper():
            slot = s
            break
    if slot is None:
        raise ValueError(f"slot {slot_name!r} not found in calibration")
    intr = slot["intrinsics"]
    dist = slot.get("distortion_coeffs") or []
    fx = float(intr[0][0])
    fy = float(intr[1][1])
    cx = float(intr[0][2])
    cy = float(intr[1][2])
    image_size = slot.get("image_size") or e2e_calib.get("calibrated_image_size") or [640, 480]
    return {
        "intrinsics": {
            "model": "pinhole",
            "fx": fx, "fy": fy, "cx": cx, "cy": cy,
            "k1": float(dist[0]) if len(dist) > 0 else 0.0,
            "k2": float(dist[1]) if len(dist) > 1 else 0.0,
            "resolution": [int(image_size[0]), int(image_size[1])],
        },
        "extrinsics": {
            "height_m": float(height_m),
            "pitch_deg": float(slot.get("camera_pitch_deg", 0.0)),
            "roll_deg": float(slot.get("camera_roll_deg", 0.0)),
            "yaw_deg": float(slot.get("camera_yaw_deg", 0.0)),
            "ego_to_camera": slot.get("ego_to_camera"),
        },
        "bev_range": {
            "forward_ft": range_fwd_ft,
            "side_ft": range_side_ft,
        },
    }


def build_multi_cam_remaps(
    e2e_calib: dict,
    slots: list[str],
    img_h: int,
    img_w: int,
    bev_size: int,
    height_m: float,
    range_fwd_ft: float = 100.0,
    range_side_ft: float = 50.0,
) -> list[seg_fast.BevRemap]:
    remaps = []
    for slot_name in slots:
        calib = _slot_to_calib(e2e_calib, slot_name, height_m,
                               range_fwd_ft, range_side_ft)
        remap = seg_fast.build_bev_remap(calib, img_h, img_w, bev_size)
        remaps.append(remap)
    return remaps


def fused_bev_class_map(
    seg_maps: list[np.ndarray],
    remaps: list[seg_fast.BevRemap],
) -> np.ndarray:
    bev_size = remaps[0].bev_size
    out = np.full((bev_size, bev_size), 255, dtype=np.uint8)
    for seg_map, remap in zip(seg_maps, remaps):
        cls_ids = seg_map[remap.map_v, remap.map_u]
        unfilled = out == 255
        fill = remap.valid & unfilled
        out[fill] = np.clip(cls_ids[fill], 0, 254).astype(np.uint8)
    return out


def fused_bev_colored(
    seg_maps: list[np.ndarray],
    remaps: list[seg_fast.BevRemap],
    palette: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    bev_size = remaps[0].bev_size
    cls_map = fused_bev_class_map(seg_maps, remaps)
    combined_valid = np.zeros((bev_size, bev_size), dtype=bool)
    for remap in remaps:
        combined_valid |= remap.valid
    dilated = cv2.dilate(combined_valid.astype(np.uint8),
                         np.ones((3, 3), dtype=np.uint8)) > 0
    combined_border = dilated & (~combined_valid)

    bev = np.full((bev_size, bev_size, 3), 30, dtype=np.uint8)
    bev[combined_border] = (80, 80, 80)
    filled = cls_map != 255
    safe_ids = np.clip(cls_map, 0, len(palette) - 1)
    np.copyto(bev, palette[safe_ids], where=filled[..., None])

    ex, ey = bev_size // 2, bev_size - 8
    ego_pts = np.array([[ex, ey - 14], [ex - 7, ey], [ex + 7, ey]], dtype=np.int32)
    cv2.fillPoly(bev, [ego_pts], (255, 255, 255))

    return bev, cls_map

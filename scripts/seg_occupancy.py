#!/usr/bin/env python3
"""Predicted future-occupancy obstacle tracking + environment braking.

Vendored, self-contained (numpy + cv2 only) port of the drive-by-segmentation
``unified-planner`` collision-avoidance stack:

  * ``predicted_occupancy.py``  — BEV obstacle blobs -> nearest-neighbour tracks
    -> constant-velocity future occupancy -> risk mask.
  * the brake helpers from ``render_trajectories.py`` — trajectory/risk conflict
    fraction, on-path semantic (VRU / sign) cues, graded pedal target, and a
    smoother.

This replaces the old corridor-based ``evaluate_protective_stop`` + UniAD
motion predictions. Moving objects (pedestrians, riders, bikes, vehicles) are
detected from the SegFormer BEV class map, tracked to estimate velocity, and
extrapolated forward; the planner's trajectory is braked in proportion to how
much of it conflicts with predicted obstacle occupancy, with an immediate
semantic override for vulnerable road users sitting directly on the path.

Coordinate convention matches the cart's BEV: ego at bottom-centre, +forward
toward the top of the image. ``BevGeometry`` is constructed directly from the
sidecar's ``rt.BEV_SIZE`` / ``rt.RANGE_FWD`` / ``rt.RANGE_SIDE`` so risk-mask
pixels line up 1:1 with the planned trajectory polyline.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

FT_TO_M = 0.3048

# Cityscapes / SegFormer class IDs that count as physical obstacles.
OBSTACLE_CLASS_IDS = (11, 12, 13, 14, 15, 17, 18)  # person, rider, vehicles, motorcycle, bicycle
# Vulnerable road users that force an immediate stop when on the planned path.
SEMANTIC_IMMEDIATE_STOP_CLASS_IDS = frozenset({11, 12, 18})  # person, rider, bicycle
SEMANTIC_SIGN_AND_LIGHT_IDS = frozenset({6, 7})              # traffic light, traffic sign


@dataclass(frozen=True)
class BevGeometry:
    """Metric BEV geometry. +forward, +left; ego at bottom-centre of the image."""

    bev_size: int
    range_fwd_m: float
    range_side_m: float
    range_fwd_ft: float
    range_side_ft: float

    @property
    def meters_per_px_fwd(self) -> float:
        return self.range_fwd_m / self.bev_size

    @property
    def meters_per_px_side(self) -> float:
        return (2.0 * self.range_side_m) / self.bev_size

    @property
    def px_per_meter_side(self) -> float:
        return self.bev_size / (2.0 * self.range_side_m)

    @classmethod
    def from_ranges(cls, bev_size: int, range_fwd_m: float, range_side_m: float) -> "BevGeometry":
        return cls(
            bev_size=int(bev_size),
            range_fwd_m=float(range_fwd_m),
            range_side_m=float(range_side_m),
            range_fwd_ft=float(range_fwd_m) / FT_TO_M,
            range_side_ft=float(range_side_m) / FT_TO_M,
        )

    def bev_to_local(self, bx, by):
        bx_arr = np.asarray(bx, dtype=np.float32)
        by_arr = np.asarray(by, dtype=np.float32)
        local_left = (bx_arr / self.bev_size - 0.5) * 2.0 * self.range_side_m
        local_fwd = (1.0 - by_arr / self.bev_size) * self.range_fwd_m
        return local_fwd, local_left

    def local_to_bev(self, local_fwd, local_left, as_int: bool = False):
        fwd = np.asarray(local_fwd, dtype=np.float32)
        left = np.asarray(local_left, dtype=np.float32)
        bx = (left / self.range_side_m * 0.5 + 0.5) * self.bev_size
        by = (1.0 - fwd / self.range_fwd_m) * self.bev_size
        if as_int:
            return bx.astype(np.int32), by.astype(np.int32)
        return bx, by


@dataclass
class _Track:
    track_id: int
    pos_m: np.ndarray
    vel_mps: np.ndarray
    age: int = 1
    hits: int = 1
    missed: int = 0
    confidence: float = 0.0


@dataclass(frozen=True)
class MotionDetection:
    fwd_m: float
    left_m: float
    confidence: float = 1.0


@dataclass(frozen=True)
class TrackPrediction:
    track_id: int
    pos_m: tuple[float, float]
    vel_mps: tuple[float, float]
    future_m: tuple[tuple[float, float], ...]
    confidence: float


@dataclass(frozen=True)
class PredictedOccupancy:
    probabilities: np.ndarray
    risk_mask: np.ndarray
    current_mask: np.ndarray
    track_count: int
    tracks: tuple[TrackPrediction, ...] = ()


class PredictedOccupancyTracker:
    """Track obstacle components in metric BEV and predict future occupancy.

    This is intentionally conservative: it does not need object identities to be
    perfect. Any plausible moving obstacle expands into a future risk mask that
    can be removed from the static road mask before planning.
    """

    def __init__(
        self,
        horizon_steps: int = 8,
        step_s: float = 0.35,
        min_area_px: int = 8,
        max_match_m: float = 2.5,
        obstacle_radius_m: float = 0.25,
        cart_half_width_m: float = 0.30,
        uncertainty_growth_mps: float = 0.06,
        reaction_time_s: float = 0.2,
        risk_threshold: float = 0.65,
        planning_horizon_steps: int = 3,
        min_track_hits: int = 3,
        min_track_confidence: float = 0.62,
        min_motion_speed_mps: float = 0.25,
        max_motion_speed_mps: float = 8.0,
    ):
        self.horizon_steps = max(1, int(horizon_steps))
        self.step_s = max(0.05, float(step_s))
        self.min_area_px = max(1, int(min_area_px))
        self.max_match_m = float(max_match_m)
        self.obstacle_radius_m = float(obstacle_radius_m)
        self.cart_half_width_m = float(cart_half_width_m)
        self.uncertainty_growth_mps = float(uncertainty_growth_mps)
        self.reaction_time_s = float(reaction_time_s)
        self.risk_threshold = float(risk_threshold)
        self.planning_horizon_steps = max(1, int(planning_horizon_steps))
        self.min_track_hits = max(1, int(min_track_hits))
        self.min_track_confidence = float(min_track_confidence)
        self.min_motion_speed_mps = float(min_motion_speed_mps)
        self.max_motion_speed_mps = float(max_motion_speed_mps)
        self._tracks: list[_Track] = []
        self._next_track_id = 1

    def reset(self) -> None:
        """Clear internal tracks (call on video restart / ego reset)."""
        self._tracks.clear()
        self._next_track_id = 1

    def update(
        self,
        class_map: np.ndarray,
        geom: BevGeometry,
        dt_s: float,
        ego_speed_mps: float = 0.0,
        extra_detections_m: list[tuple[float, float] | MotionDetection] | None = None,
        use_segmentation_obstacles: bool = True,
    ) -> PredictedOccupancy:
        detections = self._components_to_detections(class_map, geom) if use_segmentation_obstacles else []
        detection_confidences = [0.55] * len(detections)
        if extra_detections_m:
            for det in extra_detections_m:
                if isinstance(det, MotionDetection):
                    fwd_m, left_m, det_conf = det.fwd_m, det.left_m, det.confidence
                else:
                    fwd_m, left_m = det
                    det_conf = 1.0
                if fwd_m > 0.0 and abs(left_m) <= geom.range_side_m and fwd_m <= geom.range_fwd_m:
                    detections.append(np.array([float(fwd_m), float(left_m)], dtype=np.float32))
                    detection_confidences.append(float(np.clip(det_conf, 0.0, 1.0)))
        self._update_tracks(detections, detection_confidences, max(1e-3, float(dt_s)))
        probabilities = self._render_predictions(class_map.shape, geom, max(0.0, float(ego_speed_mps)))
        current_mask = np.isin(class_map, OBSTACLE_CLASS_IDS)
        planning_steps = min(self.planning_horizon_steps, probabilities.shape[0])
        risk_mask = np.max(probabilities[:planning_steps], axis=0) >= self.risk_threshold
        return PredictedOccupancy(
            probabilities=probabilities,
            risk_mask=risk_mask,
            current_mask=current_mask,
            track_count=len(self._tracks),
            tracks=self._track_predictions(geom),
        )

    def _components_to_detections(self, class_map: np.ndarray, geom: BevGeometry) -> list[np.ndarray]:
        obstacle_mask = np.isin(class_map, OBSTACLE_CLASS_IDS).astype(np.uint8)
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(obstacle_mask, connectivity=8)
        detections: list[np.ndarray] = []
        for label in range(1, n_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < self.min_area_px:
                continue
            cx, cy = centroids[label]
            fwd, left = geom.bev_to_local(cx, cy)
            if float(fwd) <= 0.0:
                continue
            detections.append(np.array([float(fwd), float(left)], dtype=np.float32))
        return detections

    def _update_tracks(self, detections: list[np.ndarray], detection_confidences: list[float], dt_s: float) -> None:
        unmatched_tracks = set(range(len(self._tracks)))
        unmatched_detections = set(range(len(detections)))
        if self._tracks and detections:
            pairs = []
            for ti, track in enumerate(self._tracks):
                for di, det in enumerate(detections):
                    dist = float(np.linalg.norm(det - track.pos_m))
                    if dist <= self.max_match_m:
                        pairs.append((dist, ti, di))
            for _, ti, di in sorted(pairs):
                if ti not in unmatched_tracks or di not in unmatched_detections:
                    continue
                track = self._tracks[ti]
                disp = detections[di] - track.pos_m
                dn = float(np.linalg.norm(disp))
                # Segmentation centroids can jitter pixel-to-pixel; don't invent multi-m/s motion.
                max_jump_m = max(0.35, 12.0 * dt_s)
                if dn > max_jump_m and dn > 1e-6:
                    disp = disp * (max_jump_m / dn)
                measured_vel = disp / dt_s
                det_conf = detection_confidences[di] if di < len(detection_confidences) else 1.0
                alpha = 0.25 + 0.25 * det_conf
                track.vel_mps = (1.0 - alpha) * track.vel_mps + alpha * measured_vel
                speed = float(np.linalg.norm(track.vel_mps))
                if speed > self.max_motion_speed_mps:
                    track.vel_mps *= self.max_motion_speed_mps / max(speed, 1e-6)
                track.pos_m = detections[di]
                track.age += 1
                track.hits += 1
                track.missed = 0
                track.confidence = min(
                    1.0,
                    0.72 * track.confidence
                    + 0.18 * det_conf
                    + 0.10 * min(track.hits / self.min_track_hits, 1.0),
                )
                unmatched_tracks.remove(ti)
                unmatched_detections.remove(di)

        for ti in unmatched_tracks:
            self._tracks[ti].missed += 1
            self._tracks[ti].pos_m = self._tracks[ti].pos_m + self._tracks[ti].vel_mps * dt_s
            self._tracks[ti].confidence *= 0.55

        for di in unmatched_detections:
            self._tracks.append(
                _Track(
                    track_id=self._next_track_id,
                    pos_m=detections[di],
                    vel_mps=np.zeros(2, dtype=np.float32),
                    confidence=0.25 * (detection_confidences[di] if di < len(detection_confidences) else 1.0),
                )
            )
            self._next_track_id += 1

        self._tracks = [track for track in self._tracks if track.missed <= 4]

    def _track_predictions(self, geom: BevGeometry) -> tuple[TrackPrediction, ...]:
        out: list[TrackPrediction] = []
        for track in self._tracks:
            if not self._track_is_predictable(track):
                continue
            future = []
            for step in range(self.horizon_steps):
                t_s = (step + 1) * self.step_s
                fwd_m, left_m = track.pos_m + track.vel_mps * t_s
                if fwd_m <= 0.0 or fwd_m > geom.range_fwd_m or abs(left_m) > geom.range_side_m:
                    continue
                future.append((float(fwd_m), float(left_m)))
            out.append(
                TrackPrediction(
                    track_id=track.track_id,
                    pos_m=(float(track.pos_m[0]), float(track.pos_m[1])),
                    vel_mps=(float(track.vel_mps[0]), float(track.vel_mps[1])),
                    future_m=tuple(future),
                    confidence=float(track.confidence),
                )
            )
        return tuple(out)

    def _track_is_predictable(self, track: _Track) -> bool:
        speed = float(np.linalg.norm(track.vel_mps))
        return (
            track.hits >= self.min_track_hits
            and track.missed == 0
            and track.confidence >= self.min_track_confidence
            and self.min_motion_speed_mps <= speed <= self.max_motion_speed_mps
        )

    def _render_predictions(self, shape: tuple[int, int], geom: BevGeometry, ego_speed_mps: float) -> np.ndarray:
        masks = np.zeros((self.horizon_steps, shape[0], shape[1]), dtype=np.float32)
        radius_base_m = self.obstacle_radius_m + self.cart_half_width_m + ego_speed_mps * self.reaction_time_s
        px_per_m = max(geom.px_per_meter_side, 1.0 / max(geom.meters_per_px_fwd, 1e-3))
        for track in self._tracks:
            if not self._track_is_predictable(track):
                continue
            for step in range(self.horizon_steps):
                t_s = (step + 1) * self.step_s
                pred_fwd, pred_left = track.pos_m + track.vel_mps * t_s
                if pred_fwd <= 0.0 or pred_fwd > geom.range_fwd_m:
                    continue
                bx, by = geom.local_to_bev(pred_fwd, pred_left)
                bx_i = int(round(float(bx)))
                by_i = int(round(float(by)))
                if bx_i < 0 or bx_i >= geom.bev_size or by_i < 0 or by_i >= geom.bev_size:
                    continue
                radius_m = radius_base_m + self.uncertainty_growth_mps * t_s
                radius_px = max(2, int(round(radius_m * px_per_m)))
                risk = max(0.2, track.confidence * (1.0 - 0.08 * step))
                cv2.circle(masks[step], (bx_i, by_i), radius_px, float(risk), -1, cv2.LINE_AA)
        return np.clip(masks, 0.0, 1.0)


# --------------------------------------------------------------------------- #
# Brake decision: planned-path vs risk-mask conflict + on-path semantic cues.
# --------------------------------------------------------------------------- #
def trajectory_conflict_frac(
    traj_bev: np.ndarray | None,
    risk_mask: np.ndarray | None,
    *,
    lateral_half_width_px: int = 5,
    max_points: int = 40,
) -> float:
    """Fraction of near-term BEV trajectory samples that intersect predicted occupancy risk."""
    if traj_bev is None or risk_mask is None:
        return 0.0
    tr = np.asarray(traj_bev, dtype=np.int32)
    if tr.ndim != 2 or tr.shape[0] < 2:
        return 0.0
    h, w = risk_mask.shape
    n_take = min(int(max_points), tr.shape[0])
    hits = 0
    for i in range(n_take):
        bx, by = int(tr[i, 0]), int(tr[i, 1])
        if not (0 <= by < h and 0 <= bx < w):
            continue
        y0 = max(0, by - 2)
        y1 = min(h, by + 3)
        x0 = max(0, bx - lateral_half_width_px)
        x1 = min(w, bx + lateral_half_width_px + 1)
        patch = risk_mask[y0:y1, x0:x1]
        if patch.size and np.any(patch):
            hits += 1
    return float(hits / max(n_take, 1))


def trajectory_semantic_stop_fractions(
    traj_bev: np.ndarray | None,
    class_map: np.ndarray | None,
    *,
    lateral_half_width_px: int = 8,
    max_points: int = 40,
    immediate_stop_ids: frozenset[int] | None = None,
    sign_light_ids: frozenset[int] | None = None,
    unknown_class_id: int = 255,
) -> tuple[float, float]:
    """Approximate fractions of planned path corridor covered by pedestrians / signs & signals.

    Cityscapes lumps all signage into ``traffic sign``; we bias toward braking when signs
    or lights occupy the ego trajectory ahead without requiring a dedicated stop-sign ID.
    """
    if traj_bev is None or class_map is None:
        return 0.0, 0.0

    stops = SEMANTIC_IMMEDIATE_STOP_CLASS_IDS if immediate_stop_ids is None else immediate_stop_ids
    signage = SEMANTIC_SIGN_AND_LIGHT_IDS if sign_light_ids is None else sign_light_ids

    tr = np.asarray(traj_bev, dtype=np.int32)
    if tr.ndim != 2 or tr.shape[0] < 2:
        return 0.0, 0.0
    cls = np.asarray(class_map, dtype=np.uint8)
    h, w = cls.shape[:2]

    if stops & signage:
        raise ValueError("immediate_stop_ids and sign_light_ids must not overlap")

    n_take = min(int(max_points), tr.shape[0])
    person_hits = 0
    sign_hits = 0
    used = 0
    for i in range(n_take):
        bx, by = int(tr[i, 0]), int(tr[i, 1])
        if not (0 <= by < h and 0 <= bx < w):
            continue
        y0 = max(0, by - 2)
        y1 = min(h, by + 3)
        x0 = max(0, bx - lateral_half_width_px)
        x1 = min(w, bx + lateral_half_width_px + 1)
        patch = cls[y0:y1, x0:x1]
        if patch.size == 0:
            continue
        used += 1
        pv = patch[patch != unknown_class_id]
        if pv.size == 0:
            continue
        if np.any(np.isin(pv, list(stops))):
            person_hits += 1
            continue  # vuln VRU dominates this sample — don't double-count signage
        if np.any(np.isin(pv, list(signage))):
            sign_hits += 1
    denom = max(used, 1)
    return float(person_hits / denom), float(sign_hits / denom)


def pedal_brake_from_conflict(
    conflict_frac: float,
    *,
    bleed: float = 0.06,
    span: float = 0.26,
    cap: float = 1.0,
) -> float:
    """Map conflict rate along path to commanded brake pedal depth in [0, 1]."""
    if conflict_frac <= bleed:
        return 0.0
    depth = min(cap, float((conflict_frac - bleed) / max(span, 1e-6)))
    return max(0.0, min(1.0, depth))


def environment_brake_target(
    occupancy_path_frac: float,
    traj_bev: np.ndarray | None,
    class_map: np.ndarray | None,
    *,
    lateral_half_width_px: int = 8,
    max_semantic_samples: int = 42,
) -> float:
    """Combine motion-occupancy path conflict with BEV semantic cues (people, signals, signs).

    Intended for golf-cart speeds: pedestrians / riders / bikes on-plan → aggressive brake;
    traffic lights/signs on-plan → biased stop (Cityscapes cannot isolate stop-only signs).
    """
    brake = pedal_brake_from_conflict(occupancy_path_frac)
    if class_map is None:
        return float(min(1.0, brake))

    crit, signage = trajectory_semantic_stop_fractions(
        traj_bev,
        class_map,
        lateral_half_width_px=lateral_half_width_px,
        max_points=max_semantic_samples,
    )

    brake = max(brake, pedal_brake_from_conflict(crit * 3.5, bleed=0.02, span=0.10))
    if crit >= 0.07:
        brake = max(brake, 1.0)

    brake = max(brake, pedal_brake_from_conflict(signage * 2.0, bleed=0.05, span=0.24))
    if signage >= 0.10:
        brake = max(brake, 0.92)
    return float(min(1.0, brake))


class PedalCommandSmoother:
    """Exponential smoothing for brake pedal to avoid twitchy STOP/GO labels."""

    def __init__(self, alpha: float = 0.32):
        self.alpha = float(np.clip(alpha, 0.02, 0.98))
        self.brake_smooth = 0.0

    def step(self, brake_target: float) -> tuple[float, float]:
        bw = float(np.clip(brake_target, 0.0, 1.0))
        self.brake_smooth = (1.0 - self.alpha) * self.brake_smooth + self.alpha * bw
        self.brake_smooth = float(np.clip(self.brake_smooth, 0.0, 1.0))
        throttle = float(np.clip(1.0 - self.brake_smooth, 0.0, 1.0))
        return throttle, self.brake_smooth

    def snapshot(self) -> tuple[float, float]:
        """Latest smoothed throttle and brake (called after zero or more step() calls)."""
        return (
            float(np.clip(1.0 - self.brake_smooth, 0.0, 1.0)),
            float(self.brake_smooth),
        )

    def reset(self, brake: float = 0.0) -> None:
        self.brake_smooth = float(np.clip(brake, 0.0, 1.0))

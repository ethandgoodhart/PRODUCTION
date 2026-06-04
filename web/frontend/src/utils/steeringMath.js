const PATH_WHEELBASE_M = 0.8;
const PATH_STEERING_COLUMN_RATIO = 15.0;
const PATH_STEER_GAIN = 1.6;
const PATH_STEER_FIT_MIN_M = 1.0;
const PATH_STEER_FIT_MAX_M = 10.0;

export function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

export function smoothPathPoints(points, win = 5) {
  if (points.length < win) return points;
  const half = Math.floor(win / 2);
  return points.map((_, i) => {
    let sx = 0;
    let sy = 0;
    for (let k = 0; k < win; k++) {
      const idx = clamp(i + k - half, 0, points.length - 1);
      sx += points[idx][0];
      sy += points[idx][1];
    }
    return [sx / win, sy / win];
  });
}

export function lookaheadPointFromPath(points, lookaheadM) {
  const target = clamp(Number(lookaheadM) || 2.5, 1.0, 10.0);
  let prev = [0, 0];
  let traveled = 0;
  for (const cur of points) {
    const seg = Math.hypot(cur[0] - prev[0], cur[1] - prev[1]);
    if (traveled + seg >= target && seg > 1e-6) {
      const t = (target - traveled) / seg;
      return [
        prev[0] + t * (cur[0] - prev[0]),
        prev[1] + t * (cur[1] - prev[1]),
      ];
    }
    traveled += seg;
    prev = cur;
  }
  return points.length ? points[points.length - 1] : null;
}

export function wheelDegFromDisplayedPath(path, lookaheadM) {
  if (!Array.isArray(path) || path.length < 2) return null;
  let points = path
    .map((p) => [Number(p && p[0]), Number(p && p[1])])
    .filter(
      (p) =>
        Number.isFinite(p[0]) &&
        Number.isFinite(p[1]) &&
        p[0] > PATH_STEER_FIT_MIN_M &&
        p[0] < PATH_STEER_FIT_MAX_M
    );
  if (points.length < 2) return null;
  points = smoothPathPoints(points, 5);
  const p = lookaheadPointFromPath(points, lookaheadM);
  if (!p) return null;
  const ld = Math.hypot(p[0], p[1]);
  if (ld < 1e-3) return null;
  const alpha = Math.atan2(p[1], p[0]);
  const delta = Math.atan2(2.0 * PATH_WHEELBASE_M * Math.sin(alpha), ld);
  const columnDeg =
    ((delta * 180) / Math.PI) * PATH_STEERING_COLUMN_RATIO * PATH_STEER_GAIN;
  return clamp(columnDeg, -270.0, 270.0);
}

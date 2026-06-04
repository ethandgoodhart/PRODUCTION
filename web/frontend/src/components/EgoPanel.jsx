import React, { useRef, useEffect } from 'react';

const EGO_TRAIL_SECONDS = 6.0;
const EGO_TRAIL_MAX = 80;
const EGO_HALF_MIN = 1.0;
const EGO_CAM_AZ = (32 * Math.PI) / 180;
const EGO_CAM_EL = (18 * Math.PI) / 180;
const EGO_COS_AZ = Math.cos(EGO_CAM_AZ);
const EGO_SIN_AZ = Math.sin(EGO_CAM_AZ);
const EGO_COS_EL = Math.cos(EGO_CAM_EL);
const EGO_SIN_EL = Math.sin(EGO_CAM_EL);

function isoProject(x, y, z, half, w, h) {
  const nx = x / half;
  const ny = y / half;
  const nz = -z / half;
  const x1 = nx * EGO_COS_AZ + nz * EGO_SIN_AZ;
  const y1 = ny;
  const z1 = -nx * EGO_SIN_AZ + nz * EGO_COS_AZ;
  const x2 = x1;
  const y2 = y1 * EGO_COS_EL - z1 * EGO_SIN_EL;
  const cx = w / 2;
  const cy = h / 2 + 4;
  const scale = Math.min(w, h) * 0.34;
  return [cx + x2 * scale, cy - y2 * scale];
}

function drawEgoCube(ctx, half, w, h) {
  const C = [
    [-1, -1, -1], [+1, -1, -1], [+1, +1, -1], [-1, +1, -1],
    [-1, -1, +1], [+1, -1, +1], [+1, +1, +1], [-1, +1, +1],
  ].map(([a, b, c]) => isoProject(a * half, b * half, c * half, half, w, h));
  const E = [
    [0, 1], [1, 2], [2, 3], [3, 0],
    [4, 5], [5, 6], [6, 7], [7, 4],
    [0, 4], [1, 5], [2, 6], [3, 7],
  ];
  ctx.lineWidth = 1;
  ctx.strokeStyle = 'rgba(60, 70, 90, 0.45)';
  ctx.beginPath();
  for (const [a, b] of E) {
    ctx.moveTo(C[a][0], C[a][1]);
    ctx.lineTo(C[b][0], C[b][1]);
  }
  ctx.stroke();

  ctx.strokeStyle = 'rgba(60, 70, 90, 0.14)';
  ctx.beginPath();
  const N = 4;
  for (let i = 1; i < N; i++) {
    const t = -half + 2 * half * (i / N);
    let p1 = isoProject(t, -half, -half, half, w, h);
    let p2 = isoProject(t, -half, +half, half, w, h);
    ctx.moveTo(p1[0], p1[1]);
    ctx.lineTo(p2[0], p2[1]);
    p1 = isoProject(-half, -half, t, half, w, h);
    p2 = isoProject(+half, -half, t, half, w, h);
    ctx.moveTo(p1[0], p1[1]);
    ctx.lineTo(p2[0], p2[1]);
  }
  ctx.stroke();

  const o = isoProject(0, 0, 0, half, w, h);
  ctx.fillStyle = 'rgba(60, 70, 90, 0.35)';
  ctx.beginPath();
  ctx.arc(o[0], o[1], 1.6, 0, Math.PI * 2);
  ctx.fill();
}

export default function EgoPanel({ state }) {
  const canvasRef = useRef(null);
  const trailRef = useRef([]);
  const lastTsRef = useRef(null);

  const connected = !!(state && state.ego_connected);
  const sample = state?.ego?.sample;

  useEffect(() => {
    if (
      sample &&
      typeof sample.t_s === 'number' &&
      sample.t_s !== lastTsRef.current
    ) {
      lastTsRef.current = sample.t_s;
      trailRef.current.push({
        x: Number(sample.x_m) || 0,
        y: Number(sample.y_m) || 0,
        z: Number(sample.z_m) || 0,
        t_s: sample.t_s,
      });
      const cutoff = sample.t_s - EGO_TRAIL_SECONDS;
      while (trailRef.current.length > 0 && trailRef.current[0].t_s < cutoff)
        trailRef.current.shift();
      while (trailRef.current.length > EGO_TRAIL_MAX)
        trailRef.current.shift();
    }

    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const cssW = canvas.clientWidth || 220;
    const cssH = canvas.clientHeight || 180;
    if (canvas.width !== cssW * dpr || canvas.height !== cssH * dpr) {
      canvas.width = cssW * dpr;
      canvas.height = cssH * dpr;
      ctx.scale(dpr, dpr);
    }
    const w = cssW;
    const h = cssH;

    ctx.clearRect(0, 0, w, h);
    const trail = trailRef.current;
    const last = trail.length > 0 ? trail[trail.length - 1] : null;
    const cx0 = last ? last.x : 0;
    const cy0 = last ? last.y : 0;
    const cz0 = last ? last.z : 0;

    let half = EGO_HALF_MIN;
    for (const p of trail) {
      half = Math.max(half, Math.abs(p.x - cx0), Math.abs(p.y - cy0), Math.abs(p.z - cz0));
    }
    half *= 1.15;

    drawEgoCube(ctx, half, w, h);

    if (trail.length >= 2) {
      ctx.lineWidth = 1.6;
      ctx.strokeStyle = 'rgba(216, 74, 74, 0.85)';
      ctx.beginPath();
      const p0 = isoProject(trail[0].x - cx0, trail[0].y - cy0, trail[0].z - cz0, half, w, h);
      ctx.moveTo(p0[0], p0[1]);
      for (let i = 1; i < trail.length; i++) {
        const p = isoProject(trail[i].x - cx0, trail[i].y - cy0, trail[i].z - cz0, half, w, h);
        ctx.lineTo(p[0], p[1]);
      }
      ctx.stroke();
    }

    const p = isoProject(0, 0, 0, half, w, h);
    ctx.fillStyle = '#111215';
    ctx.beginPath();
    ctx.arc(p[0], p[1], 3.4, 0, Math.PI * 2);
    ctx.fill();
  }, [sample]);

  const egoX = sample ? (Number(sample.x_m) || 0).toFixed(2) : '—';
  const egoY = sample ? (Number(sample.y_m) || 0).toFixed(2) : '—';
  const egoZ = sample ? (Number(sample.z_m) || 0).toFixed(2) : '—';
  const egoSpeed = sample
    ? ((Number(sample.speed_mps) || 0) * 2.23694).toFixed(2)
    : '—';
  const egoYaw = sample
    ? (((Number(sample.yaw_rad) || 0) * 180) / Math.PI).toFixed(1)
    : '—';
  const egoCurv = sample
    ? (Number(sample.curvature_inv_m) || 0).toFixed(3)
    : '—';

  return (
    <div className="fixed top-[86px] right-3 w-[232px] p-[10px_12px_8px] bg-white/86 backdrop-blur-md border border-line rounded-[14px] z-5 text-[10px] text-ink-soft tracking-[0.05em]">
      <div className="flex items-center justify-between font-semibold text-[9.5px] uppercase text-muted mb-1">
        <span>EGO MOTION</span>
        <span
          className={`w-1.5 h-1.5 rounded-full transition-all duration-[180ms] ${
            connected
              ? 'bg-[#7ed488] shadow-[0_0_0_1.5px_rgba(126,212,136,0.18)]'
              : 'bg-[#d84a4a]'
          }`}
        />
      </div>
      <canvas
        className="block w-full h-[180px] bg-[rgba(245,247,250,0.6)] rounded-[10px]"
        ref={canvasRef}
        width="220"
        height="180"
      />
      <div className="grid grid-cols-2 gap-[2px_10px] mt-1.5 tabular-nums text-[11px]">
        {[
          { k: 'x', v: egoX, u: 'm' },
          { k: 'y', v: egoY, u: 'm' },
          { k: 'z', v: egoZ, u: 'm' },
          { k: 'speed', v: egoSpeed, u: 'mph' },
          { k: 'yaw', v: egoYaw, u: '°' },
          { k: 'curv', v: egoCurv, u: '/m' },
        ].map(({ k, v, u }) => (
          <div key={k} className="flex items-baseline gap-1">
            <span className="text-muted font-medium min-w-[32px]">{k}</span>
            <span className="text-ink font-medium ml-auto">{v}</span>
            <span className="text-muted text-[9.5px]">{u}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

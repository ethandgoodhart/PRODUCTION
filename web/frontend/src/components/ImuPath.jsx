import React, { useEffect, useRef, useState } from 'react';

const TRAIL_SECONDS = 15;

export default function ImuPath({ state }) {
  const canvasRef = useRef(null);
  const traceRef = useRef(null);
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    fetch('/ego-trace')
      .then((r) => r.json())
      .then((data) => {
        traceRef.current = data;
        setLoaded(true);
      })
      .catch(() => setLoaded(false));
  }, []);

  const videoPos = state?.autoware?.video?.position_s ?? null;

  useEffect(() => {
    const canvas = canvasRef.current;
    const trace = traceRef.current;
    if (!canvas || !trace || !trace.length || videoPos == null) return;

    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const cssW = canvas.clientWidth;
    const cssH = canvas.clientHeight;
    if (canvas.width !== cssW * dpr || canvas.height !== cssH * dpr) {
      canvas.width = cssW * dpr;
      canvas.height = cssH * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }
    const w = cssW;
    const h = cssH;
    ctx.clearRect(0, 0, w, h);

    // Find current index
    let curIdx = 0;
    for (let i = 0; i < trace.length; i++) {
      if (trace[i].t <= videoPos) curIdx = i;
      else break;
    }

    const cur = trace[curIdx];

    // Compute bounds of full trace for consistent scale
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (const p of trace) {
      if (p.x < minX) minX = p.x;
      if (p.x > maxX) maxX = p.x;
      if (p.y < minY) minY = p.y;
      if (p.y > maxY) maxY = p.y;
    }
    const rangeX = maxX - minX || 1;
    const rangeY = maxY - minY || 1;
    const cx = cur.x;
    const cy = cur.y;
    const pad = 20;
    const plotW = w - pad * 2;
    const plotH = h - pad * 2;
    const scale = Math.min(plotW / rangeX, plotH / rangeY) * 2.7;

    function toScreen(x, y) {
      return [
        w / 2 - (x - cx) * scale,
        h / 2 - (y - cy) * scale,
      ];
    }

    // Full trace (dim)
    ctx.strokeStyle = 'rgba(255,255,255,0.12)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    const [fx, fy] = toScreen(trace[0].x, trace[0].y);
    ctx.moveTo(fx, fy);
    const step = Math.max(1, Math.floor(trace.length / 500));
    for (let i = step; i < trace.length; i += step) {
      const [px, py] = toScreen(trace[i].x, trace[i].y);
      ctx.lineTo(px, py);
    }
    ctx.stroke();

    // Trail
    const trailStart = videoPos - TRAIL_SECONDS;
    let startIdx = curIdx;
    for (let i = curIdx; i >= 0; i--) {
      if (trace[i].t < trailStart) break;
      startIdx = i;
    }
    const pts = trace.slice(startIdx, curIdx + 1);

    if (pts.length >= 2) {
      for (let i = 1; i < pts.length; i++) {
        const age = (cur.t - pts[i].t) / TRAIL_SECONDS;
        const alpha = 0.3 + 0.7 * (1 - age);
        ctx.strokeStyle = `rgba(59, 130, 246, ${alpha.toFixed(2)})`;
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        const [x1, y1] = toScreen(pts[i - 1].x, pts[i - 1].y);
        const [x2, y2] = toScreen(pts[i].x, pts[i].y);
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();
      }
    }

    // History path (from start to trail start)
    if (startIdx > 1) {
      ctx.strokeStyle = 'rgba(59, 130, 246, 0.2)';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      const [hx, hy] = toScreen(trace[0].x, trace[0].y);
      ctx.moveTo(hx, hy);
      const hStep = Math.max(1, Math.floor(startIdx / 200));
      for (let i = hStep; i <= startIdx; i += hStep) {
        const [px, py] = toScreen(trace[i].x, trace[i].y);
        ctx.lineTo(px, py);
      }
      ctx.stroke();
    }

    // Current position
    const [sx, sy] = toScreen(cur.x, cur.y);
    ctx.fillStyle = '#3b82f6';
    ctx.beginPath();
    ctx.arc(sx, sy, 5, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = '#fff';
    ctx.beginPath();
    ctx.arc(sx, sy, 2, 0, Math.PI * 2);
    ctx.fill();

    // Start marker
    const [s0x, s0y] = toScreen(trace[0].x, trace[0].y);
    ctx.fillStyle = 'rgba(255,255,255,0.35)';
    ctx.beginPath();
    ctx.arc(s0x, s0y, 3, 0, Math.PI * 2);
    ctx.fill();

    // Scale bar
    const scaleBarM = rangeX > 200 ? 100 : rangeX > 50 ? 50 : 10;
    const barPx = scaleBarM * scale;
    ctx.strokeStyle = 'rgba(255,255,255,0.3)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(pad, h - 10);
    ctx.lineTo(pad + barPx, h - 10);
    ctx.stroke();
    ctx.fillStyle = 'rgba(255,255,255,0.35)';
    ctx.font = '9px Inter, system-ui, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(`${scaleBarM}m`, pad + barPx / 2, h - 14);
  }, [videoPos, loaded]);

  if (!loaded || !traceRef.current?.length) return null;

  return (
    <div className="relative bg-[#0b0c0f] border border-line rounded-[10px] overflow-hidden shadow-[0_6px_22px_rgba(0,0,0,0.08)]">
      <canvas
        ref={canvasRef}
        className="block w-full h-full"
      />
      <div className="absolute top-2 left-2.5 text-[10px] font-semibold tracking-[0.14em] uppercase text-white bg-black/55 px-2 py-[3px] rounded">
        IMU Path
      </div>
    </div>
  );
}

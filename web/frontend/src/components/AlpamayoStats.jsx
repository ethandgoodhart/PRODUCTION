import React from 'react';

function fmtMs(v) {
  if (v == null) return '—';
  return v >= 1000 ? `${(v / 1000).toFixed(2)}s` : `${v.toFixed(0)}ms`;
}

function fmtMbps(v) {
  if (v == null) return '—';
  return `${v.toFixed(2)} MB/s`;
}

export default function AlpamayoStats({ state }) {
  const aw = state?.autoware || {};
  const alp = aw.alpamayo || {};
  const lat = alp.latency_ms || {};
  const bw = alp.bandwidth_mbps || {};
  const hasLat = lat.gpu !== undefined && aw.running;

  if (!hasLat) return null;

  const stats = [
    { lbl: 'INFERENCE', val: fmtMs(lat.gpu) },
    { lbl: 'SERVER TOTAL', val: fmtMs(lat.srv_total) },
    { lbl: 'DECODE', val: fmtMs(lat.srv_pre) },
    { lbl: 'RTT', val: lat.rtt && lat.rtt > 0 ? fmtMs(lat.rtt) : '—' },
    { lbl: 'NET', val: lat.net != null && lat.rtt > 0 ? fmtMs(lat.net) : '—' },
    { lbl: 'AGE', val: alp.recv_age_s == null ? '—' : `${(alp.recv_age_s * 1000).toFixed(0)}ms` },
    { lbl: 'HZ', val: aw.fps != null ? `${aw.fps.toFixed(2)}` : '—' },
    { lbl: '▲ UP', val: fmtMbps(bw.up), cls: 'text-[#4cc38a]' },
    { lbl: '▼ DOWN', val: fmtMbps(bw.down), cls: 'text-[#58a6ff]' },
    { lbl: 'REGION', val: alp.region || '—' },
  ];

  return (
    <div className="fixed top-3.5 left-1/2 -translate-x-1/2 z-[11] flex gap-[18px] px-3.5 py-1.5 bg-[rgba(18,20,26,0.72)] border border-white/[0.06] rounded-lg text-muted text-[11px] tracking-[0.06em] tabular-nums pointer-events-none backdrop-blur-md">
      {stats.map(({ lbl, val, cls }) => (
        <span key={lbl} className="inline-flex gap-1.5 items-baseline">
          <span className={`uppercase font-semibold tracking-[0.14em] text-[10px] ${cls || 'text-[rgba(220,224,232,0.55)]'}`}>
            {lbl}
          </span>
          <span className="text-white font-semibold">{val}</span>
        </span>
      ))}
    </div>
  );
}

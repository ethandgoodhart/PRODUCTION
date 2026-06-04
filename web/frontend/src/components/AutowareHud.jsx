import React from 'react';

export default function AutowareHud({ state }) {
  const aw = state?.autoware || {};
  const running = !!aw.running;
  const model = (aw.model || 'autoware').toLowerCase();
  const modelUpper = model.toUpperCase();

  let rawStatus = state?.autosteer_status || (running ? 'autoware' : 'idle');
  if (state?.human_override) rawStatus = 'human';
  const engaged = rawStatus === 'autoware';

  const angleText =
    running && aw.inference
      ? `${(Number(aw.steer_deg) || 0) >= 0 ? '+' : ''}${(Number(aw.steer_deg) || 0).toFixed(0)}°`
      : '—';

  const fpsText = aw.fps ? `${aw.fps.toFixed(0)} FPS` : '';

  const seg = aw.segmentation || {};
  const confs = Array.isArray(seg.clrnet_confidences)
    ? seg.clrnet_confidences.map(Number).filter(Number.isFinite)
    : [];
  const confThreshold = Number.isFinite(Number(seg.clrnet_conf_threshold))
    ? Number(seg.clrnet_conf_threshold)
    : 0.4;
  const bestConf = confs.length ? Math.max(...confs) : null;
  const lanesAbove = Number(seg.clrnet_lanes_above_threshold) || 0;
  const hasClrnet = !!seg.clrnet_enabled || seg.clrnet_lanes > 0;
  const showLaneConfidence = running && model === 'segmentation' && hasClrnet;
  const laneOk =
    bestConf != null && bestConf >= confThreshold && lanesAbove > 0;

  const protective = seg.protective_stop || {};
  const protectiveActive = running && !!protective.active;
  let protectiveText = 'STOP';
  if (protectiveActive) {
    const threat = protective.threat || {};
    const label = threat.label ? String(threat.label).toUpperCase() : 'OBJECT';
    const dist = Number.isFinite(Number(threat.x_m))
      ? `${Number(threat.x_m).toFixed(1)}m`
      : '';
    protectiveText = `STOP ${label} ${dist}`.trim();
  }

  const dotColor =
    rawStatus === 'autoware'
      ? 'bg-accent shadow-[0_0_0_3px_var(--color-accent-soft)]'
      : rawStatus === 'human'
        ? 'bg-warn shadow-[0_0_0_3px_rgba(255,138,0,0.18)]'
        : rawStatus === 'stale'
          ? 'bg-[#d84a4a]'
          : 'bg-[#c9ccd3]';

  return (
    <div className="fixed bottom-[calc(50vh+8px)] right-10 z-10 flex items-center gap-4 pointer-events-none text-muted text-[11px] tracking-[0.14em] uppercase tabular-nums transition-all duration-[220ms]">
      <span className={`w-2 h-2 rounded-full transition-all duration-[180ms] ${dotColor}`} />
      <span>{engaged ? modelUpper : rawStatus.toUpperCase()}</span>
      <span className="text-ink font-semibold tracking-normal text-[13px] normal-case">{angleText}</span>
      <span>{fpsText}</span>
      {showLaneConfidence && (
        <span className="inline-flex items-center gap-[7px] min-w-[210px]">
          <span className="text-muted text-[10px] font-semibold tracking-[0.12em] whitespace-nowrap">LANE CONF</span>
          <span className={`min-w-[30px] text-[13px] font-bold tracking-normal text-right ${laneOk ? 'text-[#1f9d68]' : 'text-[#d84a4a]'}`}>
            {bestConf == null ? '—' : bestConf.toFixed(2)}
          </span>
          <span className="relative w-[58px] h-1.5 overflow-hidden rounded-[3px] bg-[rgba(20,24,32,0.16)]">
            <span
              className={`block h-full rounded-[inherit] transition-all duration-[90ms] ${laneOk ? 'bg-[#1f9d68]' : 'bg-[#d84a4a]'}`}
              style={{ width: `${Math.max(0, Math.min(1, bestConf || 0)) * 100}%` }}
            />
          </span>
          <span className="text-muted text-[10px] font-semibold tracking-[0.12em] whitespace-nowrap">
            min {confThreshold.toFixed(2)} · {lanesAbove} lanes
          </span>
        </span>
      )}
      {protectiveActive && (
        <span className="protective-pulse px-2 py-1 rounded bg-[#d84a4a] text-white text-[11px] font-extrabold tracking-[0.12em] whitespace-nowrap">
          {protectiveText}
        </span>
      )}
    </div>
  );
}

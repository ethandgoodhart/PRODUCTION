import React from 'react';
import { clamp } from '../utils/steeringMath';

export default function AutospeedPanel({ state }) {
  const aw = state?.autoware || {};
  const as = aw.autospeed || {};
  const running = !!aw.running && !!aw.inference;
  const segMode = (aw.model || '').toLowerCase() === 'segmentation';
  const show = running && segMode && as.commanded_speed_mph !== undefined;

  if (!show) return null;

  const cmdMph = Number(as.commanded_speed_mph) || 0;
  const maxMph = Number(as.max_speed_mph) || 0;
  const emergency = !!as.emergency_active;
  const accel = Number(as.desired_accel) || 0;
  const range = 2.95;
  const pct = clamp(((accel + 1.75) / range) * 100, 2, 98);

  let accelLabel, accelColor;
  if (emergency) {
    accelLabel = 'EMERGENCY STOP';
    accelColor = 'text-red-600';
  } else if (accel < -0.3) {
    accelLabel = `DECEL ${accel.toFixed(1)} m/s²`;
    accelColor = 'text-red-600';
  } else if (accel > 0.3) {
    accelLabel = `ACCEL ${accel.toFixed(1)} m/s²`;
    accelColor = 'text-emerald-500';
  } else {
    accelLabel = 'COAST';
    accelColor = 'text-muted';
  }

  const limits = as.speed_limits || [];

  return (
    <div className={`bg-[#f8f9fa] border rounded-[10px] overflow-hidden shrink-0 p-[10px_14px_8px] text-xs text-ink transition-colors duration-200 ${
      emergency ? 'border-red-600 bg-red-50/95' : 'border-line'
    }`}>
      <div className="flex items-center gap-2 mb-2">
        <span className="font-bold text-[11px] tracking-[0.08em] text-muted">AUTOSPEED</span>
        {emergency && (
          <span className="font-extrabold text-[11px] tracking-[0.06em] text-red-600 turn-pulse">EMERGENCY</span>
        )}
      </div>
      <div className="flex gap-[18px] mb-2.5">
        <div className="text-center">
          <div className="text-[26px] font-bold leading-[1.1] tabular-nums text-accent">{cmdMph.toFixed(1)}</div>
          <div className="text-[9px] font-semibold tracking-[0.1em] text-muted mt-0.5">CMD MPH</div>
        </div>
        <div className="text-center">
          <div className="text-[20px] font-bold leading-[1.1] tabular-nums text-muted">{maxMph.toFixed(1)}</div>
          <div className="text-[9px] font-semibold tracking-[0.1em] text-muted mt-0.5">MAX MPH</div>
        </div>
      </div>
      <div className="relative h-1.5 rounded-[3px] bg-line mb-1 overflow-visible flex">
        <div className="flex-[1.75] as-accel-decel rounded-l-[3px] opacity-40" />
        <div className="flex-[0.5] bg-gray-300" />
        <div className="flex-[1.2] as-accel-accel rounded-r-[3px] opacity-40" />
        <div
          className="absolute -top-[3px] w-1 h-3 bg-ink rounded-sm -translate-x-0.5 transition-[left] duration-150"
          style={{ left: `${pct}%` }}
        />
      </div>
      <div className={`text-[10px] font-bold tracking-[0.06em] text-center mb-1.5 ${accelColor}`}>
        {accelLabel}
      </div>
      <div className="flex flex-col gap-[3px]">
        {limits.slice(0, 4).map((lim, i) => {
          const isMostLimiting =
            as.most_limiting &&
            as.most_limiting.track_id === lim.track_id &&
            as.most_limiting.distance_m === lim.distance_m;
          return (
            <div
              key={i}
              className={`flex items-baseline justify-between text-[10px] text-ink-soft p-[2px_4px] rounded ${
                isMostLimiting ? 'bg-amber-400/12 font-semibold' : 'bg-black/[0.03]'
              }`}
            >
              <span className="capitalize">{lim.obstacle_class || '?'}</span>
              <span className="tabular-nums text-muted text-[9px]">{Number(lim.distance_m).toFixed(1)}m</span>
              {lim.ttc_s != null ? (
                <span className={`tabular-nums text-[9px] font-semibold ${lim.ttc_s < 2 ? 'text-red-600' : lim.ttc_s < 4 ? 'text-amber-500' : 'text-muted'}`}>
                  {Number(lim.ttc_s).toFixed(1)}s
                </span>
              ) : (
                <span className="tabular-nums text-[9px] text-muted">—</span>
              )}
              <span className="font-bold tabular-nums">
                {Number(lim.speed_mph).toFixed(0)}
                <span className="font-normal text-muted"> mph</span>
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

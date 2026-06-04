import React, { useEffect, useRef } from 'react';
import SteeringWheel from './SteeringWheel';
import PedalStack from './PedalStack';

const MONO3D_REFRESH_MS = 250;

export default function ControlStack({ state }) {
  const aw = state?.autoware || {};
  const mono3dImgRef = useRef(null);
  const steerDeg = Number(state?.steer_deg) || 0;
  const segmentationActive =
    aw.running &&
    aw.inference &&
    (aw.model || '').toLowerCase() === 'segmentation';
  const mono3dReady = segmentationActive && !!aw.viz && (aw.viz_streams || []).includes('mono3d');

  useEffect(() => {
    if (!mono3dReady) return undefined;
    const refresh = () => {
      if (mono3dImgRef.current) {
        mono3dImgRef.current.src = `/cam/mono3d.jpg?t=${Date.now()}`;
      }
    };
    refresh();
    const id = setInterval(refresh, MONO3D_REFRESH_MS);
    return () => clearInterval(id);
  }, [mono3dReady]);

  const predictedWheelDeg = segmentationActive
    ? Number(aw.steer_deg) || 0
    : steerDeg;

  const predGas = Math.max(0, Math.min(1, Number(aw.target_gas) || 0));
  const predBrake = Math.max(0, Math.min(1, Number(aw.target_brake) || 0));

  const autospeed = aw.autospeed || {};
  const cmdSpeedMph = Number(autospeed.commanded_speed_mph) || 0;
  const targetMph = Number(aw.target_speed_mph) || 0;
  const predictedLabel = `autospeed ${cmdSpeedMph.toFixed(1)} mph · target ${targetMph.toFixed(1)} mph`;

  let modelAngleText = 'SEG —';
  let modelAngleHidden = true;
  let clrOverride = false;
  if (segmentationActive) {
    const cmdDeg = Number(aw.steer_deg) || 0;
    const seg = aw.segmentation || {};
    clrOverride = !!seg.clrnet_override;
    const srcLabel = clrOverride ? 'CLR' : 'SEG';
    modelAngleText = `${srcLabel} ${cmdDeg >= 0 ? '+' : ''}${cmdDeg.toFixed(0)}°`;
    modelAngleHidden = false;
  }

  const gt = aw.ground_truth_control || null;
  const gtSteer = gt ? Number(gt.steer_deg) || 0 : 0;
  const gtAngle = gt
    ? `${gtSteer >= 0 ? '+' : ''}${gtSteer.toFixed(0)}°`
    : '—';
  const gtGas = gt ? Math.max(0, Math.min(1, Number(gt.gas_frac) || 0)) : 0;
  const gtBrake = gt
    ? Math.max(0, Math.min(1, Number(gt.brake_frac) || 0))
    : 0;
  const gtMph = gt ? Number(gt.mph) || 0 : 0;
  const gtLabel = gt ? `recorded ${gtMph.toFixed(1)} mph` : 'recorded — mph';

  return (
    <div className="w-[clamp(320px,24vw,420px)] flex flex-col gap-1.5 pointer-events-none p-[8px_10px] border border-line rounded-[10px] bg-[#f8f9fa] shrink-0">
      {/* Predicted row */}
      <div className="grid grid-cols-[64px_90px_1fr] items-center gap-3 min-h-[80px]">
        <div className="text-muted text-[10px] font-extrabold tracking-[0.14em] uppercase text-right">
          Predicted
        </div>
        <div className="flex flex-col items-center gap-2.5 text-ink">
          <SteeringWheel angle={predictedWheelDeg} />
          <div className="flex items-baseline gap-2.5 min-h-3.5 tabular-nums">
            <span className="text-[11px] font-medium tracking-[0.14em] text-muted">
              {predictedWheelDeg >= 0 ? '+' : ''}
              {predictedWheelDeg.toFixed(0)}°
            </span>
            {!modelAngleHidden && (
              <span
                className={`text-xs font-bold tracking-[0.04em] ${
                  clrOverride
                    ? 'text-[#4ade80] [text-shadow:0_0_6px_rgba(74,222,128,0.4)]'
                    : 'text-[#00bcd4] [text-shadow:0_0_10px_rgba(0,188,212,0.35)]'
                }`}
              >
                {modelAngleText}
              </span>
            )}
          </div>
        </div>
        <div className="flex items-center gap-3">
          <PedalStack gasFrac={predGas} brakeFrac={predBrake} />
          <div className="min-w-[120px] text-ink-soft text-xs font-bold tracking-[0.02em] tabular-nums">
            {predictedLabel}
          </div>
        </div>
      </div>
      {/* Ground truth row */}
      <div className="grid grid-cols-[64px_90px_1fr] items-center gap-3 min-h-[80px] opacity-[0.86]">
        <div className="text-muted text-[10px] font-extrabold tracking-[0.14em] uppercase text-right">
          Recorded
        </div>
        <div className="flex flex-col items-center gap-2.5 text-ink">
          <SteeringWheel angle={gtSteer} />
          <div className="flex items-baseline gap-2.5 min-h-3.5 tabular-nums">
            <span className="text-[11px] font-medium tracking-[0.14em] text-muted">
              {gtAngle}
            </span>
            <span className="text-xs font-bold tracking-[0.04em] text-[#667085]">
              {gt ? 'REC' : 'REC —'}
            </span>
          </div>
        </div>
        <div className="flex items-center gap-3">
          <PedalStack gasFrac={gtGas} brakeFrac={gtBrake} />
          <div className="min-w-[120px] text-ink-soft text-xs font-bold tracking-[0.02em] tabular-nums">
            {gtLabel}
          </div>
        </div>
      </div>
      <div className="relative overflow-hidden rounded-[8px] border border-line bg-[#101419] aspect-video">
        <img
          ref={mono3dImgRef}
          alt=""
          className={`block w-full h-full object-cover ${mono3dReady ? 'opacity-100' : 'opacity-0'}`}
        />
        <div className="absolute top-1.5 left-2 text-[9px] font-extrabold tracking-[0.16em] uppercase text-white bg-black/45 px-1.5 py-[2px] rounded">
          3D detections
        </div>
        {!mono3dReady && (
          <div className="absolute inset-0 flex items-center justify-center text-[10px] font-bold tracking-[0.14em] uppercase text-[#7c8490]">
            Waiting for 3D
          </div>
        )}
      </div>
    </div>
  );
}

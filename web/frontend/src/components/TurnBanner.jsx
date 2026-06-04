import React from 'react';

export default function TurnBanner({ state, gps }) {
  const aw = state?.autoware || {};
  const gpsRoute = (aw.segmentation && aw.segmentation.gps_route) || {};
  let turnMsg = aw.running ? gpsRoute.turn_text || '' : '';
  let turnDir = gpsRoute.turn_dir || '';

  // Fall back to GPS-trace-derived turn info
  if (!turnMsg && gps?.turn_text) {
    turnMsg = gps.turn_text;
    turnDir = gps.turn_dir || '';
  }

  if (!turnMsg) return null;

  const arrow =
    turnDir === 'right'
      ? '➡'
      : turnDir === 'left'
        ? '⬅'
        : turnDir === 'straight'
          ? '⬆'
          : '';
  const isNow = /NOW/.test(turnMsg);
  const isStraight = turnDir === 'straight';

  return (
    <div
      className={`absolute top-[18px] left-1/2 -translate-x-1/2 z-[60] flex items-center gap-3.5 px-[26px] py-3.5 rounded-[14px] text-white font-bold text-[26px] leading-none shadow-[0_3px_18px_rgba(0,0,0,0.5)] ${
        isNow
          ? 'bg-[rgba(220,38,38,0.95)] turn-pulse'
          : isStraight
            ? 'bg-[rgba(20,20,20,0.65)]'
            : 'bg-[rgba(20,20,20,0.85)]'
      }`}
    >
      <span className="text-[34px] leading-none">{arrow}</span>
      <span>{turnMsg}</span>
    </div>
  );
}

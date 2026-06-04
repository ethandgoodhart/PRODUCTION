import React from 'react';

export default function StopSignIndicator({ state }) {
  const aw = state?.autoware || {};
  const ss = aw.stop_sign || {};
  const ssState = ss.state || 'clear';
  const show = ssState !== 'clear' && !!aw.running && !!aw.inference;

  if (!show) return null;

  const labels = {
    approaching: 'APPROACHING',
    stopped: 'STOPPED',
    departing: 'DEPARTING',
  };
  let detail = '';
  if (ssState === 'approaching') {
    const d = Number(ss.stop_target_m) || 0;
    detail = d.toFixed(1) + 'm to stop';
  } else if (ssState === 'stopped') {
    const w = ss.wait_remaining_s;
    detail = w != null ? w.toFixed(1) + 's remaining' : 'waiting...';
  } else if (ssState === 'departing') {
    detail = 'resuming...';
  }

  const minConf = ss.min_confidence;
  const curConf = ss.curr_confidence;
  let confText = '';
  if (minConf != null) {
    const curStr = curConf != null ? curConf.toFixed(2) : '—';
    confText = `conf ${curStr} / min ${minConf.toFixed(2)}`;
  }

  const bgColor =
    ssState === 'approaching'
      ? 'border-amber-400 bg-amber-100'
      : ssState === 'stopped'
        ? 'border-red-600 bg-red-100'
        : 'border-green-500 bg-green-100';

  return (
    <div className={`bg-[#f8f9fa] border rounded-[10px] overflow-hidden shrink-0 p-2 px-3 flex items-center gap-2 ${bgColor}`}>
      <div className="text-lg leading-none shrink-0">🛑</div>
      <div className="min-w-0">
        <div className="font-bold text-xs tracking-[0.5px] text-ink">
          {labels[ssState] || ssState.toUpperCase()}
        </div>
        <div className="text-[10px] text-muted tabular-nums">{detail}</div>
        {confText && <div className="text-[9px] text-muted tabular-nums mt-px">{confText}</div>}
      </div>
    </div>
  );
}

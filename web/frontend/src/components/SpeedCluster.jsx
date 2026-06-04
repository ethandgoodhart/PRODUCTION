import React from 'react';

export default function SpeedCluster({ mph }) {
  return (
    <div className="fixed top-[108px] left-0 right-0 z-5 pointer-events-none">
      <div className="absolute left-1/2 top-0 -translate-x-1/2 flex flex-col items-center gap-1">
        <div className="text-[132px] font-light leading-[0.95] tracking-[-0.06em] text-ink tabular-nums">
          {Math.round(mph || 0)}
        </div>
        <div className="text-[15px] font-bold tracking-[0.22em] text-muted">MPH</div>
      </div>
      <div className="absolute left-[calc(50%+130px)] top-[18px]">
        <div className="flex flex-col items-center justify-center bg-white rounded-[10px] p-[8px_12px_10px] w-[66px] min-h-[78px] shadow-[0_1px_2px_rgba(0,0,0,0.04),0_6px_18px_rgba(0,0,0,0.06)]">
          <div className="text-[8px] font-semibold tracking-[0.08em] leading-[1.05] text-center text-ink">
            SPEED<br />LIMIT
          </div>
          <div className="text-[32px] font-semibold leading-none mt-1.5 text-ink tabular-nums">25</div>
        </div>
      </div>
    </div>
  );
}

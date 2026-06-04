import React from 'react';
import useClock from '../hooks/useClock';

export default function TopBar({ state }) {
  const time = useClock();
  const aw = state?.autoware || {};
  const model = (aw.model || 'autoware').toLowerCase();
  let rawStatus = state?.autosteer_status || (aw.running ? 'autoware' : 'idle');
  if (state?.human_override) rawStatus = 'human';
  const modeLabel = state?.teleop_active ? 'Teleop Mode' : 'Controller Mode';
  const teleopUrl = state?.teleop_active ? state.teleop_url || '' : '';

  const alp = aw.alpamayo || {};
  const reasoning = (alp.reasoning || '').trim();

  return (
    <header className="fixed top-0 left-0 right-0 grid grid-cols-3 items-center px-10 py-[22px] z-10">
      <div className="justify-self-start">
        <div className="flex flex-col items-start leading-none">
          <div className="font-light text-[38px] tracking-[-0.1em] text-ink">Caddy</div>
          <div className="mt-1.5 text-[13px] font-medium tracking-[-0.1em] text-muted">v1.0.0</div>
        </div>
        {reasoning && (
          <div className="mt-2.5 max-w-[320px] p-[6px_10px] bg-[rgba(18,20,26,0.72)] border border-white/[0.06] rounded-md pointer-events-none backdrop-blur-md">
            <div className="text-[9px] tracking-[0.18em] text-white/[0.55] font-bold mb-0.5">REASONING</div>
            <div className="text-xs leading-[1.32] text-white font-mono whitespace-pre-wrap break-words">{reasoning}</div>
          </div>
        )}
      </div>
      <div className="justify-self-center flex flex-col items-center">
        <div className="flex items-center gap-2.5 px-5 py-[9px] bg-white text-ink-soft border border-[#e3e5ea] rounded-full text-[13.5px] font-medium tracking-[0.02em] shadow-[0_1px_2px_rgba(0,0,0,0.03),0_6px_18px_rgba(0,0,0,0.05)] backdrop-blur-2xl">
          <span className="relative inline-flex shrink-0">
            <img
              className="w-[26px] h-auto shrink-0 block object-contain"
              src="/img/ps5-controller.png"
              alt=""
              aria-hidden="true"
            />
            <span
              className={`absolute left-1/2 -bottom-0.5 -translate-x-1/2 w-[7px] h-[7px] rounded-full border-[1.5px] border-white transition-all duration-[180ms] pointer-events-none ${
                state?.controller_connected
                  ? 'bg-[#7ed488] shadow-[0_0_0_2px_rgba(126,212,136,0.22)]'
                  : 'bg-[#d84a4a]'
              }`}
              title="Controller"
            />
          </span>
          <span>{modeLabel}</span>
        </div>
        {state?.teleop_active && (
          <span className="text-[11px] text-muted font-normal ml-2 tracking-[0.01em]">
            {teleopUrl || 'tunnel starting…'}
          </span>
        )}
        <div className="flex justify-center gap-3.5 mt-2 text-[10.5px] font-medium tracking-[0.04em] text-muted">
          {[
            { key: 'motor_connected', label: 'Motor' },
            { key: 'arduino_connected', label: 'Arduino' },
            { key: 'ego_connected', label: 'Ego Motion Data Stream' },
          ].map(({ key, label }) => (
            <span key={key} className="inline-flex items-center gap-[5px]">
              <span
                className={`w-1.5 h-1.5 rounded-full transition-all duration-[180ms] ${
                  state?.[key]
                    ? 'bg-[#7ed488] shadow-[0_0_0_1.5px_rgba(126,212,136,0.18)]'
                    : 'bg-[#d84a4a]'
                }`}
              />
              <span>{label}</span>
            </span>
          ))}
        </div>
      </div>
      <div className="justify-self-end flex items-center gap-6">
        <div className="text-[15px] font-medium text-ink-soft tabular-nums">{time}</div>
        <div className="flex items-center gap-2 text-sm font-semibold text-ink-soft">
          <span>87%</span>
          <div className="relative w-8 h-3.5 border-[1.5px] border-ink-soft rounded-[3px] p-px">
            <div className="h-full w-[87%] bg-ink rounded-[1px]" />
            <div className="absolute -right-1 top-[3px] w-0.5 h-1.5 bg-ink-soft rounded-sm" />
          </div>
        </div>
      </div>
    </header>
  );
}

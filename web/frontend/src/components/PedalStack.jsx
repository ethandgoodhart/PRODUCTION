import React from 'react';

const PEDAL_COUNT = 10;

export default function PedalStack({ gasFrac = 0, brakeFrac = 0 }) {
  const gasLit = Math.round(gasFrac * PEDAL_COUNT);
  const brakeLit = Math.round(brakeFrac * PEDAL_COUNT);
  const gasEdge = gasLit > 0 ? PEDAL_COUNT - gasLit : -1;
  const brakeEdge = brakeLit > 0 ? brakeLit - 1 : -1;

  const gasDashes = [];
  for (let i = 0; i < PEDAL_COUNT; i++) {
    const active = i >= PEDAL_COUNT - gasLit;
    const edge = i === gasEdge;
    gasDashes.push(
      <div
        key={`gas-${i}`}
        className={`flex-1 min-h-[2px] rounded-sm transition-all duration-[160ms] ${
          active
            ? 'bg-accent/75 shadow-[0_0_10px_rgba(31,111,235,0.35)]'
            : 'bg-accent/[0.07]'
        } ${edge ? 'pedal-edge' : ''}`}
      />
    );
  }

  const brakeDashes = [];
  for (let i = 0; i < PEDAL_COUNT; i++) {
    const active = i < brakeLit;
    const edge = i === brakeEdge;
    brakeDashes.push(
      <div
        key={`brake-${i}`}
        className={`flex-1 min-h-[2px] rounded-sm transition-all duration-[160ms] ${
          active
            ? 'bg-accent/75 shadow-[0_0_10px_rgba(31,111,235,0.35)]'
            : 'bg-accent/[0.07]'
        } ${edge ? 'pedal-edge' : ''}`}
      />
    );
  }

  return (
    <div className="flex flex-col gap-[3px] h-[74px] w-[26px] py-0.5 justify-between" aria-hidden="true">
      {gasDashes}
      {brakeDashes}
    </div>
  );
}

import React from 'react';

export default function SteeringWheel({ angle = 0 }) {
  return (
    <svg className="w-[70px] h-[70px] block text-ink drop-shadow-[0_4px_14px_rgba(0,0,0,0.08)]" viewBox="0 0 100 100" aria-hidden="true">
      <g
        className="wheel-rotate"
        style={{ transform: `rotate(${angle}deg)` }}
      >
        <circle cx="50" cy="50" r="40" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
        <circle cx="50" cy="50" r="10.5" fill="none" stroke="currentColor" strokeWidth="3" />
        <line x1="11" y1="50" x2="39" y2="50" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
        <line x1="61" y1="50" x2="89" y2="50" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
        <line x1="50" y1="61" x2="50" y2="89" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
      </g>
    </svg>
  );
}

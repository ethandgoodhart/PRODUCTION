import React, { useRef, useCallback, useEffect, useState } from 'react';

const QUIT_DWELL_MS = 1500;

export default function QuitCorner() {
  const rafRef = useRef(0);
  const startRef = useRef(0);
  const fgRef = useRef(null);
  const [active, setActive] = useState(false);

  const doQuit = useCallback(() => {
    fetch('/quit', { method: 'POST' });
  }, []);

  const tick = useCallback(() => {
    const elapsed = Date.now() - startRef.current;
    const frac = Math.min(1, elapsed / QUIT_DWELL_MS);
    const circ = 2 * Math.PI * 15.5;
    if (fgRef.current) {
      fgRef.current.style.strokeDashoffset = circ * (1 - frac);
    }
    if (frac >= 1) {
      doQuit();
      return;
    }
    rafRef.current = requestAnimationFrame(tick);
  }, [doQuit]);

  const handleEnter = useCallback(() => {
    setActive(true);
    startRef.current = Date.now();
    const circ = 2 * Math.PI * 15.5;
    if (fgRef.current) {
      fgRef.current.style.strokeDasharray = circ;
      fgRef.current.style.strokeDashoffset = circ;
    }
    rafRef.current = requestAnimationFrame(tick);
  }, [tick]);

  const handleLeave = useCallback(() => {
    setActive(false);
    cancelAnimationFrame(rafRef.current);
    const circ = 2 * Math.PI * 15.5;
    if (fgRef.current) {
      fgRef.current.style.strokeDashoffset = circ;
    }
  }, []);

  useEffect(() => {
    const handler = (e) => {
      if (e.key === 'q' || e.key === 'Q') doQuit();
    };
    document.addEventListener('keydown', handler);
    return () => document.removeEventListener('keydown', handler);
  }, [doQuit]);

  return (
    <div
      className={`fixed top-0 right-0 w-[60px] h-[60px] z-[100] flex items-center justify-center cursor-pointer transition-opacity duration-200 ${
        active ? 'opacity-100' : 'opacity-0 hover:opacity-100'
      }`}
      onMouseEnter={handleEnter}
      onMouseLeave={handleLeave}
    >
      <svg className="absolute w-9 h-9" viewBox="0 0 36 36">
        <circle className="quit-ring-bg" cx="18" cy="18" r="15.5" />
        <circle className="quit-ring-fg" ref={fgRef} cx="18" cy="18" r="15.5" />
      </svg>
      <span className={`relative text-base font-light leading-none transition-colors duration-[180ms] ${active ? 'text-[#d84a4a]' : 'text-muted'}`}>
        &times;
      </span>
    </div>
  );
}

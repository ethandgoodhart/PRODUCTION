import { useState, useEffect, useRef } from 'react';

export default function useGpsPolling(intervalMs = 1000) {
  const [gps, setGps] = useState(null);
  const intervalRef = useRef(null);

  useEffect(() => {
    async function poll() {
      try {
        const r = await fetch('/gps', { cache: 'no-store' });
        const d = await r.json();
        setGps(d);
      } catch (e) {
        // keep last state
      }
    }
    poll();
    intervalRef.current = setInterval(poll, intervalMs);
    return () => clearInterval(intervalRef.current);
  }, [intervalMs]);

  return gps;
}

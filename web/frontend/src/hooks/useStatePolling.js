import { useState, useEffect, useRef } from 'react';

export default function useStatePolling(intervalMs = 200) {
  const [state, setState] = useState(null);
  const intervalRef = useRef(null);

  useEffect(() => {
    async function poll() {
      try {
        const r = await fetch('/state', { cache: 'no-store' });
        if (r.ok) {
          const s = await r.json();
          setState(s);
        }
      } catch (e) {
        setState((prev) =>
          prev
            ? {
                ...prev,
                controller_connected: false,
                motor_connected: false,
                arduino_connected: false,
                ego_connected: false,
              }
            : null
        );
      }
    }
    poll();
    intervalRef.current = setInterval(poll, intervalMs);
    return () => clearInterval(intervalRef.current);
  }, [intervalMs]);

  return state;
}

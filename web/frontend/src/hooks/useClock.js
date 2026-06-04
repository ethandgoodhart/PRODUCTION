import { useState, useEffect } from 'react';

function formatTime() {
  const d = new Date();
  const h = d.getHours();
  const m = d.getMinutes().toString().padStart(2, '0');
  const hh = ((h + 11) % 12) + 1;
  const ap = h >= 12 ? 'PM' : 'AM';
  return `${hh}:${m} ${ap}`;
}

export default function useClock() {
  const [time, setTime] = useState(formatTime);

  useEffect(() => {
    const id = setInterval(() => setTime(formatTime()), 1000);
    return () => clearInterval(id);
  }, []);

  return time;
}

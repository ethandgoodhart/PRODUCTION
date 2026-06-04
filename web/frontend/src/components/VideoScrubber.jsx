import React, { useState, useRef, useEffect } from 'react';

function formatVideoTime(seconds) {
  const s = Math.max(0, Number(seconds) || 0);
  const whole = Math.floor(s);
  const mins = Math.floor(whole / 60);
  const secs = String(whole % 60).padStart(2, '0');
  return `${mins}:${secs}`;
}

async function postVideoControl(seekS, pause, lastPostRef, force = false) {
  const now = performance.now();
  if (!force && now - lastPostRef.current < 120) return;
  lastPostRef.current = now;
  try {
    await fetch('/offline-video-control', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ seek_s: seekS, pause }),
    });
  } catch (_) {}
}

export default function VideoScrubber({ state }) {
  const aw = state?.autoware || {};
  const video = aw.video || {};
  const duration = Number(video.duration_s) || 0;
  const scrubbable = !!video.scrubbable && duration > 0;

  const [dragging, setDragging] = useState(false);
  const [localPaused, setLocalPaused] = useState(false);
  const [localPos, setLocalPos] = useState(0);
  const [pendingPaused, setPendingPaused] = useState(null);
  const [pendingSeek, setPendingSeek] = useState(null);
  const [predictions, setPredictions] = useState({ runs: [], active_id: null });
  const [predictionBusy, setPredictionBusy] = useState(false);
  const wasPausedRef = useRef(false);
  const lastPostRef = useRef(0);
  const rangeRef = useRef(null);

  const pos = Math.max(0, Math.min(duration, Number(video.position_s) || 0));
  const paused = dragging ? localPaused : pendingPaused ?? !!video.paused;
  const displayPos = dragging ? localPos : pendingSeek ?? pos;
  const pct = duration > 0 ? (displayPos / duration) * 100 : 0;

  const seekFromClientX = (clientX) => {
    const rect = rangeRef.current?.getBoundingClientRect();
    if (!rect || rect.width <= 0 || duration <= 0) return displayPos;
    const frac = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
    return frac * duration;
  };

  useEffect(() => {
    if (pendingPaused === null) return;
    if (!!video.paused === pendingPaused) {
      setPendingPaused(null);
    }
  }, [pendingPaused, video.paused]);

  useEffect(() => {
    if (pendingSeek === null || dragging) return;
    if (Math.abs((Number(video.position_s) || 0) - pendingSeek) < 0.35) {
      setPendingSeek(null);
    }
  }, [dragging, pendingSeek, video.position_s]);

  useEffect(() => {
    let cancelled = false;
    async function loadPredictions() {
      try {
        const r = await fetch('/offline-predictions', { cache: 'no-store' });
        if (!r.ok) return;
        const data = await r.json();
        if (!cancelled) setPredictions(data || { runs: [], active_id: null });
      } catch (_) {}
    }
    loadPredictions();
    const id = setInterval(loadPredictions, 2500);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  useEffect(() => {
    if (!dragging) return;
    const move = (e) => {
      const seekS = seekFromClientX(e.clientX);
      setLocalPos(seekS);
      postVideoControl(seekS, true, lastPostRef);
    };
    const finish = (e) => {
      const seekS = seekFromClientX(e.clientX);
      setDragging(false);
      setLocalPos(seekS);
      setPendingSeek(seekS);
      setPendingPaused(wasPausedRef.current);
      postVideoControl(seekS, wasPausedRef.current, lastPostRef, true);
    };
    window.addEventListener('pointermove', move);
    window.addEventListener('pointerup', finish);
    return () => {
      window.removeEventListener('pointermove', move);
      window.removeEventListener('pointerup', finish);
    };
  }, [dragging, displayPos, duration]);

  const runs = Array.isArray(predictions.runs) ? predictions.runs : [];
  const activePredictionId =
    predictions.active_id || state?.offline_prediction?.id || '';

  if (!scrubbable && runs.length === 0) return null;

  const handlePlay = () => {
    const nextPaused = !paused;
    setLocalPaused(nextPaused);
    setPendingPaused(nextPaused);
    postVideoControl(displayPos, nextPaused, lastPostRef, true);
  };

  const handlePointerDown = (e) => {
    const seekS = seekFromClientX(e.clientX);
    e.preventDefault();
    setDragging(true);
    wasPausedRef.current = paused;
    setLocalPaused(paused);
    setLocalPos(seekS);
    setPendingPaused(true);
    postVideoControl(seekS, true, lastPostRef, true);
  };

  const handleInput = (e) => {
    const seekS = Number(e.target.value) || 0;
    setLocalPos(seekS);
    if (dragging) {
      postVideoControl(seekS, true, lastPostRef);
    }
  };

  const handleChange = (e) => {
    const seekS = Number(e.target.value) || 0;
    setLocalPos(seekS);
    setDragging(false);
    setPendingSeek(seekS);
    setPendingPaused(wasPausedRef.current);
    postVideoControl(seekS, wasPausedRef.current, lastPostRef, true);
  };

  const handlePredictionChange = async (e) => {
    const id = e.target.value;
    if (!id) return;
    setPredictionBusy(true);
    try {
      const r = await fetch('/offline-predictions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id }),
      });
      if (r.ok) {
        const data = await r.json();
        setPredictions((prev) => ({
          ...prev,
          active_id: data?.selection?.id || id,
          active: data?.active || prev.active,
        }));
      }
    } catch (_) {
    } finally {
      setPredictionBusy(false);
    }
  };

  return (
    <div className="grid grid-cols-[minmax(0,1fr)_minmax(220px,300px)] items-center gap-2.5 px-3.5 py-2.5 border border-ink/10 rounded-[10px] bg-[rgba(248,249,250,0.95)] backdrop-blur-md">
      {scrubbable ? (
        <div className="grid grid-cols-[34px_48px_1fr_48px] items-center gap-2.5 min-w-0">
          <button
            className="w-[34px] h-[30px] border border-ink/12 rounded-[7px] bg-white text-ink text-sm font-bold leading-none cursor-pointer hover:bg-[#f0f1f3] hover:border-ink/22 active:bg-[#e4e6e9] transition-all duration-150"
            type="button"
            aria-label={paused ? 'Play video' : 'Pause video'}
            onClick={handlePlay}
          >
            {paused ? '▶' : 'Ⅱ'}
          </button>
          <span className="text-ink-soft text-xs font-semibold tabular-nums text-center select-none">
            {formatVideoTime(displayPos)}
          </span>
          <input
            ref={rangeRef}
            className="video-range"
            type="range"
            min="0"
            max={duration.toFixed(1)}
            step="0.1"
            value={displayPos.toFixed(1)}
            aria-label="Video position"
            style={{ '--pct': `${pct}%` }}
            onPointerDown={handlePointerDown}
            onInput={handleInput}
            onChange={handleChange}
          />
          <span className="text-ink-soft text-xs font-semibold tabular-nums text-center select-none">
            {formatVideoTime(duration)}
          </span>
        </div>
      ) : <div />}
      <div className="grid grid-cols-[80px_1fr] items-center gap-2 min-w-0">
        <span className="text-[10px] font-bold tracking-[0.16em] text-muted uppercase select-none">
          Predictions
        </span>
        <select
          className="h-[30px] min-w-0 rounded-[7px] border border-ink/12 bg-white px-2 text-xs font-semibold text-ink-soft outline-none cursor-pointer disabled:opacity-55"
          value={activePredictionId}
          aria-label="Offline prediction set"
          disabled={predictionBusy || runs.length === 0}
          onChange={handlePredictionChange}
        >
          {runs.length === 0 ? (
            <option value="">No runs</option>
          ) : (
            runs.map((run) => (
              <option key={run.id} value={run.id} disabled={!run.available}>
                {run.label || run.id}
              </option>
            ))
          )}
        </select>
      </div>
    </div>
  );
}

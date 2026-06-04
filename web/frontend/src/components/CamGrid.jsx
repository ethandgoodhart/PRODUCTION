import React, { useEffect, useRef, useState } from 'react';
import ImuPath from './ImuPath';

const FRAME_REFRESH_MS = 66;

const ALL_CAMS = [
  { slug: 'live_1', label: 'Camera 1', isViz: false, isModelRaw: false },
  { slug: 'live_2', label: 'Camera 2', isViz: false, isModelRaw: false },
  { slug: 'live_3', label: 'Camera 3', isViz: false, isModelRaw: false },
  { slug: 'live_4', label: 'Camera 4', isViz: false, isModelRaw: false },
  { slug: 'live_5', label: 'Camera 5', isViz: false, isModelRaw: false },
  { slug: 'live_6', label: 'Camera 6', isViz: false, isModelRaw: false },
  { slug: 'front', label: 'Front Center', isViz: false, isModelRaw: true },
  { slug: 'lanes', label: 'Lanes', isViz: true, isModelRaw: false },
  { slug: 'depth', label: 'Depth', isViz: true, isModelRaw: false },
  { slug: 'objects', label: 'Objects + Futures', isViz: true, isModelRaw: false },
  { slug: 'seg', label: 'Segmentation', isViz: true, isModelRaw: false },
  { slug: 'bev', label: 'BEV Planner', isViz: true, isModelRaw: false },
];

export default function CamGrid({ state }) {
  const aw = state?.autoware || {};
  const running = !!aw.running;
  const model = (aw.model || 'autoware').toLowerCase();
  const modelUpper = model.toUpperCase();
  const segmentationMode = model === 'segmentation';
  const vizLive = !!aw.viz;
  const liveStreams = new Set(aw.viz_streams || []);
  const activeCam = running && aw.inference ? aw.active_cam : null;
  const badgeText = running ? modelUpper : '—';

  const imgRefs = useRef({});
  const pendingRef = useRef({});
  const lastRefreshRef = useRef(0);
  const intervalRef = useRef(null);
  const [objectsZoom, setObjectsZoom] = useState(1);

  useEffect(() => {
    function refreshTiles() {
      const now = performance.now();
      if (now - lastRefreshRef.current < FRAME_REFRESH_MS) return;
      lastRefreshRef.current = now;
      const t = Date.now();

      for (const cam of ALL_CAMS) {
        const { slug, isViz, isModelRaw } = cam;
        if (isModelRaw && !segmentationMode) continue;
        if (!isModelRaw && !isViz && segmentationMode) continue;
        if (isViz && !(vizLive && liveStreams.has(slug))) continue;

        const img = imgRefs.current[slug];
        if (!img) continue;
        const pending = pendingRef.current[slug] || 0;
        if (pending >= 2) continue;
        pendingRef.current[slug] = pending + 1;
        img.src = `/cam/${slug}.jpg?t=${t}`;
      }
    }

    refreshTiles();
    intervalRef.current = setInterval(refreshTiles, FRAME_REFRESH_MS);
    return () => clearInterval(intervalRef.current);
  }, [segmentationMode, vizLive, liveStreams]);

  const handleLoad = (slug) => {
    pendingRef.current[slug] = Math.max(0, (pendingRef.current[slug] || 0) - 1);
  };
  const handleError = (slug) => {
    pendingRef.current[slug] = Math.max(0, (pendingRef.current[slug] || 0) - 1);
  };

  useEffect(() => {
    window.__showLanesTexture = running && vizLive && liveStreams.has('lanes');
  }, [running, vizLive, liveStreams]);

  return (
    <div className="grid grid-cols-3 grid-rows-2 gap-1.5 w-full h-full pointer-events-none">
      {ALL_CAMS.flatMap((cam) => {
        const { slug, label, isViz, isModelRaw } = cam;

        let display = true;
        if (isModelRaw && !segmentationMode) display = false;
        if (!isModelRaw && !isViz && segmentationMode) display = false;
        if (isViz && !(vizLive && liveStreams.has(slug))) display = false;

        if (!display) return [];

        const isActive = slug === activeCam;
        const vizReady = isViz && vizLive && liveStreams.has(slug);
        const isObjectsViz = slug === 'objects';
        const containFit = slug === 'bev' || slug === 'seg' || slug === 'objects' || segmentationMode;

        const tile = (
          <div
            key={slug}
            className={`relative ${isObjectsViz ? 'bg-white' : 'bg-[#0b0c0f]'} border rounded-[10px] overflow-hidden shadow-[0_6px_22px_rgba(0,0,0,0.08)] transition-all duration-[180ms] ${
              isActive
                ? 'border-accent shadow-[0_0_0_2px_var(--color-accent-soft),0_8px_28px_rgba(31,111,235,0.22)]'
                : 'border-line'
            }`}
            data-cam={slug}
          >
            <img
              alt=""
              className={`block w-full h-full ${containFit ? `${isObjectsViz ? 'object-contain bg-white' : 'object-contain bg-black'}` : 'object-cover bg-[#0b0c0f]'} ${
                isViz && !vizReady ? 'opacity-0' : ''
              }`}
              style={
                isObjectsViz
                  ? {
                      transform: `scale(${objectsZoom})`,
                      transformOrigin: '50% 100%',
                      objectPosition: '50% 100%',
                      transition: 'transform 120ms ease-out',
                    }
                  : undefined
              }
              ref={(el) => {
                if (el) imgRefs.current[slug] = el;
              }}
              onLoad={() => handleLoad(slug)}
              onError={() => handleError(slug)}
            />
            {isObjectsViz && (
              <div className="absolute top-2 right-2 pointer-events-auto flex items-center gap-1.5 rounded-[8px] border border-ink/10 bg-white/85 px-2 py-1 shadow-[0_2px_10px_rgba(0,0,0,0.12)] backdrop-blur-sm">
                <span className="text-[11px] font-bold text-[#1d2430] leading-none">⌕</span>
                <input
                  className="w-[70px] accent-[#2b87d2]"
                  type="range"
                  min="0.75"
                  max="2.5"
                  step="0.05"
                  value={objectsZoom}
                  aria-label="Objects BEV zoom"
                  onChange={(e) => setObjectsZoom(Number(e.target.value) || 1)}
                />
                <span className="min-w-[26px] text-right text-[10px] font-semibold tabular-nums text-[#344054]">
                  {objectsZoom.toFixed(1)}
                </span>
              </div>
            )}
            {!isObjectsViz && (
              <div className="absolute top-2 left-2.5 text-[10px] font-semibold tracking-[0.14em] uppercase text-white bg-black/55 px-2 py-[3px] rounded">
                {label}
              </div>
            )}
            {!isViz && (
              <div
                className={`absolute top-2 right-2.5 text-[10px] font-bold tracking-[0.18em] text-white bg-accent px-2 py-[3px] rounded transition-all duration-[160ms] ${
                  isActive ? 'opacity-100 translate-y-0' : 'opacity-0 -translate-y-0.5'
                }`}
              >
                {badgeText}
              </div>
            )}
            {isViz && !vizReady && (
              <div className="absolute inset-0 flex items-center justify-center text-[#6d727c] bg-[#f6f7f9] text-[10px] tracking-[0.18em] uppercase">
                model loading…
              </div>
            )}
          </div>
        );

        if (slug === 'objects') {
          return [tile, <ImuPath key="imu-path" state={state} />];
        }
        return [tile];
      })}
    </div>
  );
}

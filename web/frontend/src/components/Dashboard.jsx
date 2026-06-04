import React from 'react';
import MiniMap from './MiniMap';
import AutospeedPanel from './AutospeedPanel';
import StopSignIndicator from './StopSignIndicator';
import CamGrid from './CamGrid';
import ControlStack from './ControlStack';
import VideoScrubber from './VideoScrubber';

export default function Dashboard({ state, gps }) {
  return (
    <div className="fixed bottom-0 left-0 right-0 h-[50vh] z-[9] bg-white border-t border-line grid grid-cols-[180px_1fr_auto] grid-rows-[1fr_auto] gap-2 p-[10px_12px] shadow-[0_-2px_20px_rgba(0,0,0,0.06)]">
      <div className="row-start-1 col-start-1 flex flex-col gap-2.5 overflow-y-auto min-h-0">
        <MiniMap gps={gps} />
        <AutospeedPanel state={state} />
        <StopSignIndicator state={state} />
      </div>
      <div className="row-start-1 col-start-2 min-w-0 min-h-0 overflow-hidden">
        <CamGrid state={state} />
      </div>
      <div className="row-start-1 col-start-3 min-h-0 overflow-y-auto flex flex-col gap-1.5">
        <ControlStack state={state} />
      </div>
      <div className="row-start-2 col-span-full">
        <VideoScrubber state={state} />
      </div>
    </div>
  );
}

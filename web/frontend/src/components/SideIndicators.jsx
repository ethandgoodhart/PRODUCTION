import React from 'react';

export default function SideIndicators() {
  return (
    <>
      <div className="fixed left-9 top-1/2 -translate-y-1/2 flex flex-col gap-2.5 z-[4]">
        <div className="w-1.5 h-1.5 bg-line rounded-full" />
        <div className="w-1.5 h-1.5 bg-[#d0d4dc] rounded-full" />
        <div className="w-1.5 h-1.5 bg-line rounded-full" />
      </div>
      <div className="fixed right-9 top-1/2 -translate-y-1/2 flex flex-col gap-2.5 z-[4]">
        <div className="w-1.5 h-1.5 bg-line rounded-full" />
        <div className="w-1.5 h-1.5 bg-[#d0d4dc] rounded-full" />
        <div className="w-1.5 h-1.5 bg-line rounded-full" />
      </div>
    </>
  );
}

import React from 'react';
import useStatePolling from './hooks/useStatePolling';
import useGpsPolling from './hooks/useGpsPolling';
import ThreeScene from './components/ThreeScene';
import TopBar from './components/TopBar';
import EgoPanel from './components/EgoPanel';
import SpeedCluster from './components/SpeedCluster';
import Dashboard from './components/Dashboard';
import SideIndicators from './components/SideIndicators';
import AlpamayoStats from './components/AlpamayoStats';
import AutowareHud from './components/AutowareHud';
import TurnBanner from './components/TurnBanner';
import QuitCorner from './components/QuitCorner';

export default function App() {
  const state = useStatePolling(200);
  const gps = useGpsPolling(1000);
  const mph = Number(state?.mph) || 0;

  return (
    <>
      <ThreeScene state={state} />
      <TopBar state={state} />
      <EgoPanel state={state} />
      <SpeedCluster mph={mph} />
      <Dashboard state={state} gps={gps} />
      <SideIndicators />
      <AlpamayoStats state={state} />
      <AutowareHud state={state} />
      <TurnBanner state={state} gps={gps} />
      <QuitCorner />
    </>
  );
}

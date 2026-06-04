import React, { useEffect, useRef } from 'react';
import L from 'leaflet';

export default function MiniMap({ gps }) {
  const mapRef = useRef(null);
  const mapInstanceRef = useRef(null);
  const markerRef = useRef(null);
  const firstFixRef = useRef(true);
  const hadFixRef = useRef(false);

  useEffect(() => {
    if (mapInstanceRef.current) return;
    const map = L.map(mapRef.current, {
      zoomControl: false,
      attributionControl: false,
      dragging: false,
      scrollWheelZoom: false,
      doubleClickZoom: false,
      boxZoom: false,
      keyboard: false,
      touchZoom: false,
    }).setView([37.4275, -122.1697], 19);

    const MAPBOX_TOKEN = import.meta.env.VITE_MAPBOX_TOKEN || '';
    L.tileLayer(
      `https://api.mapbox.com/styles/v1/mapbox/streets-v12/tiles/{z}/{x}/{y}@2x?access_token=${MAPBOX_TOKEN}`,
      { maxZoom: 22, tileSize: 512, zoomOffset: -1 }
    ).addTo(map);

    const miniIcon = L.divIcon({
      className: 'mini-ego-marker',
      iconSize: [14, 14],
    });
    markerRef.current = L.marker([37.4275, -122.1697], {
      icon: miniIcon,
    }).addTo(map);

    mapInstanceRef.current = map;
  }, []);

  useEffect(() => {
    if (!gps || !mapInstanceRef.current) return;
    const map = mapInstanceRef.current;
    const marker = markerRef.current;

    if (gps.has_fix && gps.lat != null && gps.lon != null) {
      const ll = [gps.lat, gps.lon];
      marker.setLatLng(ll);
      map.panTo(ll, { animate: !firstFixRef.current, duration: 0.25 });
      if (firstFixRef.current) {
        map.setView(ll, 19);
        firstFixRef.current = false;
      }
      hadFixRef.current = true;
    }
  }, [gps]);

  const hasFix = gps?.has_fix;
  const noGps = !hasFix && !hadFixRef.current;
  const labelText = hasFix ? 'LIVE MAP' : 'MAP · OFFLINE';

  return (
    <div className="relative h-[160px] bg-[#f8f9fa] border border-line rounded-[10px] overflow-hidden shrink-0">
      <div className="absolute top-1 left-1.5 z-[1000] text-[8px] font-semibold tracking-[0.8px] text-black/60 bg-white/70 px-[5px] py-0.5 rounded-[3px] pointer-events-none">
        {labelText}
      </div>
      <div ref={mapRef} id="mini-map" />
      {noGps && (
        <div className="absolute bottom-2 right-2 z-[1000] text-[10px] font-semibold tracking-[0.5px] text-red-400/90 bg-black/60 px-1.5 py-[3px] rounded-[3px] pointer-events-none">
          NO GPS
        </div>
      )}
    </div>
  );
}

import React, { useEffect, useRef, useCallback } from 'react';

const REFRESH_MS = 66;
const FT_TO_M = 0.3048;

const CITYSCAPES_PALETTE = [
  [128,  64, 128], // road
  [244,  35, 232], // sidewalk
  [ 70,  70,  70], // building
  [102, 102, 156], // wall
  [190, 153, 153], // fence
  [153, 153, 153], // pole
  [250, 170,  30], // traffic light
  [220, 220,   0], // traffic sign
  [107, 142,  35], // vegetation
  [152, 251, 152], // terrain
  [ 70, 130, 180], // sky
  [220,  20,  60], // person
  [255,   0,   0], // rider
  [  0,   0, 142], // car
  [  0,   0,  70], // truck
  [  0,  60, 100], // bus
  [  0,  80, 100], // train
  [  0,   0, 230], // motorcycle
  [119,  11,  32], // bicycle
];

function buildPaletteLut() {
  const lut = new Uint8ClampedArray(256 * 4);
  for (let i = 0; i < 256; i++) {
    const c = i < CITYSCAPES_PALETTE.length ? CITYSCAPES_PALETTE[i] : [30, 30, 30];
    lut[i * 4 + 0] = c[0];
    lut[i * 4 + 1] = c[1];
    lut[i * 4 + 2] = c[2];
    lut[i * 4 + 3] = 255;
  }
  lut[255 * 4 + 0] = 30;
  lut[255 * 4 + 1] = 30;
  lut[255 * 4 + 2] = 30;
  lut[255 * 4 + 3] = 255;
  return lut;
}

const PALETTE_LUT = buildPaletteLut();

function localToBev(fwd, left, rangeFwdM, rangeSideM, bevSize) {
  const bx = (left / rangeSideM * 0.5 + 0.5) * bevSize;
  const by = (1 - fwd / rangeFwdM) * bevSize;
  return [bx, by];
}

export default function BevCanvas({ state }) {
  const canvasRef = useRef(null);
  const offscreenRef = useRef(null);
  const pendingRef = useRef(false);
  const intervalRef = useRef(null);

  const renderFrame = useCallback((classMapData, w, h) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d');

    const imageData = ctx.createImageData(w, h);
    const pixels = imageData.data;
    for (let i = 0; i < w * h; i++) {
      const cls = classMapData[i];
      const off = cls * 4;
      pixels[i * 4 + 0] = PALETTE_LUT[off];
      pixels[i * 4 + 1] = PALETTE_LUT[off + 1];
      pixels[i * 4 + 2] = PALETTE_LUT[off + 2];
      pixels[i * 4 + 3] = 255;
    }
    ctx.putImageData(imageData, 0, 0);

    const aw = state?.autoware || {};
    const bev = aw.bev || {};
    const bevSize = bev.bev_size || w;
    const rangeFwdFt = bev.range_fwd_ft || 100;
    const rangeSideFt = bev.range_side_ft || 50;
    const rangeFwdM = rangeFwdFt * FT_TO_M;
    const rangeSideM = rangeSideFt * FT_TO_M;

    drawGrid(ctx, w, h, rangeFwdFt, rangeSideFt);
    drawEgo(ctx, w, h);
    drawTrajectory(ctx, aw, bevSize, rangeFwdM, rangeSideM);
    drawBrakeStatus(ctx, w, h, aw);
  }, [state]);

  useEffect(() => {
    const offImg = new Image();
    offscreenRef.current = offImg;

    function poll() {
      if (pendingRef.current) return;
      pendingRef.current = true;
      offImg.src = `/cam/bev_classmap.png?t=${Date.now()}`;
    }

    offImg.onload = () => {
      pendingRef.current = false;
      const w = offImg.naturalWidth;
      const h = offImg.naturalHeight;
      if (w === 0 || h === 0) return;

      const tmpCanvas = document.createElement('canvas');
      tmpCanvas.width = w;
      tmpCanvas.height = h;
      const tmpCtx = tmpCanvas.getContext('2d');
      tmpCtx.drawImage(offImg, 0, 0);
      const imgData = tmpCtx.getImageData(0, 0, w, h);
      const classMap = new Uint8Array(w * h);
      for (let i = 0; i < w * h; i++) {
        classMap[i] = imgData.data[i * 4];
      }
      renderFrame(classMap, w, h);
    };

    offImg.onerror = () => {
      pendingRef.current = false;
    };

    poll();
    intervalRef.current = setInterval(poll, REFRESH_MS);
    return () => {
      clearInterval(intervalRef.current);
      offImg.onload = null;
      offImg.onerror = null;
    };
  }, [renderFrame]);

  return (
    <canvas
      ref={canvasRef}
      className="block w-full h-full object-contain bg-[#1e1e1e]"
      style={{ imageRendering: 'pixelated' }}
    />
  );
}

function drawGrid(ctx, w, h, rangeFwdFt, rangeSideFt) {
  ctx.lineWidth = 1;

  for (let ft = 5; ft <= rangeFwdFt; ft += 5) {
    const by = Math.round((1 - ft * FT_TO_M / (rangeFwdFt * FT_TO_M)) * h);
    if (by < 0 || by >= h) continue;
    const isMajor = ft % 10 === 0;
    ctx.strokeStyle = isMajor ? 'rgba(180,180,180,0.5)' : 'rgba(90,90,90,0.4)';
    ctx.beginPath();
    ctx.moveTo(0, by + 0.5);
    ctx.lineTo(w, by + 0.5);
    ctx.stroke();
    if (isMajor) {
      ctx.font = '11px sans-serif';
      ctx.fillStyle = 'rgba(255,255,255,0.85)';
      ctx.fillText(`${ft}ft`, w / 2 + 6, by - 3);
    }
  }

  for (let ft = -Math.floor(rangeSideFt); ft <= Math.floor(rangeSideFt); ft += 5) {
    if (ft === 0) continue;
    const bx = Math.round((ft / rangeSideFt * 0.5 + 0.5) * w);
    if (bx < 0 || bx >= w) continue;
    const isMajor = ft % 10 === 0;
    ctx.strokeStyle = isMajor ? 'rgba(180,180,180,0.5)' : 'rgba(90,90,90,0.4)';
    ctx.beginPath();
    ctx.moveTo(bx + 0.5, 0);
    ctx.lineTo(bx + 0.5, h);
    ctx.stroke();
    if (isMajor) {
      ctx.font = '11px sans-serif';
      ctx.fillStyle = 'rgba(255,255,255,0.85)';
      ctx.fillText(`${ft > 0 ? '+' : ''}${ft}ft`, bx + 3, h - 8);
    }
  }

  ctx.strokeStyle = 'rgba(220,220,220,0.6)';
  ctx.beginPath();
  ctx.moveTo(w / 2 + 0.5, 0);
  ctx.lineTo(w / 2 + 0.5, h);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(0, h - 0.5);
  ctx.lineTo(w, h - 0.5);
  ctx.stroke();

  ctx.font = '11px sans-serif';
  ctx.fillStyle = 'rgba(255,255,255,0.85)';
  ctx.fillText('x=forward (ft)', w / 2 + 6, 14);
  ctx.fillText('y=lateral (ft)', 6, h - 8);
}

function drawEgo(ctx, w, h) {
  const cx = w / 2;
  const cy = h - 8;
  ctx.fillStyle = '#ffffff';
  ctx.beginPath();
  ctx.moveTo(cx, cy - 14);
  ctx.lineTo(cx - 7, cy);
  ctx.lineTo(cx + 7, cy);
  ctx.closePath();
  ctx.fill();
}

function drawTrajectory(ctx, aw, bevSize, rangeFwdM, rangeSideM) {
  const path = aw.predicted_path;
  if (!Array.isArray(path) || path.length < 2) return;

  const CART_HALF_WIDTH_M = 24.0;

  const pts = [];
  for (const pt of path) {
    if (!Array.isArray(pt) || pt.length < 2) continue;
    const [bx, by] = localToBev(pt[0], pt[1], rangeFwdM, rangeSideM, bevSize);
    if (bx < 0 || bx >= bevSize || by < 0 || by >= bevSize) continue;
    pts.push({ fwd: pt[0], left: pt[1], bx, by });
  }
  if (pts.length < 2) return;

  const leftEdge = [];
  const rightEdge = [];
  for (let i = 0; i < pts.length; i++) {
    let dx, dy;
    if (i === 0) {
      dx = pts[1].bx - pts[0].bx;
      dy = pts[1].by - pts[0].by;
    } else if (i === pts.length - 1) {
      dx = pts[i].bx - pts[i - 1].bx;
      dy = pts[i].by - pts[i - 1].by;
    } else {
      dx = pts[i + 1].bx - pts[i - 1].bx;
      dy = pts[i + 1].by - pts[i - 1].by;
    }
    const len = Math.sqrt(dx * dx + dy * dy) || 1;
    const nx = -dy / len;
    const ny = dx / len;
    const halfPx = CART_HALF_WIDTH_M / rangeSideM * 0.5 * bevSize;
    leftEdge.push([pts[i].bx + nx * halfPx, pts[i].by + ny * halfPx]);
    rightEdge.push([pts[i].bx - nx * halfPx, pts[i].by - ny * halfPx]);
  }

  ctx.fillStyle = 'rgba(255,255,0,0.35)';
  ctx.beginPath();
  ctx.moveTo(leftEdge[0][0], leftEdge[0][1]);
  for (let i = 1; i < leftEdge.length; i++) ctx.lineTo(leftEdge[i][0], leftEdge[i][1]);
  for (let i = rightEdge.length - 1; i >= 0; i--) ctx.lineTo(rightEdge[i][0], rightEdge[i][1]);
  ctx.closePath();
  ctx.fill();

  ctx.strokeStyle = 'rgba(255,255,0,0.9)';
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  ctx.moveTo(leftEdge[0][0], leftEdge[0][1]);
  for (let i = 1; i < leftEdge.length; i++) ctx.lineTo(leftEdge[i][0], leftEdge[i][1]);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(rightEdge[0][0], rightEdge[0][1]);
  for (let i = 1; i < rightEdge.length; i++) ctx.lineTo(rightEdge[i][0], rightEdge[i][1]);
  ctx.stroke();
}

function drawBrakeStatus(ctx, w, h, aw) {
  const seg = aw.segmentation || {};
  const ps = seg.protective_stop || {};
  const active = ps.active;
  const brake01 = parseFloat(ps.brake_01) || 0;

  if (active) {
    ctx.font = 'bold 32px sans-serif';
    ctx.fillStyle = 'rgba(0,0,0,0.7)';
    ctx.fillText('STOP', w / 2 - 48, 44);
    ctx.fillStyle = '#ff2828';
    ctx.fillText('STOP', w / 2 - 50, 42);
  } else if (brake01 > 0.02) {
    const txt = `BRAKE ${brake01.toFixed(2)}`;
    ctx.font = 'bold 18px sans-serif';
    ctx.fillStyle = 'rgba(0,0,0,0.7)';
    ctx.fillText(txt, w / 2 - 52, 40);
    ctx.fillStyle = '#ffd200';
    ctx.fillText(txt, w / 2 - 54, 38);
  }
}

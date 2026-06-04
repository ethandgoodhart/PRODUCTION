import React, { useEffect, useRef } from 'react';
import * as THREE from 'three';

export default function ThreeScene({ state }) {
  const containerRef = useRef(null);
  const sceneRef = useRef(null);
  const rafRef = useRef(null);

  useEffect(() => {
    if (sceneRef.current) return;

    const container = containerRef.current;
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xffffff);
    scene.fog = new THREE.Fog(0xffffff, 14, 42);

    const camera = new THREE.PerspectiveCamera(
      38,
      window.innerWidth / window.innerHeight,
      0.1,
      500
    );
    camera.position.set(0, 7.5, 13.0);
    camera.lookAt(0, 0.6, -8);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    container.appendChild(renderer.domElement);

    // Lighting
    scene.add(new THREE.HemisphereLight(0xffffff, 0xe5e7eb, 1.0));
    const key = new THREE.DirectionalLight(0xffffff, 1.25);
    key.position.set(6, 14, 8);
    key.castShadow = true;
    key.shadow.mapSize.set(2048, 2048);
    key.shadow.camera.left = -10;
    key.shadow.camera.right = 10;
    key.shadow.camera.top = 10;
    key.shadow.camera.bottom = -10;
    key.shadow.camera.near = 1;
    key.shadow.camera.far = 40;
    key.shadow.bias = -0.0004;
    key.shadow.radius = 6;
    scene.add(key);
    const fill = new THREE.DirectionalLight(0xffffff, 0.4);
    fill.position.set(-5, 6, 4);
    scene.add(fill);

    // Road
    const roadWidth = 10;
    const roadLength = 150;
    const road = new THREE.Mesh(
      new THREE.PlaneGeometry(roadWidth, roadLength, 1, 1),
      new THREE.MeshStandardMaterial({ color: 0xeaecef, roughness: 0.95, metalness: 0.0 })
    );
    road.rotation.x = -Math.PI / 2;
    road.position.set(0, 0, -roadLength / 2 + 18);
    road.receiveShadow = true;
    scene.add(road);

    const shoulder = new THREE.Mesh(
      new THREE.PlaneGeometry(200, roadLength),
      new THREE.MeshStandardMaterial({ color: 0xf6f7f9, roughness: 1, metalness: 0 })
    );
    shoulder.rotation.x = -Math.PI / 2;
    shoulder.position.set(0, -0.01, -roadLength / 2 + 18);
    shoulder.receiveShadow = true;
    scene.add(shoulder);

    // Lane dashes
    const DASH_LEN = 2.2;
    const DASH_GAP = 2.6;
    const DASH_STRIDE = DASH_LEN + DASH_GAP;
    const DASH_Z_FAR = -roadLength + 18;
    const DASH_Z_NEAR = DASH_Z_FAR + roadLength;
    const laneDashes = [];

    function addLine({ x, dashed = false, color = 0x1a1b1f, width = 0.18 }) {
      const length = roadLength;
      if (!dashed) {
        const m = new THREE.Mesh(
          new THREE.PlaneGeometry(width, length),
          new THREE.MeshBasicMaterial({ color })
        );
        m.rotation.x = -Math.PI / 2;
        m.position.set(x, 0.012, DASH_Z_FAR + length / 2);
        scene.add(m);
      } else {
        const count = Math.floor(length / DASH_STRIDE);
        const geo = new THREE.PlaneGeometry(width, DASH_LEN);
        const mat = new THREE.MeshBasicMaterial({ color });
        for (let i = 0; i < count; i++) {
          const m = new THREE.Mesh(geo, mat);
          m.rotation.x = -Math.PI / 2;
          m.position.set(x, 0.012, DASH_Z_FAR + i * DASH_STRIDE + DASH_LEN / 2);
          scene.add(m);
          laneDashes.push(m);
        }
      }
    }

    addLine({ x: -roadWidth / 2, dashed: false, color: 0x1f2328, width: 0.2 });
    addLine({ x: roadWidth / 2, dashed: false, color: 0x1f2328, width: 0.2 });
    addLine({ x: -roadWidth / 6, dashed: true, color: 0x9aa0a8, width: 0.16 });
    addLine({ x: roadWidth / 6, dashed: true, color: 0x9aa0a8, width: 0.16 });

    // Lanes texture projection
    const LANES_PLANE_W = 6.5;
    const LANES_PLANE_LEN = 24;
    const LANES_PLANE_NEAR_Z = -2.5;
    const LANES_PLANE_FAR_Z = LANES_PLANE_NEAR_Z - LANES_PLANE_LEN;
    const LANES_REFRESH_MS = 500;

    const lanesImg = new Image();
    lanesImg.crossOrigin = 'anonymous';
    const lanesTexture = new THREE.Texture(lanesImg);
    lanesTexture.colorSpace = THREE.SRGBColorSpace;
    lanesTexture.minFilter = THREE.LinearFilter;
    lanesTexture.magFilter = THREE.LinearFilter;
    lanesTexture.generateMipmaps = false;

    const LANES_OPACITY = 0.45;
    const lanesMaterial = new THREE.MeshBasicMaterial({
      map: lanesTexture,
      transparent: true,
      depthWrite: false,
      opacity: LANES_OPACITY,
    });
    lanesMaterial.onBeforeCompile = (shader) => {
      shader.fragmentShader = shader.fragmentShader.replace(
        '#include <alphamap_fragment>',
        "diffuseColor.a *= smoothstep(0.06, 0.40, max(max(diffuseColor.r, diffuseColor.g), diffuseColor.b));"
      );
    };
    const lanesPlane = new THREE.Mesh(
      new THREE.PlaneGeometry(LANES_PLANE_W, LANES_PLANE_LEN),
      lanesMaterial
    );
    lanesPlane.rotation.x = -Math.PI / 2;
    lanesPlane.position.set(0, 0.025, (LANES_PLANE_NEAR_Z + LANES_PLANE_FAR_Z) / 2);
    lanesPlane.renderOrder = 3;
    lanesPlane.visible = false;
    scene.add(lanesPlane);

    lanesImg.onload = () => {
      lanesTexture.needsUpdate = true;
      lanesPlane.visible = true;
    };
    lanesImg.onerror = () => {
      lanesPlane.visible = false;
    };

    function refreshLanesTexture() {
      if (!window.__showLanesTexture) {
        lanesPlane.visible = false;
        return;
      }
      lanesImg.src = `/cam/lanes_solo.jpg?t=${Date.now()}`;
    }
    refreshLanesTexture();
    const lanesInterval = setInterval(refreshLanesTexture, LANES_REFRESH_MS);

    // Predicted path
    const PATH_SEGS = 60;
    const PATH_TUBE_SEGS = 80;
    const PATH_LENGTH = 70;
    const PATH_TONES = {
      human: { core: 0x1f6feb, halo: 0x4b8dff, haloOpacity: 0.12 },
      autoware: { core: 0xd4a017, halo: 0xf2c94c, haloOpacity: 0.2 },
    };
    const PATH_TRAJ_SCALE = 1.0;
    const TUBE_RADIUS_CORE = 1.0;
    const TUBE_RADIUS_HALO = 1.47;

    function curveFromTrajectory(traj) {
      const pts = [new THREE.Vector3(0, 0.09, 1.5)];
      const xSign = window.__predictedPathInvertX ? -1 : 1;
      for (const [xf, yl] of traj) {
        const sx = xSign * -yl * PATH_TRAJ_SCALE;
        const sz = -xf * PATH_TRAJ_SCALE;
        pts.push(new THREE.Vector3(sx, 0.09, sz));
      }
      return new THREE.CatmullRomCurve3(pts);
    }

    function curveFromSteerDeg(steerDeg) {
      const lateral = (steerDeg / 45) * 7;
      const pts = [];
      for (let i = 0; i <= PATH_SEGS; i++) {
        const t = i / PATH_SEGS;
        const z = 1.5 - t * PATH_LENGTH;
        const bend = t * t * lateral;
        pts.push(new THREE.Vector3(bend, 0.09, z));
      }
      return new THREE.CatmullRomCurve3(pts);
    }

    function buildCurve() {
      const traj = window.__predictedPath;
      if (Array.isArray(traj) && traj.length >= 2) {
        return curveFromTrajectory(traj);
      }
      return curveFromSteerDeg(window.__steerDeg || 0);
    }

    const initialTone = PATH_TONES.human;
    const pathCoreMat = new THREE.MeshBasicMaterial({
      color: initialTone.core,
      side: THREE.DoubleSide,
    });
    const pathHaloMat = new THREE.MeshBasicMaterial({
      color: initialTone.halo,
      transparent: true,
      opacity: initialTone.haloOpacity,
      side: THREE.DoubleSide,
      depthWrite: false,
    });
    const pathCore = new THREE.Mesh(
      new THREE.TubeGeometry(buildCurve(), PATH_TUBE_SEGS, TUBE_RADIUS_CORE, 8, false),
      pathCoreMat
    );
    const pathHalo = new THREE.Mesh(
      new THREE.TubeGeometry(buildCurve(), PATH_TUBE_SEGS, TUBE_RADIUS_HALO, 8, false),
      pathHaloMat
    );
    pathCore.renderOrder = 2;
    pathHalo.renderOrder = 1;
    scene.add(pathCore);
    scene.add(pathHalo);

    let lastFingerprint = '';
    function trajFingerprint() {
      const traj = window.__predictedPath;
      if (Array.isArray(traj) && traj.length >= 2) {
        const inv = window.__predictedPathInvertX ? 'I' : 'N';
        return 'T:' + inv + ':' + traj.map((p) => `${p[0].toFixed(2)},${p[1].toFixed(2)}`).join('|');
      }
      return 'S:' + Math.round((window.__steerDeg || 0) * 12) / 12;
    }

    function updatePaths() {
      const fp = trajFingerprint();
      if (fp === lastFingerprint) return;
      lastFingerprint = fp;
      const curve = buildCurve();
      const nextCore = new THREE.TubeGeometry(curve, PATH_TUBE_SEGS, TUBE_RADIUS_CORE, 8, false);
      const nextHalo = new THREE.TubeGeometry(curve, PATH_TUBE_SEGS, TUBE_RADIUS_HALO, 8, false);
      pathCore.geometry.dispose();
      pathHalo.geometry.dispose();
      pathCore.geometry = nextCore;
      pathHalo.geometry = nextHalo;
    }

    let currentTone = 'human';
    window.__setPathTone = function (tone) {
      if (!PATH_TONES[tone] || tone === currentTone) return;
      currentTone = tone;
      const t = PATH_TONES[tone];
      pathCoreMat.color.setHex(t.core);
      pathHaloMat.color.setHex(t.halo);
      pathHaloMat.opacity = t.haloOpacity;
    };

    // Golf cart model
    const cart = new THREE.Group();
    const cartWhite = new THREE.MeshStandardMaterial({ color: 0xfafbfc, roughness: 0.38, metalness: 0.05 });
    const cartAccent = new THREE.MeshStandardMaterial({ color: 0x1f2328, roughness: 0.45, metalness: 0.2 });
    const seatMat = new THREE.MeshStandardMaterial({ color: 0x2a2d33, roughness: 0.92 });
    const seatCushion = new THREE.MeshStandardMaterial({ color: 0x3a3e45, roughness: 0.85 });
    const tireMat = new THREE.MeshStandardMaterial({ color: 0x15171b, roughness: 0.9 });
    const rimMat = new THREE.MeshStandardMaterial({ color: 0xcbd0d8, roughness: 0.3, metalness: 0.75 });

    function roundedBox(w, h, d, r, mat) {
      const shape = new THREE.Shape();
      shape.moveTo(-w / 2 + r, -h / 2);
      shape.lineTo(w / 2 - r, -h / 2);
      shape.quadraticCurveTo(w / 2, -h / 2, w / 2, -h / 2 + r);
      shape.lineTo(w / 2, h / 2 - r);
      shape.quadraticCurveTo(w / 2, h / 2, w / 2 - r, h / 2);
      shape.lineTo(-w / 2 + r, h / 2);
      shape.quadraticCurveTo(-w / 2, h / 2, -w / 2, h / 2 - r);
      shape.lineTo(-w / 2, -h / 2 + r);
      shape.quadraticCurveTo(-w / 2, -h / 2, -w / 2 + r, -h / 2);
      const geo = new THREE.ExtrudeGeometry(shape, {
        depth: d, bevelEnabled: true, bevelSegments: 3, bevelSize: 0.035, bevelThickness: 0.035,
      });
      geo.translate(0, 0, -d / 2);
      const m = new THREE.Mesh(geo, mat);
      m.castShadow = true;
      m.receiveShadow = true;
      return m;
    }

    const deck = roundedBox(1.85, 0.12, 0.95, 0.06, cartWhite);
    deck.position.set(0, 0.6, -1.35);
    cart.add(deck);
    const rearWall = roundedBox(1.85, 0.55, 0.08, 0.04, cartWhite);
    rearWall.position.set(0, 0.82, -1.8);
    cart.add(rearWall);
    const chassis = roundedBox(1.9, 0.32, 2.3, 0.1, cartWhite);
    chassis.position.set(0, 0.52, 0);
    cart.add(chassis);
    const cowl = roundedBox(1.85, 0.35, 0.9, 0.14, cartWhite);
    cowl.position.set(0, 0.72, 1.05);
    cart.add(cowl);
    const floorMesh = new THREE.Mesh(new THREE.BoxGeometry(1.75, 0.05, 1.0), cartAccent);
    floorMesh.position.set(0, 0.37, 0.25);
    floorMesh.castShadow = true;
    floorMesh.receiveShadow = true;
    cart.add(floorMesh);
    const seat = new THREE.Mesh(new THREE.BoxGeometry(1.75, 0.18, 0.7), seatCushion);
    seat.position.set(0, 0.75, -0.45);
    seat.castShadow = true;
    cart.add(seat);
    const seatBack = roundedBox(1.75, 0.7, 0.14, 0.06, seatMat);
    seatBack.position.set(0, 1.2, -0.8);
    seatBack.rotation.x = -0.08;
    cart.add(seatBack);
    const divider = new THREE.Mesh(new THREE.BoxGeometry(0.04, 0.3, 0.65), cartAccent);
    divider.position.set(0, 1.0, -0.45);
    cart.add(divider);
    const roof = roundedBox(2.0, 0.08, 2.1, 0.08, cartWhite);
    roof.position.set(0, 2.0, -0.1);
    cart.add(roof);

    function pillar(x, z) {
      const p = new THREE.Mesh(new THREE.CylinderGeometry(0.045, 0.045, 1.3, 14), cartAccent);
      p.position.set(x, 1.3, z);
      p.castShadow = true;
      cart.add(p);
    }
    pillar(-0.92, 0.85);
    pillar(0.92, 0.85);
    pillar(-0.92, -1.1);
    pillar(0.92, -1.1);

    const steering = new THREE.Group();
    const wheelRing = new THREE.Mesh(new THREE.TorusGeometry(0.2, 0.028, 12, 32), cartAccent);
    steering.add(wheelRing);
    const wheelHub = new THREE.Mesh(new THREE.CylinderGeometry(0.04, 0.04, 0.04, 16), cartAccent);
    wheelHub.rotation.x = Math.PI / 2;
    steering.add(wheelHub);
    const column = new THREE.Mesh(new THREE.CylinderGeometry(0.03, 0.03, 0.55, 12), cartAccent);
    column.rotation.x = Math.PI / 2 - 0.35;
    column.position.set(0, -0.28, 0.15);
    steering.add(column);
    steering.position.set(-0.42, 1.05, 0.45);
    steering.rotation.x = -0.3;
    cart.add(steering);

    const dash = roundedBox(1.75, 0.22, 0.12, 0.03, cartAccent);
    dash.position.set(0, 1.05, 0.7);
    cart.add(dash);

    function wheel(x, z) {
      const g = new THREE.Group();
      const tire = new THREE.Mesh(new THREE.CylinderGeometry(0.34, 0.34, 0.24, 28), tireMat);
      tire.rotation.z = Math.PI / 2;
      tire.castShadow = true;
      g.add(tire);
      const rim = new THREE.Mesh(new THREE.CylinderGeometry(0.19, 0.19, 0.25, 16), rimMat);
      rim.rotation.z = Math.PI / 2;
      g.add(rim);
      for (let i = 0; i < 5; i++) {
        const s = new THREE.Mesh(new THREE.BoxGeometry(0.35, 0.025, 0.06), rimMat);
        s.rotation.x = (i / 5) * Math.PI * 2;
        g.add(s);
      }
      g.position.set(x, 0.34, z);
      return g;
    }
    cart.add(wheel(-0.92, 0.9));
    cart.add(wheel(0.92, 0.9));
    cart.add(wheel(-0.92, -1.1));
    cart.add(wheel(0.92, -1.1));

    const bumper = new THREE.Mesh(new THREE.BoxGeometry(1.85, 0.08, 0.06), cartAccent);
    bumper.position.set(0, 0.55, 1.52);
    cart.add(bumper);

    function headlight(x) {
      const h = new THREE.Mesh(
        new THREE.CircleGeometry(0.09, 24),
        new THREE.MeshStandardMaterial({ color: 0xffffff, emissive: 0xfff7e0, emissiveIntensity: 0.6, roughness: 0.2 })
      );
      h.position.set(x, 0.78, 1.51);
      cart.add(h);
      const socket = new THREE.Mesh(
        new THREE.RingGeometry(0.09, 0.11, 24),
        new THREE.MeshBasicMaterial({ color: 0x1f2328, side: THREE.DoubleSide })
      );
      socket.position.set(x, 0.78, 1.512);
      cart.add(socket);
    }
    headlight(-0.7);
    headlight(0.7);

    function tail(x) {
      const r = new THREE.Mesh(
        new THREE.BoxGeometry(0.18, 0.08, 0.04),
        new THREE.MeshStandardMaterial({ color: 0xc0413a, roughness: 0.5 })
      );
      r.position.set(x, 0.9, -1.84);
      cart.add(r);
    }
    tail(-0.75);
    tail(0.75);

    cart.position.set(0, 0, 0);
    cart.rotation.y = Math.PI;
    scene.add(cart);

    const contact = new THREE.Mesh(
      new THREE.CircleGeometry(2.0, 32),
      new THREE.MeshBasicMaterial({ color: 0x000000, transparent: true, opacity: 0.1 })
    );
    contact.rotation.x = -Math.PI / 2;
    contact.position.y = 0.005;
    scene.add(contact);

    // Animate
    const SPEED_WORLD_PER_MPH = 0.79;
    let t = 0;
    let prevFrame = performance.now();

    function animate(now) {
      if (now === undefined) now = performance.now();
      const dt = Math.min(0.05, (now - prevFrame) / 1000);
      prevFrame = now;
      t += dt;
      cart.position.y = Math.sin(t * 2.0) * 0.008;

      updatePaths();

      const mph = Math.max(0, Number(window.__mph) || 0);
      if (mph > 0) {
        const slide = mph * SPEED_WORLD_PER_MPH * dt;
        const wrap = (laneDashes.length / 2) * DASH_STRIDE;
        for (const m of laneDashes) {
          m.position.z += slide;
          if (m.position.z > DASH_Z_NEAR) {
            m.position.z -= wrap;
          }
        }
      }

      renderer.render(scene, camera);
      rafRef.current = requestAnimationFrame(animate);
    }
    animate();

    // Resize
    const handleResize = () => {
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    };
    window.addEventListener('resize', handleResize);

    sceneRef.current = {
      renderer,
      scene,
      camera,
      handleResize,
      lanesInterval,
    };

    return () => {
      cancelAnimationFrame(rafRef.current);
      clearInterval(lanesInterval);
      window.removeEventListener('resize', handleResize);
      renderer.dispose();
      container.removeChild(renderer.domElement);
      sceneRef.current = null;
    };
  }, []);

  // Update window globals from state for the 3D scene to read
  useEffect(() => {
    if (!state) return;
    const aw = state.autoware || {};
    const mph = Number(state.mph) || 0;
    window.__mph = mph;

    const steerDeg = Number(state.steer_deg) || 0;
    if (aw.running && aw.inference) {
      window.__steerDeg = Number(aw.steer_deg_raw) || 0;
    } else {
      window.__steerDeg = steerDeg;
    }

    if (
      aw.running &&
      aw.inference &&
      Array.isArray(aw.predicted_path) &&
      aw.predicted_path.length >= 2
    ) {
      window.__predictedPath = aw.predicted_path;
      window.__predictedPathInvertX =
        (aw.model || '').toLowerCase() === 'segmentation';
    } else {
      window.__predictedPath = null;
      window.__predictedPathInvertX = false;
    }

    let rawStatus = state.autosteer_status || (aw.running ? 'autoware' : 'idle');
    if (state.human_override) rawStatus = 'human';
    const engaged = rawStatus === 'autoware';
    if (typeof window.__setPathTone === 'function') {
      window.__setPathTone(engaged ? 'autoware' : 'human');
    }
  }, [state]);

  return <div id="scene-container" className="fixed inset-0 z-0" ref={containerRef} />;
}

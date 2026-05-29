#!/usr/bin/env python3
"""Simple multi-camera viewer + recorder web UI.

Usage:
    python record_cameras.py
    python record_cameras.py --front-left 10 --front 16 --front-right 4 --back-right 0
    python record_cameras.py --back-left 8
    python record_cameras.py --front-narrow 14

Open http://<jetson-ip>:8080 in a browser.

Cameras are auto-discovered by USB topology (e.g. usb-...-4.1.2) so the
/dev/videoN numbers can shift across reboots / hub rebinds without
breaking the mapping. Pass --front-narrow N etc to override discovery
for a single channel; pass --front-narrow -1 to disable it.

Current Thor wiring: five cameras on the dock hub and one camera plugged
directly into the Thor. The auto-discovery table below is keyed by USB
topology, not /dev/videoN, so unplug/replug ordering should not matter.

Also captures the iPhone ARKit ego-motion stream (TCP via usbmuxd; same
source as PRODUCTION's ego_state_writer). When recording, every fresh
sample is appended to ego.jsonl in the recording folder.
"""
import argparse
import datetime as dt
import json
import math
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template_string, request

try:
    import pyrealsense2 as rs
    _HAVE_RS = True
except ImportError:
    rs = None  # type: ignore[assignment]
    _HAVE_RS = False

# Same source PRODUCTION's web UI reads. ego_state_writer.py mirrors the
# iPhone JSONL stream into this file at 10 Hz; we read it the same way.
EGO_STATE_FILE = os.environ.get("EGO_STATE_FILE", "/tmp/ego_state.json")
EGO_FRESH_S = 1.0  # writer publishes at 10 Hz; >1 s = writer died
EGO_LINK_SCRIPT = os.path.expanduser("~/ego_sensor/ego_link.sh")
EGO_WRITER_SCRIPT = str(Path(__file__).resolve().parent / "ego_state_writer.py")

# GPS lives on a sibling TCP port to the ego-motion stream. Same usbmuxd
# tunnel pattern (iproxy 5006 5006), separate iOS NWListener. Wire format
# matches jetson/gps_sensor.py in Georg-Stanford-GC-iOS-Sensor.
GPS_FRESH_S = 5.0  # fixes arrive 1-5 Hz depending on phone state

# ps5_drive.py publishes a full cart-state snapshot (steer_deg, gas,
# brake, gas_frac, brake_frac, mph, ...) here every control tick. Same
# file PRODUCTION's start.sh + web UI use, so running both is harmless
# as long as only one ps5_drive is alive.
CART_STATE_FILE = os.environ.get("CART_STATE_FILE", "/tmp/cart_state.json")
CART_FRESH_S = 0.5  # ps5_drive publishes at CONTROL_HZ (~60 Hz)
AUTOWARE_STATE_FILE = os.environ.get("AUTOWARE_STATE_FILE", "/tmp/autoware_state.json")
SEGMENTATION_MAP_FILE = os.environ.get(
    "SEGMENTATION_MAP_FILE", "/tmp/cart_frames/segmentation_map.json"
)
SEGMENTATION_MAP_FRESH_S = 1.0
MODEL_FRAMES_DIR = Path(os.environ.get("CART_FRAMES_DIR", "/tmp/cart_frames"))
MODEL_TILE_FILES = (
    ("model-front", "front.jpg"),
    ("segmentation", "seg.jpg"),
    ("bev-planner", "bev.jpg"),
)
PRODUCTION_DIR = Path(__file__).resolve().parent.parent
PS5_DRIVE_SCRIPT = str(Path(__file__).resolve().parent / "ps5_drive.py")

# Grid order: row 1 = front-left / front / front-right; row 2 = back-left
# / back-right / front-narrow. CAM_OPEN_ORDER matches so the most-likely
# -needed cams open first.
CAM_NAMES = ["front-left", "front", "front-right", "back-left", "back-right", "front-narrow"]
CAM_OPEN_ORDER = list(CAM_NAMES)

# Per-cam OpenCV flip codes. The old cart had front-narrow mounted
# upside-down (flip -1); on the current rig it's mounted right-side up,
# so the entry is removed.
CAMERA_FLIP: dict[str, int] = {}

# Per-cam v4l2 control overrides applied after _open(). Each entry is
# (control_name, value). auto_exposure=3 (Aperture Priority) hands exposure
# back to the cam's firmware; brightness=32 lifts the front view without
# pushing it to the top of the UVC [-64, 64] range.
CAMERA_CONTROLS: dict[str, tuple[tuple[str, int], ...]] = {
    "front": (
        ("auto_exposure", 3),
        ("brightness", 32),
    ),
    "back-right": (
        ("auto_exposure", 3),
        ("brightness", 32),
        ("gain", 200),
    ),
}

# Logical camera name -> stable USB sub-path suffixes from the bus_info
# string v4l2-ctl prints. New Thor wiring is five cameras on the dock hub
# (usb-4.2.*) plus one direct Thor camera (usually usb-4.1.3). The old
# usb-4.1.* suffixes stay as fallbacks so the recorder still works if the
# cart is moved back to the previous cabling.
CAM_USB_PATHS: dict[str, tuple[str, ...]] = {
    # Current Thor wiring (6 cams: 3 GS on the chained hub, 2 GS + 1 H264
    # on the dock hub). Earlier suffixes kept as fallbacks for older
    # wirings of the same cart.
    "front": (
        "usb-4.1.1.1",  # cam_front symlink
        "usb-4.1.3",
    ),
    "front-right": (
        "usb-4.1.1.2",  # cam_front_right symlink
        "usb-4.2.1.2",
    ),
    "front-left": (
        "usb-4.1.1.3",  # cam_front_left symlink
        "usb-4.2.1.4",
        "usb-4.1.1.4",
    ),
    "back-left": (
        "usb-2.1",      # unlabelled GS on hub port 1
        "usb-4.2.1.3",
    ),
    "back-right": (
        "usb-2.2",      # unlabelled GS on hub port 2
        "usb-4.2.1.1",
    ),
    "front-narrow": (
        "usb-2.3",      # H264 USB Cam on hub port 3
        "usb-4.2.2",
        "usb-4.1.2",
    ),
}


def discover_video_devices() -> dict[str, int]:
    """Return {logical_name: /dev/videoN index} by parsing v4l2-ctl.

    UVC cameras usually expose one capture node and one metadata node;
    older H264 cameras can expose more. Only the capture node advertises
    MJPG, so we probe each node for MJPG support and return the lowest
    matching index per USB path.
    """
    try:
        out = subprocess.run(
            ["v4l2-ctl", "--list-devices"],
            capture_output=True, text=True, timeout=5,
        ).stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return {}
    # Parse blocks: "Name (bus_info):\n\t/dev/videoN\n\t..."
    by_path: dict[str, list[int]] = {}
    current_path: str | None = None
    for line in out.splitlines():
        if line and not line.startswith("\t"):
            m = re.search(r"\(([^)]+)\)\s*:?\s*$", line.strip())
            current_path = m.group(1) if m else None
        elif line.startswith("\t/dev/video"):
            try:
                idx = int(line.strip().removeprefix("/dev/video"))
            except ValueError:
                continue
            if current_path is not None:
                by_path.setdefault(current_path, []).append(idx)

    def first_mjpg(indices: list[int]) -> int | None:
        for idx in sorted(indices):
            try:
                fmts = subprocess.run(
                    ["v4l2-ctl", "-d", f"/dev/video{idx}", "--list-formats"],
                    capture_output=True, text=True, timeout=2,
                ).stdout
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue
            if "MJPG" in fmts:
                return idx
        return None

    mjpg_by_path: dict[str, int] = {}
    for path, indices in by_path.items():
        idx = first_mjpg(indices)
        if idx is not None:
            mjpg_by_path[path] = idx

    mapping: dict[str, int] = {}
    used_paths: set[str] = set()
    for name, suffixes in CAM_USB_PATHS.items():
        for suffix in suffixes:
            match = next(
                (p for p in mjpg_by_path if p.endswith(suffix) and p not in used_paths),
                None,
            )
            if match is None:
                continue
            mapping[name] = mjpg_by_path[match]
            used_paths.add(match)
            break

    # If the direct Thor camera lands on a topology suffix we have not seen
    # yet, still show and record it. Assign remaining MJPG-capable devices
    # to any unfilled camera slots in the declared grid order.
    remaining = [
        (path, idx) for path, idx in sorted(mjpg_by_path.items())
        if path not in used_paths
    ]
    for name in CAM_NAMES:
        if name in mapping or not remaining:
            continue
        path, idx = remaining.pop(0)
        mapping[name] = idx
        used_paths.add(path)
        print(f"[CAM] {name}: using unmatched USB path {path} -> /dev/video{idx}")
    return mapping


class Camera:
    def __init__(self, name: str, device: int, width: int, height: int, fps: int,
                 force_mjpg: bool = True):
        self.name = name
        self.device = device
        self.width = width
        self.height = height
        self.fps = fps
        self.force_mjpg = force_mjpg
        self.flip_code = CAMERA_FLIP.get(name)
        self.lock = threading.Lock()
        self.frame = None
        self.frames_ok = 0
        self.frames_fail = 0
        self.last_ok_t = 0.0
        self.cap = self._open()
        self.writer: cv2.VideoWriter | None = None
        self.writer_thread: threading.Thread | None = None
        self.writer_stop = threading.Event()
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _open(self):
        cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        if self.force_mjpg:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            print(f"[WARN] Could not open /dev/video{self.device} for {self.name}")
        # Apply per-cam v4l2 control overrides (out-of-band — OpenCV can't
        # touch most of these). v4l2-ctl talks to the same /dev/videoN
        # node and the values stick for the lifetime of the open fd.
        for ctrl, value in CAMERA_CONTROLS.get(self.name, ()):
            try:
                subprocess.run(
                    ["v4l2-ctl", "-d", f"/dev/video{self.device}",
                     "-c", f"{ctrl}={value}"],
                    check=False, timeout=2,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass
        return cap

    def _loop(self):
        consecutive_fail = 0
        period = 1.0 / max(1, self.fps)
        next_tick = time.monotonic()
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                self.frames_fail += 1
                consecutive_fail += 1
                # If a camera streams blank for a while, reopen the device.
                # Some H264 USB hubs hand out an opened-but-silent fd after
                # contention; fastest path back to live frames is reopen.
                if consecutive_fail >= 60:
                    print(f"[CAM] {self.name} (/dev/video{self.device}) stalled — reopening")
                    try:
                        self.cap.release()
                    except Exception:
                        pass
                    time.sleep(0.3)
                    self.cap = self._open()
                    consecutive_fail = 0
                else:
                    time.sleep(0.05)
                next_tick = time.monotonic()
                continue
            consecutive_fail = 0
            self.frames_ok += 1
            self.last_ok_t = time.monotonic()
            if self.flip_code is not None:
                frame = cv2.flip(frame, self.flip_code)
            with self.lock:
                self.frame = frame
            next_tick += period
            sleep_for = next_tick - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_tick = time.monotonic()

    def get_jpeg(self) -> bytes | None:
        frame = self.get_frame()
        if frame is None:
            return None
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        return buf.tobytes() if ok else None

    def get_frame(self):
        with self.lock:
            frame = None if self.frame is None else self.frame.copy()
        return frame

    def _write_loop(self):
        # Write at a fixed wall-clock cadence so saved video duration ==
        # real-time duration. If the camera delivers slower than fps,
        # frames get duplicated; if faster, they get dropped. Either way
        # playback matches reality.
        #
        # Critical: when a tick is late we MUST NOT reset the baseline —
        # we let the loop burst-write to catch up so the total frame
        # count over T real seconds is exactly round(T * fps). Otherwise
        # the file ends up shorter than the recording and plays fast.
        period = 1.0 / self.fps
        next_tick = time.monotonic()
        while not self.writer_stop.is_set():
            with self.lock:
                frame = None if self.frame is None else self.frame
            if frame is not None and self.writer is not None:
                self.writer.write(frame)
            next_tick += period
            sleep_for = next_tick - time.monotonic()
            if sleep_for > 0:
                self.writer_stop.wait(sleep_for)

    def start_recording(self, out_path: Path):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(
            str(out_path), fourcc, self.fps, (self.width, self.height)
        )
        self.writer_stop.clear()
        self.writer_thread = threading.Thread(target=self._write_loop, daemon=True)
        self.writer_thread.start()

    def stop_recording(self):
        if self.writer_thread is not None:
            self.writer_stop.set()
            self.writer_thread.join(timeout=2)
            self.writer_thread = None
        if self.writer is not None:
            self.writer.release()
            self.writer = None

    def release(self):
        self.running = False
        self.thread.join(timeout=1)
        self.stop_recording()
        self.cap.release()


class ModelFrameRecorder:
    """Record a model-published JPEG stream, e.g. /tmp/cart_frames/front.jpg."""

    def __init__(self, name: str, frame_path: Path, fps: int = 15):
        self.name = name
        self.frame_path = frame_path
        self.fps = fps
        self.writer: cv2.VideoWriter | None = None
        self.writer_thread: threading.Thread | None = None
        self.writer_stop = threading.Event()
        self.out_path: Path | None = None
        self.size: tuple[int, int] | None = None
        self.last_frame: np.ndarray | None = None

    def _read_frame(self) -> np.ndarray | None:
        frame = cv2.imread(str(self.frame_path), cv2.IMREAD_COLOR)
        if frame is None:
            return None
        return frame

    def _write_loop(self):
        period = 1.0 / max(1, self.fps)
        next_tick = time.monotonic()
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        while not self.writer_stop.is_set():
            frame = self._read_frame()
            if frame is not None:
                self.last_frame = frame
            elif self.last_frame is not None:
                frame = self.last_frame

            if frame is not None and self.out_path is not None:
                h, w = frame.shape[:2]
                if self.writer is None:
                    self.size = (w, h)
                    self.writer = cv2.VideoWriter(
                        str(self.out_path), fourcc, self.fps, self.size
                    )
                elif self.size is not None and (w, h) != self.size:
                    frame = cv2.resize(frame, self.size)
                if self.writer is not None:
                    self.writer.write(frame)

            next_tick += period
            sleep_for = next_tick - time.monotonic()
            if sleep_for > 0:
                self.writer_stop.wait(sleep_for)
            else:
                next_tick = time.monotonic()

    def start_recording(self, out_path: Path):
        self.out_path = out_path
        self.size = None
        self.last_frame = None
        self.writer = None
        self.writer_stop.clear()
        self.writer_thread = threading.Thread(target=self._write_loop, daemon=True)
        self.writer_thread.start()

    def stop_recording(self):
        if self.writer_thread is not None:
            self.writer_stop.set()
            self.writer_thread.join(timeout=2)
            self.writer_thread = None
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        self.out_path = None


class EgoRecorder:
    """Mirror of PRODUCTION's ego pipeline: spawn ego_link.sh (USB tunnel)
    + ego_state_writer.py (TCP -> /tmp/ego_state.json), then read that
    file just like web/app.py does. While recording, append every fresh
    sample (deduped by t_s) to ego.jsonl in the run folder."""

    def __init__(self, host: str = "127.0.0.1", port: int = 5005,
                 state_file: str = EGO_STATE_FILE,
                 spawn_helpers: bool = True):
        self.host = host
        self.port = port
        self.state_file = state_file
        self._link_proc: subprocess.Popen | None = None
        self._writer_proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._fp = None  # type: ignore[var-annotated]
        self._last_t_s: float | None = None
        self._rec_stop = threading.Event()
        self._rec_thread: threading.Thread | None = None
        if spawn_helpers:
            self._spawn_helpers()

    def _is_running(self, pattern: str) -> bool:
        try:
            r = subprocess.run(["pgrep", "-f", pattern], capture_output=True)
            return r.returncode == 0
        except FileNotFoundError:
            return False

    def _spawn_helpers(self):
        # ego_link.sh — USB tunnel supervisor (idevice_id + iproxy 5005 5005).
        # Skip if PRODUCTION's start.sh is already running it.
        env = os.environ.copy()
        env["EGO_STATE_FILE"] = self.state_file
        log_dir = Path("/tmp")
        if os.path.exists(EGO_LINK_SCRIPT) and not self._is_running("ego_sensor/ego_link.sh"):
            try:
                self._link_proc = subprocess.Popen(
                    ["bash", EGO_LINK_SCRIPT, str(self.port)],
                    stdout=open(log_dir / "ego_link.log", "ab"),
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                print(f"[EGO] spawned ego_link.sh pid={self._link_proc.pid}")
            except Exception as e:
                print(f"[WARN] could not spawn ego_link.sh: {e!r}")
        else:
            print("[EGO] ego_link.sh already running (or missing); not spawning")

        # ego_state_writer.py — TCP reader -> /tmp/ego_state.json.
        if not self._is_running("scripts/ego_state_writer.py"):
            python = shutil.which("python3") or sys.executable
            try:
                self._writer_proc = subprocess.Popen(
                    [python, EGO_WRITER_SCRIPT,
                     "--state-file", self.state_file,
                     "--host", self.host, "--port", str(self.port)],
                    stdout=open(log_dir / "ego_state_writer.log", "ab"),
                    stderr=subprocess.STDOUT,
                    env=env,
                    start_new_session=True,
                )
                print(f"[EGO] spawned ego_state_writer.py pid={self._writer_proc.pid}")
            except Exception as e:
                print(f"[WARN] could not spawn ego_state_writer.py: {e!r}")
        else:
            print("[EGO] ego_state_writer.py already running; not spawning")

    @property
    def available(self) -> bool:
        return True

    def _read_state(self) -> tuple[dict, bool]:
        """Return (payload, stale). Stale if file missing or older than EGO_FRESH_S."""
        try:
            st = os.stat(self.state_file)
        except FileNotFoundError:
            return {}, True
        try:
            with open(self.state_file) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            return {}, True
        ts = float(data.get("ts", 0.0))
        stale = (time.time() - ts) > EGO_FRESH_S
        return data, stale

    def latest(self) -> dict:
        """Snapshot for /ego endpoint — same shape PRODUCTION web/app.py emits."""
        data, stale = self._read_state()
        connected = bool(data.get("connected", False)) and not stale
        out: dict = {
            "connected": connected,
            "host": data.get("host", f"{self.host}:{self.port}"),
            "stale": stale,
        }
        if "sample" in data:
            out["sample"] = data["sample"]
        if "history_len" in data:
            out["history_len"] = data["history_len"]
        return out

    @staticmethod
    def _alpamayo_fields(sample: dict) -> dict:
        """Per-sample fields in Alpamayo-R1 (PhysicalAI-AV) frame.

        Mirrors scripts/alpamayo_infer.py:build_ego_tensors. Axis swap
        from the iPhone publisher frame (+X right, +Y up, -Z forward) to
        the training frame (x=forward, y=left, z=up):

            Alp.x = -iPhone.z
            Alp.y = -iPhone.x
            Alp.z = +iPhone.y

        Yaw stays in the same convention (rotation about vertical). The
        per-sample 3x3 rotation matrix is the absolute yaw rotation in
        the Alpamayo frame; recentering to t0 (so the last frame is
        identity / origin) is done offline from this stream, the same
        way build_ego_tensors does it live.
        """
        x_m = float(sample.get("x_m", 0.0))
        y_m = float(sample.get("y_m", 0.0))
        z_m = float(sample.get("z_m", 0.0))
        yaw = float(sample.get("yaw_rad", 0.0))
        ax = -z_m
        ay = -x_m
        az = y_m
        c, s = math.cos(yaw), math.sin(yaw)
        rot = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
        return {
            "xyz_m": [ax, ay, az],   # forward, left, up
            "yaw_rad": yaw,
            "rot3x3": rot,
            "speed_mps": float(sample.get("speed_mps", 0.0)),
            "yaw_rate_rad_s": float(sample.get("yaw_rate_rad_s", 0.0)),
            "curvature_inv_m": float(sample.get("curvature_inv_m", 0.0)),
        }

    def _record_loop(self, started_wall: float):
        # Header row documents the schema + axis convention so anyone
        # reading the JSONL later doesn't have to reverse-engineer it.
        with self._lock:
            if self._fp is not None:
                header = {
                    "_schema": "caddy.ego.v1",
                    "_iphone_frame": "+X=right, +Y=up, -Z=forward (metres)",
                    "_alpamayo_frame": "x=forward, y=left, z=up (PhysicalAI-AV)",
                    "_axis_map": "alp.x=-iphone.z, alp.y=-iphone.x, alp.z=+iphone.y",
                    "_history_len_for_alpamayo_r1": 16,
                    "_note": "Each row carries both the raw iPhone-frame sample "
                              "and an 'alpamayo' block in PhysicalAI-AV frame. "
                              "Recenter the last 16 alpamayo.xyz_m to (0,0,0) and "
                              "alpamayo.rot3x3 to identity at inference time to "
                              "match scripts/alpamayo_infer.py:build_ego_tensors.",
                    "started_wall_t": started_wall,
                }
                self._fp.write(json.dumps(header) + "\n")
                self._fp.flush()
        # Poll the state file faster than its 10 Hz publish rate.
        while not self._rec_stop.is_set():
            data, stale = self._read_state()
            sample = data.get("sample") if not stale else None
            if sample is not None:
                t_s = sample.get("t_s")
                if t_s is not None and t_s != self._last_t_s:
                    self._last_t_s = t_s
                    rec = {
                        "wall_t": time.time(),
                        "rel_t": time.time() - started_wall,
                        **sample,
                        "alpamayo": self._alpamayo_fields(sample),
                    }
                    with self._lock:
                        if self._fp is not None:
                            self._fp.write(json.dumps(rec) + "\n")
                            self._fp.flush()
            self._rec_stop.wait(0.02)

    def start_recording(self, out_path: Path):
        self._last_t_s = None
        self._fp = open(out_path, "w", buffering=1)
        self._rec_stop.clear()
        self._rec_thread = threading.Thread(
            target=self._record_loop, args=(time.time(),), daemon=True
        )
        self._rec_thread.start()

    def stop_recording(self):
        if self._rec_thread is not None:
            self._rec_stop.set()
            self._rec_thread.join(timeout=2)
            self._rec_thread = None
        with self._lock:
            if self._fp is not None:
                self._fp.close()
                self._fp = None

    def shutdown(self):
        self.stop_recording()
        for proc, name in ((self._writer_proc, "ego_state_writer"),
                           (self._link_proc, "ego_link")):
            if proc is not None and proc.poll() is None:
                try:
                    proc.terminate()
                    proc.wait(timeout=2)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass


class CartRecorder:
    """Drives the cart from a PS5 controller (via scripts/ps5_drive.py) and
    records its full live state — steer_deg, gas, brake, gas_frac,
    brake_frac, mph, etc. — to control.jsonl while a recording is active.

    ps5_drive.py is the same binary PRODUCTION/start.sh runs. We spawn it
    in a bash supervisor loop so a controller drop / reconnect re-arms the
    drive without operator action, matching start.sh behavior."""

    def __init__(self, state_file: str = CART_STATE_FILE,
                 spawn_helper: bool = True):
        self.state_file = state_file
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._fp = None  # type: ignore[var-annotated]
        self._last_ts: float | None = None
        self._rec_stop = threading.Event()
        self._rec_thread: threading.Thread | None = None
        if spawn_helper:
            self._spawn_helper()

    @staticmethod
    def _is_running(pattern: str) -> bool:
        try:
            r = subprocess.run(["pgrep", "-f", pattern], capture_output=True)
            return r.returncode == 0
        except FileNotFoundError:
            return False

    def _spawn_helper(self):
        # Skip if PRODUCTION/start.sh is already managing ps5_drive.
        if self._is_running("scripts/ps5_drive.py"):
            print("[CART] ps5_drive.py already running; not spawning")
            return
        # Mirror start.sh's wait-for-controller + respawn loop. uv run is
        # required because ps5_drive.py uses PRODUCTION's uv-managed
        # Python 3.13 environment.
        supervisor = (
            "while true; do "
            "  while ! grep -qi -E '(sony|dualsense|wireless controller|ps5)' "
            "      /proc/bus/input/devices 2>/dev/null; do sleep 1; done; "
            "  echo \"[record] controller detected — launching ps5_drive\"; "
            f"  uv run python {PS5_DRIVE_SCRIPT} --headless "
            f"      --state-file {self.state_file} || true; "
            "  echo \"[record] ps5_drive exited — waiting for reconnect\"; "
            "  sleep 1; "
            "done"
        )
        try:
            log = open("/tmp/ps5_drive.log", "ab")
            self._proc = subprocess.Popen(
                ["bash", "-c", supervisor],
                cwd=str(PRODUCTION_DIR),
                stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            print(f"[CART] spawned ps5_drive supervisor pid={self._proc.pid}")
        except Exception as e:
            print(f"[WARN] could not spawn ps5_drive supervisor: {e!r}")

    def _read_state(self) -> tuple[dict, bool]:
        try:
            with open(self.state_file) as f:
                data = json.load(f)
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return {}, True
        ts = float(data.get("ts", 0.0))
        stale = (time.time() - ts) > CART_FRESH_S
        return data, stale

    def latest(self) -> dict:
        data, stale = self._read_state()
        return {"connected": (not stale) and bool(data), "stale": stale, "sample": data}

    def _record_loop(self, started_wall: float):
        # Poll faster than ps5_drive's ~60 Hz control rate.
        while not self._rec_stop.is_set():
            data, stale = self._read_state()
            if not stale and data:
                ts = data.get("ts")
                if ts is not None and ts != self._last_ts:
                    self._last_ts = ts
                    rec = {
                        "wall_t": time.time(),
                        "rel_t": time.time() - started_wall,
                        **data,
                    }
                    with self._lock:
                        if self._fp is not None:
                            self._fp.write(json.dumps(rec) + "\n")
                            self._fp.flush()
            self._rec_stop.wait(0.01)

    def start_recording(self, out_path: Path):
        self._last_ts = None
        self._fp = open(out_path, "w", buffering=1)
        self._rec_stop.clear()
        self._rec_thread = threading.Thread(
            target=self._record_loop, args=(time.time(),), daemon=True
        )
        self._rec_thread.start()

    def stop_recording(self):
        if self._rec_thread is not None:
            self._rec_stop.set()
            self._rec_thread.join(timeout=2)
            self._rec_thread = None
        with self._lock:
            if self._fp is not None:
                self._fp.close()
                self._fp = None

    def shutdown(self):
        self.stop_recording()
        if self._proc is not None and self._proc.poll() is None:
            try:
                # The supervisor is a bash loop; killing the group also
                # takes down the inner uv/ps5_drive process.
                os.killpg(os.getpgid(self._proc.pid), 15)
                self._proc.wait(timeout=3)
            except Exception:
                try:
                    os.killpg(os.getpgid(self._proc.pid), 9)
                except Exception:
                    pass
        # Belt-and-braces: kill any straggler ps5_drive we own.
        subprocess.run(["pkill", "-f", "scripts/ps5_drive.py"],
                       capture_output=True)


class SegmentationMapRecorder:
    """Records raw semantic segmentation label maps published by
    scripts/segmentation_infer.py.

    The sidecar writes one latest-map JSON file atomically. While the camera
    recorder is active, this class snapshots each new infer_count into
    segmentation_maps.jsonl. Rows keep the map's RLE JSON payload and include
    the current autosteer state when it is fresh so offline analysis can align
    labels, planned path, steering, and pedals.
    """

    def __init__(self, map_file: str = SEGMENTATION_MAP_FILE,
                 state_file: str = AUTOWARE_STATE_FILE):
        self.map_file = map_file
        self.state_file = state_file
        self._lock = threading.Lock()
        self._fp = None  # type: ignore[var-annotated]
        self._last_key: tuple[int | None, float | None] | None = None
        self._rec_stop = threading.Event()
        self._rec_thread: threading.Thread | None = None

    @staticmethod
    def _read_json(path: str) -> tuple[dict, bool]:
        try:
            with open(path) as f:
                data = json.load(f)
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return {}, True
        if not isinstance(data, dict):
            return {}, True
        try:
            ts = float(data.get("ts", 0.0))
        except (TypeError, ValueError):
            return data, True
        return data, (time.time() - ts) > SEGMENTATION_MAP_FRESH_S

    def latest(self) -> dict:
        data, stale = self._read_json(self.map_file)
        return {
            "connected": bool(data) and not stale,
            "stale": stale,
            "map_file": self.map_file,
            "infer_count": data.get("infer_count"),
            "shape": data.get("shape"),
            "encoding": data.get("encoding"),
        }

    def _record_loop(self, started_wall: float):
        while not self._rec_stop.is_set():
            data, stale = self._read_json(self.map_file)
            if data and not stale:
                key = (data.get("infer_count"), data.get("ts"))
                if key != self._last_key:
                    self._last_key = key
                    rec = {
                        "wall_t": time.time(),
                        "rel_t": time.time() - started_wall,
                        **data,
                    }
                    autosteer, autosteer_stale = self._read_json(self.state_file)
                    if autosteer and not autosteer_stale:
                        rec["autosteer_state"] = autosteer
                    with self._lock:
                        if self._fp is not None:
                            self._fp.write(json.dumps(rec) + "\n")
                            self._fp.flush()
            self._rec_stop.wait(0.01)

    def start_recording(self, out_path: Path):
        self._last_key = None
        self._fp = open(out_path, "w", buffering=1)
        header = {
            "_schema": "caddy.segmentation_maps_jsonl.v1",
            "_row_schema": "caddy.segmentation_map.v1",
            "_source": "scripts/segmentation_infer.py latest-map JSON",
            "_source_file": self.map_file,
            "_autosteer_state_file": self.state_file,
            "_encoding": "rle_flat_c_order",
            "_note": "Data rows contain the raw semantic label map as RLE JSON. "
                     "Decode with np.repeat(row['rle']['values'], "
                     "row['rle']['counts']).astype(np.uint8).reshape(row['shape']).",
            "started_wall_t": time.time(),
        }
        self._fp.write(json.dumps(header) + "\n")
        self._fp.flush()
        self._rec_stop.clear()
        self._rec_thread = threading.Thread(
            target=self._record_loop, args=(time.time(),), daemon=True
        )
        self._rec_thread.start()

    def stop_recording(self):
        if self._rec_thread is not None:
            self._rec_stop.set()
            self._rec_thread.join(timeout=2)
            self._rec_thread = None
        with self._lock:
            if self._fp is not None:
                self._fp.close()
                self._fp = None

    def shutdown(self):
        self.stop_recording()


class GpsRecorder:
    """iPhone GPS stream over TCP (usbmuxd tunneled). Mirrors EgoRecorder's
    shape — auto-spawns the iproxy supervisor (ego_link.sh with the GPS
    port arg), maintains the latest fix in a background reader, and while
    recording appends every fix to gps.jsonl in the run folder.

    Wire format (one JSON object per line) matches GPSFix in
    jetson/gps_sensor.py from Georg-Stanford-GC-iOS-Sensor:
      t_unix, lat_deg, lon_deg, alt_m, h_acc_m, v_acc_m,
      speed_mps, speed_acc_mps, course_deg, course_acc_deg
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 5006,
                 spawn_helper: bool = True):
        self.host = host
        self.port = port
        self._link_proc: subprocess.Popen | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._latest: dict | None = None
        self._latest_wall: float = 0.0  # jetson wall-clock when fix arrived
        self._connected = False
        self._rec_lock = threading.Lock()
        self._fp = None  # type: ignore[var-annotated]
        self._rec_started_wall: float = 0.0
        self._last_t_unix: float | None = None
        if spawn_helper:
            self._spawn_helper()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    @staticmethod
    def _is_running(pattern: str) -> bool:
        try:
            r = subprocess.run(["pgrep", "-f", pattern], capture_output=True)
            return r.returncode == 0
        except FileNotFoundError:
            return False

    def _spawn_helper(self):
        # Port-specific pgrep so we don't collide with the ego_link.sh that
        # supervises the 5005 ego-motion port.
        pattern = f"ego_link.sh {self.port}"
        if not os.path.exists(EGO_LINK_SCRIPT):
            print(f"[GPS] {EGO_LINK_SCRIPT} missing; not spawning iproxy")
            return
        if self._is_running(pattern):
            print(f"[GPS] ego_link.sh {self.port} already running; not spawning")
            return
        try:
            self._link_proc = subprocess.Popen(
                ["bash", EGO_LINK_SCRIPT, str(self.port)],
                stdout=open(f"/tmp/gps_link_{self.port}.log", "ab"),
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            print(f"[GPS] spawned ego_link.sh {self.port} pid={self._link_proc.pid}")
        except Exception as e:
            print(f"[GPS] could not spawn ego_link.sh {self.port}: {e!r}")

    def _loop(self):
        buf = b""
        while not self._stop.is_set():
            try:
                with socket.create_connection((self.host, self.port), timeout=2.0) as s:
                    s.settimeout(1.0)
                    self._connected = True
                    buf = b""
                    while not self._stop.is_set():
                        try:
                            chunk = s.recv(4096)
                        except socket.timeout:
                            continue
                        if not chunk:
                            break
                        buf += chunk
                        while b"\n" in buf:
                            line, buf = buf.split(b"\n", 1)
                            if not line.strip():
                                continue
                            try:
                                fix = json.loads(line)
                            except json.JSONDecodeError:
                                continue
                            self._ingest(fix)
            except (ConnectionRefusedError, OSError):
                pass
            finally:
                self._connected = False
            if not self._stop.is_set():
                time.sleep(0.5)

    def _ingest(self, fix: dict):
        now = time.time()
        with self._lock:
            self._latest = fix
            self._latest_wall = now
        # Dedupe by t_unix — same fix can sometimes resend on reconnect.
        t_unix = fix.get("t_unix")
        if t_unix is None or t_unix == self._last_t_unix:
            return
        self._last_t_unix = t_unix
        with self._rec_lock:
            if self._fp is not None:
                rec = {
                    "wall_t": now,
                    "rel_t": now - self._rec_started_wall,
                    **fix,
                }
                self._fp.write(json.dumps(rec) + "\n")
                self._fp.flush()

    def latest(self) -> dict:
        with self._lock:
            fix = dict(self._latest) if self._latest is not None else None
            wall = self._latest_wall
        stale = (time.time() - wall) > GPS_FRESH_S if fix is not None else True
        return {
            "connected": self._connected and not stale,
            "tcp_connected": self._connected,
            "stale": stale,
            "host": f"{self.host}:{self.port}",
            "fix": fix,
        }

    def start_recording(self, out_path: Path):
        with self._rec_lock:
            self._last_t_unix = None
            self._rec_started_wall = time.time()
            self._fp = open(out_path, "w", buffering=1)
            header = {
                "_schema": "caddy.gps.v1",
                "_source": "iPhone CoreLocation via usbmuxd tunnel "
                           "(TCP :5006), publisher = Georg-Stanford-GC-iOS-Sensor "
                           "GPSServer (commit 1eab4c53).",
                "_fields": ["t_unix", "lat_deg", "lon_deg", "alt_m",
                            "h_acc_m", "v_acc_m", "speed_mps", "speed_acc_mps",
                            "course_deg", "course_acc_deg"],
                "_note": "Each row also carries wall_t (jetson recv time) and "
                         "rel_t (offset from recording start). h_acc_m / "
                         "v_acc_m are 1-sigma in metres; negative means "
                         "invalid per Apple convention.",
                "started_wall_t": self._rec_started_wall,
            }
            self._fp.write(json.dumps(header) + "\n")
            self._fp.flush()

    def stop_recording(self):
        with self._rec_lock:
            if self._fp is not None:
                self._fp.close()
                self._fp = None

    def shutdown(self):
        self.stop_recording()
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        if self._link_proc is not None and self._link_proc.poll() is None:
            try:
                self._link_proc.terminate()
                self._link_proc.wait(timeout=2)
            except Exception:
                try:
                    self._link_proc.kill()
                except Exception:
                    pass


class RealSenseRecorder:
    """Intel RealSense D4xx capture — color (BGR) + aligned depth (uint16).

    While recording:
      - color.mp4              H.264-fallback mp4v at fps, BGR frames
      - depth.bin              concatenated raw uint16 frames (H*W*2 bytes each)
      - depth_meta.json        {width, height, fps, depth_scale_m, intrinsics, ...}
      - depth_ts.jsonl         one row per depth frame: {idx, wall_t, rel_t, rs_t_ms}

    Loading depth offline:
        meta = json.load(open('depth_meta.json'))
        H, W = meta['height'], meta['width']
        arr = np.fromfile('depth.bin', dtype=np.uint16).reshape(-1, H, W)
        depth_metres = arr * meta['depth_scale_m']

    Depth is aligned to color so a (row, col) in color.mp4 indexes the
    same (row, col) in the depth array.
    """

    def __init__(self, width: int = 640, height: int = 480, fps: int = 30):
        self.width = width
        self.height = height
        self.fps = fps
        self.flip_code: int | None = None
        self.pipeline: "rs.pipeline | None" = None
        self.align: "rs.align | None" = None
        self.depth_scale_m: float = 0.001  # filled in from device on start
        self.intrinsics: dict = {}
        self.device_name: str = ""
        self.serial: str = ""
        self.frames_ok = 0
        self.frames_fail = 0
        self.last_ok_t = 0.0
        self.lock = threading.Lock()
        self.color_frame: np.ndarray | None = None
        self.depth_frame: np.ndarray | None = None
        self.depth_ts_ms: float = 0.0
        self.running = False
        self.thread: threading.Thread | None = None
        # Recording state
        self.writer: cv2.VideoWriter | None = None
        self.depth_fp = None  # type: ignore[var-annotated]
        self.depth_ts_fp = None  # type: ignore[var-annotated]
        self.writer_stop = threading.Event()
        self.writer_thread: threading.Thread | None = None
        self.rec_started_wall: float = 0.0
        self.rec_frame_idx: int = 0

    @staticmethod
    def detect() -> bool:
        if not _HAVE_RS:
            return False
        try:
            return len(rs.context().query_devices()) > 0
        except Exception:
            return False

    def open(self) -> bool:
        if not _HAVE_RS:
            return False
        try:
            self.pipeline = rs.pipeline()
            cfg = rs.config()
            cfg.enable_stream(rs.stream.color, self.width, self.height,
                              rs.format.bgr8, self.fps)
            cfg.enable_stream(rs.stream.depth, self.width, self.height,
                              rs.format.z16, self.fps)
            profile = self.pipeline.start(cfg)
            dev = profile.get_device()
            self.device_name = dev.get_info(rs.camera_info.name)
            try:
                self.serial = dev.get_info(rs.camera_info.serial_number)
            except Exception:
                self.serial = ""
            depth_sensor = dev.first_depth_sensor()
            self.depth_scale_m = float(depth_sensor.get_depth_scale())
            # Align depth -> color so pixel (r,c) corresponds across streams.
            self.align = rs.align(rs.stream.color)
            color_prof = profile.get_stream(rs.stream.color).as_video_stream_profile()
            intr = color_prof.get_intrinsics()
            self.intrinsics = {
                "width": intr.width, "height": intr.height,
                "fx": intr.fx, "fy": intr.fy,
                "ppx": intr.ppx, "ppy": intr.ppy,
                "model": str(intr.model),
                "coeffs": list(intr.coeffs),
            }
            self.running = True
            self.thread = threading.Thread(target=self._loop, daemon=True)
            self.thread.start()
            print(f"[RS] {self.device_name} sn={self.serial} "
                  f"{self.width}x{self.height}@{self.fps} "
                  f"depth_scale={self.depth_scale_m}")
            return True
        except Exception as e:
            print(f"[RS] could not start RealSense: {e!r}")
            self.pipeline = None
            return False

    def _loop(self):
        assert self.pipeline is not None
        while self.running:
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=2000)
            except Exception:
                self.frames_fail += 1
                continue
            if self.align is not None:
                frames = self.align.process(frames)
            color = frames.get_color_frame()
            depth = frames.get_depth_frame()
            if not color or not depth:
                self.frames_fail += 1
                continue
            color_np = np.asanyarray(color.get_data())
            depth_np = np.asanyarray(depth.get_data())
            ts_ms = float(depth.get_timestamp())
            self.frames_ok += 1
            self.last_ok_t = time.monotonic()
            with self.lock:
                self.color_frame = color_np
                self.depth_frame = depth_np
                self.depth_ts_ms = ts_ms
                # Write while we still hold the lock so the depth frame
                # we record is the exact one paired with the color frame.
                if self.writer is not None and self.depth_fp is not None:
                    self.writer.write(color_np)
                    self.depth_fp.write(depth_np.tobytes())
                    if self.depth_ts_fp is not None:
                        rec = {
                            "idx": self.rec_frame_idx,
                            "wall_t": time.time(),
                            "rel_t": time.time() - self.rec_started_wall,
                            "rs_t_ms": ts_ms,
                        }
                        self.depth_ts_fp.write(json.dumps(rec) + "\n")
                    self.rec_frame_idx += 1

    def get_color(self) -> np.ndarray | None:
        with self.lock:
            return None if self.color_frame is None else self.color_frame.copy()

    def get_depth_preview(self) -> np.ndarray | None:
        with self.lock:
            d = None if self.depth_frame is None else self.depth_frame.copy()
        if d is None:
            return None
        # Colorize: clip to ~6 m for a stable preview.
        clip_mm = int(6.0 / self.depth_scale_m)
        d = np.clip(d, 0, clip_mm)
        d8 = (d.astype(np.float32) * (255.0 / max(1, clip_mm))).astype(np.uint8)
        return cv2.applyColorMap(d8, cv2.COLORMAP_JET)

    def latest(self) -> dict:
        ok = self.running and (time.monotonic() - self.last_ok_t) < 2.0
        return {
            "connected": bool(ok),
            "device": self.device_name,
            "serial": self.serial,
            "frames_ok": self.frames_ok,
            "frames_fail": self.frames_fail,
            "depth_scale_m": self.depth_scale_m,
        }

    def start_recording(self, folder: Path):
        if not self.running:
            return
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(
            str(folder / "realsense_color.mp4"), fourcc, self.fps,
            (self.width, self.height),
        )
        self.depth_fp = open(folder / "realsense_depth.bin", "wb", buffering=0)
        self.depth_ts_fp = open(folder / "realsense_depth_ts.jsonl", "w", buffering=1)
        with open(folder / "realsense_depth_meta.json", "w") as f:
            json.dump({
                "width": self.width,
                "height": self.height,
                "fps": self.fps,
                "dtype": "uint16",
                "depth_scale_m": self.depth_scale_m,
                "aligned_to": "color",
                "color_file": "realsense_color.mp4",
                "depth_file": "realsense_depth.bin",
                "timestamps_file": "realsense_depth_ts.jsonl",
                "intrinsics": self.intrinsics,
                "device": self.device_name,
                "serial": self.serial,
                "note": "Load: np.fromfile(depth_file, np.uint16)"
                        ".reshape(-1, height, width); metres = arr * depth_scale_m.",
            }, f, indent=2)
        self.rec_started_wall = time.time()
        self.rec_frame_idx = 0

    def stop_recording(self):
        # Take the lock so the capture loop can't write into a half-closed
        # writer between our None-assignments.
        with self.lock:
            writer, dfp, tfp = self.writer, self.depth_fp, self.depth_ts_fp
            self.writer = None
            self.depth_fp = None
            self.depth_ts_fp = None
        if writer is not None:
            writer.release()
        if dfp is not None:
            dfp.close()
        if tfp is not None:
            tfp.close()

    def release(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2)
        self.stop_recording()
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except Exception:
                pass
            self.pipeline = None


app = Flask(__name__)
cameras: dict[str, Camera] = {}
model_recorders: dict[str, ModelFrameRecorder] = {
    "front": ModelFrameRecorder("front", MODEL_FRAMES_DIR / "front.jpg")
}
ego: EgoRecorder | None = None
cart: CartRecorder | None = None
gps: GpsRecorder | None = None
segmentation_maps: SegmentationMapRecorder | None = None
realsense: RealSenseRecorder | None = None
show_realsense_color = True
state = {"recording": False, "folder": None, "started_at": None}
state_lock = threading.Lock()


PAGE = """
<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<title>Caddy Camera Recorder</title>
<style>
  body { background:#111; color:#eee; font-family: system-ui, sans-serif; margin:0; padding:16px; }
  h1 { margin: 0 0 12px 0; font-size: 20px; }
  .layout { display:grid; grid-template-columns: 1fr 260px; gap:16px; align-items:start; }
  .grid { background:#000; border:1px solid #333; }
  .grid img { width:100%; display:block; }
  .cell { background:#000; border:1px solid #333; position:relative; }
  .cell img { width:100%; display:block; }
  .label { position:absolute; top:6px; left:8px; background:rgba(0,0,0,0.6); padding:2px 8px; font-size:13px; border-radius:3px; }
  .controls { margin-top:14px; display:flex; align-items:center; gap:12px; }
  button { font-size:18px; padding:10px 24px; border:0; border-radius:4px; cursor:pointer; }
  .rec { background:#c0392b; color:#fff; }
  .stop { background:#444; color:#fff; }
  .status { font-family: monospace; }
  .dot { display:inline-block; width:10px; height:10px; border-radius:50%; background:#666; margin-right:6px; vertical-align:middle; }
  .dot.on { background:#e74c3c; animation: pulse 1s infinite; }
  @keyframes pulse { 50% { opacity:0.3; } }

  /* Ego-motion panel — mirror of PRODUCTION web UI, restyled for dark bg. */
  .ego-panel {
    background: #1a1c20; border: 1px solid #2a2d33; border-radius: 12px;
    padding: 10px 12px 8px 12px; font-size: 10px; color: #aab; letter-spacing: 0.05em;
  }
  .ego-title {
    display:flex; align-items:center; justify-content:space-between;
    font-weight:600; font-size:9.5px; text-transform:uppercase; color:#7e8896; margin-bottom:6px;
  }
  .ego-conn-dot { width:6px; height:6px; border-radius:50%; background:#d84a4a; transition:background-color 180ms ease, box-shadow 180ms ease; }
  .ego-conn-dot.connected { background:#7ed488; box-shadow:0 0 0 1.5px rgba(126,212,136,0.25); }
  .ego-canvas { display:block; width:100%; height:180px; background:#0e1013; border-radius:10px; }
  .ego-readout { display:grid; grid-template-columns:repeat(2, 1fr); gap:2px 10px; margin-top:6px; font-variant-numeric:tabular-nums; font-size:11px; }
  .ego-readout > div { display:flex; align-items:baseline; gap:4px; }
  .ego-k { color:#7e8896; font-weight:500; min-width:32px; }
  .ego-v { color:#e6e9ef; font-weight:500; margin-left:auto; }
  .ego-u { color:#7e8896; font-size:9.5px; }

  /* Steering panel — sits under the ego cube, mirrors PRODUCTION HUD wheel. */
  .steer-panel {
    margin-top: 12px;
    background: #1a1c20; border: 1px solid #2a2d33; border-radius: 12px;
    padding: 10px 12px 10px 12px; color: #aab;
    display: flex; flex-direction: column; align-items: center;
  }
  .steer-title {
    align-self: stretch;
    display:flex; align-items:center; justify-content:space-between;
    font-weight:600; font-size:9.5px; text-transform:uppercase; color:#7e8896;
    letter-spacing: 0.05em; margin-bottom: 8px;
  }
  .steer-conn-dot { width:6px; height:6px; border-radius:50%; background:#d84a4a;
    transition:background-color 180ms ease, box-shadow 180ms ease; }
  .steer-conn-dot.connected { background:#7ed488;
    box-shadow:0 0 0 1.5px rgba(126,212,136,0.25); }
  .steer-wheel-svg { width: 110px; height: 110px; color: #cfd6e2; display:block; }
  .steer-wheel-svg #steer-wheel-rotate { transform-origin: 50px 50px;
    transition: transform 80ms linear; }
  .steer-angle {
    font-family: monospace; font-variant-numeric: tabular-nums;
    font-size: 18px; color: #e6e9ef; margin-top: 4px;
  }
  .pedal-row {
    display: grid; grid-template-columns: 36px 1fr 36px;
    gap: 6px; align-items: center; width: 100%; margin-top: 6px;
    font-size: 10px; font-variant-numeric: tabular-nums;
  }
  .pedal-row .pk { color:#7e8896; }
  .pedal-row .pv { color:#e6e9ef; text-align:right; }
  .pedal-bar { height: 6px; background:#0e1013; border-radius: 3px; overflow:hidden; }
  .pedal-bar > div { height: 100%; width: 0%; transition: width 80ms linear; }
  .pedal-bar.gas > div { background:#7ed488; }
  .pedal-bar.brake > div { background:#e8605c; }

  /* GPS panel — lat/lon/alt + accuracy + local-meters trail. */
  .gps-panel {
    margin-top: 12px;
    background: #1a1c20; border: 1px solid #2a2d33; border-radius: 12px;
    padding: 10px 12px 10px 12px; color: #aab;
  }
  .gps-title {
    display:flex; align-items:center; justify-content:space-between;
    font-weight:600; font-size:9.5px; text-transform:uppercase; color:#7e8896;
    letter-spacing: 0.05em; margin-bottom: 8px;
  }
  .gps-conn-dot { width:6px; height:6px; border-radius:50%; background:#d84a4a;
    transition:background-color 180ms ease, box-shadow 180ms ease; }
  .gps-conn-dot.connected { background:#7ed488;
    box-shadow:0 0 0 1.5px rgba(126,212,136,0.25); }
  .gps-canvas { display:block; width:100%; height:140px; background:#0e1013; border-radius:10px; }
  .gps-readout { display:grid; grid-template-columns:repeat(2, 1fr); gap:2px 10px; margin-top:6px; font-variant-numeric:tabular-nums; font-size:11px; }
  .gps-readout > div { display:flex; align-items:baseline; gap:4px; }
  .gps-k { color:#7e8896; font-weight:500; min-width:36px; }
  .gps-v { color:#e6e9ef; font-weight:500; margin-left:auto; font-family: monospace; }
  .gps-u { color:#7e8896; font-size:9.5px; }
  .gps-acc-pill {
    display:inline-block; padding:1px 6px; border-radius:8px;
    font-size:10px; font-weight:600; font-variant-numeric:tabular-nums;
    background:#3a3d44; color:#e6e9ef;
  }
  .gps-acc-pill.good   { background:rgba(126,212,136,0.20); color:#a8e6b3; }
  .gps-acc-pill.medium { background:rgba(232,196,96,0.20); color:#f0d68a; }
  .gps-acc-pill.poor   { background:rgba(232,96,96,0.20);  color:#f0a6a6; }
</style>
</head>
<body>
  <h1>Caddy Camera Recorder</h1>
  <div class="layout">
    <div>
      <div class="grid"><img src="/grid_feed" /></div>
      <div class="controls">
        <button id="btn" class="rec" onclick="toggle()">Record</button>
        <span class="status"><span id="dot" class="dot"></span><span id="status">idle</span></span>
      </div>
    </div>
    <div class="ego-panel" id="ego-panel">
      <div class="ego-title">
        <span>EGO MOTION</span>
        <span class="ego-conn-dot" id="ego-conn-dot"></span>
      </div>
      <canvas class="ego-canvas" id="ego-canvas" width="220" height="180"></canvas>
      <div class="ego-readout">
        <div><span class="ego-k">x</span><span class="ego-v" id="ego-x">—</span><span class="ego-u">m</span></div>
        <div><span class="ego-k">y</span><span class="ego-v" id="ego-y">—</span><span class="ego-u">m</span></div>
        <div><span class="ego-k">z</span><span class="ego-v" id="ego-z">—</span><span class="ego-u">m</span></div>
        <div><span class="ego-k">speed</span><span class="ego-v" id="ego-speed">—</span><span class="ego-u">m/s</span></div>
        <div><span class="ego-k">yaw</span><span class="ego-v" id="ego-yaw">—</span><span class="ego-u">°</span></div>
        <div><span class="ego-k">curv</span><span class="ego-v" id="ego-curv">—</span><span class="ego-u">/m</span></div>
      </div>
      <div class="steer-panel">
        <div class="steer-title">
          <span>STEERING</span>
          <span class="steer-conn-dot" id="steer-conn-dot"></span>
        </div>
        <svg class="steer-wheel-svg" viewBox="0 0 100 100" aria-hidden="true">
          <g id="steer-wheel-rotate">
            <circle cx="50" cy="50" r="40" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round"/>
            <circle cx="50" cy="50" r="10.5" fill="none" stroke="currentColor" stroke-width="3"/>
            <line x1="11" y1="50" x2="39" y2="50" stroke="currentColor" stroke-width="3" stroke-linecap="round"/>
            <line x1="61" y1="50" x2="89" y2="50" stroke="currentColor" stroke-width="3" stroke-linecap="round"/>
            <line x1="50" y1="61" x2="50" y2="89" stroke="currentColor" stroke-width="3" stroke-linecap="round"/>
          </g>
        </svg>
        <div class="steer-angle" id="steer-angle">—°</div>
        <div class="pedal-row">
          <span class="pk">gas</span>
          <div class="pedal-bar gas"><div id="pedal-gas"></div></div>
          <span class="pv" id="pedal-gas-v">—</span>
        </div>
        <div class="pedal-row">
          <span class="pk">brake</span>
          <div class="pedal-bar brake"><div id="pedal-brake"></div></div>
          <span class="pv" id="pedal-brake-v">—</span>
        </div>
      </div>
      <div class="gps-panel">
        <div class="gps-title">
          <span>GPS</span>
          <span><span class="gps-acc-pill" id="gps-acc-pill">— m</span>
            <span class="gps-conn-dot" id="gps-conn-dot" style="margin-left:6px"></span></span>
        </div>
        <canvas class="gps-canvas" id="gps-canvas" width="220" height="140"></canvas>
        <div class="gps-readout">
          <div><span class="gps-k">lat</span><span class="gps-v" id="gps-lat">—</span></div>
          <div><span class="gps-k">lon</span><span class="gps-v" id="gps-lon">—</span></div>
          <div><span class="gps-k">alt</span><span class="gps-v" id="gps-alt">—</span><span class="gps-u">m</span></div>
          <div><span class="gps-k">±v</span><span class="gps-v" id="gps-vacc">—</span><span class="gps-u">m</span></div>
          <div><span class="gps-k">crs</span><span class="gps-v" id="gps-crs">—</span><span class="gps-u">°</span></div>
        </div>
      </div>
    </div>
  </div>
<script>
async function refresh() {
  const r = await fetch('/status'); const j = await r.json();
  const btn = document.getElementById('btn');
  const dot = document.getElementById('dot');
  const status = document.getElementById('status');
  if (j.recording) {
    btn.textContent = 'Stop'; btn.className = 'stop';
    dot.classList.add('on');
    status.textContent = 'REC  ' + j.folder + '   elapsed ' + j.elapsed + 's';
  } else {
    btn.textContent = 'Record'; btn.className = 'rec';
    dot.classList.remove('on');
    status.textContent = j.last_folder ? ('saved: ' + j.last_folder) : 'idle';
  }
}
async function toggle() {
  const r = await fetch('/status'); const j = await r.json();
  await fetch(j.recording ? '/stop' : '/start', { method:'POST' });
  refresh();
}
setInterval(refresh, 500); refresh();

// ===== Ego-motion 3D mini-cube (ported from PRODUCTION web UI) =====
const egoCanvas = document.getElementById('ego-canvas');
const egoCtx = egoCanvas.getContext('2d');
const egoConnDot = document.getElementById('ego-conn-dot');
const egoX = document.getElementById('ego-x');
const egoY = document.getElementById('ego-y');
const egoZ = document.getElementById('ego-z');
const egoSpeed = document.getElementById('ego-speed');
const egoYaw = document.getElementById('ego-yaw');
const egoCurv = document.getElementById('ego-curv');

(function fitCanvas() {
  const dpr = window.devicePixelRatio || 1;
  const cssW = egoCanvas.clientWidth || 220;
  const cssH = egoCanvas.clientHeight || 180;
  egoCanvas.width = cssW * dpr; egoCanvas.height = cssH * dpr;
  egoCtx.scale(dpr, dpr);
})();
const EGO_W = egoCanvas.clientWidth || 220;
const EGO_H = egoCanvas.clientHeight || 180;
const egoTrail = [];
const EGO_TRAIL_SECONDS = 6.0;
const EGO_TRAIL_MAX = 80;
const EGO_HALF_MIN = 1.0;
let egoLastTs = null;
const EGO_CAM_AZ = 32 * Math.PI / 180;
const EGO_CAM_EL = 18 * Math.PI / 180;
const EGO_COS_AZ = Math.cos(EGO_CAM_AZ), EGO_SIN_AZ = Math.sin(EGO_CAM_AZ);
const EGO_COS_EL = Math.cos(EGO_CAM_EL), EGO_SIN_EL = Math.sin(EGO_CAM_EL);

function isoProject(x, y, z, half) {
  const nx = x / half, ny = y / half, nz = -z / half;
  const x1 = nx * EGO_COS_AZ + nz * EGO_SIN_AZ;
  const y1 = ny;
  const z1 = -nx * EGO_SIN_AZ + nz * EGO_COS_AZ;
  const x2 = x1;
  const y2 = y1 * EGO_COS_EL - z1 * EGO_SIN_EL;
  const cx = EGO_W / 2, cy = EGO_H / 2 + 4;
  const scale = Math.min(EGO_W, EGO_H) * 0.34;
  return [cx + x2 * scale, cy - y2 * scale];
}
function drawEgoCube(half) {
  const C = [
    [-1,-1,-1],[+1,-1,-1],[+1,+1,-1],[-1,+1,-1],
    [-1,-1,+1],[+1,-1,+1],[+1,+1,+1],[-1,+1,+1],
  ].map(([a,b,c]) => isoProject(a*half,b*half,c*half,half));
  const E = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];
  egoCtx.lineWidth = 1;
  egoCtx.strokeStyle = 'rgba(150, 165, 195, 0.55)';
  egoCtx.beginPath();
  for (const [a,b] of E) { egoCtx.moveTo(C[a][0],C[a][1]); egoCtx.lineTo(C[b][0],C[b][1]); }
  egoCtx.stroke();
  egoCtx.strokeStyle = 'rgba(150, 165, 195, 0.18)';
  egoCtx.beginPath();
  const N = 4;
  for (let i = 1; i < N; i++) {
    const t = -half + (2*half)*(i/N);
    let p1 = isoProject(t,-half,-half,half), p2 = isoProject(t,-half,+half,half);
    egoCtx.moveTo(p1[0],p1[1]); egoCtx.lineTo(p2[0],p2[1]);
    p1 = isoProject(-half,-half,t,half); p2 = isoProject(+half,-half,t,half);
    egoCtx.moveTo(p1[0],p1[1]); egoCtx.lineTo(p2[0],p2[1]);
  }
  egoCtx.stroke();
  const o = isoProject(0,0,0,half);
  egoCtx.fillStyle = 'rgba(150, 165, 195, 0.45)';
  egoCtx.beginPath(); egoCtx.arc(o[0],o[1],1.6,0,Math.PI*2); egoCtx.fill();
}
function renderEgoScene() {
  egoCtx.clearRect(0, 0, EGO_W, EGO_H);
  const last = egoTrail.length > 0 ? egoTrail[egoTrail.length-1] : null;
  const cx0 = last ? last.x : 0, cy0 = last ? last.y : 0, cz0 = last ? last.z : 0;
  let half = EGO_HALF_MIN;
  for (const p of egoTrail) {
    half = Math.max(half, Math.abs(p.x-cx0), Math.abs(p.y-cy0), Math.abs(p.z-cz0));
  }
  half *= 1.15;
  drawEgoCube(half);
  if (egoTrail.length >= 2) {
    egoCtx.lineWidth = 1.6;
    egoCtx.strokeStyle = 'rgba(232, 96, 96, 0.9)';
    egoCtx.beginPath();
    const p0 = isoProject(egoTrail[0].x-cx0, egoTrail[0].y-cy0, egoTrail[0].z-cz0, half);
    egoCtx.moveTo(p0[0], p0[1]);
    for (let i = 1; i < egoTrail.length; i++) {
      const p = isoProject(egoTrail[i].x-cx0, egoTrail[i].y-cy0, egoTrail[i].z-cz0, half);
      egoCtx.lineTo(p[0], p[1]);
    }
    egoCtx.stroke();
  }
  const p = isoProject(0, 0, 0, half);
  egoCtx.fillStyle = '#f0f3f8';
  egoCtx.beginPath(); egoCtx.arc(p[0], p[1], 3.4, 0, Math.PI*2); egoCtx.fill();
}
async function pollEgo() {
  try {
    const r = await fetch('/ego'); const s = await r.json();
    const connected = !!s.connected;
    egoConnDot.classList.toggle('connected', connected);
    const sample = s.sample;
    if (sample && typeof sample.t_s === 'number' && sample.t_s !== egoLastTs) {
      egoLastTs = sample.t_s;
      egoTrail.push({ x: +sample.x_m||0, y: +sample.y_m||0, z: +sample.z_m||0, t_s: sample.t_s });
      const cutoff = sample.t_s - EGO_TRAIL_SECONDS;
      while (egoTrail.length > 0 && egoTrail[0].t_s < cutoff) egoTrail.shift();
      while (egoTrail.length > EGO_TRAIL_MAX) egoTrail.shift();
    }
    if (sample) {
      egoX.textContent = (+sample.x_m||0).toFixed(2);
      egoY.textContent = (+sample.y_m||0).toFixed(2);
      egoZ.textContent = (+sample.z_m||0).toFixed(2);
      egoSpeed.textContent = (+sample.speed_mps||0).toFixed(2);
      egoYaw.textContent = ((+sample.yaw_rad||0) * 180 / Math.PI).toFixed(1);
      egoCurv.textContent = (+sample.curvature_inv_m||0).toFixed(3);
    } else if (!connected) {
      egoX.textContent = egoY.textContent = egoZ.textContent = '—';
      egoSpeed.textContent = egoYaw.textContent = egoCurv.textContent = '—';
    }
    renderEgoScene();
  } catch (e) { /* keep polling */ }
}
setInterval(pollEgo, 100); pollEgo();

// ===== Steering wheel + pedal panel =====
const steerWheel = document.getElementById('steer-wheel-rotate');
const steerAngleEl = document.getElementById('steer-angle');
const steerConnDot = document.getElementById('steer-conn-dot');
const pedalGasBar = document.getElementById('pedal-gas');
const pedalBrakeBar = document.getElementById('pedal-brake');
const pedalGasV = document.getElementById('pedal-gas-v');
const pedalBrakeV = document.getElementById('pedal-brake-v');
async function pollCart() {
  try {
    const r = await fetch('/cart'); const s = await r.json();
    const connected = !!s.connected;
    steerConnDot.classList.toggle('connected', connected);
    const sample = s.sample || {};
    if (connected && Object.keys(sample).length) {
      const deg = Number(sample.steer_deg) || 0;
      steerWheel.style.transform = `rotate(${deg}deg)`;
      steerAngleEl.textContent = `${deg >= 0 ? '+' : ''}${deg.toFixed(0)}°`;
      const gFrac = Math.max(0, Math.min(1, Number(sample.gas_frac) || 0));
      const bFrac = Math.max(0, Math.min(1, Number(sample.brake_frac) || 0));
      pedalGasBar.style.width = (gFrac * 100).toFixed(0) + '%';
      pedalBrakeBar.style.width = (bFrac * 100).toFixed(0) + '%';
      pedalGasV.textContent = (Number(sample.gas) || 0).toFixed(0);
      pedalBrakeV.textContent = (Number(sample.brake) || 0).toFixed(0);
    } else {
      steerWheel.style.transform = 'rotate(0deg)';
      steerAngleEl.textContent = '—°';
      pedalGasBar.style.width = '0%'; pedalBrakeBar.style.width = '0%';
      pedalGasV.textContent = '—'; pedalBrakeV.textContent = '—';
    }
  } catch (e) { /* keep polling */ }
}
setInterval(pollCart, 100); pollCart();

// ===== GPS panel (lat/lon/alt + accuracy pill + local-meters trail) =====
const gpsCanvas = document.getElementById('gps-canvas');
const gpsCtx = gpsCanvas.getContext('2d');
const gpsConnDot = document.getElementById('gps-conn-dot');
const gpsAccPill = document.getElementById('gps-acc-pill');
const gpsLat = document.getElementById('gps-lat');
const gpsLon = document.getElementById('gps-lon');
const gpsAlt = document.getElementById('gps-alt');
const gpsVacc = document.getElementById('gps-vacc');
const gpsCrs = document.getElementById('gps-crs');

(function fitGpsCanvas() {
  const dpr = window.devicePixelRatio || 1;
  const cssW = gpsCanvas.clientWidth || 220;
  const cssH = gpsCanvas.clientHeight || 140;
  gpsCanvas.width = cssW * dpr; gpsCanvas.height = cssH * dpr;
  gpsCtx.scale(dpr, dpr);
})();
const GPS_W = gpsCanvas.clientWidth || 220;
const GPS_H = gpsCanvas.clientHeight || 140;
const gpsTrail = [];                 // {lat, lon, t_unix, h_acc_m}
const GPS_TRAIL_MAX = 240;
const GPS_TRAIL_SECONDS = 600;       // keep last 10 min of fixes
let gpsLastTUnix = null;

// Local ENU projection centred on the latest fix. Small-angle equirect
// is fine for any trail short enough to fit in this card.
function latLonToLocalMetres(lat, lon, cLat, cLon) {
  const R = 6378137.0;
  const dLat = (lat - cLat) * Math.PI / 180;
  const dLon = (lon - cLon) * Math.PI / 180;
  const meanLat = (lat + cLat) * 0.5 * Math.PI / 180;
  return [dLon * R * Math.cos(meanLat), dLat * R];  // east, north
}
function renderGpsTrail(latestAccM) {
  gpsCtx.clearRect(0, 0, GPS_W, GPS_H);
  // Faint axes.
  gpsCtx.strokeStyle = 'rgba(150, 165, 195, 0.18)';
  gpsCtx.lineWidth = 1;
  gpsCtx.beginPath();
  gpsCtx.moveTo(GPS_W/2, 0); gpsCtx.lineTo(GPS_W/2, GPS_H);
  gpsCtx.moveTo(0, GPS_H/2); gpsCtx.lineTo(GPS_W, GPS_H/2);
  gpsCtx.stroke();
  if (gpsTrail.length === 0) return;
  const last = gpsTrail[gpsTrail.length - 1];
  const pts = gpsTrail.map(p => latLonToLocalMetres(p.lat, p.lon, last.lat, last.lon));
  let half = 5;  // metres
  for (const [e, n] of pts) half = Math.max(half, Math.abs(e), Math.abs(n));
  half *= 1.15;
  const cx = GPS_W / 2, cy = GPS_H / 2;
  const scale = Math.min(GPS_W, GPS_H) * 0.45 / half;
  // Scale bar.
  const barM = Math.pow(10, Math.floor(Math.log10(half)));
  const barPx = barM * scale;
  gpsCtx.strokeStyle = 'rgba(150, 165, 195, 0.45)';
  gpsCtx.beginPath();
  gpsCtx.moveTo(8, GPS_H - 10); gpsCtx.lineTo(8 + barPx, GPS_H - 10);
  gpsCtx.stroke();
  gpsCtx.fillStyle = 'rgba(150, 165, 195, 0.75)';
  gpsCtx.font = '9px monospace';
  gpsCtx.fillText(`${barM}m`, 8 + barPx + 4, GPS_H - 6);
  // Trail.
  if (pts.length >= 2) {
    gpsCtx.strokeStyle = 'rgba(126, 212, 136, 0.85)';
    gpsCtx.lineWidth = 1.6;
    gpsCtx.beginPath();
    gpsCtx.moveTo(cx + pts[0][0]*scale, cy - pts[0][1]*scale);
    for (let i = 1; i < pts.length; i++) {
      gpsCtx.lineTo(cx + pts[i][0]*scale, cy - pts[i][1]*scale);
    }
    gpsCtx.stroke();
  }
  // Accuracy ring around the latest fix.
  if (latestAccM > 0 && isFinite(latestAccM)) {
    const r = Math.min(Math.min(GPS_W, GPS_H) * 0.45, latestAccM * scale);
    gpsCtx.strokeStyle = 'rgba(126, 212, 136, 0.35)';
    gpsCtx.lineWidth = 1;
    gpsCtx.beginPath(); gpsCtx.arc(cx, cy, r, 0, Math.PI*2); gpsCtx.stroke();
  }
  // Current position marker.
  gpsCtx.fillStyle = '#f0f3f8';
  gpsCtx.beginPath(); gpsCtx.arc(cx, cy, 3.2, 0, Math.PI*2); gpsCtx.fill();
}
function gpsAccClass(m) {
  if (!(m >= 0) || !isFinite(m)) return '';
  if (m < 5)  return 'good';
  if (m < 15) return 'medium';
  return 'poor';
}
async function pollGps() {
  try {
    const r = await fetch('/gps'); const s = await r.json();
    const connected = !!s.connected;
    gpsConnDot.classList.toggle('connected', connected);
    const fix = s.fix;
    if (fix && typeof fix.t_unix === 'number' && fix.t_unix !== gpsLastTUnix) {
      gpsLastTUnix = fix.t_unix;
      gpsTrail.push({ lat: +fix.lat_deg, lon: +fix.lon_deg,
                      t_unix: fix.t_unix, h_acc_m: +fix.h_acc_m });
      const cutoff = fix.t_unix - GPS_TRAIL_SECONDS;
      while (gpsTrail.length > 0 && gpsTrail[0].t_unix < cutoff) gpsTrail.shift();
      while (gpsTrail.length > GPS_TRAIL_MAX) gpsTrail.shift();
    }
    if (fix) {
      gpsLat.textContent = (+fix.lat_deg).toFixed(7);
      gpsLon.textContent = (+fix.lon_deg).toFixed(7);
      gpsAlt.textContent = (+fix.alt_m).toFixed(1);
      gpsVacc.textContent = (+fix.v_acc_m).toFixed(1);
      const crs = +fix.course_deg;
      gpsCrs.textContent = (crs < 0 ? '—' : crs.toFixed(0));
      const hAcc = +fix.h_acc_m;
      gpsAccPill.textContent = (hAcc >= 0 ? `±${hAcc.toFixed(1)} m` : 'no fix');
      gpsAccPill.className = 'gps-acc-pill ' + gpsAccClass(hAcc);
      renderGpsTrail(hAcc);
    } else {
      gpsLat.textContent = gpsLon.textContent = gpsAlt.textContent = '—';
      gpsVacc.textContent = gpsCrs.textContent = '—';
      gpsAccPill.textContent = '— m';
      gpsAccPill.className = 'gps-acc-pill';
      renderGpsTrail(-1);
    }
  } catch (e) { /* keep polling */ }
}
setInterval(pollGps, 250); pollGps();
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(PAGE, names=CAM_NAMES)


@app.route("/video_feed/<name>")
def video_feed(name):
    if name not in cameras:
        return "unknown camera", 404
    cam = cameras[name]

    def gen():
        while True:
            jpeg = cam.get_jpeg()
            if jpeg is None:
                time.sleep(0.05)
                continue
            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            )
            time.sleep(1 / 20)

    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


def _tile_for(name: str, cam: Camera) -> np.ndarray:
    frame = cam.get_frame()
    if frame is None:
        frame = np.zeros((cam.height, cam.width, 3), dtype=np.uint8)
        cv2.putText(
            frame, "NO FRAME", (18, cam.height // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 60, 255), 2, cv2.LINE_AA,
        )
    if frame.shape[1] != cam.width or frame.shape[0] != cam.height:
        frame = cv2.resize(frame, (cam.width, cam.height))
    cv2.rectangle(frame, (0, 0), (cam.width, 28), (0, 0, 0), -1)
    cv2.putText(
        frame, f"{name}  /dev/video{cam.device}", (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
    )
    return frame


def _rs_tile(label: str, frame: np.ndarray | None, w: int, h: int) -> np.ndarray:
    if frame is None:
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.putText(frame, "NO FRAME", (18, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 60, 255), 2, cv2.LINE_AA)
    if frame.shape[1] != w or frame.shape[0] != h:
        frame = cv2.resize(frame, (w, h))
    cv2.rectangle(frame, (0, 0), (w, 28), (0, 0, 0), -1)
    cv2.putText(frame, label, (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def _missing_tile(name: str, w: int, h: int) -> np.ndarray:
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.rectangle(frame, (0, 0), (w, 28), (0, 0, 0), -1)
    cv2.putText(frame, f"{name}  NO DEVICE", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (190, 190, 190), 1, cv2.LINE_AA)
    cv2.putText(frame, "NO DEVICE", (18, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 60, 255), 2, cv2.LINE_AA)
    return frame


def _model_tile(label: str, filename: str, w: int, h: int) -> np.ndarray | None:
    path = MODEL_FRAMES_DIR / filename
    try:
        st = path.stat()
    except FileNotFoundError:
        return None
    frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if frame is None:
        return None
    if time.time() - st.st_mtime > 2.0:
        cv2.putText(
            frame, "STALE", (18, min(frame.shape[0] - 18, 58)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 60, 255), 2, cv2.LINE_AA,
        )
    if frame.shape[1] != w or frame.shape[0] != h:
        frame = cv2.resize(frame, (w, h))
    cv2.rectangle(frame, (0, 0), (w, 28), (0, 0, 0), -1)
    cv2.putText(
        frame, label, (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
    )
    return frame


@app.route("/grid_feed")
def grid_feed():
    def gen():
        names = CAM_NAMES
        while True:
            first_cam = next(iter(cameras.values()), None)
            tile_w = first_cam.width if first_cam is not None else 640
            tile_h = first_cam.height if first_cam is not None else 480
            tiles = []
            for label, filename in MODEL_TILE_FILES:
                tile = _model_tile(label, filename, tile_w, tile_h)
                if tile is not None:
                    tiles.append(tile)
            tiles.extend(
                _tile_for(name, cameras[name])
                for name in names
                if name in cameras
            )
            if realsense is not None:
                if show_realsense_color:
                    tiles.append(_rs_tile("realsense color",
                                          realsense.get_color(), tile_w, tile_h))
                tiles.append(_rs_tile("realsense depth",
                                      realsense.get_depth_preview(), tile_w, tile_h))
            if not tiles:
                time.sleep(0.1)
                continue
            cols = 3 if len(tiles) > 4 else 2
            rows = (len(tiles) + cols - 1) // cols
            blank = np.zeros_like(tiles[0])
            while len(tiles) < rows * cols:
                tiles.append(blank.copy())
            row_imgs = [
                np.hstack(tiles[i * cols:(i + 1) * cols])
                for i in range(rows)
            ]
            grid = np.vstack(row_imgs)
            ok, buf = cv2.imencode(".jpg", grid, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if ok:
                yield (
                    b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                    + buf.tobytes() + b"\r\n"
                )
            time.sleep(1 / 10)

    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/status")
def status():
    with state_lock:
        elapsed = (
            int(time.time() - state["started_at"]) if state["started_at"] else 0
        )
        return jsonify(
            recording=state["recording"],
            folder=state["folder"],
            elapsed=elapsed,
            last_folder=state.get("last_folder"),
        )


@app.route("/debug")
def debug():
    now = time.monotonic()
    return jsonify({
        name: {
            "device": cam.device,
            "frames_ok": cam.frames_ok,
            "frames_fail": cam.frames_fail,
            "age_s": round(now - cam.last_ok_t, 2) if cam.last_ok_t else None,
            "has_frame": cam.frame is not None,
        }
        for name, cam in cameras.items()
    })


@app.route("/ego")
def ego_state():
    return jsonify(ego.latest() if ego is not None else {"connected": False})


@app.route("/cart")
def cart_state():
    return jsonify(cart.latest() if cart is not None else {"connected": False, "sample": {}})


@app.route("/realsense")
def realsense_state():
    return jsonify(realsense.latest() if realsense is not None else {"connected": False})


@app.route("/gps")
def gps_state():
    return jsonify(gps.latest() if gps is not None else {"connected": False, "fix": None})


@app.route("/segmentation_maps")
def segmentation_maps_state():
    return jsonify(
        segmentation_maps.latest()
        if segmentation_maps is not None
        else {"connected": False}
    )


@app.route("/start", methods=["POST"])
def start():
    with state_lock:
        if state["recording"]:
            return jsonify(ok=True)
        ts = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        folder = Path.home() / f"CADDY-6-CAM-EVAL-{ts}"
        folder.mkdir(parents=True, exist_ok=True)
        for name, cam in cameras.items():
            cam.start_recording(folder / f"{name}.mp4")
        for name, rec in model_recorders.items():
            rec.start_recording(folder / f"{name}.mp4")
        if ego is not None:
            ego.start_recording(folder / "ego.jsonl")
        if cart is not None:
            cart.start_recording(folder / "control.jsonl")
        if gps is not None:
            gps.start_recording(folder / "gps.jsonl")
        if segmentation_maps is not None:
            segmentation_maps.start_recording(folder / "segmentation_maps.jsonl")
        if realsense is not None:
            realsense.start_recording(folder)
        # Wall-clock start time captured AFTER the writers/log threads are
        # armed so it bounds the actual recording window from below.
        start_t = time.time()
        with open(folder / "timestamps.json", "w") as f:
            json.dump({
                "start_ms": int(round(start_t * 1000)),
                "start_iso": dt.datetime.fromtimestamp(start_t).isoformat(timespec="milliseconds"),
            }, f, indent=2)
        state["recording"] = True
        state["folder"] = str(folder)
        state["started_at"] = start_t
        print(f"[REC] started -> {folder}")
        return jsonify(ok=True, folder=str(folder))


@app.route("/stop", methods=["POST"])
def stop():
    with state_lock:
        if not state["recording"]:
            return jsonify(ok=True)
        # Capture stop BEFORE we tear down writers so it bounds the actual
        # recording window from above.
        stop_t = time.time()
        for cam in cameras.values():
            cam.stop_recording()
        for rec in model_recorders.values():
            rec.stop_recording()
        if ego is not None:
            ego.stop_recording()
        if cart is not None:
            cart.stop_recording()
        if gps is not None:
            gps.stop_recording()
        if segmentation_maps is not None:
            segmentation_maps.stop_recording()
        if realsense is not None:
            realsense.stop_recording()
        folder = state["folder"]
        start_t = state["started_at"]
        if folder is not None:
            with open(Path(folder) / "timestamps.json", "w") as f:
                json.dump({
                    "start_ms": int(round(start_t * 1000)),
                    "stop_ms":  int(round(stop_t  * 1000)),
                    "duration_ms": int(round((stop_t - start_t) * 1000)),
                    "start_iso": dt.datetime.fromtimestamp(start_t).isoformat(timespec="milliseconds"),
                    "stop_iso":  dt.datetime.fromtimestamp(stop_t ).isoformat(timespec="milliseconds"),
                }, f, indent=2)
        state["recording"] = False
        state["last_folder"] = folder
        state["folder"] = None
        state["started_at"] = None
        print(f"[REC] stopped -> {folder}")
        return jsonify(ok=True, folder=folder)


def parse_args():
    p = argparse.ArgumentParser()
    # Default = -1 = "auto-discover via USB path"; pass an explicit index
    # to override, or -2 to disable that channel entirely.
    p.add_argument("--front-left", type=int, default=-1)
    p.add_argument("--front", type=int, default=-1)
    p.add_argument("--front-right", type=int, default=-1)
    p.add_argument("--back-left", type=int, default=-1)
    p.add_argument("--back-right", type=int, default=-1)
    p.add_argument("--front-narrow", type=int, default=-1)
    # 640x480 matches production inference (alpamayo_infer / autoware_infer).
    # Keep this conservative for six simultaneous USB streams. Bump only
    # after confirming the dock/direct split has enough bus bandwidth.
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--no-fourcc", action="store_true",
                   help="Do not force MJPG; let OpenCV/V4L2 negotiate format.")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--ego-host", default="127.0.0.1")
    p.add_argument("--ego-port", type=int, default=5005)
    p.add_argument("--no-browser", action="store_true",
                   help="Don't auto-open the UI in a browser at startup")
    p.add_argument("--no-ps5", action="store_true",
                   help="Don't spawn ps5_drive.py — disables PS5 cart "
                        "control + control.jsonl logging")
    p.add_argument("--cart-state-file", default=CART_STATE_FILE,
                   help="Path ps5_drive.py writes its live state to "
                        "(also read while recording).")
    p.add_argument("--no-gps", action="store_true",
                   help="Don't spawn the GPS iproxy/TCP reader.")
    p.add_argument("--gps-host", default="127.0.0.1")
    p.add_argument("--gps-port", type=int, default=5006)
    p.add_argument("--no-segmentation-maps", action="store_true",
                   help="Do not record raw segmentation map JSON snapshots "
                        "from scripts/segmentation_infer.py.")
    p.add_argument("--segmentation-map-file", default=SEGMENTATION_MAP_FILE,
                   help="Latest-map JSON path published by segmentation_infer.py.")
    p.add_argument("--autoware-state-file", default=AUTOWARE_STATE_FILE,
                   help="Autosteer state file to attach to segmentation map rows.")
    p.add_argument("--no-realsense", action="store_true",
                   help="Skip auto-detection and disable RealSense capture.")
    p.add_argument("--rs-width", type=int, default=640)
    p.add_argument("--rs-height", type=int, default=480)
    p.add_argument("--rs-fps", type=int, default=30)
    p.add_argument("--rs-depth-only", action="store_true",
                   help="Show RealSense depth in the grid without adding the "
                        "RealSense color tile.")
    return p.parse_args()


def main():
    global ego, cart, gps, segmentation_maps, realsense, show_realsense_color
    args = parse_args()
    show_realsense_color = not args.rs_depth_only
    overrides = {
        "front-left":   args.front_left,
        "front":        args.front,
        "front-right":  args.front_right,
        "back-left":    args.back_left,
        "back-right":   args.back_right,
        "front-narrow": args.front_narrow,
    }
    discovered = discover_video_devices()
    if discovered:
        print(f"[CAM] discovered: {discovered}")
    devices: dict[str, int] = {}
    for name in CAM_NAMES:
        v = overrides[name]
        if v == -2:
            continue  # explicitly disabled
        if v >= 0:
            devices[name] = v
            continue
        if name in discovered:
            devices[name] = discovered[name]
        else:
            print(f"[CAM] {name}: not discovered and no override — skipping")
    CAM_OPEN_ORDER[:] = [n for n in CAM_OPEN_ORDER if n in devices]
    for name in CAM_OPEN_ORDER:
        cameras[name] = Camera(
            name, devices[name], args.width, args.height, args.fps,
            force_mjpg=not args.no_fourcc,
        )
        print(f"[CAM] {name} -> /dev/video{devices[name]}")
    ego = EgoRecorder(host=args.ego_host, port=args.ego_port)
    print(f"[EGO] reading {EGO_STATE_FILE} (writer feeds it from {args.ego_host}:{args.ego_port})")
    if not args.no_ps5:
        cart = CartRecorder(state_file=args.cart_state_file)
        print(f"[CART] reading {args.cart_state_file} (PS5 drive)")
    else:
        print("[CART] --no-ps5: PS5 control + control.jsonl logging disabled")

    if args.no_gps:
        print("[GPS] --no-gps: GPS capture disabled")
    else:
        # Init AFTER EgoRecorder so its (less specific) pgrep doesn't
        # false-match our ego_link.sh 5006 child.
        gps = GpsRecorder(host=args.gps_host, port=args.gps_port)
        print(f"[GPS] reading TCP {args.gps_host}:{args.gps_port}")

    if args.no_segmentation_maps:
        print("[SEG] --no-segmentation-maps: segmentation_maps.jsonl disabled")
    else:
        segmentation_maps = SegmentationMapRecorder(
            map_file=args.segmentation_map_file,
            state_file=args.autoware_state_file,
        )
        print(f"[SEG] recording maps from {args.segmentation_map_file}")

    if args.no_realsense:
        print("[RS] --no-realsense: RealSense capture disabled")
    elif not _HAVE_RS:
        print("[RS] pyrealsense2 not installed — skipping RealSense")
    elif not RealSenseRecorder.detect():
        print("[RS] no RealSense device found — skipping")
    else:
        rs_rec = RealSenseRecorder(
            width=args.rs_width, height=args.rs_height, fps=args.rs_fps,
        )
        if rs_rec.open():
            realsense = rs_rec

    # Auto-open the UI in the local browser. Connect to localhost regardless
    # of bind host so it works when --host is 0.0.0.0.
    if not args.no_browser:
        url = f"http://localhost:{args.port}"

        def _open():
            time.sleep(1.2)  # let Flask bind the socket first
            try:
                webbrowser.open(url)
            except Exception as e:
                print(f"[WARN] could not auto-open browser: {e!r}  ({url})")

        threading.Thread(target=_open, daemon=True).start()

    try:
        app.run(host=args.host, port=args.port, threaded=True, debug=False)
    finally:
        for cam in cameras.values():
            cam.release()
        for rec in model_recorders.values():
            rec.stop_recording()
        if ego is not None:
            ego.shutdown()
        if cart is not None:
            cart.shutdown()
        if gps is not None:
            gps.shutdown()
        if segmentation_maps is not None:
            segmentation_maps.shutdown()
        if realsense is not None:
            realsense.release()


if __name__ == "__main__":
    main()

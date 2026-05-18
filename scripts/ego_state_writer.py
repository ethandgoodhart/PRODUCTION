"""Mirror the iPhone ego-motion stream into /tmp/ego_state.json so the web
UI can show a connected/disconnected dot next to Motor + Arduino.

The actual TCP reader lives in ~/ego_sensor/ego_sensor.py (stdlib-only).
This script is a thin loop: every ~100 ms write the latest sample plus a
``connected`` flag and a ``ts`` field (epoch seconds). The Flask /state
endpoint treats stale files (>1 s old) as "writer died" and surfaces
disconnected to the UI — same convention used by cart_state.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

# ego_sensor is at ~/ego_sensor/ego_sensor.py — outside this repo because
# it's also used standalone. Add the dir to sys.path rather than copying.
_EGO_DIR = os.path.expanduser("~/ego_sensor")
if _EGO_DIR not in sys.path:
    sys.path.insert(0, _EGO_DIR)

from ego_sensor import EgoSensor  # noqa: E402


def _atomic_write(path: str, payload: dict) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f)
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state-file", default=os.environ.get("EGO_STATE_FILE", "/tmp/ego_state.json"))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5005)
    ap.add_argument("--write-hz", type=float, default=10.0)
    args = ap.parse_args()

    sensor = EgoSensor(host=args.host, port=args.port, verbose=False)
    sensor.start()
    period = 1.0 / args.write_hz

    last_t_s: float | None = None
    try:
        while True:
            sample = sensor.read()
            # Connected = TCP socket open AND we've gotten a fresh sample
            # in the last second. iOS app sometimes accepts and goes silent;
            # treat that as disconnected for the indicator.
            now_mono = time.monotonic()
            fresh = sample is not None and (now_mono - sample.t_recv) < 1.0
            connected = bool(sensor.connected and fresh)

            payload: dict = {
                "ts": time.time(),
                "connected": connected,
                "host": f"{args.host}:{args.port}",
            }
            if sample is not None:
                payload["sample"] = {
                    "t_s": sample.t_s,
                    "x_m": sample.x_m,
                    "y_m": sample.y_m,
                    "z_m": sample.z_m,
                    "speed_mps": sample.speed_mps,
                    "yaw_rad": sample.yaw_rad,
                    "yaw_rate_rad_s": sample.yaw_rate_rad_s,
                    "curvature_inv_m": sample.curvature_inv_m,
                    "age_s": now_mono - sample.t_recv,
                }
                payload["history_len"] = len(sensor.history())
                if last_t_s is not None and sample.t_s != last_t_s:
                    pass
                last_t_s = sample.t_s
            _atomic_write(args.state_file, payload)
            time.sleep(period)
    except KeyboardInterrupt:
        pass
    finally:
        sensor.stop()


if __name__ == "__main__":
    main()

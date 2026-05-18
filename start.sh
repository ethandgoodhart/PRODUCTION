#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

PROFILE="$(mktemp -d)"
STATE_FILE="/tmp/cart_state.json"
AUTOWARE_STATE_FILE="/tmp/autoware_state.json"
EGO_STATE_FILE="/tmp/ego_state.json"
FRAMES_DIR="/tmp/cart_frames"

# --mockspeed=N (or --mockspeed N) short-circuits the real PS5/Arduino/ODrive
# stack and runs scripts/mock_state.py instead, so the UI animates (lane
# dashes, wheel, pedals, green dots) with a synthetic MPH. Handy for UI
# work when the hardware isn't plugged in.
# --autosteer turns on the Autoware inference sidecar + ps5_drive --autosteer.
# --video=PATH replays a clip into autoware_infer.py instead of opening live
# USB cameras. Combine with --autosteer to demo / evaluate the predictor on
# canned footage with the wheel commanded from the inferred steer.
MOCK_SPEED=""
AUTOSTEER=""
VIDEO=""
NO_LOOP=""
DRY_RUN=""
# AutoSteer / EgoLanes were trained on a ~30° forward-narrow view. Default
# the front_narrow center-crop to 30° so the model gets the FOV it likes;
# pass --narrow-fov-deg 0 (or any value >= source) to opt out.
NARROW_FOV_DEG="30"
NARROW_SOURCE_FOV_DEG=""
# --model {autoware|alpamayo|clrnet|segmentation} picks the steering brain. All write the
# same /tmp/autoware_state.json schema so ps5_drive.py --autosteer is
# agnostic. autoware = on-device perception stack; alpamayo = remote
# Modal-hosted Alpamayo-R1 trajectory predictor reached over a Modal
# forward tunnel (raw TCP); the deployed app name is passed via
# --app-name (default 'alpamayo-live-demo', which is the name baked
# into mayo/scripts/live_demo_server.py).
# --mph N sets the drive-by-segmentation constant speed target; omitted
# means 8 mph. It is only used with --model segmentation.
MODEL="autoware"
ALPAMAYO_APP_NAME="${ALPAMAYO_APP_NAME:-alpamayo-live-demo}"
SEGMENTATION_MPH="8"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --mockspeed=*)             MOCK_SPEED="${1#*=}"; shift ;;
        --mockspeed)               MOCK_SPEED="$2"; shift 2 ;;
        --autosteer)               AUTOSTEER=1; shift ;;
        --video=*)                 VIDEO="${1#*=}"; shift ;;
        --video)                   VIDEO="$2"; shift 2 ;;
        --no-loop)                 NO_LOOP=1; shift ;;
        --dry-run)                 DRY_RUN=1; shift ;;
        --narrow-fov-deg=*)        NARROW_FOV_DEG="${1#*=}"; shift ;;
        --narrow-fov-deg)          NARROW_FOV_DEG="$2"; shift 2 ;;
        --narrow-source-fov-deg=*) NARROW_SOURCE_FOV_DEG="${1#*=}"; shift ;;
        --narrow-source-fov-deg)   NARROW_SOURCE_FOV_DEG="$2"; shift 2 ;;
        --model=*)                 MODEL="${1#*=}"; shift ;;
        --model)                   MODEL="$2"; shift 2 ;;
        --app-name=*)              ALPAMAYO_APP_NAME="${1#*=}"; shift ;;
        --app-name)                ALPAMAYO_APP_NAME="$2"; shift 2 ;;
        --mph=*)                   SEGMENTATION_MPH="${1#*=}"; shift ;;
        --mph)                     SEGMENTATION_MPH="$2"; shift 2 ;;
        *) echo "start.sh: unknown arg $1" >&2; exit 2 ;;
    esac
done

if [[ "$MODEL" != "autoware" && "$MODEL" != "alpamayo" && "$MODEL" != "clrnet" && "$MODEL" != "segmentation" ]]; then
    echo "start.sh: --model must be 'autoware', 'alpamayo', 'clrnet', or 'segmentation' (got '$MODEL')" >&2
    exit 2
fi
if [[ "$MODEL" == "alpamayo" && -z "$ALPAMAYO_APP_NAME" ]]; then
    echo "start.sh: --model alpamayo requires --app-name <modal-app> " \
         "(or \$ALPAMAYO_APP_NAME)" >&2
    exit 2
fi
# When the operator picks alpamayo or clrnet as the brain, they obviously
# want it to drive — auto-engage --autosteer in ps5_drive so the wheel +
# pedals follow the model. The PS5 stick + trigger overrides remain in
# effect.
if [[ "$MODEL" == "alpamayo" || "$MODEL" == "clrnet" || "$MODEL" == "segmentation" ]]; then
    AUTOSTEER=1
fi

if [[ -n "$VIDEO" && ! -f "$VIDEO" ]]; then
    echo "start.sh: --video file not found: $VIDEO" >&2
    exit 2
fi

cleanup() {
    kill "$SRV" 2>/dev/null || true
    kill "$DRIVE_LOOP" 2>/dev/null || true
    kill "$MOCK_PID" 2>/dev/null || true
    kill "$AUTOWARE_PID" 2>/dev/null || true
    kill "$EGO_PID" 2>/dev/null || true
    kill "$EGO_LINK_PID" 2>/dev/null || true
    kill "$TUNNEL_PID" 2>/dev/null || true
    pkill -P "$DRIVE_LOOP" 2>/dev/null || true
    pkill -P "$EGO_LINK_PID" 2>/dev/null || true
    pkill -f "scripts/ps5_drive.py" 2>/dev/null || true
    pkill -f "scripts/mock_state.py" 2>/dev/null || true
    pkill -f "scripts/autoware_infer.py" 2>/dev/null || true
    pkill -f "scripts/alpamayo_infer.py" 2>/dev/null || true
    pkill -f "scripts/clrnet_infer.py" 2>/dev/null || true
    pkill -f "scripts/segmentation_infer.py" 2>/dev/null || true
    pkill -f "scripts/ego_state_writer.py" 2>/dev/null || true
    pkill -f "ego_sensor/ego_link.sh" 2>/dev/null || true
    # Any iproxy spawned by ego_link.sh.
    pkill -f "iproxy 5005 5005" 2>/dev/null || true
    rm -rf "$PROFILE"
    rm -f "$STATE_FILE" "$STATE_FILE.tmp"
    rm -f "$AUTOWARE_STATE_FILE" "$AUTOWARE_STATE_FILE.tmp"
    rm -f "$EGO_STATE_FILE" "$EGO_STATE_FILE.tmp"
    rm -f /tmp/teleop_cmd.json /tmp/teleop_cmd.json.tmp
    rm -rf "$FRAMES_DIR"
}
trap cleanup EXIT

export CART_STATE_FILE="$STATE_FILE"
export AUTOWARE_STATE_FILE
export EGO_STATE_FILE
export CART_FRAMES_DIR="$FRAMES_DIR"
mkdir -p "$FRAMES_DIR"

# iPhone ARKit ego-motion publisher. ego_link.sh supervises the
# usbmuxd/iproxy USB tunnel — waits for the phone to appear, launches
# iproxy 5005 5005, and restarts it if the cable is replugged.
# ego_state_writer.py then reads the JSONL stream off 127.0.0.1:5005 and
# mirrors connected/sample state into $EGO_STATE_FILE. If the tunnel
# isn't yet up or the iOS app isn't streaming, the writer just keeps
# retrying and the UI dot stays red.
~/ego_sensor/ego_link.sh 5005 >>/tmp/ego_link.log 2>&1 &
EGO_LINK_PID=$!
echo "[start] ego_link pid=$EGO_LINK_PID (USB iproxy supervisor)" \
    >>/tmp/ego_link.log
/usr/bin/python3 scripts/ego_state_writer.py --state-file "$EGO_STATE_FILE" \
    >>/tmp/ego_state_writer.log 2>&1 &
EGO_PID=$!
echo "[start] ego_state_writer pid=$EGO_PID -> $EGO_STATE_FILE" \
    >>/tmp/ego_state_writer.log

# Inference sidecar — picked by --model. Both write to
# $AUTOWARE_STATE_FILE so ps5_drive.py --autosteer reads from one place
# regardless of which brain is driving. Both publish to $FRAMES_DIR for
# the web UI tiles.
if [[ "$MODEL" == "autoware" ]]; then
    # Autoware: full on-device perception stack (seg/3d/lanes/steer/speed)
    # so the lane/depth/objects/seg viz tiles populate regardless of
    # --autosteer. The --autosteer flag only controls whether ps5_drive
    # USES the steering output.
    INFER_ARGS=(--frames-dir "$FRAMES_DIR" --state-file "$AUTOWARE_STATE_FILE")
    if [[ -n "$VIDEO" ]]; then
        INFER_ARGS+=(--video "$VIDEO")
        [[ -n "$NO_LOOP" ]] && INFER_ARGS+=(--no-loop)
        echo "[start] VIDEO mode — replaying $VIDEO into autoware pipeline" \
            >>/tmp/autoware_infer.log
    fi
    # A value of 0 (or empty) means "don't crop". Anything > 0 narrows the
    # front_narrow stream's apparent FOV to that many degrees.
    if [[ -n "$NARROW_FOV_DEG" && "$NARROW_FOV_DEG" != "0" ]]; then
        INFER_ARGS+=(--narrow-fov-deg "$NARROW_FOV_DEG")
        if [[ -n "$NARROW_SOURCE_FOV_DEG" ]]; then
            INFER_ARGS+=(--narrow-source-fov-deg "$NARROW_SOURCE_FOV_DEG")
        fi
    fi
    # autoware_infer.py needs torch — runs on system Python 3.12 with the
    # Jetson CUDA wheel, not PRODUCTION's uv-managed 3.13.
    /usr/bin/python3 scripts/autoware_infer.py "${INFER_ARGS[@]}" \
        >>/tmp/autoware_infer.log 2>&1 &
    AUTOWARE_PID=$!
    echo "[start] model=autoware (on-device perception)" >>/tmp/autoware_infer.log
elif [[ "$MODEL" == "clrnet" ]]; then
    # CLRerNet: ON-DEVICE single-image lane detector (Jetson Thor torch
    # 2.10/cu130 + mmcv 2.x). Uses front_wide only; the steering loop is
    # the centerline+lookahead controller from lane-detection/visualize.py.
    # Pedals are a constant-7-mph closed loop using cart_state.json's mph.
    INFER_ARGS=(--frames-dir "$FRAMES_DIR" --state-file "$AUTOWARE_STATE_FILE"
                --cart-state-file "$STATE_FILE")
    if [[ -n "$VIDEO" ]]; then
        INFER_ARGS+=(--video "$VIDEO")
        [[ -n "$NO_LOOP" ]] && INFER_ARGS+=(--no-loop)
        echo "[start] VIDEO mode — replaying $VIDEO into clrnet pipeline" \
            >>/tmp/autoware_infer.log
    fi
    /usr/bin/python3 scripts/clrnet_infer.py "${INFER_ARGS[@]}" \
        >>/tmp/autoware_infer.log 2>&1 &
    AUTOWARE_PID=$!
    echo "[start] model=clrnet (CLRerNet, on-device, target 7 mph)" \
        >>/tmp/autoware_infer.log
elif [[ "$MODEL" == "segmentation" ]]; then
    # drive-by-segmentation: ON-DEVICE SegFormer semantic segmentation,
    # BEV path planner, and steering estimator from
    # /home/caddy/drive-by-segmentation/live.py. Uses front_wide and
    # publishes seg + bev viz tiles. Pedals hold a constant target
    # speed; default is 8 mph unless --mph overrides it.
    INFER_ARGS=(--frames-dir "$FRAMES_DIR" --state-file "$AUTOWARE_STATE_FILE"
                --target-mph "$SEGMENTATION_MPH"
                --source realsense --no-depth)
    if [[ -n "$VIDEO" ]]; then
        INFER_ARGS+=(--video "$VIDEO")
        [[ -n "$NO_LOOP" ]] && INFER_ARGS+=(--no-loop)
        echo "[start] VIDEO mode — replaying $VIDEO into segmentation pipeline" \
            >>/tmp/autoware_infer.log
    fi
    /usr/bin/python3 scripts/segmentation_infer.py "${INFER_ARGS[@]}" \
        >>/tmp/autoware_infer.log 2>&1 &
    AUTOWARE_PID=$!
    echo "[start] model=segmentation (drive-by-segmentation, target ${SEGMENTATION_MPH} mph)" \
        >>/tmp/autoware_infer.log
else
    # Alpamayo: remote Modal-hosted Alpamayo-R1 reached over a Modal
    # forward tunnel (raw TCP, region=us-west on the server side). No
    # GPU on the Jetson; needs the `modal` SDK + a working modal auth.
    # We deliberately use mayo's .venv-client interpreter (Python 3.12)
    # because it's the only one on this box with `modal` installed —
    # /usr/bin/python3 doesn't have it and PRODUCTION's uv-managed env
    # would require an extra `uv add modal` round-trip. cv2/msgpack/
    # numpy are present in that venv too.
    ALPAMAYO_PY="${ALPAMAYO_PY:-/home/caddy/mayo/.venv-client/bin/python}"
    if [[ ! -x "$ALPAMAYO_PY" ]]; then
        echo "[start] ERROR: $ALPAMAYO_PY not found; install modal there" \
             "or set \$ALPAMAYO_PY to a python that has modal+cv2+msgpack" \
            >>/tmp/autoware_infer.log
        exit 2
    fi
    INFER_ARGS=(--frames-dir "$FRAMES_DIR" --state-file "$AUTOWARE_STATE_FILE"
                --app-name "$ALPAMAYO_APP_NAME")
    # Optional codec / quality overrides — honor env vars so callers
    # like `ALPAMAYO_CODEC=hevc ALPAMAYO_QUALITY=36 ./alpamayo.sh` can
    # pick the wire format without editing the script. Defaults to the
    # values baked into alpamayo_infer.py if unset.
    if [[ -n "${ALPAMAYO_CODEC:-}" ]]; then
        INFER_ARGS+=(--codec "$ALPAMAYO_CODEC")
    fi
    if [[ -n "${ALPAMAYO_QUALITY:-}" ]]; then
        INFER_ARGS+=(--quality "$ALPAMAYO_QUALITY")
    fi
    if [[ -n "$VIDEO" ]]; then
        INFER_ARGS+=(--video "$VIDEO")
        [[ -n "$NO_LOOP" ]] && INFER_ARGS+=(--no-loop)
        echo "[start] VIDEO mode — replaying $VIDEO into alpamayo pipeline" \
            >>/tmp/autoware_infer.log
    fi
    "$ALPAMAYO_PY" scripts/alpamayo_infer.py "${INFER_ARGS[@]}" \
        >>/tmp/autoware_infer.log 2>&1 &
    AUTOWARE_PID=$!
    echo "[start] model=alpamayo app=$ALPAMAYO_APP_NAME py=$ALPAMAYO_PY" \
        "(modal tunnel transport)" >>/tmp/autoware_infer.log
fi

# Web UI (Flask) — reads $CART_STATE_FILE for the MPH readout.
(cd web && exec python3 app.py) >/tmp/flask.log 2>&1 &
SRV=$!

if [[ -n "$MOCK_SPEED" ]]; then
    echo "[start] MOCK mode — mph=$MOCK_SPEED (skipping PS5/Arduino/ODrive)" >>/tmp/ps5_drive.log
    python3 scripts/mock_state.py --mph "$MOCK_SPEED" --state-file "$STATE_FILE" \
        >>/tmp/ps5_drive.log 2>&1 &
    MOCK_PID=$!
else
    # PS5 drive loop — ps5_drive.py needs a DualSense already paired before
    # pygame can init its joystick layer. Wait for one to show up in
    # /proc/bus/input/devices, launch the driver, and if the driver exits
    # (BT drop, out of range) loop back and wait for reconnect. State-file
    # staleness tells the web UI when the controller is gone.
    (
        while true; do
            while ! grep -qi -E '(sony|dualsense|wireless controller|ps5)' /proc/bus/input/devices 2>/dev/null; do
                sleep 1
            done
            echo "[start] controller detected — launching ps5_drive" >>/tmp/ps5_drive.log
            PS5_ARGS=(--headless --state-file "$STATE_FILE")
            if [[ -n "$DRY_RUN" ]]; then
                PS5_ARGS+=(--dry-run)
            fi
            if [[ -n "$AUTOSTEER" ]]; then
                PS5_ARGS+=(--autosteer --autosteer-state-file "$AUTOWARE_STATE_FILE")
            fi
            uv run python scripts/ps5_drive.py "${PS5_ARGS[@]}" >>/tmp/ps5_drive.log 2>&1 || true
            echo "[start] ps5_drive exited — waiting for controller to reconnect" >>/tmp/ps5_drive.log
            sleep 1
        done
    ) &
    DRIVE_LOOP=$!
fi

for i in {1..30}; do
    curl -s -o /dev/null http://127.0.0.1:5050 && break
    sleep 0.2
done

# Cloudflare named tunnel — exposes the Flask server at
# https://caddy.ethandgoodhart.com so the teleop page (/teleop) is
# reachable from any network. Persistent outbound connection; no port
# forwarding or firewall changes needed.
TUNNEL_PID=""
if command -v cloudflared &>/dev/null; then
    cloudflared tunnel run caddy \
        >>/tmp/cloudflared.log 2>&1 &
    TUNNEL_PID=$!
    echo "[start] cloudflared tunnel pid=$TUNNEL_PID (caddy.ethandgoodhart.com)"
else
    echo "[start] cloudflared not installed — teleop tunnel disabled."
fi

firefox --no-remote --new-instance --profile "$PROFILE" --kiosk http://127.0.0.1:5050

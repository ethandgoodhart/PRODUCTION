#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

PROFILE="$(mktemp -d)"
STATE_FILE="/tmp/cart_state.json"
AUTOWARE_STATE_FILE="/tmp/autoware_state.json"
EGO_STATE_FILE="/tmp/ego_state.json"
GPS_STATE_FILE="/tmp/gps_state.json"
FRAMES_DIR="/tmp/cart_frames"
VIDEO_CONTROL_FILE="/tmp/video_control.json"
OBJECT_PREDICTOR_URL=""
OBJECT_PREDICTOR_PID=""

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
OFFLINE=""
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
SEGMENTATION_CAMERA_SLOT="${SEGMENTATION_CAMERA_SLOT:-CAM_FRONT}"
SEGMENTATION_CAMERA_INDEX="${SEGMENTATION_CAMERA_INDEX:-}"
SEGMENTATION_WITH_CLRNET=""
SEGMENTATION_CLRNET_CACHE="${SEGMENTATION_CLRNET_CACHE:-}"
SEGMENTATION_YOLO_LIVE="${SEGMENTATION_YOLO_LIVE:-}"
SEGMENTATION_YOLO_MODEL="${SEGMENTATION_YOLO_MODEL:-yolo11x.pt}"
SEGMENTATION_YOLO_IMGSZ="${SEGMENTATION_YOLO_IMGSZ:-960}"
SEGMENTATION_YOLO_DEVICE="${SEGMENTATION_YOLO_DEVICE:-0}"
SEGMENTATION_PUBLISH_HZ="${SEGMENTATION_PUBLISH_HZ:-}"
SEGMENTATION_INFER_HZ="${SEGMENTATION_INFER_HZ:-}"
SEGMENTATION_CACHE_META="${SEGMENTATION_CACHE_META:-}"
SEGMENTATION_YOLO_CACHE="${SEGMENTATION_YOLO_CACHE:-}"
SEGMENTATION_YOLO_MIN_CONF="${SEGMENTATION_YOLO_MIN_CONF:-0.20}"
SEGMENTATION_OBJECT_PREDICTOR_URL="${SEGMENTATION_OBJECT_PREDICTOR_URL:-}"
SEGMENTATION_OBJECT_PREDICTOR_TIMEOUT_MS="${SEGMENTATION_OBJECT_PREDICTOR_TIMEOUT_MS:-250}"
SEGMENTATION_WITH_SOCIAL_STGCNN="${SEGMENTATION_WITH_SOCIAL_STGCNN:-}"
SOCIAL_STGCNN_HOST="${SOCIAL_STGCNN_HOST:-127.0.0.1}"
SOCIAL_STGCNN_PORT="${SOCIAL_STGCNN_PORT:-8766}"
SOCIAL_STGCNN_REPO="${SOCIAL_STGCNN_REPO:-$PWD/.cache/third_party/social-nce-stgcnn}"
SOCIAL_STGCNN_CHECKPOINT_DIR="${SOCIAL_STGCNN_CHECKPOINT_DIR:-$PWD/.cache/third_party/social-nce-stgcnn/checkpoint-snce/snce-social-stgcnn-univ}"
SOCIAL_STGCNN_DEVICE="${SOCIAL_STGCNN_DEVICE:-cpu}"
SEGMENTATION_WITH_TRAJECTRONPP="${SEGMENTATION_WITH_TRAJECTRONPP:-}"
TRAJECTRONPP_HOST="${TRAJECTRONPP_HOST:-127.0.0.1}"
TRAJECTRONPP_PORT="${TRAJECTRONPP_PORT:-8765}"
TRAJECTRONPP_REPO="${TRAJECTRONPP_REPO:-$PWD/.cache/third_party/Trajectron-plus-plus}"
TRAJECTRONPP_MODEL_DIR="${TRAJECTRONPP_MODEL_DIR:-$PWD/.cache/third_party/Trajectron-plus-plus/experiments/nuScenes/models/int_ee_me}"
TRAJECTRONPP_CHECKPOINT="${TRAJECTRONPP_CHECKPOINT:-12}"
SEGMENTATION_HOMOGRAPHY_CALIB="${SEGMENTATION_HOMOGRAPHY_CALIB:-$HOME/Programming/drive-by-segmentation/camera_calibration.json}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --mockspeed=*)             MOCK_SPEED="${1#*=}"; shift ;;
        --mockspeed)               MOCK_SPEED="$2"; shift 2 ;;
        --autosteer)               AUTOSTEER=1; shift ;;
        --video=*)                 VIDEO="${1#*=}"; shift ;;
        --video)                   VIDEO="$2"; shift 2 ;;
        --no-loop)                 NO_LOOP=1; shift ;;
        --dry-run)                 DRY_RUN=1; shift ;;
        --offline|--ofline)        OFFLINE=1; shift ;;
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
        --with-clrnet)             SEGMENTATION_WITH_CLRNET=1; shift ;;
        --clrnet-cache=*)          SEGMENTATION_CLRNET_CACHE="${1#*=}"; shift ;;
        --clrnet-cache)            SEGMENTATION_CLRNET_CACHE="$2"; shift 2 ;;
        --yolo-live)               SEGMENTATION_YOLO_LIVE=1; shift ;;
        --yolo-model=*)            SEGMENTATION_YOLO_MODEL="${1#*=}"; shift ;;
        --yolo-model)              SEGMENTATION_YOLO_MODEL="$2"; shift 2 ;;
        --yolo-imgsz=*)            SEGMENTATION_YOLO_IMGSZ="${1#*=}"; shift ;;
        --yolo-imgsz)              SEGMENTATION_YOLO_IMGSZ="$2"; shift 2 ;;
        --yolo-device=*)           SEGMENTATION_YOLO_DEVICE="${1#*=}"; shift ;;
        --yolo-device)             SEGMENTATION_YOLO_DEVICE="$2"; shift 2 ;;
        --seg-cache-meta=*)       SEGMENTATION_CACHE_META="${1#*=}"; shift ;;
        --seg-cache-meta)         SEGMENTATION_CACHE_META="$2"; shift 2 ;;
        --seg-yolo-cache=*)       SEGMENTATION_YOLO_CACHE="${1#*=}"; shift ;;
        --seg-yolo-cache)         SEGMENTATION_YOLO_CACHE="$2"; shift 2 ;;
        --seg-yolo-min-conf=*)    SEGMENTATION_YOLO_MIN_CONF="${1#*=}"; shift ;;
        --seg-yolo-min-conf)      SEGMENTATION_YOLO_MIN_CONF="$2"; shift 2 ;;
        --seg-object-predictor-url=*) SEGMENTATION_OBJECT_PREDICTOR_URL="${1#*=}"; shift ;;
        --seg-object-predictor-url) SEGMENTATION_OBJECT_PREDICTOR_URL="$2"; shift 2 ;;
        --seg-object-predictor-timeout-ms=*) SEGMENTATION_OBJECT_PREDICTOR_TIMEOUT_MS="${1#*=}"; shift ;;
        --seg-object-predictor-timeout-ms) SEGMENTATION_OBJECT_PREDICTOR_TIMEOUT_MS="$2"; shift 2 ;;
        --with-social-stgcnn)     SEGMENTATION_WITH_SOCIAL_STGCNN=1; shift ;;
        --social-stgcnn-repo=*)   SOCIAL_STGCNN_REPO="${1#*=}"; shift ;;
        --social-stgcnn-repo)     SOCIAL_STGCNN_REPO="$2"; shift 2 ;;
        --social-stgcnn-checkpoint-dir=*) SOCIAL_STGCNN_CHECKPOINT_DIR="${1#*=}"; shift ;;
        --social-stgcnn-checkpoint-dir) SOCIAL_STGCNN_CHECKPOINT_DIR="$2"; shift 2 ;;
        --social-stgcnn-port=*)   SOCIAL_STGCNN_PORT="${1#*=}"; shift ;;
        --social-stgcnn-port)     SOCIAL_STGCNN_PORT="$2"; shift 2 ;;
        --social-stgcnn-device=*) SOCIAL_STGCNN_DEVICE="${1#*=}"; shift ;;
        --social-stgcnn-device)   SOCIAL_STGCNN_DEVICE="$2"; shift 2 ;;
        --with-trajectronpp)      SEGMENTATION_WITH_TRAJECTRONPP=1; shift ;;
        --trajectronpp-repo=*)    TRAJECTRONPP_REPO="${1#*=}"; shift ;;
        --trajectronpp-repo)      TRAJECTRONPP_REPO="$2"; shift 2 ;;
        --trajectronpp-model-dir=*) TRAJECTRONPP_MODEL_DIR="${1#*=}"; shift ;;
        --trajectronpp-model-dir) TRAJECTRONPP_MODEL_DIR="$2"; shift 2 ;;
        --trajectronpp-checkpoint=*) TRAJECTRONPP_CHECKPOINT="${1#*=}"; shift ;;
        --trajectronpp-checkpoint) TRAJECTRONPP_CHECKPOINT="$2"; shift 2 ;;
        --trajectronpp-port=*)    TRAJECTRONPP_PORT="${1#*=}"; shift ;;
        --trajectronpp-port)      TRAJECTRONPP_PORT="$2"; shift 2 ;;
        --seg-camera-slot=*)       SEGMENTATION_CAMERA_SLOT="${1#*=}"; shift ;;
        --seg-camera-slot)         SEGMENTATION_CAMERA_SLOT="$2"; shift 2 ;;
        --seg-camera-index=*)      SEGMENTATION_CAMERA_INDEX="${1#*=}"; shift ;;
        --seg-camera-index)        SEGMENTATION_CAMERA_INDEX="$2"; shift 2 ;;
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
    kill "$OBJECT_PREDICTOR_PID" 2>/dev/null || true
    kill "$EGO_PID" 2>/dev/null || true
    kill "$EGO_LINK_PID" 2>/dev/null || true
    kill "$GPS_LINK_PID" 2>/dev/null || true
    pkill -P "$DRIVE_LOOP" 2>/dev/null || true
    pkill -P "$EGO_LINK_PID" 2>/dev/null || true
    pkill -P "$GPS_LINK_PID" 2>/dev/null || true
    pkill -f "scripts/ps5_drive.py" 2>/dev/null || true
    pkill -f "scripts/mock_state.py" 2>/dev/null || true
    pkill -f "scripts/autoware_infer.py" 2>/dev/null || true
    pkill -f "scripts/alpamayo_infer.py" 2>/dev/null || true
    pkill -f "scripts/clrnet_infer.py" 2>/dev/null || true
    pkill -f "scripts/segmentation_infer.py" 2>/dev/null || true
    pkill -f "scripts/object_predictor_social_stgcnn.py" 2>/dev/null || true
    pkill -f "scripts/object_predictor_trajectronpp.py" 2>/dev/null || true
    pkill -f "scripts/ego_state_writer.py" 2>/dev/null || true
    pkill -f "ego_sensor/ego_link.sh" 2>/dev/null || true
    # Any iproxy spawned by ego_link.sh.
    pkill -f "iproxy 5005 5005" 2>/dev/null || true
    pkill -f "iproxy 5006 5006" 2>/dev/null || true
    rm -rf "$PROFILE"
    rm -f "$STATE_FILE" "$STATE_FILE.tmp"
    rm -f "$AUTOWARE_STATE_FILE" "$AUTOWARE_STATE_FILE.tmp"
    rm -f "$EGO_STATE_FILE" "$EGO_STATE_FILE.tmp"
    rm -f "$GPS_STATE_FILE" "$GPS_STATE_FILE.tmp"
    rm -f "$VIDEO_CONTROL_FILE" "$VIDEO_CONTROL_FILE.tmp"
    rm -f /tmp/teleop_cmd.json /tmp/teleop_cmd.json.tmp
    rm -rf "$FRAMES_DIR"
}
trap cleanup EXIT

export CART_STATE_FILE="$STATE_FILE"
export AUTOWARE_STATE_FILE
export EGO_STATE_FILE
export GPS_STATE_FILE
export VIDEO_CONTROL_FILE
export CART_FRAMES_DIR="$FRAMES_DIR"
mkdir -p "$FRAMES_DIR"

if [[ -n "$SEGMENTATION_WITH_SOCIAL_STGCNN" ]]; then
    OBJECT_PREDICTOR_URL="http://${SOCIAL_STGCNN_HOST}:${SOCIAL_STGCNN_PORT}/predict"
    /opt/homebrew/bin/python3 scripts/object_predictor_social_stgcnn.py \
        --host "$SOCIAL_STGCNN_HOST" \
        --port "$SOCIAL_STGCNN_PORT" \
        --repo "$SOCIAL_STGCNN_REPO" \
        --checkpoint-dir "$SOCIAL_STGCNN_CHECKPOINT_DIR" \
        --device "$SOCIAL_STGCNN_DEVICE" \
        >>/tmp/object_predictor_social_stgcnn.log 2>&1 &
    OBJECT_PREDICTOR_PID=$!
    echo "[start] social-stgcnn provider pid=$OBJECT_PREDICTOR_PID url=$OBJECT_PREDICTOR_URL" \
        >>/tmp/object_predictor_social_stgcnn.log
    if /opt/homebrew/bin/python3 - "$SOCIAL_STGCNN_HOST" "$SOCIAL_STGCNN_PORT" <<'PY'
import json
import sys
import time
import urllib.request

host, port = sys.argv[1], sys.argv[2]
deadline = time.monotonic() + 8.0
while time.monotonic() < deadline:
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/health", timeout=1.0) as r:
            data = json.loads(r.read().decode("utf-8"))
        if data.get("ready"):
            raise SystemExit(0)
    except Exception:
        pass
    time.sleep(0.25)
raise SystemExit(1)
PY
    then
        SEGMENTATION_OBJECT_PREDICTOR_URL="$OBJECT_PREDICTOR_URL"
        echo "[start] social-stgcnn provider healthy; routing object futures to it" \
            >>/tmp/object_predictor_social_stgcnn.log
    else
        echo "[start] social-stgcnn provider unavailable; keeping constant-velocity object fallback" \
            >>/tmp/object_predictor_social_stgcnn.log
    fi
elif [[ -n "$SEGMENTATION_WITH_TRAJECTRONPP" ]]; then
    OBJECT_PREDICTOR_URL="http://${TRAJECTRONPP_HOST}:${TRAJECTRONPP_PORT}/predict"
    /opt/homebrew/bin/python3 scripts/object_predictor_trajectronpp.py \
        --host "$TRAJECTRONPP_HOST" \
        --port "$TRAJECTRONPP_PORT" \
        --repo "$TRAJECTRONPP_REPO" \
        --model-dir "$TRAJECTRONPP_MODEL_DIR" \
        --checkpoint "$TRAJECTRONPP_CHECKPOINT" \
        >>/tmp/object_predictor_trajectronpp.log 2>&1 &
    OBJECT_PREDICTOR_PID=$!
    echo "[start] trajectron++ provider pid=$OBJECT_PREDICTOR_PID url=$OBJECT_PREDICTOR_URL" \
        >>/tmp/object_predictor_trajectronpp.log
    sleep 1
    if /opt/homebrew/bin/python3 - "$TRAJECTRONPP_HOST" "$TRAJECTRONPP_PORT" <<'PY'
import json
import sys
import urllib.error
import urllib.request

host, port = sys.argv[1], sys.argv[2]
try:
    with urllib.request.urlopen(f"http://{host}:{port}/health", timeout=0.5) as r:
        data = json.loads(r.read().decode("utf-8"))
    raise SystemExit(0 if data.get("ready") else 1)
except Exception:
    raise SystemExit(1)
PY
    then
        SEGMENTATION_OBJECT_PREDICTOR_URL="$OBJECT_PREDICTOR_URL"
        echo "[start] trajectron++ provider healthy; routing object futures to it" \
            >>/tmp/object_predictor_trajectronpp.log
    else
        echo "[start] trajectron++ provider unavailable; keeping constant-velocity object fallback" \
            >>/tmp/object_predictor_trajectronpp.log
    fi
fi

# iPhone ARKit ego-motion publisher. ego_link.sh supervises the
# usbmuxd/iproxy USB tunnel — waits for the phone to appear, launches
# iproxy 5005 5005, and restarts it if the cable is replugged.
# ego_state_writer.py then reads the JSONL stream off 127.0.0.1:5005 and
# mirrors connected/sample state into $EGO_STATE_FILE. If the tunnel
# isn't yet up or the iOS app isn't streaming, the writer just keeps
# retrying and the UI dot stays red.
if [[ -z "$OFFLINE" ]]; then
    ~/ego_sensor/ego_link.sh 5005 >>/tmp/ego_link.log 2>&1 &
    EGO_LINK_PID=$!
    echo "[start] ego_link pid=$EGO_LINK_PID (USB iproxy supervisor)" \
        >>/tmp/ego_link.log
    /usr/bin/python3 scripts/ego_state_writer.py --state-file "$EGO_STATE_FILE" \
        >>/tmp/ego_state_writer.log 2>&1 &
    EGO_PID=$!
    echo "[start] ego_state_writer pid=$EGO_PID -> $EGO_STATE_FILE" \
        >>/tmp/ego_state_writer.log
else
    echo '{"connected":false,"ts":0}' > "$EGO_STATE_FILE"
    echo "[start] OFFLINE mode — skipping iPhone ARKit ego link" \
        >>/tmp/ego_state_writer.log
fi

# iPhone CoreLocation GPS publisher. The iOS app exposes this as a separate
# JSONL stream on port 5006; web/app.py connects to it and mirrors the latest
# fix into $GPS_STATE_FILE for segmentation route bias.
if [[ -z "$OFFLINE" ]]; then
    ~/ego_sensor/ego_link.sh 5006 >>/tmp/gps_link.log 2>&1 &
    GPS_LINK_PID=$!
    echo "[start] gps_link pid=$GPS_LINK_PID (USB iproxy supervisor)" \
        >>/tmp/gps_link.log
else
    echo '{"connected":false,"ts":0}' > "$GPS_STATE_FILE"
    echo "[start] OFFLINE mode — skipping iPhone GPS link" >>/tmp/gps_link.log
fi

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
    # /home/caddy/drive-by-segmentation/live.py. Uses the center-front
    # E2E-calibrated PVC camera only and publishes seg + bev viz tiles.
    # Pedals target the requested speed; when iPhone ARKit ego-speed is fresh,
    # segmentation_infer closes the loop on actual MPH. If the phone stream is
    # unavailable, it falls back to the calibrated feed-forward pedal pot.
    INFER_ARGS=(--frames-dir "$FRAMES_DIR" --state-file "$AUTOWARE_STATE_FILE"
                --target-mph "$SEGMENTATION_MPH"
                --source uvc
                --calib "$SEGMENTATION_HOMOGRAPHY_CALIB"
                --camera-slot "$SEGMENTATION_CAMERA_SLOT"
                --gps-route-gain 0.6
                --gps-route-max-bias-deg 150
                --gps-route-lookahead-m 5)
    if [[ -z "$SEGMENTATION_WITH_CLRNET" && -z "$SEGMENTATION_CLRNET_CACHE" ]]; then
        INFER_ARGS+=(--no-clrnet)
    fi
    if [[ -n "$SEGMENTATION_CLRNET_CACHE" ]]; then
        INFER_ARGS+=(--clrnet-cache-file "$SEGMENTATION_CLRNET_CACHE")
    fi
    if [[ -n "$SEGMENTATION_YOLO_LIVE" ]]; then
        INFER_ARGS+=(--yolo-live
                     --yolo-model "$SEGMENTATION_YOLO_MODEL"
                     --yolo-imgsz "$SEGMENTATION_YOLO_IMGSZ"
                     --yolo-device "$SEGMENTATION_YOLO_DEVICE"
                     --yolo-min-conf "$SEGMENTATION_YOLO_MIN_CONF")
    fi
    if [[ -n "$SEGMENTATION_CACHE_META" ]]; then
        INFER_ARGS+=(--segmentation-cache-meta "$SEGMENTATION_CACHE_META")
    fi
    if [[ -n "$SEGMENTATION_YOLO_CACHE" ]]; then
        INFER_ARGS+=(--yolo-cache-file "$SEGMENTATION_YOLO_CACHE"
                     --yolo-min-conf "$SEGMENTATION_YOLO_MIN_CONF")
        if [[ -n "$SEGMENTATION_OBJECT_PREDICTOR_URL" ]]; then
            INFER_ARGS+=(--object-predictor-url "$SEGMENTATION_OBJECT_PREDICTOR_URL"
                         --object-predictor-timeout-ms "$SEGMENTATION_OBJECT_PREDICTOR_TIMEOUT_MS")
        fi
    fi
    if [[ -n "$SEGMENTATION_PUBLISH_HZ" ]]; then
        INFER_ARGS+=(--publish-hz "$SEGMENTATION_PUBLISH_HZ")
    fi
    if [[ -n "$SEGMENTATION_INFER_HZ" ]]; then
        INFER_ARGS+=(--infer-hz "$SEGMENTATION_INFER_HZ")
    fi
    if [[ -n "$SEGMENTATION_CAMERA_INDEX" ]]; then
        INFER_ARGS+=(--camera-index "$SEGMENTATION_CAMERA_INDEX")
    fi
    # In segmentation mode the model sidecar owns the center-front camera and
    # publishes /tmp/cart_frames/front.jpg. Do not let Flask's raw live-camera
    # probe race it for /dev/video4.
    export LIVE_CAMERA_COUNT=0
    if [[ -n "$VIDEO" ]]; then
        INFER_ARGS+=(--video "$VIDEO" --video-control-file "$VIDEO_CONTROL_FILE")
        [[ -n "$NO_LOOP" ]] && INFER_ARGS+=(--no-loop)
        echo "[start] VIDEO mode — replaying $VIDEO into segmentation pipeline" \
            >>/tmp/autoware_infer.log
    fi
    SEGMENTATION_PY="$(command -v python3)"
    "$SEGMENTATION_PY" scripts/segmentation_infer.py "${INFER_ARGS[@]}" \
        >>/tmp/autoware_infer.log 2>&1 &
    AUTOWARE_PID=$!
    echo "[start] model=segmentation (target ${SEGMENTATION_MPH} mph, homography_calib=$SEGMENTATION_HOMOGRAPHY_CALIB, yolo_live=${SEGMENTATION_YOLO_LIVE:-0}, py=$SEGMENTATION_PY)" \
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

# Web UI (Flask) — reads $CART_STATE_FILE for controls/status, but the
# dashboard MPH readout is overridden with fresh iPhone ARKit ego-motion
# speed from $EGO_STATE_FILE, falling back to iPhone CoreLocation GPS speed
# from port 5006 when ARKit speed is unavailable. The old gas/brake-derived
# estimate remains in /state as drive_mph_estimate.
(cd web && exec python3 app.py) >/tmp/flask.log 2>&1 &
SRV=$!
echo "[start] web ui pid=$SRV; MPH source=iPhone ARKit ego speed, GPS fallback" \
    >>/tmp/flask.log

if [[ -n "$OFFLINE" && -z "$MOCK_SPEED" ]]; then
    MOCK_SPEED="$SEGMENTATION_MPH"
fi

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

URL="http://127.0.0.1:5050"
if [[ -n "$OFFLINE" ]]; then
    if command -v open >/dev/null 2>&1; then
        open "$URL" >/dev/null 2>&1 || true
    elif command -v firefox >/dev/null 2>&1; then
        firefox --no-remote --new-instance --profile "$PROFILE" "$URL" >/dev/null 2>&1 || true
    else
        echo "[start] web ui ready at $URL"
    fi
    echo "[start] OFFLINE mode — web ui ready at $URL; waiting until interrupted" \
        >>/tmp/flask.log
    wait
else
    if command -v firefox >/dev/null 2>&1; then
        firefox --no-remote --new-instance --profile "$PROFILE" --kiosk "$URL"
    elif command -v open >/dev/null 2>&1; then
        open "$URL" >/dev/null 2>&1 || true
        wait
    else
        echo "[start] web ui ready at $URL"
        wait
    fi
fi

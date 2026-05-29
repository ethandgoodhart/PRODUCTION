#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

EXPECTED_CAMERAS="${EXPECTED_CAMERAS:-6}"
USB_ROOT_PATH="${USB_ROOT_PATH:-1-4}"
UVC_QUIRKS="${UVC_QUIRKS:-0}"
WIDTH="${WIDTH:-320}"
HEIGHT="${HEIGHT:-240}"
FPS="${FPS:-5}"
START_RECORDER="${START_RECORDER:-1}"
NO_FOURCC="${NO_FOURCC:-0}"

log() {
  printf '\n== %s ==\n' "$*"
}

run_sudo() {
  if [ "$(id -u)" -eq 0 ]; then
    "$@"
  else
    sudo "$@"
  fi
}

camera_count() {
  lsusb | grep -c '32e4:0234' || true
}

log "Stopping camera applications"
pkill -f 'scripts/record_cameras.py' || true
pkill -f 'scripts/camera_view.py' || true
pkill -f 'scripts/alpamayo_infer.py' || true
pkill -f 'scripts/autoware_infer.py' || true
pkill -f 'scripts/segmentation_infer.py' || true
sleep 2

log "Killing processes still holding /dev/video*"
run_sudo fuser -k /dev/video* 2>/dev/null || true
sleep 2

log "Reloading uvcvideo with bandwidth quirk ${UVC_QUIRKS}"
run_sudo modprobe -r uvcvideo || true
sleep 2
run_sudo modprobe uvcvideo "quirks=${UVC_QUIRKS}"

log "Resetting USB path ${USB_ROOT_PATH}"
if [ ! -e "/sys/bus/usb/devices/${USB_ROOT_PATH}" ]; then
  echo "USB path ${USB_ROOT_PATH} not found under /sys/bus/usb/devices."
  echo "Current USB tree:"
  lsusb -t
  exit 1
fi

run_sudo bash -c "echo '${USB_ROOT_PATH}' > /sys/bus/usb/drivers/usb/unbind"
sleep 5
run_sudo bash -c "echo '${USB_ROOT_PATH}' > /sys/bus/usb/drivers/usb/bind"
sleep 8

log "Current camera inventory"
count="$(camera_count)"
echo "Camera count: ${count}/${EXPECTED_CAMERAS}"
lsusb | grep '32e4:0234' || true
echo
lsusb -t

if [ "$count" != "$EXPECTED_CAMERAS" ]; then
  log "Still missing cameras"
  echo "Expected ${EXPECTED_CAMERAS}, but USB currently sees ${count}."
  echo "Do not start the recorder yet. Move cameras across another USB root path or unplug/replug the missing hub ports."
  exit 2
fi

if [ "$START_RECORDER" != "1" ]; then
  log "All cameras visible"
  echo "START_RECORDER=${START_RECORDER}; leaving recorder stopped."
  exit 0
fi

log "Starting recorder"
args=(python3 scripts/record_cameras.py --width "$WIDTH" --height "$HEIGHT" --fps "$FPS")
if [ "$NO_FOURCC" = "1" ]; then
  args+=(--no-fourcc)
fi
printf 'Command:'
printf ' %q' "${args[@]}"
printf '\n'
exec "${args[@]}"

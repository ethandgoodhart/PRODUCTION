#!/usr/bin/env bash
# Shortcut: launch the cart in alpamayo mode (Modal-hosted Alpamayo-R1
# trajectory predictor as the steering brain, reached over a Modal
# forward tunnel — no WebSocket). All extra args are passed straight
# through to start.sh, so e.g. `./alpamayo.sh --no-loop` works.
set -e
cd "$(dirname "$0")"
exec ./start.sh \
    --model alpamayo \
    --app-name "${ALPAMAYO_APP_NAME:-alpamayo-live-demo}" \
    "$@"

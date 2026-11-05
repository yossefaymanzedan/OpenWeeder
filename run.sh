#!/usr/bin/env bash
# OpenWeeder launcher for the Raspberry Pi: starts the web app on port 5000.
set -e
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  echo "[OpenWeeder] No .venv yet: see docs/START_HERE_PI.md section 2"; exit 1
fi
source .venv/bin/activate
exec python tools/app.py --host 0.0.0.0 --port 5000

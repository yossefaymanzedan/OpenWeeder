# OpenWeeder

A low-cost precision weeding robot for small farms. A small car drives along rails
over a planting tray. A phone camera looks ahead, Meta **SAM 3.1** finds every plant
and **Muse Spark** judges crop vs. weed. A servo arm aims a nozzle at the weeds only.
It runs on a **Raspberry Pi 4**, controlled from a web app.

## What it does

- **Teach a crop:** the farmer uploads ~10 photos. The AI labels them, the farmer
  fixes mistakes, and the AI learns from those fixes (Results, Fix, Learning).
- **Collect a dataset:** the robot drives the tray and records photos from the phone.
  The tray is cropped automatically and the AI labels the photos.
- **Train YOLO** on the collected images, with live per-epoch metrics.
- **Run on the field:** the robot stops, takes a picture, finds weeds, aims the servo
  at each one and moves on. The run is saved as a video with the segmentation drawn on.
- **Safety:** it sprays only weeds at ≥ 80% confidence and keeps a safe distance
  from crops. It stops when the encoder shows the wheels are stuck.

## Hardware

Raspberry Pi 4, L298N with 4 DC motors, a single-channel wheel encoder, an SG90
servo arm, a pump (a dummy GPIO / LED for now), and an Android or iPhone as the
camera (over WebSocket on HTTPS). The pins are in [`pins.py`](pins.py) and
[`calib/hardware.json`](calib/hardware.json).

## Quick start (Raspberry Pi)

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements-pi.txt
cp .env.example .env          # put your META_API_KEY in .env (never commit it)
sudo systemctl enable --now pigpiod
./run.sh                      # open http://<pi-ip>:5000
```

Open the camera link or QR code from the Train page on your phone (same Wi-Fi).
For details, see [`START_HERE.md`](START_HERE.md) and [`docs/`](docs/).

## Layout

| Path | What |
|---|---|
| `tools/app.py` | Flask web app (port 5000) |
| `tools/hardware.py` | motors, encoder, servo, pump, watchdog |
| `tools/phone_cam.py` | phone camera server (HTTPS + WebSocket, port 8444) |
| `tools/planner.py`, `tools/geometry.py` | stop-and-go planning, arm aiming |
| `tools/runner.py` | field run: vision, spraying, videos |
| `tools/trainer.py` | dataset recording and YOLO training |
| `tools/tray.py` | finds the tray walls in each frame |
| `calib/` | pins, limits and measured robot dimensions |
| `templates/`, `static/` | web pages |
| `tests/` | `python -m pytest -q` (runs dry, no motors move) |

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

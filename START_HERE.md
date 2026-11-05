# START HERE: importing OpenWeeder on the Raspberry Pi 4

This bundle was built on the owner's Windows laptop to be unpacked **on the owner's own Raspberry Pi 4**.
It contains the whole platform, the molokhia dataset (200 images), the trained YOLO model, the crop's learned
data, and the API key (`.env`). Nothing here is to be shared outside the Pi.

## 0. If you are Claude Code on the Pi, do this in order

1. Read `docs/ROBOT_BRIEF.md` (what the platform is), then `docs/PRD_ROBOT_INTEGRATION.md` (what to build).
2. Do **section 1-4 below** (import and verify). Do not start on the robot code until the checks pass.
3. Then follow the PRD, in this order: R10 (config + pin map) -> R1/R2 (`PiHardware` with the dummy pump pin) ->
   R5/R11 (calibration) -> R12 (tests) -> R9 (run on the field) -> R8 (recording).
4. **Never print, log or commit `.env`.** Never connect anything to the pump GPIO except a bench LED.
5. Report honestly: misses, crop hits, anything that failed.

## 1. Unpack

```bash
cd ~
unzip openweeder_pi_bundle.zip -d openweeder          # or: tar / your file manager
cd openweeder
chmod 600 .env                                         # it holds the API key
rm ~/openweeder_pi_bundle.zip                          # optional: the zip also contains the key
```

The folder must be used **as the working directory** (paths in the code are relative to it).

## 2. Python environment

Raspberry Pi OS (Bookworm, 64-bit) ships Python 3.11, which is what the code targets.

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip python3-opencv python3-picamera2 python3-lgpio pigpio python3-pigpio ffmpeg
python3 -m venv --system-site-packages .venv           # system packages give you OpenCV and picamera2
source .venv/bin/activate
pip install -r requirements-pi.txt
```

- `requirements-pi.txt` leaves out `ultralytics`/`torch` (heavy). Install them **only** if you want YOLO inference on the Pi:
  `pip install ultralytics` (large; PyTorch CPU wheels for aarch64). Not needed for the SAM + Muse mode.
- Use pigpio (`sudo systemctl enable --now pigpiod`) or lgpio, whichever you prefer, for servo PWM and the encoder.

## 3. Check the import

```bash
source .venv/bin/activate
python -m pytest tests/test_planner.py tests/test_geometry.py tests/test_trainer.py tests/test_runner.py -q
```
Expected: all pass (about 50 tests). If a test fails, fix the environment first; do not edit the tests.

```bash
python - <<'PY'
from pathlib import Path
import sys; sys.path.insert(0, "tools")
import trainer, runner, setup
crop = "Molokhia"
i = trainer.info(crop)
print("images:", i["images"], "labeled:", i["labeled"], "model:", i["model"])
print("setup done:", setup.active_model(crop)[0] is not None)
print("yolo mAP50:", (i["metrics"] or {}).get("best", {}).get("map50"))
print("advice:", runner.options([crop])[0]["text"])
PY
```
Expected: `images: 200 labeled: 200 model: True`, `setup done: True`, `yolo mAP50` about 0.47, and the advice
"use SAM + Muse".

## 4. Start the web app

```bash
python tools/app.py            # listens on 0.0.0.0:5000
```
Open `http://<pi-ip>:5000/` from a phone or laptop on the same network and check:
- **Crops** shows **Molokhia**; **Train model -> Molokhia** shows 200 images, the metrics table and the test-set example images.
- **Run on the field** lists Molokhia with the advice "use SAM + Muse" and the measured numbers; Start says
  "No robot is connected yet" (that message is the blank robot function you will replace).

API check (costs about a cent): `python tools/sam_probe.py data/farmer_refs/ref_1.jpg` should return plant objects.
If it fails, check the Pi's internet and that `.env` has `META_API_KEY`.

## 5. What is in the bundle

```
START_HERE.md                this file (also docs/START_HERE_PI.md)
.env  .env.example           META_API_KEY (the real key is in .env)
docs/                        ROBOT_BRIEF.md, PRD_ROBOT_INTEGRATION.md
PROJECT.md                   findings, lessons, geometry + simulator documentation
tools/ templates/ static/    the platform (Python 3.11)
tests/                       unit tests (test_ui.py needs Playwright: skip it on the Pi)
calib/                       robot.json, servo.json  (PLACEHOLDERS: you measure and fill them)
fields/Molokhia/             the crop: farmer photos + labels, setup (checklist), the 200-image dataset
                             (train/images, train/labels), yolo/best.pt, metrics.json,
                             examples/, detector_compare.json
data/farmer_refs/            the 5 real photos used to generate the dataset
data/synthetic/*/manifest.json  how the 200 images were generated (prompts, seeds)
out/cache/                   cached SAM answers for these images (saves API cost)
yolo11n.pt                   the base YOLO weights (so training/inference works offline)
requirements*.txt  run.sh    install list and a launcher
```

Not included (rebuilt or unnecessary): `yolo/data` (a copy of the dataset made at training time), `yolo/run`
(training scratch), other crops, the simulator's videos, the Windows `.venv`, `run.bat`.

## 6. Prompt to give Claude Code on the Pi (copy-paste)

> You are on the owner's Raspberry Pi 4. The folder `~/openweeder` is the OpenWeeder platform. Read
> `docs/ROBOT_BRIEF.md` and `docs/PRD_ROBOT_INTEGRATION.md` completely. First follow `docs/START_HERE_PI.md`
> sections 1-4 to install and verify. Then implement the PRD in the order given in `START_HERE_PI.md` section 0,
> starting with `calib/hardware.json` (ask me for the pin map and the measured numbers), then `PiHardware`
> in `tools/hardware.py` with the pump as a **dummy GPIO output** (no real pump is connected to the Pi: it is
> switched by a push button outside the Pi). Keep `tools/planner.py` and `tools/geometry.py` behaviour unchanged;
> all existing tests must keep passing. Never print or commit `.env`. Report results honestly, including any crop hit.

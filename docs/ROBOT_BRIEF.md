# OpenWeeder / WeedBot: platform brief for the robot engineer

Read this first, then `PRD_ROBOT_INTEGRATION.md` (what you must build), then `START_HERE_PI.md`
(how to import and run everything on the Raspberry Pi 4).

You know the hardware and how to drive it from code. You do **not** know this software.
This document explains what the platform is, how it decides, what already works in simulation,
and exactly where the real robot plugs in.

---

## 1. What this is

**WeedBot** is a low-cost precision weeding robot for small farms. A small 4-wheel car drives
along rails over a row of plants. A camera looks straight down. The software finds every plant,
decides which are weeds, and sprays **only the weeds** with a small wiper arm that swings a
nozzle across the row. Crops are never sprayed.

**OpenWeeder** is the platform (a web app + Python tools) that:
1. teaches the system what one crop looks like (a farmer labels a few photos),
2. builds a dataset and trains a small offline detector (YOLO) for that crop,
3. runs the robot in the field using either that detector or the cloud AI.

Hackathon context: Meta Global AI Developer Hackathon (Indie track), deadline **4 Nov 9 PM ET**.
The robot is the demo. Safety first: spraying a crop is the worst possible mistake; leaving a weed
is a minor one. Every rule below follows from that.

**The bar we are aiming for:** find about 50% of the weeds and **never spray a crop**. It does not
have to be perfect.

## 2. The AI models (and what each is good for)

| Model | What it does | Where it runs | Speed / cost |
|---|---|---|---|
| **SAM 3.1** (Meta, API `sam-3.1`) | Given the prompt `"plant"`, outlines every plant (mask + box). It cannot tell species. | cloud API | ~2.4 s at 480 px, ~12 s at full size; $0.0025/image |
| **Muse Spark 1.3** (Meta, API `muse-spark-1.3`) | Looks at each outlined plant with the crop's saved checklist and calls it crop / weed / mixed / unsure with a confidence. | cloud API | ~18-30 s per picture (batches of up to 8 plants); ~$0.02/picture |
| **Muse Image 1.0** (`muse-image-1.0`) | Generates synthetic training photos from the farmer's reference photos. Used only to build datasets. | cloud API | $0.01/image |
| **YOLO11n** (Ultralytics) | Small offline detector trained per crop on our dataset. Meant to run on the robot with no internet. | trained on a laptop; runs on the Pi | trained; weak today (see section 6) |

Hard lessons already learned (do not repeat them):
- SAM **cannot** pick a species from a descriptive prompt. We tested: a crop-specific prompt returns
  0 objects or also outlines the weeds. It is always called with `"plant"`; Muse decides crop vs weed.
- Touching plants come back as **one** SAM object. Never spray a whole object blindly: spray a point
  on weed pixels kept 15 px away from crop pixels.
- When unsure, **protect**. Only a weed called with confidence >= 0.85 (adjustable per crop on the
  Results page) becomes a spray.

## 3. The physical model (coordinates and the wiper arm)

All positions are in **centimetres**, in the robot's frame:
- `x` = along the driving direction, **forward positive**.
- `y` = across the row, **left positive**.
- origin = the ground point under the arm's **pivot**, with the car at rest.
- Camera: straight down, ~25 cm above the ridge top, mounted `camera_to_nozzle_cm` (D, placeholder
  12 cm) **ahead** of the pivot. **The top of the image points the way the car drives.**
  Image up = +x, image left = +y.
- Two ground planes: `ridge` (crop on the ridge top) and `furrow` (8 cm lower, where weeds often
  grow). The photo scale differs between them, so the same pixel is a different place on each.

**The wiper arm.** One SG90 servo swings an arm of length `L` (placeholder 25 cm) that carries the
nozzle. At arm angle `phi` (0 = straight ahead, + = left) the nozzle is at
`(L cos phi, L sin phi)` from the pivot. To hit a weed `y` cm to the side:

```
phi          = asin(y / L)
stop at tick = weed_tick - L * cos(phi)      (the car must stop L cos(phi) BEFORE the weed)
```

Servo angle = `90 + trim + sign * phi` (`calib/servo.json`, key `arm`; `park_deg` 70 = arm parked
off to the side toward the furrow; `max_deg` 80). The old pan/tilt model remains only for the optional
laser mode.

> **Mechanical constraint you must know.** The nozzle can only reach points on an arc of radius `L`
> around the pivot, and that arc is **ahead** of the car. A weed that is already closer than `L`
> cannot be hit without backing up. With the camera only 12 cm ahead of the pivot, almost every weed in
> frame is "too close" until the car has passed it. That is why the tested plan for the first field runs
> is **two passes** (map the row, drive back, then stop-and-spray), and why a one-pass real-time mode
> needs the camera about `L + half a frame` (~40 cm) ahead of the pivot. Details in the PRD.

## 4. How the software decides (pipeline)

```
frame -> ExG green mask -> SAM("plant") masks/boxes -> shape + colour features
      -> Muse Spark judge (crop checklist learned for THIS crop)  -> label + confidence
      -> safety rule (spray only a sure weed; mixed/unsure = protect)
      -> spray point on weed pixels, 15 px clear of crop pixels
      -> geometry.py: pixel -> cm in robot coordinates -> arm angle + stop tick
      -> planner.py: stop, settle 0.3 s, swing the arm, pump 150 ms, drip 0.1 s, park the arm
```

Two ways to detect (the Run-on-the-field page chooses):
1. **YOLO model** (offline, fast): detects crop / weed / unsure boxes. Needs a trained model.
2. **SAM + Muse** (cloud, slow, more accurate today): the pipeline above on every picture.

A rule that holds for both: **crops and "unsure" plants are never sprayed.** After detection a veto
removes any spray whose centre is inside a detected crop/unsure box (grown by a margin), and the
planner re-checks the real landing point (encoder + arm angle) against every protected zone before
each spray, logging `skipped: near crop`.

## 5. The platform (web app, `python tools/app.py`, port 5000)

Left sidebar pages:
- **Home / Crops**: list of crop workspaces and their learning status.
- **New crop** (`/crop/new`): define a crop (name, crop type, planting style: single row / double /
  scattered). Then a guided flow: photo guide -> upload 10 photos -> label 3 (tap each plant: crop /
  weed) -> the AI labels the other 7 -> the farmer fixes mistakes, adds notes, re-runs -> confirm.
  This produces the **checklist** (what the crop and its weeds look like) used by Muse.
- **Train new crop** (`/crop/train`): placeholder for the future "robot drives three columns and gathers
  data". Disabled. Also lists crops that already have a dataset with a Continue link.
- **Train model** (`/train/<crop>`): dataset (robot recording = **blank**, or import images) ->
  **Auto label** (SAM + Muse label every image, costs ~$0.02/image) -> **Train YOLO** (<= 30 epochs,
  live metrics, test-set example images). Saves `fields/<crop>/yolo/best.pt`.
- **Fields / New field** (`/fields`, `/fields/new`): a field = the real ground; it picks which finished
  crop's model to use. Stored in `farm/<name>.json`.
- **Run on the field** (`/run-field`): pick a crop and a detector. If the model's mAP50 is below 0.70 it
  advises SAM + Muse. **Start/Stop call a blank robot function** (you fill it). Shows measured accuracy,
  a step-size / row-length estimate (time and cost) and will list recorded videos.

Data on disk (everything is plain files, relative to the repo root; run the app from the repo root):

```
fields/<crop>/field.json           crop name, planting style, which photos are labeled
fields/<crop>/model.json           saved field model (after Confirm)
fields/<crop>/setup_log.json       setup: checklist versions, scores
fields/<crop>/photos/              the farmer's photos
fields/<crop>/labels/              the farmer's labels (masks + boxes)  <- real ground truth
fields/<crop>/runs/                test runs of the pipeline
fields/<crop>/train/images/        the training dataset
fields/<crop>/train/labels/        YOLO labels (0 crop, 1 weed, 2 unsure)
fields/<crop>/yolo/best.pt         the trained model;  metrics.json = results
fields/<crop>/detector_compare.json  measured SAM+Muse vs prompt-only on real photos
fields/<crop>/field_runs/*.mp4     (future) videos from field runs
calib/robot.json, calib/servo.json  robot measurements (PLACEHOLDERS until you measure them)
out/cache/                         cached SAM answers (do not re-pay for the same image)
.env                               META_API_KEY (the Meta API key)
```

## 6. Where things stand (be honest about this)

**Works, tested in simulation:** the vision pipeline, geometry (pixel -> cm -> servo angle -> encoder
tick, validated against OpenCV), the planner (two-pass map-then-spray, stop-and-go, continuous, and a
one-pass real-time mode), the simulator and its scoring, the web app, dataset generator, auto labeling,
YOLO training.

**Simulator results** (field `tomato-3`, 73 cm strip, placeholder calibration, cached vision):
24/40 weeds sprayed, 0 crop hits, 7 ml vs 101 ml for broadcasting; one-pass real-time mode 25/40 in 44.7 s.
These use placeholder robot numbers and the vision was taught from the same 3 photos, so treat them as
optimistic.

**Crop "Molokhia"** (the real crop you will see; photos of molokhia in a tray, scattered plants with tall
grass as the weed):
- YOLO (200 synthetic images, 960 px, 30 epochs): crop AP 0.85, but **weed recall is only ~7-12%**, and
  mAP50 ~0.45. Crops were never sprayed (0 crop hits) at the safe setting. **Not good enough to spray
  with.**
- SAM + Muse on 3 real labeled photos: **45-61% of weeds found, 0-1 crop hits over three runs**
  (Muse is not perfectly repeatable), **~33-35 s per picture**, ~2 cents each. This is the
  recommended detector for the first field tests.
- SAM with a crop-description prompt ("prompt only"): returned **no crops**, sprayed everything,
  10 crop hits. Rejected.

## 7. What is missing (your job)

Blank placeholders in the code, each marked `BLANK` / `STUB`:
1. `tools/hardware.py` -> `class PiHardware` (every method raises `NotImplementedError`). **The big one.**
2. `tools/trainer.py` -> `record_images()` (record 2 images per second while driving).
3. `tools/runner.py` -> `run_on_field()` (drive, detect, spray, record video).
4. The pump: the real pump is switched by a **push button outside the Pi**; in software it is a **dummy
   GPIO output** (see PRD section 5.2).

Also required: measure the real numbers into `calib/robot.json` / `calib/servo.json` (they are all
placeholders), and make the planner call detection **asynchronously** (see PRD R9, it matters).

## 8. Code map

| File | Role |
|---|---|
| `tools/hardware.py` | `Hardware` interface; `FakeHardware` (the simulator's robot); **`PiHardware` stub** |
| `tools/planner.py` | The brain. Talks to the world **only** through `Hardware` + a vision backend. |
| `tools/geometry.py` | Pixels <-> cm <-> servo angle <-> encoder ticks; wiper-arm maths; A4 calibration |
| `tools/vision.py` | The AI pipeline (ExG + SAM + Muse + safety rule) |
| `tools/sam_client.py` | SAM 3.1 streaming client |
| `tools/sim.py` | Simulator: stitches photos into a strip, runs the planner on `FakeHardware`, scores it, writes a video |
| `tools/setup.py` | The per-crop learning (checklist) |
| `tools/trainer.py` | Dataset, auto label, YOLO training, the spray rule, scoring |
| `tools/runner.py` | Run-on-the-field options/advice; **blank robot function** |
| `tools/app.py` + `templates/` | The web app |
| `tools/compare_detectors.py`, `sam_video_latency.py` | Measurements (accuracy, latency) |
| `tools/measure_ticks.py`, `calibrate_a4.py`, `check_points.py` | Calibration tools to run on the robot |
| `tests/` | `test_planner.py`, `test_geometry.py`, `test_trainer.py`, `test_runner.py`, `test_ui.py` |
| `PROJECT.md` | Findings, lessons, geometry and simulator documentation (long, worth reading) |

## 9. Commands you will use

```bash
python tools/app.py                                   # the web app on http://<pi-ip>:5000
python -m pytest tests/test_planner.py tests/test_geometry.py tests/test_trainer.py tests/test_runner.py -q
python tools/sim.py --field Molokhia                  # run the planner on a simulated robot
python tools/measure_ticks.py --ticks 96              # after driving exactly 100 cm
python tools/calibrate_a4.py photo.jpg --plane ridge  # camera calibration from an A4 sheet
python tools/compare_detectors.py Molokhia            # SAM+Muse accuracy on real photos (costs ~$0.07)
```

(`tests/test_ui.py` needs Playwright and a browser; it is slow and not needed on the Pi.)

## 10. Rules of the house

- Python 3.11. Everything that touches hardware goes through `Hardware`. The simulator and the real
  robot run the **same** `planner.py`.
- Never call the API in a tight loop. Cache. Every API call is logged with its cost (`out/api_log.jsonl`).
- Do not change vision rules or the geometry behaviour to make a number look better. Report misses and
  crop hits honestly (`CROP_HITS` must be 0; if it is not, find the cause, do not hide it).
- Generated/synthetic images and AI labels are **not** ground truth. The only real truth is the farmer's
  own labels on real photos (`fields/<crop>/labels/`). Do not quote synthetic numbers as results.
- Safety is a hard requirement, not a feature (PRD section 6).

## 11. Glossary

- **Crop / weed / unsure**: the three classes. Unsure = the AI could not tell; treated as crop (never sprayed).
- **Checklist**: the short text describing how to tell this crop from its weeds, learned during setup.
- **mAP50**: a detector accuracy score from 0 to 1 (0.7+ is good). Not the same as "weeds found".
- **Weeds found / crop hits**: what we really care about. Crop hit = a spray landing on a crop.
- **Tick**: one encoder count (~1 cm). `cm_per_tick` converts.
- **Ridge / furrow**: the raised crop bed and the lower ground beside it.
- **Two-pass**: drive forward mapping weeds, drive back, then stop-and-spray each one.

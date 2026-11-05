# PRD: connecting OpenWeeder to the real robot (Raspberry Pi 4)

| | |
|---|---|
| Owner (platform) | Yossef (OpenWeeder) |
| Implementer | the robot engineer / Claude Code running on the Pi |
| Status | draft v1, 2026-11-04 |
| Deadline context | Meta Global AI Developer Hackathon, Indie track, 4 Nov 9 PM ET |
| Read first | `docs/ROBOT_BRIEF.md` |

---

## 1. Goal

Make the **same planner that already runs in simulation** drive the **real robot** along a row of plants,
record data, detect weeds and trigger the spray, **without changing the planner's logic**. The
only seam between simulation and reality is the `Hardware` interface (`tools/hardware.py`).

**Success = a demo run in which the robot drives a short row (about 1 m) over real molokhia plants, finds
roughly half the weeds, never sprays a crop, and writes an event log and a video.** It need not be perfect.

### Non-goals (this phase)
- Spraying real liquid under software control. The pump is switched by a **push button outside the Pi**; in
  software the pump is a **dummy GPIO output** (section 5.2).
- Training models on the Pi. Training stays on the laptop (it is heavy and CPU-bound).
- Laser mode (optional, section 5.7).
- Farm-scale speed. SAM + Muse is ~35 s per picture; a long row takes hours. This is a demo-scale system.
- Weather/water-proofing, multi-row navigation, GPS.

## 2. What already exists (do not rebuild)

- `Hardware` abstract class + `FakeHardware` (simulated robot) + a `PiHardware` **stub** in `tools/hardware.py`.
- `tools/planner.py`: Pass 1 (map) / Pass 2 (stop-and-spray), `--continuous`, and a one-pass real-time mode
  (`run_realtime`). It uses **only** `Hardware` + a vision backend.
- `tools/geometry.py` (pixel -> cm -> servo angle -> tick, wiper-arm maths), `tools/sim.py` (simulator + scoring
  + video), `tools/vision.py` (SAM + Muse pipeline), `tools/trainer.py`, `tools/runner.py`, the web app.
- ~110 tests (planner, geometry, trainer, runner, UI). **The planner/geometry tests must keep passing.**

## 3. Hardware (confirm and complete in the pin map, section 5.10)

From `PROJECT.md`, to be confirmed by you (you know the real build):
- Raspberry **Pi 4** (the only controller), **Pi Camera Module 3**.
- 4WD TT-motor car on an **L298N**, powered by 3x18650.
- LM2596 buck to **5 V** for the SG90 servos (and the pump supply).
- 2x SG90 servos: **arm servo** (wiper arm; this phase) and a second servo for the optional laser tilt.
- **Wheel encoder**: ~1 cm per tick (nominal, to be measured).
- **Red button** = hard cut of pump/laser power (hardware, not software).
- **Green button** = mode select (optional).
- **Pump**: switched by a push button outside the Pi (see 5.2).

## 4. Architecture: where code runs

```
Raspberry Pi 4 (on the robot)                          Cloud (Meta API)            Laptop
 planner.py  --Hardware-->  PiHardware  --> GPIO/PWM    SAM 3.1  (segmentation)     trains YOLO
 vision backend  --HTTPS-->  ......................->    Muse Spark 1.3 (judge)      (fields/<crop>/yolo/best.pt
 web app (Flask :5000)  <-- phone / laptop browser                                    is copied to the Pi)
 camera, encoder, servo, dummy pump pin
```

Decision for you (recommended default in bold): **run the app, planner and PiHardware on the Pi.**
It needs only Python, OpenCV, Flask and an internet connection (for SAM + Muse). YOLO inference on
the Pi is optional (section 5.9). Keep training on the laptop and copy `best.pt` over.

## 5. Requirements

Priorities: **P0** = needed for the first demo, **P1** = needed for a reliable demo, **P2** = nice to have.

### 5.1 R1 (P0) `PiHardware` implements `Hardware`

File: `tools/hardware.py` (replace the stub). Constructor suggestion:
`PiHardware(config="calib/hardware.json", dry_run=False)`. With `dry_run=True` it must run the **same code
path with no motion** (motors off, simulated ticks) so the planner can be exercised on a bench.

| Method | Contract |
|---|---|
| `drive(speed_cm_s)` | **Non-blocking.** Start driving; `+` forward, `-` backward, `0` = stop. Hold the speed with encoder feedback (at least a simple P/PI loop) to within about +/-15% at 3 and 5 cm/s. Clamp to `max_speed_cm_s` from the config. Ramp, do not slam. |
| `stop()` | Stop driving (actively brake the L298N). Return immediately. The planner assumes the car **coasts ~0.1 s** after `stop()` and compensates by stopping `speed * 0.1 s` early. **Measure your real coast distance at 3 and 5 cm/s** and report it (it sets `COAST_S`). |
| `ticks() -> int` | Cumulative encoder count since power-on. **Forward driving increases it, backward decreases it.** Thread-safe, readable at >= 50 Hz. `cm_per_tick` (calib) converts to cm. |
| `capture() -> np.ndarray` | One camera frame, **RGB `uint8`, shape (972, 1296, 3)** (matches `calib/robot.json`: `image_width_px` 1296, `image_height_px` 972), **taken now**, returns in <= ~0.15 s. Image **top = forward**, image left = +y (left). Fixed exposure and a **fixed manual focus for 25 cm** (Module 3 has autofocus: turn it off). Short exposure (motion blur at 3-5 cm/s). |
| `servo_set(angle_deg)` | **Non-blocking.** Command the wiper-arm servo (0-180, already mapped by `geometry.arm_to_servo`: `90 + trim + sign*phi`). The planner itself waits the travel time (0.1 s per 60 degrees), so just hold a **jitter-free** PWM (use hardware/DMA PWM, e.g. pigpio, not software PWM). Refuse angles outside the configured safe range. |
| `pump(ms)` | **Blocks for `ms` milliseconds**, then returns. See 5.2 (dummy GPIO). Reject `ms` above `pump_max_ms`. |
| `laser(on)` | GPIO output (optional, 5.7). |
| `now() -> float` | Seconds on a monotonic clock (`time.monotonic()`). |
| `sleep(s)` | `time.sleep(s)`. The car keeps moving meanwhile if it is driving. |

**Event log parity.** Like `FakeHardware.events`, keep a list of `{t, type, value, tick, arm}` for every
action (`start, drive, stop, capture, servo, pump, laser`) and dump it to `out/run_<stamp>_events.json`
at the end of a run. This lets us compare a real run with the simulated one line by line.

### 5.2 R2 (P0) The pump is a **dummy GPIO output**

The real pump is wired to a **push button outside the Pi** (an operator presses it). The Pi does **not**
switch the pump in this phase. We still need the software path, so:

- Configure one GPIO pin as an **output** (`pump_gpio_bcm` in `calib/hardware.json`; pick any free pin,
  default BCM 17) and leave **nothing wired** to it (an LED + resistor on a breadboard is fine for bench
  checks).
- `pump(ms)`: set the pin HIGH, `sleep(ms/1000)`, set it LOW, log the event `{type:"pump", value:ms}`
  with the true tick and arm angle. Guarantee the pin ends LOW even if the call is interrupted (`try/finally`).
- The simulator and the scoring treat this pump event as a spray. So the run log shows exactly where and
  when the robot **would** have sprayed, and the operator can press the real button at those moments
  during tests if they want to wet the weed.
- **Never** connect a pump, valve or motor directly to a Pi GPIO pin. If a real software-switched pump
  is added later it needs a MOSFET/relay driver and the red-button hard cut.
- Cap `ms` (`pump_max_ms`, default 500) and refuse to fire unless the robot is `armed` (5.6).

### 5.3 R3 (P0) The wiper arm servo

- Measure and record in `calib/servo.json` (`arm` block): `center_trim_deg` (servo angle at which the arm
  points **straight ahead** is `90 + trim`), `sign` (flip to +1/-1 so that **+phi swings left**),
  `max_deg` (largest safe swing, default 80) and `park_deg` (arm angle where it is parked off to the side
  toward the furrow, default 70).
- Measure the arm length `L` (pivot to nozzle) and write `arm_length_cm`; measure the wet radius on the ground
  (`spray_radius_cm`) and the pump flow (`pump_flow_ml_per_s`) when a real pump is used.
- Park the arm on `start()` and on any stop/fault.

### 5.4 R4 (P0) Camera

- Use the Pi Camera Module 3 through `picamera2`. Output 1296x972 RGB. Mounted **straight down** about 25 cm
  above the ridge top, `camera_to_nozzle_cm` (D) ahead of the arm pivot (measure it, placeholder 12 cm).
- Lock exposure, gain, white balance and focus for the field lighting; document the settings.
- Optionally save the raw frames of a run to `out/run_<stamp>/frames/` (needed for R8 anyway).

### 5.5 R5 (P0) Motion and encoder

- The planner drives at 3 cm/s by default (up to 5). Acceptance: drive exactly 100 cm on the rails, read
  `ticks()`, run `python tools/measure_ticks.py --ticks <n>`. Repeat three times; the counts must agree within 2%.
- Report: coast distance after `stop()` at 3 and 5 cm/s; the minimum reliable speed; whether the car goes
  straight (drift over 100 cm).

### 5.6 R6 (P0) Safety (hard requirements)

1. The **red button** cuts pump/laser power in hardware. Software must not depend on it, but must never
   fight it.
2. `PiHardware` starts **disarmed**. Spraying (the dummy pump pin) and arm motion only work after an explicit
   `arm()` / "armed" flag set by the operator (a confirmation on the web page or a key press).
3. **Watchdog**: if the planner stops calling the interface (crash, exception) for more than 2 s, the motors stop,
   the pump pin goes LOW and the arm parks. Any uncaught exception in a run does the same.
4. **Never spray a crop.** The software already vetoes; you add nothing here except: do not bypass the planner's
   protected-zone check, and never invent an extra spray path.
5. Hard limits in the config: `max_speed_cm_s`, `pump_max_ms`, servo angle limits. Values outside raise an error.
6. The Pi must be able to **abort a run** (Stop button on the web page, Ctrl-C) and leave the hardware safe.

### 5.7 R7 (P2) Laser (optional)

`laser(on)` toggles a GPIO output (dummy pin as for the pump unless a MOSFET driver exists). Tilt servo is only for
laser mode; ignore it for the spray demo.

### 5.8 R8 (P1) Dataset recording: `trainer.record_images()`

File `tools/trainer.py`, function `record_images(crop, stop, fps=2, root=Path("fields"))` (currently raises
`NotImplementedError`). Drive the robot along the column and save **2 camera frames per second** as
`fields/<crop>/train/images/img_<n>.jpg` (continue numbering after the existing images; use
`import_images`' naming) until `stop` (a `threading.Event`) is set; return the number saved. JPEG quality ~92.
Also write the encoder tick next to each frame (`fields/<crop>/train/ticks.json`) for later use.

**App wiring (known gap):** `tools/app.py` route `/api/train/<crop>/collect/<action>` currently calls this function
**synchronously** inside the request. A real implementation blocks until stopped, so the route must start it in a
background thread, keep its status (`on`, frames saved) in `TRAIN[crop]["collect"]`, and return immediately
(the page already polls `/api/train/<crop>`). Stop sets the event.

### 5.9 R9 (P0/P1) Run on the field: `runner.run_on_field()`

File `tools/runner.py`, function `run_on_field(model_path, stop, video_path, mode, step_cm)` (blank today).
Two modes:

- **`sam_muse`** (recommended first; it works today): drive, take pictures, run the pipeline
  (`vision.analyze(photo, cache_dir, log_path, refs, field_model=..., spray_conf=...)`) on each, and let
  `planner.py` decide where to spray. Load the crop's field model with `setup.active_model(crop)` and
  `setup.ref_list(...)`; see `tools/sim.py` `LiveVision` for the exact calls. ~35 s per picture.
- **`yolo`**: run `ultralytics.YOLO(model_path).predict(frame, conf=0.03, imgsz=960)` on each frame, then
  apply `trainer.sprays(pred, conf_w, margin, crop_conf)` with the **operating point saved in
  `fields/<crop>/yolo/metrics.json`** (`operating_point`). On a Pi 4 this is likely 1-3 s per frame (**unmeasured:
  please measure it**); consider exporting to NCNN/ONNX. The current Molokhia model is weak (section 7), so
  treat this mode as a pipeline test, not a spraying mode.

**Both modes must:**
1. Record every frame with its detections drawn (crop green, weed red, unsure grey) into an **mp4** at `video_path`
   (`fields/<crop>/field_runs/<stamp>.mp4`, via `runner.new_video_path`). OpenCV `mp4v` is fine; H.264 via ffmpeg optional.
2. Write the events log (5.1) and a JSON summary `{frames, sprays, skipped, seconds, usd}` next to the video.
3. Stop cleanly on the `stop` event and leave the hardware safe.

**Planner integration (important).** `Planner.map_pass` / `run_realtime` call `self.vision.detect(img, tick)`
**synchronously**; in simulation the 10-35 s latency is only *accounted for* (`ready_s`), not waited. On the
real robot a 35 s blocking call inside the drive loop would freeze the control loop while the car keeps rolling.
Required change (planner stays testable): run `detect` in a **worker thread pool** (`vision_workers`, default 3),
let the loop keep polling `ticks()`, and collect each frame's result when it is actually ready. Add a test with a
backend whose `detect` really sleeps, proving the planner keeps reading ticks and does not overshoot the stop tick.

**Mechanical constraint (see Brief section 3).** The arm reaches only the arc of radius `L` ahead of the pivot.
- With the camera D = 12 cm ahead (current), use the **two-pass** flow that is already implemented and tested:
  `Planner.run()`: Pass 1 drive forward taking a picture every `frame_length * 0.8` (~19 cm), wait for the
  results, **drive back**, Pass 2 stop-and-spray each weed. Set `vision_s` to the real latency.
- A **one-pass** run (`Planner.run_realtime`) needs the camera about `L + half a frame` (~40 cm) ahead of the
  pivot (`tools/sim.py --realtime --camera-cm 40` demonstrates it). Choose one and **write the chosen D into
  `calib/robot.json`**.

**App wiring (known gap):** `/api/field_run/<action>/<crop>` currently calls `run_on_field` synchronously. As with
R8, run it in a background thread, publish status (`running`, frames, sprays so far, last event) and expose it so
the page can show it; Stop sets the event.

### 5.10 R10 (P0) Config and pin map

Create `calib/hardware.json` (and a documented template) with **every pin and limit in one place**. Fill the TBDs:

| Function | Pin / value | Notes |
|---|---|---|
| Motor driver | L298N ENA/ENB/IN1-IN4 = **TBD** | PWM frequency, brake mode |
| Encoder | channel A/B (or single) = **TBD** | ticks per revolution = **TBD**; wheel diameter = **TBD** |
| Arm servo (SG90) | signal pin = **TBD** | pulse range 500-2400 us, hardware PWM |
| Tilt servo (laser, optional) | **TBD** | |
| Pump (dummy) | `pump_gpio_bcm` = **17 (default)** | nothing wired; LED for bench |
| Laser (optional) | **TBD** | |
| Green button / red button sense | **TBD** | red is a hardware cut; sensing optional |
| Camera | CSI ribbon | 1296x972 RGB |
| Limits | `max_speed_cm_s` 8, `pump_max_ms` 500, servo safe range [10, 170], `watchdog_s` 2 | |

### 5.11 R11 (P0) Calibration deliverables

Replace the placeholders in `calib/robot.json` / `calib/servo.json` with measured values and add each to the
`measured` map (instructions in `PROJECT.md`, section "Geometry and calibration"):

1. `camera_height_cm`, `ridge_height_cm`, `camera_to_nozzle_cm` (D), `camera_side_offset_cm`, `nozzle_height_cm`.
2. `cm_per_tick` via `tools/measure_ticks.py`.
3. A4 homography for the ridge plane (and the furrow plane): `python tools/calibrate_a4.py photo.jpg --plane ridge`
   (A4 flat on the ridge, long side across the rails, whole sheet in view, photo taken at 1296x972). Then verify with
   `tools/check_points.py` (target about 1 cm average error).
4. Arm: `arm_length_cm`, servo trim/sign/max/park (5.3); `spray_radius_cm`.
5. Timing: real coast, settle, servo travel, capture time (update the constants at the top of `tools/hardware.py`).

### 5.12 R12 (P1) Tests you must add

- `tests/test_pi_hardware.py`: with `dry_run=True`, the interface contract (ticks sign, pump pin ends LOW,
  caps/refusals, watchdog, event log).
- The planner's async-vision test (5.9).
- A **sim-vs-dry-run equivalence test**: the same planner run against `FakeHardware` and `PiHardware(dry_run=True)`
  produces the same sequence of event types and the same stop ticks (within 1 tick).

## 6. Test plan and acceptance criteria

| Phase | What | Pass when |
|---|---|---|
| **0 Bench (no motion)** | Wheels off the ground. Encoder counts by hand, servo sweep, dummy pump pin toggles an LED, camera frame saved | `ticks()` direction correct; servo angles match `phi` within 2 degrees; `pump(150)` gives a 150 +/-10 ms pulse and ends LOW; frame is 1296x972 and sharp at 25 cm |
| **1 Dry run** | `PiHardware(dry_run=True)` and then real motion on the rails with the arm parked, planner `Planner.run()` with a stub vision that returns canned weeds | event log has the right stop ticks; the car stops within +/-1 cm of each planned stop; the watchdog works (kill the planner: car stops, pin LOW) |
| **2 Calibration** | R11 | `measure_ticks` repeatable within 2%; A4 check about 1 cm; arm hits a marked point on the ground within 1.5 cm (the spray radius) |
| **3 Data recording** | `record_images` along a column | about 2 images/s saved, numbered, not blurred; stop works |
| **4 Field demo (SAM + Muse, two-pass)** | 1 m of real molokhia in the tray/row, dummy pump | run completes; video and event log written; **no spray event lands on a crop** (check against the video); roughly half the visible weeds get a spray event |
| **5 YOLO mode** | same with the saved model | pipeline works end to end; accuracy is reported honestly, not assumed |

Reporting format for every run: weeds visible, weeds sprayed, crop hits (from the video), time, API cost.
If `crop hits > 0`, stop and find the cause. Do not hide it.

## 7. Current model status (what to expect)

- **SAM + Muse** (Molokhia, 3 real labeled photos, three runs): 45-61% of weeds found, 0-1 crop hits, ~33-35 s per picture,
  ~$0.02 each. The checklist was learned from those same photos, so this is an upper bound.
- **YOLO** (Molokhia, 200 synthetic images): crop AP 0.85; **weeds found 7-12%** at the safe setting, 0 crop hits.
  Not usable for spraying until the weed labels and dataset improve (planned on the laptop).
- `tools/sam_video_latency.py` (measured): SAM at 480 px ~2.4 s, at full size ~12 s; 4 frames at once ~0.85 fps. The
  Muse judge is the bottleneck. **Send SAM smaller frames if you need speed**: plant count barely changes (11 vs 12).

## 8. Constraints and assumptions

- Python 3.11, Linux (Raspberry Pi OS). No GPU. Internet needed for SAM + Muse (Wi-Fi or phone hotspot).
- `.env` holds `META_API_KEY`. Treat it as a secret (never commit, never print).
- Run from the repo root (paths are relative). The app listens on `0.0.0.0:5000`; open it from a phone on the same network.
- Disk: models and the dataset are ~300 MB. Videos can be large; clean `fields/*/field_runs` between demos.

## 9. Open questions for you (please answer in `docs/ROBOT_ANSWERS.md`)

1. Exact hardware: motor driver wiring, encoder type/resolution, wheel diameter, battery, servo mounting and horn zero.
2. Where is the camera mounted relative to the arm pivot (D) and how high? Can it be moved to ~40 cm ahead?
3. Real arm length `L`, and the real maximum safe arm angle.
4. Real measured coast distance and drift.
5. Which GPIO/PWM library do you prefer (pigpio daemon, lgpio, gpiozero)? Any conflict with the camera?
6. Network on the field: Wi-Fi, phone hotspot? Is the API reachable from the robot?
7. How will the operator arm/disarm the robot (web button, physical key)?
8. YOLO on the Pi: measured seconds per frame at 960 px, and is ONNX/NCNN export worth it?

## 10. Interface reference (copy of the contract)

```python
class Hardware(abc.ABC):
    def drive(self, speed_cm_s: float) -> None      # non-blocking, + forward, - back
    def stop(self) -> None
    def ticks(self) -> int                          # forward +, signed cumulative
    def capture(self) -> np.ndarray                 # RGB uint8 (972, 1296, 3), top = forward
    def servo_set(self, angle_deg: float) -> None   # non-blocking; 90 + trim + sign*phi
    def pump(self, ms: float) -> None               # BLOCKS ms; dummy GPIO output now
    def laser(self, on: bool) -> None
    def now(self) -> float                          # monotonic seconds
    def sleep(self, s: float) -> None
```

Timing constants the planner uses (top of `tools/hardware.py`, all guesses until you measure): settle 0.3 s, spray 0.15 s,
drip 0.1 s, servo 0.1 s per 60 degrees, capture 0.1 s, coast 0.1 s, vision 10 s (use ~35 s for SAM + Muse).
Planner defaults (`planner.Config`): speed 3 cm/s, frame overlap 20%, merge radius 2 cm, 3 vision workers.

Coordinates: x forward, y left, origin under the arm pivot; image top = forward. Geometry helpers:
`geometry.wiper_arm(y)`, `arm_to_servo(phi)`, `servo_to_arm(angle)`, `arm_landing(phi)`, `locate(u, v, model, size, row_y)`,
`pixel_to_ground`, `plan_target`.

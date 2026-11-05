# WeedBot: project brief (read this first every session)

Low-cost precision weeding robot for smallholder farms. A small car drives along
rails over a row of potted plants. A camera looks ~25 cm ahead. AI finds the weeds
with zero training. A pan-tilt nozzle sprays only the weeds (mode 1) or a laser
pointer tracks them (mode 2, demo of laser weeding).

Hackathon: Meta Global AI Developer Hackathon, Indie track. Deadline Nov 4 9 PM ET.
Judging: Impact 30%, Use of Meta AI tooling 25%, Demo 25%, Technical depth 20%.

## Meta models (OpenAI-compatible, base_url https://api.meta.ai/v1, key in META_API_KEY)
- sam-3.1 (Responses API): text prompt + image -> objects as tokens like
  `0<|box;x1=..;y1=..;x2=..;y2=..;w=..;h=..|><|mask;...|>`. Masks decode with meta-sam-parser.
- muse-spark-1.3: multimodal judge for uncertain plants, field calibration, farmer report.

## Vision pipeline
1. ExG green mask (2G-R-B > threshold) = every plant pixel, no API.
2. SAM 3.1 with prompt "plant" finds per-plant objects (masks + boxes).
3. Shape features per object (size, position, row line fit on the same image).
4. Muse Spark judges: field notes first (crop/weed/mixed per plant + reasons),
   then final labels.
5. Safety rule: spray only label == weed at confidence >= 0.85; mixed, unsure,
   or missing = protect the crop.
6. Spray point: sprayed weed objects -> queued encoder-tick commands with
   pan/tilt angles.

## Findings (night one, measured)
- SAM 3.1 prompt "plant" gives clean per-plant masks. Crop-specific prompts
  (9 tested) either return nothing or include weeds (86-96% leak). SAM finds
  plants; it does not judge species.
- Touching plants come back as one SAM object. Never spray a whole object blindly.
- ExG green filter: ~97% of plant pixels found on CWFID, threshold ~60, no API.
- Muse Spark judges crop vs weed. With spray only at weed >= 0.85 and
  mixed/unsure = protect: 0 crop hits, 2/5 weeds caught on CWFID.
- Hints (reference crop, row line) must come from the SAME image/field.
  Transferred from another image they made results worse.
- SAM latency 5-6.5 s/call, $0.0025/call. Plan car speed ~3 cm/s with frames
  in parallel.
- CWFID 035 has a doubtful label (carrot-like plant labeled weed); excluded
  in reports.

## Lessons (what failed and why)
Read these before building; each rule exists because the opposite failed.

1. SAM cannot judge species.
   Failed: 9 crop prompts ("carrot", "feathery leaves", "plant with thin divided
   leaves"...) returned either 0 objects or masks that also covered 86-96% of the
   weeds. A box around one carrot as a visual example returned only that carrot;
   a pasted reference patch returned only the patch.
   Rule: SAM is only called with the prompt "plant". Muse Spark decides crop vs weed.
2. Touching plants come back as one SAM object.
   Failed: a carrot and a weed tangled together were one mask; spraying the
   object would hit the carrot.
   Rule: never spray a whole object. Spray points come from weed green pixels
   with a 15 px safety margin from crop/protected pixels. Mixed = protect.
3. Hints from a different image make things worse.
   Failed: a row line fitted on one photo and drawn on another landed on bare
   soil; Spark then called two real carrots "weed, off row" at 0.85. Reference
   crops from another photo (different growth stage) added a crop hit.
   Rule: every hint (row, reference close-ups) comes from THIS field's own setup photos.
4. Spark over-trusts the row when it is given as text.
   Failed: with the row in the prompt, crop hits rose to 5-6, all "off row"
   reasoning on tangled plants.
   Rule: the row is used only by geometry, conservatively: weed by geometry only
   if far outside the crop band AND touching no other plant. Spark's prompt gives
   measured facts but never says "off row means weed". Geometry can only confirm
   a weed Spark already calls weed; it never sprays a plant Spark calls crop,
   mixed or unsure. Every geometry/Spark disagreement is logged.
5. Re-thinking the field on every frame hurt.
   Failed: per-frame "field notes" dropped accuracy from 32% to 23% and brought
   crop hits back.
   Rule: understanding happens once, in setup. Runtime uses the saved checklist.
6. Long answers got cut off.
   Failed: with 12+ plants per call, Spark's JSON was truncated; 89 of 195 plants
   were never judged.
   Rule: batches of max 8 plants per call, minimal reasoning + JSON schema
   output, retry once, and if it still fails every plant in that batch = protect.
7. Inputs must be clean.
   Failed: extra green blobs duplicated SAM objects; tiny close-ups were mostly
   the number tag.
   Rule: merge blobs overlapping a SAM object by IoU > 0.6; upscale close-ups to
   at least 64 px on the short side.
8. Look-alike weeds fool any model.
   Failed: a feathery weed was called carrot at 0.95-0.98 by both Muse Spark and Gemini.
   Rule: the farmer's labels and reference close-ups of THIS field's weeds go
   into every judge prompt. When unsure, protect.
9. Object accuracy was a misleading score.
   Failed: protecting a tangled plant counted as "wrong" even though it is
   correct behavior.
   Rule: main metrics are CROP_HITS (physical: crop pixels inside the 15 px spray
   circle), % of weed area sprayed, % of weed area left inside protected plants.
   Object accuracy is secondary.
10. SAM is slow (7 s per call, ~13 s when 3 run in parallel).
    Rule: cache SAM results per photo. During iteration only Spark calls repeat.
11. The carrot dataset (CWFID) is no longer used to decide anything: doubtful
    labels, near-infrared camera. Truth now = the farmer's labels on our own tray.
    CWFID truth may only simulate a farmer to check that the pipeline runs;
    those numbers never go into the pitch.

## Motion and timing
- Encoder ~1 cm per tick. Car ~5 cm/s. Camera look-ahead hides AI latency.
- Each weed becomes a queued command: fire when encoder reaches tick N at pan/tilt angle A.
- Tilted camera -> homography (4 marked points) -> bird's-eye cm coordinates.

## Geometry and calibration (step 5: tools/geometry.py)
Pixels -> centimeters -> servo angles and encoder ticks. Until the robot exists
every measurement in calib/robot.json (and calib/servo.json) is a PLACEHOLDER
(`"placeholder": true`) and a warning prints whenever they are used.

**Setup.** Camera straight down (90 degrees), ~25 cm above the ridge top, mounted
10-15 cm in front of the nozzle; the image TOP points the way the car drives.
Nozzle at rest also straight down (pan 90, tilt 90). Spraying: the encoder
handles forward/back (fire when the weed is under the nozzle), so tilt stays 90
and only pan moves; laser mode also moves tilt.

**Coordinates (cm).** x = along the driving direction, forward positive;
y = across the row, LEFT positive; origin = the ground point under the nozzle
at rest. Image: up = forward (+x), left = +y. The ground under the image center
is camera_to_nozzle_cm ahead of the origin.

**Two ground planes**, "ridge" (crop on the ridge top) and "furrow"
(ridge_height_cm lower, weeds often there). A plant's plane changes the photo
scale (lower ground is farther from the camera: height 25 -> 33 cm) and the
aiming angle (the nozzle is 15 -> 23 cm above it). `plane_for()` picks "ridge"
inside the field model's crop band, else "furrow"; with only one plane
calibrated it uses that one. `locate(u, v, model)` places the point on the
ridge, picks the plane, then places it again on that plane.

**The five measurements** (calib/robot.json; add each to `measured` when done):
1. camera_height_cm: camera lens to the ridge top.
2. ridge_height_cm: ridge top down to the furrow floor (~8).
3. camera_to_nozzle_cm (D, along x) and camera_side_offset_cm (along y, left
   positive): ground point under the lens to ground point under the nozzle.
4. nozzle_height_cm: nozzle tip above the ridge top (~15).
5. cm_per_tick: `python tools/measure_ticks.py --ticks N` after driving exactly
   100 cm.
(camera_hfov_deg 66 and the resolution 1296x972 come from the camera and vision.py.)

**Formulas.** scale = 2 h tan(hfov/2) / width (h = camera height above that
plane), unless an A4 calibration exists (then its homography is used, which also
handles a slightly tilted mount). h_nozzle = nozzle_height (ridge) or
nozzle_height + ridge_height (furrow); pan = 90 + atan(y / h_nozzle),
tilt = 90 + atan(x / h_nozzle), then calib/servo.json (center trim, direction
sign), clamped to 0-180; beyond max_reach_deg (45) off center -> OutOfReach.
fire_tick = tick_at_capture + round(x / cm_per_tick).

**Calibration when the robot is assembled** (every step prints what to check):
1. Measure the five numbers above, write them into calib/robot.json.
2. A4 on the ridge top, edges parallel and square to the RAILS (long side across them;
   a sheet turned 2 degrees turns every position 2 degrees: ~0.5 cm at 15 cm out), whole sheet in view; take a photo with the real camera at the real
   resolution. `python tools/calibrate_a4.py photo.jpg --plane ridge`, click the
   4 corners (top-left, top-right, bottom-right, bottom-left). It prints cm per
   pixel, the tilt from straight down, and the camera height it estimates (compare
   with step 1), and saves calib/homography_ridge.json + preview_ridge.jpg:
   the 1 cm grid must sit on the sheet's edges.
3. Same with the A4 at the bottom of the furrow: `--plane furrow`.
4. Mark 5 dots on a sheet, measure them with a ruler, then
   `python tools/check_points.py photo.jpg --plane ridge` (click each dot, type its
   X Y in cm from the sheet's top-left). Target: about 1 cm or less on average.
5. Servos: with the nozzle pointing straight down at the ground, adjust
   center_trim_deg in calib/servo.json until it really is straight down; flip
   `sign` if a servo turns the wrong way.
6. Drive 100 cm and measure_ticks (step 5 of the measurements).

**Things the numbers already say** (placeholders, so recheck on the robot):
- With D = 12 cm and the camera 25 cm up, the photo's bottom edge on the ridge is
  at x ~ 0, right at the nozzle: ridge plants in frame are all still ahead of it.
  Only furrow ground at the corners is already behind (plan_target: 'behind').
- On the ridge the nozzle is 15 cm up, so reach 45 degrees = 15 cm sideways, less
  than the photo's half-width (16 cm): the extreme edges of the ridge are out of
  reach (OutOfReach). In the furrow (23 cm up) the whole frame is reachable.
- Tests (tests/test_geometry.py, no hardware): straight-down grid error 0.006 cm
  max; 5 degree tilt with the A4 homography 0.007 cm max (0.4 cm without it).

## Simulator (step 6: tools/hardware.py, planner.py, sim.py)
The robot before the robot exists. `planner.py` is the real brain and talks to
the world ONLY through `hardware.py`; the simulator gives it a `FakeHardware`,
the robot will give it `PiHardware` (a stub until step 8).

**Hardware interface:** drive(speed), stop(), ticks(), capture(), servo_set(angle),
pump(ms), laser(on), now(), sleep(s). FakeHardware owns the clock (sleep = time
passes), the true position, the encoder (1 cm/tick), a servo that takes time to
turn (SG90, 0.1 s per 60 degrees), a camera that renders the right window of the
strip, and an event log {time, type, value, tick, arm angle}. Latency constants
are at the top of hardware.py (all guesses until measured): settle 0.3 s, spray
0.15 s, drip 0.1 s, servo 0.1 s/60 deg, capture 0.1 s, SAM+Spark 10 s/frame
(`--vision-s`), the car coasts 0.1 s after stop().

**The wiper arm.** The nozzle sits at the tip of an arm of length L (calib
`arm_length_cm`, placeholder 25) that pivots at the robot: at arm angle phi the
nozzle is at (L cos phi, L sin phi) from the pivot. To hit a weed y cm to the side:
phi = asin(y / L), and the car must stop L cos(phi) BEFORE the weed:
stop tick = weed tick - L cos(phi). geometry.py: `wiper_arm`, `arm_to_servo`,
`servo_to_arm`, `arm_landing`, `ArmOutOfReach`; calib/servo.json has an `arm`
block (trim, sign, max_deg 80, park_deg 70 = parked off to the side toward the
furrow). The old pan/tilt model is still there for the laser mode.

**Planner.** Pass 1 maps: drive forward, a frame every N ticks (20% overlap),
vision on each frame (results arrive `vision_s` later, 3 in parallel), pixels ->
world cm with geometry.locate, duplicates merged (within 2 cm), `out/weed_map.json`.
Crops, mixed, unsure (and weeds the judge was not sure about) become protected
zones. Pass 2 drives back and walks the weed list: stop, settle, swing the arm,
pump, wait for the drips, park. **Safety check before every spray:** the real
landing point (encoder + arm angle) must be farther than spray radius + margin +
half a tick from every protected zone, else "skipped: near crop". `--continuous`
never stops: it starts the pump half a spray early and the servo early enough,
from speed and the latencies; a weed it cannot reach in time is "skipped: late".

**Scoring (sim.py).** Truth = the farmer's labels; the planner never sees it. A
spray is judged by where the nozzle PHYSICALLY was (true car position, real servo
angle every 20 ms of the pump), not where the plan said: CROP_HITS = sprays whose
wet disc (spray_radius_cm) touches a crop; plus weeds sprayed/missed, sprays on
nothing, % of weed area covered, liquid (pump ms x pump_flow_ml_per_s) vs
broadcasting the same dose over the whole row. Output: out/sim.mp4,
out/sim_report.json (with the reason for every missed weed), out/weed_map.json.
`python tools/sim.py --field tomato-3 [--continuous] [--speed 5] [--live-vision]`
(--tag name writes out/sim_<tag>.*). Tests: tests/test_planner.py (no API).

**Skipped weeds, and re-aiming.** A weed is "skipped: near crop" when its spray
point is closer to a protected plant than spray radius + margin + half a tick.
Before giving up the planner looks for another spot ON THE SAME WEED's outline
that is far enough from every protected zone (a big weed beside a crop is
usually sprayable on its far side). Only if the judge was >= 0.95 sure
(`reaim_min_conf`): the first version re-aimed everything and sprayed a crop
Spark had called a weed at 0.9 (CROP_HITS 1; the old near-crop skip had hidden
that vision error by luck). The 0.95 was chosen after seeing that one case, so it
is a safety setting, not a measured optimum. Weeds the judge never called a sure
weed are not in the map at all (grey rings in the video): that is the vision's
decision, not the planner's.

**One pass, real time (the plan for real columns).** `--realtime` /
`Planner.run_realtime`: the first columns are driven once to GATHER DATA (slow
SAM + Spark labels, a person reviews, nothing is sprayed), an offline model is
trained from them (YOLO: NOT trained yet; the sim replays the cached AI results at
0.1 s per frame as a stand-in), then each real column is driven ONCE: detect each
frame, merge, spray as the arm gets level; no map, no return trip, no waiting for
the AI. Finding: the wiper nozzle sits L cos(phi) AHEAD of the pivot, so the camera
must see a weed before the pivot is that far behind it: camera about arm length +
half a frame ahead of the pivot (25 + 12 = ~40 cm, `--camera-cm`). With the camera
12 cm ahead every central weed is seen too late ("skipped: passed", tested).
`python tools/sim.py --field tomato-3 --lifecycle` writes out/sim_lifecycle.mp4
(data columns -> training card -> real column -> result card, ~60 s).

**Results** (field tomato-3, the 3 farmer-labeled photos as a 73 cm strip,
placeholder calibration, cached vision):

| run | weeds sprayed | CROP_HITS | liquid vs broadcast | total time |
|---|---|---|---|---|
| two passes, stop-and-spray, 3 cm/s | 24/40 | 0 | 7.2 ml vs 100.7 ml | 94.4 s |
| two passes, continuous, 3 cm/s | 24/40 | 0 | 7.2 ml vs 100.7 ml | 78.4 s |
| two passes, stop-and-spray, 5 cm/s | 23/40 | 0 | 6.9 ml vs 100.7 ml | 62.7 s |
| ONE pass real time, 3 cm/s, camera 40 cm | 25/40 | 0 | 7.5 ml vs 100.7 ml | 44.7 s |

Read these honestly: (1) the 15-17 misses are mostly vision (11 weeds Spark did not
call a sure weed) plus 3-5 skipped near a crop; every weed the planner chose to
spray was hit, except one "late" (continuous) and one "passed" (5 cm/s). (2) The
field model was learned from these same 3 photos, so the vision part is optimistic;
the mechanics (timing, arm, landing) are not. (3) With the safety check off the same
run had 1 crop hit, so CROP_HITS = 0 is not vacuous. (4) Every number depends on
placeholders (arm length, spray radius, pump flow, latencies): redo on the robot.
(5) Cached vision shows each frame the objects whose centre is in it; plants cut by
a frame edge are only tested by --live-vision. (6) Truth only covers plants SAM
found as objects. (7) The data columns in the lifecycle video are other photos of the
same field (their AI labels come from the evaluation run), not separate rows.

## Platform pages (crop vs field)
- **Crop** = what the app used to call a "field": the workspace where a crop is defined, photographed, labeled and learned (folder `fields/<name>`, URLs `/field/<name>/...` unchanged internally). **New crop** (`/crop/new`, old `/field/new` still works) is that form.
- **Train new crop** (`/crop/train`): a clearly marked placeholder for the future "robot drives three columns, AI labels, you review, train the offline model". Everything on it is disabled.
- **Field** = the real ground the robot weeds (`farm/<name>.json`: name, chosen crop, columns). **New field** (`/fields/new`) picks one crop that has finished training (has `model.json`); `/fields` lists them. The detector is a stand-in until the offline YOLO model exists.

## Hardware (arriving soon)
Raspberry Pi (only controller) + Pi camera, 4WD TT-motor car on L298N, 3x18650,
LM2596 at 5 V for 2x SG90 servos (pan, tilt) + 5 V pump + laser via IRLZ44N MOSFETs.
Red button = hard cut of pump/laser power. Green button = mode select.

## Code rules
- Python 3.11. Everything that touches hardware goes through hardware.py with a
  FakeHardware class, so the simulator and the real robot run the same code.
- Never call the API in a tight loop. Cache results. Log every API call and its cost.
- tests/images/carrot_field_*.png have ground truth in *_truth.png
  (green = crop, red = weed). Use them for evaluation.
- scratch/ holds the first test scripts. Reuse ideas, do not import from there.

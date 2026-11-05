"""The planner: the robot's brain, shared by the simulator and the real robot.

It touches the world ONLY through the Hardware interface (hardware.py) and
the vision backend it is given. Two passes over the row:

  Pass 1 (map)   drive forward at `speed`, take a frame every N ticks (about
                 20% overlap), have vision find weeds and protected plants,
                 convert to world cm with geometry.py, merge duplicates, save
                 out/weed_map.json. Crops, mixed and unsure plants (and weeds
                 the judge was not sure about) become PROTECTED zones.
  Pass 2 (spray) drive back to the start, then forward again. The wiper arm
                 (length L, pivot on the robot) puts the nozzle at
                 (L cos(phi), L sin(phi)) from the pivot, so for a weed y cm to
                 the side phi = asin(y / L) and the car must stop
                 L cos(phi) BEFORE the weed. Stop, settle, swing the arm, pump,
                 wait for the drips, park the arm toward the furrow. Before
                 every spray a safety check skips weeds too close to a
                 protected zone ("skipped: near crop").
                 --continuous does not stop: it predicts when to start the
                 pump from the speed and the latencies.

Vision on the real robot (Config.async_vision): detect() takes 10-35 s, so it
runs in a pool of vision_workers threads while the driving loop keeps reading the
encoder; a frame's detections are used when they are really ready. The simulator
leaves it off: there detect() returns at once and the latency is only counted.

Vision backend: detect(image, tick) -> list of detections
    {"label": crop|weed|unsure, "action": spray|protect,
     "spray_px": (u, v) | None, "polygon_px": [(u, v), ...], "box": [...]}
in the pixels of the frame the camera gave. Cached or live, same shape.
"""
from __future__ import annotations

import json
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

import geometry as g
import hardware as hw_mod
from hardware import Hardware


class Stalled(Exception):
    """The car is told to drive but the encoder has stopped counting: it hit
    the wall at the end of the rails, or it is stuck."""


@dataclass
class Config:
    speed: float = 3.0                      # cm/s while mapping and spraying
    overlap: float = 0.20                   # between consecutive frames
    spray_ms: float = hw_mod.SPRAY_S * 1000
    settle_s: float = hw_mod.SETTLE_S
    drip_s: float = hw_mod.DRIP_S
    brake_s: float = hw_mod.COAST_S         # coast after stop(): stop this early
    vision_s: float = hw_mod.VISION_S       # seconds per frame (SAM + Spark)
    vision_workers: int = 3                 # frames processed in parallel
    async_vision: bool = False              # real robot: detect in threads, keep driving
    max_travel_cm: float | None = None      # stop-and-go: never drive further than this from the start
    stall_s: float | None = None            # real robot: no encoder count this long while driving = Stalled
    aim_hold_s: float = 0.0                 # calibration: hold the arm aimed at each weed this long
    continuous: bool = False                # spray without stopping
    merge_cm: float = 2.0                   # same plant seen twice
    late_cm: float = 1.5                    # continuous: skip if this far late
    reaim_min_conf: float = 0.95            # a weed beside a crop is re-aimed only if this sure
    poll_s: float = 0.02
    out_dir: Path = Path("out")
    image_size: tuple = (1296, 972)


class Planner:
    def __init__(self, hw: Hardware, vision, calib: g.Calib | None = None,
                 model: dict | None = None, cfg: Config | None = None):
        self.hw, self.vision = hw, vision
        self.calib = calib or g.default_calib()
        self.model = model
        self.cfg = cfg or Config()
        r = self.calib.robot
        self.cm = float(r["cm_per_tick"])
        self.L = float(r["arm_length_cm"])
        self.spray_r = float(r["spray_radius_cm"])
        self.margin = float(r["protect_margin_cm"])
        self.park_deg = g.arm_to_servo(float(self.calib.servo["arm"]["park_deg"]), self.calib)
        self.arm_deg = None                 # last servo angle commanded
        self.frames: list[dict] = []
        self.weeds: list[dict] = []
        self.protected: list[dict] = []
        self.row_y_cm = 0.0
        self.log: list[dict] = []           # decisions (skips, sprays)
        self.start_tick = 0
        self._worker_free = [0.0] * max(1, self.cfg.vision_workers)
        self.ready_at = 0.0
        self.phases: list[tuple[str, float]] = []
        self._pool = (ThreadPoolExecutor(max(1, self.cfg.vision_workers), thread_name_prefix="vision")
                      if self.cfg.async_vision else None)

    # ---------- small helpers ----------
    def _phase(self, name: str) -> None:
        self.phases.append((name, self.hw.now()))

    def _note(self, kind: str, **kw) -> None:
        self.log.append({"t": round(self.hw.now(), 3), "type": kind, **kw})

    def frame_length_cm(self) -> float:
        w, h = self.cfg.image_size
        x0, _ = g.pixel_to_ground(w / 2, 0, "ridge", (w, h), self.calib)
        x1, _ = g.pixel_to_ground(w / 2, h, "ridge", (w, h), self.calib)
        return abs(x0 - x1)

    def capture_every_ticks(self) -> int:
        return max(1, int(self.frame_length_cm() * (1 - self.cfg.overlap) / self.cm))

    def _wait_ticks(self, target: int, up: bool = True) -> None:
        """Poll the encoder until it reaches target (counting up or down). With
        cfg.stall_s, raises Stalled when the count stops changing that long."""
        last, since = self.hw.ticks(), self.hw.now()
        while (last < target) if up else (last > target):
            self.hw.sleep(self.cfg.poll_s)
            now_ticks = self.hw.ticks()
            if now_ticks != last:
                last, since = now_ticks, self.hw.now()
            elif self.cfg.stall_s is not None and self.hw.now() - since > self.cfg.stall_s:
                raise Stalled(f"no encoder count for {self.cfg.stall_s:.1f} s at tick {last}")

    def _vision_ready(self, now: float) -> float:
        i = min(range(len(self._worker_free)), key=self._worker_free.__getitem__)
        start = max(now, self._worker_free[i])
        self._worker_free[i] = start + self.cfg.vision_s
        return self._worker_free[i]

    def _detect(self, img, tick: int) -> dict:
        """A frame record. Simulator: detected now, ready after vision_s (counted).
        Real robot (async_vision): detected in a worker thread, ready when done."""
        now = self.hw.now()
        f = {"tick": tick, "t": round(now, 2)}
        if self._pool is None:
            f["dets"] = self.vision.detect(img, tick)
            f["ready_s"] = round(self._vision_ready(now), 2)
            self._note("capture", tick=tick, detections=len(f["dets"]))
        else:
            f["dets"], f["ready_s"] = None, None
            f["future"] = self._pool.submit(self.vision.detect, img, tick)
            self._note("capture", tick=tick, detections=None)
        return f

    def _ready(self, f: dict) -> bool:
        """Are this frame's detections in? (Collects them from the worker.) A
        detection that failed gives no detections at all: nothing is sprayed
        from that frame."""
        if "future" not in f:
            return f["ready_s"] <= self.hw.now()
        if not f["future"].done():
            return False
        try:
            f["dets"] = f.pop("future").result()
            self._note("detected", tick=f["tick"], detections=len(f["dets"]))
        except Exception as e:  # noqa: BLE001  (an API error must not stop the robot)
            f["dets"] = []
            self._note("vision_error", tick=f["tick"], error=f"{type(e).__name__}: {e}")
        f["ready_s"] = round(self.hw.now(), 2)
        self.ready_at = max(self.ready_at, f["ready_s"])
        return True

    def _wait_vision(self) -> None:
        """Wait until every frame's detections are in."""
        if self._pool is None:
            if self.hw.now() < self.ready_at:
                self._phase("waiting for the last frames")
                self._note("wait_vision", seconds=round(self.ready_at - self.hw.now(), 2))
                self.hw.sleep(self.ready_at - self.hw.now())
            return
        pending = [f for f in self.frames if not self._ready(f)]
        if pending:
            self._phase("waiting for the last frames")
            t0 = self.hw.now()
            while pending:
                self.hw.sleep(0.1)  # keeps the robot's watchdog fed
                pending = [f for f in pending if not self._ready(f)]
            self._note("wait_vision", seconds=round(self.hw.now() - t0, 2))

    def close(self) -> None:
        """Stop the vision threads (the real robot; nothing to do in simulation)."""
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)

    # ---------- Pass 1: map ----------
    def map_pass(self, end_cm: float) -> None:
        """Drive forward until the camera has seen ground up to end_cm."""
        hw, cfg = self.hw, self.cfg
        self.start_tick = hw.ticks()
        self._phase("pass 1: mapping")
        half = self.frame_length_cm() / 2
        D = float(self.calib.robot["camera_to_nozzle_cm"])
        end_tick = int(math.ceil((end_cm - D - half) / self.cm))
        every = self.capture_every_ticks()
        nxt = self.start_tick
        hw.drive(cfg.speed)
        while True:
            t = hw.ticks()
            if t >= nxt:
                f = self._detect(hw.capture(), t)
                self.frames.append(f)
                if f["ready_s"] is not None:
                    self.ready_at = max(self.ready_at, f["ready_s"])
                nxt += every
            if t >= end_tick:
                break
            hw.sleep(cfg.poll_s)
        hw.stop()
        hw.sleep(cfg.brake_s)

    def return_to_start(self) -> None:
        hw = self.hw
        self._phase("driving back")
        hw.drive(-self.cfg.speed)
        self._wait_ticks(self.start_tick + int(round(self.cfg.speed * self.cfg.brake_s / self.cm)),
                         up=False)
        hw.stop()
        hw.sleep(self.cfg.brake_s)
        self._wait_vision()                    # results from the last frames

    # ---------- the weed map ----------
    def ingest(self, frames: list[dict]) -> None:
        """Pixels -> world cm for these frames; merge them into the weed list
        and the protected zones (a plant seen twice is one plant)."""
        size = self.cfg.image_size
        crops_y = []
        for f in self.frames:
            for d in f["dets"]:
                if d["label"] == "crop" and d.get("box") is not None:
                    u = (d["box"][0] + d["box"][2]) / 2
                    v = (d["box"][1] + d["box"][3]) / 2
                    crops_y.append(g.pixel_to_ground(u, v, "ridge", size, self.calib)[1])
        if crops_y:
            self.row_y_cm = float(np.median(crops_y))

        def to_world(u, v, tick):
            p = g.locate(u, v, self.model, size, self.row_y_cm, self.calib)
            return tick * self.cm + p["x"], p["y"], p["plane"]

        for f in frames:
            for d in f["dets"]:
                if d["action"] == "spray" and d.get("spray_px"):
                    x, y, plane = to_world(*d["spray_px"], f["tick"])
                    for m in self.weeds:
                        if math.dist((x, y), (m["x"], m["y"])) < self.cfg.merge_cm:
                            m["sightings"] += 1
                            if "status" not in m:      # not decided yet: refine
                                n = m["sightings"]
                                m["x"] += (x - m["x"]) / n
                                m["y"] += (y - m["y"]) / n
                                m["tick"] = int(round(m["x"] / self.cm))
                            break
                    else:
                        poly = [to_world(u, v, f["tick"])[:2] for u, v in d.get("polygon_px") or []]
                        self.weeds.append({"x": x, "y": y, "plane": plane, "sightings": 1,
                                           "tick": int(round(x / self.cm)), "poly": poly,
                                           "conf": d.get("conf")})
                elif d["action"] != "spray":
                    pts = [to_world(u, v, f["tick"])[:2] for u, v in d["polygon_px"]]
                    if len(pts) < 3:
                        continue
                    c = np.mean(pts, axis=0)
                    if not any(math.dist(c, q["centroid"]) < self.cfg.merge_cm
                               and q["label"] == d["label"] for q in self.protected):
                        self.protected.append({"label": d["label"], "polygon": pts,
                                               "centroid": (float(c[0]), float(c[1]))})
        self.weeds.sort(key=lambda m: m["x"])
        for i, m in enumerate(self.weeds):
            m["id"] = i

    def build_map(self) -> dict:
        self.ingest(self.frames)
        return self.save_map()

    def save_map(self) -> dict:
        d = {"cm_per_tick": self.cm, "row_y_cm": round(self.row_y_cm, 2),
             "arm_length_cm": self.L,
             "weeds": [{k: (round(v, 2) if isinstance(v, float) else v)
                        for k, v in w.items() if k != "poly"} for w in self.weeds],
             "protected": [{"label": p["label"],
                            "polygon": [[round(x, 2), round(y, 2)] for x, y in p["polygon"]]}
                           for p in self.protected],
             "frames": [{"tick": f["tick"], "t": f["t"], "ready_s": f["ready_s"],
                         "detections": len(f["dets"])} for f in self.frames]}
        out = Path(self.cfg.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "weed_map.json").write_text(json.dumps(d, indent=1), encoding="utf-8")
        return d

    # ---------- planning one spray ----------
    def plan_stop(self, wx_cm: float, wy_cm: float) -> tuple[int, float]:
        """(tick at which the car must stop, arm angle phi in degrees).
        stop tick = weed tick - L cos(phi) in ticks; phi = asin(y / L)."""
        phi, off = g.wiper_arm(wy_cm, self.calib)
        return int(round(wx_cm / self.cm)) - int(round(off / self.cm)), phi

    def dist_to_protected(self, x: float, y: float) -> float:
        """Distance (cm) from a point to the nearest protected zone; negative
        or zero inside one."""
        best = float("inf")
        for p in self.protected:
            poly = np.asarray(p["polygon"], np.float32)
            lo, hi = poly.min(axis=0), poly.max(axis=0)
            if (x < lo[0] - 5 or x > hi[0] + 5 or y < lo[1] - 5 or y > hi[1] + 5):
                continue
            best = min(best, -cv2.pointPolygonTest(poly.reshape(-1, 1, 2), (float(x), float(y)), True))
        return best

    def safe_distance(self) -> float:
        """How far a spray centre must be from a protected zone: the wet
        radius, a margin, and half a tick of not knowing where the car is."""
        return self.spray_r + self.margin + 0.5 * self.cm

    # ---------- spraying ----------
    def _park(self) -> None:
        self.hw.servo_set(self.park_deg)
        self.arm_deg = self.park_deg

    def reaim(self, w: dict) -> bool:
        """The planned spot is too close to a protected plant. Look for another
        spot ON THE SAME WEED (inside its own outline) that is far enough from
        every protected zone, nearest to where we first aimed. A big weed
        beside a crop is usually sprayable on its far side."""
        poly = w.get("poly") or []
        if len(poly) < 3:
            return False
        arr = np.asarray(poly, np.float32)
        lo, hi = arr.min(axis=0), arr.max(axis=0)
        need = self.safe_distance() + 0.3          # slack for stopping error
        best = None
        for x in np.arange(lo[0], hi[0] + 1e-6, 0.25):
            for y in np.arange(lo[1], hi[1] + 1e-6, 0.25):
                if cv2.pointPolygonTest(arr.reshape(-1, 1, 2), (float(x), float(y)), False) < 0:
                    continue
                if self.dist_to_protected(float(x), float(y)) < need:
                    continue
                d = math.dist((x, y), (w["x"], w["y"]))
                if best is None or d < best[0]:
                    best = (d, float(x), float(y))
        if best is None:
            return False
        w["aimed_from"] = [round(w["x"], 2), round(w["y"], 2)]
        w["x"], w["y"] = best[1], best[2]
        w["tick"] = int(round(w["x"] / self.cm))
        self._note("reaim", weed=w["id"], moved_cm=round(best[0], 2))
        return True

    def _job(self, w: dict):
        """(stop_tick, phi) for a weed the arm can reach, else None (and why)."""
        if (not w.get("aim_checked")
                and self.dist_to_protected(w["x"], w["y"]) < self.safe_distance() + 0.3):
            w["aim_checked"] = True
            # a weed this close to a crop is where the judge errs most (a crop
            # called a weed at 0.9 sat exactly here), so it must be very sure
            if (w.get("conf") or 0.0) < self.cfg.reaim_min_conf or not self.reaim(w):
                w["status"] = "skipped: near crop"
                self._note("skip", weed=w["id"], reason=w["status"], detail="no safe spot on the weed, or not sure enough to re-aim")
                return None
        try:
            stop_tick, phi = self.plan_stop(w["x"], w["y"])
        except g.OutOfReach as e:
            w["status"] = "skipped: out of reach"
            self._note("skip", weed=w["id"], reason=w["status"], detail=str(e))
            return None
        if stop_tick < self.start_tick:
            w["status"] = "skipped: before the start"
            self._note("skip", weed=w["id"], reason=w["status"])
            return None
        return stop_tick, phi

    def _jobs(self):
        """Weeds the arm can reach, in driving order: (stop_tick, phi, weed)."""
        jobs = []
        for w in self.weeds:
            j = self._job(w)
            if j:
                jobs.append((j[0], j[1], w))
        return sorted(jobs, key=lambda j: j[0])

    def _brake_ticks(self) -> int:
        return int(round(self.cfg.speed * self.cfg.brake_s / self.cm))

    def _spray_one_stop(self, stop_tick: int, phi: float, w: dict) -> None:
        """Drive to the stop tick, stop, settle, swing the arm, pump, drip, park."""
        hw, cfg = self.hw, self.cfg
        if hw.ticks() > stop_tick + self._brake_ticks():
            w["status"] = "skipped: passed"
            self._note("skip", weed=w["id"], reason=w["status"])
            return
        hw.drive(cfg.speed)
        self._wait_ticks(stop_tick - self._brake_ticks())
        hw.stop()
        hw.sleep(cfg.brake_s)               # coasting
        hw.sleep(cfg.settle_s)
        self._spray_here(phi, w)

    def _spray_here(self, phi: float, w: dict) -> None:
        """The car is stopped near the weed: check where the nozzle will really
        land, swing the arm, pump, drip, park."""
        hw, cfg = self.hw, self.cfg
        # where the nozzle will really be: encoder now + arm at phi
        land_x = (hw.ticks() + 0.5) * self.cm + self.L * math.cos(math.radians(phi))
        land_y = self.L * math.sin(math.radians(phi))
        d = self.dist_to_protected(land_x, land_y)
        if d < self.safe_distance():
            w["status"] = "skipped: near crop"
            self._note("skip", weed=w["id"], reason=w["status"], dist_cm=round(d, 2))
            return
        target = g.arm_to_servo(phi, self.calib)
        hw.servo_set(target)
        hw.sleep(hw_mod.servo_travel_s(self.arm_deg, target))
        self.arm_deg = target
        if cfg.aim_hold_s > 0:  # calibration: look where the nozzle really points
            self._phase(f"holding the arm on weed {w['id']} for {cfg.aim_hold_s:g} s")
            self._note("hold", weed=w["id"], seconds=cfg.aim_hold_s, x_cm=round(w["x"], 1),
                       y_cm=round(w["y"], 1), phi=round(phi, 1), servo=round(target, 1), tick=hw.ticks())
            hw.sleep(cfg.aim_hold_s)
        hw.pump(cfg.spray_ms)
        hw.sleep(cfg.drip_s)
        self._park()
        w["status"] = "sprayed"
        self._note("spray", weed=w["id"], tick=hw.ticks(), phi=round(phi, 1))

    def _continuous_plan(self, phi: float, w: dict) -> tuple[int, int, float]:
        """(tick to start the pump, ticks of lead the servo needs, servo angle)."""
        cfg = self.cfg
        half_spray = cfg.speed * cfg.spray_ms / 2000.0
        off = self.L * math.cos(math.radians(phi))
        fire_tick = int(round((w["x"] - off - half_spray) / self.cm))
        target = g.arm_to_servo(phi, self.calib)
        lead = int(math.ceil(hw_mod.servo_travel_s(self.arm_deg, target) * cfg.speed / self.cm)) + 1
        return fire_tick, lead, target

    def _spray_one_continuous(self, phi: float, w: dict) -> None:
        """Never stop. Predict where the car will be when the pump runs: start
        the pump half a spray early so the middle of the spray is on the weed,
        and start the servo early enough to arrive."""
        hw, cfg = self.hw, self.cfg
        half_spray = cfg.speed * cfg.spray_ms / 2000.0
        fire_tick, lead, target = self._continuous_plan(phi, w)
        self._wait_ticks(fire_tick - lead)
        if hw.ticks() > fire_tick + cfg.late_cm / self.cm:
            w["status"] = "skipped: late"
            self._note("skip", weed=w["id"], reason=w["status"])
            return
        # the nozzle will be at the weed in the middle of the spray
        if self.dist_to_protected(w["x"], w["y"]) < self.safe_distance() + half_spray:
            w["status"] = "skipped: near crop"
            self._note("skip", weed=w["id"], reason=w["status"])
            return
        hw.servo_set(target)
        self.arm_deg = target
        self._wait_ticks(fire_tick)
        hw.pump(cfg.spray_ms)
        self._park()
        w["status"] = "sprayed"
        self._note("spray", weed=w["id"], tick=hw.ticks(), phi=round(phi, 1))

    # ---------- Pass 2: spray what Pass 1 mapped ----------
    def spray_pass(self) -> None:
        self._phase("pass 2: continuous spray" if self.cfg.continuous
                    else "pass 2: stop-and-spray")
        self._park()
        if self.cfg.continuous:
            self.hw.drive(self.cfg.speed)
        for stop_tick, phi, w in self._jobs():
            if self.cfg.continuous:
                self._spray_one_continuous(phi, w)
            else:
                self._spray_one_stop(stop_tick, phi, w)
        self.hw.stop()
        if self.cfg.continuous:
            self.hw.sleep(self.cfg.brake_s)

    # ---------- stop-and-go, one picture at a time ----------
    def _go_to(self, target: int) -> None:
        """Drive forward or back to the target tick and stop there."""
        hw, cfg = self.hw, self.cfg
        target = min(target, self._travel_limit())  # never off the end of the rails
        target = max(target, self.start_tick)
        now = hw.ticks()
        if abs(now - target) <= self._brake_ticks():
            return
        forward = target > now
        hw.drive(cfg.speed if forward else -cfg.speed)
        try:
            if forward:
                self._wait_ticks(target - self._brake_ticks(), up=True)
            else:
                self._wait_ticks(target + self._brake_ticks(), up=False)
        except Stalled as e:
            hw.stop()
            hw.sleep(cfg.brake_s)
            if forward:  # the wall at the end of the rails: nothing beyond it
                self.wall_tick = hw.ticks()
                self._note("wall", tick=self.wall_tick, detail=str(e))
            raise
        hw.stop()
        hw.sleep(cfg.brake_s)
        hw.sleep(cfg.settle_s)

    def _travel_limit(self) -> int:
        """The furthest tick the car may drive to: the travel limit, or just
        before the wall once the car has found it."""
        limit = 10 ** 9
        if self.cfg.max_travel_cm is not None:
            limit = self.start_tick + int(self.cfg.max_travel_cm / self.cm)
        if getattr(self, "wall_tick", None) is not None:
            limit = min(limit, self.wall_tick - 1)
        return limit

    def _spray_known(self, before_x: float | None) -> None:
        """Spray every weed not decided yet whose spot is before before_x (cm;
        None = all), in driving order, backing up first if one is behind."""
        todo = []
        for w in self.weeds:
            if "status" in w or (before_x is not None and w["x"] >= before_x):
                continue
            j = self._job(w)
            if j and j[0] > self._travel_limit() + self._brake_ticks():
                w["status"] = "skipped: beyond the rails"
                self._note("skip", weed=w["id"], reason=w["status"])
                continue
            if j:
                todo.append((j[0], j[1], w))
        for stop_tick, phi, w in sorted(todo, key=lambda j: j[0]):
            try:
                self._go_to(stop_tick)
            except Stalled:
                w["status"] = "skipped: beyond the rails"
                self._note("skip", weed=w["id"], reason=w["status"], tick=self.hw.ticks())
                continue
            if abs(self.hw.ticks() - stop_tick) > self._brake_ticks() + 1:
                w["status"] = "skipped: missed the stop"
                self._note("skip", weed=w["id"], reason=w["status"], tick=self.hw.ticks())
                continue
            self._spray_here(phi, w)

    def run_stepwise(self, end_cm: float, step_cm: float | None = None) -> dict:
        """Stop, take a picture, wait for its detections, spray its weeds, move on
        step_cm (default: one picture's length, no overlap), repeat until the
        camera has seen up to end_cm. The picture starts D - half a picture ahead
        of the arm's pivot but the arm reaches only L ahead of it, so the car
        backs up for weeds near the back of a picture. A weed near the front edge
        waits for the next picture: what grows just beyond it (a crop?) must be
        known before spraying it."""
        hw, cfg = self.hw, self.cfg
        self.start_tick = hw.ticks()
        frame = self.frame_length_cm()
        step_ticks = max(1, int(round((step_cm or frame) / self.cm)))
        D = float(self.calib.robot["camera_to_nozzle_cm"])
        end_tick = self.start_tick + int(math.ceil(max(0.0, end_cm - D - frame / 2) / self.cm))
        guard = self.safe_distance() + cfg.merge_cm
        self._park()
        position = self.start_tick
        self.wall_tick = None
        while True:
            self._phase(f"picture {len(self.frames) + 1}")
            try:
                self._go_to(position)
            except Stalled:
                position = hw.ticks()  # at the wall: this is the last picture
            f = self._detect(hw.capture(), hw.ticks())
            self.frames.append(f)
            self._phase(f"picture {len(self.frames)}: detecting")
            while not self._ready(f):
                hw.sleep(0.1)
            self.ingest([f])
            limit = self._travel_limit()
            # this picture reaches the end of the stretch, or the car can go no further
            last = position >= end_tick or position >= limit or self.wall_tick is not None
            self._phase(f"picture {len(self.frames)}: spraying")
            front = f["tick"] * self.cm + D + frame / 2
            self._spray_known(None if last else front - guard)
            if last:
                break
            position = min(position + step_ticks, limit)
        hw.stop()
        hw.sleep(cfg.brake_s)
        return self.save_map()

    # ---------- everything, two passes ----------
    def run(self, end_cm: float) -> dict:
        self.map_pass(end_cm)
        self.return_to_start()
        d = self.build_map()
        self.spray_pass()
        return d

    # ---------- everything, ONE pass (a trained fast detector) ----------
    def run_realtime(self, end_cm: float) -> dict:
        """Drive the column once. Each frame is detected a moment after it is
        taken (cfg.vision_s, ~0.1 s for a small offline model, not 10 s for
        SAM + Spark); weeds are merged into the list as they appear and
        sprayed as the arm comes level with them. The camera must see a weed
        before the pivot gets to L cos(phi) behind it, so it has to be mounted
        about arm length + half a frame ahead of the pivot (tools/sim.py
        --camera-cm); a weed that shows up too late is "skipped: passed"."""
        hw, cfg = self.hw, self.cfg
        self.start_tick = hw.ticks()
        self._phase("real-time pass: detect + spray")
        half = self.frame_length_cm() / 2
        D = float(self.calib.robot["camera_to_nozzle_cm"])
        end_tick = int(math.ceil((end_cm - D - half) / self.cm))
        every = self.capture_every_ticks()
        nxt = self.start_tick
        waiting: list[dict] = []          # frames whose detections are not ready
        self._park()
        hw.drive(cfg.speed)
        while True:
            t = hw.ticks()
            if t >= nxt and t <= end_tick + every:
                f = self._detect(hw.capture(), t)
                self.frames.append(f)
                waiting.append(f)
                nxt += every
            ready = [f for f in waiting if self._ready(f)]
            if ready:
                waiting = [f for f in waiting if f not in ready]
                self.ingest(ready)
            todo = []
            for w in self.weeds:
                if "status" in w:
                    continue
                j = self._job(w)
                if j:
                    todo.append((j[0], j[1], w))
            if todo:
                stop_tick, phi, w = min(todo, key=lambda j: j[0])
                if cfg.continuous:
                    fire_tick, lead, _ = self._continuous_plan(phi, w)
                    trigger = fire_tick - lead - 1
                else:
                    trigger = stop_tick - self._brake_ticks()
                if hw.ticks() >= trigger:
                    if cfg.continuous:
                        self._spray_one_continuous(phi, w)
                    else:
                        self._spray_one_stop(stop_tick, phi, w)
                    hw.drive(cfg.speed)
                    continue
            if t >= end_tick and not waiting and not todo:
                break
            if t >= end_tick + 4 * every:       # safety stop
                break
            hw.sleep(cfg.poll_s)
        hw.stop()
        hw.sleep(cfg.brake_s)
        return self.save_map()

"""Run a trained model on the field (the robot).

    fields/<crop>/yolo/best.pt                  the model to run (see trainer.py)
    fields/<crop>/field_runs/<stamp>.mp4        what YOLO saw, as video, per run
    fields/<crop>/field_runs/<stamp>.json       the run's log (when it ends)

run_on_field() drives the real robot (hardware.py PiHardware) with planner.py,
stop and go, one picture at a time (planner.run_stepwise):
    1. Stop, take a picture, cut it to the tray (tray.py), detect (SAM + Muse,
       or YOLO) while the car waits.
    2. Spray the weeds in it: drive (back or forward) to each, swing the arm,
       pump (a dummy GPIO pin: the log shows where it would have sprayed), park.
       A weed at the picture's front edge waits for the next picture.
    3. Move on one picture's length and repeat.
    4. Write every picture with its detections drawn (crop green, weed red,
       unsure grey, the tray's walls blue) as an H.264 .mp4, the event log, and
       a summary .json next to it.
    until `stop` is set (the Stop button) or the column ends.
"""
from __future__ import annotations

import json
import subprocess
import threading
import time
from pathlib import Path


STEP_CM_DEFAULT = 19   # about 20% overlap between pictures (a frame covers ~24 cm)
MIN_MAP50 = 0.70   # below this a YOLO model is not trusted on its own
STEP_CM = 10       # SAM + Muse mode: move this far, stop, take a picture, decide


def advice(map50: float | None) -> dict:
    """Which detector to recommend for a model with this mAP50 (None = no model)."""
    if map50 is None:
        return {"use": "sam_muse", "level": "warn",
                "text": "No trained YOLO model for this crop: use SAM + Muse."}
    if map50 < MIN_MAP50:
        return {"use": "sam_muse", "level": "warn",
                "text": f"mAP50 {map50:.2f} is below {MIN_MAP50:.2f}: use SAM + Muse instead. "
                        "The model misses too many weeds."}
    return {"use": "yolo", "level": "ok",
            "text": f"mAP50 {map50:.2f} is {MIN_MAP50:.2f} or better: the YOLO model can run alone."}


def options(ready: list[str], root: Path = Path("fields")) -> list[dict]:
    """One entry per crop that finished its setup: its YOLO model (if any), the model's
    mAP50 and the advice. `ready` = crop names that finished their steps."""
    import json
    out = []
    for crop in ready:
        pt = root / crop / "yolo" / "best.pt"
        map50 = None
        mj = root / crop / "yolo" / "metrics.json"
        if pt.is_file() and mj.is_file():
            try:
                map50 = (json.loads(mj.read_text(encoding="utf-8")).get("best") or {}).get("map50")
            except json.JSONDecodeError:
                pass
        out.append({"crop": crop, "has_model": pt.is_file(), "map50": map50,
                    "saved": time.strftime("%Y-%m-%d %H:%M", time.localtime(pt.stat().st_mtime))
                    if pt.is_file() else None, **advice(map50 if pt.is_file() else None)})
    return out


def compare(crop: str, root: Path = Path("fields")) -> dict | None:
    """The saved head-to-head of the detectors on this crop's real photos
    (tools/compare_detectors.py), or None."""
    import json
    f = root / crop / "detector_compare.json"
    try:
        return json.loads(f.read_text(encoding="utf-8")) if f.is_file() else None
    except json.JSONDecodeError:
        return None


def models(root: Path = Path("fields")) -> list[dict]:
    """Every crop with a saved YOLO model, newest first."""
    out = []
    for p in sorted(root.glob("*/yolo/best.pt")):
        crop = p.parent.parent.name
        out.append({"crop": crop, "path": p, "saved": time.strftime(
            "%Y-%m-%d %H:%M", time.localtime(p.stat().st_mtime)), "ts": p.stat().st_mtime})
    return sorted(out, key=lambda m: -m["ts"])


def videos(crop: str, root: Path = Path("fields")) -> list[str]:
    """Each run's segmentation video (its real-time video is <name>_live.mp4)."""
    d = root / crop / "field_runs"
    return sorted((p.name for p in d.glob("*.mp4") if not p.stem.endswith("_live")), reverse=True) if d.is_dir() else []


def summary(crop: str, video: str, root: Path = Path("fields")) -> dict | None:
    """The saved summary of a recorded run (the .json next to its video), or None."""
    f = root / crop / "field_runs" / Path(video).with_suffix(".json").name
    try:
        d = json.loads(f.read_text(encoding="utf-8")) if f.is_file() else None
    except json.JSONDecodeError:
        return None
    if d is not None:
        d.pop("log", None)
        d.pop("weed_list", None)
    return d


def picture_length_cm() -> float:
    """How much ground one picture covers, front to back (from calib/)."""
    import geometry as g
    import planner as pl
    return pl.Planner(None, None, g.Calib(quiet=True)).frame_length_cm()


def new_video_path(crop: str, root: Path = Path("fields")) -> Path:
    d = root / crop / "field_runs"
    d.mkdir(parents=True, exist_ok=True)
    return d / (time.strftime("%Y%m%d_%H%M%S") + ".mp4")


LENGTH_CM_DEFAULT = 100   # a demo column
VISION_S = {"sam_muse": 35.0, "yolo": 2.0}  # seconds per picture, for the planner's estimates
COLORS = {"crop": (60, 180, 60), "weed": (40, 40, 230), "unsure": (150, 150, 150)}  # BGR


class RunStopped(Exception):
    """The Stop button was pressed during a field run."""


class FieldVision:
    """The planner's vision backend on the real robot: each picture is cut to the
    tray, detected with SAM + Muse (or YOLO), and the detections are mapped back
    to the picture's pixels for geometry.py. A spray point outside the tray is
    dropped. Every picture is kept with its detections drawn, for the video.
    detect() is called from several threads at once."""

    def __init__(self, crop: str, mode: str, model_path: Path | None = None,
                 cache_dir: Path = Path("out/cache"), log_path: Path = Path("out/api_log.jsonl"),
                 work_dir: Path = Path("out/field_run")):
        import tray as tray_mod
        self.crop, self.mode = crop, mode
        self.cache_dir, self.log_path = Path(cache_dir), Path(log_path)
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.tracker = tray_mod.TrayTracker()
        self.lock = threading.Lock()
        self.pictures: dict[int, object] = {}   # tick -> picture with detections drawn
        self.latest_jpeg: bytes | None = None
        self.usd = 0.0
        self.n = 0
        self.with_tray = 0
        self.dropped_outside = 0
        self.below_threshold = 0
        try:  # the crop's weed-confidence threshold (Results page; field.json spray_conf)
            import setup
            self.min_conf = setup.spray_conf_for(crop)
        except Exception:  # noqa: BLE001
            self.min_conf = 0.85
        self._load_detector(mode, model_path)

    def _load_detector(self, mode: str, model_path: Path | None) -> None:
        crop = self.crop
        if mode == "sam_muse":
            import setup
            self.fmodel, srefs = setup.active_model(crop)
            if self.fmodel is None:
                raise RuntimeError(f"{crop} has no field model yet: finish its steps first")
            self.refs = setup.ref_list(srefs, [])
            self.spray_conf = setup.spray_conf_for(crop)
        elif mode == "yolo":
            try:
                from ultralytics import YOLO
            except ImportError as e:
                raise RuntimeError("YOLO mode needs the ultralytics package on the Pi "
                                   "(pip install ultralytics, large). Use SAM + Muse.") from e
            if model_path is None or not Path(model_path).is_file():
                raise RuntimeError(f"{crop} has no trained YOLO model")
            self.yolo = YOLO(str(model_path))
            metrics = json.loads((Path(model_path).parent / "metrics.json").read_text(encoding="utf-8"))
            self.op = metrics.get("operating_point") or {"conf_w": 0.5, "margin": 0.1, "crop_conf": 0.25}
        else:
            raise ValueError(f"unknown mode {mode}")

    def detect(self, image, tick: int) -> list[dict]:
        import cv2
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        with self.lock:
            self.n += 1
            n = self.n
            found = self.tracker.update(bgr)
            self.with_tray += found is not None
        view = found.rectify(bgr) if found is not None else bgr
        local = self._sam_muse(view, n) if self.mode == "sam_muse" else self._yolo(view)

        def back(points):
            if found is None or not points:
                return [(float(x), float(y)) for x, y in points]
            return [(float(x), float(y)) for x, y in found.to_frame(points)]

        dets = []
        for d in local:
            d = dict(d)
            d["polygon_px"] = back(d.get("polygon_px") or [])
            x0, y0, x1, y1 = d["box"]
            corners = back([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
            xs, ys = [p[0] for p in corners], [p[1] for p in corners]
            d["box"] = [min(xs), min(ys), max(xs), max(ys)]
            if d.get("action") == "spray" and (d.get("conf") or 0.0) < self.min_conf:
                # not sure enough it is a weed: protect it like a crop, never aim at it
                d.update(action="protect", label="unsure", spray_px=None)
                with self.lock:
                    self.below_threshold += 1
            if d.get("spray_px") is not None:
                d["spray_px"] = back([d["spray_px"]])[0]
                if found is not None and not found.contains([d["spray_px"]])[0]:
                    with self.lock:
                        self.dropped_outside += 1
                    continue  # never spray outside the tray
            dets.append(d)
        self._keep(bgr, found, dets, tick)
        return dets

    def _sam_muse(self, view, n: int) -> list[dict]:
        import cv2
        import vision
        from sim import _contour_px
        path = self.work_dir / f"picture_{n:03d}.png"
        cv2.imwrite(str(path), view)
        r = vision.analyze(path, cache_dir=self.cache_dir, log_path=self.log_path, refs=self.refs,
                           field_model=self.fmodel, spray_conf=self.spray_conf)
        with self.lock:
            self.usd += r.spark_usd + (0 if r.sam_cached else vision.sam_client.USD_PER_IMAGE)
        boxes = {o["id"]: o for o in r.objects}
        dets = []
        for t in r.targets:
            o = boxes[t.id]
            dets.append({"label": {"crop": "crop", "weed": "weed"}.get(t.label, "unsure"),
                         "action": t.action, "conf": float(t.confidence),
                         "box": [float(v) for v in o["box"]],
                         "spray_px": (t.x, t.y) if t.action == "spray" else None,
                         "polygon_px": _contour_px(o["mask"]), "id": f"{n}#{t.id}"})
        return dets

    def _yolo(self, view) -> list[dict]:
        import trainer
        res = self.yolo.predict(view, conf=0.03, imgsz=trainer.IMGSZ, verbose=False)[0]
        pred = [(int(c), float(s), [float(v) for v in b])
                for c, s, b in zip(res.boxes.cls.tolist(), res.boxes.conf.tolist(), res.boxes.xyxy.tolist())]
        op = self.op
        spray_boxes = trainer.sprays(pred, op["conf_w"], op["margin"], op["crop_conf"])
        dets = []
        for c, s, b in pred:
            label = trainer.CLASSES[c]
            spray = b in spray_boxes
            if not spray and (c not in trainer.PROTECTED or s < op["crop_conf"]):
                continue
            x0, y0, x1, y1 = b
            dets.append({"label": label, "action": "spray" if spray else "protect", "conf": s, "box": b,
                         "spray_px": ((x0 + x1) / 2, (y0 + y1) / 2) if spray else None,
                         "polygon_px": [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]})
        return dets

    def _keep(self, bgr, found, dets, tick: int) -> None:
        """The picture with its detections drawn, for the video and the live view."""
        import cv2
        import numpy as np
        pic = found.draw(bgr, color=(230, 120, 30)) if found is not None else bgr.copy()
        fill = pic.copy()  # the segmentation: each plant's area filled in its colour
        for d in dets:
            poly = np.int32(d["polygon_px"]).reshape(-1, 1, 2) if d["polygon_px"] else None
            if poly is not None and len(poly) >= 3:
                cv2.fillPoly(fill, [poly], COLORS.get(d["label"], COLORS["unsure"]))
        pic = cv2.addWeighted(fill, 0.4, pic, 0.6, 0)
        for d in dets:
            color = COLORS.get(d["label"], COLORS["unsure"])
            poly = np.int32(d["polygon_px"]).reshape(-1, 1, 2) if d["polygon_px"] else None
            if poly is not None and len(poly) >= 3:
                cv2.polylines(pic, [poly], True, color, 3)
            else:
                x0, y0, x1, y1 = (int(v) for v in d["box"])
                cv2.rectangle(pic, (x0, y0), (x1, y1), color, 3)
            if d.get("spray_px") is not None:
                cv2.drawMarker(pic, (int(d["spray_px"][0]), int(d["spray_px"][1])), (0, 0, 255),
                               cv2.MARKER_CROSS, 28, 4)
        cv2.putText(pic, f"tick {tick}", (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)
        ok, jpeg = cv2.imencode(".jpg", pic, [cv2.IMWRITE_JPEG_QUALITY, 80])
        with self.lock:
            self.pictures[tick] = pic
            if ok:
                self.latest_jpeg = jpeg.tobytes()


class _Guarded:
    """The robot as the planner sees it during a field run: every call first
    checks the Stop button, and reports progress now and then."""

    def __init__(self, hw, stop: threading.Event, report):
        self._hw, self._stop, self._report = hw, stop, report
        self._last = 0.0

    def _check(self) -> None:
        if self._stop.is_set():
            raise RunStopped()
        now = time.monotonic()
        if now - self._last > 0.5:
            self._last = now
            self._report()

    def drive(self, speed_cm_s):
        self._check()
        self._hw.drive(speed_cm_s)

    def stop(self):
        self._hw.stop()

    def ticks(self):
        self._check()
        return self._hw.ticks()

    def capture(self):
        self._check()
        return self._hw.capture()

    def servo_set(self, angle_deg):
        self._check()
        self._hw.servo_set(angle_deg)

    def pump(self, ms):
        self._check()
        self._hw.pump(ms)

    def laser(self, on):
        self._hw.laser(on)

    def now(self):
        return self._hw.now()

    def sleep(self, s):
        end = self._hw.now() + s
        while True:
            self._check()
            left = end - self._hw.now()
            if left <= 0:
                return
            self._hw.sleep(min(left, 0.1))


def live_path(video_path: Path) -> Path:
    """The real-time video of a run, next to its segmentation video."""
    return video_path.with_name(video_path.stem + "_live.mp4")


def _live_video(hw, backend, video_path: Path, status):
    """Record the whole run as it happens (video_rec.py): the camera with the
    tray's walls, the run's status, and the latest segmented picture in the
    corner. None when the camera cannot stream (a dry run without the phone)."""
    camera = getattr(hw, "camera", None)
    if camera is None or not hasattr(camera, "subscribe"):
        return None
    import cv2
    import numpy as np
    import tray as tray_mod
    import video_rec
    live_tray = tray_mod.TrayTracker()
    inset = {"jpeg": None, "image": None}

    def draw(bgr):
        found = live_tray.update(bgr)
        if found is not None:
            bgr = found.draw(bgr, color=(230, 120, 30))
        latest = getattr(backend, "latest_jpeg", None)
        if latest is not None and latest is not inset["jpeg"]:
            inset.update(jpeg=latest, image=cv2.imdecode(np.frombuffer(latest, np.uint8), cv2.IMREAD_COLOR))
        if inset["image"] is not None:
            h, w = bgr.shape[:2]
            iw = w // 3
            small = cv2.resize(inset["image"], (iw, max(1, int(inset["image"].shape[0] * iw / inset["image"].shape[1]))))
            ih = min(small.shape[0], h)
            bgr[h - ih:h, w - iw:w] = small[:ih]
            cv2.rectangle(bgr, (w - iw, h - ih), (w - 1, h - 1), (255, 255, 255), 2)
        s = status()
        return video_rec.caption(bgr, [f"{s['phase']}", f"pictures {s['frames']}  weeds {s['weeds']}  "
                                                         f"sprays {s['sprays']}  skipped {s['skipped']}  {s['seconds']} s"])

    return video_rec.LiveRecorder(camera, live_path(video_path), draw).start()


def write_video(pictures: dict, path: Path, fps: float = 2.0) -> bool:
    """The pictures in driving order as an H.264 .mp4 that browsers play."""
    import cv2
    if not pictures:
        return False
    tmp = path.with_suffix("")
    tmp.mkdir(parents=True, exist_ok=True)
    for i, tick in enumerate(sorted(pictures)):
        cv2.imwrite(str(tmp / f"{i:05d}.jpg"), pictures[tick], [cv2.IMWRITE_JPEG_QUALITY, 90])
    result = subprocess.run(
        ["nice", "-n", "15", "ffmpeg", "-loglevel", "error", "-y", "-framerate", str(fps),
         "-i", str(tmp / "%05d.jpg"), "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
         "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", str(path)],
        capture_output=True)
    for f in tmp.glob("*.jpg"):
        f.unlink()
    tmp.rmdir()
    return result.returncode == 0


def run_on_field(model_path: Path | None, stop: threading.Event, video_path: Path,
                 mode: str = "yolo", step_cm: float = STEP_CM_DEFAULT,
                 length_cm: float = LENGTH_CM_DEFAULT, hardware=None, vision=None,
                 progress=None, hold_s: float | None = None) -> dict:
    """Drive the column stop and go, detect, spray (dummy pump), and record. mode
    "sam_muse" (advised while the YOLO model's mAP50 is below MIN_MAP50) or "yolo".
    step_cm = ground between pictures (None: one picture's length, no overlap);
    length_cm = how much of the column to cover. hold_s = seconds the arm stays
    aimed at each weed before the pump (calibration; default field_run.aim_hold_s). Runs until done or
    `stop` is set; the robot is left stopped, pump pin LOW, arm parked, disarmed.
    progress(status_dict) is called during the run. Returns the summary, also
    written next to the video as .json (and the robot's events as _events.json).

    hardware / vision: tests pass a dry-run robot and a fake backend."""
    import geometry as g
    import planner as pl
    import setup

    video_path = Path(video_path)
    crop = video_path.parent.parent.name
    if hardware is None:
        from hardware import robot
        hardware = robot()
    hw = hardware
    backend = vision or FieldVision(crop, mode, model_path, work_dir=video_path.with_name(video_path.stem + "_pictures"))
    report = progress or (lambda status: None)
    calib = g.Calib(quiet=True)
    try:  # the crop's field model (row band etc.) for geometry.locate, as in the simulator
        fmodel = setup.active_model(crop)[0]
    except Exception:  # noqa: BLE001  (tests use a crop that has none)
        fmodel = None
    rec = getattr(hw, "cfg", {}).get("field_run", {})
    length_cm = min(float(length_cm), float(rec.get("max_cm", 200)))
    cfg = pl.Config(speed=float(rec.get("speed_cm_s", 3)), vision_s=VISION_S.get(mode, 10.0),
                    async_vision=True, out_dir=video_path.parent,
                    max_travel_cm=float(rec["travel_cm"]) if rec.get("travel_cm") is not None else None,
                    stall_s=float(rec.get("stall_s", 2.0)),
                    aim_hold_s=max(0.0, float(rec.get("aim_hold_s", 0) if hold_s is None else hold_s)),
                    reaim_min_conf=float(rec.get("reaim_min_conf", pl.Config.reaim_min_conf)))
    p = pl.Planner(None, backend, calib, fmodel, cfg)
    first_event = len(getattr(hw, "events", []))
    t0 = time.monotonic()
    state = {"phase": "starting", "stopped": False, "error": None}

    def status() -> dict:
        notes = p.log
        return {"phase": p.phases[-1][0] if p.phases else state["phase"],
                "frames": len(p.frames),
                "detected": sum(1 for f in p.frames if f.get("dets") is not None),
                "weeds": len(p.weeds),
                "sprays": sum(1 for e in notes if e["type"] == "spray"),
                "skipped": sum(1 for e in notes if e["type"] == "skip"),
                "usd": round(getattr(backend, "usd", 0.0), 3),
                "seconds": round(time.monotonic() - t0)}

    p.hw = _Guarded(hw, stop, lambda: report(status()))
    live = _live_video(hw, backend, video_path, status)
    if hasattr(hw, "arm"):
        hw.arm()  # the operator confirmed on the page before the run started
    try:
        p.run_stepwise(length_cm, float(step_cm) if step_cm else None)
    except RunStopped:
        state["stopped"] = True
    except Exception as e:  # noqa: BLE001  (the run ends safely and says why)
        state["error"] = f"{type(e).__name__}: {e}"
    finally:
        p.close()
        if hasattr(hw, "safe_stop"):
            hw.safe_stop("field run ended")
            hw.disarm()
        else:
            hw.stop()
    report({**status(), "phase": "saving the videos"})
    live_saved = live.stop() if live is not None else False
    summary = {"crop": crop, "mode": mode, **status(), "length_cm": length_cm, "step_cm": step_cm,
               "aim_hold_s": cfg.aim_hold_s,
               "aims": [e for e in p.log if e["type"] == "hold"],
               "stopped_early": state["stopped"], "error": state["error"],
               "tray_found": getattr(backend, "with_tray", None),
               "dropped_outside_tray": getattr(backend, "dropped_outside", None),
               "weed_threshold": getattr(backend, "min_conf", None),
               "below_threshold": getattr(backend, "below_threshold", None),
               "weed_list": [{k: w.get(k) for k in ("id", "x", "y", "status", "conf")} for w in p.weeds],
               "log": p.log}
    summary["phase"] = "stopped" if state["stopped"] else ("failed" if state["error"] else "done")
    pictures = getattr(backend, "pictures", {})
    summary["video"] = video_path.name if write_video(pictures, video_path) else None
    summary["live_video"] = live.path.name if live_saved else None
    video_path.with_suffix(".json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    if hasattr(hw, "save_events"):
        hw.save_events(video_path.with_name(video_path.stem + "_events.json"), first_event)
    report(summary)
    return summary

"""Train an offline detector (YOLO) for a crop that finished its setup.

    fields/<crop>/train/images/*.jpg    the dataset (recorded by the robot, or imported)
    fields/<crop>/train/labels/*.txt    YOLO labels written by auto-label
                                        (0 = crop, 1 = weed, 2 = unsure: never spray)
    fields/<crop>/train/labels.json     what auto-label decided per image
    fields/<crop>/yolo/best.pt          the trained model
    fields/<crop>/yolo/metrics.json     per-epoch metrics + final result + operating point

Three steps, each its own function so the web page (tools/app.py) and the tests
call the same code:
    record_images()   the robot drives a column and saves FPS photos per second
    auto_label()      SAM + Muse Spark with the crop's saved field model (costs API money)
    train()           YOLO, at most MAX_EPOCHS epochs, metrics reported after every epoch

Auto-labels are the AI's, not a person's. Plants the AI was not sure about (or a
crop/weed call below LABEL_MIN_CONF) become class "unsure" instead of background:
YOLO then learns that those plants exist and the robot treats them like crops.

Split: 70% train / 10% calibration / 20% test. The calibration images choose the
spray settings (weed confidence, crop-veto margin); the test images, never used for
either, report the result. The farmer's own labeled photos are a second, real test.
"""
from __future__ import annotations

import json
import random
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

MAX_EPOCHS = 30
FPS = 2                    # the robot records two images every second
LABEL_MIN_CONF = 0.60      # below this the AI's crop/weed call becomes "unsure"
CLASSES = ["crop", "weed", "unsure"]
PROTECTED = (0, 2)         # class ids the robot must never spray
BASE_MODEL = "yolo11n.pt"  # the small YOLO; downloaded on first use
IMG_EXT = (".jpg", ".jpeg", ".png")
USD_PER_IMAGE = 0.022      # measured on the app's runs: SAM + Spark per photo
IMGSZ = 960                # training/inference size (small weeds need more than 640)
LARGE_WEED = 0.03          # a "large" weed: box long side >= this fraction of the image's long side
WEED_WINDOW = 0.16         # a weed's box is at most this fraction of the image's long side
FRACS = (0.70, 0.10, 0.20)  # train / calibration / test


def tdir(crop: str, root: Path = Path("fields")) -> Path:
    return root / crop / "train"


def images(crop: str, root: Path = Path("fields")) -> list[Path]:
    d = tdir(crop, root) / "images"
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXT) if d.is_dir() else []


def info(crop: str, root: Path = Path("fields")) -> dict:
    imgs = images(crop, root)
    lab = tdir(crop, root) / "labels"
    labeled = sum(1 for p in imgs if (lab / f"{p.stem}.txt").is_file())
    ymd = root / crop / "yolo"
    metrics = None
    if (ymd / "metrics.json").is_file():
        try:
            metrics = json.loads((ymd / "metrics.json").read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {"images": len(imgs), "labeled": labeled,
            "est_usd": round((len(imgs) - labeled) * USD_PER_IMAGE, 2),
            "model": (ymd / "best.pt").is_file(), "metrics": metrics}


# ---------- 1. recording with the robot ----------

def next_image_number(crop: str, root: Path = Path("fields")) -> int:
    """The number for the next img_<n>.jpg: after the highest existing one, so
    nothing is overwritten even if some images were deleted."""
    numbers = [int(p.stem[4:]) for p in images(crop, root) if p.stem.startswith("img_") and p.stem[4:].isdigit()]
    return max(numbers + [len(images(crop, root))]) + 1


def record_images(crop: str, stop: threading.Event, fps: int = FPS,
                  root: Path = Path("fields"), hardware=None, progress=None) -> int:
    """Drive the robot along the column and save `fps` camera frames per second
    into fields/<crop>/train/images (img_<n>.jpg, JPEG quality 92) until `stop` is
    set, the car has gone recording.max_cm, or the wheels stop turning. The
    encoder tick of each frame goes to fields/<crop>/train/ticks.json. Returns
    how many frames were saved.

    With recording.crop_to_tray (the default) each saved image is the tray cut
    out straight and centred (tools/tray.py), the same view the detectors get in
    the field; the full frame goes to train/raw/ and the tray's walls to
    train/trays.json, so the crop can always be redone. A frame where no tray is
    found is saved whole and marked null in trays.json.

    The whole drive is also saved as a real-time video (video_rec.py) in
    train/videos/<time>.mp4, with the tray's walls drawn, when the camera is the
    phone (which can stream).

    The first photo is taken before the car moves, so a missing camera stops the
    recording before anything moves. progress(saved, cm, message) is called after
    every frame and once at the end."""
    import cv2

    if hardware is None:
        from hardware import robot
        hardware = robot()
    hw = hardware
    rec = getattr(hw, "cfg", {}).get("recording", {})
    speed = float(rec.get("speed_cm_s", 3))
    max_cm = float(rec.get("max_cm", 150))
    stall_s = float(rec.get("stall_s", 3))
    cm_per_tick = float(getattr(hw, "cm_per_tick", 1.0))
    report = progress or (lambda saved, cm, message=None: None)

    crop_to_tray = bool(rec.get("crop_to_tray", True))
    out = tdir(crop, root) / "images"
    out.mkdir(parents=True, exist_ok=True)
    raw = tdir(crop, root) / "raw"

    def load_log(path: Path) -> dict:
        try:
            return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        except json.JSONDecodeError:
            return {}

    ticks_path = tdir(crop, root) / "ticks.json"
    trays_path = tdir(crop, root) / "trays.json"
    ticks_log, trays_log = load_log(ticks_path), load_log(trays_path)
    tracker = None
    if crop_to_tray:
        import tray as tray_mod
        tracker = tray_mod.TrayTracker()
        raw.mkdir(parents=True, exist_ok=True)
    number = next_image_number(crop, root)
    saved = 0
    with_tray = 0
    first_event = len(getattr(hw, "events", []))
    start_tick = hw.ticks()
    reason = "Stopped."
    period = 1.0 / fps
    quality = [cv2.IMWRITE_JPEG_QUALITY, 92]

    def save(frame) -> None:
        nonlocal number, saved, with_tray
        name = f"img_{number:05d}.jpg"
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        image = bgr
        if tracker is not None:
            cv2.imwrite(str(raw / name), bgr, quality)
            found = tracker.update(bgr)
            if found is None:
                trays_log[name] = None
            else:
                image = found.rectify(bgr)
                with_tray += 1
                trays_log[name] = {"top_l": round(found.top_l, 1), "bot_l": round(found.bot_l, 1),
                                   "top_r": round(found.top_r, 1), "bot_r": round(found.bot_r, 1),
                                   "offset": round(found.offset, 4), "tilt_deg": round(found.tilt_deg, 2)}
        cv2.imwrite(str(out / name), image, quality)
        ticks_log[name] = hw.ticks()
        number += 1
        saved += 1

    video = None
    video_saved = False
    try:
        save(hw.capture())  # proves the camera works before anything moves
        camera = getattr(hw, "camera", None)
        if camera is not None and hasattr(camera, "subscribe"):
            import video_rec
            import tray as tray_mod
            live_tray = tray_mod.TrayTracker()
            videos = tdir(crop, root) / "videos"
            videos.mkdir(parents=True, exist_ok=True)

            def draw(bgr):
                found = live_tray.update(bgr)
                bgr = found.draw(bgr) if found is not None else bgr
                return video_rec.caption(bgr, [f"Recording {crop}: {saved} photos"])

            video = video_rec.LiveRecorder(camera, videos / f"{time.strftime('%Y%m%d_%H%M%S')}.mp4", draw).start()
        report(saved, 0.0, "Camera OK, driving...")
        hw.drive(speed)
        last_tick, last_move = hw.ticks(), hw.now()
        next_time = hw.now() + period
        while not stop.is_set():
            wait = next_time - hw.now()
            if wait > 0:
                hw.sleep(wait)
            next_time = max(next_time + period, hw.now())
            if stop.is_set():
                break
            save(hw.capture())
            tick = hw.ticks()
            cm = (tick - start_tick) * cm_per_tick
            report(saved, cm, f"Recording: {saved} images, {cm:.0f} cm"
                   + (f", tray found in {with_tray}" if tracker is not None else ""))
            if tick != last_tick:
                last_tick, last_move = tick, hw.now()
            elif hw.now() - last_move > stall_s:
                reason = (f"Stopped: the encoder saw no movement for {stall_s:.0f} s "
                          "(the car reached the wall at the end, is stuck, or the encoder is not counting).")
                break
            if cm >= max_cm:
                reason = f"Stopped at the {max_cm:.0f} cm limit."
                break
    finally:
        hw.stop()
        ticks_path.write_text(json.dumps(ticks_log, indent=1), encoding="utf-8")
        if tracker is not None:
            trays_path.write_text(json.dumps(trays_log, indent=1), encoding="utf-8")
        if hasattr(hw, "save_events"):
            hw.save_events(Path("out") / f"record_{crop}_{time.strftime('%Y%m%d_%H%M%S')}_events.json", first_event)
        if video is not None:
            report(saved, (hw.ticks() - start_tick) * cm_per_tick, "Saving the video...")
            video_saved = video.stop()
    cm = (hw.ticks() - start_tick) * cm_per_tick
    tray_note = f" Tray found and cropped in {with_tray} of them." if tracker is not None else ""
    if video is not None:
        tray_note += f" Video: train/videos/{video.path.name}." if video_saved else " The video could not be saved."
    report(saved, cm, f"{reason} Saved {saved} images over {cm:.0f} cm.{tray_note}")
    return saved


def import_images(crop: str, files: list[tuple[str, bytes]], root: Path = Path("fields")) -> int:
    """Add images to the dataset. files = [(filename, bytes)]. Returns how many."""
    from io import BytesIO

    from PIL import Image
    d = tdir(crop, root) / "images"
    d.mkdir(parents=True, exist_ok=True)
    n = len(images(crop, root))
    added = 0
    for name, raw in files:
        if Path(name).suffix.lower() not in IMG_EXT:
            continue
        try:
            im = Image.open(BytesIO(raw)).convert("RGB")
        except Exception:  # noqa: BLE001  (not an image)
            continue
        n += 1
        im.save(d / f"img_{n:05d}.jpg", quality=92)
        added += 1
    return added


# ---------- 2. auto-label with SAM + Spark (the crop's learned model) ----------

def weed_box(mask, size: tuple[int, int]) -> list[int]:
    """Tight box for a weed (or an unsure plant). SAM masks of tall grass are one
    huge thin shape whose bounding box covers a quarter of the tray and says
    nothing about where the plant is. Take the thickest part of the mask (where
    the blades gather, so on the plant itself) and keep only the mask inside a
    window around it."""
    import cv2
    import numpy as np
    W, H = size
    m = (np.asarray(mask) > 0).astype(np.uint8)
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return [0, 0, 0, 0]
    dist = cv2.distanceTransform(m, cv2.DIST_L2, 3)
    py, px = np.unravel_index(int(dist.argmax()), dist.shape)
    half = int(WEED_WINDOW * max(W, H) / 2)
    x0, x1, y0, y1 = max(0, px - half), min(W, px + half), max(0, py - half), min(H, py + half)
    sub = m[y0:y1, x0:x1]
    sy, sx = np.nonzero(sub)
    pad = 6
    return [max(0, x0 + int(sx.min()) - pad), max(0, y0 + int(sy.min()) - pad),
            min(W, x0 + int(sx.max()) + pad), min(H, y0 + int(sy.max()) + pad)]


def class_of(label: str, conf: float) -> int:
    """AI call -> YOLO class. Anything not a confident crop/weed is 'unsure'."""
    if label in ("crop", "weed") and conf >= LABEL_MIN_CONF:
        return CLASSES.index(label)
    return CLASSES.index("unsure")


def yolo_lines(objs: list[dict], size: tuple[int, int]) -> tuple[list[str], dict]:
    """objs: [{id, box, label, conf}] -> YOLO txt lines + counts per class."""
    W, H = size
    lines, kept, skipped = [], {c: 0 for c in CLASSES}, 0
    for o in objs:
        cls = class_of(o["label"], o["conf"])
        x1, y1, x2, y2 = o["box"]
        x1, x2 = max(0, x1), min(W, x2)
        y1, y2 = max(0, y1), min(H, y2)
        if x2 - x1 < 4 or y2 - y1 < 4:
            skipped += 1
            continue
        lines.append(f"{cls} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} "
                     f"{(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
        kept[CLASSES[cls]] += 1
    return lines, {"kept": kept, "skipped": skipped}


def auto_label(crop: str, cache_dir: Path, log_path: Path, progress=None,
               root: Path = Path("fields"), stop: threading.Event | None = None,
               workers: int = 4) -> dict:
    """Label every unlabeled dataset image with SAM + Spark using the crop's
    saved checklist. progress(done, total, usd, name). Safe to stop and re-run."""
    import setup
    import vision
    from PIL import Image
    model, srefs = setup.active_model(crop)
    if model is None:
        raise RuntimeError(f"{crop} has not finished its setup (no field model).")
    refs = setup.ref_list(srefs, [])
    spray = setup.spray_conf_for(crop)
    t = tdir(crop, root)
    (t / "labels").mkdir(parents=True, exist_ok=True)
    log_f = t / "labels.json"
    log = json.loads(log_f.read_text(encoding="utf-8")) if log_f.is_file() else {}
    todo = [p for p in images(crop, root) if not (t / "labels" / f"{p.stem}.txt").is_file()]
    usd, done = 0.0, 0
    lock = threading.Lock()

    def one(p: Path) -> None:
        nonlocal usd, done
        if stop is not None and stop.is_set():
            return
        r = vision.analyze(p, cache_dir=cache_dir, log_path=log_path, refs=refs,
                           field_model=model, spray_conf=spray)
        size = Image.open(p).size
        boxes = {o["id"]: [int(v) for v in o["box"]] for o in r.objects}
        masks = {o["id"]: o["mask"] for o in r.objects}
        objs = [{"id": x.id, "box": boxes[x.id],
                 "label": {"crop": "crop", "weed": "weed"}.get(x.label, "unsure"),
                 "conf": float(x.confidence)} for x in r.targets if x.id in boxes]
        for o in objs:
            if class_of(o["label"], o["conf"]) != 0:      # weed or unsure: tight box
                o["box"] = weed_box(masks[o["id"]], size)
        lines, stats = yolo_lines(objs, size)
        with lock:
            usd += r.spark_usd + (0 if r.sam_cached else vision.sam_client.USD_PER_IMAGE)
            (t / "labels" / f"{p.stem}.txt").write_text("\n".join(lines), encoding="utf-8")
            log[p.name] = {**stats, "plants": len(objs), "tight": True, "unsure_done": True}
            log_f.write_text(json.dumps(log, indent=1), encoding="utf-8")
            done += 1
            if progress:
                progress(done, len(todo), usd, p.name)

    with ThreadPoolExecutor(max(1, workers)) as ex:
        list(ex.map(one, todo))
    return {"labeled": len(todo), "usd": round(usd, 4), "log": log}


def covered(obj_boxes: list[list[float]], labels: list[list[float]], min_iou: float = 0.5) -> bool:
    """Is this SAM object already a labeled plant? obj_boxes = its full box and its
    tight box; labels = pixel boxes of every labeled plant in the image."""
    return any(_iou(o, lb) >= min_iou for o in obj_boxes for lb in labels)


def add_unsure(crop: str, cache_dir: Path, log_path: Path, root: Path = Path("fields"),
               workers: int = 4) -> int:
    """For images labeled before class 'unsure' existed: every SAM object that has
    no label becomes an unsure box. Free: SAM results are cached and the judge is
    replaced by a stub, so the same objects come back."""
    import setup
    import vision
    from PIL import Image

    def _no_judge(*a, **k):  # the pipeline's own "judge failed" result: no call, no cost
        return {}, {}, {}, 0.0, False, ""

    model, _srefs = setup.active_model(crop)
    t = tdir(crop, root)
    log_f = t / "labels.json"
    log = json.loads(log_f.read_text(encoding="utf-8")) if log_f.is_file() else {}
    todo = [p for p in images(crop, root)
            if (t / "labels" / f"{p.stem}.txt").is_file() and not log.get(p.name, {}).get("unsure_done")]
    lock, changed = threading.Lock(), [0]

    def one(p: Path) -> None:
        W, H = Image.open(p).size
        real = vision.spark_judge
        vision.spark_judge = _no_judge
        try:
            r = vision.analyze(p, cache_dir=cache_dir, log_path=log_path, field_model=model)
        finally:
            vision.spark_judge = real
        f = t / "labels" / f"{p.stem}.txt"
        lines = f.read_text(encoding="utf-8").splitlines()
        labels = [b for _c, b in _parse(lines, W, H)]
        new = []
        for o in r.objects:
            full = [float(v) for v in o["box"]]
            tight = [float(v) for v in weed_box(o["mask"], (W, H))]
            if covered([full, tight], labels):
                continue
            if tight[2] - tight[0] < 4 or tight[3] - tight[1] < 4:
                continue
            new.append(f"2 {(tight[0] + tight[2]) / 2 / W:.6f} {(tight[1] + tight[3]) / 2 / H:.6f} "
                       f"{(tight[2] - tight[0]) / W:.6f} {(tight[3] - tight[1]) / H:.6f}")
        with lock:
            f.write_text("\n".join(lines + new), encoding="utf-8")
            log.setdefault(p.name, {}).update(unsure_done=True, unsure_added=len(new))
            log_f.write_text(json.dumps(log, indent=1), encoding="utf-8")
            changed[0] += 1

    with ThreadPoolExecutor(max(1, workers)) as ex:
        list(ex.map(one, todo))
    return changed[0]


# ---------- 3. train YOLO ----------

def split3(names: list[str], fracs=FRACS, seed: int = 0) -> tuple[list[str], list[str], list[str]]:
    """Deterministic train / calibration / test split. With few images the test and
    calibration sets keep at least 1 image each (when there are 4+)."""
    rng = random.Random(seed)
    names = sorted(names)
    rng.shuffle(names)
    n = len(names)
    n_test = max(1, round(n * fracs[2])) if n >= 3 else 0
    n_cal = max(1, round(n * fracs[1])) if n >= 4 else 0
    return names[n_test + n_cal:], names[n_test:n_test + n_cal], names[:n_test]


def build_yolo_dataset(crop: str, root: Path = Path("fields"), labels: str = "labels") -> Path:
    """Copy labeled images into fields/<crop>/yolo/data/{train,val,test}; write data.yaml.
    'val' is the calibration set (picks the best epoch and the spray settings)."""
    t = tdir(crop, root)
    labeled = [p for p in images(crop, root) if (t / labels / f"{p.stem}.txt").is_file()]
    if len(labeled) < 4:
        raise RuntimeError(f"Need at least 4 labeled images to train (have {len(labeled)}). "
                           "Import more images and run auto label.")
    if not any((t / labels / f"{p.stem}.txt").read_text(encoding="utf-8").strip() for p in labeled):
        raise RuntimeError("Auto label found no plants in any image: nothing to learn.")
    base = root / crop / "yolo" / "data"
    if base.exists():
        shutil.rmtree(base)
    tr, ca, te = split3([p.name for p in labeled])
    for part, names in (("train", tr), ("val", ca), ("test", te)):
        (base / "images" / part).mkdir(parents=True)
        (base / "labels" / part).mkdir(parents=True)
        for n in names:
            stem = Path(n).stem
            shutil.copy(t / "images" / n, base / "images" / part / n)
            shutil.copy(t / labels / f"{stem}.txt", base / "labels" / part / f"{stem}.txt")
    yaml = base / "data.yaml"
    yaml.write_text(f"path: {base.resolve().as_posix()}\ntrain: images/train\nval: images/val\n"
                    f"test: images/test\nnames:\n" + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES)),
                    encoding="utf-8")
    return yaml


def train(crop: str, epochs: int = MAX_EPOCHS, on_epoch=None,
          stop: threading.Event | None = None, root: Path = Path("fields"),
          imgsz: int = IMGSZ, batch: int = 4, labels: str = "labels") -> dict:
    """Train yolo11n for at most MAX_EPOCHS epochs. on_epoch(row) is called after
    every epoch with the metrics; stop.set() ends training after the current epoch.
    Saves best.pt + metrics.json in fields/<crop>/yolo/."""
    from ultralytics import YOLO
    epochs = max(1, min(int(epochs), MAX_EPOCHS))
    yaml = build_yolo_dataset(crop, root, labels)
    out = root / crop / "yolo"
    rows: list[dict] = []
    live = out / "live.json"   # the page reads this, so a run started anywhere shows live

    def write_live(status: str, msg: str = "") -> None:
        live.write_text(json.dumps({"status": status, "epochs": epochs, "rows": rows,
                                    "msg": msg, "updated": time.time()}), encoding="utf-8")

    out.mkdir(parents=True, exist_ok=True)
    write_live("running", "Preparing... (the first epoch also downloads the base model)")
    model = YOLO(BASE_MODEL)

    def cb(tr) -> None:
        if tr.epoch + 1 > epochs:  # the final validation of best.pt is not an epoch
            return
        m = tr.metrics or {}
        li = {}
        if getattr(tr, "tloss", None) is not None:
            li = {k.split("/")[-1]: float(v) for k, v in
                  tr.label_loss_items(tr.tloss, prefix="train").items()}
        row = {"epoch": tr.epoch + 1, "epochs": epochs,
               "box_loss": round(li["box_loss"], 4) if "box_loss" in li else None,
               "cls_loss": round(li["cls_loss"], 4) if "cls_loss" in li else None,
               "precision": round(float(m.get("metrics/precision(B)", 0)), 4),
               "recall": round(float(m.get("metrics/recall(B)", 0)), 4),
               "map50": round(float(m.get("metrics/mAP50(B)", 0)), 4),
               "map50_95": round(float(m.get("metrics/mAP50-95(B)", 0)), 4),
               "t": round(time.time() - t0, 1)}
        rows.append(row)
        write_live("running", f"epoch {row['epoch']}/{epochs}")
        if on_epoch:
            on_epoch(row)
        if stop is not None and stop.is_set():
            tr.stop = True

    model.add_callback("on_fit_epoch_end", cb)
    t0 = time.time()
    model.train(data=str(yaml), epochs=epochs, imgsz=imgsz, batch=batch, workers=0,
                project=str(out.resolve()), name="run", exist_ok=True, patience=0,
                plots=False, verbose=False, device="cpu" if not _cuda() else 0)
    best = out / "run" / "weights" / "best.pt"
    if not best.is_file():
        raise RuntimeError("Training ended without a saved model.")
    shutil.copy(best, out / "best.pt")
    write_live("running", "Scoring on the test images...")
    return finish(crop, root, yaml, rows, epochs, imgsz, time.time() - t0, write_live)


def finish(crop: str, root: Path, yaml: Path, rows: list, epochs: int, imgsz: int,
           seconds: float, write_live=None) -> dict:
    """Everything after training: per-class scores, examples, the spray settings
    chosen on the calibration images and what they give on the test images and on
    the farmer's own photos. Also usable on its own for an already saved model."""
    out = root / crop / "yolo"
    per_class, examples = evaluate(crop, root, yaml)
    op = operating_point(crop, root, imgsz)
    final = max(rows, key=lambda r: r["map50"]) if rows else {}
    res = {"crop": crop, "base_model": BASE_MODEL, "imgsz": imgsz, "epochs_run": len(rows),
           "epochs_max": epochs, "seconds": round(seconds, 1), "best": final, "rows": rows,
           "classes": CLASSES, "per_class": per_class, "examples": examples,
           "operating_point": op,
           "n_train": len(list((out / "data" / "images" / "train").iterdir())),
           "n_calib": len(list((out / "data" / "images" / "val").iterdir())),
           "n_test": len(list((out / "data" / "images" / "test").iterdir())),
           "note": ("Scored on the AI's labels of the 20% test images, so these numbers "
                    "measure agreement with the AI, not with a person. The farmer's own "
                    "labeled photos are the real check."),
           "saved_at": time.strftime("%Y-%m-%d %H:%M")}
    (out / "metrics.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    if write_live:
        write_live("done", f"Saved fields/{crop}/yolo/best.pt")
    return res


def evaluate(crop: str, root: Path, yaml: Path, n_examples: int = 12) -> tuple[list, list]:
    """Score best.pt on the held-out test images (per class) and save example
    pictures: left = the AI's labels (what the model was scored against), right =
    what YOLO predicts. Pictures go to fields/<crop>/yolo/examples/."""
    import cv2
    from ultralytics import YOLO
    out = root / crop / "yolo"
    model = YOLO(str(out / "best.pt"))
    dev = "cpu" if not _cuda() else 0
    v = model.val(data=str(yaml), split="test", imgsz=IMGSZ, verbose=False, plots=False,
                  workers=0, device=dev)
    per_class = []
    for k, ci in enumerate(v.box.ap_class_index):
        per_class.append({"class": CLASSES[int(ci)], "precision": round(float(v.box.p[k]), 4),
                          "recall": round(float(v.box.r[k]), 4),
                          "map50": round(float(v.box.ap50[k]), 4),
                          "map50_95": round(float(v.box.ap[k]), 4)})
    ex_dir = out / "examples"
    shutil.rmtree(ex_dir, ignore_errors=True)
    ex_dir.mkdir(parents=True)
    test_imgs = sorted((out / "data" / "images" / "test").iterdir())[:n_examples]
    colors = {0: (60, 200, 60), 1: (60, 60, 230), 2: (170, 170, 170)}  # BGR
    names = []
    for p in test_imgs:
        im = cv2.imread(str(p))
        h, w = im.shape[:2]
        gt, pr = im.copy(), im.copy()
        lab = out / "data" / "labels" / "test" / f"{p.stem}.txt"
        for c, (x1, y1, x2, y2) in _parse(lab.read_text(encoding="utf-8").splitlines(), w, h):
            cv2.rectangle(gt, (int(x1), int(y1)), (int(x2), int(y2)), colors[c], max(2, w // 300))
        res = model.predict(str(p), conf=0.25, imgsz=IMGSZ, verbose=False, device=dev)[0]
        for b in res.boxes:
            x1, y1, x2, y2 = (int(t) for t in b.xyxy[0].tolist())
            c = int(b.cls[0])
            cv2.rectangle(pr, (x1, y1), (x2, y2), colors[c], max(2, w // 300))
            cv2.putText(pr, f"{CLASSES[c]} {float(b.conf[0]):.2f}", (x1, max(20, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, max(0.6, w / 1800), colors[c], 2, cv2.LINE_AA)
        for img, txt in ((gt, "AI label"), (pr, "YOLO prediction")):
            cv2.putText(img, txt, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 4, cv2.LINE_AA)
            cv2.putText(img, txt, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)
        both = cv2.hconcat([gt, pr])
        both = cv2.resize(both, (both.shape[1] // 2, both.shape[0] // 2))
        cv2.imwrite(str(ex_dir / f"{p.stem}.jpg"), both, [cv2.IMWRITE_JPEG_QUALITY, 85])
        names.append(f"{p.stem}.jpg")
    return per_class, names


# ---------- the spray rule: what the robot does with the detections ----------

def _parse(lines: list[str], W: int, H: int) -> list[tuple[int, list[float]]]:
    out = []
    for line in lines:
        if not line.strip():
            continue
        c, cx, cy, bw, bh = line.split()
        out.append((int(c), [(float(cx) - float(bw) / 2) * W, (float(cy) - float(bh) / 2) * H,
                             (float(cx) + float(bw) / 2) * W, (float(cy) + float(bh) / 2) * H]))
    return out


def _read_boxes(f: Path, W: int, H: int) -> list[tuple[int, list[float]]]:
    return _parse(f.read_text(encoding="utf-8").splitlines(), W, H)


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def _ctr(b):
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


def _inside(pt, b) -> bool:
    return b[0] <= pt[0] <= b[2] and b[1] <= pt[1] <= b[3]


def sprays(pred: list[tuple[int, float, list[float]]], conf_w: float, margin: float,
           crop_conf: float) -> list[list[float]]:
    """The spray rule. pred = [(class, confidence, box)] for one image.
    Spray a predicted weed (confidence >= conf_w) unless its centre lies inside any
    detected crop or unsure box (confidence >= crop_conf) grown by `margin` times that
    box's long side. Crops and unsure plants are never sprayed."""
    guards = [b for c, s, b in pred if c in PROTECTED and s >= crop_conf]
    out = []
    for c, s, b in pred:
        if c != 1 or s < conf_w:
            continue
        ctr = _ctr(b)
        blocked = False
        for g in guards:
            m = margin * max(g[2] - g[0], g[3] - g[1])
            if g[0] - m <= ctr[0] <= g[2] + m and g[1] - m <= ctr[1] <= g[3] + m:
                blocked = True
                break
        if not blocked:
            out.append(b)
    return out


def score(data: list, conf_w: float, margin: float, crop_conf: float) -> dict:
    """data = [(W, H, gt, pred)]. Counts, against the (AI or farmer) labels:
    weeds_found (all, and large ones), crop_hits / unsure_hits (a spray centre inside a
    labeled crop / unsure box: must be 0), false_sprays (on nothing labeled)."""
    found = found_l = total = total_l = ch = uh = fs = n = 0
    for W, H, gt, pred in data:
        sp = sprays(pred, conf_w, margin, crop_conf)
        gw = [b for c, b in gt if c == 1]
        big = [b for b in gw if max(b[2] - b[0], b[3] - b[1]) >= LARGE_WEED * max(W, H)]
        hit = lambda g_: any(_iou(g_, q) >= 0.3 or _inside(_ctr(q), g_) for q in sp)  # noqa: E731
        found += sum(1 for g_ in gw if hit(g_))
        found_l += sum(1 for g_ in big if hit(g_))
        total += len(gw)
        total_l += len(big)
        for q in sp:
            on_crop = any(_inside(_ctr(q), b) for c, b in gt if c == 0)
            on_uns = any(_inside(_ctr(q), b) for c, b in gt if c == 2)
            ch += on_crop
            uh += on_uns and not on_crop
            fs += not on_crop and not on_uns and not any(
                _iou(g_, q) >= 0.3 or _inside(_ctr(q), g_) for g_ in gw)
        n += len(sp)
    pct = lambda a, b: round(100 * a / b, 1) if b else None  # noqa: E731
    return {"weeds_total": total, "weeds_found": found, "weeds_found_pct": pct(found, total),
            "large_total": total_l, "large_found": found_l, "large_found_pct": pct(found_l, total_l),
            "crop_hits": ch, "unsure_hits": uh, "false_sprays": fs, "sprays": n}


def choose(cal: list) -> dict:
    """Pick the spray settings on the calibration images: no crop or unsure plant
    sprayed, then as many large weeds found as possible, then fewest sprays on nothing.
    If no setting reaches zero hits, the one with the fewest hits."""
    best = None
    for conf_w in (0.05, 0.07, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5):
        for margin in (0.0, 0.15, 0.3, 0.5):
            for crop_conf in (0.05, 0.1, 0.25):
                s = score(cal, conf_w, margin, crop_conf)
                key = (s["crop_hits"] + s["unsure_hits"], -s["large_found"], s["false_sprays"])
                if best is None or key < best[0]:
                    best = (key, {"conf_w": conf_w, "margin": margin, "crop_conf": crop_conf}, s)
    return {**best[1], "calibration": best[2]}


def predictions(model, files: list[Path], labels_dir: Path | None, imgsz: int = IMGSZ,
                gt_for=None) -> list:
    """[(W, H, gt, pred)] with predictions at a low confidence, so the spray rule can
    pick its own threshold. gt comes from labels_dir/<stem>.txt, or gt_for(path)."""
    from PIL import Image
    dev = "cpu" if not _cuda() else 0
    out = []
    for p in files:
        W, H = Image.open(p).size
        gt = gt_for(p, W, H) if gt_for else _read_boxes(labels_dir / f"{p.stem}.txt", W, H)
        r = model.predict(str(p), conf=0.03, imgsz=imgsz, verbose=False, device=dev)[0]
        pred = [(int(b.cls[0]), float(b.conf[0]), [float(v) for v in b.xyxy[0].tolist()])
                for b in r.boxes]
        out.append((W, H, gt, pred))
    return out


def real_photos(crop: str, root: Path = Path("fields")) -> list:
    """The farmer's own labeled photos of this crop: [(path, labels file)]."""
    d = root / crop / "labels"
    out = []
    for f in sorted(d.glob("*.labels.json")):
        d_ = json.loads(f.read_text(encoding="utf-8"))
        for ext in (".jpg", ".jpeg", ".png"):
            p = root / crop / "photos" / f"{d_['name']}{ext}"
            if p.is_file():
                out.append((p, d_))
                break
    return out


def operating_point(crop: str, root: Path = Path("fields"), imgsz: int = IMGSZ) -> dict:
    """Choose the spray settings on the calibration images, then report them on the
    test images and on the farmer's own labeled photos (never used to choose)."""
    from ultralytics import YOLO
    out = root / crop / "yolo"
    model = YOLO(str(out / "best.pt"))

    def split_data(part: str) -> list:
        files = sorted((out / "data" / "images" / part).iterdir())
        return predictions(model, files, out / "data" / "labels" / part, imgsz)

    cal = split_data("val")
    op = choose(cal) if cal else {"conf_w": 0.25, "margin": 0.15, "crop_conf": 0.25}
    args = (op["conf_w"], op["margin"], op["crop_conf"])
    test = split_data("test")
    op["test"] = score(test, *args)
    real = real_photos(crop, root)
    if real:
        lab = {p.name: d_ for p, d_ in real}

        def gt_for(p, W, H):
            d_ = lab[p.name]
            sx, sy = W / d_["image_size"][0], H / d_["image_size"][1]
            m = {"crop": 0, "weed": 1}
            return [(m.get(o["label"], 2), [o["box"][0] * sx, o["box"][1] * sy,
                                            o["box"][2] * sx, o["box"][3] * sy])
                    for o in d_["labels"] if o.get("box")]
        rd = predictions(model, [p for p, _ in real], None, imgsz, gt_for)
        op["real_photos"] = {"n": len(real), **score(rd, *args)}
    return op


def _cuda() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except ImportError:
        return False

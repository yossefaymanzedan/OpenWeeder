"""OpenWeeder app: farmer labeling platform (Stitch UI).

Home dashboard lists fields and trained crops. Setup flow per field:
guide -> upload photos -> label some (field_plan) -> learning -> results -> ready.
Pages live in templates/ (converted from the template/ export), images in
static/. Backend logic is unchanged: ExG + SAM "plant", Spark pre-labels,
.labels.json saving with farmer-change counts.

Storage (all plain files, no database):
    fields/<field>/field.json             name, crop, style, selection
    fields/<field>/photos/<name>.jpg      uploaded originals (<=1600 px)
    fields/<field>/labels/<name>.labels.json + masks/   farmer labels
    fields/<field>/setup_log.json         step 4 learning log (rounds, notes)
    crops/<crop>/refs/*.png               confirmed-crop close-ups
    crops/<crop>/crop.json                ref provenance

Run from the repo root:
    python tools/app.py                    platform on http://0.0.0.0:5000/
    python tools/app.py photo.jpg --field F --name N    single-photo mode
    python tools/app.py photo.jpg --field F --terminal  no browser
Double-click run.bat: starts the app, opens a browser, and shares it on a
Cloudflare quick-tunnel URL for your phone.
"""

import argparse
import io
import json
import re
import secrets
import socket
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from flask import (Flask, jsonify, redirect, render_template,
                   render_template_string, request, send_file,
                   send_from_directory)
from PIL import Image, ImageOps

import runner
import setup
import trainer
import vision

ROOT = Path(__file__).resolve().parent.parent
NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,40}")
MAX_DIM = 1600
MAX_UPLOAD = 16 * 1024 * 1024

SESSIONS: dict[str, dict] = {}  # key -> prepare() session + extras
JOBS: dict[str, dict] = {}       # id -> {status, url, error, field, name}
NEXT_AFTER: dict[str, str] = {}  # session key -> URL after save
PREV_AFTER: dict[str, str] = {}  # session key -> previous label URL
RUNS: dict[str, dict] = {}  # field -> live evaluation run state
LEARNING: set[str] = set()  # fields whose step-4 setup thread is running
HARDWARE_CONFIG = Path("calib") / "hardware.json"


def robot_recording_config() -> dict:
    """Recording speed and distance limit from calib/hardware.json, for the page."""
    try:
        cfg = json.loads(HARDWARE_CONFIG.read_text(encoding="utf-8"))
        return {"speed_cm_s": cfg["recording"]["speed_cm_s"], "max_cm": cfg["recording"]["max_cm"],
                "crop_to_tray": cfg["recording"].get("crop_to_tray", True)}
    except (OSError, KeyError, json.JSONDecodeError):
        return {"speed_cm_s": 3, "max_cm": 150, "crop_to_tray": True}


def stop_robot_on_exit() -> None:
    """Ctrl-C, a kill, or closing the terminal leaves the robot safe: the motor
    pins keep their state in pigpiod after this program ends, so stop them first."""
    import atexit
    import signal
    import sys

    def shut_down() -> None:
        import hardware
        if hardware._ROBOT is not None:
            hardware._ROBOT.close()

    atexit.register(shut_down)
    for signum in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, lambda *_: sys.exit(0))


def start_phone_camera() -> None:
    """The robot's camera is a phone page served by tools/phone_cam.py."""
    import phone_cam
    try:
        cfg = json.loads(HARDWARE_CONFIG.read_text(encoding="utf-8")).get("camera", {})
        cam = phone_cam.shared(cfg)
        print("Robot camera: open this on the phone on the robot: " + " or ".join(cam.links))
    except Exception as e:  # noqa: BLE001  (the platform still works without it)
        print(f"Robot camera not started: {e}")


def load_run(field: str) -> dict | None:
    """Last completed evaluation run, or None."""
    p = Path("fields") / field / "runs" / "run.json"
    if not p.is_file():
        return None
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return doc if isinstance(doc, dict) else None


def describe_stage(stage: str, info: dict) -> str:
    if stage == "exg":
        return f"Green filter (ExG {info.get('threshold')})"
    if stage == "sam":
        extra = ", cached" if info.get("cached") else ""
        return (f"SAM found {info.get('objects')} plants "
                f"({info.get('seconds')}s{extra})")
    if stage == "judge":
        return f"Spark judging batch {info.get('batch')}/{info.get('batches')}…"
    if stage == "decide":
        return f"{info.get('plants')} plants decided"
    return stage


def ok_name(s: str) -> bool:
    return bool(NAME_RE.fullmatch(s or ""))


def fields() -> list[str]:
    d = Path("fields")
    return sorted(p.name for p in d.iterdir() if p.is_dir()) if d.is_dir() else []


def crops() -> list[str]:
    d = Path("crops")
    return sorted(p.name for p in d.iterdir() if p.is_dir()) if d.is_dir() else []


def load_field_meta(field: str) -> dict:
    p = Path("fields") / field / "field.json"
    if p.is_file():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {}


def save_field_meta(field: str, meta: dict) -> None:
    d = Path("fields") / field
    d.mkdir(parents=True, exist_ok=True)
    (d / "field.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")


def field_crop(field: str) -> str:
    """The crop name the judge is told. Never silently 'carrot': a field with
    no crop name gets a neutral phrase (the judge then relies on the images)."""
    return (load_field_meta(field).get("crop") or "").strip() or (
        "the main crop planted in the row (its name is not given)")


def field_code(field: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", field).upper()[:6] or "FIELD"


def save_setup_log(field: str, doc: dict) -> None:
    d = Path("fields") / field
    d.mkdir(parents=True, exist_ok=True)
    (d / "setup_log.json").write_text(json.dumps(doc, indent=1),
                                      encoding="utf-8")


REFS_PER_LABEL = 2  # max corrected examples per label fed to each judge call


def load_corrections(field: str) -> dict:
    """Farmer fixes on run overlays: {photo: {obj_id: label}}."""
    p = Path("fields") / field / "runs" / "corrections.json"
    if not p.is_file():
        return {}
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return doc if isinstance(doc, dict) else {}


def save_corrections(field: str, doc: dict) -> None:
    d = Path("fields") / field / "runs"
    d.mkdir(parents=True, exist_ok=True)
    (d / "corrections.json").write_text(json.dumps(doc, indent=1),
                                        encoding="utf-8")


def load_notes(field: str) -> dict:
    """Farmer notes for the judge: {"field": [{round, text, at}],
    "photos": {photo: {"text": str, "plants": {obj_id: str}}}}."""
    p = Path("fields") / field / "runs" / "notes.json"
    doc = {}
    if p.is_file():
        try:
            doc = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            doc = {}
    if not isinstance(doc, dict):
        doc = {}
    doc.setdefault("field", [])
    doc.setdefault("photos", {})
    return doc


def save_notes(field: str, doc: dict) -> None:
    d = Path("fields") / field / "runs"
    d.mkdir(parents=True, exist_ok=True)
    (d / "notes.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False),
                                  encoding="utf-8")


def notes_text(notes: dict, corrections: dict | None = None) -> str:
    """All field, photo and plant notes as bullet lines for the judge prompt.

    Plant notes also ride with their example picture when that fix is picked
    as a reference, but refs are capped, so every note is listed here too.
    """
    corrections = corrections or {}
    lines = [f"- {n['text']}" for n in notes.get("field", []) if n.get("text")]
    for photo, pn in sorted(notes.get("photos", {}).items()):
        if pn.get("text"):
            lines.append(f"- (about photo {photo}) {pn['text']}")
        for oid, txt in sorted(pn.get("plants", {}).items()):
            if txt:
                lab = corrections.get(photo, {}).get(str(oid))
                why = f"a plant the farmer marked {lab}" if lab else "a plant"
                lines.append(f"- ({why}, photo {photo}) {txt}")
    return "\n".join(lines)


def labeled_examples(field: str) -> list:
    """Farmer-verified crop/weed examples from labeled photos.

    Farmer-changed objects first (explicit fixes), then kept pre-labels
    (implicit agreement). Each item: {photo, box, label, tag}.
    """
    out = []
    for p in project_photos(field):
        if not p["labeled"]:
            continue
        lp = Path("fields") / field / "labels" / f"{p['name']}.labels.json"
        if not lp.is_file():
            continue
        try:
            doc = json.loads(lp.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        items = [o for o in doc.get("labels", [])
                 if o.get("label") in ("crop", "weed") and o.get("box")]
        items.sort(key=lambda o: (not o.get("changed", False), o["id"]))
        for o in items:
            out.append({"photo": p["file"], "box": o["box"],
                        "label": o["label"],
                        "tag": f"lab_{p['name']}_{o['id']}"})
    return out


def ref_closeup(photo: Path, box: list, dest: Path) -> Path | None:
    """Untagged padded close-up for judge reference; reused once saved."""
    if dest.is_file():
        return dest
    try:
        img = Image.open(photo).convert("RGB")
    except OSError:
        return None
    x1, y1, x2, y2 = (int(v) for v in box)
    p = int(0.2 * max(x2 - x1, y2 - y1, 1))
    crop = img.crop((max(x1 - p, 0), max(y1 - p, 0),
                     min(x2 + p, img.width), min(y2 + p, img.height)))
    w, h = crop.size
    if min(w, h) < 64:
        s = 64 / min(w, h)
        crop = crop.resize((int(w * s), int(h * s)), Image.LANCZOS)
    dest.parent.mkdir(parents=True, exist_ok=True)
    crop.save(dest)
    return dest


def build_run_refs(field: str, name: str, photos: dict, prev_boxes: dict,
                   corrections: dict, runs_dir: Path,
                   labeled: list | None = None,
                   plant_notes: dict | None = None) -> tuple:
    """Labeled judge refs: fix corrections first, then labeled photos fill.

    Same-photo corrections first, then other photos; within a photo, fixes
    the farmer explained come first. Labeled-photo examples fill remaining
    cap slots. prev_boxes maps (photo, obj_id) -> box from the previous
    round; plant_notes maps the same key to the farmer's note, which rides
    along with the ref. Caps at REFS_PER_LABEL crop + REFS_PER_LABEL weed;
    skip labels never teach. Returns (refs, n_corrections, n_labeled).
    """
    plant_notes = plant_notes or {}
    got: dict[str, list] = {"crop": [], "weed": []}
    n_corr = 0
    names = [name] + [n for n in corrections if n != name]
    for pname in names:
        fixes = sorted(corrections.get(pname, {}).items(),
                       key=lambda kv: (not plant_notes.get((pname, str(kv[0]))),
                                       kv[0]))
        for oid, lab in fixes:
            if lab not in got or len(got[lab]) >= REFS_PER_LABEL:
                continue
            box = prev_boxes.get((pname, str(oid)))
            info = photos.get(pname)
            if box is None or info is None:
                continue
            saved = ref_closeup(Path("fields") / field / "photos" / info["file"],
                                box, runs_dir / f"ref_{pname}_{oid}.png")
            if saved is not None:
                got[lab].append((saved, plant_notes.get((pname, str(oid)), "")))
                n_corr += 1
    for ex in labeled or []:
        lab = ex["label"]
        if lab not in got or len(got[lab]) >= REFS_PER_LABEL:
            continue
        saved = ref_closeup(Path("fields") / field / "photos" / ex["photo"],
                            ex["box"], runs_dir / f"ref_{ex['tag']}.png")
        if saved is not None:
            got[lab].append((saved, ""))
    refs = [(lab, p, note) for lab in ("crop", "weed") for p, note in got[lab]]
    return refs, n_corr, len(refs) - n_corr


def load_setup_log(field: str) -> dict | None:
    """Step 4 learning log; None until step 4 writes it."""
    p = Path("fields") / field / "setup_log.json"
    if not p.is_file():
        return None
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return doc if isinstance(doc, dict) else None


def photo_quality(path: Path) -> tuple[str, str]:
    """(dims 'WxH', quality 'OK'|'A bit dark'|'Blurry'|'Very bright')."""
    try:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        h, w = img.shape
        mean = float(img.mean())
        lap = float(cv2.Laplacian(img, cv2.CV_64F).var())
    except Exception:
        return "?", "OK"
    # Calibrated on CWFID daylight photos (mean ~70, lapvar ~25): those read OK.
    if mean < 50:
        q = "A bit dark"
    elif mean > 210:
        q = "Very bright"
    elif lap < 12:
        q = "Blurry"
    else:
        q = "OK"
    return f"{w}x{h}", q


def project_photos(field: str) -> list[dict]:
    """Uploaded photos with label status, dims, quality; newest first."""
    ph = Path("fields") / field / "photos"
    if not ph.is_dir():
        return []
    selected = set(load_field_meta(field).get("selected", []))
    out = []
    for img in sorted(ph.iterdir(), key=lambda p: p.stat().st_mtime,
                      reverse=True):
        if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        lab = Path("fields") / field / "labels" / f"{img.stem}.labels.json"
        dims, quality = photo_quality(img)
        info = {"name": img.stem, "file": img.name, "labeled": False,
                "changed": 0, "total": 0, "dims": dims, "quality": quality,
                "selected": img.stem in selected}
        if lab.is_file():
            try:
                doc = json.loads(lab.read_text(encoding="utf-8"))
                info.update(labeled=True, changed=doc.get("changed", 0),
                            total=doc.get("total", 0))
            except (json.JSONDecodeError, KeyError):
                pass
        out.append(info)
    return out


def crop_card(crop: str) -> dict:
    d = Path("crops") / crop
    refs = sorted((d / "refs").glob("*.png")) if (d / "refs").is_dir() else []
    fields_seen: list[str] = []
    meta = d / "crop.json"
    if meta.is_file():
        try:
            for r in json.loads(meta.read_text(encoding="utf-8")).get("refs", []):
                if r.get("field") not in fields_seen:
                    fields_seen.append(r.get("field"))
        except json.JSONDecodeError:
            pass
    if fields_seen:
        shown = ", ".join(f for f in fields_seen[:2] if f)
        fields_text = f"{len(fields_seen)} field(s) ({shown})"
    else:
        fields_text = "—"
    return {"name": crop, "n_refs": len(refs),
            "thumbs": [r.name for r in refs[:4]], "fields_text": fields_text}


def crop_info(crop: str) -> dict:
    d = Path("crops") / crop
    refs = sorted((d / "refs").glob("*.png")) if (d / "refs").is_dir() else []
    return {"name": crop, "n_refs": len(refs),
            "files": [r.name for r in refs]}


PLAN_NEW = {"photos": 10, "label": 3}  # you label 3, the AI is tested on 7
PLAN_LEGACY = {"photos": 5, "label": 2}  # fields made before the 10-photo plan


def field_plan(field: str | None) -> dict:
    """Photos per field and how many the farmer labels; the rest test the AI."""
    meta = load_field_meta(field) if field else {}
    plan = meta.get("plan") if field else PLAN_NEW
    if not plan:
        # Field made before plans existed: one already holding 1-5 photos
        # keeps 5/2; an empty one (or one with more) is on 10/3, and that is
        # saved so it can't flip back once photos arrive.
        ph = Path("fields") / field / "photos"
        n = sum(1 for _ in ph.iterdir()) if ph.is_dir() else 0
        if 0 < n <= PLAN_LEGACY["photos"]:
            plan = PLAN_LEGACY
        else:
            plan = PLAN_NEW
            if (Path("fields") / field).is_dir():
                meta["plan"] = PLAN_NEW
                save_field_meta(field, meta)
    return {"photos": plan["photos"], "label": plan["label"],
            "test": plan["photos"] - plan["label"]}


def field_status(field: str, photos: list[dict]) -> str:
    labeled = sum(1 for p in photos if p["labeled"])
    if labeled >= field_plan(field)["label"]:
        return "Labeled"
    if photos:
        return "Labeling"
    return "New"


def field_next(field: str, photos: list[dict]) -> tuple[str, str]:
    """(url, label) of the next setup step for a field."""
    labeled = sum(1 for p in photos if p["labeled"])
    n_label = field_plan(field)["label"]
    if not photos:
        return f"/field/{field}/guide", "Start with the guide"
    if labeled < n_label:
        return f"/field/{field}/upload", f"Upload + pick {n_label}"
    return f"/field/{field}/learning", "Open Learning"


def field_steps(field: str, photos: list[dict]) -> list[dict]:
    labeled = [p for p in photos if p["labeled"]]
    log = load_setup_log(field)
    S = (log or {}).get("setup") or {}
    learned = S.get("state") == "done"
    results = bool(log and log.get("results"))
    done_log = bool(log and log.get("done"))
    plan = field_plan(field)
    order = ["guide", "upload", "label", "learning", "results", "ready"]
    if not photos:
        nxt = "guide"
    elif len(photos) < plan["photos"]:
        nxt = "upload"
    elif len(labeled) < plan["label"]:
        nxt = "label"
    elif not learned and not results:
        nxt = "learning"
    else:
        nxt = "results"
    reached = len(order) if done_log else order.index(nxt)
    if field in LEARNING:
        learn_note = "running"
    elif learned:
        learn_note = f"v{S.get('active_version')}"
    elif S.get("state") == "error":
        learn_note = "error"
    else:
        learn_note = "start"
    reviewed = len(setup.load_reviewed(field))
    defs = [
        ("guide", "Guide", "photo_camera", f"/field/{field}/guide", "do & don't"),
        ("upload", "Upload", "upload", f"/field/{field}/upload",
         f"{len(photos)}/{plan['photos']}"),
        ("label", "Label", "draw", f"/field/{field}/upload",
         f"{len(labeled)}/{plan['label']}"),
        ("learning", "Learning", "psychiatry",
         f"/field/{field}/learning", learn_note),
        ("results", "Results", "fact_check",
         f"/field/{field}/results",
         f"{reviewed}/{plan['test']} reviewed" if learned else
         ("scored" if results else "after learning")),
        ("ready", "Ready", "verified",
         f"/field/{field}/ready", "done" if done_log else "confirm"),
    ]
    out = []
    for i, (key, label, icon, url, note) in enumerate(defs):
        state = "done" if i < reached else ("next" if i == reached else "todo")
        out.append({"key": key, "label": label, "icon": icon, "url": url,
                    "note": note, "state": state})
    return out


def spark_to_pre(label: str) -> str:
    return {"crop": "crop", "weed": "weed"}.get(label, "unsure")


def prepare(photo: Path, field: str, name: str, cache_dir: Path,
            log_path: Path) -> dict:
    """Run ExG + SAM + Spark pre-labels; save masks; return session dict."""
    labels_dir = Path("fields") / field / "labels"
    masks_dir = labels_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    r = vision.analyze(photo, cache_dir=cache_dir, log_path=log_path,
                       crop=field_crop(field))
    img = Image.open(photo).convert("RGB")
    objects = []
    for o, t in zip(r.objects, r.targets):
        Image.fromarray((o["mask"] * 255).astype(np.uint8)).save(
            masks_dir / f"{name}_id{o['id']}.png")
        objects.append({"id": o["id"], "box": [int(v) for v in o["box"]],
                        "pre": spark_to_pre(t.label), "reason": t.reason,
                        "conf": round(float(t.confidence), 2)})
    print(f"{len(objects)} plants, Spark pre-labels done "
          f"({r.spark_seconds:.0f}s, ${r.spark_usd:.4f})", flush=True)
    return {"photo": str(photo), "field": field, "name": name,
            "size": list(img.size), "objects": objects,
            "labels_dir": str(labels_dir)}


def save_labels(session: dict, final: dict[int, str],
                tapped: set[int]) -> tuple[Path, int]:
    """Write the .labels.json file. Returns (path, changed count)."""
    labels_dir = Path(session["labels_dir"])
    pre = {o["id"]: o["pre"] for o in session["objects"]}
    out = []
    for o in session["objects"]:
        oid = o["id"]
        label = final.get(oid, pre[oid])
        if label not in ("crop", "weed", "skip", "unsure"):
            raise ValueError(f"Bad label for {oid}: {label}")
        out.append({"id": oid, "mask": f"masks/{session['name']}_id{oid}.png",
                    "label": label,
                    "decided_by": "farmer" if oid in tapped else "spark",
                    "spark_label": pre[oid], "changed": label != pre[oid],
                    "box": o["box"]})
    changed = sum(1 for o in out if o["changed"])
    doc = {"photo": session["photo"], "field": session["field"],
           "name": session["name"], "image_size": session["size"],
           "labels": out, "changed": changed, "total": len(out)}
    path = labels_dir / f"{session['name']}.labels.json"
    path.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    return path, changed


def save_upload(field: str, name: str, data: bytes) -> tuple[Path, str]:
    """Validate + downscale an uploaded photo. Returns (path, final name)."""
    try:
        img = Image.open(io.BytesIO(data))
        img.verify()
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise ValueError("not a readable image (JPG/PNG)")
    img = ImageOps.exif_transpose(img)
    img.thumbnail((MAX_DIM, MAX_DIM), Image.LANCZOS)
    labels_dir = Path("fields") / field / "labels"
    photos_dir = Path("fields") / field / "photos"
    base, i = name, 2
    while ((labels_dir / f"{name}.labels.json").exists()
           or (photos_dir / f"{name}.jpg").exists()
           or (photos_dir / f"{name}.png").exists()):
        name = f"{base}_{i}"
        i += 1
    photo_path = Path("fields") / field / "photos" / f"{name}.jpg"
    photo_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(photo_path, quality=88)
    return photo_path, name


def photo_stem(filename: str) -> str:
    stem = Path(filename).stem
    return re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_") or "photo"


def remove_photo(field: str, name: str) -> None:
    """Delete a photo and everything derived from it (labels, masks,
    fixes, run entry), and drop it from the labeling selection."""
    base = Path("fields") / field
    for ext in (".jpg", ".jpeg", ".png"):
        (base / "photos" / f"{name}{ext}").unlink(missing_ok=True)
    (base / "labels" / f"{name}.labels.json").unlink(missing_ok=True)
    for m in (base / "labels" / "masks").glob(f"{name}_id*.png"):
        m.unlink()
    for ext in (".png", ".jpg"):
        (base / "runs" / f"{name}.overlay{ext}").unlink(missing_ok=True)
    corr = load_corrections(field)
    if corr.pop(name, None) is not None:
        save_corrections(field, corr)
    run_path = base / "runs" / "run.json"
    if run_path.is_file():
        try:
            doc = json.loads(run_path.read_text(encoding="utf-8"))
            doc["entries"] = [e for e in doc.get("entries", [])
                              if e.get("name") != name]
            run_path.write_text(json.dumps(doc, indent=1), encoding="utf-8")
        except json.JSONDecodeError:
            pass
    RUNS.pop(field, None)
    meta = load_field_meta(field)
    if name in meta.get("selected", []):
        meta["selected"] = [n for n in meta["selected"] if n != name]
        save_field_meta(field, meta)


def padded_crop(img: Image.Image, box: list, pad: float = 0.20) -> Image.Image:
    """Untagged padded close-up (same geometry as vision.closeup)."""
    x1, y1, x2, y2 = (int(v) for v in box)
    p = int(pad * max(x2 - x1, y2 - y1))
    crop = img.crop((max(x1 - p, 0), max(y1 - p, 0),
                     min(x2 + p, img.width), min(y2 + p, img.height)))
    w, h = crop.size
    if min(w, h) < 64:
        s = 64 / min(w, h)
        crop = crop.resize((int(w * s), int(h * s)), Image.LANCZOS)
    return crop


def export_refs(session: dict, final: dict[int, str],
                tapped: set[int]) -> int:
    """Save final-label crop plants as <crop> reference exemplars."""
    crop = session["crop"]
    refs_dir = Path("crops") / crop / "refs"
    refs_dir.mkdir(parents=True, exist_ok=True)
    img = Image.open(session["photo"]).convert("RGB")
    meta_path = refs_dir.parent / "crop.json"
    meta = {"name": crop, "refs": []}
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    have = {r["file"] for r in meta.get("refs", [])}
    pre = {o["id"]: o["pre"] for o in session["objects"]}
    n = 0
    for o in session["objects"]:
        oid = o["id"]
        if final.get(oid, pre[oid]) != "crop":
            continue
        fname = (f"{session['field']}_{session['name']}_id{oid}.png")
        if fname not in have:
            padded_crop(img, o["box"]).save(refs_dir / fname)
            meta.setdefault("refs", []).append(
                {"file": fname, "field": session["field"],
                 "photo": session["name"], "id": oid,
                 "decided_by": "farmer" if oid in tapped else "spark"})
            have.add(fname)
        n += 1
    meta_path.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return n


def start_job(field: str, name: str, photo: Path, crop: str | None,
              cache_dir: Path, log_path: Path,
              extra: dict | None = None) -> tuple[str, str]:
    jid, key = secrets.token_urlsafe(6), secrets.token_urlsafe(8)
    JOBS[jid] = {"status": "working", "url": "", "error": "",
                 "field": field, "name": name}
    args = (jid, key, field, name, photo, crop, cache_dir, log_path,
            extra or {})
    threading.Thread(target=_run_job, args=args, daemon=True).start()
    return jid, key


def _run_job(jid: str, key: str, field: str, name: str, photo: Path,
             crop: str | None, cache_dir: Path, log_path: Path,
             extra: dict) -> None:
    try:
        session = prepare(photo, field, name, cache_dir, log_path)
        session["crop"] = crop
        session.update(extra)
        SESSIONS[key] = session
        JOBS[jid] = {"status": "done", "url": f"/label/{key}", "error": "",
                     "field": field, "name": name}
    except Exception as e:  # noqa: BLE001 - surfaced on the job page
        JOBS[jid] = {"status": "error", "url": "",
                     "error": f"{type(e).__name__}: {e}",
                     "field": field, "name": name}


def _run_eval(field: str, names: list[str], cache_dir: Path,
              log_path: Path) -> None:
    """Evaluate test photos through vision.analyze with live progress."""
    run = RUNS[field]
    runs_dir = Path("fields") / field / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    photos = {p["name"]: p for p in project_photos(field)}
    prev = load_run(field)
    round_no = (prev.get("round", 0) + 1) if prev else 1
    run["round"] = round_no
    corrections = load_corrections(field)
    taught = sum(len(v) for v in corrections.values())
    labeled = labeled_examples(field)
    notes = load_notes(field)
    farmer_notes = notes_text(notes, corrections)
    plant_notes = {(p, str(oid)): txt
                   for p, pn in notes["photos"].items()
                   for oid, txt in pn.get("plants", {}).items() if txt}
    n_notes = len(farmer_notes.splitlines())
    run["notes_used"] = n_notes
    max_lab = 0
    prev_boxes = {}
    if prev:
        for e in prev.get("entries", []):
            for o in e.get("objects", []):
                prev_boxes[(e["name"], str(o["id"]))] = o["box"]
    totals = {"plants": 0, "crop": 0, "weed": 0, "unsure": 0, "spray": 0,
              "protect": 0, "seconds": 0.0, "cost": 0.0}
    t0 = time.time()

    def say(msg: str) -> None:
        run["log"].append(f"{time.time() - t0:5.1f}s  {msg}")
        run["log"] = run["log"][-60:]

    # A learned field model (step 4) turns this into the farmer loop: his
    # fixes + notes first rewrite the checklist, then every photo is judged
    # with the active checklist, his fixed plants and the setup references.
    fmodel, srefs = setup.active_model(field)
    version = None
    try:
        if fmodel:
            stamp = json.dumps([corrections, farmer_notes], sort_keys=True)
            S = setup.get_setup(field)
            if (corrections or farmer_notes) and stamp != S.get("last_farmer_input"):
                say("revising the checklist with your fixes and notes…")
                for e in run["entries"]:
                    e["stage"] = "waiting: checklist is being revised"
                new = setup.farmer_revision(field, prev, corrections, farmer_notes,
                                            plant_notes, cache_dir, log_path)
                S = setup.get_setup(field)
                S["last_farmer_input"] = stamp
                setup.write_setup(field, S)
                if new and new.get("activated"):
                    say(f"checklist v{new['v']}: {new['what_changed']}")
                elif new:
                    say(f"checklist v{new['v']} scored worse on your labeled photos, "
                        f"so v{S['active_version']} stays active; your fixes still "
                        "teach as examples")
                fmodel, srefs = setup.active_model(field)
            version = setup.get_setup(field).get("active_version")
            say(f"judging with checklist v{version}")
        elif n_notes:
            say(f"farmer notes: {n_notes} (sent with every judge call)")
        for i, name in enumerate(names):
            entry = run["entries"][i]
            entry["status"] = "running"
            photo = (Path("fields") / field / "photos" / photos[name]["file"])
            say(f"photo {i + 1}/{len(names)}: {name}")

            def progress(stage: str, info: dict, e=entry) -> None:
                e["stage"] = describe_stage(stage, info)
                if stage == "sam":
                    e["sam_objects"] = info.get("objects", 0)

            refs, n_corr, n_lab = build_run_refs(
                field, name, photos, prev_boxes, corrections, runs_dir,
                None if fmodel else labeled, plant_notes)
            if fmodel:
                refs = setup.ref_list(srefs, refs)
                n_lab = len(refs) - n_corr
            max_lab = max(max_lab, n_lab)
            entry["taught"] = {"corrections": n_corr, "labeled": n_lab}
            if refs:
                bits = []
                if n_corr:
                    bits.append(f"{n_corr} fixes")
                if n_lab:
                    bits.append(f"{n_lab} labeled")
                say(f"{name}: teaching with {len(refs)} examples "
                    f"({' + '.join(bits)})")
            start = time.time()
            if fmodel:
                res = setup.judge_photo(
                    field, name, fmodel, refs,
                    {"cache_dir": cache_dir, "log_path": log_path}, runs_dir,
                    farmer_notes=farmer_notes or None, progress=progress)
                objs, cost = res["objects"], res["usd"]
            else:
                r = vision.analyze(photo, cache_dir=cache_dir, log_path=log_path,
                                   overlay_path=runs_dir / f"{name}.overlay.png",
                                   refs=refs, progress=progress,
                                   farmer_notes=farmer_notes or None,
                                   crop=field_crop(field),
                                   spray_conf=setup.spray_conf_for(field))
                sam_usd = 0.0 if r.sam_cached else vision.sam_client.USD_PER_IMAGE
                cost = round(r.spark_usd + sam_usd, 4)
                box_by_id = {o["id"]: [int(v) for v in o["box"]]
                             for o in r.objects}
                objs = [{"id": t.id, "box": box_by_id.get(t.id, [0, 0, 0, 0]),
                         "label": spark_to_pre(t.label),
                         "conf": round(t.confidence, 2), "reason": t.reason,
                         "decided_by": t.decided_by, "action": t.action,
                         "x": t.x, "y": t.y}
                        for t in r.targets]
            secs = time.time() - start
            counts = {k: sum(1 for o in objs if o["label"] == k)
                      for k in ("crop", "weed", "unsure")}
            spray = sum(1 for o in objs if o["action"] == "spray")
            entry.update(status="done", stage=f"done in {secs:.0f}s",
                         plants=len(objs), crop=counts["crop"],
                         weed=counts["weed"], unsure=counts["unsure"],
                         spray=spray, protect=len(objs) - spray,
                         seconds=round(secs, 1), cost=cost,
                         overlay=True, objects=objs)
            totals["plants"] += len(objs)
            for k in ("crop", "weed", "unsure"):
                totals[k] += counts[k]
            totals["spray"] += spray
            totals["protect"] += len(objs) - spray
            totals["seconds"] = round(time.time() - t0, 1)
            totals["cost"] = round(totals["cost"] + cost, 4)
            say(f"{name}: {len(objs)} plants "
                f"({counts['crop']} crop, {counts['weed']} weed, "
                f"{counts['unsure']} unsure), {spray} spray")
        run["status"] = "done"
        run["totals"] = totals
        run["taught"] = {"corrections": taught, "labeled": max_lab}
        doc = {"status": "done", "round": round_no,
               "ran_at": time.strftime("%Y-%m-%d %H:%M"),
               "source": "farmer" if fmodel else "plain",
               "checklist_version": version,
               "corrections_used": taught, "taught": run["taught"],
               "notes_used": n_notes,
               "entries": run["entries"], "totals": totals}
        (runs_dir / "run.json").write_text(json.dumps(doc, indent=1),
                                           encoding="utf-8")
        log = load_setup_log(field) or {}
        taught_bits = []
        if taught:
            taught_bits.append(f"{taught} fixes")
        if max_lab:
            taught_bits.append(f"{max_lab} labeled examples")
        log["results"] = {
            "frames": len(names), "plants": totals["plants"],
            "crop": totals["crop"], "weed": totals["weed"],
            "unsure": totals["unsure"], "spray": totals["spray"],
            "protect": totals["protect"], "seconds": totals["seconds"],
            "cost": totals["cost"], "ran_at": doc["ran_at"],
            "round": round_no, "corrections": taught,
            "taught": run["taught"],
            "confirmed": False,
            "audit": (f"Round {round_no}: AI evaluated {len(names)} frames: "
                      f"{totals['plants']} plants, {totals['spray']} spray "
                      f"decisions" +
                      (f", taught by {' + '.join(taught_bits)}."
                       if taught_bits else "."))}
        save_setup_log(field, log)
        say(f"done: {totals['plants']} plants, ${totals['cost']:.4f}")
    except Exception as e:  # noqa: BLE001 - surfaced on the results page
        run["status"] = "error"
        run["error"] = f"{type(e).__name__}: {e}"
        say(f"ERROR: {run['error']}")


CROP_PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{crop}} — OpenWeeder</title><style>
body{font-family:sans-serif;margin:8px;background:#222;color:#eee}
a{color:#7db8ff}
.card{background:#333;border-radius:8px;padding:10px;margin:8px 0;max-width:900px}
.row{display:flex;gap:8px;flex-wrap:wrap;align-items:center}
input,select,button{font-size:16px;padding:6px;margin:2px}
.grid{display:flex;gap:8px;flex-wrap:wrap}
.grid figure{margin:0;text-align:center;font-size:12px}
.grid img{width:140px;height:140px;object-fit:cover}
</style></head><body>
<h3><a href="/">OpenWeeder</a> / crop {{crop}} ({{n_refs}} references)</h3>
<div class="card"><form method="post" action="/crop/{{crop}}/add"
enctype="multipart/form-data"><div class="row">
<select name="project" required>
{% for f in fields %}<option value="{{f}}">{{f}}</option>{% endfor %}
</select>
<input type="file" name="photo" accept="image/*" required>
<button>Add training photo</button></div>
<p><small>Label the photo; every plant left as crop becomes a reference.
{% if not fields %}Create a project first.{% endif %}</small></p></form></div>
<div class="grid">
{% for r in refs %}<figure><img src="/cref/{{crop}}/{{r.file}}"
title="{{r.field}}/{{r.photo}} id {{r.id}} ({{r.decided_by}})">
<figcaption>{{r.photo}} #{{r.id}}</figcaption></figure>{% endfor %}
</div></body></html>"""


def create_app(cache_dir: Path, log_path: Path,
               solo_key: str | None = None) -> Flask:
    app = Flask(__name__, template_folder=str(ROOT / "templates"),
                static_folder=str(ROOT / "static"))
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD
    app.config["CACHE_DIR"] = cache_dir
    app.config["LOG_PATH"] = log_path
    # Template edits show on the next page load, no server restart needed.
    app.config["TEMPLATES_AUTO_RELOAD"] = True
    app.jinja_env.auto_reload = True

    @app.get("/favicon.ico")
    def favicon():
        return "", 204

    @app.after_request
    def no_stale_images(resp):
        # Photos and overlays get replaced under the same URL (re-runs,
        # replaced uploads): make the browser re-check each time. Unchanged
        # files still come back as a cheap 304.
        if request.path.startswith(("/p/", "/run_overlay/", "/img/",
                                    "/closeup/", "/cref/", "/setup_img/")):
            resp.headers["Cache-Control"] = "no-cache"
        return resp

    @app.context_processor
    def nav_helpers():
        def nav_steps(field: str) -> list[dict]:
            if field not in fields():
                return []
            return field_steps(field, project_photos(field))
        return {"nav_steps": nav_steps, "field_plan": field_plan}

    def err(msg: str, back: str = "/", code: int = 400):
        return render_template("error.html", msg=msg, back=back), code

    @app.errorhandler(413)
    def too_big(_e):
        return err("Photo too big (max 16 MB).", "/", 413)

    def label_context(key: str) -> dict:
        s = SESSIONS[key]
        w, h = s["size"]
        objects = []
        for o in s["objects"]:
            x1, y1, x2, y2 = o["box"]
            objects.append({
                "id": o["id"], "pre": o["pre"], "reason": o["reason"],
                "conf": f"{o.get('conf', 0) * 100:.0f}%",
                "box": o["box"],
                "top": round(y1 / h * 100, 2), "left": round(x1 / w * 100, 2),
                "w": round((x2 - x1) / w * 100, 2),
                "h": round((y2 - y1) / h * 100, 2),
                "cx": int((x1 + x2) / 2), "cy": int((y1 + y2) / 2)})
        counts = {"crop": 0, "weed": 0, "unsure": 0}
        for o in objects:
            if o["pre"] in counts:
                counts[o["pre"]] += 1
        first = objects[0] if objects else {
            "id": "—", "cx": "—", "cy": "—", "pre": "—",
            "reason": "no plants found", "conf": "—"}
        return {"key": key, "field": s["field"], "name": s["name"],
                "pos": s.get("pos", 1), "total": s.get("total", 1),
                "dims": f"{w}x{h}", "prev_url": PREV_AFTER.get(key),
                "next_label": s.get("next_label", "Done"),
                "crop": s.get("crop"), "objects": objects, "counts": counts,
                "first": first,
                "payload": json.dumps({"key": key, "objects": objects})}

    if solo_key is None:
        @app.get("/")
        def index():
            projs = []
            for f in fields():
                photos = project_photos(f)
                meta = load_field_meta(f)
                url, label = field_next(f, photos)
                log = load_setup_log(f) or {}
                if RUNS.get(f, {}).get("status") == "running":
                    phase = "running"
                elif log.get("done"):
                    phase = "ready"
                else:
                    phase = "learning"
                run = load_run(f) or {}
                projs.append({
                    "phase": phase,
                    "rounds": run.get("round", 0) or 0,
                    "n_label": field_plan(f)["label"],
                    "plants": (run.get("totals") or {}).get("plants", 0),
                    "name": f, "code": field_code(f),
                    "crop": meta.get("crop", "—"),
                    "style": meta.get("planting_style", ""),
                    "status": field_status(f, photos),
                    "photos": len(photos),
                    "labeled": sum(1 for p in photos if p["labeled"]),
                    "changed": sum(p["changed"] for p in photos),
                    "next_url": url, "next_label": label})
            return render_template("home.html", fields=projs,
                                   crops=[crop_card(c) for c in crops()])

        # A "crop" is what used to be called a field here: the workspace where a
        # crop is defined, photographed, labeled and learned (folder fields/<name>).
        # A "field" (folder farm/<name>.json) is the real ground the robot weeds;
        # it only picks which finished crop to use. /field/new is the old address.
        @app.get("/crop/new")
        @app.get("/field/new")
        def field_new_form():
            return render_template("field_new.html", crops=crops())

        @app.post("/crop/new")
        @app.post("/field/new")
        def field_new():
            name = (request.form.get("name") or "").strip()
            crop = (request.form.get("crop") or "").strip()
            style = request.form.get("planting_style") or "single"
            if not ok_name(name):
                return err("Bad name: letters, numbers, - and _ only.",
                           "/crop/new")
            if not crop or len(crop) > 40:
                return err("Crop name is required (max 40 chars).",
                           "/crop/new")
            if style not in ("single", "double", "scattered"):
                return err("Bad planting style.", "/crop/new")
            if (Path("fields") / name).exists():
                return err(f"A crop called '{name}' already exists.", "/crop/new")
            save_field_meta(name, {
                "crop": crop, "planting_style": style,
                "cultivar": (request.form.get("cultivar") or "").strip(),
                "selected": [], "plan": PLAN_NEW})
            return redirect(f"/field/{name}/guide")

        @app.get("/crop/train")
        def train_crop():
            # Placeholder: the robot drives three columns to gather data. Not built.
            out = []
            for f in fields():
                i = trainer.info(f)
                out.append({"name": f, "crop": load_field_meta(f).get("crop", "—"),
                            "ready": trainable(f), **i})
            return render_template("train_crop.html", crops=out)

        def crop_choices() -> list[dict]:
            out = []
            for f in fields():
                ready = (Path("fields") / f / "model.json").is_file()
                out.append({"name": f, "crop": load_field_meta(f).get("crop", "—"),
                            "ready": ready,
                            "note": ("Trained. Detector: stand-in until the offline model "
                                     "is trained." if ready else
                                     "Not trained yet: finish its setup first.")})
            return out

        def farm_all() -> list[dict]:
            out = []
            for f in sorted(Path("farm").glob("*.json")) if Path("farm").is_dir() else []:
                try:
                    out.append(json.loads(f.read_text(encoding="utf-8")))
                except json.JSONDecodeError:
                    pass
            return out

        @app.get("/fields")
        def farm_list():
            return render_template("farm_list.html", farm=farm_all())

        @app.get("/fields/new")
        def farm_new_form():
            return render_template("farm_new.html", crops=crop_choices())

        @app.post("/fields/new")
        def farm_new():
            name = (request.form.get("name") or "").strip()
            crop_name = (request.form.get("crop_name") or "").strip()
            if not ok_name(name):
                return err("Bad field name: letters, numbers, - and _ only.", "/fields/new")
            ok = {c["name"]: c for c in crop_choices() if c["ready"]}
            if crop_name not in ok:
                return err("Pick a crop that has finished training.", "/fields/new")
            try:
                columns = int(request.form.get("columns") or 0)
            except ValueError:
                columns = 0
            if not 1 <= columns <= 500:
                return err("Columns must be a number from 1 to 500.", "/fields/new")
            path = Path("farm") / f"{name}.json"
            if path.exists():
                return err(f"A field called '{name}' already exists.", "/fields/new")
            path.parent.mkdir(exist_ok=True)
            path.write_text(json.dumps({
                "name": name, "crop_name": crop_name, "crop_type": ok[crop_name]["crop"],
                "columns": columns, "created": time.strftime("%Y-%m-%d")}, indent=1),
                encoding="utf-8")
            return redirect(f"/fields/{name}")

        @app.get("/fields/<name>")
        def farm_page(name: str):
            f = next((x for x in farm_all() if x["name"] == name), None)
            if f is None:
                return err("Unknown field.", "/fields", 404)
            return render_template("farm_page.html", f=f)

        # ---- Train model: dataset -> auto label (SAM + Spark) -> YOLO ----
        TRAIN: dict[str, dict] = {}

        def tstate(crop: str) -> dict:
            return TRAIN.setdefault(crop, {
                "collect": {"on": False, "msg": ""},
                "label": {"status": "idle", "done": 0, "total": 0, "usd": 0.0, "msg": ""},
                "fit": {"status": "idle", "rows": [], "msg": ""},
                "stop": {"label": threading.Event(), "fit": threading.Event(),
                         "collect": threading.Event()}})

        def trainable(crop: str) -> bool:
            # "finished its steps" = confirmed (model.json) or the setup kept a checklist
            return crop in fields() and (
                (Path("fields") / crop / "model.json").is_file()
                or setup.active_model(crop)[0] is not None)

        @app.get("/train")
        def train_pick():
            return render_template("train_model.html", crop=None, crops=crop_choices())

        @app.get("/train/<crop>")
        def train_page(crop: str):
            if not trainable(crop):
                return err("Pick a crop that has finished its steps.", "/train", 404)
            imgs = trainer.images(crop)
            vdir = Path("fields") / crop / "train" / "videos"
            rec_videos = sorted((p.name for p in vdir.glob("*.mp4")), reverse=True) if vdir.is_dir() else []
            return render_template("train_model.html", crop=crop, crops=crop_choices(), rec_videos=rec_videos,
                                   info=trainer.info(crop), thumbs=[p.name for p in imgs[-24:]],
                                   max_epochs=trainer.MAX_EPOCHS, fps=trainer.FPS,
                                   recording=robot_recording_config())

        @app.get("/train_video/<crop>/<file>")
        def train_video(crop: str, file: str):
            if not trainable(crop) or not file.endswith(".mp4"):
                return err("Not found.", "/", 404)
            return send_from_directory((Path("fields") / crop / "train" / "videos").resolve(), file)

        @app.get("/train_img/<crop>/<file>")
        def train_img(crop: str, file: str):
            if not trainable(crop):
                return err("Not found.", "/", 404)
            return send_from_directory((trainer.tdir(crop) / "images").resolve(), file)

        @app.get("/train_example/<crop>/<file>")
        def train_example(crop: str, file: str):
            if not trainable(crop):
                return err("Not found.", "/", 404)
            return send_from_directory((Path("fields") / crop / "yolo" / "examples").resolve(), file)

        @app.get("/api/train/<crop>")
        def api_train(crop: str):
            if not trainable(crop):
                return jsonify({"error": "unknown crop"}), 404
            s = tstate(crop)
            fit = s["fit"]
            lv = Path("fields") / crop / "yolo" / "live.json"
            if fit["status"] == "idle" and lv.is_file():  # a run started outside the page
                try:
                    d = json.loads(lv.read_text(encoding="utf-8"))
                    stale = time.time() - d.get("updated", 0) > 900
                    fit = {"status": "idle" if stale and d["status"] == "running" else d["status"],
                           "rows": d["rows"], "epochs": d["epochs"], "msg": d.get("msg", "")}
                except (json.JSONDecodeError, KeyError):
                    pass
            return jsonify({"collect": s["collect"], "label": s["label"], "fit": fit,
                            "info": trainer.info(crop)})

        @app.post("/api/train/<crop>/collect/<action>")
        def train_collect(crop: str, action: str):
            if not trainable(crop):
                return jsonify({"error": "unknown crop"}), 404
            s = tstate(crop)["collect"]
            stop = tstate(crop)["stop"]["collect"]
            if action == "stop":
                stop.set()
                if s.get("on"):
                    s.update(msg="Stopping...")
                return jsonify(s)
            if s.get("on"):
                return jsonify(s)
            if any(r.get("on") for r in FIELD_RUN.values()):
                s.update(msg="The robot is on a field run (Run on the field): stop that first.")
                return jsonify(s)
            # The robot drives while recording, so this runs in the background and
            # the page polls /api/train/<crop> for the status.
            stop.clear()
            s.update(on=True, saved=0, cm=0.0, msg="Starting: asking the phone for a photo...")

            def progress(saved: int, cm: float, message: str | None = None) -> None:
                s.update(saved=saved, cm=round(cm, 1),
                         msg=message or f"Recording: {saved} images, {cm:.0f} cm")

            def work() -> None:
                try:
                    trainer.record_images(crop, stop, progress=progress)
                except Exception as e:  # noqa: BLE001  (shown on the page)
                    s.update(msg=f"Not recorded: {e}")
                finally:
                    s.update(on=False)

            threading.Thread(target=work, daemon=True, name=f"record-{crop}").start()
            return jsonify(s)

        @app.get("/api/camera")
        def camera_status():
            import phone_cam
            cam = phone_cam.running()
            if cam is None:
                return jsonify({"connected": False, "links": [], "error": "The phone camera server is not running."})
            return jsonify(cam.status())

        @app.get("/camera/preview.jpg")
        def camera_preview():
            import phone_cam
            cam = phone_cam.running()
            jpeg = cam.preview() if cam else None
            if jpeg is None:
                return ("", 204)
            return send_file(io.BytesIO(jpeg), mimetype="image/jpeg", max_age=0)

        @app.get("/camera/qr.svg")
        def camera_qr():
            """A QR code of the phone camera link: scan it with the phone."""
            import phone_cam
            import segno
            cam = phone_cam.running()
            if cam is None or not cam.links:
                return ("", 204)
            out = io.BytesIO()
            segno.make(cam.links[0], error="m").save(out, kind="svg", scale=6, border=2)
            return app.response_class(out.getvalue(), mimetype="image/svg+xml")

        tray_view = {"tracker": None, "jpeg": None, "picture": None, "info": {"found": False}}
        tray_lock = threading.Lock()

        def tray_now() -> dict | None:
            """The tray found in the phone's latest preview (each preview once)."""
            import phone_cam
            import tray as tray_mod
            cam = phone_cam.running()
            jpeg = cam.preview() if cam else None
            if jpeg is None:
                return None
            with tray_lock:
                if jpeg is not tray_view["jpeg"]:
                    import hardware
                    frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
                    # Turn and crop it the way capture() does, so the view shows
                    # exactly what the robot's photos will show.
                    try:
                        cam_cfg = json.loads(HARDWARE_CONFIG.read_text(encoding="utf-8"))["camera"]
                    except (OSError, KeyError, ValueError):
                        cam_cfg = {}
                    turn = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
                    if int(cam_cfg.get("rotate_deg", 0)) in turn:
                        frame = cv2.rotate(frame, turn[int(cam_cfg["rotate_deg"])])
                    aspect = float(cam_cfg.get("width", 1296)) / float(cam_cfg.get("height", 972))
                    width = frame.shape[1]
                    frame = hardware.fit_frame(frame, width, max(2, int(round(width / aspect))))
                    tracker = tray_view["tracker"] or tray_mod.TrayTracker()
                    tray_view["tracker"] = tracker
                    found = tracker.update(frame)
                    ok, buf = cv2.imencode(".jpg", tray_mod.view(frame, found, height=360),
                                           [cv2.IMWRITE_JPEG_QUALITY, 80])
                    info = {"found": found is not None}
                    if found is not None:
                        info.update(offset_pct=round(found.offset * 100, 1), tilt_deg=round(found.tilt_deg, 1),
                                    width_pct=round(found.tray_width / found.width * 100))
                    tray_view.update(jpeg=jpeg, picture=buf.tobytes() if ok else None, info=info)
                return tray_view

        @app.get("/camera/tray.jpg")
        def camera_tray_picture():
            state = tray_now()
            if state is None or state["picture"] is None:
                return ("", 204)
            return send_file(io.BytesIO(state["picture"]), mimetype="image/jpeg", max_age=0)

        @app.get("/api/camera/tray")
        def camera_tray():
            state = tray_now()
            return jsonify(state["info"] if state else {"found": False, "camera": False})

        @app.post("/api/train/<crop>/import")
        def train_import(crop: str):
            if not trainable(crop):
                return jsonify({"error": "unknown crop"}), 404
            files = [(f.filename or "", f.read()) for f in request.files.getlist("images")]
            folder = (request.form.get("folder") or "").strip()
            if folder:
                d = Path(folder)
                if not d.is_dir():
                    return jsonify({"error": f"Folder not found: {folder}"}), 400
                files += [(p.name, p.read_bytes()) for p in sorted(d.iterdir())
                          if p.suffix.lower() in trainer.IMG_EXT]
            if not files:
                return jsonify({"error": "No images chosen."}), 400
            n = trainer.import_images(crop, files)
            return jsonify({"added": n, "info": trainer.info(crop)})

        @app.post("/api/train/<crop>/label")
        def train_label(crop: str):
            if not trainable(crop):
                return jsonify({"error": "unknown crop"}), 404
            s = tstate(crop)
            if s["label"]["status"] == "running" or s["fit"]["status"] == "running":
                return jsonify({"error": "Something is already running."}), 409
            if not trainer.images(crop):
                return jsonify({"error": "The dataset is empty."}), 400
            s["stop"]["label"].clear()
            s["label"].update(status="running", done=0, total=0, usd=0.0, msg="Starting...")
            cache, logp = app.config["CACHE_DIR"], app.config["LOG_PATH"]

            def work():
                def prog(i, n, usd, name):
                    s["label"].update(done=i, total=n, usd=round(usd, 4), msg=name)
                try:
                    r = trainer.auto_label(crop, cache, logp, prog, stop=s["stop"]["label"])
                    stopped = s["stop"]["label"].is_set()
                    s["label"].update(status="stopped" if stopped else "done",
                                      usd=r["usd"], msg=f"{r['labeled']} labeled")
                except Exception as e:  # noqa: BLE001
                    s["label"].update(status="error", msg=str(e))

            threading.Thread(target=work, daemon=True).start()
            return jsonify(s["label"])

        @app.post("/api/train/<crop>/fit")
        def train_fit(crop: str):
            if not trainable(crop):
                return jsonify({"error": "unknown crop"}), 404
            s = tstate(crop)
            if s["label"]["status"] == "running" or s["fit"]["status"] == "running":
                return jsonify({"error": "Something is already running."}), 409
            if trainer.info(crop)["labeled"] < 4:
                return jsonify({"error": "Needs at least 4 labeled images: import images "
                                         "and run Auto label first."}), 400
            try:
                epochs = int(request.form.get("epochs") or trainer.MAX_EPOCHS)
            except ValueError:
                epochs = trainer.MAX_EPOCHS
            epochs = max(1, min(epochs, trainer.MAX_EPOCHS))
            s["stop"]["fit"].clear()
            s["fit"].update(status="running", rows=[], epochs=epochs, msg="Preparing the dataset...",
                            started=time.time())

            def work():
                try:
                    s["fit"]["msg"] = "Training (the first epoch also downloads the base model)..."
                    r = trainer.train(crop, epochs, on_epoch=lambda row: s["fit"]["rows"].append(row),
                                      stop=s["stop"]["fit"])
                    s["fit"].update(status="done", msg=f"Saved fields/{crop}/yolo/best.pt",
                                    result=r["best"])
                except Exception as e:  # noqa: BLE001
                    s["fit"].update(status="error", msg=str(e))

            threading.Thread(target=work, daemon=True).start()
            return jsonify(s["fit"])

        @app.post("/api/train/<crop>/stop/<what>")
        def train_stop(crop: str, what: str):
            if not trainable(crop) or what not in ("label", "fit"):
                return jsonify({"error": "unknown"}), 404
            tstate(crop)["stop"][what].set()
            return jsonify({"ok": True})

        # ---- Run on the field: pick a trained model, start/stop the robot ----
        FIELD_RUN: dict[str, dict] = {}

        def run_models() -> list[dict]:
            out = []
            for m in runner.models():
                met = trainer.info(m["crop"]).get("metrics") or {}
                out.append({**m, "map50": (met.get("best") or {}).get("map50")})
            return out

        @app.get("/run-field")
        def run_field_page():
            ms = run_models()
            vids = {m["crop"]: runner.videos(m["crop"]) for m in ms}
            opts = runner.options([f for f in fields() if trainable(f)])
            for o in opts:
                vids.setdefault(o["crop"], runner.videos(o["crop"]))
            cmp_ = {o["crop"]: runner.compare(o["crop"]) for o in opts}
            sums = {crop: {v: runner.summary(crop, v) for v in vs} for crop, vs in vids.items()}
            lives = {crop: {v: runner.live_path(Path(v)).name for v in vs
                            if (Path("fields") / crop / "field_runs" / runner.live_path(Path(v)).name).is_file()}
                     for crop, vs in vids.items()}
            try:
                fr = json.loads(HARDWARE_CONFIG.read_text(encoding="utf-8"))["field_run"]
                field_limit, aim_hold = fr["max_cm"], fr.get("aim_hold_s", 0)
            except (OSError, KeyError, ValueError):
                field_limit, aim_hold = 200, 0
            try:
                picture_cm = round(runner.picture_length_cm())
            except Exception:  # noqa: BLE001  (calibration files unreadable)
                picture_cm = runner.STEP_CM_DEFAULT
            return render_template("run_field.html", options=opts, videos=vids, summaries=sums, lives=lives,
                                   has_videos=any(vids.values()), min_map50=runner.MIN_MAP50,
                                   step_cm=picture_cm, picture_cm=picture_cm, compare=cmp_,
                                   field_limit=field_limit, aim_hold=aim_hold)

        @app.get("/field_run_video/<crop>/<file>")
        def field_run_video(crop: str, file: str):
            if crop not in fields() or not file.endswith(".mp4"):
                return err("Not found.", "/", 404)
            return send_from_directory((Path("fields") / crop / "field_runs").resolve(), file)

        @app.post("/api/field_run/<action>/<crop>")
        def field_run(action: str, crop: str):
            mode = request.args.get("mode") or request.form.get("mode") or "yolo"
            has_model = (Path("fields") / crop / "yolo" / "best.pt").is_file()
            if action not in ("start", "stop") or mode not in ("yolo", "sam_muse"):
                return jsonify({"error": "Unknown action or mode."}), 404
            if not trainable(crop) or (mode == "yolo" and not has_model):
                return jsonify({"error": "That crop has no trained YOLO model (or has not "
                                         "finished its steps)."}), 404
            st = FIELD_RUN.setdefault(crop, {"stop": threading.Event(), "on": False})
            if action == "stop":
                st["stop"].set()
                if st.get("on"):
                    st["msg"] = "Stopping: the robot stops, then the video is saved..."
                return jsonify(field_run_status(crop))
            if st.get("on"):
                return jsonify(field_run_status(crop))
            if any(s["collect"].get("on") for s in TRAIN.values()):
                return jsonify({"msg": "The robot is recording a dataset (Train model): stop that first.",
                                "on": False})
            # The run drives the robot for minutes, so it runs in the background;
            # the page polls /api/field_run/status/<crop>.
            try:
                step = float(request.args.get("step") or runner.picture_length_cm())
                length = float(request.args.get("length") or runner.LENGTH_CM_DEFAULT)
                hold = request.args.get("hold")
                hold = None if hold in (None, "") else min(120.0, max(0.0, float(hold)))
            except ValueError:
                return jsonify({"error": "Step, length and hold must be numbers."}), 400
            model_path = Path("fields") / crop / "yolo" / "best.pt" if has_model else None
            video = runner.new_video_path(crop)
            st["stop"].clear()
            st.update(on=True, msg="Starting: checking the camera and the detector...", status={},
                      video=None, backend=None)
            try:
                backend = runner.FieldVision(crop, mode, model_path,
                                             work_dir=video.with_name(video.stem + "_pictures"))
            except Exception as e:  # noqa: BLE001  (shown on the page; nothing moved)
                st.update(on=False, msg=f"Not started: {e}")
                return jsonify(field_run_status(crop))
            st["backend"] = backend

            def progress(status: dict) -> None:
                st["status"] = {k: status.get(k) for k in (
                    "phase", "frames", "detected", "weeds", "sprays", "skipped", "usd", "seconds")}

            def work() -> None:
                try:
                    result = runner.run_on_field(model_path, st["stop"], video, mode, step_cm=step,
                                                 length_cm=length, vision=backend, progress=progress,
                                                 hold_s=hold)
                    progress(result)
                    why = {"stopped": "Stopped.", "failed": f"Failed: {result.get('error')}."}.get(
                        result["phase"], "Done.")
                    st.update(msg=f"{why} {result['frames']} pictures, {result['sprays']} sprays, "
                                  f"{result['skipped']} skipped, {result['seconds']} s, ${result['usd']}.",
                              video=result.get("video"))
                except Exception as e:  # noqa: BLE001  (shown on the page)
                    st["msg"] = f"Not run: {e}"
                finally:
                    st["on"] = False

            threading.Thread(target=work, daemon=True, name=f"field-run-{crop}").start()
            return jsonify(field_run_status(crop))

        def field_run_status(crop: str) -> dict:
            st = FIELD_RUN.get(crop) or {}
            return {"on": bool(st.get("on")), "msg": st.get("msg", ""), "status": st.get("status", {}),
                    "video": st.get("video")}

        @app.get("/api/field_run/status/<crop>")
        def field_run_state(crop: str):
            return jsonify(field_run_status(crop))

        @app.get("/field_run_live/<crop>.jpg")
        def field_run_live(crop: str):
            backend = (FIELD_RUN.get(crop) or {}).get("backend")
            jpeg = backend.latest_jpeg if backend is not None else None
            if jpeg is None:
                return ("", 204)
            return send_file(io.BytesIO(jpeg), mimetype="image/jpeg", max_age=0)

        @app.get("/field/<field>")
        def field_page(field: str):
            if field not in fields():
                return err("Unknown field.", "/", 404)
            meta = load_field_meta(field)
            photos = project_photos(field)
            return render_template(
                "field.html", field=field, crop=meta.get("crop", "—"),
                style=meta.get("planting_style", ""),
                status=field_status(field, photos), photos=photos,
                labeled_count=sum(1 for p in photos if p["labeled"]),
                steps=field_steps(field, photos))

        @app.get("/field/<field>/guide")
        def guide(field: str):
            if field not in fields():
                return err("Unknown field.", "/", 404)
            return render_template("guide.html", field=field)

        def _render_upload(field: str, error: str = "", code: int = 200):
            photos = project_photos(field)
            meta = load_field_meta(field)
            selected = [n for n in meta.get("selected", [])
                        if n in {p["name"] for p in photos}]
            return render_template("upload.html", field=field, photos=photos,
                                   selected=selected, error=error), code

        @app.get("/field/<field>/upload")
        def upload_form(field: str):
            if field not in fields():
                return err("Unknown field.", "/", 404)
            return _render_upload(field)

        @app.post("/field/<field>/upload")
        def upload(field: str):
            if field not in fields():
                return err("Unknown field.", "/", 404)
            files = [f for f in request.files.getlist("photos") if f.filename]
            have = len(project_photos(field))
            max_photos = field_plan(field)["photos"]
            if have + len(files) > max_photos:
                return _render_upload(
                    field, f"Only {max_photos} photos per field "
                           f"({have} already).", 400)
            if not files:
                return _render_upload(field, "No photos chosen.", 400)
            for f in files:
                try:
                    save_upload(field, photo_stem(f.filename), f.read())
                except ValueError as e:
                    return _render_upload(field, f"{f.filename}: {e}", 400)
            return redirect(f"/field/{field}/upload")

        def _busy(field: str) -> bool:
            return RUNS.get(field, {}).get("status") == "running"

        @app.post("/field/<field>/photo/<name>/remove")
        def photo_remove(field: str, name: str):
            if field not in fields():
                return err("Unknown field.", "/", 404)
            if _busy(field):
                return _render_upload(field, "A test run is in progress — "
                                             "wait for it to finish.", 409)
            if name not in {p["name"] for p in project_photos(field)}:
                return _render_upload(field, "Photo not found.", 404)
            remove_photo(field, name)
            return redirect(f"/field/{field}/upload")

        @app.post("/field/<field>/photo/<name>/replace")
        def photo_replace(field: str, name: str):
            if field not in fields():
                return err("Unknown field.", "/", 404)
            if _busy(field):
                return _render_upload(field, "A test run is in progress — "
                                             "wait for it to finish.", 409)
            if name not in {p["name"] for p in project_photos(field)}:
                return _render_upload(field, "Photo not found.", 404)
            f = request.files.get("photo")
            if not f or not f.filename:
                return _render_upload(field, "No photo chosen.", 400)
            was_selected = name in load_field_meta(field).get("selected", [])
            try:  # save the new one first so a bad file loses nothing
                _, new = save_upload(field, photo_stem(f.filename), f.read())
            except ValueError as e:
                return _render_upload(field, f"{f.filename}: {e}", 400)
            remove_photo(field, name)
            if was_selected:
                meta = load_field_meta(field)
                meta["selected"] = meta.get("selected", []) + [new]
                save_field_meta(field, meta)
            return redirect(f"/field/{field}/upload")

        @app.post("/field/<field>/photos/replace_all")
        def photos_replace_all(field: str):
            if field not in fields():
                return err("Unknown field.", "/", 404)
            if _busy(field):
                return _render_upload(field, "A test run is in progress — "
                                             "wait for it to finish.", 409)
            files = [f for f in request.files.getlist("photos") if f.filename]
            if not files:
                return _render_upload(field, "No photos chosen.", 400)
            if len(files) > field_plan(field)["photos"]:
                return _render_upload(field, f"Pick at most {field_plan(field)['photos']} "
                                             f"photos.", 400)
            blobs = [(f.filename, f.read()) for f in files]
            for fname, data in blobs:  # validate all before deleting anything
                try:
                    Image.open(io.BytesIO(data)).verify()
                except Exception:
                    return _render_upload(
                        field, f"{fname}: not a readable image (JPG/PNG)", 400)
            for p in project_photos(field):
                remove_photo(field, p["name"])
            for fname, data in blobs:
                save_upload(field, photo_stem(fname), data)
            return redirect(f"/field/{field}/upload")

        @app.post("/field/<field>/select")
        def select(field: str):
            if field not in fields():
                return err("Unknown field.", "/", 404)
            photos = {p["name"]: p for p in project_photos(field)}
            picks = [n for n in request.form.getlist("pick") if n in photos]
            n = field_plan(field)["label"]
            if len(picks) != n:
                return err(f"Pick exactly {n} photos to label.",
                           f"/field/{field}/upload")
            meta = load_field_meta(field)
            meta["selected"] = picks
            save_field_meta(field, meta)
            jids = []
            keys = []
            for i, name in enumerate(picks):
                photo = Path("fields") / field / "photos" / f"{name}.jpg"
                if not photo.is_file():
                    photo = Path("fields") / field / "photos" / f"{name}.png"
                extra = {"pos": i + 1, "total": n,
                         "next_label": (f"Next Photo ({i + 2}/{n})"
                                        if i < n - 1 else "Finish → Learning")}
                jid, key = start_job(field, name, photo, None,
                                     app.config["CACHE_DIR"],
                                     app.config["LOG_PATH"], extra)
                jids.append(jid)
                keys.append(key)
            for i, key in enumerate(keys):  # chain: photo 1 -> 2 -> ... -> learning
                NEXT_AFTER[key] = (f"/label/{keys[i + 1]}" if i < n - 1
                                   else f"/field/{field}/learning?auto=1")
                if i:
                    PREV_AFTER[key] = f"/label/{keys[i - 1]}"
            return redirect(f"/job/{','.join(jids)}")

        @app.post("/field/<field>/relabel/<name>")
        def relabel(field: str, name: str):
            if field not in fields():
                return err("Unknown field.", "/", 404)
            photo = Path("fields") / field / "photos" / f"{name}.jpg"
            if not photo.is_file():
                photo = Path("fields") / field / "photos" / f"{name}.png"
            if not photo.is_file():
                return err("Photo not found.", f"/field/{field}", 404)
            extra = {"pos": 1, "total": 1, "next_label": "Done"}
            jid, key = start_job(field, name, photo, None,
                                 app.config["CACHE_DIR"],
                                 app.config["LOG_PATH"], extra)
            NEXT_AFTER[key] = f"/field/{field}"
            return redirect(f"/job/{jid}")

        @app.get("/field/<field>/learning")
        def learning(field: str):
            if field not in fields():
                return err("Unknown field.", "/", 404)
            meta = load_field_meta(field)
            photos = project_photos(field)
            S = setup.get_setup(field)
            ctx = dict(field=field, crop=meta.get("crop", "—"),
                       labeled=[p for p in photos if p["labeled"]],
                       test_photos=[p for p in photos if not p["labeled"]],
                       S=S, running=field in LEARNING,
                       active=next((v for v in S.get("versions", [])
                                    if v["v"] == S.get("active_version")), None))
            if request.args.get("partial"):  # live refresh of the panel only
                return render_template("_learning_panel.html", **ctx)
            return render_template("learning.html", **ctx)

        @app.post("/field/<field>/learn")
        def learn(field: str):
            if field not in fields():
                return err("Unknown field.", "/", 404)
            if field in LEARNING:
                return redirect(f"/field/{field}/learning")
            photos = project_photos(field)
            labeled = [p["name"] for p in photos if p["labeled"]]
            test = [p["name"] for p in photos if not p["labeled"]]
            if len(labeled) < field_plan(field)["label"] or not test:
                return err("Label your photos first, and keep some unlabeled "
                           "ones for the test.", f"/field/{field}/upload")
            if RUNS.get(field, {}).get("status") == "running":
                return err("A test run is in progress — wait for it to finish.",
                           f"/field/{field}/results")
            crop = load_field_meta(field).get("crop") or "crop"
            LEARNING.add(field)
            RUNS.pop(field, None)

            def work():
                try:
                    setup.run_setup(field, crop, labeled, test,
                                    app.config["CACHE_DIR"], app.config["LOG_PATH"])
                except Exception:  # noqa: BLE001 - written to setup_log.json
                    pass
                finally:
                    LEARNING.discard(field)

            threading.Thread(target=work, daemon=True).start()
            return redirect(f"/field/{field}/learning")

        @app.get("/api/learn/<field>")
        def api_learn(field: str):
            S = setup.get_setup(field)
            return jsonify({"running": field in LEARNING,
                            "state": S.get("state", "none"),
                            "message": S.get("message", ""),
                            "progress": S.get("progress", {}),
                            "rounds": len(S.get("rounds", [])),
                            "versions": len(S.get("versions", []))})

        @app.get("/setup_img/<field>/<path:sub>")
        def setup_img(field: str, sub: str):
            """Learning-screen images (round overlays, references) as JPEG,
            scaled to ?w= (default 640) so a round of 7 loads fast."""
            if field not in fields():
                return err("Not found.", "/", 404)
            base = (Path("fields") / field).resolve()
            p = (base / sub).resolve()
            if base not in p.parents or not p.is_file():
                return err("Not found.", "/", 404)
            w = min(max(int(request.args.get("w", 640)), 64), 1600)
            img = Image.open(p).convert("RGB")
            if img.width > w:
                img = img.resize((w, int(img.height * w / img.width)), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=82)
            buf.seek(0)
            return send_file(buf, mimetype="image/jpeg")

        @app.get("/field/<field>/setup_log.json")
        def setup_log_raw(field: str):
            if field not in fields():
                return err("Unknown field.", "/", 404)
            return jsonify({"setup_log": load_setup_log(field),
                            "run": load_run(field),
                            "notes": load_notes(field),
                            "corrections": load_corrections(field)})

        @app.get("/field/<field>/results")
        def results(field: str):
            if field not in fields():
                return err("Unknown field.", "/", 404)
            meta = load_field_meta(field)
            photos = project_photos(field)
            log = load_setup_log(field)
            run = RUNS.get(field)
            if run is None:
                run = load_run(field)
            entries = {}
            if run:
                entries = {e["name"]: e for e in run.get("entries", [])}
            corr = load_corrections(field)
            # Card boxes show the farmer's fixes over the AI's labels.
            card_boxes = {}
            for p in photos:
                e = entries.get(p["name"])
                if not e or not e.get("objects") or "x" not in p["dims"]:
                    continue
                iw, ih = (int(v) for v in p["dims"].split("x"))
                fixes = corr.get(p["name"], {})
                boxes = []
                for o in e["objects"]:
                    x1, y1, x2, y2 = o["box"]
                    lab = fixes.get(str(o["id"]), o["label"])
                    boxes.append({
                        "id": o["id"], "label": lab,
                        "changed": str(o["id"]) in fixes,
                        "spray": o.get("action") == "spray"
                        and str(o["id"]) not in fixes,
                        "l": round(x1 / iw * 100, 2),
                        "t": round(y1 / ih * 100, 2),
                        "w": round((x2 - x1) / iw * 100, 2),
                        "h": round((y2 - y1) / ih * 100, 2),
                        "area": (x2 - x1) * (y2 - y1)})
                card_boxes[p["name"]] = sorted(boxes, key=lambda b: -b["area"])
            notes = load_notes(field)
            t = (run or {}).get("taught", {}) or {}
            taught_bits = []
            if t.get("corrections"):
                taught_bits.append(f"{t['corrections']} fixes")
            if t.get("labeled"):
                taught_bits.append(f"{t['labeled']} labeled examples")
            if (run or {}).get("notes_used"):
                taught_bits.append(f"{run['notes_used']} farmer notes")
            run_taught = (f"Taught by {' + '.join(taught_bits)}"
                          if taught_bits else None)
            return render_template(
                "results.html", field=field, crop=meta.get("crop", "—"),
                test_photos=[p for p in photos if not p["labeled"]],
                results=(log or {}).get("results"),
                spray_conf=setup.spray_conf_for(field),
                run_status=(run or {}).get("status", "none"),
                run_entries=entries,
                run_totals=(run or {}).get("totals"),
                run_round=(run or {}).get("round"),
                corr_counts={n: len(v) for n, v in corr.items()},
                corr_total=sum(len(v) for v in corr.values()),
                run_taught=run_taught, card_boxes=card_boxes,
                field_notes=notes["field"],
                photo_notes={n: pn for n, pn in notes["photos"].items()
                             if pn.get("text") or pn.get("plants")},
                learned=setup.get_setup(field).get("state") == "done",
                learning=field in LEARNING,
                active=_active_version(field),
                reviewed=setup.load_reviewed(field),
                honest=setup.honest_score(field))

        def _active_version(field: str) -> dict | None:
            S = setup.get_setup(field)
            return next((v for v in S.get("versions", [])
                         if v["v"] == S.get("active_version")), None)

        @app.post("/field/<field>/reviewed/<name>")
        def mark_reviewed(field: str, name: str):
            if field not in fields():
                return jsonify({"ok": False, "error": "unknown field"}), 404
            data = request.get_json(silent=True) or {}
            doc = setup.load_reviewed(field)
            if data.get("reviewed"):
                doc[name] = True
            else:
                doc.pop(name, None)
            setup.save_reviewed(field, doc)
            return jsonify({"ok": True, "reviewed": bool(doc.get(name)),
                            "honest": setup.honest_score(field)})

        @app.get("/api/honest/<field>")
        def api_honest(field: str):
            return jsonify(setup.honest_score(field) if field in fields() else None)

        @app.post("/field/<field>/run")
        def run_eval(field: str):
            if field not in fields():
                return err("Unknown field.", "/", 404)
            if RUNS.get(field, {}).get("status") == "running":
                return redirect(f"/field/{field}/results")
            if field in LEARNING:
                return err("The AI is still learning — wait for it to finish.",
                           f"/field/{field}/learning")
            names = [p["name"] for p in project_photos(field)
                     if not p["labeled"]]
            if not names:
                return err("No test photos — upload photos first.",
                           f"/field/{field}/upload")
            try:
                sc = round(min(0.99, max(0.5, float(request.form.get("spray_conf", ""))))
                           , 2)
                meta = load_field_meta(field)
                if meta.get("spray_conf") != sc:
                    meta["spray_conf"] = sc
                    save_field_meta(field, meta)
            except ValueError:
                pass
            note = (request.form.get("note") or "").strip()[:1000]
            if note:
                notes = load_notes(field)
                prev = load_run(field)
                notes["field"].append({
                    "round": ((prev or {}).get("round", 0) or 0) + 1,
                    "text": note, "at": time.strftime("%Y-%m-%d %H:%M")})
                save_notes(field, notes)
            RUNS[field] = {
                "status": "running", "error": "",
                "entries": [{"name": n, "status": "queued", "stage": "queued",
                             "plants": 0, "crop": 0, "weed": 0, "unsure": 0,
                             "spray": 0, "protect": 0, "seconds": 0,
                             "cost": 0, "overlay": False} for n in names],
                "totals": None, "log": ["run started"],
                "started": time.time()}
            threading.Thread(
                target=_run_eval,
                args=(field, names, app.config["CACHE_DIR"],
                      app.config["LOG_PATH"]),
                daemon=True).start()
            return redirect(f"/field/{field}/results")

        @app.get("/api/run/<field>")
        def api_run(field: str):
            run = RUNS.get(field)
            if run is None:
                run = load_run(field)
            if run is None:
                return jsonify({"status": "none"})
            elapsed = round(time.time() - run.get("started", time.time()))
            return jsonify({**run, "elapsed": elapsed})

        @app.get("/run_overlay/<field>/<name>")
        def run_overlay(field: str, name: str):
            # Served as a JPEG copy (~10x smaller than the 1.5 MB PNG) so
            # cards fill in quickly over the phone tunnel; rebuilt whenever
            # a new run rewrites the PNG.
            if field not in fields() or not ok_name(name):
                return err("Not found.", "/", 404)
            d = (Path("fields") / field / "runs").resolve()
            png, jpg = d / f"{name}.overlay.png", d / f"{name}.overlay.jpg"
            if not png.is_file():
                return err("Not found.", "/", 404)
            if not jpg.is_file() or jpg.stat().st_mtime < png.stat().st_mtime:
                Image.open(png).convert("RGB").save(jpg, quality=82)
            return send_file(jpg, mimetype="image/jpeg")

        @app.post("/field/<field>/confirm")
        def confirm(field: str):
            if field not in fields():
                return err("Unknown field.", "/", 404)
            run = RUNS.get(field) or load_run(field)
            if not run or run.get("status") != "done":
                return err("Run the test first, then confirm.",
                           f"/field/{field}/results")
            log = load_setup_log(field) or {}
            res = log.get("results", {})
            res["confirmed"] = True
            res["audit"] = (res.get("audit", "") +
                            " Farmer reviewed the overlays and confirmed.")
            log["results"] = res
            log["done"] = True
            save_setup_log(field, log)
            if setup.get_setup(field).get("state") == "done":
                setup.save_model(field)  # D: the field model every run uses
            return redirect(f"/field/{field}/ready")

        @app.get("/field/<field>/fix/<name>")
        def fix(field: str, name: str):
            if field not in fields():
                return err("Unknown field.", "/", 404)
            run = RUNS.get(field) or load_run(field)
            entry = None
            if run:
                entry = next((e for e in run.get("entries", [])
                              if e["name"] == name
                              and e.get("status") == "done"
                              and e.get("objects")), None)
            if entry is None:
                return err("Run the test first, then fix labels.",
                           f"/field/{field}/results")
            fixed = load_corrections(field).get(name, {})
            pnotes = load_notes(field)["photos"].get(name, {})
            plant_notes = pnotes.get("plants", {})
            photo = Path("fields") / field / "photos" / f"{name}.jpg"
            if not photo.is_file():
                photo = Path("fields") / field / "photos" / f"{name}.png"
            iw, ih = Image.open(photo).size
            objects = []
            for o in entry["objects"]:
                x1, y1, x2, y2 = o["box"]
                objects.append({
                    "id": o["id"], "ai": o["label"],
                    "conf": o.get("conf", 0),
                    "label": fixed.get(str(o["id"]), o["label"]),
                    "note": plant_notes.get(str(o["id"]), ""),
                    "l": round(x1 / iw * 100, 2),
                    "t": round(y1 / ih * 100, 2),
                    "w": round((x2 - x1) / iw * 100, 2),
                    "h": round((y2 - y1) / ih * 100, 2),
                    "area": (x2 - x1) * (y2 - y1)})
            return render_template(
                "fix.html", field=field, name=name, photo_file=photo.name,
                round_no=(run or {}).get("round", 1), objects=objects,
                n_fixed=len(fixed), photo_note=pnotes.get("text", ""))

        @app.post("/field/<field>/fix/<name>")
        def fix_save(field: str, name: str):
            if field not in fields():
                return jsonify({"ok": False, "error": "unknown field"}), 404
            data = request.get_json(silent=True) or {}
            labels = data.get("labels", {})
            clean = {str(k): v for k, v in labels.items()
                     if v in ("crop", "weed", "skip")}
            doc = load_corrections(field)
            doc[name] = clean
            save_corrections(field, doc)
            if "photo_note" in data or "plant_notes" in data:
                notes = load_notes(field)
                plants = {str(k): str(v).strip()[:300]
                          for k, v in (data.get("plant_notes") or {}).items()
                          if str(v).strip()}
                text = str(data.get("photo_note") or "").strip()[:1000]
                if text or plants:
                    notes["photos"][name] = {"text": text, "plants": plants}
                else:
                    notes["photos"].pop(name, None)
                save_notes(field, notes)
            n = sum(1 for v in clean.values() if v in ("crop", "weed"))
            return jsonify({"ok": True, "saved": len(clean), "teach": n})

        @app.get("/field/<field>/ready")
        def ready(field: str):
            if field not in fields():
                return err("Unknown field.", "/", 404)
            meta = load_field_meta(field)
            photos = project_photos(field)
            log = load_setup_log(field)
            return render_template(
                "ready.html", field=field, crop=meta.get("crop", "—"),
                ready=bool(log and log.get("done")),
                results=(log or {}).get("results"),
                labeled=[p for p in photos if p["labeled"]],
                test_photos=[p for p in photos if not p["labeled"]],
                changed_total=sum(p["changed"] for p in photos),
                model=_load_model(field))

        def _load_model(field: str) -> dict | None:
            p = Path("fields") / field / "model.json"
            if not p.is_file():
                return None
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return None

        @app.get("/job/<jids>")
        def job(jids: str):
            ids = [j for j in jids.split(",") if j in JOBS]
            if not ids:
                return err("Unknown job.", "/", 404)
            return render_template("job.html", field=JOBS[ids[0]]["field"],
                                   jids=ids)

        @app.get("/api/job/<jid>")
        def api_job(jid: str):
            if jid not in JOBS:
                return jsonify({"status": "error",
                                "error": "unknown job"}), 404
            return jsonify(JOBS[jid])

        @app.post("/crop")
        def new_crop():
            name = (request.form.get("name") or "").strip().lower()
            if not ok_name(name):
                return err("Bad crop name: letters, numbers, - and _ only.")
            (Path("crops") / name / "refs").mkdir(parents=True, exist_ok=True)
            return redirect(f"/crop/{name}")

        @app.get("/crop/<crop>")
        def crop(crop: str):
            if crop not in crops():
                return err("Unknown crop.", "/", 404)
            meta_path = Path("crops") / crop / "crop.json"
            refs = []
            if meta_path.is_file():
                try:
                    refs = json.loads(meta_path.read_text(
                        encoding="utf-8")).get("refs", [])
                except json.JSONDecodeError:
                    pass
            return render_template_string(
                CROP_PAGE, crop=crop, fields=fields(), refs=refs,
                n_refs=crop_info(crop)["n_refs"])

        @app.post("/crop/<crop>/add")
        def crop_add(crop: str):
            if crop not in crops():
                return err("Unknown crop.", "/", 404)
            field = request.form.get("project") or ""
            if field not in fields():
                return err("Unknown field.", f"/crop/{crop}", 404)
            f = request.files.get("photo")
            if not f or not f.filename:
                return err("No photo chosen.", f"/crop/{crop}")
            try:
                photo, name = save_upload(field, photo_stem(f.filename),
                                          f.read())
            except ValueError as e:
                return err(str(e), f"/crop/{crop}")
            extra = {"pos": 1, "total": 1, "next_label": "Done"}
            jid, key = start_job(field, name, photo, crop,
                                 app.config["CACHE_DIR"],
                                 app.config["LOG_PATH"], extra)
            NEXT_AFTER[key] = f"/crop/{crop}"
            return redirect(f"/job/{jid}")

        @app.get("/p/<field>/<file>")
        def photo_file(field: str, file: str):
            # Absolute: Flask resolves relative dirs against tools/, not CWD.
            d = (Path("fields") / field / "photos").resolve()
            return send_from_directory(d, file)

        @app.get("/cref/<crop>/<file>")
        def crop_file(crop: str, file: str):
            d = (Path("crops") / crop / "refs").resolve()
            return send_from_directory(d, file)
    else:
        @app.get("/")
        def solo_index():
            return render_template("label.html",
                                   **label_context(solo_key))

    @app.get("/label/<key>")
    def label(key: str):
        if key not in SESSIONS:
            return err("Session expired — upload the photo again.", "/")
        return render_template("label.html", **label_context(key))

    @app.get("/img/<key>")
    def session_img(key: str):
        if key not in SESSIONS:
            return jsonify({"error": "unknown session"}), 404
        return send_file(Path(SESSIONS[key]["photo"]).resolve())

    @app.get("/closeup/<key>/<int:oid>")
    def closeup(key: str, oid: int):
        if key not in SESSIONS:
            return jsonify({"error": "unknown session"}), 404
        s = SESSIONS[key]
        o = next((x for x in s["objects"] if x["id"] == oid), None)
        if o is None:
            return jsonify({"error": "unknown plant"}), 404
        img = Image.open(s["photo"]).convert("RGB")
        buf = io.BytesIO()
        padded_crop(img, o["box"]).save(buf, format="JPEG", quality=85)
        buf.seek(0)
        return send_file(buf, mimetype="image/jpeg")

    @app.post("/api/save")
    def save():
        body = request.get_json(force=True)
        try:
            key = body.get("key") or solo_key
            session = SESSIONS[key]
            final = {int(k): v for k, v in body.get("labels", {}).items()}
            tapped = {int(i) for i in body.get("tapped", [])}
            path, changed = save_labels(session, final, tapped)
            refs = (export_refs(session, final, tapped)
                    if session.get("crop") else None)
        except (ValueError, KeyError, AttributeError, TypeError) as e:
            return jsonify({"error": str(e)}), 400
        nxt = NEXT_AFTER.get(key, f"/field/{session['field']}")
        return jsonify({"path": str(path), "changed": changed,
                        "total": len(session["objects"]), "refs": refs,
                        "next": nxt})

    return app


def lan_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return "127.0.0.1"


def run_terminal(session: dict) -> None:
    print("Type c=crop, w=weed, s=skip, Enter keeps Spark's label.")
    final, tapped = {}, set()
    for o in session["objects"]:
        while True:
            ans = input(f"id {o['id']} box {o['box']} "
                        f"spark={o['pre']} ({o['reason']}) [c/w/s/Enter]: "
                        ).strip().lower()
            if ans == "":
                final[o["id"]] = o["pre"]
                break
            if ans in ("c", "w", "s"):
                final[o["id"]] = {"c": "crop", "w": "weed", "s": "skip"}[ans]
                tapped.add(o["id"])
                break
            print("  answer c, w, s, or Enter")
    path, changed = save_labels(session, final, tapped)
    print(f"Saved {path} — {changed}/{len(session['objects'])} changed")


def main() -> None:
    ap = argparse.ArgumentParser(description="OpenWeeder labeling platform.")
    ap.add_argument("photo", type=Path, nargs="?",
                    help="single-photo mode: label just this image")
    ap.add_argument("--field", default="default")
    ap.add_argument("--name", default=None)
    ap.add_argument("--terminal", action="store_true")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--out", type=Path, default=Path("out"))
    args = ap.parse_args()
    cache_dir = args.out / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.out / "api_calls.log"

    if args.photo is not None:
        name = args.name or args.photo.stem
        session = prepare(args.photo, args.field, name, cache_dir, log_path)
        session["crop"] = None
        session.update({"pos": 1, "total": 1, "next_label": "Done"})
        if args.terminal:
            run_terminal(session)
            return
        SESSIONS["solo"] = session
        NEXT_AFTER["solo"] = f"/field/{args.field}"
        app = create_app(cache_dir, log_path, solo_key="solo")
        url = f"http://{lan_ip()}:{args.port}/"
        print(f"Labeling {args.field}/{name} ({len(session['objects'])} plants)")
        print(f"Open on your phone (same Wi-Fi): {url}")
        app.run(host=args.host, port=args.port, threaded=True)
        return

    app = create_app(cache_dir, log_path)
    url = f"http://{lan_ip()}:{args.port}/"
    print("OpenWeeder platform — projects, labeling, crop training.")
    print(f"Open on your phone (same Wi-Fi): {url}")
    start_phone_camera()
    stop_robot_on_exit()
    print("Ctrl-C stops.")
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()

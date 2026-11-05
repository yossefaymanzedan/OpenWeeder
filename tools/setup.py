"""Step 4: field learning. Study the farmer's labeled photos, write a field
checklist, iterate on the unlabeled photos, score honestly, save the model.

A. STUDY   the labeled photos with machine vision (row band, sizes, shape and
           colour features, best single thresholds), then Spark writes a plain
           checklist (<= 150 words) from crop and weed close-ups + the numbers,
           and the clearest 4 crop + 4 weed close-ups become references.
B. ITERATE on the unlabeled photos, max 3 rounds: judge (checklist + refs +
           measured facts; geometry only confirms weeds), Spark reviews the
           overlays and revises the checklist. Every checklist version is
           scored leave-one-out on the labeled photos; the best one is kept.
           Stops when a round changes no labels.
C. FARMER  check on the Results page: honest score (before fixes) on the
           photos marked reviewed; fixes + notes revise the checklist.
D. SAVE    fields/<field>/model.json when the farmer confirms.

Writes fields/<field>/setup_log.json after every step (the Learning screen
reads it live), round images under fields/<field>/setup/, and the run the
Results page shows under fields/<field>/runs/. Logs every API call + cost.

    python tools/setup.py <field>      run A+B from the command line
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from openai import OpenAI
from PIL import Image

import sam_client
import vision

MAX_ROUNDS = 3
REFS_EACH = 4          # reference close-ups per class in every judge call
STUDY_EACH = 8         # close-ups per class shown when writing the checklist
CHECKLIST_WORDS = 150
PARALLEL = 3           # photos judged at once (SAM is cached, Spark repeats)
SPRAY_R = vision.SAFETY_MARGIN_PX
MIN_PLANT_PX = vision.MIN_BLOB_PX
REVIEW_PX = 960        # overlays shown to the reviewer are scaled to this width
PRE = {"crop": "crop", "weed": "weed", "mixed": "unsure"}
FEATURES = ("area", "solidity", "elongation", "leaf_width", "hue", "sat")
FEATURE_NAMES = {"area": "size (green px)", "solidity": "solidity",
                 "elongation": "elongation", "leaf_width": "leaf width (px)",
                 "hue": "hue", "sat": "saturation"}
_LOCK = threading.Lock()


# ---------- files ----------

def spray_conf_for(field: str) -> float:
    """The farmer's weed-confidence threshold for this crop (field.json), else the default."""
    try:
        v = json.loads((fdir(field) / "field.json").read_text(encoding="utf-8")).get("spray_conf")
        return min(0.99, max(0.5, float(v)))
    except (OSError, ValueError, TypeError):
        return vision.SPRAY_CONF


def fdir(field: str) -> Path:
    return Path("fields") / field


def photo_path(field: str, name: str) -> Path:
    for ext in (".jpg", ".jpeg", ".png"):
        p = fdir(field) / "photos" / f"{name}{ext}"
        if p.is_file():
            return p
    raise FileNotFoundError(f"photo {name} not found in {field}")


def load_log(field: str) -> dict:
    p = fdir(field) / "setup_log.json"
    if p.is_file():
        try:
            doc = json.loads(p.read_text(encoding="utf-8"))
            return doc if isinstance(doc, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _save_log(field: str, log: dict) -> None:
    p = fdir(field) / "setup_log.json"
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(log, indent=1, ensure_ascii=False, default=_jsonable),
                   encoding="utf-8")
    tmp.replace(p)


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o).__name__)


def write_setup(field: str, setup: dict) -> None:
    """Persist the setup section of setup_log.json (other keys untouched)."""
    with _LOCK:
        log = load_log(field)
        log["setup"] = setup
        _save_log(field, log)


def get_setup(field: str) -> dict:
    return load_log(field).get("setup") or {}


def rel(p: Path) -> str:
    return Path(p).as_posix()


# ---------- Spark calls for study / review / revise ----------

def spark_json(content: list, schema: dict, name: str, stage: str,
               log_path: Path | None, max_tokens: int = 1500) -> tuple[dict, float]:
    """One structured Spark call, one retry. Returns (data, usd). Raises on
    a second unusable reply (the caller keeps the previous checklist)."""
    client = OpenAI(base_url=sam_client.BASE_URL, api_key=sam_client.api_key())
    total = 0.0
    for attempt in (1, 2):
        started = time.perf_counter()
        resp = client.responses.create(
            model=vision.SPARK_MODEL, max_output_tokens=max_tokens,
            reasoning={"effort": "minimal"},
            text={"format": {"type": "json_schema", "name": name, "schema": schema}},
            input=[{"type": "message", "role": "user", "content": content}])
        seconds = time.perf_counter() - started
        usage = {}
        u = getattr(resp, "usage", None)
        if u is not None:
            d = u.model_dump() if hasattr(u, "model_dump") else dict(u)
            usage = {"input_tokens": d.get("input_tokens", 0),
                     "output_tokens": d.get("output_tokens", 0)}
        usd = (usage.get("input_tokens", 0) / 1e6 * vision.SPARK_USD_PER_1M_IN
               + usage.get("output_tokens", 0) / 1e6 * vision.SPARK_USD_PER_1M_OUT)
        total += usd
        vision.log_call(log_path, {"model": vision.SPARK_MODEL,
                                   "stage": f"setup-{stage}", "attempt": attempt,
                                   "seconds": round(seconds, 1), "usage": usage,
                                   "cached": False, "usd": round(usd, 6)})
        try:
            return json.loads(resp.output_text or ""), total
        except json.JSONDecodeError:
            continue
    raise RuntimeError(f"Spark {stage}: reply unusable twice")


def _text(s: str) -> dict:
    return {"type": "input_text", "text": s}


def _image(img: Image.Image, max_w: int | None = None) -> dict:
    if max_w and img.width > max_w:
        img = img.resize((max_w, int(img.height * max_w / img.width)), Image.LANCZOS)
    return {"type": "input_image", "image_url": vision.jpeg_data_url(img)}


def trim_words(text: str, limit: int = CHECKLIST_WORDS) -> str:
    words = text.split()
    if len(words) <= limit:
        return text.strip()
    cut = " ".join(words[:limit])
    end = max(cut.rfind(". "), cut.rfind(".\n"))
    return (cut[:end + 1] if end > len(cut) // 2 else cut).strip()


# ---------- A1. study the labeled photos ----------

def cutout(img: Image.Image, box, pad: float = 0.20) -> Image.Image:
    """Untagged padded close-up, upscaled to >= 64 px (lesson 7)."""
    x1, y1, x2, y2 = (int(v) for v in box)
    p = int(pad * max(x2 - x1, y2 - y1, 1))
    crop = img.crop((max(x1 - p, 0), max(y1 - p, 0),
                     min(x2 + p, img.width), min(y2 + p, img.height)))
    if min(crop.size) < 64:
        s = 64 / min(crop.size)
        crop = crop.resize((int(crop.width * s), int(crop.height * s)), Image.LANCZOS)
    return crop


def label_objects(field: str, name: str) -> list[dict]:
    """The farmer's plants for one labeled photo: id, label, mask, box."""
    lp = fdir(field) / "labels" / f"{name}.labels.json"
    doc = json.loads(lp.read_text(encoding="utf-8"))
    out = []
    for o in doc.get("labels", []):
        mp = fdir(field) / "labels" / o["mask"]
        if not mp.is_file():
            continue
        m = np.asarray(Image.open(mp).convert("L")) > 127
        out.append({"id": o["id"], "label": o["label"], "mask": m,
                    "box": [int(v) for v in o["box"]],
                    "decided_by": o.get("decided_by", "")})
    return out


def row_fit(crop_px: np.ndarray) -> dict | None:
    """Crop row on ONE photo from its crop pixels: line, half band width
    holding 95% of crop pixels, and how continuous the crop is along it."""
    ys, xs = np.nonzero(crop_px)
    if len(xs) < 50:
        return None
    pts = np.column_stack((xs, ys)).astype(float)
    c = pts.mean(axis=0)
    vals, vecs = np.linalg.eigh(np.cov((pts - c).T))
    d = vecs[:, int(np.argmax(vals))]
    dist = np.abs((pts - c)[:, 0] * d[1] - (pts - c)[:, 1] * d[0])
    proj = (pts - c) @ d
    lo, hi = np.percentile(proj, [2, 98])
    bins = np.arange(lo, hi + 20, 20)
    covered = float((np.histogram(proj, bins)[0] > 0).mean()) if len(bins) > 1 else 0.0
    return {"center": [float(c[0]), float(c[1])], "dir": [float(d[0]), float(d[1])],
            "half_px": float(np.percentile(dist, 95)), "covered": round(covered, 2),
            "angle_deg": round(math.degrees(math.atan2(abs(d[0]), abs(d[1]))), 1)}


def study_photo(field: str, name: str) -> dict:
    """Machine-vision facts for one labeled photo (farmer labels = truth)."""
    img = Image.open(photo_path(field, name)).convert("RGB")
    green, _thr = vision.exg_mask(img)
    hsv = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2HSV)
    objs = label_objects(field, name)
    crop_px = np.zeros_like(green)
    for o in objs:
        if o["label"] == "crop":
            crop_px |= o["mask"] & green
    row = row_fit(crop_px)
    kernel = np.ones((5, 5), np.uint8)
    plants = []
    for o in objs:
        if o["label"] not in ("crop", "weed"):
            continue
        g = o["mask"] & green
        if g.sum() < MIN_PLANT_PX:
            continue
        f = {**vision.shape_features(g), **vision.color_features(hsv, g)}
        ys, xs = np.nonzero(g)
        cx, cy = float(xs.mean()), float(ys.mean())
        others = np.zeros_like(green)
        for o2 in objs:
            if o2["id"] != o["id"]:
                others |= o2["mask"]
        touching = bool((cv2.dilate(o["mask"].astype(np.uint8), kernel).astype(bool)
                         & others).any())
        dist = None
        if row:
            (rx, ry), (dx, dy) = row["center"], row["dir"]
            dist = abs((cx - rx) * dy - (cy - ry) * dx)
        plants.append({"photo": name, "id": o["id"], "label": o["label"],
                       "box": o["box"], "features": f, "touching": touching,
                       "row_dist": dist})
    return {"name": name, "row": row, "plants": plants,
            "size": list(img.size)}


MIN_SPLIT = 0.65  # thresholds weaker than this are not shown to Spark


def best_threshold(crop_vals: list, weed_vals: list) -> dict | None:
    """Best single split between crop and weed on one feature.

    Scored by balanced accuracy (mean of the crop and weed hit rates): with
    4 crops and 21 weeds, plain accuracy rewards "everything is weed".
    """
    vals = sorted(set(crop_vals + weed_vals))
    if len(vals) < 2 or not crop_vals or not weed_vals:
        return None
    nc, nw = len(crop_vals), len(weed_vals)
    best = None
    for a, b in zip(vals, vals[1:]):
        t = (a + b) / 2
        crop_hi = sum(v >= t for v in crop_vals) / nc
        weed_lo = sum(v < t for v in weed_vals) / nw
        for side, bal in (("above", (crop_hi + weed_lo) / 2),
                          ("below", ((1 - crop_hi) + (1 - weed_lo)) / 2)):
            if best is None or bal > best[0]:
                best = (bal, t, side)
    bal, t, side = best
    return {"threshold": round(t, 2), "crop_is": side,
            "balanced_accuracy": round(bal, 2)}


def aggregate(photos: list[dict]) -> dict:
    """Field-level numbers from a set of studied photos."""
    plants = [p for ph in photos for p in ph["plants"]]
    crop = [p for p in plants if p["label"] == "crop"]
    weed = [p for p in plants if p["label"] == "weed"]
    rows = [ph["row"] for ph in photos if ph["row"]]

    def q(vals, k):
        return round(float(np.percentile(vals, k)), 1) if vals else None

    sizes = {lab: {"n": len(ps), "median": q([p["features"]["area"] for p in ps], 50),
                   "p25": q([p["features"]["area"] for p in ps], 25),
                   "p75": q([p["features"]["area"] for p in ps], 75)}
             for lab, ps in (("crop", crop), ("weed", weed))}
    thresholds = {}
    for f in FEATURES:
        t = best_threshold([p["features"][f] for p in crop],
                           [p["features"][f] for p in weed])
        if t:
            thresholds[f] = t
    half = float(np.median([r["half_px"] for r in rows])) if rows else None
    outside = [p for ph in photos for p in ph["plants"]
               if p["label"] == "weed" and ph["row"] and p["row_dist"] is not None
               and p["row_dist"] > ph["row"]["half_px"]]
    covered = float(np.median([r["covered"] for r in rows])) if rows else 0.0
    return {
        "photos": [ph["name"] for ph in photos],
        "n_crop": len(crop), "n_weed": len(weed),
        "band_half_px": round(half, 1) if half else None,
        "band_width_px": round(2 * half, 1) if half else None,
        "band_angle_deg": [r["angle_deg"] for r in rows],
        "crop_layout": "continuous band" if covered >= 0.7 else "separate plants",
        "crop_coverage_along_row": round(covered, 2),
        "sizes": sizes,
        "crop_median_area": sizes["crop"]["median"],
        "thresholds": thresholds,
        "weeds_outside_band_pct": round(100 * len(outside) / len(weed), 1) if weed else None,
    }


def pick_examples(photos: list[dict], label: str, k: int) -> list[dict]:
    """Clearest examples: untouched plants of at least median size, biggest
    first, alternating between photos so every photo contributes. Plants that
    touch others (the farmer still called them crop/weed) come last."""
    per = []
    for ph in photos:
        ps = [p for p in ph["plants"] if p["label"] == label]
        areas = sorted(p["features"]["area"] for p in ps)
        med = areas[len(areas) // 2] if areas else 0
        good = sorted([p for p in ps if not p["touching"] and p["features"]["area"] >= med],
                      key=lambda p: -p["features"]["area"])
        rest = sorted([p for p in ps if p not in good and not p["touching"]],
                      key=lambda p: -p["features"]["area"])
        touching = sorted([p for p in ps if p["touching"]],
                          key=lambda p: -p["features"]["area"])
        per.append(good + rest + touching)
    out = []
    for i in range(max((len(x) for x in per), default=0)):
        for x in per:
            if i < len(x) and len(out) < k:
                out.append(x[i])
    return out


def numbers_text(agg: dict) -> str:
    """Study numbers for Spark. Row/band position is left out on purpose:
    position is handled only by geometry (lesson 4)."""
    s = agg["sizes"]
    lines = [f"Measured on the farmer's {len(agg['photos'])} labeled photos "
             f"({agg['n_crop']} crop plants, {agg['n_weed']} weed plants):",
             f"- size in green pixels: crop median {s['crop']['median']} "
             f"(middle half {s['crop']['p25']}-{s['crop']['p75']}); weed median "
             f"{s['weed']['median']} (middle half {s['weed']['p25']}-{s['weed']['p75']})",
             f"- the crop grows as {agg['crop_layout']}"]
    strong = {f: t for f, t in agg["thresholds"].items()
              if t["balanced_accuracy"] >= MIN_SPLIT}
    for f, t in strong.items():
        side = ">=" if t["crop_is"] == "above" else "<"
        lines.append(f"- {FEATURE_NAMES[f]}: crop if {side} {t['threshold']} "
                     f"(separates crop from weed {round(100 * t['balanced_accuracy'])}% "
                     "of the time; 50% = no better than a coin)")
    if not strong:
        lines.append("- no single measured number separates crop from weed well; "
                     "rely on what the close-ups show")
    if min(agg["n_crop"], agg["n_weed"]) < 5:
        lines.append(f"- careful: only {agg['n_crop']} crop and {agg['n_weed']} weed "
                     "plants were labeled, so these numbers are rough")
    return "\n".join(lines)


CHECKLIST_SCHEMA = {"type": "object", "properties": {
    "checklist": {"type": "string", "maxLength": 1400}},
    "required": ["checklist"], "additionalProperties": False}
REVISE_SCHEMA = {"type": "object", "properties": {
    "checklist": {"type": "string", "maxLength": 1400},
    "what_changed": {"type": "string", "maxLength": 300}},
    "required": ["checklist", "what_changed"], "additionalProperties": False}
REVIEW_SCHEMA = {"type": "object", "properties": {
    "suspects": {"type": "array", "maxItems": 12, "items": {
        "type": "object", "properties": {
            "photo": {"type": "string"}, "id": {"type": "integer"},
            "now": {"type": "string"},
            "should_be": {"type": "string", "enum": ["crop", "weed", "mixed"]},
            "rule": {"type": "string", "maxLength": 80},
            "why": {"type": "string", "maxLength": 80}},
        "required": ["photo", "id", "now", "should_be", "rule", "why"],
        "additionalProperties": False}},
    "summary": {"type": "string", "maxLength": 200}},
    "required": ["suspects", "summary"], "additionalProperties": False}
RULES = ("Rules for the checklist: plain words a farmer understands; at most "
         f"{CHECKLIST_WORDS} words; cover distinguishing traits (leaf shape, edges, "
         "arrangement, colour, size), each weed type you see and how to tell it from "
         "the crop, and which rule to trust most. Never use the plant's position or "
         "distance from the row as a rule; position is checked separately. When "
         "traits conflict, say to protect the plant.")


def write_checklist(field: str, crop: str, photos: dict, agg: dict,
                    log_path) -> tuple[str, float]:
    """A2: Spark studies crop and weed close-ups side by side + the numbers."""
    studied = list(photos.values())
    content = [_text(f"You are an agronomist setting up a weeding robot for one "
                     f"field. The crop is {crop}. Below are close-ups of plants the "
                     f"farmer labeled in this field, then measured numbers. Write the "
                     f"field checklist the robot's judge will use on every plant.\n"
                     + RULES)]
    imgs = {}
    for lab in ("crop", "weed"):
        ex = pick_examples(studied, lab, STUDY_EACH)
        content.append(_text(f"{lab.upper()} plants confirmed by the farmer "
                             f"({len(ex)}):"))
        for p in ex:
            if p["photo"] not in imgs:
                imgs[p["photo"]] = Image.open(
                    photo_path(field, p["photo"])).convert("RGB")
            content.append(_image(cutout(imgs[p["photo"]], p["box"]), 320))
    content.append(_text(numbers_text(agg)))
    content.append(_text('Reply ONLY JSON: {"checklist": "..."}'))
    data, usd = spark_json(content, CHECKLIST_SCHEMA, "FieldChecklist", "checklist",
                           log_path, max_tokens=2500)
    return trim_words(data["checklist"]), usd


def save_refs(field: str, photos: dict, out: Path, k: int = REFS_EACH) -> dict:
    """A3: clearest k crop + k weed close-ups saved as reference files."""
    out.mkdir(parents=True, exist_ok=True)
    refs = {}
    imgs = {}
    for lab in ("crop", "weed"):
        refs[lab] = []
        for i, p in enumerate(pick_examples(list(photos.values()), lab, k)):
            if p["photo"] not in imgs:
                imgs[p["photo"]] = Image.open(photo_path(field, p["photo"])).convert("RGB")
            dst = out / f"{lab}_{i}_{p['photo']}_{p['id']}.png"
            cutout(imgs[p["photo"]], p["box"]).save(dst)
            refs[lab].append({"file": rel(dst), "photo": p["photo"], "id": p["id"]})
    return refs


# ---------- scoring (lesson 9: physical metrics first) ----------

def disk(shape, x: int, y: int) -> np.ndarray:
    m = np.zeros(shape, np.uint8)
    cv2.circle(m, (int(x), int(y)), SPRAY_R, 1, -1)
    return m.astype(bool)


def score(objects: list[dict], crop_px: np.ndarray, weed_px: np.ndarray,
          weed_plants: list[np.ndarray]) -> dict:
    """objects: [{label, action, x, y, mask, truth}] for one photo.

    crop_hits    spray points whose 15 px circle touches truth crop pixels
    weeds_hit    truth weed plants with a spray circle on their green pixels
    weed_sprayed_px / weed_protected_px   weed area sprayed / left in protected plants
    n, ok        object labels vs truth (secondary)
    """
    sprayed = np.zeros(crop_px.shape, bool)
    protected = np.zeros(crop_px.shape, bool)
    hits = 0
    for o in objects:
        if o["action"] == "spray" and o.get("x") is not None:
            d = disk(crop_px.shape, o["x"], o["y"])
            hits += int((d & crop_px).any())
            sprayed |= d
        elif o.get("mask") is not None:
            protected |= o["mask"]
    n = ok = 0
    for o in objects:
        if o.get("truth") in ("crop", "weed", "mixed"):
            n += 1
            ok += int(o["label"] == o["truth"])
    return {"crop_hits": hits,
            "spray_points": sum(1 for o in objects if o["action"] == "spray"),
            "weeds": len(weed_plants),
            "weeds_hit": sum(int((sprayed & w).any()) for w in weed_plants),
            "weed_px": int(weed_px.sum()),
            "weed_sprayed_px": int((sprayed & weed_px).sum()),
            "weed_protected_px": int((protected & weed_px).sum()),
            "n": n, "ok": ok}


def total(scores: list[dict]) -> dict:
    keys = ("crop_hits", "spray_points", "weeds", "weeds_hit", "weed_px",
            "weed_sprayed_px", "weed_protected_px", "n", "ok")
    t = {k: sum(s[k] for s in scores) for k in keys}
    t["weeds_hit_pct"] = round(100 * t["weeds_hit"] / t["weeds"], 1) if t["weeds"] else 0.0
    t["weed_sprayed_pct"] = round(100 * t["weed_sprayed_px"] / t["weed_px"], 1) if t["weed_px"] else 0.0
    t["weed_protected_pct"] = round(100 * t["weed_protected_px"] / t["weed_px"], 1) if t["weed_px"] else 0.0
    t["accuracy_pct"] = round(100 * t["ok"] / t["n"], 1) if t["n"] else 0.0
    return t


def rank(t: dict) -> tuple:
    """Lower is better: fewest crop hits, then most weeds hit, then accuracy."""
    return (t["crop_hits"], -t["weeds_hit_pct"], -t["accuracy_pct"])


def truth_of(mask: np.ndarray, crop_px: np.ndarray, weed_px: np.ndarray) -> str:
    c, w = int((mask & crop_px).sum()), int((mask & weed_px).sum())
    if c + w == 0:
        return "none"
    if c / (c + w) > 0.7:
        return "crop"
    if w / (c + w) > 0.7:
        return "weed"
    return "mixed"


# ---------- running the judge ----------

def model_for(crop: str, checklist: str, agg: dict) -> dict:
    return {"crop": crop, "checklist": checklist,
            "crop_median_area": agg.get("crop_median_area"),
            "band_half_px": agg.get("band_half_px")}


def ref_list(refs: dict, extra: list | None = None) -> list:
    """Judge refs: extra (label, path, note) first (farmer fixes), then the
    setup references, capped at REFS_EACH per class."""
    out, count = [], {"crop": 0, "weed": 0}
    for lab, p, note in extra or []:
        if count.get(lab, REFS_EACH) < REFS_EACH:
            out.append((lab, p, note))
            count[lab] += 1
    for lab in ("crop", "weed"):
        for r in refs.get(lab, []):
            if count[lab] < REFS_EACH and Path(r["file"]).is_file():
                out.append((lab, r["file"], ""))
                count[lab] += 1
    return out


def judge_photo(field: str, name: str, model: dict, refs: list, ctx: dict,
                out_dir: Path, farmer_notes: str | None = None,
                progress=None) -> dict:
    """Run the field model on one photo; save overlay + masks; summary dict."""
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    r = vision.analyze(photo_path(field, name), cache_dir=ctx["cache_dir"],
                       log_path=ctx["log_path"], refs=refs, field_model=model,
                       farmer_notes=farmer_notes, progress=progress,
                       spray_conf=spray_conf_for(field),
                       overlay_path=out_dir / f"{name}.overlay.png")
    masks = {o["id"]: o["mask"] for o in r.objects}
    boxes = {o["id"]: [int(v) for v in o["box"]] for o in r.objects}
    ids = sorted(masks)
    np.savez_compressed(out_dir / f"{name}.masks.npz", ids=np.array(ids, dtype=np.int32),
                        masks=np.stack([masks[i] for i in ids]) if ids
                        else np.zeros((0, 1, 1), bool))
    objs = [{"id": t.id, "box": boxes.get(t.id, [0, 0, 0, 0]), "label": PRE.get(t.label, "unsure"),
             "spark_label": t.label, "conf": round(t.confidence, 2), "reason": t.reason,
             "action": t.action, "decided_by": t.decided_by, "x": t.x, "y": t.y}
            for t in r.targets]
    sam_usd = 0.0 if r.sam_cached else sam_client.USD_PER_IMAGE
    counts = {k: sum(1 for o in objs if o["label"] == k) for k in ("crop", "weed", "unsure")}
    spray = sum(1 for o in objs if o["action"] == "spray")
    return {"name": name, "objects": objs, "geometry": r.geometry,
            "usd": round(r.spark_usd + sam_usd, 4), "seconds": round(time.time() - started, 1),
            "overlay": rel(out_dir / f"{name}.overlay.png"),
            "masks": rel(out_dir / f"{name}.masks.npz"),
            "plants": len(objs), **counts, "spray": spray, "protect": len(objs) - spray}


def load_masks(path: str | Path) -> dict[int, np.ndarray]:
    d = np.load(path)
    return {int(i): m for i, m in zip(d["ids"], d["masks"])}


def leave_one_out(field: str, crop: str, checklist: str, studied: dict, ctx: dict,
                  tag: str) -> dict:
    """Score a checklist on the labeled photos: each photo is judged with refs
    and band numbers from the OTHER labeled photos, against the farmer's labels.
    The checklist itself was written from all of them, so this is a little
    optimistic (noted on the Learning screen)."""
    names = list(studied)
    out_dir = fdir(field) / "setup" / "loo"

    def one(name):
        others = {n: studied[n] for n in names if n != name} or studied
        agg = aggregate(list(others.values()))
        refs = save_refs(field, others, fdir(field) / "setup" / "loo_refs" / name)
        res = judge_photo(field, name, model_for(crop, checklist, agg),
                          ref_list(refs), ctx, out_dir / tag)
        img = Image.open(photo_path(field, name)).convert("RGB")
        green, _ = vision.exg_mask(img)
        truth = label_objects(field, name)
        crop_px = np.zeros_like(green)
        weed_px = np.zeros_like(green)
        weeds = []
        for o in truth:
            g = o["mask"] & green
            if o["label"] == "crop":
                crop_px |= g
            elif o["label"] == "weed":
                weed_px |= g
                if g.sum() >= MIN_PLANT_PX:
                    weeds.append(g)
        masks = load_masks(res["masks"])
        objs = [{**o, "mask": masks.get(o["id"]),
                 "truth": truth_of(masks[o["id"]], crop_px, weed_px)
                 if o["id"] in masks else "none"} for o in res["objects"]]
        for o in objs:  # compare in judge terms: unsure == mixed
            o["label"] = {"unsure": "mixed"}.get(o["label"], o["label"])
        return {"photo": name, **score(objs, crop_px, weed_px, weeds),
                "usd": res["usd"]}

    with ThreadPoolExecutor(PARALLEL) as ex:
        per = list(ex.map(one, names))
    t = total(per)
    t["per_photo"] = per
    t["usd"] = round(sum(p["usd"] for p in per), 4)
    return t


# ---------- B. review + revise ----------

def review(crop: str, checklist: str, results: list[dict], log_path) -> tuple[dict, float]:
    """Spark looks at every overlay and names labels that look wrong + the
    checklist rule that caused each; flags = low confidence or geometry
    disagreement."""
    content = [_text(
        f"You check a weeding robot's work in a {crop} field. Each image is one "
        "photo with every plant boxed and tagged 'id label confidence' (red = will "
        "be sprayed, green = crop, yellow = protected). The robot used this field "
        f"checklist:\n{checklist}\n\nList plants whose label looks wrong, what they "
        "should be, and which checklist rule caused the mistake (or 'missing rule'). "
        "Spraying a crop is the worst mistake; a weed left alone is minor. Look "
        "hardest at the flagged plants. Reply ONLY JSON.")]
    for r in results:
        flags = [o for o in r["objects"]
                 if o["conf"] < vision.SPRAY_CONF or o["label"] == "unsure"]
        dis = {d["id"] for d in r["geometry"].get("disagree", [])}
        lines = [f"Photo {r['name']}:"]
        for o in r["objects"]:
            mark = []
            if o in flags:
                mark.append("low confidence")
            if o["id"] in dis:
                mark.append("geometry disagrees")
            lines.append(f"  #{o['id']} {o['spark_label']} {o['conf']:.2f} "
                         f"{o['action']} ({o['reason']})"
                         + (f"  FLAG: {', '.join(mark)}" if mark else ""))
        content.append(_text("\n".join(lines)))
        content.append(_image(Image.open(r["overlay"]).convert("RGB"), REVIEW_PX))
    return spark_json(content, REVIEW_SCHEMA, "ChecklistReview", "review",
                      log_path, max_tokens=4000)


def revise(crop: str, checklist: str, suspects: list, log_path,
           farmer_fixes: str = "", farmer_notes: str = "",
           fix_images: list | None = None) -> tuple[dict, float]:
    """Spark rewrites the checklist from review suspects and/or farmer fixes."""
    content = [_text(
        f"You maintain the field checklist a weeding robot's judge uses in a {crop} "
        f"field. Current checklist:\n{checklist}\n\n" + RULES + " Keep rules that "
        "work; change only what the evidence below shows is wrong or missing. "
        "Then say in at most 30 plain words what you changed.")]
    if suspects:
        content.append(_text("A reviewer thinks these labels were wrong:\n" + "\n".join(
            f"- photo {s['photo']} plant #{s['id']}: labeled {s['now']}, should be "
            f"{s['should_be']} (rule: {s['rule']}; {s['why']})" for s in suspects)))
    if farmer_fixes:
        content.append(_text("The FARMER corrected these plants (the farmer is "
                             "always right):\n" + farmer_fixes))
        for lab, img in fix_images or []:
            content.append(_text(f"Plant the farmer says is {lab}:"))
            content.append(_image(img, 320))
    if farmer_notes:
        content.append(_text("The farmer's notes about this field:\n" + farmer_notes))
    content.append(_text('Reply ONLY JSON: {"checklist": "...", "what_changed": "..."}'))
    data, usd = spark_json(content, REVISE_SCHEMA, "ChecklistRevision", "revise",
                           log_path, max_tokens=2500)
    data["checklist"] = trim_words(data["checklist"])
    return data, usd


def labels_changed(prev: list[dict] | None, cur: list[dict]) -> int | None:
    if prev is None:
        return None
    before = {(r["name"], o["id"]): (o["label"], o["action"])
              for r in prev for o in r["objects"]}
    return sum(1 for r in cur for o in r["objects"]
               if before.get((r["name"], o["id"])) != (o["label"], o["action"]))


# ---------- publish a round to the Results page ----------

def publish(field: str, results: list[dict], round_no: int, source: str,
            version: int, baseline: bool = False, extra: dict | None = None) -> dict:
    """Write runs/run.json (+ overlays, masks) in the Results page format.
    baseline=True also freezes it as the run the honest score is taken on."""
    runs = fdir(field) / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    entries = []
    for r in results:
        shutil.copy(r["overlay"], runs / f"{r['name']}.overlay.png")
        (runs / f"{r['name']}.overlay.jpg").unlink(missing_ok=True)
        shutil.copy(r["masks"], runs / f"{r['name']}.masks.npz")
        entries.append({"name": r["name"], "status": "done",
                        "stage": f"done in {r['seconds']:.0f}s", "plants": r["plants"],
                        "crop": r["crop"], "weed": r["weed"], "unsure": r["unsure"],
                        "spray": r["spray"], "protect": r["protect"],
                        "seconds": r["seconds"], "cost": r["usd"], "overlay": True,
                        "objects": r["objects"], "geometry": r["geometry"]})
    totals = {k: sum(e[k] for e in entries)
              for k in ("plants", "crop", "weed", "unsure", "spray", "protect")}
    totals["seconds"] = round(sum(e["seconds"] for e in entries), 1)
    totals["cost"] = round(sum(e["cost"] for e in entries), 4)
    doc = {"status": "done", "round": round_no, "source": source,
           "checklist_version": version, "ran_at": time.strftime("%Y-%m-%d %H:%M"),
           "entries": entries, "totals": totals, "taught": {}, **(extra or {})}
    (runs / "run.json").write_text(json.dumps(doc, indent=1, default=_jsonable),
                                   encoding="utf-8")
    with _LOCK:  # the Results page's summary (confirm needs it)
        log = load_log(field)
        log["results"] = {
            "frames": len(entries), **{k: totals[k] for k in (
                "plants", "crop", "weed", "unsure", "spray", "protect",
                "seconds", "cost")},
            "ran_at": doc["ran_at"], "round": round_no, "confirmed": False,
            "audit": (f"Round {round_no}: the AI judged {len(entries)} photos with "
                      f"checklist v{version}: {totals['plants']} plants, "
                      f"{totals['spray']} spray decisions.")}
        log.pop("done", None)
        _save_log(field, log)
    if baseline:
        bdir = runs / "baseline"
        bdir.mkdir(exist_ok=True)
        for r in results:
            shutil.copy(r["masks"], bdir / f"{r['name']}.masks.npz")
        (bdir / "run.json").write_text(json.dumps(doc, indent=1, default=_jsonable),
                                       encoding="utf-8")
    return doc


# ---------- the setup run (A + B) ----------

def run_setup(field: str, crop: str, labeled: list[str], test: list[str],
              cache_dir: Path, log_path: Path) -> dict:
    """Study + iterate. Updates setup_log.json after every step."""
    ctx = {"cache_dir": Path(cache_dir), "log_path": Path(log_path)}
    sdir = fdir(field) / "setup"
    if sdir.exists():
        shutil.rmtree(sdir)
    S = {"state": "studying", "message": "Measuring your labeled photos",
         "started": time.time(), "usd": 0.0, "crop": crop,
         "labeled": labeled, "test": test, "versions": [], "rounds": [],
         "disagreements": [], "active_version": None, "best_version": None,
         "loo_note": "Each labeled photo is judged with examples from the other "
                     "labeled photos. The checklist was written from all of them, "
                     "so this score is a little optimistic.",
         "progress": {}}
    (fdir(field) / "runs" / "reviewed.json").unlink(missing_ok=True)  # new run: review again

    def save():
        # Per-round results stay in memory; the log gets the public view.
        write_setup(field, {**S, "versions": [_public(v) for v in S["versions"]]})

    def step(state, message, **progress):
        S["state"], S["message"], S["progress"] = state, message, progress
        S["seconds"] = round(time.time() - S["started"], 1)
        save()

    save()

    try:
        # A1. machine vision study of the farmer's labels.
        studied = {n: study_photo(field, n) for n in labeled}
        agg = aggregate(list(studied.values()))
        S["study"] = {**agg, "per_photo": [
            {"name": n, "row": studied[n]["row"],
             "plants": len(studied[n]["plants"])} for n in labeled]}
        step("studying", "Spark is writing the field checklist")
        # A2. checklist v1.
        checklist, usd = write_checklist(field, crop, studied, agg, ctx["log_path"])
        S["usd"] += usd
        # A3. references.
        S["refs"] = save_refs(field, studied, sdir / "refs")
        step("scoring", "Scoring checklist v1 on your labeled photos")
        loo = leave_one_out(field, crop, checklist, studied, ctx, "v1")
        S["usd"] += loo["usd"]
        S["versions"].append({"v": 1, "checklist": checklist, "source": "study",
                              "what_changed": "First checklist, written from your "
                                              "labeled photos.", "loo": loo})
        S["best_version"] = S["active_version"] = 1
        save()

        # B. rounds on the unlabeled photos.
        prev = None
        refs = ref_list(S["refs"])
        for rnd in range(1, MAX_ROUNDS + 1):
            ver = S["versions"][-1]
            model = model_for(crop, ver["checklist"], agg)
            done = []
            step("round", f"Round {rnd}: judging {len(test)} photos", round=rnd,
                 done=0, total=len(test))

            def go(name, rnd=rnd, model=model):
                res = judge_photo(field, name, model, refs, ctx,
                                  sdir / f"round{rnd}")
                done.append(name)
                step("round", f"Round {rnd}: judging {len(test)} photos",
                     round=rnd, done=len(done), total=len(test))
                return res

            with ThreadPoolExecutor(PARALLEL) as ex:
                results = list(ex.map(go, test))
            S["usd"] += sum(r["usd"] for r in results)
            changed = labels_changed(prev, results)
            dis = [{"round": rnd, "photo": r["name"], **d}
                   for r in results for d in r["geometry"].get("disagree", [])]
            S["disagreements"] += dis
            R = {"round": rnd, "version": ver["v"], "labels_changed": changed,
                 "overlays": [r["overlay"] for r in results],
                 "photos": [{k: r[k] for k in ("name", "plants", "crop", "weed",
                                               "unsure", "spray", "protect")}
                            for r in results],
                 "flags": sum(1 for r in results for o in r["objects"]
                              if o["conf"] < vision.SPRAY_CONF or o["label"] == "unsure"),
                 "disagreements": len(dis), "geometry_confirmed": sum(
                     len(r["geometry"].get("confirmed", [])) for r in results),
                 "usd": round(sum(r["usd"] for r in results), 4)}
            S["rounds"].append(R)
            ver["results"] = results  # kept in memory for publishing
            save()
            prev = results
            if changed == 0:
                R["stop"] = "No labels changed: the checklist has settled."
                break
            if rnd == MAX_ROUNDS:
                R["stop"] = f"Reached {MAX_ROUNDS} rounds."
                break
            try:
                step("reviewing", f"Round {rnd}: Spark is reviewing the overlays")
                rev, usd = review(crop, ver["checklist"], results, ctx["log_path"])
                S["usd"] += usd
                R["suspects"] = rev.get("suspects", [])
                R["review_summary"] = rev.get("summary", "")
                step("revising", f"Round {rnd}: Spark is revising the checklist")
                new, usd = revise(crop, ver["checklist"], R["suspects"], ctx["log_path"])
                S["usd"] += usd
            except RuntimeError as e:  # unusable reply twice: keep what we have
                R["stop"] = f"Stopped: {e}. Kept the checklists so far."
                break
            v = len(S["versions"]) + 1
            step("scoring", f"Scoring checklist v{v} on your labeled photos")
            loo = leave_one_out(field, crop, new["checklist"], studied, ctx, f"v{v}")
            S["usd"] += loo["usd"]
            S["versions"].append({"v": v, "checklist": new["checklist"],
                                  "source": "review",
                                  "what_changed": new["what_changed"], "loo": loo})

        # Keep the best-scoring version, not the last one.
        run_versions = [v for v in S["versions"] if "results" in v]
        best = min(run_versions, key=lambda v: (rank(v["loo"]), -v["v"]))
        S["best_version"] = S["active_version"] = best["v"]
        publish(field, best["results"], 1, "setup", best["v"], baseline=True,
                extra={"note": f"From learning: checklist v{best['v']} (best score "
                               "on your labeled photos)."})
        S["state"], S["message"] = "done", (
            f"Done: kept checklist v{best['v']} (best score on your labeled photos).")
        S["finished"] = time.time()
        S["seconds"] = round(S["finished"] - S["started"], 1)
        S["usd"] = round(S["usd"], 4)
        S["versions"] = [_public(v) for v in S["versions"]]
        write_setup(field, S)
        return S
    except Exception as e:  # noqa: BLE001 - shown on the Learning screen
        S["state"], S["message"] = "error", f"{type(e).__name__}: {e}"
        S["versions"] = [_public(v) for v in S["versions"]]
        write_setup(field, S)
        raise


def _public(v: dict) -> dict:
    return {k: val for k, val in v.items() if k != "results"}


# ---------- helpers the app uses ----------

def active_model(field: str) -> tuple[dict, dict] | tuple[None, None]:
    """(field_model, refs) of the active checklist, or (None, None)."""
    S = get_setup(field)
    if S.get("state") != "done":
        return None, None
    ver = next((v for v in S["versions"] if v["v"] == S["active_version"]), None)
    if ver is None:
        return None, None
    study = S.get("study", {})
    return model_for(S.get("crop", "crop"), ver["checklist"], study), S.get("refs", {})


def load_reviewed(field: str) -> dict:
    p = fdir(field) / "runs" / "reviewed.json"
    if p.is_file():
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            return d if isinstance(d, dict) else {}
        except json.JSONDecodeError:
            pass
    return {}


def save_reviewed(field: str, doc: dict) -> None:
    p = fdir(field) / "runs" / "reviewed.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, indent=1), encoding="utf-8")


def load_corrections(field: str) -> dict:
    p = fdir(field) / "runs" / "corrections.json"
    if p.is_file():
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            return d if isinstance(d, dict) else {}
        except json.JSONDecodeError:
            pass
    return {}


def honest_score(field: str) -> dict | None:
    """C: the learning run's labels vs the farmer's review, BEFORE any of his
    fixes taught the model. Only photos marked reviewed count. Truth per
    plant = his fix, else the AI's crop/weed label he left standing; plants
    he skipped or left as unsure are not scored."""
    bdir = fdir(field) / "runs" / "baseline"
    if not (bdir / "run.json").is_file():
        return None
    base = json.loads((bdir / "run.json").read_text(encoding="utf-8"))
    reviewed = load_reviewed(field)
    corr = load_corrections(field)
    scores = []
    for e in base["entries"]:
        if not reviewed.get(e["name"]):
            continue
        mp = bdir / f"{e['name']}.masks.npz"
        if not mp.is_file():
            continue
        masks = load_masks(mp)
        img = Image.open(photo_path(field, e["name"])).convert("RGB")
        green, _ = vision.exg_mask(img)
        fixes = corr.get(e["name"], {})
        crop_px = np.zeros_like(green)
        weed_px = np.zeros_like(green)
        weeds = []
        objs = []
        for o in e["objects"]:
            truth = fixes.get(str(o["id"]), o["label"])
            m = masks.get(o["id"])
            if m is None:
                continue
            g = m & green
            if truth == "crop":
                crop_px |= g
            elif truth == "weed":
                weed_px |= g
                if g.sum() >= MIN_PLANT_PX:
                    weeds.append(g)
            objs.append({"label": o["label"], "action": o["action"], "x": o.get("x"),
                         "y": o.get("y"), "mask": m,
                         "truth": truth if truth in ("crop", "weed") else None})
        scores.append({"photo": e["name"], **score(objs, crop_px, weed_px, weeds)})
    t = total(scores) if scores else total([score([], np.zeros((1, 1), bool),
                                                  np.zeros((1, 1), bool), [])])
    t["scored_on"] = len(scores)
    t["of"] = len(base["entries"])
    t["per_photo"] = scores
    t["checklist_version"] = base.get("checklist_version")
    return t


def farmer_revision(field: str, prev_run: dict, corrections: dict, notes_text: str,
                    plant_notes: dict, cache_dir: Path, log_path: Path) -> dict | None:
    """C: the farmer's fixes + notes rewrite the checklist. The new version is
    scored on the labeled photos and only becomes active if it scores at least
    as well as the current one (fewest crop hits, then most weeds, then
    labels right). Not a veto on his labels: his fixes always teach as
    reference close-ups."""
    S = get_setup(field)
    if S.get("state") != "done":
        return None
    cur = next(v for v in S["versions"] if v["v"] == S["active_version"])
    lines, fix_imgs = [], []
    by_name = {e["name"]: e for e in (prev_run or {}).get("entries", [])}
    for name, fixes in corrections.items():
        e = by_name.get(name)
        if not e:
            continue
        objs = {str(o["id"]): o for o in e.get("objects", [])}
        img = None
        for oid, lab in fixes.items():
            o = objs.get(str(oid))
            if not o or lab not in ("crop", "weed"):
                continue
            note = plant_notes.get((name, str(oid)), "")
            lines.append(f"- photo {name} plant #{oid}: AI said {o['label']} "
                         f"({o.get('reason', '')}), farmer says {lab}"
                         + (f". Farmer's note: {note}" if note else ""))
            if len(fix_imgs) < 8:
                img = img or Image.open(photo_path(field, name)).convert("RGB")
                fix_imgs.append((lab, cutout(img, o["box"])))
    if not lines and not notes_text:
        return None
    new, usd = revise(S["crop"], cur["checklist"], [], log_path,
                      farmer_fixes="\n".join(lines), farmer_notes=notes_text,
                      fix_images=fix_imgs)
    studied = {n: study_photo(field, n) for n in S["labeled"]}
    v = len(S["versions"]) + 1
    loo = leave_one_out(field, S["crop"], new["checklist"], studied,
                        {"cache_dir": Path(cache_dir), "log_path": Path(log_path)},
                        f"v{v}")
    # Same rule as the learning rounds: keep the best-scoring version. A
    # rewrite that "plays safe" by calling everything unsure scores worse on
    # weeds and stays inactive.
    safe = rank(loo) <= rank(cur["loo"])
    S["versions"].append({"v": v, "checklist": new["checklist"], "source": "farmer",
                          "what_changed": new["what_changed"], "loo": loo,
                          "activated": safe})
    if safe:
        S["active_version"] = v
    S["usd"] = round(S.get("usd", 0) + usd + loo["usd"], 4)
    S.setdefault("farmer_rounds", []).append(
        {"version": v, "fixes": len(lines), "notes": bool(notes_text),
         "activated": safe, "usd": round(usd + loo["usd"], 4),
         "at": time.strftime("%Y-%m-%d %H:%M")})
    write_setup(field, S)
    return S["versions"][-1]


def save_model(field: str) -> dict:
    """D: fields/<field>/model.json, the field model every run uses."""
    S = get_setup(field)
    ver = next(v for v in S["versions"] if v["v"] == S["active_version"])
    mdir = fdir(field) / "model"
    rdir = mdir / "refs"
    if rdir.exists():
        shutil.rmtree(rdir)
    rdir.mkdir(parents=True)
    refs = {}
    for lab, items in S.get("refs", {}).items():
        refs[lab] = []
        for r in items:
            if Path(r["file"]).is_file():
                dst = rdir / Path(r["file"]).name
                shutil.copy(r["file"], dst)
                refs[lab].append(rel(dst))
    st = S.get("study", {})
    honest = honest_score(field) or {}
    model = {
        "field": field, "crop": S.get("crop"),
        "saved_at": time.strftime("%Y-%m-%d %H:%M"),
        "band": {"half_width_px": st.get("band_half_px"),
                 "width_px": st.get("band_width_px"),
                 "angle_deg": st.get("band_angle_deg"),
                 "crop_layout": st.get("crop_layout")},
        "sizes": st.get("sizes"), "thresholds": st.get("thresholds"),
        "crop_median_area": st.get("crop_median_area"),
        "references": refs,
        "checklist": ver["checklist"], "checklist_version": ver["v"],
        "checklist_score_on_labeled_photos": ver.get("loo"),
        "honest_score": {k: honest.get(k) for k in (
            "accuracy_pct", "crop_hits", "weeds_hit", "weeds", "weeds_hit_pct",
            "weed_sprayed_pct", "weed_protected_pct", "scored_on", "of")},
        "setup_seconds": S.get("seconds"),
        "cost_usd": S.get("usd"),
        "labeled_photos": S.get("labeled"), "test_photos": S.get("test"),
    }
    (fdir(field) / "model.json").write_text(
        json.dumps(model, indent=1, ensure_ascii=False, default=_jsonable),
        encoding="utf-8")
    return model


def main() -> None:
    ap = argparse.ArgumentParser(description="Step 4 field learning (A + B).")
    ap.add_argument("field")
    ap.add_argument("--out", type=Path, default=Path("out"))
    args = ap.parse_args()
    meta = json.loads((fdir(args.field) / "field.json").read_text(encoding="utf-8"))
    names = sorted(p.stem for p in (fdir(args.field) / "photos").iterdir())
    labeled = [n for n in names
               if (fdir(args.field) / "labels" / f"{n}.labels.json").is_file()]
    test = [n for n in names if n not in labeled]
    S = run_setup(args.field, meta.get("crop", "crop"), labeled, test,
                  args.out / "cache", args.out / "api_calls.log")
    for v in S["versions"]:
        lo = v["loo"]
        print(f"v{v['v']} ({v['source']}): crop hits {lo['crop_hits']}, weeds hit "
              f"{lo['weeds_hit']}/{lo['weeds']}, accuracy {lo['accuracy_pct']}%  "
              f"| {v['what_changed']}")
    for r in S["rounds"]:
        print(f"round {r['round']} (v{r['version']}): labels changed "
              f"{r['labels_changed']}, flags {r['flags']}, geometry disagreements "
              f"{r['disagreements']}")
    print(f"kept v{S['best_version']}; {S['seconds']}s, ${S['usd']:.4f}")


if __name__ == "__main__":
    main()

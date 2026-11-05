"""Step 2 hybrid vision: geometry locally, judgment via one Muse Spark call.

Pipeline (PROJECT.md):
  ExG green mask (Otsu threshold, morphological opening)
  -> SAM 3.1 prompt "plant" (cached) + uncovered green blobs as extra objects
  -> per-object shape features from green pixels
  -> optional --split (watershed) and --row (RANSAC row fit, same image)
  -> batched muse-spark-1.3 calls (<=8 objects each; overview + close-ups
     + features as text)
  -> safety rule (spray only weed @ >=0.85) -> spray point with safety margin

Run from the repo root:
    python tools/vision.py tests/images/carrot_field_035.png --split --row

Library use:
    from tools import vision  # or: import vision when tools/ is on sys.path
    result = vision.analyze("tests/images/carrot_field_035.png", cache_dir="out/cache")
"""

import argparse
import asyncio
import base64
import hashlib
import io
import json
import math
import random
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from openai import AsyncOpenAI, OpenAI
from PIL import Image, ImageDraw

import sam_client
from prompt_sweep import cache_path

SAM_PROMPT = "plant"
SPARK_MODEL = "muse-spark-1.3"
SPARK_USD_PER_1M_IN = 1.25
SPARK_USD_PER_1M_OUT = 4.25

SPRAY_CONF = 0.85
MIN_BLOB_PX = 150
SAFETY_MARGIN_PX = 15
MAX_OUTPUT_TOKENS = 1000

LABELS = ("crop", "weed", "mixed")


@dataclass
class Target:
    id: int
    x: int | None  # spray point; None unless action == "spray"
    y: int | None
    area: int  # sprayable green px; 0 unless action == "spray"
    features: dict = field(default_factory=dict)
    label: str = "mixed"
    confidence: float = 0.0
    reason: str = ""
    action: str = "protect"  # "spray" or "protect"
    decided_by: str = "other"  # spark_085 / margin_veto / below_threshold /
    # not_weed / judge_failed. NOTE: no geometry spray rule exists; row
    # distance only informs the judge as text, it never fires a spray.


@dataclass
class VisionResult:
    targets: list[Target]
    field_notes: dict
    judge_ok: bool
    exg_threshold: float
    n_sam_objects: int
    n_extra_objects: int
    row: dict | None  # {"point": (x,y) px, "dir": (dx,dy), "inliers": n}
    sam_seconds: float
    sam_cached: bool
    spark_seconds: float
    spark_usage: dict
    spark_usd: float
    spark_raw: str = ""
    objects: list = field(default_factory=list)  # {id, mask, box, features}
    judge_batches: int = 1
    judge_batches_ok: int = 1
    n_merged: int = 0
    geometry: dict = field(default_factory=dict)  # candidates/confirmed/disagree


# ---------- local geometry ----------

def exg_mask(img: Image.Image) -> tuple[np.ndarray, float]:
    """ExG green mask. Threshold = Otsu clamped to [40, 90], plus opening."""
    rgb = np.asarray(img.convert("RGB")).astype(np.int32)
    exg = 2 * rgb[..., 1] - rgb[..., 0] - rgb[..., 2]
    exg8 = np.clip(exg, 0, 255).astype(np.uint8)
    thr, _ = cv2.threshold(exg8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = float(np.clip(thr, 40, 90))
    mask = exg > thr
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    return opened.astype(bool), thr


def shape_features(green: np.ndarray) -> dict:
    """area, solidity, elongation, mean leaf width from green pixels."""
    area = int(green.sum())
    if area == 0:
        return {"area": 0, "solidity": 0.0, "elongation": 0.0, "leaf_width": 0.0}
    ys, xs = np.nonzero(green)
    pts = np.column_stack((xs, ys)).astype(np.int32)
    hull_area = float(cv2.contourArea(cv2.convexHull(pts))) if len(pts) >= 3 else 0.0
    solidity = area / hull_area if hull_area > 0 else 0.0
    _center, (w, h), _angle = cv2.minAreaRect(pts)
    elongation = (max(w, h) / min(w, h)) if min(w, h) > 0 else 1.0
    dist = cv2.distanceTransform(green.astype(np.uint8), cv2.DIST_L2, 5)
    leaf_width = float(2 * dist[green].mean())
    return {"area": area, "solidity": round(solidity, 3),
            "elongation": round(elongation, 2), "leaf_width": round(leaf_width, 1)}


def color_features(hsv: np.ndarray, green: np.ndarray) -> dict:
    """Mean hue (OpenCV 0-180) and saturation (0-255) of the green pixels."""
    if not green.any():
        return {"hue": 0.0, "sat": 0.0}
    px = hsv[green]
    return {"hue": round(float(px[:, 0].mean()), 1),
            "sat": round(float(px[:, 1].mean()), 1)}


def watershed_pieces(green: np.ndarray, min_px: int = MIN_BLOB_PX,
                     erode_kernel: int = 35) -> list[np.ndarray]:
    """Split touching plants via watershed from eroded markers.

    Heavy erosion breaks thin leaf contacts first, leaving one marker blob
    per plant body (leaflets vanish instead of becoming false markers).
    Returns green-pixel pieces (each > min_px, slivers absorbed into the
    largest piece), or [] when there is nothing to split (< 2 markers).
    Kernel 35 chosen by probe: best purity gain that keeps counts sane
    (erode-25 purifies slightly more but yields 17-23 pieces/image, which
    risks truncating the judge's 1000-token reply).
    """
    bin8 = (green * 255).astype(np.uint8)
    if bin8.sum() == 0:
        return []
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (erode_kernel, erode_kernel))
    eroded = cv2.erode(bin8, kernel)
    n, lab, stats, _c = cv2.connectedComponentsWithStats(eroded, 8)
    big = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= 20]
    if len(big) < 2:
        return []
    markers = np.ones_like(lab, dtype=np.int32)  # background 1
    for new, old in enumerate(big, start=2):
        markers[lab == old] = new
    markers[(bin8 > 0) & (eroded == 0)] = 0  # unknown
    cv2.watershed(cv2.cvtColor(bin8, cv2.COLOR_GRAY2BGR), markers)
    whole = bin8 > 0
    pieces = [(markers == lbl) & whole for lbl in range(2, 2 + len(big))]
    kept = [p for p in pieces if p.sum() > min_px]
    if not kept:
        return [whole]
    dropped = whole & ~np.logical_or.reduce(kept)
    if dropped.any():
        biggest = int(np.argmax([p.sum() for p in kept]))
        kept[biggest] = kept[biggest] | dropped
    return kept


def fit_row_ransac(centroids: list[tuple[float, float]], weights: list[float],
                   tol: float = 50.0, iters: int = 300,
                   seed: int = 0) -> tuple | None:
    """RANSAC line through centroids weighted by area. Returns (point, dir, n_inliers)."""
    pts = np.array(centroids, dtype=float)
    w = np.array(weights, dtype=float)
    if len(pts) < 2:
        return None
    rng = random.Random(seed)
    best = None
    for _ in range(iters):
        i, j = rng.sample(range(len(pts)), 2)
        d = pts[j] - pts[i]
        L = float(np.linalg.norm(d))
        if L < 1e-6:
            continue
        d /= L
        dists = np.abs((pts - pts[i])[:, 0] * d[1] - (pts - pts[i])[:, 1] * d[0])
        inl = dists < tol
        if inl.sum() < 2:
            continue
        W = w[inl].sum()
        c = (w[inl][:, None] * pts[inl]).sum(axis=0) / W
        C = pts[inl] - c
        cov = (C * w[inl][:, None]).T @ C / W
        vals, vecs = np.linalg.eigh(cov)
        dd = vecs[:, int(np.argmax(vals))]
        d2 = np.abs((pts - c)[:, 0] * dd[1] - (pts - c)[:, 1] * dd[0])
        inl2 = d2 < tol
        score = float(w[inl2].sum())
        if best is None or score > best[0]:
            best = (score, tuple(c), tuple(dd), int(inl2.sum()))
    if best is None:
        return None
    return best[1], best[2], best[3]


def point_line_dist(px: float, py: float, line) -> float:
    (cx, cy), (dx, dy) = line[0], line[1]
    return abs((px - cx) * dy - (py - cy) * dx)


def box_iou(a: tuple, b: tuple) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    if iw == 0 or ih == 0:
        return 0.0
    inter = iw * ih
    return inter / ((a[2] - a[0]) * (a[3] - a[1])
                    + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def angle_from_vertical(line) -> float:
    (_cx, _cy), (dx, dy) = line[0], line[1]
    return math.degrees(math.atan2(abs(dx), abs(dy)))


def geometry_weed(objects: list, row_line, band_half: float | None = None) -> set:
    """Ids of green blobs that geometry considers weed candidates.

    A blob qualifies only if it is farther than 2x the crop band width from
    the row line AND touches no other object. With band_half (the field
    model's half band width, measured on the farmer's labeled photos) the
    trigger is 2x that; without it the band is data-driven: 2x the median
    distance of crop-sized (green area >= median) objects to the row, so the
    trigger distance is 4x that median. Empty-green objects never trigger.
    Candidates never spray on their own: see the verdict rule in analyze().
    """
    if row_line is None or not objects:
        return set()
    cents = {}
    areas = {}
    for o in objects:
        ys, xs = np.nonzero(o["green"])
        if len(xs) == 0:
            continue
        cents[o["id"]] = (float(xs.mean()), float(ys.mean()))
        areas[o["id"]] = int(o["green"].sum())
    if not cents:
        return set()
    if band_half:
        band = 2 * band_half
    else:
        med_area = sorted(areas.values())[len(areas) // 2]
        big_dists = sorted(point_line_dist(*cents[oid], row_line)
                           for oid in cents if areas[oid] >= med_area)
        if not big_dists:
            return set()
        band = 2 * big_dists[len(big_dists) // 2]
    if band <= 0:
        return set()
    thresh = 2 * band
    all_masks = {o["id"]: o["mask"] for o in objects}
    kernel = np.ones((3, 3), np.uint8)
    out = set()
    for oid, (cx, cy) in cents.items():
        if point_line_dist(cx, cy, row_line) <= thresh:
            continue
        others = np.zeros_like(all_masks[oid])
        for oid2, m in all_masks.items():
            if oid2 != oid:
                others |= m
        grown = cv2.dilate(
            next(o["green"] for o in objects if o["id"] == oid).astype(np.uint8),
            kernel).astype(bool)
        if (grown & others).any():
            continue  # touches another object -> not geometry-weed
        out.add(oid)
    return out


# ---------- two-zone runtime (step 4b) ----------

GEO_FAR_BANDS = 2.0     # geometry-alone needs distance > this x band half-width
SIZE_SLACK = 2.0        # crop-like size: p25/2 .. p75*2 of the labeled crop plants
STRONG_SPLIT = 0.65     # a model threshold counts only if it separated >= this


class RowTracker:
    """Keeps the crop-row line across frames: the line of recent frames
    predicts the next one, and a new fit that agrees is blended in (the
    band moves slowly); a fit that disagrees only wins with more inliers."""

    def __init__(self) -> None:
        self.line = None  # (point, dir, inliers) as fit_row_ransac returns

    def update(self, fit, band_half: float):
        if fit is None:
            return self.line, ("tracked" if self.line else None)
        if self.line is None:
            self.line = fit
            return fit, "ok"
        (c0, d0, n0), (c1, d1, n1) = self.line, fit
        dot = abs(d0[0] * d1[0] + d0[1] * d1[1])
        angle = math.degrees(math.acos(min(1.0, dot)))
        if angle <= 8.0 and point_line_dist(c1[0], c1[1], self.line) <= 2 * band_half:
            sign = 1.0 if d0[0] * d1[0] + d0[1] * d1[1] >= 0 else -1.0
            dx, dy = d0[0] + sign * d1[0], d0[1] + sign * d1[1]
            n = math.hypot(dx, dy) or 1.0
            self.line = (((c0[0] + c1[0]) / 2, (c0[1] + c1[1]) / 2),
                         (dx / n, dy / n), n1)
            return self.line, "blended"
        if n1 >= n0:
            self.line = fit
            return fit, "reset"
        return self.line, "tracked-kept"


def crop_like(f: dict, model: dict) -> bool:
    """Does a plant's size and shape match the labeled crop in the model?
    Errs toward True: 'crop-like' only blocks a geometry spray or skips the
    AI for a plant that is then protected, so a wrong True costs a missed
    weed, never a sprayed crop."""
    sz = (model.get("sizes") or {}).get("crop") or {}
    med = sz.get("median") or model.get("crop_median_area")
    if med:
        lo = min(sz.get("p25") or med, med) / SIZE_SLACK
        hi = max(sz.get("p75") or med, med) * SIZE_SLACK
        if not lo <= f["area"] <= hi:
            return False
    strong = {k: t for k, t in (model.get("thresholds") or {}).items()
              if k != "area" and k in f and t.get("balanced_accuracy", 0) >= STRONG_SPLIT}
    if not strong:
        return True
    votes = sum((f[k] >= t["threshold"]) == (t["crop_is"] == "above")
                for k, t in strong.items())
    return votes * 2 >= len(strong)


def blob_objects(green: np.ndarray, hsv: np.ndarray) -> list[dict]:
    """Free (no SAM) plants: connected green blobs with the same fields the
    SAM path produces."""
    n, lab, stats, _c = cv2.connectedComponentsWithStats(green.astype(np.uint8), 8)
    out = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] <= MIN_BLOB_PX:
            continue
        m = lab == i
        x, y, w, h = (int(v) for v in stats[i, :4])
        o = {"id": len(out), "mask": m, "green": m, "box": (x, y, x + w, y + h),
             "parent": len(out)}
        o["features"] = {**shape_features(m), **color_features(hsv, m)}
        out.append(o)
    return out


def fit_band_row(objs: list[dict], model: dict, tracker: "RowTracker | None"):
    """Row line for the two-zone runtime: RANSAC through the crop-like
    plants (all plants if fewer than 2), then the tracker."""
    def pts(sub):
        out = []
        for o in sub:
            ys, xs = np.nonzero(o["green"])
            if len(xs):
                out.append((float(xs.mean()), float(ys.mean()), float(o["green"].sum())))
        return out
    like = [o for o in objs if crop_like(o["features"], model)]
    items = pts(like) if len(like) >= 2 else pts(objs)
    fit = None
    if len(items) >= 2:
        fit = fit_row_ransac([(x, y) for x, y, _w in items], [w for _x, _y, w in items])
        if fit is not None and angle_from_vertical(fit) > 20.0:
            fit = None
    band_half = model.get("band_half_px") or 50.0
    if tracker is not None:
        line, sanity = tracker.update(fit, band_half)
        return line, sanity or "skipped"
    return fit, ("ok" if fit else "skipped")


def touches_others(o: dict, objs: list[dict]) -> bool:
    grown = cv2.dilate(o["green"].astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    return any((grown & p["mask"]).any() for p in objs if p["id"] != o["id"])


def zone_of(o: dict, objs: list[dict], row_line, model: dict) -> tuple[str, str]:
    """('geometry' | 'crop_stats' | 'ai', reason). Geometry alone may spray
    ONLY a plant that is far outside the band AND touches nothing AND does
    not look like the crop. Crop-like plants in the band are protected
    without an AI call. Everything else is judged by the AI."""
    half = model["band_half_px"]
    d = point_line_dist(*_centroid(o), row_line) if row_line is not None else None
    like = crop_like(o["features"], model)
    if d is None:
        return "ai", "no row"
    if d > GEO_FAR_BANDS * half and not like and not touches_others(o, objs):
        return "geometry", f"{d / half:.1f}x band away, no crop traits"
    if d <= half + SAFETY_MARGIN_PX and like and not touches_others(o, objs):
        return "crop_stats", "in band, matches crop size and shape"
    return "ai", "unclear"


def _centroid(o: dict) -> tuple[float, float]:
    ys, xs = np.nonzero(o["green"])
    if len(xs) == 0:
        ys, xs = np.nonzero(o["mask"])
    return float(xs.mean()), float(ys.mean())


# ---------- SAM fetch (cached) ----------

def log_call(log_path: Path | None, entry: dict) -> None:
    if log_path is None:
        return
    entry = {"ts": datetime.now(timezone.utc).isoformat(), **entry}
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


async def fetch_plant(client: AsyncOpenAI, image: Path, cache_dir: Path,
                      log_path: Path | None):
    image_bytes = image.read_bytes()
    cpath = cache_path(cache_dir, image_bytes, SAM_PROMPT)
    if cpath.is_file():
        meta = json.loads(cpath.read_text(encoding="utf-8"))
        # The key is the image's content, so the same photo under another file
        # name (uploaded again, or a robot picture) is a valid hit. Only a
        # different content or prompt is a real collision.
        stored_sha = meta.get("image_sha256")
        if meta.get("prompt") != SAM_PROMPT or (
                stored_sha is not None and stored_sha != hashlib.sha256(image_bytes).hexdigest()):
            raise RuntimeError(f"Cache key collision: {cpath}")
        result = await sam_client.parse_events(meta["events"])
        seconds, usage, cached = meta.get("seconds", 0.0), meta.get("usage", {}), True
    else:
        result, seconds, usage, events = await sam_client.segment_image(
            image, SAM_PROMPT, client
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        cpath.write_text(json.dumps({
            "image": image.name,
            "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
            "prompt": SAM_PROMPT,
            "usage": usage,
            "seconds": round(seconds, 1),
            "events": events,
        }, default=str), encoding="utf-8")
        cached = False
    n = len({r.object_id for r in result.records if r.kind == "box"})
    log_call(log_path, {"model": sam_client.MODEL, "stage": "sam",
                        "image": str(image), "prompt": SAM_PROMPT,
                        "seconds": round(seconds, 1), "objects": n,
                        "outcome": result.outcome.status, "usage": usage,
                        "cached": cached,
                        "usd": 0.0 if cached else sam_client.USD_PER_IMAGE})
    return result, seconds, cached


# ---------- Spark judge ----------

def jpeg_data_url(img: Image.Image, quality: int = 85) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def closeup(img: Image.Image, box: tuple, tag: str, pad: float = 0.20) -> Image.Image:
    x1, y1, x2, y2 = (int(v) for v in box)
    p = int(pad * max(x2 - x1, y2 - y1))
    crop = img.crop((max(x1 - p, 0), max(y1 - p, 0),
                     min(x2 + p, img.width), min(y2 + p, img.height)))
    w, h = crop.size
    if min(w, h) < 64:
        # Upscale tiny close-ups so the plant (not the id tag) fills the frame.
        s = 64 / min(w, h)
        crop = crop.resize((int(w * s), int(h * s)), Image.LANCZOS)
    d = ImageDraw.Draw(crop)
    d.rectangle((0, 0, 44, 26), fill=(0, 0, 0))
    d.text((12, 5), str(tag), fill=(0, 255, 255))
    return crop


def annotate(img: Image.Image, objects: list) -> Image.Image:
    out = img.copy()
    d = ImageDraw.Draw(out)
    for o in objects:
        x1, y1, x2, y2 = (int(v) for v in o["box"])
        d.rectangle((x1, y1, x2, y2), outline=(0, 255, 255), width=3)
        d.rectangle((x1, y1 - 22, x1 + 40, y1), fill=(0, 0, 0))
        d.text((x1 + 9, y1 - 20), str(o["id"]), fill=(0, 255, 255))
    return out


def draw_row_line(img: Image.Image, line, color=(255, 255, 0)) -> None:
    """Draw a pixel-space line clipped to the image."""
    (cx, cy), (dx, dy) = line[0], line[1]
    w, h = img.size
    cands = []
    if abs(dx) > 1e-9:
        for t in ((0 - cx) / dx, (w - cx) / dx):
            if 0 <= cy + t * dy <= h:
                cands.append(t)
    if abs(dy) > 1e-9:
        for t in ((0 - cy) / dy, (h - cy) / dy):
            if 0 <= cx + t * dx <= w:
                cands.append(t)
    if len(cands) < 2:
        return
    t1, t2 = min(cands), max(cands)
    d = ImageDraw.Draw(img)
    d.line([(cx + t1 * dx, cy + t1 * dy), (cx + t2 * dx, cy + t2 * dy)],
           fill=color, width=3)
    d.rectangle((8, 28, 110, 52), fill=(0, 0, 0))
    d.text((14, 33), "crop row", fill=color)


JUDGE_HEAD = (
    "Reply with ONLY a JSON object. No other text, no preamble."
    " You are an agronomist. The field crop is {crop}. Study the numbered "
    "overview, each close-up, and the measured features. Features: area_px (green "
    "pixels), solidity 0-1 (1 = solid blob), elongation >= 1 (higher = longer "
    "thinner), leaf_width_px (typical leaf width), hue (0-180, green ~35-85), "
    "sat (0-255){size_note}."
    "{notes_ask}"
    " Then label each object: id, label (crop/weed/mixed), confidence 0-1, "
    "reason (at most 8 words, naming the trait you used: leaf shape, edges, "
    "arrangement, color or size). Crop and weed touching = mixed. When unsure, "
    "give a low confidence."
    " {reply_fmt}"
)
SIZE_NOTE = (", size_vs_crop (green area / median crop plant area in this "
             "field's labeled photos)")
CHECKLIST_HEAD = (
    "Field checklist, written from the farmer's labeled photos of THIS field. "
    "Apply it to every plant; when it does not settle a plant, protect it "
    "(mixed or low confidence):\n")
GEO_CONFIRM_MIN = 0.60  # geometry may confirm a Spark "weed" at >= this
NOTES_ASK = (
    " First write field_notes: crop (what carrot looks like here, at most 12 "
    "words), weeds (weed types present, at most 12 words)."
)


def build_text(objects: list, no_notes: bool, crop: str = "carrot",
               crop_median_area: float | None = None) -> str:
    """Judge instructions + measured per-object facts.

    Row position is deliberately never included (lesson 4): the row is used
    only by geometry, which can confirm but never decide a weed.
    """
    head = JUDGE_HEAD.format(
        crop=crop,
        size_note=SIZE_NOTE if crop_median_area else "",
        notes_ask="" if no_notes else NOTES_ASK,
        reply_fmt=("Reply ONLY JSON: {objects: [{id, label, confidence, reason}]}."
                   if no_notes else
                   "Reply ONLY JSON: {field_notes: {crop, weeds}, "
                   "objects: [{id, label, confidence, reason}]}."))
    lines = [head, "Per-object features:"]
    for o in objects:
        f = o["features"]
        line = (f"Object {o['id']}: area_px={f['area']} solidity={f['solidity']:.2f} "
                f"elongation={f['elongation']:.2f} leaf_width_px={f['leaf_width']:.1f}")
        if "hue" in f:
            line += f" hue={f['hue']:.0f} sat={f['sat']:.0f}"
        if crop_median_area:
            line += f" size_vs_crop={f['area'] / crop_median_area:.2f}"
        lines.append(line)
    return "\n".join(lines)


def normalize_label(s) -> str | None:
    s = str(s or "").strip().lower()
    if not s:
        return None
    if "mix" in s or "both" in s or "touch" in s:
        return "mixed"
    if "weed" in s:
        return "weed"
    if "crop" in s:
        return "crop"
    return None


def parse_verdicts(raw: str) -> tuple[dict, dict]:
    """Returns (field_notes, {id: verdict}). Raises ValueError when unusable."""
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        raise ValueError("no JSON object in reply")
    data = json.loads(m.group(0))
    objs = data.get("objects")
    if not isinstance(objs, list) or not objs:
        raise ValueError("no objects array in reply")
    by_id = {}
    for v in objs:
        if not isinstance(v, dict) or "id" not in v:
            continue
        try:
            by_id[int(v["id"])] = v
        except (TypeError, ValueError):
            continue
    if not by_id:
        raise ValueError("no usable ids in reply")
    notes = data.get("field_notes", {})
    return (notes if isinstance(notes, dict) else {}), by_id


def judge_schema(no_notes: bool) -> dict:
    """JSON Schema for the verdicts (structured output: text.format).

    Per docs: every object sets additionalProperties false and required
    lists every key. Labels are an enum so no normalization is needed.
    """
    verdict = {
        "type": "object",
        "properties": {
            "id": {"type": "integer"},
            "label": {"type": "string", "enum": ["crop", "weed", "mixed"]},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "reason": {"type": "string", "maxLength": 80},
        },
        "required": ["id", "label", "confidence", "reason"],
        "additionalProperties": False,
    }
    props: dict = {"objects": {"type": "array", "items": verdict}}
    required = ["objects"]
    if not no_notes:
        props["field_notes"] = {
            "type": "object",
            "properties": {
                "crop": {"type": "string", "maxLength": 150},
                "weeds": {"type": "string", "maxLength": 150},
            },
            "required": ["crop", "weeds"],
            "additionalProperties": False,
        }
        required = ["field_notes", "objects"]
    return {"type": "object", "properties": props, "required": required,
            "additionalProperties": False}


def spark_judge(client: OpenAI, content: list, log_path: Path | None,
                image: Path, no_notes: bool, batch: str = ""):
    """One Spark call with one retry. Returns (notes, by_id, usage, sec, ok, raw).

    Muse Spark is a reasoning model whose hidden thinking consumes the output
    budget, so reasoning effort is "minimal" (the visible field_notes carry
    the reasoning instead) and decoding is constrained to the verdict schema.
    On double failure returns ({}, {}, usage, seconds, False, last_raw): the
    caller must protect everything.
    """
    usage: dict = {}
    total_sec = 0.0
    last_raw = ""
    text_format = {"format": {"type": "json_schema", "name": "PlantVerdicts",
                              "schema": judge_schema(no_notes)}}
    for attempt in (1, 2):
        started = time.perf_counter()
        resp = client.responses.create(
            model=SPARK_MODEL,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            reasoning={"effort": "minimal"},
            text=text_format,
            input=[{"type": "message", "role": "user", "content": content}],
        )
        seconds = time.perf_counter() - started
        total_sec += seconds
        raw = resp.output_text or ""
        u = getattr(resp, "usage", None)
        if u is not None:
            d = u.model_dump() if hasattr(u, "model_dump") else dict(u)
            details = d.get("output_tokens_details", {}) or {}
            usage = {"input_tokens": d.get("input_tokens", d.get("prompt_tokens", 0)),
                     "output_tokens": d.get("output_tokens", d.get("completion_tokens", 0)),
                     "reasoning_tokens": details.get("reasoning_tokens", 0)}
        usd = (usage.get("input_tokens", 0) / 1e6 * SPARK_USD_PER_1M_IN
               + usage.get("output_tokens", 0) / 1e6 * SPARK_USD_PER_1M_OUT)
        log_call(log_path, {"model": SPARK_MODEL, "stage": "spark-judge",
                            "image": str(image), "batch": batch,
                            "attempt": attempt, "seconds": round(seconds, 1),
                            "usage": usage, "cached": False,
                            "usd": round(usd, 6)})
        try:
            notes, by_id = parse_verdicts(raw)
            return notes, by_id, usage, total_sec, True, raw
        except (ValueError, json.JSONDecodeError):
            last_raw = raw
            continue
    return {}, {}, usage, total_sec, False, last_raw


# ---------- main entry ----------

def analyze(image: str | Path, *, cache_dir: str | Path, split: bool = False,
            row: bool = False, refs: list | None = None, no_notes: bool = True,
            overlay_path: str | Path | None = None,
            log_path: str | Path | None = None,
            debug_dir: str | Path | None = None,
            progress=None, farmer_notes: str | None = None,
            field_model: dict | None = None, crop: str | None = None,
            spray_conf: float | None = None) -> VisionResult:
    """Run the hybrid pipeline on one image. Returns VisionResult.

    progress(stage, info) is called at stage boundaries (exg, sam, judge,
    decide) when given; used for live run updates. None disables it.
    refs items: image/path (= crop), (label, image) or (label, image, note).
    farmer_notes: free text from the farmer about this field, sent with
    every judge batch. Neither changes the spray rule.
    crop: the field's crop name for the judge prompt (field_model's wins);
    without either the prompt falls back to "carrot", the old test crop.
    field_model (from setup.py): {"crop", "checklist", "crop_median_area",
    "band_half_px"}. The checklist goes into every judge batch; the band
    turns on geometry (row fitted on THIS image, band width from setup).
    Geometry only ever confirms a Spark "weed"; see res.geometry.
    """
    field_model = field_model or {}
    spray_conf = SPRAY_CONF if spray_conf is None else min(0.99, max(0.5, float(spray_conf)))
    if field_model.get("band_half_px"):
        row = True
    image = Path(image)
    cache_dir = Path(cache_dir)
    log_path = Path(log_path) if log_path else None

    def _emit(stage: str, **info):
        if progress is not None:
            progress(stage, info)

    img = Image.open(image).convert("RGB")
    W, H = img.size

    # 1. ExG mask.
    green, exg_thr = exg_mask(img)
    _emit("exg", threshold=round(float(exg_thr), 1))

    # 2. SAM objects (cached) + uncovered green blobs as extras.
    async def run_sam():
        async with AsyncOpenAI(base_url=sam_client.BASE_URL,
                               api_key=sam_client.api_key()) as client:
            return await fetch_plant(client, image, cache_dir, log_path)

    result, sam_seconds, sam_cached = asyncio.run(run_sam())
    _w, _h, sx, sy = sam_client.api_space(result.raw_output, W, H)
    sam_masks = {r.object_id: r for r in result.records if r.kind == "mask"}
    objects: list[dict] = []
    for r in result.records:
        if r.kind != "box":
            continue
        m = sam_masks.get(r.object_id)
        box = sam_client.scale_box(r, sx, sy)
        if m is None:
            full = np.zeros((H, W), bool)
            x1, y1, x2, y2 = (max(0, v) for v in
                              (box[0], box[1], min(box[2], W), min(box[3], H)))
            full[y1:y2, x1:x2] = True
        else:
            bl, bt, bw, bh = sam_client.scale_bounds(m, sx, sy)
            arr = sam_client.mask_raster_array(m.mask)
            small = np.asarray(Image.fromarray((arr * 255).astype(np.uint8))
                               .resize((bw, bh), Image.NEAREST)) > 0 if bw > 0 and bh > 0 else None
            full = np.zeros((H, W), bool)
            if small is not None:
                x1, y1 = max(bl, 0), max(bt, 0)
                x2, y2 = min(bl + bw, W), min(bt + bh, H)
                if x2 > x1 and y2 > y1:
                    full[y1:y2, x1:x2] = small[y1 - bt:y2 - bt, x1 - bl:x2 - bl]
        objects.append({"id": int(r.object_id), "mask": full, "box": box})
    n_sam = len(objects)
    _emit("sam", objects=n_sam, cached=bool(sam_cached),
          seconds=round(sam_seconds, 1))
    sam_union = np.logical_or.reduce([o["mask"] for o in objects]) if objects else np.zeros((H, W), bool)
    n_lab, lab, stats, _cent = cv2.connectedComponentsWithStats(
        green.astype(np.uint8), 8)
    next_id = max([o["id"] for o in objects], default=-1) + 1
    sam_boxes = [(o["id"], o["box"]) for o in objects]
    n_extra = 0
    n_merged = 0
    for lbl in range(1, n_lab):
        if stats[lbl, cv2.CC_STAT_AREA] <= MIN_BLOB_PX:
            continue
        rest = (lab == lbl) & ~sam_union
        if rest.sum() <= MIN_BLOB_PX:
            continue
        ys, xs = np.nonzero(rest)
        ebox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        best, best_iou = None, 0.0
        for oid, sbox in sam_boxes:
            iou = box_iou(ebox, sbox)
            if iou > best_iou:
                best, best_iou = oid, iou
        if best is not None and best_iou > 0.6:
            # Duplicate of a SAM object (mask missed green it should own):
            # merge instead of adding a nested twin.
            for o in objects:
                if o["id"] == best:
                    o["mask"] = o["mask"] | rest
                    my, mx = np.nonzero(o["mask"])
                    o["box"] = (int(mx.min()), int(my.min()),
                                int(mx.max()) + 1, int(my.max()) + 1)
                    break
            n_merged += 1
        else:
            objects.append({"id": next_id, "mask": rest, "box": ebox})
            next_id += 1
            n_extra += 1

    # 3+4. Green pixels, optional watershed split, shape features.
    final: list[dict] = []
    for o in objects:
        g = o["mask"] & green
        pieces = watershed_pieces(g) if split else []
        if len(pieces) > 1:
            for p in pieces:
                ys, xs = np.nonzero(p)
                box = (int(xs.min()), int(ys.min()),
                       int(xs.max()) + 1, int(ys.max()) + 1)
                final.append({"id": next_id, "mask": p, "box": box,
                              "green": p, "parent": o["id"]})
                next_id += 1
        else:
            # No split: whole object keeps its SAM id and full mask.
            final.append({"id": o["id"], "mask": o["mask"], "box": o["box"],
                          "green": g, "parent": o["id"]})
    hsv = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2HSV)
    for o in final:
        o["features"] = {**shape_features(o["green"]),
                         **color_features(hsv, o["green"])}

    # 5. Optional row fit on THIS image, with sanity check.
    row_line = None
    row_sanity = "not_requested"
    if row:
        row_sanity = "skipped"
        items = []  # (cx, cy, weight, green area)
        for o in final:
            ys, xs = np.nonzero(o["green"])
            if len(xs) == 0:
                ys, xs = np.nonzero(o["mask"])
            if len(xs) == 0:
                continue
            items.append((float(xs.mean()), float(ys.mean()),
                          float(o["green"].sum()), int(o["green"].sum())))

        def _fit(sub):
            if len(sub) < 2:
                return None
            return fit_row_ransac([(x, y) for x, y, _w, _a in sub],
                                  [w for _x, _y, w, _a in sub])

        fit = _fit(items)
        if fit is not None and angle_from_vertical(fit) <= 20.0:
            row_line, row_sanity = fit, "ok"
        else:
            # Refit on crop-sized (large) objects only.
            areas = sorted(it[3] for it in items)
            med = areas[len(areas) // 2] if areas else 0
            fit2 = _fit([it for it in items if it[3] >= med])
            if fit2 is not None and angle_from_vertical(fit2) <= 20.0:
                row_line, row_sanity = fit2, "refit-ok"
    for o in final:
        if row_line is not None:
            ys, xs = np.nonzero(o["green"])
            if len(xs) == 0:
                ys, xs = np.nonzero(o["mask"])
            o["row_dist"] = (point_line_dist(float(xs.mean()), float(ys.mean()), row_line)
                             if len(xs) else 0.0)
        else:
            o["row_dist"] = 0.0

    if row:
        row_info: dict | None = (
            {"point": row_line[0], "dir": row_line[1],
             "inliers": row_line[2], "sanity": row_sanity}
            if row_line is not None else
            {"point": None, "dir": None, "inliers": 0,
             "sanity": row_sanity})
    else:
        row_info = None

    # 6. Batched Spark calls (skipped when there is nothing to judge).
    if not final:
        return VisionResult([], {}, True, exg_thr, n_sam, n_extra, row_info,
                            sam_seconds, sam_cached, 0.0, {}, 0.0, final)
    # NOTE: the judge overview never shows the row line. The row is used
    # ONLY by geometry; the output overlay (draw_overlay) still draws it.
    overview = annotate(img, final)
    judged_cups = [closeup(img, o["box"], o["id"]) for o in final]
    if debug_dir is not None:
        dd = Path(debug_dir)
        dd.mkdir(parents=True, exist_ok=True)
        overview.save(dd / f"{image.stem}_overview.png")
        for o, cup in zip(final, judged_cups):
            cup.save(dd / f"{image.stem}_id{o['id']}.png")
    cup_by_id = {o["id"]: cup for o, cup in zip(final, judged_cups)}
    ref_urls = []  # (label, data-url, note); bare path/image = confirmed crop
    for rp in refs or []:
        lab, note = "crop", ""
        if isinstance(rp, (tuple, list)):
            lab, rp, *rest = rp
            note = rest[0] if rest else ""
        rimg = Image.open(rp).convert("RGB") if isinstance(rp, (str, Path)) else rp
        ref_urls.append((lab, jpeg_data_url(rimg), note or ""))
    overview_url = jpeg_data_url(overview)
    spark = OpenAI(base_url=sam_client.BASE_URL, api_key=sam_client.api_key())
    batches = [final[i:i + 8] for i in range(0, len(final), 8)]
    notes, by_id = {}, {}
    usage = {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0}
    spark_seconds = 0.0
    batches_ok = 0
    raws = []
    for bi, chunk in enumerate(batches):
        _emit("judge", batch=bi + 1, batches=len(batches))
        content: list[dict] = []
        if ref_urls:
            for lab, kind in (("crop", "crop"), ("weed", "weed")):
                items = [(u, n) for lb, u, n in ref_urls if lb == lab]
                if not items:
                    continue
                content.append({"type": "input_text",
                                "text": f"Reference images: confirmed {kind} plants "
                                        "from the same field. "
                                        "Compare each numbered plant to them."})
                for url, note in items:
                    content.append({"type": "input_image", "image_url": url})
                    if note:
                        content.append({"type": "input_text",
                                        "text": f"Farmer's note on this {kind}: "
                                                f"{note}"})
        if field_model.get("checklist"):
            content.append({"type": "input_text",
                            "text": CHECKLIST_HEAD + field_model["checklist"]})
        if farmer_notes:
            content.append({"type": "input_text",
                            "text": "Farmer's notes about this field (they know "
                                    "it best; use them):\n" + farmer_notes})
        content.append({"type": "input_text",
                        "text": build_text(chunk, no_notes,
                                           crop=field_model.get("crop") or crop or "carrot",
                                           crop_median_area=field_model.get("crop_median_area"))})
        content.append({"type": "input_image", "image_url": overview_url})
        for o in chunk:
            content.append({"type": "input_image",
                            "image_url": jpeg_data_url(cup_by_id[o["id"]])})
        n2, b2, u2, s2, ok2, raw2 = spark_judge(
            spark, content, log_path, image, no_notes, batch=f"{bi + 1}/{len(batches)}")
        if not notes and n2:
            notes = n2  # field notes from the first batch that has them
        by_id.update(b2)
        for k in usage:
            usage[k] += u2.get(k, 0)
        spark_seconds += s2
        batches_ok += ok2
        raws.append(f"--- batch {bi + 1}/{len(batches)} ---\n{raw2}")
    judge_ok = batches_ok == len(batches)
    spark_raw = "\n".join(raws)
    if overlay_path is not None:
        Path(overlay_path).with_name(Path(overlay_path).stem + "_raw.txt").write_text(
            spark_raw, encoding="utf-8")
    spark_usd = (usage.get("input_tokens", 0) / 1e6 * SPARK_USD_PER_1M_IN
                 + usage.get("output_tokens", 0) / 1e6 * SPARK_USD_PER_1M_OUT)

    # 7+8. Safety rule + spray points with margin.
    # Geometry can only CONFIRM a weed (lesson 4, and an isolated off-row
    # carrot it once sprayed): it lifts a Spark "weed" at >= GEO_CONFIRM_MIN
    # to spray, and never touches a plant Spark calls crop or mixed or failed
    # to judge. Every disagreement is recorded in res.geometry.
    geo_ids = (geometry_weed(final, row_line, field_model.get("band_half_px"))
               if row_line is not None else set())
    geometry = {"candidates": sorted(geo_ids), "confirmed": [], "disagree": []}
    verdicts = {}
    for o in final:
        v = by_id.get(o["id"])  # per-object: a failed batch only sinks its own
        label = normalize_label(v.get("label")) if v else None
        try:
            conf = float(v.get("confidence", 0.0)) if v else 0.0
        except (TypeError, ValueError):
            conf = 0.0
        conf = min(max(conf, 0.0), 1.0)
        missing = v is None or label is None
        reason = " ".join(str(v.get("reason", "") if v else "").split()[:8])
        spray = (not missing) and label == "weed" and conf >= spray_conf
        geo = False
        if o["id"] in geo_ids:
            if not missing and label == "weed" and not spray and conf >= GEO_CONFIRM_MIN:
                spray = geo = True
                geometry["confirmed"].append(o["id"])
            elif missing or label != "weed":
                geometry["disagree"].append(
                    {"id": o["id"], "spark_label": label or "judge_failed",
                     "confidence": round(conf, 2), "reason": reason})
        verdicts[o["id"]] = {"label": label or "mixed", "confidence": conf,
                             "reason": reason or ("judge failed" if missing else ""),
                             "spray": spray, "missing": missing, "geo": geo}
    protected_union = np.zeros((H, W), bool)
    for o in final:
        if not verdicts[o["id"]]["spray"]:
            protected_union |= o["mask"]
    margin_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * SAFETY_MARGIN_PX + 1, 2 * SAFETY_MARGIN_PX + 1))
    danger = cv2.dilate(protected_union.astype(np.uint8), margin_kernel).astype(bool)
    targets = []
    for o in final:
        vd = verdicts[o["id"]]
        if vd["missing"]:
            targets.append(Target(o["id"], None, None, 0, o["features"],
                                  vd["label"], vd["confidence"], vd["reason"],
                                  "protect", "judge_failed"))
        elif vd["spray"]:
            sprayable = o["green"] & ~danger
            if sprayable.sum() == 0:
                targets.append(Target(o["id"], None, None, 0, o["features"],
                                      vd["label"], vd["confidence"],
                                      (vd["reason"] + " no safe point").strip(),
                                      "protect", "margin_veto"))
            else:
                ys, xs = np.nonzero(sprayable)
                targets.append(Target(o["id"], int(xs.mean()), int(ys.mean()),
                                      int(sprayable.sum()), o["features"],
                                      vd["label"], vd["confidence"], vd["reason"],
                                      "spray",
                                      "geometry_confirmed" if vd.get("geo") else "spark_085"))
        elif vd["label"] != "weed":
            targets.append(Target(o["id"], None, None, 0, o["features"],
                                  vd["label"], vd["confidence"], vd["reason"],
                                  "protect", "not_weed"))
        else:
            targets.append(Target(o["id"], None, None, 0, o["features"],
                                  vd["label"], vd["confidence"], vd["reason"],
                                  "protect", "below_threshold"))

    _emit("decide", plants=len(targets))
    if overlay_path is not None:
        draw_overlay(img, final, {t.id: t for t in targets}, row_line,
                     Path(overlay_path))

    res = VisionResult(targets, notes, judge_ok, exg_thr, n_sam, n_extra,
                       row_info, sam_seconds, sam_cached, spark_seconds,
                       usage, spark_usd, spark_raw, final)
    res.judge_batches = len(batches)
    res.judge_batches_ok = batches_ok
    res.n_merged = n_merged
    res.geometry = geometry
    return res


def draw_overlay(img: Image.Image, objects: list, targets: dict,
                 row_line, path: Path) -> None:
    """Red = spray, green = crop, yellow = protected, + row line + spray dots."""
    film = Image.new("RGBA", img.size, (0, 0, 0, 0))
    for o in objects:
        t = targets[o["id"]]
        if t.action == "spray":
            color = (230, 0, 0)
        elif t.label == "crop":
            color = (0, 200, 0)
        else:
            color = (255, 200, 0)
        m = Image.fromarray((o["mask"] * 255).astype(np.uint8))
        layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        layer.paste(Image.new("RGBA", img.size, color + (110,)), mask=m)
        film = Image.alpha_composite(film, layer)
    out = Image.alpha_composite(img.convert("RGBA"), film)
    d = ImageDraw.Draw(out)
    for o in objects:
        t = targets[o["id"]]
        color = (230, 0, 0) if t.action == "spray" else (
            (0, 200, 0) if t.label == "crop" else (255, 200, 0))
        x1, y1, x2, y2 = (int(v) for v in o["box"])
        d.rectangle((x1, y1, x2, y2), outline=color, width=3)
        tag1 = f"{o['id']} {t.label} {t.confidence:.2f}"
        tag2 = f"{t.action} {t.decided_by}"
        tw = int(max(d.textlength(tag1), d.textlength(tag2))) + 12
        d.rectangle((x1, y1, x1 + tw, y1 + 38), fill=(0, 0, 0))
        d.text((x1 + 6, y1 + 3), tag1, fill=(255, 255, 255))
        d.text((x1 + 6, y1 + 20), tag2, fill=(255, 255, 255))
        if t.action == "spray" and t.x is not None:
            d.ellipse((t.x - 7, t.y - 7, t.x + 7, t.y + 7),
                      outline=(255, 255, 255), width=2)
            d.ellipse((t.x - 4, t.y - 4, t.x + 4, t.y + 4), fill=(230, 0, 0))
    if row_line is not None:
        rgb = out.convert("RGB")
        draw_row_line(rgb, row_line)
        rgb.save(path)
    else:
        out.convert("RGB").save(path)


def main() -> None:
    global SAM_PROMPT
    ap = argparse.ArgumentParser(description="Step 2 hybrid vision on one image.")
    ap.add_argument("image", type=Path)
    ap.add_argument("--prompt", default=SAM_PROMPT)
    ap.add_argument("--split", action="store_true")
    ap.add_argument("--row", action="store_true")
    ap.add_argument("--refs", nargs="+", type=Path, default=[])
    ap.add_argument("--notes", action="store_true",
                        help="enable per-frame field notes (default off; "
                             "field understanding belongs to setup)")
    ap.add_argument("--out", type=Path, default=Path("out"))
    args = ap.parse_args()
    SAM_PROMPT = args.prompt
    args.out.mkdir(parents=True, exist_ok=True)
    overlay = args.out / f"vision_{args.image.stem}.png"
    r = analyze(args.image, cache_dir=args.out / "cache", split=args.split,
                row=args.row, refs=args.refs, no_notes=(not args.notes),
                overlay_path=overlay, log_path=args.out / "api_calls.log")
    print(f"ExG threshold: {r.exg_threshold:.0f}; SAM objects: {r.n_sam_objects} "
          f"(+{r.n_extra_objects} extra, {r.n_merged} merged); judged: {len(r.targets)} "
          f"in {r.judge_batches_ok}/{r.judge_batches} batches; judge_ok: {r.judge_ok}")
    if r.row is not None:
        if r.row.get("point") is not None:
            print(f"row ({r.row['sanity']}): inliers={r.row['inliers']}")
        else:
            print(f"row: skipped ({r.row['sanity']})")
    if r.field_notes:
        print(f"field_notes: {r.field_notes}")
    print(f"SAM: {r.sam_seconds:.1f}s ({'cached' if r.sam_cached else 'fresh'}); "
          f"Spark: {r.spark_seconds:.1f}s, "
          f"{r.spark_usage.get('input_tokens', 0)}+{r.spark_usage.get('output_tokens', 0)} tok, "
          f"${r.spark_usd:.4f}")
    print(f"{'id':>3} {'action':<7} {'label':<6} {'conf':>5}  "
          f"{'spray@':<12} reason")
    for t in r.targets:
        at = f"({t.x},{t.y})" if t.action == "spray" else "-"
        print(f"{t.id:>3} {t.action:<7} {t.label:<6} {t.confidence:>5.2f}  "
              f"{at:<12} {t.reason}")
    print(f"Saved {overlay}")


if __name__ == "__main__":
    main()

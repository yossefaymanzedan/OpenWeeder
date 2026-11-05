"""Step 2 variant comparison: vision.py configurations judged on CWFID.

Variants (safety defaults: no per-frame notes; row is geometry-only):
  base       labels only
  split      + watershed split of touching plants
  row        + RANSAC row fit feeding the geometry weed rule
  split+row  + both

Exclusion for doubtful truth: the truth WEED component in carrot_field_035
with bbox x 611-990, y 656-883 (~27,600 px). Objects where it exceeds 30% of
plant pixels are excluded from object metrics; its pixels are removed from
the weed mask for pixel metrics.

Main metrics per variant: CROP_HITS, % of weed pixels within 15 px of a
spray point, % of weed pixels inside protected objects, decided_by
histogram. Object accuracy is secondary.

Run from the repo root:
    python tools/classify_test.py [--labels fields/<field>/labels/<name>.labels.json]

With --labels, truth comes from a farmer labels file (single image) instead
of CWFID *_truth.png files; outputs go to variant_labels_*.png and
classify_report_labels.md so the CWFID report is untouched.

Outputs (all under --out, default out/):
    variant_<stem>_<tag>.png  vision overlay per (image, variant)
    classify_report.md        summary + per-variant details
    api_calls.log             JSONL line per call (PROJECT.md: log every call)
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import vision
from prompt_sweep import image_pairs, load_truth

VARIANTS = {
    "base": {},
    "split": {"split": True},
    "row": {"row": True},
    "split+row": {"split": True, "row": True},
}
LABELS = ("crop", "weed", "mixed")

# Doubtful truth: carrot-like plant labeled weed in 035 (PROJECT.md).
EXCL_IMAGE = "carrot_field_035.png"
EXCL_BBOX = (611, 656, 990, 883)  # x1, y1, x2, y2
EXCL_SIZE = 27600


def find_exclusion(weed: np.ndarray) -> np.ndarray:
    """Mask of the doubtful weed component, located by its bbox + size."""
    n, lab, stats, _c = cv2.connectedComponentsWithStats(weed.astype(np.uint8), 8)
    x1t, y1t, x2t, y2t = EXCL_BBOX
    for i in range(1, n):
        x1, y1 = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
        x2 = x1 + stats[i, cv2.CC_STAT_WIDTH] - 1
        y2 = y1 + stats[i, cv2.CC_STAT_HEIGHT] - 1
        size = stats[i, cv2.CC_STAT_AREA]
        if (abs(x1 - x1t) <= 3 and abs(y1 - y1t) <= 3
                and abs(x2 - x2t) <= 3 and abs(y2 - y2t) <= 3
                and abs(size - EXCL_SIZE) / EXCL_SIZE <= 0.15):
            print(f"  exclusion component: bbox ({x1},{y1},{x2},{y2}) px={size}")
            return lab == i
    raise RuntimeError("Doubtful 035 weed component not found in truth")


def load_labels_truth(path: Path):
    """Truth pixel masks from a farmer .labels.json file.

    Only farmer-decided crop/weed labels count as truth; spark pre-labels
    the farmer never touched are excluded (judging Spark against its own
    output would be circular). Farmer skip/unsure labels are excluded too:
    unknown pixels stay out of both masks, so spraying them counts as a
    hit via truth_label's n == 0 -> "mixed" fallback.
    Returns (image path, crop mask, weed mask, meta dict).
    """
    doc = json.loads(path.read_text(encoding="utf-8"))
    w, h = doc["image_size"]
    crop = np.zeros((h, w), bool)
    weed = np.zeros((h, w), bool)
    for e in doc["labels"]:
        if e["decided_by"] != "farmer" or e["label"] not in ("crop", "weed"):
            continue
        m = np.array(Image.open(path.parent / e["mask"]).convert("L")) > 127
        if e["label"] == "crop":
            crop |= m
        else:
            weed |= m
    meta = {"changed": doc["changed"], "total": doc["total"],
            "n_farmer": sum(1 for e in doc["labels"]
                            if e["decided_by"] == "farmer")}
    return Path(doc["photo"]), crop, weed, meta


def truth_label(mask: np.ndarray, crop: np.ndarray, weed: np.ndarray) -> str:
    plant = mask & (crop | weed)
    n = int(plant.sum())
    if n == 0:
        return "mixed"
    if (mask & crop).sum() / n > 0.7:
        return "crop"
    if (mask & weed).sum() / n > 0.7:
        return "weed"
    return "mixed"


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare vision.py variants.")
    ap.add_argument("--tests", type=Path, default=Path("tests/images"))
    ap.add_argument("--out", type=Path, default=Path("out"))
    ap.add_argument("--labels", type=Path, default=None,
                    help="Farmer .labels.json file to use as truth "
                    "(single image) instead of CWFID truth files.")
    args = ap.parse_args()
    cache_dir = args.out / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.out / "api_calls.log"

    labels_mode = args.labels is not None
    truths = {}
    if labels_mode:
        image, crop, weed, meta = load_labels_truth(args.labels)
        pairs = [(image, args.labels)]
        truths[image.name] = {"crop": crop, "weed": weed, "weed_full": weed,
                              "excl": None, "size": (crop.shape[1],
                                                     crop.shape[0])}
        print(f"labels truth: {args.labels} "
              f"({meta['n_farmer']}/{meta['total']} farmer-decided, "
              f"{meta['changed']} changed)")
    else:
        pairs = image_pairs(args.tests)
        for image, truth in pairs:
            img = Image.open(image)
            crop, weed = load_truth(truth, img.size)
            excl = find_exclusion(weed) if image.name == EXCL_IMAGE else None
            weed_clean = weed & ~excl if excl is not None else weed
            truths[image.name] = {"crop": crop, "weed": weed_clean,
                                  "weed_full": weed, "excl": excl,
                                  "size": img.size}

    summary = {}
    details: dict[str, list[str]] = {}
    for tag, kw in VARIANTS.items():
        s = {"hits": [], "near_px": 0, "prot_px": 0, "weed_px": 0,
             "correct": 0, "n": 0, "rules": Counter(), "usd": 0.0,
             "sec": 0.0, "failed": 0, "excluded": 0, "errors": [],
             "per_image": []}
        tag_fs = tag.replace("+", "_")
        for image, _truth in pairs:
            t = truths[image.name]
            ov_prefix = "variant_labels_" if labels_mode else "variant_"
            r = vision.analyze(
                image, cache_dir=cache_dir, log_path=log_path,
                overlay_path=args.out / f"{ov_prefix}{image.stem}_{tag_fs}.png",
                **kw)
            s["usd"] += r.spark_usd
            s["sec"] += r.spark_seconds
            by_obj = {o["id"]: o for o in r.objects}
            h, w = t["size"][1], t["size"][0]
            prot_union = np.zeros((h, w), bool)
            spray_pts: list[tuple[int, int]] = []
            img_hits, img_corr, img_n = [], 0, 0
            for tgt in r.targets:
                o = by_obj[tgt.id]
                truth = truth_label(o["mask"], t["crop"], t["weed_full"])
                plant = o["mask"] & (t["crop"] | t["weed_full"])
                excl = t["excl"]
                frac = ((o["mask"] & excl).sum() / plant.sum()
                        if excl is not None and plant.sum() else 0.0)
                if frac > 0.30:
                    s["excluded"] += 1
                    continue
                img_n += 1
                s["n"] += 1
                s["rules"][tgt.decided_by] += 1
                if tgt.decided_by == "judge_failed":
                    s["failed"] += 1
                if tgt.label == truth:
                    img_corr += 1
                    s["correct"] += 1
                else:
                    s["errors"].append((image.name, tgt.id, truth, tgt.label,
                                        tgt.confidence, tgt.action, tgt.reason))
                if tgt.action == "spray":
                    if truth in ("crop", "mixed"):
                        s["hits"].append((image.name, tgt.id, tgt.confidence))
                        img_hits.append(tgt.id)
                    spray_pts.append((tgt.x, tgt.y))
                else:
                    prot_union |= o["mask"]
            disks = np.zeros((h, w), np.uint8)
            for x, y in spray_pts:
                cv2.circle(disks, (x, y), 15, 1, -1)
            disks = disks > 0
            wsum = int(t["weed"].sum())
            near = int((t["weed"] & disks).sum())
            prot = int((t["weed"] & prot_union).sum())
            s["near_px"] += near
            s["prot_px"] += prot
            s["weed_px"] += wsum
            acc = img_corr / img_n * 100 if img_n else 0.0
            s["per_image"].append(
                (image.name, img_n, acc, len(img_hits),
                 near / wsum * 100 if wsum else 0.0,
                 prot / wsum * 100 if wsum else 0.0,
                 r.judge_batches_ok, r.judge_batches))
            print(f"  [{tag}] {image.name}: {img_n} obj, acc {acc:.0f}%, "
                  f"hits {len(img_hits)}, batches "
                  f"{r.judge_batches_ok}/{r.judge_batches}", flush=True)
        summary[tag] = s
        d = [f"## Variant {tag}",
             f"Cost ${s['usd']:.4f}, Spark seconds {s['sec']:.0f}, "
             f"judge_failed {s['failed']}, excluded {s['excluded']}.",
             f"decided_by: {dict(s['rules'])}", "",
             "Per image (n, acc, hits, weed-px near spray / protected):"]
        for row in s["per_image"]:
            d.append(f"- {row[0]}: {row[1]} obj, {row[2]:.0f}%, {row[3]} hits, "
                     f"{row[4]:.1f}% / {row[5]:.1f}%")
        d += ["", "Errors (pred != truth):",
              "| image | id | truth | pred | conf | action | reason |"]
        d += [f"| {n} | {o} | {t} | {p} | {c:.2f} | {a} | {r_} |"
              for n, o, t, p, c, a, r_ in s["errors"]]
        d.append("")
        details[tag] = d

    report_name = ("classify_report_labels.md" if labels_mode
                   else "classify_report.md")
    rep = ["# Step 2 variant comparison (safety defaults)", "",
           f"Judge: {vision.SPARK_MODEL} via vision.py, max 8 objects/call, "
           "no per-frame notes. Spray rule: weed @ "
           f">= {vision.SPRAY_CONF} + spray-point safety margin; else protect. "
           "Row variants: row feeds ONLY the geometry rule (weed iff >2x crop "
           "band width off-row AND isolated); the judge never sees the row."]
    if labels_mode:
        rep += [f"Truth: farmer labels from {args.labels} "
                f"({meta['n_farmer']}/{meta['total']} farmer-decided); only "
                "farmer-decided crop/weed labels count, spark pre-labels and "
                "skip/unsure are excluded.", ""]
    else:
        rep += [f"Exclusion: truth weed component in {EXCL_IMAGE} bbox "
                f"{EXCL_BBOX}; objects where it exceeds 30% of plant pixels "
                "are excluded, and its pixels are removed from the weed "
                "mask.", ""]
    rep += ["## Summary",
            "| variant | CROP_HITS | weed px near spray | weed px protected | "
            "acc | cost | sec |",
            "|---|---|---|---|---|---|---|"]
    for tag in VARIANTS:
        s = summary[tag]
        acc = s["correct"] / s["n"] * 100 if s["n"] else 0.0
        rep.append(f"| {tag} | {len(s['hits'])} | "
                   f"{s['near_px'] / s['weed_px'] * 100:.1f}% | "
                   f"{s['prot_px'] / s['weed_px'] * 100:.1f}% | "
                   f"{s['correct']}/{s['n']} ({acc:.1f}%) | "
                   f"${s['usd']:.4f} | {s['sec']:.0f} |")
    rep.append("")
    for tag in VARIANTS:
        rep += details[tag]
    (args.out / report_name).write_text("\n".join(rep), encoding="utf-8")

    print("\nvariant     HITS  near-spray  protected  acc        cost     sec")
    for tag in VARIANTS:
        s = summary[tag]
        acc = s["correct"] / s["n"] * 100 if s["n"] else 0.0
        print(f"{tag:<11} {len(s['hits']):<5} "
              f"{s['near_px'] / s['weed_px'] * 100:>5.1f}%     "
              f"{s['prot_px'] / s['weed_px'] * 100:>5.1f}%    "
              f"{s['correct']}/{s['n']} ({acc:.0f}%)  "
              f"${s['usd']:.4f}  {s['sec']:.0f}")
    print(f"\nSaved {args.out / report_name}")


if __name__ == "__main__":
    main()

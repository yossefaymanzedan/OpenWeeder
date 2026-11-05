"""Does teaching from labeled photos help the judge? Measured on CWFID truth.

Teaching pool: truth plants from --teach images (CWFID train split), i.e.
what a farmer gives us by labeling those photos: pure crop / pure weed
connected components cut as padded close-ups (same cutter as the app).
Each config feeds the judge N crop + N weed references, round-robin across
the teaching photos, biggest plants first. Test images are never used to
teach. Config "0b" repeats the no-teaching baseline to show run-to-run noise.

Metrics per config:
  CROP_HITS   spray points whose 15 px disk touches truth crop pixels
  weeds hit   truth weed plants (>= 150 px) with a spray point within 15 px
  near-spray  % of truth weed pixels within 15 px of a spray point
  accuracy    object label vs truth label (70% rule), secondary
Analyses are cached per (config, image) in out/teach/ so reruns are free.

Usage: python tools/teach_test.py [--configs 0 0b 1 2 4] [--teach 2 5 7]
Data:  data/cwfid/images/NNN_image.png + annotations/NNN_annotation.png
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import vision
from app import ref_closeup
from prompt_sweep import load_truth

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "cwfid"
OUT = ROOT / "out" / "teach"
CACHE = ROOT / "out" / "cache"
LOG = ROOT / "out" / "api_calls.log"
# CWFID test split minus 028 (also listed in train) and 035 (doubtful label).
TEST = [1, 3, 4, 9, 10, 13, 15, 21, 22, 26, 29, 30, 32, 39, 44, 47, 48, 54, 60]
MIN_REF_PX = 400
MIN_WEED_PX = 150
R = vision.SAFETY_MARGIN_PX


def paths(n: int) -> tuple[Path, Path]:
    return (DATA / "images" / f"{n:03d}_image.png",
            DATA / "annotations" / f"{n:03d}_annotation.png")


def components(mask: np.ndarray, min_px: int) -> list[tuple]:
    """(area, box x1,y1,x2,y2, label mask) for each blob >= min_px."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    out = []
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if a >= min_px:
            out.append((int(a), [int(x), int(y), int(x + w), int(y + h)],
                        lab == i))
    return out


def teaching_pool(teach: list[int]) -> dict[str, list[Path]]:
    """Per label, close-ups ordered round-robin across photos, big first."""
    per = {"crop": [], "weed": []}
    for n in teach:
        img_p, tr_p = paths(n)
        crop, weed = load_truth(tr_p, Image.open(img_p).size)
        for lab, m in (("crop", crop), ("weed", weed)):
            comps = sorted(components(m, MIN_REF_PX), key=lambda c: -c[0])
            per[lab].append([ref_closeup(img_p, c[1],
                                         OUT / "refs" / f"{n:03d}_{lab}_{i}.png")
                             for i, c in enumerate(comps)])
    pool = {}
    for lab, lists in per.items():
        order = []
        for i in range(max(len(x) for x in lists)):
            order += [x[i] for x in lists if i < len(x)]
        pool[lab] = order
    return pool


def truth_label(mask: np.ndarray, crop: np.ndarray, weed: np.ndarray) -> str:
    c, w = int((mask & crop).sum()), int((mask & weed).sum())
    if c + w == 0:
        return "empty"
    if c / (c + w) > 0.7:
        return "crop"
    if w / (c + w) > 0.7:
        return "weed"
    return "mixed"


def run_one(cfg: str, n: int, refs: list) -> dict:
    dst = OUT / f"cfg_{cfg}" / f"{n:03d}.json"
    if dst.is_file():
        return json.loads(dst.read_text(encoding="utf-8"))
    img_p, tr_p = paths(n)
    crop, weed = load_truth(tr_p, Image.open(img_p).size)
    dst.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    r = vision.analyze(img_p, cache_dir=CACHE, log_path=LOG, refs=refs,
                       overlay_path=OUT / f"cfg_{cfg}" / f"{n:03d}.png")
    masks = {o["id"]: o["mask"] for o in r.objects}
    doc = {"image": n, "config": cfg, "seconds": round(time.time() - t0, 1),
           "spark_usd": r.spark_usd, "judge_ok": r.judge_ok,
           "targets": [{"id": t.id, "x": t.x, "y": t.y, "label": t.label,
                        "conf": t.confidence, "action": t.action,
                        "decided_by": t.decided_by,
                        "truth": truth_label(masks[t.id], crop, weed)
                        if t.id in masks else "empty"}
                       for t in r.targets]}
    dst.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    return doc


def score(docs: list[dict]) -> dict:
    s = {"hits": 0, "spray": 0, "weeds": 0, "weeds_hit": 0, "near": 0,
         "weed_px": 0, "n": 0, "ok": 0, "usd": 0.0, "sec": 0.0,
         "per_image_hits": []}
    for d in docs:
        img_p, tr_p = paths(d["image"])
        crop, weed = load_truth(tr_p, Image.open(img_p).size)
        disks = np.zeros(crop.shape, np.uint8)
        hits = 0
        pts = [(t["x"], t["y"]) for t in d["targets"] if t["action"] == "spray"]
        for x, y in pts:
            one = np.zeros(crop.shape, np.uint8)
            cv2.circle(one, (int(x), int(y)), R, 1, -1)
            hits += int((one.astype(bool) & crop).any())
            disks |= one
        s["hits"] += hits
        s["per_image_hits"].append(hits)
        s["spray"] += len(pts)
        near = disks.astype(bool)
        s["near"] += int((near & weed).sum())
        s["weed_px"] += int(weed.sum())
        for _a, _box, m in components(weed, MIN_WEED_PX):
            s["weeds"] += 1
            s["weeds_hit"] += int((near & m).any())
        for t in d["targets"]:
            if t["truth"] != "empty":
                s["n"] += 1
                s["ok"] += int(t["label"] == t["truth"])
        s["usd"] += d["spark_usd"]
        s["sec"] += d["seconds"]
    return s


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--configs", nargs="+", default=["0", "0b", "1", "2", "4"],
                    help="refs per label; a trailing letter marks a repeat")
    ap.add_argument("--teach", nargs="+", type=int, default=[2, 5, 7])
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    assert not set(args.teach) & set(TEST), "teaching images must not be tested"

    pool = teaching_pool(args.teach)
    print(f"teaching pool from {args.teach}: {len(pool['crop'])} crop, "
          f"{len(pool['weed'])} weed plants")
    results = {}
    for cfg in args.configs:  # configs in sequence: the first warms SAM cache
        k = int(cfg.rstrip("abcdefgh"))
        if k > min(len(pool["crop"]), len(pool["weed"])):
            raise SystemExit(f"config {cfg}: pool too small")
        refs = ([("crop", p) for p in pool["crop"][:k]]
                + [("weed", p) for p in pool["weed"][:k]])
        with ThreadPoolExecutor(args.workers) as ex:
            docs = list(ex.map(lambda n: run_one(cfg, n, refs), TEST))
        results[cfg] = score(docs)
        s = results[cfg]
        print(f"config {cfg:>3}: hits {s['hits']}, weeds hit "
              f"{s['weeds_hit']}/{s['weeds']}, ${s['usd']:.3f}")

    rows = ["| refs (crop+weed) | CROP_HITS | images with a hit | spray points | "
            "weed plants hit | weed px near spray | accuracy | Spark cost |",
            "|---|---|---|---|---|---|---|---|"]
    for cfg, s in results.items():
        k = int(cfg.rstrip("abcdefgh"))
        rep = " (repeat)" if cfg[-1].isalpha() else ""
        rows.append(
            f"| {k}+{k}{rep} | {s['hits']} | "
            f"{sum(1 for h in s['per_image_hits'] if h)}/{len(TEST)} | "
            f"{s['spray']} | {s['weeds_hit']}/{s['weeds']} "
            f"({s['weeds_hit'] / max(s['weeds'], 1) * 100:.0f}%) | "
            f"{s['near'] / max(s['weed_px'], 1) * 100:.1f}% | "
            f"{s['ok']}/{s['n']} ({s['ok'] / max(s['n'], 1) * 100:.0f}%) | "
            f"${s['usd']:.3f} |")
    report = [
        "# Teaching test: do labeled examples help the judge?", "",
        f"Teaching photos (CWFID train): {', '.join(f'{n:03d}' for n in args.teach)}"
        f" — pool {len(pool['crop'])} crop / {len(pool['weed'])} weed plants.",
        f"Test photos (CWFID test split, never taught): {len(TEST)} — "
        f"{', '.join(f'{n:03d}' for n in TEST)}.",
        "Spray rule: weed >= 0.85, 15 px safety margin. CROP_HITS is physical "
        "(spray disk r=15 px touches truth crop).", "", *rows, ""]
    (ROOT / "out" / "teach_report.md").write_text("\n".join(report),
                                                  encoding="utf-8")
    print("\n".join(rows))
    print(f"\nSaved {ROOT / 'out' / 'teach_report.md'}")


if __name__ == "__main__":
    main()

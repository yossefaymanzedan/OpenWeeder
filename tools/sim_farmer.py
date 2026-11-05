"""Simulated farmer for PIPELINE CHECKS ONLY (lesson 11).

Uses CWFID truth (data/cwfid/annotations/NNN_annotation.png, green = crop,
red = weed) in place of the farmer, so the step-4 pipeline can be run end to
end before real tray photos exist. These numbers never go into the pitch.
Photo names must look like carrot_NNN (NNN = CWFID image number).

    python tools/sim_farmer.py make <field> --from <field>   new sim field
    python tools/sim_farmer.py label <field>    farmer labels = truth (labeled photos)
    python tools/sim_farmer.py review <field>   review the current run: fix every
                                                wrong label, mark all reviewed
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

import setup

DATA = Path("data") / "cwfid" / "annotations"


def truth(name: str, size: tuple) -> tuple[np.ndarray, np.ndarray]:
    m = re.search(r"(\d{3})$", name)
    if not m:
        raise ValueError(f"{name}: not a CWFID photo name (carrot_NNN)")
    t = np.asarray(Image.open(DATA / f"{m.group(1)}_annotation.png").convert("RGB")
                   .resize(size, Image.NEAREST)).astype(int)
    r, g, b = t[..., 0], t[..., 1], t[..., 2]
    return (g > 150) & (r < 100) & (b < 100), (r > 150) & (g < 100) & (b < 100)


def farmer_label(mask: np.ndarray, crop: np.ndarray, weed: np.ndarray) -> str:
    """What a careful farmer taps: crop / weed, or skip when tangled or empty."""
    lab = setup.truth_of(mask, crop, weed)
    return lab if lab in ("crop", "weed") else "skip"


def cmd_make(field: str, src: str) -> None:
    """Copy photos + the labeled photos' SAM objects from an existing field."""
    dst = setup.fdir(field)
    if dst.exists():
        raise SystemExit(f"{dst} exists")
    s = setup.fdir(src)
    shutil.copytree(s / "photos", dst / "photos")
    shutil.copytree(s / "labels", dst / "labels")
    meta = json.loads((s / "field.json").read_text(encoding="utf-8"))
    meta.update(crop="carrot", plan={"photos": 10, "label": 3})
    (dst / "field.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"made {dst}: {len(list((dst / 'photos').iterdir()))} photos, "
          f"labels for {len(list((dst / 'labels').glob('*.labels.json')))}")


def cmd_label(field: str) -> None:
    for lp in sorted((setup.fdir(field) / "labels").glob("*.labels.json")):
        doc = json.loads(lp.read_text(encoding="utf-8"))
        crop, weed = truth(doc["name"], tuple(doc["image_size"]))
        counts = {"crop": 0, "weed": 0, "skip": 0}
        for o in doc["labels"]:
            m = np.asarray(Image.open(lp.parent / o["mask"]).convert("L")) > 127
            o["label"] = farmer_label(m, crop, weed)
            o["decided_by"] = "farmer"
            o["changed"] = o["label"] != o.get("spark_label")
            counts[o["label"]] += 1
        doc["changed"] = sum(o["changed"] for o in doc["labels"])
        lp.write_text(json.dumps(doc, indent=1), encoding="utf-8")
        print(f"{doc['name']}: {counts}")


def cmd_review(field: str) -> None:
    runs = setup.fdir(field) / "runs"
    run = json.loads((runs / "run.json").read_text(encoding="utf-8"))
    corr = setup.load_corrections(field)
    reviewed = setup.load_reviewed(field)
    for e in run["entries"]:
        img = Image.open(setup.photo_path(field, e["name"]))
        crop, weed = truth(e["name"], img.size)
        masks = setup.load_masks(runs / f"{e['name']}.masks.npz")
        fixes = {}
        for o in e["objects"]:
            want = farmer_label(masks[o["id"]], crop, weed)
            if want != o["label"] and not (want == "skip" and o["label"] == "unsure"):
                fixes[str(o["id"])] = want
        corr[e["name"]] = fixes
        reviewed[e["name"]] = True
        print(f"{e['name']}: {len(fixes)} fixes")
    (runs / "corrections.json").write_text(json.dumps(corr, indent=1), encoding="utf-8")
    setup.save_reviewed(field, reviewed)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["make", "label", "review"])
    ap.add_argument("field")
    ap.add_argument("--from", dest="src")
    a = ap.parse_args()
    if a.cmd == "make":
        cmd_make(a.field, a.src)
    elif a.cmd == "label":
        cmd_label(a.field)
    else:
        cmd_review(a.field)


if __name__ == "__main__":
    main()

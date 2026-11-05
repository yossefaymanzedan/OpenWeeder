"""Make clean soil base photos (no plants) for one field, with Muse Image.

    python tools/make_soil_bases.py field --light sun  3 hilled-ridge bases (default shape)
    python tools/make_soil_bases.py potting --bed raised  3 wide raised-bed bases
    python tools/make_soil_bases.py potting --bed flat 3 flat bases
    python tools/make_soil_bases.py potting --dry      show the prompts, no cost

Base A is generated from text. Bases B, C... are EDITS of A ("a different spot
of the same field"), so they share the same soil type, color, moisture, light
and bed shape, but the small surface details differ, like real spots in one field.
Ridge (default) = a narrow hilled ridge with steep sides and deep furrows, like
a potato-ridge field; raised = a wide flat-topped bed; flat = level soil.
Output: soil/<soil>_ridge_a.jpg ... (flat: soil/<soil>_a.jpg ...), a .json
next to each saying its bed shape (make_dataset.py reads it), and a _sheet.jpg.
$0.01 per image. Replace them with real photos of your tray soil when you can.
"""
from __future__ import annotations

import argparse
import json
import string
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_image  # noqa: E402
from make_dataset import (BED_TEXT, CAMERA, COLOR_KEYS, LIGHT_KEYS, SOIL_KEYS,  # noqa: E402
                          TAIL, bed_text, edit_image, fill_color, parse_width,
                          soil_for)

SPOT_EDIT = ("Edit this photo to show a DIFFERENT spot of the same field, a few "
             "steps away: keep exactly the same soil type, color, moisture, "
             "lighting, camera height and angle{shape}. Only rearrange the small "
             "surface details (crumbs, specks, tiny clods, faint marks) the way "
             "another area of the same soil would look. {extra} No plants, no "
             "weeds, no leaves, no text, no frame, no vignette.")
KEEP_SHAPE = {
    "raised": (", and the same raised bed: its nearly flat top down the middle "
               "of the frame running past the top and bottom edges, the gentle "
               "slopes at the left and right edges, same width and position"),
    "ridge": (", and the same hilled ridge: its narrow top down the middle of "
              "the frame running past the top and bottom edges, the steep slopes "
              "down into the furrows at the left and right, same width and position"),
}
EXTRAS = ["", "Add one faint slightly darker damp patch.",
          "Add a few slightly smaller crumbs.",
          "Add a few slightly larger crumbs.", ""]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("soil", help="soil key (" + ", ".join(SOIL_KEYS)
                                 + ") or your own description in quotes")
    ap.add_argument("--n", type=int, default=3, help="how many bases (default 3)")
    ap.add_argument("--light", default="indoor", help=", ".join(LIGHT_KEYS))
    ap.add_argument("--bed", default="ridge", choices=list(BED_TEXT),
                    help="ridge (narrow hilled ridge, default), raised (wide "
                         "flat-topped bed) or flat")
    ap.add_argument("--color", help="soil color: " + ", ".join(COLOR_KEYS)
                                    + " or your own words (default: the soil's own)")
    ap.add_argument("--top-width", default=None,
                    help='width of the raised top in cm, e.g. "10" (a range like '
                         '"8-14" picks one value). Default: 11 for ridge, 23 for raised')
    ap.add_argument("--name", help="file prefix (default: the soil key + "
                                   "_<shape>, e.g. field_ridge; flat: just the soil key)")
    ap.add_argument("--size", default="1296x972")
    ap.add_argument("--out", type=Path, default=Path("soil"))
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()

    color = COLOR_KEYS.get(a.color.lower(), a.color) if a.color else None
    soil = soil_for(fill_color(SOIL_KEYS.get(a.soil, a.soil), color), a.bed)
    light = LIGHT_KEYS.get(a.light, a.light)
    name = a.name or ((a.soil if a.soil in SOIL_KEYS else "custom")
                      + ("" if a.bed == "flat" else f"_{a.bed}"))
    lo, hi = parse_width(a.top_width, a.bed)
    top = (lo + hi) // 2 if BED_TEXT[a.bed] else None  # one base = one width
    shape = f" {bed_text(a.bed, top)}" if BED_TEXT[a.bed] else ""
    prompt_a = (f"{CAMERA} The ground is {soil}.{shape} Only bare soil: no plants, "
                f"no weeds, no seedlings, no leaves, no roots. {light} {TAIL}")
    keep = KEEP_SHAPE.get(a.bed, "")
    edits = [SPOT_EDIT.format(shape=keep, extra=EXTRAS[i % len(EXTRAS)])
             .replace("  ", " ") for i in range(1, a.n)]
    if a.dry:
        print(f"--- {name}_a (generate)\n{prompt_a}\n")
        for i, e in enumerate(edits, 1):
            print(f"--- {name}_{string.ascii_lowercase[i]} (edit of a)\n{e}\n")
        print(f"{a.n} images, about ${a.n * 0.01:.2f}. Nothing sent.")
        return

    a.out.mkdir(parents=True, exist_ok=True)
    log = Path("out") / "api_calls.log"
    first = a.out / f"{name}_a.jpg"
    if not first.is_file():
        saved = gen_image.generate(prompt_a, a.out, 1, a.size, "jpeg",
                                   name=f"tmp_{name}", log_path=log)
        saved[0].replace(first)
        saved[0].with_suffix(".txt").unlink(missing_ok=True)
    made = [first]
    for i, e in enumerate(edits, 1):
        out = a.out / f"{name}_{string.ascii_lowercase[i]}.jpg"
        if not out.is_file():
            edit_image(first, e, out, log)
        made.append(out)
    for p in made:  # make_dataset.py reads the shape from here
        p.with_suffix(".json").write_text(json.dumps(
            {"bed": a.bed, "soil": a.soil, "light": a.light, "color": a.color,
             "top_width_cm": top}, indent=1),
            encoding="utf-8")

    thumb, th = 400, 300
    sheet = Image.new("RGB", (thumb * len(made), th + 20), "white")
    d = ImageDraw.Draw(sheet)
    for n, p in enumerate(made):
        im = Image.open(p).convert("RGB")
        im.thumbnail((thumb, th))
        sheet.paste(im, (n * thumb, 0))
        d.text((n * thumb + 4, th + 4), p.name, fill="black")
    sheet_path = a.out / f"{name}_sheet.jpg"
    sheet.save(sheet_path, quality=88)
    print("bases:", " ".join(str(p) for p in made))
    print(f"check {sheet_path}: same soil, no plants, small differences only.")


if __name__ == "__main__":
    main()
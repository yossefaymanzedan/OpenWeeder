"""Make a synthetic image dataset with Muse Image: as many images as you ask.

    python tools/make_dataset.py 20                       20 tomato images
    python tools/make_dataset.py 30 --dry                 show the prompts, no cost
    python tools/make_dataset.py 50 --weeds purslane=4,grass=2,pigweed=1,dandelion=1
    python tools/make_dataset.py 10 --soil potting --light indoor
                          one soil and one light for every image (like one field)
    python tools/make_dataset.py 40 --bed ridge --soil field --light sun
                          all on hilled ridges, like a potato-ridge field
    python tools/make_dataset.py 40 --bed ridge --soil field --color light=3,dark=1
                          soil color: one name, a weighted mix, or your own words
                          (black, dark, brown, light, tan, sand, red, grey)
    python tools/make_dataset.py 40 --bed ridge --top-width 6-12
                          width of the raised top in cm: every image gets a random
                          width in the range ("10" = always 10 cm)
    python tools/make_dataset.py 40 --bed ridge=3,raised=1,flat=1   (default mix)
                          ridge  = narrow hilled ridge, steep sides, crop on top,
                                   deep furrows both sides (main focus)
                          raised = wide flat-topped bed with gentle slopes
                          flat   = level ground
                          weeds go in the furrows, on the slopes or in the row
    python tools/make_dataset.py 10 --base-image soil/potting_a.jpg soil/potting_b.jpg soil/potting_c.jpg --field syn-1
                          EDIT mode: images start from these soil photos of one
                          field (rotated), only plants are added
    python tools/make_dataset.py 20 --reference farmer1.jpg farmer2.jpg --layout scattered
                          REFERENCE mode: the farmer's own photos (any angle) guide the
                          plants, weeds, soil and light; every image is a NEW top view
                          in a different arrangement. --layout row | scattered | mixed
                          (scattered = no rows, flat ground; also works without --reference)
    python tools/make_dataset.py --make-base data/synthetic/tomato_v1/img_004.jpg photos/base_soil.jpg
                          remove all plants from a photo to get a clean soil base

Every image gets its own prompt, built from a weighted random mix of growth
stage, spacing, weed types, how much the weeds touch the crop, lighting and
soil, plus crop-only and weed-only scenes. The camera text describes a
Raspberry Pi Camera Module 3 pointing straight down from 25 cm (no fisheye).
Output:

    data/synthetic/<name>/img_001.jpg ...    the images
    data/synthetic/<name>/manifest.json      prompt + settings of every image
    data/synthetic/<name>/contact_sheet.jpg  all images as numbered thumbnails

Safe to stop and re-run: finished images are kept and only missing ones are
made (same --seed gives the same prompts). $0.01 per image; failed or
safety-blocked images are not billed.

NO PLANT-LEVEL ANSWERS COME WITH THE IMAGES. Only the scenario is known:
crop_only images should get zero sprays, weed_only images should have zero
plants called crop (use these as free sanity checks). These are generated
pictures: use them to test the pipeline and to train, never for numbers you
show as results (PROJECT.md lesson 11).
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import random
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw

import gen_image

IMAGE_MODEL = "muse-image-1.0"
BASE_URL = "https://api.meta.ai/v1"
USD_PER_IMAGE = 0.01

CAMERA = ("A realistic photo from a Raspberry Pi Camera Module 3 (standard "
          "rectilinear lens, about 66 degrees field of view) mounted 25 cm above "
          "the soil, pointing straight down. The frame shows about 32 cm x 24 cm "
          "of ground.")
TAIL = ("Flat rectangular 4:3 image, edge to edge: no fisheye, no barrel "
        "distortion, no vignette, no circular frame, no dark corners. Sharp focus "
        "across the frame. Plain, unstaged, like a robot test photo. No people, "
        "no hands, no tools, no text, no watermark.")
EDIT_HEAD = ("Edit this photo. Keep the soil, its texture, color, moisture, the "
             "lighting and the camera angle exactly as they are. Do not move or "
             "restyle the soil. Only add plants:")
EDIT_TAIL = ("The plants must sit naturally in this soil with matching light and "
             "soft shadows. No other changes, no text, no frame, no vignette.")
REF_HEAD = ("This photo was taken by a farmer in their own field. Make a NEW photo: "
            "the same kind of plants, weeds, soil color and texture and lighting as "
            "this reference, but a different patch of ground, seen from straight "
            "ABOVE (top view, camera pointing straight down, no horizon, no tilt, "
            "no side view). Do not copy the reference's layout or camera angle. "
            "Show:")
REF_TAIL = ("Sharp, realistic, unstaged, like a robot test photo, flat rectangular "
            "4:3 image edge to edge, no fisheye, no vignette, no people, no hands, "
            "no tools, no text, no watermark.")
REF_KEEP_HEAD = ("Edit this photo into a DIFFERENT photo taken a moment later with the same "
                 "phone in the same place, aimed at a different spot. Keep exactly as in "
                 "the photo: the camera angle, perspective and framing style, the lighting "
                 "and shadows, the photo quality, grain and sharpness, the soil type, color "
                 "and clods, the tray edge and the background. Do NOT make it top-down, "
                 "do not clean it up, do not restyle it. The image must stay a VERTICAL "
                 "portrait, taller than it is wide, like the original. Change only the plants: remove "
                 "the current plants and show:")
REF_KEEP_TAIL = ("It must look like a real, unedited phone photo of a real tray, not an "
                 "illustration, not a 3D render, not painted, with the same slightly "
                 "imperfect everyday quality as the original. No text, no watermark.")
PLACEMENT_SCATTERED = [
    ("spread randomly between and around the crop plants", 2),
    ("some right next to a crop plant, some far from any", 2),
    ("tangled with a crop plant, leaves overlapping the crop leaves", 1),
    ("only a few small ones here and there", 1),
    ("dense, covering much of the soil", 1),
]
MAKE_BASE_PROMPT = ("Edit this photo. Remove every plant, weed and leaf. Fill those "
                    "areas with the same soil so it looks untouched. Keep the soil "
                    "texture, color, moisture, lighting and camera angle exactly as "
                    "they are. No other changes, no text, no frame, no vignette.")

CROPS = {
    "molokhia": ("molokhia (jute mallow) plants: each a low rosette of soft, slightly "
                 "hairy, oval pale-green leaves with finely toothed edges and visible "
                 "veins, leaves held close to the ground"),
    "tomato": ("tomato seedlings (broad toothed leaflets, not feathery, compound "
               "leaves, slightly hairy stems, two narrow seed leaves at the base)"),
    "carrot": "carrot seedlings (thin feathery finely divided leaves on thin stems)",
    "basil": "basil seedlings (smooth glossy oval leaves with a pointed tip)",
    "chives": "chive seedlings (thin upright hollow grass-like green blades)",
    "onion": "green onion plants (thin upright hollow tubular leaves)",
    "lettuce": "lettuce seedlings (soft rounded ruffled leaves in a small rosette)",
}
# (text, weight): the 15-20 cm stage is rare because the real tray is small.
STAGES = [
    ("just emerged and only 3 to 5 cm tall", 3),
    ("young and 5 to 10 cm tall", 4),
    ("larger, 15 to 20 cm tall, with leaves starting to overlap", 1),
]
SPACING = [
    ("well spaced with clear soil between them", 1),
    ("close together", 1),
    ("touching each other", 1),
]
WEEDS = {  # --weeds names -> how the prompt describes them
    "purslane": "purslane (small round fleshy leaves, reddish stems, lying flat)",
    "grass": "thin grass blades",
    "clover": "clover (three small round leaflets)",
    "chickweed": "chickweed (small pointed oval leaves on thin trailing stems)",
    "dandelion": "young dandelions (jagged leaves in a flat rosette)",
    "pigweed": "pigweed seedlings (diamond-shaped leaves, upright stem)",
}
PLACEMENT = [
    ("all between the rows, away from the crop", 1),
    ("some between the rows and some squeezed inside the row next to the crop", 2),
    ("tangled with the crop, leaves overlapping the crop leaves", 1),
    ("only a few small scattered ones", 1),
    ("dense, covering much of the soil around the row", 1),
]
SOIL_KEYS = {  # {color} is filled from --color, else the soil's own default
    "potting": ("uniform {color} peat-based potting mix with a fine, crumbly "
                "texture, a few tiny white perlite specks, slightly moist, a flat "
                "level surface, no stones, no clods, no cracks, no mulch"),
    "sandy": ("{color} dry sandy soil with a fine even grain, a flat level surface, "
              "no stones, no cracks"),
    "garden": ("{color} garden soil, fine and loose, a few small pebbles, "
               "a flat level surface"),
    "field": ("{color} dry cultivated field soil, freshly hoed, loose and "
              "crumbly with small clods and a few tiny pebbles"),
    # Built from two real potato-ridge photos (soil/refs/): colors measured
    # (Lab L 36 / 59, almost no yellow: grayish, not orange) and described by
    # Muse Spark; the tractor-tread detail was added by eye.
    "loam": ("{color} damp, finely tilled loam, matte with no shine. Almost all of "
             "the surface is loose granular soil made of fine crumbs mixed with "
             "tiny fibrous organic bits, with a light scattering of 1 to 3 cm "
             "slightly lighter soil clods and "
             "no stones. The ridges are rough and crumbly everywhere, with a "
             "natural uneven grain, never smooth, never molded or sculpted. The "
             "soil in the narrow bottom of the furrow is only slightly smoother. "
             "The surface has no tire tracks, no wheel ruts, no tread patterns, "
             "no footprints, no puddles, no wet or dark patches, no stains"),
    "silt": ("{color} dry, soft silt loam, matte and powdery, in even soft light. "
             "The surface is mostly fine smooth dust and tiny crumbs, with only a "
             "few small soft clods up to 2 cm that crush easily, exactly the same "
             "color as the dust around them (never whiter or grayer, never like "
             "white rocks or gravel). The furrow bottom is loose, churned "
             "and slightly uneven. The whole surface is free of any marks: no tire "
             "tracks, no wheel ruts, no tread patterns, no footprints. No cracks, "
             "no stones, no gravel, no shine, no harsh contrast"),
}
SOIL_DEFAULT_COLOR = {"potting": "dark brown", "sandy": "pale yellow-beige",
                      "garden": "medium brown", "field": "light brown",
                      "loam": "extremely dark brown, almost black-brown, like wet "
                              "peat compost in shade or freshly ground dark coffee: "
                              "a warm very dark brown (about RGB 36, 21, 9). Not "
                              "red, not orange, not clay, not gray, not light "
                              "brown",
                      "silt": "warm light tan, like pale khaki, only a slight "
                              "gray cast"}
COLOR_KEYS = {  # --color names -> words in the prompt
    "black": "very dark brown, almost black",
    "dark": "dark brown",
    "brown": "medium brown",
    "light": "light brown",
    "tan": "tan, yellowish brown",
    "sand": "pale yellow-beige",
    "red": "reddish brown, iron-rich",
    "grey": "grey-brown",
}
# Top width of the hilled part, in cm, as (low, high); one value is picked per
# image. The camera frame is only about 32 cm wide.
TOP_WIDTH_DEFAULT = {"ridge": (8, 14), "raised": (18, 28)}


def fill_color(soil: str, color: str | None) -> str:
    """Put a color into a soil text: the chosen one, else that soil's default."""
    if color is None:
        key = next((k for k, v in SOIL_KEYS.items() if v == soil), None)
        color = SOIL_DEFAULT_COLOR.get(key, "brown")
    return soil.replace("{color}", color)


def parse_colors(spec: str | None) -> list[tuple[str, float]] | None:
    """--color "light" / "light=3,dark=1" / "reddish clay" -> [(words, weight)].
    No spec = None (each soil keeps its own default color)."""
    if not spec:
        return None
    out = []
    for part in spec.split(","):
        name, _, w = part.strip().partition("=")
        name = name.strip()
        if not name:
            continue
        try:
            weight = float(w) if w else 1.0
        except ValueError:
            raise SystemExit(f"--color: bad weight in '{part}' (use light=3)")
        if weight <= 0:
            raise SystemExit(f"--color: weight must be > 0 in '{part}'")
        out.append((COLOR_KEYS.get(name.lower(), name), weight))
    if not out:
        raise SystemExit("--color: no colors given")
    return out


def parse_width(spec: str | None, bed: str) -> tuple[int, int]:
    """--top-width "8-14" or "10" (cm) -> (low, high) for this shape."""
    if not spec:
        return TOP_WIDTH_DEFAULT.get(bed, (10, 10))
    try:
        lo, _, hi = spec.replace(" ", "").partition("-")
        lo_i, hi_i = int(float(lo)), int(float(hi or lo))
    except ValueError:
        raise SystemExit(f'--top-width: use "10" or "8-14" (cm), not "{spec}"')
    if lo_i > hi_i:
        lo_i, hi_i = hi_i, lo_i
    if lo_i < 3 or hi_i > 28:
        raise SystemExit("--top-width: between 3 and 28 cm (the camera frame is "
                         "only about 32 cm wide, so wider leaves no room for furrows)")
    return lo_i, hi_i
LIGHT_KEYS = {
    "indoor": "Indoor even lighting from above, soft shadows.",
    "overcast": "Overcast daylight, soft shadows.",
    "afternoon": "Warm late-afternoon daylight with mild soft shadows.",
    "sun": ("Bright direct sunlight from one side: one side of each slope lit, "
            "the other in soft shade, crisp short shadows under the leaves."),
    # For dark soils: asking for a "dark scene" makes the whole picture look
    # like dusk, so say the soil is dark but the exposure is a normal bright day.
    "noon": ("Bright, even, diffuse midday daylight under a high overcast sky, "
             "normal well-exposed photo with correct exposure. The soil is "
             "naturally dark brown, but the lighting is bright and clear, with "
             "visible soft shadows between the crumbs, and the highlights on the "
             "soil are bright, like a photo taken at noon."),
}
LIGHT = [(LIGHT_KEYS["indoor"], 2), (LIGHT_KEYS["overcast"], 1),
         (LIGHT_KEYS["afternoon"], 1), (LIGHT_KEYS["sun"], 2)]
SOIL = [(SOIL_KEYS["potting"], 1), (SOIL_KEYS["sandy"], 1), (SOIL_KEYS["garden"], 1),
        (SOIL_KEYS["field"], 1), (SOIL_KEYS["loam"], 4), (SOIL_KEYS["silt"], 4)]
# Scene mix: mostly crop + weeds; crop-only and weed-only scenes double as
# free sanity checks (no sprays allowed / no crop allowed).
SCENARIOS = [("mixed", 0.70), ("crop_only", 0.15), ("weed_only", 0.15)]

# Bed shape. Real fields mostly grow the crop on a raised bed: a ridge of soil
# (an upside-down U in cross-section) with lower furrows on both sides.
BED_TEXT = {
    # Written as the robot sees it from 25 cm: only a short piece of a long
    # bed, its top filling the middle. Earlier wording ("mound", "10 cm high",
    # "upside-down U") produced a short loaf-shaped dome with black furrows.
    "raised": ("The camera looks straight down at the top of a long raised "
               "planting bed in a vegetable field. The bed is straight and runs "
               "through the whole frame, past the top edge and past the bottom "
               "edge, so neither end of it is visible. Its top is nearly flat, "
               "only slightly rounded, about {top} cm wide, and fills the middle of "
               "the frame. Only along the left and right edges does the soil "
               "slope gently down toward the furrows beside the bed, a few "
               "centimeters lower. It looks like an ordinary hoe-made garden bed: "
               "slightly uneven, a few faint tool marks, the sides a little more "
               "compact. Not a dome, not a mound, not a loaf shape; the light is "
               "even, with only soft shade on the slopes."),
    # Hilled ridges like a potato/vegetable ridge field: narrow, steep sides,
    # deep furrows. From 25 cm the robot sees one ridge top and its two slopes.
    "ridge": ("The camera looks straight down at the top of one long hilled "
              "ridge in a cultivated vegetable field, like a field of potato "
              "ridges. The ridge is straight and runs through the whole frame, "
              "past the top edge and past the bottom edge. Its top is narrow, "
              "about {top} cm wide and slightly rounded, a strip down the middle of "
              "the frame where the plants grow. From there both sides slope "
              "steeply down, about 15 cm, into the furrows that fill the left "
              "and right parts of the frame, so the soil is lowest at the left "
              "and right edges. In cross-section the ridge is a triangle with a "
              "rounded top. The soil looks freshly hilled with a hoe, with small "
              "clods on the slopes. Not a dome, not a loaf shape, not a mound "
              "with an end. The camera axis is exactly vertical (a straight "
              "top-down view): the ridge edges are parallel straight lines and do "
              "not converge toward the top of the frame."),
    "flat": "",  # the soil text already says flat and level
}
BEDS_DEFAULT = "ridge=3,raised=1,flat=1"  # main focus: hilled ridges
HILLED = ("ridge", "raised")  # shapes with a top, slopes and furrows


def bed_text(bed: str, top: int | None = None) -> str:
    """Prompt text for a bed shape, with its top width (cm) filled in."""
    return BED_TEXT[bed].replace("{top}", str(top or TOP_WIDTH_DEFAULT.get(bed, (10, 10))[0]))
PLACEMENT_RAISED = [
    ("down in the furrows on both sides, away from the crop", 1),
    ("on the sloping sides of the bed and in the furrows", 2),
    ("on top of the bed, squeezed into the row next to the crop", 2),
    ("tangled with the crop on top of the bed, leaves overlapping the crop leaves", 1),
    ("only a few small scattered ones", 1),
    ("dense, filling the furrows and climbing the sides of the bed", 1),
]


# --placement names -> weed layout text (or pass your own words)
PLACEMENT_KEYS = {
    "separated": ("every weed is its own small individual plant, clearly apart "
                  "from the other weeds with bare soil visible around it. About "
                  "a third of the weeds grow right beside the crop row, one or "
                  "two centimeters from a tomato plant, close but not touching "
                  "or overlapping it; the rest grow farther away on the slopes "
                  "and in the furrows at different distances. No mats, no dense "
                  "clusters, never covering a crop plant"),
}


def parse_beds(spec: str) -> list[tuple[str, float]]:
    """--bed "ridge" / "flat" / "ridge=3,raised=1,flat=1" -> [(bed, weight)]."""
    out = []
    for part in spec.split(","):
        name, _, w = part.strip().partition("=")
        name = name.strip().lower()
        if name not in BED_TEXT:
            raise SystemExit(f"--bed: unknown shape '{name}' (use "
                             + ", ".join(BED_TEXT) + ")")
        try:
            weight = float(w) if w else 1.0
        except ValueError:
            raise SystemExit(f"--bed: bad weight in '{part}' (use raised=3)")
        if weight <= 0:
            raise SystemExit(f"--bed: weight must be > 0 in '{part}'")
        out.append((name, weight))
    return out


def base_bed(base: Path) -> str:
    """Shape of a soil base photo: its .json sidecar (make_soil_bases.py writes
    one), else the shape named in the file name (ridge/raised), else flat."""
    side = base.with_suffix(".json")
    if side.is_file():
        try:
            bed = json.loads(side.read_text(encoding="utf-8")).get("bed")
            if bed in BED_TEXT:
                return bed
        except json.JSONDecodeError:
            pass
    stem = base.stem.lower()
    return next((b for b in HILLED if b in stem), "flat")


def soil_for(soil: str, bed: str) -> str:
    """Drop 'flat level surface' from a soil description on a hilled shape."""
    if bed not in HILLED:
        return soil
    return (soil.replace("a flat level surface, ", "")
            .replace(", a flat level surface", "")
            .replace("a flat level surface", "").strip(", "))


def wchoice(rng: random.Random, items: list[tuple[str, float]]) -> str:
    return rng.choices([t for t, _ in items], [w for _, w in items])[0]


def parse_weeds(spec: str | None) -> dict[str, float]:
    """--weeds "purslane=3,grass=1" -> {description: weight}. A name that is
    not in WEEDS is used as its own description (e.g. "bindweed"). No spec =
    every known weed, equally likely."""
    if not spec:
        return {d: 1.0 for d in WEEDS.values()}
    out = {}
    for part in spec.split(","):
        name, _, w = part.strip().partition("=")
        if not name.strip():
            continue
        try:
            weight = float(w) if w else 1.0
        except ValueError:
            raise SystemExit(f"--weeds: bad weight in '{part}' (use name=3)")
        if weight <= 0:
            raise SystemExit(f"--weeds: weight must be > 0 in '{part}'")
        out[WEEDS.get(name.strip().lower(), name.strip())] = weight
    if not out:
        raise SystemExit("--weeds: no weed names given")
    return out


def pick_weeds(rng: random.Random, pool: dict[str, float], k: int) -> list[str]:
    """k different weeds, chosen by weight; the first is the main weed."""
    names, weights = list(pool), list(pool.values())
    out = []
    while names and len(out) < k:
        i = rng.choices(range(len(names)), weights)[0]
        out.append(names.pop(i))
        weights.pop(i)
    return out


def plan_image(rng: random.Random, crop: str, pool: dict[str, float],
               max_types: int = 3, edit: bool = False, bed: str = "flat",
               look: random.Random | None = None,
               colors: list[tuple[str, float]] | None = None,
               width: str | None = None, force_scenario: str | None = None,
               force_placement: str | None = None, layout: str = "row",
               reference: bool = False, ref_style: str = "top") -> dict:
    """`look` is a separate random stream for soil color and top width, so
    --color / --top-width never change the weeds, stages or scenes a seed gives.
    force_scenario / force_placement override the random pick (the random draws
    are still made, so the rest of a seed's plan does not change)."""
    look = look or random.Random(0)
    raised = bed in HILLED
    word = "ridge" if bed == "ridge" else "bed"  # what the prompt calls the shape
    scenario = rng.choices([s for s, _ in SCENARIOS],
                           [w for _, w in SCENARIOS])[0]
    scenario = force_scenario or scenario
    k = min(rng.choice([1, 2, 2, 3]), max_types, len(pool))
    scattered = layout == "scattered"
    placement = wchoice(rng, PLACEMENT_SCATTERED if scattered else
                        PLACEMENT_RAISED if raised else PLACEMENT)
    placement = PLACEMENT_KEYS.get(force_placement, force_placement) or placement
    lo, hi = parse_width(width, bed)
    top = look.randint(lo, hi) if raised else None
    color = wchoice(look, colors) if colors else None
    soil_text = fill_color(wchoice(rng, SOIL), color)
    if not raised:  # a level surface has no furrows
        soil_text = soil_text.replace(" The furrow bottom is loose, churned and slightly uneven.", "")
    f = {"scenario": scenario, "crop": crop, "bed": bed, "top_width_cm": top,
         "stage": wchoice(rng, STAGES), "spacing": wchoice(rng, SPACING),
         "weeds": pick_weeds(rng, pool, k),
         "placement": placement.replace(" bed", f" {word}"),
         "light": wchoice(rng, LIGHT), "soil": soil_for(soil_text, bed)}
    look = CROPS.get(crop, f"{crop} seedlings")
    where = (f"running along the top of the {'raised bed' if word == 'bed' else word}, "
             "through the center of the frame from top to bottom" if raised else
             "running through the center of the frame from top to bottom")
    n_scat = rng.randint(6, 12)
    scat = (f"{n_scat} SEPARATE {look} scattered at random positions across the whole "
            f"frame, NO rows and NO lines: they are not aligned, the gaps are very "
            f"uneven, some stand alone and some in loose clusters; each grows from the "
            f"soil on its own short stem, seen from above as small rosettes; the "
            f"plants are {f['stage']}, {f['spacing']}")
    row = scat if scattered else (f"a straight line of 6 to 8 SEPARATE {look} {where}, each growing from "
           f"the soil on its own short stem, seen from above as small rosettes; "
           f"the plants are {f['stage']}, {f['spacing']}. The row looks hand-"
           f"planted, not machine-perfect: the gaps between plants are uneven, "
           f"the plants differ a little in size, and one or two sit slightly off "
           f"the line")
    main, others = f["weeds"][0], f["weeds"][1:]
    weeds = (f"mostly {main}, with a few {' and '.join(others)}" if others
             else main)
    soil = "the soil"  # the soil itself is described once (generate) or kept (edit)
    if scenario == "mixed":
        plants = f"{row}. Small weeds grow in {soil}: {weeds}; {f['placement']}."
    elif scenario == "crop_only":
        rest = f"the {word}, its sides and the furrows stay clean" if raised else \
            f"the rest of {soil} stays clean"
        plants = f"{row}. No weeds at all: {rest}."
    else:
        across = f"on the {word}, its sides and in the furrows" if raised else \
            f"scattered across {soil}"
        plants = f"no crop plants at all, only small weeds {across}: {weeds}."
    f["plants"] = plants
    f["layout"] = layout
    if reference and ref_style == "keep":
        n_crop, n_grass = rng.choice([1, 2, 2, 3]), rng.choice([2, 3, 4, 5, 6])
        sizes = rng.choice(["one larger and one smaller", "all of similar size",
                            "different sizes, some young and small"])
        crop_t = (f"{n_crop} {look} growing from the soil, {sizes}, a low leafy rosette "
                  "just like the crop plant in the photo")
        grass_t = (f"{n_grass} TALL grass weeds: long narrow green blades 15 to 35 cm tall, "
                   "arching and leaning in different directions, growing from the soil "
                   "alone or in small tufts, exactly like the tall grass in the photo, "
                   "plus a few tiny broadleaf seedlings and dry brown leaves on the soil")
        apart = (" Keep every weed well APART from every crop plant, at least 10 cm of "
                 "bare soil between them: no grass blade grows out of, through, behind "
                 "or across a crop plant and no weed touches or overlaps a crop leaf; "
                 "the weeds stand in the open soil on their own.")
        if scenario == "crop_only":
            shown = f"{crop_t}. No grass and no other weeds."
        elif scenario == "weed_only":
            shown = f"no crop rosette at all, only {grass_t}."
        else:
            shown = f"{crop_t}, and also {grass_t}.{apart}"
        f["plants"] = shown
        f["prompt"] = f"{REF_KEEP_HEAD} {shown} {REF_KEEP_TAIL}"
    elif reference:
        f["prompt"] = f"{REF_HEAD} {plants} {REF_TAIL}"
    elif edit:
        f["prompt"] = f"{EDIT_HEAD} {plants} {EDIT_TAIL}"
    else:
        shape = f" {bed_text(bed, top)}" if BED_TEXT[bed] else ""
        f["prompt"] = (f"{CAMERA} The ground is {f['soil']}.{shape} "
                       f"{plants[0].upper() + plants[1:]} {f['light']} {TAIL}")
    return f


# ---------- Muse Image edit endpoint (used by --base-image and --make-base) ----------

def _api_key() -> str:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    key = os.environ.get("META_API_KEY") or os.environ.get("MODEL_API_KEY")
    if not key:
        raise SystemExit("No API key: set META_API_KEY in .env")
    return key


def _log_call(log_path: Path, prompt: str, seconds: float, ok: bool) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({
            "ts": datetime.now(timezone.utc).isoformat(), "model": IMAGE_MODEL,
            "endpoint": "images/edits", "prompt": prompt[:200],
            "seconds": round(seconds, 1), "ok": ok,
            "usd": USD_PER_IMAGE if ok else 0.0}) + "\n")


def edit_image(base: Path, prompt: str, out: Path, log_path: Path) -> Path:
    """One Muse Image edit call: base photo + prompt -> out (JPEG)."""
    from openai import OpenAI
    client = OpenAI(base_url=BASE_URL, api_key=_api_key())
    img = Image.open(base).convert("RGB")
    img.thumbnail((1600, 1600), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    buf.seek(0)
    buf.name = "base.png"
    started = time.time()
    try:
        resp = client.images.edit(model=IMAGE_MODEL, image=buf, prompt=prompt)
    except Exception as e:  # noqa: BLE001  (blocked, rate limit, API error)
        _log_call(log_path, prompt, time.time() - started, False)
        raise SystemExit(f"edit failed: {e}")
    _log_call(log_path, prompt, time.time() - started, True)
    item = resp.data[0]
    if getattr(item, "b64_json", None):
        raw = base64.b64decode(item.b64_json)
    elif getattr(item, "url", None):
        with urllib.request.urlopen(item.url, timeout=60) as r:
            raw = r.read()
    else:
        raise SystemExit("edit returned no image")
    Image.open(io.BytesIO(raw)).convert("RGB").save(out, quality=92)
    return out


# ---------- helpers ----------

def load_manifest(path: Path) -> dict:
    if path.is_file():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {}


def contact_sheet(images: list[Path], manifest: dict, out: Path,
                  thumb: int = 240, cols: int = 8,
                  firsts: list[Path] | None = None) -> None:
    """All images as small numbered thumbnails, scenario written under each.
    `firsts` (the base soil photos) are shown first for comparison."""
    firsts = firsts or []
    items = list(firsts) + images
    if not items:
        return
    th = int(thumb * 3 / 4)
    rows = (len(items) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb, rows * (th + 18)), "white")
    draw = ImageDraw.Draw(sheet)
    for n, p in enumerate(items):
        img = Image.open(p).convert("RGB")
        img.thumbnail((thumb, th))
        x, y = (n % cols) * thumb, (n // cols) * (th + 18)
        sheet.paste(img, (x, y))
        m = manifest.get(p.stem, {})
        label = (f"BASE {n + 1}: {p.name}" if n < len(firsts)
                 else f"{p.stem} {m.get('scenario', '?')} {m.get('bed', '')}")
        draw.text((x + 4, y + th + 2), label, fill="black")
    sheet.save(out, quality=88)


def make_field(field: str, crop: str, images: list[Path], n_label: int) -> None:
    """Copy images into fields/<field>/photos so they can be labeled in the app."""
    base = Path("fields") / field
    if (base / "field.json").exists():
        raise SystemExit(f"fields/{field} already exists; pick another --field")
    (base / "photos").mkdir(parents=True)
    for p in images:
        img = Image.open(p).convert("RGB")
        img.thumbnail((1600, 1600), Image.LANCZOS)
        img.save(base / "photos" / f"{p.stem}.jpg", quality=88)
    (base / "field.json").write_text(json.dumps({
        "crop": crop, "planting_style": "single", "cultivar": "", "selected": [],
        "synthetic": True,
        "plan": {"photos": len(images), "label": n_label}}, indent=1),
        encoding="utf-8")
    print(f"field '{field}': {len(images)} photos; open it in the app, upload "
          f"step is done, pick {n_label} to label")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("n", type=int, nargs="?", help="how many images to make")
    ap.add_argument("--crop", default="tomato",
                    help="crop name (known looks: " + ", ".join(CROPS) + ")")
    ap.add_argument("--name", help="dataset folder name (default: <crop>_syn)")
    ap.add_argument("--weeds", help="which weeds, optionally weighted: "
                                    '"purslane" or "purslane,grass" or '
                                    '"purslane=3,grass=1" (known: ' + ", ".join(WEEDS)
                                    + "; other names are used as written). "
                                    "Default: all, equally likely")
    ap.add_argument("--scenario", choices=[s for s, _ in SCENARIOS],
                    help="force every image to this scene (default: a mix of "
                         "mixed / crop_only / weed_only)")
    ap.add_argument("--placement", help='where the weeds grow: "separated" (each '
                                        "weed its own plant, some close to the "
                                        "crop, some far, none overlapping) or "
                                        "your own words (default: random)")
    ap.add_argument("--max-weed-types", type=int, default=3,
                    help="most weed kinds in one image (default 3)")
    ap.add_argument("--soil", help="lock the soil for every image: "
                                   + ", ".join(SOIL_KEYS) + " (default: vary)")
    ap.add_argument("--light", help="lock the light: " + ", ".join(LIGHT_KEYS)
                                    + " (default: vary)")
    ap.add_argument("--color", help="soil color, one or a weighted mix: "
                                    + ", ".join(COLOR_KEYS) + ', e.g. "light" or '
                                    '"light=3,dark=1" or your own words like '
                                    '"pale grey clay" (default: each soil keeps its '
                                    "own color). Ignored in EDIT mode")
    ap.add_argument("--top-width", help="width of the raised top part in cm: one "
                                        'value "10" or a range "8-14" (a random '
                                        "width in the range for every image). "
                                        "Defaults: ridge 8-14, raised 18-28. "
                                        "Ignored in EDIT mode (the base photo decides)")
    ap.add_argument("--bed", default=BEDS_DEFAULT,
                    help='shape: "ridge" (narrow hilled ridge, steep sides, deep '
                         'furrows), "raised" (wide flat-topped bed), "flat", or a '
                         f'mix like "ridge=3,flat=1" (default {BEDS_DEFAULT}). In '
                         "EDIT mode each base photo's own shape is used instead")
    ap.add_argument("--base-image", type=Path, nargs="+",
                    help="EDIT mode: start every image from these soil photos "
                         "(same field, 2-3 spots; image i uses base i %% count); "
                         "--soil and --light are then ignored")
    ap.add_argument("--reference", type=Path, nargs="+",
                    help="the farmer's own photos (any angle): every image is a NEW top-view "
                         "photo of the same plants, weeds and soil look, in a different "
                         "arrangement (image i uses photo i %% count). Replaces --soil, "
                         "--light, --bed, --color")
    ap.add_argument("--ref-style", choices=["keep", "top"], default="top",
                    help="with --reference: keep = same camera angle, tray, soil and phone "
                         "quality as the farmer's photo, only the plants change (tall weeds "
                         "stay tall); top = a clean straight-down view (default)")
    ap.add_argument("--layout", choices=["row", "scattered", "mixed"], default="row",
                    help="row = crop in a line (default); scattered = crop plants spread "
                         "at random, no rows, flat ground; mixed = a random choice per image")
    ap.add_argument("--make-base" , nargs=2, metavar=("SRC", "OUT"), type=Path,
                    help="remove all plants from SRC and save clean soil as OUT, then exit")
    ap.add_argument("--size", default="1296x972",
                    help='aspect ratio "WxH" (default 1296x972, 4:3 like the Pi camera)')
    ap.add_argument("--seed", type=int, default=0,
                    help="same seed = same prompts (images still differ)")
    ap.add_argument("--workers", type=int, default=4, help="parallel calls")
    ap.add_argument("--field", help="also create this app field from the images")
    ap.add_argument("--field-n", type=int, default=10,
                    help="how many images go into the app field (default 10)")
    ap.add_argument("--label", type=int, default=3,
                    help="photos you will label in the app (default 3)")
    ap.add_argument("--dry", action="store_true", help="print prompts only, no calls")
    ap.add_argument("--yes", action="store_true", help="skip the cost check for n > 50")
    ap.add_argument("--out", type=Path, default=Path("data") / "synthetic")
    a = ap.parse_args()
    log_path = Path("out") / "api_calls.log"

    if a.make_base:
        src, out = a.make_base
        if not src.is_file():
            raise SystemExit(f"--make-base: {src} not found")
        out.parent.mkdir(parents=True, exist_ok=True)
        edit_image(src, MAKE_BASE_PROMPT, out, log_path)
        print(f"clean soil base saved: {out} (cost ${USD_PER_IMAGE:.2f}). "
              "Open it and check that no plants are left.")
        return
    if a.n is None or a.n < 1:
        ap.error("n must be at least 1")

    refs: list[Path] = list(a.reference or [])
    if refs and a.base_image:
        ap.error("use --reference or --base-image, not both")
    bases: list[Path] = refs or list(a.base_image or [])
    edit = bool(bases)
    for b in bases:
        if not b.is_file():
            raise SystemExit(f"--base-image: {b} not found")
    base_names = [str(b) for b in bases]
    if not edit:
        if a.soil:
            SOIL[:] = [(SOIL_KEYS.get(a.soil, a.soil), 1)]
        if a.light:
            LIGHT[:] = [(LIGHT_KEYS.get(a.light, a.light), 1)]

    name = a.name or f"{a.crop}_syn"
    dest = a.out / name
    manifest_path = dest / "manifest.json"
    rng = random.Random(a.seed)
    pool = parse_weeds(a.weeds)
    beds = parse_beds(a.bed)
    base_beds = [base_bed(b) for b in bases]

    lay_rng = random.Random(f"{a.seed}-layout")  # own stream: seeds keep their weeds
    layouts = [lay_rng.choice(["row", "scattered"]) if a.layout == "mixed" else a.layout
               for _ in range(a.n)]

    def bed_of(i: int) -> str:  # i = 1-based image number
        if layouts[i - 1] == "scattered" or refs:
            return "flat"  # no ridge or furrow: the reference/scatter decides the ground
        return base_beds[(i - 1) % len(bases)] if edit else wchoice(rng, beds)

    colors = parse_colors(a.color)
    for shape in HILLED:  # fail early on a bad --top-width
        parse_width(a.top_width, shape)
    if edit and (a.color or a.top_width):
        print("note: --color and --top-width are ignored in EDIT mode: the base "
              "photos decide the soil and the shape.")
    look_rng = random.Random(f"{a.seed}-look")  # color + width: own random stream
    plans = [plan_image(rng, a.crop, pool, max(1, a.max_weed_types), edit, bed_of(i),
                        look_rng, None if edit else colors,
                        None if edit else a.top_width, a.scenario, a.placement,
                        layouts[i - 1], bool(refs), a.ref_style)
             for i in range(1, a.n + 1)]

    if a.dry:
        for i, p in enumerate(plans, 1):
            top = f", top {p['top_width_cm']} cm" if p["top_width_cm"] else ""
            print(f"--- img_{i:03d} [{p['scenario']}, {p['bed']} bed{top}]\n"
                  f"{p['prompt']}\n")
        counts = {s: sum(p["scenario"] == s for p in plans) for s, _ in SCENARIOS}
        bed_counts = {b: sum(p["bed"] == b for p in plans) for b in BED_TEXT}
        widths = [p["top_width_cm"] for p in plans if p["top_width_cm"]]
        wtxt = f", top widths {min(widths)}-{max(widths)} cm" if widths else ""
        mode = (f"EDIT from {list(zip(base_names, base_beds))}" if edit
                else "generate")
        print(f"{a.n} prompts ({counts}, beds {bed_counts}{wtxt}), mode: {mode}, "
              f"nothing sent. Would cost about ${a.n * USD_PER_IMAGE:.2f}.")
        return

    dest.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(manifest_path)
    images = manifest.get("images", {})
    if images and (manifest.get("weeds") != a.weeds
                   or manifest.get("base_images") != (base_names or None)
                   or manifest.get("bed", BEDS_DEFAULT) != a.bed
                   or manifest.get("color") != a.color
                   or manifest.get("top_width") != a.top_width):
        print(f"note: {len(images)} existing images in {dest} were made with other "
              "settings. Use another --name to keep datasets separate.")
    todo = [i for i in range(1, a.n + 1)
            if not (dest / f"img_{i:03d}.jpg").is_file()]
    if len(todo) > 50 and not a.yes:
        raise SystemExit(f"{len(todo)} images will cost about "
                         f"${len(todo) * USD_PER_IMAGE:.2f}. Add --yes to go ahead.")
    print(f"{a.n - len(todo)} already done, making {len(todo)} more "
          f"(about ${len(todo) * USD_PER_IMAGE:.2f}), mode: "
          f"{'EDIT from ' + str(base_names) if edit else 'generate'}")

    lock = threading.Lock()
    failed: dict[int, str] = {}
    done = [0]

    def save_manifest() -> None:
        manifest_path.write_text(json.dumps({
            "crop": a.crop, "size": a.size, "seed": a.seed, "weeds": a.weeds,
            "soil": a.soil, "light": a.light, "bed": a.bed,
            "color": a.color, "top_width": a.top_width,
            "base_images": base_names or None,
            "note": ("Generated pictures. No plant-level labels exist. "
                     "crop_only = no sprays allowed, weed_only = no crop allowed."),
            "images": images}, indent=1), encoding="utf-8")

    def work(i: int) -> None:
        p = plans[i - 1]
        final = dest / f"img_{i:03d}.jpg"
        try:
            if edit:
                for attempt in range(3):  # keep only vertical (portrait) pictures
                    edit_image(bases[(i - 1) % len(bases)], p["prompt"], final, log_path)
                    w0, h0 = Image.open(final).size
                    if not refs or a.ref_style != "keep" or h0 > w0:
                        break
                    print(f"  img_{i:03d} came out {w0}x{h0} (not vertical), retrying")
                else:
                    final.unlink(missing_ok=True)
                    raise SystemExit("still not vertical after 3 tries")
            else:
                saved = gen_image.generate(p["prompt"], dest, 1, a.size, "jpeg",
                                           name=f"tmp{i:03d}", log_path=log_path)
                saved[0].replace(final)
                saved[0].with_suffix(".txt").unlink(missing_ok=True)
        except SystemExit as e:  # blocked / API error: skip, keep going
            with lock:
                failed[i] = str(e)
                print(f"  img_{i:03d} failed: {e}")
            return
        w, h = Image.open(final).size
        keys = ("scenario", "bed", "top_width_cm", "stage", "spacing", "weeds",
                "placement", "layout", "prompt")
        if not edit:
            keys += ("light", "soil")
        extra = {"width": w, "height": h}
        if edit:
            extra["base"] = base_names[(i - 1) % len(bases)]
        with lock:
            images[f"img_{i:03d}"] = {k: p[k] for k in keys} | extra
            done[0] += 1
            save_manifest()
            print(f"  img_{i:03d} [{p['scenario']}] saved "
                  f"({done[0]}/{len(todo)})")

    started = time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(work, todo))
    made = sorted(dest.glob("img_*.jpg"))
    print(f"\n{len(made)} images in {dest} ({time.time() - started:.0f}s, "
          f"cost ${done[0] * USD_PER_IMAGE:.2f}); manifest: {manifest_path}")
    if failed:
        print(f"{len(failed)} failed (not billed); re-run the same command to "
              f"retry: {sorted(failed)}")
    if made:
        sheet = dest / "contact_sheet.jpg"
        contact_sheet(made, images, sheet, firsts=bases)
        print(f"contact sheet: {sheet}  (check it before making more)")
    if a.field and made:
        chosen = made[:max(2, a.field_n)]
        if len(chosen) < 2:
            raise SystemExit("--field needs at least 2 images (some to label, "
                             "some to test the AI on)")
        n_label = min(a.label, len(chosen) - 1)
        if n_label < a.label:
            print(f"only {len(chosen)} images: you will label {n_label} so at "
                  "least one is left to test the AI on")
        make_field(a.field, a.crop, chosen, n_label)


if __name__ == "__main__":
    main()
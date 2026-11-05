"""Which is better on real photos: SAM with a crop prompt, or SAM + Muse? Accuracy and latency.

    python tools/compare_detectors.py Molokhia

Truth = the farmer's own labels on that crop's labeled photos (fields/<crop>/labels).
Both options see each photo cold (no cache, every call hits the API) and are timed.

  1  "Muse's prompt": Muse Spark reads the crop's saved checklist and writes ONE short
     noun phrase for SAM that outlines only the crop. SAM("<phrase>") finds the crops,
     SAM("plant") finds every plant, and every plant that is not a crop is a weed to
     spray. No judge call per frame. Two SAM calls, run at once.
  2  SAM + Muse: SAM("plant") finds every plant and Muse Spark judges each one with the
     checklist (the pipeline the app and the simulator use); spray = a sure weed.

Scored the way trainer.py scores YOLO: weeds found, crop hits (a spray centre inside a
farmer-labeled crop), unsure hits, sprays on nothing. NOTE both options were taught by
these same photos (the checklist was learned from them), so treat the numbers as an upper
bound, not a field result.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

from openai import AsyncOpenAI
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sam_client  # noqa: E402
import setup  # noqa: E402
import trainer  # noqa: E402
import vision  # noqa: E402

LOG = Path("out/api_log.jsonl")


def ask_muse_for_prompt(checklist: str, crop: str) -> tuple[str, float, float]:
    """Muse writes the one noun phrase SAM should be given to outline only the crop."""
    content = [setup._text(
        f"This is what the farmer's {crop} crop looks like and how to tell it from the weeds:\n\n"
        f"{checklist}\n\nGive ONE short noun phrase (2 to 4 words, no punctuation) to type "
        "into an image segmentation model so that it outlines ONLY the crop plants and not "
        "the weeds. Describe the crop's look, for example 'broad toothed leaf rosette'.")]
    schema = {"type": "object", "properties": {"phrase": {"type": "string"}},
              "required": ["phrase"], "additionalProperties": False}
    t0 = time.perf_counter()
    data, usd = setup.spark_json(content, schema, "sam_prompt", "sam-prompt", LOG, max_tokens=200)
    return data["phrase"].strip().strip(".\"'"), time.perf_counter() - t0, usd


def boxes_of(result, W: int, H: int) -> list[list[float]]:
    _w, _h, sx, sy = sam_client.api_space(result.raw_output, W, H)
    return [[float(v) for v in sam_client.scale_box(r, sx, sy)]
            for r in result.records if r.kind == "box"]


async def sam_boxes(path: Path, phrase: str, W: int, H: int) -> tuple[list, float]:
    async with AsyncOpenAI(base_url=sam_client.BASE_URL, api_key=sam_client.api_key()) as c:
        result, seconds, _u, _e = await sam_client.segment_image(path, phrase, c)
    return boxes_of(result, W, H), seconds


async def option1(path: Path, phrase: str, W: int, H: int) -> dict:
    """SAM(crop phrase) and SAM('plant') at once; weeds = plants that are not crops."""
    t0 = time.perf_counter()
    (crops, s1), (plants, s2) = await asyncio.gather(sam_boxes(path, phrase, W, H),
                                                     sam_boxes(path, "plant", W, H))
    wall = time.perf_counter() - t0
    weeds = [p for p in plants
             if not any(trainer._iou(p, c) >= 0.3 or trainer._inside(trainer._ctr(p), c)
                        or trainer._inside(trainer._ctr(c), p) for c in crops)]
    return {"crops": crops, "plants": plants, "sprays": weeds, "seconds": wall,
            "sam_crop": s1, "sam_plant": s2}


def option2(path: Path, model: dict, refs: list, spray_conf: float, cache: Path) -> dict:
    t0 = time.perf_counter()
    r = vision.analyze(path, cache_dir=cache, log_path=LOG, refs=refs, field_model=model,
                       spray_conf=spray_conf)
    wall = time.perf_counter() - t0
    boxes = {o["id"]: [float(v) for v in o["box"]] for o in r.objects}
    # the pipeline sprays a chosen POINT on the weed's green pixels (kept clear of crops),
    # not the centre of its box: score that point (a tiny box around it)
    sprays = [[t.x - 3.0, t.y - 3.0, t.x + 3.0, t.y + 3.0] for t in r.targets
              if t.action == "spray" and t.x is not None]
    return {"sprays": sprays, "plants": list(boxes.values()), "seconds": wall,
            "sam": r.sam_seconds, "spark": r.spark_seconds, "usd": r.spark_usd + sam_client.USD_PER_IMAGE}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("crop")
    a = ap.parse_args()
    model, srefs = setup.active_model(a.crop)
    refs = setup.ref_list(srefs, [])
    spray_conf = setup.spray_conf_for(a.crop)
    S = json.loads((Path("fields") / a.crop / "setup_log.json").read_text(encoding="utf-8"))["setup"]
    ver = next(v for v in S["versions"] if v["v"] == S["active_version"])
    phrase, mt, mu = ask_muse_for_prompt(ver["checklist"], a.crop)
    print(f'Muse wrote the SAM prompt: "{phrase}" ({mt:.1f}s, ${mu:.4f})\n', flush=True)

    cache = Path("out/cmp_cache")          # fresh: no cached answers
    cache.mkdir(parents=True, exist_ok=True)
    rows = {1: [], 2: []}
    data = {1: [], 2: []}
    for p, d_ in trainer.real_photos(a.crop):
        W, H = Image.open(p).size
        sx, sy = W / d_["image_size"][0], H / d_["image_size"][1]
        gt = [({"crop": 0, "weed": 1}.get(o["label"], 2),
               [o["box"][0] * sx, o["box"][1] * sy, o["box"][2] * sx, o["box"][3] * sy])
              for o in d_["labels"] if o.get("box")]
        r1 = asyncio.run(option1(p, phrase, W, H))
        r2 = option2(p, model, refs, spray_conf, cache)
        for k, r in ((1, r1), (2, r2)):
            data[k].append((W, H, gt, [(1, 1.0, b) for b in r["sprays"]]))
            rows[k].append(r)
        print(f'{p.name[-22:]:>22}: opt1 {len(r1["crops"])} crops + {len(r1["plants"])} plants -> '
              f'{len(r1["sprays"])} sprays in {r1["seconds"]:.1f}s | '
              f'opt2 {len(r2["plants"])} plants -> {len(r2["sprays"])} sprays in {r2["seconds"]:.1f}s '
              f'(SAM {r2["sam"]:.1f}s + Spark {r2["spark"]:.1f}s)', flush=True)

    print()
    out = {"prompt": phrase}
    for k, name in ((1, f'1  SAM("{phrase}") + SAM("plant")'), (2, "2  SAM + Muse (the pipeline)")):
        s = trainer.score(data[k], conf_w=0.5, margin=-1, crop_conf=1.1)
        secs = [r["seconds"] for r in rows[k]]
        out[str(k)] = {**s, "seconds_each": [round(x, 1) for x in secs]}
        print(f'{name}\n   weeds found {s["weeds_found"]}/{s["weeds_total"]} ({s["weeds_found_pct"]}%), '
              f'large {s["large_found"]}/{s["large_total"]}, crop hits {s["crop_hits"]}, '
              f'unsure hits {s["unsure_hits"]}, sprays on nothing {s["false_sprays"]}, '
              f'sprays {s["sprays"]}; latency per photo median {sorted(secs)[len(secs)//2]:.1f}s '
              f'(min {min(secs):.1f}, max {max(secs):.1f})')
    out["variation"] = "Muse is not perfectly repeatable: run it a few times before trusting one result."
    out["photos"] = len(rows[1])
    out["at"] = time.strftime("%Y-%m-%d %H:%M")
    out["note"] = ("Real labeled photos of this crop, farmer's labels as truth. The checklist was "
                   "learned from the same photos, so these are upper bounds.")
    Path("out/compare_detectors.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    (Path("fields") / a.crop / "detector_compare.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

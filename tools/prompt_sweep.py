"""Score SAM 3.1 crop prompts against ground-truth field images.

For each image in tests/images with a *_truth.png (green = crop, red = weed),
runs every crop prompt, then scores the union of returned masks:
crop recall (% of crop pixels covered) and weed leak (% of weed pixels
wrongly covered). Prints a table sorted by (recall - 2 * leak).

Every response is cached in out/cache/ by (image hash, prompt), so reruns
make no API calls. Up to --workers requests fly in parallel.

Run from the repo root:
    python tools/prompt_sweep.py

Outputs (all under --out, default out/):
    sweep.csv      one row per (image, prompt), sorted by score desc
    cache/         one JSON file per (image hash, prompt): raw stream events
    api_calls.log  JSONL line per call, cached or not (PROJECT.md: log every call)
"""

import argparse
import asyncio
import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from openai import AsyncOpenAI
from PIL import Image

import sam_client

PROMPTS = [
    "carrot",
    "carrot plant",
    "young carrot",
    "carrot seedling",
    "feathery leaves",
    "fern-like plant",
    "plant with thin divided leaves",
    "row of small plants",
    "parsley-like plant",
]


def image_pairs(tests_dir: Path) -> list[tuple[Path, Path]]:
    """(image, truth) for each *_truth.png that has a matching image."""
    pairs = []
    for truth in sorted(tests_dir.glob("*_truth.png")):
        img = truth.with_name(truth.stem[: -len("_truth")] + ".png")
        if img.is_file():
            pairs.append((img, truth))
        else:
            print(f"warning: {truth.name} has no matching image, skipped")
    if not pairs:
        raise RuntimeError(f"No image/truth pairs in {tests_dir}")
    return pairs


def load_truth(truth_path: Path, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """(crop, weed) bool arrays at image size. Green = crop, red = weed."""
    t = np.asarray(Image.open(truth_path).convert("RGB")).astype(int)
    r, g, b = t[..., 0], t[..., 1], t[..., 2]
    crop = (g > 150) & (r < 100) & (b < 100)
    weed = (r > 150) & (g < 100) & (b < 100)
    if (t.shape[1], t.shape[0]) != size:
        print(f"warning: {truth_path.name} is {t.shape[1]}x{t.shape[0]}, "
              f"resizing truth to image size {size[0]}x{size[1]}")
        crop = np.asarray(Image.fromarray(crop).resize(size, Image.NEAREST))
        weed = np.asarray(Image.fromarray(weed).resize(size, Image.NEAREST))
    return crop, weed


def cache_path(cache_dir: Path, image_bytes: bytes, prompt: str) -> Path:
    sha = hashlib.sha256(image_bytes).hexdigest()[:16]
    slug = re.sub(r"[^a-z0-9]+", "_", prompt.lower()).strip("_")[:40]
    return cache_dir / f"{sha}_{slug}.json"


async def run_one(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    log_lock: asyncio.Lock,
    image: Path,
    image_bytes: bytes,
    size: tuple[int, int],
    crop: np.ndarray,
    weed: np.ndarray,
    prompt: str,
    cache_dir: Path,
    log_path: Path,
) -> dict:
    cpath = cache_path(cache_dir, image_bytes, prompt)
    if cpath.is_file():
        meta = json.loads(cpath.read_text(encoding="utf-8"))
        if meta.get("prompt") != prompt or meta.get("image") != image.name:
            raise RuntimeError(f"Cache key collision: {cpath}")
        result = await sam_client.parse_events(meta["events"])
        seconds = meta.get("seconds", 0.0)  # original network latency
        usage = meta.get("usage", {})
        cached = True
    else:
        async with sem:
            result, seconds, usage, events = await sam_client.segment_image(
                image, prompt, client
            )
        cpath.write_text(json.dumps({
            "image": image.name,
            "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
            "prompt": prompt,
            "usage": usage,
            "seconds": round(seconds, 1),
            "events": events,
        }, default=str), encoding="utf-8")
        cached = False

    boxes = [r for r in result.records if r.kind == "box"]
    n_objects = len({r.object_id for r in boxes})
    _w, _h, sx, sy = sam_client.api_space(result.raw_output, *size)
    union = sam_client.mask_union(result.records, *size, sx, sy)
    recall = (union & crop).sum() / crop.sum() * 100 if crop.sum() else 0.0
    leak = (union & weed).sum() / weed.sum() * 100 if weed.sum() else 0.0
    row = {
        "image": image.name,
        "prompt": prompt,
        "objects": n_objects,
        "crop_recall_pct": round(recall, 1),
        "weed_leak_pct": round(leak, 1),
        "score": round(recall - 2 * leak, 1),
        "seconds": round(seconds, 1),
        "cached": cached,
    }
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "model": sam_client.MODEL,
        "image": str(image),
        "prompt": prompt,
        "seconds": round(seconds, 1),
        "objects": n_objects,
        "outcome": result.outcome.status,
        "usage": usage,
        "cached": cached,
        "usd": 0.0 if cached else sam_client.USD_PER_IMAGE,
    }
    async with log_lock:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    print(f"  {image.name}  {prompt!r}: {n_objects} objects, "
          f"recall {recall:.1f}%, leak {leak:.1f}%"
          f"{'  (cached)' if cached else ''}", flush=True)
    return row


async def sweep(images: list[dict], prompts: list[str], workers: int,
                cache_dir: Path, log_path: Path) -> list[dict]:
    sem = asyncio.Semaphore(workers)
    log_lock = asyncio.Lock()
    async with AsyncOpenAI(
        base_url=sam_client.BASE_URL, api_key=sam_client.api_key()
    ) as client:
        tasks = [
            run_one(client, sem, log_lock, info["path"], info["bytes"],
                    info["size"], info["crop"], info["weed"],
                    prompt, cache_dir, log_path)
            for info in images
            for prompt in prompts
        ]
        return list(await asyncio.gather(*tasks))


def main() -> None:
    ap = argparse.ArgumentParser(description="Score SAM 3.1 crop prompts vs truth.")
    ap.add_argument("--tests", type=Path, default=Path("tests/images"))
    ap.add_argument("--out", type=Path, default=Path("out"))
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    cache_dir = args.out / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    images = []
    for image, truth in image_pairs(args.tests):
        img = Image.open(image)
        crop, weed = load_truth(truth, img.size)
        print(f"{image.name} {img.size[0]}x{img.size[1]}: "
              f"crop {crop.mean() * 100:.1f}% of px, weed {weed.mean() * 100:.1f}% of px")
        images.append({"path": image, "bytes": image.read_bytes(),
                       "size": img.size, "crop": crop, "weed": weed})

    print(f"\nSweeping {len(images)} images x {len(PROMPTS)} prompts "
          f"({args.workers} parallel)...")
    rows = asyncio.run(sweep(images, PROMPTS, args.workers, cache_dir,
                             args.out / "api_calls.log"))
    rows.sort(key=lambda r: r["score"], reverse=True)

    csv_path = args.out / "sweep.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image", "prompt", "objects",
                                          "crop_recall_pct", "weed_leak_pct",
                                          "score", "seconds", "cached"])
        w.writeheader()
        w.writerows(rows)

    print(f"\n{'image':<22} {'prompt':<28} {'obj':>3} "
          f"{'recall':>6} {'leak':>6} {'score':>6} {'sec':>5}  cached")
    for r in rows:
        print(f"{r['image']:<22} {r['prompt']:<28} {r['objects']:>3} "
              f"{r['crop_recall_pct']:>6.1f} {r['weed_leak_pct']:>6.1f} "
              f"{r['score']:>6.1f} {r['seconds']:>5.1f}  {r['cached']}")
    best = rows[0]
    network = sum(1 for r in rows if not r["cached"])
    print(f"\nBest: {best['prompt']!r} on {best['image']} "
          f"(recall {best['crop_recall_pct']}%, leak {best['weed_leak_pct']}%)")
    print(f"Network calls this run: {network}, cached: {len(rows) - network}, "
          f"cost: ${network * sam_client.USD_PER_IMAGE:.4f}")
    print(f"Saved {csv_path}")


if __name__ == "__main__":
    main()

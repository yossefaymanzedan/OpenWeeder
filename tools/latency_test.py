"""Measure sam-3.1 latency vs parallelism: 1, 2, 3 concurrent requests.

Sends the same image with N requests in flight, 3 rounds per level, NO cache
(every call hits the API). Reports mean seconds per call and frames per
minute per level, and logs every call with its cost.

Run from the repo root:
    python tools/latency_test.py [image] [--prompt plant]

18 API calls total (~$0.045 at $0.0025/call).
"""

import argparse
import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from openai import AsyncOpenAI

import sam_client

LEVELS = (1, 2, 3)
ROUNDS = 3


def log_call(log_path: Path, entry: dict) -> None:
    entry = {"ts": datetime.now(timezone.utc).isoformat(), **entry}
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


async def one_round(client: AsyncOpenAI, image: Path, prompt: str, level: int,
                    log_path: Path) -> tuple[list[float], float]:
    """One round of `level` concurrent calls. Returns (call seconds, wall)."""
    started = time.perf_counter()
    results = await asyncio.gather(*[
        sam_client.segment_image(image, prompt, client) for _ in range(level)
    ])
    wall = time.perf_counter() - started
    secs = []
    for result, seconds, usage, _events in results:
        n = len({r.object_id for r in result.records if r.kind == "box"})
        secs.append(seconds)
        log_call(log_path, {"model": sam_client.MODEL, "stage": "latency",
                            "image": str(image), "prompt": prompt,
                            "parallel": level, "seconds": round(seconds, 1),
                            "objects": n, "usage": usage, "cached": False,
                            "usd": sam_client.USD_PER_IMAGE})
    return secs, wall


async def main_async(image: Path, prompt: str, log_path: Path) -> dict:
    stats: dict[int, dict] = {}
    async with AsyncOpenAI(base_url=sam_client.BASE_URL,
                           api_key=sam_client.api_key()) as client:
        for level in LEVELS:
            all_secs: list[float] = []
            walls: list[float] = []
            for rnd in range(1, ROUNDS + 1):
                secs, wall = await one_round(client, image, prompt, level, log_path)
                all_secs += secs
                walls.append(wall)
                print(f"  level {level} round {rnd}: "
                      f"calls {[f'{s:.1f}s' for s in secs]} wall {wall:.1f}s",
                      flush=True)
            stats[level] = {"secs": all_secs, "walls": walls}
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description="SAM latency vs parallelism.")
    ap.add_argument("image", type=Path, nargs="?",
                    default=Path("tests/images/carrot_field_035.png"))
    ap.add_argument("--prompt", default="plant")
    ap.add_argument("--out", type=Path, default=Path("out"))
    args = ap.parse_args()
    print(f"Image: {args.image} ({args.image.stat().st_size / 1e6:.1f} MB), "
          f"prompt {args.prompt!r}, no cache")
    stats = asyncio.run(main_async(args.image, args.prompt,
                                   args.out / "api_calls.log"))
    total_calls = sum(len(v["secs"]) for v in stats.values())
    print("\nlevel  calls  sec/call  wall/round  frames/min")
    for level in LEVELS:
        secs, walls = stats[level]["secs"], stats[level]["walls"]
        mean_sec = sum(secs) / len(secs)
        mean_wall = sum(walls) / len(walls)
        fpm = 60 * level / mean_wall
        print(f"{level:<6} {len(secs):<6} {mean_sec:<9.1f} {mean_wall:<10.1f} {fpm:<10.1f}")
    print(f"Total: {total_calls} calls, "
          f"${total_calls * sam_client.USD_PER_IMAGE:.4f}")


if __name__ == "__main__":
    main()

"""How fast could SAM 3.1 follow a video? Measures frames sent as a stream.

    python tools/sam_video_latency.py [--frames 4] [--parallel 4] [images...]

The API takes one still image per request (no video endpoint), so "video" means a
stream of frames, each its own request. For each frame size it sends N distinct
frames one after another (no cache: every call hits the API) and records
  - time to first event (the server started answering),
  - total time (all objects and masks received),
  - payload size sent.
Then it sends PARALLEL frames at once at the smallest size to see how many frames
per second the API can sustain. About 20 calls, ~$0.05; each call is logged to
out/api_log.jsonl with its cost.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import statistics
import time
from contextlib import aclosing
from datetime import datetime, timezone
from pathlib import Path

from openai import AsyncOpenAI
from PIL import Image

import sam_client
from meta_sam_parser import image_segmentation_format, parse_responses_stream

SIZES = (None, 960, 640, 480)   # long side in pixels; None = the frame as it is
PROMPT = "plant"


def jpeg_url(img: Image.Image, long_side: int | None) -> tuple[str, int, tuple[int, int]]:
    im = img.copy()
    if long_side:
        im.thumbnail((long_side, long_side), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=90)
    raw = buf.getvalue()
    return "data:image/jpeg;base64," + base64.b64encode(raw).decode("ascii"), len(raw), im.size


async def one_call(client: AsyncOpenAI, url: str) -> dict:
    """One streaming SAM request. Returns timings in seconds."""
    usage: dict = {}
    events: list = []
    t0 = time.perf_counter()
    first = None
    stream = await client.responses.create(
        model=sam_client.MODEL, stream=True,
        input=[{"type": "message", "role": "user",
                "content": [{"type": "input_text", "text": PROMPT},
                            {"type": "input_image", "image_url": url}]}])
    async with stream, aclosing(sam_client.stream_events(stream, usage, events)) as adapted:
        async def timed():
            nonlocal first
            async for ev in adapted:
                if first is None:
                    first = time.perf_counter() - t0
                yield ev
        result = await parse_responses_stream(timed(), image_segmentation_format()).final_result()
    total = time.perf_counter() - t0
    n = len({r.object_id for r in result.records if r.kind == "box"})
    return {"first": first if first is not None else total, "total": total, "objects": n}


def log(entry: dict, log_path: Path) -> None:
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(), "model": sam_client.MODEL,
                            "stage": "video-latency", "usd": sam_client.USD_PER_IMAGE, **entry}) + "\n")


def stats(xs: list[float]) -> str:
    return f"median {statistics.median(xs):5.1f}s  min {min(xs):5.1f}s  max {max(xs):5.1f}s"


async def main_async(paths: list[Path], frames: int, parallel: int, log_path: Path) -> dict:
    imgs = [Image.open(p).convert("RGB") for p in paths]
    report: dict = {"sizes": {}}
    async with AsyncOpenAI(base_url=sam_client.BASE_URL, api_key=sam_client.api_key()) as client:
        for ls in SIZES:
            firsts, totals, objs, kb, dim = [], [], [], 0, None
            for i in range(frames):
                url, nbytes, dim = jpeg_url(imgs[i % len(imgs)], ls)
                kb = nbytes // 1024
                r = await one_call(client, url)
                firsts.append(r["first"]); totals.append(r["total"]); objs.append(r["objects"])
                log({"size": list(dim), "kb": kb, "first": round(r["first"], 2),
                     "seconds": round(r["total"], 2), "objects": r["objects"]}, log_path)
            name = f"{dim[0]}x{dim[1]} ({kb} KB)"
            report["sizes"][name] = {"first": firsts, "total": totals, "objects": objs}
            print(f"{name:>20}: first event {stats(firsts)} | whole answer {stats(totals)} | "
                  f"objects {statistics.mean(objs):.0f}", flush=True)
        # sustained throughput: `parallel` frames in flight at the smallest size
        urls = [jpeg_url(imgs[i % len(imgs)], SIZES[-1])[0] for i in range(parallel)]
        t0 = time.perf_counter()
        res = await asyncio.gather(*[one_call(client, u) for u in urls])
        wall = time.perf_counter() - t0
        for r in res:
            log({"parallel": parallel, "seconds": round(r["total"], 2), "first": round(r["first"], 2)}, log_path)
        report["parallel"] = {"n": parallel, "wall": wall, "each": [r["total"] for r in res]}
        print(f"{parallel} frames at once (480 px): wall {wall:.1f}s, each {stats([r['total'] for r in res])}"
              f" -> {parallel / wall * 60:.0f} frames/min = {parallel / wall:.2f} fps", flush=True)
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("images", nargs="*", type=Path)
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--log", type=Path, default=Path("out/api_log.jsonl"))
    a = ap.parse_args()
    paths = a.images or sorted(Path("data/synthetic/molokhia_100").glob("img_0*.jpg"))[:a.frames]
    if not paths:
        raise SystemExit("no images: pass some")
    rep = asyncio.run(main_async(paths, a.frames, a.parallel, a.log))
    (Path("out") / "sam_video_latency.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

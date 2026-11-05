"""Shared SAM 3.1 client: one streaming request + parse + geometry helpers.

Request/response format per:
  https://dev.meta.ai/docs/sam/reading-segmentation (wire grammar)
  https://dev.meta.ai/docs/sam/client-libraries (request shape, parser usage)
  https://github.com/meta-models/meta-model-cookbook (07_sam/01_api_basics)

Used by tools/sam_probe.py (single interactive probe) and
tools/prompt_sweep.py (scored prompt comparison with caching).
"""

import base64
import mimetypes
import os
import re
import time
from collections.abc import AsyncIterable, AsyncIterator
from contextlib import aclosing
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from meta_sam_parser import (
    ImageSegmentationResult,
    ResponsesEventLike,
    SegmentationBoxRecord,
    SegmentationMask,
    SegmentationMaskRecord,
    decode_mask_to_raster,
    image_segmentation_format,
    parse_responses_stream,
)
from openai import AsyncOpenAI
from PIL import Image

BASE_URL = "https://api.meta.ai/v1"
MODEL = "sam-3.1"

# https://dev.meta.ai/docs/pricing-rate-limits: image segmentation is
# $2.50 per 1,000 images (streaming and non-streaming priced the same).
USD_PER_IMAGE = 2.50 / 1000

# Documented wire grammar: every box token repeats the coordinate space as
# w=<frameW>;h=<frameH> (docs: "Source frame or image dimensions").
WH_RE = re.compile(r"w=(\d+);h=(\d+)")


def api_key() -> str:
    load_dotenv()
    key = os.environ.get("META_API_KEY")
    if not key:
        raise RuntimeError("Set META_API_KEY in .env")
    return key


def image_data_url(path: Path) -> str:
    media, _ = mimetypes.guess_type(path.name)
    if media is None:
        raise ValueError(f"Could not infer a media type for {path}.")
    return f"data:{media};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


async def stream_events(
    stream: AsyncIterable[Any],
    usage_box: dict[str, Any],
    events_out: list | None = None,
) -> AsyncIterator[ResponsesEventLike]:
    """Adapt SDK events to the plain dicts the parser consumes.

    From the cookbook (07_sam/01_api_basics/python/sam_api.py): the API streams
    `response.content_part.done` but not `response.output_text.done`, which the
    parser uses to close a lane, so synthesize it from the content part.
    Captures token usage from the terminal event when present, and optionally
    collects every yielded event (for response caching).
    """
    completed_lanes: set[tuple[object, object, object]] = set()

    async def emit(data: dict[str, Any]) -> dict[str, Any]:
        if events_out is not None:
            events_out.append(data)
        return data

    async for event in stream:
        data = event.model_dump() if hasattr(event, "model_dump") else dict(event)
        lane = (data.get("item_id"), data.get("output_index"), data.get("content_index"))
        if data.get("type") == "response.output_text.done":
            if lane not in completed_lanes:
                completed_lanes.add(lane)
                yield await emit(data)
            continue
        yield await emit(data)
        if data.get("type") == "response.completed":
            usage = (data.get("response") or {}).get("usage")
            if isinstance(usage, dict):
                usage_box.update(usage)
        part = data.get("part")
        if (
            data.get("type") == "response.content_part.done"
            and lane not in completed_lanes
            and isinstance(part, dict)
            and part.get("type") == "output_text"
            and isinstance(part.get("text"), str)
        ):
            completed_lanes.add(lane)
            yield await emit({
                "type": "response.output_text.done",
                "item_id": data.get("item_id"),
                "output_index": data.get("output_index"),
                "content_index": data.get("content_index"),
                "text": part["text"],
            })


async def _request(
    api: AsyncOpenAI, path: Path, phrase: str
) -> tuple[ImageSegmentationResult, float, dict[str, Any], list[dict[str, Any]]]:
    usage: dict[str, Any] = {}
    events: list[dict[str, Any]] = []
    started = time.perf_counter()
    stream = await api.responses.create(
        model=MODEL,
        stream=True,
        input=[
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": phrase},
                    {"type": "input_image", "image_url": image_data_url(path)},
                ],
            }
        ],
    )
    # See cookbook: close the event generator before the response, or the
    # loop's asyncgen finalizer aclose()s a generator that is running.
    async with stream, aclosing(stream_events(stream, usage, events)) as adapted:
        result = await parse_responses_stream(
            adapted, image_segmentation_format()
        ).final_result()
    return result, time.perf_counter() - started, usage, events


async def segment_image(
    path: Path, prompt: str, client: AsyncOpenAI | None = None
) -> tuple[ImageSegmentationResult, float, dict[str, Any], list[dict[str, Any]]]:
    """One SAM 3.1 streaming request.

    Returns (result, seconds, usage, events). Pass a shared client for
    parallel callers; otherwise a short-lived client is created and closed.
    """
    phrase = prompt.strip()
    if not phrase:
        raise ValueError("Prompt must be a short noun phrase.")
    if client is None:
        async with AsyncOpenAI(base_url=BASE_URL, api_key=api_key()) as api:
            return await _request(api, path, phrase)
    return await _request(client, path, phrase)


async def parse_events(events: list[dict[str, Any]]) -> ImageSegmentationResult:
    """Rebuild a result from cached stream events, without any API call."""

    async def replay() -> AsyncIterator[ResponsesEventLike]:
        for event in events:
            yield event

    return await parse_responses_stream(replay(), image_segmentation_format()).final_result()


def api_space(raw: str, img_w: int, img_h: int) -> tuple[int, int, float, float]:
    """Coordinate space the API used, from the documented w/h on box tokens.

    Returns (w, h, sx, sy) where sx/sy scale API pixels to image pixels.
    An empty lane (zero matches) carries no w/h; fall back to image size.
    """
    m = WH_RE.search(raw)
    w, h = (int(m.group(1)), int(m.group(2))) if m else (img_w, img_h)
    return w, h, img_w / w, img_h / h


def scale_box(r: SegmentationBoxRecord, sx: float, sy: float) -> tuple[int, int, int, int]:
    return (int(r.left * sx), int(r.top * sy), int(r.right * sx), int(r.bottom * sy))


def scale_bounds(r: SegmentationMaskRecord, sx: float, sy: float) -> tuple[int, int, int, int]:
    """Scaled mask placement as (left, top, width, height) in image pixels."""
    bl, bt = int(r.bounds.left * sx), int(r.bounds.top * sy)
    return bl, bt, int(r.bounds.right * sx) - bl, int(r.bounds.bottom * sy) - bt


def mask_raster_array(mask: SegmentationMask) -> np.ndarray:
    """Decoded mask as a uint8 0/1 array of shape (height, width)."""
    raster = decode_mask_to_raster(mask)  # bytes-like row-major 0/1
    if isinstance(raster, (bytes, bytearray, memoryview)):
        arr = np.frombuffer(raster, dtype=np.uint8)
    else:
        arr = np.asarray(raster, dtype=np.uint8)
    return arr.reshape((mask.height, mask.width))


def mask_union(records: list, img_w: int, img_h: int, sx: float, sy: float) -> np.ndarray:
    """Union of all mask records as a bool (img_h, img_w) array in image pixels."""
    union = np.zeros((img_h, img_w), dtype=bool)
    for r in records:
        if r.kind != "mask":
            continue
        bl, bt, bw, bh = scale_bounds(r, sx, sy)
        if bw <= 0 or bh <= 0:
            continue
        small = Image.fromarray((mask_raster_array(r.mask) * 255).astype(np.uint8))
        m = np.asarray(small.resize((bw, bh), Image.NEAREST)) > 0
        x1, y1 = max(bl, 0), max(bt, 0)
        x2, y2 = min(bl + bw, img_w), min(bt + bh, img_h)
        if x2 <= x1 or y2 <= y1:
            continue
        union[y1:y2, x1:x2] |= m[y1 - bt:y2 - bt, x1 - bl:x2 - bl]
    return union

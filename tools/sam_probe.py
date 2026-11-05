"""Probe the real SAM 3.1 API: one image + one concept prompt.

Sends a single streaming Responses request, saves the raw output-text lane,
draws detected boxes, and renders decoded mask overlays.

Run from the repo root:
    python tools/sam_probe.py tests/images/carrot_field_035.png --prompt "carrot seedling"

Outputs (all under --out, default out/):
    raw_sam.txt    exact output-text lane from the API
    boxes.png      input image with one rectangle per detected box
    masks.png      input image with one semi-transparent mask overlay per object
    api_calls.log  JSONL line for this call (PROJECT.md: log every API call + cost)
"""

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import sam_client

PALETTE = [
    (230, 0, 0),
    (0, 200, 0),
    (0, 120, 255),
    (255, 200, 0),
    (200, 0, 200),
    (0, 200, 200),
    (255, 120, 0),
    (120, 0, 255),
]


def main() -> None:
    ap = argparse.ArgumentParser(description="Probe SAM 3.1 on one image + prompt.")
    ap.add_argument("image", type=Path)
    ap.add_argument("--prompt", required=True, help='short noun phrase, e.g. "carrot seedling"')
    ap.add_argument("--out", type=Path, default=Path("out"))
    args = ap.parse_args()
    if not args.image.is_file():
        raise FileNotFoundError(args.image)
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"API key: META_API_KEY ({len(sam_client.api_key())} chars)")
    result, seconds, usage, _events = asyncio.run(
        sam_client.segment_image(args.image, args.prompt)
    )

    raw = result.raw_output
    (args.out / "raw_sam.txt").write_text(raw, encoding="utf-8")

    img = Image.open(args.image).convert("RGB")
    boxes = [r for r in result.records if r.kind == "box"]
    masks = [r for r in result.records if r.kind == "mask"]
    ids = sorted({r.object_id for r in boxes}, key=int)

    api_w, api_h, sx, sy = sam_client.api_space(raw, img.width, img.height)

    print(f"outcome: {result.outcome.status}  media: {result.media}")
    print(f"objects: {len(ids)}  (boxes: {len(boxes)}, masks: {len(masks)})")
    print(f"API coordinate space: w={api_w} h={api_h}  (image: {img.width}x{img.height})")
    print(f"request time: {seconds:.1f}s  cost: ${sam_client.USD_PER_IMAGE:.4f}")
    if usage:
        print(f"usage: {usage}")
    for d in result.diagnostics:
        print(f"diagnostic [{d.severity}/{d.code}] line {d.line}: {d.message}")

    # --- boxes.png: parser box records are half-open [left,right)x[top,bottom)
    # (docs: builder writes x2 as right-1, same shape as a parsed box record),
    # so draw the inclusive outline (left, top, right-1, bottom-1).
    overlay = img.copy()
    draw = ImageDraw.Draw(overlay)
    width = max(2, img.width // 400)
    box_by_id = {}
    for r in boxes:
        x1, y1, x2, y2 = sam_client.scale_box(r, sx, sy)
        box_by_id[r.object_id] = (x1, y1, x2, y2)
        draw.rectangle((x1, y1, max(x1, x2 - 1), max(y1, y2 - 1)),
                       outline=PALETTE[int(r.object_id) % len(PALETTE)], width=width)
        draw.text((x1 + 3, y1 + 3), str(r.object_id), fill=(255, 255, 255))
    overlay.save(args.out / "boxes.png")

    # --- masks.png: each raster is placed at the record's scaled bounds.
    film = Image.new("RGBA", img.size, (0, 0, 0, 0))
    mask_px_by_id: dict[str, int] = {}
    in_box_by_id: dict[str, float] = {}
    for r in masks:
        bl, bt, bw, bh = sam_client.scale_bounds(r, sx, sy)
        if bw <= 0 or bh <= 0:
            print(f"warning: object {r.object_id} has empty mask bounds, skipped")
            continue
        arr = sam_client.mask_raster_array(r.mask)
        alpha = Image.fromarray((arr * 255).astype(np.uint8)).resize((bw, bh), Image.NEAREST)
        color = PALETTE[int(r.object_id) % len(PALETTE)]
        layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        layer.paste(Image.new("RGBA", (bw, bh), color + (110,)), (bl, bt), mask=alpha)
        film = Image.alpha_composite(film, layer)
        fg = np.asarray(alpha) > 0
        mask_px_by_id[r.object_id] = int(fg.sum())
        b = box_by_id.get(r.object_id)
        if b is not None:
            x1, y1, x2, y2 = (max(0, v) for v in (b[0], b[1], min(b[2], img.width), min(b[3], img.height)))
            # fg is bounds-sized at offset (bl, bt); intersect with the box.
            ix1, iy1 = max(x1 - bl, 0), max(y1 - bt, 0)
            ix2, iy2 = min(x2 - bl, bw), min(y2 - bt, bh)
            if ix2 > ix1 and iy2 > iy1 and fg.sum() > 0:
                in_box_by_id[r.object_id] = fg[iy1:iy2, ix1:ix2].sum() / fg.sum()
    Image.alpha_composite(img.convert("RGBA"), film).convert("RGB").save(args.out / "masks.png")

    print(f"\n{'id':>3}  {'box (x1,y1,x2,y2) px':<24} {'mask px':>8}  {'in box':>7}")
    for oid in ids:
        b = box_by_id.get(oid, ("?", "?", "?", "?"))
        px = mask_px_by_id.get(oid, 0)
        frac = in_box_by_id.get(oid)
        print(f"{oid:>3}  ({b[0]},{b[1]},{b[2]},{b[3]})"
              f"  {px:>8}  {(f'{frac * 100:.1f}%') if frac is not None else 'n/a':>7}")

    with open(args.out / "api_calls.log", "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "ts": datetime.now(timezone.utc).isoformat(),
            "model": sam_client.MODEL,
            "image": str(args.image),
            "prompt": args.prompt,
            "seconds": round(seconds, 1),
            "objects": len(ids),
            "outcome": result.outcome.status,
            "usage": usage,
            "cached": False,
            "usd": sam_client.USD_PER_IMAGE,
        }) + "\n")
    print(f"\nSaved {args.out / 'raw_sam.txt'}, boxes.png, masks.png, api_calls.log")


if __name__ == "__main__":
    main()

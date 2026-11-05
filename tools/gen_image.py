"""Generate images with Muse Image (muse-image-1.0) from a prompt you give.

    python tools/gen_image.py "a watercolor fox in snow"
    python tools/gen_image.py --file prompts/tomato_row.txt -n 3
    python tools/gen_image.py --file prompts/tomato_row.txt --size 1536x1024 --format png

Each image is saved to out/images/<name>_<time>_<i>.<ext> next to a .txt copy
of the prompt, and the call is logged with its cost in out/api_calls.log.
$0.01 per image (failed or safety-blocked images are not billed).
"""
from __future__ import annotations

import argparse
import base64
import re
import sys
import time
from pathlib import Path

from openai import APIStatusError, OpenAI

import sam_client
import vision

MODEL = "muse-image-1.0"
USD_PER_IMAGE = 0.01
RETRIES = 3  # 503 = retryable backend failure: back off and try again


def slug(text: str, n: int = 40) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:n] or "image"


def generate(prompt: str, out_dir: Path, n: int = 1, size: str | None = None,
             fmt: str = "png", name: str | None = None,
             log_path: Path | None = None) -> list[Path]:
    """Call Muse Image and save each returned image. Returns the saved paths."""
    client = OpenAI(base_url=sam_client.BASE_URL, api_key=sam_client.api_key())
    args = {"model": MODEL, "prompt": prompt, "n": n,
            "response_format": "b64_json", "output_format": fmt}
    if size:
        args["size"] = size
    started = time.perf_counter()
    for attempt in range(1, RETRIES + 1):
        try:
            resp = client.images.generate(**args)
            break
        except APIStatusError as e:
            if e.status_code == 503 and attempt < RETRIES:
                time.sleep(2 ** attempt)
                continue
            body = getattr(e, "body", None) or {}
            code = body.get("code") if isinstance(body, dict) else None
            if code == "content_policy_violation":
                raise SystemExit("Blocked by Meta's content-safety filter "
                                 "(not billed). Reword the prompt.")
            raise SystemExit(f"API error {e.status_code}: {e.message}")
    seconds = time.perf_counter() - started

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{name or slug(prompt)}_{time.strftime('%Y%m%d_%H%M%S')}"
    saved = []
    for i, item in enumerate(resp.data, start=1):
        ext = resp.output_format or fmt
        path = out_dir / f"{stem}_{i}.{ext}"
        path.write_bytes(base64.b64decode(item.b64_json))
        path.with_suffix(".txt").write_text(prompt, encoding="utf-8")
        saved.append(path)
    vision.log_call(log_path, {"model": MODEL, "stage": "image-generate",
                               "prompt": prompt[:200], "n": n, "size": size,
                               "seconds": round(seconds, 1), "images": len(saved),
                               "cached": False,
                               "usd": round(USD_PER_IMAGE * len(saved), 4)})
    return saved


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("prompt", nargs="?", help="the prompt text")
    ap.add_argument("--file", "-f", type=Path, help="read the prompt from a text file")
    ap.add_argument("-n", type=int, default=1, help="images to make, 1-10 (default 1)")
    ap.add_argument("--size", help='aspect ratio as "WxH", e.g. 1536x1024 '
                                   "(sets the ratio, not exact pixels)")
    ap.add_argument("--format", choices=["png", "jpeg", "webp"], default="png")
    ap.add_argument("--name", help="file name prefix (default: from the prompt)")
    ap.add_argument("--out", type=Path, default=Path("out"))
    a = ap.parse_args()
    if bool(a.prompt) == bool(a.file):
        ap.error("give the prompt as text OR with --file")
    prompt = (a.file.read_text(encoding="utf-8") if a.file else a.prompt).strip()
    if not 1 <= a.n <= 10:
        ap.error("-n must be 1 to 10")
    name = a.name or (a.file.stem if a.file else None)
    paths = generate(prompt, a.out / "images", a.n, a.size, a.format, name,
                     a.out / "api_calls.log")
    for p in paths:
        print(f"saved {p}  ({p.stat().st_size // 1024} KB)")
    print(f"cost ${USD_PER_IMAGE * len(paths):.2f}")


if __name__ == "__main__":
    sys.exit(main())

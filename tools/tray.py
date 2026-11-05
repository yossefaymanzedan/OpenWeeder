"""Find the tray (the strip of field the robot drives along) in a camera frame,
and crop it out straight and centred.

The camera looks down at a long tray of soil that runs top to bottom in the
picture. Around it are things that are not field: the table, cardboard, a ruler,
other plants. Cropping to the tray means the detectors (SAM + Muse, YOLO) only
see the field, anything outside it can never be sprayed, and the tray's middle is
the middle of the picture, which makes "how far left or right" simple for the arm.

    find(frame)              -> Tray | None     one frame
    TrayTracker().update(f)  -> Tray | None     a video: faster, steadier
    Tray.rectify(frame)      -> the tray only, upright, centred
    Tray.to_frame(points)    -> crop pixels back to frame pixels (for geometry.py)
    Tray.contains(points)    -> which frame pixels are inside the tray
    Tray.offset, tilt_deg    -> where the tray is: for steering and the arm

How: the soil gives a rough position of each wall; then both walls are snapped
together to the strongest pair of long straight lines near them. The tray is a
rectangle seen from above, so the two walls are parallel (one shared lean) and
the lean is small (at most MAX_TILT_DEG). Plants, shadows and things beside the
tray do not make two long parallel straight lines. Works on a 320 px copy.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

WORK_WIDTH = 320       # pixels: everything is found on a copy this wide
SEARCH_FRAC = 0.08     # first frame: look this far (of the width) around each rough wall
TRACK_FRAC = 0.03      # later frames: look this far around last frame's wall
MIN_WIDTH_FRAC = 0.2   # a tray narrower than this (of the picture) is not believed
MIN_EDGE = 60.0        # mean edge strength along a wall below this: not a wall
MIN_SUPPORT = 0.6      # a wall must be a clear edge along at least this share of its length
EDGE_ON = 40.0         # edge strength that counts as "a clear edge" at one point of a wall
MAX_TILT_DEG = 8.0     # the tray leans at most this much in the picture
WIDTH_CHANGE = 0.2     # while tracking, a tray this much wider or narrower is not believed
SMOOTH = 0.5           # how much a new frame moves the tracked walls (1 = no smoothing)
HOLD_FRAMES = 0        # never reuse an old tray: a frame without one (past the end) has none


@dataclass
class Tray:
    """The tray in a frame of size (width, height): its left wall from top_l to
    bot_l and right wall from top_r to bot_r (x pixels at the top and bottom row)."""
    width: int
    height: int
    top_l: float
    bot_l: float
    top_r: float
    bot_r: float
    strength: float = 0.0

    @property
    def corners(self) -> np.ndarray:
        """Frame pixels: top-left, top-right, bottom-right, bottom-left."""
        h = self.height - 1
        return np.float32([[self.top_l, 0], [self.top_r, 0], [self.bot_r, h], [self.bot_l, h]])

    @property
    def tray_width(self) -> float:
        return ((self.top_r - self.top_l) + (self.bot_r - self.bot_l)) / 2

    @property
    def offset(self) -> float:
        """Tray centre minus picture centre at the middle row, as a fraction of the
        picture width: negative = the tray is left of centre."""
        centre = (self.top_l + self.top_r + self.bot_l + self.bot_r) / 4
        return (centre - self.width / 2) / self.width

    @property
    def tilt_deg(self) -> float:
        """How much the tray leans in the picture (positive = its bottom is further
        right than its top), from the mean of both walls."""
        dx = ((self.bot_l - self.top_l) + (self.bot_r - self.top_r)) / 2
        return float(np.degrees(np.arctan2(dx, self.height - 1)))

    def out_size(self) -> tuple[int, int]:
        return max(1, int(round(self.tray_width))), self.height

    def matrix(self) -> np.ndarray:
        """Frame -> crop (perspective transform)."""
        w, h = self.out_size()
        dst = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
        return cv2.getPerspectiveTransform(self.corners, dst)

    def rectify(self, frame: np.ndarray) -> np.ndarray:
        """The tray only: walls straight up, the tray's middle the crop's middle."""
        return cv2.warpPerspective(frame, self.matrix(), self.out_size(), flags=cv2.INTER_LINEAR)

    def to_frame(self, points) -> np.ndarray:
        """Crop pixels (x, y) -> frame pixels."""
        pts = np.asarray(points, np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, np.linalg.inv(self.matrix())).reshape(-1, 2)

    def from_frame(self, points) -> np.ndarray:
        """Frame pixels (x, y) -> crop pixels."""
        pts = np.asarray(points, np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.matrix()).reshape(-1, 2)

    def contains(self, points, margin_px: float = 0.0) -> np.ndarray:
        """Which frame pixels are inside the tray (shrunk by margin_px)."""
        crop = self.from_frame(points)
        w, h = self.out_size()
        return ((crop[:, 0] >= margin_px) & (crop[:, 0] <= w - 1 - margin_px)
                & (crop[:, 1] >= 0) & (crop[:, 1] <= h - 1))

    def scaled(self, width: int, height: int) -> "Tray":
        sx, sy = width / self.width, (height - 1) / max(1, self.height - 1)
        del sy  # rows map top to top and bottom to bottom
        return Tray(width, height, self.top_l * sx, self.bot_l * sx, self.top_r * sx, self.bot_r * sx, self.strength)

    def draw(self, frame: np.ndarray, color=(0, 0, 255)) -> np.ndarray:
        """A copy of the frame (BGR or RGB) with the walls and the picture centre drawn."""
        out = frame.copy()
        h, w = out.shape[:2]
        t = self.scaled(w, h)
        thick = max(2, w // 200)
        for a, b in ((t.top_l, t.bot_l), (t.top_r, t.bot_r)):
            cv2.line(out, (int(round(a)), 0), (int(round(b)), h - 1), color, thick)
        cv2.line(out, (w // 2, 0), (w // 2, h - 1), (255, 255, 0), max(1, thick // 2))
        return out


def _soil_mask(small_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    soil = ((h >= 3) & (h <= 28) & (s >= 45) & (v >= 25) & (v <= 150)).astype(np.uint8)
    soil = cv2.morphologyEx(soil, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    return cv2.morphologyEx(soil, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))


def _rough_walls(soil: np.ndarray):
    """Left and right edge of the biggest patch of soil, one line each, ignoring
    rows where something hangs over a wall. None if there is no such patch."""
    hgt, wid = soil.shape
    n, labels, stats, _ = cv2.connectedComponentsWithStats(soil)
    if n < 2:
        return None
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    region = labels == big
    ys = np.flatnonzero(region.sum(1) > MIN_WIDTH_FRAC * 0.75 * wid)
    if len(ys) < hgt * 0.3:
        return None
    lefts = region[ys].argmax(1).astype(float)
    rights = (wid - 1 - region[ys, ::-1].argmax(1)).astype(float)

    def wall(x):
        keep = np.ones(len(ys), bool)
        for _ in range(4):
            a, b = np.polyfit(ys[keep], x[keep], 1)
            res = np.abs(x - (a * ys + b))
            keep = res <= max(2.0, 2.0 * np.median(res[keep]))
        return b, a * (hgt - 1) + b

    return wall(lefts), wall(rights)


def _edge_strength(gray: np.ndarray) -> np.ndarray:
    gx = np.abs(cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)).astype(np.float32)
    return cv2.dilate(gx, np.ones((1, 3), np.uint8))


def _line_scores(edges: np.ndarray, top: float, band: int, leans: np.ndarray):
    """For straight lines starting within band pixels of `top` and leaning by
    each of `leans` (pixels from top to bottom): mean edge strength, and the share
    of the line where the edge is clear. Arrays (tops, leans)."""
    hgt, wid = edges.shape
    rows = np.arange(0, hgt, 2)
    frac = rows / max(1, hgt - 1)
    tops = np.clip(np.round(top) + np.arange(-band, band + 1), 0, wid - 1)
    xs = np.round(tops[:, None, None] + leans[None, :, None] * frac[None, None, :])
    inside = (xs >= 0) & (xs <= wid - 1)
    vals = edges[rows[None, None, :], np.clip(xs, 0, wid - 1).astype(np.intp)] * inside
    return tops, vals.mean(2), (vals >= EDGE_ON).mean(2)


def _snap_pair(edges: np.ndarray, guess, band: int, lean_range: tuple[int, int]):
    """Both walls at once: the two parallel lines (one shared lean) near the
    guesses whose edges are strongest together. (top_l, top_r, lean, strength_l,
    strength_r, support_l, support_r)."""
    leans = np.arange(lean_range[0], lean_range[1] + 1)
    tops_l, score_l, sup_l = _line_scores(edges, guess[0][0], band, leans)
    tops_r, score_r, sup_r = _line_scores(edges, guess[1][0], band, leans)
    best_l, best_r = score_l.argmax(0), score_r.argmax(0)
    cols = np.arange(len(leans))
    k = int((score_l[best_l, cols] + score_r[best_r, cols]).argmax())
    i, j = best_l[k], best_r[k]
    return (float(tops_l[i]), float(tops_r[j]), float(leans[k]), float(score_l[i, k]), float(score_r[j, k]),
            float(sup_l[i, k]), float(sup_r[j, k]))


def find(frame: np.ndarray, previous: Tray | None = None, bgr: bool = True) -> Tray | None:
    """The tray in one frame (BGR by default; bgr=False for RGB), or None.
    With previous (a tray from the frame before), only looks close to it."""
    hgt, wid = frame.shape[:2]
    scale = WORK_WIDTH / wid
    small = cv2.resize(frame, (WORK_WIDTH, max(2, int(round(hgt * scale)))), interpolation=cv2.INTER_AREA)
    if not bgr:
        small = cv2.cvtColor(small, cv2.COLOR_RGB2BGR)
    sh, sw = small.shape[:2]
    max_lean = int(np.ceil(np.tan(np.radians(MAX_TILT_DEG)) * (sh - 1)))
    if previous is not None:
        p = previous.scaled(sw, sh)
        guess, band = ((p.top_l, p.bot_l), (p.top_r, p.bot_r)), max(2, int(TRACK_FRAC * sw))
        lean = int(round(((p.bot_l - p.top_l) + (p.bot_r - p.top_r)) / 2))
        lean_range = (max(-max_lean, lean - band), min(max_lean, lean + band))
    else:
        guess, band = _rough_walls(_soil_mask(small)), max(3, int(SEARCH_FRAC * sw))
        if guess is None:
            return None
        lean_range = (-max_lean, max_lean)
    edges = _edge_strength(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
    top_l, top_r, lean, strength_l, strength_r, support_l, support_r = _snap_pair(edges, guess, band, lean_range)
    tray = Tray(sw, sh, top_l, top_l + lean, top_r, top_r + lean, min(strength_l, strength_r))
    if (tray.strength < MIN_EDGE or min(support_l, support_r) < MIN_SUPPORT
            or top_r - top_l < MIN_WIDTH_FRAC * sw):
        return None
    if previous is not None:
        old_width = previous.scaled(sw, sh).tray_width
        if abs(tray.tray_width - old_width) > WIDTH_CHANGE * old_width:
            return None
    return tray.scaled(wid, hgt)


class TrayTracker:
    """The tray through a video: the first frame searches widely, later frames
    only near the last walls (faster), and the walls are smoothed between
    frames. A frame where the tray is not found has no tray (hold_frames > 0
    would reuse the last one, which drew trays on frames past the field's end)."""

    def __init__(self, smooth: float = SMOOTH, hold_frames: int = HOLD_FRAMES):
        self.smooth = smooth
        self.hold_frames = hold_frames
        self.tray: Tray | None = None
        self.missed = 0

    def update(self, frame: np.ndarray, bgr: bool = True) -> Tray | None:
        hgt, wid = frame.shape[:2]
        last = self.tray.scaled(wid, hgt) if self.tray is not None else None
        found = find(frame, last, bgr) if last is not None else None
        if found is None:
            found = find(frame, None, bgr)  # lost it: search the whole picture again
        if found is None:
            self.missed += 1
            if self.missed > self.hold_frames:
                self.tray = None
            return self.tray.scaled(wid, hgt) if self.tray is not None else None
        self.missed = 0
        if last is not None:
            a = self.smooth
            found = Tray(wid, hgt, *(a * new + (1 - a) * old for new, old in (
                (found.top_l, last.top_l), (found.bot_l, last.bot_l),
                (found.top_r, last.top_r), (found.bot_r, last.bot_r))), found.strength)
        self.tray = found
        return found


def view(frame_bgr: np.ndarray, tray: Tray | None, height: int = 360) -> np.ndarray:
    """One picture for people: the frame with the walls drawn, and the cropped tray
    next to it (or 'no tray found')."""
    h, w = frame_bgr.shape[:2]
    left = cv2.resize(tray.draw(frame_bgr) if tray else frame_bgr, (int(w * height / h), height))
    if tray is None:
        cv2.putText(left, "no tray found", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return left
    crop = tray.rectify(frame_bgr)
    crop = cv2.resize(crop, (max(1, int(crop.shape[1] * height / crop.shape[0])), height))
    gap = np.full((height, 12, 3), 255, np.uint8)
    return np.hstack([left, gap, crop])


if __name__ == "__main__":
    import argparse
    import time

    ap = argparse.ArgumentParser(description="Find the tray in photos and write the crop next to each.")
    ap.add_argument("photos", nargs="+")
    ap.add_argument("--out", default="out/tray")
    args = ap.parse_args()
    from pathlib import Path
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in args.photos:
        frame = cv2.imread(name)
        t0 = time.perf_counter()
        tray = find(frame)
        ms = (time.perf_counter() - t0) * 1000
        if tray is None:
            print(f"{name}: no tray found ({ms:.0f} ms)")
        else:
            print(f"{name}: tray {tray.tray_width / tray.width:.0%} of the width, centre {tray.offset:+.1%}, "
                  f"tilt {tray.tilt_deg:+.1f} deg, edge strength {tray.strength:.0f} ({ms:.0f} ms)")
        cv2.imwrite(str(out / (Path(name).stem + "_tray.jpg")), view(frame, tray))

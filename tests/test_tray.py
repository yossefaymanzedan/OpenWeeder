"""tools/tray.py on drawn trays whose walls are known.

    python -m pytest tests/test_tray.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import tray  # noqa: E402

W, H = 640, 860


def drawn_tray(top_l=150, bot_l=170, top_r=470, bot_r=480, seed=0) -> np.ndarray:
    """White table, a wooden plank and cardboard outside, a soil tray with dark
    glass walls and a few plants (some hanging over a wall), as BGR."""
    rng = np.random.default_rng(seed)
    img = np.full((H, W, 3), 235, np.uint8)
    cv2.rectangle(img, (0, 500), (60, H), (90, 150, 200), -1)          # wooden plank, left
    cv2.rectangle(img, (540, 40), (630, 300), (110, 160, 200), -1)     # cardboard, right
    soil = np.array([[top_l, 0], [top_r, 0], [bot_r, H - 1], [bot_l, H - 1]], np.int32)
    cv2.fillPoly(img, [soil], (45, 75, 110))
    noise = rng.integers(-18, 18, img.shape, dtype=np.int16)
    inside = np.zeros((H, W), np.uint8)
    cv2.fillPoly(inside, [soil], 1)
    img = np.where(inside[..., None] == 1, np.clip(img.astype(np.int16) + noise, 0, 255), img).astype(np.uint8)
    for a, b in ((top_l, bot_l), (top_r, bot_r)):                      # glass walls
        cv2.line(img, (a, 0), (b, H - 1), (40, 60, 40), 5)
    for cx, cy, r in ((260, 200, 45), (380, 520, 60), (top_r, 400, 50), (230, 700, 35)):
        cv2.circle(img, (cx, cy), r, (60, 170, 70), -1)                # plants
    cv2.line(img, (90, 300), (300, 330), (70, 190, 90), 9)             # grass blade over the left wall
    return img


def test_finds_both_walls_within_a_few_pixels():
    found = tray.find(drawn_tray())
    assert found is not None
    for got, want in ((found.top_l, 150), (found.bot_l, 170), (found.top_r, 470), (found.bot_r, 480)):
        assert abs(got - want) <= 6
    assert found.tray_width == pytest.approx(315, abs=8)
    assert found.offset == pytest.approx(((150 + 470 + 170 + 480) / 4 - W / 2) / W, abs=0.01)
    assert found.tilt_deg > 0  # both walls lean right going down


def test_rectify_and_map_back():
    frame = drawn_tray()
    found = tray.find(frame)
    crop = found.rectify(frame)
    assert crop.shape[0] == H and abs(crop.shape[1] - found.tray_width) <= 1
    points = np.float32([[10, 10], [crop.shape[1] / 2, H / 2], [crop.shape[1] - 5, H - 5]])
    back = found.from_frame(found.to_frame(points))
    assert np.allclose(back, points, atol=0.5)
    middle = found.to_frame([[crop.shape[1] / 2, H / 2]])[0]
    assert middle[0] == pytest.approx((found.top_l + found.top_r + found.bot_l + found.bot_r) / 4, abs=2)


def test_contains_only_the_tray():
    found = tray.find(drawn_tray())
    inside = found.contains([[320, 400], [30, 700], [600, 100], [found.top_l + 3, 2]])
    assert inside.tolist() == [True, False, False, True]
    assert not found.contains([[found.top_l + 3, 2]], margin_px=10)[0]


def test_no_tray_gives_none():
    assert tray.find(np.full((H, W, 3), 235, np.uint8)) is None


def test_tracker_is_steady_and_a_frame_without_a_tray_has_none():
    tracker = tray.TrayTracker()
    frames = [drawn_tray(seed=i) for i in range(6)]
    positions = [tracker.update(f).top_l for f in frames]
    assert max(positions) - min(positions) <= 3
    blank = np.full((H, W, 3), 235, np.uint8)
    assert tracker.update(blank) is None          # past the end of the field: no tray
    assert tracker.update(frames[0]) is not None  # and it finds it again


def test_walls_are_always_parallel_and_lean_little():
    # the drawn walls lean differently (20 and 10 px); the tray is a rectangle,
    # so the answer is one shared lean, and never more than MAX_TILT_DEG
    found = tray.find(drawn_tray())
    assert (found.bot_l - found.top_l) == pytest.approx(found.bot_r - found.top_r)
    assert abs(found.tilt_deg) <= tray.MAX_TILT_DEG
    steep = drawn_tray(150, 330, 470, 650)  # leans about 12 degrees: too much for a tray
    found = tray.find(steep)
    assert found is None or abs(found.tilt_deg) <= tray.MAX_TILT_DEG


def test_follows_a_moving_tray():
    tracker = tray.TrayTracker(smooth=1.0)
    for shift in range(0, 60, 10):
        found = tracker.update(drawn_tray(150 + shift, 170 + shift, 470 + shift, 480 + shift))
        assert abs(found.top_l - (150 + shift)) <= 6


def test_view_draws_frame_and_crop_side_by_side():
    frame = drawn_tray()
    picture = tray.view(frame, tray.find(frame), height=300)
    assert picture.shape[0] == 300 and picture.shape[1] > 300 * W / H + 50
    assert tray.view(frame, None, height=300).shape[0] == 300

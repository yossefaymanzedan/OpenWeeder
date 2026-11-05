"""The planner on the real robot's timing (PRD R9): detection takes seconds, so
it runs in threads while the car keeps driving. Uses PiHardware as a dry run on
a real clock and a vision backend whose detect() really sleeps.

    python -m pytest tests/test_planner_async.py -q
"""
from __future__ import annotations

import shutil
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import geometry as g  # noqa: E402
import hardware  # noqa: E402
import planner  # noqa: E402

DETECT_S = 1.5


class SlowVision:
    """detect() sleeps like SAM + Muse; one weed in the first frame taken at or after
    weed_tick (the arm reaches L ahead of its pivot, so a weed in the very first frame
    would need the car to stop before its start)."""

    def __init__(self, weed_tick: int = 0):
        self.weed_tick = weed_tick
        self.given = False
        self.calls = []
        self.lock = threading.Lock()

    def detect(self, image, tick):
        start = time.monotonic()
        time.sleep(DETECT_S)
        with self.lock:
            self.calls.append((tick, start, time.monotonic()))
            give = tick >= self.weed_tick and not self.given
            self.given = self.given or give
        if not give:
            return []
        h, w = image.shape[:2]
        u, v = w / 2, h * 0.25
        return [{"label": "weed", "action": "spray", "conf": 0.97, "box": [u - 5, v - 5, u + 5, v + 5],
                 "spray_px": (u, v), "polygon_px": [(u - 5, v - 5), (u + 5, v - 5), (u + 5, v + 5), (u - 5, v + 5)]}]


class TickLog(hardware.PiHardware):
    """Remembers when ticks() was read, to prove the loop never froze."""

    def __init__(self, *args, **kwargs):
        self.reads = []
        super().__init__(*args, **kwargs)

    def ticks(self):
        value = super().ticks()
        self.reads.append(time.monotonic())
        return value


@pytest.fixture
def robot(tmp_path):
    calib = tmp_path / "calib"
    shutil.copytree(ROOT / "calib", calib)
    hw = TickLog(calib / "hardware.json", dry_run=True, calib_dir=calib)
    hw.arm()
    yield hw, g.Calib(calib_dir=calib, quiet=True)
    hw.close()


def test_slow_detection_does_not_freeze_the_drive(robot, tmp_path):
    hw, calib = robot
    vision = SlowVision()
    cfg = planner.Config(speed=4.0, async_vision=True, out_dir=tmp_path)
    p = planner.Planner(hw, vision, calib, cfg=cfg)
    try:
        p.map_pass(end_cm=70)  # long enough for several pictures whatever D is
        p.return_to_start()
    finally:
        p.close()
    every = p.capture_every_ticks()
    ticks = [f["tick"] for f in p.frames]
    assert len(ticks) >= 2
    # pictures at the planned spacing: a blocking detect would make them late
    assert all(abs((b - a) - every) <= 1 for a, b in zip(ticks, ticks[1:]))
    # the encoder was read many times while each detection was running
    for _, start, end in vision.calls:
        assert sum(start < t < end for t in hw.reads) > 20
    # every frame's detections arrived, and the weed made it into the map
    assert all(f["dets"] is not None and "future" not in f for f in p.frames)
    assert any(e["type"] == "detected" for e in p.log)


def test_two_passes_stop_where_planned_while_vision_is_slow(robot, tmp_path):
    hw, calib = robot
    cfg = planner.Config(speed=5.0, async_vision=True, out_dir=tmp_path)
    p = planner.Planner(hw, SlowVision(weed_tick=15), calib, cfg=cfg)  # the 2nd picture
    try:
        d = p.run(end_cm=70)
    finally:
        p.close()
    assert len(d["weeds"]) == 1
    sprays = [e for e in p.log if e["type"] == "spray"]
    weed = p.weeds[0]
    if weed.get("status", "").startswith("skipped"):
        pytest.fail(f"the weed was skipped: {weed['status']}")
    stop_tick, _ = p.plan_stop(weed["x"], weed["y"])
    assert len(sprays) == 1
    assert abs(sprays[0]["tick"] - stop_tick) <= 1
    pumps = [e for e in hw.events if e["type"] == "pump"]
    assert len(pumps) == 1 and abs(pumps[0]["tick"] - stop_tick) <= 1


def test_a_failed_detection_sprays_nothing_from_that_frame(robot, tmp_path):
    hw, calib = robot

    class Broken:
        def detect(self, image, tick):
            time.sleep(0.2)
            raise RuntimeError("API down")

    p = planner.Planner(hw, Broken(), calib, cfg=planner.Config(speed=4.0, async_vision=True, out_dir=tmp_path))
    try:
        d = p.run(end_cm=30)
    finally:
        p.close()
    assert d["weeds"] == []
    assert any(e["type"] == "vision_error" and "API down" in e["error"] for e in p.log)
    assert not [e for e in hw.events if e["type"] == "pump"]

"""runner.run_on_field on a dry-run robot with a fake detector (no API money):
the tray crop, mapping detections back, the two passes, Stop, the video, the logs.

    python -m pytest tests/test_field_run.py -q
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import hardware  # noqa: E402
import runner  # noqa: E402

# A phone frame (1600x1200, landscape) of a soil tray with glass walls.
WALLS = (500, 1100, 520, 1120)  # top_l, top_r, bot_l, bot_r


def tray_jpeg() -> bytes:
    image = np.full((1200, 1600, 3), 235, np.uint8)
    top_l, top_r, bot_l, bot_r = WALLS
    soil = np.array([[top_l, 0], [top_r, 0], [bot_r, 1199], [bot_l, 1199]], np.int32)
    cv2.fillPoly(image, [soil], (45, 75, 110))
    noise = np.random.default_rng(1).integers(-18, 18, image.shape, dtype=np.int16)
    inside = np.zeros(image.shape[:2], np.uint8)
    cv2.fillPoly(inside, [soil], 1)
    image = np.where(inside[..., None] == 1, np.clip(image.astype(np.int16) + noise, 0, 255), image).astype(np.uint8)
    cv2.line(image, (top_l, 0), (bot_l, 1199), (40, 60, 40), 9)
    cv2.line(image, (top_r, 0), (bot_r, 1199), (40, 60, 40), 9)
    return cv2.imencode(".jpg", image)[1].tobytes()


class TrayCamera:
    def __init__(self):
        self.jpeg = tray_jpeg()

    def connected(self):
        return True

    def capture(self, timeout=2.0):
        return self.jpeg


class FakeDetector(runner.FieldVision):
    """FieldVision with the detector swapped for canned answers in crop pixels:
    a weed in the middle of the first picture at or after weed_tick, a crop
    beside it, and a 'weed' just outside the tray's left wall."""

    def __init__(self, work_dir, weed_tick=30):
        self.weed_tick = weed_tick
        self.given = False
        super().__init__("Crop", "sam_muse", None, work_dir=work_dir)

    def _load_detector(self, mode, model_path):
        pass

    def _sam_muse(self, view, n):
        h, w = view.shape[:2]
        out = [{"label": "weed", "action": "spray", "conf": 0.97, "box": [-60, 300, -20, 340],
                "spray_px": (-40, 320), "polygon_px": [(-60, 300), (-20, 300), (-20, 340), (-60, 340)]},
               {"label": "crop", "action": "protect", "conf": 0.9, "box": [20, h - 200, 120, h - 100],
                "spray_px": None, "polygon_px": [(20, h - 200), (120, h - 200), (120, h - 100), (20, h - 100)]}]
        with self.lock:
            give = not self.given and self._tick >= self.weed_tick
            self.given = self.given or give
        if give:
            u, v = w / 2, h * 0.25
            out.append({"label": "weed", "action": "spray", "conf": 0.97, "box": [u - 8, v - 8, u + 8, v + 8],
                        "spray_px": (u, v), "polygon_px": [(u - 8, v - 8), (u + 8, v - 8), (u + 8, v + 8), (u - 8, v + 8)]})
        return out

    def detect(self, image, tick):
        self._tick = tick
        return super().detect(image, tick)


@pytest.fixture
def setup_run(tmp_path):
    calib = tmp_path / "calib"
    shutil.copytree(ROOT / "calib", calib)
    cfg = json.loads((calib / "hardware.json").read_text())
    cfg["field_run"] = {"speed_cm_s": 6, "max_cm": 200}
    (calib / "hardware.json").write_text(json.dumps(cfg))
    hw = hardware.PiHardware(calib / "hardware.json", dry_run=True, calib_dir=calib, camera=TrayCamera())
    video = tmp_path / "fields" / "Crop" / "field_runs" / "run.mp4"
    video.parent.mkdir(parents=True)
    yield hw, video
    hw.close()


def frames_in(video: Path) -> int:
    out = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                          "-show_entries", "stream=codec_name,nb_read_frames", "-of", "json", str(video)],
                         capture_output=True, text=True).stdout
    stream = json.loads(out)["streams"][0]
    assert stream["codec_name"] == "h264"
    return int(stream["nb_read_frames"])


def test_detections_come_back_to_frame_pixels_and_never_outside_the_tray(setup_run, tmp_path):
    hw, video = setup_run
    det = FakeDetector(tmp_path / "pics", weed_tick=0)
    frame = hw.capture()
    dets = det.detect(frame, 0)
    labels = sorted(d["label"] for d in dets)
    assert labels == ["crop", "weed"]  # the weed outside the left wall was dropped
    assert det.dropped_outside == 1
    weed = next(d for d in dets if d["label"] == "weed")
    # the crop's middle column is the tray's middle in the 1296x972 frame
    scale = 1296 / 1600
    middle = (sum(WALLS) / 4) * scale
    assert abs(weed["spray_px"][0] - middle) < 8
    assert det.with_tray == 1 and det.latest_jpeg[:2] == b"\xff\xd8"


def test_full_run_stop_and_go_video_and_logs(setup_run, tmp_path):
    hw, video = setup_run
    statuses = []
    summary = runner.run_on_field(None, threading.Event(), video, "sam_muse", step_cm=19, length_cm=70,
                                  hardware=hw, vision=FakeDetector(tmp_path / "pics"),
                                  progress=statuses.append)
    assert summary["phase"] == "done" and summary["error"] is None
    assert summary["frames"] >= 3 and summary["detected"] == summary["frames"]
    assert summary["sprays"] == 1 and summary["dropped_outside_tray"] >= summary["frames"]
    assert summary["tray_found"] == summary["frames"]
    pumps = [e for e in hw.events if e["type"] == "pump"]
    assert len(pumps) == 1
    assert frames_in(video) == summary["frames"]
    saved = json.loads(video.with_suffix(".json").read_text())
    assert saved["sprays"] == 1 and saved["video"] == "run.mp4"
    events = json.loads(video.with_name("run_events.json").read_text())
    assert events[0]["type"] != "start" and any(e["type"] == "pump" for e in events)
    assert any(s.get("phase", "").startswith("picture") for s in statuses)  # reported twice a second
    assert not hw.armed and hw._target == 0


def test_stop_button_ends_the_run_safely(setup_run, tmp_path):
    hw, video = setup_run
    stop = threading.Event()
    threading.Timer(2.0, stop.set).start()
    summary = runner.run_on_field(None, stop, video, "sam_muse", step_cm=19, length_cm=150,
                                  hardware=hw, vision=FakeDetector(tmp_path / "pics"))
    assert summary["phase"] == "stopped" and summary["stopped_early"]
    assert summary["seconds"] < 6
    assert not hw.armed and hw._target == 0
    assert video.with_suffix(".json").is_file()
    assert not [e for e in hw.events if e["type"] == "pump"]


def test_the_page_length_is_capped(setup_run, tmp_path):
    hw, video = setup_run
    stop = threading.Event()
    stop.set()
    summary = runner.run_on_field(None, stop, video, "sam_muse", length_cm=5000,
                                  hardware=hw, vision=FakeDetector(tmp_path / "pics"))
    assert summary["length_cm"] == 200


def test_weeds_below_the_threshold_are_protected_not_aimed_at(setup_run, tmp_path):
    hw, video = setup_run

    class Unsure(FakeDetector):
        def _sam_muse(self, view, n):
            h, w = view.shape[:2]
            u, v = w / 2, h / 2
            box = [u - 8, v - 8, u + 8, v + 8]
            poly = [(u - 8, v - 8), (u + 8, v - 8), (u + 8, v + 8), (u - 8, v + 8)]
            return [{"label": "weed", "action": "spray", "conf": 0.79, "box": box, "spray_px": (u, v), "polygon_px": poly},
                    {"label": "weed", "action": "spray", "conf": 0.80, "box": box, "spray_px": (u, v), "polygon_px": poly}]

    det = Unsure(tmp_path / "pics")
    det.min_conf = 0.80
    dets = det.detect(hw.capture(), 0)
    by_conf = {d["conf"]: d for d in dets}
    assert by_conf[0.79]["action"] == "protect" and by_conf[0.79]["label"] == "unsure"
    assert by_conf[0.79]["spray_px"] is None
    assert by_conf[0.80]["action"] == "spray"
    assert det.below_threshold == 1

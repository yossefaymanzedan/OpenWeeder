"""Real-time videos (video_rec.py): the recorder itself, a training recording,
and a field run (live video + segmentation video). A fake phone streams frames.

    python -m pytest tests/test_videos.py -q
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import hardware  # noqa: E402
import runner  # noqa: E402
import trainer  # noqa: E402
import video_rec  # noqa: E402
from test_field_run import FakeDetector, TrayCamera  # noqa: E402


class StreamingCamera(TrayCamera):
    """TrayCamera that also streams like the phone page: frames at the asked
    rate to every subscriber, while a stream is on."""

    def __init__(self):
        super().__init__()
        full = cv2.imdecode(np.frombuffer(self.jpeg, np.uint8), cv2.IMREAD_COLOR)
        self.small = cv2.imencode(".jpg", cv2.resize(full, (640, 480)))[1].tobytes()
        self.subscribers = []
        self.stream = None
        self.sent = 0
        threading.Thread(target=self._run, daemon=True).start()

    def subscribe(self, fn):
        self.subscribers.append(fn)

    def unsubscribe(self, fn):
        self.subscribers.remove(fn)

    def start_stream(self, fps=5, width=640, quality=0.7):
        self.stream = fps

    def stop_stream(self):
        self.stream = None

    def _run(self):
        while True:
            if self.stream:
                for fn in list(self.subscribers):
                    fn(self.small, time.monotonic(), "stream")
                self.sent += 1
                time.sleep(1 / self.stream)
            else:
                time.sleep(0.05)


def probe(path: Path) -> tuple[str, float]:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=codec_name:format=duration", "-of", "json", str(path)],
                         capture_output=True, text=True).stdout
    d = json.loads(out)
    return d["streams"][0]["codec_name"], float(d["format"]["duration"])


def test_recorder_makes_a_real_time_video(tmp_path):
    cam = StreamingCamera()
    marks = []
    rec = video_rec.LiveRecorder(cam, tmp_path / "v.mp4",
                                 overlay=lambda bgr: marks.append(1) or video_rec.caption(bgr, ["hello"])).start()
    time.sleep(2.0)
    assert rec.stop()
    codec, seconds = probe(tmp_path / "v.mp4")
    assert codec == "h264" and 1.5 <= seconds <= 2.8
    assert marks and not (tmp_path / "v_frames").exists()
    assert cam.stream is None and not cam.subscribers  # the stream was stopped


def test_recorder_without_frames_writes_nothing(tmp_path):
    cam = StreamingCamera()
    cam.start_stream = lambda *a, **k: None  # an old phone page that cannot stream
    rec = video_rec.LiveRecorder(cam, tmp_path / "v.mp4").start()
    time.sleep(0.3)
    assert not rec.stop() and not (tmp_path / "v.mp4").exists()


@pytest.fixture
def calib(tmp_path):
    d = tmp_path / "calib"
    shutil.copytree(ROOT / "calib", d)
    cfg = json.loads((d / "hardware.json").read_text())
    cfg["recording"]["max_cm"] = 12
    cfg["recording"]["speed_cm_s"] = 6
    cfg["field_run"] = {"speed_cm_s": 6, "max_cm": 40, "aim_hold_s": 0}
    (d / "hardware.json").write_text(json.dumps(cfg))
    return d


def test_training_recording_saves_its_video(calib, tmp_path):
    robot = hardware.PiHardware(calib / "hardware.json", dry_run=True, calib_dir=calib, camera=StreamingCamera())
    root = tmp_path / "fields"
    messages = []
    try:
        trainer.record_images("Crop", threading.Event(), fps=4, root=root, hardware=robot,
                              progress=lambda n, cm, msg=None: messages.append(msg))
    finally:
        robot.close()
    videos = list((root / "Crop" / "train" / "videos").glob("*.mp4"))
    assert len(videos) == 1
    codec, seconds = probe(videos[0])
    assert codec == "h264" and seconds > 1
    assert "Video: train/videos/" in messages[-1]


def test_field_run_saves_live_and_segmentation_videos(calib, tmp_path):
    robot = hardware.PiHardware(calib / "hardware.json", dry_run=True, calib_dir=calib, camera=StreamingCamera())
    video = tmp_path / "fields" / "Crop" / "field_runs" / "run.mp4"
    video.parent.mkdir(parents=True)
    try:
        summary = runner.run_on_field(None, threading.Event(), video, "sam_muse", length_cm=40,
                                      hardware=robot, vision=FakeDetector(tmp_path / "pics", weed_tick=0))
    finally:
        robot.close()
    assert summary["video"] == "run.mp4" and summary["live_video"] == "run_live.mp4"
    assert probe(video)[0] == "h264"
    codec, seconds = probe(video.with_name("run_live.mp4"))
    assert codec == "h264" and seconds >= summary["seconds"] * 0.5
    assert runner.videos("Crop", tmp_path / "fields") == ["run.mp4"]  # the live one is listed with it, not alone

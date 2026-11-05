"""runner.py: the blank robot function and the model/video listing.

    python -m pytest tests/test_runner.py -v
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import runner  # noqa: E402


def test_yolo_mode_without_ultralytics_says_so_and_never_moves(tmp_path):
    # Was a blank function until the robot was built; tests/test_field_run.py covers
    # the real run. The Pi has no ultralytics, so YOLO mode must explain itself.
    pytest.importorskip("hardware")
    import importlib.util
    if importlib.util.find_spec("ultralytics"):
        pytest.skip("ultralytics is installed here")
    from hardware import PiHardware
    hw = PiHardware(dry_run=True)
    video = tmp_path / "fields" / "Crop" / "field_runs" / "x.mp4"
    try:
        with pytest.raises(RuntimeError, match="ultralytics"):
            runner.run_on_field(tmp_path / "best.pt", threading.Event(), video, "yolo", hardware=hw)
    finally:
        hw.close()
    assert "drive" not in [e["type"] for e in hw.events]


def test_models_lists_only_crops_with_a_saved_model_newest_first(tmp_path):
    (tmp_path / "a" / "yolo").mkdir(parents=True)
    (tmp_path / "b" / "yolo").mkdir(parents=True)
    (tmp_path / "c").mkdir()
    (tmp_path / "a" / "yolo" / "best.pt").write_bytes(b"1")
    (tmp_path / "b" / "yolo" / "best.pt").write_bytes(b"2")
    import os
    os.utime(tmp_path / "a" / "yolo" / "best.pt", (1, 1))
    assert [m["crop"] for m in runner.models(tmp_path)] == ["b", "a"]


def test_videos_and_new_path(tmp_path):
    assert runner.videos("a", tmp_path) == []
    v = runner.new_video_path("a", tmp_path)
    assert v.parent == tmp_path / "a" / "field_runs" and v.suffix == ".mp4"
    v.write_bytes(b"x")
    assert runner.videos("a", tmp_path) == [v.name]


def test_advice_switches_at_map50_0_7():
    assert runner.advice(None)["use"] == "sam_muse"
    assert runner.advice(0.47)["use"] == "sam_muse" and "below 0.70" in runner.advice(0.47)["text"]
    assert runner.advice(0.69)["use"] == "sam_muse"
    assert runner.advice(0.70)["use"] == "yolo"
    assert runner.advice(0.9)["level"] == "ok"


def test_options_read_the_models_map50(tmp_path):
    import json
    (tmp_path / "a" / "yolo").mkdir(parents=True)
    (tmp_path / "a" / "yolo" / "best.pt").write_bytes(b"1")
    (tmp_path / "a" / "yolo" / "metrics.json").write_text(json.dumps({"best": {"map50": 0.47}}))
    (tmp_path / "b").mkdir()
    o = {x["crop"]: x for x in runner.options(["a", "b"], tmp_path)}
    assert o["a"]["has_model"] and o["a"]["map50"] == 0.47 and o["a"]["use"] == "sam_muse"
    assert not o["b"]["has_model"] and o["b"]["use"] == "sam_muse"

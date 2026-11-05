"""The web app's Run on the field routes: the run goes to the background, the
page polls its status, Stop works, two robot jobs never run at once. Uses Flask's
test client, a dry-run robot and a fake detector (no API money, nothing moves).

    python -m pytest tests/test_field_run_app.py -q
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import app as app_mod  # noqa: E402
import hardware  # noqa: E402
import runner  # noqa: E402
from test_field_run import FakeDetector, TrayCamera  # noqa: E402

CROP = "Molokhia"


@pytest.fixture
def client(tmp_path, monkeypatch):
    if not (ROOT / "fields" / CROP).is_dir():
        pytest.skip("needs the Molokhia crop")
    monkeypatch.chdir(ROOT)
    calib = tmp_path / "calib"
    shutil.copytree(ROOT / "calib", calib)
    cfg = json.loads((calib / "hardware.json").read_text())
    cfg["field_run"] = {"speed_cm_s": 6, "max_cm": 70}
    (calib / "hardware.json").write_text(json.dumps(cfg))
    robot = hardware.PiHardware(calib / "hardware.json", dry_run=True, calib_dir=calib, camera=TrayCamera())
    monkeypatch.setattr(hardware, "_ROBOT", robot)
    monkeypatch.setattr(runner, "FieldVision", lambda crop, mode, model_path, work_dir: FakeDetector(tmp_path / "pics"))
    videos = tmp_path / "fields" / CROP / "field_runs"
    videos.mkdir(parents=True)
    monkeypatch.setattr(runner, "new_video_path", lambda crop, root=None: videos / "run.mp4")
    app = app_mod.create_app(tmp_path / "cache", tmp_path / "api.log")
    yield app.test_client(), robot, videos
    robot.close()


def wait_done(client, limit=60):
    end = time.monotonic() + limit
    while time.monotonic() < end:
        r = client.get(f"/api/field_run/status/{CROP}").get_json()
        if not r["on"]:
            return r
        time.sleep(0.3)
    return r


def test_start_runs_in_the_background_and_finishes(client):
    c, robot, videos = client
    t0 = time.monotonic()
    first = c.post(f"/api/field_run/start/{CROP}?mode=sam_muse&step=19&length=70").get_json()
    assert time.monotonic() - t0 < 2 and first["on"]
    time.sleep(3)
    live = c.get(f"/api/field_run/status/{CROP}").get_json()
    assert live["on"] and live["status"].get("phase", "").startswith("picture")
    assert c.get(f"/field_run_live/{CROP}.jpg").data[:2] == b"\xff\xd8"
    again = c.post(f"/api/field_run/start/{CROP}?mode=sam_muse").get_json()
    assert again["on"]  # a second Start while running changes nothing
    done = wait_done(c)
    assert done["msg"].startswith("Done.") and "1 sprays" in done["msg"], done["msg"]
    assert done["video"] == "run.mp4" and (videos / "run.mp4").is_file()
    assert runner.summary(CROP, "run.mp4", videos.parent.parent)["sprays"] == 1
    assert not robot.armed and robot._target == 0


def test_stop_button(client):
    c, robot, videos = client
    c.post(f"/api/field_run/start/{CROP}?mode=sam_muse&step=19&length=70")
    time.sleep(2)
    stopping = c.post(f"/api/field_run/stop/{CROP}").get_json()
    assert stopping["msg"].startswith("Stopping")
    done = wait_done(c, 15)
    assert done["msg"].startswith("Stopped.")
    assert not [e for e in robot.events if e["type"] == "pump"]
    assert robot._target == 0


def test_recording_and_field_run_never_drive_at_once(client):
    c, robot, videos = client
    c.post(f"/api/field_run/start/{CROP}?mode=sam_muse&step=19&length=70")
    time.sleep(1)
    rec = c.post(f"/api/train/{CROP}/collect/start").get_json()
    assert not rec["on"] and "field run" in rec["msg"]
    c.post(f"/api/field_run/stop/{CROP}")
    wait_done(c, 15)


def test_page_shows_the_run_panel_and_the_limit(client):
    c, robot, videos = client
    html = c.get("/run-field").get_data(as_text=True)
    limit = json.loads((ROOT / "calib" / "hardware.json").read_text())["field_run"]["max_cm"]
    assert 'id="run-box"' in html and f"at most {limit} cm per run" in html
    assert "No robot is connected yet" not in html

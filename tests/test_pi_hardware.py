"""PiHardware in dry run (no GPIO, no motion), and recording with it (PRD R12, R8).

    python -m pytest tests/test_pi_hardware.py -q
"""
from __future__ import annotations

import json
import shutil
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import hardware  # noqa: E402
import trainer  # noqa: E402


@pytest.fixture
def calib(tmp_path: Path) -> Path:
    """A copy of calib/ with a short, fast recording so the tests are quick."""
    d = tmp_path / "calib"
    shutil.copytree(ROOT / "calib", d)
    cfg = json.loads((d / "hardware.json").read_text())
    cfg["recording"] = {"speed_cm_s": 8, "max_cm": 6, "stall_s": 3}
    (d / "hardware.json").write_text(json.dumps(cfg))
    return d


@pytest.fixture
def hw(calib: Path):
    robot = hardware.PiHardware(calib / "hardware.json", dry_run=True, calib_dir=calib)
    yield robot
    robot.close()


class FakeCamera:
    """Stands in for the phone: returns a JPEG of a known size."""

    def __init__(self, width: int = 1600, height: int = 1200, fail: bool = False):
        import cv2
        self.fail = fail
        self.calls = 0
        image = np.zeros((height, width, 3), np.uint8)
        image[: height // 2] = (0, 200, 0)  # top half green (BGR)
        self.jpeg = cv2.imencode(".jpg", image)[1].tobytes()

    def connected(self) -> bool:
        return not self.fail

    def capture(self, timeout: float = 2.0) -> bytes:
        self.calls += 1
        if self.fail:
            raise RuntimeError("The phone camera is not connected")
        return self.jpeg


def wait_for(condition, timeout: float = 5.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if condition():
            return True
        time.sleep(0.02)
    return False


def test_starts_disarmed_with_the_arm_parked(hw):
    assert not hw.armed
    assert hw.events[0]["type"] == "start"
    assert hw._servo_deg == pytest.approx(hw.park_angle())


def test_forward_ticks_up_backward_ticks_down(hw):
    start = hw.ticks()
    hw.drive(5)
    assert wait_for(lambda: hw.ticks() >= start + 3)
    hw.stop()
    time.sleep(0.5)
    middle = hw.ticks()
    hw.drive(-5)
    assert wait_for(lambda: hw.ticks() <= middle - 3)
    hw.stop()


def test_drive_returns_immediately_and_clamps_to_the_limit(hw):
    t0 = time.monotonic()
    hw.drive(100)
    assert time.monotonic() - t0 < 0.05
    assert hw._target == hw.max_speed
    hw.stop()


def test_stop_brakes_then_releases(hw):
    hw.drive(5)
    time.sleep(0.3)
    hw.stop()
    front = hw.cfg["motors"]["front"]
    assert wait_for(lambda: hw._pins[front["forward"]] == 1 and hw._pins[front["backward"]] == 1, 0.2)
    assert wait_for(lambda: hw._pins[front["forward"]] == 0 and hw._pins[front["enable"]] == 0, 1.0)


def test_pump_refuses_while_disarmed(hw):
    with pytest.raises(RuntimeError, match="disarmed"):
        hw.pump(150)


def test_pump_pulse_length_and_pin_ends_low(hw):
    hw.arm()
    pin = hw.cfg["pump_gpio_bcm"]
    seen = []
    watcher = threading.Thread(target=lambda: [seen.append(hw._pins[pin]) or time.sleep(0.005) for _ in range(60)])
    watcher.start()
    t0 = time.monotonic()
    hw.pump(150)
    took = time.monotonic() - t0
    watcher.join()
    assert 0.14 <= took <= 0.2
    assert 1 in seen
    assert hw._pins[pin] == 0
    assert hw.events[-1]["type"] == "pump" and hw.events[-1]["value"] == 150


def test_pump_caps(hw):
    hw.arm()
    with pytest.raises(ValueError):
        hw.pump(hw.pump_max_ms + 1)
    with pytest.raises(ValueError):
        hw.pump(-1)


def test_servo_limits_and_arming(hw):
    with pytest.raises(ValueError):
        hw.servo_set(hw.servo_max + 1)
    with pytest.raises(ValueError):
        hw.servo_set(hw.servo_min - 1)
    with pytest.raises(RuntimeError, match="disarmed"):
        hw.servo_set(90)
    hw.servo_set(hw.park_angle())  # parking is always allowed
    hw.arm()
    hw.servo_set(90)
    assert hw.events[-1] == {**hw.events[-1], "type": "servo", "value": 90.0}


def test_watchdog_stops_a_car_nobody_is_driving(hw):
    hw.drive(5)
    hw.arm()
    assert wait_for(lambda: hw._target == 0, hw.watchdog_s + 1.5)
    assert any(e["type"] == "stop" and e["value"] == "watchdog" for e in hw.events)
    assert hw._pins[hw.cfg["pump_gpio_bcm"]] == 0
    assert hw._servo_deg == pytest.approx(hw.park_angle())


def test_sleep_keeps_the_watchdog_fed(hw):
    hw.drive(5)
    hw.sleep(hw.watchdog_s + 0.5)
    assert hw._target == 5
    hw.stop()


def test_error_inside_a_run_leaves_the_robot_safe(hw):
    with pytest.raises(ZeroDivisionError):
        with hw:
            hw.drive(5)
            1 / 0
    assert hw._target == 0
    assert hw.events[-1]["type"] in ("stop", "servo")


def test_event_log_has_every_action_and_saves(hw, tmp_path):
    hw.drive(3)
    hw.capture()
    hw.stop()
    hw.arm()
    hw.pump(10)
    kinds = [e["type"] for e in hw.events]
    for kind in ("start", "drive", "capture", "stop", "pump"):
        assert kind in kinds
    assert set(hw.events[0]) == {"t", "type", "value", "tick", "arm"}
    path = hw.save_events(tmp_path / "events.json")
    assert json.loads(path.read_text())[1:] == json.loads(json.dumps(hw.events))[1:]


def test_capture_without_a_phone_in_dry_run_is_a_grey_frame(hw):
    frame = hw.capture()
    assert frame.shape == (972, 1296, 3) and frame.dtype == np.uint8


def test_capture_crops_to_four_by_three_rgb(calib):
    robot = hardware.PiHardware(calib / "hardware.json", dry_run=True, calib_dir=calib,
                                camera=FakeCamera(1920, 1080))
    try:
        frame = robot.capture()
        assert frame.shape == (972, 1296, 3)
        assert frame[10, 648, 1] > 150 and frame[10, 648, 0] < 60  # green on top, as RGB
        assert frame[-10, 648].max() < 60
    finally:
        robot.close()


def test_recording_saves_frames_and_ticks_and_stops_at_the_limit(calib, tmp_path):
    robot = hardware.PiHardware(calib / "hardware.json", dry_run=True, calib_dir=calib, camera=FakeCamera())
    root = tmp_path / "fields"
    (root / "Crop" / "train" / "images").mkdir(parents=True)
    (root / "Crop" / "train" / "images" / "img_00007.jpg").write_bytes(FakeCamera().jpeg)
    messages = []
    try:
        saved = trainer.record_images("Crop", threading.Event(), fps=4, root=root, hardware=robot,
                                      progress=lambda n, cm, msg=None: messages.append(msg))
    finally:
        robot.close()
    names = sorted(p.name for p in (root / "Crop" / "train" / "images").iterdir())
    assert saved >= 2 and len(names) == saved + 1
    assert names[1] == "img_00008.jpg"  # numbering continues after the highest one
    ticks = json.loads((root / "Crop" / "train" / "ticks.json").read_text())
    assert sorted(ticks) == names[1:]
    assert ticks[names[-1]] >= 6 / robot.cm_per_tick - 1
    assert "limit" in messages[-1]
    kinds = [e["type"] for e in robot.events]
    assert kinds.index("capture") < kinds.index("drive")  # photo before moving
    assert robot._target == 0


class TrayCamera(FakeCamera):
    """A phone looking down at a soil tray with glass walls (as in tests/test_tray.py)."""

    def __init__(self):
        import cv2
        super().__init__()
        image = np.full((1200, 1600, 3), 235, np.uint8)
        soil = np.array([[500, 0], [1100, 0], [1120, 1199], [520, 1199]], np.int32)
        cv2.fillPoly(image, [soil], (45, 75, 110))
        rng = np.random.default_rng(1)
        noise = rng.integers(-18, 18, image.shape, dtype=np.int16)
        inside = np.zeros(image.shape[:2], np.uint8)
        cv2.fillPoly(inside, [soil], 1)
        image = np.where(inside[..., None] == 1, np.clip(image.astype(np.int16) + noise, 0, 255), image).astype(np.uint8)
        cv2.line(image, (500, 0), (520, 1199), (40, 60, 40), 9)
        cv2.line(image, (1100, 0), (1120, 1199), (40, 60, 40), 9)
        cv2.circle(image, (800, 600), 90, (60, 170, 70), -1)
        self.jpeg = cv2.imencode(".jpg", image)[1].tobytes()


def test_recording_saves_the_tray_crop_and_keeps_the_full_frame(calib, tmp_path):
    import cv2
    robot = hardware.PiHardware(calib / "hardware.json", dry_run=True, calib_dir=calib, camera=TrayCamera())
    root = tmp_path / "fields"
    try:
        saved = trainer.record_images("Crop", threading.Event(), fps=4, root=root, hardware=robot)
    finally:
        robot.close()
    train = root / "Crop" / "train"
    names = sorted(p.name for p in (train / "images").iterdir())
    assert len(names) == saved and sorted(p.name for p in (train / "raw").iterdir()) == names
    full = cv2.imread(str(train / "raw" / names[0]))
    cropped = cv2.imread(str(train / "images" / names[0]))
    assert full.shape[:2] == (972, 1296)
    # the 1600x1200 phone frame becomes 1296x972: the tray is about 600/1600 of the width
    assert cropped.shape[0] == 972 and abs(cropped.shape[1] - 1296 * 600 / 1600) < 30
    trays = json.loads((train / "trays.json").read_text())
    assert sorted(trays) == names and all(t is not None for t in trays.values())
    assert abs(trays[names[0]]["offset"] - ((500 + 1100 + 520 + 1120) / 4 - 800) / 1600) < 0.02


def test_recording_without_a_tray_saves_whole_frames(calib, tmp_path):
    robot = hardware.PiHardware(calib / "hardware.json", dry_run=True, calib_dir=calib, camera=FakeCamera())
    root = tmp_path / "fields"
    messages = []
    try:
        trainer.record_images("Crop", threading.Event(), fps=4, root=root, hardware=robot,
                              progress=lambda n, cm, msg=None: messages.append(msg))
    finally:
        robot.close()
    import cv2
    first = sorted((root / "Crop" / "train" / "images").iterdir())[0]
    assert cv2.imread(str(first)).shape[:2] == (972, 1296)
    trays = json.loads((root / "Crop" / "train" / "trays.json").read_text())
    assert all(t is None for t in trays.values())
    assert "Tray found and cropped in 0" in messages[-1]


def test_recording_does_not_move_without_a_camera(calib, tmp_path):
    robot = hardware.PiHardware(calib / "hardware.json", dry_run=True, calib_dir=calib,
                                camera=FakeCamera(fail=True))
    try:
        robot.dry_run = False  # no GPIO was set up; now capture() behaves as on the real robot
        with pytest.raises(RuntimeError, match="not connected"):
            trainer.record_images("Crop", threading.Event(), root=tmp_path / "fields", hardware=robot)
        assert "drive" not in [e["type"] for e in robot.events]
    finally:
        robot.close()


def test_recording_stops_when_asked(calib, tmp_path):
    cfg = json.loads((calib / "hardware.json").read_text())
    cfg["recording"]["max_cm"] = 1000
    (calib / "hardware.json").write_text(json.dumps(cfg))
    robot = hardware.PiHardware(calib / "hardware.json", dry_run=True, calib_dir=calib, camera=FakeCamera())
    stop = threading.Event()
    threading.Timer(1.0, stop.set).start()
    try:
        t0 = time.monotonic()
        saved = trainer.record_images("Crop", stop, fps=2, root=tmp_path / "fields", hardware=robot)
        assert time.monotonic() - t0 < 2.5
        assert 2 <= saved <= 4
    finally:
        robot.close()


class Blocked(hardware.PiHardware):
    """A dry-run robot whose wheels cannot turn: the encoder never counts."""

    def _simulate(self, dt):
        pass


def test_motors_stop_when_the_wheels_do_not_turn(calib):
    robot = Blocked(calib / "hardware.json", dry_run=True, calib_dir=calib)
    try:
        robot.drive(3)
        t0 = time.monotonic()
        while robot._target != 0 and time.monotonic() - t0 < robot.stall_s + 2:
            robot.ticks()  # keep the watchdog fed: the program is fine, the wheels are not
            time.sleep(0.05)
        took = time.monotonic() - t0
        assert robot._target == 0 and robot.stalled
        assert robot.stall_s - 0.2 <= took <= robot.stall_s + 0.5
        assert any(e["type"] == "stop" and "stalled" in str(e["value"]) for e in robot.events)
        robot.drive(3)  # driving again is allowed (and gets its own stall_s)
        assert not robot.stalled and robot._target == 3
    finally:
        robot.close()


def test_turning_wheels_never_count_as_stalled(calib):
    robot = hardware.PiHardware(calib / "hardware.json", dry_run=True, calib_dir=calib)
    try:
        robot.drive(3)
        robot.sleep(robot.stall_s + 1)
        assert robot._target == 3 and not robot.stalled
        robot.stop()
    finally:
        robot.close()

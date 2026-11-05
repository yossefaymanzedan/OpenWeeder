"""planner.run_stepwise: stop, picture, detect, spray, move one picture on.
On the simulated robot (FakeHardware) with this robot's measured layout:
camera 16.5 cm ahead of the arm's pivot, arm 8 cm, camera 29 cm high.

    python -m pytest tests/test_stepwise.py -q
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import geometry as g  # noqa: E402
import hardware as hw_mod  # noqa: E402
import planner as pl  # noqa: E402

SIZE = (1296, 972)


def calib() -> g.Calib:
    robot = {"camera_height_cm": 29, "ridge_height_cm": 8, "camera_to_nozzle_cm": 16.5,
             "camera_side_offset_cm": 0, "nozzle_height_cm": 15, "cm_per_tick": 1.0,
             "camera_hfov_deg": 66, "image_width_px": 1296, "image_height_px": 972,
             "arm_length_cm": 8, "spray_radius_cm": 1.5, "protect_margin_cm": 0.5,
             "placeholder": False}
    servo = {"arm": {"center_trim_deg": 0, "sign": 1, "max_deg": 80, "park_deg": 70}}
    return g.Calib(robot=robot, servo=servo, homographies={}, quiet=True)


class Field:
    """Plants at known places (cm along the row, cm to the side). The vision
    reports those inside the picture, in the picture's pixels, like SAM + Muse."""

    def __init__(self, hw: hw_mod.FakeHardware, c: g.Calib, weeds, crops=()):
        self.hw, self.c = hw, c
        self.weeds, self.crops = list(weeds), list(crops)
        self.pictures = []

    def detect(self, image, tick):
        pivot = self.hw.pos
        self.pictures.append(pivot)
        w, h = SIZE

        def px(x, y):
            return g.ground_to_pixel(x - pivot, y, "ridge", SIZE, self.c)

        def inside(u, v):
            return 0 <= u < w and 0 <= v < h

        out = []
        for x, y in self.weeds:
            u, v = px(x, y)
            if inside(u, v):
                out.append({"label": "weed", "action": "spray", "conf": 0.97, "box": [u - 4, v - 4, u + 4, v + 4],
                            "spray_px": (u, v), "polygon_px": [px(x - .5, y - .5), px(x + .5, y - .5), px(x + .5, y + .5), px(x - .5, y + .5)]})
        for x, y in self.crops:
            u, v = px(x, y)
            if inside(u, v):
                poly = [px(x - 2, y - 2), px(x + 2, y - 2), px(x + 2, y + 2), px(x - 2, y + 2)]
                out.append({"label": "crop", "action": "protect", "conf": 0.9, "box": [u - 20, v - 20, u + 20, v + 20],
                            "spray_px": None, "polygon_px": poly})
        return out


def run(weeds, crops=(), length=80.0, step=None):
    c = calib()
    hw = hw_mod.FakeHardware(camera_to_nozzle_cm=16.5, park_deg=g.arm_to_servo(70, c))
    field = Field(hw, c, weeds, crops)
    p = pl.Planner(hw, field, c, cfg=pl.Config(speed=3.0, out_dir=Path("/tmp")))
    p.run_stepwise(length, step)
    return p, hw, field


def landings(hw: hw_mod.FakeHardware, c: g.Calib):
    """Where each pump shot really landed: the car's true position + the arm's
    real angle at that moment."""
    out = []
    for e in hw.events:
        if e["type"] == "pump":
            phi = g.servo_to_arm(e["arm"], c)
            pos = next(t[1] for t in hw.trace if t[0] >= e["t"])
            out.append((pos + 8 * math.cos(math.radians(phi)), 8 * math.sin(math.radians(phi))))
    return out


def test_pictures_are_one_picture_apart_and_cover_the_row():
    p, hw, field = run([], length=100)
    frame = p.frame_length_cm()
    gaps = [b - a for a, b in zip(field.pictures, field.pictures[1:])]
    assert gaps and all(abs(gap - round(frame)) <= 1.5 for gap in gaps)
    last_front = field.pictures[-1] + 16.5 + frame / 2
    assert last_front >= 100  # the camera saw the whole row
    assert field.pictures[-2] + 16.5 + frame / 2 < 100  # and took no picture more than needed


def test_every_reachable_weed_is_hit_including_by_backing_up():
    c = calib()
    # 12 cm: needs a stop at 4 cm; 28 cm: front edge of picture 1, waits for
    # picture 2 and then needs the car to back up; 45 and 60 cm: later pictures.
    weeds = [(12, 0), (28, 3), (45, -5), (60, 6)]
    p, hw, field = run(weeds, length=90)
    statuses = {round(w["x"]): w.get("status") for w in p.weeds}
    assert all(s == "sprayed" for s in statuses.values()), statuses
    hits = landings(hw, c)
    assert len(hits) == len(weeds)
    for x, y in weeds:
        assert min(math.dist((x, y), hit) for hit in hits) <= 1.5  # inside the wet radius
    assert any(e["type"] == "drive" and e["value"] < 0 for e in hw.events)  # it backed up


def test_out_of_reach_and_near_crop_and_before_start_are_skipped():
    weeds = [(40, 9.5), (55, 0), (4, 0)]
    crops = [(56.5, 0.5)]
    p, hw, field = run(weeds, crops, length=90)
    status = {round(w["x"]): w.get("status") for w in p.weeds}
    assert status[40] == "skipped: out of reach"
    assert status[55] == "skipped: near crop"
    assert status[4] == "skipped: before the start"
    assert not [e for e in hw.events if e["type"] == "pump"]


def test_a_weed_at_the_front_edge_waits_for_the_next_picture():
    # 29.5 cm is inside picture 1 (it reaches 30.6 cm) but within the guard of
    # its front edge; the crop just beyond it shows only in picture 2.
    weeds, crops = [(29.5, 0)], [(31.5, 0)]
    p, hw, field = run(weeds, crops, length=70)
    assert p.weeds[0]["status"] == "skipped: near crop"
    assert not [e for e in hw.events if e["type"] == "pump"]


def test_step_can_be_set():
    p, hw, field = run([], length=100, step=10)
    gaps = [b - a for a, b in zip(field.pictures, field.pictures[1:])]
    assert gaps and all(abs(gap - 10) <= 1.5 for gap in gaps)


def test_never_drives_beyond_the_rails():
    c = calib()
    hw = hw_mod.FakeHardware(camera_to_nozzle_cm=16.5, park_deg=g.arm_to_servo(70, c))
    field = Field(hw, c, [(40, 0), (90, 0)])
    p = pl.Planner(hw, field, c, cfg=pl.Config(speed=3.0, out_dir=Path("/tmp"), max_travel_cm=75))
    p.run_stepwise(100)
    assert [round(x) for x in field.pictures] == [0, 28, 56, 75]
    assert max(t[1] for t in hw.trace) <= 75 + 1          # the car stayed on the rails
    status = {round(w["x"]): w["status"] for w in p.weeds}
    assert status == {40: "sprayed", 90: "skipped: beyond the rails"}


class WallHardware(hw_mod.FakeHardware):
    """The simulated car with a wall at the end of the rails: it cannot get
    past wall_cm, so its encoder stops counting there."""

    def __init__(self, wall_cm, **kwargs):
        super().__init__(**kwargs)
        self.wall = wall_cm

    def sleep(self, s):
        super().sleep(s)
        self.pos = min(self.pos, self.wall)


def test_stops_at_the_wall_and_treats_it_as_the_end():
    c = calib()
    hw = WallHardware(75, camera_to_nozzle_cm=16.5, park_deg=g.arm_to_servo(70, c))
    field = Field(hw, c, [(40, 0), (80, 0), (90, 0)])
    p = pl.Planner(hw, field, c, cfg=pl.Config(speed=3.0, out_dir=Path("/tmp"), max_travel_cm=90, stall_s=2.0))
    p.run_stepwise(100)
    assert [round(x) for x in field.pictures] == [0, 28, 56, 75]  # last picture at the wall
    assert any(e["type"] == "wall" for e in p.log)
    status = {round(w["x"]): w["status"] for w in p.weeds}
    assert status == {40: "sprayed", 80: "sprayed", 90: "skipped: beyond the rails"}
    assert not any(e["type"] == "drive" and e["value"] > 0 for e in hw.events[-3:])  # not still pushing


def test_without_stall_detection_nothing_changes():
    p, hw, field = run([(40, 0)], length=90)
    assert p.cfg.stall_s is None and [w["status"] for w in p.weeds] == ["sprayed"]


def test_the_arm_holds_its_aim_before_the_pump():
    c = calib()
    hw = hw_mod.FakeHardware(camera_to_nozzle_cm=16.5, park_deg=g.arm_to_servo(70, c))
    field = Field(hw, c, [(40, 3)])
    p = pl.Planner(hw, field, c, cfg=pl.Config(speed=3.0, out_dir=Path("/tmp"), aim_hold_s=20))
    p.run_stepwise(70)
    holds = [e for e in p.log if e["type"] == "hold"]
    assert len(holds) == 1 and holds[0]["x_cm"] == pytest.approx(40, abs=0.5) and holds[0]["y_cm"] == pytest.approx(3, abs=0.5)
    aim = next(e for e in hw.events if e["type"] == "servo" and abs(e["value"] - holds[0]["servo"]) <= 0.05)  # the log rounds to 0.1 deg
    pump = next(e for e in hw.events if e["type"] == "pump")
    assert pump["t"] - aim["t"] >= 20                      # held still for 20 s
    moves = [e for e in hw.events if aim["t"] < e["t"] < pump["t"] and e["type"] in ("servo", "drive")]
    assert not moves                                       # nothing moved meanwhile

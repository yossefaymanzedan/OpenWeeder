"""Geometry tests, no hardware and no API: a synthetic camera looks at a grid
of dots at known positions; we find the dots in the image and check that
pixel_to_ground puts them back where they were.

    python -m pytest tests/test_geometry.py -v
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import geometry as g  # noqa: E402
from geometry import Calib, OutOfReach  # noqa: E402

W, H = 1296, 972
HFOV = 66.0
D = 12.0  # camera_to_nozzle_cm


def calib(**robot) -> Calib:
    r = {"placeholder": False, "camera_height_cm": 25.0, "ridge_height_cm": 8.0,
         "camera_to_nozzle_cm": D, "camera_side_offset_cm": 0.0,
         "nozzle_height_cm": 15.0, "cm_per_tick": 1.0, "camera_hfov_deg": HFOV,
         "image_width_px": W, "image_height_px": H}
    r.update(robot)
    return Calib(robot=r, servo={}, homographies={}, quiet=True)


class Camera:
    """Pinhole camera over the ground. World = nozzle coordinates (x forward,
    y left, z up). Straight down (tilt 0): image top = forward, image left =
    +y. tilt_deg pitches the camera about its own X axis; the camera is
    placed so that its optical axis hits the ground at (D, 0)."""

    def __init__(self, height_cm: float = 25.0, tilt_deg: float = 0.0):
        self.f = (W / 2) / math.tan(math.radians(HFOV) / 2)
        r0 = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]], float)
        t = math.radians(tilt_deg)
        rx = np.array([[1, 0, 0], [0, math.cos(t), -math.sin(t)],
                       [0, math.sin(t), math.cos(t)]])
        self.R = rx @ r0
        axis = self.R.T @ np.array([0, 0, 1.0])
        self.C = np.array([D, 0, 0]) - (height_cm / -axis[2]) * axis

    def project(self, x, y, z=0.0):
        pc = self.R @ (np.array([x, y, z], float) - self.C)
        return W / 2 + self.f * pc[0] / pc[2], H / 2 + self.f * pc[1] / pc[2]


def render_dots(cam: Camera, pts, z=0.0):
    """Draw each world point as a filled dot; return the image."""
    img = np.zeros((H, W), np.uint8)
    for x, y in pts:
        u, v = cam.project(x, y, z)
        cv2.circle(img, (int(round(u * 16)), int(round(v * 16))), 5 * 16, 255, -1,
                   lineType=cv2.LINE_AA, shift=4)
    return img


def detect(img, n):
    """Dot centers (u, v) found in the image."""
    num, _l, stats, cents = cv2.connectedComponentsWithStats((img > 127).astype(np.uint8))
    out = [tuple(cents[i]) for i in range(1, num) if stats[i, cv2.CC_STAT_AREA] > 20]
    assert len(out) == n, f"found {len(out)} dots, expected {n}"
    return out


def grid_points(rows=5, cols=7, dx=3.5, dy=4.0):
    """Dots around the camera's view: x from D-7 to D+7, y from -12 to +12."""
    return [(D + (r - rows // 2) * dx, (c - cols // 2) * dy)
            for r in range(rows) for c in range(cols)]


def match(found, cam, pts, plane, c, size=None, z=0.0):
    """Max error (cm) between pixel_to_ground(dot) and the true position."""
    worst = 0.0
    for (x, y) in pts:
        u0, v0 = cam.project(x, y, z)
        u, v = min(found, key=lambda p: (p[0] - u0) ** 2 + (p[1] - v0) ** 2)
        gx, gy = g.pixel_to_ground(u, v, plane, size, c)
        worst = max(worst, math.hypot(gx - x, gy - y))
    return worst


# ---------- pixel -> ground ----------

def test_straight_down_grid_ridge_and_furrow():
    c = calib()
    pts = grid_points()
    cam = Camera(25.0)
    err = match(detect(render_dots(cam, pts), len(pts)), cam, pts, "ridge", c)
    assert err < 0.3, err
    # furrow ground is 8 cm lower: same camera, the scale is larger
    cam = Camera(25.0)
    err = match(detect(render_dots(cam, pts, z=-8.0), len(pts)), cam, pts, "furrow", c, z=-8.0)
    assert err < 0.3, err


def test_scale_estimate_and_resolution_rescale():
    c = calib()
    s = c.cm_per_px("ridge")
    assert s == pytest.approx(2 * 25 * math.tan(math.radians(33)) / W)
    assert c.cm_per_px("furrow") == pytest.approx(s * 33 / 25)
    # a photo at half resolution lands on the same ground point
    full = g.pixel_to_ground(400, 300, "ridge", calib=c)
    half = g.pixel_to_ground(200, 150, "ridge", image_size=(W / 2, H / 2), calib=c)
    assert half == pytest.approx(full)
    # image center = camera_to_nozzle ahead of the nozzle, no side offset
    assert g.pixel_to_ground(W / 2, H / 2, "ridge", calib=c) == pytest.approx((D, 0.0))
    # up in the image = forward, left in the image = +y
    x_up, _ = g.pixel_to_ground(W / 2, 100, "ridge", calib=c)
    _, y_left = g.pixel_to_ground(100, H / 2, "ridge", calib=c)
    assert x_up > D and y_left > 0
    # round trip
    u, v = g.ground_to_pixel(*g.pixel_to_ground(321.5, 123.25, "furrow", calib=c),
                             "furrow", calib=c)
    assert (u, v) == pytest.approx((321.5, 123.25))


def test_side_offset_shifts_y():
    c = calib(camera_side_offset_cm=2.5)
    assert g.pixel_to_ground(W / 2, H / 2, "ridge", calib=c) == pytest.approx((D, 2.5))


def test_tilted_5_degrees_with_a4_homography():
    c = calib()
    cam = Camera(25.0, tilt_deg=5.0)
    # A4 flat on the ridge top, long side across the photo (landscape),
    # edges parallel to the photo edges; corners TL, TR, BR, BL
    paper = [(D + 10.5, 14.85), (D + 10.5, -14.85), (D - 10.5, -14.85), (D - 10.5, 14.85)]
    corners = [cam.project(x, y) for x, y in paper]
    hom = g.compute_homography(corners, (W, H), c)
    assert hom["paper_cm"] == [29.7, 21.0]
    assert hom["tilt_deg"] == pytest.approx(5.0, abs=0.5)
    assert hom["camera_height_est_cm"] == pytest.approx(25.0, abs=1.0)
    c2 = Calib(robot=c.robot, servo={}, homographies={"ridge": hom}, quiet=True)
    pts = grid_points()
    err_h = match(detect(render_dots(cam, pts), len(pts)), cam, pts, "ridge", c2)
    assert err_h < 0.5, err_h
    # without the homography the straight-down scale is visibly worse
    err_plain = match(detect(render_dots(cam, pts), len(pts)), cam, pts, "ridge", c)
    assert err_plain > err_h


def test_plane_for_and_locate():
    c = calib()
    model = {"band_half_px": 100.0}  # 100 px * 0.02505 cm/px = 2.5 cm
    assert g.plane_for(D, 0.0, model, calib=c) == "ridge"
    assert g.plane_for(D, 2.4, model, calib=c) == "ridge"
    assert g.plane_for(D, 3.0, model, calib=c) == "furrow"
    assert g.plane_for(D, -3.0, model, row_y_cm=-2.0, calib=c) == "ridge"
    assert g.plane_for(D, 9.0, None, calib=c) == "ridge"  # no band known
    assert g.plane_for(D, 0.0, {"band": {"half_width_px": 100}}, calib=c) == "ridge"
    # only one plane calibrated: that plane, whatever the position
    only = Calib(robot=c.robot, servo={}, quiet=True,
                 homographies={"furrow": {"H_px2paper": np.eye(3).tolist(),
                                          "image_size": [W, H], "cm_per_px": 0.03,
                                          "center_paper": [0, 0]}})
    assert g.plane_for(D, 0.0, model, calib=only) == "furrow"
    # locate re-places a furrow point with the furrow scale (farther = larger)
    u = W / 2 - 400
    ridge_pt = g.pixel_to_ground(u, H / 2, "ridge", calib=c)
    p = g.locate(u, H / 2, model, calib=c)
    assert p["plane"] == "furrow"
    assert abs(p["y"]) > abs(ridge_pt[1])
    assert p["y"] == pytest.approx(g.pixel_to_ground(u, H / 2, "furrow", calib=c)[1])


def test_plant_point_is_mask_centroid():
    m = np.zeros((50, 80), bool)
    m[10:20, 30:50] = True
    assert g.plant_point(m) == pytest.approx((39.5, 14.5))
    with pytest.raises(ValueError):
        g.plant_point(np.zeros((5, 5), bool))


# ---------- servo + encoder ----------

def test_target_to_servo_angles():
    c = calib()  # nozzle 15 cm above the ridge top, 23 cm above the furrow
    assert g.target_to_servo(0.0, 0.0, "ridge", c) == pytest.approx((90.0, 90.0))
    pan_ridge, tilt = g.target_to_servo(0.0, 15.0, "ridge", c)  # y = h
    assert pan_ridge == pytest.approx(135.0) and tilt == pytest.approx(90.0)
    pan, tilt = g.target_to_servo(15.0, -15.0, "ridge", c)
    assert pan == pytest.approx(45.0) and tilt == pytest.approx(135.0)
    # furrow: the nozzle is higher above the ground, the same y needs less angle
    pan_f, _ = g.target_to_servo(0.0, 15.0, "furrow", c)
    assert pan_f == pytest.approx(90 + math.degrees(math.atan(15 / 23)))
    assert 90 < pan_f < pan_ridge
    with pytest.raises(OutOfReach) as e:
        g.target_to_servo(0.0, 16.0, "ridge", c)  # just beyond 45 degrees
    assert "pan" in str(e.value) and "45" in str(e.value)
    with pytest.raises(OutOfReach):
        g.target_to_servo(-20.0, 0.0, "ridge", c)  # tilt too far
    g.target_to_servo(0.0, 16.0, "furrow", c)  # reachable from the higher plane
    with pytest.raises(ValueError):
        g.target_to_servo(0, 0, "roof", c)


def test_servo_trim_sign_and_clamp():
    c = Calib(robot=calib().robot, homographies={}, quiet=True,
              servo={"pan": {"center_trim_deg": 3.0, "sign": -1},
                     "tilt": {"center_trim_deg": -2.0, "sign": 1},
                     "max_reach_deg": 60})
    pan, tilt = g.target_to_servo(0.0, 15.0, "ridge", c)
    assert pan == pytest.approx(90 + 3 - 45) and tilt == pytest.approx(88.0)
    far = Calib(robot=calib().robot, homographies={}, quiet=True,
                servo={"pan": {"center_trim_deg": 100.0, "sign": 1}, "max_reach_deg": 45})
    assert g.target_to_servo(0.0, 15.0, "ridge", far)[0] == 180.0  # clamped


def test_fire_tick():
    assert g.fire_tick(12.0, 1000, calib(cm_per_tick=1.0)) == 1012
    assert g.fire_tick(12.0, 1000, calib(cm_per_tick=0.5)) == 1024
    assert g.fire_tick(12.4, 500, calib(cm_per_tick=1.0)) == 512
    assert g.fire_tick(-3.0, 1000, calib()) == 997  # already behind the nozzle


def test_plan_target_spray_and_edge_cases():
    c = calib()
    model = {"band_half_px": 100.0}
    m = np.zeros((H, W), bool)
    m[480:500, 700:720] = True  # near the image center, slightly right of it
    p = g.plan_target(m, 1000, model, calib=c)
    assert p["status"] == "ok" and p["plane"] == "ridge"
    assert p["tilt"] == pytest.approx(90.0)  # spraying: tilt stays at 90
    assert p["fire_tick"] == 1000 + round(p["x"])
    assert p["pan"] < 90  # right of center = negative y = pan below 90
    far = np.zeros((H, W), bool)
    far[480:500, 10:30] = True  # far left edge: 15.7 cm sideways
    # on the ridge the nozzle is only 15 cm above it: beyond 45 degrees
    assert g.plan_target(far, 0, None, calib=c)["status"] == "out_of_reach"
    # the same pixel in the furrow is reachable (furrow ground is lower, so the
    # nozzle is higher above it)
    assert g.plan_target(far, 0, model, calib=c)["plane"] == "furrow"
    assert g.plan_target(far, 0, model, calib=c)["status"] == "ok"
    # bottom of the frame on the ridge is x = D - 12.2 = -0.2 .. 0.3: at the
    # nozzle, not behind it. The lower furrow ground at the edge is behind.
    ridge_bottom = np.zeros((H, W), bool)
    ridge_bottom[950:960, 600:620] = True
    assert g.plan_target(ridge_bottom, 0, model, calib=c)["x"] == pytest.approx(0.25, abs=0.1)
    low = np.zeros((H, W), bool)
    low[950:960, 10:30] = True  # bottom-left corner, furrow: x < 0, already passed
    assert g.plan_target(low, 0, model, calib=c)["status"] == "behind"
    laser = g.plan_target(low, 0, model, mode="laser", calib=c)
    assert laser["status"] in ("ok", "out_of_reach") and laser["tilt"] != 90.0


# ---------- independent reference: OpenCV's own projection and pose solver ----------

def cv_pixels(pts_xy, tilt_deg=0.0, z=0.0, height=25.0):
    """Pixels of ground points from cv2.projectPoints (not our Camera class)."""
    cam = Camera(height, tilt_deg)
    rvec, _ = cv2.Rodrigues(cam.R)
    tvec = (-cam.R @ cam.C).reshape(3, 1)
    K = np.array([[cam.f, 0, W / 2], [0, cam.f, H / 2], [0, 0, 1.0]])
    obj = np.array([[x, y, z] for x, y in pts_xy], np.float64)
    px, _ = cv2.projectPoints(obj, rvec, tvec, K, np.zeros(5))
    return px.reshape(-1, 2), K


def test_matches_opencv_projectpoints_exactly():
    c = calib()
    pts = grid_points(7, 9, 3.0, 3.5)
    for plane, z in (("ridge", 0.0), ("furrow", -8.0)):
        px, _ = cv_pixels(pts, z=z)
        for (x, y), (u, v) in zip(pts, px):
            gx, gy = g.pixel_to_ground(u, v, plane, calib=c)
            assert math.hypot(gx - x, gy - y) < 1e-6


def test_tilt_and_height_agree_with_opencv_solvepnp():
    c = calib()
    paper = [(D + 10.5, 14.85), (D + 10.5, -14.85), (D - 10.5, -14.85), (D - 10.5, 14.85)]
    for tilt in (2.0, 5.0, 8.0):
        corners, K = cv_pixels(paper, tilt_deg=tilt)
        hom = g.compute_homography(corners.tolist(), (W, H), c)
        obj = np.array([[x, y, 0.0] for x, y in paper])
        ok, rv, tv = cv2.solvePnP(obj, corners, K, None, flags=cv2.SOLVEPNP_IPPE)
        assert ok
        Rm, _ = cv2.Rodrigues(rv)
        pnp_tilt = math.degrees(math.acos(abs((Rm.T @ np.array([0, 0, 1.0]))[2])))
        pnp_height = float((-Rm.T @ tv)[2, 0])
        assert hom["tilt_deg"] == pytest.approx(pnp_tilt, abs=0.01)
        assert hom["camera_height_est_cm"] == pytest.approx(abs(pnp_height), abs=0.05)
        # and the calibrated positions are exact
        c2 = Calib(robot=c.robot, servo={}, homographies={"ridge": hom}, quiet=True)
        pts = grid_points(7, 9, 3.0, 3.5)
        px, _ = cv_pixels(pts, tilt_deg=tilt)
        worst = max(math.hypot(g.pixel_to_ground(u, v, "ridge", calib=c2)[0] - x,
                               g.pixel_to_ground(u, v, "ridge", calib=c2)[1] - y)
                    for (x, y), (u, v) in zip(pts, px))
        assert worst < 1e-3, worst


def test_corner_click_noise_tolerance():
    """Clicking the 4 A4 corners 2 px off (1 sigma) must still give < 0.3 cm."""
    c = calib()
    paper = [(D + 10.5, 14.85), (D + 10.5, -14.85), (D - 10.5, -14.85), (D - 10.5, 14.85)]
    pts = grid_points(7, 9, 3.0, 3.5)
    corners, _ = cv_pixels(paper, tilt_deg=4.0)
    px, _ = cv_pixels(pts, tilt_deg=4.0)
    rng = np.random.default_rng(0)
    errs = []
    for _ in range(100):
        hom = g.compute_homography((corners + rng.normal(0, 2.0, corners.shape)).tolist(),
                                   (W, H), c)
        c2 = Calib(robot=c.robot, servo={}, homographies={"ridge": hom}, quiet=True)
        errs += [math.hypot(g.pixel_to_ground(u, v, "ridge", calib=c2)[0] - x,
                            g.pixel_to_ground(u, v, "ridge", calib=c2)[1] - y)
                 for (x, y), (u, v) in zip(pts, px)]
    assert np.percentile(errs, 95) < 0.15 and max(errs) < 0.5


# ---------- the calibration tools ----------

def test_calibration_tools_end_to_end(tmp_path):
    import json

    import calibrate_a4
    import check_points
    import measure_ticks

    robot = dict(calib().robot)
    (tmp_path / "robot.json").write_text(json.dumps(robot))
    cam = Camera(25.0, tilt_deg=4.0)
    paper = [(D + 10.5, 14.85), (D + 10.5, -14.85), (D - 10.5, -14.85), (D - 10.5, 14.85)]
    corners = [cam.project(x, y) for x, y in paper]
    img = np.full((H, W, 3), (60, 90, 60), np.uint8)
    cv2.fillConvexPoly(img, np.array(corners, np.int32), (255, 255, 255))
    photo = tmp_path / "a4.jpg"
    cv2.imwrite(str(photo), img)

    hom = calibrate_a4.calibrate(photo, "ridge", corners, None, tmp_path, show=False)
    assert (tmp_path / "homography_ridge.json").is_file()
    prev = cv2.imread(str(tmp_path / "preview_ridge.jpg"))
    assert prev.shape == img.shape and np.abs(prev.astype(int) - img).sum() > 0  # grid drawn
    assert hom["tilt_deg"] == pytest.approx(4.0, abs=0.5)

    # marks on the sheet, measured from its top-left corner (X right, Y down)
    marks = [(5, 5), (15, 5), (25, 10), (10, 16), (20, 18)]
    world = [(D + 10.5 - Y, 14.85 - X) for X, Y in marks]
    pts = [cam.project(x, y) for x, y in world]
    c = g.Calib(calib_dir=tmp_path, quiet=True)
    res = check_points.check(pts, marks, "ridge", (W, H), c)
    assert res["mean"] < 0.5 and res["max"] < 1.0, res["mean"]
    bad = check_points.check(pts, [(X + 2, Y) for X, Y in marks], "ridge", (W, H), c)
    assert bad["mean"] == pytest.approx(2.0, abs=0.5)  # a wrong ruler reading shows up
    check_points.report(res)

    out = measure_ticks.save_cm_per_tick(100 / 96, tmp_path / "robot.json")
    assert out["cm_per_tick"] == pytest.approx(1.04167, abs=1e-4)
    assert out["measured"]["cm_per_tick"] and out["placeholder"] is True
    assert g.Calib(calib_dir=tmp_path, quiet=True).robot["cm_per_tick"] == out["cm_per_tick"]


# ---------- placeholders ----------

def test_placeholder_warning(capsys, tmp_path):
    (tmp_path / "robot.json").write_text('{"placeholder": true, "measured": {"cm_per_tick": true}}')
    c = Calib(calib_dir=tmp_path)
    c.cm_per_px("ridge")
    err = capsys.readouterr().err
    assert "PLACEHOLDER" in err and "camera_height_cm" in err
    assert "cm_per_tick" not in err.split("PLACEHOLDER")[1].split("\n")[0]
    assert "ESTIMATE" in err
    c.cm_per_px("ridge")  # warned once only
    assert capsys.readouterr().err == ""
    # measured and flagged real: no placeholder warning
    (tmp_path / "robot.json").write_text('{"placeholder": false}')
    Calib(calib_dir=tmp_path)
    assert "PLACEHOLDER" not in capsys.readouterr().err

"""Run: python tests/validate_geometry.py  (writes out/geometry_validation.md)
Independent validation of tools/geometry.py (reference = OpenCV's own
projectPoints / solvePnP and hand trigonometry, not the code under test)."""
import math
import sys

import cv2
import numpy as np

ROOT = __import__("pathlib").Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import geometry as g

W, H, HFOV, D = 1296, 972, 66.0, 12.0
F = (W / 2) / math.tan(math.radians(HFOV) / 2)
K = np.array([[F, 0, W / 2], [0, F, H / 2], [0, 0, 1.0]])
R0 = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]], float)  # straight down, image up = +x, left = +y


def robot(**kw):
    r = {"placeholder": False, "camera_height_cm": 25.0, "ridge_height_cm": 8.0,
         "camera_to_nozzle_cm": D, "camera_side_offset_cm": 0.0, "nozzle_height_cm": 15.0,
         "cm_per_tick": 1.0, "camera_hfov_deg": HFOV, "image_width_px": W, "image_height_px": H}
    r.update(kw)
    return r


def mk(rob=None, homs=None):
    return g.Calib(robot=rob or robot(), servo={}, homographies=homs or {}, quiet=True)


def pose(tilt_deg=0.0, height=25.0, yaw_deg=0.0):
    """OpenCV rvec/tvec; the optical axis hits the ground at (D, 0)."""
    t, y = math.radians(tilt_deg), math.radians(yaw_deg)
    rx = np.array([[1, 0, 0], [0, math.cos(t), -math.sin(t)], [0, math.sin(t), math.cos(t)]])
    rz = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
    R = rz @ rx @ R0
    a = R.T @ np.array([0, 0, 1.0])
    C = np.array([D, 0, 0]) - (height / -a[2]) * a
    rvec, _ = cv2.Rodrigues(R)
    return rvec, (-R @ C).reshape(3, 1), R, C


def project(pts_xy, z=0.0, tilt=0.0, height=25.0, dist=None, yaw=0.0):
    rvec, tvec, *_ = pose(tilt, height, yaw)
    obj = np.array([[x, y, z] for x, y in pts_xy], np.float64)
    px, _ = cv2.projectPoints(obj, rvec, tvec, K, np.array(dist, float) if dist is not None else np.zeros(5))
    return px.reshape(-1, 2)


def dots(rows=7, cols=9, dx=3.0, dy=3.5):
    return [(D + (r - rows // 2) * dx, (c - cols // 2) * dy) for r in range(rows) for c in range(cols)]


def errors(calib, plane, pts_xy, px, z_unused=None):
    out = []
    for (x, y), (u, v) in zip(pts_xy, px):
        gx, gy = g.pixel_to_ground(u, v, plane, None, calib)
        out.append((math.hypot(x - D, y), math.hypot(gx - x, gy - y)))  # (radius from center, error)
    return out


def band_stats(errs):
    res = []
    for lo, hi in ((0, 5), (5, 10), (10, 17)):
        e = [d for r, d in errs if lo <= r < hi]
        res.append((f"{lo}-{hi} cm", np.mean(e) if e else float('nan'), max(e) if e else float('nan')))
    return res


def fmt_bands(errs):
    return "  ".join(f"{b}: mean {m:.2f} max {mx:.2f}" for b, m, mx in band_stats(errs))


def paper_corners(rot_deg=0.0):
    pts = [(10.5, 14.85), (10.5, -14.85), (-10.5, -14.85), (-10.5, 14.85)]
    a = math.radians(rot_deg)
    return [(D + x * math.cos(a) - y * math.sin(a), x * math.sin(a) + y * math.cos(a)) for x, y in pts]


def calibrated(tilt=0.0, height=25.0, dist=None, rot=0.0, ridge_h_assumed=None):
    corners = project(paper_corners(rot), 0.0, tilt, height, dist)
    hom = g.compute_homography(corners.tolist(), (W, H), mk(), None)
    return mk(homs={"ridge": hom}), hom


out = []


def say(s=""):
    print(s)
    out.append(s)


say("# Geometry validation (placeholder robot numbers: camera 25 cm up, FOV 66, nozzle 15 cm up)\n")

# ---------------------------------------------------------------- A. hand math
say("## A. Formulas vs hand calculation")
c = mk()
rows = [
    ("scale, ridge  2*25*tan(33deg)/1296  cm/px", 2 * 25 * math.tan(math.radians(33)) / 1296, c.cm_per_px("ridge")),
    ("scale, furrow 2*33*tan(33deg)/1296  cm/px", 2 * 33 * math.tan(math.radians(33)) / 1296, c.cm_per_px("furrow")),
    ("frame width on the ridge (cm)", 2 * 25 * math.tan(math.radians(33)), c.cm_per_px("ridge") * W),
    ("frame height on the ridge (cm)", 2 * 25 * math.tan(math.radians(33)) * 972 / 1296, c.cm_per_px("ridge") * H),
    ("frame bottom edge x on the ridge (cm)", D - 12.1753, g.pixel_to_ground(W / 2, H, "ridge", calib=c)[0]),
    ("pan for y=7.5 on ridge (h=15): 90+atan(.5)", 90 + math.degrees(math.atan(0.5)), g.target_to_servo(0, 7.5, "ridge", c)[0]),
    ("pan for y=7.5 in furrow (h=23)", 90 + math.degrees(math.atan(7.5 / 23)), g.target_to_servo(0, 7.5, "furrow", c)[0]),
    ("tilt for x=10 on ridge: 90+atan(10/15)", 90 + math.degrees(math.atan(10 / 15)), g.target_to_servo(10, 0, "ridge", c)[1]),
    ("fire tick: 1000 + 12.0/1.0", 1012, g.fire_tick(12.0, 1000, c)),
    ("pixel 100 px left of center -> y (cm)", 100 * 0.0250527, g.pixel_to_ground(W / 2 - 100, H / 2, "ridge", calib=c)[1]),
]
say("| quantity | by hand | code | diff |\n|---|---|---|---|")
for name, hand, code in rows:
    say(f"| {name} | {hand:.4f} | {code:.4f} | {abs(hand - code):.1e} |")

# ---------------------------------------------------- B. OpenCV as the reference
say("\n## B. Independent reference: OpenCV projectPoints / solvePnP")
pts = dots()
for label, tilt, plane, z, height in (("straight down, ridge", 0.0, "ridge", 0.0, 25.0),
                                      ("straight down, furrow (-8 cm)", 0.0, "furrow", -8.0, 25.0)):
    px = project(pts, z, tilt, height)
    e = errors(mk(), plane, pts, px)
    say(f"- {label}, FOV scale only: max error {max(d for _, d in e):.4f} cm over {len(pts)} dots")
for tilt in (2.0, 5.0, 8.0):
    corners = project(paper_corners(), 0.0, tilt)
    hom = g.compute_homography(corners.tolist(), (W, H), mk(), None)
    cal = mk(homs={"ridge": hom})
    px = project(pts, 0.0, tilt)
    e = errors(cal, "ridge", pts, px)
    # solvePnP: pose recovered by OpenCV from the same 4 corners
    obj = np.array([[x, y, 0.0] for x, y in paper_corners()], np.float64)
    ok, rv, tv = cv2.solvePnP(obj, corners, K, None, flags=cv2.SOLVEPNP_IPPE)
    Rm, _ = cv2.Rodrigues(rv)
    cam_c = (-Rm.T @ tv).ravel()
    axis_w = Rm.T @ np.array([0, 0, 1.0])
    pnp_tilt = math.degrees(math.acos(abs(axis_w[2])))
    say(f"- tilt {tilt:.0f} deg + A4 homography: max error {max(d for _, d in e):.4f} cm | "
        f"tilt: code {hom['tilt_deg']:.3f} vs solvePnP {pnp_tilt:.3f} | "
        f"height: code {hom['camera_height_est_cm']:.2f} vs solvePnP {cam_c[2]:.2f} (true {25.0})")

# --------------------------------------------- C. clicking accuracy (Monte Carlo)
say("\n## C. How accurate your corner clicks must be (tilt 4 deg, 400 trials each)")
rng = np.random.default_rng(0)
corners0 = project(paper_corners(), 0.0, 4.0)
px_dots = project(pts, 0.0, 4.0)
say("| click noise (px, 1 sigma) | mean error cm | 95% of dots under cm | worst cm |\n|---|---|---|---|")
for sigma in (0.5, 1, 2, 4, 8):
    allerr = []
    for _ in range(400):
        noisy = corners0 + rng.normal(0, sigma, corners0.shape)
        hom = g.compute_homography(noisy.tolist(), (W, H), mk(), None)
        cal = mk(homs={"ridge": hom})
        allerr += [d for _, d in errors(cal, "ridge", pts, px_dots)]
    a = np.array(allerr)
    say(f"| {sigma} | {a.mean():.3f} | {np.percentile(a, 95):.3f} | {a.max():.3f} |")

# ---------------------------------------- D. wrong numbers / imperfect setup
say("\n## D. What goes wrong with imperfect measurements (error in cm by distance from the image center)")
say("Scale-only = no A4 calibration yet; homography = calibrated on the real setup.\n")
cases = []
# scale only, wrong assumptions
for lab, rob, kw in (
    ("real camera 27 cm up, assumed 25", robot(), dict(height=27.0)),
    ("real camera 23 cm up, assumed 25", robot(), dict(height=23.0)),
    ("real FOV 62, assumed 66", robot(), dict(fov=62.0)),
    ("camera tilted 3 deg, no calibration", robot(), dict(tilt=3.0)),
    ("camera tilted 5 deg, no calibration", robot(), dict(tilt=5.0)),
):
    fov = kw.get("fov", HFOV)
    fx = (W / 2) / math.tan(math.radians(fov) / 2)
    Kx = np.array([[fx, 0, W / 2], [0, fx, H / 2], [0, 0, 1.0]])
    rvec, tvec, *_ = pose(kw.get("tilt", 0.0), kw.get("height", 25.0))
    obj = np.array([[x, y, 0.0] for x, y in pts], np.float64)
    px, _ = cv2.projectPoints(obj, rvec, tvec, Kx, np.zeros(5))
    cases.append((lab, "scale only", errors(mk(), "ridge", pts, px.reshape(-1, 2))))
# homography, imperfections during calibration
for lab, kw in (("A4 sheet rotated 2 deg when photographed", dict(rot=2.0, tilt=0.0)),
                ("A4 sheet rotated 5 deg when photographed", dict(rot=5.0, tilt=0.0)),
                ("lens barrel distortion k1=-0.03 (mild)", dict(dist=[-0.03, 0, 0, 0, 0])),
                ("lens barrel distortion k1=-0.10 (strong)", dict(dist=[-0.10, 0, 0, 0, 0]))):
    cal, _ = calibrated(tilt=kw.get("tilt", 3.0), dist=kw.get("dist"), rot=kw.get("rot", 0.0))
    px = project(pts, 0.0, kw.get("tilt", 3.0), 25.0, kw.get("dist"))
    cases.append((lab, "homography", errors(cal, "ridge", pts, px)))
# furrow plane with a wrong ridge height
for real_rh in (10.0, 6.0):
    px = project(pts, -real_rh, 0.0, 25.0)  # furrow floor is real_rh below the ridge top
    cases.append((f"ridge height really {real_rh:.0f} cm, robot.json says 8 (furrow points)", "scale only",
                  errors(mk(), "furrow", pts, px)))
say("| case | method | error cm by distance from the center (mean / max) |\n|---|---|---|")
for lab, meth, e in cases:
    say(f"| {lab} | {meth} | {fmt_bands(e)} |")

# --------------------------------------------- E. servo + timing consequences
say("\n## E. What a position error costs at the nozzle")
say("| target y (cm) | plane | servo angle per 1 cm of position error |\n|---|---|---|")
for y, plane, h in ((5, "ridge", 15), (10, "ridge", 15), (14, "ridge", 15), (10, "furrow", 23), (20, "furrow", 23)):
    deg = math.degrees(h / (h * h + y * y))
    say(f"| {y} | {plane} | {deg:.2f} deg |")
say("")
say("| nozzle height error | at y = 10 cm on the ridge: target moves by |\n|---|---|")
for dh in (1, 2, 3):
    # servo set for nozzle 15 but really 15+dh: lands at y_real = 15+dh... tan(angle)*real h
    ang = math.atan(10 / 15)
    say(f"| {dh} cm too low assumed ({15}->{15 + dh}) | {math.tan(ang) * (15 + dh) - 10:.2f} cm farther out |")
say("")
say("Reach: ridge +/-%.1f cm (frame half-width %.1f cm), furrow +/-%.1f cm (frame half-width %.1f cm)" % (
    15.0, c.cm_per_px('ridge') * W / 2, 23.0, c.cm_per_px('furrow') * W / 2))
say("Encoder: cm_per_tick = 1.0 -> position rounded to +/-0.5 cm; a 2%% error in cm_per_tick over the 12 cm "
    "look-ahead = %.2f cm." % (0.02 * 12))

open(ROOT / "out" / "geometry_validation.md", "w", encoding="utf-8").write("\n".join(out))

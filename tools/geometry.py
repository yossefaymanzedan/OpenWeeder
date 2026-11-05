"""Step 5: from pixels to centimeters to servo angles and encoder ticks.

Coordinates (nozzle coordinates, all in cm):
    x = along the driving direction, forward positive
    y = across the row, LEFT positive
    origin = the ground point directly under the nozzle at rest

Camera: points straight down, with the TOP of the image toward the direction
of travel and the LEFT of the image toward +y. The ground point under the
image center is camera_to_nozzle_cm ahead of the nozzle (and
camera_side_offset_cm to the left of it).

Two ground planes: "ridge" (the ridge top, where the crop is) and "furrow"
(ridge_height_cm lower). A target's plane changes both its scale in the photo
(the lower ground is farther from the camera) and the aiming angle (the
nozzle is farther above it).

Until the robot exists every measurement in calib/robot.json is a
PLACEHOLDER, and a warning is printed whenever they are used.

    from geometry import locate, target_to_servo, fire_tick
    p = locate(u, v, model, image_size=(w, h))          # x, y, plane
    pan, tilt = target_to_servo(0.0, p["y"], p["plane"])  # spraying: tilt 90
    tick = fire_tick(p["x"], encoder_ticks_at_capture)
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CALIB_DIR = ROOT / "calib"
A4_CM = (21.0, 29.7)  # (short, long)
PLANES = ("ridge", "furrow")

DEFAULT_ROBOT = {
    "placeholder": True, "measured": {},
    "camera_height_cm": 25.0, "ridge_height_cm": 8.0,
    "camera_to_nozzle_cm": 12.0, "camera_side_offset_cm": 0.0,
    "nozzle_height_cm": 15.0, "cm_per_tick": 1.0,
    "camera_hfov_deg": 66.0, "image_width_px": 1296, "image_height_px": 972,
    # wiper arm (step 6): pivot -> nozzle length, splash radius on the ground,
    # extra clearance around crops, and the pump's flow
    "arm_length_cm": 25.0, "spray_radius_cm": 1.5, "protect_margin_cm": 0.5,
    "pump_flow_ml_per_s": 2.0,
}
ROBOT_KEYS = ("camera_height_cm", "ridge_height_cm", "camera_to_nozzle_cm",
              "camera_side_offset_cm", "nozzle_height_cm", "cm_per_tick",
              "camera_hfov_deg", "image_width_px", "image_height_px",
              "arm_length_cm", "spray_radius_cm", "protect_margin_cm",
              "pump_flow_ml_per_s")
DEFAULT_SERVO = {"pan": {"center_trim_deg": 0.0, "sign": 1},
                 "tilt": {"center_trim_deg": 0.0, "sign": 1},
                 # wiper arm: angle phi from straight ahead, + = left
                 "arm": {"center_trim_deg": 0.0, "sign": 1, "max_deg": 80.0,
                         "park_deg": 70.0},
                 "max_reach_deg": 45.0}


class OutOfReach(Exception):
    """The target needs a nozzle angle beyond the allowed reach."""

    def __init__(self, axis: str, offset_deg: float, max_deg: float,
                 x_cm: float, y_cm: float, plane: str):
        self.axis, self.offset_deg, self.max_deg = axis, offset_deg, max_deg
        super().__init__(
            f"target (x={x_cm:.1f}, y={y_cm:.1f} cm, {plane}) needs the {axis} "
            f"servo {offset_deg:+.1f} deg off center; the limit is "
            f"{max_deg:.0f} deg. Skip it or let the robot get closer.")


class Calib:
    """Robot measurements, servo trims and per-plane A4 calibrations.
    Everything is read from calib/ unless passed in (tests do that)."""

    def __init__(self, robot: dict | None = None, servo: dict | None = None,
                 homographies: dict | None = None, calib_dir: Path = CALIB_DIR,
                 quiet: bool = False):
        self.dir = Path(calib_dir)
        self.quiet = quiet
        self._warned: set[str] = set()
        if robot is None:
            robot = _read_json(self.dir / "robot.json")
        self.robot = {**DEFAULT_ROBOT, **robot}
        if servo is None:
            servo = _read_json(self.dir / "servo.json")
        self.servo = {**DEFAULT_SERVO, **servo}
        for ax in ("pan", "tilt", "arm"):
            self.servo[ax] = {**DEFAULT_SERVO[ax], **self.servo.get(ax, {})}
        self._homs = dict(homographies) if homographies is not None else None
        if self.robot.get("placeholder", True):
            todo = [k for k in ROBOT_KEYS if not self.robot.get("measured", {}).get(k)]
            self.warn("placeholder", "calib/robot.json holds PLACEHOLDER values (not "
                      "measured on the robot yet): " + ", ".join(todo))

    def warn(self, key: str, msg: str) -> None:
        if key not in self._warned:
            self._warned.add(key)
            if not self.quiet:
                print(f"WARNING (geometry): {msg}", file=sys.stderr)

    def hom(self, plane: str) -> dict | None:
        """The saved A4 calibration of a plane, or None."""
        if plane not in PLANES:
            raise ValueError(f"plane must be one of {PLANES}, not {plane!r}")
        if self._homs is None:
            self._homs = {}
            for p in PLANES:
                path = self.dir / f"homography_{p}.json"
                if path.is_file():
                    self._homs[p] = _read_json(path)
        return self._homs.get(plane)

    def calibrated_planes(self) -> list[str]:
        return [p for p in PLANES if self.hom(p)]

    def cal_size(self, plane: str) -> tuple[float, float]:
        """Resolution the calibration (or the robot.json estimate) refers to."""
        h = self.hom(plane)
        if h:
            return float(h["image_size"][0]), float(h["image_size"][1])
        return float(self.robot["image_width_px"]), float(self.robot["image_height_px"])

    def camera_height(self, plane: str) -> float:
        r = self.robot
        return r["camera_height_cm"] + (r["ridge_height_cm"] if plane == "furrow" else 0.0)

    def cm_per_px(self, plane: str) -> float:
        h = self.hom(plane)
        if h:
            return float(h["cm_per_px"])
        r = self.robot
        self.warn(f"scale-{plane}", f"cm per pixel for the {plane} plane is an ESTIMATE "
                  "from the camera FOV and height (no A4 calibration found: run "
                  f"tools/calibrate_a4.py <photo> --plane {plane}).")
        w = float(r["image_width_px"])
        return 2 * self.camera_height(plane) * math.tan(math.radians(r["camera_hfov_deg"]) / 2) / w


_DEFAULT: Calib | None = None


def default_calib() -> Calib:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = Calib()
    return _DEFAULT


def reset_default() -> None:
    global _DEFAULT
    _DEFAULT = None


def _read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _apply_h(H, u: float, v: float) -> tuple[float, float]:
    p = np.asarray(H, float) @ np.array([u, v, 1.0])
    return float(p[0] / p[2]), float(p[1] / p[2])


# ---------- pixels <-> ground ----------

def pixel_to_ground(u: float, v: float, plane: str = "ridge",
                    image_size: tuple | None = None,
                    calib: Calib | None = None) -> tuple[float, float]:
    """Pixel (u right, v down) -> (x_cm, y_cm) in nozzle coordinates, for a
    point on `plane`. image_size=(w, h) of the photo the pixel came from; it
    is rescaled to the resolution the calibration refers to."""
    c = calib or default_calib()
    cw, ch = c.cal_size(plane)
    if image_size is not None:
        u, v = u * cw / image_size[0], v * ch / image_size[1]
    D, side = c.robot["camera_to_nozzle_cm"], c.robot["camera_side_offset_cm"]
    hom = c.hom(plane)
    if hom:  # full homography: handles a slightly tilted mount
        Xp, Yp = _apply_h(hom["H_px2paper"], u, v)
        Xc, Yc = hom["center_paper"]
    else:  # straight-down scale
        s = c.cm_per_px(plane)
        Xp, Yp = (u - cw / 2) * s, (v - ch / 2) * s
        Xc = Yc = 0.0
    return D + (Yc - Yp), side + (Xc - Xp)


def ground_to_pixel(x_cm: float, y_cm: float, plane: str = "ridge",
                    image_size: tuple | None = None,
                    calib: Calib | None = None) -> tuple[float, float]:
    """Inverse of pixel_to_ground."""
    c = calib or default_calib()
    cw, ch = c.cal_size(plane)
    D, side = c.robot["camera_to_nozzle_cm"], c.robot["camera_side_offset_cm"]
    hom = c.hom(plane)
    if hom:
        Xc, Yc = hom["center_paper"]
        Xp, Yp = Xc - (y_cm - side), Yc - (x_cm - D)
        u, v = _apply_h(np.linalg.inv(np.asarray(hom["H_px2paper"], float)), Xp, Yp)
    else:
        s = c.cm_per_px(plane)
        u = cw / 2 - (y_cm - side) / s
        v = ch / 2 - (x_cm - D) / s
    if image_size is not None:
        u, v = u * image_size[0] / cw, v * image_size[1] / ch
    return u, v


def ground_to_paper(x_cm: float, y_cm: float, plane: str,
                    calib: Calib | None = None) -> tuple[float, float]:
    """Nozzle coordinates -> position on the calibration A4 (cm from its
    top-left corner, X right, Y down). Used by tools/check_points.py."""
    c = calib or default_calib()
    hom = c.hom(plane)
    if not hom:
        raise ValueError(f"no A4 calibration for the {plane} plane")
    D, side = c.robot["camera_to_nozzle_cm"], c.robot["camera_side_offset_cm"]
    Xc, Yc = hom["center_paper"]
    return Xc - (y_cm - side), Yc - (x_cm - D)


def plant_point(mask: np.ndarray) -> tuple[float, float]:
    """Mask centroid (u, v). The camera looks straight down, so the center of
    a rosette is where the stem is."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise ValueError("empty mask")
    return float(xs.mean()), float(ys.mean())


def _band_half_px(model: dict | None) -> float | None:
    if not model:
        return None
    if model.get("band_half_px"):
        return float(model["band_half_px"])
    return (model.get("band") or {}).get("half_width_px")


def plane_for(x_cm: float, y_cm: float, model: dict | None = None,
              row_y_cm: float | None = None, calib: Calib | None = None) -> str:
    """'ridge' if the point is inside the field model's crop band, else
    'furrow'. With only one plane calibrated, that plane. Without a band in
    the model, 'ridge'. The band is centered on row_y_cm (the row's lateral
    position at that x; default: model['row_y_cm'] or 0)."""
    c = calib or default_calib()
    cal = c.calibrated_planes()
    if len(cal) == 1:
        return cal[0]
    half_cm = (model or {}).get("band_half_cm")
    if half_cm is None:
        px = _band_half_px(model)
        if px is None:
            return "ridge"
        half_cm = px * c.cm_per_px("ridge")
    center = row_y_cm if row_y_cm is not None else (model or {}).get("row_y_cm", 0.0)
    return "ridge" if abs(y_cm - center) <= half_cm else "furrow"


def locate(u: float, v: float, model: dict | None = None,
           image_size: tuple | None = None, row_y_cm: float | None = None,
           calib: Calib | None = None) -> dict:
    """Pixel -> {x, y, plane}. Which plane a point is on depends on where it
    is, and where it is depends on the plane (the furrow is farther from the
    camera), so: place the point on the ridge plane, pick the plane from
    that position, then place it again on the chosen plane."""
    x, y = pixel_to_ground(u, v, "ridge", image_size, calib)
    plane = plane_for(x, y, model, row_y_cm, calib)
    if plane != "ridge":
        x, y = pixel_to_ground(u, v, plane, image_size, calib)
    return {"x": x, "y": y, "plane": plane}


# ---------- servo + encoder ----------

def target_to_servo(x_cm: float, y_cm: float, plane: str = "ridge",
                    calib: Calib | None = None) -> tuple[float, float]:
    """(pan_deg, tilt_deg) that point the nozzle at (x_cm, y_cm) on `plane`.
    h = nozzle height above that plane; pan = 90 + atan(y/h),
    tilt = 90 + atan(x/h), then calib/servo.json (center trim, direction sign).
    Spraying fires when the target is under the nozzle (x = 0), so tilt = 90
    and only pan moves. Raises OutOfReach beyond max_reach_deg off center."""
    c = calib or default_calib()
    if plane not in PLANES:
        raise ValueError(f"plane must be one of {PLANES}, not {plane!r}")
    h = c.robot["nozzle_height_cm"] + (c.robot["ridge_height_cm"] if plane == "furrow" else 0.0)
    limit = float(c.servo["max_reach_deg"])
    out = []
    for axis, dist in (("pan", y_cm), ("tilt", x_cm)):
        off = math.degrees(math.atan2(dist, h))
        if abs(off) > limit + 1e-9:
            raise OutOfReach(axis, off, limit, x_cm, y_cm, plane)
        sv = c.servo[axis]
        angle = 90.0 + float(sv["center_trim_deg"]) + float(sv["sign"]) * off
        out.append(min(180.0, max(0.0, angle)))
    return out[0], out[1]


class ArmOutOfReach(OutOfReach):
    """The weed is farther across than the wiper arm can swing."""

    def __init__(self, y_cm: float, reach_cm: float):
        Exception.__init__(
            self, f"weed {abs(y_cm):.1f} cm to the side is beyond the arm's reach "
                  f"({reach_cm:.1f} cm). Skip it.")
        self.axis, self.offset_deg, self.max_deg = "arm", float("nan"), float("nan")


def wiper_arm(y_cm: float, calib: Calib | None = None) -> tuple[float, float]:
    """Wiper arm of length L pivoting at the robot (the geometry origin).
    The nozzle at the arm's tip is at (L cos(phi), L sin(phi)) from the pivot,
    phi measured from straight ahead, positive to the left. To hit a weed
    y_cm to the side: phi = asin(y / L), and the pivot must stop
    L cos(phi) BEHIND the weed. Returns (phi_deg, forward_offset_cm).
    Raises ArmOutOfReach beyond the arm's max swing."""
    c = calib or default_calib()
    L = float(c.robot["arm_length_cm"])
    reach = L * math.sin(math.radians(float(c.servo["arm"]["max_deg"])))
    if abs(y_cm) > reach + 1e-9:
        raise ArmOutOfReach(y_cm, reach)
    phi = math.degrees(math.asin(y_cm / L))
    return phi, L * math.cos(math.radians(phi))


def arm_to_servo(phi_deg: float, calib: Calib | None = None) -> float:
    """Servo angle for arm angle phi (90 + trim + sign * phi, clamped 0-180)."""
    sv = (calib or default_calib()).servo["arm"]
    return min(180.0, max(0.0, 90.0 + float(sv["center_trim_deg"]) + float(sv["sign"]) * phi_deg))


def servo_to_arm(angle_deg: float, calib: Calib | None = None) -> float:
    """Inverse of arm_to_servo: the arm angle a servo angle really gives."""
    sv = (calib or default_calib()).servo["arm"]
    return (angle_deg - 90.0 - float(sv["center_trim_deg"])) / float(sv["sign"])


def arm_landing(phi_deg: float, calib: Calib | None = None) -> tuple[float, float]:
    """Where the nozzle is on the ground for arm angle phi: (forward, y) cm
    from the pivot. The inverse of wiper_arm; used to score sprays."""
    L = float((calib or default_calib()).robot["arm_length_cm"])
    return L * math.cos(math.radians(phi_deg)), L * math.sin(math.radians(phi_deg))


def fire_tick(x_cm_at_capture: float, tick_at_capture: int,
              calib: Calib | None = None) -> int:
    """Encoder tick at which the target is directly under the nozzle.
    (Then aim with tilt = 90.) A negative x means it is already behind."""
    c = calib or default_calib()
    return int(tick_at_capture + round(x_cm_at_capture / c.robot["cm_per_tick"]))


def plan_target(mask: np.ndarray, tick_at_capture: int, model: dict | None = None,
                image_size: tuple | None = None, mode: str = "spray",
                row_y_cm: float | None = None, calib: Calib | None = None) -> dict:
    """One weed mask -> {u, v, x, y, plane, fire_tick, pan, tilt, status}.
    status: 'ok', 'behind' (already passed the nozzle) or 'out_of_reach'.
    mode 'spray': aim at x = 0 (tilt 90) and fire at fire_tick.
    mode 'laser': aim at the current (x, y), tilt follows."""
    u, v = plant_point(mask)
    p = locate(u, v, model, image_size, row_y_cm, calib)
    out = {"u": u, "v": v, **p, "fire_tick": fire_tick(p["x"], tick_at_capture, calib),
           "pan": None, "tilt": None, "status": "ok"}
    if mode == "spray" and p["x"] < 0:
        out["status"] = "behind"
        return out
    try:
        out["pan"], out["tilt"] = target_to_servo(
            0.0 if mode == "spray" else p["x"], p["y"], p["plane"], calib)
    except OutOfReach as e:
        out["status"] = "out_of_reach"
        out["reason"] = str(e)
    return out


# ---------- A4 calibration (used by tools/calibrate_a4.py) ----------

def paper_size_for(corners_px, landscape: bool | None = None) -> tuple[float, float]:
    """(width, height) of the A4 as seen: long side across the photo
    (landscape) or along it (portrait). Auto: the longer edge pair wins."""
    c = np.asarray(corners_px, float)
    top, bot = np.linalg.norm(c[1] - c[0]), np.linalg.norm(c[2] - c[3])
    left, right = np.linalg.norm(c[3] - c[0]), np.linalg.norm(c[2] - c[1])
    if landscape is None:
        landscape = (top + bot) > (left + right)
    return (A4_CM[1], A4_CM[0]) if landscape else (A4_CM[0], A4_CM[1])


def compute_homography(corners_px, image_size: tuple, calib: Calib | None = None,
                       landscape: bool | None = None) -> dict:
    """The 4 A4 corners (top-left, top-right, bottom-right, bottom-left, in
    the photo) -> homography, cm per pixel, and how far the camera is from
    straight down. The A4 edges must be parallel to the photo edges."""
    c = calib or default_calib()
    w, h = paper_size_for(corners_px, landscape)
    src = np.asarray(corners_px, np.float32)
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
    H = cv2.getPerspectiveTransform(src, dst)
    W_, H_ = image_size
    cx, cy = W_ / 2, H_ / 2
    center = _apply_h(H, cx, cy)
    # cm per pixel at the image center: sqrt of the local area scale
    px = np.array(_apply_h(H, cx + 1, cy)) - np.array(_apply_h(H, cx - 1, cy))
    py = np.array(_apply_h(H, cx, cy + 1)) - np.array(_apply_h(H, cx, cy - 1))
    cm_per_px = math.sqrt(abs(px[0] * py[1] - px[1] * py[0]) / 4.0)
    # camera tilt from the plane's normal in camera coordinates
    f = (W_ / 2) / math.tan(math.radians(c.robot["camera_hfov_deg"]) / 2)
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
    M = np.linalg.inv(K) @ np.linalg.inv(H)
    lam = 1.0 / np.linalg.norm(M[:, 0])
    r1, r2, t = lam * M[:, 0], lam * M[:, 1], lam * M[:, 2]
    n = np.cross(r1, r2)
    n /= np.linalg.norm(n)
    tilt = math.degrees(math.acos(min(1.0, abs(n[2]))))
    return {"H_px2paper": H.tolist(), "image_size": [W_, H_],
            "center_paper": list(center), "cm_per_px": cm_per_px,
            "tilt_deg": tilt, "camera_height_est_cm": float(abs(t @ n)),
            "paper_cm": [w, h], "corners_px": np.asarray(corners_px, float).tolist()}


def save_homography(plane: str, hom: dict, calib_dir: Path = CALIB_DIR) -> Path:
    calib_dir = Path(calib_dir)
    calib_dir.mkdir(parents=True, exist_ok=True)
    path = calib_dir / f"homography_{plane}.json"
    path.write_text(json.dumps({"plane": plane, **hom}, indent=1), encoding="utf-8")
    return path

"""Calibrate the camera's scale from the tray itself: its width is known.

The camera's field of view is what turns pixels into centimetres when there is no
A4 calibration (geometry.Calib.cm_per_px). A phone's field of view depends on its
zoom, so the placeholder is usually far off. The tray finder (tray.py) measures
the tray's width in pixels in every photo; with the real width in cm this gives
the true scale, and from it the field of view, written to calib/robot.json.

    python tools/calibrate_tray.py --crop Molokhia --width-cm 6
    python tools/calibrate_tray.py photo1.jpg photo2.jpg --width-cm 6

With --crop it uses the full photos the robot recorded (fields/<crop>/train/raw).
Run it again whenever the phone's zoom or height changes.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
import geometry as g  # noqa: E402
import tray  # noqa: E402


def tray_widths(photos: list[Path], image_width: int) -> list[float]:
    """The tray's width in pixels (scaled to image_width) in each photo where it is found."""
    tracker = tray.TrayTracker()
    widths = []
    for path in photos:
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        found = tracker.update(frame)
        if found is not None:
            widths.append(found.tray_width * image_width / frame.shape[1])
    return widths


def hfov_for(width_px: float, width_cm: float, height_cm: float, image_width: int) -> float:
    """The horizontal field of view (degrees) at which width_px pixels are width_cm
    on a plane height_cm below the camera (geometry.Calib.cm_per_px, inverted)."""
    cm_per_px = width_cm / width_px
    return math.degrees(2 * math.atan(cm_per_px * image_width / (2 * height_cm)))


def save(hfov: float, robot_path: Path, flat: bool) -> dict:
    robot = json.loads(robot_path.read_text(encoding="utf-8"))
    robot["camera_hfov_deg"] = round(hfov, 2)
    robot.setdefault("measured", {})["camera_hfov_deg"] = True
    if flat:  # the field is flat: plants and weeds are on one plane
        robot["ridge_height_cm"] = 0.0
        robot["measured"]["ridge_height_cm"] = True
    robot["placeholder"] = not all(robot["measured"].get(k) for k in g.ROBOT_KEYS)
    robot_path.write_text(json.dumps(robot, indent=2), encoding="utf-8")
    return robot


def main() -> None:
    ap = argparse.ArgumentParser(description="Set the camera's field of view from the tray's known width.")
    ap.add_argument("photos", nargs="*", type=Path, help="full photos from the robot's camera")
    ap.add_argument("--crop", help="use the full photos recorded for this crop (fields/<crop>/train/raw)")
    ap.add_argument("--width-cm", type=float, required=True, help="the tray's real width, wall to wall")
    ap.add_argument("--not-flat", action="store_true", help="keep ridge_height_cm (the field has a raised bed)")
    ap.add_argument("--calib", type=Path, default=Path("calib"))
    args = ap.parse_args()

    photos = list(args.photos)
    if args.crop:
        photos += sorted((Path("fields") / args.crop / "train" / "raw").glob("*.jpg"))
    if not photos:
        sys.exit("No photos: give photos or --crop")
    calib = g.Calib(calib_dir=args.calib, quiet=True)
    image_width = int(calib.robot["image_width_px"])
    widths = tray_widths(photos, image_width)
    if len(widths) < 3:
        sys.exit(f"The tray was found in only {len(widths)} of {len(photos)} photos: not enough to calibrate")
    width_px = statistics.median(widths)
    height = float(calib.robot["camera_height_cm"])
    old = float(calib.robot["camera_hfov_deg"])
    hfov = hfov_for(width_px, args.width_cm, height, image_width)
    robot = save(hfov, args.calib / "robot.json", flat=not args.not_flat)
    g._DEFAULT = None
    new = g.Calib(calib_dir=args.calib, quiet=True)
    w, h = image_width, int(new.robot["image_height_px"])
    picture_w = new.cm_per_px("ridge") * w
    picture_h = new.cm_per_px("ridge") * h
    print(f"tray found in {len(widths)} of {len(photos)} photos: {width_px:.0f} px wide "
          f"(spread {min(widths):.0f}-{max(widths):.0f}) = {args.width_cm:g} cm")
    print(f"camera height {height:g} cm -> field of view {hfov:.1f} deg (was {old:g})")
    print(f"one photo now covers {picture_w:.1f} x {picture_h:.1f} cm of ground "
          f"({1 / new.cm_per_px('ridge'):.0f} px per cm)")
    if robot.get("ridge_height_cm") == 0.0:
        print("ridge_height_cm set to 0: the field is flat")


if __name__ == "__main__":
    main()

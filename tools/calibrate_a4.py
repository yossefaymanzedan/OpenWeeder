"""Calibrate one ground plane from a photo of an A4 sheet.

    python tools/calibrate_a4.py photo.jpg --plane ridge
    python tools/calibrate_a4.py photo.jpg --plane furrow

How to take the photo (on the robot, camera mounted as it will run):
  - Lay a flat A4 sheet on the plane: ON the ridge top for --plane ridge,
    at the bottom of the furrow for --plane furrow.
  - Keep its edges parallel and square to the RAILS (the driving direction),
    long side across the rails: A4 is 29.7 cm long and the photo is only
    ~32 cm wide. The tool treats the sheet's edges as the robot's x and y
    axes, so a sheet turned by 2 degrees turns every position by 2 degrees
    (about 0.5 cm at 15 cm from the center). Aligned to the rails rather than
    to the photo edges, it also absorbs a camera that is mounted slightly
    rotated.
  - The whole sheet in view, corners clear.

Then click its 4 corners IN ORDER: top-left, top-right, bottom-right,
bottom-left (as you see them in the photo). Keys: u = undo, Enter = accept,
Esc = quit. The tool prints the scale (cm per pixel), how far the camera is
from straight down, and saves

    calib/homography_<plane>.json
    calib/preview_<plane>.jpg     a 1 cm grid drawn on the ground: check by
                                  eye that it fits the sheet and the plants

No window (or scripting)? Pass the corners: --corners "x,y x,y x,y x,y"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import geometry as g  # noqa: E402

NAMES = ["top-left", "top-right", "bottom-right", "bottom-left"]


def click_points(img: np.ndarray, names: list[str], title: str,
                 max_w: int = 1400, max_h: int = 850) -> list[tuple[float, float]]:
    """Show the photo (scaled to fit the screen), let the user click
    len(names) points, return them in PHOTO pixels. u = undo, Enter = accept,
    Esc = quit."""
    h, w = img.shape[:2]
    s = min(1.0, max_w / w, max_h / h)
    shown = cv2.resize(img, (int(w * s), int(h * s))) if s < 1 else img.copy()
    pts: list[tuple[int, int]] = []

    def draw() -> None:
        view = shown.copy()
        for i, (x, y) in enumerate(pts):
            cv2.circle(view, (x, y), 6, (0, 0, 255), -1)
            cv2.putText(view, f"{i + 1} {names[i]}", (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        if len(pts) > 1:
            cv2.polylines(view, [np.array(pts, np.int32)], len(pts) == len(names),
                          (0, 200, 255), 1)
        msg = (f"click {names[len(pts)]} corner" if len(pts) < len(names)
               else "Enter = accept, u = undo, Esc = quit")
        cv2.rectangle(view, (0, 0), (view.shape[1], 30), (0, 0, 0), -1)
        cv2.putText(view, msg, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1)
        cv2.imshow(title, view)

    def on_mouse(event, x, y, _f, _p) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < len(names):
            pts.append((x, y))
            draw()

    cv2.namedWindow(title)
    cv2.setMouseCallback(title, on_mouse)
    draw()
    while True:
        k = cv2.waitKey(50) & 0xFF
        if k == 27:
            cv2.destroyAllWindows()
            raise SystemExit("cancelled")
        if k == ord("u") and pts:
            pts.pop()
            draw()
        if k in (13, 10) and len(pts) == len(names):
            break
    cv2.destroyAllWindows()
    return [(x / s, y / s) for x, y in pts]


def parse_corners(text: str) -> list[tuple[float, float]]:
    pts = [tuple(float(v) for v in p.split(",")) for p in text.split()]
    if len(pts) != 4 or any(len(p) != 2 for p in pts):
        raise SystemExit('--corners needs 4 points: "x,y x,y x,y x,y"')
    return pts


def draw_grid_preview(img: np.ndarray, plane: str, calib: g.Calib,
                      corners: list[tuple[float, float]]) -> np.ndarray:
    """Draw a 1 cm grid on the ground (in A4-sheet coordinates), heavier every
    5 cm, plus the sheet's outline, using the calibration just computed."""
    hom = calib.hom(plane)
    H = np.asarray(hom["H_px2paper"], float)
    Hinv = np.linalg.inv(H)
    h, w = img.shape[:2]
    out = img.copy()
    # ground range covered by the photo (paper cm), and a little more
    ground = np.array([g._apply_h(H, x, y) for x, y in
                       [(0, 0), (w, 0), (w, h), (0, h)]])
    x0, y0 = np.floor(ground.min(axis=0)) - 1
    x1, y1 = np.ceil(ground.max(axis=0)) + 1
    x0, x1, y0, y1 = (int(v) for v in (max(x0, -60), min(x1, 90), max(y0, -60), min(y1, 90)))

    def line(a, b, color, thick) -> None:
        pa = tuple(int(round(v)) for v in g._apply_h(Hinv, *a))
        pb = tuple(int(round(v)) for v in g._apply_h(Hinv, *b))
        ok, p1, p2 = cv2.clipLine((0, 0, w, h), pa, pb)
        if ok:
            cv2.line(out, p1, p2, color, thick, cv2.LINE_AA)

    for X in range(x0, x1 + 1):
        major = X % 5 == 0
        line((X, y0), (X, y1), (255, 255, 0) if major else (200, 200, 0), 2 if major else 1)
    for Y in range(y0, y1 + 1):
        major = Y % 5 == 0
        line((x0, Y), (x1, Y), (255, 255, 0) if major else (200, 200, 0), 2 if major else 1)
    pw, ph = hom["paper_cm"]
    for a, b in (((0, 0), (pw, 0)), ((pw, 0), (pw, ph)), ((pw, ph), (0, ph)), ((0, ph), (0, 0))):
        line(a, b, (0, 0, 255), 3)
    cv2.circle(out, (w // 2, h // 2), 8, (0, 255, 0), 2)  # the optical axis
    cv2.putText(out, "red = A4 sheet, cyan = 1 cm grid (thick every 5 cm), green = image center",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
    cv2.putText(out, "red = A4 sheet, cyan = 1 cm grid (thick every 5 cm), green = image center",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return out


def calibrate(photo: Path, plane: str, corners: list[tuple[float, float]] | None = None,
              landscape: bool | None = None, calib_dir: Path = g.CALIB_DIR,
              show: bool = True) -> dict:
    img = cv2.imread(str(photo))
    if img is None:
        raise SystemExit(f"cannot read {photo}")
    h, w = img.shape[:2]
    c = g.Calib(calib_dir=calib_dir)
    exp = (c.robot["image_width_px"], c.robot["image_height_px"])
    if (w, h) != tuple(exp):
        print(f"note: the photo is {w}x{h} but robot.json says {exp[0]}x{exp[1]}. "
              "The calibration is saved for this photo's size and other sizes are "
              "rescaled; use the camera resolution you will really run at.")
    if corners is None:
        corners = click_points(img, NAMES, f"A4 corners ({plane})")
    hom = g.compute_homography(corners, (w, h), c, landscape)
    path = g.save_homography(plane, hom, calib_dir)
    c2 = g.Calib(calib_dir=calib_dir, quiet=True)
    prev = draw_grid_preview(img, plane, c2, corners)
    prev_path = Path(calib_dir) / f"preview_{plane}.jpg"
    cv2.imwrite(str(prev_path), prev)
    pw, ph = hom["paper_cm"]
    print(f"plane: {plane}   sheet seen as {pw:.1f} x {ph:.1f} cm "
          f"({'landscape' if pw > ph else 'portrait'})")
    print(f"scale at the image center: {hom['cm_per_px']:.4f} cm per pixel "
          f"(image {w / 1 * hom['cm_per_px']:.1f} cm wide)")
    print(f"camera tilt from straight down: {hom['tilt_deg']:.1f} degrees")
    want = c.camera_height(plane)
    print(f"camera height estimated from the sheet: {hom['camera_height_est_cm']:.1f} cm "
          f"(robot.json: {want:.1f} cm for the {plane} plane)")
    if hom["tilt_deg"] > 10:
        print("WARNING: more than 10 degrees from straight down; re-mount the camera "
              "or accept that near the photo edges the error grows.")
    if abs(hom["camera_height_est_cm"] - want) > 0.15 * want:
        print("WARNING: the estimated height differs from robot.json by more than 15%. "
              "Either the real height differs from the placeholder (measure it), the "
              "FOV is wrong, or a corner was clicked in the wrong place.")
    print(f"saved {path}\nsaved {prev_path}  <- open it: the grid must sit on the sheet's "
          "edges and look straight along the ridge")
    if show:
        try:
            cv2.imshow("preview (any key closes)", cv2.resize(prev, None, fx=min(1, 1400 / w), fy=min(1, 1400 / w)))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except cv2.error:
            pass
    return hom


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("photo", type=Path)
    ap.add_argument("--plane", choices=g.PLANES, required=True)
    ap.add_argument("--corners", help='"x,y x,y x,y x,y" in photo pixels: skip clicking')
    ap.add_argument("--landscape", dest="landscape", action="store_true", default=None,
                    help="the long side of the sheet runs across the photo")
    ap.add_argument("--portrait", dest="landscape", action="store_false",
                    help="the long side runs up the photo (default: auto)")
    ap.add_argument("--no-window", action="store_true", help="do not show the preview")
    ap.add_argument("--calib-dir", type=Path, default=g.CALIB_DIR,
                    help="where robot.json is read and the calibration saved "
                         "(default calib/; use another folder to practice)")
    a = ap.parse_args()
    calibrate(a.photo, a.plane, parse_corners(a.corners) if a.corners else None,
              a.landscape, a.calib_dir, show=not a.no_window)


if __name__ == "__main__":
    main()

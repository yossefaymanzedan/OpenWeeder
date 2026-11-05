"""Check a calibration with points you marked on the A4 sheet.

    python tools/check_points.py photo.jpg --plane ridge

Use the SAME photo setup as the calibration (or a new photo of the sheet with
extra marks). Draw 5 small dots on the A4 with a pen and measure where they
are with a ruler: X = cm from the sheet's LEFT edge, Y = cm from its TOP edge
(as you see the sheet in the photo). Click each dot in the photo, then type
its real position. The tool predicts each point with pixel_to_ground() (the
same function the robot uses), converts it back to sheet coordinates, and
prints the error. Target: about 1 cm or less on average.

No window (or scripting)? --points "u,v u,v ..." --real "X,Y X,Y ..."
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
import geometry as g  # noqa: E402
from calibrate_a4 import click_points  # noqa: E402

TARGET_CM = 1.0


def check(points_px, real_cm, plane: str, image_size, calib: g.Calib | None = None) -> dict:
    """Per-point error between where the calibration puts each point and
    where you measured it. Both are in sheet coordinates (cm)."""
    c = calib or g.default_calib()
    rows = []
    for i, ((u, v), (X, Y)) in enumerate(zip(points_px, real_cm), 1):
        x, y = g.pixel_to_ground(u, v, plane, image_size, c)
        Xp, Yp = g.ground_to_paper(x, y, plane, c)
        rows.append({"n": i, "u": u, "v": v, "real": (X, Y), "pred": (Xp, Yp),
                     "err": math.hypot(Xp - X, Yp - Y), "ground": (x, y)})
    errs = [r["err"] for r in rows]
    return {"rows": rows, "mean": sum(errs) / len(errs), "max": max(errs)}


def report(res: dict) -> None:
    print(f"{'#':>2} {'real X,Y (cm)':>15} {'predicted':>15} {'error cm':>9}   nozzle x,y (cm)")
    for r in res["rows"]:
        print(f"{r['n']:>2} {r['real'][0]:>7.1f},{r['real'][1]:>6.1f} "
              f"{r['pred'][0]:>7.1f},{r['pred'][1]:>6.1f} {r['err']:>9.2f}   "
              f"{r['ground'][0]:>6.1f}, {r['ground'][1]:>6.1f}")
    verdict = "GOOD" if res["mean"] <= TARGET_CM else "TOO BIG"
    print(f"average error {res['mean']:.2f} cm, worst {res['max']:.2f} cm "
          f"(target: about {TARGET_CM:.0f} cm or less): {verdict}")
    if res["mean"] > TARGET_CM:
        print("Try again: make sure the sheet lay flat, its edges are parallel to the "
              "photo edges, and the corners were clicked exactly. Remeasure the dots.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("photo", type=Path)
    ap.add_argument("--plane", choices=g.PLANES, required=True)
    ap.add_argument("--points", help='"u,v u,v ..." photo pixels (skip clicking)')
    ap.add_argument("--real", help='"X,Y X,Y ..." real cm from the sheet top-left')
    ap.add_argument("--calib-dir", type=Path, default=g.CALIB_DIR,
                    help="folder with the calibration to check (default calib/)")
    a = ap.parse_args()
    img = cv2.imread(str(a.photo))
    if img is None:
        raise SystemExit(f"cannot read {a.photo}")
    h, w = img.shape[:2]
    c = g.Calib(calib_dir=a.calib_dir)
    if not c.hom(a.plane):
        raise SystemExit(f"no calibration for the {a.plane} plane: run "
                         f"tools/calibrate_a4.py <photo> --plane {a.plane} first")
    if a.points:
        pts = [tuple(float(v) for v in p.split(",")) for p in a.points.split()]
        real = [tuple(float(v) for v in p.split(",")) for p in (a.real or "").split()]
        if len(real) != len(pts):
            raise SystemExit("--real needs one X,Y per point")
    else:
        n = 5
        pts = click_points(img, [f"dot {i + 1}" for i in range(n)], f"check points ({a.plane})")
        real = []
        for i in range(n):
            while True:
                try:
                    X, Y = (float(v) for v in input(
                        f"dot {i + 1}: real position in cm from the sheet's top-left, "
                        "'X Y' (X to the right, Y down): ").replace(",", " ").split())
                    break
                except ValueError:
                    print("  type two numbers, e.g. 10.5 14.2")
            real.append((X, Y))
    report(check(pts, real, a.plane, (w, h), c))


if __name__ == "__main__":
    main()

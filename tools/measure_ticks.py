"""Measure cm per encoder tick on the real robot.

On the robot, on the rails:
  1. Put the car at a start mark. Note or reset the encoder count.
  2. Drive it EXACTLY 100 cm along the rails (use a tape measure; drive
     slowly, and stop on the end mark).
  3. Read the encoder: ticks counted over that distance.
  4. Run ONE of

         python tools/measure_ticks.py --ticks 96            ticks for 100 cm
         python tools/measure_ticks.py --start 120 --end 218  encoder readings
         python tools/measure_ticks.py --ticks 96 --distance 200   other distance
         python tools/measure_ticks.py                        asks you

     It saves cm_per_tick (distance / ticks) in calib/robot.json and marks it
     as measured. Repeat 3 times and use the average if the counts differ.

hardware.py (the encoder reader) does not exist yet. Once it does, the same
two numbers come from reading it before and after the drive.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import geometry as g  # noqa: E402


def save_cm_per_tick(cm_per_tick: float, path: Path | None = None) -> dict:
    """Write cm_per_tick into robot.json and mark it measured; when every
    value is measured, placeholder becomes false."""
    path = path or g.CALIB_DIR / "robot.json"
    robot = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    robot = {**g.DEFAULT_ROBOT, **robot}
    robot["cm_per_tick"] = round(cm_per_tick, 5)
    robot.setdefault("measured", {})["cm_per_tick"] = True
    robot["placeholder"] = not all(robot["measured"].get(k) for k in g.ROBOT_KEYS)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(robot, indent=2), encoding="utf-8")
    return robot


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ticks", type=float, help="ticks counted over the distance")
    ap.add_argument("--start", type=float, help="encoder reading at the start")
    ap.add_argument("--end", type=float, help="encoder reading at the end")
    ap.add_argument("--distance", type=float, default=100.0, help="cm driven (default 100)")
    a = ap.parse_args()
    ticks = a.ticks
    if ticks is None and a.start is not None and a.end is not None:
        ticks = a.end - a.start
    if ticks is None:
        print(f"Drive exactly {a.distance:g} cm along the rails, then read the encoder.")
        ticks = float(input("ticks counted over that distance: "))
    if ticks <= 0:
        raise SystemExit("ticks must be positive (did the encoder count backwards?)")
    cm = a.distance / ticks
    if not 0.05 <= cm <= 20:
        print(f"WARNING: {cm:.3f} cm per tick looks wrong (expected ~0.5 to 2). "
              "Check the distance and the count.")
    robot = save_cm_per_tick(cm)
    print(f"{ticks:g} ticks over {a.distance:g} cm = {cm:.4f} cm per tick, saved to "
          f"calib/robot.json")
    left = [k for k in g.ROBOT_KEYS if not robot["measured"].get(k)]
    print("still placeholders: " + (", ".join(left) if left else "none"))


if __name__ == "__main__":
    main()

"""Simulator: the planner drives a FakeHardware over a field strip.

    python tools/sim.py --field tomato-3
    python tools/sim.py --field tomato-3 --continuous
    python tools/sim.py --field tomato-3 --speed 5 --tag speed5
    python tools/sim.py --field tomato-3 --live-vision      # real SAM + Spark per frame ($)

The strip: the field's photos laid end to end (the farmer-labeled ones by
default: only they have ground truth), a stretch of bare soil before and
after. Each photo is one camera frame wide, so the strip is the row as a
camera driving along it would see it. Ground truth = the farmer's labels.
The planner never sees the truth; only the scorer does.

Outputs (out/, or out/sim_<tag>.* with --tag):
    weed_map.json      Pass 1 result (written by the planner)
    sim_report.json    metrics, per-weed table, decisions, latencies used
    sim.mp4            camera window / top-down car + arm + splashes / counters
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import geometry as g  # noqa: E402
import hardware as hw_mod  # noqa: E402
import planner as pl  # noqa: E402

PX_PER_CM = 40          # resolution of the strip picture
GRID_CM = 0.2           # resolution of the scoring maps
LEAD_CM = 30.0          # bare soil before the first photo (the arm reaches ahead)
TAIL_CM = 30.0
FIELDS = Path("fields")


# ---------- the field strip ----------

def _contour_px(mask: np.ndarray, eps: float = 2.0) -> list[tuple[float, float]]:
    """Outline of the biggest blob of a mask, simplified (pixels)."""
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return []
    c = max(cs, key=cv2.contourArea)
    c = cv2.approxPolyDP(c, eps, True)
    return [(float(p[0][0]), float(p[0][1])) for p in c]


class Strip:
    """Photos end to end in world cm (x forward from the start, y left)."""

    def __init__(self, field: str, names: list[str], calib: g.Calib, model: dict | None,
                 cache_dir: Path, log_path: Path, refresh: bool = False):
        self.field, self.names, self.calib, self.model = field, names, calib, model
        self.D = float(calib.robot["camera_to_nozzle_cm"])
        self.tiles = []
        self.photos = {}
        fdir = FIELDS / field
        photo_files = {p.stem: p for p in (fdir / "photos").iterdir()
                       if p.suffix.lower() in (".jpg", ".jpeg", ".png")}
        x_cursor = LEAD_CM
        for n in names:
            img = Image.open(photo_files[n]).convert("RGB")
            W, H = img.size
            xt, _ = g.pixel_to_ground(W / 2, 0, "ridge", (W, H), calib)
            xb, _ = g.pixel_to_ground(W / 2, H, "ridge", (W, H), calib)
            _, yl = g.pixel_to_ground(0, H / 2, "ridge", (W, H), calib)
            _, yr = g.pixel_to_ground(W, H / 2, "ridge", (W, H), calib)
            Hg, Wg = xt - xb, yl - yr
            self.tiles.append({"name": n, "size": (W, H), "Hg": Hg, "Wg": Wg,
                               "center": x_cursor + Hg / 2, "path": photo_files[n]})
            self.photos[n] = img
            x_cursor += Hg
        self.row_end = x_cursor
        self.length = x_cursor + TAIL_CM
        self.width = max(t["Wg"] for t in self.tiles)
        # furrow ground is farther from the camera, so a plant at the photo's
        # edge is up to 33/25 times farther out than the ridge-plane picture shows
        self.score_width = self.width * 1.4
        self._build_canvas()
        self.cache_dir, self.log_path = Path(cache_dir), Path(log_path)
        self.vision_dir = Path("out") / "sim_cache" / field
        self.vision_dir.mkdir(parents=True, exist_ok=True)

    # picture: row 0 = far end (forward), column 0 = left (+y)
    def _build_canvas(self) -> None:
        rows, cols = int(round(self.length * PX_PER_CM)), int(round(self.width * PX_PER_CM))
        med = np.median(np.concatenate([np.asarray(im.resize((64, 48))).reshape(-1, 3)
                                        for im in self.photos.values()]), axis=0)
        self.canvas = np.empty((rows, cols, 3), np.uint8)
        self.canvas[:] = np.clip(med * 0.9, 0, 255).astype(np.uint8)
        for t in self.tiles:
            h, w = int(round(t["Hg"] * PX_PER_CM)), int(round(t["Wg"] * PX_PER_CM))
            tile = np.asarray(self.photos[t["name"]].resize((w, h), Image.LANCZOS))
            top = int(round((self.length - (t["center"] + t["Hg"] / 2)) * PX_PER_CM))
            left = (cols - w) // 2
            self.canvas[top:top + h, left:left + w] = tile[:rows - top, :cols - left]

    def window(self, cam_x: float, out_size=(1296, 972)) -> np.ndarray:
        """What the camera sees with the ground under its lens at cam_x."""
        w, h = out_size
        F = self.frame_len(out_size)
        r0 = int(round((self.length - (cam_x + F / 2)) * PX_PER_CM))
        r1 = r0 + int(round(F * PX_PER_CM))
        rows = self.canvas.shape[0]
        piece = np.empty((r1 - r0, self.canvas.shape[1], 3), np.uint8)
        piece[:] = self.canvas[-1, 0]
        a, b = max(r0, 0), min(r1, rows)
        if b > a:
            piece[a - r0:b - r0] = self.canvas[a:b]
        return cv2.resize(piece, (w, h), interpolation=cv2.INTER_AREA)

    def frame_len(self, size=(1296, 972)) -> float:
        w, h = size
        x0, _ = g.pixel_to_ground(w / 2, 0, "ridge", size, self.calib)
        x1, _ = g.pixel_to_ground(w / 2, h, "ridge", size, self.calib)
        return abs(x0 - x1)

    def to_world(self, tile: dict, u: float, v: float, row_y: float | None) -> tuple[float, float, str]:
        p = g.locate(u, v, self.model, tile["size"], row_y, self.calib)
        return tile["center"] + p["x"] - self.D, p["y"], p["plane"]

    # ---- truth: the farmer's labels ----
    def truth(self) -> tuple[list[dict], float]:
        """[{photo,id,label,polygon(world cm),centroid}] and the row's y."""
        raw, crop_y = [], []
        for t in self.tiles:
            f = FIELDS / self.field / "labels" / f"{t['name']}.labels.json"
            if not f.is_file():
                continue
            d = json.loads(f.read_text(encoding="utf-8"))
            for o in d["labels"]:
                m = np.asarray(Image.open(FIELDS / self.field / "labels" / o["mask"]).convert("L")) > 127
                poly = _contour_px(m)
                if len(poly) < 3:
                    continue
                raw.append((t, o, poly))
                if o["label"] == "crop":
                    b = o["box"]
                    crop_y.append(g.pixel_to_ground((b[0] + b[2]) / 2, (b[1] + b[3]) / 2,
                                                    "ridge", t["size"], self.calib)[1])
        row_y = float(np.median(crop_y)) if crop_y else 0.0
        out = []
        for t, o, poly in raw:
            wp = [self.to_world(t, u, v, row_y)[:2] for u, v in poly]
            c = np.mean(wp, axis=0)
            out.append({"photo": t["name"], "id": o["id"], "label": o["label"],
                        "polygon": wp, "centroid": (float(c[0]), float(c[1]))})
        return out, row_y

    # ---- what the vision pipeline said (cached once; costs money the first time) ----
    def ensure_cache(self, fmodel: dict | None, refs: list) -> None:
        """Make sure every photo has its pipeline output on disk. Order: our
        own cache, then the field's last evaluation run (fields/<f>/runs, free),
        then the real pipeline (API, costs a few cents per photo)."""
        import shutil
        import setup  # tools/setup.py (needs the API only on a cache miss)
        run_json = FIELDS / self.field / "runs" / "run.json"
        entries = {}
        if run_json.is_file():
            entries = {e["name"]: e for e in json.loads(run_json.read_text(encoding="utf-8")).get("entries", [])
                       if e.get("objects")}
        for t in self.tiles:
            cache = self.vision_dir / f"{t['name']}.json"
            masks = self.vision_dir / f"{t['name']}.masks.npz"
            if cache.is_file() and masks.is_file():
                continue
            run_masks = FIELDS / self.field / "runs" / f"{t['name']}.masks.npz"
            if t["name"] in entries and run_masks.is_file():
                shutil.copy(run_masks, masks)
                cache.write_text(json.dumps(entries[t["name"]]["objects"]), encoding="utf-8")
                continue
            print(f"vision on {t['name']} (cached after this)...", flush=True)
            res = setup.judge_photo(self.field, t["name"], fmodel, refs,
                                    {"cache_dir": self.cache_dir, "log_path": self.log_path},
                                    self.vision_dir)
            cache.write_text(json.dumps(res["objects"]), encoding="utf-8")

    def cached_row_y(self) -> float:
        """Row position from the cached pipeline's own crop calls (no truth)."""
        ys = []
        for t in self.tiles:
            for o in json.loads((self.vision_dir / f"{t['name']}.json").read_text(encoding="utf-8")):
                if o["label"] == "crop":
                    b = o["box"]
                    ys.append(g.pixel_to_ground((b[0] + b[2]) / 2, (b[1] + b[3]) / 2,
                                                "ridge", t["size"], self.calib)[1])
        return float(np.median(ys)) if ys else 0.0

    def vision_objects(self, row_y: float, fmodel: dict | None, refs: list) -> list[dict]:
        self.ensure_cache(fmodel, refs)
        out = []
        for t in self.tiles:
            cache = self.vision_dir / f"{t['name']}.json"
            masks = self.vision_dir / f"{t['name']}.masks.npz"
            objs = json.loads(cache.read_text(encoding="utf-8"))
            z = np.load(masks)
            by_id = {int(i): m for i, m in zip(z["ids"], z["masks"])}
            for o in objs:
                m = by_id.get(o["id"])
                poly = _contour_px(m) if m is not None else []
                world_poly = [self.to_world(t, u, v, row_y)[:2] for u, v in poly]
                spray = None
                plane = "ridge"
                if o["action"] == "spray" and o.get("x") is not None:
                    sx, sy, plane = self.to_world(t, o["x"], o["y"], row_y)
                    spray = (sx, sy)
                b = o["box"]
                cx, cy, plane2 = self.to_world(t, (b[0] + b[2]) / 2, (b[1] + b[3]) / 2, row_y)
                out.append({"photo": t["name"], "id": o["id"], "label": o["label"],
                            "action": o["action"], "conf": o.get("conf"),
                            "polygon": world_poly, "spray": spray, "center": (cx, cy),
                            "plane": plane if spray else plane2})
        return out


class CachedVision:
    """Replays the cached pipeline output as if it were run on each frame:
    every object whose centre is inside the frame is reported, in the PIXELS
    of that frame (the planner converts them back with geometry.py)."""

    def __init__(self, strip: Strip, objs: list[dict], calib: g.Calib, image_size=(1296, 972)):
        self.strip, self.objs, self.calib, self.size = strip, objs, calib, image_size
        self.cam_x = 0.0           # set by the frame source on every capture
        self.F = strip.frame_len(image_size)

    def detect(self, image, tick: int) -> list[dict]:
        p = self.cam_x - self.strip.D
        w, h = self.size
        dets = []
        for o in self.objs:
            cx, cy = o["center"]
            if abs(cx - self.cam_x) > self.F / 2:
                continue

            def px(x, y):
                return g.ground_to_pixel(x - p, y, o["plane"], self.size, self.calib)

            u, v = px(cx, cy)
            dets.append({"label": o["label"], "action": o["action"], "conf": o.get("conf"),
                         "box": [u - 1, v - 1, u + 1, v + 1],
                         "spray_px": px(*o["spray"]) if o["spray"] else None,
                         "polygon_px": [px(x, y) for x, y in o["polygon"]],
                         "id": f"{o['photo']}#{o['id']}"})
        return dets


class LiveVision:
    """The real pipeline (SAM + Muse Spark) on every frame. Costs money."""

    def __init__(self, strip: Strip, fmodel, refs, cache_dir, log_path, image_size=(1296, 972)):
        self.strip, self.fmodel, self.refs = strip, fmodel, refs
        self.cache_dir, self.log_path, self.size = Path(cache_dir), Path(log_path), image_size
        self.n = 0
        self.usd = 0.0

    def detect(self, image, tick: int) -> list[dict]:
        import vision
        self.n += 1
        tmp = Path("out") / "sim_cache" / f"live_{self.n:03d}.png"
        Image.fromarray(image).save(tmp)
        r = vision.analyze(tmp, cache_dir=self.cache_dir, log_path=self.log_path,
                           refs=self.refs, field_model=self.fmodel)
        self.usd += r.spark_usd + (0 if r.sam_cached else vision.sam_client.USD_PER_IMAGE)
        boxes = {o["id"]: o for o in r.objects}
        dets = []
        for t in r.targets:
            o = boxes[t.id]
            dets.append({"label": {"crop": "crop", "weed": "weed"}.get(t.label, "unsure"),
                         "action": t.action, "conf": float(t.confidence),
                         "box": [float(v) for v in o["box"]],
                         "spray_px": (t.x, t.y) if t.action == "spray" else None,
                         "polygon_px": _contour_px(o["mask"]), "id": f"live{self.n}#{t.id}"})
        return dets


# ---------- scoring ----------

class Scorer:
    """Rasterises the truth and every physical spray on a 0.2 cm grid."""

    def __init__(self, strip: Strip, truth: list[dict], calib: g.Calib):
        self.strip, self.calib = strip, calib
        self.ny = int(math.ceil(strip.score_width / GRID_CM))
        self.nx = int(math.ceil(strip.length / GRID_CM))
        self.weed_id = np.full((self.nx, self.ny), -1, np.int32)
        self.crop = np.zeros((self.nx, self.ny), bool)
        self.truth = truth
        self.weeds = [t for t in truth if t["label"] == "weed"]
        self.crops = [t for t in truth if t["label"] in ("crop", "unsure")]
        for i, t in enumerate(self.weeds):
            self._fill(self.weed_id, t["polygon"], i)
        c8 = np.zeros((self.nx, self.ny), np.uint8)
        for t in self.crops:
            self._fill(c8, t["polygon"], 1)
        self.crop = c8.astype(bool)

    def cell(self, x: float, y: float) -> tuple[int, int]:
        return int(round(x / GRID_CM)), int(round((self.strip.score_width / 2 - y) / GRID_CM))

    def _fill(self, arr, poly, val) -> None:
        pts = np.array([[self.cell(x, y)[1], self.cell(x, y)[0]] for x, y in poly], np.int32)
        cv2.fillPoly(arr, [pts], int(val))

    def footprint(self, centers: list[tuple[float, float]]) -> np.ndarray:
        m = np.zeros((self.nx, self.ny), np.uint8)
        r = int(round(float(self.calib.robot["spray_radius_cm"]) / GRID_CM))
        for x, y in centers:
            i, j = self.cell(x, y)
            cv2.circle(m, (j, i), r, 1, -1)
        return m.astype(bool)

    def pump_centers(self, hw: hw_mod.FakeHardware, pump: dict) -> list[tuple[float, float]]:
        """Where the nozzle really was during this pump: true position of the
        car and the real servo angle at each 20 ms step."""
        L = float(self.calib.robot["arm_length_cm"])
        out = []
        t1 = pump["t0"] + pump["ms"] / 1000.0
        for t, pos, servo, _drv, _pmp in hw.trace:
            if pump["t0"] - 1e-9 < t <= t1 + 1e-9:
                phi = math.radians(g.servo_to_arm(servo, self.calib))
                out.append((pos + L * math.cos(phi), L * math.sin(phi)))
        if not out:
            phi = math.radians(g.servo_to_arm(hw.servo_deg(pump["t0"]), self.calib))
            out.append((pump["pos"] + L * math.cos(phi), L * math.sin(phi)))
        return out

    def score(self, hw: hw_mod.FakeHardware, planner: pl.Planner, cfg: pl.Config) -> dict:
        rob = self.calib.robot
        flow = float(rob["pump_flow_ml_per_s"])
        hit_weed = set()
        all_cover = np.zeros((self.nx, self.ny), bool)
        pumps, crop_hits, nothing = [], 0, 0
        for k, p in enumerate(hw.pumps):
            fp = self.footprint(self.pump_centers(hw, p))
            all_cover |= fp
            ids = {int(i) for i in np.unique(self.weed_id[fp]) if i >= 0}
            on_crop = bool((fp & self.crop).any())
            hit_weed |= ids
            crop_hits += on_crop
            nothing += (not ids)
            c = self.pump_centers(hw, p)[len(self.pump_centers(hw, p)) // 2]
            pumps.append({"t": round(p["t0"], 2), "ms": p["ms"], "tick": p["tick"],
                          "landing_cm": [round(c[0], 2), round(c[1], 2)],
                          "weeds_hit": sorted(ids), "crop_hit": on_crop})
        weed_cells = self.weed_id >= 0
        covered = int((weed_cells & all_cover).sum())
        total_cells = int(weed_cells.sum())
        per_weed = []
        for i, t in enumerate(self.weeds):
            near = [w for w in planner.weeds
                    if math.dist((w["x"], w["y"]), t["centroid"]) <= 4.0]
            if near:
                w = min(near, key=lambda w: math.dist((w["x"], w["y"]), t["centroid"]))
                why = w.get("status", "not reached")
                if why == "sprayed":
                    why = "sprayed, but the spray landed beside the weed"
            else:
                why = "not in the weed map (vision did not call it a sure weed)"
            per_weed.append({"photo": t["photo"], "id": t["id"],
                             "x_cm": round(t["centroid"][0], 1), "y_cm": round(t["centroid"][1], 1),
                             "sprayed": i in hit_weed,
                             "status": "sprayed" if i in hit_weed else why})
        pump_ms = sum(p["ms"] for p in hw.pumps)
        ml = pump_ms / 1000.0 * flow
        dose = flow * (cfg.spray_ms / 1000.0) / (math.pi * float(rob["spray_radius_cm"]) ** 2)
        row_area = (self.strip.row_end - LEAD_CM) * self.strip.width
        broadcast = dose * row_area
        return {"weeds_total": len(self.weeds), "weeds_sprayed": len(hit_weed),
                "weeds_missed": len(self.weeds) - len(hit_weed),
                "CROP_HITS": int(crop_hits), "sprays": len(hw.pumps),
                "sprays_on_nothing": int(nothing),
                "pct_weed_area_covered": round(100.0 * covered / max(1, total_cells), 1),
                "liquid_ml": round(ml, 2), "broadcast_ml": round(broadcast, 1),
                "liquid_saved_pct": round(100.0 * (1 - ml / broadcast), 1) if broadcast else 0.0,
                "skipped": {k: sum(1 for w in planner.weeds if w.get("status") == k)
                            for k in sorted({w.get("status") for w in planner.weeds
                                             if w.get("status") not in (None, "sprayed")})},
                "total_time_s": round(hw.now(), 1)}, per_weed, pumps


# ---------- video ----------

LABEL_COLOR = {"crop": (60, 200, 60), "weed": (60, 60, 230), "unsure": (170, 170, 170)}
W, Hh = 1280, 720


def render_segment(vw, strip: Strip, scorer: Scorer | None, hw: hw_mod.FakeHardware,
                   planner: pl.Planner, pumps: list[dict], calib: g.Calib,
                   seconds: float = 45.0, fps: int = 15, caption: str = "",
                   data_objs: list[dict] | None = None, data_title: str = "") -> None:
    """One drive of one column, drawn into the open video writer `vw`. With
    data_objs the column is a data-gathering drive (what the pipeline labeled
    is drawn as it is seen, nothing is sprayed); otherwise the truth is drawn
    and the counters follow the sprays."""
    weeds_t = scorer.weeds if scorer else []
    crops_t = scorer.crops if scorer else []
    L = float(calib.robot["arm_length_cm"])
    r_spray = float(calib.robot["spray_radius_cm"])
    flow = float(calib.robot["pump_flow_ml_per_s"])
    xmin, xmax = -L - 8, strip.length + 4
    s = W / (xmax - xmin)
    panel_y0, panel_h = 300, 300
    cy = panel_y0 + panel_h / 2

    def P(x, y):
        return int(round((x - xmin) * s)), int(round(cy - y * s))

    rot = cv2.rotate(strip.canvas, cv2.ROTATE_90_CLOCKWISE)
    rot = cv2.resize(rot, (int(strip.length * s), int(strip.width * s)), interpolation=cv2.INTER_AREA)
    base = np.full((Hh, W, 3), 28, np.uint8)
    x0, y0 = P(0, strip.width / 2)
    base[y0:y0 + rot.shape[0], x0:x0 + rot.shape[1]] = rot[:Hh - y0, :W - x0]
    cv2.rectangle(base, (0, panel_y0), (W - 1, panel_y0 + panel_h), (90, 90, 90), 1)

    # time of each weed's fate
    spray_t = {}
    for p in pumps:
        for i in p["weeds_hit"]:
            spray_t.setdefault(i, p["t"])
    skip_t = {}
    for e in planner.log:
        if e["type"] == "skip":
            skip_t[e["weed"]] = e["t"]
    weed_to_truth = {}
    for i, t in enumerate(weeds_t):
        near = [w for w in planner.weeds if math.dist((w["x"], w["y"]), t["centroid"]) <= 4.0]
        if near:
            weed_to_truth[i] = min(near, key=lambda w: math.dist((w["x"], w["y"]), t["centroid"]))["id"]
    pump_t = [(p["t"], p["landing_cm"]) for p in pumps]
    trace = hw.trace
    times = np.array([r[0] for r in trace])
    T = hw.now()
    n = int(seconds * fps)
    cap_times = [e["t"] for e in hw.events if e["type"] == "capture"]
    phases = planner.phases
    for k in range(n):
        t = T * (k + 1) / n
        i = min(int(np.searchsorted(times, t)), len(trace) - 1)
        _t, pos, servo, driving, pumping = trace[i]
        phi = math.radians(g.servo_to_arm(servo, calib))
        tip = (pos + L * math.cos(phi), L * math.sin(phi))
        fr = base.copy()
        # camera window (top panel)
        cam = strip.window(pos + strip.D)
        cam = cv2.resize(cam, (400, 300))
        fr[0:300, 0:400] = cam
        if any(0 <= t - ct < 0.4 for ct in cap_times):
            cv2.rectangle(fr, (0, 0), (399, 299), (255, 255, 255), 6)
        # planned weeds on the camera view
        for w in planner.weeds:
            u, v = g.ground_to_pixel(w["x"] - pos, w["y"], w["plane"], (400, 300), calib)
            if 0 <= u < 400 and 0 <= v < 300:
                cv2.drawMarker(fr, (int(u), int(v)), (255, 255, 0), cv2.MARKER_CROSS, 14, 2)
        cv2.putText(fr, "camera window (cyan + = planned spray)", (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        # truth markers (middle panel)
        seen = 0
        counts = {"crop": 0, "weed": 0, "unsure": 0}
        for o in data_objs or []:
            if o["center"][0] <= pos + strip.D + strip.frame_len() / 2 and len(o["polygon"]) >= 3:
                pts = np.array([P(x, y) for x, y in o["polygon"]], np.int32)
                lab = o["label"] if o["label"] in LABEL_COLOR else "unsure"
                cv2.polylines(fr, [pts], True, LABEL_COLOR[lab], 2, cv2.LINE_AA)
                counts[lab] += 1
                seen += 1
        for c in crops_t:
            pts = np.array([P(x, y) for x, y in c["polygon"]], np.int32)
            cv2.polylines(fr, [pts], True, (60, 200, 60), 2, cv2.LINE_AA)
        for j, wd in enumerate(weeds_t):
            pts = np.array([P(x, y) for x, y in wd["polygon"]], np.int32)
            if j in spray_t and spray_t[j] <= t:
                cv2.fillPoly(fr, [pts], (0, 140, 255))
            else:
                cv2.polylines(fr, [pts], True, (60, 60, 230), 2, cv2.LINE_AA)
            wid = weed_to_truth.get(j)
            if wid is None and pos >= wd["centroid"][0] - L and not (j in spray_t and spray_t[j] <= t):
                cx, cyy = P(*wd["centroid"])      # the vision did not call it a sure weed
                cv2.circle(fr, (cx, cyy), 14, (190, 190, 190), 2, cv2.LINE_AA)
            if wid is not None and wid in skip_t and skip_t[wid] <= t and not (j in spray_t and spray_t[j] <= t):
                cx, cyy = P(*wd["centroid"])
                cv2.circle(fr, (cx, cyy), 14, (0, 230, 255), 3, cv2.LINE_AA)
        # splashes so far
        ov = fr.copy()
        for pt, c in pump_t:
            if pt <= t:
                cv2.circle(ov, P(*c), max(2, int(r_spray * s)), (255, 120, 20), -1, cv2.LINE_AA)
        fr = cv2.addWeighted(ov, 0.6, fr, 0.4, 0)
        # car, arm, camera window
        cv2.rectangle(fr, P(pos - 18, 8), P(pos + 2, -8), (200, 200, 200), -1)
        cv2.rectangle(fr, P(pos - 18, 8), P(pos + 2, -8), (40, 40, 40), 2)
        cv2.line(fr, P(pos, 0), P(*tip), (255, 255, 255), 3, cv2.LINE_AA)
        cv2.circle(fr, P(*tip), 5, (255, 140, 30) if pumping else (255, 255, 255), -1, cv2.LINE_AA)
        F = strip.frame_len()
        cv2.rectangle(fr, P(pos + strip.D - F / 2, strip.width / 2),
                      P(pos + strip.D + F / 2, -strip.width / 2), (255, 255, 0), 1)
        # counters
        done = sum(1 for v in spray_t.values() if v <= t)
        skipped = sum(1 for v in skip_t.values() if v <= t)
        sprays = sum(1 for pt, _c in pump_t if pt <= t)
        ml = sum(p["ms"] for p in pumps if p["t"] <= t) / 1000.0 * flow
        phase = next((nm for nm, pt in reversed(phases) if pt <= t), "")
        hits = sum(1 for p in pumps if p["t"] <= t and p["crop_hit"])
        cv2.rectangle(fr, (0, 610), (W, Hh), (20, 20, 20), -1)
        head = f"t = {t:6.1f} s   {phase}   car {pos:6.1f} cm   arm {math.degrees(phi):+5.1f} deg"
        if data_objs is not None:
            nf = sum(1 for ct in cap_times if ct <= t)
            rows = [head,
                    f"frames saved {nf}    plants labeled by the AI {seen}  (crop {counts['crop']} / "
                    f"weed {counts['weed']} / unsure {counts['unsure']})    sprayed 0",
                    "green = crop, red = weed, grey = unsure. A person will review these labels."]
        else:
            rows = [head,
                    f"weeds sprayed {done}/{len(weeds_t)}    sprays {sprays}    "
                    f"skipped near crop (yellow ring) {skipped}    CROP HITS {hits}",
                    f"liquid {ml:5.2f} ml   orange = sprayed, red = weed, green = crop, "
                    "grey ring = vision not sure, protected"]
        for q, txt in enumerate(rows):
            col = (80, 80, 255) if (q == 1 and hits) else (255, 255, 255)
            cv2.putText(fr, txt, (14, 640 + q * 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2, cv2.LINE_AA)
        for q, txt in enumerate((data_title or caption).split("|")):
            cv2.putText(fr, txt.strip(), (430, 50 + q * 44), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0 if q == 0 else 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        vw.write(fr)


def render_video(path: Path, strip, scorer, hw, planner, pumps, calib, seconds: float = 45.0,
                 fps: int = 15, caption: str = "") -> None:
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, Hh))
    render_segment(vw, strip, scorer, hw, planner, pumps, calib, seconds, fps, caption)
    vw.release()


def card(vw, lines: list[str], seconds: float, fps: int = 15, images: list | None = None,
         progress: bool = False) -> None:
    """A title card: big text, optional pictures, optional filling bar."""
    n = int(seconds * fps)
    for k in range(n):
        fr = np.full((Hh, W, 3), 24, np.uint8)
        y = 90
        for i, txt in enumerate(lines):
            cv2.putText(fr, txt, (60, y), cv2.FONT_HERSHEY_SIMPLEX, 1.25 if i == 0 else 0.8,
                        (255, 255, 255) if i == 0 else (200, 200, 200), 2, cv2.LINE_AA)
            y += 70 if i == 0 else 42
        for i, im in enumerate(images or []):
            x0 = 60 + i * 400
            fr[300:600, x0:x0 + 380] = cv2.resize(im, (380, 300))
        if progress:
            f = (k + 1) / n
            cv2.rectangle(fr, (60, 650), (1220, 685), (90, 90, 90), 2)
            cv2.rectangle(fr, (62, 652), (62 + int(1156 * f), 683), (255, 170, 60), -1)
        vw.write(fr)


# ---------- run ----------

def load_field_model(field: str):
    import setup
    return setup.active_model(field)


def run(field: str, speed: float = 3.0, continuous: bool = False, live: bool = False,
        tag: str = "", names: list[str] | None = None, vision_s: float = hw_mod.VISION_S,
        video: bool = True, out_dir: Path = Path("out"), calib: g.Calib | None = None,
        cache_dir: Path = Path("out/cache"), log_path: Path = Path("out/api_log.jsonl"),
        quiet: bool = False, video_seconds: float = 45.0, realtime: bool = False,
        camera_cm: float | None = None, ctx_out: dict | None = None) -> dict:
    calib = calib or g.Calib(quiet=True)
    if camera_cm is not None:
        calib = with_camera(calib, camera_cm)
    fmodel, srefs = load_field_model(field)
    if fmodel is None:
        raise SystemExit(f"{field} has no field model yet (run the setup in the app first)")
    import setup
    refs = setup.ref_list(srefs, [])
    if names is None:
        lab = FIELDS / field / "labels"
        names = sorted(p.name.split(".labels.json")[0] for p in lab.glob("*.labels.json"))
    if not names:
        raise SystemExit(f"{field} has no farmer-labeled photos: no ground truth to score")
    strip = Strip(field, names, calib, fmodel, cache_dir, log_path)
    truth, row_y = strip.truth()
    size = (int(calib.robot["image_width_px"]), int(calib.robot["image_height_px"]))
    cv = None
    if live:
        backend = LiveVision(strip, fmodel, refs, cache_dir, log_path, size)
    else:
        objs = strip.vision_objects(row_y, fmodel, refs)
        backend = cv = CachedVision(strip, objs, calib, size)

    def frames(cam_x: float) -> np.ndarray:
        if cv is not None:
            cv.cam_x = cam_x
        else:
            backend.cam_x = cam_x
        return strip.window(cam_x, size)

    park = g.arm_to_servo(float(calib.servo["arm"]["park_deg"]), calib)
    hw = hw_mod.FakeHardware(frames, camera_to_nozzle_cm=float(calib.robot["camera_to_nozzle_cm"]),
                             cm_per_tick=float(calib.robot["cm_per_tick"]), park_deg=park)
    cfg = pl.Config(speed=speed, continuous=continuous, vision_s=vision_s,
                    vision_workers=1 if realtime else 3, out_dir=out_dir, image_size=size)
    p = pl.Planner(hw, backend, calib, fmodel, cfg)
    if realtime:
        p.run_realtime(strip.row_end)
    else:
        p.run(strip.row_end)
    if tag:
        (Path(out_dir) / "weed_map.json").replace(Path(out_dir) / f"weed_map_{tag}.json")
    scorer = Scorer(strip, truth, calib)
    metrics, per_weed, pumps = scorer.score(hw, p, cfg)
    sfx = f"_{tag}" if tag else ""
    report = {"field": field, "photos": names, "mode": ("real-time one pass, " if realtime else "two passes, ")
                              + ("continuous" if continuous else "stop-and-spray"),
              "camera_to_nozzle_cm": float(calib.robot["camera_to_nozzle_cm"]),
              "speed_cm_s": speed, "live_vision": live,
              "truth": "farmer labels on the photos above (the field model was learned from "
                       "these same photos: vision numbers are optimistic, the mechanics are not)",
              "latencies": {"settle_s": cfg.settle_s, "spray_ms": cfg.spray_ms, "drip_s": cfg.drip_s,
                            "servo_s_per_60deg": hw_mod.SERVO_S_PER_60DEG, "capture_s": hw_mod.CAPTURE_S,
                            "vision_s": cfg.vision_s, "coast_s": cfg.brake_s},
              "calib_placeholder": bool(calib.robot.get("placeholder")),
              "metrics": metrics, "weeds": per_weed, "sprays": pumps,
              "decisions": p.log, "events": hw.events}
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / f"sim_report{sfx}.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    if ctx_out is not None:
        ctx_out.update(strip=strip, scorer=scorer, hw=hw, planner=p, pumps=pumps, calib=calib,
                       fmodel=fmodel, refs=refs, metrics=metrics)
    if video:
        render_video(Path(out_dir) / f"sim{sfx}.mp4", strip, scorer, hw, p, pumps, calib,
                     seconds=video_seconds,
                     caption="REAL-TIME PASS|detect and spray in one go" if realtime else "")
    if not quiet:
        print(json.dumps(metrics, indent=1))
    return report


def with_camera(calib: g.Calib, camera_cm: float) -> g.Calib:
    """The same robot with the camera camera_cm ahead of the arm's pivot."""
    return g.Calib(robot={**calib.robot, "camera_to_nozzle_cm": float(camera_cm)},
                   servo=calib.servo, homographies=calib._homs, calib_dir=calib.dir, quiet=True)


def lifecycle(field: str, out_dir: Path = Path("out"), speed: float = 3.0,
              camera_cm: float = 40.0, continuous: bool = False, columns: int = 3,
              per_column: int = 2, video_seconds_real: float = 26.0) -> dict:
    """The story in one video: columns 1..N are driven once to GATHER DATA (the
    slow SAM + Spark pipeline labels them, nothing is sprayed), an offline model
    is trained from them, then a real column is driven ONCE: the fast model
    detects in real time and the arm sprays as it goes, no map, no return."""
    import setup
    out_dir = Path(out_dir)
    calib = with_camera(g.Calib(quiet=True), camera_cm)
    fmodel, srefs = load_field_model(field)
    refs = setup.ref_list(srefs, [])
    size = (int(calib.robot["image_width_px"]), int(calib.robot["image_height_px"]))
    cache_dir, log_path = Path("out/cache"), Path("out/api_log.jsonl")
    lab = FIELDS / field / "labels"
    labeled = {p.name.split(".labels.json")[0] for p in lab.glob("*.labels.json")}
    run_json = json.loads((FIELDS / field / "runs" / "run.json").read_text(encoding="utf-8"))
    pool = sorted(e["name"] for e in run_json["entries"] if e.get("objects") and e["name"] not in labeled)
    groups = [pool[i * per_column:(i + 1) * per_column] for i in range(columns)]
    groups = [x for x in groups if x]
    segs, all_objs = [], []
    for k, names in enumerate(groups, 1):
        strip = Strip(field, names, calib, fmodel, cache_dir, log_path)
        strip.ensure_cache(fmodel, refs)
        objs = strip.vision_objects(strip.cached_row_y(), fmodel, refs)
        cv = CachedVision(strip, objs, calib, size)

        def frames(cam_x, strip=strip, cv=cv):
            cv.cam_x = cam_x
            return strip.window(cam_x, size)

        hw = hw_mod.FakeHardware(frames, camera_to_nozzle_cm=camera_cm,
                                 cm_per_tick=float(calib.robot["cm_per_tick"]))
        p = pl.Planner(hw, cv, calib, fmodel, pl.Config(speed=speed, out_dir=out_dir / "_data", image_size=size))
        p.map_pass(strip.row_end)
        segs.append((k, strip, hw, p, objs, cv))
        all_objs += objs
    ctx: dict = {}
    run("%s" % field, speed, continuous, False, "lifecycle", None, 0.1, False, out_dir, calib,
        cache_dir, log_path, True, realtime=True, ctx_out=ctx)
    m = ctx["metrics"]
    # ---- the video ----
    path = out_dir / "sim_lifecycle.mp4"
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 15, (W, Hh))
    n_frames = sum(len(s[3].frames) for s in segs)
    n_obj = len(all_objs)
    n_weed = sum(1 for o in all_objs if o["label"] == "weed")
    n_crop = sum(1 for o in all_objs if o["label"] == "crop")
    card(vw, ["How WeedBot learns a field", f"Columns 1-{len(segs)}: drive once, gather data",
              "Then: train an offline model.  Then: detect and spray in REAL TIME, one pass.",
              "(all numbers are simulated with placeholder robot values)"], 3.5)
    for k, strip, hw, p, objs, _cv in segs:
        render_segment(vw, strip, None, hw, p, [], calib, seconds=6.5, caption="",
                       data_objs=objs, data_title=f"COLUMN {k} of {len(segs)}: GATHER DATA|"
                                                  "drive once, the AI labels every plant, no spraying")
    imgs = []
    for k, strip, hw, p, objs, cv in segs[:3]:
        cam = strip.row_end - 20
        cv.cam_x = cam
        im = strip.window(cam, size).copy()
        for d in cv.detect(im, 0):
            pts = np.array(d["polygon_px"], np.int32)
            if len(pts) >= 3:
                cv2.polylines(im, [pts], True, LABEL_COLOR.get(d["label"], (170, 170, 170)), 5)
        imgs.append(im)
    card(vw, ["OFFLINE TRAINING (illustration)",
              f"{n_frames} frames, {n_obj} plants ({n_crop} crop, {n_weed} weed) labeled by the AI and reviewed",
              "A small detector (YOLO) learns them offline, on a laptop. No model is trained yet in this project;",
              "in the next frames its detections are stood in by the cached AI results, replayed at ~0.1 s per frame."],
         5.0, images=imgs, progress=True)
    render_segment(vw, ctx["strip"], ctx["scorer"], ctx["hw"], ctx["planner"], ctx["pumps"], calib,
                   seconds=video_seconds_real,
                   caption="REAL COLUMN: ONE PASS|the offline model detects in real time, the arm sprays as it goes")
    card(vw, ["Result on the real column",
              f"weeds sprayed {m['weeds_sprayed']}/{m['weeds_total']}    crop hits {m['CROP_HITS']}    "
              f"sprays on nothing {m['sprays_on_nothing']}",
              f"liquid {m['liquid_ml']} ml instead of {m['broadcast_ml']} ml for spraying the whole row "
              f"({m['liquid_saved_pct']}% less)",
              f"{m['total_time_s']} s for the whole column in one pass; no return trip, no waiting for the AI",
              "Weeds the AI was not sure about stay protected (grey rings): the farmer's review teaches the model."],
         5.0)
    vw.release()
    print(json.dumps(m, indent=1))
    print(f"video: {path}")
    return m


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--field", required=True)
    ap.add_argument("--speed", type=float, default=3.0, help="cm/s (default 3)")
    ap.add_argument("--continuous", action="store_true", help="spray without stopping")
    ap.add_argument("--live-vision", action="store_true",
                    help="run SAM + Spark on every frame (costs money; default replays a cached run)")
    ap.add_argument("--vision-s", type=float, default=None,
                    help="seconds of processing per frame (default 10; 0.1 with --realtime)")
    ap.add_argument("--realtime", action="store_true",
                    help="ONE pass: detect each frame as it comes and spray as the arm gets level "
                         "(a fast offline model; here the cached results stand in for it)")
    ap.add_argument("--camera-cm", type=float, default=None,
                    help="camera distance ahead of the arm pivot (default robot.json; --realtime "
                         "and --lifecycle need about arm length + half a frame: 40)")
    ap.add_argument("--lifecycle", action="store_true",
                    help="make out/sim_lifecycle.mp4: data columns -> offline training -> real column")
    ap.add_argument("--photos", help="comma list of photo names (default: the farmer-labeled ones)")
    ap.add_argument("--tag", default="", help="suffix for the output files")
    ap.add_argument("--no-video", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    if a.lifecycle:
        lifecycle(a.field, speed=a.speed, camera_cm=a.camera_cm or 40.0, continuous=a.continuous)
    else:
        vs = a.vision_s if a.vision_s is not None else (0.1 if a.realtime else hw_mod.VISION_S)
        run(a.field, a.speed, a.continuous, a.live_vision, a.tag,
            a.photos.split(",") if a.photos else None, vs, not a.no_video,
            realtime=a.realtime, camera_cm=a.camera_cm or (40.0 if a.realtime else None))
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()

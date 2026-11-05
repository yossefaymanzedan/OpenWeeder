"""Trainer pieces that need no API and no GPU.

    python -m pytest tests/test_trainer.py -v
"""
from __future__ import annotations

import sys
import threading
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import trainer  # noqa: E402


def test_yolo_lines_make_unsure_a_class_instead_of_background():
    objs = [{"id": 0, "box": [10, 20, 110, 120], "label": "crop", "conf": 0.95},
            {"id": 1, "box": [200, 200, 300, 260], "label": "weed", "conf": 0.7},
            {"id": 2, "box": [0, 0, 50, 50], "label": "unsure", "conf": 0.9},
            {"id": 3, "box": [0, 0, 50, 50], "label": "weed", "conf": 0.3},
            {"id": 4, "box": [5, 5, 7, 7], "label": "weed", "conf": 0.9}]
    lines, st = trainer.yolo_lines(objs, (400, 400))
    assert st == {"kept": {"crop": 1, "weed": 1, "unsure": 2}, "skipped": 1}
    assert [ln.split()[0] for ln in lines] == ["0", "1", "2", "2"]
    cls, cx, cy, w, h = lines[0].split()
    assert float(cx) == pytest.approx(60 / 400) and float(w) == pytest.approx(0.25)


def test_split3_is_deterministic_and_each_part_is_used():
    names = [f"img_{i}.jpg" for i in range(100)]
    a = trainer.split3(names)
    assert a == trainer.split3(names)
    tr, ca, te = a
    assert (len(tr), len(ca), len(te)) == (70, 10, 20)
    assert not (set(tr) & set(ca) or set(tr) & set(te) or set(ca) & set(te))
    tr, ca, te = trainer.split3([f"{i}" for i in range(4)])
    assert len(ca) == 1 and len(te) == 1 and len(tr) == 2


def test_record_takes_a_photo_before_moving(tmp_path):
    # Was a blank function until the robot was built; tests/test_pi_hardware.py
    # covers recording in full. A dry-run robot with no phone gives grey frames.
    from hardware import PiHardware
    hw = PiHardware(dry_run=True)
    stop = threading.Event()
    stop.set()
    try:
        assert trainer.record_images("x", stop, root=tmp_path, hardware=hw) == 1
    finally:
        hw.close()
    assert (tmp_path / "x" / "train" / "images" / "img_00001.jpg").is_file()


def test_import_skips_non_images_and_numbers_files(tmp_path):
    buf = BytesIO()
    Image.new("RGB", (8, 8), (0, 200, 0)).save(buf, "PNG")
    n = trainer.import_images("c", [("a.png", buf.getvalue()), ("notes.txt", b"x"),
                                    ("bad.jpg", b"not an image"), ("b.png", buf.getvalue())],
                              root=tmp_path)
    assert n == 2
    assert [p.name for p in trainer.images("c", tmp_path)] == ["img_00001.jpg", "img_00002.jpg"]
    assert trainer.info("c", tmp_path)["images"] == 2


def test_training_needs_enough_labeled_images(tmp_path):
    (tmp_path / "c" / "train" / "images").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="at least 4"):
        trainer.build_yolo_dataset("c", tmp_path)


def test_epochs_are_capped_at_30():
    assert trainer.MAX_EPOCHS == 30


# ---- the spray rule ----
CROP, WEED, UNS = 0, 1, 2


def test_spray_rule_never_sprays_inside_a_detected_crop_or_unsure_box():
    crop = [100, 100, 200, 200]
    unsure = [400, 400, 450, 450]
    pred = [(CROP, 0.9, crop), (UNS, 0.5, unsure),
            (WEED, 0.9, [140, 140, 160, 160]),      # centre inside the crop: veto
            (WEED, 0.9, [410, 410, 430, 430]),      # centre inside an unsure plant: veto
            (WEED, 0.9, [600, 600, 640, 640]),      # open soil: spray
            (WEED, 0.05, [700, 700, 740, 740])]     # below the confidence: no
    out = trainer.sprays(pred, conf_w=0.25, margin=0.0, crop_conf=0.25)
    assert out == [[600, 600, 640, 640]]            # the unsure box (0.5) is >= crop_conf 0.25
    # the veto grows with the margin: a weed just beside the crop is vetoed by margin 0.3
    pred2 = [(CROP, 0.9, crop), (WEED, 0.9, [205, 140, 215, 160])]
    assert trainer.sprays(pred2, 0.25, 0.0, 0.25) != []
    assert trainer.sprays(pred2, 0.25, 0.3, 0.25) == []
    # a crop detected below crop_conf does not guard
    pred3 = [(CROP, 0.1, crop), (WEED, 0.9, [140, 140, 160, 160])]
    assert trainer.sprays(pred3, 0.25, 0.0, 0.25) != []


def test_score_counts_found_weeds_and_crop_hits():
    W = H = 1000
    gt = [(CROP, [100, 100, 200, 200]), (WEED, [500, 500, 600, 600]),
          (WEED, [800, 800, 815, 815]), (UNS, [300, 300, 350, 350])]
    pred = [(WEED, 0.9, [510, 510, 590, 590]),      # finds the large weed
            (WEED, 0.9, [140, 140, 160, 160]),      # lands on the crop
            (WEED, 0.9, [320, 320, 330, 330]),      # lands on the unsure plant
            (WEED, 0.9, [900, 100, 940, 140])]      # on nothing
    s = trainer.score([(W, H, gt, pred)], conf_w=0.25, margin=-1, crop_conf=1.1)
    assert s["weeds_found"] == 1 and s["weeds_total"] == 2
    assert s["large_total"] == 1 and s["large_found"] == 1      # the 15 px weed is not "large"
    assert s["crop_hits"] == 1 and s["unsure_hits"] == 1 and s["false_sprays"] == 1
    # with the veto on (the detected crop and unsure boxes guard), both hits disappear
    pred2 = pred + [(CROP, 0.9, [100, 100, 200, 200]), (UNS, 0.9, [300, 300, 350, 350])]
    s2 = trainer.score([(W, H, gt, pred2)], conf_w=0.25, margin=0.0, crop_conf=0.25)
    assert s2["crop_hits"] == 0 and s2["unsure_hits"] == 0 and s2["weeds_found"] == 1


def test_choose_prefers_zero_hits_over_more_weeds():
    W = H = 1000
    gt = [(CROP, [100, 100, 200, 200]), (WEED, [500, 500, 600, 600])]
    # a low-confidence weed box sits on the crop and is NOT guarded (crop never detected)
    pred = [(WEED, 0.9, [510, 510, 590, 590]), (WEED, 0.08, [140, 140, 160, 160])]
    op = trainer.choose([(W, H, gt, pred)])
    assert op["calibration"]["crop_hits"] == 0
    assert op["conf_w"] > 0.08 and op["calibration"]["large_found"] == 1


def test_covered_matches_an_object_to_its_label():
    labels = [[100, 100, 200, 200]]
    assert trainer.covered([[98, 98, 202, 202], [120, 120, 180, 180]], labels)
    assert not trainer.covered([[500, 500, 600, 600], [510, 510, 590, 590]], labels)

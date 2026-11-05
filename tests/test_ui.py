"""Browser UI tests: every page, every button, the full farmer flow.

Runs the real Flask app in a throwaway working folder (your fields/ and
crops/ are never touched) with vision.analyze replaced by a fast fake, so
no API calls and no cost. Drives the installed Microsoft Edge via
Playwright (no browser download).

    pip install -r requirements-dev.txt
    python -m pytest tests/test_ui.py -v            (headless)
    set UI_HEADED=1 & python -m pytest tests/test_ui.py -v   (watch it)
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageOps
from playwright.sync_api import Page, expect, sync_playwright
from werkzeug.serving import make_server

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import app as appmod  # noqa: E402
import setup  # noqa: E402
import vision  # noqa: E402
from vision import Target, VisionResult  # noqa: E402

STAGE_DELAY = 0.4  # seconds per fake pipeline stage: slow enough to watch live
CALLS: list[dict] = []
SPARK: list[dict] = []  # setup's checklist / review / revise calls
FAKE_LABELS = ["crop", "weed", "crop", "mixed", "weed", "crop"]


def fake_spark_json(content, schema, name, stage, log_path, max_tokens=1500):
    """Setup's Spark calls: a fixed checklist, one review suspect, a revision."""
    text = "\n".join(c["text"] for c in content if c["type"] == "input_text")
    SPARK.append({"name": name, "text": text,
                  "images": sum(c["type"] == "input_image" for c in content)})
    if name == "FieldChecklist":
        return {"checklist": "Carrot: thin feathery leaves. Weeds: round leaves. "
                             "If unsure, protect."}, 0.0
    if name == "ChecklistReview":
        return {"suspects": [{"photo": "x", "id": 1, "now": "weed",
                              "should_be": "crop", "rule": "round leaves",
                              "why": "stub"}], "summary": "stub review"}, 0.0
    n = sum(s["name"] == "ChecklistRevision" for s in SPARK)
    return {"checklist": f"Checklist revision {n}: feathery = carrot.",
            "what_changed": f"stub change {n}"}, 0.0


def fake_analyze(image, *, progress=None, overlay_path=None, refs=None,
                 farmer_notes=None, field_model=None, **_kw) -> VisionResult:
    """6 square plants on a grid; weeds are sprayed. Records what it got."""
    CALLS.append({"image": Path(image).name, "refs": list(refs or []),
                  "notes": farmer_notes, "model": field_model})
    img = Image.open(image).convert("RGB")
    w, h = img.size
    for stage, info in (("exg", {"threshold": 60.0}),
                        ("sam", {"objects": 6, "seconds": 0.1}),
                        ("judge", {"batch": 1, "batches": 1})):
        if progress:
            progress(stage, info)
        time.sleep(STAGE_DELAY)
    if overlay_path:
        img.save(overlay_path)
    objects, targets = [], []
    for i, lab in enumerate(FAKE_LABELS):
        x1, y1 = 60 + (i % 3) * 380, 80 + (i // 3) * 420
        box = (x1, y1, x1 + 300, y1 + 300)
        m = np.zeros((h, w), bool)
        m[y1:y1 + 300, x1:x1 + 300] = True
        objects.append({"id": i, "mask": m, "box": box})
        spray = lab == "weed"
        targets.append(Target(
            id=i, x=x1 + 150 if spray else None, y=y1 + 150 if spray else None,
            area=90000 if spray else 0, label=lab, confidence=0.9,
            reason="stub", action="spray" if spray else "protect",
            decided_by="spark_085" if spray else "not_weed"))
    if progress:
        progress("decide", {"plants": len(targets)})
    return VisionResult(targets=targets, field_notes={}, judge_ok=True,
                        exg_threshold=60.0, n_sam_objects=6, n_extra_objects=0,
                        row=None, sam_seconds=0.1, sam_cached=True,
                        spark_seconds=0.1, spark_usage={}, spark_usd=0.0,
                        objects=objects)


# ---------- fixtures ----------

@pytest.fixture(scope="session")
def workdir(tmp_path_factory) -> Path:
    """Fresh app working folder + 12 distinct test photos."""
    d = tmp_path_factory.mktemp("openweeder")
    (d / "fields").mkdir()
    (d / "crops").mkdir()
    (d / "out").mkdir()
    src = sorted(p for p in (ROOT / "tests" / "images").glob("carrot_field_*.png")
                 if "truth" not in p.name)
    photos = d / "photos"
    photos.mkdir()
    for p in src:
        shutil.copy(p, photos / p.name)
        im = Image.open(p)
        ImageOps.mirror(im).save(photos / f"mirror_{p.name}")
        ImageOps.flip(im).save(photos / f"flip_{p.name}")
        im.rotate(180).save(photos / f"rot_{p.name}")
    return d


@pytest.fixture(scope="session")
def base_url(workdir) -> str:
    old_cwd = os.getcwd()
    os.chdir(workdir)
    real, real_spark = vision.analyze, setup.spark_json
    vision.analyze = fake_analyze
    setup.spark_json = fake_spark_json
    flask_app = appmod.create_app(cache_dir=workdir / "out" / "cache",
                                  log_path=workdir / "out" / "api_calls.log")
    srv = make_server("127.0.0.1", 0, flask_app, threaded=True)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()
    vision.analyze, setup.spark_json = real, real_spark
    os.chdir(old_cwd)


@pytest.fixture(scope="session")
def browser():
    with sync_playwright() as p:
        b = p.chromium.launch(channel="msedge",
                              headless=not os.environ.get("UI_HEADED"))
        yield b
        b.close()


@pytest.fixture()
def page(browser, base_url):
    ctx = browser.new_context(viewport={"width": 1440, "height": 900},
                              base_url=base_url)
    pg = ctx.new_page()
    pg.errors = []
    pg.on("pageerror", lambda e: pg.errors.append(str(e)))
    pg.on("console", lambda m: pg.errors.append(m.text)
          if m.type == "error" and "favicon" not in m.text else None)
    pg.on("dialog", lambda d: d.accept())  # confirm() on Remove / Replace all
    yield pg
    ctx.close()


def photos(workdir: Path) -> list[Path]:
    return sorted((workdir / "photos").iterdir())


# ---------- the farmer flow (runs first, builds the field the rest use) ----------

FIELD = "ui-plot"


def test_01_create_field(page: Page):
    page.goto("/")
    page.locator('.ow-side a[href="/crop/new"]').click()
    expect(page).to_have_url(re.compile(r"/crop/new$"))
    page.fill("input[name=name]", FIELD)
    page.fill("input[name=crop]", "carrot")
    page.check("input[name=planting_style][value=double]")
    page.get_by_text("Continue to Photo Guide").click()
    expect(page).to_have_url(re.compile(rf"/field/{FIELD}/guide$"))
    assert not page.errors, page.errors


def test_02_guide_checklist_to_upload(page: Page):
    page.goto(f"/field/{FIELD}/guide")
    for box in page.locator("aside input[type=checkbox]").all():
        box.check()
    page.get_by_text("I'm Ready to Upload").click()
    expect(page).to_have_url(re.compile(rf"/field/{FIELD}/upload$"))
    assert not page.errors, page.errors


def test_03_upload_remove_replace(page: Page, workdir):
    ph = photos(workdir)
    page.goto(f"/field/{FIELD}/upload")
    # Upload picker submits on choose.
    # A new field: 10 photos, the farmer labels 3, the AI is tested on 7.
    page.set_input_files("#uploadform input[type=file]", [str(p) for p in ph[:10]])
    expect(page.locator("[data-remove]")).to_have_count(10)
    expect(page.locator("#uploadform")).to_have_count(0)  # full: no add box
    # Remove one -> 9 left and the upload box returns.
    page.locator("[data-remove]").first.click()
    expect(page.locator("[data-remove]")).to_have_count(9)
    # Re-upload button opens the add picker when there is room.
    with page.expect_file_chooser() as fc:
        page.click("#reupload-btn")
    fc.value.set_files(str(ph[10]))
    expect(page.locator("[data-remove]")).to_have_count(10)
    # Replace one photo in place.
    first = page.locator("[data-replace]").first.get_attribute("data-replace")
    with page.expect_file_chooser() as fc:
        page.locator("[data-replace]").first.click()
    fc.value.set_files(str(ph[11]))
    expect(page.locator(f'[data-replace="{first}"]')).to_have_count(0)
    expect(page.locator("[data-remove]")).to_have_count(10)
    # With all 10 photos, Re-upload replaces the whole set (after confirm).
    with page.expect_file_chooser() as fc:
        page.click("#reupload-btn")
    fc.value.set_files([str(p) for p in ph[:10]])
    expect(page.locator("[data-remove]")).to_have_count(10)
    assert not page.errors, page.errors


def test_03b_old_empty_field_stays_on_10_photos(page: Page, workdir):
    """A field made before plans existed (no plan saved) that was empty must
    stay on 10 photos / pick 3 after its photos arrive."""
    old = workdir / "fields" / "old-empty"
    old.mkdir()
    (old / "field.json").write_text('{"crop": "carrot", "selected": []}')
    page.goto("/field/old-empty/upload")
    page.set_input_files("#uploadform input[type=file]",
                         [str(p) for p in photos(workdir)[:10]])
    expect(page.locator("[data-remove]")).to_have_count(10)
    expect(page.get_by_text("OF 10", exact=True)).to_be_visible()
    picks = page.locator("input[name=pick]")
    for i in range(3):
        picks.nth(i).check()
    expect(page.locator("#sel-count")).to_have_text("3")
    expect(page.locator("#proceed-btn")).to_be_enabled()
    expect(picks.nth(3)).to_be_disabled()
    page.reload()  # and it survives a reload
    expect(page.get_by_text("OF 10", exact=True)).to_be_visible()
    assert not page.errors, page.errors


def test_04_pick_three_and_label(page: Page):
    page.goto(f"/field/{FIELD}/upload")
    go = page.locator("#proceed-btn")
    expect(go).to_be_disabled()
    picks = page.locator("input[name=pick]")
    picks.nth(0).check()
    picks.nth(1).check()
    expect(go).to_be_disabled()  # 2 is not enough
    picks.nth(2).check()
    expect(page.locator("#sel-count")).to_have_text("3")
    expect(picks.nth(3)).to_be_disabled()  # max 3
    go.click()
    page.wait_for_url(re.compile(r"/label/"), timeout=30000)
    for photo_no in (1, 2, 3):
        boxes = page.locator('.plant-box[data-view="d"]')
        expect(boxes).to_have_count(len(FAKE_LABELS))
        changed = page.locator("#d-m-changed")
        # Tap -> label changes; undo restores; redo re-applies.
        boxes.nth(0).click()
        expect(changed).to_contain_text("1")
        page.click("#tb-undo")
        expect(changed).to_contain_text("0")
        page.click("#tb-redo")
        expect(changed).to_contain_text("1")
        # Zoom + pan.
        page.click("#tb-zoomin")
        expect(page.locator("#tb-zoom")).to_have_text("125%")
        page.click("#tb-pan")
        expect(page.locator("#tb-pan")).to_have_attribute("aria-pressed", "true")
        boxes.nth(1).click()  # pan mode: must NOT relabel
        expect(changed).to_contain_text("1")
        page.click("#tb-pan")
        page.click("#tb-zoomout")
        expect(page.locator("#tb-zoom")).to_have_text("100%")
        # Inspector buttons assign directly.
        page.click("#d-btn-weed")
        before = page.url
        page.click("#d-save-btn")
        if photo_no < 3:
            page.wait_for_url(lambda u: u != before and "/label/" in u,
                              timeout=15000)
    page.wait_for_url(re.compile(rf"/field/{FIELD}/learning$"), timeout=15000)
    assert not page.errors, page.errors


def test_05_learning_runs_live_then_results(page: Page):
    """Step 4 A+B: started by itself after labeling; the panel updates live."""
    page.goto(f"/field/{FIELD}/learning")
    state = page.locator("#learn-state")
    # No reload: the panel re-renders itself until DONE.
    expect(state).to_have_text("DONE", timeout=60000)
    names = [s["name"] for s in SPARK]
    assert names[0] == "FieldChecklist", names
    assert "ChecklistReview" in names and "ChecklistRevision" in names, names
    study = SPARK[0]["text"]
    assert "separates crop from weed" in study or "no single measured number" in study
    assert "outside the crop band" not in study and "% of weeds" not in study  # lesson 4
    assert SPARK[0]["images"] >= 2  # crop + weed close-ups side by side
    # Checklist versions with their leave-one-out score, best one in use.
    expect(page.get_by_text("IN USE", exact=True)).to_be_visible()
    expect(page.get_by_text("BEST SCORE", exact=True)).to_be_visible()
    expect(page.locator("details[data-key^=v]")).to_have_count(2)
    expect(page.locator("details[data-key^=r]")).to_have_count(2)  # round 2 changed nothing
    expect(page.get_by_text(re.compile("No labels changed"))).to_be_visible()
    expect(page.locator("img[alt^='crop example']").first).to_be_visible()
    log = json.loads((Path("fields") / FIELD / "setup_log.json").read_text())
    rnd = log["setup"]["rounds"][0]
    assert rnd["round"] == 1 and rnd["version"] == 1 and rnd["overlays"], rnd
    assert all(v["loo"]["per_photo"] for v in log["setup"]["versions"])
    # Every judge call during learning carried the checklist + references
    # (CALLS also holds step 3's pre-label calls, which have no model).
    learn = [c for c in CALLS if c["model"]]
    assert len(learn) >= 2 * 7 + 2 * 3, len(learn)  # 2 rounds + 2 scorings
    assert all(c["model"]["checklist"] and c["refs"] for c in learn)
    assert all(c["model"]["band_half_px"] for c in learn)
    page.get_by_role("link", name="Check the results").click()
    expect(page).to_have_url(re.compile(rf"/field/{FIELD}/results$"))
    expect(page.locator("#m-plants")).to_have_text(str(7 * len(FAKE_LABELS)))
    expect(page.locator("#honest")).to_be_visible()
    expect(page.locator("#h-scored")).to_contain_text("Scored on 0 of 7")
    expect(page.locator("input.reviewed")).to_have_count(7)
    assert not page.errors, page.errors


def test_06_fix_autosaves_and_persists(page: Page):
    page.goto(f"/field/{FIELD}/results")
    page.get_by_role("link", name="Fix labels", exact=True).first.click()
    expect(page).to_have_url(re.compile(rf"/field/{FIELD}/fix/"))
    box = page.locator(".pbox[data-id='0']")
    expect(box).to_have_attribute("data-label", "crop")
    box.click()  # on the photo
    expect(box).to_have_attribute("data-label", "weed")
    chip = page.locator(".plant[data-id='0']")
    expect(chip).to_have_attribute("data-label", "weed")  # list follows photo
    page.locator(".plant[data-id='1'] .chip").click()  # from the list
    expect(page.locator(".pbox[data-id='1']")).to_have_attribute("data-label", "skip")
    chip.locator(".why").fill("purslane, thick round leaves")
    page.fill("#photo-note", "grass blades here are weeds")
    expect(page.locator("#save-state")).to_contain_text("All changes saved",
                                                        timeout=5000)
    # Reload: everything is still there (saved without pressing anything).
    page.reload()
    expect(page.locator(".pbox[data-id='0']")).to_have_attribute("data-label", "weed")
    expect(page.locator(".pbox[data-id='1']")).to_have_attribute("data-label", "skip")
    expect(page.locator(".plant[data-id='0'] .why")).to_have_value(
        "purslane, thick round leaves")
    expect(page.locator("#photo-note")).to_have_value("grass blades here are weeds")
    expect(page.locator("#fixed-count")).to_have_text("You fixed 2 plants")
    # AI overlay toggle hides the editable boxes and back.
    page.click("#view-toggle")
    expect(page.locator("#pboxes")).to_have_class(re.compile("hide"))
    page.click("#view-toggle")
    page.click("#save")  # Done
    expect(page).to_have_url(re.compile(rf"/field/{FIELD}/results$"))
    assert not page.errors, page.errors


def test_07_results_cards_show_fixes_and_honest_score(page: Page):
    page.goto(f"/field/{FIELD}/results")
    expect(page.locator(".cbox.changed")).to_have_count(2)
    toggle = page.locator(".ai-toggle").first
    toggle.click()
    expect(toggle).to_have_text("Your labels")
    toggle.click()
    expect(toggle).to_have_text("AI view")
    # Honest score counts only reviewed photos. On the fixed photo: plant 0
    # AI crop -> farmer weed (missed weed), plant 1 -> skip (not scored),
    # plant 3 unsure left alone (not scored): weeds 1/2, 3/4 labels right.
    first = page.locator("input.reviewed").first
    first.check()
    expect(page.locator("#h-scored")).to_contain_text("Scored on 1 of 7")
    expect(page.locator("#h-hits")).to_have_text("0")
    expect(page.locator("#h-weeds")).to_have_text("1/2")
    expect(page.locator("#h-acc")).to_have_text("75%")
    page.reload()  # persisted
    expect(page.locator("input.reviewed").first).to_be_checked()
    expect(page.locator("#h-scored")).to_contain_text("Scored on 1 of 7")
    second = page.locator("input.reviewed").nth(1)
    second.check()
    expect(page.locator("#h-scored")).to_contain_text("Scored on 2 of 7")
    second.uncheck()
    expect(page.locator("#h-scored")).to_contain_text("Scored on 1 of 7")
    assert not page.errors, page.errors


def test_08_rerun_with_note_teaches(page: Page):
    page.goto(f"/field/{FIELD}/results")
    CALLS.clear()
    SPARK.clear()
    page.fill("#run-note", "broad round leaves are always weeds")
    page.get_by_role("button", name="Re-run with my fixes").click()
    # Live: elapsed ticks, the log says the checklist is being revised, cards
    # go Running -> Done (green) without reloads.
    elapsed = page.locator("#run-elapsed")
    expect(elapsed).to_be_visible()
    first = elapsed.inner_text()
    expect(elapsed).not_to_have_text(first, timeout=5000)
    expect(page.locator("#run-log")).to_contain_text("checklist", timeout=10000)
    expect(page.locator("#run-log")).to_contain_text("photo 1/7", timeout=15000)
    expect(page.locator("[id^=chip-]").first).to_contain_text("Done", timeout=15000)
    expect(page.locator("[id^=chip-]").first).to_have_class(re.compile(r"2E9E5B"))
    expect(page.get_by_role("button", name=re.compile("Re-run"))).to_be_visible(timeout=40000)
    # C: his fixes + notes rewrote the checklist before judging.
    rev = [s for s in SPARK if s["name"] == "ChecklistRevision"]
    assert len(rev) == 1, [s["name"] for s in SPARK]
    assert "FARMER corrected" in rev[0]["text"] and "farmer says weed" in rev[0]["text"]
    assert "purslane, thick round leaves" in rev[0]["text"]
    assert "broad round leaves are always weeds" in rev[0]["text"]
    assert rev[0]["images"] >= 1  # close-up of the fixed plant
    assert all(c["model"]["checklist"].startswith("Checklist revision") for c in CALLS)
    # Honest score did not move: it is the learning run's labels, before fixes.
    expect(page.locator("#h-weeds")).to_have_text("1/2")
    # Unfixed photos show this round's new detection image (fast JPEG,
    # never served from a stale browser copy); fixed ones show your labels.
    imgs = page.locator("img[id^=img-]")
    srcs = [imgs.nth(i).get_attribute("src") for i in range(imgs.count())]
    fresh = [s for s in srcs if s.startswith("/run_overlay/") and s.endswith("?r=2")]
    assert len(fresh) == len(srcs) - 1, srcs
    r = page.request.get(fresh[0])
    assert r.ok and r.headers["content-type"] == "image/jpeg", r.headers
    assert r.headers.get("cache-control") == "no-cache", r.headers
    assert len(r.body()) < 600_000, len(r.body())
    r = page.request.get(page.locator("img[src^='/p/']").first.get_attribute("src"))
    assert r.headers.get("cache-control") == "no-cache", r.headers
    run_calls = [c for c in CALLS if c["notes"]]  # test photos (not the scoring)
    assert len(run_calls) == 7, [c["image"] for c in CALLS]
    notes = run_calls[0]["notes"]
    assert "broad round leaves are always weeds" in notes
    assert "grass blades here are weeds" in notes
    refs = run_calls[0]["refs"]
    assert any(len(r) == 3 and r[2] == "purslane, thick round leaves"
               for r in refs), refs
    expect(page.get_by_text(re.compile("farmer notes"))).to_be_visible()
    page.get_by_text(re.compile("Your notes the AI reads")).click()
    expect(page.get_by_text("Round 2: broad round leaves are always weeds")).to_be_visible()
    assert not page.errors, page.errors


def test_09_confirm_ready_and_raw_log(page: Page):
    page.goto(f"/field/{FIELD}/learning")  # the fix-taught version is listed
    expect(page.get_by_text("from your fixes")).to_be_visible()
    page.goto(f"/field/{FIELD}/results")
    page.get_by_role("button", name=re.compile("Confirm Results")).click()
    expect(page).to_have_url(re.compile(rf"/field/{FIELD}/ready$"))
    r = page.request.get(f"/field/{FIELD}/setup_log.json")
    assert r.ok and r.json()["setup_log"]["done"] is True
    # D: model.json saved and shown.
    model = json.loads((Path("fields") / FIELD / "model.json").read_text())
    assert model["checklist"] and model["checklist_version"] >= 1
    assert model["honest_score"]["scored_on"] == 1
    assert model["honest_score"]["weeds_hit"] == 1
    assert model["references"]["crop"] and model["band"]
    assert all(Path(f).is_file() for f in model["references"]["crop"])
    expect(page.get_by_text(re.compile(r"Field model \(saved"))).to_be_visible()
    expect(page.get_by_text("Checklist the robot will use")).to_be_visible()
    assert not page.errors, page.errors


# ---------- navigation ----------

PAGES = {  # path -> active sidebar item
    "/": "Home",
    "/crop/new": "New crop",
    "/crop/train": "Train new crop",
    "/train": "Train model",
    "/run-field": "Run on the field",
    "/fields": "Fields",
    "/fields/new": "New field",
    f"/field/{FIELD}": "Overview",
    f"/field/{FIELD}/guide": "Guide",
    f"/field/{FIELD}/upload": "Upload",
    f"/field/{FIELD}/learning": "Learning",
    f"/field/{FIELD}/results": "Results",
    f"/field/{FIELD}/ready": "Ready",
}


@pytest.mark.parametrize("path,active", PAGES.items())
def test_nav_active_and_no_js_errors(page: Page, path, active):
    page.goto(path)
    expect(page.locator(".ow-side")).to_have_count(1)
    on = page.locator(".ow-side .ow-link.on")
    expect(on).to_have_count(1)
    expect(on).to_contain_text(active)
    assert not page.errors, page.errors


def test_nav_links_go_where_they_say(page: Page):
    page.goto(f"/field/{FIELD}/results")
    for name, url in (("Overview", rf"/field/{FIELD}$"),
                      ("Guide", rf"/field/{FIELD}/guide$"),
                      ("Learning", rf"/field/{FIELD}/learning$"),
                      ("Ready", rf"/field/{FIELD}/ready$"),
                      ("Home", r"/$")):
        page.locator(".ow-side .ow-link", has_text=name).first.click()
        expect(page).to_have_url(re.compile(url))
        page.goto(f"/field/{FIELD}/results")


def test_nav_home_sections_follow_hash(page: Page):
    on = page.locator(".ow-side .ow-link.on")
    page.goto("/#crops")  # opened directly
    expect(on).to_have_count(1)
    expect(on).to_contain_text("Crops")
    page.locator(".ow-side [data-nav=home]").click()
    expect(page).to_have_url(re.compile(r"/$"))
    expect(on).to_contain_text("Home")
    page.go_back()  # back to /#crops
    expect(on).to_contain_text("Crops")


def test_mobile_menu_drawer(browser, base_url):
    ctx = browser.new_context(viewport={"width": 390, "height": 844},
                              base_url=base_url, has_touch=True)
    pg = ctx.new_page()
    pg.goto(f"/field/{FIELD}/results")
    side = pg.locator(".ow-side")
    assert side.bounding_box()["x"] < 0  # off-screen
    pg.locator(".ow-mbar button").click()
    pg.wait_for_timeout(300)
    assert side.bounding_box()["x"] == 0
    pg.mouse.click(380, 400)  # tap the dimmed area
    pg.wait_for_timeout(300)
    assert side.bounding_box()["x"] < 0
    # Page fits the phone: no sideways scrolling.
    assert pg.evaluate("document.documentElement.scrollWidth") <= 390 + 1
    ctx.close()


def test_home_filters(page: Page):
    page.goto("/")
    cards = page.locator(".field-card")
    total = cards.count()
    assert total >= 1
    page.locator(".field-filter[data-filter=ready]").click()
    expect(page.locator(".field-card:not(.hidden)")).to_have_count(1)  # ui-plot
    page.locator(".field-filter[data-filter=running]").click()
    expect(page.locator(".field-card:not(.hidden)")).to_have_count(0)
    expect(page.locator("#filter-empty")).to_be_visible()
    page.locator(".field-filter[data-filter=all]").click()
    expect(page.locator(".field-card:not(.hidden)")).to_have_count(total)


# ---------- no dead buttons anywhere ----------

DEAD_JS = """
() => [...document.querySelectorAll('button, a')].filter(el => {
  if (!el.offsetParent && getComputedStyle(el).position !== 'fixed') return false;
  if (el.disabled || el.getAttribute('aria-disabled') === 'true') return false;
  if (el.tagName === 'A') {
    const h = el.getAttribute('href');
    return !h || h === '#';
  }
  if (el.getAttribute('onclick')) return false;
  if (el.form && (el.type || 'submit') === 'submit') return false;
  return !el.__hasClick;
}).map(el => (el.id ? '#' + el.id + ' ' : '') + el.textContent.trim().slice(0, 40))
"""

# Records every element that gets a click listener, before page scripts run.
TRACK_CLICKS = """
const _add = EventTarget.prototype.addEventListener;
EventTarget.prototype.addEventListener = function (type, fn, opts) {
  if (type === 'click' || type === 'pointerdown') this.__hasClick = true;
  return _add.call(this, type, fn, opts);
};
"""


def test_dead_button_detector_catches_one(page: Page):
    page.add_init_script(TRACK_CLICKS)
    page.goto("/")
    page.evaluate("""() => {
      const b = document.createElement('button'); b.textContent = 'planted dead';
      const a = document.createElement('a'); a.href = '#'; a.textContent = 'planted link';
      const ok = document.createElement('button'); ok.textContent = 'planted live';
      ok.addEventListener('click', () => {});
      document.querySelector('main').append(b, a, ok);
    }""")
    dead = page.evaluate(DEAD_JS)
    assert "planted dead" in dead and "planted link" in dead, dead
    assert "planted live" not in dead, dead


@pytest.mark.parametrize("path", [p for p in PAGES] + ["FIX", "LABEL"])
def test_no_dead_buttons(page: Page, path):
    page.add_init_script(TRACK_CLICKS)
    if path == "FIX":
        page.goto(f"/field/{FIELD}/results")
        page.get_by_role("link", name="Fix labels", exact=True).first.click()
    elif path == "LABEL":
        page.goto(f"/field/{FIELD}/upload")  # the 2 picks are remembered
        page.click("#proceed-btn")
        page.wait_for_url(re.compile(r"/label/"), timeout=30000)
    else:
        page.goto(path)
    page.wait_for_load_state("networkidle")
    dead = page.evaluate(DEAD_JS)
    assert not dead, f"buttons/links that do nothing on {page.url}: {dead}"


# ---------- crops, train new crop, fields ----------

def test_new_crop_page_is_called_new_crop(page: Page):
    page.goto("/crop/new")
    expect(page).to_have_title(re.compile("New crop"))
    expect(page.get_by_text("+ New crop")).to_be_visible()
    assert page.locator("form").get_attribute("action") == "/crop/new"
    page.goto("/field/new")            # the old address still works
    expect(page.get_by_text("+ New crop")).to_be_visible()


def test_train_new_crop_is_a_disabled_placeholder(page: Page):
    page.goto("/crop/train")
    expect(page.get_by_text("NOT BUILT YET")).to_be_visible()
    expect(page.locator("#train-start")).to_be_disabled()
    expect(page.locator("#train-crop")).to_be_disabled()


def test_new_field_uses_a_trained_crop(page: Page):
    page.goto("/fields/new")
    choice = page.locator(f"input[name=crop_name][value={FIELD}]")
    expect(choice).to_be_enabled()          # test_09 saved its model.json
    page.fill("input[name=name]", "ui-ground")
    page.fill("input[name=columns]", "4")
    choice.check()
    page.click("#farm-save")
    expect(page).to_have_url(re.compile(r"/fields/ui-ground$"))
    expect(page.get_by_text("4 columns to weed")).to_be_visible()
    expect(page.get_by_role("link", name=FIELD, exact=True)).to_be_visible()
    page.goto("/fields")
    expect(page.locator("[data-field=ui-ground]")).to_be_visible()
    assert not page.errors, page.errors


def test_new_field_refuses_an_untrained_crop(page: Page):
    Path("fields", "ui-untrained").mkdir(exist_ok=True)
    Path("fields", "ui-untrained", "field.json").write_text(
        json.dumps({"crop": "pea", "planting_style": "single", "selected": []}))
    try:
        page.goto("/fields/new")
        expect(page.locator("input[name=crop_name][value=ui-untrained]")).to_be_disabled()
        page.request.post("/fields/new", form={"name": "bad-one", "columns": "3",
                                               "crop_name": "ui-untrained"})
        page.goto("/fields")
        expect(page.locator("[data-field=bad-one]")).to_have_count(0)
    finally:
        import shutil
        shutil.rmtree(Path("fields", "ui-untrained"), ignore_errors=True)


# ---------- train model / run on the field (fast: own tiny crop, no farmer flow) ----------

@pytest.fixture()
def tcrop(workdir):
    """A finished crop folder made on the spot (field.json + model.json)."""
    name = "ui-quick"
    d = workdir / "fields" / name
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    (d / "field.json").write_text(json.dumps({"crop": "pea", "planting_style": "single",
                                              "selected": []}))
    (d / "model.json").write_text("{}")
    yield name
    shutil.rmtree(d, ignore_errors=True)


def test_train_model_picks_only_finished_crops(page: Page, tcrop):
    page.goto("/train")
    expect(page.locator(f"a[data-crop={tcrop}]")).to_be_visible()
    page.goto("/train/not-a-crop")
    expect(page.get_by_text("Pick a crop that has finished")).to_be_visible()


def test_train_model_page_flow_without_api(page: Page, tcrop, workdir):
    page.goto(f"/train/{tcrop}")
    page.click("#rec-start")                       # the blank robot function
    expect(page.locator("#rec-msg")).to_contain_text("No robot is connected")
    page.click("#rec-stop")
    expect(page.locator("#fit-btn")).to_be_disabled()
    expect(page.locator("#fit-msg")).to_contain_text("at least 4 labeled")
    imgs = sorted((workdir / "photos").iterdir())[:3]
    page.set_input_files("#imp-files", [str(p) for p in imgs])
    page.click("#imp-btn")
    page.wait_for_load_state("networkidle")
    expect(page.locator("#summary")).to_contain_text("3 images, 0 labeled")
    assert page.locator("#thumbs img").count() == 3
    expect(page.locator("#label-msg")).to_contain_text("3 images to label")
    r = page.request.post(f"/api/train/{tcrop}/fit", form={"epochs": "30"})
    assert r.status == 400 and "at least 4" in r.json()["error"]
    assert page.request.post("/api/train/nope/label").status == 404
    assert not page.errors, page.errors


def test_train_new_crop_offers_to_continue_an_existing_dataset(page: Page, tcrop, workdir):
    d = workdir / "fields" / tcrop / "train" / "images"
    d.mkdir(parents=True)
    for i in (1, 2):
        Image.new("RGB", (8, 8)).save(d / f"img_{i:05d}.jpg")
    page.goto("/crop/train")
    row = page.locator(f"[data-existing={tcrop}]")
    expect(row).to_contain_text("2 images, 0 labeled")
    row.get_by_role("link", name="Continue").click()
    expect(page).to_have_url(re.compile(rf"/train/{tcrop}$"))
    expect(page.locator("#summary")).to_contain_text("2 images")


def test_run_field_advises_sam_muse_below_map50_0_7(page: Page, tcrop, workdir):
    page.goto("/run-field")
    opt = page.locator(f".opt[data-crop={tcrop}]")
    expect(opt.locator(".advice")).to_contain_text("No trained YOLO model")   # finished setup, no YOLO yet
    assert page.locator("input[name=mode][value=sam_muse]").is_checked()
    assert page.locator("input[name=mode][value=yolo]").is_disabled()
    y = workdir / "fields" / tcrop / "yolo"
    y.mkdir()
    (y / "best.pt").write_bytes(b"x")                                          # only its presence matters
    for map50, use in ((0.47, "sam_muse"), (0.82, "yolo")):
        (y / "metrics.json").write_text(json.dumps({"best": {"map50": map50}}))
        page.goto("/run-field")
        expect(page.locator(f".opt[data-crop={tcrop}] .advice")).to_contain_text(f"mAP50 {map50:.2f}")
        assert page.locator(f"input[name=mode][value={use}]").is_checked(), (map50, use)
        assert page.locator(f".opt[data-crop={tcrop}] .advice").get_attribute("data-use") == use
    expect(page.locator(".advice").first).to_contain_text("YOLO model is good enough")
    assert not page.errors, page.errors


def test_run_field_start_stop_are_blank_in_both_modes(page: Page, tcrop, workdir):
    y = workdir / "fields" / tcrop / "yolo"
    y.mkdir()
    (y / "best.pt").write_bytes(b"x")
    page.goto("/run-field")
    expect(page.locator(f"input[name=model][value={tcrop}]")).to_be_checked()
    for m in ("yolo", "sam_muse"):
        page.check(f"input[name=mode][value={m}]")
        page.click("#go")
        expect(page.locator("#msg")).to_contain_text("No robot is connected")
        page.click("#halt")
        expect(page.locator("#msg")).to_contain_text("Stopped")
    shutil.rmtree(y)
    assert page.request.post(f"/api/field_run/start/{tcrop}?mode=yolo").status == 404   # no model: refused
    assert page.request.post(f"/api/field_run/start/{tcrop}?mode=bogus").status == 404
    assert not page.errors, page.errors


def test_run_field_sam_muse_panel_shows_measured_numbers_and_estimate(page: Page, tcrop, workdir):
    (workdir / "fields" / tcrop / "detector_compare.json").write_text(json.dumps({
        "photos": 3, "at": "2026-11-04 12:00", "variation": "Muse varies.", "note": "n.",
        "2": {"weeds_found": 20, "weeds_total": 44, "weeds_found_pct": 45.5, "crop_hits": 1,
              "seconds_each": [31.0, 35.4, 39.5]}}))
    page.goto("/run-field")
    expect(page.locator("#sm-measured")).to_contain_text("20/44 (45.5%)")
    expect(page.locator("#sm-measured")).to_contain_text("crop hits 1")
    expect(page.locator("#sm-measured")).to_contain_text("35.4 s")
    page.fill("#rowlen", "100")
    page.fill("#step", "19")
    expect(page.locator("#sm-est")).to_contain_text("527 pictures")      # ceil(10000 / 19)
    page.fill("#step", "10")
    expect(page.locator("#sm-est")).to_contain_text("1000 pictures")
    assert not page.errors, page.errors

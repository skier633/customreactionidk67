# the air pen with a made-up pinching hand
import importlib
import math

import numpy as np
import pytest

import settings
from fake_hands import pen_grip_points


@pytest.fixture
def pen():
    import pen
    return importlib.reload(pen)


def hand(tip, pinched, ready_made=("None", 0.5), fingers_curled=True):
    # a pen grip: index out, middle/ring/pinky curled in, thumb touching the index tip when pinched
    world = pen_grip_points()
    if not fingers_curled:  # relaxed hand: other fingers straight
        for k, y in [(9, 0.02), (13, 0.04), (17, 0.06)]:
            world[k:k + 4] = [(0, y, 0), (0.03, y, 0), (0.055, y, 0), (0.075, y, 0)]
    if pinched:
        world[4] = world[8] + (0.01, 0, 0)
    # screen points: the same hand shape, placed so the index tip is at `tip` (1500 pixels per meter)
    pts = np.asarray(tip, float) + (world - world[8])[:, :2] * 1500
    return {"label": "Right", "pts": pts, "world": world, "ready_made": ready_made}


def draw_circle(pen, cfg, t=0.0, n=40):
    for i in range(n):
        a = i / (n - 1) * 2 * math.pi
        pen.update([hand((640 + 100 * math.cos(a), 360 + 100 * math.sin(a)), True)], t, 1280, 720, cfg)
        t += 1 / 30
    pen.update([hand((640, 360), False)], t, 1280, 720, cfg)
    return t + 1 / 30


def test_pinch_draws_one_stroke(pen):
    cfg = dict(settings.PEN_DEFAULTS)
    draw_circle(pen, cfg)
    assert len(pen.strokes) == 1
    assert len(pen.strokes[0]["points"]) > 20
    frame = np.zeros((720, 1280, 3), np.uint8)
    pen.draw(frame, 2.0, cfg)
    assert frame.any()


def test_no_pinch_no_ink(pen):
    cfg = dict(settings.PEN_DEFAULTS)
    for i in range(30):
        pen.update([hand((600 + i * 5, 300), False)], i / 30, 1280, 720, cfg)
    assert pen.strokes == []


def test_relaxed_fingers_dont_draw(pen):
    # the accidental pinches in the real recordings all had the other fingers loose
    cfg = dict(settings.PEN_DEFAULTS)
    for i in range(30):
        pen.update([hand((600 + i * 5, 300), True, fingers_curled=False)], i / 30, 1280, 720, cfg)
    assert pen.strokes == []


def test_any_grip_setting_allows_relaxed_fingers(pen):
    cfg = dict(settings.PEN_DEFAULTS, grip="any")
    for i in range(30):
        pen.update([hand((600 + i * 5, 300), True, fingers_curled=False)], i / 30, 1280, 720, cfg)
    assert len(pen.strokes) == 1


def test_quick_pinch_doesnt_draw(pen):
    cfg = dict(settings.PEN_DEFAULTS)
    pen.update([hand((600, 300), True)], 0.0, 1280, 720, cfg)
    pen.update([hand((620, 300), True)], 1 / 30, 1280, 720, cfg)   # 2 frames is less than the hold time
    pen.update([hand((640, 300), False)], 2 / 30, 1280, 720, cfg)
    assert pen.strokes == []


def test_line_starts_where_the_pinch_started(pen):
    cfg = dict(settings.PEN_DEFAULTS)
    for i in range(20):
        pen.update([hand((300 + i * 20, 300), True)], i / 30, 1280, 720, cfg)
    assert pen.strokes[0]["points"][0][0] < 300 / 1280 + 0.02


@pytest.mark.parametrize("clear_with,shown,cleared", [
    ("fist", "Closed_Fist", True),        # the default
    ("fist", "Open_Palm", False),
    ("open_palm", "Open_Palm", True),
    ("none", "Closed_Fist", False),
])
def test_clearing(pen, clear_with, shown, cleared):
    cfg = dict(settings.PEN_DEFAULTS, clear_with=clear_with)
    t = draw_circle(pen, cfg)
    for i in range(25):
        pen.update([hand((640, 360), False, (shown, 0.9))], t + i / 30, 1280, 720, cfg)
    assert (pen.strokes == []) == cleared


def test_fade_after_removes_old_ink(pen):
    cfg = dict(settings.PEN_DEFAULTS, fade_after=1)
    t = draw_circle(pen, cfg)
    for i in range(60):
        pen.update([], t + i / 30, 1280, 720, cfg)
    assert pen.strokes == []


def test_turned_off_draws_nothing(pen):
    cfg = dict(settings.PEN_DEFAULTS, enabled=False)
    draw_circle(pen, cfg)
    assert pen.strokes == []


def test_call_sees_the_drawing_the_way_you_do(pen):
    # you write an "L" while looking at your mirrored preview. readable_for_call (the default) means
    # the call gets the ink the same way round as your preview, so the L isnt backwards for them
    cfg = dict(settings.PEN_DEFAULTS, glow=False)
    w, h = 1280, 720
    # in camera space a preview-"L" goes down then to the LEFT (the preview is flipped)
    path = [(900, 200 + i * 20) for i in range(15)] + [(900 - i * 20, 480) for i in range(15)]
    for i, tip in enumerate(path):
        pen.update([hand(tip, True)], i / 30, w, h, cfg)
    camera = np.zeros((h, w, 3), np.uint8)
    preview = camera[:, ::-1].copy()                          # what gesture_cam shows you
    pen.draw(preview, 1.0, cfg, mirror=True)
    call = camera.copy()
    pen.draw(call, 1.0, cfg, mirror=cfg["readable_for_call"])
    assert np.array_equal(preview.any(2), call.any(2))      # same drawing for both of you
    ys, xs = np.nonzero(preview.any(2))
    bottom = xs[ys > 470]
    assert bottom.mean() > xs[ys < 300].mean()                # the L's foot goes right: readable

    follow = camera.copy()                                    # readable_for_call off: ink follows your hand for them
    pen.draw(follow, 1.0, cfg, mirror=False)
    a, b = follow.any(2), preview.any(2)[:, ::-1]
    assert (a & b).sum() / (a | b).sum() > 0.9               # same line (give or take edge pixels)


def test_clearing_while_drawing_keeps_going(pen):
    cfg = dict(settings.PEN_DEFAULTS)
    for i in range(10):
        pen.update([hand((300 + i * 10, 300), True)], i / 30, 1280, 720, cfg)
    pen.clear()   # C key or the studio button, mid-line
    for i in range(10, 20):
        pen.update([hand((300 + i * 10, 300), True)], i / 30, 1280, 720, cfg)
    assert len(pen.strokes) == 1 and len(pen.strokes[0]["points"]) > 3


def test_turning_off_clears_the_ink(pen):
    cfg = dict(settings.PEN_DEFAULTS)
    t = draw_circle(pen, cfg)
    off = dict(cfg, enabled=False)
    frame = np.zeros((720, 1280, 3), np.uint8)
    pen.draw(frame, t, off)
    assert not frame.any()                 # hidden right away
    pen.update([], t + 0.1, 1280, 720, off)
    assert pen.strokes == []               # and gone

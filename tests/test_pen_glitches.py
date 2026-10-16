# the air pen should survive tracking glitches: your real drawing clip (4 drawings) with made-up
# glitches added, like mediapipe putting the index tip on a knuckle or losing the hand for a frame
import copy
import importlib
import os

import numpy as np
import pytest

import settings
from replay import load_clip

CLIP = os.path.join(os.path.dirname(__file__), "clips", "pen_draw.npz")
pytestmark = pytest.mark.skipif(not os.path.exists(CLIP), reason="needs tests/clips/pen_draw.npz")


def draw(frames, w, h):
    import pen
    pen = importlib.reload(pen)
    cfg = dict(settings.PEN_DEFAULTS)
    for t, hands in frames:
        pen.update(hands, t, w, h, cfg)
    return [np.array(s["points"]) * [w, h] for s in pen.strokes]


def add_glitches(frames, rng, kind):
    out = copy.deepcopy(frames)
    pinched = [i for i, (_, hands) in enumerate(out) if hands and
               np.linalg.norm(hands[0]["world"][4] - hands[0]["world"][8]) / np.linalg.norm(hands[0]["world"][9] - hands[0]["world"][0]) < 0.4]
    for i in rng.choice(pinched, 6, replace=False):
        for k in range(rng.integers(1, 4 if kind == "knuckle" else 3)):
            if i + k >= len(out) or not out[i + k][1]:
                continue
            if kind == "knuckle":   # index tip reported on the index finger's knuckle
                hand = out[i + k][1][0]
                j = rng.choice([5, 6])
                hand["pts"][8] = hand["pts"][j]
                hand["world"][8] = hand["world"][j]
            else:                   # hand lost for a frame or two
                out[i + k] = (out[i + k][0], [])
    return out


def worst_spike(strokes, clean):
    ref = np.concatenate(clean)
    return max(np.min(np.linalg.norm(ref - p, axis=1)) for s in strokes for p in s)


@pytest.mark.parametrize("kind", ["knuckle", "dropout"])
def test_glitches_dont_break_the_drawing(kind):
    _, (h, w), frames = load_clip(CLIP)
    clean = draw(frames, w, h)
    assert len(clean) == 4
    right_count, spikes = 0, []
    for seed in range(10):
        strokes = draw(add_glitches(frames, np.random.default_rng(seed), kind), w, h)
        right_count += len(strokes) == 4
        spikes.append(worst_spike(strokes, clean))
    # before the fix: knuckle glitches gave the wrong number of lines in 70% of runs, spikes ~50px
    assert right_count >= 8, f"only {right_count}/10 runs kept the 4 drawings intact"
    if kind == "knuckle":
        assert np.median(spikes) < 40, f"line jumps too far: median spike {np.median(spikes):.0f}px"


def test_letting_go_is_quick():
    # in the real clip, the pen should lift soon after you start opening your fingers
    # (an earlier version took up to 0.9s, which felt like the pen was stuck)
    import pen
    pen = importlib.reload(pen)
    _, (h, w), frames = load_clip(CLIP)
    cfg = dict(settings.PEN_DEFAULTS)

    def pinch(hand):
        world = hand["world"]
        return np.linalg.norm(world[4] - world[8]) / np.linalg.norm(world[9] - world[0])

    was_down, delays = False, []
    for i, (t, hands) in enumerate(frames):
        pen.update(hands, t, w, h, cfg)
        down = any(p["down"] for p in pen.pens.values())
        if was_down and not down:
            opened = [s for s, (t0, hs) in enumerate(frames[:i + 1]) if s > 0 and hs and frames[s - 1][1]
                      and pinch(frames[s - 1][1][0]) < 0.4 <= pinch(hs[0])]
            delays.append(t - frames[opened[-1]][0])
        was_down = down
    assert len(delays) == 4
    assert max(delays) <= 0.25, f"lifting took {max(delays):.2f}s"

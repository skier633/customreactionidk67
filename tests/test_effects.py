# every effect option runs without crashing, alone and stacked on top of each other
import importlib

import numpy as np
import pytest

import settings

W, H = 640, 360


@pytest.fixture
def effects(project):
    import effects
    effects = importlib.reload(effects)
    effects.load_media({"demo_explosion.gif"})
    return effects


def options():
    picture = "demo_explosion.gif"  # the one picture every copy of the repo has
    combos = [{"gif": picture, "gif_where": w} for w in settings.GIF_WHERE]
    combos += [{"distortion": d, "distortion_where": w, "strength": 100} for d in settings.DISTORTIONS for w in settings.DISTORTION_WHERE]
    combos += [{"filter": f} for f in settings.FILTERS]
    combos += [{"particles": p} for p in settings.PARTICLES]
    combos += [{"shake": 100, "text": "hello"}, {"fade": False, "duration": 0.1}]
    return combos


@pytest.mark.parametrize("cfg", options(), ids=lambda c: "-".join(f"{k}={v}" for k, v in c.items()))
def test_option_runs(effects, cfg):
    frame = np.full((H, W, 3), 100, np.uint8)
    mask = np.zeros((H, W, 1), np.float32)
    face = (np.array([320.0, 160.0]), 120.0)
    effects.trigger("x", settings.effect(cfg), 400, 200, W, H)
    for i in range(10):
        out = effects.update_and_draw(frame.copy(), i / 30, 1 / 30, mask, face)
        effects.draw_text(out)
        assert out.shape == frame.shape


def test_overlapping_effects_and_bursts(effects):
    # this used to crash: picture particles from 2 bursts dying at different times
    for key in ["a", "b", "c"]:
        effects.trigger(key, settings.effect({"gif": "demo_explosion.gif", "picture": "demo_explosion.gif", "picture_where": "burst",
                                              "particles": "fireworks", "distortion": "swirl"}), 300, 200, W, H)
    for i in range(120):
        effects.update_and_draw(np.zeros((H, W, 3), np.uint8), i / 30, 1 / 30)
    assert effects.particles == []


def test_effect_ends_after_its_duration(effects):
    effects.trigger("x", settings.effect({"duration": 0.5, "filter": "gray"}), 0, 0, W, H)
    for i in range(20):
        effects.update_and_draw(np.zeros((H, W, 3), np.uint8), i / 30, 1 / 30)
    assert effects.running == []

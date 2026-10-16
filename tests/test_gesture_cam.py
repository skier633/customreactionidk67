# when a gesture goes off, whether gesture_cam plays it
import settings
from gesture_cam import should_fire


def test_normal_gesture_fires():
    assert should_fire(settings.effect({"particles": "hearts"}), "x", 10.0, {})


def test_switched_off_gesture_doesnt_fire():
    assert not should_fire(settings.effect({"particles": "hearts", "enabled": False}), "x", 10.0, {})


def test_try_it_works_even_when_switched_off():
    assert should_fire(settings.effect({"enabled": False}), "x", 10.0, {}, test=True)


def test_cooldown():
    cfg = settings.effect({"cooldown": 2})
    assert not should_fire(cfg, "x", 10.0, {"x": 9.0})
    assert should_fire(cfg, "x", 10.0, {"x": 7.5})


def test_gesture_with_nothing_set_up_doesnt_fire():
    assert not should_fire(None, "x", 10.0, {})

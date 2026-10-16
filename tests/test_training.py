# the features, augmentation and training for your own gestures
import json

import numpy as np

import features
from fake_hands import pen_grip_points, tilt


def as_vec(points):
    p = points - points[0]
    return (p / np.linalg.norm(p[9] - p[0])).ravel()


def test_tilt_proof_parts_dont_change_when_you_tilt_your_hand():
    hand = as_vec(pen_grip_points())
    tilted = as_vec(tilt(pen_grip_points(), 30, 20))
    a, b = features.pose_features(hand), features.pose_features(tilted)
    shape_only = slice(None, 63)                 # hand turned to face the same way
    bends_and_gaps = slice(63 + 6, None)         # bends, fingertip gaps, spread
    assert np.allclose(a[shape_only], b[shape_only], atol=1e-6)
    assert np.allclose(a[bends_and_gaps], b[bends_and_gaps], atol=1e-6)
    assert not np.allclose(a[63:69], b[63:69])   # but which way it faces does change


def test_augment_makes_different_but_similar_hands():
    rng = np.random.default_rng(0)
    hand = as_vec(pen_grip_points())
    copies = [features.augment(hand, rng) for _ in range(20)]
    assert all(c.shape == (63,) for c in copies)
    assert all(not np.allclose(c, hand) for c in copies)
    assert max(np.abs(c - hand).max() for c in copies) < 1.0


def fake_recordings(folder):
    # "point" vs "flat" vs "none" with some wobble, saved like record.py does
    rng = np.random.default_rng(1)
    flat = pen_grip_points().copy()
    flat[9:21, 1] = flat[9:21, 1] * 0 + np.repeat([0.02, 0.04, 0.06], 4)  # other fingers straight too
    flat[9:21, 0] = np.tile([0, 0.03, 0.055, 0.075], 3)
    shapes = {"point": pen_grip_points(), "flat": flat}
    for name, base in shapes.items():
        samples = [as_vec(tilt(base, rng.uniform(-20, 20), rng.uniform(-20, 20)) + rng.normal(0, 0.002, (21, 3))) for _ in range(150)]
        np.save(folder / "data" / (name + ".npy"), np.array(samples))
    np.save(folder / "data" / "none.npy", np.array([as_vec(rng.normal(0, 0.05, (21, 3))) for _ in range(60)]))


def test_training_learns_and_reports(project):
    import train
    fake_recordings(project)
    train.main()
    import joblib
    model = joblib.load("models/my_gestures.joblib")
    report = json.load(open("models/my_gestures_report.json"))
    assert set(model.classes_) == {"point", "flat", "none"}
    assert report["accuracy"] > 0.9
    assert set(report["gestures"]) == {"point", "flat", "none"}
    # it takes the plain 63 numbers (the features are worked out inside the model)
    assert model.predict([as_vec(tilt(pen_grip_points(), 10, 0))])[0] == "point"


def test_balancing_gives_every_gesture_the_same_amount():
    import train
    rng = np.random.default_rng(0)
    groups = {"a": np.zeros((100, 63)), "b": np.zeros((10, 63))}
    X, y = train.balanced(groups, rng)
    assert (y == "a").sum() == (y == "b").sum()

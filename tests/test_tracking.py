"""Deterministic tracking sequences: no camera or model download needed."""

import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from fake_hands import pen_grip_points
from tracking import HandStabilizer, HandTracker, LandmarkFilter, find_hands, hand_features


def hand(x=300, y=300, label="Left"):
    world = pen_grip_points()
    pts = (world[:, :2] - world[0, :2]) * 800 + [x, y]
    return {"label": label, "pts": pts, "world": world,
            "ready_made": ("Victory", 0.95)}


def test_adaptive_filter_reduces_stationary_jitter():
    filt = LandmarkFilter()
    rng = np.random.default_rng(12)
    positions = np.full((21, 2), 0.5)
    raw, filtered = [], []
    for frame in range(180):
        noisy = positions + rng.normal(0, 0.0025, positions.shape)
        result = filt.update(noisy, frame / 30)
        if frame > 30:
            raw.append(noisy)
            filtered.append(result)
    assert np.std(filtered) < np.std(raw) * 0.65


def test_adaptive_filter_follows_fast_movement():
    filt = LandmarkFilter()
    filt.update(np.full((21, 2), 0.2), 0)
    for frame in range(1, 31):
        position = np.full((21, 2), 0.2 + frame * 0.015)
        result = filt.update(position, frame / 30)
    # Less than two frames of lag at a fast 0.45 frame-widths / second.
    assert np.max(np.abs(result - position)) < 0.03


def test_filter_resets_instead_of_dragging_from_stale_position():
    filt = LandmarkFilter()
    filt.update(np.full((21, 2), 0.1), 0)
    current = np.full((21, 2), 0.8)
    np.testing.assert_array_equal(filt.update(current, 0.2), current)
    np.testing.assert_array_equal(filt.update(current + 0.1, 0.1), current + 0.1)


def test_identity_survives_reordered_results_and_handedness_flicker():
    tracker = HandStabilizer()
    initial = tracker.update([hand(200, label="Left"), hand(800, label="Right")], 0, 1280, 720)
    result = tracker.update([hand(810, label="Left"), hand(210, label="Right")], 1 / 30, 1280, 720)
    assert [h["track_id"] for h in result] == [h["track_id"] for h in initial]
    assert [h["label"] for h in result] == ["Left", "Right"]
    assert result[0]["pts"][0, 0] < 300
    assert result[1]["pts"][0, 0] > 700


def test_motion_matching_preserves_identity_when_hands_cross():
    tracker = HandStabilizer()
    for frame in range(15):
        left_x, right_x = 300 + frame * 30, 700 - frame * 30
        detections = [hand(left_x, label="Left"), hand(right_x, label="Right")]
        if 5 <= frame <= 10:
            # The overlap can make both handedness estimates agree temporarily.
            detections[1]["label"] = "Left"
        if frame % 2:
            detections.reverse()
        result = tracker.update(detections, frame / 30, 1280, 720)
    assert result[0]["label"] == "Left"
    assert result[0]["track_id"] == 0
    assert result[0]["pts"][0, 0] > result[1]["pts"][0, 0]


def test_duplicate_handedness_has_unique_stable_labels():
    tracker = HandStabilizer()
    first = tracker.update([hand(200), hand(800)], 0, 1280, 720)
    second = tracker.update([hand(790), hand(210)], 1 / 30, 1280, 720)
    assert {h["label"] for h in first} == {"Left", "Right"}
    assert [h["label"] for h in second] == [h["label"] for h in first]
    assert second[0]["pts"][0, 0] < second[1]["pts"][0, 0]


def test_missing_hands_are_not_fabricated_and_long_loss_gets_new_id():
    tracker = HandStabilizer()
    first = tracker.update([hand()], 0, 1280, 720)[0]
    assert tracker.update([], 0.05, 1280, 720) == []
    recovered = tracker.update([hand(305)], 0.1, 1280, 720)[0]
    assert recovered["track_id"] == first["track_id"]
    assert tracker.update([], 0.2, 1280, 720) == []
    returned = tracker.update([hand(305)], 0.5, 1280, 720)[0]
    assert returned["track_id"] != first["track_id"]
    np.testing.assert_allclose(returned["pts"], hand(305)["pts"])


def test_large_displacement_starts_new_track_without_smoothing_bridge():
    tracker = HandStabilizer()
    first = tracker.update([hand(100)], 0, 1280, 720)[0]
    returned = tracker.update([hand(1000)], 1 / 30, 1280, 720)[0]
    assert returned["track_id"] != first["track_id"]
    np.testing.assert_allclose(returned["pts"], hand(1000)["pts"])


def test_world_geometry_and_classifier_features_are_not_smoothed():
    tracker = HandStabilizer()
    tracker.update([hand()], 0, 1280, 720)
    current = hand(305)
    current["world"][8] += [0.03, -0.02, 0.01]
    original_points = current["pts"].copy()
    result = tracker.update([current], 1 / 30, 1280, 720)[0]
    np.testing.assert_array_equal(result["world"], current["world"])
    np.testing.assert_array_equal(hand_features(result), hand_features(current))
    np.testing.assert_array_equal(current["pts"], original_points)
    assert result["ready_made"] == current["ready_made"]


def test_resolution_or_clock_reset_starts_fresh_tracks():
    tracker = HandStabilizer()
    old_id = tracker.update([hand()], 1, 1280, 720)[0]["track_id"]
    resized = tracker.update([hand()], 2, 640, 480)[0]
    assert resized["track_id"] != old_id
    rewound = tracker.update([hand()], 0, 640, 480)[0]
    assert rewound["track_id"] != resized["track_id"]


def test_invalid_landmarks_are_ignored():
    tracker = HandStabilizer()
    malformed = hand()
    malformed["pts"][8] = np.nan
    assert tracker.update([malformed], 0, 1280, 720) == []


def test_find_hands_uses_tracker_state_and_closes_recognizer():
    detection = hand()
    points = [SimpleNamespace(x=x / 1280, y=y / 720) for x, y in detection["pts"]]
    world = [SimpleNamespace(x=x, y=y, z=z) for x, y, z in detection["world"]]
    category = SimpleNamespace(category_name="Left")
    gesture = SimpleNamespace(category_name="Victory", score=0.95)

    class Recognizer:
        closed = False

        def recognize_for_video(self, image, time_ms):
            return SimpleNamespace(hand_landmarks=[points], hand_world_landmarks=[world],
                                   handedness=[[category]], gestures=[[gesture]])

        def close(self):
            self.closed = True

    recognizer = Recognizer()
    with HandTracker(recognizer) as tracker:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        first = find_hands(tracker, frame, 0)[0]
        category.category_name = "Right"
        second = find_hands(tracker, frame, 33)[0]
        assert first["track_id"] == second["track_id"]
        assert second["label"] == "Left"
    assert recognizer.closed


def test_new_track_cannot_inherit_a_drawing_with_the_same_label():
    import pen
    import settings
    pen = importlib.reload(pen)
    tracker = HandStabilizer()
    cfg = dict(settings.PEN_DEFAULTS)
    for frame in range(24):
        current = hand()
        current["world"][4] = current["world"][8] + (0.01, 0, 0)
        x = 200 + frame * 5 if frame < 12 else 1000 + (frame - 12) * 5
        current["pts"] = (current["world"] - current["world"][8])[:, :2] * 800 + [x, 300]
        stable = tracker.update([current], frame / 30, 1280, 720)
        pen.update(stable, frame / 30, 1280, 720, cfg)
    assert len(pen.strokes) == 2
    assert pen.strokes[0]["end"] == 12 / 30
    # Neither stroke may include the big jump between appearances.
    assert all(np.ptp(np.array(stroke["points"])[:, 0]) < 0.15 for stroke in pen.strokes)


@pytest.mark.parametrize("name", ["pen_draw", "pen_accidental"])
def test_recorded_pen_clips_through_full_tracking_pipeline(name):
    import pen
    import settings
    from replay import load_clip, matches
    pen = importlib.reload(pen)
    tracker = HandStabilizer()
    cfg = dict(settings.PEN_DEFAULTS)
    expect, (height, width), frames = load_clip(Path(__file__).parent / "clips" / (name + ".npz"))
    started = 0
    for now, hands in frames:
        stabilized = tracker.update(hands, now, width, height)
        before = len(pen.strokes)
        pen.update(stabilized, now, width, height, cfg)
        started += len(pen.strokes) > before
    assert matches(expect["pen_strokes"], started)

"""Deterministic gesture timelines, independent of a camera or model download."""
import importlib

import numpy as np
import pytest

import gestures


@pytest.fixture
def detector():
    return importlib.reload(gestures)


def hand(track=1, name="Thumb_Up", score=0.95, x=100):
    world = np.zeros((21, 3))
    world[9, 1] = 0.1
    return {"track_id": track, "label": "Left" if track == 1 else "Right",
            "pts": np.tile([x, 200.0], (21, 1)), "world": world,
            "ready_made": (name, score)}


def timeline(detector, samples):
    return [(t, event) for t, hands in samples for event in detector.detect(hands, t)]


@pytest.mark.parametrize("fps", [15, 30, 60])
def test_hold_duration_is_consistent_across_frame_rates(detector, fps):
    events = timeline(detector, [(i / fps, [hand()]) for i in range(fps * 3)])
    assert len(events) == 1
    assert 0.4 - 1e-9 <= events[0][0] <= 0.4 + 1 / fps


@pytest.mark.parametrize("missing", [[], [hand(score=0.2)], [hand(name="Victory")]])
def test_brief_uncertainty_pauses_hold_without_losing_progress(detector, missing):
    samples = [(i / 100, [hand()]) for i in range(21)]
    samples += [(0.21, missing), (0.25, missing)]
    samples += [(i / 100, [hand()]) for i in range(26, 71)]
    events = timeline(detector, samples)
    assert len(events) == 1
    # Twenty hundredths before the gap + twenty after it. Missing time is not evidence.
    assert events[0][0] == pytest.approx(0.46)


def test_brief_dropout_does_not_rearm_after_cooldown(detector):
    samples = [(i / 20, [hand()]) for i in range(60)]
    samples += [(3.0, []), (3.05, []), (3.1, [hand()])]
    samples += [(i / 20, [hand()]) for i in range(63, 100)]
    assert len(timeline(detector, samples)) == 1


def test_sustained_release_rearms(detector):
    samples = [(i / 20, [hand()]) for i in range(60)]
    samples += [(i / 20, []) for i in range(60, 66)]
    samples += [(i / 20, [hand()]) for i in range(66, 90)]
    events = timeline(detector, samples)
    assert [t for t, _ in events] == pytest.approx([0.4, 3.7])


def test_single_hand_effect_uses_its_own_location(detector):
    events = timeline(detector, [(i / 20, [hand(x=75), hand(2, name="None", x=900)])
                                 for i in range(12)])
    assert len(events) == 1
    assert events[0][1]["x"] == 75


def test_hands_cannot_pass_a_hold_to_each_other(detector):
    samples = [(i / 20, [hand()]) for i in range(7)]
    samples += [(i / 20, [hand(2, x=900)]) for i in range(7, 25)]
    events = timeline(detector, samples)
    assert len(events) == 1
    assert events[0][0] >= 0.35 + detector.RELEASE_GRACE + detector.TRAINED_HOLD
    assert events[0][1]["x"] == 900


def test_detection_order_does_not_change_active_participant(detector):
    left, right = hand(), hand(2, name="Victory")
    samples = [(i / 30, [left, right] if i % 2 == 0 else [right, left]) for i in range(20)]
    events = timeline(detector, samples)
    assert len(events) == 1
    assert events[0][1]["name"] == "thumbs_up"


def test_two_handed_pose_survives_short_loss_and_anchors_between_hands(detector):
    samples = [(i / 20, [hand(x=100), hand(2, x=900)]) for i in range(5)]
    samples += [(0.25, [hand(x=100)])]
    samples += [(i / 20, [hand(x=100), hand(2, x=900)]) for i in range(6, 20)]
    events = timeline(detector, samples)
    assert len(events) == 1
    assert events[0][1]["name"] == "thumbs_up_x2"
    assert events[0][1]["x"] == 500


def test_stalled_stream_does_not_count_as_held_time(detector):
    samples = [(0, [hand()]), (0.1, [hand()]), (2.0, [hand()])]
    assert timeline(detector, samples) == []
    assert detector.status()[0]["progress"] == 0


def test_clock_reset_starts_a_fresh_hold(detector):
    assert timeline(detector, [(2, [hand()]), (2.2, [hand()]), (0, [hand()])]) == []
    assert detector.status()[0]["progress"] == 0


class FakeClassifier:
    classes_ = np.array(["none", "wave"])

    def __init__(self):
        self.probs = np.array([0.05, 0.95])

    def predict_proba(self, features):
        return np.array([self.probs])


def test_trained_smoothing_resets_after_hand_loss(detector):
    detector.classifier = FakeClassifier()
    detector.check_trained([hand()], 0)
    assert detector.state["probs"]
    detector.check_trained([], 0.01)
    assert detector.state["probs"] == {}
    detector.classifier.probs = np.array([0.95, 0.05])
    assert detector.guess_pose(hand(), now=0.02) is None


@pytest.mark.parametrize("fps", [15, 30, 60])
def test_probability_smoothing_uses_time_not_frame_count(detector, fps):
    detector.classifier = FakeClassifier()
    detector.guess_pose(hand(), now=0)
    detector.classifier.probs = np.array([0.95, 0.05])
    for i in range(1, fps + 1):
        detector.guess_pose(hand(), now=i / fps)
    expected = 0.95 + (0.05 - 0.95) * np.exp(-1 / detector.SMOOTH_SECONDS)
    assert detector.state["probs"][1][0][0] == pytest.approx(expected)


def test_new_track_does_not_inherit_classifier_probabilities(detector):
    detector.classifier = FakeClassifier()
    detector.guess_pose(hand(), now=0)
    replacement = hand()
    replacement["track_id"] = 99
    detector.classifier.probs = np.array([0.95, 0.05])
    assert detector.guess_pose(replacement, now=0.01) is None


def test_failed_classifier_reload_clears_old_model_and_state(detector, tmp_path, monkeypatch):
    broken = tmp_path / "broken.joblib"
    broken.write_text("invalid model")
    monkeypatch.setattr(detector, "CLASSIFIER_PATH", str(broken))
    detector.classifier = FakeClassifier()
    detector.check_trained([hand()], 0)
    detector.load_classifier()
    assert detector.classifier is None
    assert detector.state["probs"] == {}
    assert detector.state["held"]["trained"] == {}

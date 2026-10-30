import numpy as np
import pytest
import settings
from tools.evaluate_background import evaluate, save_frames, load_frames


def test_background_counts_actual_configured_triggers_and_cooldown(monkeypatch):
    import gestures
    monkeypatch.setattr(gestures, 'detect', lambda hands, now: [
        {'type': 'motion', 'name': name} for name in ('swipe', 'disabled', 'unconfigured')])
    config = {'gestures': {'swipe': settings.effect({'text': 'Hi', 'cooldown': 2}),
                           'disabled': settings.effect({'enabled': False})}}
    report = evaluate([(0, []), (1, []), (3, [])], 300, config)
    assert report['candidate_count'] == 9
    assert report['configured_trigger_count'] == 2
    assert report['false_triggers_per_minute'] == 0.4
    assert report['hand_visibility_fraction'] == 0
    assert report['completed_five_minutes']


def test_short_background_session_is_labeled(monkeypatch):
    import gestures
    monkeypatch.setattr(gestures, 'detect', lambda *args: [])
    report = evaluate([(0, []), (30, [])], 30, {'gestures': {}})
    assert not report['completed_five_minutes']
    assert report['false_triggers_per_minute'] == 0
    with pytest.raises(ValueError):
        evaluate([], 0, {'gestures': {}})


def test_landmark_replay_preserves_empty_frames_and_track_identity(tmp_path):
    hand = {'label': 'Left', 'track_id': 42, 'pts': np.ones((21,2)),
            'world': np.ones((21,3)), 'ready_made': ('Victory', 0.9)}
    path = tmp_path / 'background.npz'
    save_frames(path, [(0, []), (1, [hand]), (2, [])], 2)
    frames, duration = load_frames(path)
    assert duration == 2
    assert frames[0][1] == frames[2][1] == []
    assert frames[1][1][0]['track_id'] == 42
    np.testing.assert_array_equal(frames[1][1][0]['world'], hand['world'])


def test_replay_with_no_hands_is_valid(tmp_path):
    path = tmp_path / 'empty.npz'
    save_frames(path, [(0, []), (1, [])], 1)
    frames, duration = load_frames(path)
    assert frames == [(0, []), (1, [])]

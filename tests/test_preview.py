"""Preview guides must never leak into the virtual camera output."""
import pytest
import sys
from types import SimpleNamespace

import numpy as np

import gesture_cam
import settings


@pytest.mark.parametrize("studio_preview", [False, True])
def test_status_display_stays_in_local_preview(monkeypatch, project, studio_preview):
    frame = np.full((540, 960, 3), 100, dtype=np.uint8)
    output, shown = [], []
    camera = SimpleNamespace(read=lambda: (True, frame.copy()), set=lambda *args: None,
                             release=lambda: None)
    virtual = SimpleNamespace(device="test", send=lambda image: output.append(image.copy()),
                              close=lambda: None)
    monkeypatch.setitem(sys.modules, "pyvirtualcam", SimpleNamespace(Camera=lambda **kw: virtual))
    monkeypatch.setattr(gesture_cam.cv2, "VideoCapture", lambda _: camera)
    monkeypatch.setattr(gesture_cam, "LatestCapture", lambda cap: cap)
    monkeypatch.setattr(gesture_cam.cv2, "imshow", lambda title, image: shown.append(image.copy()))
    monkeypatch.setattr(gesture_cam.cv2, "waitKey", lambda _: ord("q"))
    monkeypatch.setattr(gesture_cam.cv2, "destroyAllWindows", lambda: None)
    for name in ("make_hand_tracker", "make_face_tracker", "make_person_cutter"):
        monkeypatch.setattr(gesture_cam, name, lambda: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(gesture_cam, "find_hands", lambda *args: [])
    monkeypatch.setattr(gesture_cam, "load_settings", lambda: {
        "gestures": {}, "pen": {**settings.PEN_DEFAULTS, "enabled": False}})
    monkeypatch.setattr(gesture_cam.gestures, "detect", lambda *args: [])
    monkeypatch.setattr(gesture_cam.gestures, "status", lambda: [])
    monkeypatch.setattr(gesture_cam.effects, "needs_face", lambda: False)
    monkeypatch.setattr(gesture_cam.effects, "needs_person", lambda: False)
    monkeypatch.setattr(gesture_cam.effects, "update_and_draw", lambda frame, *args: frame)
    monkeypatch.setattr(gesture_cam.effects, "draw_text", lambda *args: None)
    monkeypatch.setattr(gesture_cam.sounds, "start_call_mix", lambda *args: None)
    monkeypatch.setattr(gesture_cam.sounds, "stop_call_mix", lambda: None)
    streamed = []
    stream = SimpleNamespace(update=lambda frame: streamed.append(frame.copy()), close=lambda: None)
    monkeypatch.setitem(sys.modules, "preview_stream", SimpleNamespace(PreviewServer=lambda: stream))
    gesture_cam.main(studio_preview=studio_preview)
    assert len(streamed) == int(studio_preview)
    assert len(output) == len(shown) == 1
    np.testing.assert_array_equal(output[0], frame)
    assert not np.array_equal(shown[0], frame)

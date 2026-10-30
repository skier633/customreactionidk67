# the studio's web api, in a throwaway folder
import json

import numpy as np
import pytest


@pytest.fixture
def client(project):
    import importlib
    import studio
    studio = importlib.reload(studio)
    np.save(project / "data" / "none.npy", np.zeros((50, 63)))
    np.save(project / "data" / "wave.npy", np.ones((50, 63)))
    return studio.app.test_client()


def test_state_lists_everything(client):
    s = client.get("/api/state").get_json()
    assert "thumbs_up" in s["ready_made"]
    assert [g["name"] for g in s["gestures"]] == ["wave"]
    assert {"wave", "wave_x2", "thumbs_up_x2"} <= set(s["effects"])
    assert "finger_gun" not in s["effects"] and "cheek_pull" not in s["effects"]


def test_set_saves_to_settings_json(client):
    r = client.post("/api/set", json={"key": "wave_x2", "field": "particles", "value": "confetti"})
    assert r.status_code == 200
    assert json.load(open("settings.json"))["gestures"]["wave_x2"]["particles"] == "confetti"


@pytest.mark.parametrize("body,error", [
    ({"key": "nope", "field": "text", "value": "x"}, "unknown gesture"),
    ({"key": "none", "field": "text", "value": "x"}, "unknown gesture"),
    ({"key": "wave", "field": "evil", "value": "x"}, "unknown setting"),
    ({"key": "wave", "field": "gif", "value": "../settings.json"}, "no file"),
    ({"key": "wave", "field": "filter", "value": "nope"}, "not an option"),
    ({"key": "wave", "field": "duration", "value": "abc"}, "number"),
])
def test_set_rejects_bad_input(client, body, error):
    r = client.post("/api/set", json=body)
    assert r.status_code == 400 and error in r.get_json()["error"]


def test_numbers_get_clamped(client):
    s = client.post("/api/set", json={"key": "wave", "field": "strength", "value": "250"}).get_json()
    assert s["effects"]["wave"]["strength"] == 100


def test_pen_settings(client):
    assert client.post("/api/pen", json={"field": "color", "value": "#00FF88"}).get_json()["pen"]["color"] == "#00ff88"
    assert client.post("/api/pen", json={"field": "color", "value": "red"}).status_code == 400
    assert client.post("/api/pen", json={"field": "thickness", "value": "999"}).get_json()["pen"]["thickness"] == 30


def test_try_it_and_clear_talk_to_gesture_cam(client):
    client.post("/api/test", json={"key": "thumbs_up"})
    assert json.load(open("test_trigger.json")) == {"gesture": "thumbs_up"}
    client.post("/api/clear_pen", json={})
    assert json.load(open("test_trigger.json")) == {"command": "clear_pen"}


def test_names_that_are_taken(client):
    for name in ["thumbs_up", "wave_x2", "../x"]:
        assert client.post("/api/record", json={"name": name}).status_code == 400


def test_delete_moves_recording_and_settings(client, project):
    client.post("/api/set", json={"key": "wave", "field": "text", "value": "hi"})
    s = client.post("/api/delete", json={"name": "wave"}).get_json()
    assert s["gestures"] == []
    assert (project / "data" / "deleted" / "wave.npy").exists()
    assert "wave" not in json.load(open("settings.json"))["gestures"]


def test_switch_a_gesture_off_and_on(client):
    s = client.post("/api/set", json={"key": "thumbs_up_x2", "field": "enabled", "value": False}).get_json()
    assert s["effects"]["thumbs_up_x2"]["enabled"] is False
    assert s["effects"]["thumbs_up"]["enabled"] is True          # one hand is separate
    assert json.load(open("settings.json"))["gestures"]["thumbs_up_x2"]["enabled"] is False
    s = client.post("/api/set", json={"key": "thumbs_up_x2", "field": "enabled", "value": True}).get_json()
    assert s["effects"]["thumbs_up_x2"]["enabled"] is True


def test_single_reaction_replaces_effects_and_preserves_timing(client):
    client.post("/api/set", json={"key":"wave", "field":"text", "value":"Old caption"})
    client.post("/api/set", json={"key":"wave", "field":"motion_early", "value":True})
    result = client.post("/api/reaction", json={"key":"wave", "field":"particles", "value":"confetti"})
    effect = result.get_json()["effects"]["wave"]
    assert effect["particles"] == "confetti" and effect["text"] == ""
    assert effect["motion_early"] is True
    assert client.post("/api/reaction", json={"key":"wave", "field":"bad"}).status_code == 400


def test_presets_removed_and_camera_validation(client):
    assert client.post("/api/preset", json={"name":"celebrate"}).status_code == 404
    assert client.post("/api/camera", json={"action":"start", "camera":-1}).status_code == 400
    assert client.get("/api/preview").status_code == 404


def test_camera_lifecycle(client, monkeypatch):
    import studio
    class Process:
        stopped = False
        def poll(self): return 0 if self.stopped else None
        def terminate(self): self.stopped = True
        def wait(self, timeout): pass
    process = Process()
    calls = []
    def launch(args):
        calls.append(args)
        return process
    monkeypatch.setattr(studio.subprocess, "Popen", launch)
    assert client.post("/api/camera", json={"action":"start", "camera":2}).get_json()["running"]
    assert calls[0][-3:] == ["--camera", "2", "--studio-preview"]
    assert not client.post("/api/camera", json={"action":"stop"}).get_json()["running"]
    assert process.stopped


def test_record_uses_selected_camera_and_releases_preview(client, monkeypatch):
    import studio
    calls = []
    monkeypatch.setattr(studio, 'stop_camera', lambda: calls.append('stop'))
    monkeypatch.setattr(studio, 'start_job', lambda label, args, **kw: calls.append(args))
    response = client.post('/api/record', json={'name':'swipe','camera':2})
    assert response.status_code == 200
    assert calls == ['stop', ['record.py', 'swipe', '--camera', '2']]

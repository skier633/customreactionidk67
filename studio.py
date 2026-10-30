# studio.py
# a web page for adding your own gestures and picking what they do (picture, gif, sound, distortion...)
# run: python studio.py   (it opens in your browser)
#
# you can leave gesture_cam.py running while you use this. it picks up the changes by itself
# under the hood this just runs record.py and train.py for you and edits settings.json

import glob
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import webbrowser

import numpy as np
from flask import Flask, abort, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename

import settings
import motion

PORT = 5050
IMAGE_TYPES = (".png", ".jpg", ".jpeg")
GIF_TYPES = (".gif", ".mp4", ".mov", ".m4v")
SOUND_TYPES = (".wav", ".mp3", ".m4a")
TEST_FILE = "test_trigger.json"  # gesture_cam.py looks for this
NAME_OK = re.compile(r"^[a-z0-9_]{1,30}$")

app = Flask(__name__)
job = {"label": None, "proc": None, "output": ""}  # the record.py / train.py thats running
model_cache = {"time": None, "classes": []}


# ---------- helpers ----------

def files_in(folder, types):
    if not os.path.isdir(folder):
        return []
    return sorted(f for f in os.listdir(folder) if f.lower().endswith(types))


def recorded_gestures():
    # {name: number of samples} for everything in data/
    counts = {}
    for path in sorted(glob.glob("data/*.npy")):
        name = os.path.basename(path)[:-4]
        counts[name] = int(np.load(path, mmap_mode="r").shape[0])
    for name, path in motion.recordings().items():
        with np.load(path) as data:
            counts[name] = len(data["clips"])
    return counts


def trained_classes():
    # which gestures the saved model knows about (only reloads it when the file changes)
    path = motion.MODEL_PATH if motion.recordings() else settings.CLASSIFIER_PATH
    if not os.path.exists(path):
        return None
    t = os.path.getmtime(path)
    if model_cache["time"] != t:
        import joblib
        try:
            model_cache["classes"] = sorted(str(c) for c in joblib.load(path).classes_)
            model_cache["time"] = t
        except Exception:
            return []
    return model_cache["classes"]


def needs_training(counts):
    classes = trained_classes()
    if classes is None:
        return len(counts) >= 2
    if sorted(counts) != classes:
        return True
    paths = list(motion.recordings().values()) or list(glob.glob("data/*.npy"))
    newest_data = max([os.path.getmtime(p) for p in paths], default=0)
    model_path = motion.MODEL_PATH if motion.recordings() else settings.CLASSIFIER_PATH
    return newest_data > os.path.getmtime(model_path)


def job_running():
    return job["proc"] is not None and job["proc"].poll() is None


def start_job(label, args, then=None, keep_output=False):
    # runs a script in the background. then() runs after it if it worked
    proc = subprocess.Popen([sys.executable, "-u"] + args, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    job["label"] = label
    job["proc"] = proc
    if not keep_output:
        job["output"] = ""

    def read_output():
        for line in proc.stdout:
            # skip mediapipe's startup spam
            if line.startswith(("W0000", "I0000", "INFO: Created")):
                continue
            job["output"] += line
        proc.wait()
        if proc.returncode == 0 and then is not None:
            then()

    threading.Thread(target=read_output, daemon=True).start()


def train_after_recording():
    # so you dont have to remember to hit train. needs at least 2 recordings (like "none" + one gesture)
    if len(recorded_gestures()) >= 2:
        job["output"] += "\nnow training...\n"
        start_job("training", ["train_motion.py"] if motion.recordings() else ["train.py"], keep_output=True)


def check_name(name):
    if not NAME_OK.match(name or ""):
        abort(400, "names can only have lowercase letters, numbers and _")
    if name in settings.READY_MADE or name.endswith("_x2"):
        abort(400, "that name is taken")


def check_gesture_key(key):
    # ready-made gestures or a recorded one, with or without _x2 for both hands
    base = key[:-3] if key.endswith("_x2") else key
    if base in settings.READY_MADE:
        return
    if base == "none" or base not in recorded_gestures():
        abort(400, "unknown gesture " + key)


def clean_value(field, value):
    # checks a value from the page before it goes in settings.json
    choices = {
        "picture_where": settings.PICTURE_WHERE, "gif_where": settings.GIF_WHERE,
        "distortion": settings.DISTORTIONS, "distortion_where": settings.DISTORTION_WHERE,
        "particles": settings.PARTICLES, "filter": settings.FILTERS,
    }
    files = {"picture": ("assets", IMAGE_TYPES), "gif": ("assets", GIF_TYPES), "sound": ("sounds", SOUND_TYPES)}
    if field in choices:
        if value != "" and value not in choices[field]:
            abort(400, "not an option: " + str(value))
        if value == "" and field.endswith("_where"):
            abort(400, "pick where it goes")
        return value
    if field in files:
        folder, types = files[field]
        if value != "" and value not in files_in(folder, types):
            abort(400, "no file called " + str(value))
        return value
    if field in settings.NUMBER_RANGES:
        low, high = settings.NUMBER_RANGES[field]
        try:
            number = float(value)
        except (TypeError, ValueError):
            abort(400, field + " has to be a number")
        number = min(max(number, low), high)
        return number if field == "duration" else int(round(number))
    if field in ("fade", "enabled"):
        return bool(value)
    if field == "text":
        return str(value).strip()[:60]
    abort(400, "unknown setting " + str(field))


def clean_pen_value(field, value):
    if field in ("enabled", "glow", "readable_for_call"):
        return bool(value)
    if field == "grip":
        if value not in settings.PEN_GRIPS:
            abort(400, "not an option: " + str(value))
        return value
    if field == "color":
        if not re.match(r"^#[0-9a-fA-F]{6}$", str(value)):
            abort(400, "color has to look like #ff3b6b")
        return str(value).lower()
    if field == "clear_with":
        if value not in settings.PEN_CLEAR:
            abort(400, "not an option: " + str(value))
        return value
    if field in settings.PEN_RANGES:
        low, high = settings.PEN_RANGES[field]
        try:
            number = float(value)
        except (TypeError, ValueError):
            abort(400, field + " has to be a number")
        number = min(max(number, low), high)
        return int(round(number)) if field == "thickness" else number
    abort(400, "unknown pen setting " + str(field))


def training_report():
    # the test results train.py saved, if theyre for the current model
    path = motion.REPORT_PATH if motion.recordings() else "models/my_gestures_report.json"
    model_path = motion.MODEL_PATH if motion.recordings() else settings.CLASSIFIER_PATH
    if not os.path.exists(path) or not os.path.exists(model_path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def send_to_cam(message):
    with open(TEST_FILE, "w") as f:
        json.dump(message, f)


# ---------- api ----------

@app.errorhandler(400)
@app.errorhandler(404)
@app.errorhandler(409)
def error(e):
    return jsonify({"error": e.description}), e.code


@app.get("/")
def page():
    return send_from_directory(".", "studio.html")


@app.get("/api/state")
def state():
    counts = recorded_gestures()
    config = settings.load()
    # every gesture shows up, even ones that dont have settings yet
    effects = {}
    for name in settings.READY_MADE + [n for n in counts if n != "none"]:
        for key in [name, name + "_x2"]:
            effects[key] = config["gestures"].get(key, settings.effect({}))
    return jsonify({
        "ready_made": settings.READY_MADE,
        "motion_names": sorted(motion.recordings()),
        "pen": config["pen"],
        "pen_clear": settings.PEN_CLEAR,
        "pen_grips": settings.PEN_GRIPS,
        "report": training_report(),
        "gestures": [{"name": n, "samples": c} for n, c in counts.items() if n != "none"],
        "none_samples": counts.get("none", 0),
        "trained": trained_classes() is not None,
        "needs_training": needs_training(counts),
        "effects": effects,
        "options": {
            "picture_where": settings.PICTURE_WHERE, "gif_where": settings.GIF_WHERE,
            "distortion": settings.DISTORTIONS, "distortion_where": settings.DISTORTION_WHERE,
            "particles": settings.PARTICLES, "filter": settings.FILTERS, "ranges": settings.NUMBER_RANGES,
        },
        "images": files_in("assets", IMAGE_TYPES),
        "gifs": files_in("assets", GIF_TYPES),
        "sound_files": files_in("sounds", SOUND_TYPES),
        "job": {"label": job["label"], "running": job_running(), "output": job["output"],
                "ok": job["proc"] is not None and job["proc"].poll() == 0},
    })


@app.post("/api/record")
def record():
    name = (request.json or {}).get("name", "").strip().lower()
    if name != "none":
        check_name(name)
    if job_running():
        abort(409, "wait for the other thing to finish first")
    start_job("recording " + name, ["record.py", name], then=train_after_recording)
    return state()


@app.post("/api/train")
def train():
    if job_running():
        abort(409, "wait for the other thing to finish first")
    start_job("training", ["train_motion.py"] if motion.recordings() else ["train.py"])
    return state()


@app.post("/api/stop")
def stop():
    if job_running():
        job["proc"].terminate()
    return state()


@app.post("/api/set")
def set_value():
    # changes one option for one gesture, like {"key": "thumbs_up", "field": "sound", "value": "boom.wav"}
    body = request.json or {}
    key, field = body.get("key", ""), body.get("field", "")
    check_gesture_key(key)
    value = clean_value(field, body.get("value", ""))
    config = settings.load()
    cfg = config["gestures"].get(key, settings.effect({}))
    cfg[field] = value
    config["gestures"][key] = cfg
    settings.save(config)
    return state()


@app.post("/api/reset")
def reset():
    # puts a gesture back how it started (does nothing)
    key = (request.json or {}).get("key", "")
    check_gesture_key(key)
    config = settings.load()
    config["gestures"][key] = settings.effect(settings.DEFAULTS["gestures"].get(key, {}))
    settings.save(config)
    return state()


@app.post("/api/preset")
def apply_preset():
    name = (request.get_json(silent=True) or {}).get("name")
    if not isinstance(name, str) or name not in settings.PRESETS:
        abort(400, "unknown preset")
    config = settings.load()
    for key, value in settings.PRESETS[name].items():
        current = config["gestures"].get(key)
        # An untouched/reset slot is safe to fill. Keep any edits, including disabled slots.
        if current is None or current == settings.effect({}):
            config["gestures"][key] = settings.effect(value)
    settings.save(config)
    return state()


@app.post("/api/test")
def test():
    # tells gesture_cam.py to play this gesture's effect right now
    key = (request.json or {}).get("key", "")
    check_gesture_key(key)
    send_to_cam({"gesture": key})
    return jsonify({"ok": True})


@app.post("/api/pen")
def set_pen():
    # changes one air pen setting, like {"field": "color", "value": "#00ff88"}
    body = request.json or {}
    field = body.get("field", "")
    value = clean_pen_value(field, body.get("value", ""))
    config = settings.load()
    config["pen"][field] = value
    settings.save(config)
    return state()


@app.post("/api/clear_pen")
def clear_pen():
    send_to_cam({"command": "clear_pen"})
    return jsonify({"ok": True})


@app.post("/api/delete")
def delete_gesture():
    # moves the recording to data/deleted/ instead of really deleting it (move it back to undo)
    name = (request.json or {}).get("name", "")
    if name != "none":
        check_name(name)
    path = str(motion.recordings().get(name, os.path.join("data", name + ".npy")))
    if not os.path.exists(path):
        abort(404, "no gesture called " + name)
    if job_running():
        abort(409, "wait for the other thing to finish first")
    os.makedirs("data/deleted", exist_ok=True)
    shutil.move(path, os.path.join("data/deleted", os.path.basename(path)))
    config = settings.load()
    config["gestures"].pop(name, None)
    config["gestures"].pop(name + "_x2", None)
    settings.save(config)
    return state()


@app.post("/api/upload")
def upload():
    file = request.files.get("file")
    if file is None or not file.filename:
        abort(400, "no file")
    name = secure_filename(file.filename)
    ext = os.path.splitext(name)[1].lower()
    if ext in IMAGE_TYPES or ext in GIF_TYPES:
        folder = "assets"
    elif ext in SOUND_TYPES:
        folder = "sounds"
    else:
        abort(400, "only pictures (png, jpg), gifs/videos (gif, mp4, mov) and sounds (wav, mp3, m4a)")
    os.makedirs(folder, exist_ok=True)
    # an mp3 gets turned into a wav with the same name the first time. if you upload a new
    # version, throw away the old wav so it converts the new one
    if folder == "sounds" and ext != ".wav":
        old_wav = os.path.join(folder, os.path.splitext(name)[0] + ".wav")
        if os.path.exists(old_wav) and os.path.exists(os.path.join(folder, name)):
            os.remove(old_wav)
    file.save(os.path.join(folder, name))
    # touch settings.json so gesture_cam reloads (in case you replaced a file thats already used)
    settings.save(settings.load())
    return state()


@app.get("/files/<folder>/<name>")
def get_file(folder, name):
    # pictures and sounds so the page can show and play them
    if folder not in ("assets", "sounds"):
        abort(404)
    return send_from_directory(folder, name)


def main():
    url = "http://localhost:" + str(PORT)
    print("gesture studio is at", url, " (ctrl+c to quit)")
    threading.Timer(1, lambda: webbrowser.open(url)).start()
    app.run(host="127.0.0.1", port=PORT)


if __name__ == "__main__":
    main()

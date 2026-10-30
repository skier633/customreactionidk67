# settings.py
# what happens for each gesture. saved in settings.json
# studio.py changes this file, and gesture_cam.py notices and reloads it while it's running
#
# every gesture gets the same options (see EFFECT_DEFAULTS). trained gestures can have a
# separate set for doing it with both hands: "rock_on" and "rock_on_x2"

import json
import os

PATH = "settings.json"
CLASSIFIER_PATH = "models/my_gestures.joblib"
# mediapipe's ready-made gestures (see gestures.READY_MADE). they can have a both-hands "_x2" version too
READY_MADE = ["thumbs_up", "thumbs_down", "open_palm", "fist", "pointing_up", "victory", "i_love_you"]

# everything a gesture can do. "" = off
EFFECT_DEFAULTS = {
    "enabled": True,            # False = this gesture is switched off (it does nothing)
    "picture": "",              # png/jpg in assets/
    "picture_where": "full",    # one of PICTURE_WHERE
    "gif": "",                  # gif/mp4/mov in assets/. loops until the duration is up
    "gif_where": "full",        # one of GIF_WHERE
    "opacity": 100,             # for the picture and gif, 0-100
    "sound": "",                # wav/mp3 in sounds/
    "volume": 100,              # 0-100
    "distortion": "",           # one of DISTORTIONS
    "distortion_where": "face", # one of DISTORTION_WHERE
    "strength": 50,             # how strong the distortion is, 0-100
    "particles": "",            # one of PARTICLES
    "filter": "",               # color filter, one of FILTERS
    "shake": 0,                 # screen shake, 0-100
    "text": "",                 # caption across the bottom
    "duration": 1.0,            # seconds the picture/gif/distortion/filter/shake/text last
    "fade": True,               # fade in and out instead of popping
    "cooldown": 0,              # seconds before this gesture can go off again
}

# the choices (value: what the studio shows)
PICTURE_WHERE = {"full": "cover the screen", "behind": "behind you", "face": "on your face",
                 "hand": "at your hand", "burst": "burst out of your hand"}
GIF_WHERE = {"full": "cover the screen", "behind": "behind you", "face": "on your face",
             "hand": "at your hand"}
DISTORTIONS = {"bulge": "bulge", "pinch": "pinch", "swirl": "swirl", "pixelate": "pixelate",
               "wave": "wavy (whole screen)", "mirror": "mirror (whole screen)"}
DISTORTION_WHERE = {"face": "your face", "hand": "your hand", "center": "middle of the screen"}
PARTICLES = {"hearts": "hearts", "confetti": "confetti", "fireworks": "fireworks"}
FILTERS = {"gray": "black & white", "sepia": "old photo", "invert": "inverted", "deepfried": "deep fried",
           "thermal": "thermal camera", "blur": "blurry", "red": "red", "blue": "blue"}
NUMBER_RANGES = {"opacity": (0, 100), "volume": (0, 100), "strength": (0, 100), "shake": (0, 100),
                 "duration": (0.1, 10), "cooldown": (0, 30)}

# the air pen (see pen.py)
PEN_DEFAULTS = {
    "enabled": True,
    "color": "#ff3b6b",
    "thickness": 8,             # pixels at 720p
    "glow": True,
    "fade_after": 0,            # seconds after you lift the pen. 0 = stays until you clear it
    "clear_with": "fist",       # one of PEN_CLEAR
    "grip": "curled",           # one of PEN_GRIPS
    "readable_for_call": True,  # flip the ink in the call so writing reads the right way for them
}
PEN_GRIPS = {"curled": "other fingers curled in, like holding a pen (fewer accidents)",
             "any": "any pinch (easier, but more accidental lines)"}
PEN_CLEAR = {"fist": "hold a fist", "open_palm": "hold an open palm", "none": "only the C key / studio button"}
PEN_RANGES = {"thickness": (2, 30), "fade_after": (0, 60)}

# what you get the first time (before settings.json exists)
DEFAULTS = {"gestures": {}}

# Optional starter scenes use built-in effects, so no downloads or recordings are needed.
PRESETS = {
    "celebrate": {
        "thumbs_up": {"particles": "confetti", "text": "NICE!", "duration": 2.5},
        "victory": {"particles": "fireworks", "text": "LET'S GO", "duration": 2.5},
        "i_love_you": {"particles": "hearts", "duration": 2.5},
    },
    "present": {
        "thumbs_up": {"text": "Great question", "duration": 2.5},
        "victory": {"text": "Two things to remember", "duration": 3},
        "pointing_up": {"text": "Here's the key idea", "duration": 3},
    },
    "play": {
        "thumbs_down": {"filter": "gray", "text": "Well, that happened", "duration": 2},
        "victory": {"particles": "confetti", "shake": 20, "duration": 2},
        "i_love_you": {"filter": "sepia", "particles": "hearts", "duration": 2.5},
    },
}

last_good = None


def effect(cfg):
    # fills in anything missing with the defaults
    full = dict(EFFECT_DEFAULTS)
    full.update(cfg or {})
    return full


def load():
    # if the file is broken (like half written), keep using the last one that worked
    global last_good
    if not os.path.exists(PATH):
        save(DEFAULTS)
    try:
        with open(PATH) as f:
            data = json.load(f)
    except (OSError, ValueError) as e:
        print("couldnt read", PATH, "-", e)
        return last_good or {"gestures": {k: effect(v) for k, v in DEFAULTS["gestures"].items()},
                             "pen": dict(PEN_DEFAULTS)}
    data.setdefault("gestures", {})
    data["gestures"] = {k: effect(v) for k, v in data["gestures"].items()}
    data["pen"] = {**PEN_DEFAULTS, **data.get("pen", {})}
    last_good = data
    return data


def save(data):
    # only keeps the options that arent the default, so settings.json stays short
    short = {"gestures": {}}
    for name, cfg in data["gestures"].items():
        short["gestures"][name] = {k: v for k, v in cfg.items() if EFFECT_DEFAULTS.get(k) != v}
    pen = {k: v for k, v in data.get("pen", {}).items() if PEN_DEFAULTS.get(k) != v}
    if pen:
        short["pen"] = pen
    # write to a temp file first so gesture_cam never reads a half written file
    tmp = PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(short, f, indent=2)
    os.replace(tmp, PATH)


def stamp():
    # changes whenever the settings or the trained model change
    times = []
    for path in [PATH, CLASSIFIER_PATH]:
        times.append(os.path.getmtime(path) if os.path.exists(path) else 0)
    return tuple(times)

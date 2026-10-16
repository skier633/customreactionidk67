# tests/replay.py
# loads a clip from tools/record_clip.py and plays it back through the gesture code

import importlib
import json

import numpy as np


def load_clip(path):
    # returns (expect, (h, w), [(time, hands), ...])
    d = np.load(path)
    expect = json.loads(str(d["expect"]))
    frames = [(float(t), []) for t in d["times"]]
    for i in range(len(d["frame"])):
        frames[int(d["frame"][i])][1].append({
            "label": str(d["label"][i]),
            "pts": d["pts"][i],
            "world": d["world"][i],
            "ready_made": (str(d["ready_made_name"][i]), float(d["ready_made_score"][i])),
        })
    return expect, tuple(int(x) for x in d["size"]), frames


def replay(path):
    # counts what goes off: gesture names (thumbs_up, victory...) and "pen_strokes"
    import gestures
    import pen
    import settings
    gestures = importlib.reload(gestures)  # fresh state for every clip
    pen = importlib.reload(pen)
    gestures.classifier = None             # trained gestures depend on your own model, so skip them
    pen_cfg = dict(settings.PEN_DEFAULTS)
    expect, (h, w), frames = load_clip(path)
    counts = {}
    for t, hands in frames:
        events = gestures.detect(hands, t)
        for e in events:
            counts[e["name"]] = counts.get(e["name"], 0) + 1
        before = len(pen.strokes)
        pen.update(hands, t, w, h, pen_cfg)
        if len(pen.strokes) > before:
            counts["pen_strokes"] = counts.get("pen_strokes", 0) + 1
    return expect, counts


def matches(expected, actual):
    # expected is "5" or a range like "4-6"
    low, _, high = str(expected).partition("-")
    return int(low) <= actual <= int(high or low)

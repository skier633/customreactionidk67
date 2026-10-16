# tools/record_clip.py
# records a test clip: the hand points mediapipe finds every frame (just numbers, no video)
# the tests in tests/ replay these clips through the gesture code and check what goes off
#
# usage (from the project folder):
#   python tools/record_clip.py pen_heart --expect pen_strokes=1
#   python tools/record_clip.py talking --expect pen_strokes=0
#   python tools/record_clip.py thumbs --expect thumbs_up=3 --seconds 20
#
# --expect says what SHOULD happen in the clip (you can give a range like thumbs_up=2-3).
# names: pen_strokes, or any ready-made gesture (thumbs_up, victory, open_palm...)

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np

import gestures
import pen
import settings
from tracking import make_hand_tracker, find_hands

parser = argparse.ArgumentParser()
parser.add_argument("name", help="what to call the clip, like flicks")
parser.add_argument("--expect", nargs="*", default=[], help="like thumbs_up=3 or pen_strokes=1")
parser.add_argument("--seconds", type=float, default=15)
parser.add_argument("--camera", type=int, default=0)
args = parser.parse_args()

expect = {}
for item in args.expect:
    key, _, value = item.partition("=")
    expect[key] = value

cap = cv2.VideoCapture(args.camera)
tracker = make_hand_tracker()
pen_cfg = dict(settings.PEN_DEFAULTS)
rows = []          # one per hand per frame
times = []         # one per frame
counts = {}
start = time.time()
last_ms = -1
size = None

while True:
    ok, frame = cap.read()
    if not ok:
        print("camera problem")
        break
    size = frame.shape[:2]
    h, w = size
    now = time.time() - start
    ms = max(int(now * 1000), last_ms + 1)
    last_ms = ms
    hands = find_hands(tracker, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), ms)
    recording = 3 <= now < 3 + args.seconds

    # run the real gesture code live so you can see what it thinks
    events = gestures.detect(hands, now)
    strokes_before = len(pen.strokes)
    pen.update(hands, now, w, h, pen_cfg)
    if recording:
        times.append(now - 3)
        for hand in hands:
            rows.append((len(times) - 1, hand["label"], hand["pts"], hand["world"], *hand["ready_made"]))
        for e in events:
            counts[e["name"]] = counts.get(e["name"], 0) + 1
            print(f"{now - 3:5.1f}s  {e['name']}", flush=True)
        if len(pen.strokes) > strokes_before:
            counts["pen_strokes"] = counts.get("pen_strokes", 0) + 1
            print(f"{now - 3:5.1f}s  pen down", flush=True)

    view = cv2.flip(frame, 1)
    pen.draw(view, now, pen_cfg, mirror=True)
    if now < 3:
        msg = "get ready " + str(3 - int(now))
    elif recording:
        msg = "RECORDING " + str(int(3 + args.seconds - now)) + "s"
    else:
        break
    cv2.putText(view, msg, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    cv2.putText(view, "should happen: " + (", ".join(f"{k}={v}" for k, v in expect.items()) or "(nothing given)"),
                (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.imshow("record a test clip (q to stop)", view)
    if (cv2.waitKey(1) & 0xFF) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

out_dir = os.path.join(os.path.dirname(__file__), "..", "tests", "clips")
os.makedirs(out_dir, exist_ok=True)
path = os.path.join(out_dir, args.name + ".npz")
np.savez_compressed(
    path,
    size=np.array(size),
    times=np.array(times),
    frame=np.array([r[0] for r in rows], dtype=np.int32),
    label=np.array([r[1] for r in rows]),
    pts=np.array([r[2] for r in rows]).reshape(-1, 21, 2),
    world=np.array([r[3] for r in rows]).reshape(-1, 21, 3),
    ready_made_name=np.array([r[4] for r in rows]),
    ready_made_score=np.array([r[5] for r in rows], dtype=np.float32),
    expect=json.dumps(expect),
)
print()
print("saved", len(times), "frames to", os.path.relpath(path))
print("live it counted:", counts or "nothing")
print("you said it should be:", expect or "(nothing)")

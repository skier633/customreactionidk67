# record.py
# records examples of a hand pose so you can train it
#
# usage:
#   python record.py rock_on
#   python record.py none     <- do this one too!! random hand stuff that ISNT a gesture
#
# while its recording, move your hand around, tilt it, go closer and farther from the camera.
# the more different the examples are the better it works
# running it again with the same name adds more examples instead of replacing them

import os
import sys
import time

import cv2
import numpy as np

from tracking import make_hand_tracker, find_hands, hand_features

SECONDS = 15

if len(sys.argv) < 2:
    print("usage: python record.py <gesture name>")
    sys.exit()

name = sys.argv[1]
os.makedirs("data", exist_ok=True)
path = os.path.join("data", name + ".npy")

cap = cv2.VideoCapture(0)
tracker = make_hand_tracker()
samples = []
start = time.time()
last_ms = -1

while True:
    ok, frame = cap.read()
    if not ok:
        print("camera problem")
        break
    now = time.time() - start
    time_ms = max(int(now * 1000), last_ms + 1)
    last_ms = time_ms

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hands = find_hands(tracker, rgb, time_ms)

    preview = frame.copy()
    for hand in hands:
        for x, y in hand["pts"]:
            cv2.circle(preview, (int(x), int(y)), 4, (0, 255, 0), -1)
    preview = cv2.flip(preview, 1)

    if now < 3:
        text = "get ready: " + str(3 - int(now))
        color = (0, 255, 255)
    elif now < 3 + SECONDS:
        # record every hand we see
        for hand in hands:
            samples.append(hand_features(hand))
        text = "RECORDING " + name + "  " + str(int(3 + SECONDS - now)) + "s  samples: " + str(len(samples))
        color = (0, 0, 255)
    else:
        break

    cv2.putText(preview, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow("record (q to stop early)", preview)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

if len(samples) == 0:
    print("didnt see any hands, nothing saved")
    sys.exit()

samples = np.array(samples)
if os.path.exists(path):
    samples = np.concatenate([np.load(path), samples])
np.save(path, samples)
print("saved", len(samples), "samples to", path)

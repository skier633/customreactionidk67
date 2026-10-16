# gestures.py
# all the gesture checks. each one looks at the hand points and decides if you're doing the gesture
#
# two kinds: mediapipe's ready-made ones (thumbs up, peace...), and anything you train with
# record.py + train.py (checked in check_trained())

import os

import numpy as np

from tracking import WRIST, hand_features

# ---------- settings (change these if stuff triggers too much or not enough) ----------

TRAINED_HOLD = 0.4       # hold a trained (or ready-made) gesture this long before it triggers
TRAINED_COOLDOWN = 2.0
TRAINED_MIN_CONFIDENCE = 0.8
SMOOTH_FRAMES = 5        # average the network's guesses over this many frames (fewer flickers)

# mediapipe's ready-made gestures: their name -> our name
READY_MADE = {"Thumb_Up": "thumbs_up", "Thumb_Down": "thumbs_down", "Open_Palm": "open_palm",
              "Closed_Fist": "fist", "Pointing_Up": "pointing_up", "Victory": "victory", "ILoveYou": "i_love_you"}
READY_MADE_MIN_SCORE = 0.6
CLASSIFIER_PATH = "models/my_gestures.joblib"

# ---------- stuff we remember between frames ----------

state = {
    "probs": {},  # hand label -> the network's last few guesses (for smoothing)
    "held": {"trained": {}, "ready_made": {}},  # what pose each kind is seeing, and since when
    "last_trained": {},  # gesture name -> when it last went off
}

classifier = None


# ---------- helper math ----------

def angle_between(a, b):
    cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def finger_bend(world, knuckle):
    # adds up how much a finger bends at its two middle joints
    # knuckle = 5 for index, 9 for middle, 13 for ring, 17 for pinky
    a, b, c, d = world[knuckle], world[knuckle + 1], world[knuckle + 2], world[knuckle + 3]
    return angle_between(b - a, c - b) + angle_between(c - b, d - c)


# ---------- trained gestures ----------

def load_classifier():
    global classifier
    if os.path.exists(CLASSIFIER_PATH):
        import joblib
        try:
            classifier = joblib.load(CLASSIFIER_PATH)
        except Exception as e:
            print("couldnt load", CLASSIFIER_PATH, "-", e)
            return
        print("loaded trained gestures:", [str(c) for c in classifier.classes_ if c != "none"])
    else:
        classifier = None
        print("no trained gestures yet (make some with: python studio.py)")
    state["probs"].clear()  # the old guesses were for the old classifier


def guess_pose(hand, smooth=True):
    # the trained gesture this hand is doing, or None. averages the last few frames so one
    # weird frame doesnt count
    probs = classifier.predict_proba([hand_features(hand)])[0]
    if smooth:
        recent = state["probs"].setdefault(hand["label"], [])
        recent.append(probs)
        del recent[:-SMOOTH_FRAMES]
        probs = np.mean(recent, axis=0)
    best = int(np.argmax(probs))
    name = str(classifier.classes_[best])
    if name == "none" or probs[best] < TRAINED_MIN_CONFIDENCE:
        return None
    return name


def ready_made_pose(hand):
    name, score = hand.get("ready_made", ("None", 0.0))
    if name in READY_MADE and score >= READY_MADE_MIN_SCORE:
        return READY_MADE[name]
    return None


def held_pose(kind, names, hands, now):
    # turns per-hand guesses into an event once one is held long enough. both hands = "_x2"
    held = state["held"][kind]
    if len(names) == 2 and names[0] is not None and names[0] == names[1]:
        current = names[0] + "_x2"
    else:
        current = next((n for n in names if n is not None), None)

    if current != held.get("name"):
        held.update(name=current, since=now, done=False)
    if current is None or held["done"] or now - held["since"] < TRAINED_HOLD:
        return None
    if now - state["last_trained"].get(current, -99) < TRAINED_COOLDOWN:
        return None

    held["done"] = True
    state["last_trained"][current] = now
    middle = np.mean([hand["pts"][WRIST] for hand in hands], axis=0)
    return {"type": kind, "name": current, "x": middle[0], "y": middle[1]}


def check_trained(hands, now):
    # forget the smoothing for hands that left
    seen = {hand["label"] for hand in hands}
    for label in list(state["probs"]):
        if label not in seen:
            del state["probs"][label]
    if classifier is None or len(hands) == 0:
        state["held"]["trained"] = {}
        return None
    return held_pose("trained", [guess_pose(hand) for hand in hands], hands, now)


def check_ready_made(hands, now):
    if len(hands) == 0:
        state["held"]["ready_made"] = {}
        return None
    return held_pose("ready_made", [ready_made_pose(hand) for hand in hands], hands, now)


# ---------- run everything ----------

def detect(hands, now):
    # returns the gestures that just went off, like [{"type": "trained", "name": "peace", "x": .., "y": ..}]
    events = []
    for event in [check_trained(hands, now), check_ready_made(hands, now)]:
        if event:
            events.append(event)
    return events


def debug_text(hand):
    # what the debug view shows next to each hand
    world = hand["world"]
    bends = [int(finger_bend(world, k)) for k in [5, 9, 13, 17]]
    text = hand["label"] + " bends:" + str(bends)
    if classifier is not None:
        text += " pose:" + str(guess_pose(hand, smooth=False))
    name, score = hand.get("ready_made", ("None", 0))
    if name != "None":
        text += " mp:" + name + " " + str(int(score * 100)) + "%"
    return text

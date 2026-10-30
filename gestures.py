# gestures.py
# all the gesture checks. each one looks at the hand points and decides if you're doing the gesture
#
# two kinds: mediapipe's ready-made ones (thumbs up, peace...), and anything you train with
# record.py + train.py (checked in check_trained())

import os
import time

import numpy as np

from tracking import WRIST, hand_features

# ---------- settings (change these if stuff triggers too much or not enough) ----------

TRAINED_HOLD = 0.4       # hold a trained (or ready-made) gesture this long before it triggers
TRAINED_COOLDOWN = 2.0
TRAINED_MIN_CONFIDENCE = 0.8
SMOOTH_SECONDS = 0.08   # time-based probability smoothing, independent of camera frame rate
RELEASE_GRACE = 0.18    # a brief uncertain frame pauses a hold instead of cancelling it
MAX_SAMPLE_GAP = 0.25   # never infer a continuous hold across a stalled camera

# mediapipe's ready-made gestures: their name -> our name
READY_MADE = {"Thumb_Up": "thumbs_up", "Thumb_Down": "thumbs_down", "Open_Palm": "open_palm",
              "Closed_Fist": "fist", "Pointing_Up": "pointing_up", "Victory": "victory", "ILoveYou": "i_love_you"}
READY_MADE_MIN_SCORE = 0.6
CLASSIFIER_PATH = "models/my_gestures.joblib"

# ---------- stuff we remember between frames ----------

state = {
    "probs": {},  # track identity -> smoothed probabilities and sample time
    "held": {"trained": {}, "ready_made": {}},  # what pose each kind is seeing, and since when
    "last_trained": {},  # gesture name -> when it last went off
}

classifier = None
motion_recognizer = None


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
    global classifier, motion_recognizer
    from motion import MODEL_PATH, MotionRecognizer
    motion_recognizer = None
    if os.path.exists(MODEL_PATH):
        import joblib
        try:
            motion_recognizer = MotionRecognizer(joblib.load(MODEL_PATH))
        except Exception as e:
            print("Could not load motion model:", e)
    state["probs"].clear()
    state["held"]["trained"] = {}
    if os.path.exists(CLASSIFIER_PATH):
        import joblib
        try:
            classifier = joblib.load(CLASSIFIER_PATH)
        except Exception as e:
            classifier = None
            print("couldnt load", CLASSIFIER_PATH, "-", e)
            return
        print("loaded trained gestures:", [str(c) for c in classifier.classes_ if c != "none"])
    else:
        classifier = None
        print("no trained gestures yet (make some with: python studio.py)")


def identity(hand):
    return hand.get("track_id", hand["label"])


def guess_pose(hand, smooth=True, now=None):
    # Smooth by elapsed time so the response is consistent at different frame rates.
    probs = classifier.predict_proba([hand_features(hand)])[0]
    if smooth:
        now = time.monotonic() if now is None else now
        key = identity(hand)
        recent = state["probs"].get(key)
        if recent is not None:
            previous, previous_time = recent
            dt = now - previous_time
            if 0 <= dt <= MAX_SAMPLE_GAP and previous.shape == probs.shape:
                alpha = -np.expm1(-dt / SMOOTH_SECONDS)
                probs = previous + alpha * (probs - previous)
        state["probs"][key] = (probs.copy(), now)
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
    # Count only observed pose time. Brief uncertainty pauses the timer and retains a
    # fired latch; a deliberate release rearms it. Ownership follows physical tracks.
    held = state["held"][kind]
    previous_time = held.get("time", now)
    dt = now - previous_time
    if dt < 0 or dt > MAX_SAMPLE_GAP:
        held.clear()
        dt = 0

    if len(names) == 2 and names[0] is not None and names[0] == names[1]:
        current = names[0] + "_x2"
        participants = hands
    else:
        choices = [(name, [hand]) for name, hand in zip(names, hands) if name is not None]
        # Detection order can change. Keep following the same participant while its
        # pose is still visible, even when the other hand is making another gesture.
        selected = next(((name, hs) for name, hs in choices
                         if (name, frozenset(identity(h) for h in hs)) == held.get("signature")),
                        choices[0] if choices else (None, []))
        current, participants = selected
    signature = (current, frozenset(identity(h) for h in participants))

    if held and (current is None or signature != held.get("signature")):
        if held.get("missing_since") is None:
            held["missing_since"] = now
        held.update(time=now, observed=False)
        if now - held["missing_since"] < RELEASE_GRACE:
            return None
        held.clear()

    if current is None:
        return None
    # A returning sample after a long observed absence is a new hold, even when no
    # intermediate empty frames were delivered during the absence.
    if held.get("missing_since") is not None and now - held["missing_since"] >= RELEASE_GRACE:
        held.clear()

    if not held:
        held.update(name=current, signature=signature, duration=0.0, done=False,
                    time=now, observed=True, missing_since=None)
    else:
        if held.get("observed"):
            held["duration"] += dt
        held.update(time=now, observed=True, missing_since=None)

    if held["done"] or held["duration"] + 1e-9 < TRAINED_HOLD:
        return None
    if now - state["last_trained"].get(current, -99) < TRAINED_COOLDOWN:
        return None

    held["done"] = True
    state["last_trained"][current] = now
    middle = np.mean([hand["pts"][WRIST] for hand in participants], axis=0)
    return {"type": kind, "name": current, "x": middle[0], "y": middle[1]}


def check_trained(hands, now):
    # forget the smoothing for hands that left
    seen = {identity(hand) for hand in hands}
    for key in list(state["probs"]):
        if key not in seen:
            del state["probs"][key]
    if classifier is None:
        state["held"]["trained"] = {}
        return None
    names = [guess_pose(hand, now=now) for hand in hands]
    if motion_recognizer is not None:
        # A re-recorded motion owns its name; a legacy pose model must not replay it.
        motion_names = set(motion_recognizer.model.classes_)
        names = [name if name not in motion_names else None for name in names]
    return held_pose("trained", names, hands, now)


def check_ready_made(hands, now):
    return held_pose("ready_made", [ready_made_pose(hand) for hand in hands], hands, now)


def status():
    """Progress for a preview overlay; does not advance recognition or trigger events."""
    poses = [{"name": held["name"], "progress": min(1.0, held["duration"] / TRAINED_HOLD),
              "done": held["done"]}
             for held in state["held"].values() if held.get("name")]
    if motion_recognizer is not None:
        active = any(track["segment"].active for track in motion_recognizer.tracks.values())
        if active or not poses:
            return [{"name": "Move, then pause" if active else "Make a motion or hold a pose",
                     "progress": 0, "done": False, "motion": True}]
    return poses


# ---------- run everything ----------

def detect(hands, now):
    # returns the gestures that just went off, like [{"type": "trained", "name": "peace", "x": .., "y": ..}]
    events = []
    pose_hands = hands
    if motion_recognizer is not None:
        events.extend(motion_recognizer.update_all(hands, now))
        busy = motion_recognizer.busy_ids
        pose_hands = [hand for hand in hands if identity(hand) not in busy]
        # A partially held pose cannot accumulate time through a motion.
        for held in state["held"].values():
            signature = held.get("signature")
            if signature and signature[1] & busy:
                held.clear()
    for event in [check_trained(pose_hands, now), check_ready_made(pose_hands, now)]:
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

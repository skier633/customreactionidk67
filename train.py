# train.py
# trains a small neural network on everything in the data/ folder
# usage: python train.py
#
# what it does:
#   1. tests itself fairly: trains on the first 80% of each recording and tests on the last 20%
#      (testing on random frames would be cheating, since frames right next to each other look the same)
#   2. makes extra made-up examples (tilted, resized, wobbly versions of your recordings), and makes
#      enough of them that every gesture gets the same amount, so it doesnt favor the one you recorded most
#   3. saves the test results to models/my_gestures_report.json (the studio shows them)
#   4. trains the final version on everything and saves it to models/my_gestures.joblib

import glob
import json
import os

import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from features import augment, pose_features_batch

MODEL_PATH = "models/my_gestures.joblib"
REPORT_PATH = "models/my_gestures_report.json"
TEST_PART = 0.2        # last 20% of each recording is for testing
COPIES = 3             # the biggest gesture ends up with this many made-up examples per real one
MIN_SAMPLES = 60       # warn if a gesture has fewer than this


def make_model():
    return make_pipeline(
        FunctionTransformer(pose_features_batch),   # the 63 numbers -> tilt-proof features (see features.py)
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3, max_iter=500, random_state=0),
    )


def balanced(groups, rng):
    # groups = {name: samples}. returns X, y where every gesture has the same number of examples,
    # made of its real samples plus augmented copies
    target = max(len(s) for s in groups.values()) * COPIES
    X, y = [], []
    for name, samples in groups.items():
        X.extend(samples)
        picks = rng.integers(0, len(samples), target - len(samples))
        X.extend(augment(samples[i], rng) for i in picks)
        y.extend([name] * target)
    return np.array(X), np.array(y)


def main():
    groups = {}
    for file in sorted(glob.glob("data/*.npy")):
        name = os.path.basename(file)[:-4]
        groups[name] = np.load(file)
        print(name + ":", len(groups[name]), "samples" + ("  <- kinda few, record some more" if len(groups[name]) < MIN_SAMPLES else ""))

    if len(groups) < 2:
        print("need at least 2 gestures recorded (one of them should be 'none')")
        raise SystemExit(1)
    if "none" not in groups:
        print("warning: no 'none' recorded, so it'll think every hand is one of your gestures")
        print("run: python record.py none")

    rng = np.random.default_rng(0)

    # 1. fair test: recordings are saved in order, so the end of each one is "later" than the rest
    train, test_X, test_y = {}, [], []
    for name, samples in groups.items():
        cut = int(len(samples) * (1 - TEST_PART))
        train[name] = samples[:cut]
        test_X.extend(samples[cut:])
        test_y.extend([name] * (len(samples) - cut))
    test_X, test_y = np.array(test_X), np.array(test_y)

    model = make_model()
    model.fit(*balanced(train, rng))
    guesses = model.predict(test_X)

    names = sorted(groups)
    report = {"accuracy": float(np.mean(guesses == test_y)), "gestures": {}}
    print()
    print("test accuracy (on the last 20% of each recording):", round(report["accuracy"] * 100, 1), "%")
    for name in names:
        mask = test_y == name
        right = float(np.mean(guesses[mask] == name))
        # what it guessed instead when it got this one wrong
        mixups = {}
        for other in names:
            if other != name:
                share = float(np.mean(guesses[mask] == other))
                if share > 0:
                    mixups[other] = share
        report["gestures"][name] = {"accuracy": right, "mixed_up_with": mixups, "samples": len(groups[name])}
        line = "  " + name + ": " + str(round(right * 100, 1)) + " %"
        if mixups:
            line += "   (mistaken for " + ", ".join(o + " " + str(round(s * 100)) + "%" for o, s in sorted(mixups.items(), key=lambda kv: -kv[1])) + ")"
        print(line)

    # 4. final version, trained on everything
    final = make_model()
    final.fit(*balanced(groups, rng))
    os.makedirs("models", exist_ok=True)
    # save to a temp file first so gesture_cam.py never loads a half written one
    joblib.dump(final, MODEL_PATH + ".tmp")
    os.replace(MODEL_PATH + ".tmp", MODEL_PATH)
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print()
    print("saved to", MODEL_PATH)
    print("give your gestures effects and sounds in studio.py (or settings.json)")


if __name__ == "__main__":
    main()

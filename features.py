# features.py
# turns a hand (the 63 numbers from tracking.hand_features) into numbers that are easier for the
# neural network to learn from, and makes extra training examples (augmentation)
#
# the problem with raw points: tilt your hand a little and all 63 numbers change, even though the
# gesture is the same. so on top of the raw points we give the network things that DONT change
# when you tilt your hand:
#   - the hand turned to face the same way every time ("canonical" points)
#   - how much each finger joint bends
#   - how far apart the fingertips are
#   - how spread the fingers are
# plus which way the hand is facing (so thumbs up and thumbs down are still different)

import numpy as np

WRIST = 0
TIPS = [4, 8, 12, 16, 20]
# each finger as (joint before knuckle, knuckle, middle joints..., tip)
FINGERS = [[0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [0, 9, 10, 11, 12], [0, 13, 14, 15, 16], [0, 17, 18, 19, 20]]


def unit(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)


def angle(a, b):
    # angle between two directions, 0 to 1 (0 = same direction, 1 = opposite)
    cos = np.sum(unit(a) * unit(b), axis=-1)
    return np.arccos(np.clip(cos, -1, 1)) / np.pi


def hand_frame(p):
    # 3 directions attached to the hand: up the palm, across the palm, out of the palm
    up = unit(p[9] - p[WRIST])                       # wrist to middle knuckle
    across = p[5] - p[17]                            # pinky knuckle to index knuckle
    across = unit(across - np.dot(across, up) * up)  # make it exactly sideways to "up"
    out = np.cross(across, up)                       # sticks out of the palm
    return np.stack([across, up, out])


def pose_features(vec):
    # vec = 63 numbers from tracking.hand_features -> about 100 numbers for the network
    p = np.asarray(vec, np.float64).reshape(21, 3)
    frame = hand_frame(p)
    canonical = (p - p[WRIST]) @ frame.T             # same hand, turned to face the same way
    facing = np.concatenate([frame[1], frame[2]])    # which way the hand points and faces
    bends = []
    for f in FINGERS:
        for i in range(3):
            bends.append(angle(p[f[i + 1]] - p[f[i]], p[f[i + 2]] - p[f[i + 1]]))
    tips = p[TIPS]
    gaps = [np.linalg.norm(tips[i] - tips[j]) for i in range(5) for j in range(i + 1, 5)]
    directions = [p[f[4]] - p[f[1]] for f in FINGERS[1:]]
    spread = [angle(directions[i], directions[i + 1]) for i in range(3)]
    spread.append(angle(p[4] - p[2], directions[0]))  # thumb vs index
    return np.concatenate([canonical.ravel(), facing, bends, gaps, spread])


def pose_features_batch(X):
    # for the scikit-learn pipeline: many hands at once
    return np.array([pose_features(x) for x in np.asarray(X)])


def random_rotation(rng, max_degrees):
    ax, ay, az = np.radians(rng.uniform(-max_degrees, max_degrees, 3))
    rx = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]])
    ry = np.array([[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]])
    rz = np.array([[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1]])
    return rz @ ry @ rx


def augment(vec, rng, max_degrees=15, scale=0.08, jitter=0.015):
    # one made-up example: the same hand a bit tilted, a bit bigger/smaller, and a bit wobbly
    # (no mirroring: left hands are already flipped to look like right hands in hand_features)
    p = np.asarray(vec, np.float64).reshape(21, 3)
    p = p @ random_rotation(rng, max_degrees).T
    p = p * rng.uniform(1 - scale, 1 + scale)
    p = p + rng.normal(0, jitter, p.shape)
    p = p - p[WRIST]
    return p.ravel()

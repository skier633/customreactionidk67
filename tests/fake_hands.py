# made-up hands for tests: index finger out, the rest curled (a pen grip), that you can tilt

import math

import numpy as np


def pen_grip_points():
    # a hand in its own frame: index points along +x, y is down (like images), meters
    p = np.zeros((21, 3))
    p[0] = (-0.08, 0.03, 0)                                                                   # wrist
    p[1:5] = [(-0.06, 0.0, 0.02), (-0.04, -0.02, 0.02), (-0.025, -0.04, 0.02), (-0.015, -0.055, 0.02)]  # thumb up
    p[5:9] = [(0, 0, 0), (0.03, 0, 0), (0.055, 0, 0), (0.075, 0, 0)]                         # index straight
    for k, y in [(9, 0.02), (13, 0.04), (17, 0.06)]:                                         # the rest curled
        p[k:k + 4] = [(0, y, 0), (0.025, y, 0), (0.025, y + 0.02, 0), (0.005, y + 0.02, 0)]
    return p


def tilt(p, up_degrees, toward_camera_degrees=0):
    t = math.radians(up_degrees)
    x, y = p[:, 0] * math.cos(t) + p[:, 1] * math.sin(t), -p[:, 0] * math.sin(t) + p[:, 1] * math.cos(t)
    a = math.radians(toward_camera_degrees)
    z = p[:, 2]
    x, z = x * math.cos(a) + z * math.sin(a), -x * math.sin(a) + z * math.cos(a)
    return np.stack([x, y, z], 1)

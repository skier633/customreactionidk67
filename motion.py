"""Few-example, ordered motion recognition. One visible hand, pause between repetitions."""
from pathlib import Path
import numpy as np
from tracking import hand_features, palm_size

MODEL_PATH = 'models/motions.joblib'
REPORT_PATH = 'models/motions_report.json'
STEPS = 32


def sample(hand):
    return np.r_[hand['pts'][0], palm_size(hand['pts']), hand_features(hand)]


def encode(values, times):
    values, times = np.asarray(values), np.asarray(times)
    if len(values) < 6 or not np.isfinite(values).all() or np.any(np.diff(times) <= 0):
        raise ValueError('Need at least six finite, ordered observations')
    if not .25 <= times[-1] - times[0] <= 4.5:
        raise ValueError('Motion must last 0.25–4.5 seconds')
    # Preserve direction and amplitude; remove screen origin and camera distance.
    xy = (values[:, :2] - values[0, :2]) / max(np.median(values[:, 2]), 10)
    shape = values[:, 3:] * .3
    features = np.c_[xy, shape]
    grid = np.linspace(times[0], times[-1], STEPS)
    return np.column_stack([np.interp(grid, times, col) for col in features.T])


def distance(a, b):
    # Banded DTW. Normalize by fixed sequence length; never reorder or reverse time.
    cost = np.mean((a[:, None, :] - b[None, :, :]) ** 2, axis=2)
    cost += np.mean((a[:, None, :2] - b[None, :, :2]) ** 2, axis=2)
    n, m = cost.shape
    dp = np.full((n + 1, m + 1), np.inf)
    dp[0, 0] = 0
    for i in range(1, n + 1):
        for j in range(max(1, i - 8), min(m, i + 8) + 1):
            dp[i, j] = cost[i-1, j-1] + min(dp[i-1, j], dp[i, j-1], dp[i-1, j-1])
    return float(np.sqrt(dp[n, m] / n))


class MotionModel:
    def __init__(self, groups):
        self.groups = {k: np.asarray(v) for k, v in groups.items()}
        self.classes_ = np.array(sorted(groups))
        self.radius = {}
        for name, clips in self.groups.items():
            nearest = [min(distance(a, b) for j, b in enumerate(clips) if i != j)
                       for i, a in enumerate(clips)]
            self.radius[name] = max(.04, float(np.quantile(nearest, .9)) * 1.8)

    def predict_one(self, clip):
        scores = sorted((min(distance(clip, t) for t in clips), name)
                        for name, clips in self.groups.items())
        best, name = scores[0]
        if name == 'none' or best > self.radius[name]:
            return None
        if len(scores) > 1 and best >= scores[1][0] * .8:
            return None  # ambiguous; a distance is not a calibrated probability
        return name


class Segmenter:
    """Require a resting hand, motion, then a pause. Never join across missing tracks."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.key = None
        self.last = None
        self.anchor = None
        self.rest = None
        self.active = False
        self.values, self.times = [], []

    def update(self, hands, now):
        if len(hands) != 1:
            self.reset()
            return None
        hand = hands[0]
        key, value = hand.get('track_id', hand['label']), sample(hand)
        if not np.isfinite(value).all():
            self.reset()
            return None
        if key != self.key or self.last is None or not 0 < now - self.last <= .15:
            self.reset()
            self.key, self.last, self.anchor, self.rest = key, now, value, now
            return None
        self.last = now
        moved = (np.linalg.norm(value[:2] - self.anchor[:2]) / max(value[2], 10) > .12
                 or np.sqrt(np.mean((value[3:] - self.anchor[3:]) ** 2)) > .065)
        if moved:
            if not self.active and now - self.rest >= .3:
                self.active = True
                self.values, self.times = [self.anchor.copy()], [now - min(.1, now - self.rest)]
            self.anchor, self.rest = value.copy(), now
        if self.active:
            self.values.append(value)
            self.times.append(now)
            if now - self.times[0] > 4.5:
                self.reset()
                return None
            if now - self.rest >= .3:
                values, times = self.values, self.times
                self.active = False
                self.values, self.times = [], []
                try:
                    return encode(values, times)
                except ValueError:
                    return None
        return None


class MotionRecognizer:
    def __init__(self, model):
        self.model, self.segmenter, self.last_fired = model, Segmenter(), {}

    def update(self, hands, now):
        clip = self.segmenter.update(hands, now)
        if clip is None:
            return None
        name = self.model.predict_one(clip)
        if name is None or now - self.last_fired.get(name, -1e9) < 1:
            return None
        self.last_fired[name] = now
        x, y = hands[0]['pts'][0]
        return {'type': 'motion', 'name': name, 'x': x, 'y': y}


def recordings():
    return {p.name.removesuffix('.motion.npz'): p for p in Path('data').glob('*.motion.npz')}

"""Learn ordered movements from examples, with independent state for each hand."""
from pathlib import Path

import numpy as np

from tracking import hand_features, palm_size

MODEL_PATH = "models/motions.joblib"
REPORT_PATH = "models/motions_report.json"
STEPS = 32
WARP_WINDOW = 8              # Maximum timing displacement, in resampled observations.
MIN_DURATION = 0.25
MAX_DURATION = 4.5
REST_SECONDS = 0.3
MAX_FRAME_GAP = 0.15
MIN_OBSERVATIONS = 6
MIN_PALM_PIXELS = 10
SHAPE_WEIGHT = 0.3
WRIST_MOVEMENT = 0.12        # Palm lengths from the last movement anchor.
FINGER_MOVEMENT = 0.065      # RMS change in wrist-relative world landmarks.
EARLY_INTERVAL = 0.08
MOTION_COOLDOWN = 1.0


def identity(hand):
    return hand.get("track_id", hand["label"])


def sample(hand):
    return np.concatenate((hand["pts"][0], [palm_size(hand["pts"])], hand_features(hand)))


def encode(values, times):
    values = np.asarray(values)
    times = np.asarray(times)
    if (values.ndim != 2 or values.shape[1] != 66 or len(values) < MIN_OBSERVATIONS
            or times.shape != (len(values),) or not np.isfinite(values).all()
            or not np.isfinite(times).all() or np.any(np.diff(times) <= 0)):
        raise ValueError("Need at least six finite, ordered observations")
    if not MIN_DURATION <= times[-1] - times[0] <= MAX_DURATION:
        raise ValueError("Motion must last 0.25–4.5 seconds")
    # Remove the starting screen position and normalize distance by palm size.
    # Do not rotate, reverse, or resize the path: direction and amplitude matter.
    palm = max(np.median(values[:, 2]), MIN_PALM_PIXELS)
    wrist_path = (values[:, :2] - values[0, :2]) / palm
    finger_shapes = values[:, 3:] * SHAPE_WEIGHT
    features = np.column_stack((wrist_path, finger_shapes))
    timeline = np.linspace(times[0], times[-1], STEPS)
    return np.column_stack([np.interp(timeline, times, column) for column in features.T])


def resample_features(features):
    old_timeline = np.linspace(0, 1, len(features))
    new_timeline = np.linspace(0, 1, STEPS)
    return np.column_stack([np.interp(new_timeline, old_timeline, column)
                            for column in features.T])


def distance(first, second):
    """Align two sequences in time without changing the order of their observations."""
    differences = first[:, None, :] - second[None, :, :]
    costs = np.mean(differences ** 2, axis=2)
    # The 63 shape coordinates must not drown out the two wrist coordinates.
    costs += np.mean(differences[:, :, :2] ** 2, axis=2)
    rows, columns = costs.shape
    best_cost = np.full((rows + 1, columns + 1), np.inf)
    best_cost[0, 0] = 0
    for row in range(1, rows + 1):
        for column in range(max(1, row - WARP_WINDOW), min(columns, row + WARP_WINDOW) + 1):
            previous = min(best_cost[row - 1, column], best_cost[row, column - 1],
                           best_cost[row - 1, column - 1])
            best_cost[row, column] = costs[row - 1, column - 1] + previous
    return float(np.sqrt(best_cost[rows, columns] / rows))


class MotionModel:
    def __init__(self, groups):
        self.groups = {name: np.asarray(clips) for name, clips in groups.items()}
        self.classes_ = np.array(sorted(groups))
        self.radius = {}
        for name, clips in self.groups.items():
            if len(clips) < 2:
                raise ValueError("Each class needs at least two examples")
            nearest = [min(distance(clip, other) for j, other in enumerate(clips) if i != j)
                       for i, clip in enumerate(clips)]
            self.radius[name] = max(0.04, float(np.quantile(nearest, 0.9)) * 1.8)

    def predict_one(self, clip, early=False):
        scores = sorted((min(distance(clip, template) for template in clips), name)
                        for name, clips in self.groups.items())
        best, name = scores[0]
        radius_factor = 0.6 if early else 1.0
        separation = 0.6 if early else 0.8
        if name == "none" or best > self.radius[name] * radius_factor:
            return None
        if len(scores) > 1 and best >= scores[1][0] * separation:
            return None
        if early:
            # A completed short motion may also be the beginning of another class.
            # Reject matches near those prefixes. This is conservative, not proof
            # that an unrecorded longer movement cannot share the same beginning.
            for other_name, templates in self.groups.items():
                if other_name == name:
                    continue
                for template in templates:
                    for end in (12, 18, 24):
                        prefix = resample_features(template[:end])
                        if distance(clip, prefix) <= max(best / separation, self.radius[name]):
                            return None
        return name


class Segmenter:
    """Collect rest → movement → pause for a single physical hand."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.key = None
        self.last = None
        self.anchor = None
        self.rest = None
        self.active = False
        self.values = []
        self.times = []

    def update(self, hands, now):
        if len(hands) != 1:
            self.reset()
            return None
        hand = hands[0]
        key = identity(hand)
        value = sample(hand)
        if not np.isfinite(value).all():
            self.reset()
            return None
        if key != self.key or self.last is None or not 0 < now - self.last <= MAX_FRAME_GAP:
            self.reset()
            self.key = key
            self.last = now
            self.anchor = value
            self.rest = now
            return None
        self.last = now
        wrist_change = np.linalg.norm(value[:2] - self.anchor[:2]) / max(value[2], MIN_PALM_PIXELS)
        finger_change = np.sqrt(np.mean((value[3:] - self.anchor[3:]) ** 2))
        moved = wrist_change > WRIST_MOVEMENT or finger_change > FINGER_MOVEMENT
        if moved:
            if not self.active and now - self.rest >= REST_SECONDS:
                self.active = True
                self.values = [self.anchor.copy()]
                self.times = [now - min(0.1, now - self.rest)]
            self.anchor = value.copy()
            self.rest = now
        if self.active:
            self.values.append(value)
            self.times.append(now)
            if now - self.times[0] > MAX_DURATION:
                self.reset()
                return None
            if now - self.rest >= REST_SECONDS:
                values, times = self.values, self.times
                self.active = False
                self.values, self.times = [], []
                try:
                    return encode(values, times)
                except ValueError:
                    return None
        return None

    def early_candidate(self):
        if not self.active or len(self.values) < MIN_OBSERVATIONS:
            return None
        if self.times[-1] - self.times[0] < MIN_DURATION:
            return None
        # Compare against a hypothetical ending pause, without waiting for it.
        # This padding is only a matching hypothesis; it is never saved as observed data.
        values = self.values + [self.values[-1]]
        times = self.times + [self.times[-1] + REST_SECONDS]
        try:
            return encode(values, times)
        except ValueError:
            return None


class MotionRecognizer:
    def __init__(self, model, early_names=()):
        self.model = model
        self.early_names = set(early_names)
        self.tracks = {}
        self.last_fired = {}
        self.busy_ids = set()

    def update_all(self, hands, now):
        seen = {identity(hand) for hand in hands}
        self.tracks = {key: track for key, track in self.tracks.items() if key in seen}
        self.busy_ids = set()
        events = []
        for hand in hands:
            key = identity(hand)
            track = self.tracks.setdefault(key, {"segment": Segmenter(), "fired": False,
                                                 "checked": -1e9})
            segment = track["segment"]
            was_active = segment.active
            completed = segment.update([hand], now)
            if segment.active and not was_active:
                track["fired"] = False
            moving = segment.rest is not None and now - segment.rest < 0.2
            if segment.active or moving or completed is not None:
                self.busy_ids.add(key)
            name = None
            if not track["fired"]:
                if completed is not None:
                    name = self.model.predict_one(completed)
                elif self.early_names and now - track["checked"] >= EARLY_INTERVAL:
                    candidate = segment.early_candidate()
                    if candidate is not None:
                        track["checked"] = now
                        candidate_name = self.model.predict_one(candidate, early=True)
                        if candidate_name in self.early_names:
                            name = candidate_name
            if name is not None:
                track["fired"] = True  # Also latch a match suppressed by the cooldown.
                if now - self.last_fired.get(name, -1e9) >= MOTION_COOLDOWN:
                    self.last_fired[name] = now
                    x, y = hand["pts"][0]
                    events.append({"type": "motion", "name": name, "x": x, "y": y})
            if completed is not None:
                track["fired"] = False
        return events

    def update(self, hands, now):
        """Compatibility for callers expecting at most one event."""
        events = self.update_all(hands, now)
        return events[0] if events else None


def recordings():
    return {path.name.removesuffix(".motion.npz"): path for path in Path("data").glob("*.motion.npz")}

# pen.py
# the air pen: pinch your thumb and index finger together (other fingers curled in, like holding a
# pen) to draw with your index fingertip, let go to stop. the ink stays on the video until you clear
# it (or fades, if you set that up)
#
# clear it by holding a fist (or an open palm, see settings), pressing C in the preview window,
# or the "clear drawing" button in the studio
#
# the numbers below come from real recordings (tests/clips/pen_draw and pen_accidental): when
# drawing on purpose the other 3 fingers were always curled 130+ degrees, and in accidental pinches
# (touching your face, picking stuff up, talking with your hands) never more than about 105

import numpy as np
import cv2

from gestures import finger_bend
from tracking import THUMB_TIP, INDEX_TIP, palm_size

PINCH_DOWN = 0.3     # thumb and index tips closer than this (in palm sizes) starts a pinch
PINCH_UP = 0.45      # farther than this lifts the pen. in between it keeps doing what it was doing (no flicker)
PINCH_HOLD = 0.15    # the pinch has to last this long before ink shows up (quick accidental pinches dont count)
GRIP_CURL = 120      # "curled" grip: middle, ring and pinky have to bend at least this much (degrees)...
GRIP_RELEASE = 100   # ...and the pen lifts if they open up past this
RELEASE_GRACE = 0.1  # letting go only counts once its been open this long in total (a wobble back for a frame
                     # doesnt restart the count, it just takes a little off), so one glitchy frame doesnt break
                     # your line but letting go still feels instant
INDEX_FOLDED = 170   # when mediapipe misplaces the hidden index tip (usually onto a knuckle), the index finger
                     # suddenly looks folded way past what a real finger does (real: up to ~150, glitch: 185+).
                     # those frames dont count for letting go, and dont move the pen point
LOST_GRACE = 0.2     # same if mediapipe loses your hand for a moment
TIP_GAP_MAX = 0.5    # when pinching, index tip and thumb tip are next to each other on screen. if mediapipe puts
                     # the index tip farther than this from the thumb (in palm sizes), its probably guessing
                     # (the tip is hidden behind your thumb) so the pen follows your thumb instead
OFFSET_JUMP = 0.1    # the thumb -> pen point offset barely changes while you draw. a sudden jump bigger than
                     # this (in palm sizes) is the index tip getting misplaced (usually onto a knuckle), so
                     # its ignored, unless it stays that way for RELEASE_GRACE (then its a real change)
SMOOTH_SLOW = 1.5    # "one euro" smoothing: how steady the line is when you move slowly (lower = steadier)...
SMOOTH_FAST = 3.0    # ...and how much it lets fast moves through without lagging (higher = less lag)
MIN_STEP = 0.003     # dont add a point unless the pen moved this much (fraction of the screen)
CLEAR_HOLD = 0.6     # hold the clear gesture this long
FADE_TIME = 0.5      # how long the ink takes to fade away (when fade_after is on)

strokes = []   # each: {"points": [(x, y), ...] as fractions of the screen, "end": time or None, "color", "thickness"}
pens = {}      # hand label -> state of that hand's pen (see new_pen)
clearing = {"since": None, "done": False}


def pinch(hand):
    # how far apart the thumb and index tips are, in palm sizes
    world = hand["world"]
    return np.linalg.norm(world[THUMB_TIP] - world[INDEX_TIP]) / palm_size(world)


def curl(hand):
    # how curled the middle, ring and pinky fingers are (average bend in degrees)
    return np.mean([finger_bend(hand["world"], k) for k in (9, 13, 17)])


class OneEuro:
    # smoothing that follows fast moves closely but irons out shaking when you move slowly
    # (from "1 euro filter" by casiez, roussel & vogel, 2012)
    def __init__(self):
        self.x = None

    def __call__(self, x, t):
        if self.x is None:
            self.x, self.speed, self.t = x, np.zeros_like(x), t
            return x
        dt = max(t - self.t, 1e-3)
        self.t = t
        speed = (x - self.x) / dt
        self.speed = self.speed + blend(dt, 1.0) * (speed - self.speed)
        cutoff = SMOOTH_SLOW + SMOOTH_FAST * np.linalg.norm(self.speed)
        self.x = self.x + blend(dt, cutoff) * (x - self.x)
        return self.x


def blend(dt, cutoff):
    return 1 / (1 + 1 / (2 * np.pi * cutoff * dt))


def new_pen():
    return {
        "down": False, "stroke": None,
        "since": None, "pending": [],              # pinch started (before the ink shows up) and its points
        "open_for": 0.0, "unsure": [],             # how long its looked like you let go, and the points meanwhile
        "last_t": None,
        "last_seen": None, "offset": None,         # last time we saw this hand, last good thumb -> pen point offset
        "track_id": None,                         # a new hand must never inherit an old stroke
        "odd_offset": None, "not_pinching": None,  # since when the offset looked off / the pinch dropped while waiting
        "smooth": OneEuro(), "point": None,
    }


def pen_point(hand, pen, w, h, now):
    # halfway between the thumb tip and index tip (steadier than the index tip alone, which is often
    # hidden behind your thumb). if the index tip looks like a bad guess, follow the thumb instead,
    # using the last good offset between them
    pts = hand["pts"]
    palm = np.linalg.norm(pts[9] - pts[0]) + 1e-6
    offset = (pts[INDEX_TIP] - pts[THUMB_TIP]) / 2 / palm   # in palm sizes, so it works at any distance
    gap = np.linalg.norm(pts[INDEX_TIP] - pts[THUMB_TIP]) / palm
    folded = finger_bend(hand["world"], 5) > INDEX_FOLDED
    if pen["offset"] is None:
        pen["offset"] = offset
    elif gap <= TIP_GAP_MAX and not folded:
        if np.linalg.norm(offset - pen["offset"]) <= OFFSET_JUMP:
            pen.update(offset=offset, odd_offset=None)
        elif pen["odd_offset"] is None:
            pen["odd_offset"] = now
        elif now - pen["odd_offset"] >= RELEASE_GRACE:
            pen.update(offset=offset, odd_offset=None)   # its been like this a while, so its real
    return (pts[THUMB_TIP] + pen["offset"] * palm) / [w, h]


def is_down(hand):
    pen = pens.get(hand["label"])
    same_hand = pen and ("track_id" not in hand or hand["track_id"] == pen.get("track_id"))
    return bool(same_hand and pen["down"])


def clear():
    _ink_cache.size = None
    strokes.clear()
    for pen in pens.values():
        pen["stroke"] = None


def hex_to_bgr(color):
    color = color.lstrip("#")
    r, g, b = (int(color[i:i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)


def update(hands, now, w, h, cfg):
    # call once a frame with the hands from tracking.find_hands and the "pen" settings
    if not cfg["enabled"]:
        # turned off: no pinching, and the ink goes away
        for pen in pens.values():
            lift(pen, now)
        strokes.clear()
        return

    seen = set()
    for hand in hands:
        label = hand["label"]
        seen.add(label)
        pen = pens.setdefault(label, new_pen())
        track_id = hand.get("track_id")
        if track_id is not None and track_id != pen.get("track_id"):
            lift(pen, now)
            pen = pens[label] = new_pen()
            pen["track_id"] = track_id
        amount = pinch(hand)
        fingers = curl(hand)
        any_grip = cfg["grip"] == "any"
        previous_seen = pen["last_seen"]
        came_back = previous_seen is not None and now - previous_seen > 0.1
        pen["last_seen"] = now
        point = pen["smooth"](pen_point(hand, pen, w, h, now), now)

        if pen["down"] and pen["stroke"] is None:
            start_stroke(pen, cfg, [])  # the drawing got cleared while you were still drawing
        if pen["down"]:
            gone_for = now - previous_seen if previous_seen is not None else 0
            if came_back and np.linalg.norm(point - pen["point"]) > 0.08 + 1.5 * gone_for:
                # lost the hand for a moment and it came back somewhere else: dont draw a line across
                end_stroke(pen, now)
                start_stroke(pen, cfg, [])
            dt = now - pen["last_t"] if pen["last_t"] is not None else 0
            glitch = finger_bend(hand["world"], 5) > INDEX_FOLDED
            letting_go = amount > PINCH_UP or (not any_grip and fingers < GRIP_RELEASE)
            if glitch:
                pass  # cant tell what your fingers are doing this frame, so it doesnt count either way
            elif letting_go:
                pen["open_for"] += dt
            else:
                pen["open_for"] = max(0.0, pen["open_for"] - dt)
            if pen["open_for"] >= RELEASE_GRACE:
                lift(pen, now)  # really let go. the unsure points get thrown away
            elif pen["open_for"] > 0 or glitch:
                add_point(pen["unsure"], point)  # not sure yet if youre still drawing
            else:
                for p in pen["unsure"]:  # it was just a glitch, keep those points
                    add_point(pen["stroke"]["points"], p)
                pen["unsure"] = []
                add_point(pen["stroke"]["points"], point)
        elif amount < PINCH_DOWN and (any_grip or fingers >= GRIP_CURL):
            # pinching. remember where the pen goes, and start the ink once it lasts long enough
            if pen["since"] is None:
                pen.update(since=now, pending=[])
            pen["not_pinching"] = None
            add_point(pen["pending"], point)
            if now - pen["since"] >= PINCH_HOLD:
                start_stroke(pen, cfg, pen["pending"])
        elif pen["since"] is not None and (pen["not_pinching"] is None or now - pen["not_pinching"] < RELEASE_GRACE):
            # was starting a pinch and it dropped for a frame. give it a moment before giving up
            if pen["not_pinching"] is None:
                pen["not_pinching"] = now
        else:
            pen.update(since=None, pending=[], not_pinching=None)
            pen["smooth"] = OneEuro()  # start fresh next time, so the line doesnt lag in from the old spot
        pen["point"] = point
        pen["last_t"] = now

    # hands that left the screen lift their pen (after a moment, in case its just a tracking hiccup)
    for label, pen in pens.items():
        if label not in seen and pen["last_seen"] is not None and now - pen["last_seen"] >= LOST_GRACE:
            if pen["down"]:
                lift(pen, now)
            pen.update(since=None, pending=[], last_seen=None, offset=None, smooth=OneEuro())

    check_clear(hands, now, cfg)

    # with fade_after on, old ink fades and then goes away
    if cfg["fade_after"] > 0:
        strokes[:] = [s for s in strokes if s["end"] is None or now - s["end"] < cfg["fade_after"] + FADE_TIME]


def add_point(points, pos):
    if not points or np.linalg.norm(pos - points[-1]) > MIN_STEP:
        points.append(pos.copy())


def start_stroke(pen, cfg, points):
    pen["stroke"] = {"points": points, "end": None, "color": cfg["color"], "thickness": cfg["thickness"]}
    strokes.append(pen["stroke"])
    pen.update(down=True, since=None, pending=[], open_for=0.0, unsure=[], not_pinching=None)


def end_stroke(pen, now):
    if pen["stroke"] is not None:
        pen["stroke"]["end"] = now
        if len(pen["stroke"]["points"]) < 2:  # just a tap, not a line
            strokes[:] = [s for s in strokes if s is not pen["stroke"]]
    pen["stroke"] = None


def lift(pen, now):
    end_stroke(pen, now)
    pen.update(down=False, since=None, pending=[], open_for=0.0, unsure=[])


def check_clear(hands, now, cfg):
    gesture = {"open_palm": "Open_Palm", "fist": "Closed_Fist"}.get(cfg["clear_with"])
    showing = gesture is not None and any(
        hand.get("ready_made", ("None", 0))[0] == gesture and hand["ready_made"][1] >= 0.6 and not is_down(hand)
        for hand in hands)
    if not showing:
        clearing.update(since=None, done=False)
        return
    if clearing["since"] is None:
        clearing["since"] = now
    if not clearing["done"] and now - clearing["since"] >= CLEAR_HOLD and strokes:
        clear()
        clearing["done"] = True
        print("cleared the drawing")


def _draw_fading(frame, now, cfg, mirror=False):
    # draws all the ink on the frame. glow = a soft colored halo around the line
    # mirror=True flips it left to right (for the mirrored preview, and for the call so writing
    # reads the right way for the other person)
    if not cfg["enabled"] or (not strokes and not any(p["down"] for p in pens.values())):
        return
    h, w = frame.shape[:2]

    def to_pixels(points):
        p = np.array(points) * [w, h]
        if mirror:
            p[..., 0] = (w - 1) - p[..., 0]  # same as cv2.flip does to the video
        return np.rint(p).astype(np.int32)

    scale = h / 720  # thickness is in pixels at 720p
    glow = np.zeros_like(frame) if cfg["glow"] else None
    for s in strokes:
        if len(s["points"]) < 2:
            continue
        fade = 1.0
        if cfg["fade_after"] > 0 and s["end"] is not None:
            fade = 1 - min(1.0, max(0.0, (now - s["end"] - cfg["fade_after"]) / FADE_TIME))
        if fade <= 0:
            continue
        pts = to_pixels(s["points"])
        color = hex_to_bgr(s["color"])
        thick = max(1, int(s["thickness"] * scale))
        if fade < 1:
            layer = frame.copy()
            cv2.polylines(layer, [pts], False, color, thick, cv2.LINE_AA)
            frame[:] = cv2.addWeighted(layer, fade, frame, 1 - fade, 0)
        else:
            cv2.polylines(frame, [pts], False, color, thick, cv2.LINE_AA)
        if glow is not None:
            cv2.polylines(glow, [pts], False, tuple(int(c * fade) for c in color), thick * 3, cv2.LINE_AA)
    if glow is not None:
        small = cv2.GaussianBlur(cv2.resize(glow, (w // 4, h // 4)), (0, 0), 3)
        frame[:] = cv2.add(frame, cv2.resize(small, (w, h)))
    # a white dot on the fingertip while drawing, so you can see where the pen is
    for pen in pens.values():
        if pen["down"] and pen["point"] is not None:
            x, y = to_pixels(pen["point"])
            cv2.circle(frame, (x, y), max(3, int(cfg["thickness"] * scale * 0.6)), (255, 255, 255), -1, cv2.LINE_AA)


class InkCache:
    """Rasterize new segments once; reuse the ink for both video outputs."""
    def __init__(self):
        self.size = None
        self.entries = []

    def refresh(self, width, height, glow):
        entries = [(id(stroke), len(stroke['points']), stroke['color'], stroke['thickness'])
                   for stroke in strokes]
        reset = self.size != (width, height, glow)
        if not reset:
            reset = len(entries) < len(self.entries)
            for index, old in enumerate(self.entries):
                if index >= len(entries) or entries[index][0] != old[0] or entries[index][2:] != old[2:] or entries[index][1] < old[1]:
                    reset = True
                # Preserve overlap order when two pens append to different strokes.
                if index < len(self.entries) - 1 and index < len(entries) and entries[index] != old:
                    reset = True
        if reset:
            self.size = (width, height, glow)
            self.entries = []
            self.ink = np.zeros((height, width, 3), np.uint8)
            self.mask = np.zeros((height, width), np.uint8)
            self.glow_ink = np.zeros((max(1, height // 4), max(1, width // 4), 3), np.uint8)
        if entries == self.entries and not reset:
            return
        for index, stroke in enumerate(strokes):
            previous = self.entries[index][1] if index < len(self.entries) else 0
            if len(stroke['points']) < 2 or len(stroke['points']) == previous:
                continue
            points = np.rint(np.asarray(stroke['points'][max(0, previous - 1):]) * [width, height]).astype(np.int32)
            color = hex_to_bgr(stroke['color'])
            thickness = max(1, int(stroke['thickness'] * height / 720))
            cv2.polylines(self.ink, [points], False, color, thickness, cv2.LINE_AA)
            cv2.polylines(self.mask, [points], False, 255, thickness, cv2.LINE_AA)
            if glow:
                small_points = np.rint(points / [width, height] * self.glow_ink.shape[1::-1]).astype(np.int32)
                cv2.polylines(self.glow_ink, [small_points], False, color,
                              max(1, round(thickness * 3 / 4)), cv2.LINE_AA)
        self.entries = entries
        self.sources = list(strokes)
        self.inverse_alpha = cv2.cvtColor(1 - self.mask.astype(np.float32) / 255, cv2.COLOR_GRAY2BGR)
        self.layer = self.ink
        if glow:
            blurred = cv2.GaussianBlur(self.glow_ink, (0, 0), 3)
            self.layer = cv2.add(self.ink, cv2.resize(blurred, (width, height)))


_ink_cache = InkCache()


def draw(frame, now, cfg, mirror=False):
    if not cfg['enabled'] or (not strokes and not any(p['down'] for p in pens.values())):
        return
    # Fading changes old strokes every frame; retain its existing compositing behavior.
    if cfg['fade_after'] > 0:
        return _draw_fading(frame, now, cfg, mirror)
    height, width = frame.shape[:2]
    _ink_cache.refresh(width, height, cfg['glow'])
    layer = _ink_cache.layer[:, ::-1] if mirror else _ink_cache.layer
    inverse = _ink_cache.inverse_alpha[:, ::-1] if mirror else _ink_cache.inverse_alpha
    cv2.add(cv2.multiply(frame, inverse, dtype=cv2.CV_8U), layer, dst=frame)
    for state in pens.values():
        if state['down'] and state['point'] is not None:
            x, y = np.rint(state['point'] * [width, height]).astype(int)
            if mirror:
                x = width - 1 - x
            cv2.circle(frame, (x, y), max(3, int(cfg['thickness'] * height / 720 * 0.6)),
                       (255, 255, 255), -1, cv2.LINE_AA)

# effects.py
# everything that gets drawn on the screen
#
# when a gesture goes off, trigger() starts a "running" effect using that gesture's settings
# (see settings.py). every frame update_and_draw() draws all the running ones, in this order:
#   gif/picture behind you > particles > picture/gif on top > distortion > color filter > shake
# and draw_text() adds the captions last (it's separate so the text doesnt come out backwards
# in the mirrored preview)
#
# particles are just dicts with a position, speed, how long they live, and what kind they are

import os
import math
import random

import numpy as np
import cv2

particles = []
waiting = []  # effects that start a little later (like the fireworks going off one by one)
running = []  # gesture effects that are going right now
media = {}    # pictures and gifs from assets/: name -> {"frames": [...], "frame_time": seconds}
last_face = {"center": None, "width": None}

GRAVITY = 300  # pixels per second per second
FADE_TIME = 0.15
MAX_MEDIA_SIZE = 1280  # big gifs get shrunk to this when loading so they dont eat all the memory

# heart shape (the heart curve formula from wikipedia)
t = np.linspace(0, 2 * np.pi, 40)
HEART_SHAPE = np.stack([16 * np.sin(t) ** 3,
                        -(13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t))], axis=1) / 17.0

# colors are BGR in opencv not RGB!
HEART_COLORS = [(180, 105, 255), (60, 60, 255), (147, 20, 255), (203, 192, 255)]
CONFETTI_COLORS = [(0, 215, 255), (255, 144, 30), (60, 60, 255), (50, 205, 50), (255, 0, 255), (0, 165, 255)]
FIREWORK_COLORS = [(0, 215, 255), (255, 191, 0), (147, 20, 255), (50, 255, 50), (60, 60, 255), (255, 255, 255)]

# old photo look (BGR in, BGR out)
SEPIA = np.array([[0.131, 0.534, 0.272],
                  [0.168, 0.686, 0.349],
                  [0.189, 0.769, 0.393]], np.float32)

# fonts for the captions. first one that exists wins
FONTS = ["/System/Library/Fonts/Supplemental/Arial Rounded Bold.ttf",
         "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
         "/Library/Fonts/Arial Bold.ttf"]


# ---------- loading ----------

def shrink(img):
    h, w = img.shape[:2]
    scale = MAX_MEDIA_SIZE / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def read_media(path):
    # pictures are just 1 frame. gifs keep their see-through parts
    name = path.lower()
    if name.endswith(".gif"):
        from PIL import Image, ImageSequence
        gif = Image.open(path)
        frames, times = [], []
        for f in ImageSequence.Iterator(gif):
            frames.append(shrink(cv2.cvtColor(np.array(f.convert("RGBA")), cv2.COLOR_RGBA2BGRA)))
            times.append(f.info.get("duration", 50) or 50)
        return frames, sum(times) / len(times) / 1000
    if name.endswith((".mp4", ".mov", ".m4v")):
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(shrink(f))
        cap.release()
        return frames, 1 / fps if fps > 0 else 1 / 30
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return [], 1
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return [shrink(img)], 1


def load_media(names, folder="assets"):
    # loads the pictures/gifs the settings use. skips ones that are already loaded and didnt change
    for name in list(media):
        if name not in names:
            del media[name]
    for name in names:
        path = os.path.join(folder, name)
        if not os.path.exists(path):
            print("cant find", path)
            media.pop(name, None)
            continue
        mtime = os.path.getmtime(path)
        if name in media and media[name]["mtime"] == mtime:
            continue
        frames, frame_time = read_media(path)
        if not frames:
            print("couldnt read", path)
            continue
        media[name] = {"frames": frames, "frame_time": max(frame_time, 0.01), "mtime": mtime}
        print("loaded", name + (" (" + str(len(frames)) + " frames)" if len(frames) > 1 else ""))


# ---------- starting effects ----------

def trigger(key, cfg, x, y, w, h):
    # starts everything a gesture is set up to do. key = gesture name like "thumbs_up" or "rock_on_x2"
    if cfg["particles"] == "hearts":
        spawn_hearts(x, y, h)
    elif cfg["particles"] == "confetti":
        spawn_confetti(w, h)
    elif cfg["particles"] == "fireworks":
        spawn_fireworks(w, h)
    if cfg["picture"] and cfg["picture_where"] == "burst":
        spawn_image_burst(cfg["picture"], x, y, h)
    # doing the same gesture again restarts it instead of stacking
    running[:] = [r for r in running if r["key"] != key]
    running.append({"key": key, "cfg": cfg, "t": 0.0, "x": x, "y": y, "text_img": None})


def stop_all():
    running.clear()


def needs_person():
    # true if something is going behind you, so gesture_cam knows to cut you out
    return any(uses(r["cfg"], "behind") for r in running)


def needs_face():
    return any(uses(r["cfg"], "face") or (r["cfg"]["distortion"] and r["cfg"]["distortion_where"] == "face")
               for r in running)


def uses(cfg, where):
    return ((cfg["picture"] and cfg["picture_where"] == where) or
            (cfg["gif"] and cfg["gif_where"] == where))


def spawn_hearts(x, y, h, count=25):
    for i in range(count):
        particles.append({
            "kind": "heart",
            "x": x + random.uniform(-40, 40), "y": y + random.uniform(-20, 20),
            "vx": random.uniform(-120, 120), "vy": random.uniform(-260, -90),
            "age": -random.uniform(0, 0.6),  # negative age = wait a bit before showing up
            "life": random.uniform(1.8, 2.8),
            "size": random.uniform(0.03, 0.07) * h,
            "color": random.choice(HEART_COLORS),
            "wobble": random.uniform(0, 6.28),
        })


def spawn_confetti(w, h, count=160):
    for i in range(count):
        particles.append({
            "kind": "confetti",
            "x": random.uniform(0, w), "y": random.uniform(-h * 0.5, 0),
            "vx": random.uniform(-60, 60), "vy": random.uniform(120, 280),
            "age": 0, "life": 4.0,
            "size": random.uniform(0.012, 0.022) * h,
            "color": random.choice(CONFETTI_COLORS),
            "angle": random.uniform(0, 6.28), "spin": random.uniform(-8, 8),
        })


def spawn_fireworks(w, h):
    for i in range(5):
        x = random.uniform(0.15, 0.85) * w
        y = random.uniform(0.15, 0.5) * h
        waiting.append({"delay": i * 0.25, "func": firework_burst, "args": (x, y, h)})


def firework_burst(x, y, h):
    color = random.choice(FIREWORK_COLORS)
    for i in range(70):
        angle = random.uniform(0, 6.28)
        speed = random.uniform(0.25, 0.6) * h
        particles.append({
            "kind": "spark",
            "x": x, "y": y,
            "vx": math.cos(angle) * speed, "vy": math.sin(angle) * speed,
            "age": 0, "life": random.uniform(1.0, 1.6),
            "size": 3, "color": color, "gravity": True,
        })


def spawn_image_burst(name, x, y, h, count=12):
    if name not in media:
        return
    picture = media[name]["frames"][0]
    for i in range(count):
        size = int(random.uniform(0.06, 0.12) * h)
        img = cv2.resize(picture, (size, int(size * picture.shape[0] / picture.shape[1])))
        angle = random.uniform(0, 6.28)
        speed = random.uniform(100, 300)
        particles.append({
            "kind": "image", "img": img,
            "x": x, "y": y,
            "vx": math.cos(angle) * speed, "vy": math.sin(angle) * speed - 150,
            "age": 0, "life": random.uniform(1.2, 2.0),
        })


def draw_heart(img, x, y, size, color):
    pts = (HEART_SHAPE * size + [x, y]).astype(np.int32)
    cv2.fillPoly(img, [pts], color, cv2.LINE_AA)
    # little shiny spot
    cv2.circle(img, (int(x - size * 0.35), int(y - size * 0.35)), max(1, int(size * 0.15)), (255, 255, 255), -1, cv2.LINE_AA)


def draw_confetti(img, p):
    s = p["size"]
    c, si = math.cos(p["angle"]), math.sin(p["angle"])
    # rectangle that's 1 wide and 0.5 tall, rotated. the cos(age) part makes it look like its flipping
    flip = abs(math.cos(p["age"] * 5 + p["spin"]))
    corners = []
    for cx, cy in [(-1, -0.5), (1, -0.5), (1, 0.5), (-1, 0.5)]:
        cx *= s
        cy *= s * flip
        corners.append((p["x"] + cx * c - cy * si, p["y"] + cx * si + cy * c))
    cv2.fillPoly(img, [np.array(corners, np.int32)], p["color"], cv2.LINE_AA)


def draw_png(frame, img, cx, cy, opacity=1.0):
    # pastes a picture (with see-through parts if it has them) onto the frame, centered on cx, cy
    H, W = frame.shape[:2]
    h, w = img.shape[:2]
    x0 = int(cx - w / 2)
    y0 = int(cy - h / 2)
    fx0, fy0 = max(x0, 0), max(y0, 0)
    fx1, fy1 = min(x0 + w, W), min(y0 + h, H)
    if fx1 <= fx0 or fy1 <= fy0 or opacity <= 0:
        return
    part = img[fy0 - y0:fy1 - y0, fx0 - x0:fx1 - x0]
    if part.shape[2] == 4:
        alpha = part[:, :, 3:4] / 255.0 * opacity
        color = part[:, :, :3]
    else:
        alpha = opacity
        color = part
    area = frame[fy0:fy1, fx0:fx1]
    frame[fy0:fy1, fx0:fx1] = (area * (1 - alpha) + color * alpha).astype(np.uint8)


# ---------- helpers for the running effects ----------

def level(r):
    # 0 to 1. how "on" the effect is right now (for fading in and out)
    if not r["cfg"]["fade"]:
        return 1.0
    left = r["cfg"]["duration"] - r["t"]
    return max(0.0, min(1.0, r["t"] / FADE_TIME, left / FADE_TIME))


def anchor(r, where, w, h):
    # where on the screen something goes, and how big (in pixels)
    if where == "face" and last_face["center"] is not None:
        return last_face["center"][0], last_face["center"][1], last_face["width"]
    if where in ("hand", "face"):  # no face found yet, so use the hand
        return r["x"], r["y"], 0.3 * h
    return w / 2, h / 2, 0.45 * h  # middle of the screen


def current_frame(name, t):
    m = media[name]
    return m["frames"][int(t / m["frame_time"]) % len(m["frames"])]


def media_layers(r):
    # the picture and/or gif this effect shows: [(name, where)]
    cfg = r["cfg"]
    layers = []
    if cfg["picture"] in media and cfg["picture_where"] != "burst":
        layers.append((cfg["picture"], cfg["picture_where"]))
    if cfg["gif"] in media:
        layers.append((cfg["gif"], cfg["gif_where"]))
    return layers


def draw_behind(frame, img, person_mask, opacity):
    # puts img behind you. person_mask is 1 where you are, 0 for the background
    h, w = frame.shape[:2]
    big = cv2.resize(img, (w, h))
    weight = (1 - person_mask.reshape(h, w, 1).astype(np.float32)) * opacity
    if big.shape[2] == 4:
        weight = weight * (big[:, :, 3:4] / 255.0)
        big = big[:, :, :3]
    frame[:] = (frame * (1 - weight) + big * weight).astype(np.uint8)


def draw_on_top(frame, img, r, where, opacity):
    h, w = frame.shape[:2]
    if where == "full":
        draw_png(frame, cv2.resize(img, (w, h)), w / 2, h / 2, opacity)
        return
    cx, cy, size = anchor(r, where, w, h)
    if where == "face":
        size *= 1.3  # a bit bigger than your face so it covers it
    scale = size / max(img.shape[:2])
    small = cv2.resize(img, (max(1, int(img.shape[1] * scale)), max(1, int(img.shape[0] * scale))))
    draw_png(frame, small, cx, cy, opacity)


grids = {}


def grid(y0, y1, x0, x1):
    # pixel coordinates for remap. cached since its the same every frame
    k = (y0, y1, x0, x1)
    if k not in grids:
        if len(grids) > 20:
            grids.clear()
        ys, xs = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        grids[k] = (ys, xs)
    return grids[k]


def distort(frame, kind, cx, cy, radius, amount, t):
    # funhouse mirror stuff. amount is 0 to 1
    h, w = frame.shape[:2]
    if amount <= 0.01:
        return
    if kind == "wave":
        ys, xs = grid(0, h, 0, w)
        size = amount * 0.04 * h
        map_x = xs + size * np.sin(ys / (0.06 * h) + t * 8)
        map_y = ys + size * 0.5 * np.sin(xs / (0.09 * h) + t * 6)
        frame[:] = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        return
    if kind == "mirror":
        # right half becomes a flipped copy of the left half
        half = w // 2
        mirrored = cv2.flip(frame[:, :w - half], 1)
        frame[:, half:] = cv2.addWeighted(mirrored, amount, frame[:, half:], 1 - amount, 0)
        return

    x0, x1 = int(max(0, cx - radius)), int(min(w, cx + radius))
    y0, y1 = int(max(0, cy - radius)), int(min(h, cy + radius))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return
    if kind == "pixelate":
        block = int(4 + amount * 0.12 * radius)
        part = frame[y0:y1, x0:x1]
        small = cv2.resize(part, (max(1, (x1 - x0) // block), max(1, (y1 - y0) // block)), interpolation=cv2.INTER_AREA)
        frame[y0:y1, x0:x1] = cv2.resize(small, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)
        return

    # inside the circle we grab pixels from somewhere else
    ys, xs = grid(y0, y1, x0, x1)
    dx = xs - cx
    dy = ys - cy
    d = np.sqrt(dx * dx + dy * dy) / radius
    inside = d < 1
    if kind == "bulge":
        # grab from closer to the middle, which stretches the middle out
        f = np.where(inside, 1 - 0.7 * amount * (1 - d * d), 1.0)
        map_x, map_y = cx + dx * f, cy + dy * f
    elif kind == "pinch":
        # grab from farther out, which squishes the middle
        f = np.where(inside, 1 + 0.8 * amount * (1 - d) ** 2, 1.0)
        map_x, map_y = cx + dx * f, cy + dy * f
    elif kind == "swirl":
        angle = np.where(inside, amount * 4 * (1 - d) ** 2, 0.0)
        c, s = np.cos(angle), np.sin(angle)
        map_x, map_y = cx + dx * c - dy * s, cy + dx * s + dy * c
    else:
        return
    frame[y0:y1, x0:x1] = cv2.remap(frame, map_x.astype(np.float32), map_y.astype(np.float32),
                                    cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def color_filter(frame, kind, amount):
    if amount <= 0.01:
        return
    if kind == "gray":
        out = cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    elif kind == "sepia":
        out = cv2.transform(frame, SEPIA)
    elif kind == "invert":
        out = 255 - frame
    elif kind == "deepfried":
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 2.5, 0, 255)
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        out = cv2.convertScaleAbs(out, alpha=1.6, beta=-40)
        out = cv2.filter2D(out, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32))
        ok, jpg = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 6])
        out = cv2.imdecode(jpg, cv2.IMREAD_COLOR)
    elif kind == "thermal":
        out = cv2.applyColorMap(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLORMAP_JET)
    elif kind == "blur":
        h, w = frame.shape[:2]
        small = cv2.GaussianBlur(cv2.resize(frame, (w // 4, h // 4)), (0, 0), 3)
        out = cv2.resize(small, (w, h))
    elif kind == "red":
        out = cv2.convertScaleAbs(frame * np.array([0.4, 0.4, 1.3], np.float32))
    elif kind == "blue":
        out = cv2.convertScaleAbs(frame * np.array([1.3, 0.6, 0.4], np.float32))
    else:
        return
    frame[:] = cv2.addWeighted(out, amount, frame, 1 - amount, 0)


def shake(frame, amount):
    h, w = frame.shape[:2]
    size = amount * 0.05 * h
    if size < 1:
        return
    m = np.float32([[1, 0, random.uniform(-size, size)], [0, 1, random.uniform(-size, size)]])
    frame[:] = cv2.warpAffine(frame, m, (w, h), borderMode=cv2.BORDER_REFLECT)


# ---------- drawing everything ----------

def update_and_draw(frame, now, dt, person_mask=None, face=None):
    # face = (center, width) if gesture_cam found your face this frame
    h, w = frame.shape[:2]
    glow = np.zeros_like(frame)  # bright stuff gets drawn here too, then blurred so it glows
    used_glow = False

    if face is not None:
        last_face["center"], last_face["width"] = face

    # start any effects that were waiting
    for item in waiting[:]:
        item["delay"] -= dt
        if item["delay"] <= 0:
            item["func"](*item["args"])
            waiting.remove(item)

    # move the running effects along and drop the finished ones
    # (filtering the list instead of .remove() because comparing dicts with pictures in them crashes)
    for r in running:
        r["t"] += dt
    running[:] = [r for r in running if r["t"] <= r["cfg"]["duration"]]

    # stuff behind you
    for r in running:
        for name, where in media_layers(r):
            if where == "behind" and person_mask is not None:
                draw_behind(frame, current_frame(name, r["t"]), person_mask, r["cfg"]["opacity"] / 100 * level(r))

    # particles. drop the dead ones first (same reason as above)
    for p in particles:
        p["age"] += dt
    particles[:] = [p for p in particles if p["age"] <= p["life"] and p["y"] <= h + 100]
    for p in particles:
        if p["age"] < 0:
            continue  # not born yet
        if p["kind"] == "spark" and p.get("gravity"):
            p["vy"] += GRAVITY * dt
        if p["kind"] == "image":
            p["vy"] += GRAVITY * 0.5 * dt
        if p["kind"] == "heart":
            p["vx"] += math.sin(p["age"] * 4 + p["wobble"]) * 200 * dt
            p["vx"] *= 0.98
        if p["kind"] == "spark":
            p["vx"] *= 0.97
            p["vy"] *= 0.97
        p["x"] += p["vx"] * dt
        p["y"] += p["vy"] * dt
        left = p["life"] - p["age"]
        shrink_amount = min(1.0, left / 0.4)  # shrinks away at the end of its life

        if p["kind"] == "heart":
            pop = min(1.0, p["age"] / 0.15)  # pops in at the start
            draw_heart(frame, p["x"], p["y"], p["size"] * shrink_amount * pop, p["color"])
        elif p["kind"] == "confetti":
            p["angle"] += p["spin"] * dt
            draw_confetti(frame, p)
        elif p["kind"] == "spark":
            r = max(1, int(p["size"] * shrink_amount + 0.5))
            pos = (int(p["x"]), int(p["y"]))
            cv2.circle(frame, pos, r, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(glow, pos, r * 3, p["color"], -1, cv2.LINE_AA)
            used_glow = True
        elif p["kind"] == "image":
            draw_png(frame, p["img"], p["x"], p["y"])

    # make the glow layer blurry and add it on top. doing the blur on a small copy is way faster
    if used_glow:
        small = cv2.resize(glow, (w // 4, h // 4))
        small = cv2.GaussianBlur(small, (0, 0), 4)
        frame[:] = cv2.add(frame, cv2.resize(small, (w, h)))

    # pictures and gifs on top
    for r in running:
        for name, where in media_layers(r):
            if where != "behind":
                draw_on_top(frame, current_frame(name, r["t"]), r, where, r["cfg"]["opacity"] / 100 * level(r))

    # distortion warps everything under it (you, the pictures, the particles)
    for r in running:
        cfg = r["cfg"]
        if cfg["distortion"]:
            cx, cy, size = anchor(r, cfg["distortion_where"], w, h)
            radius = size * 0.75 if cfg["distortion_where"] == "face" else size * 0.6
            distort(frame, cfg["distortion"], cx, cy, radius, cfg["strength"] / 100 * level(r), r["t"])

    # color filters and shaking go over everything
    for r in running:
        if r["cfg"]["filter"]:
            color_filter(frame, r["cfg"]["filter"], level(r))
    for r in running:
        if r["cfg"]["shake"]:
            # starts strong and calms down
            calm = 1 - r["t"] / r["cfg"]["duration"]
            shake(frame, r["cfg"]["shake"] / 100 * calm)

    return frame


def make_text_image(text, height):
    # white text with a black outline, as a see-through picture (pillow does nicer text than opencv)
    from PIL import Image, ImageDraw, ImageFont
    size = int(height * 0.09)
    font = None
    for path in FONTS:
        if os.path.exists(path):
            font = ImageFont.truetype(path, size)
            break
    if font is None:
        font = ImageFont.load_default(size=size)
    stroke = max(2, size // 12)
    box = ImageDraw.Draw(Image.new("RGBA", (1, 1))).textbbox((0, 0), text, font=font, stroke_width=stroke)
    img = Image.new("RGBA", (box[2] - box[0] + 4, box[3] - box[1] + 4), (0, 0, 0, 0))
    ImageDraw.Draw(img).text((2 - box[0], 2 - box[1]), text, font=font, fill=(255, 255, 255, 255),
                             stroke_width=stroke, stroke_fill=(0, 0, 0, 255))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGRA)


def draw_text(frame):
    # captions along the bottom. call this on the final picture (and again on the mirrored preview)
    h, w = frame.shape[:2]
    y = h * 0.86
    for r in reversed(running):
        text = r["cfg"]["text"]
        if not text:
            continue
        if r["text_img"] is None or r["text_img"][0] != h:
            img = make_text_image(text, h)
            if img.shape[1] > w * 0.94:  # too wide, shrink it to fit
                scale = w * 0.94 / img.shape[1]
                img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))
            r["text_img"] = (h, img)
        img = r["text_img"][1]
        draw_png(frame, img, w / 2, y, level(r))
        y -= img.shape[0] * 1.1

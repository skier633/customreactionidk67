# tools/make_demo_assets.py
# makes the demo explosion and bang sound that come with the repo, from scratch with code
# (so theyre original and fine to share under the repo's license)
#
# usage (from the project folder): python tools/make_demo_assets.py
#   -> assets/demo_explosion.gif and sounds/demo_bang.wav

import os
import wave

import numpy as np
from PIL import Image
from scipy.ndimage import zoom

ROOT = os.path.join(os.path.dirname(__file__), "..")
W, H = 640, 360
FRAMES = 36
FRAME_MS = 60
RATE = 44100


def smooth_noise(rng, shape, cells):
    # random values on a coarse grid, smoothly blown up to `shape`. a few sizes added together
    # looks like fire/smoke turbulence
    total = np.zeros(shape)
    for i, c in enumerate(cells):
        coarse = rng.random([max(2, int(s / c)) for s in shape])
        fine = zoom(coarse, [s / n for s, n in zip(shape, coarse.shape)], order=3)
        total += fine[:shape[0], :shape[1], :shape[2]] / (i + 1)
    total -= total.min()
    return total / total.max()


def heat_to_color(heat):
    # 0 = dark smoke, 1 = white hot
    stops = np.array([0.0, 0.15, 0.35, 0.6, 0.85, 1.0])
    colors = np.array([[40, 30, 30], [90, 40, 25], [200, 60, 20], [255, 150, 40], [255, 230, 140], [255, 255, 240]], float)
    return np.stack([np.interp(heat, stops, colors[:, c]) for c in range(3)], -1)


def make_explosion(path):
    rng = np.random.default_rng(7)
    # turbulence that drifts over time (frames x height x width), computed at half size for speed
    turb = smooth_noise(rng, (FRAMES, H // 2, W // 2), [8, 24, 60])
    ys, xs = np.mgrid[0:H, 0:W].astype(float)

    # sky: dark at the top, glowing orange at the horizon, then dark ground
    sky_t = np.clip(ys / (H * 0.75), 0, 1)[..., None]
    sky = (1 - sky_t) * np.array([25, 20, 45]) + sky_t * np.array([140, 70, 50])
    sky[ys > H * 0.75] = [35, 25, 25]

    cx, ground = W / 2, H * 0.75
    sparks = rng.normal(0, 1, (60, 2)) * [180, 120] + [0, -160]  # speeds in pixels per second
    frames = []
    for f in range(FRAMES):
        t = f * FRAME_MS / 1000
        grow = 1 - np.exp(-t * 4)                      # fast at first, then slows down
        radius = 40 + 150 * grow
        cy = ground - radius * 0.55 - 40 * t           # the fireball rises a bit
        cool = np.exp(-t * 0.9)                        # gets darker and smokier over time
        tb = zoom(turb[f], 2, order=1)[:H, :W]

        d = np.hypot(xs - cx, (ys - cy) * 1.15) / radius
        edge = 1 - d + 0.35 * (tb - 0.5)               # ragged edge from the turbulence
        heat = np.clip(edge * 1.6, 0, 1) * (0.35 + 0.65 * cool) * (0.8 + 0.4 * tb)
        body = np.clip(edge * 4, 0, 1)                 # how much of the fireball covers this pixel

        img = sky * (1 + 0.8 * cool * np.exp(-d * 1.5))[..., None]   # fire lights up the sky

        # column of smoke going down to the ground, behind the fireball, with ragged edges
        width = radius * 0.3 * (1 + 0.6 * (tb - 0.5))
        stem = np.clip(1 - np.abs(xs - cx) / width, 0, 1) ** 0.5 * (ys > cy) * (ys < ground)
        stem = stem * np.clip(grow * 1.5, 0, 1) * 0.85
        img = img * (1 - stem[..., None]) + heat_to_color(0.15 + 0.3 * cool * tb) * stem[..., None]
        fire = heat_to_color(heat)
        img = img * (1 - body[..., None]) + fire * body[..., None]

        # shockwave ring
        ring_r = 300 * t + 30
        ring = np.exp(-((np.hypot(xs - cx, (ys - ground) * 2.5) - ring_r) / 6) ** 2) * np.exp(-t * 2.5)
        img += ring[..., None] * np.array([255, 220, 170]) * 0.8

        # sparks flying out and falling
        for vx, vy in sparks:
            sx, sy = cx + vx * t, ground - 30 + vy * t + 250 * t * t
            if 0 <= sx < W - 2 and 0 <= sy < H - 2 and t < 1.6:
                img[int(sy):int(sy) + 2, int(sx):int(sx) + 2] = [255, 200, 90]

        # flash at the start
        img += 255 * np.exp(-t * 14)
        frames.append(Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).convert("P", palette=Image.ADAPTIVE, colors=128))
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=FRAME_MS, loop=0, optimize=True)


def lowpass(x, amount):
    # simple smoothing filter: higher amount = darker sound
    out = np.empty_like(x)
    acc = 0.0
    a = 1 - amount
    for i, v in enumerate(x):
        acc = acc * amount + v * a
        out[i] = acc
    return out


def make_bang(path):
    rng = np.random.default_rng(3)
    n = int(RATE * 1.4)
    t = np.arange(n) / RATE
    noise = rng.normal(0, 1, n)
    crack = lowpass(noise, 0.6) * np.exp(-t / 0.01)                        # short snap at the start
    body = lowpass(lowpass(noise, 0.96), 0.96) * np.exp(-t / 0.2) * 12     # the main boom (muffled noise)
    freq = 40 + 90 * np.exp(-t / 0.08)                                     # thump sliding down in pitch
    thump = np.sin(2 * np.pi * np.cumsum(freq) / RATE) * np.exp(-t / 0.22)
    rumble = lowpass(noise, 0.985) * np.exp(-t / 0.6) * 6                  # low rumble that hangs around
    sound = 0.3 * crack + 0.8 * body + 1.2 * thump + 0.6 * rumble
    for delay, gain in [(0.047, 0.35), (0.11, 0.25), (0.19, 0.15)]:      # a bit of echo
        k = int(delay * RATE)
        sound[k:] += sound[:-k] * gain
    sound = np.tanh(sound / np.abs(sound).max() * 1.8)                     # squash it so it hits harder
    sound *= np.minimum(1, (n - np.arange(n)) / (0.05 * RATE))             # fade the very end
    sound = (sound / np.abs(sound).max() * 0.9 * 32767).astype(np.int16)
    with wave.open(path, "w") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(RATE)
        f.writeframes(sound.tobytes())


if __name__ == "__main__":
    gif = os.path.join(ROOT, "assets", "demo_explosion.gif")
    wav = os.path.join(ROOT, "sounds", "demo_bang.wav")
    make_explosion(gif)
    make_bang(wav)
    print("made", os.path.relpath(gif), "and", os.path.relpath(wav))

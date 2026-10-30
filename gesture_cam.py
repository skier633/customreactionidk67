# gesture_cam.py
# main file. run this: python gesture_cam.py
# press q to quit, d to show the debug view (hand points + what it thinks you're doing), c to clear the drawing,
# p to turn the air pen on/off

import json
import os
import time

import cv2
import numpy as np

import effects
import gestures
import pen
import preview as preview_ui
import settings
import sounds
from tracking import (make_hand_tracker, make_face_tracker, make_person_cutter,
                      find_hands, find_face, find_person, FACE_LEFT_EDGE, FACE_RIGHT_EDGE)

# ---------- settings ----------

CAMERA_NUMBER = 0        # try 1 if it opens the wrong camera
WIDTH = 1280
HEIGHT = 720
USE_VIRTUAL_CAM = True   # sends the video to OBS Virtual Camera so facetime/zoom can use it
USE_FACE = True          # needed for "on your face" effects. turn off if its laggy
SHOW_DEBUG = False

# what each gesture does (picture, gif, sound, distortion...) is in settings.json.
# change it with: python studio.py (or edit settings.json yourself). this picks up changes while its running

# where you hear the sounds. None = your normal speakers/headphones
SOUND_DEVICE = None

# sends your mic + the sounds into BlackHole, so the call hears both. in facetime/discord pick
# "BlackHole 2ch" as your mic (needs BlackHole, see the README). False = sounds only play for you
SEND_SOUNDS_TO_CALL = True
CALL_MIC = None  # which mic to send. None = your normal mic, or a name like "MacBook Air Microphone"

# studio.py writes this file when you click "try it" (or "clear drawing"), so you can see an effect
# without doing the gesture
TEST_FILE = "test_trigger.json"


def draw_debug(frame, hands):
    preview_ui.skeleton(frame, hands)


def load_settings():
    # loads settings.json plus everything it points to. runs again whenever it changes
    config = settings.load()
    gestures.load_classifier()
    if gestures.motion_recognizer is not None:
        gestures.motion_recognizer.early_names = {name for name, cfg in config["gestures"].items()
                                                if cfg.get("motion_early", False)}
    pictures = set()
    for cfg in config["gestures"].values():
        pictures.update(name for name in [cfg["picture"], cfg["gif"]] if name)
    effects.load_media(pictures)
    sounds.load_sounds({key: cfg["sound"] for key, cfg in config["gestures"].items() if cfg["sound"]},
                       SOUND_DEVICE)
    return config


def should_fire(cfg, name, now, last_fired, test=False):
    # whether a gesture that just went off should play its sound and effects
    if cfg is None:
        return False  # a trained gesture you havent given anything to do yet
    if test:
        return True   # "try it" in the studio always works, even for switched-off gestures
    if not cfg["enabled"]:
        return False  # switched off in the studio
    return now - last_fired.get(name, -99) >= cfg["cooldown"]


def read_test_file():
    # returns what studio.py asked for: {"gesture": name} or {"command": "clear_pen"}, or None
    if not os.path.exists(TEST_FILE):
        return None
    try:
        with open(TEST_FILE) as f:
            name = json.load(f)
    except (OSError, ValueError):
        name = None
    try:
        os.remove(TEST_FILE)
    except OSError:
        pass
    return name


def main(camera_number=CAMERA_NUMBER, studio_preview=False):
    cap = cv2.VideoCapture(camera_number)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    ok, frame = cap.read()
    if not ok:
        print("couldnt open the camera. on mac check System Settings > Privacy > Camera for Terminal/VS Code")
        cap.release()
        return
    h, w = frame.shape[:2]
    print("camera is", w, "x", h)

    hand_tracker = make_hand_tracker()
    face_tracker = make_face_tracker() if USE_FACE else None
    person_cutter = make_person_cutter()
    config = load_settings()
    if SEND_SOUNDS_TO_CALL:
        sounds.start_call_mix(CALL_MIC)
    stamp = settings.stamp()
    last_check = time.monotonic()
    if os.path.exists(TEST_FILE):
        os.remove(TEST_FILE)  # old leftover

    cam = None
    if USE_VIRTUAL_CAM:
        try:
            import pyvirtualcam
            cam = pyvirtualcam.Camera(width=w, height=h, fps=30)
            print("virtual camera is on:", cam.device)
        except Exception as e:
            print("couldnt start the virtual camera so its just the preview window for now")
            print("  reason:", e)

    stream = None
    if studio_preview:
        from preview_stream import PreviewServer
        stream = PreviewServer()
    show_debug = SHOW_DEBUG
    show_hud = True
    display_fps = 30.0
    last_fired = {}  # gesture name -> time, for the cooldowns
    start = time.monotonic()
    last = start
    last_ms = -1

    def fire(name, x, y, now, test=False):
        cfg = config["gestures"].get(name)
        print("gesture:", name, "(switched off)" if cfg is not None and not cfg["enabled"] and not test else "")
        if not should_fire(cfg, name, now, last_fired, test):
            return
        last_fired[name] = now
        sounds.play(name, cfg["volume"] / 100)
        effects.trigger(name, cfg, x, y, w, h)

    while True:
        ok, frame = cap.read()
        if not ok:
            print("camera stopped")
            break

        now = time.monotonic()
        dt = now - last
        display_fps += min(1.0, dt * 3) * (1 / max(dt, 1e-6) - display_fps)
        last = now

        # check once a second if studio.py changed the settings or retrained the gestures
        if now - last_check > 1:
            last_check = now
            if settings.stamp() != stamp:
                stamp = settings.stamp()
                print("settings changed, reloading")
                config = load_settings()
        # mediapipe wants a time in milliseconds that always goes up
        time_ms = int((now - start) * 1000)
        if time_ms <= last_ms:
            time_ms = last_ms + 1
        last_ms = time_ms

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hands = find_hands(hand_tracker, rgb, time_ms)
        face = None
        if face_tracker is not None and effects.needs_face():
            # only need the face while an effect is on your face. saves a lot of speed
            face = find_face(face_tracker, rgb, time_ms)

        for e in gestures.detect(hands, now):
            fire(e["name"], e["x"], e["y"], now)

        test = read_test_file()
        if test and test.get("gesture"):
            fire(test["gesture"], w / 2, h / 2, now, test=True)
        if test and test.get("command") == "clear_pen":
            pen.clear()

        pen.update(hands, now, w, h, config["pen"])

        # only cut you out from the background while something is going behind you (its kinda slow)
        person_mask = None
        if effects.needs_person():
            person_mask = find_person(person_cutter, rgb, time_ms)

        face_info = None
        if face is not None:
            face_info = (np.mean(face, axis=0), np.linalg.norm(face[FACE_LEFT_EDGE] - face[FACE_RIGHT_EDGE]))
        out = effects.update_and_draw(frame, now, dt, person_mask, face_info)

        # the preview window is mirrored so it feels like a mirror
        shown = out
        if show_debug:
            shown = out.copy()
            draw_debug(shown, hands)
        preview = cv2.flip(shown, 1)

        # the drawing and captions go on after flipping so theyre readable in both. in the preview the
        # ink follows your finger. in the call its flipped the same way (unless readable_for_call is off),
        # since otherwise the other person would see your writing backwards
        pen.draw(preview, now, config["pen"], mirror=True)
        pen.draw(out, now, config["pen"], mirror=config["pen"]["readable_for_call"])
        effects.draw_text(out)
        effects.draw_text(preview)

        # the virtual camera gets the normal picture. facetime mirrors it on your side by itself
        if cam is not None:
            cam.send(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))

        if stream is not None:
            studio_frame = preview.copy()
            # Small fingertip markers belong only to Studio, never to the call output.
            for hand in hands:
                for tip in (4, 8):
                    x, y = hand["pts"][tip]
                    cv2.circle(studio_frame, (int(w - 1 - x), int(y)), 3,
                               (180, 220, 170), 1, cv2.LINE_AA)
            stream.update(studio_frame)

        if show_hud:
            preview_ui.draw(preview, hands, display_fps, config["pen"]["enabled"],
                            cam is not None, gestures.status())

        if show_debug:
            for i, hand in enumerate(hands):
                cv2.putText(preview, gestures.debug_text(hand), (10, 130 + 30 * i),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            fps = 1 / max(dt, 1e-6)
            cv2.putText(preview, "fps: " + str(int(fps)), (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("gesture cam (q quit, d debug, c clear drawing, p pen on/off)", preview)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("d"):
            show_debug = not show_debug
        if key == ord("h"):
            show_hud = not show_hud
        if key == ord("c"):
            pen.clear()
        if key == ord("p"):
            # turn the air pen on/off and save it, so the studio shows the same thing
            config["pen"]["enabled"] = not config["pen"]["enabled"]
            settings.save(config)
            print("air pen", "on" if config["pen"]["enabled"] else "off")

    cap.release()
    if stream is not None:
        stream.close()
    hand_tracker.close()
    if face_tracker is not None:
        face_tracker.close()
    person_cutter.close()
    sounds.stop_call_mix()
    if cam is not None:
        cam.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=CAMERA_NUMBER)
    parser.add_argument("--studio-preview", action="store_true")
    args = parser.parse_args()
    main(args.camera, args.studio_preview)

# tracking.py
# wraps mediapipe so the rest of the code just gets simple numpy arrays
# the hand model finds 21 points on each hand, the face model finds 478 points on your face,
# and the person model cuts you out from the background (for the explosion)

import os
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

MODEL_DIR = "models"
# the gesture recognizer finds the same 21 hand points as mediapipe's hand tracker, and also
# recognizes a few common gestures (thumbs up, peace, open palm...) for free
HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task"
FACE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
PERSON_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite"

# hand point numbers (from the mediapipe docs)
# 0 = wrist
# thumb: 1 2 3 4 (4 is the tip)
# index: 5 6 7 8 (5 is the knuckle, 8 is the tip)
# middle: 9 10 11 12
# ring: 13 14 15 16
# pinky: 17 18 19 20
WRIST = 0
THUMB_TIP = 4
INDEX_KNUCKLE = 5
INDEX_TIP = 8
MIDDLE_KNUCKLE = 9
MIDDLE_TIP = 12

# face point numbers
FACE_LEFT_EDGE = 234
FACE_RIGHT_EDGE = 454


def download_model(url, filename):
    path = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(path):
        os.makedirs(MODEL_DIR, exist_ok=True)
        print("downloading " + filename + " (only happens the first time)...")
        urllib.request.urlretrieve(url, path)
        print("done")
    return path


def make_hand_tracker():
    path = download_model(HAND_MODEL_URL, "gesture_recognizer.task")
    options = vision.GestureRecognizerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=path),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.GestureRecognizer.create_from_options(options)


def make_face_tracker():
    path = download_model(FACE_MODEL_URL, "face_landmarker.task")
    options = vision.FaceLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=path),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
    )
    return vision.FaceLandmarker.create_from_options(options)


def make_person_cutter():
    # this one figures out which pixels are you and which are the background
    path = download_model(PERSON_MODEL_URL, "selfie_segmenter.tflite")
    options = vision.ImageSegmenterOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=path),
        running_mode=vision.RunningMode.VIDEO,
        output_confidence_masks=True,
    )
    return vision.ImageSegmenter.create_from_options(options)


def find_person(cutter, frame_rgb, time_ms):
    # returns an array the size of the frame. 1 = you, 0 = background, in between = edges
    h, w = frame_rgb.shape[:2]
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = cutter.segment_for_video(image, time_ms)
    # the last mask is the "person" one
    # (if the explosion shows up in FRONT of you instead of behind, change this to 1 - mask)
    mask = result.confidence_masks[-1].numpy_view().copy()
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h))
    return np.clip(mask, 0, 1)


def find_hands(tracker, frame_rgb, time_ms):
    # returns a list of hands. each hand is a dict:
    #   "label": "Left" or "Right"
    #   "pts":   21 x 2 array of pixel positions (x, y)
    #   "world": 21 x 3 array of positions in meters (good for checking finger shapes)
    #   "ready_made": (name, score) of mediapipe's own guess, like ("Thumb_Up", 0.93) or ("None", 0.8)
    h, w = frame_rgb.shape[:2]
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = tracker.recognize_for_video(image, time_ms)

    hands = []
    for i in range(len(result.hand_landmarks)):
        pts = np.array([[p.x * w, p.y * h] for p in result.hand_landmarks[i]])
        world = np.array([[p.x, p.y, p.z] for p in result.hand_world_landmarks[i]])
        label = result.handedness[i][0].category_name
        guess = result.gestures[i][0] if result.gestures and result.gestures[i] else None
        ready_made = (guess.category_name, guess.score) if guess else ("None", 0.0)
        hands.append({"label": label, "pts": pts, "world": world, "ready_made": ready_made})

    # if both hands get the same label (happens sometimes) rename one so they dont mix up
    if len(hands) == 2 and hands[0]["label"] == hands[1]["label"]:
        if hands[0]["pts"][WRIST][0] < hands[1]["pts"][WRIST][0]:
            hands[0]["label"], hands[1]["label"] = "Left", "Right"
        else:
            hands[0]["label"], hands[1]["label"] = "Right", "Left"
    return hands


def find_face(tracker, frame_rgb, time_ms):
    # returns a 478 x 2 array of pixel positions, or None if no face
    h, w = frame_rgb.shape[:2]
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = tracker.detect_for_video(image, time_ms)
    if len(result.face_landmarks) == 0:
        return None
    return np.array([[p.x * w, p.y * h] for p in result.face_landmarks[0]])


def palm_size(points):
    # distance from wrist to middle knuckle. used to make things work at any distance from the camera
    return np.linalg.norm(points[MIDDLE_KNUCKLE] - points[WRIST]) + 1e-6


def hand_features(hand):
    # turns one hand into 63 numbers for the trained classifier
    # moves the wrist to (0,0,0) and scales by palm size so it doesnt matter how close you are
    world = hand["world"].copy()
    world = world - world[WRIST]
    world = world / palm_size(world)
    # flip left hands so a left peace sign looks the same as a right one
    if hand["label"] == "Left":
        world[:, 0] = -world[:, 0]
    return world.flatten()

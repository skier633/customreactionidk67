"""Measure false triggers while talking normally, without playing effects or saving video.

Run from the project folder:
    python tools/evaluate_background.py --seconds 300
    python tools/evaluate_background.py --replay data/background/SESSION.npz
"""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

import gestures
from gesture_cam import should_fire
import motion
import settings
from tracking import make_hand_tracker, find_hands


def evaluate(frames, duration, config):
    """Replay every observed frame, including no-hand frames, through live detection."""
    if duration <= 0:
        raise ValueError("A positive observation duration is required")
    candidates = []
    triggered = []
    last_fired = {}
    total_frames = 0
    hand_frames = 0
    for timestamp, hands in frames:
        total_frames += 1
        hand_frames += bool(hands)
        for event in gestures.detect(hands, timestamp):
            entry = {"time": float(timestamp), "name": event["name"], "kind": event["type"]}
            candidates.append(entry)
            name = event["name"]
            if should_fire(config["gestures"].get(name), name, timestamp, last_fired):
                triggered.append(entry)
                last_fired[name] = timestamp
    return {
        "duration_seconds": duration,
        "completed_five_minutes": duration >= 300,
        "frames": total_frames,
        "frames_with_hands": hand_frames,
        "hand_visibility_fraction": hand_frames / max(1, total_frames),
        "candidate_count": len(candidates),
        "configured_trigger_count": len(triggered),
        "false_triggers_per_minute": len(triggered) * 60 / duration,
        "candidates_per_minute": len(candidates) * 60 / duration,
        "configured_triggers_by_name": dict(Counter(event["name"] for event in triggered)),
        "events": candidates,
        "configured_events": triggered,
        "protocol": "Ordinary conversation with no intentional commands; all triggers count as false.",
        "limitation": "One session only. Zero triggers does not establish a zero long-term false-alarm rate.",
    }


def save_frames(path, frames, duration):
    rows = [(index, hand) for index, (_, hands) in enumerate(frames) for hand in hands]
    np.savez_compressed(path, duration=duration, times=[time for time, _ in frames],
        frame=[index for index, _ in rows],
        label=[hand["label"] for _, hand in rows],
        track_id=[hand["track_id"] for _, hand in rows],
        pts=np.array([hand["pts"] for _, hand in rows]).reshape(-1, 21, 2),
        world=np.array([hand["world"] for _, hand in rows]).reshape(-1, 21, 3),
        ready_name=[hand["ready_made"][0] for _, hand in rows],
        ready_score=[hand["ready_made"][1] for _, hand in rows])


def load_frames(path):
    with np.load(path, allow_pickle=False) as data:
        frames = [(float(timestamp), []) for timestamp in data["times"]]
        for row, frame in enumerate(data["frame"]):
            frames[int(frame)][1].append({
                "label": str(data["label"][row]), "track_id": int(data["track_id"][row]),
                "pts": data["pts"][row].copy(), "world": data["world"][row].copy(),
                "ready_made": (str(data["ready_name"][row]), float(data["ready_score"][row])),
            })
        return frames, float(data["duration"])


def capture(seconds, camera_number):
    camera = cv2.VideoCapture(camera_number)
    tracker = None
    frames = []
    try:
        tracker = make_hand_tracker()
        start = time.monotonic()
        last_ms = -1
        print("Talk normally. Do not intentionally perform commands. Q ends early.", flush=True)
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Camera stopped; no complete evaluation was produced")
            elapsed = time.monotonic() - start
            timestamp_ms = max(last_ms + 1, int(elapsed * 1000))
            last_ms = timestamp_ms
            hands = find_hands(tracker, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), timestamp_ms)
            frames.append((elapsed, hands))
            shown = cv2.flip(frame, 1)
            text = f'Talk normally | {max(0, seconds-elapsed):.0f}s left | Q ends early'
            cv2.putText(shown, text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 255, 200), 2)
            cv2.imshow("Background evaluation (no video saved)", shown)
            if cv2.waitKey(1) & 255 == ord("q") or elapsed >= seconds:
                return frames, elapsed
    finally:
        camera.release()
        if tracker is not None:
            tracker.close()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=300)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--replay", type=Path)
    args = parser.parse_args()
    if not np.isfinite(args.seconds) or not 1 <= args.seconds <= 3600:
        parser.error("seconds must be between 1 and 3600")
    if not Path(motion.MODEL_PATH).exists():
        parser.error("Train a motion model first")
    config = settings.load()
    gestures.load_classifier()
    if gestures.motion_recognizer is None:
        parser.error("The motion model could not be loaded")
    gestures.motion_recognizer.early_names = {
        name for name, effect in config["gestures"].items() if effect.get("motion_early", False)}
    model_hash = hashlib.sha256(Path(motion.MODEL_PATH).read_bytes()).hexdigest()
    if args.replay:
        frames, duration = load_frames(args.replay)
        clip_path = args.replay
    else:
        frames, duration = capture(args.seconds, args.camera)
        folder = Path("data/background")
        folder.mkdir(parents=True, exist_ok=True)
        clip_path = folder / (datetime.now().strftime("%Y%m%d-%H%M%S-%f") + ".npz")
        save_frames(clip_path, frames, duration)
    report = evaluate(frames, duration, config)
    report.update(model_sha256=model_hash, settings=config, clip=str(clip_path))
    # Keep reports for repeated evaluations instead of overwriting earlier results.
    report_path = clip_path.with_name(clip_path.stem + "-" + datetime.now().strftime("%H%M%S-%f") + ".json")
    report_path.write_text(json.dumps(report, indent=2))
    print(f'{report["configured_trigger_count"]} configured triggers in {duration/60:.2f} minutes')
    print(f'{report["false_triggers_per_minute"]:.3f} false triggers/minute; '
          f'{report["hand_visibility_fraction"]:.0%} of frames have visible hands')
    print("Saved landmarks:", clip_path)
    print("Saved report:", report_path)
    if duration < 300:
        print("Short session: this is not a complete five-minute evaluation.")


if __name__ == "__main__":
    main()

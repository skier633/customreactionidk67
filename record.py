"""Record complete motion repetitions using the same segmenter as live recognition."""
import os
import sys
import time
import uuid
import cv2
import numpy as np
from tracking import make_hand_tracker, find_hands
from motion import Segmenter


def main(name, camera_number=0):
    os.makedirs('data', exist_ok=True)
    path = 'data/' + name + '.motion.npz'
    clips, sessions = [], []
    if os.path.exists(path):
        with np.load(path) as data:
            clips, sessions = list(data['clips']), list(data['sessions'])
    initial = len(clips)
    session = uuid.uuid4().hex
    segmenter = Segmenter()
    cap, tracker = cv2.VideoCapture(camera_number), make_hand_tracker()
    start, last_ms = time.monotonic(), -1
    print('One hand only. Rest, perform the full motion, then pause. Repeat 8–12 times. Q saves.')
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            now = time.monotonic() - start
            ms = max(int(now * 1000), last_ms + 1)
            last_ms = ms
            hands = find_hands(tracker, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), ms)
            clip = segmenter.update(hands, now)
            if clip is not None:
                clips.append(clip)
                sessions.append(session)
                print('Saved repetition', len(clips) - initial, flush=True)
            text = ('MOVE, then pause' if segmenter.active else 'REST, then perform motion')
            if len(hands) != 1:
                text = 'Show exactly ONE hand'
            preview = cv2.flip(frame, 1)
            cv2.putText(preview, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, .7, (100, 255, 180), 2)
            cv2.putText(preview, f'{name}: {len(clips)-initial} new examples | Q saves and quits',
                        (20, 75), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 2)
            cv2.imshow('Record a motion', preview)
            if cv2.waitKey(1) & 255 == ord('q'):
                break
    finally:
        cap.release()
        tracker.close()
        cv2.destroyAllWindows()
    if len(clips) > initial:
        with open(path + '.tmp', 'wb') as f:
            np.savez_compressed(f, clips=np.array(clips), sessions=np.array(sessions))
        os.replace(path + '.tmp', path)
    print(f'{len(clips)} complete repetitions. At least 6 per motion and none are required.')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()
    main(args.name, args.camera)

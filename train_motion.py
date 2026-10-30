"""Train templates and evaluate on entire held-out repetitions or recording sessions."""
import json
import os
import joblib
import numpy as np
from motion import MotionModel, recordings, MODEL_PATH, REPORT_PATH


def main():
    groups, train, test, splits = {}, {}, {}, {}
    for name, path in recordings().items():
        with np.load(path) as data:
            clips, sessions = data['clips'], data['sessions']
        if clips.ndim != 3 or clips.shape[1:] != (32, 65) or not np.isfinite(clips).all() or len(clips) < 6:
            raise ValueError(f'{name}: record at least six valid complete repetitions')
        distinct = list(dict.fromkeys(sessions))
        mask = sessions == distinct[-1] if len(distinct) > 1 else np.arange(len(clips)) >= int(len(clips)*.75)
        if (~mask).sum() < 3:
            raise ValueError(f'{name}: need at least three repetitions in earlier sessions')
        groups[name], train[name], test[name] = clips, clips[~mask], clips[mask]
        splits[name] = 'held-out session' if len(distinct) > 1 else 'held-out repetitions (same session)'
    if 'none' not in groups or len(groups) < 2:
        raise ValueError('Record a motion and none (ordinary non-command movements), six repetitions each')
    model = MotionModel(train)
    report = {'kind': 'motion', 'gestures': {}, 'split': splits}
    correct, total = 0, 0
    for name, clips in test.items():
        guesses = [model.predict_one(c) or 'none' for c in clips]
        count = guesses.count(name)
        correct += count
        total += len(clips)
        report['gestures'][name] = {'accuracy': count/len(clips), 'samples': len(groups[name]),
            'mixed_up_with': {n: guesses.count(n)/len(clips) for n in set(guesses) if n != name}}
    report['accuracy'] = correct / total
    report['validation_examples'] = total
    report['background_false_accept_rate'] = 1 - report['gestures']['none']['accuracy']
    os.makedirs('models', exist_ok=True)
    joblib.dump(MotionModel(groups), MODEL_PATH + '.tmp')
    os.replace(MODEL_PATH + '.tmp', MODEL_PATH)
    with open(REPORT_PATH + '.tmp', 'w') as f:
        json.dump(report, f, indent=2)
    os.replace(REPORT_PATH + '.tmp', REPORT_PATH)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

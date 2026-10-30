# Gesture Cam

**Teach your camera a movement.** Live reactions, glowing air drawings, and custom effects for your webcam and video calls.

Gesture Cam runs on your computer, with a browser-based Studio for mixing effects. Raise a thumbs-up for confetti, draw a quick explanation in the air, or build a signature reaction of your own.

![Gesture Studio](docs/studio.png)

## What you can do

- **React without reaching for a button.** Seven built-in gestures, plus separate two-hand combinations.
- **Draw in the air.** Pinch with your other fingers curled to write. Release to lift the pen. Ink can glow, fade, and stay readable for people on the call.
- **Make each reaction yours.** Combine particles, captions, images, animated overlays, sounds, distortions, and color filters.
- **Start with a scene.** Studio's Celebrate, Present, and Play presets fill unused gesture slots without replacing your edits.
- **Teach a custom gesture.** Record complete movements in Studio, train, and inspect the held-out accuracy report.
- **See what the tracker is doing.** The local preview shows hand count, frame rate, and gesture hold progress. Toggle the landmark skeleton to inspect tracking. These guides stay out of the virtual camera feed.

## Quick start

Use **Python 3.12**. The virtual-camera and call-audio setup below is for macOS; a webcam preview works without OBS or BlackHole.

```sh
git clone https://github.com/skier633/customreactionidk67.git
cd customreactionidk67
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python gesture_cam.py
```

Allow camera access when prompted. Tracking models download on the first run. In a second terminal, from the same folder:

```sh
source venv/bin/activate
python studio.py
```

Studio opens at **http://localhost:5050**. Pick **Celebrate**, then hold a thumbs-up for about half a second. You can also use **Try it** on a configured gesture to preview its effect while Gesture Cam is running. Configuration changes load automatically.

### A 30-second demo

1. Apply **Celebrate** in Studio and hold a thumbs-up for confetti.
2. Release, then make a peace sign for fireworks.
3. Pinch your thumb and index finger with your other fingers curled; draw a shape, then release.
4. Press **C** to clear. Press **H** to hide the preview controls for a clean presentation.

Presets only fill untouched slots. To give an existing gesture a new preset, reset that slot in its editor first, then apply the preset. Trained gestures and pen settings are preserved.

## Smoother tracking, deliberate reactions

The tracker matches hands between frames using palm position and motion, so a changing detection order or a brief handedness-label flicker does not swap your controls. Adaptive landmark smoothing steadies small movements while responding more quickly to larger motions. World-space finger geometry remains unfiltered for pinch checks and custom gesture features.

Gesture holds tolerate brief recognition dropouts, but missing observations do not count toward the hold. A sustained release is required before repeating a reaction. This prevents a momentary tracking hiccup from replaying a held gesture. Lost hands are never drawn as predicted detections.

The air pen adds its own pinch hysteresis and hidden-fingertip handling. Tracking still depends on lighting, framing, and visibility: keep the whole hand in view and avoid covering one hand with the other. Extremely fast motion or complete occlusion can still break identity.

## Preview controls

Click the camera preview so it receives keyboard input.

| Key | Action |
| --- | --- |
| **Q** | Quit |
| **H** | Show or hide the preview status display |
| **D** | Show or hide hand landmarks and debug details |
| **C** | Clear the drawing |
| **P** | Toggle the air pen |

## Use it in FaceTime, Zoom, or another call

The call needs to offer a camera selector. On macOS, Gesture Cam sends its video through the OBS virtual camera:

1. Install [OBS Studio](https://obsproject.com/).
2. Open OBS, click **Start Virtual Camera**, then stop it and quit OBS. This registers the virtual camera.
3. Run `python gesture_cam.py`.
4. Select **OBS Virtual Camera** in your call. In FaceTime, use the **Video** menu.

If the virtual camera cannot start, Gesture Cam keeps running as a local preview and prints the reason in the terminal.

### Send voice and reaction sounds to the call

A virtual camera carries video only. Gesture Cam can mix your microphone and reaction sounds into BlackHole:

1. Install BlackHole 2ch: `brew install --cask blackhole-2ch`, then restart your Mac.
2. Run Gesture Cam and allow microphone access. Check the terminal for the message that your mic and sounds are being sent to BlackHole.
3. Select **BlackHole 2ch** as the call's microphone.
4. If effects are cut off, reduce the call's noise filtering. In FaceTime, use **Control Center → Mic Mode → Standard**.

## Teach your own motions

Custom recording now learns **complete movements**, not independent hand poses. Try a wave, directional swipe, circle, or changing finger configuration. Use exactly one visible hand. Pause for at least half a second before and after each repetition; a pause ends the motion. Movements should last roughly 0.3–4 seconds. Continuous motion with internal pauses over 0.3 seconds is treated as separate gestures. Two-hand choreography is not supported yet.

1. Open Studio and record `none`: ordinary movements you want ignored. Collect 8–12 varied repetitions, then press **Q** to save.
2. Name your motion (for example `circle`) and record 8–12 complete repetitions. Vary speed and position. Press **Q** to save; Studio trains automatically when both classes exist.
3. Assign an effect to the motion. Restart the camera if you stopped it for recording; otherwise the model reloads automatically.
4. Record another session under different lighting or at another distance, then inspect the validation results. Each class needs at least six repetitions, with at least three in earlier sessions for session-based validation.

Once a motion model is loaded, automatic pose triggers are disabled so intermediate poses cannot fire unrelated reactions. The air pen and manual effect previews still work. Old pose recordings and models are preserved; they cannot be converted into motions because they contain no trajectory or repetition boundaries. The old CLI recorder remains available as `record_pose.py`.

### Recognition design and honest accuracy

The feature sequence combines wrist travel in palm-size units with all 21 world-space hand landmarks relative to the wrist. It retains direction, movement order, and finger-shape changes while removing screen position and much of camera-distance variation. Timestamp resampling produces 32 observations per repetition; banded dynamic time warping aligns local speed differences. This is a small-data template approach informed by [DTW gesture recognition research](https://www.eecs.ucf.edu/isuelab/research/jackknife/), not an implementation of the complete Jackknife algorithm.

Each class learns an acceptance radius from distances between its examples. Unfamiliar sequences, close competing matches, and background movements are rejected. Hand loss, a second visible hand, identity changes, or long gaps abort the current sequence. Completion triggers once after the ending pause, with a one-second cooldown.

Training validates on the latest **entire session** when multiple sessions exist, or on held-out **whole repetitions** otherwise. The report labels the split and records background false accepts. Same-session validation is weaker evidence of real-world performance. No webcam accuracy percentage is claimed without your recordings. Improve low scores with varied positives and difficult background negatives; inspect both missed motions and false accepts.

Motion clips live in `data/*.motion.npz`; the model and report are `models/motions.joblib` and `models/motions_report.json`. These files remain local and ignored by Git. To train from the terminal, use `python train_motion.py`.

## Local data

Camera frames are processed locally; the app does not upload them. The first launch downloads the tracking models. Studio listens only on `127.0.0.1`. Your recordings, trained model, settings, and uploaded assets are excluded from Git. The repository includes only the demo assets.

## Development and tests

```sh
pip install -r requirements-dev.txt
python -m pytest -q
```

The suite covers recorded pen clips, simulated tracking failures, identity changes, jitter filtering, gesture timing, effects, training, and Studio APIs. Most tests run without a camera. To capture another landmark-only test clip, see `tools/record_clip.py`.

| File | Purpose |
| --- | --- |
| `tracking.py` | Landmark detection, hand association, adaptive smoothing |
| `gestures.py` | Gesture recognition and temporal trigger handling |
| `pen.py` | Pinch drawing and stroke rendering |
| `preview.py` | Local status display and landmark skeleton |
| `effects.py` / `sounds.py` | Video effects and call audio |
| `studio.py` / `studio.html` | Local configuration and recording interface |

## License

[MIT](LICENSE).

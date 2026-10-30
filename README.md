# Gesture Cam

**Turn a hand gesture into a moment.** Live reactions, glowing air drawings, and custom effects for your webcam and video calls.

Gesture Cam runs on your computer, with a browser-based Studio for mixing effects. Raise a thumbs-up for confetti, draw a quick explanation in the air, or build a signature reaction of your own.

![Gesture Studio](docs/studio.png)

## What you can do

- **React without reaching for a button.** Seven built-in gestures, plus separate two-hand combinations.
- **Draw in the air.** Pinch with your other fingers curled to write. Release to lift the pen. Ink can glow, fade, and stay readable for people on the call.
- **Make each reaction yours.** Combine particles, captions, images, animated overlays, sounds, distortions, and color filters.
- **Start with a scene.** Studio's Celebrate, Present, and Play presets fill unused gesture slots without replacing your edits.
- **Teach a custom gesture.** Record examples in Studio, train, and inspect the held-out accuracy report.
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

## Create your own gestures

In Studio, record **none** first: relaxed hands and ordinary movements that should not trigger reactions. Then add a named gesture and record it with different positions, angles, and distances. Stop Gesture Cam while recording if your camera cannot be shared by two processes.

Studio trains after recordings when enough classes are available. Its report evaluates the final 20% of each recording separately from the training examples. This is a useful check, not a guarantee under new lighting or for another person. Add more varied recordings when two poses get confused.

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

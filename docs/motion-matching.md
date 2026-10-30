# Following a motion through the code

Start with `motion.py`. It uses NumPy and a small dynamic-programming loop; there is no hidden sequence library.

1. **`sample`** collects the wrist’s screen position, palm size, and the 21 hand landmarks relative to the wrist.
2. **`Segmenter.update`** waits for a resting hand, collects its movement, and finishes after a 0.3-second pause. It throws away a sequence if tracking disappears or changes identity. `MotionRecognizer` gives each visible hand its own segmenter.
3. **`encode`** subtracts the starting wrist position and divides travel by palm size. This makes moving elsewhere on screen or changing camera distance less disruptive. It samples 32 evenly spaced times. Direction and finger changes remain in the sequence.
4. **`distance`** compares each observation with nearby observations in the other sequence. At each position it keeps the cheapest path from the previous row, column, or diagonal. This allows different speeds without changing the order of the movement. `WARP_WINDOW = 8` limits how far those timing matches can drift. Wider windows permit more variation, but also more accidental matches. The wrist gets extra weight so its two coordinates are not overwhelmed by 63 finger coordinates.
5. **`MotionModel.predict_one`** finds the closest example. The acceptance radius comes from how different the recorded examples of that motion are. A candidate outside that radius, too close to another class, or closest to `none` is ignored. Distances are not probabilities.
6. **`MotionRecognizer.update_all`** turns accepted matches into events. It remembers which hand has already fired and applies a one-second cooldown per motion. Pose detection can still use stationary hands.

## Early firing

Early firing is an opt-in hypothesis: “what if the hand stopped here?” The candidate is padded with a stationary ending for comparison with the recorded examples. No padded landmarks are saved as real observations. The normal segmenter still waits for the actual pause.

Early matching requires a smaller distance and wider separation from competitors. It also checks several prefixes of other motions. Once it fires, the ending pause cannot fire the same movement again. These checks reduce risk; they cannot prove an unknown movement will not share a familiar beginning. Normal-pause matching remains the default.

## What the tests establish

Synthetic sequences check direction, changing finger order, different speeds, two visible hands, interruptions, coexistence with pose reactions, and duplicate prevention. They do not establish real webcam accuracy. Training reports evaluate held-out examples; `tools/evaluate_background.py` measures false triggers during ordinary conversation. Keep those results separate.

This template approach is informed by [the Jackknife gesture-recognition research](https://www.eecs.ucf.edu/isuelab/research/jackknife/). It does not reproduce the full Jackknife algorithm.

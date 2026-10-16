# replays every real recording in tests/clips through the gesture code and checks that what went
# off matches what the clip says should happen (record new ones with tools/record_clip.py)
import glob
import os

import pytest

from replay import matches, replay

CLIPS = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "clips", "*.npz")))

# clips that show a problem nobody has fixed yet. they still run, and if one starts passing,
# pytest says so (XPASS) so you know you can take it off this list
KNOWN_ISSUES = {}


def name_of(path):
    return os.path.basename(path)[:-4]


@pytest.mark.parametrize("path", [
    pytest.param(p, id=name_of(p), marks=pytest.mark.xfail(reason=KNOWN_ISSUES[name_of(p)], strict=False))
    if name_of(p) in KNOWN_ISSUES else pytest.param(p, id=name_of(p))
    for p in CLIPS
])
def test_clip(path):
    expect, counts = replay(path)
    assert expect, "this clip doesnt say what should happen (record it with --expect)"
    wrong = {k: (v, counts.get(k, 0)) for k, v in expect.items() if not matches(v, counts.get(k, 0))}
    assert not wrong, "expected vs what happened: " + ", ".join(f"{k}: wanted {w}, got {g}" for k, (w, g) in wrong.items())

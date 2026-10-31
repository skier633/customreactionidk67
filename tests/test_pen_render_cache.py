import importlib
import cv2
import numpy as np
import pen
import settings


def stroke(points):
    return {'points':[np.array(p) for p in points], 'end':None, 'color':'#ff3b6b','thickness':8}


def test_only_new_segments_are_rasterized_and_two_outputs_share_cache(monkeypatch):
    drawing = importlib.reload(pen)
    cfg = dict(settings.PEN_DEFAULTS, enabled=True)
    drawing.strokes.append(stroke([(0.2,0.2),(0.3,0.3)]))
    original = cv2.polylines
    lengths = []
    def capture(image, points, *args, **kwargs):
        lengths.append(len(points[0]))
        return original(image, points, *args, **kwargs)
    monkeypatch.setattr(cv2, 'polylines', capture)
    frame = np.zeros((360,640,3),np.uint8)
    drawing.draw(frame,0,cfg,True)
    initial = len(lengths)
    drawing.draw(frame.copy(),0,cfg,True)
    assert len(lengths) == initial
    drawing.strokes[0]['points'].append(np.array([0.4,0.4]))
    drawing.draw(frame.copy(),1,cfg,True)
    assert lengths[initial:] == [2,2,2]  # ink, coverage, low-resolution glow


def test_cache_clear_resize_and_mirror():
    drawing = importlib.reload(pen)
    cfg = dict(settings.PEN_DEFAULTS, enabled=True, glow=False)
    drawing.strokes.append(stroke([(0.2,0.2),(0.3,0.4)]))
    a = np.full((360,640,3),50,np.uint8)
    b = a.copy()
    drawing.draw(a,0,cfg,False)
    drawing.draw(b,0,cfg,True)
    np.testing.assert_array_equal(a[:,::-1],b)
    drawing.clear()
    # An active pen may have no stroke immediately after clearing.
    drawing.pens['Left'] = dict(down=True,point=np.array([0.7,0.7]))
    c = np.full((720,1280,3),50,np.uint8)
    drawing.draw(c,1,cfg,True)
    assert not (c[200:300,200:300] != 50).any()


def test_fade_mode_still_fades():
    drawing = importlib.reload(pen)
    cfg = dict(settings.PEN_DEFAULTS, enabled=True, glow=False, fade_after=1)
    item = stroke([(0.2,0.2),(0.3,0.4)])
    item['end'] = 0
    drawing.strokes.append(item)
    a = np.zeros((360,640,3),np.uint8)
    drawing.draw(a,0,cfg)
    b = np.zeros_like(a)
    drawing.draw(b,2,cfg)
    assert a.any() and not b.any()

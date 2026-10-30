import numpy as np
import pytest
from motion import encode, distance, MotionModel, Segmenter, MotionRecognizer


def clip(kind='right', count=50, duration=1, origin=0, scale=100):
    t = np.linspace(0, duration, count)
    u = t / duration
    xy = {'right': np.c_[u*2, u*0], 'left': np.c_[-u*2, u*0],
          'up': np.c_[u*0, -u*2], 'circle': np.c_[np.sin(u*2*np.pi), np.cos(u*2*np.pi)-1]}[kind]
    values = np.c_[xy*scale+origin, np.full(count,scale), np.zeros((count,63))]
    return encode(values, t)


def test_sequence_order_and_speed():
    assert distance(clip(), clip(count=95,duration=2,origin=700,scale=160)) < 1e-8
    assert distance(clip(), clip('left')) > 1
    assert distance(clip('circle'), clip('up')) > .4


def test_unknown_and_background_are_rejected():
    model = MotionModel({'right':[clip()]*4, 'left':[clip('left')]*4, 'none':[clip('up')]*4})
    assert model.predict_one(clip(duration=2)) == 'right'
    assert model.predict_one(clip('left')) == 'left'
    assert model.predict_one(clip('up')) is None
    assert model.predict_one(clip('circle')) is None
    assert model.predict_one(np.zeros((32,65))) is None


def test_ambiguous_classes_rejected():
    model = MotionModel({'a':[clip()]*4,'b':[clip()]*4})
    assert model.predict_one(clip()) is None


def hand(x, key=1):
    world = np.zeros((21,3)); world[9,1] = 1
    pts = np.tile([x,300.],(21,1)); pts[9,1] += 100
    return {'pts':pts,'world':world,'track_id':key,'label':'Left'}


def sequence():
    return [(i/30, [hand(100 if i<15 else 100+(i-15)*10 if i<35 else 290)]) for i in range(60)]


def test_segmenter_requires_motion_and_rest_and_emits_once():
    seg = Segmenter()
    clips = [c for t,h in sequence() if (c:=seg.update(h,t)) is not None]
    assert len(clips) == 1 and clips[0].shape == (32,65)
    seg = Segmenter()
    assert all(seg.update([hand(100)],i/30) is None for i in range(120))


@pytest.mark.parametrize('broken', [[], [hand(300,99)], [hand(300),hand(500,2)]])
def test_loss_or_identity_change_aborts_sequence(broken):
    seg = Segmenter()
    result = []
    for i,(t,h) in enumerate(sequence()):
        c = seg.update(broken if i==30 else h,t)
        if c is not None: result.append(c)
    assert not result


def test_training_splits_sessions_and_persists(tmp_path,monkeypatch):
    import train_motion, joblib, json
    monkeypatch.chdir(tmp_path)
    (tmp_path/'data').mkdir()
    for name,kind in [('swipe','right'),('none','up')]:
        np.savez(tmp_path/'data'/f'{name}.motion.npz', clips=[clip(kind)]*8,
                 sessions=['first']*6+['second']*2)
    train_motion.main()
    report = json.loads((tmp_path/'models/motions_report.json').read_text())
    assert report['accuracy'] == 1
    assert report['background_false_accept_rate'] == 0
    assert report['split']['swipe'] == 'held-out session'
    assert joblib.load(tmp_path/'models/motions.joblib').predict_one(clip()) == 'swipe'


def test_live_recognizer_emits_motion_event():
    seg=Segmenter()
    example=next(c for t,h in sequence() if (c:=seg.update(h,t)) is not None)
    model=MotionModel({'swipe':[example]*4,'none':[clip('up')]*4})
    live=MotionRecognizer(model)
    events=[e for t,h in sequence() if (e:=live.update(h,t)) is not None]
    assert len(events)==1 and events[0]['type']=='motion' and events[0]['name']=='swipe'


def test_local_speed_variation_preserves_path_but_not_reverse():
    t=np.linspace(0,1,70)
    def encoded(u):
        values=np.c_[np.sin(u*2*np.pi)*100, (np.cos(u*2*np.pi)-1)*100,
                     np.full(len(u),100), np.zeros((len(u),63))]
        return encode(values,t)
    original=encoded(t)
    assert distance(original,encoded(t**1.3)) < .15
    assert distance(original,encoded(-t)) > .4


def test_finger_sequence_order_matters_even_with_same_wrist_and_endpoints():
    t=np.linspace(0,1,50)
    values=np.zeros((50,66)); values[:,2]=100
    values[8:20,10:20]=2
    values[25:38,30:40]=2
    forward=encode(values,t)
    reverse=encode(values[::-1],t)
    assert distance(forward,reverse) > .05


def test_motion_studio_lists_examples_and_deletes_correct_file(project):
    import studio, importlib
    studio=importlib.reload(studio)
    np.savez('data/swipe.motion.npz', clips=[clip()]*8, sessions=['a']*8)
    client=studio.app.test_client()
    data=client.get('/api/state').get_json()
    assert data['motion_names']==['swipe']
    assert data['gestures']==[{'name':'swipe','samples':8}]
    assert client.post('/api/set',json={'key':'swipe','field':'text','value':'Hello'}).status_code==200
    assert client.post('/api/delete',json={'name':'swipe'}).status_code==200
    assert (project/'data/deleted/swipe.motion.npz').exists()

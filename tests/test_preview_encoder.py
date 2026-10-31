import threading
import time
from types import SimpleNamespace
import numpy as np
import preview_stream


def test_encoder_keeps_latest_frame_without_blocking_camera(monkeypatch):
    monkeypatch.setattr(preview_stream, 'ThreadingHTTPServer', lambda *args:
        SimpleNamespace(serve_forever=lambda:None,shutdown=lambda:None,server_close=lambda:None))
    started, release = threading.Event(), threading.Event()
    encoded = []
    original = preview_stream.cv2.imencode
    def slow_encode(ext, frame, options):
        encoded.append(int(frame[0,0,0]))
        if len(encoded)==1:
            started.set()
            assert release.wait(3)
        return original(ext,frame,options)
    monkeypatch.setattr(preview_stream.cv2,'imencode',slow_encode)
    server=preview_stream.PreviewServer()
    try:
        server.update(np.full((32,32,3),10,np.uint8))
        assert started.wait(2)
        for color in (20,30):
            server.offered=0
            server.update(np.full((32,32,3),color,np.uint8))
        assert int(server.pending[0,0,0])==30
        release.set()
        deadline=time.monotonic()+3
        while server.sequence<2 and time.monotonic()<deadline:
            time.sleep(.01)
        assert server.sequence==2
        assert encoded==[10,30]
    finally:
        release.set()
        server.close()
    assert not server.worker.is_alive()

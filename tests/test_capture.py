import threading
from capture import LatestCapture


def test_latest_capture_discards_backlog_and_releases_device():
    class Camera:
        def __init__(self):
            self.count = 0
            self.released = False
            self.produced = threading.Event()
            self.finish = threading.Event()
        def read(self):
            self.count += 1
            if self.count <= 20:
                if self.count == 20:
                    self.produced.set()
                return True, self.count
            self.finish.wait(2)
            return False, None
        def release(self):
            self.released = True
    camera = Camera()
    reader = LatestCapture(camera)
    try:
        assert camera.produced.wait(2)
        with reader.condition:
            assert reader.condition.wait_for(lambda: reader.sequence == 20, timeout=2)
        ok, frame = reader.read()
        assert ok and frame == 20
    finally:
        camera.finish.set()
        reader.release()
    assert camera.released and not reader.worker.is_alive()

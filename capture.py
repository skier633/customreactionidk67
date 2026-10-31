"""Continuously drain the camera; processing consumes the newest complete frame."""
import threading


class LatestCapture:
    def __init__(self, camera):
        self.camera = camera
        self.condition = threading.Condition()
        self.stopped = False
        self.failed = False
        self.sequence = 0
        self.consumed = 0
        self.frame = None
        self.worker = threading.Thread(target=self._read, daemon=True)
        self.worker.start()

    def _read(self):
        try:
            while not self.stopped:
                ok, frame = self.camera.read()
                with self.condition:
                    if not ok:
                        self.failed = True
                        self.condition.notify_all()
                        return
                    self.frame = frame
                    self.sequence += 1
                    self.condition.notify_all()
        finally:
            self.camera.release()

    def read(self):
        with self.condition:
            ready = self.condition.wait_for(lambda: self.failed or self.stopped or
                                           self.sequence > self.consumed, timeout=2)
            if not ready or self.failed or self.stopped:
                return False, None
            self.consumed = self.sequence
            return True, self.frame

    def release(self):
        with self.condition:
            self.stopped = True
            self.condition.notify_all()
        self.worker.join(timeout=2)

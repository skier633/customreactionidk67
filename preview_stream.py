"""A latest-frame-only preview: encoding and slow clients never block tracking."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import time
import cv2


class PreviewServer:
    def __init__(self, port=5051):
        self.jpeg = None
        self.updated = 0
        self.sequence = 0
        self.pending = None
        self.closed = False
        self.condition = threading.Condition()
        self.offered = 0
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == '/stream':
                    self.send_response(200)
                    self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
                    self.send_header('Cache-Control', 'no-store')
                    self.end_headers()
                    previous = 0
                    try:
                        while not owner.closed:
                            with owner.condition:
                                owner.condition.wait_for(lambda: owner.closed or owner.sequence != previous, timeout=3)
                                if owner.closed or time.monotonic() - owner.updated > 3:
                                    break
                                data, previous = owner.jpeg, owner.sequence
                            if data is None:
                                continue
                            self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: ' +
                                             str(len(data)).encode() + b'\r\n\r\n' + data + b'\r\n')
                            self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError, TimeoutError):
                        pass
                    return
                if self.path != '/frame' or owner.jpeg is None or time.monotonic() - owner.updated > 2:
                    self.send_error(503)
                    return
                data = owner.jpeg
                self.send_response(200)
                self.send_header('Content-Type', 'image/jpeg')
                self.send_header('Cache-Control', 'no-store')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                try:
                    self.wfile.write(data)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
        self.worker = threading.Thread(target=self.encode, daemon=True)
        self.worker.start()
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def update(self, frame):
        now = time.monotonic()
        if now - self.offered < 1 / 30:
            return
        self.offered = now
        with self.condition:
            self.pending = frame.copy()  # Replace an older pending frame instead of queuing it.
            self.condition.notify_all()

    def encode(self):
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.closed or self.pending is not None)
                if self.closed:
                    return
                frame, self.pending = self.pending, None
            height, width = frame.shape[:2]
            small = cv2.resize(frame, (800, max(1, int(height * 800 / width))))
            ok, encoded = cv2.imencode('.jpg', small, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if ok:
                with self.condition:
                    self.jpeg = encoded.tobytes()
                    self.updated = time.monotonic()
                    self.sequence += 1
                    self.condition.notify_all()

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()
        self.worker.join(timeout=2)
        self.server.shutdown()
        self.server.server_close()

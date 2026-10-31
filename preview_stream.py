"""On-demand preview encoding with one latest frame and no capture-thread copies."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit, parse_qs
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
        self.requested = False
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                target = urlsplit(self.path)
                if target.path != '/frame':
                    self.send_error(404)
                    return
                try:
                    after = int(parse_qs(target.query).get('after', ['-1'])[0])
                except ValueError:
                    self.send_error(400)
                    return
                with owner.condition:
                    owner.requested = True
                    owner.condition.notify_all()
                    owner.condition.wait_for(lambda: owner.closed or
                        (owner.jpeg is not None and owner.sequence > after), timeout=1.5)
                    if owner.jpeg is None or owner.sequence <= after or time.monotonic() - owner.updated > 2:
                        self.send_error(503)
                        return
                    data, sequence = owner.jpeg, owner.sequence
                self.send_response(200)
                self.send_header('Content-Type', 'image/jpeg')
                self.send_header('Cache-Control', 'no-store')
                self.send_header('Content-Length', str(len(data)))
                self.send_header('X-Frame-Sequence', str(sequence))
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
        with self.condition:
            self.pending = frame  # Replace an older pending frame instead of queuing it.
            self.condition.notify_all()

    def encode(self):
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.closed or (self.requested and self.pending is not None))
                if self.closed:
                    return
                frame, self.pending = self.pending, None
                self.requested = False
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

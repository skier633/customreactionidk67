"""Keep the Studio preview in memory and serve it only on loopback."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import time
import cv2


class PreviewServer:
    def __init__(self, port=5051):
        self.jpeg = None
        self.updated = 0
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
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
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def update(self, frame):
        if time.monotonic() - self.updated < 0.1:
            return
        height, width = frame.shape[:2]
        small = cv2.resize(frame, (800, max(1, int(height * 800 / width))))
        ok, encoded = cv2.imencode('.jpg', small, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ok:
            self.jpeg = encoded.tobytes()
            self.updated = time.monotonic()

    def close(self):
        self.server.shutdown()
        self.server.server_close()

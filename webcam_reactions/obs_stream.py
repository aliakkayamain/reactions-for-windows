"""Local MJPEG stream of the output, added to OBS as a "Browser" source (http://127.0.0.1:<port>/)."""
import http.server
import socketserver
import threading

import cv2

from . import config as C

PAGE = b"""<!doctype html>
<html><head><meta charset="utf-8"><title>Webcam Reactions</title>
<style>html,body{margin:0;height:100%;overflow:hidden;background:transparent}
img{width:100vw;height:100vh;object-fit:cover;display:block}</style></head>
<body><img id="v" alt="">
<script>
// Reconnects automatically when the app is restarted.
var v = document.getElementById('v');
function connect(){ v.style.visibility='visible'; v.src='/stream?t=' + Date.now(); }
v.onerror = function(){ v.style.visibility='hidden'; setTimeout(connect, 1000); };
connect();
</script></body></html>"""


class ObsStream:
    """Serves the latest frame as multipart MJPEG. Bound to 127.0.0.1 only (not reachable from the network)."""

    def __init__(self, port):
        self.port = port
        self.cond = threading.Condition()
        self.jpeg = None
        self.seq = 0
        self.viewers = 0
        self.snapshot_requests = 0
        self.running = True
        stream = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                path = self.path.split("?")[0]
                if path in ("/", "/index.html"):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(PAGE)
                elif path == "/stream":
                    self._stream()
                elif path == "/frame.jpg":
                    stream.snapshot_requests += 1               # make the next frame get encoded
                    try:
                        jpeg, _ = stream.wait_frame(stream.seq, 2.0)
                    finally:
                        stream.snapshot_requests -= 1
                    if jpeg is None:
                        self.send_error(503)
                        return
                    self.send_response(200)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(jpeg)))
                    self.end_headers()
                    self.wfile.write(jpeg)
                else:
                    self.send_error(404)

            def _stream(self):
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                with stream.cond:
                    stream.viewers += 1
                last = -1
                try:
                    while stream.running:
                        jpeg, last = stream.wait_frame(last, 2.0)
                        if jpeg is None:
                            continue
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(jpeg))
                        self.wfile.write(jpeg)
                        self.wfile.write(b"\r\n")
                except (ConnectionError, OSError):
                    pass
                finally:
                    with stream.cond:
                        stream.viewers -= 1

        class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
            daemon_threads = True
            allow_reuse_address = False

        self.server = Server(("127.0.0.1", port), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    @property
    def url(self):
        return "http://127.0.0.1:%d/" % self.port

    def wait_frame(self, last_seq, timeout):
        """Wait for a frame newer than last_seq; returns (jpeg, seq) or (None, last_seq)."""
        with self.cond:
            if self.seq == last_seq or self.jpeg is None:
                self.cond.wait(timeout)
            if self.seq == last_seq:
                return None, last_seq
            return self.jpeg, self.seq

    def send(self, frame):
        """Encode and publish the frame - only when someone (OBS) is watching."""
        if self.viewers <= 0 and self.snapshot_requests <= 0 and self.jpeg is not None:
            return
        ok, data = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, C.OBS_JPEG_QUALITY])
        if not ok:
            return
        with self.cond:
            self.jpeg = data.tobytes()
            self.seq += 1
            self.cond.notify_all()

    def close(self):
        self.running = False
        with self.cond:
            self.cond.notify_all()
        try:
            self.server.shutdown()
            self.server.server_close()
        except Exception:
            pass

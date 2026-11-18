"""
Object detection on the GPU: phone, bottle, pen, cup and headphones.

Model: D-FINE-L pretrained on Objects365 (models/dfine_l_obj365.onnx, made by tools/export_detector.py),
run with ONNX Runtime + DirectML, so it uses the discrete GPU on any DirectX 12 card (falls back to
the CPU if no GPU is available). Inference runs on a background thread on the newest frame, so the
video loop never waits for it.
"""
import threading
import time

import cv2
import numpy as np

from . import config as C

# Objects365 label ids (the exported model outputs category ids, 1-based) -> our class name
OBJECTS365_IDS = {
    62: "phone",         # Cell Phone
    9: "bottle",         # Bottle
    55: "pen",           # Pen/Pencil
    11: "cup",           # Cup
    126: "headphones",   # Head Phone
    208: "headphones",   # Earphone
}
COLORS = {                # BGR
    "phone": (255, 132, 10), "bottle": (88, 209, 48), "pen": (10, 214, 255),
    "cup": (242, 90, 191), "headphones": (58, 69, 255),
}
INPUT_SIZE = 640


def _iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class ObjectDetector:
    def __init__(self, width, height):
        import onnxruntime as ort

        self.w, self.h = width, height
        so = ort.SessionOptions()
        so.log_severity_level = 3
        providers = ["CPUExecutionProvider"]
        if C.DETECTION_USE_GPU and "DmlExecutionProvider" in ort.get_available_providers():
            # high_performance = the discrete GPU on laptops with integrated + discrete graphics
            providers.insert(0, ("DmlExecutionProvider", {"performance_preference": "high_performance",
                                                          "device_filter": "gpu"}))
        self.session = ort.InferenceSession(C.DETECTION_MODEL, so, providers=providers)
        self.device = "GPU (DirectML)" if self.session.get_providers()[0] == "DmlExecutionProvider" else "CPU"
        self.size = np.array([[width, height]], np.int64)
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.pending = None                 # newest preprocessed frame waiting for the worker
        self.tracks = []                    # [{name, score, box, seen}]
        self.infer_ms = 0.0
        self.running = True
        self.thread = threading.Thread(target=self._worker, name="object-detector", daemon=True)
        self.thread.start()

    # --- main thread ---
    def submit(self, frame):
        """Queue the frame (only the newest one is kept). Resize + colour conversion happen here,
        which also copies the frame, so effects drawn later do not leak into the detection."""
        img = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        with self.lock:
            self.pending = img
        self.wake.set()

    def detections(self):
        """Current objects: list of (name, score, (x0, y0, x1, y1)) in frame pixels."""
        now = time.perf_counter()
        with self.lock:
            return [(t["name"], t["score"], tuple(t["box"])) for t in self.tracks
                    if now - t["seen"] < C.DETECTION_HOLD and t["hits"] >= C.DETECTION_MIN_HITS]

    def close(self):
        self.running = False
        self.wake.set()
        self.thread.join(timeout=2)

    # --- worker thread ---
    def _worker(self):
        while self.running:
            self.wake.wait(0.5)
            self.wake.clear()
            with self.lock:
                img, self.pending = self.pending, None
            if img is None or not self.running:
                continue
            t0 = time.perf_counter()
            blob = (img.transpose(2, 0, 1)[None].astype(np.float32) * (1 / 255.0))
            try:
                labels, boxes, scores = self.session.run(None, {"images": blob, "orig_target_sizes": self.size})
            except Exception as e:
                print("WARNING: object detection failed: %s" % e)
                time.sleep(1.0)
                continue
            self.infer_ms = self.infer_ms * 0.9 + (time.perf_counter() - t0) * 1000 * 0.1
            self._update(self._filter(labels[0], boxes[0], scores[0]))

    def _filter(self, labels, boxes, scores):
        keep = scores >= C.DETECTION_SCORE
        found = []
        for label, box, score in zip(labels[keep], boxes[keep], scores[keep]):
            name = OBJECTS365_IDS.get(int(label))
            if name is None:
                continue
            x0, y0, x1, y1 = (float(v) for v in box)
            x0, x1 = max(0.0, min(x0, x1)), min(float(self.w), max(x0, x1))
            y0, y1 = max(0.0, min(y0, y1)), min(float(self.h), max(y0, y1))
            if min(x1 - x0, y1 - y0) < max(4.0, C.DETECTION_MIN_SIZE * self.h):
                continue
            found.append((name, float(score), [x0, y0, x1, y1]))
        found.sort(key=lambda d: -d[1])
        result = []                                         # per-class overlap removal
        for d in found:
            if all(not (r[0] == d[0] and _iou(r[2], d[2]) > 0.5) for r in result):
                result.append(d)
        return result

    def _update(self, found):
        """Match to the previous boxes and smooth them, so the boxes do not jitter."""
        now = time.perf_counter()
        a = C.DETECTION_SMOOTHING
        with self.lock:
            tracks = [t for t in self.tracks if now - t["seen"] < C.DETECTION_HOLD]
            used = set()
            for name, score, box in found:
                best, best_iou = None, 0.3
                for i, t in enumerate(tracks):
                    if i in used or t["name"] != name:
                        continue
                    iou = _iou(t["box"], box)
                    if iou > best_iou:
                        best, best_iou = i, iou
                if best is None:
                    tracks.append({"name": name, "score": score, "box": box, "seen": now, "hits": 1})
                    used.add(len(tracks) - 1)
                else:
                    t = tracks[best]
                    t["box"] = [p * a + q * (1 - a) for p, q in zip(t["box"], box)]
                    t["score"], t["seen"] = score, now
                    t["hits"] += 1
                    used.add(best)
            for i, t in enumerate(tracks):                      # missed this time: must re-earn its hits
                if i not in used and t["hits"] < C.DETECTION_MIN_HITS:
                    t["hits"] = 0
            self.tracks = [t for t in tracks if t["hits"] > 0]


# =============================================================================
# Drawing (on the output frame)
# =============================================================================
_LABEL_CACHE = {}


def _label_sprite(text, color):
    """Rounded label image (BGRA) rendered once with PIL (proper font, any language)."""
    key = (text, color)
    sprite = _LABEL_CACHE.get(key)
    if sprite is None:
        from PIL import Image, ImageDraw, ImageFont
        try:
            font = ImageFont.truetype(C.FONT_BOLD, 18)
        except Exception:
            font = ImageFont.load_default()
        l, t, r, b = ImageDraw.Draw(Image.new("RGBA", (1, 1))).textbbox((0, 0), text, font=font)
        w, h = r - l + 20, b - t + 12
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.rounded_rectangle([0, 0, w - 1, h - 1], radius=h // 2, fill=(color[2], color[1], color[0], 235))
        d.text((10 - l, 6 - t), text, font=font, fill=(255, 255, 255, 255))
        sprite = np.asarray(img)[..., [2, 1, 0, 3]].copy()
        _LABEL_CACHE[key] = sprite
    return sprite


def _paste(frame, sprite, x, y):
    h, w = frame.shape[:2]
    sh, sw = sprite.shape[:2]
    x, y = int(min(max(0, x), w - sw)), int(min(max(0, y), h - sh))
    region = frame[y:y + sh, x:x + sw]
    a = sprite[..., 3:4].astype(np.float32) / 255.0
    region[:] = (sprite[..., :3] * a + region * (1 - a)).astype(np.uint8)


def visible_in(dets, mask, min_cover=0.15):
    """Drop boxes that lie in a background the effect has painted over (person mask ~0 there):
    the camera still sees those objects, but on screen they are hidden behind the flag."""
    if mask is None:
        return dets
    keep = []
    for d in dets:
        x0, y0, x1, y1 = (int(v) for v in d[2])
        region = mask[max(0, y0):max(y0 + 1, y1), max(0, x0):max(x0 + 1, x1)]
        if region.size and float(region.mean()) >= min_cover:
            keep.append(d)
    return keep


def draw_detections(frame, dets):
    """Corner-style boxes with a label pill, drawn in place on the BGR frame."""
    for name, score, (x0, y0, x1, y1) in dets:
        color = COLORS.get(name, (255, 255, 255))
        p0, p1 = (int(x0), int(y0)), (int(x1), int(y1))
        cv2.rectangle(frame, p0, p1, color, 2, cv2.LINE_AA)
        c = int(min(x1 - x0, y1 - y0) * 0.22)                  # thicker corners
        for (cx, cy, dx, dy) in ((p0[0], p0[1], 1, 1), (p1[0], p0[1], -1, 1),
                                 (p0[0], p1[1], 1, -1), (p1[0], p1[1], -1, -1)):
            cv2.line(frame, (cx, cy), (cx + dx * c, cy), color, 5, cv2.LINE_AA)
            cv2.line(frame, (cx, cy), (cx, cy + dy * c), color, 5, cv2.LINE_AA)
        text = "%s  %d%%" % (C.DETECTION_LABELS.get(name, name), round(score * 100))
        sprite = _label_sprite(text, color)
        _paste(frame, sprite, x0, y0 - sprite.shape[0] - 6 if y0 > sprite.shape[0] + 6 else y0 + 6)

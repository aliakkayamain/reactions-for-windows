"""Hand gesture detection: MediaPipe Gesture Recognizer plus custom landmark-based gestures."""
import threading
import time

import cv2
import numpy as np

from . import config as C
from .effects import (HEART, LASERS, PARTY, THUMBS_DOWN, THUMBS_UP, TWO_THUMBS_DOWN, TWO_THUMBS_UP,
                      TWO_VICTORY, VICTORY, WOLF)

# MediaPipe hand landmark indices: 0 wrist, 4 thumb tip, 8 index tip, 12 middle tip,
# 16 ring tip, 20 pinky tip; 6/10/14/18 are the middle (PIP) joints, 9 the middle finger base.


def _dist(p, a, b):
    return float(np.linalg.norm(p[a] - p[b]))


def _extended(p, tip, pip):
    """A finger is extended when its tip is clearly farther from the wrist than its middle joint."""
    return _dist(p, 0, tip) > _dist(p, 0, pip) * 1.15


def is_horns(p):
    """🤘 index and pinky extended, middle and ring folded (thumb free)."""
    return (_extended(p, 8, 6) and _extended(p, 20, 18)
            and not _extended(p, 12, 10) and not _extended(p, 16, 14))


def is_wolf_sign(p):
    """Grey Wolf sign: index + pinky up (ears), middle + ring fingertips touching the thumb tip (snout)."""
    size = _dist(p, 0, 9)
    if size < 1 or not is_horns(p):
        return False
    return (_dist(p, 4, 12) < 0.5 * size and _dist(p, 4, 16) < 0.5 * size
            and _dist(p, 4, 8) > 0.6 * size and _dist(p, 4, 20) > 0.6 * size)


class GestureDetector:
    """Runs MediaPipe Gesture Recognizer asynchronously (LIVE_STREAM mode)."""

    def __init__(self):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self.mp = mp
        with open(C.GESTURE_MODEL, "rb") as f:
            model = f.read()
        self.lock = threading.Lock()
        self.result = None
        self.result_count = 0
        self.last_ts = 0
        options = vision.GestureRecognizerOptions(
            base_options=mp_python.BaseOptions(model_asset_buffer=model),
            running_mode=vision.RunningMode.LIVE_STREAM,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            result_callback=self._on_result,
        )
        self.recognizer = vision.GestureRecognizer.create_from_options(options)

    def _on_result(self, result, _image, _ts):
        with self.lock:
            self.result = result
            self.result_count += 1

    def submit(self, frame_bgr):
        """Send a (downscaled) frame to the recognizer without waiting."""
        h, w = frame_bgr.shape[:2]
        if w > C.DETECTION_WIDTH:
            frame_bgr = cv2.resize(frame_bgr, (C.DETECTION_WIDTH, int(h * C.DETECTION_WIDTH / w)),
                                   interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        ts = max(self.last_ts + 1, int(time.monotonic() * 1000))
        self.last_ts = ts
        self.recognizer.recognize_async(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb), ts)

    def detect(self, w, h):
        """Return (gesture | None, center_px, hand_count) for the latest result."""
        with self.lock:
            result = self.result
        if result is None or not result.hand_landmarks:
            return None, None, 0
        names = [g[0].category_name if g and g[0].score >= C.MIN_GESTURE_SCORE else "None"
                 for g in result.gestures]
        hands = [np.array([[lm.x * w, lm.y * h] for lm in hand]) for hand in result.hand_landmarks]
        center = np.mean([p[9] for p in hands], axis=0)
        center = (int(center[0]), int(center[1]))

        if len(hands) == 2:
            a, b = hands
            size = (_dist(a, 0, 9) + _dist(b, 0, 9)) / 2
            if size > 1:
                # Heart: thumb tips touching at the bottom, index tips touching at the top
                thumbs = np.linalg.norm(a[4] - b[4]) / size
                index = np.linalg.norm(a[8] - b[8]) / size
                opening = (_dist(a, 4, 8) + _dist(b, 4, 8)) / 2 / size
                if thumbs < 0.45 and index < 0.45 and (a[4][1] + b[4][1]) > (a[8][1] + b[8][1]) \
                        and opening > 0.35 and not ({"Victory", "Thumb_Up", "Thumb_Down"} & set(names)):
                    m = (a[4] + b[4] + a[8] + b[8]) / 4
                    return HEART, (int(m[0]), int(m[1])), 2
            if is_horns(a) and is_horns(b):
                return LASERS, center, 2
            if names.count("Thumb_Up") == 2:
                return TWO_THUMBS_UP, center, 2
            if names.count("Thumb_Down") == 2:
                return TWO_THUMBS_DOWN, center, 2
            if names.count("Victory") == 2:
                return TWO_VICTORY, center, 2
            if names.count("Open_Palm") == 2:
                return PARTY, center, 2

        for i, p in enumerate(hands):
            if names[i] not in ("Thumb_Up", "Thumb_Down", "Victory") and is_wolf_sign(p):
                return WOLF, (int(p[9][0]), int(p[9][1])), len(hands)

        for label, gesture in (("Thumb_Up", THUMBS_UP), ("Thumb_Down", THUMBS_DOWN), ("Victory", VICTORY)):
            if names.count(label) == 1 and all(n in (label, "None", "Open_Palm", "Closed_Fist") for n in names):
                i = names.index(label)
                return gesture, (int(hands[i][9][0]), int(hands[i][9][1])), len(hands)
        return None, center, len(hands)

    def close(self):
        try:
            self.recognizer.close()
        except Exception:
            pass

"""
macOS-style video effects, all based on MediaPipe person segmentation:
  - Portrait      : blurred background, sharp person
  - Studio Light  : brightened person, dimmed background
  - Center Stage  : smooth auto-framing that keeps the person centered
"""
import time

import cv2
import numpy as np

from . import config as C
from .graphics import blend


class PersonSegmenter:
    """Produces a per-frame person mask (0 = background, 1 = person), float32, same size as the frame."""

    def __init__(self):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self.mp = mp
        with open(C.SEGMENTATION_MODEL, "rb") as f:
            model = f.read()
        options = vision.ImageSegmenterOptions(
            base_options=mp_python.BaseOptions(model_asset_buffer=model),
            running_mode=vision.RunningMode.VIDEO,
            output_confidence_masks=True,
            output_category_mask=False,
        )
        self.segmenter = vision.ImageSegmenter.create_from_options(options)
        self.previous = None
        self.last_ts = 0

    def mask(self, frame):
        h, w = frame.shape[:2]
        small = cv2.cvtColor(cv2.resize(frame, (256, 144), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        ts = max(self.last_ts + 1, int(time.monotonic() * 1000))
        self.last_ts = ts
        result = self.segmenter.segment_for_video(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=small), ts)
        m = result.confidence_masks[-1].numpy_view().astype(np.float32)   # last mask = person
        if m.ndim == 3:
            m = m[..., 0]
        if self.previous is not None and self.previous.shape == m.shape:
            m = self.previous * 0.35 + m * 0.65                 # temporal smoothing against flicker
        self.previous = m
        m = cv2.resize(m, (w // 2, h // 2), interpolation=cv2.INTER_LINEAR)
        m = np.clip((m - 0.25) / 0.5, 0, 1)                      # sharpen the edge
        m = cv2.GaussianBlur(m, (0, 0), 2.0)
        return cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)

    def close(self):
        try:
            self.segmenter.close()
        except Exception:
            pass


def apply_portrait(frame, mask):
    """Blur the background (shallow depth of field)."""
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), max(1, C.PORTRAIT_BLUR) / 4.0)
    blurred = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    blend(frame, blurred, 1.0 - mask)


def apply_studio_light(frame, mask):
    """Brighten the person and dim the background."""
    person = cv2.convertScaleAbs(frame, alpha=1.12, beta=10)
    background = cv2.convertScaleAbs(frame, alpha=C.STUDIO_BACKGROUND_LEVEL)
    cv2.blendLinear(person, background, np.ascontiguousarray(mask), np.ascontiguousarray(1.0 - mask), dst=frame)


class CenterStage:
    """Keeps the person centered and zooms in when they are far away, with smooth camera motion."""

    def __init__(self, w, h):
        self.w, self.h = w, h
        self.cx, self.cy, self.zoom = w / 2.0, h / 2.0, 1.0

    def update(self, mask, dt):
        ys, xs = np.nonzero(mask[::8, ::8] > 0.5)
        if len(xs) > 30:
            x0, x1 = np.percentile(xs, [3, 97]) * 8
            top = np.percentile(ys, 2) * 8
            target_cx = (x0 + x1) / 2
            # head ~12% from the top of the crop, person ~55% of the crop width
            target_zoom = np.clip(min(self.w * 0.55 / max(1.0, x1 - x0), self.h * 0.9 / max(1.0, self.h - top)),
                                  1.0, C.CENTER_STAGE_MAX_ZOOM)
            view_h = self.h / target_zoom
            target_cy = top - view_h * 0.12 + view_h / 2
        else:
            target_cx, target_cy, target_zoom = self.w / 2, self.h / 2, 1.0
        k = 1 - np.exp(-dt * 2.2)
        self.cx += (target_cx - self.cx) * k
        self.cy += (target_cy - self.cy) * k
        self.zoom += (target_zoom - self.zoom) * k

    def apply(self, *images):
        vw, vh = self.w / self.zoom, self.h / self.zoom
        x0 = float(np.clip(self.cx - vw / 2, 0, self.w - vw))
        y0 = float(np.clip(self.cy - vh / 2, 0, self.h - vh))
        if self.zoom < 1.01 and abs(x0) < 1 and abs(y0) < 1:
            return images
        m = np.float32([[self.zoom, 0, -x0 * self.zoom], [0, self.zoom, -y0 * self.zoom]])
        return tuple(cv2.warpAffine(img, m, (self.w, self.h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                     for img in images)

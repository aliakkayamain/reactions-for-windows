"""Modern on-screen UI for the preview window (glass panels, progress ring, shortcut list)."""
import math
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import config as C
from .effects import LABELS
from .graphics import blend

BLUE = (10, 132, 255)
GREEN = (48, 209, 88)
RED = (255, 69, 58)
PURPLE = (191, 90, 242)
GLASS = (20, 20, 24)

SHORTCUTS = [
    ("1", "Thumbs up"), ("2", "Thumbs down"), ("3", "Hearts"), ("4", "Balloons"),
    ("5", "Rain"), ("6", "Confetti"), ("7", "Fireworks"), ("8", "Lasers"),
    ("9", "Grey Wolf + flag"), ("P", "Party mode on/off"),
    ("B", "Portrait (blur background)"), ("L", "Studio light"), ("C", "Center Stage"),
    ("E", "Gestures on/off"), ("H", "Hide this list"), ("Q", "Quit"),
]


def _font(path, size):
    for p in (path, "arial.ttf"):
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            pass
    return ImageFont.load_default()


class Overlay:
    def __init__(self):
        self.small = _font(C.FONT, 13)
        self.large = _font(C.FONT_BOLD, 18)
        self.show_help = False

    def _pill(self, d, x, y, text, font, fill=GLASS + (150,), color=(255, 255, 255), pad=(12, 6), right=False):
        """Rounded 'pill' label; returns its (width, height)."""
        l, t, r, b = d.textbbox((0, 0), text, font=font)
        w, h = r - l + pad[0] * 2, b - t + pad[1] * 2
        if right:
            x -= w
        d.rounded_rectangle([x, y, x + w, y + h], radius=h // 2, fill=fill)
        d.text((x + pad[0] - l, y + pad[1] - t), text, font=font, fill=color)
        return w, h

    def render(self, frame, s):
        ph = int(C.HEIGHT * C.PREVIEW_WIDTH / C.WIDTH)
        preview = cv2.resize(frame, (C.PREVIEW_WIDTH, ph), interpolation=cv2.INTER_AREA)
        if C.MIRROR_PREVIEW != C.MIRROR_OUTPUT:
            preview = cv2.flip(preview, 1)
        W, H = C.PREVIEW_WIDTH, ph
        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)

        # top left: live indicator
        pulse = 0.5 + 0.5 * math.sin(time.time() * 4)
        _, h = self._pill(d, 12, 12, "      LIVE   %d FPS   Hands %d" % (round(s["fps"]), s["hands"]), self.small)
        d.ellipse([22, 12 + h / 2 - 5, 32, 12 + h / 2 + 5], fill=RED + (int(150 + 105 * pulse),))

        # top right: mode chips
        x = W - 12
        for label, on, color in (("Party", s["party"], PURPLE), ("Center Stage", s["center_stage"], BLUE),
                                 ("Studio", s["studio"], BLUE), ("Portrait", s["portrait"], BLUE),
                                 ("Gestures", s["gestures"], GREEN)):
            w, _ = self._pill(d, x, 12, label, self.small, color + (230,) if on else GLASS + (140,),
                              (255, 255, 255) if on else (170, 170, 175), pad=(10, 5), right=True)
            x -= w + 6

        # bottom center: gesture progress or active effect
        if s["active"]:
            text = LABELS.get(s["active"], s["active"])
            self._pill(d, W // 2 - d.textlength(text, font=self.large) // 2 - 16, H - 52, text, self.large,
                       GLASS + (170,), pad=(16, 8))
        elif s["candidate"]:
            text = LABELS.get(s["candidate"], s["candidate"])
            x0 = int(W // 2 - (d.textlength(text, font=self.large) + 60) // 2)
            _, h = self._pill(d, x0, H - 52, "      " + text, self.large, GLASS + (170,), pad=(16, 8))
            cx, cy, r = x0 + 24, H - 52 + h / 2, 9
            d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(255, 255, 255, 70), width=3)
            d.arc([cx - r, cy - r, cx + r, cy + r], -90, -90 + 360 * min(1.0, s["progress"]), fill=BLUE + (255,), width=3)

        # bottom: outputs and help hint
        outputs = []
        if s["virtual_cam"]:
            outputs.append("Virtual camera")
        if s["obs"] >= 0:
            outputs.append("OBS connected" if s["obs"] > 0 else "OBS waiting")
        self._pill(d, 12, H - 34, "  ·  ".join(outputs) or "No output", self.small, GLASS + (130,), (200, 200, 205), pad=(10, 5))
        self._pill(d, W - 12, H - 34, "H  shortcuts", self.small, GLASS + (130,), (200, 200, 205), pad=(10, 5), right=True)

        if self.show_help:
            pw, ph2 = 300, 26 + len(SHORTCUTS) * 19
            px, py = (W - pw) // 2, max(50, (H - ph2) // 2)
            d.rounded_rectangle([px, py, px + pw, py + ph2], radius=16, fill=(16, 16, 20, 215))
            for i, (key, what) in enumerate(SHORTCUTS):
                y = py + 14 + i * 19
                d.rounded_rectangle([px + 16, y, px + 40, y + 16], radius=5, fill=(255, 255, 255, 40))
                d.text((px + 28, y + 8), key, font=self.small, fill=(255, 255, 255), anchor="mm")
                d.text((px + 52, y + 8), what, font=self.small, fill=(230, 230, 235), anchor="lm")

        rgba = np.asarray(layer)
        return blend(np.ascontiguousarray(preview), cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR),
                     rgba[..., 3].astype(np.float32) * (1 / 255.0))

"""Drawing helpers: emoji sprites (static + animated), easing curves, glow / light layers."""
import os

import cv2
import numpy as np


# =============================================================================
# Easing
# =============================================================================
def clamp(x, lo=0.0, hi=1.0):
    return lo if x < lo else hi if x > hi else x


def ease_out_back(x):
    """0 -> 1 with a slight overshoot at the end."""
    x = clamp(x)
    c1 = 1.70158
    return 1 + (c1 + 1) * (x - 1) ** 3 + c1 * (x - 1) ** 2


def ease_out_cubic(x):
    return 1 - (1 - clamp(x)) ** 3


def fade_in_out(t, duration, fade_in=0.4, fade_out=0.6):
    """Overall visibility of an effect: fades in at the start and out at the end (0-1)."""
    return clamp(t / fade_in if fade_in > 0 else 1) * clamp((duration - t) / fade_out if fade_out > 0 else 1)


def hsv_color(h, s=1.0, v=1.0):
    """h in 0-1 -> BGR tuple (0-255)."""
    i = int(h * 6) % 6
    f = h * 6 - int(h * 6)
    p, q, t = v * (1 - s), v * (1 - f * s), v * (1 - (1 - f) * s)
    r, g, b = [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)][i]
    return (int(b * 255), int(g * 255), int(r * 255))


# =============================================================================
# Blending
# =============================================================================
def blend(base, top, alpha, dst=None):
    """base*(1-alpha) + top*alpha, where alpha is a float32 2-D weight map (0-1).
    Uses cv2.blendLinear, which is ~30x faster than the numpy equivalent."""
    alpha = np.ascontiguousarray(alpha, np.float32)
    return cv2.blendLinear(base, np.ascontiguousarray(top), 1.0 - alpha, alpha,
                           dst=dst if dst is not None else base)


def darken(frame, amount, tint=None):
    """Darken the frame in place by `amount` (0 = unchanged, 1 = fully dark), optionally towards a tint."""
    if amount <= 0.005:
        return
    if tint is None:
        cv2.convertScaleAbs(frame, dst=frame, alpha=1 - amount)
    else:
        color = np.empty_like(frame)
        color[:] = tint
        cv2.addWeighted(frame, 1 - amount, color, amount, 0, dst=frame)


class LightLayer:
    """Additive light layer drawn at reduced resolution, blurred (bloom) and added to the frame."""

    def __init__(self, w, h, scale=0.5):
        self.w, self.h = w, h
        self.scale = scale
        self.canvas = np.zeros((int(h * scale), int(w * scale), 3), np.uint8)

    def clear(self, decay=0.0):
        """Clear the layer, or with `decay` fade the previous content (light trails)."""
        if decay > 0:
            cv2.convertScaleAbs(self.canvas, dst=self.canvas, alpha=decay)
        else:
            self.canvas[:] = 0

    def p(self, x, y):
        """Full-resolution point -> layer point."""
        return (int(x * self.scale), int(y * self.scale))

    def apply(self, frame, glow=1.0, bloom=13, gain=1.0):
        layer = self.canvas
        if bloom:
            blurred = cv2.GaussianBlur(layer, (0, 0), bloom * self.scale * 2)
            layer = cv2.addWeighted(layer, gain, blurred, glow * 1.6, 0)
        elif gain != 1.0:
            layer = cv2.convertScaleAbs(layer, alpha=gain)
        full = cv2.resize(layer, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        cv2.add(frame, full, dst=frame)


# =============================================================================
# Emoji store
# =============================================================================
class EmojiStore:
    """
    assets/emoji/static/*.png   -> static 3D emoji (Microsoft Fluent Emoji, MIT)
    assets/emoji/animated/*.npz -> animated emoji (Google Noto Animated Emoji, CC BY 4.0),
                                   built by tools/setup_assets.py
    Animated emoji are addressed with the "anim_" prefix: EMOJI.draw(frame, "anim_wolf", ..., time=t)
    """
    BALLOON_HUES = (0, 12, 26, 60, 105, 140)   # red, orange, yellow, green, blue, purple (OpenCV hue shift)

    def __init__(self, emoji_dir=""):
        self.static = {}       # name -> BGRA image
        self.animated = {}     # "anim_name" -> (frames[N,H,W,4], frame_start_times[N], total_duration)
        self.cache = {}
        if not emoji_dir or not os.path.isdir(emoji_dir):
            return
        static_dir = os.path.join(emoji_dir, "static")
        if os.path.isdir(static_dir):
            for name in os.listdir(static_dir):
                if name.lower().endswith(".png"):
                    img = cv2.imread(os.path.join(static_dir, name), cv2.IMREAD_UNCHANGED)
                    if img is not None and img.ndim == 3 and img.shape[2] == 4:
                        self.static[os.path.splitext(name)[0]] = img
        anim_dir = os.path.join(emoji_dir, "animated")
        if os.path.isdir(anim_dir):
            for name in os.listdir(anim_dir):
                if name.lower().endswith(".npz"):
                    try:
                        data = np.load(os.path.join(anim_dir, name))
                        frames, durations = data["frames"], data["durations"].astype(np.float64)
                        starts = np.concatenate([[0.0], np.cumsum(durations)[:-1]])
                        self.animated["anim_" + os.path.splitext(name)[0]] = (frames, starts, float(durations.sum()))
                    except Exception:
                        pass
        if "balloon" in self.static:                    # recolored balloon variants
            hsv = cv2.cvtColor(self.static["balloon"][..., :3], cv2.COLOR_BGR2HSV)
            for i, shift in enumerate(self.BALLOON_HUES):
                tinted = hsv.copy()
                tinted[..., 0] = (tinted[..., 0].astype(np.int16) + shift) % 180
                bgr = cv2.cvtColor(tinted, cv2.COLOR_HSV2BGR)
                self.static["balloon_%d" % i] = np.dstack([bgr, self.static["balloon"][..., 3]])

    def has(self, *names):
        return all(n in self.static or n in self.animated for n in names)

    def _source(self, name, time):
        if name in self.animated:
            frames, starts, total = self.animated[name]
            i = max(0, int(np.searchsorted(starts, (time or 0.0) % total, side="right") - 1))
            return frames[i], (name, i)
        return self.static[name], (name, -1)

    def _scaled(self, name, size, mirror, time):
        src, key0 = self._source(name, time)
        step = 8 if name in self.animated else 4
        size = max(step, int(size) // step * step)
        key = key0 + (size, mirror)
        img = self.cache.get(key)
        if img is None:
            interp = cv2.INTER_AREA if size < src.shape[0] else cv2.INTER_LINEAR
            img = cv2.resize(src, (size, size), interpolation=interp)
            if mirror:
                img = cv2.flip(img, 1)
            if len(self.cache) > 900:
                self.cache.clear()
            self.cache[key] = img
        return img

    def draw(self, frame, name, cx, cy, size, alpha=1.0, mirror=False, time=None, angle=0.0):
        """Draw emoji `name` centered at (cx, cy), `size` px wide. angle in degrees."""
        if size < 4 or alpha <= 0.01 or not self.has(name):
            return
        img = self._scaled(name, size, mirror, time)
        if abs(angle) > 0.5:
            s = img.shape[0]
            m = cv2.getRotationMatrix2D((s / 2, s / 2), angle, 1.0)
            img = cv2.warpAffine(img, m, (s, s), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))
        self._paste(frame, img[..., :3], img[..., 3], cx, cy, min(1.0, alpha) / 255.0)

    def draw_shadow(self, frame, name, cx, cy, size, alpha=1.0, time=None, offset=0.06):
        """Soft drop shadow under an emoji (gives a 3D feel)."""
        if size < 16 or alpha <= 0.01 or not self.has(name):
            return
        _, key0 = self._source(name, time)
        size = int(size) // 8 * 8
        key = ("shadow",) + key0 + (size,)
        mask = self.cache.get(key)
        if mask is None:
            img = self._scaled(name, size * 0.5, False, time)
            s = img.shape[0]
            pad = int(s * 0.2)                          # room so the blur fades out before the edge
            mask = cv2.copyMakeBorder(img[..., 3], pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
            mask = cv2.GaussianBlur(mask, (0, 0), max(1.0, s * 0.06))
            mask = cv2.resize(mask, (int(size * 1.4) // 2 * 2,) * 2, interpolation=cv2.INTER_LINEAR)
            self.cache[key] = mask
        self._paste(frame, None, mask, cx, cy + size * offset, 0.45 * alpha / 255.0)

    @staticmethod
    def _paste(frame, color, alpha_u8, cx, cy, factor):
        s = alpha_u8.shape[0]
        h, w = frame.shape[:2]
        x0, y0 = int(round(cx - s / 2)), int(round(cy - s / 2))
        fx0, fy0, fx1, fy1 = max(0, x0), max(0, y0), min(w, x0 + s), min(h, y0 + s)
        if fx1 <= fx0 or fy1 <= fy0:
            return
        sy, sx = slice(fy0 - y0, fy1 - y0), slice(fx0 - x0, fx1 - x0)
        a = alpha_u8[sy, sx].astype(np.float32) * factor
        region = frame[fy0:fy1, fx0:fx1]
        blend(region, color[sy, sx] if color is not None else np.zeros_like(region), a, dst=region)


EMOJI = EmojiStore()


def load_emoji(emoji_dir):
    """Called once at startup; every module shares the same EMOJI instance."""
    EMOJI.__dict__.update(EmojiStore(emoji_dir).__dict__)
    return EMOJI

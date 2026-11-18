"""
Howling grey wolf for the Grey Wolf effect, drawn BEHIND the person (on top of the flag).

Two sources:
  1. A real clip, if one is present in assets/wolf/  (first file found is used):
       .gif / .webp / .png (animated, with transparency)  -> used as-is
       .mp4 / .mov / .webm / .avi / .mkv                    -> plain-colour (green screen) background
                                                              is keyed out automatically
  2. Otherwise a procedurally animated wolf (sitting, raises its head and howls).

Both return BGRA frames; the effect composites them onto the flag before the person mask is
applied, so the wolf always stays behind you and the person cut-out is untouched.
"""
import math
import os

import cv2
import numpy as np

from . import config as C
from .graphics import clamp, ease_out_cubic

CLIP_EXTS_ALPHA = (".gif", ".webp", ".png", ".apng")
CLIP_EXTS_VIDEO = (".mp4", ".mov", ".webm", ".avi", ".mkv", ".m4v")
CLIP_MAX_FRAMES = 300
CUTOUT_VERSION = 2


def smoothstep(x):
    x = clamp(x)
    return x * x * (3 - 2 * x)


def clip_max_height():
    return int(min(720, C.HEIGHT * C.WOLF_SIZE * 1.05))


# =============================================================================
# Real clip (GIF / WebP / green-screen video / ordinary video)
# =============================================================================
class WolfClip:
    """Frames (BGRA, cropped to the wolf) + per-frame durations. Played forwards then backwards
    (ping-pong) so the loop has no jump. `touches` = edges of the source frame the wolf is cut off at."""

    def __init__(self, frames, durations, path, touches=()):
        self.path = path
        self.frames = frames
        self.touches = set(touches)
        d = np.asarray(durations, np.float64)
        self.order = list(range(len(frames))) + list(range(len(frames) - 2, 0, -1))
        dur = np.array([d[i] for i in self.order])
        self.starts = np.concatenate([[0.0], np.cumsum(dur)[:-1]])
        self.total = float(dur.sum())
        self.aspect = frames[0].shape[1] / frames[0].shape[0]   # w / h
        self.cache = {}

    def frame(self, t, height):
        k = max(0, int(np.searchsorted(self.starts, t % self.total, side="right") - 1))
        i = self.order[k]
        height = max(8, int(height) // 4 * 4)
        key = (i, height)
        img = self.cache.get(key)
        if img is None:
            src = self.frames[i]
            w = max(4, int(round(height * self.aspect)))
            interp = cv2.INTER_AREA if height < src.shape[0] else cv2.INTER_LINEAR
            img = cv2.resize(src, (w, height), interpolation=interp)
            if len(self.cache) > 2 * len(self.frames):
                self.cache.clear()
            self.cache[key] = img
        return img


def _key_background(bgr):
    """Alpha for a clip shot on a plain background (green screen etc.): the background colour
    is sampled from the frame border; pixels close to it become transparent."""
    h, w = bgr.shape[:2]
    b = max(2, min(h, w) // 40)
    border = np.concatenate([bgr[:b].reshape(-1, 3), bgr[-b:].reshape(-1, 3),
                             bgr[:, :b].reshape(-1, 3), bgr[:, -b:].reshape(-1, 3)])
    key = np.median(border, axis=0)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    key_lab = cv2.cvtColor(key.reshape(1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)[0, 0]
    dist = np.sqrt(((lab - key_lab) ** 2).sum(axis=2))
    alpha = np.clip((dist - 18.0) / 30.0, 0, 1)
    alpha = cv2.GaussianBlur(alpha, (0, 0), 0.8)
    # remove colour spill (green fringe) on the edges
    is_green = key[1] > key[0] + 20 and key[1] > key[2] + 20
    if is_green:
        g = bgr[..., 1].astype(np.float32)
        limit = np.maximum(bgr[..., 0], bgr[..., 2]).astype(np.float32)
        bgr = bgr.copy()
        bgr[..., 1] = np.minimum(g, limit + 8).astype(np.uint8)
    return bgr, (alpha * 255).astype(np.uint8)


def _is_plain_background(bgr):
    """True for a green screen / studio backdrop: nearly all border pixels have the same colour."""
    h, w = bgr.shape[:2]
    b = max(2, min(h, w) // 40)
    border = np.concatenate([bgr[:b].reshape(-1, 3), bgr[-b:].reshape(-1, 3),
                             bgr[:, :b].reshape(-1, 3), bgr[:, -b:].reshape(-1, 3)])
    lab = cv2.cvtColor(border.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    dist = np.sqrt(((lab - np.median(lab, axis=0)) ** 2).sum(axis=1))
    return float((dist < 18).mean()) > 0.85


def _largest_component(binary):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 1:
        return binary
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == best).astype(np.uint8)


def _segment_subject(frames_bgr):
    """Cut the wolf out of an ordinary video (natural background) with MediaPipe's interactive
    segmenter (models/magic_touch.tflite). The first frame is seeded at WOLF_CLIP_SEED; every next
    frame is seeded at the deepest point inside the previous frame's mask, so the wolf is tracked."""
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.components.containers.keypoint import NormalizedKeypoint

    roi_cls = vision.InteractiveSegmenterLegacyRegionOfInterest
    options = vision.InteractiveSegmenterLegacyOptions(base_options=BaseOptions(model_asset_path=C.CUTOUT_MODEL))
    segmenter = vision.InteractiveSegmenterLegacy.create_from_options(options)
    seed = C.WOLF_CLIP_SEED
    out = []
    try:
        for n, bgr in enumerate(frames_bgr):
            h, w = bgr.shape[:2]
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            roi = roi_cls(format=roi_cls.Format.KEYPOINT, keypoint=NormalizedKeypoint(x=seed[0], y=seed[1]))
            m = segmenter.segment(image, roi).confidence_masks[0].numpy_view().astype(np.float32)
            if m.ndim == 3:
                m = m[..., 0]
            if m.shape != (h, w):
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
            core = _largest_component((m > 0.5).astype(np.uint8))
            if core.any():
                dist = cv2.distanceTransform(core, cv2.DIST_L2, 5)
                y, x = np.unravel_index(int(np.argmax(dist)), dist.shape)
                seed = (x / w, y / h)                           # track the wolf into the next frame
            keep = cv2.dilate(core, np.ones((15, 15), np.uint8)).astype(np.float32)
            alpha = np.clip((m - 0.3) / 0.4, 0, 1) * keep
            alpha = cv2.GaussianBlur(alpha, (0, 0), 0.7)
            out.append(np.dstack([bgr, (alpha * 255).astype(np.uint8)]))
            if n % 50 == 0:
                print("  cutting out the wolf: frame %d / %d" % (n + 1, len(frames_bgr)))
    finally:
        segmenter.close()
    return out


def _read_alpha_animation(path):
    from PIL import Image
    img = Image.open(path)
    frames, durations = [], []
    n = getattr(img, "n_frames", 1)
    step = max(1, int(math.ceil(n / CLIP_MAX_FRAMES)))
    for i in range(0, n, step):
        img.seek(i)
        rgba = np.asarray(img.convert("RGBA"))
        bgra = rgba[..., [2, 1, 0, 3]].copy()
        if bgra[..., 3].min() == 255:                            # no transparency: key the background
            bgr, a = _key_background(np.ascontiguousarray(bgra[..., :3]))
            bgra = np.dstack([bgr, a])
        frames.append(bgra)
        durations.append(max(0.02, img.info.get("duration", 80) / 1000.0) * step)
    return frames, durations


def _read_video(path):
    """Raw BGR frames + durations."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return [], []
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(math.ceil(n / CLIP_MAX_FRAMES))) if n > 0 else 1
    frames, i = [], 0
    while len(frames) < CLIP_MAX_FRAMES:
        ok, bgr = cap.read()
        if not ok or bgr is None:
            break
        if i % step == 0:
            frames.append(bgr)
        i += 1
    cap.release()
    return frames, [step / fps] * len(frames)


def _crop_and_limit(frames):
    """Crop all frames to the union of their visible area and limit the height (memory).
    Also returns which edges of the source frame the subject touches (it is cut off there)."""
    union = np.zeros(frames[0].shape[:2], np.uint8)
    for f in frames:
        np.maximum(union, (f[..., 3] > 20).astype(np.uint8), out=union)
    ys, xs = np.nonzero(union)
    if len(ys) == 0:
        return frames, ()
    H, W = union.shape
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    m = 3
    touches = [e for e, hit in (("left", x0 <= m), ("right", x1 >= W - m), ("top", y0 <= m),
                                ("bottom", y1 >= H - m)) if hit]
    max_h = clip_max_height()
    out = []
    for f in frames:
        f = f[y0:y1, x0:x1]
        if f.shape[0] > max_h:
            s = max_h / f.shape[0]
            f = cv2.resize(f, (max(1, int(f.shape[1] * s)), max_h), interpolation=cv2.INTER_AREA)
        f = np.ascontiguousarray(f)
        far = cv2.dilate((f[..., 3] > 0).astype(np.uint8), np.ones((7, 7), np.uint8)) == 0
        f[far, :3] = 0                                          # empty background compresses well
        out.append(f)
    return out, touches


def find_clip_file(folder):
    if not os.path.isdir(folder):
        return None
    for name in sorted(os.listdir(folder)):
        if name.lower().startswith("song"):                    # the background song, not a clip
            continue
        if name.lower().endswith(CLIP_EXTS_ALPHA + CLIP_EXTS_VIDEO):
            return os.path.join(folder, name)
    return None


def _cache_key(path):
    st = os.stat(path)
    return "%d|%d|%d|%d|%s|%s" % (CUTOUT_VERSION, st.st_size, int(st.st_mtime), clip_max_height(),
                                  tuple(C.WOLF_CLIP_SEED), C.WOLF_CLIP_CUTOUT)


def _load_cache(path):
    cache = path + ".cutout.npz"
    if not os.path.isfile(cache):
        return None
    try:
        data = np.load(cache, allow_pickle=False)
        if str(data["key"]) != _cache_key(path):
            return None
        frames = list(data["frames"])
        return WolfClip(frames, data["durations"], path, [str(t) for t in data["touches"]])
    except Exception:
        return None


def _save_cache(path, clip, durations):
    try:
        np.savez_compressed(path + ".cutout.npz", key=_cache_key(path), frames=np.stack(clip.frames),
                 durations=np.asarray(durations), touches=np.array(sorted(clip.touches), dtype=str))
    except Exception as e:
        print("WARNING: could not save the wolf cut-out cache (%s)" % e)


def load_wolf_clip(folder):
    path = find_clip_file(folder)
    if path is None:
        return None
    try:
        if path.lower().endswith(CLIP_EXTS_ALPHA):
            frames, durations = _read_alpha_animation(path)
            if not frames:
                return None
            frames, touches = _crop_and_limit(frames)
            return WolfClip(frames, durations, path, touches)
        clip = _load_cache(path)
        if clip is not None:
            return clip
        raw, durations = _read_video(path)
        if not raw:
            print("WARNING: could not read wolf clip %s" % path)
            return None
        if _is_plain_background(raw[0]) or not C.WOLF_CLIP_CUTOUT:
            frames = [np.dstack(_key_background(f)) for f in raw]
        elif os.path.isfile(C.CUTOUT_MODEL):
            print("Wolf clip has a natural background: cutting the wolf out (one time, cached)...")
            frames = _segment_subject(raw)
        else:
            print("WARNING: the wolf clip has a natural background and %s is missing; "
                  "run scripts\\setup.bat to download it." % C.CUTOUT_MODEL)
            return None
        del raw
        frames, touches = _crop_and_limit(frames)
        clip = WolfClip(frames, durations, path, touches)
        _save_cache(path, clip, durations)
        return clip
    except Exception as e:
        print("WARNING: could not load wolf clip %s (%s)" % (path, e))
        return None


# =============================================================================
# Procedural wolf
# =============================================================================
class ProceduralWolf:
    """Sitting grey wolf in side view, facing left. render(pose) -> BGRA image (size x size).

    pose: head  0 = looking ahead, 1 = head thrown back (howl)
          jaw   0 = closed, 1 = open
          breath 0-1 chest expansion
    """
    SS = 2   # supersampling for smooth edges

    def __init__(self, size):
        self.size = int(size)
        n = self.size * self.SS
        rng = np.random.default_rng(7)
        # fur texture: noise stretched along the fur direction (downwards / backwards)
        noise = rng.normal(0, 1, (self.size, self.size)).astype(np.float32)
        streaks = cv2.GaussianBlur(noise, (0, 0), sigmaX=0.6, sigmaY=3.0)
        streaks /= max(1e-6, float(np.abs(streaks).max()))
        coarse = cv2.GaussianBlur(rng.normal(0, 1, (self.size, self.size)).astype(np.float32), (0, 0), self.size * 0.03)
        coarse /= max(1e-6, float(np.abs(coarse).max()))
        self.fur = 1.0 + streaks * 0.22 + coarse * 0.12
        # jagged-edge noise for the fur outline
        edge = cv2.GaussianBlur(rng.normal(0, 1, (n, n)).astype(np.float32), (0, 0), 1.2 * self.SS)
        self.edge_noise = edge / max(1e-6, float(np.abs(edge).max()))
        yy = np.linspace(0, 1, self.size, dtype=np.float32)[:, None]
        self.vertical = np.repeat(yy, self.size, axis=1)

    # --- geometry helpers (unit coordinates, 0-1 in the image) ---
    def _p(self, x, y):
        s = self.size * self.SS
        return int(round(x * s)), int(round(y * s))

    def _poly(self, img, pts, value=255):
        cv2.fillPoly(img, [np.array([self._p(x, y) for x, y in pts], np.int32)], value, cv2.LINE_AA)

    def _ellipse(self, img, c, axes, angle=0, value=255):
        s = self.size * self.SS
        cv2.ellipse(img, self._p(*c), (max(1, int(axes[0] * s)), max(1, int(axes[1] * s))), angle, 0, 360,
                    value, -1, cv2.LINE_AA)

    def _limb(self, img, pts, widths, value=255):
        """Tapered thick stroke through pts."""
        s = self.size * self.SS
        for (x0, y0), (x1, y1), w0, w1 in zip(pts[:-1], pts[1:], widths[:-1], widths[1:]):
            steps = 8
            for k in range(steps + 1):
                f = k / steps
                cv2.circle(img, self._p(x0 + (x1 - x0) * f, y0 + (y1 - y0) * f),
                           max(1, int((w0 + (w1 - w0) * f) * s / 2)), value, -1, cv2.LINE_AA)

    def render(self, head=0.0, jaw=0.0, breath=0.0):
        n = self.size * self.SS
        body = np.zeros((n, n), np.uint8)       # whole silhouette
        light = np.zeros((n, n), np.uint8)      # pale fur areas (chest, cheeks, muzzle underside)
        dark = np.zeros((n, n), np.uint8)       # darker saddle on the back
        b = breath * 0.012

        # ---- body (sitting) ----
        self._ellipse(body, (0.66, 0.79), (0.21 + b * 0.5, 0.19))                    # haunch
        self._ellipse(body, (0.53, 0.56), (0.15 + b, 0.29), angle=-28)               # torso
        self._ellipse(body, (0.39, 0.46), (0.115 + b, 0.17), angle=-10)              # chest
        self._limb(body, [(0.39, 0.55), (0.375, 0.80), (0.37, 0.965)], [0.075, 0.055, 0.05])   # front legs
        self._limb(body, [(0.46, 0.60), (0.455, 0.82), (0.45, 0.965)], [0.07, 0.05, 0.045])
        self._ellipse(body, (0.355, 0.975), (0.045, 0.022))                          # paws
        self._ellipse(body, (0.435, 0.975), (0.042, 0.02))
        self._ellipse(body, (0.58, 0.975), (0.11, 0.028))                            # hind paw
        tail = [(0.82, 0.84), (0.93, 0.92), (0.88, 0.985), (0.72, 0.995), (0.6, 0.99)]
        self._limb(body, tail, [0.09, 0.08, 0.07, 0.05, 0.02])                       # bushy tail
        self._ellipse(dark, (0.6, 0.5), (0.12, 0.3), angle=-32)                      # saddle
        self._ellipse(light, (0.37, 0.5), (0.08, 0.15), angle=-8)                    # chest ruff

        # ---- head + neck: rotate around the back of the skull ----
        angle = math.radians(18 + 50 * head)                  # snout elevation above horizontal
        c, s = math.cos(angle), math.sin(angle)
        px, py = 0.36 + 0.02 * head, 0.215 - 0.035 * head      # skull pivot

        def head_pt(lx, ly):
            # local: snout towards -x, back of the head +x, +y down; rotated so the snout
            # (-1, 0) points up-left at `angle` above the horizontal
            return px + lx * c - ly * s, py + lx * s + ly * c

        deg = math.degrees(angle)
        neck = [head_pt(0.075, -0.04), (0.56, 0.30), (0.50, 0.40), (0.33, 0.45), head_pt(0.02, 0.09)]
        self._poly(body, neck)
        tx, ty = head_pt(0.0, 0.08)
        self._ellipse(light, ((tx + 0.35) / 2, (ty + 0.42) / 2), (0.055, 0.09), angle=deg * 0.5)   # throat ruff

        def head_poly(img, pts):
            self._poly(img, [head_pt(x, y) for x, y in pts])

        self._ellipse(body, head_pt(0.0, 0.0), (0.095, 0.08), angle=deg)             # skull
        self._ellipse(body, head_pt(0.03, 0.055), (0.085, 0.07), angle=deg)          # cheek ruff
        self._ellipse(light, head_pt(0.0, 0.06), (0.06, 0.045), angle=deg)
        head_poly(body, [(-0.03, -0.065), (-0.17, -0.045), (-0.235, -0.03), (-0.255, -0.015),
                         (-0.255, 0.0), (-0.20, 0.012), (-0.04, 0.045)])                  # upper muzzle
        # lower jaw, hinged at the back of the mouth
        ja = math.radians(28 * jaw)
        jc, js = math.cos(ja), math.sin(ja)
        hx, hy = -0.04, 0.03

        def jaw_pt(x, y):
            dx, dy = x - hx, y - hy
            return hx + dx * jc - dy * js, hy + dx * js + dy * jc

        jaw_shape = [jaw_pt(x, y) for x, y in [(-0.03, 0.02), (-0.20, 0.01), (-0.215, 0.022),
                                                (-0.19, 0.035), (-0.05, 0.065)]]
        head_poly(body, jaw_shape)
        head_poly(light, jaw_shape)
        if jaw > 0.05:                                                                   # mouth interior
            mouth = np.zeros((n, n), np.uint8)
            head_poly(mouth, [(-0.04, 0.03), (-0.20, 0.008)] + [jaw_pt(-0.19, 0.012), jaw_pt(-0.04, 0.04)])
        else:
            mouth = None
        # ears (one behind the other)
        head_poly(body, [(0.005, -0.06), (0.045, -0.175), (0.075, -0.05)])
        head_poly(body, [(0.03, -0.055), (0.085, -0.15), (0.10, -0.035)])
        head_poly(dark, [(0.02, -0.07), (0.048, -0.15), (0.065, -0.07)])

        # ---- furry outline: perturb the silhouette edge with noise ----
        soft = cv2.GaussianBlur(body, (0, 0), 1.6 * self.SS).astype(np.float32) / 255.0
        fur_edge = np.clip((soft - 0.5 + self.edge_noise * 0.22) * 4 + 0.5, 0, 1)
        alpha = np.maximum(fur_edge, body.astype(np.float32) / 255.0 * (soft > 0.75))
        alpha = cv2.resize(alpha, (self.size, self.size), interpolation=cv2.INTER_AREA)

        def small(m, blur):
            m = cv2.resize(m, (self.size, self.size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
            return cv2.GaussianBlur(m, (0, 0), blur) if blur else m

        light_m = small(light, self.size * 0.02)
        dark_m = small(dark, self.size * 0.04)

        # ---- shading: grey wolf colours, lit from the upper left, fur texture ----
        base = np.array([118, 120, 126], np.float32)        # BGR grey
        pale = np.array([200, 204, 210], np.float32)
        saddle = np.array([62, 64, 70], np.float32)
        col = base[None, None, :] * np.ones((self.size, self.size, 1), np.float32)
        col += (pale - base)[None, None, :] * light_m[..., None]
        col += (saddle - base)[None, None, :] * (dark_m * (1 - light_m))[..., None]
        shade = 1.05 - 0.35 * self.vertical                  # darker towards the ground
        col *= (shade * self.fur)[..., None]
        # rim light from the flag behind (warm red-white edge)
        inner = cv2.GaussianBlur(alpha, (0, 0), self.size * 0.012)
        rim = np.clip(alpha - inner, 0, 1) * 2.2
        col += np.array([60, 70, 150], np.float32)[None, None, :] * rim[..., None]
        if mouth is not None:
            mm = small(mouth, 0)
            col = col * (1 - mm[..., None]) + np.array([40, 30, 70], np.float32) * mm[..., None]
        # nose and eye
        nose = head_pt(-0.245, -0.012)
        eye = head_pt(-0.035, -0.022)
        ns = self.size
        cv2.circle(col, (int(nose[0] * ns), int(nose[1] * ns)), max(1, int(ns * 0.012)), (20, 20, 22), -1, cv2.LINE_AA)
        eye_open = 1.0 - 0.7 * head                          # eyes half-closed while howling
        cv2.ellipse(col, (int(eye[0] * ns), int(eye[1] * ns)),
                    (max(1, int(ns * 0.012)), max(1, int(ns * 0.007 * eye_open))),
                    deg - 10, 0, 360, (30, 120, 170), -1, cv2.LINE_AA)
        out = np.empty((self.size, self.size, 4), np.uint8)
        out[..., :3] = np.clip(col, 0, 255).astype(np.uint8)
        out[..., 3] = (alpha * 255).astype(np.uint8)
        return out


# =============================================================================
# The wolf actor used by WolfEffect
# =============================================================================
_CLIP = {"loaded": False, "clip": None}


def get_clip():
    if not _CLIP["loaded"]:
        _CLIP["loaded"] = True
        _CLIP["clip"] = load_wolf_clip(C.WOLF_CLIP_DIR)
        if _CLIP["clip"] is not None:
            print("Wolf clip loaded: %s (%d frames)" % (_CLIP["clip"].path, len(_CLIP["clip"].frames)))
    return _CLIP["clip"]


class HowlingWolf:
    """Emerges from behind the person, raises its head and howls; draw(canvas, t, vis)."""

    def __init__(self, w, h):
        self.w, self.h = w, h
        self.clip = get_clip()
        self.side = 1 if C.WOLF_SIDE != "left" else -1
        self.height = h * C.WOLF_SIZE
        self.proc = None if self.clip else ProceduralWolf(int(min(self.height, 480)) // 4 * 4)
        self.cache = {}

    # timeline (seconds)
    T_EMERGE = (0.35, 1.7)
    T_HOWL = (1.9, 2.6, 5.4, 6.1)      # head up starts, fully up, starts lowering, down

    def pose(self, t):
        a, b, c, d = self.T_HOWL
        if t < a or t > d:
            head = 0.0
        elif t < b:
            head = smoothstep((t - a) / (b - a))
        elif t < c:
            head = 1.0
        else:
            head = 1.0 - smoothstep((t - c) / (d - c))
        jaw = clamp((head - 0.35) / 0.65) * (0.85 + 0.15 * math.sin(t * 23))   # vibrating howl
        breath = 0.5 + 0.5 * math.sin(t * 2.4) if head < 0.5 else 1.0 - 0.6 * clamp((t - b) / (c - b))
        return head, jaw, breath

    def howling(self, t):
        return self.T_HOWL[1] - 0.3 < t < self.T_HOWL[2] + 0.2

    def draw(self, canvas, t, vis):
        e = ease_out_cubic((t - self.T_EMERGE[0]) / (self.T_EMERGE[1] - self.T_EMERGE[0]))
        if e <= 0 or vis <= 0.01:
            return None
        if self.clip is not None:
            img = self.clip.frame(t, self.height)
            content_x = 0.5
        else:
            img = self._procedural(t)
            content_x = 0.62                                  # body centre inside the square image
        if self.side < 0:
            img = cv2.flip(img, 1)
            content_x = 1 - content_x
        ih, iw = img.shape[:2]
        # start hidden behind the person (centre, lower), slide out to the side and rise
        x_end = self.w * (0.5 + self.side * C.WOLF_OFFSET)
        cx = self.w * 0.5 + (x_end - self.w * 0.5) * e
        bottom = self.h * 1.04 + self.h * 0.35 * (1 - e)
        x0 = int(round(cx - iw * content_x))
        x0 = min(x0, int(self.w - iw * (0.97 if self.side > 0 else 1.0)))   # keep the tail in frame
        x0 = max(x0, int(-iw * (0.03 if self.side < 0 else 0.0)))
        if self.clip is not None:
            edge = "right" if self.side > 0 else "left"         # side the clip is cut off at (after flip)
            if self.clip_edge(edge):
                x0 = self.w - iw if self.side > 0 else 0        # hide the cut against the frame edge
            if "bottom" in self.clip.touches:
                bottom = self.h + self.h * 0.35 * (1 - e)
        y0 = int(round(bottom - ih))
        self._paste(canvas, img, x0, y0, vis * clamp(e * 1.6))
        # mouth position (for the howl rings), in frame coordinates
        if self.clip is None:
            head, _, _ = self.pose(t)
            ang = math.radians(18 + 50 * head)
            mx = 0.36 + 0.02 * head + (-0.25) * math.cos(ang)
            my = 0.215 - 0.035 * head - 0.25 * math.sin(ang)
            if self.side < 0:
                mx = 1 - mx
            return x0 + mx * iw, y0 + my * ih, ang
        return None                                             # real footage: no drawn sound waves

    def clip_edge(self, edge):
        """Is the clip cut off at this output side (taking the left/right flip into account)?"""
        src = {"right": "left", "left": "right"}[edge] if self.side < 0 else edge
        return src in self.clip.touches

    def _procedural(self, t):
        head, jaw, breath = self.pose(t)
        key = (round(head * 24), round(jaw * 5), round(breath * 3) if head < 0.05 else 0)
        img = self.cache.get(key)
        if img is None:
            img = self.proc.render(key[0] / 24, key[1] / 5, key[2] / 3)
            size = int(self.height) // 4 * 4
            if img.shape[0] != size:
                img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
            if len(self.cache) > 96:
                self.cache.clear()
            self.cache[key] = img
        return img

    @staticmethod
    def _paste(canvas, img, x0, y0, opacity):
        h, w = canvas.shape[:2]
        ih, iw = img.shape[:2]
        fx0, fy0, fx1, fy1 = max(0, x0), max(0, y0), min(w, x0 + iw), min(h, y0 + ih)
        if fx1 <= fx0 or fy1 <= fy0:
            return
        sub = img[fy0 - y0:fy1 - y0, fx0 - x0:fx1 - x0]
        a = sub[..., 3].astype(np.float32) * (opacity / 255.0)
        region = canvas[fy0:fy1, fx0:fx1]
        cv2.blendLinear(region, np.ascontiguousarray(sub[..., :3]), 1.0 - a, a, dst=region)

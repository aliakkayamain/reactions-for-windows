"""Waving Turkish flag: drawn from the official proportions, cloth motion generated in real time."""
import math

import cv2
import numpy as np

FLAG_RED = (23, 10, 227)   # #E30A17 (BGR)
WHITE = (255, 255, 255)


def draw_crescent_and_star(img, cx, cy, g, scale=1):
    """Proportions from the Turkish Flag Law, g = flag height:
    outer circle diameter 0.5g, inner circle diameter 0.4g, centers g/16 apart, star diameter 0.25g."""
    def pt(x, y):
        return (int(round(x * scale)), int(round(y * scale)))

    r_outer, r_inner = 0.25 * g, 0.2 * g
    inner_x = cx + g / 16.0
    cv2.circle(img, pt(cx, cy), int(r_outer * scale), WHITE, -1, cv2.LINE_AA)
    cv2.circle(img, pt(inner_x, cy), int(r_inner * scale), FLAG_RED, -1, cv2.LINE_AA)
    # star: center g/3 from the inner circle's left edge, one point facing the crescent
    star_x = (inner_x - r_inner) + g / 3.0
    big, small = 0.125 * g, 0.125 * g * 0.381966
    points = []
    for i in range(10):
        a = math.pi + i * math.pi / 5
        r = big if i % 2 == 0 else small
        points.append(pt(star_x + r * math.cos(a), cy + r * math.sin(a)))
    cv2.fillPoly(img, [np.array(points, np.int32)], WHITE, cv2.LINE_AA)


class WavingFlag:
    """Full-frame waving Turkish flag. The crescent sits on the left so it is not hidden
    behind a person sitting in the middle of the frame."""

    def __init__(self, w, h, work_width=480, crescent_x=0.2):
        self.w, self.h = w, h
        self.cw = work_width
        self.ch = int(round(h * work_width / w))
        ss = 2                                                  # draw at 2x for smooth edges
        base = np.empty((self.ch * ss, self.cw * ss, 3), np.uint8)
        base[:] = FLAG_RED
        draw_crescent_and_star(base, self.cw * crescent_x, self.ch * 0.48, self.ch * 0.96, scale=ss)
        self.base = cv2.resize(base, (self.cw, self.ch), interpolation=cv2.INTER_AREA)
        weave = 1.0 + 0.025 * np.sin(np.arange(self.ch, dtype=np.float32) * 2.1)[:, None]   # fabric texture
        self.base = np.clip(self.base * weave[..., None], 0, 255).astype(np.uint8)
        yy, xx = np.mgrid[0:self.ch, 0:self.cw].astype(np.float32)
        self.xx, self.yy = xx, yy
        self.xn, self.yn = xx / self.cw, yy / self.ch           # 0 = hoist side, 1 = fly end
        self.out = np.empty((h, w, 3), np.uint8)

    def frame(self, t):
        """Flag image at time t (w x h, BGR)."""
        xn, yn = self.xn, self.yn
        freedom = 0.35 + 0.65 * xn                              # hoist side moves less
        f1 = xn * 9.0 + yn * 1.5 - t * 4.2
        f2 = xn * 15.0 - yn * 3.0 - t * 6.3
        wave = np.sin(f1) * 0.65 + np.sin(f2) * 0.35
        slope = np.cos(f1) * 0.65 + np.cos(f2) * 0.35 * 1.6
        map_x = self.xx + (np.cos(f1) * 0.008 * self.cw) * freedom
        map_y = self.yy + (wave * 0.035 * self.ch) * freedom
        warped = cv2.remap(self.base, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        shade = 1.0 + slope * 0.24 * freedom                    # light and shadow in the folds
        lit = np.clip(warped * shade[..., None], 0, 255).astype(np.uint8)
        return cv2.resize(lit, (self.w, self.h), dst=self.out, interpolation=cv2.INTER_LINEAR)

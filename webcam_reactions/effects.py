"""
All effects. Every effect implements:
  update(dt)                -> advance time
  draw_behind(frame, mask)  -> (optional) draw BEHIND the person (mask: 1 = person)
  draw(frame)               -> draw in front of everything
"""
import math
import random

import cv2
import numpy as np

from . import config as C
from .graphics import (EMOJI, LightLayer, blend, clamp, darken, ease_out_back, ease_out_cubic,
                       fade_in_out, hsv_color)

# Gesture / effect identifiers
THUMBS_UP, TWO_THUMBS_UP = "thumbs_up", "two_thumbs_up"
THUMBS_DOWN, TWO_THUMBS_DOWN = "thumbs_down", "two_thumbs_down"
VICTORY, TWO_VICTORY = "victory", "two_victory"
HEART, WOLF, LASERS, PARTY = "heart", "wolf", "lasers", "party"

LABELS = {
    THUMBS_UP: "Thumbs up", TWO_THUMBS_UP: "Fireworks",
    THUMBS_DOWN: "Thumbs down", TWO_THUMBS_DOWN: "Rain",
    VICTORY: "Balloons", TWO_VICTORY: "Confetti",
    HEART: "Hearts", WOLF: "Grey Wolf", LASERS: "Lasers", PARTY: "Party mode",
}

# Keyboard triggers (like the reactions menu on macOS)
KEY_EFFECTS = {
    ord("1"): THUMBS_UP, ord("2"): THUMBS_DOWN, ord("3"): HEART, ord("4"): VICTORY,
    ord("5"): TWO_THUMBS_DOWN, ord("6"): TWO_VICTORY, ord("7"): TWO_THUMBS_UP, ord("8"): LASERS,
    ord("9"): WOLF,
}

PALETTE = [(60, 60, 255), (60, 170, 255), (50, 230, 255), (90, 220, 80),
           (255, 180, 50), (255, 90, 150), (220, 80, 255)]


class Effect:
    duration = 3.0
    uses_background = False    # True: draw_behind() paints behind the person (needs the person mask)

    def __init__(self, w, h, center=None):
        self.w, self.h = w, h
        self.center = center or (w // 2, h // 2)
        self.t = 0.0

    @property
    def finished(self):
        return self.t >= self.duration

    def update(self, dt):
        self.t += dt

    def draw_behind(self, frame, mask):
        pass

    def draw(self, frame):
        raise NotImplementedError

    def close(self):
        """Called when the effect is removed (stop sounds etc.)."""

    def stop(self):
        """Fade out an open-ended effect."""
        self.duration = min(self.duration, self.t + 0.8)


# =============================================================================
# Thumbs up / down
# =============================================================================
class ThumbsEffect(Effect):
    duration = 2.4

    def __init__(self, w, h, center=None, up=True):
        super().__init__(w, h, center)
        self.up = up
        self.name = "thumbs_up" if up else "thumbs_down"
        self.light = LightLayer(w, h, 0.5)
        self.color = (80, 200, 255) if up else (255, 170, 90)
        self.minis = [{
            "dx": random.uniform(-0.36, 0.36) * w, "dy": random.uniform(-0.05, 0.15) * h,
            "delay": random.uniform(0.15, 0.9), "speed": random.uniform(0.18, 0.32) * h,
            "size": random.uniform(0.07, 0.11) * h, "phase": random.uniform(0, 6.28),
            "angle": random.uniform(-18, 18),
        } for _ in range(10)]

    def draw(self, frame):
        direction = -1 if self.up else 1
        cx, cy0 = self.w / 2, self.h * 0.5
        # expanding light rings
        self.light.clear()
        for delay in (0.05, 0.25):
            o = (self.t - delay) / 0.9
            if 0 < o < 1:
                r = self.h * (0.12 + 0.45 * ease_out_cubic(o))
                c = tuple(int(v * (1 - o)) for v in self.color)
                cv2.circle(self.light.canvas, self.light.p(cx, cy0), int(r * 0.5), c, 3, cv2.LINE_AA)
        self.light.apply(frame, glow=0.8, bloom=8)
        for m in self.minis:
            age = self.t - m["delay"]
            o = age / 1.3
            if not 0 < o < 1:
                continue
            x = cx + m["dx"] + math.sin(age * 4 + m["phase"]) * self.h * 0.015
            y = cy0 + m["dy"] + direction * m["speed"] * age
            alpha = clamp(o / 0.15) * (1.0 if o < 0.6 else (1 - o) / 0.4)
            EMOJI.draw(frame, self.name, x, y, m["size"] * ease_out_back(o / 0.3), alpha, angle=m["angle"])
        o = clamp(self.t / self.duration)
        size = self.h * 0.44 * ease_out_back(o / 0.33) * (1 + 0.1 * max(0.0, o - 0.5))
        alpha = 1.0 if o < 0.65 else max(0.0, 1 - (o - 0.65) / 0.35)
        cy = cy0 + direction * self.h * 0.06 * o + math.sin(self.t * 6) * self.h * 0.008
        wobble = math.sin(self.t * 9) * 8 * max(0.0, 1 - self.t / 0.8)
        EMOJI.draw_shadow(frame, self.name, cx, cy, size, alpha)
        EMOJI.draw(frame, self.name, cx, cy, size, alpha, angle=wobble)


# =============================================================================
# Hearts
# =============================================================================
class HeartsEffect(Effect):
    duration = 4.2
    STATIC = ("red_heart", "red_heart", "pink_heart", "sparkling_heart", "two_hearts")

    def __init__(self, w, h, center=None):
        super().__init__(w, h, center)
        names = [n for n in self.STATIC if EMOJI.has(n)] or ["red_heart"]
        animated = [n for n in ("anim_sparkling_heart", "anim_two_hearts") if EMOJI.has(n)]
        self.hearts = []
        for _ in range(46):
            self.hearts.append({
                "born": random.uniform(0, 1.9), "angle": random.uniform(0, 2 * math.pi),
                "speed": random.uniform(0.25, 0.7) * h, "size": random.uniform(0.04, 0.1) * h,
                "life": random.uniform(1.6, 2.2), "spin": random.uniform(-25, 25),
                "name": random.choice(animated) if animated and random.random() < 0.18 else random.choice(names),
            })
        self.hearts.sort(key=lambda k: k["size"])
        self.light = LightLayer(w, h, 0.25)

    def draw(self, frame):
        mx, my = self.center
        o = self.t / 0.8
        if o < 1:                                               # warm pink flash at the start
            self.light.clear()
            cv2.circle(self.light.canvas, self.light.p(mx, my), int(self.h * 0.25 * 0.25),
                       tuple(int(c * (1 - o)) for c in (130, 60, 255)), -1, cv2.LINE_AA)
            self.light.apply(frame, glow=1.2, bloom=40, gain=0.4)
        for k in self.hearts:
            age = self.t - k["born"]
            if age < 0 or age > k["life"]:
                continue
            o = age / k["life"]
            dist = k["speed"] * k["life"] * (1 - (1 - o) ** 2)
            x = mx + math.cos(k["angle"]) * dist
            y = my + math.sin(k["angle"]) * dist - self.h * 0.12 * o
            scale = ease_out_back(o / 0.25) * (1.0 if o < 0.75 else max(0.0, (1 - o) / 0.25))
            EMOJI.draw(frame, k["name"], x, y, k["size"] * 2.2 * scale, time=age, angle=k["spin"] * o)


# =============================================================================
# Balloons
# =============================================================================
class BalloonsEffect(Effect):
    duration = 5.0

    def __init__(self, w, h, center=None):
        super().__init__(w, h, center)
        colors = [n for n in ("balloon_%d" % i for i in range(6)) if EMOJI.has(n)] or ["balloon"]
        self.balloons = []
        for _ in range(16):
            r = random.uniform(0.055, 0.09) * h
            self.balloons.append({
                "x": random.uniform(0.06, 0.94) * w, "y": h + r * 1.5 + random.uniform(0, 0.35) * h,
                "r": r, "speed": random.uniform(0.36, 0.52) * h, "phase": random.uniform(0, 6.28),
                "name": random.choice(colors),
            })
        self.balloons.sort(key=lambda b: b["r"])

    def update(self, dt):
        super().update(dt)
        for b in self.balloons:
            b["y"] -= b["speed"] * dt

    def draw(self, frame):
        for b in self.balloons:
            r = b["r"]
            sway = math.sin(self.t * 2 + b["phase"])
            EMOJI.draw(frame, b["name"], b["x"] + sway * r * 0.5, b["y"] + r, r * 3.2, angle=sway * 8)


# =============================================================================
# Rain
# =============================================================================
class RainEffect(Effect):
    duration = 4.8

    def __init__(self, w, h, center=None):
        super().__init__(w, h, center)
        self.clouds = [{
            "x": w * (0.12 + 0.76 * i / 4) + random.uniform(-0.03, 0.03) * w,
            "y": h * random.uniform(0.08, 0.15), "size": h * random.uniform(0.26, 0.34),
            "phase": random.uniform(0, 6.28),
        } for i in range(5)]
        n = 160
        self.x = np.random.uniform(0.0, 1.0, n) * w
        self.y = np.random.uniform(0.15, 1.0, n) * h
        self.v = np.random.uniform(1.1, 1.6, n) * h
        self.length = np.random.uniform(0.03, 0.06, n) * h
        self.drops = [{"x": random.uniform(0.1, 0.9) * w, "y": random.uniform(0.2, 1.0) * h,
                       "v": random.uniform(0.5, 0.8) * h, "size": random.uniform(0.035, 0.055) * h}
                      for _ in range(18)]

    def update(self, dt):
        super().update(dt)
        self.y += self.v * dt
        for d in self.drops:
            d["y"] += d["v"] * dt
        if self.t < self.duration - 1.0:                       # stop respawning near the end
            self.y[self.y > self.h] = self.h * 0.18
            for d in self.drops:
                if d["y"] > self.h + 30:
                    d["y"] = self.h * 0.2

    def draw(self, frame):
        vis = fade_in_out(self.t, self.duration, 0.5, 0.6)
        darken(frame, 0.28 * vis, tint=(70, 40, 20))          # gloomy blue tone
        streaks = np.zeros_like(frame)
        for x, y, ln in zip(self.x, self.y, self.length):
            if y >= self.h * 0.18:
                cv2.line(streaks, (int(x), int(y)), (int(x - ln * 0.15), int(y - ln)), (255, 220, 190), 1, cv2.LINE_AA)
        cv2.addWeighted(frame, 1.0, streaks, 0.55 * vis, 0, dst=frame)
        for d in self.drops:
            if d["y"] > self.h * 0.2:
                EMOJI.draw(frame, "droplet", d["x"], d["y"], d["size"], 0.9 * vis)
        shift = -(1 - vis) * self.h * 0.35
        for c in self.clouds:
            y = c["y"] + shift + math.sin(self.t * 2 + c["phase"]) * self.h * 0.01
            EMOJI.draw(frame, "cloud_with_rain", c["x"], y, c["size"])


# =============================================================================
# Confetti
# =============================================================================
class ConfettiEffect(Effect):
    duration = 4.6

    def __init__(self, w, h, center=None):
        super().__init__(w, h, center)
        n = 260
        half = n // 2
        # first half bursts out of the party poppers in the corners, the rest falls from the top
        self.x = np.empty(n, np.float32)
        self.y = np.empty(n, np.float32)
        self.vx = np.empty(n, np.float32)
        self.vy = np.empty(n, np.float32)
        left = np.arange(half) % 2 == 0
        self.x[:half] = np.where(left, 0.12 * w, 0.88 * w)
        self.y[:half] = 0.8 * h
        angle = np.random.uniform(-1.25, -0.55, half)
        speed = np.random.uniform(0.9, 1.6, half) * h
        self.vx[:half] = np.where(left, 1, -1) * np.cos(angle) * speed * 0.65
        self.vy[:half] = np.sin(angle) * speed
        self.x[half:] = np.random.uniform(0, w, n - half)
        self.y[half:] = np.random.uniform(-0.9 * h, -0.05 * h, n - half)
        self.vx[half:] = np.random.uniform(-0.04, 0.04, n - half) * w
        self.vy[half:] = np.random.uniform(0.4, 0.7, n - half) * h
        self.angle = np.random.uniform(0, 6.28, n)
        self.spin = np.random.uniform(-7, 7, n)
        self.phase = np.random.uniform(0, 6.28, n)
        self.width = np.random.uniform(0.011, 0.02, n) * h
        self.height = np.random.uniform(0.006, 0.011, n) * h
        self.color = np.array([random.choice(PALETTE) for _ in range(n)], np.float32)

    def update(self, dt):
        super().update(dt)
        self.vy += 0.9 * self.h * dt                            # gravity
        self.vy = np.minimum(self.vy, 0.45 * self.h)            # paper flutters down slowly
        self.vx *= (1 - 1.5 * dt)
        self.x += (self.vx + np.sin(self.t * 3 + self.phase) * self.w * 0.03) * dt
        self.y += self.vy * dt
        self.angle += self.spin * dt

    def draw(self, frame):
        for i in range(len(self.x)):
            y = self.y[i]
            if y < -20 or y > self.h + 20:
                continue
            flip = math.cos(self.t * 6 + self.phase[i])
            shade = 0.55 + 0.45 * abs(flip)                     # catches the light while tumbling
            color = tuple(int(c * shade) for c in self.color[i])
            hw, hh = self.width[i], self.height[i] * (0.2 + 0.8 * abs(flip))
            ca, sa = math.cos(self.angle[i]), math.sin(self.angle[i])
            pts = [[self.x[i] + kx * ca - ky * sa, y + kx * sa + ky * ca]
                   for kx, ky in ((-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh))]
            cv2.fillConvexPoly(frame, np.array(pts, np.int32), color, cv2.LINE_AA)
        size = self.h * 0.3 * ease_out_back(self.t / 0.4)
        alpha = clamp((self.duration - self.t) / 0.6)
        shake = math.sin(self.t * 30) * self.h * 0.006 * max(0.0, 1 - self.t / 0.6)
        EMOJI.draw(frame, "party_popper", self.w * 0.11 + shake, self.h * 0.82, size, alpha)
        EMOJI.draw(frame, "party_popper", self.w * 0.89 - shake, self.h * 0.82, size, alpha, mirror=True)


# =============================================================================
# Fireworks (with bloom)
# =============================================================================
class FireworksEffect(Effect):
    duration = 5.0

    def __init__(self, w, h, center=None):
        super().__init__(w, h, center)
        self.light = LightLayer(w, h, 0.5)
        self.rockets = [{
            "start": i * 0.38 + random.uniform(0, 0.15),
            "x": random.uniform(0.1, 0.9) * w, "y": random.uniform(0.12, 0.45) * h,
            "hue": random.random(), "sparks": None,
        } for i in range(9)]

    def update(self, dt):
        super().update(dt)
        for r in self.rockets:
            s = r["sparks"]
            if s is not None:
                s["v"][:, 1] += 240 * dt
                s["v"] *= (1 - 1.5 * dt)
                s["pos"] += s["v"] * dt
                s["age"] += dt

    def draw(self, frame):
        darken(frame, 0.38 * fade_in_out(self.t, self.duration, 0.4, 0.8), tint=(40, 10, 10))
        L = self.light
        L.clear(decay=0.78)                                     # leaves fading trails
        for r in self.rockets:
            elapsed = self.t - r["start"]
            if elapsed < 0:
                continue
            if elapsed < 0.55:                                  # rocket rising
                o = elapsed / 0.55
                y = self.h + (r["y"] - self.h) * (1 - (1 - o) ** 2)
                cv2.circle(L.canvas, L.p(r["x"], y), 2, (200, 235, 255), -1, cv2.LINE_AA)
                continue
            if r["sparks"] is None:                             # explosion
                n = 110
                angle = np.random.uniform(0, 2 * math.pi, n)
                speed = np.random.uniform(0.3, 1.0, n) ** 0.5 * self.h * 0.55
                r["sparks"] = {
                    "pos": np.tile(np.array([r["x"], r["y"]], np.float32), (n, 1)),
                    "v": np.stack([np.cos(angle) * speed, np.sin(angle) * speed], 1).astype(np.float32),
                    "age": 0.0, "life": np.random.uniform(1.0, 1.6, n),
                    "color": [hsv_color((r["hue"] + random.uniform(-0.06, 0.06)) % 1, 0.75, 1.0) for _ in range(n)],
                }
            s = r["sparks"]
            for (x, y), life, (b, g, rr) in zip(s["pos"], s["life"], s["color"]):
                left = 1 - s["age"] / life
                if left <= 0:
                    continue
                k = left * (0.6 + 0.4 * random.random() if left < 0.4 else 1.0)   # twinkle while dying
                cv2.circle(L.canvas, L.p(x, y), 1, (int(b * k), int(g * k), int(rr * k)), -1, cv2.LINE_AA)
        L.apply(frame, glow=1.0, bloom=6)
        for r in self.rockets:
            s = r["sparks"]
            if s is not None and s["age"] < 0.8:
                o = s["age"] / 0.8
                EMOJI.draw(frame, "sparkles", r["x"], r["y"], self.h * 0.15 * ease_out_back(o / 0.4), 1 - o ** 2)


# =============================================================================
# Lasers (two-hand rock sign, as on macOS)
# =============================================================================
class LasersEffect(Effect):
    duration = 5.5

    def __init__(self, w, h, center=None):
        super().__init__(w, h, center)
        self.light = LightLayer(w, h, 0.5)
        self.sources = []
        for i, (x, y, base) in enumerate([(0.15, -0.02, 70), (0.5, -0.03, 90), (0.85, -0.02, 110),
                                          (-0.02, 1.0, 35), (1.02, 1.0, 145)]):
            self.sources.append({"x": x * w, "y": y * h, "base": base, "hue": i / 5.0,
                                 "speed": random.uniform(0.7, 1.3) * random.choice((-1, 1)),
                                 "phase": random.uniform(0, 6.28)})

    def draw(self, frame):
        vis = fade_in_out(self.t, self.duration, 0.3, 0.7)
        darken(frame, 0.5 * vis, tint=(30, 0, 20))
        L = self.light
        L.clear()
        length = math.hypot(self.w, self.h) * 1.2
        for s in self.sources:
            color = hsv_color((s["hue"] + self.t * 0.15) % 1, 0.85, vis)
            spread = 22 + 12 * math.sin(self.t * 0.8 + s["phase"])
            mid = s["base"] + 28 * math.sin(self.t * s["speed"] + s["phase"])
            for j in range(-3, 4):
                a = math.radians(mid + j * spread / 3)
                x1, y1 = s["x"] + math.cos(a) * length, s["y"] + math.sin(a) * length
                cv2.line(L.canvas, L.p(s["x"], s["y"]), L.p(x1, y1), color, 1, cv2.LINE_AA)
            cv2.circle(L.canvas, L.p(s["x"], s["y"]), 6, color, -1, cv2.LINE_AA)
        L.apply(frame, glow=1.4, bloom=10, gain=1.3)           # bloom doubles as haze


# =============================================================================
# Party mode: mirror ball, colored spotlights, reflected light dots
# =============================================================================
class PartyEffect(Effect):
    BPM = 124

    def __init__(self, w, h, center=None, duration=None):
        super().__init__(w, h, center)
        self.duration = duration if duration is not None else C.PARTY_DURATION
        self.light = LightLayer(w, h, 0.5)
        self.spots = [{"x": x * w, "hue": i / 4.0, "phase": random.uniform(0, 6.28),
                       "speed": random.uniform(0.6, 1.1) * random.choice((-1, 1))}
                      for i, x in enumerate((0.08, 0.36, 0.64, 0.92))]
        n = 90
        self.dot_x = np.random.uniform(0, 1, n)
        self.dot_y = np.random.uniform(0.05, 1.0, n)
        self.dot_speed = np.random.uniform(0.06, 0.12, n)
        self.dot_size = np.random.uniform(2.0, 5.0, n)
        self.dot_hue = np.random.uniform(0, 1, n)
        self.pops = []
        self.next_pop = 0.8

    def update(self, dt):
        super().update(dt)
        if self.t >= self.next_pop and self.t < self.duration - 1.5:
            left = random.random() < 0.5
            self.pops.append({"x": (random.uniform(0.06, 0.24) if left else random.uniform(0.76, 0.94)) * self.w,
                              "y": random.uniform(0.35, 0.85) * self.h, "t0": self.t,
                              "name": random.choice(("anim_partying_face", "anim_partying_face",
                                                     "anim_star_struck", "anim_fire")),
                              "size": random.uniform(0.16, 0.22) * self.h})
            self.next_pop = self.t + random.uniform(0.9, 1.6)
        self.pops = [p for p in self.pops if self.t - p["t0"] < 1.6]

    def draw(self, frame):
        vis = fade_in_out(self.t, self.duration, 0.8, 0.8)
        beat = 60.0 / self.BPM
        pulse = math.exp(-((self.t % beat) / beat) * 5)         # flashes on every beat
        darken(frame, (0.42 - 0.12 * pulse) * vis, tint=(40, 0, 25))
        L = self.light
        L.clear()
        for s in self.spots:                                    # colored spotlight cones
            a = math.radians(90 + 35 * math.sin(self.t * s["speed"] + s["phase"]))
            half = math.radians(9)
            reach = self.h * 1.3
            color = hsv_color((s["hue"] + self.t * 0.12) % 1, 0.8, (0.45 + 0.35 * pulse) * vis)
            p0 = L.p(s["x"], -10)
            p1 = L.p(s["x"] + math.cos(a - half) * reach, -10 + math.sin(a - half) * reach)
            p2 = L.p(s["x"] + math.cos(a + half) * reach, -10 + math.sin(a + half) * reach)
            cv2.fillConvexPoly(L.canvas, np.array([p0, p1, p2], np.int32), tuple(int(c * 0.35) for c in color), cv2.LINE_AA)
            cv2.circle(L.canvas, p0, 8, color, -1, cv2.LINE_AA)
        xs = ((self.dot_x + self.t * self.dot_speed) % 1.0) * self.w   # mirror-ball reflections sweep the room
        for x, y, size, hue in zip(xs, self.dot_y * self.h, self.dot_size, self.dot_hue):
            level = (0.6 + 0.4 * math.sin(self.t * 7 + hue * 30)) * vis
            color = hsv_color((hue + self.t * 0.05) % 1, 0.35, level)
            cv2.ellipse(L.canvas, L.p(x, y), (int(size * 1.6), int(size)), 0, 0, 360, color, -1, cv2.LINE_AA)
        L.apply(frame, glow=1.0, bloom=9)
        for p in self.pops:                                     # party emoji popping up at the sides
            o = (self.t - p["t0"]) / 1.6
            EMOJI.draw(frame, p["name"], p["x"], p["y"] - o * self.h * 0.05, p["size"] * ease_out_back(o / 0.25),
                       clamp((1 - o) / 0.3) * vis, time=self.t - p["t0"])
        drop = ease_out_cubic(self.t / 1.2) if self.duration - self.t > 0.8 else ease_out_cubic((self.duration - self.t) / 0.8)
        ball_y = -self.h * 0.15 + self.h * 0.3 * drop             # mirror ball hanging from the ceiling
        cv2.line(frame, (self.w // 2, 0), (self.w // 2, int(ball_y)), (190, 190, 190), 2, cv2.LINE_AA)
        size = self.h * (0.22 + 0.01 * pulse)
        EMOJI.draw(frame, "anim_mirror_ball", self.w / 2, ball_y + size * 0.45, size, time=self.t)


# =============================================================================
# Grey Wolf: waving Turkish flag behind the person, howling wolf animation
# =============================================================================
class WolfEffect(Effect):
    duration = 7.5
    uses_background = True

    def __init__(self, w, h, center=None):
        super().__init__(w, h, center)
        from .flag import WavingFlag
        from .wolf import HowlingWolf
        from .sound import get_song
        self.flag = WavingFlag(w, h)
        self.wolf = HowlingWolf(w, h)
        self.song = get_song()
        if self.song is not None:
            if self.song.duration:                              # effect lasts as long as the song segment
                self.duration = max(self.duration, self.song.duration + 0.3)
            self.song.play()

    def close(self):
        if self.song is not None:
            self.song.stop()

    def draw_behind(self, frame, mask):
        vis = fade_in_out(self.t, self.duration, 0.6, 0.8)
        if vis <= 0.01:
            return
        # background layer = waving flag + howling wolf; the person mask is applied once, at the
        # end, so the wolf is always behind you and the person cut-out is exactly the same as before
        flag = self.flag.frame(self.t)
        bg = flag.copy() if C.FLAG_OPACITY >= 0.999 else cv2.addWeighted(frame, 1 - C.FLAG_OPACITY, flag,
                                                                          C.FLAG_OPACITY, 0)
        mouth = self.wolf.draw(bg, self.t, 1.0)
        if mouth is not None and self.wolf.howling(self.t):
            self._howl_rings(bg, mouth)
        if mask is None:                                        # no person mask: translucent layer
            cv2.addWeighted(frame, 1 - 0.35 * vis, bg, 0.35 * vis, 0, dst=frame)
            return
        blend(frame, bg, (1.0 - mask) * vis)

    def _howl_rings(self, bg, mouth):
        """Faint sound-wave arcs leaving the wolf's mouth."""
        mx, my, ang = mouth
        direction = math.degrees(ang) + 180 if self.wolf.side > 0 else -math.degrees(ang)   # along the snout
        reach = int(self.h * 0.4)                               # work only in the area around the mouth
        x0, y0 = max(0, int(mx) - reach), max(0, int(my) - reach)
        x1, y1 = min(self.w, int(mx) + reach), min(self.h, int(my) + reach)
        if x1 <= x0 or y1 <= y0:
            return
        region = bg[y0:y1, x0:x1]
        layer = np.zeros_like(region)
        for k in range(4):
            o = (self.t * 0.9 + k / 4) % 1.0
            r = int(self.h * (0.04 + 0.32 * o))
            level = int(200 * (1 - o) ** 1.5)
            cv2.ellipse(layer, (int(mx) - x0, int(my) - y0), (r, r), direction, -28, 28, (level,) * 3,
                        max(2, int(self.h * 0.006)), cv2.LINE_AA)
        cv2.GaussianBlur(layer, (0, 0), 2.0, dst=layer)
        cv2.add(region, layer, dst=region)

    def draw(self, frame):
        pass                                                    # everything is drawn behind the person


# =============================================================================
def create_effect(name, w, h, center):
    factory = {
        THUMBS_UP: lambda: ThumbsEffect(w, h, center, up=True),
        THUMBS_DOWN: lambda: ThumbsEffect(w, h, center, up=False),
        TWO_THUMBS_UP: lambda: FireworksEffect(w, h, center),
        TWO_THUMBS_DOWN: lambda: RainEffect(w, h, center),
        VICTORY: lambda: BalloonsEffect(w, h, center),
        TWO_VICTORY: lambda: ConfettiEffect(w, h, center),
        HEART: lambda: HeartsEffect(w, h, center),
        LASERS: lambda: LasersEffect(w, h, center),
        WOLF: lambda: WolfEffect(w, h, center),
    }.get(name)
    return factory() if factory else None


ALL_EFFECTS = [THUMBS_UP, TWO_THUMBS_UP, THUMBS_DOWN, TWO_THUMBS_DOWN, VICTORY, TWO_VICTORY, HEART, LASERS, WOLF]

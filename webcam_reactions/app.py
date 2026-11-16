"""
Webcam Reactions - main loop.

Reads the physical webcam, detects hand gestures, draws effects and sends the result to
"OBS Virtual Camera" (Teams, Zoom, Meet...) and/or to OBS Studio as a Browser source.
Run with:  python -m webcam_reactions [--obs] [--test SECONDS]
"""
import argparse
import ctypes
import math
import os
import time

import cv2
import numpy as np

from . import config as C
from . import effects as E
from .graphics import load_emoji

SETUP_HINT = "Run scripts\\setup.bat first (see README.md)."


def show_error(message):
    """Print the error and show it in a Windows message box."""
    print("\nERROR: " + message + "\n")
    try:
        ctypes.windll.user32.MessageBoxW(0, message, "Webcam Reactions - Error", 0x10)
    except Exception:
        pass


def open_camera():
    """Open the physical webcam, or return None."""
    for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF):
        cam = cv2.VideoCapture(C.CAMERA_INDEX, backend)
        if not cam.isOpened():
            cam.release()
            continue
        cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, C.WIDTH)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, C.HEIGHT)
        cam.set(cv2.CAP_PROP_FPS, C.FPS)
        cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        for _ in range(15):                                     # the first frames can be empty
            ok, frame = cam.read()
            if ok and frame is not None and frame.size:
                return cam
            time.sleep(0.1)
        cam.release()
    return None


def fit_frame(frame):
    """Scale/crop the camera frame to the output resolution."""
    h, w = frame.shape[:2]
    if (w, h) == (C.WIDTH, C.HEIGHT):
        return frame
    scale = max(C.WIDTH / w, C.HEIGHT / h)
    sw, sh = int(math.ceil(w * scale)), int(math.ceil(h * scale))
    frame = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_LINEAR)
    x0, y0 = (sw - C.WIDTH) // 2, (sh - C.HEIGHT) // 2
    return np.ascontiguousarray(frame[y0:y0 + C.HEIGHT, x0:x0 + C.WIDTH])


def main():
    parser = argparse.ArgumentParser(prog="webcam_reactions", description="Gesture reactions for your webcam")
    parser.add_argument("--obs", action="store_true",
                        help="OBS mode: do not use the virtual camera, only stream to OBS (Browser source)")
    parser.add_argument("--test", type=float, default=0, metavar="SECONDS",
                        help="Self-test: play every effect in sequence, print statistics and exit")
    args = parser.parse_args()
    W, H = C.WIDTH, C.HEIGHT

    print("Webcam Reactions starting...")
    emoji = load_emoji(C.EMOJI_DIR)
    print("Emoji loaded: %d static, %d animated" % (len(emoji.static), len(emoji.animated)))
    if not emoji.animated:
        print("WARNING: animated emoji not built yet. " + SETUP_HINT)

    # 1) Gesture detector
    from .gestures import GestureDetector
    if not os.path.isfile(C.GESTURE_MODEL):
        show_error("Model file not found:\n%s\n\n%s" % (C.GESTURE_MODEL, SETUP_HINT))
        return 1
    try:
        detector = GestureDetector()
    except Exception as e:
        show_error("Could not start hand detection (MediaPipe).\n\nDetails: %s" % e)
        return 1

    # 2) Person segmentation (portrait, studio light, center stage, flag background)
    from .video_modes import CenterStage, PersonSegmenter, apply_portrait, apply_studio_light
    try:
        segmenter = PersonSegmenter()
    except Exception as e:
        segmenter = None
        print("WARNING: person segmentation unavailable (%s). Portrait, Studio Light and Center Stage "
              "are disabled; the flag will be translucent. %s" % (e, SETUP_HINT))

    # 3) Physical camera
    camera = open_camera()
    if camera is None:
        detector.close()
        show_error("Could not open the camera (index %d).\n\n"
                   "- Is it connected, and is another app (Teams, Zoom, Camera) using it?\n"
                   "- Windows Settings > Privacy & security > Camera: allow desktop apps.\n"
                   "- With several cameras, set CAMERA_INDEX in webcam_reactions\\config.py to 1 or 2."
                   % C.CAMERA_INDEX)
        return 1

    # 4) OBS stream
    from .obs_stream import ObsStream
    try:
        obs = ObsStream(C.OBS_STREAM_PORT)
    except OSError as e:
        obs = None
        if args.obs:
            camera.release()
            detector.close()
            show_error("Could not start the OBS stream (port %d is in use).\n\nThe app may already be running.\n\n"
                       "Details: %s" % (C.OBS_STREAM_PORT, e))
            return 1
        print("WARNING: OBS stream unavailable (port %d): %s" % (C.OBS_STREAM_PORT, e))

    # 5) Virtual camera (not used in OBS mode, so OBS can start its own virtual camera)
    vcam = None
    if not args.obs:
        try:
            import pyvirtualcam
            vcam = pyvirtualcam.Camera(width=W, height=H, fps=C.FPS, fmt=pyvirtualcam.PixelFormat.BGR, backend="obs")
        except Exception as e:
            camera.release()
            detector.close()
            if obs:
                obs.close()
            show_error("Could not start the virtual camera (OBS Virtual Camera).\n\n"
                       "- OBS Studio must be installed (it provides the virtual camera driver).\n"
                       "- If OBS's own 'Start Virtual Camera' is on, stop it, or use scripts\\run_obs.bat.\n"
                       "- The app may already be running.\n\nDetails: %s" % e)
            return 1

    print("Camera ready." + (" Virtual camera: %s" % vcam.device if vcam else " OBS mode (virtual camera off)."))
    if obs:
        print("OBS stream: %s  (OBS > Sources > + > Browser > URL)" % obs.url)
    print("Press H in the preview window for all shortcuts, Q to quit.")

    from .overlay import Overlay
    overlay = Overlay()
    center_stage = CenterStage(W, H)
    state = {"gestures": C.GESTURES_ON_AT_START,
             "portrait": C.PORTRAIT_ON_AT_START and segmenter is not None,
             "studio": C.STUDIO_LIGHT_ON_AT_START and segmenter is not None,
             "center_stage": C.CENTER_STAGE_ON_AT_START and segmenter is not None}
    effect, active, party = None, None, None
    candidate, candidate_since, last_seen = None, 0.0, 0.0
    cooldown_until = 0.0
    prev = time.perf_counter()
    started = prev
    fps = float(C.FPS)
    frames, empty_reads, test_index = 0, 0, 0
    test_sequence = E.ALL_EFFECTS + [E.PARTY]
    stats = {"hand_frames": 0, "gestures": {}, "max_ms": 0.0, "total_ms": 0.0}

    def start_effect(name, center, duration=None):
        nonlocal effect, active, party
        if name == E.PARTY:                                     # party mode toggles on / off
            if party is None or party.finished:
                party = E.PartyEffect(W, H, center, duration=duration)
                print("Party mode on")
            else:
                party.stop()
                print("Party mode off")
            return
        effect, active = E.create_effect(name, W, H, center), name
        print("Effect: " + E.LABELS.get(name, name))

    cv2.namedWindow(C.WINDOW_TITLE, cv2.WINDOW_AUTOSIZE)
    try:
        while True:
            ok, frame = camera.read()
            if not ok or frame is None:
                empty_reads += 1
                if empty_reads > 60:
                    show_error("No image from the camera. It may have been unplugged or taken by another app.")
                    return 1
                time.sleep(0.02)
                continue
            empty_reads = 0
            t0 = time.perf_counter()
            now = t0
            dt = min(0.1, now - prev)
            prev = now
            if dt > 0:
                fps = fps * 0.9 + (1.0 / dt) * 0.1

            frame = fit_frame(frame)
            if C.MIRROR_OUTPUT:
                frame = cv2.flip(frame, 1)

            # --- person mask and video modes ---
            need_mask = (state["portrait"] or state["studio"] or state["center_stage"]
                         or (effect is not None and effect.uses_background))
            mask = segmenter.mask(frame) if (segmenter and need_mask) else None
            if state["center_stage"] and mask is not None:
                center_stage.update(mask, dt)
                frame, mask = center_stage.apply(frame, mask)
                frame = np.ascontiguousarray(frame)
            if state["portrait"] and mask is not None:
                apply_portrait(frame, mask)
            if state["studio"] and mask is not None:
                apply_studio_light(frame, mask)

            # --- gestures ---
            detector.submit(frame)
            gesture, center, hands = detector.detect(W, H)
            if hands:
                stats["hand_frames"] += 1
            if gesture:
                stats["gestures"][gesture] = stats["gestures"].get(gesture, 0) + 1

            # --- trigger: hold the gesture for HOLD_TIME, then cooldown ---
            progress = 0.0
            if effect is not None or not state["gestures"] or now < cooldown_until:
                candidate = None
            elif gesture is not None and gesture == candidate:
                last_seen = now
                progress = (now - candidate_since) / C.HOLD_TIME
                if progress >= 1.0:
                    start_effect(gesture, center)
                    candidate = None
                    if gesture == E.PARTY:
                        cooldown_until = now + 1.5
            elif gesture is not None:
                candidate, candidate_since, last_seen = gesture, now, now
            elif candidate is not None:
                if now - last_seen > C.LOSS_TOLERANCE:
                    candidate = None
                else:
                    progress = (now - candidate_since) / C.HOLD_TIME

            # --- self-test: play every effect in turn ---
            if args.test and effect is None and test_index < len(test_sequence):
                start_effect(test_sequence[test_index], (W // 2, H // 2))
                test_index += 1

            # --- draw: party (back) -> effect background layer -> effect ---
            if party is not None:
                party.update(dt)
                party.draw(frame)
                if party.finished:
                    party = None
            if effect is not None:
                if not state["gestures"]:
                    effect = None
                else:
                    effect.update(dt)
                    if effect.uses_background:
                        effect.draw_behind(frame, mask)
                    effect.draw(frame)
                    if effect.finished:
                        effect = None
                        cooldown_until = now + C.COOLDOWN
            if effect is None:
                active = None

            # --- output ---
            if vcam:
                vcam.send(frame)
            if obs:
                obs.send(frame)
            ui = dict(state, fps=fps, hands=hands, candidate=candidate, progress=progress, active=active,
                      party=party is not None, virtual_cam=vcam is not None, obs=obs.viewers if obs else -1)
            cv2.imshow(C.WINDOW_TITLE, overlay.render(frame, ui))

            elapsed_ms = (time.perf_counter() - t0) * 1000
            stats["total_ms"] += elapsed_ms
            stats["max_ms"] = max(stats["max_ms"], elapsed_ms)
            frames += 1

            # --- keyboard ---
            key = cv2.waitKey(1) & 0xFF
            if key != 255:
                key = ord(chr(key).lower()) if 65 <= key <= 90 else key
                if key in (ord("q"), 27):
                    break
                if key in E.KEY_EFFECTS and state["gestures"]:
                    start_effect(E.KEY_EFFECTS[key], (W // 2, int(H * 0.55)))
                elif key == ord("p"):                           # keyboard party lasts until turned off
                    start_effect(E.PARTY, None, duration=float("inf"))
                elif key == ord("e"):
                    state["gestures"] = not state["gestures"]
                elif key == ord("h"):
                    overlay.show_help = not overlay.show_help
                elif key in (ord("b"), ord("l"), ord("c")):
                    mode = {ord("b"): "portrait", ord("l"): "studio", ord("c"): "center_stage"}[key]
                    if segmenter is None:
                        print("This feature needs the segmentation model. " + SETUP_HINT)
                    else:
                        state[mode] = not state[mode]
                        print("%s %s" % (mode, "ON" if state[mode] else "OFF"))
            if cv2.getWindowProperty(C.WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1:
                break                                           # window closed with the X button
            if args.test and now - started >= args.test:
                break
    except KeyboardInterrupt:
        pass
    finally:
        duration = max(1e-6, time.perf_counter() - started)
        camera.release()
        if vcam:
            vcam.close()
        if obs:
            obs.close()
        detector.close()
        if segmenter:
            segmenter.close()
        cv2.destroyAllWindows()
        if args.test:
            print("--- TEST RESULT ---")
            print("frames: %d, time: %.1f s, average FPS: %.1f" % (frames, duration, frames / duration))
            print("average frame time: %.1f ms, slowest: %.1f ms" % (stats["total_ms"] / max(1, frames), stats["max_ms"]))
            print("detector results: %d, frames with hands: %d" % (detector.result_count, stats["hand_frames"]))
            print("gestures seen: %s" % stats["gestures"])
            print("effects played: %d / %d" % (test_index, len(test_sequence)))
    print("Webcam Reactions closed.")
    return 0

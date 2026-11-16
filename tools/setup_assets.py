"""
Prepares runtime assets (run once after cloning; scripts\\setup.bat calls it):
  1. Downloads the MediaPipe models into models/ (skipped if already present).
  2. Converts the animated emoji in assets/emoji/animated_source/*.webp into frame packs
     (assets/emoji/animated/*.npz) that the app can load quickly.

To add a new animated emoji, drop a .webp into assets/emoji/animated_source/ and run this again.
"""
import os
import sys
import urllib.request

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models")
SOURCE_DIR = os.path.join(ROOT, "assets", "emoji", "animated_source")
TARGET_DIR = os.path.join(ROOT, "assets", "emoji", "animated")

MODELS = {
    "gesture_recognizer.task":
        "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task",
    "selfie_segmenter_landscape.tflite":
        "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter_landscape/float16/latest/selfie_segmenter_landscape.tflite",
}
FRAME_SIZE = 224     # px
MAX_FRAMES = 48      # longer animations are resampled evenly (keeps memory low)


def download_models():
    os.makedirs(MODELS_DIR, exist_ok=True)
    for name, url in MODELS.items():
        path = os.path.join(MODELS_DIR, name)
        if os.path.isfile(path) and os.path.getsize(path) > 0:
            print("  model ok          %s" % name)
            continue
        print("  downloading       %s ..." % name)
        tmp = path + ".part"
        urllib.request.urlretrieve(url, tmp)
        os.replace(tmp, path)
        print("  downloaded        %s (%d KB)" % (name, os.path.getsize(path) // 1024))


def convert_webp(path):
    img = Image.open(path)
    frames, durations = [], []
    try:
        while True:
            rgba = np.asarray(img.convert("RGBA").resize((FRAME_SIZE, FRAME_SIZE), Image.LANCZOS))
            frames.append(rgba[..., [2, 1, 0, 3]].copy())          # RGBA -> BGRA
            durations.append(max(10, img.info.get("duration", 40)))
            img.seek(img.tell() + 1)
    except EOFError:
        pass
    total = float(sum(durations))
    if len(frames) > MAX_FRAMES:                                  # resample, keeping the timing
        starts = np.cumsum([0] + durations[:-1])
        picks = [int(np.searchsorted(starts, t, side="right") - 1)
                 for t in np.linspace(0, total, MAX_FRAMES, endpoint=False)]
        frames = [frames[i] for i in picks]
        durations = [total / MAX_FRAMES] * MAX_FRAMES
    return np.stack(frames), np.array(durations, np.float32) / 1000.0


def build_animated_emoji():
    os.makedirs(TARGET_DIR, exist_ok=True)
    for name in sorted(os.listdir(SOURCE_DIR)):
        if not name.lower().endswith(".webp"):
            continue
        frames, durations = convert_webp(os.path.join(SOURCE_DIR, name))
        np.savez_compressed(os.path.join(TARGET_DIR, os.path.splitext(name)[0] + ".npz"),
                            frames=frames, durations=durations)
        print("  emoji built       %-18s %2d frames, %.2f s" % (os.path.splitext(name)[0], len(frames), durations.sum()))


def main():
    print("Models:")
    download_models()
    print("Animated emoji:")
    build_animated_emoji()
    print("Assets ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

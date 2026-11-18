"""
User settings. Edit the values below and restart the app.
"""
import os

# --- Camera and output --------------------------------------------------------
CAMERA_INDEX = 0            # Physical webcam index (0 = first camera; try 1, 2... if the wrong one opens)
WIDTH = 1280                # Output width (px)
HEIGHT = 720                # Output height (px)
FPS = 30                    # Target frame rate
MIRROR_OUTPUT = False       # True: the image sent to the virtual camera / OBS is mirrored
MIRROR_PREVIEW = True       # True: the preview window is mirrored (does not affect the output)
PREVIEW_WIDTH = 720         # Preview window width (px)

# --- Gesture detection --------------------------------------------------------
HOLD_TIME = 1.0             # Seconds a gesture must be held before its effect starts
LOSS_TOLERANCE = 0.3        # A gesture may disappear this long (s) without resetting the hold timer
COOLDOWN = 0.5              # Pause after an effect ends before looking for a new gesture (s)
DETECTION_WIDTH = 640       # Frames are downscaled to this width for hand detection (performance)
MIN_GESTURE_SCORE = 0.5     # Minimum classifier confidence for a gesture (0-1)

# --- Features enabled at startup ----------------------------------------------
GESTURES_ON_AT_START = True       # Hand gesture effects (E key)
PORTRAIT_ON_AT_START = False      # Portrait mode: blurred background (B key)
STUDIO_LIGHT_ON_AT_START = False  # Studio light: bright face, dimmed background (L key)
CENTER_STAGE_ON_AT_START = False  # Center Stage: auto-framing that follows you (C key)
DETECTION_ON_AT_START = True      # Object detection: phone, bottle, pen, cup, headphones (O key)

# --- GPU ----------------------------------------------------------------------------
USE_OPENCL = True                 # Flag / wolf / background blending on the GPU via OpenCL (False = CPU)

# --- Object detection (GPU) -------------------------------------------------------
DETECTION_USE_GPU = True          # Run the detector on the GPU (DirectML); False = CPU
DETECTION_SCORE = 0.6             # Minimum confidence to show an object (0-1); higher = fewer false boxes
DETECTION_MIN_HITS = 3            # An object must be found in this many detections in a row before it is shown
DETECTION_MIN_SIZE = 0.03         # Ignore boxes whose shorter side is below this fraction of the frame height
DETECTION_ON_OUTPUT = True        # True: boxes are also sent to the virtual camera / OBS
DETECTION_HOLD = 0.4              # Keep a box this long (s) after the object is last seen (less flicker)
DETECTION_SMOOTHING = 0.5         # Box smoothing between detections (0 = none, 0.9 = very smooth)
DETECTION_LABELS = {              # Text shown on the boxes
    "phone": "Phone", "bottle": "Bottle", "pen": "Pen", "cup": "Cup", "headphones": "Headphones",
}

# --- Feature tuning -------------------------------------------------------------
PORTRAIT_BLUR = 21                # Portrait background blur strength (5-41)
STUDIO_BACKGROUND_LEVEL = 0.45    # Studio light background brightness (0 = black, 1 = unchanged)
CENTER_STAGE_MAX_ZOOM = 1.7       # Center Stage maximum zoom
PARTY_DURATION = 12.0             # Length of party mode when started with the gesture (s)
FLAG_OPACITY = 1.0                # Opacity of the flag behind you in the wolf effect (0-1)
WOLF_SIDE = "right"               # Side the howling wolf comes out on, from behind you ("right" / "left")
WOLF_SIZE = 0.9                   # Wolf height relative to the frame height
WOLF_SONG_START = 21.0            # Background song for the wolf effect: segment start (seconds into the file)
WOLF_SONG_END = 32.0              # Segment end in seconds (0 = play to the end of the file)
WOLF_SONG_VOLUME = 0.8            # Song volume (0-1)
WOLF_CLIP_CUTOUT = True           # Clip with a natural background: cut the wolf out automatically (cached)
WOLF_CLIP_SEED = (0.62, 0.6)      # A point on the wolf in the clip's FIRST frame (x, y as 0-1 of the frame)
WOLF_OFFSET = 0.3                 # How far the wolf moves out to the side (0 = stays behind you, 0.5 = edge)

# --- OBS --------------------------------------------------------------------------
OBS_STREAM_PORT = 8765      # OBS "Browser" source URL: http://127.0.0.1:8765/
OBS_JPEG_QUALITY = 85       # OBS stream image quality (50-95)

# --- Paths (normally no need to change) -------------------------------------------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT_DIR, "models")
EMOJI_DIR = os.path.join(ROOT_DIR, "assets", "emoji")
WOLF_CLIP_DIR = os.path.join(ROOT_DIR, "assets", "wolf")   # optional real wolf clip (see assets/wolf/README.txt)
WOLF_SONG_FILE = os.path.join(WOLF_CLIP_DIR, "song.mp3")  # optional background song (see assets/wolf/README.txt)
GESTURE_MODEL = os.path.join(MODELS_DIR, "gesture_recognizer.task")
SEGMENTATION_MODEL = os.path.join(MODELS_DIR, "selfie_segmenter_landscape.tflite")
CUTOUT_MODEL = os.path.join(MODELS_DIR, "magic_touch.tflite")   # cuts the wolf out of a clip
DETECTION_MODEL = os.path.join(MODELS_DIR, "dfine_l_obj365.onnx")  # made by tools/export_detector.py
WINDOW_TITLE = "Webcam Reactions"
FONT = r"C:\Windows\Fonts\segoeui.ttf"
FONT_BOLD = r"C:\Windows\Fonts\seguisb.ttf"

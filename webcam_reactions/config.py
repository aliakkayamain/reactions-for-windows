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

# --- Feature tuning -------------------------------------------------------------
PORTRAIT_BLUR = 21                # Portrait background blur strength (5-41)
STUDIO_BACKGROUND_LEVEL = 0.45    # Studio light background brightness (0 = black, 1 = unchanged)
CENTER_STAGE_MAX_ZOOM = 1.7       # Center Stage maximum zoom
PARTY_DURATION = 12.0             # Length of party mode when started with the gesture (s)
FLAG_OPACITY = 1.0                # Opacity of the flag behind you in the wolf effect (0-1)

# --- OBS --------------------------------------------------------------------------
OBS_STREAM_PORT = 8765      # OBS "Browser" source URL: http://127.0.0.1:8765/
OBS_JPEG_QUALITY = 85       # OBS stream image quality (50-95)

# --- Paths (normally no need to change) -------------------------------------------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT_DIR, "models")
EMOJI_DIR = os.path.join(ROOT_DIR, "assets", "emoji")
GESTURE_MODEL = os.path.join(MODELS_DIR, "gesture_recognizer.task")
SEGMENTATION_MODEL = os.path.join(MODELS_DIR, "selfie_segmenter_landscape.tflite")
WINDOW_TITLE = "Webcam Reactions"
FONT = r"C:\Windows\Fonts\segoeui.ttf"
FONT_BOLD = r"C:\Windows\Fonts\seguisb.ttf"

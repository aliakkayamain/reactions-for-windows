# Webcam Reactions

macOS-style **gesture reactions and video effects for any webcam on Windows**.
Hold a hand gesture in front of your camera and animated effects (hearts, fireworks, lasers,
a disco party, a waving flag with howling wolves...) appear on your video. The result is sent to
**Teams, Zoom, Google Meet, Discord** through a virtual camera, and to **OBS Studio** for live streaming.

## Features

- **Gesture reactions** (like Reactions on macOS): thumbs up/down, balloons, hearts, rain,
  confetti, fireworks and lasers, plus extras: **party mode** and a **Grey Wolf** effect.
- **Video effects** (like on macOS): Portrait mode (blurred background), Studio Light and Center Stage
  (auto-framing).
- **Modern visuals**: 3D and animated emoji, glow and bloom lighting, soft shadows, smooth animation.
- **Object detection on the GPU**: phone, bottle, pen, cup and headphones are boxed and labelled
  live (D-FINE-L Objects365 model on ONNX Runtime + DirectML). Toggle with `O`.
- **Keyboard shortcuts** to trigger any effect without a gesture.
- **Two outputs**: virtual camera for video calls, and a local stream for OBS Studio.
- Runs fully offline, about 30 FPS at 1280×720. Object detection runs on the GPU (DirectML, any DirectX 12
  card), and the flag / wolf / background compositing on the GPU via OpenCL; both fall back to the CPU.
  MediaPipe (hand gestures, person segmentation) has no GPU support on Windows and stays on the CPU.

## Requirements

| What | Notes |
|---|---|
| Windows 10 or 11 (64-bit) | |
| A webcam | Built-in or USB |
| [Python 3.12](https://www.python.org/downloads/) | Tick **"Add python.exe to PATH"** in the installer |
| [OBS Studio](https://obsproject.com/download) 30 or newer | Needed even if you don't stream: it installs the **OBS Virtual Camera** driver |
| [Git](https://git-scm.com/download/win) | To clone the repository (or download it as a ZIP) |
| Internet connection | Only for the first setup (packages and models) |

## Installation (step by step)

### 1. Install Python 3.12
Download it from [python.org](https://www.python.org/downloads/). In the first installer screen, tick
**"Add python.exe to PATH"**, then click **Install Now**.

Check that it works by opening a new Command Prompt and running:
```bash
python --version
```
You should see `Python 3.12.x`.

### 2. Install OBS Studio
Download and install it from [obsproject.com](https://obsproject.com/download). You don't need to configure
anything; installing it adds the **OBS Virtual Camera** device to Windows.

### 3. Get the project
```bash
git clone https://github.com/aliakkayamain/reactions-for-windows.git
```
```bash
cd reactions-for-windows
```
(Or click **Code → Download ZIP** on GitHub and extract it.)

### 4. Run the setup
Double-click **`scripts\setup.bat`**, or run it from the project folder:
```bash
scripts\setup.bat
```
It does three things (takes a few minutes the first time):
1. creates a Python virtual environment in `venv\`,
2. installs the packages from `requirements.txt`,
3. downloads the MediaPipe models into `models\` and builds the animated emoji.

When it prints **"Setup complete"** you're done. You only need to do this once.

### 4b. Optional: object detection model
Object detection (phone, bottle, pen, cup, headphones) needs `models\dfine_l_obj365.onnx`. It is made once
from the public D-FINE weights (needs Git and ~1.5 GB of temporary space, takes a few minutes):
```
scripts\export_detector.bat
```
Without it the app runs normally, just without object detection. Note: the model is pretrained on the
Objects365 dataset, whose terms may restrict commercial use.

### 5. Start the app
Double-click **`scripts\run.bat`**, or:
```bash
scripts\run.bat
```
A preview window opens. Keep it open (you can minimize it) — the effects only work while the app runs.

### 6. Select the camera in your video call app
Start **Webcam Reactions first**, then open your call app and pick **OBS Virtual Camera** as the camera:

| App | Where |
|---|---|
| Microsoft Teams | Settings → Devices → Camera |
| Zoom | Settings → Video → Camera |
| Google Meet | ⋮ More options → Settings → Video → Camera |
| Discord | User Settings → Voice & Video → Camera |

Hold up a thumb for one second — a 3D thumbs-up should appear. 🎉

> Tip: create a desktop shortcut to `scripts\run.bat` (right-click → Send to → Desktop) for one-click start.

## Usage

### Hand gestures
Hold the gesture still for about **1 second**; a progress ring in the preview fills up and the effect starts.

| Gesture | Effect |
|---|---|
| 👍 Thumbs up | 3D thumbs up with a light ring |
| 👎 Thumbs down | 3D thumbs down |
| ✌️ Victory (one hand) | Colorful 3D balloons |
| 🐺 Grey Wolf sign (one hand) | Waving Turkish flag **behind you** with a howling wolf, optionally with a song (see [Grey Wolf effect](#grey-wolf-effect)) |
| 👍👍 Two thumbs up | Fireworks |
| 👎👎 Two thumbs down | Rain clouds and rain |
| ✌️✌️ Victory with both hands | Confetti and party poppers |
| 🫶 Heart with both hands | 3D hearts |
| 🤘🤘 Rock sign with both hands | Laser show |
| 🙌 Both palms open | **Party mode**: mirror ball, spotlights, disco lights (12 s; repeat to stop) |

- **Grey Wolf sign:** index and pinky fingers up (the ears), middle and ring fingertips touching the thumb tip
  (the snout), palm facing the camera.
- **Heart:** thumb tips touching at the bottom, index fingertips touching at the top.
- For two-hand gestures, raise **both hands at the same time**.

### Grey Wolf effect
The flag and the wolf are drawn **behind you**: they are composited into one background layer and your
person cut-out is applied last, so you always stay in front and your outline is not changed.

**Your own wolf clip (optional).** Put one file in `assets\wolf\` (for example `wolf.mp4`):

| Clip | What happens |
|---|---|
| GIF / WebP / PNG with transparency | Used as-is |
| Video on a green screen or plain background | The background colour is keyed out |
| Ordinary video (natural background) | The wolf is cut out frame by frame with MediaPipe's interactive segmenter (`models\magic_touch.tflite`, downloaded by setup). This runs once (~25 s) and is cached as `<file>.cutout.npz`. If the wrong thing is cut out, set `WOLF_CLIP_SEED` to a point on the wolf in the first frame. |

The clip is cropped to the wolf and looped forwards/backwards. A wolf that is cut off at the edge of the
video is placed against that edge of the picture. Without a clip, a drawn wolf is used.

**Song (optional).** Save it as `assets\wolf\song.mp3` and set the part to play in `config.py`, e.g.
`WOLF_SONG_START = 21.0`, `WOLF_SONG_END = 32.0`. The effect lasts as long as that part. The song plays on
the PC's speakers; the virtual camera carries video only.

Files in `assets\wolf\` are ignored by Git. Only use clips and music you have the right to use.

### Object detection
Phones, bottles, pens, cups and headphones get a coloured box and a label (`O` turns it on/off). It needs
the model from [step 4b](#4b-optional-object-detection-model).

- Runs on the GPU (DirectML) on a background thread, so the video does not wait for it (~20 ms per detection
  on an RTX 5060 laptop GPU; ~240 ms on the CPU).
- To avoid false boxes, an object is shown only when its confidence is at least `DETECTION_SCORE` (0.6),
  it was found in `DETECTION_MIN_HITS` (3) detections in a row and it is not tiny. While the flag is
  behind you, objects in the hidden background get no box.
- Too many false boxes: raise `DETECTION_SCORE` (e.g. 0.7). Real objects missed: lower it (e.g. 0.5).

### Keyboard shortcuts
Click the preview window first.

| Key | Action |
|---|---|
| `1` – `9` | Trigger an effect: 1 thumbs up, 2 thumbs down, 3 hearts, 4 balloons, 5 rain, 6 confetti, 7 fireworks, 8 lasers, 9 Grey Wolf |
| `P` | Party mode on/off (stays on until you press `P` again) |
| `B` | Portrait mode — blurred background |
| `L` | Studio Light — brighter face, dimmed background |
| `C` | Center Stage — follows and frames you |
| `O` | Object detection on/off (phone, bottle, pen, cup, headphones) |
| `E` | Turn gesture detection on/off |
| `H` | Show/hide the shortcut list |
| `Q` | Quit |

### Streaming with OBS Studio
1. Start the app in OBS mode: **`scripts\run_obs.bat`**.
2. In OBS: **Sources → + → Browser**, set
   - URL: `http://127.0.0.1:8765/`
   - Width: `1280`, Height: `720`
3. Select the source and press **Ctrl+F** to fit it to the screen.

Alternatively import the ready-made scene: **Scene Collection → Import →** `obs\scene_collection.json`.

Notes:
- Do **not** also add your physical webcam to OBS — the app is already using it.
- In OBS mode the app does not use the virtual camera, so you can press **Start Virtual Camera** in OBS
  to send your whole OBS scene (with the effects) to Teams/Zoom at the same time.
- The order doesn't matter: the OBS source reconnects automatically when the app starts.

## Configuration
Open `webcam_reactions\config.py` in any text editor, change values and restart the app. Common settings:

| Setting | Meaning |
|---|---|
| `CAMERA_INDEX` | Which webcam to use (0, 1, 2...) |
| `WIDTH`, `HEIGHT` | Output resolution (use 960×540 on slower PCs) |
| `HOLD_TIME` | Seconds to hold a gesture |
| `PORTRAIT_ON_AT_START`, `STUDIO_LIGHT_ON_AT_START`, `CENTER_STAGE_ON_AT_START` | Video effects enabled at startup |
| `PORTRAIT_BLUR` | Background blur strength |
| `PARTY_DURATION` | Party mode length when started by gesture |
| `OBS_STREAM_PORT` | Port of the OBS stream |
| `DETECTION_ON_AT_START`, `DETECTION_SCORE` | Object detection at startup, minimum confidence |
| `DETECTION_MIN_HITS`, `DETECTION_MIN_SIZE` | Detections in a row before a box is shown, minimum box size |
| `DETECTION_ON_OUTPUT` | Also send the object boxes to the virtual camera / OBS |
| `DETECTION_LABELS` | Text shown on the boxes (e.g. Turkish names) |
| `DETECTION_USE_GPU`, `USE_OPENCL` | GPU for detection / for compositing (`False` = CPU) |
| `FLAG_OPACITY`, `WOLF_SIDE`, `WOLF_SIZE`, `WOLF_OFFSET` | Grey Wolf effect: flag opacity, wolf side, size and position |
| `WOLF_SONG_START`, `WOLF_SONG_END`, `WOLF_SONG_VOLUME` | Song part (seconds) and volume |
| `WOLF_CLIP_CUTOUT`, `WOLF_CLIP_SEED` | Cut-out of a natural-background wolf clip |

## Troubleshooting

| Problem | Solution |
|---|---|
| The call app shows a black / empty "OBS Virtual Camera" | Webcam Reactions isn't running. Start it, then switch the camera in the call app to another one and back. |
| "Could not open the camera" | Another app is using the webcam. Close it (or set its camera to OBS Virtual Camera) and restart. Also check Windows Settings → Privacy & security → Camera → allow desktop apps. |
| Wrong camera / black preview | Set `CAMERA_INDEX` to `1` or `2` in `config.py`. |
| "Could not start the virtual camera" | Another program is using OBS Virtual Camera: usually a second copy of this app (close it) or OBS's own Start Virtual Camera (stop it, or use `scripts\run_obs.bat`). If OBS Studio is not installed, install it. |
| "OBS Virtual Camera" is not listed in the call app | Fully quit and reopen the call app while Webcam Reactions is running. If it's still missing, open OBS once and click Start/Stop Virtual Camera. |
| Gestures are not detected | Use good lighting and keep your hand clearly visible, a bit away from your face. The preview shows `Hands 0` when no hand is seen. |
| Laggy video | Turn off Portrait / Studio Light / Center Stage, or lower `WIDTH`/`HEIGHT`. |
| "Model file not found" | Run `scripts\setup.bat` again. |
| "Object detection disabled" at start | The detection model is missing: run `scripts\export_detector.bat` (step 4b). |
| Boxes on things that are not there | Raise `DETECTION_SCORE` or `DETECTION_MIN_HITS` in `config.py`. |
| The song does not play | Check `assets\wolf\song.mp3` and the `WOLF_SONG_*` times. The console shows `Wolf song loaded` when it is found. |
| The camera image looks grey / dull | The app does not change the camera's own settings. Add light in front of you and check the camera's brightness / saturation in its settings. |
| The flag or text looks mirrored in my own view | Call apps mirror your self-view; other participants see it correctly. |

## Project structure

```
reactions-for-windows/
├── README.md
├── requirements.txt
├── scripts/                  # double-click helpers
│   ├── setup.bat             # one-time setup
│   ├── export_detector.bat   # one-time export of the object detection model
│   ├── run.bat               # start (virtual camera)
│   └── run_obs.bat           # start (OBS mode)
├── webcam_reactions/         # application package (python -m webcam_reactions)
│   ├── app.py                # main loop: camera -> effects -> outputs
│   ├── config.py             # user settings
│   ├── effects.py            # all effects
│   ├── gestures.py           # gesture detection
│   ├── detector.py           # object detection (GPU, ONNX Runtime + DirectML)
│   ├── gpu.py                # OpenCL compositing helpers
│   ├── sound.py              # wolf effect song segment
│   ├── video_modes.py        # portrait, studio light, center stage
│   ├── flag.py               # waving flag renderer
│   ├── wolf.py               # howling wolf (real clip from assets/wolf/ or procedural)
│   ├── graphics.py           # emoji sprites, blending, glow
│   ├── overlay.py            # preview window UI
│   └── obs_stream.py         # local stream for OBS
├── assets/
│   ├── emoji/static/         # 3D emoji (PNG)
│   ├── emoji/animated_source/# animated emoji (WebP)
│   └── licenses/
├── models/                   # MediaPipe models (downloaded by setup)
├── obs/scene_collection.json # ready-made OBS scene
└── tools/
    ├── setup_assets.py       # downloads models, builds animated emoji
    └── export_detector.py    # D-FINE -> ONNX (DirectML-safe)
```

Command-line options (from the project folder):
```bash
venv\Scripts\python -m webcam_reactions --help
```
- `--obs` — OBS mode (no virtual camera)
- `--test SECONDS` — self-test: plays every effect and prints performance statistics

## Credits
- Hand gesture and person segmentation models: [Google MediaPipe](https://ai.google.dev/edge/mediapipe) (Apache 2.0)
- Object detection: [D-FINE](https://github.com/Peterande/D-FINE) (Apache 2.0), weights pretrained on Objects365
- 3D emoji: [Microsoft Fluent Emoji](https://github.com/microsoft/fluentui-emoji) (MIT) — `assets/licenses/FLUENT_EMOJI_LICENSE.txt`
- Animated emoji: [Google Noto Animated Emoji](https://googlefonts.github.io/noto-emoji-animation/) (CC BY 4.0)
- Virtual camera: [pyvirtualcam](https://github.com/letmaik/pyvirtualcam) with the OBS Virtual Camera driver

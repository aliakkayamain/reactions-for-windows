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
- **Keyboard shortcuts** to trigger any effect without a gesture.
- **Two outputs**: virtual camera for video calls, and a local stream for OBS Studio.
- Runs fully offline on the CPU, about 30 FPS at 1280×720.

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
| 🐺 Grey Wolf sign (one hand) | Waving Turkish flag **behind you**; a grey wolf comes out from behind you and howls (drop your own clip in `assets/wolf/`) |
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

### Keyboard shortcuts
Click the preview window first.

| Key | Action |
|---|---|
| `1` – `9` | Trigger an effect: 1 thumbs up, 2 thumbs down, 3 hearts, 4 balloons, 5 rain, 6 confetti, 7 fireworks, 8 lasers, 9 Grey Wolf |
| `P` | Party mode on/off (stays on until you press `P` again) |
| `B` | Portrait mode — blurred background |
| `L` | Studio Light — brighter face, dimmed background |
| `C` | Center Stage — follows and frames you |
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

## Troubleshooting

| Problem | Solution |
|---|---|
| The call app shows a black / empty "OBS Virtual Camera" | Webcam Reactions isn't running. Start it, then switch the camera in the call app to another one and back. |
| "Could not open the camera" | Another app is using the webcam. Close it (or set its camera to OBS Virtual Camera) and restart. Also check Windows Settings → Privacy & security → Camera → allow desktop apps. |
| Wrong camera / black preview | Set `CAMERA_INDEX` to `1` or `2` in `config.py`. |
| "Could not start the virtual camera" | Install OBS Studio. If OBS's own virtual camera is running, stop it, or use `scripts\run_obs.bat`. |
| "OBS Virtual Camera" is not listed in the call app | Fully quit and reopen the call app while Webcam Reactions is running. If it's still missing, open OBS once and click Start/Stop Virtual Camera. |
| Gestures are not detected | Use good lighting and keep your hand clearly visible, a bit away from your face. The preview shows `Hands 0` when no hand is seen. |
| Laggy video | Turn off Portrait / Studio Light / Center Stage, or lower `WIDTH`/`HEIGHT`. |
| "Model file not found" | Run `scripts\setup.bat` again. |
| The flag or text looks mirrored in my own view | Call apps mirror your self-view; other participants see it correctly. |

## Project structure

```
reactions-for-windows/
├── README.md
├── requirements.txt
├── scripts/                  # double-click helpers
│   ├── setup.bat             # one-time setup
│   ├── run.bat               # start (virtual camera)
│   └── run_obs.bat           # start (OBS mode)
├── webcam_reactions/         # application package (python -m webcam_reactions)
│   ├── app.py                # main loop: camera -> effects -> outputs
│   ├── config.py             # user settings
│   ├── effects.py            # all effects
│   ├── gestures.py           # gesture detection
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
└── tools/setup_assets.py     # downloads models, builds animated emoji
```

Command-line options (from the project folder):
```bash
venv\Scripts\python -m webcam_reactions --help
```
- `--obs` — OBS mode (no virtual camera)
- `--test SECONDS` — self-test: plays every effect and prints performance statistics

## Credits
- Hand gesture and person segmentation models: [Google MediaPipe](https://ai.google.dev/edge/mediapipe) (Apache 2.0)
- 3D emoji: [Microsoft Fluent Emoji](https://github.com/microsoft/fluentui-emoji) (MIT) — `assets/licenses/FLUENT_EMOJI_LICENSE.txt`
- Animated emoji: [Google Noto Animated Emoji](https://googlefonts.github.io/noto-emoji-animation/) (CC BY 4.0)
- Virtual camera: [pyvirtualcam](https://github.com/letmaik/pyvirtualcam) with the OBS Virtual Camera driver

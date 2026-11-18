Real wolf clip for the Grey Wolf effect (optional)
==================================================

Drop ONE file here and restart the app. The first file found (alphabetical) is used:

  wolf.gif / wolf.webp / wolf.png   animated, with a transparent background (best)
  wolf.mp4 / .mov / .webm / .avi    green screen / plain background: keyed out by colour
                                    natural background: the wolf is cut out with MediaPipe
                                    (models/magic_touch.tflite) once, cached as <file>.cutout.npz.
                                    If the wrong thing is cut out, set WOLF_CLIP_SEED in config.py
                                    to a point on the wolf in the clip's first frame.

Tips: a side view of a howling wolf, facing LEFT (towards you when WOLF_SIDE = "right"),
3-8 seconds long. The clip is looped, cropped to the wolf, and drawn behind you on the flag.
Only use footage you have the right to use.

Without a file here, a procedurally animated wolf is drawn instead.
Size / side / position: WOLF_SIZE, WOLF_SIDE, WOLF_OFFSET in webcam_reactions/config.py

Background song (optional)
==========================

Save the song here as  song.mp3  (or set WOLF_SONG_FILE in webcam_reactions/config.py; .wav and
.wma also work, .m4a/.aac if the codec is installed). Only the segment between WOLF_SONG_START and
WOLF_SONG_END (seconds) is played, and the wolf effect lasts as long as that segment.
The song plays on the PC's speakers; the virtual camera sends video only.

Files you put in this folder are ignored by git (see .gitignore), so they are not pushed to GitHub.

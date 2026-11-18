"""
Background song for the Grey Wolf effect.

Plays only a segment (WOLF_SONG_START .. WOLF_SONG_END seconds) of a local audio file through the
Windows MCI API (winmm.dll): mp3 / wav / wma, plus m4a / aac if the codec is installed. No extra
packages or ffmpeg needed. Sound goes to the PC's default speakers; the virtual camera carries
video only (see README).
"""
import ctypes
import os

from . import config as C

AUDIO_EXTS = (".mp3", ".wav", ".wma", ".m4a", ".aac")
_ALIAS = "wolfsong"


def _mci(command):
    buf = ctypes.create_unicode_buffer(256)
    err = ctypes.windll.winmm.mciSendStringW(command, buf, 255, 0)
    return err, buf.value


def _playable(path):
    """MCI refuses many MP3s with a large ID3v2 tag (cover art, UTF-8 text). Play a copy without
    the tag (created once next to the file) in that case."""
    if not path.lower().endswith(".mp3"):
        return path
    with open(path, "rb") as f:
        head = f.read(10)
    if len(head) < 10 or head[:3] != b"ID3":
        return path
    size = ((head[6] & 0x7F) << 21) | ((head[7] & 0x7F) << 14) | ((head[8] & 0x7F) << 7) | (head[9] & 0x7F)
    size += 10 + (10 if head[5] & 0x10 else 0)                  # header (+ footer)
    copy = path + ".notag.mp3"
    if not (os.path.isfile(copy) and os.path.getmtime(copy) >= os.path.getmtime(path)):
        with open(path, "rb") as f:
            data = f.read()
        with open(copy, "wb") as f:
            f.write(data[size:])
    return copy


class SongSegment:
    def __init__(self, path, start, end, volume):
        self.path, self.start, self.end = path, max(0.0, start), end
        self.volume = int(max(0, min(1000, volume * 1000)))
        self.is_open = False
        self.length = None

    @property
    def duration(self):
        end = self.end if self.end and self.end > self.start else self.length
        return (end - self.start) if end else None

    def _open(self):
        if self.is_open:
            return True
        err, _ = _mci('open "%s" type mpegvideo alias %s' % (_playable(self.path), _ALIAS))
        if err:
            print("WARNING: could not open song %s (MCI error %d). Try an .mp3 or .wav file." % (self.path, err))
            return False
        _mci("set %s time format milliseconds" % _ALIAS)
        err, value = _mci("status %s length" % _ALIAS)
        if not err and value.isdigit():
            self.length = int(value) / 1000.0
        self.is_open = True
        return True

    def play(self):
        if not self._open():
            return
        end = self.start + self.duration if self.duration else None
        _mci("setaudio %s volume to %d" % (_ALIAS, self.volume))
        command = "play %s from %d" % (_ALIAS, int(self.start * 1000))
        if end:
            command += " to %d" % int(end * 1000)
        err, _ = _mci(command)
        if err:
            print("WARNING: could not play song (MCI error %d)" % err)

    def stop(self):
        if self.is_open:
            _mci("stop %s" % _ALIAS)

    def close(self):
        if self.is_open:
            _mci("close %s" % _ALIAS)
            self.is_open = False


_SONG = {"loaded": False, "song": None}


def get_song():
    """The configured song segment, or None (no file / not on Windows)."""
    if not _SONG["loaded"]:
        _SONG["loaded"] = True
        path = C.WOLF_SONG_FILE
        if path and os.path.isfile(path) and os.name == "nt":
            song = SongSegment(os.path.abspath(path), C.WOLF_SONG_START, C.WOLF_SONG_END, C.WOLF_SONG_VOLUME)
            if song._open():
                _SONG["song"] = song
                print("Wolf song loaded: %s (%.1f-%.1f s)" % (path, song.start, song.start + (song.duration or 0)))
    return _SONG["song"]


def close_song():
    if _SONG["song"] is not None:
        _SONG["song"].close()

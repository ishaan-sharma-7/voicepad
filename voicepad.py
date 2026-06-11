#!/usr/bin/env python3
"""
VoicePad v7
Trigger key (default Right Cmd) = start/stop  |  Esc = cancel
Raw transcription only — mic is opt-in (open solely while recording).
"""

import threading
import queue
import time
import math
import random
import sys
import os
import signal
import subprocess

# When running as a py2app bundle, py2app puts lib/python311.zip and
# lib/python3.11/lib-dynload on sys.path but NOT lib/python3.11/ itself.
# Our mlx namespace package's Python sources live at lib/python3.11/mlx/
# (post-build copy from setup.py), so without adding that parent dir
# Python finds only lib-dynload/mlx/core.so and trips on a missing
# `mlx._reprlib_fix` import. Adding the path here makes the full
# namespace visible.
if os.environ.get('RESOURCEPATH'):
    _bundled_lib = os.path.join(os.environ['RESOURCEPATH'], 'lib', 'python3.11')
    if _bundled_lib not in sys.path:
        sys.path.insert(0, _bundled_lib)


def _hard_exit(*_):
    """Skip atexit handlers when the process is asked to terminate.

    Some dylibs in Python's environment (e.g. libggml-metal, dlopened by
    transitive deps of mlx/whisper backends) call abort() from their
    destructor during __cxa_finalize_ranges. That fires SIGABRT inside
    a clean shutdown and macOS reports it as "Python quit unexpectedly"
    every time Hammerspoon reloads us. We have nothing critical to flush
    on exit (audio streams and the panel are released by the OS), so we
    bypass cleanup entirely on signal- or AppleEvent-driven termination.
    """
    os._exit(0)


signal.signal(signal.SIGTERM, _hard_exit)
signal.signal(signal.SIGINT, _hard_exit)

try:
    import sounddevice as sd
    import numpy as np
except ImportError:
    sys.exit("pip install sounddevice numpy")

try:
    import pyperclip
except ImportError:
    sys.exit("pip install pyperclip")

try:
    from pynput import keyboard as kb_input
except ImportError:
    sys.exit("pip install pynput")

try:
    import requests
except ImportError:
    sys.exit("pip install requests")

try:
    import objc
    from AppKit import (
        NSApplication, NSView, NSBezierPath, NSColor,
        NSFont, NSString,
        NSFontAttributeName, NSForegroundColorAttributeName,
        NSMakeRect, NSMakePoint,
        NSBackingStoreBuffered, NSWindowStyleMaskBorderless,
        NSFloatingWindowLevel, NSStatusWindowLevel, NSPopUpMenuWindowLevel,
        NSScreenSaverWindowLevel,
        NSNonactivatingPanelMask,
        NSPanel, NSWindowCollectionBehaviorCanJoinAllSpaces,
        NSWindowCollectionBehaviorStationary,
        NSWindowCollectionBehaviorFullScreenAuxiliary,
        NSTimer, NSScreen, NSWorkspace, NSEvent,
    )
    from Foundation import (
        NSObject, NSNotificationCenter, NSProcessInfo, NSPointInRect,
        NSRunLoop, NSRunLoopCommonModes,
        NSActivityUserInitiatedAllowingIdleSystemSleep, NSActivityLatencyCritical,
    )
    HAVE_APPKIT = True
except Exception as e:
    print(f"[voicepad] AppKit unavailable ({e}), using tkinter")
    HAVE_APPKIT = False

# ── config ────────────────────────────────────────────────────────────────────
import json, re

SAMPLE_RATE  = 16_000
CHANNELS     = 1
DTYPE        = "float32"
AUTOHIDE_SEC = 0.3

# User config — created with defaults on first run, edit and restart to apply.
CONFIG_PATH = os.path.expanduser("~/.voicepad/config.json")

DEFAULT_CONFIG = {
    # pynput key name ("cmd_r", "alt_r", "f19", ...) or a single character.
    "hotkey": "cmd_r",
    # "builtin" records from the Mac's internal mic even when AirPods are
    # connected — it opens instantly (no Bluetooth HFP negotiation) and
    # sounds better than the HFP telephone codec. "default" follows the
    # system default input device instead; expect the first words to be
    # clipped while a Bluetooth mic renegotiates.
    "input_device": "builtin",
    # Remote transcription server (e.g. the office Mac Mini):
    #   "http://macmini.local:8765/transcribe"
    # Empty = transcribe locally on this machine.
    "transcribe_url": "",
    # Load the local Whisper model even when transcribe_url is set, so
    # dictation keeps working if the server is unreachable. Set false for
    # a thin client (no model download, no RAM cost).
    "local_fallback": True,
    "mlx_model": "mlx-community/whisper-small.en-mlx",
}

def load_config():
    cfg = dict(DEFAULT_CONFIG)
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH) as f:
                cfg.update(json.load(f))
        except Exception as e:
            print(f"[voicepad] bad config.json ({e}), using defaults")
    else:
        try:
            os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
            with open(CONFIG_PATH, "w") as f:
                json.dump(DEFAULT_CONFIG, f, indent=2)
        except Exception:
            pass
    return cfg

CONFIG         = load_config()
MLX_MODEL      = CONFIG["mlx_model"]
TRANSCRIBE_URL = (CONFIG["transcribe_url"] or "").strip()

def resolve_hotkey():
    name = str(CONFIG.get("hotkey", "cmd_r"))
    key = getattr(kb_input.Key, name, None)
    if key is not None:
        return key
    if len(name) == 1:
        return kb_input.KeyCode.from_char(name)
    print(f"[voicepad] unknown hotkey {name!r}, falling back to cmd_r")
    return kb_input.Key.cmd_r

HOTKEY = resolve_hotkey()

# ── vocab config ──────────────────────────────────────────────────────────────
VOCAB_PATH = os.path.expanduser("~/.voicepad/vocab.json")

DEFAULT_VOCAB = {
    # Add your own phonetic corrections here.
    # Keys are what Whisper hears; values are what you want typed.
    # Example:
    #   "jon": "John",
    #   "jon doe": "John Doe",
    #   "api": "API",
}

def load_vocab():
    """Load vocab from file, fall back to defaults. Hot-reloaded each transcription."""
    if os.path.exists(VOCAB_PATH):
        try:
            with open(VOCAB_PATH) as f:
                return json.load(f)
        except Exception:
            pass
    else:
        # create default vocab file on first run
        try:
            os.makedirs(os.path.dirname(VOCAB_PATH), exist_ok=True)
            with open(VOCAB_PATH, 'w') as f:
                json.dump(DEFAULT_VOCAB, f, indent=2)
        except Exception:
            pass
    return DEFAULT_VOCAB

def apply_vocab(text: str) -> str:
    """Case-insensitive word-boundary replacement using vocab map."""
    vocab = load_vocab()
    for wrong, right in sorted(vocab.items(), key=lambda x: -len(x[0])):
        pattern = re.compile(r'\b' + re.escape(wrong) + r'\b', re.IGNORECASE)
        text = pattern.sub(right, text)
    return text

# ── window geometry ───────────────────────────────────────────────────────────
# Small Wispr-Flow-style pill: just a waveform inside a rounded capsule.
W        = 148
H        = 36
RADIUS   = (H - 3) / 2   # full capsule (accounting for the 1.5px rim inset)
BARS     = 16

# ── palette ───────────────────────────────────────────────────────────────────
BG       = (0.11,  0.11,  0.125, 0.96)
BORDER   = (0.30,  0.30,  0.34,  0.8)
WAVE_A   = (0.86,  0.86,  0.88,  1.0)
WAVE_D   = (0.19,  0.19,  0.21,  1.0)
DOT_GREY = (0.55,  0.55,  0.60,  1.0)
DOT_GRN  = (0.22,  0.82,  0.46,  1.0)

DOT_STATE_IDLE = "idle"
DOT_STATE_REC  = "rec"
DOT_STATE_DONE = "done"

# ── waveform envelope ─────────────────────────────────────────────────────────
BAR_ENV = []
for i in range(BARS):
    t = (i / (BARS - 1)) * 2 - 1
    env = math.exp(-(t ** 2) / 0.55)
    BAR_ENV.append(0.30 + 0.70 * env)

rng = random.Random(7)
BAR_LAG = [rng.random() * 0.18 for _ in range(BARS)]

def ns_c(r, g, b, a=1.0):
    return NSColor.colorWithCalibratedRed_green_blue_alpha_(r, g, b, a)

# ── paste ─────────────────────────────────────────────────────────────────────
def paste_to_frontmost():
    time.sleep(0.04)
    subprocess.run(["osascript", "-e",
        'tell application "System Events" to keystroke "v" using {command down}'],
        capture_output=True)

# ── transcription ─────────────────────────────────────────────────────────────
_mlx_ok   = False
_fw_model = None

def load_transcription_model():
    global _mlx_ok, _fw_model
    if TRANSCRIBE_URL:
        try:
            health = TRANSCRIBE_URL.rsplit("/", 1)[0] + "/health"
            r = requests.get(health, timeout=3)
            print(f"[voicepad] remote transcription ready: {r.json()}")
        except Exception as e:
            print(f"[voicepad] remote server unreachable ({e})")
        if not CONFIG.get("local_fallback", True):
            return
    try:
        import mlx_whisper
        dummy = np.zeros(SAMPLE_RATE, dtype=np.float32)
        mlx_whisper.transcribe(dummy, path_or_hf_repo=MLX_MODEL, language="en")
        _mlx_ok = True
        print("[voicepad] mlx-whisper ready")
    except Exception as e:
        print(f"[voicepad] mlx failed ({e}), trying faster-whisper")
        try:
            from faster_whisper import WhisperModel
            _fw_model = WhisperModel("small.en", device="cpu", compute_type="int8")
            print("[voicepad] faster-whisper ready")
        except Exception as e2:
            print(f"[voicepad] no model: {e2}")

# Common Whisper hallucinations on silence/ambient noise
_HALLUCINATIONS = {
    "you", "thanks for watching", "thank you", "thank you.",
    "thanks for watching.", "you.", "bye", "bye.", "goodbye",
    "please subscribe", "like and subscribe", "subtitles by",
    "transcribed by", "www.", ".com", "hmm", "hmm.", "uh", "um",
}

def _is_hallucination(text: str) -> bool:
    t = text.strip().lower().rstrip(".")
    if t in _HALLUCINATIONS:
        return True
    # repeated phrase filter — e.g. "so so so so so"
    words = t.split()
    if len(words) >= 4:
        unique = len(set(words))
        if unique / len(words) < 0.4:   # >60% repeated words
            return True
    return False

def transcribe_remote(audio_np):
    """POST raw 16 kHz mono float32 PCM; the server returns {"text": ...}."""
    r = requests.post(
        TRANSCRIBE_URL,
        data=audio_np.astype(np.float32).tobytes(),
        headers={"Content-Type": "application/octet-stream"},
        timeout=30)
    r.raise_for_status()
    return r.json().get("text", "").strip()

def transcribe(audio_np):
    text = None
    if TRANSCRIBE_URL:
        try:
            text = transcribe_remote(audio_np)
        except Exception as e:
            print(f"[voicepad] remote transcribe failed ({e}), trying local")
    if text is None and _mlx_ok:
        try:
            import mlx_whisper
            r = mlx_whisper.transcribe(audio_np, path_or_hf_repo=MLX_MODEL, language="en")
            text = r.get("text", "").strip()
        except Exception as e:
            print(f"[voicepad] mlx error: {e}")
    if text is None and _fw_model:
        segs, _ = _fw_model.transcribe(audio_np, language="en", beam_size=5, vad_filter=True)
        text = " ".join(s.text.strip() for s in segs).strip()
    if not text or _is_hallucination(text):
        return ""
    return apply_vocab(text)

# ── silence detection ─────────────────────────────────────────────────────────
SILENCE_RMS_THRESHOLD = 0.008
SILENCE_FRAMES_NEEDED = 45
MIN_CHUNK_FRAMES      = 15

def resolve_input_device():
    """Index of the configured input device, or None for the system default.

    "builtin" pins the Mac's internal microphone. The mic is opt-in — it is
    only open while actually recording — so the device must open instantly;
    the internal mic does, while a Bluetooth default (AirPods) spends ~1s
    renegotiating HFP on every open."""
    if CONFIG.get("input_device") != "builtin":
        return None
    try:
        for i, d in enumerate(sd.query_devices()):
            name = d["name"].lower()
            if d["max_input_channels"] > 0 and (
                    "built-in" in name or "macbook" in name
                    or "imac" in name or "mac mini" in name
                    or "mac studio" in name):
                return i
    except Exception:
        pass
    return None

class AudioRecorder:
    """Opt-in microphone: no stream exists outside start()..stop().

    (An earlier version held a silent "warmup" stream open 24/7 to
    pre-negotiate Bluetooth HFP. Removed for privacy — the mic must be
    verifiably off when not dictating. Use input_device="builtin" to keep
    recording start instant with AirPods connected.)"""

    def __init__(self):
        self._frames     = []
        self._stream     = None
        self._lock       = threading.Lock()
        self._chunk_cb   = None
        self._silence_ct = 0
        self._speaking   = False
        self._active     = False  # True only when actually recording

    def set_chunk_callback(self, fn):
        self._chunk_cb = fn

    def start(self):
        with self._lock:
            self._frames = []; self._silence_ct = 0
            self._speaking = False; self._active = True
            # Refresh PortAudio's device cache. PortAudio enumerates devices
            # once at Pa_Initialize, so after AirPods (or any input) drops and
            # rejoins, stale indices/handles linger and recordings come up
            # empty until the process is restarted.
            try:
                sd._terminate(); sd._initialize()
            except Exception as e:
                print(f"[voicepad] PortAudio refresh failed ({e})")
            dev = resolve_input_device()
            try:
                info = sd.query_devices(dev, 'input')
                print(f"[voicepad] recording: {info['name']}")
            except Exception:
                pass
            self._stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS,
                dtype=DTYPE, callback=self._cb, blocksize=512, device=dev)
            self._stream.start()

    def _cb(self, indata, *_):
        chunk = indata.copy().flatten()
        rms   = float(np.sqrt(np.mean(chunk ** 2)))
        with self._lock:
            self._frames.append(chunk)
            if rms > SILENCE_RMS_THRESHOLD:
                self._speaking = True; self._silence_ct = 0
            elif self._speaking:
                self._silence_ct += 1
                if self._silence_ct >= SILENCE_FRAMES_NEEDED:
                    if len(self._frames) >= MIN_CHUNK_FRAMES:
                        audio = np.concatenate(self._frames)
                        self._frames = []; self._silence_ct = 0; self._speaking = False
                        if self._chunk_cb:
                            threading.Thread(target=self._chunk_cb,
                                args=(audio,), daemon=True).start()

    def flush(self):
        with self._lock:
            self._active = False
            if not self._frames: return np.array([], dtype=DTYPE)
            audio = np.concatenate(self._frames)
            self._frames = []; self._silence_ct = 0; self._speaking = False
        return audio

    def stop(self):
        # Detach the stream under the lock, but DO NOT hold the lock
        # through stream.stop(). PortAudio's
        # stop() blocks until the in-flight _cb returns, and _cb itself
        # acquires self._lock — holding it here would deadlock the audio
        # thread against the main thread, which presents as the process
        # hanging and getting SIGKILLed by macOS ("Python quit unexpectedly").
        with self._lock:
            stream = self._stream
            self._stream = None
            self._active = False
        if stream:
            try: stream.stop(); stream.close()
            except Exception: pass
        # After stream.stop() returns, _cb will not fire again — safe to
        # reacquire the lock and drain the captured frames.
        with self._lock:
            if self._frames:
                audio = np.concatenate(self._frames)
            else:
                audio = np.array([], dtype=DTYPE)
            self._frames = []; self._silence_ct = 0; self._speaking = False
        # mic is now fully closed — nothing is reopened until the next start()
        return audio

    def rms(self):
        with self._lock:
            if not self._frames: return 0.0
            return float(np.sqrt(np.mean(self._frames[-1] ** 2)))

# ══════════════════════════════════════════════════════════════════════════════
# AppKit UI
# ══════════════════════════════════════════════════════════════════════════════
if HAVE_APPKIT:

    class VoiceView(NSView):
        def initWithFrame_(self, frame):
            self = objc.super(VoiceView, self).initWithFrame_(frame)
            if self is None: return None
            self.wave_vals = [0.0] * BARS
            self.dot_state = DOT_STATE_IDLE
            self.drag_pt   = None
            self.anim_tick = 0
            return self

        def isFlipped(self): return True

        def mouseDown_(self, e):   self.drag_pt = e.locationInWindow()
        def mouseDragged_(self, e):
            if not self.drag_pt: return
            win = self.window(); f = win.frame()
            c = e.locationInWindow()
            win.setFrameOrigin_(NSMakePoint(
                f.origin.x + c.x - self.drag_pt.x,
                f.origin.y + c.y - self.drag_pt.y))
        def mouseUp_(self, _): self.drag_pt = None

        def drawRect_(self, rect):
            # Capsule background with a mode-colored rim. Inset so the rim
            # stroke isn't clipped at the view edge.
            inset = 1.5
            path = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                NSMakeRect(inset, inset, W - inset * 2, H - inset * 2),
                RADIUS, RADIUS)
            ns_c(*BG).setFill()
            path.fill()
            ns_c(*BORDER).setStroke()
            path.setLineWidth_(1.0)
            path.stroke()
            if self.dot_state == DOT_STATE_IDLE:
                self.vp_draw_dots()
            else:
                self.vp_draw_waveform()

        def vp_draw_waveform(self):
            pad   = 16
            avail = W - pad * 2
            slot  = avail / BARS
            bw    = max(2.0, slot * 0.46)
            ym    = H / 2
            done  = (self.dot_state == DOT_STATE_DONE)
            for i, v in enumerate(self.wave_vals):
                x  = pad + i * slot + slot / 2
                hh = max(2.5, v * (H - 14) * 0.95)
                if done:
                    col = ns_c(*DOT_GRN)
                else:
                    col = ns_c(*WAVE_A) if v > 0.07 else ns_c(*WAVE_D)
                col.setFill()
                r = min(bw / 2, hh / 2, 3.0)
                NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                    NSMakeRect(x - bw / 2, ym - hh / 2, bw, hh), r, r).fill()

        def vp_draw_dots(self):
            # three bouncing dots, centered — "processing"
            dot_r   = 2.8
            spacing = 9.0
            cx0     = W / 2 - spacing
            self.anim_tick += 1
            for di in range(3):
                phase  = (self.anim_tick - di * 7) % 21
                rise   = max(0.0, math.sin(phase / 21.0 * math.pi))
                alpha  = 0.30 + 0.55 * rise
                cx = cx0 + di * spacing
                cy = H / 2 - rise * 3.0
                ns_c(DOT_GREY[0], DOT_GREY[1], DOT_GREY[2], alpha).setFill()
                NSBezierPath.bezierPathWithOvalInRect_(
                    NSMakeRect(cx - dot_r, cy - dot_r,
                               dot_r * 2, dot_r * 2)).fill()

        def set_wave(self, v):
            self.wave_vals = v; self.setNeedsDisplay_(True)

        def set_state(self, dot):
            self.dot_state = dot
            self.setNeedsDisplay_(True)

    class AppKitUI(NSObject):
        def init(self):
            self = objc.super(AppKitUI, self).init()
            if self is None: return None
            self._q      = queue.Queue()
            self._ctrl   = None
            self._tick_n = 0
            return self

        def _panel_screen(self):
            # The screen the user is actually looking at — the one with the
            # mouse cursor. mainScreen() for a background app with no key
            # window is just screens()[0], so on multi-monitor setups the
            # panel could appear on a display the user wasn't watching.
            try:
                mouse = NSEvent.mouseLocation()
                for s in NSScreen.screens():
                    if NSPointInRect(mouse, s.frame()):
                        return s
            except Exception:
                pass
            return NSScreen.mainScreen() or NSScreen.screens()[0]

        def _panel_origin(self):
            # Hug the very bottom edge of the screen, centered — Wispr Flow
            # style. frame() (not visibleFrame) so the pill sits flush with
            # the physical bottom; at our window level it floats above the
            # Dock anyway. Origin includes the screen's global offset so
            # multi-monitor setups place it on the correct screen.
            sf = self._panel_screen().frame()
            x  = sf.origin.x + (sf.size.width - W) / 2
            y  = sf.origin.y + 4
            return NSMakePoint(x, y)

        def _apply_window_traits(self):
            # NSScreenSaverWindowLevel (1000) sits above pop-up menus and
            # every normal app window. NSPopUpMenuWindowLevel (101) was still
            # occasionally topped by other windows at the same level (menus,
            # pickers, some overlay apps), and NSStatusWindowLevel (25) was
            # covered by fullscreen Chrome.
            self.win.setLevel_(NSScreenSaverWindowLevel)
            self.win.setCollectionBehavior_(
                NSWindowCollectionBehaviorCanJoinAllSpaces |
                NSWindowCollectionBehaviorStationary |
                NSWindowCollectionBehaviorFullScreenAuxiliary)

        def _make_window(self):
            origin = self._panel_origin()
            win = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
                NSMakeRect(origin.x, origin.y, W, H),
                NSWindowStyleMaskBorderless | NSNonactivatingPanelMask,
                NSBackingStoreBuffered, False)
            win.setOpaque_(False)
            win.setBackgroundColor_(NSColor.clearColor())
            win.setAlphaValue_(0.97)
            win.setHasShadow_(True)
            win.setHidesOnDeactivate_(False)
            self.view = VoiceView.alloc().initWithFrame_(NSMakeRect(0, 0, W, H))
            win.setContentView_(self.view)
            self.win = win
            self._apply_window_traits()
            # Observe screen reconfig + wake so the panel survives without a
            # process restart. Both can drop level/collection behavior or
            # leave the panel mispositioned on a now-unplugged screen.
            NSNotificationCenter.defaultCenter(
            ).addObserver_selector_name_object_(
                self, "screenParamsChanged:",
                "NSApplicationDidChangeScreenParametersNotification", None)
            NSWorkspace.sharedWorkspace().notificationCenter(
            ).addObserver_selector_name_object_(
                self, "didWake:",
                "NSWorkspaceDidWakeNotification", None)

        def screenParamsChanged_(self, _):
            self._apply_window_traits()
            try:
                self.win.setFrameOrigin_(self._panel_origin())
                if self.win.isVisible():
                    self.win.orderFrontRegardless()
            except Exception:
                pass

        def didWake_(self, _):
            self._apply_window_traits()

        def applicationShouldTerminate_(self, _):
            # Intercept the AppleEvent-Quit path (e.g. when Hammerspoon's
            # task termination or any AE Quit reaches us) and exit before
            # NSApplication's clean shutdown invokes __cxa_finalize_ranges,
            # which triggers libggml-metal's abort(). _hard_exit never
            # returns, so the NSTerminateReply value below is unreachable.
            _hard_exit()
            return 0

        def is_visible(self):
            try: return bool(self.win.isVisible())
            except: return False

        def _pump(self):
            # Drain the queue on the main thread NOW instead of waiting for
            # the next 33ms timer tick. App Nap defers background-app timers
            # by seconds (or indefinitely), which is exactly when the panel
            # "didn't pop up" after switching windows — the show command was
            # sitting in the queue waiting for a napped timer.
            self.performSelectorOnMainThread_withObject_waitUntilDone_(
                "tick:", None, False)

        def q_show(self):          self._q.put(("show",)); self._pump()
        def q_hide(self):          self._q.put(("hide",)); self._pump()
        def q_wave(self, v):       self._q.put(("wave", v))
        def q_dot(self, d):        self._q.put(("dot", d)); self._pump()

        def tick_(self, _):
            while True:
                try:
                    msg = self._q.get_nowait()
                except queue.Empty:
                    break
                # Isolate each message: one failure (e.g. a transient screen
                # error during display reconfig) must not eat a queued show.
                try:
                    cmd = msg[0]
                    if cmd == "show":
                        # orderOut first so the WindowServer fully re-evaluates
                        # stacking/Space membership on re-show. Re-asserting on
                        # an already-visible-but-stuck window doesn't always
                        # un-stick it, which is why the panel previously
                        # required a full restart to recover.
                        self.win.orderOut_(None)
                        self.win.setFrameOrigin_(self._panel_origin())
                        self._apply_window_traits()
                        self.win.orderFrontRegardless()
                    elif cmd == "hide":
                        # Drop stale hides. _autohide/_finalize threads check
                        # state then enqueue hide; a new recording can start
                        # between the check and the enqueue, leaving the queue
                        # as [show, hide] — the hide buries the fresh panel
                        # while the mic keeps recording ("it hears me but no
                        # pill"). State is only trustworthy here, on the main
                        # thread, where commands are processed in order.
                        if self._ctrl is None or self._ctrl._state == "idle":
                            self.win.orderOut_(None)
                    elif cmd == "wave": self.view.set_wave(msg[1])
                    elif cmd == "dot":  self.view.set_state(msg[1])
                except Exception as e:
                    print(f"[voicepad] ui cmd {msg[0]} failed: {e}")
            try:
                self._tick_n += 1
                visible = self.win.isVisible()
                busy = self._ctrl is not None and self._ctrl._state != "idle"
                if visible:
                    # keep bottom bar animating while processing
                    if self.view.dot_state == DOT_STATE_IDLE:
                        self.view.setNeedsDisplay_(True)
                    # Watchdog: re-assert level + ordering ~1x/sec while
                    # visible. orderFrontRegardless on a non-activating panel
                    # never steals focus, so this is free insurance against
                    # anything ordering itself above us mid-recording.
                    if self._tick_n % 30 == 0:
                        self._apply_window_traits()
                        self.win.orderFrontRegardless()
                elif busy:
                    # Self-heal: recording/processing but the panel is hidden
                    # — a show was eaten (transient AppKit error during screen
                    # reconfig) or buried by a stale hide. Re-show every tick
                    # until it sticks; once visible this branch stops running.
                    self.win.setFrameOrigin_(self._panel_origin())
                    self._apply_window_traits()
                    self.win.orderFrontRegardless()
            except Exception:
                pass

        def run(self):
            app = NSApplication.sharedApplication()
            # When running inside the py2app bundle, LSUIElement=YES in
            # Info.plist already hides us from the Dock / menu bar / Cmd+Tab,
            # AND Prohibited (2) at runtime breaks Metal access (mlx fails to
            # initialize its extension). So we only set Prohibited for the
            # raw-script invocation; the .app uses Accessory (1, the default
            # under LSUIElement) which keeps Metal working.
            if os.environ.get('RESOURCEPATH'):
                # py2app bundle — Info.plist's LSUIElement handles visibility.
                pass
            else:
                # Raw script (e.g. Hammerspoon launching python directly):
                # Prohibited keeps "Python" out of the menu bar.
                app.setActivationPolicy_(2)
            # Become the NSApp delegate so applicationShouldTerminate_ fires
            # on AE Quit — this is the path the libggml-metal abort travels
            # through (see crash trace: _handleAEQuit → terminate: → exit).
            app.setDelegate_(self)
            # Opt out of App Nap. As a windowless LSUIElement agent we're a
            # prime nap candidate the moment the user is in another app —
            # napping defers our NSTimer (the only thing that drains the UI
            # queue) so the panel showed late or not at all. LatencyCritical
            # marks our timers as real-time; AllowingIdleSystemSleep keeps us
            # from blocking the machine's normal sleep. The token must stay
            # referenced or the assertion is dropped.
            self._activity = NSProcessInfo.processInfo(
            ).beginActivityWithOptions_reason_(
                NSActivityUserInitiatedAllowingIdleSystemSleep |
                NSActivityLatencyCritical,
                "VoicePad hotkey must show the panel instantly")
            self._make_window()
            timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
                0.033, self, "tick:", None, True)
            # Common modes so the timer keeps firing during event tracking
            # (e.g. while the user drags the panel) — default-mode-only
            # timers stall there and freeze the waveform/queue.
            NSRunLoop.mainRunLoop().addTimer_forMode_(timer, NSRunLoopCommonModes)
            app.run()

# ══════════════════════════════════════════════════════════════════════════════
# Tkinter fallback
# ══════════════════════════════════════════════════════════════════════════════
else:
    import tkinter as tk
    FB_BARS = BARS

    class TkUI:
        BW = 480; BH = 88; WH = 54

        def __init__(self):
            self.root = tk.Tk(); self.root.withdraw()
            self.root.overrideredirect(True)
            self.root.attributes("-topmost", True)
            self.root.configure(bg="#1C1C20")
            self._q        = queue.Queue()
            self.wave_vals = [0.0] * FB_BARS
            self.dot_state = DOT_STATE_IDLE
            self._dx = self._dy = 0
            self._ctrl_ref = None
            self._build(); self._place(); self._poll()

        def _build(self):
            outer = tk.Frame(self.root, bg="#1C1C20",
                             highlightbackground="#3A3A40", highlightthickness=1)
            outer.pack(fill="both", expand=True)
            self.canvas = tk.Canvas(outer, width=self.BW-2, height=self.WH,
                                    bg="#1C1C20", highlightthickness=0)
            self.canvas.pack()
            self.canvas.bind("<ButtonPress-1>",
                lambda e: (setattr(self,'_dx',e.x_root), setattr(self,'_dy',e.y_root)))
            self.canvas.bind("<B1-Motion>", self._drag)
            tk.Frame(outer, bg="#383840", height=1).pack(fill="x")
            bot = tk.Frame(outer, bg="#1C1C20"); bot.pack(fill="x", padx=10, pady=6)
            self.dot = tk.Label(bot, text="■", font=("Helvetica", 9),
                                fg="#52525A", bg="#1C1C20")
            self.dot.pack(side="left", padx=(0,5))
            self.root.geometry(f"{self.BW}x{self.BH}")

        def _drag(self, e):
            x = self.root.winfo_x() + e.x_root - self._dx
            y = self.root.winfo_y() + e.y_root - self._dy
            self.root.geometry(f"+{x}+{y}")
            self._dx, self._dy = e.x_root, e.y_root

        def _place(self):
            sw = self.root.winfo_screenwidth(); sh = self.root.winfo_screenheight()
            self.root.geometry(f"{self.BW}x{self.BH}+{(sw-self.BW)//2}+{sh-self.BH-88}")

        def _poll(self):
            try:
                while True:
                    msg = self._q.get_nowait(); cmd = msg[0]
                    if cmd == "show":   self.root.deiconify(); self.root.lift()
                    elif cmd == "hide":
                        # Drop stale hides (see AppKitUI.tick_): a hide
                        # enqueued just as a new recording started must not
                        # bury the fresh show.
                        if self._ctrl_ref is None or self._ctrl_ref._state == "idle":
                            self.root.withdraw()
                    elif cmd == "wave": self.wave_vals = msg[1]; self._draw()
                    elif cmd == "dot":
                        self.dot_state = msg[1]
                        c = {"rec":"#EB4040","done":"#38D068"}.get(msg[1],"#52525A")
                        self.dot.config(fg=c)
            except queue.Empty: pass
            # Self-heal: recording/processing but hidden — re-show until it
            # sticks (mirrors the AppKit watchdog).
            if (self._ctrl_ref is not None and self._ctrl_ref._state != "idle"
                    and not self.is_visible()):
                self.root.deiconify(); self.root.lift()
            self.root.after(30, self._poll)

        def _draw(self):
            c = self.canvas; c.delete("all")
            avail = self.BW - 2 - 36; slot = avail / FB_BARS
            bw = max(2, int(slot * 0.44)); ym = self.WH // 2
            for i, v in enumerate(self.wave_vals):
                x = 18 + int(i * slot + slot / 2)
                h = max(2, int(v * (self.WH - 10) * 0.94))
                fill = "#DCDCE0" if v > 0.07 else "#303034"
                c.create_rectangle(x-bw, ym-h//2, x+bw, ym+h//2, fill=fill, outline="")

        def is_visible(self):
            return self.root.state() == "normal"

        def q_show(self):      self._q.put(("show",))
        def q_hide(self):      self._q.put(("hide",))
        def q_wave(self, v):   self._q.put(("wave", v))
        def q_dot(self, d):    self._q.put(("dot", d))
        def run(self):         self.root.mainloop()

# ══════════════════════════════════════════════════════════════════════════════
# Controller
# ══════════════════════════════════════════════════════════════════════════════
class VoicePad:
    def __init__(self):
        if HAVE_APPKIT:
            self.ui = AppKitUI.alloc().init()
            self.ui._ctrl = self
        else:
            self.ui = TkUI()
            self.ui._ctrl_ref = self
        self.recorder    = AudioRecorder()
        self.recorder.set_chunk_callback(self._on_silence_chunk)
        self._state      = "idle"
        self._transcript = []
        self._chunk_lock = threading.Lock()
        self._smooth     = [0.0] * BARS
        self._targets    = [0.0] * BARS

    def _on_press(self, key):
        if key == HOTKEY:
            if self._state == "idle":        self._start()
            elif self._state == "recording": self._stop()
            # ignore the hotkey during processing — prevents ghost triggers on wake
            return
        if key == kb_input.Key.esc:
            self._cancel()   # dismiss in any state
            return

    def _on_release(self, key):
        pass

    def _start(self):
        self._state = "recording"
        with self._chunk_lock:
            self._transcript = []
        self.ui.q_show()
        self.ui.q_dot(DOT_STATE_REC)
        self.recorder.start()
        threading.Thread(target=self._wave_loop, daemon=True).start()

    def _stop(self):
        self._state = "processing"
        self.ui.q_dot(DOT_STATE_IDLE)
        audio = self.recorder.stop()
        threading.Thread(target=self._finalize, args=(audio,), daemon=True).start()

    def _cancel(self):
        try:
            if self._state == "recording":
                self.recorder.stop()
        except Exception:
            pass
        self._state = "idle"
        self.ui.q_wave([0.0] * BARS)
        self.ui.q_dot(DOT_STATE_IDLE)
        self.ui.q_hide()

    def _on_silence_chunk(self, audio):
        if audio.size < SAMPLE_RATE * 0.3:
            return
        text = transcribe(audio)
        if text:
            with self._chunk_lock:
                self._transcript.append(text)

    def _finalize(self, audio):
        if audio.size >= SAMPLE_RATE * 0.3:
            text = transcribe(audio)
            if text:
                with self._chunk_lock:
                    self._transcript.append(text)
        with self._chunk_lock:
            full_text = " ".join(self._transcript).strip()
        if not full_text:
            self.ui.q_dot(DOT_STATE_IDLE)
            time.sleep(0.3); self._state = "idle"; self.ui.q_hide(); return
        self._deliver(full_text)

    def _deliver(self, raw):
        pyperclip.copy(raw)
        paste_to_frontmost()
        self.ui.q_dot(DOT_STATE_DONE)
        self._state = "idle"
        threading.Thread(target=self._autohide, daemon=True).start()

    def _autohide(self):
        time.sleep(AUTOHIDE_SEC)
        if self._state == "idle":
            self.ui.q_wave([0.0] * BARS)
            self.ui.q_dot(DOT_STATE_IDLE)
            self.ui.q_hide()

    def _wave_loop(self):
        smooth   = [0.0] * BARS
        rms_hist = [0.0] * 6
        hi = 0

        while self._state == "recording":
            rms = self.recorder.rms()
            rms_hist[hi % len(rms_hist)] = rms
            hi += 1
            avg_rms = sum(rms_hist) / len(rms_hist)

            for i in range(BARS):
                lag    = BAR_LAG[i]
                s_rms  = avg_rms * (1 - lag) + rms * lag
                env    = BAR_ENV[i]
                target = min(1.0, s_rms * 28 * env + 0.03)
                if target > smooth[i]:
                    smooth[i] = smooth[i] * 0.35 + target * 0.65
                else:
                    smooth[i] = smooth[i] * 0.82 + target * 0.18

            self.ui.q_wave(list(smooth))
            time.sleep(0.033)

        for _ in range(14):
            smooth = [v * 0.78 for v in smooth]
            self.ui.q_wave(list(smooth))
            time.sleep(0.033)
        self.ui.q_wave([0.0] * BARS)

    def run(self):
        threading.Thread(target=load_transcription_model, daemon=True).start()
        listener = kb_input.Listener(
            on_press=self._on_press, on_release=self._on_release)
        listener.daemon = True; listener.start()
        print(f"[voicepad] {CONFIG['hotkey']}=record | Esc=cancel")
        self.ui.run()

# ── launchd keep-alive (replaces Hammerspoon) ─────────────────────────────────
LAUNCH_AGENT_LABEL = "com.voicepad.agent"
LAUNCH_AGENT_PATH  = os.path.expanduser(
    f"~/Library/LaunchAgents/{LAUNCH_AGENT_LABEL}.plist")

def _launch_agent_program():
    if os.environ.get('RESOURCEPATH'):
        # running inside the .app bundle — point launchd at the bundle binary
        contents = os.path.dirname(os.environ['RESOURCEPATH'])
        return [os.path.join(contents, 'MacOS', 'VoicePad')]
    return [sys.executable, os.path.abspath(__file__)]

def install_launch_agent():
    args = "\n".join(f"    <string>{p}</string>"
                     for p in _launch_agent_program())
    plist = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>             <string>{LAUNCH_AGENT_LABEL}</string>
  <key>ProgramArguments</key>
  <array>
{args}
  </array>
  <key>RunAtLoad</key>         <true/>
  <key>KeepAlive</key>         <true/>
  <key>StandardOutPath</key>   <string>/tmp/voicepad.log</string>
  <key>StandardErrorPath</key> <string>/tmp/voicepad.log</string>
</dict>
</plist>
"""
    os.makedirs(os.path.dirname(LAUNCH_AGENT_PATH), exist_ok=True)
    with open(LAUNCH_AGENT_PATH, "w") as f:
        f.write(plist)
    subprocess.run(["launchctl", "unload", LAUNCH_AGENT_PATH],
                   capture_output=True)
    subprocess.run(["launchctl", "load", LAUNCH_AGENT_PATH],
                   capture_output=True)
    print(f"[voicepad] launch agent installed: {LAUNCH_AGENT_PATH}")
    print("[voicepad] VoicePad now starts at login and restarts if it exits.")
    print("[voicepad] Remove any Hammerspoon keep-alive to avoid double launches.")

def uninstall_launch_agent():
    subprocess.run(["launchctl", "unload", LAUNCH_AGENT_PATH],
                   capture_output=True)
    try:
        os.remove(LAUNCH_AGENT_PATH)
    except FileNotFoundError:
        pass
    print("[voicepad] launch agent removed")

if __name__ == "__main__":
    if "--install-agent" in sys.argv:
        install_launch_agent(); sys.exit(0)
    if "--uninstall-agent" in sys.argv:
        uninstall_launch_agent(); sys.exit(0)
    VoicePad().run()

#!/usr/bin/env python3
"""
VoicePad v6
Right Cmd = start/stop  |  Tab = cycle mode (only when visible)  |  Esc = cancel
Status dot: grey (idle) -> red (recording) -> green (done)
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
    from Quartz import (
        CGEventTapCreate, CGEventTapEnable,
        CFMachPortCreateRunLoopSource, CFRunLoopAddSource,
        kCGSessionEventTap, kCGHeadInsertEventTap,
        kCGEventKeyDown, CGEventGetIntegerValueField,
        kCGKeyboardEventKeycode,
    )
    from CoreFoundation import CFRunLoopGetMain, kCFRunLoopCommonModes
    HAVE_QUARTZ = True
except Exception as _qe:
    print(f"[voicepad] Quartz unavailable, tab suppression disabled")
    HAVE_QUARTZ = False

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
        NSNonactivatingPanelMask,
        NSPanel, NSWindowCollectionBehaviorCanJoinAllSpaces,
        NSWindowCollectionBehaviorStationary,
        NSWindowCollectionBehaviorFullScreenAuxiliary,
        NSTimer, NSScreen, NSWorkspace,
    )
    from Foundation import NSObject, NSNotificationCenter
    HAVE_APPKIT = True
except Exception as e:
    print(f"[voicepad] AppKit unavailable ({e}), using tkinter")
    HAVE_APPKIT = False

# ── config ────────────────────────────────────────────────────────────────────
SAMPLE_RATE  = 16_000
CHANNELS     = 1
DTYPE        = "float32"
OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma4:e2b"   # Any model you have pulled; gemma4:e2b is small and fast
MLX_MODEL    = "mlx-community/whisper-small.en-mlx"
AUTOHIDE_SEC = 1.2

MODES      = ["raw", "email", "notes", "math"]

# ── vocab config ──────────────────────────────────────────────────────────────
import json, re
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

# ── context awareness ─────────────────────────────────────────────────────────
# Bundle IDs → mode auto-selection
NOTES_BUNDLE_IDS = {
    "com.apple.Notes",
}

GMAIL_URL_PATTERNS = [
    "mail.google.com",
    "gmail.com",
]

def get_frontmost_bundle_id() -> str:
    """Get the bundle ID of the currently focused app via AppleScript."""
    try:
        result = subprocess.run(
            ["osascript", "-e",
             'tell application "System Events" to return bundle identifier of first process whose frontmost is true'],
            capture_output=True, text=True, timeout=1)
        return result.stdout.strip()
    except Exception:
        return ""

def get_chrome_active_url() -> str:
    """Get the URL of the active Chrome tab via AppleScript."""
    try:
        result = subprocess.run(
            ["osascript", "-e",
             'tell application "Google Chrome" to return URL of active tab of front window'],
            capture_output=True, text=True, timeout=1)
        return result.stdout.strip()
    except Exception:
        return ""

def detect_context_mode() -> int | None:
    """
    Returns a mode index if context matches, None if no auto-switch needed.
    Only auto-switches to email (Gmail) or notes (Notes/Raycast).
    """
    bundle = get_frontmost_bundle_id()

    # Notes app or Raycast → notes mode
    if bundle in NOTES_BUNDLE_IDS:
        return MODES.index("notes")

    # Chrome → check if Gmail is active tab
    if bundle in ("com.google.Chrome", "org.chromium.Chromium"):
        url = get_chrome_active_url()
        if any(p in url for p in GMAIL_URL_PATTERNS):
            return MODES.index("email")

    return None
MODE_SHORT = {"raw": "Default", "email": "Email", "notes": "Notes", "math": "Math"}

SYSTEM_PROMPTS = {
    "raw": None,
    "email": (
        "You edit a voice-dictated email. Output ONLY the email body — no subject, no preamble, no explanation.\n\n"
        "Format:\n"
        "  [greeting line]\n"
        "  \n"
        "  [body, 1-3 short paragraphs]\n"
        "  \n"
        "  [closing line]\n\n"
        "Rules:\n"
        "1. Fix grammar, run-ons, and filler words (uh, um, like, you know, \"wait actually let me start over\").\n"
        "2. Match the sender's tone EXACTLY — do not formalize. Look at how they greeted you:\n"
        "   - Casual openers (\"yo\", \"hey\", \"dude\", \"lol\", \"sup\") → casual greeting (\"Yo\", \"Hey\") and casual closing (\"Cheers,\", \"Later,\", \"Talk soon,\").\n"
        "   - Neutral openers (\"hi\", \"hello\") → \"Hi [name],\" and \"Thanks,\".\n"
        "   - Never use \"Dear\".\n"
        "3. Keep the sender's exact words and meaning. Do not add content they did not say.\n"
        "4. For one-line messages (heads-ups, quick asks), keep the body to one short sentence.\n"
    ),
    "notes": (
        "Convert voice-dictated text to bullet-point notes. Output ONLY the notes — no preamble, no explanation.\n\n"
        "Rules:\n"
        "1. One idea per bullet (use `* `).\n"
        "2. Sub-bullets (indented `  * `) for details under a parent idea.\n"
        "3. ADD a **bold header** at the top when EITHER is true:\n"
        "   - The input starts with a topical phrase (e.g. \"thoughts on X\", \"X takeaways\", \"ideas for Y\", \"plan for Z\", \"prep for X\", \"notes from X\"). Use that phrase as the header.\n"
        "   - The notes contain 3 or more distinct topics.\n"
        "4. PRESERVE meaning exactly. NEVER drop or invert negatives or instructions like \"skip\", \"don't\", \"no\", \"avoid\". If the speaker said \"skip X\", the note must say \"Skip X\" — not \"do X\".\n"
        "5. Fix obvious transcription errors only. Do not invent, rephrase, or add content.\n\n"
        "Examples:\n\n"
        "Input: design review takeaways the new dashboard got positive feedback we need to fix mobile loading\n"
        "Output:\n"
        "**Design review takeaways**\n"
        "* New dashboard got positive feedback\n"
        "* Need to fix mobile loading\n\n"
        "Input: buy milk and eggs\n"
        "Output:\n"
        "* Buy milk\n"
        "* Buy eggs\n"
    ),
    "math": (
        "Convert dictated math to LaTeX. Output ONLY wrapped LaTeX — no prose, no code fences, no explanation.\n\n"
        "WRAPPING (CRITICAL — every output MUST be wrapped):\n"
        "  - Use $...$ for short expressions with no fractions, integrals, sums, products, limits, derivatives, matrices, or multi-line systems.\n"
        "  - Use $$...$$ for everything else.\n"
        "  - NEVER output bare LaTeX without $ or $$ wrappers.\n\n"
        "Translation cheatsheet:\n"
        "  \"x squared\" → x^2 ;  \"x to the n\" → x^n\n"
        "  \"alpha/beta/pi/theta/...\" → \\alpha, \\beta, \\pi, \\theta\n"
        "  \"a over b\" → \\frac{a}{b}\n"
        "  \"square root of x\" → \\sqrt{x}\n"
        "  \"integral from a to b of f dx\" → \\int_a^b f\\,dx\n"
        "  \"sum from i=1 to n of f\" → \\sum_{i=1}^{n} f\n"
        "  \"limit as x approaches L of f\" → \\lim_{x \\to L} f\n"
        "  \"derivative of f with respect to x\" → \\frac{df}{dx}\n\n"
        "Examples:\n"
        "  Input: x plus y\n"
        "  Output: $x + y$\n\n"
        "  Input: x squared plus y squared equals r squared\n"
        "  Output: $x^2 + y^2 = r^2$\n\n"
        "  Input: alpha plus beta equals pi over two\n"
        "  Output: $$\\alpha + \\beta = \\frac{\\pi}{2}$$\n\n"
        "  Input: integral from zero to infinity of e to the minus x dx\n"
        "  Output: $$\\int_0^\\infty e^{-x}\\,dx$$\n\n"
        "  Input: sum from i equals 1 to n of i equals n times n plus 1 over 2\n"
        "  Output: $$\\sum_{i=1}^{n} i = \\frac{n(n+1)}{2}$$\n\n"
        "  Input: limit as x approaches infinity of one over x equals zero\n"
        "  Output: $$\\lim_{x \\to \\infty} \\frac{1}{x} = 0$$\n\n"
        "  Input: two x plus three y equals seven and four x minus y equals two\n"
        "  Output: $$2x + 3y = 7$$\n          $$4x - y = 2$$\n\n"
        "Now convert the user's input. Output only the wrapped LaTeX."
    ),
}

# ── window geometry ───────────────────────────────────────────────────────────
W        = 480
WAVE_H   = 54
BOTTOM_H = 34
H        = WAVE_H + BOTTOM_H
RADIUS   = 12
BARS     = 32

# ── palette ───────────────────────────────────────────────────────────────────
BG       = (0.11,  0.11,  0.125, 0.96)
BORDER   = (0.24,  0.24,  0.27,  0.8)
DIV      = (0.22,  0.22,  0.25,  0.6)
WAVE_A   = (0.86,  0.86,  0.88,  1.0)
WAVE_D   = (0.19,  0.19,  0.21,  1.0)
DOT_GREY = (0.32,  0.32,  0.36,  1.0)
DOT_RED  = (0.92,  0.25,  0.25,  1.0)
DOT_GRN  = (0.22,  0.82,  0.46,  1.0)
TEXT     = (0.72,  0.72,  0.76,  1.0)
TEXT_DIM = (0.36,  0.36,  0.40,  1.0)
KEY_BG   = (0.22,  0.22,  0.26,  1.0)
KEY_FG   = (0.82,  0.82,  0.86,  1.0)
SEP      = (0.28,  0.28,  0.32,  0.8)

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

def transcribe(audio_np):
    if _mlx_ok:
        try:
            import mlx_whisper
            r = mlx_whisper.transcribe(audio_np, path_or_hf_repo=MLX_MODEL, language="en")
            text = r.get("text", "").strip()
            if _is_hallucination(text):
                return ""
            return apply_vocab(text)
        except Exception as e:
            print(f"[voicepad] mlx error: {e}")
    if _fw_model:
        segs, _ = _fw_model.transcribe(audio_np, language="en", beam_size=5, vad_filter=True)
        text = " ".join(s.text.strip() for s in segs).strip()
        if _is_hallucination(text):
            return ""
        return apply_vocab(text)
    return ""

def ollama_post(raw, mode):
    prompt = SYSTEM_PROMPTS.get(mode)
    if not prompt:
        return raw
    try:
        r = requests.post(OLLAMA_URL, json={
            "model": OLLAMA_MODEL, "system": prompt,
            "prompt": raw, "stream": False,
        }, timeout=60)
        r.raise_for_status()
        return r.json().get("response", raw).strip()
    except Exception as e:
        return f"[Ollama error: {e}]\n\n{raw}"

# ── silence detection ─────────────────────────────────────────────────────────
SILENCE_RMS_THRESHOLD = 0.008
SILENCE_FRAMES_NEEDED = 45
MIN_CHUNK_FRAMES      = 15

class AudioRecorder:
    def __init__(self):
        self._frames     = []
        self._stream     = None
        self._warmup     = None   # always-open stream to pre-negotiate BT HFP
        self._lock       = threading.Lock()
        self._chunk_cb   = None
        self._silence_ct = 0
        self._speaking   = False
        self._active     = False  # True only when actually recording

    def set_chunk_callback(self, fn):
        self._chunk_cb = fn

    def prewarm(self):
        """Open a silent input stream immediately on launch.
        This forces Bluetooth HFP negotiation to happen now,
        so when the user presses the hotkey the mic is already live."""
        try:
            with self._lock:
                # Don't disturb an active recording, and don't race with a
                # concurrent start() that's opening the recording stream.
                if self._active:
                    return
                if self._warmup:
                    try: self._warmup.stop(); self._warmup.close()
                    except Exception: pass
                    self._warmup = None
                # device=None always uses whatever macOS currently has as default input
                # This correctly picks AirPods if they're set in System Settings -> Sound -> Input
                info = sd.query_devices(None, 'input')
                print(f"[voicepad] mic pre-warmed: {info['name']}")
                self._warmup = sd.InputStream(
                    samplerate=SAMPLE_RATE, channels=CHANNELS,
                    dtype=DTYPE, callback=self._warmup_cb, blocksize=512, device=None)
                self._warmup.start()
        except Exception as e:
            print(f"[voicepad] prewarm failed ({e}) — mic will have slight delay")

    def _warmup_cb(self, indata, *_):
        # discard all audio — we just need the stream open
        pass

    def start(self):
        with self._lock:
            self._frames = []; self._silence_ct = 0
            self._speaking = False; self._active = True
            # close warmup stream before refreshing PortAudio
            if self._warmup:
                try: self._warmup.stop(); self._warmup.close()
                except Exception: pass
                self._warmup = None
            # Refresh PortAudio's device cache. PortAudio enumerates devices
            # once at Pa_Initialize, so after AirPods (or any input) drops and
            # rejoins, device=None still resolves to the stale handle and
            # recordings come up empty until the process is restarted.
            try:
                sd._terminate(); sd._initialize()
            except Exception as e:
                print(f"[voicepad] PortAudio refresh failed ({e})")
            try:
                info = sd.query_devices(None, 'input')
                print(f"[voicepad] recording: {info['name']}")
            except Exception:
                pass
            # device=None = always use current macOS default input (AirPods, built-in, etc)
            self._stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS,
                dtype=DTYPE, callback=self._cb, blocksize=512, device=None)
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
        # Detach the stream under the lock so prewarm() can't race with us,
        # but DO NOT hold the lock through stream.stop(). PortAudio's
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
        # re-open warmup stream so next recording has no delay
        threading.Thread(target=self.prewarm, daemon=True).start()
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
            self.mode_idx  = 0
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
            ns_c(*BG).setFill()
            bg_path = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                NSMakeRect(0, 0, W, H), RADIUS, RADIUS)
            bg_path.fill()
            ns_c(*BORDER).setStroke()
            bg_path.setLineWidth_(0.75)
            bg_path.stroke()
            self.vp_draw_waveform()
            ns_c(*DIV).setStroke()
            d = NSBezierPath.bezierPath()
            d.moveToPoint_(NSMakePoint(0, WAVE_H))
            d.lineToPoint_(NSMakePoint(W, WAVE_H))
            d.setLineWidth_(0.5)
            d.stroke()
            self.vp_draw_bottom()

        def vp_draw_waveform(self):
            pad   = 18
            avail = W - pad * 2
            slot  = avail / BARS
            bw    = max(2.0, slot * 0.44)
            ym    = WAVE_H / 2
            for i, v in enumerate(self.wave_vals):
                x  = pad + i * slot + slot / 2
                hh = max(2.5, v * (WAVE_H - 12) * 0.96)
                col = ns_c(*WAVE_A) if v > 0.07 else ns_c(*WAVE_D)
                col.setFill()
                r = min(bw / 2, hh / 2, 3.0)
                NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                    NSMakeRect(x - bw / 2, ym - hh / 2, bw, hh), r, r).fill()

        def vp_draw_bottom(self):
            strip_mid = WAVE_H + BOTTOM_H / 2
            ty = strip_mid - 7
            dy = strip_mid - 4.5
            x  = 14.0

            is_proc = (self.dot_state == DOT_STATE_IDLE)

            if is_proc:
                # three bouncing dots animation
                dot_r   = 2.8
                spacing = 7.5
                self.anim_tick += 1
                for di in range(3):
                    phase  = (self.anim_tick - di * 7) % 21
                    rise   = max(0.0, math.sin(phase / 21.0 * math.pi))
                    alpha  = 0.30 + 0.55 * rise
                    offset = rise * 3.5
                    ns_c(DOT_GREY[0], DOT_GREY[1], DOT_GREY[2], alpha).setFill()
                    cx = x + di * spacing + dot_r
                    cy = dy + dot_r - offset
                    NSBezierPath.bezierPathWithOvalInRect_(
                        NSMakeRect(cx - dot_r, cy - dot_r,
                                   dot_r * 2, dot_r * 2)).fill()
                x += 3 * spacing + 6
            elif self.dot_state == DOT_STATE_REC:
                ns_c(*DOT_RED).setFill()
                NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                    NSMakeRect(x, dy, 8.0, 8.0), 2.2, 2.2).fill()
                x += 8.0 + 8
            else:
                ns_c(*DOT_GRN).setFill()
                NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                    NSMakeRect(x, dy, 8.0, 8.0), 2.2, 2.2).fill()
                x += 8.0 + 8

            lbl   = "Recording"  if self.dot_state == DOT_STATE_REC  else \
                    "Done"       if self.dot_state == DOT_STATE_DONE else \
                    "Processing"
            lbl_c = ns_c(*TEXT) if self.dot_state == DOT_STATE_REC else ns_c(*TEXT_DIM)
            self.vp_txt(lbl, x, ty, NSFont.systemFontOfSize_(11), lbl_c)
            x += self.vp_tw(lbl, NSFont.systemFontOfSize_(11)) + 12
            x  = self.vp_sep(x, ty)

            mode_lbl = MODE_SHORT[MODES[self.mode_idx]]
            self.vp_txt(mode_lbl, x, ty, NSFont.systemFontOfSize_(11), ns_c(*TEXT))
            x += self.vp_tw(mode_lbl, NSFont.systemFontOfSize_(11)) + 5
            x  = self.vp_keycap("Tab", x, ty) + 12
            x  = self.vp_sep(x, ty)

            self.vp_txt("Stop", x, ty, NSFont.systemFontOfSize_(11), ns_c(*TEXT))
            x += self.vp_tw("Stop", NSFont.systemFontOfSize_(11)) + 5
            x  = self.vp_keycap("Cmd", x, ty) + 12
            x  = self.vp_sep(x, ty)

            self.vp_txt("Cancel", x, ty, NSFont.systemFontOfSize_(11), ns_c(*TEXT))
            x += self.vp_tw("Cancel", NSFont.systemFontOfSize_(11)) + 5
            self.vp_keycap("Esc", x, ty)

        def vp_txt(self, s, x, y, font, color):
            NSString.stringWithString_(s).drawAtPoint_withAttributes_(
                NSMakePoint(x, y),
                {NSFontAttributeName: font,
                 NSForegroundColorAttributeName: color})

        def vp_tw(self, s, font):
            return NSString.stringWithString_(s).sizeWithAttributes_(
                {NSFontAttributeName: font}).width

        def vp_keycap(self, label, x, y):
            font = NSFont.monospacedSystemFontOfSize_weight_(9, 0.3)
            tw   = self.vp_tw(label, font)
            kw   = tw + 10; kh = 16; kr = 3.5; ky = y + 1
            ns_c(*KEY_BG).setFill()
            ns_c(0.35, 0.35, 0.40, 0.7).setStroke()
            cap = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                NSMakeRect(x, ky, kw, kh), kr, kr)
            cap.fill(); cap.setLineWidth_(0.5); cap.stroke()
            self.vp_txt(label, x + 5, ky + 2, font, ns_c(*KEY_FG))
            return x + kw

        def vp_sep(self, x, y):
            ns_c(*SEP).setStroke()
            p = NSBezierPath.bezierPath()
            p.moveToPoint_(NSMakePoint(x, y - 1))
            p.lineToPoint_(NSMakePoint(x, y + 13))
            p.setLineWidth_(0.5); p.stroke()
            return x + 9

        def set_wave(self, v):
            self.wave_vals = v; self.setNeedsDisplay_(True)

        def set_state(self, dot, mode):
            self.dot_state = dot; self.mode_idx = mode
            self.setNeedsDisplay_(True)

    class AppKitUI(NSObject):
        def init(self):
            self = objc.super(AppKitUI, self).init()
            if self is None: return None
            self._q    = queue.Queue()
            self._ctrl = None
            return self

        def _panel_origin(self):
            # visibleFrame() excludes menu bar / Dock; origin includes the
            # screen's global offset, so multi-monitor setups (and post-
            # unplug/replug states) place the panel on the correct screen
            # rather than off in dead space — which presented as the panel
            # being "behind everything".
            sf = NSScreen.mainScreen().visibleFrame()
            x  = sf.origin.x + (sf.size.width - W) / 2
            y  = sf.origin.y + 90
            return NSMakePoint(x, y)

        def _apply_window_traits(self):
            # NSPopUpMenuWindowLevel (101) is reliably above fullscreen-app
            # contents on every macOS version we care about; NSStatusWindowLevel
            # (25) was sometimes covered by fullscreen Chrome.
            self.win.setLevel_(NSPopUpMenuWindowLevel)
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

        def q_show(self):          self._q.put(("show",))
        def q_hide(self):          self._q.put(("hide",))
        def q_wave(self, v):       self._q.put(("wave", v))
        def q_dot(self, d, m):     self._q.put(("dot", d, m))

        def tick_(self, _):
            try:
                while True:
                    msg = self._q.get_nowait(); cmd = msg[0]
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
                    elif cmd == "hide": self.win.orderOut_(None)
                    elif cmd == "wave": self.view.set_wave(msg[1])
                    elif cmd == "dot":  self.view.set_state(msg[1], msg[2])
            except queue.Empty: pass
            # keep bottom bar animating while processing
            try:
                if self.view.dot_state == DOT_STATE_IDLE and self.win.isVisible():
                    self.view.setNeedsDisplay_(True)
            except Exception:
                pass

        def run(self):
            app = NSApplication.sharedApplication()
            # Prohibited (2): no Dock icon, no Cmd+Tab entry, and crucially
            # no "Python" in the menu bar when frontmost. Accessory (1) was
            # tried previously to fix panel-behind-fullscreen, but the other
            # fixes here (NSPopUpMenuWindowLevel + FullScreenAuxiliary +
            # screen-change/wake observers + orderOut→re-assert→front show
            # sequence) make Accessory unnecessary for that case.
            app.setActivationPolicy_(2)
            # Become the NSApp delegate so applicationShouldTerminate_ fires
            # on AE Quit — this is the path the libggml-metal abort travels
            # through (see crash trace: _handleAEQuit → terminate: → exit).
            app.setDelegate_(self)
            self._make_window()
            NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
                0.033, self, "tick:", None, True)
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
            self.mode_idx  = 0
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
            self.mode_lbl = tk.Label(bot, text="Default",
                                     font=("Helvetica", 10), fg="#B8B8BC", bg="#1C1C20")
            self.mode_lbl.pack(side="left")
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
                    elif cmd == "hide": self.root.withdraw()
                    elif cmd == "wave": self.wave_vals = msg[1]; self._draw()
                    elif cmd == "dot":
                        self.dot_state = msg[1]; self.mode_idx = msg[2]
                        c = {"rec":"#EB4040","done":"#38D068"}.get(msg[1],"#52525A")
                        self.dot.config(fg=c)
                        self.mode_lbl.config(text=MODE_SHORT[MODES[msg[2]]])
            except queue.Empty: pass
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
        def q_dot(self, d, m): self._q.put(("dot", d, m))
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
        self._mode       = 0
        self._raw        = ""
        self._transcript = []
        self._chunk_lock = threading.Lock()
        self._smooth     = [0.0] * BARS
        self._targets    = [0.0] * BARS

    def _on_press(self, key):
        if key == kb_input.Key.cmd_r:
            if self._state == "idle":        self._start()
            elif self._state == "recording": self._stop()
            # ignore cmd_r during processing — prevents ghost triggers on wake
            return
        if key == kb_input.Key.esc:
            self._cancel()   # dismiss in any state
            return
        # Tab is handled exclusively by CGEventTap (_install_tab_tap)
        # Do NOT handle it here — would cause double-cycle

    def _on_release(self, key):
        pass

    def _start(self):
        self._state = "recording"
        self._raw   = ""
        with self._chunk_lock:
            self._transcript = []
        # auto-detect context and switch mode silently
        detected = detect_context_mode()
        if detected is not None:
            self._mode = detected
        self.ui.q_show()
        self.ui.q_dot(DOT_STATE_REC, self._mode)
        self.recorder.start()
        threading.Thread(target=self._wave_loop, daemon=True).start()

    def _stop(self):
        self._state = "processing"
        self.ui.q_dot(DOT_STATE_IDLE, self._mode)
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
        self.ui.q_dot(DOT_STATE_IDLE, self._mode)
        self.ui.q_hide()

    def _cycle_mode(self):
        self._switch_mode((self._mode + 1) % len(MODES))

    def _switch_mode(self, idx):
        self._mode = idx
        self.ui.q_dot(
            DOT_STATE_REC if self._state == "recording" else DOT_STATE_IDLE,
            idx)
        if self._raw and self._state == "idle":
            threading.Thread(target=self._reprocess, daemon=True).start()

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
            self.ui.q_dot(DOT_STATE_IDLE, self._mode)
            time.sleep(0.6); self._state = "idle"; self.ui.q_hide(); return
        self._raw = full_text
        self._deliver(full_text)

    def _reprocess(self):
        self._state = "processing"
        self.ui.q_show()
        self._deliver(self._raw)

    def _deliver(self, raw):
        mode = MODES[self._mode]
        if mode != "raw":
            self.ui.q_dot(DOT_STATE_IDLE, self._mode)
            result = ollama_post(raw, mode)
        else:
            result = raw
        pyperclip.copy(result)
        paste_to_frontmost()
        self.ui.q_dot(DOT_STATE_DONE, self._mode)
        self._state = "idle"
        threading.Thread(target=self._autohide, daemon=True).start()

    def _autohide(self):
        time.sleep(AUTOHIDE_SEC)
        if self._state == "idle":
            self.ui.q_wave([0.0] * BARS)
            self.ui.q_dot(DOT_STATE_IDLE, self._mode)
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

    def _install_tab_tap(self):
        """Install a CGEventTap ONLY for Tab suppression.
        Everything else (cmd_r, esc) still handled by pynput."""
        if not HAVE_QUARTZ:
            return
        TAB_KEYCODE = 48
        pad = self

        def cb(proxy, etype, event, refcon):
            try:
                kc = CGEventGetIntegerValueField(event, kCGKeyboardEventKeycode)
                if kc == TAB_KEYCODE and etype == kCGEventKeyDown:
                    if pad.ui.is_visible():
                        pad._cycle_mode()
                        return None   # suppress — Chrome never sees it
            except Exception:
                pass
            return event   # pass everything else through

        tap = CGEventTapCreate(
            kCGSessionEventTap, kCGHeadInsertEventTap, 0,
            (1 << kCGEventKeyDown),
            cb, None)
        if tap is None:
            print("[voicepad] CGEventTap failed — check Accessibility. Tab will pass through.")
            return
        src = CFMachPortCreateRunLoopSource(None, tap, 0)
        CFRunLoopAddSource(CFRunLoopGetMain(), src, kCFRunLoopCommonModes)
        CGEventTapEnable(tap, True)
        print("[voicepad] Tab suppression active")

    def run(self):
        threading.Thread(target=load_transcription_model, daemon=True).start()
        threading.Thread(target=self.recorder.prewarm, daemon=True).start()
        # pynput handles cmd_r and esc as before
        listener = kb_input.Listener(
            on_press=self._on_press, on_release=self._on_release)
        listener.daemon = True; listener.start()
        # CGEventTap only for tab suppression, runs on AppKit main loop
        self._install_tab_tap()
        print("[voicepad] Right Cmd=record | Tab=mode (visible only) | Esc=cancel")
        self.ui.run()

if __name__ == "__main__":
    VoicePad().run()

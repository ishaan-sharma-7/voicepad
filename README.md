# VoicePad

A lightweight macOS voice-dictation overlay powered by [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) (Apple Silicon) with optional LLM post-processing via [Ollama](https://ollama.com).

Press **Right ⌘** to start/stop recording. Text is transcribed locally, optionally cleaned up by a local LLM, and pasted directly into whatever app is in focus.

## Privacy

VoicePad never stores or transmits your audio.

- **Warmup stream** — to eliminate the lag when switching Bluetooth audio devices (e.g. AirPods) into recording mode, VoicePad keeps a silent microphone stream open at all times. Every audio chunk from this stream is immediately discarded in memory — it is never written to disk, never buffered, and never leaves your machine.
- **Recording** — audio is only actively captured while you hold Right ⌘. The raw PCM data lives in RAM for the duration of transcription, then is released.
- **Transcription** — runs entirely on-device via Apple's MLX framework. No audio or text is sent to any external server.
- **LLM post-processing** — if enabled, the transcript is sent to a local [Ollama](https://ollama.com) instance running on your machine. Nothing leaves localhost.

---

## Requirements

| Requirement | Notes |
|---|---|
| macOS 12+ | AppKit UI; Quartz event tap for Tab suppression |
| Apple Silicon (M1/M2/M3) | Required for `mlx-whisper`; see fallback below |
| Python 3.10+ | Uses `int \| None` union syntax |
| [Ollama](https://ollama.com) | Only needed for Email / Notes / Math modes |

> **No Apple Silicon?** The app automatically falls back to `faster-whisper` on CPU. Install it with `pip install faster-whisper` and remove `mlx-whisper` from `requirements.txt`.

---

## Installation

```bash
# 1. Clone the repo
git clone https://github.com/your-username/voicepad.git
cd voicepad

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Pull an Ollama model for post-processing modes
ollama pull qwen2.5:14b   # or any model you prefer
```

### Accessibility permission

VoicePad uses a CGEventTap to intercept the **Tab** key (for mode cycling) without letting it reach other apps. macOS requires Accessibility access for this.

1. Open **System Settings → Privacy & Security → Accessibility**
2. Add your terminal emulator (e.g. Terminal, iTerm2, Warp) or the Python binary

If you skip this step, Tab will still cycle modes but will also be forwarded to the active app.

---

## Usage

```bash
source venv/bin/activate
python voicepad.py
```

| Key | Action |
|---|---|
| **Right ⌘** | Start / stop recording |
| **Tab** *(while overlay visible)* | Cycle transcription mode |
| **Esc** | Cancel and dismiss |

The overlay appears at the bottom-center of your screen while recording. When you stop, the transcribed (and optionally processed) text is copied to the clipboard and pasted into the frontmost window.

---

## Transcription modes

| Mode | What it does |
|---|---|
| **Default** | Raw Whisper output — no LLM post-processing |
| **Email** | Cleans up grammar and structures as a short email body |
| **Notes** | Converts to bullet-point notes |
| **Math** | Converts spoken math to LaTeX |

VoicePad also auto-detects context: if **Apple Notes** is focused it switches to Notes mode; if **Gmail** is open in Chrome it switches to Email mode.

---

## Configuration

All tunable constants are at the top of `voicepad.py`:

```python
OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:14b"      # any model pulled in Ollama
MLX_MODEL    = "mlx-community/whisper-small.en-mlx"  # see model options below
AUTOHIDE_SEC = 1.2                # seconds before overlay hides after pasting
```

### MLX Whisper model options

Larger models are more accurate but slower. All are English-only `.en` variants unless noted.

| Model | Size | Notes |
|---|---|---|
| `mlx-community/whisper-tiny.en-mlx` | ~75 MB | Fastest; lower accuracy |
| `mlx-community/whisper-base.en-mlx` | ~145 MB | Good balance |
| `mlx-community/whisper-small.en-mlx` | ~465 MB | **Default** |
| `mlx-community/whisper-medium.en-mlx` | ~1.5 GB | Higher accuracy |
| `mlx-community/whisper-large-v3-mlx` | ~3 GB | Best accuracy; multilingual |

Models are downloaded automatically from Hugging Face on first use.

### Custom vocabulary / name corrections

Whisper sometimes mishears proper nouns, names, or jargon. Edit `DEFAULT_VOCAB` in `voicepad.py` (or the hot-reloaded file at `~/.voicepad/vocab.json`) to add your own corrections:

```json
{
  "jon": "John",
  "jon doe": "John Doe",
  "api": "API",
  "github": "GitHub"
}
```

The vocab file is hot-reloaded on every transcription — no restart needed.

---

## How it works

1. **Recording** — `sounddevice` streams audio from the default macOS input at 16 kHz.
2. **Silence detection** — chunks are flushed automatically when silence is detected (configurable RMS threshold).
3. **Transcription** — `mlx_whisper.transcribe()` runs on-device via Apple's MLX framework (no network, no API key).
4. **Post-processing** — if a non-raw mode is active, the transcript is sent to a local Ollama instance with a mode-specific system prompt.
5. **Paste** — result is written to the clipboard and pasted via AppleScript `keystroke "v" using {command down}`.

---

## Standalone `.app` bundle (optional)

If you'd rather install VoicePad as a regular macOS app — no menu-bar `Python` entry, no Cmd+Tab listing, no dependency on whatever Python your shell happens to resolve — you can build a self-contained `.app` from the included [setup.py](setup.py).

**Why bother:** the bundle has its own embedded Python interpreter and only ships voicepad's actual dependencies. If your system Python's site-packages is "polluted" with something that misbehaves at process exit (e.g. `llama-cpp-python` whose `libggml-metal` destructor calls `abort()` and produces a *Python quit unexpectedly* dialog every time the process is killed), the bundle is fully insulated from it.

```bash
# Build environment — use a fresh venv, py2app needs a working pip
python3.11 -m venv /tmp/voicepad-build
/tmp/voicepad-build/bin/pip install -r requirements.txt py2app

# Build (produces dist/VoicePad.app, ~1 GB)
/tmp/voicepad-build/bin/python setup.py py2app

# Install to ~/Applications and ad-hoc-sign so macOS will run it
mkdir -p ~/Applications
cp -R dist/VoicePad.app ~/Applications/
codesign --force --deep --sign - ~/Applications/VoicePad.app
```

First launch will prompt for **Microphone** and **Accessibility** access — grant both. The bundle is registered as `com.voicepad.app` (a separate identity from a raw Python script), so any earlier permissions granted to "Python" don't carry over.

`Info.plist` sets `LSUIElement=YES` so the app is hidden from the Dock, menu bar, and Cmd+Tab — it just shows the floating panel when you press Right ⌘.

---

## Auto-launch on login (optional)

VoicePad runs fine by launching it manually in a terminal. If you want it to start automatically on login, pick one of the options below.

### Option 1: launchd plist (no third-party dependencies)

Create `~/Library/LaunchAgents/com.voicepad.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>             <string>com.voicepad</string>
  <key>ProgramArguments</key>
  <array>
    <string>/path/to/voicepad/venv/bin/python</string>
    <string>/path/to/voicepad/voicepad.py</string>
  </array>
  <key>RunAtLoad</key>         <true/>
  <key>KeepAlive</key>         <true/>
  <key>StandardOutPath</key>   <string>/tmp/voicepad.log</string>
  <key>StandardErrorPath</key> <string>/tmp/voicepad.log</string>
</dict>
</plist>
```

Replace both `/path/to/voicepad` entries with your actual paths, then load it:

```bash
launchctl load ~/Library/LaunchAgents/com.voicepad.plist
```

`KeepAlive` means launchd will automatically restart VoicePad if it ever crashes.

### Option 2: Hammerspoon (if you already use it)

If you have [Hammerspoon](https://www.hammerspoon.org) installed, add this to your `~/.hammerspoon/init.lua`:

```lua
-- Auto-launch VoicePad on login and restart it if it exits
local voicepadTask = nil

local function startVoicePad()
  voicepadTask = hs.task.new(
    "/path/to/voicepad/venv/bin/python",
    function(exitCode, stdOut, stdErr)
      -- Restart on exit (e.g. crash)
      hs.timer.doAfter(2, startVoicePad)
    end,
    { "/path/to/voicepad/voicepad.py" }
  )
  voicepadTask:start()
end

startVoicePad()
```

Replace both `/path/to/voicepad` entries with your actual paths and reload your Hammerspoon config.

If you built the `.app` bundle (see above), point Hammerspoon at its inner binary instead:

```lua
local function startVoicePad()
  hs.task.new(
    "/Users/yourname/Applications/VoicePad.app/Contents/MacOS/VoicePad",
    function(exitCode, stdOut, stdErr)
      hs.timer.doAfter(3, startVoicePad)
    end,
    {}
  ):start()
end

startVoicePad()
```

---

## License

MIT — see [LICENSE](LICENSE).

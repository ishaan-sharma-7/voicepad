# VoicePad

A lightweight macOS voice-dictation overlay powered by [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) (Apple Silicon), with optional offloading to a shared transcription server on your LAN.

Press the trigger key (default **Right ⌘**) to start/stop recording. Text is transcribed and pasted directly into whatever app is in focus.

## Privacy

The microphone is **opt-in**: no audio stream exists except between your start and stop keypresses. There is no warmup stream, no background listening — when you're not dictating, the mic is closed.

- **Recording** — audio is captured only between trigger-key presses. The raw PCM lives in RAM for the duration of transcription, then is released. Nothing is written to disk.
- **Transcription** — by default, runs entirely on-device via Apple's MLX framework. If you configure a remote server (see below), audio is sent to that server — keep it LAN-only or behind Tailscale.
- **Output** — the transcript goes to your clipboard and is pasted into the frontmost app. No history is stored.

---

## Requirements

| Requirement | Notes |
|---|---|
| macOS 12+ | AppKit overlay UI |
| Apple Silicon (M1+) | Required for `mlx-whisper`; see fallback below |
| Python 3.10+ | Uses `int \| None` union syntax |

> **No Apple Silicon?** The app automatically falls back to `faster-whisper` on CPU (`pip install faster-whisper`), or point `transcribe_url` at a server that has Apple Silicon.

---

## Installation

```bash
git clone https://github.com/maslowtechnologies/voicepad.git
cd voicepad
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python voicepad.py
```

### Permissions

1. **Microphone** — prompted on first recording.
2. **Accessibility / Input Monitoring** (System Settings → Privacy & Security) — add your terminal (or the `.app` bundle). Needed for the global hotkey and for auto-paste.

### Start at login (no Hammerspoon needed)

```bash
python voicepad.py --install-agent
```

This writes a `launchd` LaunchAgent (`~/Library/LaunchAgents/com.voicepad.agent.plist`) with `KeepAlive`, so VoicePad starts at login and restarts automatically if it ever exits. Run with `--uninstall-agent` to remove it. If you previously used a Hammerspoon keep-alive, delete that snippet first or you'll get duplicate instances.

---

## Usage

| Key | Action |
|---|---|
| **Right ⌘** (configurable) | Start / stop recording |
| **Esc** | Cancel and dismiss |

The pill overlay appears at the bottom-center of your screen while recording. When you stop, the transcript is copied to the clipboard and pasted into the frontmost window.

---

## Configuration

`~/.voicepad/config.json` is created with defaults on first run:

```json
{
  "hotkey": "cmd_r",
  "input_device": "builtin",
  "transcribe_url": "",
  "local_fallback": true,
  "mlx_model": "mlx-community/whisper-small.en-mlx"
}
```

- **hotkey** — any [pynput key name](https://pynput.readthedocs.io/en/latest/keyboard.html#pynput.keyboard.Key) (`cmd_r`, `alt_r`, `f19`, …) or a single character.
- **input_device** — `"builtin"` (default) records from the Mac's internal mic even when AirPods are connected: it opens instantly (no Bluetooth HFP negotiation, which matters now that the mic only opens on demand) and sounds better than the HFP telephone codec. `"default"` follows the system input device instead; expect the first words to be clipped while a Bluetooth mic renegotiates.
- **transcribe_url** — empty for local transcription, or your transcription server's URL, e.g. `http://macmini.local:8765/transcribe`. Remote failures fall back to local transcription.
- **local_fallback** — set `false` for a thin client that never loads a local model (no ~500 MB download, no RAM cost) and depends entirely on the server.
- **mlx_model** — local model. Options: `whisper-tiny.en-mlx` (~75 MB), `whisper-base.en-mlx` (~145 MB), `whisper-small.en-mlx` (~465 MB, default), `whisper-medium.en-mlx` (~1.5 GB), `whisper-large-v3-mlx` (~3 GB), all under `mlx-community/`. Downloaded from Hugging Face on first use.

Restart VoicePad after editing config.json.

### Custom vocabulary / name corrections

Whisper sometimes mishears proper nouns, names, or jargon. Add corrections to `~/.voicepad/vocab.json`:

```json
{
  "eshon": "Ishaan",
  "cloud code": "Claude Code",
  "slash loop": "/loop"
}
```

Matching is case-insensitive on word boundaries, longest phrase first. The file is hot-reloaded on every transcription — no restart needed.

---

## Shared transcription server (Mac Mini)

Instead of every laptop running its own Whisper model, run one big model on a shared machine and point every client at it. The server holds the model resident in memory, so requests are fast, and clients become thin (no model download, no RAM cost).

```
┌──────────────┐  audio (HTTP POST /transcribe)   ┌───────────────────────┐
│  VoicePad     │ ───────────────────────────────▶ │  Mac Mini              │
│  (your Mac)   │ ◀─────────────────────────────── │  whisper-large-v3      │
└──────────────┘            text                   └───────────────────────┘
```

### On the Mac Mini

```bash
git clone https://github.com/maslowtechnologies/voicepad.git
cd voicepad/server
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python transcribe_server.py        # serves on 0.0.0.0:8765, loads whisper-large-v3
```

To keep it running across reboots, create `~/Library/LaunchAgents/com.voicepad.server.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>             <string>com.voicepad.server</string>
  <key>ProgramArguments</key>
  <array>
    <string>/path/to/voicepad/server/venv/bin/python</string>
    <string>/path/to/voicepad/server/transcribe_server.py</string>
  </array>
  <key>RunAtLoad</key>         <true/>
  <key>KeepAlive</key>         <true/>
  <key>StandardOutPath</key>   <string>/tmp/voicepad-server.log</string>
  <key>StandardErrorPath</key> <string>/tmp/voicepad-server.log</string>
</dict>
</plist>
```

then `launchctl load ~/Library/LaunchAgents/com.voicepad.server.plist`.

Model and port are overridable via `VOICEPAD_MODEL` and `VOICEPAD_PORT` env vars. The server never writes audio to disk and never logs transcripts. Keep it on the LAN (or Tailscale) — there is no auth layer.

### On each client

Edit `~/.voicepad/config.json`:

```json
{
  "transcribe_url": "http://macmini.local:8765/transcribe",
  "local_fallback": false
}
```

(`local_fallback: true` keeps a local model as a backup for when you're off the office network.)

Sanity check from a client: `curl http://macmini.local:8765/health`

Vocabulary corrections stay client-side — each person's `vocab.json` is applied to the text the server returns.

---

## How it works

1. **Recording** — on the trigger key, `sounddevice` opens the configured input at 16 kHz. PortAudio's device cache is refreshed on every start so Bluetooth devices that dropped and rejoined are picked up.
2. **Silence detection** — chunks are flushed for transcription automatically as you pause (configurable RMS threshold), so long dictations transcribe incrementally.
3. **Transcription** — remote server if configured (raw float32 PCM over HTTP), else `mlx_whisper` on-device. Hallucination filtering and vocab corrections are applied client-side.
4. **Paste** — the result is written to the clipboard and pasted via AppleScript `keystroke "v"`.

---

## Standalone `.app` bundle (optional)

If you'd rather install VoicePad as a regular macOS app — no terminal, no dependency on whatever Python your shell happens to resolve — build a self-contained `.app` from the included [setup.py](setup.py).

```bash
# Build environment — use a fresh venv, py2app needs a working pip
python3.11 -m venv /tmp/voicepad-build
/tmp/voicepad-build/bin/pip install -r requirements.txt py2app

# Build (produces dist/VoicePad.app)
/tmp/voicepad-build/bin/python setup.py py2app

# Install to ~/Applications and ad-hoc-sign so macOS will run it
mkdir -p ~/Applications
cp -R dist/VoicePad.app ~/Applications/
codesign --force --deep --sign - ~/Applications/VoicePad.app
```

First launch prompts for **Microphone** and **Accessibility** access. `Info.plist` sets `LSUIElement=YES` so the app is hidden from the Dock, menu bar, and Cmd+Tab — it just shows the floating pill when you press the trigger key.

To auto-start the bundle at login, run the inner binary once with the agent flag:

```bash
~/Applications/VoicePad.app/Contents/MacOS/VoicePad --install-agent
```

---

## License

MIT — see [LICENSE](LICENSE).

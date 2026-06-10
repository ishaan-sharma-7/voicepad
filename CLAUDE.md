# VoicePad — Maslow Technologies

Single-file macOS dictation app: `voicepad.py` is the whole client (recording,
transcription, AppKit pill UI, hotkey handling). `server/transcribe_server.py`
is the shared transcription server that runs on the office Mac Mini.
`setup.py` builds an optional py2app `.app` bundle.

## Architecture

- **Opt-in mic** — no audio stream exists outside `AudioRecorder.start()..stop()`.
  An earlier version held a 24/7 silent "warmup" stream to pre-negotiate
  Bluetooth HFP; it was deliberately removed for privacy. Do not reintroduce it.
  Instant start with AirPods is achieved by pinning the built-in mic instead
  (`input_device: "builtin"`, the default).
- **Raw transcription only** — Email/Notes modes, Ollama post-processing,
  context auto-detection, and the Tab-cycling CGEventTap were removed (June 2026).
- **Config** — `~/.voicepad/config.json` (hotkey, input_device, transcribe_url,
  local_fallback, mlx_model), created with defaults on first run. Read once at
  startup. `~/.voicepad/vocab.json` (mishearing corrections) is hot-reloaded on
  every transcription by `apply_vocab()` — word-boundary, case-insensitive,
  longest-key-first.
- **Transcription** — remote-first when `transcribe_url` is set (raw 16 kHz mono
  float32 PCM POSTed to the Mac Mini server), falling back to local `mlx_whisper`
  (or `faster-whisper` on non-Apple-Silicon). Hallucination filter and vocab run
  client-side in `transcribe()`.
- **Server** — FastAPI, model resident in memory, transcriptions serialized
  behind a lock (mlx is not re-entrant). No auth: LAN/Tailscale only. Never log
  audio or transcripts server-side.
- **Keep-alive** — `voicepad.py --install-agent` writes a launchd LaunchAgent
  (`com.voicepad.agent`, KeepAlive=true). This replaced Hammerspoon.
- **Shutdown** — SIGTERM/AE-Quit call `os._exit(0)` deliberately (`_hard_exit`):
  a dylib in the dependency tree aborts in its destructor during normal
  interpreter teardown. Do not "clean up" by restoring normal exit paths.

## Gotchas

- PortAudio caches the device list at init; `AudioRecorder.start()` calls
  `sd._terminate()/_initialize()` to pick up Bluetooth devices that dropped
  and rejoined.
- `AudioRecorder.stop()` must not hold the lock through `stream.stop()` —
  PortAudio blocks until the in-flight callback (which takes the same lock)
  returns. See comments before touching the lock discipline.
- The panel re-asserts window level/ordering on a watchdog tick and on
  screen-reconfig and wake notifications — flakiness here historically
  required a process restart.
- App Nap: the process holds an NSActivity assertion (`LatencyCritical`);
  without it the UI queue timer gets deferred and the panel shows late or never.
- Whisper hallucinates on silence ("thanks for watching", repeated words) —
  `_is_hallucination()` filters these; extend the set rather than removing it.

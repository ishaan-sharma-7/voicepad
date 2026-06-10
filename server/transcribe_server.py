#!/usr/bin/env python3
"""
VoicePad transcription server — runs on the office Mac Mini.

Holds a Whisper model resident in memory and serves transcriptions to every
team member's VoicePad over the LAN. Clients point at it by setting
"transcribe_url" in ~/.voicepad/config.json.

API:
  POST /transcribe   body: raw 16 kHz mono float32 PCM  ->  {"text": "..."}
  GET  /health       ->  {"status": "ok", "model": "..."}

Run:
  pip install -r requirements.txt
  python transcribe_server.py

Environment overrides:
  VOICEPAD_MODEL  (default mlx-community/whisper-large-v3-mlx)
  VOICEPAD_PORT   (default 8765)

Keep this LAN-only (or behind Tailscale). Audio is transcribed in memory and
never written to disk or logged.
"""

import os
import threading

import numpy as np
import mlx_whisper
import uvicorn
from fastapi import FastAPI, HTTPException, Request

MODEL       = os.environ.get("VOICEPAD_MODEL", "mlx-community/whisper-large-v3-mlx")
PORT        = int(os.environ.get("VOICEPAD_PORT", "8765"))
SAMPLE_RATE = 16_000

app = FastAPI(title="VoicePad transcription server")

# mlx_whisper is not re-entrant; serialize transcriptions. Requests are a few
# hundred ms each on Apple Silicon, so a queue beats a crash.
_lock = threading.Lock()


def warm():
    """Load the model now so the first real request isn't slow."""
    dummy = np.zeros(SAMPLE_RATE, dtype=np.float32)
    mlx_whisper.transcribe(dummy, path_or_hf_repo=MODEL, language="en")
    print(f"[server] {MODEL} loaded and warm")


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL}


@app.post("/transcribe")
async def transcribe(request: Request):
    body = await request.body()
    if not body or len(body) % 4 != 0:
        raise HTTPException(400, "expected raw 16 kHz mono float32 PCM")
    audio = np.frombuffer(body, dtype=np.float32)
    with _lock:
        r = mlx_whisper.transcribe(audio, path_or_hf_repo=MODEL, language="en")
    return {"text": r.get("text", "").strip()}


if __name__ == "__main__":
    warm()
    uvicorn.run(app, host="0.0.0.0", port=PORT)

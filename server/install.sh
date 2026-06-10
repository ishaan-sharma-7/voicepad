#!/bin/bash
# One-command setup for the VoicePad transcription server (run on the Mac Mini):
#
#   git clone https://github.com/maslowtechnologies/voicepad.git
#   ./voicepad/server/install.sh
#
# Creates a venv, installs deps, registers a launchd KeepAlive agent so the
# server starts at boot and restarts on crash, then waits until it's healthy.
set -euo pipefail
cd "$(dirname "$0")"

PORT="${VOICEPAD_PORT:-8765}"
PLIST="$HOME/Library/LaunchAgents/com.voicepad.server.plist"

echo "==> Creating venv and installing dependencies"
python3 -m venv venv
./venv/bin/pip install --quiet -r requirements.txt

echo "==> Installing launchd agent ($PLIST)"
mkdir -p "$HOME/Library/LaunchAgents"
cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>             <string>com.voicepad.server</string>
  <key>ProgramArguments</key>
  <array>
    <string>$(pwd)/venv/bin/python</string>
    <string>$(pwd)/transcribe_server.py</string>
  </array>
  <key>EnvironmentVariables</key>
  <dict>
    <key>VOICEPAD_PORT</key>   <string>$PORT</string>
  </dict>
  <key>RunAtLoad</key>         <true/>
  <key>KeepAlive</key>         <true/>
  <key>StandardOutPath</key>   <string>/tmp/voicepad-server.log</string>
  <key>StandardErrorPath</key> <string>/tmp/voicepad-server.log</string>
</dict>
</plist>
EOF
launchctl unload "$PLIST" 2>/dev/null || true
launchctl load "$PLIST"

echo "==> Waiting for the server (first run downloads whisper-large-v3, ~3 GB)"
echo "    progress: tail -f /tmp/voicepad-server.log"
until curl -sf "http://localhost:$PORT/health" > /dev/null; do sleep 5; done

echo "==> Ready. Point clients at:"
echo "    \"transcribe_url\": \"http://$(hostname -s).local:$PORT/transcribe\""

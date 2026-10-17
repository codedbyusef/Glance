# audio/: Glance voice I/O

Wake word, speech-to-text and text-to-speech for Glance. The spec is [CLAUDE.md](CLAUDE.md); section 17 lists the decisions made while building it.

- **Pi side:** mic capture, wake word, VAD, push-to-talk, STT client with local fallback, Piper TTS.
- **Vultr side:** `server/stt_service.py`, faster-whisper `small.en` on the GPU.

All commands below run from the **repo root**, because the code is the `audio` package.

License note: `piper-tts` is GPL-3.0, while this repo is MIT. That's acceptable for the hackathon, but keep it in mind before redistributing.

## Unit tests (any machine)

```bash
python -m venv audio/.venv
audio/.venv/bin/pip install -r audio/requirements-dev.txt     # Windows: audio\.venv\Scripts\pip
audio/.venv/bin/python -m pytest audio/tests -q
```

No mic, speaker, GPU, network or model files needed.

## Dev laptop (Windows) before the Pi exists

From the repo root in PowerShell:

```powershell
python -m venv audio\.venv
audio\.venv\Scripts\pip install -r audio\requirements-laptop.txt
audio\.venv\Scripts\pip install --no-deps openwakeword==0.6.0
audio\.venv\Scripts\python audio\scripts\fetch_models.py      # Piper voice, wake models, tiny.en
```

Run the STT service on the laptop CPU (second terminal):

```powershell
$env:GLANCE_STT_TOKEN="devtoken"; $env:GLANCE_STT_DEVICE="cpu"; $env:GLANCE_STT_MODEL="tiny.en"
audio\.venv\Scripts\python -m uvicorn audio.server.stt_service:app --host 127.0.0.1 --port 8001
```

Then point the client at it: `$env:GLANCE_STT_URL="http://127.0.0.1:8001"; $env:GLANCE_STT_TOKEN="devtoken"`.
`small.en` with beam 5 takes several seconds per clip on a laptop CPU, which is over the 2 s client timeout. Use `tiny.en` or `base.en` on the laptop, or set `stt_timeout_s: 15` in `audio/config.yaml` while testing `small.en`.

## Vultr: STT service

```bash
cd ~/Glance                     # repo checkout; only audio/__init__.py, audio/stt_client.py, audio/log.py and audio/server/ are used
python3 -m venv ~/stt-venv && . ~/stt-venv/bin/activate
pip install -r audio/requirements-server.txt
# CUDA libs from the pip wheels (skip if the image already has cuBLAS 12 + cuDNN 9):
export LD_LIBRARY_PATH=$(python -c 'import os, nvidia.cublas.lib, nvidia.cudnn.lib; print(os.path.dirname(nvidia.cublas.lib.__file__) + ":" + os.path.dirname(nvidia.cudnn.lib.__file__))')

export GLANCE_STT_TOKEN=$(openssl rand -hex 24)   # share it with the Pi; never commit it
# optional: GLANCE_STT_MODEL=small.en  GLANCE_STT_INITIAL_PROMPT="done, yes, no, stop, wait, cancel, repeat"  ("" = no prompt)
#           GLANCE_STT_DEVICE=cuda (default) or cpu
uvicorn audio.server.stt_service:app --host 0.0.0.0 --port 8001
```

A request that waited over 10 s for the model (its client already gave up) gets 503 without inference.
`GET /health` returns 503 until the model is loaded and warmed up, then `{"ready": true, "model": "small.en"}`. Both endpoints need `Authorization: Bearer $GLANCE_STT_TOKEN`. Expose port 8001 the same way as the perception server (HTTPS front door or SSH tunnel), and allow only that port through the firewall.

Quick check from anywhere:

```bash
curl -s -H "Authorization: Bearer $GLANCE_STT_TOKEN" $GLANCE_STT_URL/health
curl -s -H "Authorization: Bearer $GLANCE_STT_TOKEN" -H "Content-Type: audio/wav" --data-binary @some16kmono.wav $GLANCE_STT_URL/transcribe
```

## GPU PC stand-in for Vultr (Windows, RTX 3070, over Tailscale)

Until Vultr is set up, the STT server runs on a Windows PC with an NVIDIA GPU. The client only needs `GLANCE_STT_URL=http://100.112.104.87:8001` (decision 17). PowerShell on the GPU PC, from the repo root:

```powershell
python -m venv audio\.venv-server
audio\.venv-server\Scripts\python -m pip install --upgrade pip
audio\.venv-server\Scripts\pip install -r audio\requirements-server-windows.txt

$env:GLANCE_STT_TOKEN = "<long random token; same value on the laptop>"
$env:GLANCE_STT_MODEL = "small.en"
$env:GLANCE_STT_DEVICE = "cuda"
# $env:GLANCE_STT_INITIAL_PROMPT = "none"     # for the A/B run without the prompt
audio\.venv-server\Scripts\python -m uvicorn audio.server.stt_service:app --host 0.0.0.0 --port 8001
```

The startup log prints the CUDA DLL folders it added, then `ready in ...: {'model': 'small.en', 'device': 'cuda', 'beam_size': 5, 'initial_prompt': ...}`.

Firewall (an elevated PowerShell, once). This allows port 8001 from Tailscale addresses only:

```powershell
New-NetFirewallRule -DisplayName "Glance STT (Tailscale only)" -Direction Inbound -Protocol TCP -LocalPort 8001 -RemoteAddress 100.64.0.0/10 -Action Allow -Profile Any
```

If Windows shows a "Windows Defender Firewall has blocked some features" prompt for python.exe, you can cancel it, but cancelling creates *block* rules for python.exe, and block rules beat allow rules. Check for them and remove them:

```powershell
Get-NetFirewallRule -Direction Inbound -Action Block | Where-Object DisplayName -like "*python*" | Select-Object DisplayName, Profile
Get-NetFirewallRule -Direction Inbound -Action Block | Where-Object DisplayName -like "*python*" | Remove-NetFirewallRule
```

From the laptop:

```powershell
$env:GLANCE_STT_URL = "http://100.112.104.87:8001"; $env:GLANCE_STT_TOKEN = "<same token>"
audio\.venv\Scripts\python -m audio.cli stt-check
```

## Pi: setup from a fresh Raspberry Pi OS (64-bit)

```bash
git clone <repo> ~/Glance && cd ~/Glance
bash audio/scripts/install_pi.sh          # apt libportaudio2, venv, pip deps, openwakeword --no-deps, models
. audio/.venv/bin/activate
export GLANCE_STT_URL=https://<vultr-host>:8001     # base URL; /transcribe is appended
export GLANCE_STT_TOKEN=<token from the server>
```

openWakeWord's Linux metadata requires `tflite-runtime`, which has no aarch64 wheel for Python 3.11 or 3.13. So the install script runs `pip install --no-deps openwakeword==0.6.0`, the real runtime deps (`onnxruntime numpy scipy scikit-learn tqdm requests`) come from `requirements-pi.txt`, and the code always uses `inference_framework="onnx"`.

`scripts/fetch_models.sh` (a wrapper for `fetch_models.py`) downloads the Piper voice into `audio/models/`, openWakeWord's feature models plus the `hey_jarvis` stand-in, and caches faster-whisper `tiny.en`.

Optional overrides go in `audio/config.yaml` (gitignored). Unknown keys are an error. Defaults are in `config.py`.

## TTS player

```bash
python -m audio.cli say "Okay, cancelled." "The can is on your right."          # queued in order, prints reports and timings
python -m audio.cli say "A long sentence to cut off part way through it." --stop-after 1.0
python -m audio.cli say "Faster speech." --rate 1.3                             # speaking_rate; Piper length_scale = 1 / rate
```

`Player` implements the agent's `AudioOut`. One synth thread renders text ahead of playback (cached: the three fixed controller phrases plus an LRU of 16). One writer thread owns the output stream and writes in 50 ms chunks. `stop_speech` reports `interrupted` synchronously on the caller's thread. Every other report goes through `dispatch`. See CLAUDE.md section 17, decisions 9 and 10.

## Listening

```bash
python -m audio.cli listen                      # wake word, VAD, STT; prints every event with timings
python -m audio.cli listen --ptt key            # plus push-to-talk: Enter starts, Enter stops
python -m audio.cli listen --reply answer       # after each utterance say "Yes?" with expect=answer (reply window)
python -m audio.cli listen --task-active        # as if a task were running: no wake word needed
python -m audio.cli listen --scores             # print wake scores >= 0.2, for tuning wake_threshold
python -m audio.cli listen --wav some.wav       # replay a 16 kHz mono recording instead of the mic
python -m audio.cli echo                        # say each transcript back
```

The wake model is `models/hey_glance.onnx`. If that file is missing, the `hey_jarvis` stand-in loads and the log says STAND-IN. If no model loads at all, push-to-talk still works.

After a detection, the capture starts only when the wake phrase has ended: at the first 60 ms gap in the voice, when the score falls below `wake_end_threshold` (0.3), or after `wake_end_max_s` (1.0 s), whichever comes first. The wake model also runs during open-state captures; a detection there drops the audio up to the end of the phrase, so "hey glance, stop" arrives as "stop". A leading "hey glance" left in a transcript is stripped as a backup. Each detection plays a short tone (`wake_tone`, on by default) unless something is already playing; wait for it before the command.

Noise defaults: `vad_aggressiveness` 3, `speech_start_ms` 300 (every 20 ms frame of a 300 ms window voiced), and a capture is kept going only by voiced frames when more than half of the last 300 ms is voiced. If short commands ("no") are missed in open state, try `speech_start_ms: 260` in `config.yaml`; if mid-sentence pauses split utterances, raise `end_silence_ms`.

## Clip set and STT benchmark

```bash
python -m audio.cli record --label "get me the can" --tag quiet   # Enter to start, Enter to stop
python -m audio.cli stt-bench --out audio/clips/bench.csv          # both backends; --backends vultr or local
```

Clips and `labels.csv` go to `audio/clips/` (gitignored, since it holds voice recordings). Noise-only clips get an empty label (`--label=` in PowerShell, which drops a bare `""` argument); the `noise` column is the share of them that came back empty. `stt-bench` prints exact-match accuracy (lowercased, punctuation stripped), accuracy on the command clips, and p50/p95 latency per backend. To A/B the initial prompt, restart the server with `GLANCE_STT_INITIAL_PROMPT=none` and rerun. `stt-check` calls `/health` and transcribes the bundled `assets/stt_check.wav`.

## Selftest, logs and stats

```bash
python -m audio.cli selftest            # or: python -m audio.cli --selftest. Run it first after each boot at the venue
python -m audio.cli stats               # numbers from today's log, last listen or echo session
python -m audio.cli stats --session 164301          # one session by id (the id is printed in the summary)
python -m audio.cli stats --session all --day 20261017
python -m audio.cli echo --minutes 30   # soak test; prints the session summary at the end
```

`selftest` checks the wake model file and the env vars (it says whether the token is set, never what it is), plays a test sentence and the thinking tone, records 3 s, and transcribes the recording with the Vultr server (after `/health`) and the local fallback. It prints PASS, WARN or FAIL per line and exits 1 if anything failed. A mic peak under 3000 or an empty transcript is a WARN: speak louder or check the mic, and run it again.

Every event goes to `audio/logs/audio-YYYYMMDD.jsonl` with `mono` and `wall` timestamps. `listen`, `echo`, `say` and `selftest` write a `session` line first, and `listen` and `echo` print the session's numbers when they exit. `stats` reads them back:

- wake detections and their scores
- speech starts by source; utterances, with empty ones by reason; starts left without an utterance (should be 0)
- end of speech to `on_utterance`, per STT backend; STT time per backend; fallbacks
- user starts talking (first voiced frame) to playback actually stopped
- `stop_speech` to silence (the stream abort)
- `say` to first audio for uncached speech
- errors by place

## Acceptance run on the Pi (section 14.3)

With the USB headset, after `selftest` passes:

| Check | Command | What to do |
| --- | --- | --- |
| Wake detections, 20 tries | `python -m audio.cli listen --no-tts --scores` | Say "hey glance" 20 times at normal voice, a few seconds apart, then Ctrl+C. Detections are in the summary |
| False wakes, 10 min | `python -m audio.cli listen --no-tts --minutes 10` | Play or hold 10 min of nearby conversation. Wake detections in the summary are false wakes |
| STT exact match (Vultr) | `python -m audio.cli record ...` x 40, then `python -m audio.cli stt-bench --out audio/clips/bench-pi.csv` | Half quiet, half with background chatter (`--tag quiet` / `--tag noisy`) |
| End of speech to `on_utterance` | `python -m audio.cli listen --no-tts` | 20 commands and requests with the wake word, then Ctrl+C. Read the `vultr` line |
| Talking over TTS to stopped | `python -m audio.cli echo` | Say a long request; while it is read back, say "stop". Repeat 10 times |
| `stop_speech` to silence | same session | Read the `stop_speech -> silence` line |
| `say` to first audio | `python -m audio.cli say "<a 20-word sentence>"` x 5 | Use different sentences (uncached), then `python -m audio.cli stats --all` |
| 30 min soak | `python -m audio.cli echo --minutes 30` | Talk now and then. No crash, `unanswered starts: 0`, and it still answers at the end |

## Results (section 14.3)

Not measured yet. Laptop numbers so far (Windows, default devices, not the targets' hardware): `stop_speech` to silence 31 ms; `say` to first audio 389 ms for a 112-character sentence; GPU stand-in STT p50 320 ms, p95 514 ms over Tailscale (CLAUDE.md decision 17).

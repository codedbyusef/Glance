# audio/: wake word, speech-to-text, text-to-speech

## 1. Overview

This directory is Glance's voice I/O. It hears the wake phrase "hey glance", detects when the user starts and stops speaking, turns speech into text, and speaks the agent's replies. The agent (`guide_agent/`) decides what to say. This component only moves audio in and out and reports events.

None of this is the novel part of the project. Use proven components in their standard configuration. No custom models, except training the "hey glance" wake phrase with openWakeWord's standard notebook.

Two halves:

- **Pi side** (the Raspberry Pi the user wears or carries): mic capture, wake word, voice activity detection, endpointing, push-to-talk, STT client with a local fallback, TTS (Piper) and playback.
- **Vultr side**: one small STT service, faster-whisper on the GPU.

Working rules:

- This file is the spec. The contracts in section 5 are shared with the agent. Don't change them without asking.
- If the spec is silent or contradicts itself, ask instead of guessing.
- 36-hour hackathon: prefer the simplest thing that works reliably. No streaming STT, no custom DSP, no new ML.
- Build in isolation. Don't import from or edit anything outside `audio/` (superseded at integration by the root CLAUDE.md). The agent's interfaces are restated in section 5. Integration happens later (section 15).
- Every unit test runs with no mic, speaker, GPU, network, or model files. Hardware and models sit behind small interfaces with fakes.
- Secrets come from environment variables. Never log tokens. Don't save audio unless `save_clips` is on.
- Nothing on the Pi's audio path may crash the process. Device errors, server errors, and model errors are logged and recovered from (section 12).

## 2. Target platform

- Raspberry Pi 5 (a Pi 4 should also work), 64-bit Raspberry Pi OS. Bookworm (Python 3.11) or Trixie (Python 3.13). Every dependency in section 4 has aarch64 wheels for both.
- Audio hardware: a wired USB headset, or a USB sound card with a mic and wired headphones. The Pi 5 has no 3.5 mm jack and no mic input. Don't use Bluetooth: it adds latency, its headset mic profile reduces audio quality, and pairing is fragile at a venue.
- Headphones, not a speaker. Barge-in listens while TTS plays, so a speaker would make the system hear and interrupt itself.
- Use the system default audio devices (PipeWire on current Pi OS), which resample for us. Device names are configurable.
- CPU budget: the agent controller, camera capture, and the glove link share this Pi. Everything on the steady-state audio path is light (openWakeWord, WebRTC VAD, Piper). The only heavy piece is the fallback Whisper, and it runs only when Vultr is unreachable.

## 3. Data flow

```
 USB mic ─► capture (16 kHz mono int16, 80 ms blocks)
              │
              ├─► wake phrase (openWakeWord, "hey glance") ─┐
              ├─► VAD + endpointer (WebRTC VAD)          ├─► events ─► sink.on_speech_start(at)
              └─► push-to-talk (key or GPIO button)     ─┘            sink.on_utterance(text, ended_at, frame)
                                │ finished utterance (WAV)
                                ▼
                     STT client ──HTTPS──► Vultr STT service (faster-whisper small.en, GPU)
                         └─ on failure ─► local faster-whisper tiny.en (Pi CPU)

 agent ─► Player.say / stop_speech / cue ─► Piper (Pi CPU) ─► USB headphones
                 └─► listener(speech_id, completed | interrupted)
```

## 4. Component choices

| Job                    | Choice                                                       | Notes                                                                                                                                                                                                              |
| ---------------------- | ------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Audio I/O              | `sounddevice` (PortAudio)                                    | Needs `libportaudio2` from apt                                                                                                                                                                                     |
| Wake word              | `openwakeword` 0.6.0, ONNX backend, custom `hey_glance.onnx` | Install with `--no-deps` (see below). Feature models come from `openwakeword.utils.download_models()`                                                                                                              |
| VAD and endpointing    | `webrtcvad-wheels`                                           | Prebuilt aarch64 wheels. 20 ms frames                                                                                                                                                                              |
| STT, primary           | `faster-whisper` on the Vultr GPU, `small.en`, float16       | English-only model, since we only need English                                                                                                                                                                     |
| STT, fallback          | `faster-whisper` on the Pi CPU, `tiny.en`, int8              | Used only when the primary fails or times out                                                                                                                                                                      |
| STT server             | FastAPI + uvicorn                                            | Same Vultr instance as the perception server, separate process and port                                                                                                                                            |
| TTS                    | `piper-tts` 1.8, voice `en_US-lessac-medium`                 | Runs on the Pi CPU. Python API: `PiperVoice.load(path)`; `voice.synthesize(text)` yields `AudioChunk`s with `sample_rate` and `audio_float_array`. Check the installed version's API before relying on other names |
| Push-to-talk on the Pi | `gpiozero` (preinstalled on Pi OS)                           | Optional GPIO button                                                                                                                                                                                               |
| HTTP client            | `httpx`                                                      | Explicit timeouts                                                                                                                                                                                                  |

openWakeWord install: its Linux metadata requires `tflite-runtime`, which has no aarch64 wheel for Python 3.11 or 3.13. Install it with `pip install --no-deps openwakeword`, then install its real runtime dependencies yourself (`onnxruntime`, `numpy`, `scipy`, `scikit-learn`, `tqdm`, `requests`), and always construct the model with `inference_framework="onnx"`. Put this in `audio/README.md` and in the install script.

Ruled out:

- Porcupine: Picovoice disabled free-tier AccessKeys after June 30, 2026, and the engine won't start without a valid key.
- TTS on Vultr: the agent has to say "guidance stopped because of a connection problem" exactly when the Vultr link is down, so speech output can't depend on Vultr.
- Kokoro or other PyTorch TTS on the Pi: too heavy for a CPU that's shared with the rest of the system.
- Streaming or partial transcription: the agent only uses final utterances.
- Bluetooth audio (section 2).

License note: `piper-tts` is GPL-3.0 and the repo is MIT. That's fine for the hackathon; flag it in the README.

## 5. Contracts with the agent

These restate `guide_agent/CLAUDE.md` section 5 and section 19 decisions 5 to 7, plus `guide_agent/interfaces.py`. Signatures and semantics must match exactly, so integration is just passing objects in.

### 5.1 Output: `Player` implements `AudioOut`

```python
class AudioOut(Protocol):
    def say(self, text: str, expect: str) -> str: ...      # returns speech_id
    def stop_speech(self) -> None: ...
    def cue(self, name: str) -> None: ...
    def set_listener(self, listener: Callable[[str, str], None]) -> None: ...  # (speech_id, "completed" | "interrupted")
```

- `say` only queues. It returns a unique speech_id immediately and plays the speech after everything already queued. It never calls the listener from inside `say`.
- Every speech gets exactly one report, `completed` or `interrupted`, in queue order.
- `completed` means the last sample has actually been played: written to the device, and the stream's output latency has passed.
- `stop_speech` stops playback within about 50 ms, empties the queue, and reports `interrupted` for every speech it stopped. It reports them **synchronously, on the caller's thread, before returning**, as the agent's `StubAudio` does. The controller calls it on its own event loop thread and relies on this order.
- A speech that finished just before `stop_speech` was called is reported once, as `completed`. No double reports: guard each id with a lock.
- `expect` is `none`, `done`, or `answer`. It sets the reply window (section 6).
- `cue("thinking")` plays a short pre-rendered tone (about 150 ms, soft) right away if nothing is playing, and is dropped otherwise. It has no speech_id and no report. Unknown cue names are logged and ignored.

### 5.2 Input: events into a sink

```python
class AudioEvents(Protocol):
    def on_speech_start(self, at: float | None = None) -> None: ...
    def on_utterance(self, text: str, ended_at: float | None = None, frame: object | None = None) -> None: ...
```

At integration these are the agent controller's own methods. In isolation, a printing or recording sink stands in.

- `on_speech_start` fires when the user starts talking: on wake word detection, on VAD speech start when no wake word is needed (section 6), or on a push-to-talk press.
- `on_utterance` fires once per finished utterance, with the transcript (which may be empty), `ended_at` (the end-of-speech time), and `frame`.
- `frame` comes from an injected `frame_source()` callable, called **at end of speech, before the STT request**, so the frame shows what was in front of the user when they stopped talking. In isolation `frame_source` returns `None`. It must be thread-safe (the camera owner's job).
- Every `on_speech_start` is followed by exactly one `on_utterance`, even if empty: no speech after a wake word, audio rejected as noise, all STT backends failing, or a hallucination filtered out. The controller restores the pending step on an empty utterance, so this is always safe.
- Exception: the user starts speaking, stops before `min_utterance_ms`, and then speaks again within the same listening session. That's the same utterance.

### 5.3 Threading

The controller is asyncio-based and starts turns with `asyncio.get_running_loop().create_task`, so its methods must run on the event loop thread.

- Every callback that crosses into the agent goes through an injected `dispatch(fn, *args)`, which at integration is `loop.call_soon_threadsafe`. That covers `on_speech_start` and `on_utterance` from the mic worker, and playback reports from the player thread.
- The one exception is section 5.1: `stop_speech` reports synchronously because it's already called on the loop thread.
- In isolation, `dispatch` calls the function directly.

## 6. Listening policy

The mic is in one of these states, and the policy decides whether the wake word is needed.

| State     | Entered when                       | Behavior                                                                                                                                                       |
| --------- | ---------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `wake`    | Wake word needed (rule below)      | Run the wake word only. A detection fires `on_speech_start` and moves to `capture`                                                                             |
| `open`    | Wake word not needed               | Run VAD. A speech start fires `on_speech_start` and moves to `capture`                                                                                         |
| `capture` | Wake word, VAD start, or PTT press | Record with pre-roll until end of speech (or PTT release, or `max_utterance_s`). Then grab the frame, transcribe, fire `on_utterance`, and recompute the state |

The wake word is needed only when all of these hold:

1. The player is idle: nothing playing and nothing queued.
2. The reply window is closed.
3. `task_active()` is false. `task_active` is an injected callable; in isolation it returns `False`.

Reply window:

- It opens on `say` with `expect` of `done` or `answer`.
- It closes when an utterance with non-empty text is delivered, or when a later `say` has `expect` `none`.
- It never times out on its own. The agent's idle timer handles check-ins (agent decision 6).

So the user can say "stop" or "cancel" without the wake word while speech is playing or a reach is running, as the agent assumes.

After a wake word with no speech within `no_speech_after_wake_s`, deliver `on_utterance("")`.

Push-to-talk works in every state. A press fires `on_speech_start` and starts `capture`; release ends it, with no VAD endpointing. PTT is the fallback for a noisy hall, so it must always work, even if the wake word model is missing.

Capture after a wake detection starts at the detection, with no pre-roll, so the wake phrase stays out of the clip. Pre-roll applies only to VAD and PTT starts. If "glance" still shows up in a transcript ("hey glance, what's this"), leave it: the controller strips its `assistant_name` (Glance). A leftover "hey" would break the agent's whole-utterance intercepts ("hey stop" doesn't match "stop"), which is why there's no pre-roll after a wake.

## 7. Endpointing

Capture runs at 16 kHz mono int16 in 80 ms blocks (1280 samples). openWakeWord wants multiples of 80 ms. Each block is also split into four 20 ms frames for WebRTC VAD. If the device can't open at 16 kHz, open it at its native rate and resample with `scipy.signal.resample_poly`.

| Key                    | Default | Meaning                                                                                                         |
| ---------------------- | ------- | --------------------------------------------------------------------------------------------------------------- |
| vad_aggressiveness     | 2       | WebRTC VAD mode, 0 to 3                                                                                         |
| speech_start_ms        | 200     | Voiced audio within a 300 ms window needed to count as speech start (filters clicks)                            |
| end_silence_ms         | 700     | Unvoiced audio that ends an utterance                                                                           |
| preroll_ms             | 300     | Audio kept from before a VAD or PTT start, so the first syllable isn't clipped. Not used after a wake detection |
| min_utterance_ms       | 250     | Shorter captures are noise: send `on_utterance("")` without calling STT                                         |
| max_utterance_s        | 10      | Hard stop                                                                                                       |
| no_speech_after_wake_s | 4       | Wake word but no speech                                                                                         |
| wake_threshold         | 0.5     | openWakeWord score threshold. Tune on the Pi                                                                    |
| wake_vad_threshold     | 0.5     | openWakeWord's built-in VAD gate; 0 turns it off                                                                |
| wake_cooldown_s        | 1.5     | Ignore repeat detections after a wake                                                                           |

All values are starting points, not measurements. Tune them on the Pi with the real headset (section 14.3).

## 8. Wake phrase: "hey glance"

- A custom openWakeWord model for "hey glance" is being trained with the project's automatic training notebook (Google Colab). The output is saved at `audio/models/hey_glance.onnx` and committed (it's small).
- Development doesn't wait for it: until the file exists, load openWakeWord's pretrained `hey_jarvis` model as a stand-in, chosen by config, and log a warning. PTT works either way.
- False triggers and misses are tuned with `wake_threshold` only, measured per section 14.3. If the model misses too often, retraining with more samples is the next step. That's a model-file swap, with no code change.

## 9. STT service (Vultr)

`audio/server/stt_service.py`, run with `uvicorn audio.server.stt_service:app --host 0.0.0.0 --port 8001`.

- At startup it loads `WhisperModel(model, device="cuda", compute_type="float16")` once, then runs one warm-up transcription on a second of silence before reporting ready.
- Settings: `language="en"`, `beam_size=5`, `vad_filter=True`, `condition_on_previous_text=False`, `initial_prompt` from config (default: `"done, yes, no, stop, wait, cancel, repeat"`). Test with and without the prompt on the clip set, and keep it only if it helps.
- One transcription at a time behind a lock. Requests are seconds long and come from one user.
- Auth: header `Authorization: Bearer <token>`, with the token from the `GLANCE_STT_TOKEN` env var. Any mismatch returns 401.
- Exposed the same way as the perception server (HTTPS front door or SSH tunnel). Keep the firewall to the needed port.

API:

```
GET  /health      -> 200 {"ready": true, "model": "small.en"}            (503 while loading)
POST /transcribe  body: WAV, 16 kHz, mono, PCM16, at most 15 s           (413 if longer, 415 if not that format)
     -> 200 {"text": "get me the can", "duration_s": 1.9, "infer_ms": 142,
             "no_speech_prob": 0.02, "avg_logprob": -0.21, "model": "small.en"}
```

`no_speech_prob` is the maximum over segments and `avg_logprob` is the duration-weighted mean. With no segments, return `text: ""`, `no_speech_prob: 1.0`, `avg_logprob: 0`.

## 10. STT client (Pi)

`Transcriber` interface: `transcribe(wav_bytes) -> SttResult(text, backend, ms, dropped_reason)`.

- **Primary:** POST to `GLANCE_STT_URL` with a `stt_timeout_s` timeout (default 2.0) and no retry.
- **Fallback:** on timeout, connection error, or non-200, run local `tiny.en` int8 on the CPU. Load the model lazily on first use, and preload it at startup only if `fallback_preload: true`. Log every fallback.
- If both fail, return empty text with `dropped_reason="stt_failed"`.
- Filter, then return empty text with a reason:
  - `no_speech_prob > 0.6` and `avg_logprob < -1.0` (Whisper's usual no-speech rule).
  - Text that's empty after normalization.
  - Text entirely in brackets or parentheses (`[BLANK_AUDIO]`, `(music)`).
  - An exact match against `hallucination_blocklist` (default: "you", "thanks for watching", "thank you for watching").
- Normalize by trimming whitespace and collapsing internal spaces. Don't change case or punctuation, since the agent does its own normalization.

## 11. TTS player (Pi)

`audio/player.py`, class `Player` implementing section 5.1.

- One Piper voice loaded at startup. One synth thread turns queued text into float32 audio, working ahead on the next item while the current one plays. One writer thread owns a single `sounddevice.OutputStream`, opened at startup, and writes in 50 ms chunks, checking a stop flag between chunks.
- Resample to the output stream's rate if it differs from the voice's (`resample_poly`).
- Before synthesis, strip `* _ # \``, collapse whitespace, and skip empty text: report it `completed`asynchronously, never from inside`say`.
- Cache: synthesize the fixed controller phrases at startup ("Okay, cancelled.", "Sorry, I had a problem. Please say that again.", "Sorry, I had a problem. Do you want to keep going?"), and keep an LRU cache of the last 16 synthesized texts. The repeat intercept re-says the same text.
- `speaking_rate` config maps to Piper's length scale (1.0 is the voice default). Blind users often prefer faster speech, so make it easy to change.
- `busy` property: true while anything is playing or queued. The listening policy reads it.
- If the output device fails, log it, report the affected speeches `interrupted`, and reopen the stream every 2 s until it works.

## 12. Reliability

- Mic device lost: log it, deliver `on_utterance("")` if a capture was in progress, and retry opening every 2 s. The process keeps running.
- STT server down: fallback per section 10. `GET /health` is checked at startup and logged; it doesn't block startup.
- Wake word model missing: use the stand-in (section 8) or PTT only. Never crash.
- Any exception in a worker thread is caught, logged with a traceback, and the worker restarts its loop.
- A `--selftest` CLI flag plays a test sentence, records 3 s, transcribes it with both backends, and prints the results. Run it first after each boot at the venue.

## 13. Configuration, logging, layout

Config: defaults live in `audio/config.py`, and an optional `audio/config.yaml` overrides them. Unknown keys are an error. Keys: every key in section 7, plus `input_device`, `output_device`, `sample_rate` (16000), `wake_model` (path or a pretrained name), `ptt` (`none`, `key`, or `gpio:<pin>`), `stt_timeout_s`, `fallback_model` (`tiny.en`), `fallback_preload` (false), `initial_prompt`, `hallucination_blocklist`, `piper_voice` (path to the `.onnx`), `speaking_rate`, `save_clips` (false), `log_dir` (`logs`). Env vars: `GLANCE_STT_URL`, `GLANCE_STT_TOKEN`.

Logging: one JSON line per event in `audio/logs/audio-YYYYMMDD.jsonl`. Events: `wake` (score), `speech_start` (source: wake, vad, or ptt), `speech_end` (duration_ms), `stt` (backend, ms, chars, dropped_reason), `utterance_delivered`, `say_queued` (speech_id, chars, expect), `synth` (speech_id, ms), `playback_start`, `playback_done` (status), `cue`, and `error`. Each event carries a monotonic timestamp, so latency can be computed later. With `save_clips` on, WAVs go to `logs/clips/`. It's off by default because the expo is a public space.

Layout:

```
audio/
  CLAUDE.md  README.md  requirements-server.txt  config.py   (Pi/laptop deps: runtime/requirements.txt at the repo root)
  capture.py        mic stream, resampling, device recovery
  wake.py           openWakeWord wrapper (Detector interface + fake)
  vad.py            WebRTC VAD + endpointer state machine (pure, testable on synthetic frames)
  listen.py         listening policy + Mic (wires capture, wake, vad, ptt, stt, sink)
  ptt.py            keyboard and GPIO push-to-talk
  stt_client.py     Transcriber: VultrWhisper, LocalWhisper, filtering
  player.py         Player (AudioOut), Synth interface (Piper + fake), output backend (sounddevice + fake)
  events.py         AudioEvents protocol, PrintSink, RecordingSink
  log.py            JSONL event log
  cli.py            dev commands (section 14.2)
  server/stt_service.py
  models/           hey_glance.onnx (committed), piper voices (gitignored, fetched by script)
  scripts/fetch_models.sh
  tests/
```

## 14. Testing

### 14.1 Unit tests (no hardware, no models, no network)

- Endpointer: synthetic voiced/unvoiced frame sequences. Start needs `speech_start_ms`, end needs `end_silence_ms`, pre-roll is included, min and max lengths are enforced.
- Listening policy: every row of section 6, the reply window open and close rules, and PTT in every state.
- Mic event contract: every `on_speech_start` gets exactly one `on_utterance`, including no speech after wake, too-short captures, STT failure, and filtered text. The frame is requested before STT is called. All sink calls go through `dispatch`.
- Player with a fake synth and a fake output backend:
  - Queue order, unique ids, and one report per speech.
  - Nothing reported inside `say`.
  - `stop_speech` reports synchronously for all queued speeches.
  - The completed-versus-stop race reports once.
  - Cue dropped while speaking.
  - Cache hits.
  - Device failure path.
- STT client against a fake server (httpx mock transport): success, timeout leading to fallback, non-200 leading to fallback, both failing, and every filter rule.
- STT service with the Whisper model stubbed: auth, format checks (413, 415), response shape, and health before and after load.

### 14.2 Dev CLI (`python -m audio.cli <cmd>`)

- `say "text" [--expect done]`: speak through the real Player and print the reports.
- `listen`: real mic, wake word, VAD, and STT with a `PrintSink`. Prints each event with timings.
- `echo`: listen, then say the transcript back. This tests the full loop without the agent.
- `record --label "get me the can"`: record one utterance to `clips/` and append the label to `clips/labels.csv`.
- `stt-bench clips/`: transcribe every clip with each backend. Prints exact-match accuracy against the labels (after lowercasing and stripping punctuation) and p50/p95 latency per backend. This produces the Vultr-versus-Pi numbers for the prize writeup.
- `selftest`: section 12.

### 14.3 On-Pi acceptance (with the real headset)

Record a clip set of at least 40 utterances: the commands (done, yes, no, stop, wait, cancel, repeat), the demo requests ("what is on the desk", "guide me to the paper", "read this paper", "get me the can"), half in quiet and half with background chatter. Then measure and write the results in `audio/README.md`. Targets:

| Check                                                      | Target                                                 |
| ---------------------------------------------------------- | ------------------------------------------------------ |
| Wake word detections, 20 tries at normal voice             | at least 19                                            |
| False wakes during 10 min of nearby conversation           | at most 1                                              |
| STT exact match on command clips (Vultr)                   | at least 95%                                           |
| End of speech to `on_utterance`, Vultr path                | p50 under 1.2 s (700 ms of it is the end-silence wait) |
| User starts talking during TTS to playback stopped         | under 300 ms                                           |
| `stop_speech` to silence                                   | under 100 ms                                           |
| `say` to first audio (uncached sentence of about 20 words) | measure and record                                     |
| 30 min soak running `echo`                                 | no crash, no stuck state                               |

## 15. Integration (later, not in this build)

The runtime glue that eventually connects everything will:

- Create the event loop. Set `dispatch = loop.call_soon_threadsafe`. Pass `Player` as the controller's `audio` and the controller as the Mic's sink.
- Set `task_active = lambda: controller.task is not None` and `frame_source = camera.latest_frame`.
- Run `controller.tick()` every 100 ms.

Agent-side changes are for the agent owner, not this build: set `assistant_name: Glance` in `guide_agent/config.yaml`, and let `Harness` accept an injected audio object, so a `voice_chat` script can reuse `WebcamHarness` with real voice.

## 16. Milestones and definition of done

1. **STT:** service, client, filters, `stt-bench`, `record`, and their unit tests. The service runs on the Vultr GPU and `stt-bench` runs on a clip set.
2. **TTS:** Player with Piper, cache, cue, `say` CLI, and the contract tests. It plays on the Pi through the USB headset.
3. **Listening:** capture, wake word (stand-in until `hey_glance.onnx` exists), VAD endpointer, PTT, policy, Mic, `listen` CLI, and unit tests.
4. **Loop and hardening:** `echo`, `selftest`, local fallback STT on the Pi, device recovery, JSONL logging, then the section 14.3 acceptance run with results recorded.

Done means: all unit tests pass, every section 14.3 row is measured and recorded with misses explained, and the README covers Pi setup from a fresh OS install, including the openWakeWord `--no-deps` step and the model fetch script.

## 17. Decisions (resolved 2026-10-17)

These resolve gaps found while planning. They take precedence over earlier sections where they differ.

1. `max_utterance_s` stays 10. The agent's `speech_start_timeout_s` is raised to 20 by the agent owner; the one-`on_utterance`-per-`on_speech_start` guarantee makes that timeout only a safety net. `fallback_preload` defaults to true, so the first fallback doesn't pay the model load.
2. `at` and `ended_at` use `time.time()`, matching the agent's `RealClock`. Every log line carries both `wall` (`time.time()`) and `mono` (`time.monotonic()`).
3. Utterance length runs from the first voiced frame to the last voiced frame, excluding pre-roll and trailing silence. A burst shorter than `min_utterance_ms` doesn't end the capture: it stays open for up to `no_speech_after_wake_s`; speech that resumes is the same utterance, otherwise `on_utterance("")` is delivered.
4. Key push-to-talk (dev only): Enter toggles, press to start and press again to stop. GPIO push-to-talk uses real press and release.
5. The STT server reads only env vars: `GLANCE_STT_TOKEN`, `GLANCE_STT_MODEL` (default `small.en`) and `GLANCE_STT_INITIAL_PROMPT` (default `"done, yes, no, stop, wait, cancel, repeat"`; an empty string means no prompt). It doesn't load `config.py` or `config.yaml`. To A/B the prompt, restart the server with a different value and rerun `stt-bench`. The `initial_prompt` config key is dropped from the Pi side.
6. New config key `assistant_name` (default `"Glance"`). A transcript that normalizes (lowercase, punctuation stripped) to only "glance" or "hey glance" (using `assistant_name`) is delivered as usual but doesn't close the reply window.
7. Bystander speech can trigger VAD in `open` state. Mitigations are config only: the close-talk headset mic, `vad_aggressiveness` 3 in a loud room, and PTT. `speech_start` logs its source and `stt` logs the character count so this can be measured.
8. Extra dependencies: PyYAML for `config.yaml`; WAV handling uses the stdlib `wave` module. `sounddevice`, `openwakeword`, `piper`, `faster_whisper` and `webrtcvad` are imported lazily inside the real backends, so the unit tests run without them (including on Windows).
9. Player: a `stop_speech` that arrives while the writer is waiting out the output latency after the last sample reports `interrupted`. On an output device failure, the current speech and everything queued are reported `interrupted`; while the device is down, new speeches are reported `interrupted` asynchronously, and the stream is reopened every 2 s.
10. Player reports go through an outbox that is drained on the dispatch thread. `stop_speech` drains the outbox first, then reports `interrupted`, so a speech that completed just before the stop is reported `completed` before later ones are reported `interrupted`, keeping queue order.
11. Audio that arrives while the Mic is transcribing is discarded. An `on_utterance("")` delivered without an STT call passes `frame=None`.
12. An `expect` other than `none`, `done` or `answer` is logged and treated as `none`.
13. The Mic is constructed with the Player (its `busy` and the shared reply window), in addition to the section 15 wiring.
14. The STT server also reads `GLANCE_STT_DEVICE` (`cuda`, the default, with float16; `cpu` with int8), so it can run on a dev laptop. A request that waited more than 10 s for the model lock is answered 503 without inference: its client has already timed out, and running it would only delay the requests behind it.
15. Until the Pi exists, a Windows laptop is the test machine. `say`, `listen`, `record`, `echo` and `stt-bench` must work there with the default mic and speakers. `requirements-dev.txt` at the repo root installs everything (it replaced `requirements-laptop.txt` in the integration Step 0), and `scripts/fetch_models.py` (which `fetch_models.sh` wraps) fetches models on any OS.
16. Initial prompt A/B on `tiny.en`, 10 real laptop clips (all quiet), local CPU server: with the prompt 70% exact and 83% on commands (misses: "stop" heard as "stop, stop...", "guide me to the paper" as "and guide me to the paper.", "desk" as "disc"). Without it, 70% exact and 67% on commands (misses: "cancel" as "Can't sue.", "stop" returned empty, "desk" as "disc"). In-process local fallback (`tiny.en`, beam 1, no prompt): 90% exact, 100% on commands. The prompt stays the default; rerun the A/B with `small.en` on the GPU. `GLANCE_STT_INITIAL_PROMPT=none` also means no prompt, since PowerShell deletes an env var set to `""`. The server logs its model, device, beam size and prompt at startup and returns them from `/health`.
17. Until Vultr is set up, the STT server runs on a Windows PC with an RTX 3070, reached over Tailscale (e.g. `http://100.112.104.87:8001`). Nothing in the client changes except `GLANCE_STT_URL`. On Windows the server makes the pip `nvidia-cublas-cu12` and `nvidia-cudnn-cu12` (cuDNN 9) DLLs loadable with `os.add_dll_directory` before importing faster-whisper (`requirements-server-windows.txt`). `stt-check` calls `/health` and transcribes the bundled `assets/stt_check.wav` ("Get me the can.", rendered with Piper).
18. `audio/models/hey_glance.onnx` exists and is the default `wake_model`. The `hey_jarvis` stand-in is used only when the configured file is missing.
    GPU stand-in measured (RTX 3070 over Tailscale, `small.en`, cuda, prompt on): `stt-check` round trips 907 / 312 / 313 ms with server inference 531 / 93 / 110 ms; `stt-bench` on the 10 clips 90% exact and 100% on commands, p50 320 ms, p95 514 ms (only miss: "desk" heard as "disk"). The initial prompt stays the default.
19. The wake phrase is kept out of every clip. After a detection the capture, and the `no_speech_after_wake_s` window, start when the phrase has ended: at the first 60 ms unvoiced gap after the detection block, when the score falls below `wake_end_threshold` (0.3), or after `wake_end_max_s` (1.0 s), whichever comes first. Audio before that is dropped; there is still no pre-roll. The gap rule is there because the score lags: on Piper clips it stays above 0.9 for 300 to 400 ms after "glance" ends, so the falling edge alone cut "Hey Glance, stop" to nothing and "Hey Glance, get me the can" to "See the can". `on_speech_start` still fires at the detection. The wake model runs in every state; a detection during a VAD or wake capture (outside the cooldown) drops that capture's audio up to the end of the phrase, by the same rule, without a second `on_speech_start`. A detection during a push-to-talk capture is ignored. As a backup, the Mic strips a leading "hey glance" / "hey, glance" / "glance" (from `assistant_name`, ignoring case and punctuation) plus the punctuation and space after it from transcripts, so a transcript of only the wake phrase is empty and doesn't close the reply window. This replaces section 6's "leave it" for the wake phrase.
20. Noise rejection in `open` state: `vad_aggressiveness` defaults to 3 and `speech_start_ms` to 300, so all of a 300 ms window must be voiced to start (which also makes 300 ms the effective minimum for VAD-endpointed speech). A voiced frame keeps a capture going only when more than half of the last 300 ms is voiced, so scattered noise no longer holds it open; a capture never ends on a voiced frame, so speech resuming after a pause isn't cut while the ratio builds back up.
21. The STT server warms up with two passes on `assets/stt_check.wav` using the request settings (VAD filter and prompt), so the first real request doesn't pay for first-run setup. If the clip is missing it warms up on a second of silence with the VAD filter off.
22. Milestone 4 measurement tooling: `listen`, `echo`, `say` and `selftest` log a `session` event first; `speech_start` logs `at` (the first voiced frame, wall time); `utterance_delivered` logs the STT `backend`; the Player logs `playback_cut` (ms from `stop_speech` to the stream abort). `python -m audio.cli stats` computes the section 14.3 numbers from the JSONL log, and `listen`/`echo` print them on exit. `--minutes` bounds a `listen` or `echo` run for the soak. `selftest` is also available as the `--selftest` flag.
23. Prompt echo filter (client side, all backends): a transcript is dropped with `dropped_reason="prompt_echo"` when it has more than 2 words that all come from the default initial prompt, or any prompt word appears 3 or more times (case and punctuation ignored). It came from small.en transcribing a 2.2 s noise capture as "cancel, repeat, cancel, repeat, cancel, repeat, cancel,". The prompt words come from `stt_client.DEFAULT_PROMPT`, which the server also uses.
24. Wake tone: on every wake detection the Mic calls `player.cue("wake")`, a 90 ms 990 Hz tone, which the Player drops if anything is playing (as for "thinking"). Config key `wake_tone` (default true). "wake" is a Player-internal cue name; the agent's contract still uses only "thinking".
25. `stats --session last|all|<id>`: `last` (the default) is the last `listen` or `echo` session; session ids are the start time (HHMMSS) and are printed in each summary. `selftest` marks an empty transcript from any backend, and a mic peak under 3000, as WARN.
26. Push-to-talk only (integration Step 1, for speakers instead of headphones): config key `ptt_only` (default false). When on, the Listener's idle state is `ptt`: no wake word (no model is loaded) and no open-state VAD start, so speaker output can't start a capture. Push-to-talk works as in section 6 and decision 4, and a press during playback fires `on_speech_start` (a barge-in). The section 5 contracts are unchanged. `build_mic` also takes an optional `ptt` source, so the runtime can own stdin.

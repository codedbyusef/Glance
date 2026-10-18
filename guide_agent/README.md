# Guidance agent

The LLM-driven controller for the blind-user guidance system. `CLAUDE.md` is the spec; section 19 records decisions made while building it.

## Setup

Run everything from the **repo root**, in the root venv (see the root README):

```
python -m venv .venv
.venv\Scripts\pip install -r requirements-dev.txt
.venv\Scripts\pip install -e .
```

Put `GEMINI_API_KEY=...` in `guide_agent/.env` (gitignored) or the environment.

## Controller tests (17.1)

```
.venv\Scripts\python -m pytest guide_agent
```

No network: a scripted fake model, stub interfaces and a fake clock.

## Harness (17.2)

```
.venv\Scripts\python -m guide_agent.harness
.venv\Scripts\python -m guide_agent.harness --script guide_agent\harness_scripts\reference.txt
```

Type what the user says, or `/help` for commands (`/frame`, `/speech_start`, `/idle`, `/haptic <status>`, ...). `harness_scripts/` holds the 17.6 reference flow and its two variants. `--images DIR` loads fixture photos from another folder.

## Eval (17.3 to 17.5)

```
.venv\Scripts\python -m guide_agent.evals.run                     # every case, 3 runs
.venv\Scripts\python -m guide_agent.evals.run --case halt --runs 1
```

Cases are in `evals/cases/`, fixture photos in `fixture_frames/` (see `fixture_frames/manifest.json`). Cases whose photos are missing are reported as skipped. Reports go to `guide_agent/evals/reports/<timestamp>/`.

## Webcam chat (typed utterances, live camera, no glove)

```
.venv\Scripts\python -m guide_agent.scripts.webcam_chat [--camera 1] [--no-window]
```

## Layout

| Path | What |
|---|---|
| `controller.py` | Triggers, intercepts, task state, tool execution, restore, timers (sections 6 to 13) |
| `tools.py` | Tool schemas and validation (section 9) |
| `compose.py` | Per-turn model input (section 11) |
| `prompts/system.md` | System prompt (sections 14 and 15) |
| `models/` | Provider-neutral interface, Gemini adapter, scripted test model |
| `interfaces.py`, `stubs.py` | External interfaces (section 5) and their stubs |
| `turnlog.py` | JSONL turn log (section 16), written to `guide_agent/logs/` |
| `evals/`, `fixture_frames/`, `harness_scripts/` | Eval runner and cases, fixture photos, harness scripts |
| `scripts/webcam_chat.py` | Typed chat with live webcam frames |
| `tests/` | Controller tests (`tests/rig.py` is the shared test rig) |

Real components implement the methods in `interfaces.py`. Call `controller.tick()` every 100 ms or so to drive the idle, speech_start and inactivity timers.

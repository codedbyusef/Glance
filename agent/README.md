# Guidance agent

The LLM-driven controller for the blind-user guidance system. `CLAUDE.md` is the spec; section 19 records decisions made while building it.

## Setup

```
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

Put `GEMINI_API_KEY=...` in `agent/.env` (gitignored) or the environment. Run everything from `agent/`.

## Controller tests (17.1)

```
.venv\Scripts\python -m pytest
```

No network: a scripted fake model, stub interfaces and a fake clock.

## Harness (17.2)

```
.venv\Scripts\python -m guide_agent.harness
.venv\Scripts\python -m guide_agent.harness --script harness_scripts\reference.txt
```

Type what the user says, or `/help` for commands (`/frame`, `/speech_start`, `/idle`, `/haptic <status>`, ...). `harness_scripts/` holds the 17.6 reference flow and its two variants. `--images DIR` loads fixture photos from another folder.

## Eval (17.3 to 17.5)

```
.venv\Scripts\python -m evals.run                     # every case, 3 runs
.venv\Scripts\python -m evals.run --case halt --runs 1
```

Cases are in `evals/cases/`, fixture photos in `fixtures/` (see `fixtures/manifest.json`). Cases whose photos are missing are reported as skipped. Reports go to `evals/reports/<timestamp>/`.

## Layout

| Path | What |
|---|---|
| `guide_agent/controller.py` | Triggers, intercepts, task state, tool execution, restore, timers (sections 6 to 13) |
| `guide_agent/tools.py` | Tool schemas and validation (section 9) |
| `guide_agent/compose.py` | Per-turn model input (section 11) |
| `guide_agent/prompts/system.md` | System prompt (sections 14 and 15) |
| `guide_agent/models/` | Provider-neutral interface, Gemini adapter, scripted test model |
| `guide_agent/interfaces.py`, `stubs.py` | External interfaces (section 5) and their stubs |
| `guide_agent/turnlog.py` | JSONL turn log (section 16), written to `logs/` |

Real components implement the methods in `interfaces.py`. Call `controller.tick()` every 100 ms or so to drive the idle, speech_start and inactivity timers.

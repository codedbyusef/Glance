# Guidance agent: spec and working rules

## 1. Overview

This repo contains the **agent**: the LLM-driven controller of an assistive guidance system for blind users. The user wears glasses with a camera, talks to the system by voice, and wears a haptic glove that guides their hand during the final reach to an object. The agent answers questions about what the camera sees and runs physical tasks ("get me the can", "open the door") one step at a time, using spoken instructions for moving and handing off to haptic guidance for the reach.

Speech recognition, speech output, camera capture, the perception and haptic guidance loop, and the glove firmware are separate components. They appear here only as interfaces (section 5).

Working rules:
- This file is the spec. Tool names, fields, enums, and event schemas are contracts with other components. Don't change them without asking. When a decision changes, update this file in the same change.
- If the spec is silent or contradicts itself, ask instead of guessing.
- This is built during a 36-hour hackathon. Prefer the simplest design that meets the spec.
- Nothing depends on hardware. Every external component is a stub behind the interfaces in section 5 and can be replaced by the real one without changing controller logic.
- The model provider sits behind one adapter. Switching providers must not change controller logic.
- API keys come from environment variables. Never commit keys. Never write image bytes or keys to logs; log frame ids.
- A change to the system prompt or controller behavior is done only after the eval suite (section 17) has been run against it. Fix failures at their cause. Don't loosen a test unless it contradicts this spec.

## 2. Scope

In scope:
- Controller: triggers, intercepts, task state, notes, tool validation and execution, interruption and staleness rules.
- System prompt and per-turn input.
- Model provider adapter.
- Stubs for every external interface.
- Interactive harness, deterministic controller tests, eval runner with scripted cases, logging.

Out of scope: wake word detection, speech-to-text, text-to-speech, camera capture, the perception and haptic loop, glove firmware, deployment.

Deferred (not in v1): a `look()` tool for extra frames mid-turn; queueing a second task; streaming partial model output to speech; tasks that go beyond touching or grasping plus one simple final action.

## 3. Data flow

```
audio layer --speech_start, utterance(text, frame)--> controller <--> model API
haptic loop --haptic_result(event, frame)-----------> controller
controller  --say, stop_speech, cue----------------> audio layer
controller  --start, pause, resume, cancel---------> haptic loop
controller  --latest_frame()-----------------------> camera
```

## 4. Configuration

| Key | Default | Meaning |
|---|---|---|
| assistant_name | Nova (placeholder until the team picks one) | Used in the system prompt; also the wake word. It is stripped from every utterance, so it must not be a common word |
| glove_hand | right | Hand wearing the glove |
| provider, model | Gemini API, gemini-3.8-flash | Needs image input, function calling, and several function calls in one response. Default because the event has an MLH Best Use of Gemini API challenge |
| thinking_level | low | Gemini adapter option, chosen for turn latency |
| judge_model | same as model | Grades `judge` rubrics in the eval |
| temperature | 0.2 | |
| max_model_calls_per_turn | 4 | |
| recent_turns_max | 10 | |
| recent_turns_max_age_s | 600 | |
| history_max_steps | 8 | Most recent steps shown |
| notes_max_chars | 600 | |
| step_max_words | 25 | |
| answer_max_words | 40 | Not applied when reading text the user asked for |
| idle_timeout_done_s | 30 | |
| idle_timeout_answer_s | 15 | |
| max_idle_checkins | 2 | |
| speech_start_timeout_s | 8 | |
| task_inactivity_close_s | 600 | |
| frame_max_long_edge_px | 1024 | |

## 5. Interfaces

Inputs:

| Event | Fields | Notes |
|---|---|---|
| speech_start | at | The user started speaking (wake word, or a reply the audio layer is listening for) |
| utterance | text, ended_at, frame | Wake word removed from text. Frame captured when the user stopped speaking, or null |
| haptic_result | see section 10 | Sent when a guidance session ends |

Frame: `{id, image (JPEG), captured_at}`. Not mirrored: image left is the user's left. Null when the camera is unavailable.

Assumed of the audio layer: it delivers "stop" and "cancel" without the wake word while speech is playing or a reach is running.

Outputs:

| Call | Meaning |
|---|---|
| say(text, expect) | Speak. `expect` is none, done, or answer, and tells the audio layer whether to listen for a reply without the wake word. Reports completed or interrupted when playback ends |
| stop_speech() | Stop playback now |
| cue("thinking") | Short sound when a model turn starts after a user utterance |
| haptic.start(target, refer) | Start guidance; returns session_id. On failure the controller synthesizes a haptic_result with status stale_link |
| haptic.pause / resume / cancel (session_id) | Pause stops vibration immediately |
| camera.latest_frame() | Newest frame or null |

## 6. Turns

A turn is the controller's response to one trigger: user_utterance, haptic_result, or idle_timeout.

- Each turn starts a fresh model conversation: the system prompt plus one composed input message (section 11). Earlier turns reach the model only through that message.
- The controller calls the model, executes the tool calls in order, returns results for non-terminal calls, and calls the model again, until a terminal action executes or max_model_calls_per_turn is reached.
- Terminal actions: speak, reach, finish_task, or a response with text and no tool calls (a plain answer). Non-terminal: start_task.
- One terminal action per turn. The prompt tells the model to send start_task and the first step in one response.

Plain answers:

| State | Action |
|---|---|
| No active task | say(text, none) |
| Active task with a pending step | Say the text, then restore the pending step (section 13) |
| Active task, no pending step | Convert to a speak step with expect answer; warning no_pending_step |

Tool errors are returned to the model and the turn continues:

| Code | When |
|---|---|
| task_active | start_task while a task is active and replace_active is not true |
| no_active_task | speak, reach, or finish_task with no active task |
| invalid_args | Missing required field, empty string, enum violation, target over 5 words |

Protocol warnings are logged and counted by the eval. The runtime keeps going:

| Code | When | Runtime behavior |
|---|---|---|
| extra_terminal | More than one terminal action in a response | Execute the first, drop the rest |
| no_pending_step | Plain text in an active task with no pending step | Convert to speak with expect answer |
| missing_notes | speak or reach without notes | Keep the previous notes |
| notes_truncated | Notes over notes_max_chars | Truncate |
| step_too_long, answer_too_long | Over the word limits | Still spoken |
| turn_cap | Call limit reached without a valid terminal action | Fallback below |

Info, logged only: text_with_tool (free text next to a terminal tool call is dropped).

Turn-cap fallback: if a task is active with no pending step, record a speak step "Sorry, I had a problem. Do you want to keep going?" with expect answer. Otherwise say "Sorry, I had a problem. Please say that again." and restore any pending step.

## 7. Intercepts

Checked on every utterance before any model call. Normalize by lowercasing and removing punctuation, the wake word, and "please". A phrase matches only if it is the whole normalized utterance, optionally repeated ("stop stop"). "What does that stop sign say" and "stop, I got it" are not intercepted.

| Class | Phrases | Action |
|---|---|---|
| halt | stop, wait, hold on, pause | stop_speech; cancel any reach session; end the current step with result interrupted. If a task is active, run a model turn for the utterance with intercepted="halt". Otherwise nothing more |
| cancel | cancel, never mind, forget it | stop_speech; abandon any in-flight turn; cancel any reach session. If a task is active, close it as cancelled and say "Okay, cancelled." No model turn |
| repeat | repeat, repeat that, say that again, what, sorry, pardon | Say the last spoken output again, if any, with its original expect. If it was the pending step's speech, that speech now counts as heard fully. Then restore the pending step. No model turn |

## 8. Task state

The controller owns this. The model sees it only through the per-turn input.

```
task:
  id, goal, request              # request: the user's words that started the task
  started_at, last_user_activity_at
  current_step: step or null
  history: [step]                # ended steps, oldest first
  notes: string
  outcome: null | success | failed | cancelled | superseded

step:
  n, type: speak | reach, issued_at
  speak fields: text, expect (done | answer)
  reach fields: target, refer, say_first, session_id
  state: speaking | awaiting_reply        (speak)
         starting | running | paused      (reach; starting = say_first playing)
  heard_fully: bool
  result: null | string                   # set when the step ends
```

Lifecycle:
- start_task opens a task. Only one task is active at a time.
- A new speak or reach ends the current step and becomes the current step. The ended step's result records what ended it: "user: <utterance>" in a user turn, "no reply" in an idle turn, "interrupted" for a halt.
- A reach step ends when its haptic_result arrives. Its result is the status, and current_step is null until the model issues the next step.
- A task closes through finish_task, the cancel intercept (cancelled), start_task with replace_active (superseded), or inactivity (cancelled). Closing cancels any reach session.

Invariant: after every completed turn, either no task is active or the active task has exactly one pending step (a speak step that is speaking or awaiting a reply, or a reach step that is starting, running, or paused).

## 9. Tools

Descriptions can be expanded in the implementation. Names, fields, and enums cannot change.

```json
[
  {
    "name": "start_task",
    "description": "Open a task for a request that needs the user to move or act. Send the first step in the same response.",
    "parameters": {
      "type": "object",
      "properties": {
        "goal": {"type": "string", "description": "Short goal, at most 12 words."},
        "replace_active": {"type": "boolean", "description": "True only when the user clearly wants to drop the active task for this one."}
      },
      "required": ["goal"]
    }
  },
  {
    "name": "speak",
    "description": "Give one instruction or ask one question, then wait for the user's reply. Ends your turn.",
    "parameters": {
      "type": "object",
      "properties": {
        "text": {"type": "string"},
        "expect": {"type": "string", "enum": ["done", "answer"]},
        "notes": {"type": "string"}
      },
      "required": ["text", "expect", "notes"]
    }
  },
  {
    "name": "reach",
    "description": "Start haptic guidance of the gloved hand to a target within arm's reach. Ends your turn.",
    "parameters": {
      "type": "object",
      "properties": {
        "target": {"type": "string", "description": "1 to 4 word noun phrase for the object detector. No positions."},
        "refer": {"type": "string", "description": "What tells this object apart from similar ones."},
        "say_first": {"type": "string", "description": "Spoken before guidance starts."},
        "notes": {"type": "string"}
      },
      "required": ["target", "refer", "say_first", "notes"]
    }
  },
  {
    "name": "finish_task",
    "description": "End the active task. Ends your turn.",
    "parameters": {
      "type": "object",
      "properties": {
        "outcome": {"type": "string", "enum": ["success", "failed", "cancelled"]},
        "summary": {"type": "string", "description": "One line for the log."},
        "say": {"type": "string", "description": "Optional closing words for the user."}
      },
      "required": ["outcome", "summary"]
    }
  }
]
```

Execution:

| Tool | Effect |
|---|---|
| start_task | If replace_active is true, close the active task as superseded. Open the task. Return {task_id} |
| speak | New speak step; say(text, expect). The idle timer starts when playback completes |
| reach | New reach step in state starting; say(say_first, none); when playback completes, haptic.start(target, refer) and state running |
| finish_task | Close the task; say(say, none) if present |

## 10. Haptic result event

```json
{
  "event": "haptic_result",
  "session_id": "s-17",
  "status": "target_lost",
  "inferred": false,
  "elapsed_s": 6.1,
  "target": "red soda can",
  "last_seen": {"region": "right-middle", "depth": "near", "age_s": 2.3},
  "closest_approach": "near",
  "frame": {"id": "f-311", "captured_at": "2026-10-17T14:05:02.118"}
}
```

| Field | Values |
|---|---|
| status | success, target_not_found, target_lost, hand_lost, stalled, timeout, stale_link |
| inferred | true when success was inferred because the hand covered the target at the end |
| last_seen | null if never acquired. region: left, center, or right, plus top, middle, or bottom (image thirds). depth: near, mid, far |
| closest_approach | touching, near, far |
| frame | null on success; the frame at loop end for every other status |

Status meanings:
- success: the hand reached the target and stayed there.
- target_not_found: the target was never acquired.
- target_lost: acquired, then lost for too long without the hand covering it.
- hand_lost: the gloved hand was out of camera view for too long.
- stalled: the hand stopped getting closer.
- timeout: the time cap passed without success.
- stale_link: the guidance connection or perception service failed.

The haptic loop also has a cancelled status for sessions the controller cancels. It never reaches the model. Events whose session_id doesn't match the current step's session are dropped.

## 11. Per-turn model input

Built fresh every turn, in this order. At most one image: the trigger's frame. Exceptions: no image on haptic success or when the frame is null; idle_timeout turns use camera.latest_frame().

```
<task_state>
goal: get the red can
request: "get me the red can on the table"
current_step: n=2 type=speak state=awaiting_reply expect=done heard_fully=true age_s=12
  text: "Hold your hands out at waist height and step forward slowly until you touch the table, about two steps, then say done."
history:
  1 speak "The table is about four steps ahead, slightly to your right. Turn slightly right and take two steps forward, then say done." -> user: done
notes: Step 2. Table about 2 steps ahead, can on its right half at waist height. Remaining: walk to the table, reach for the can, grasp.
</task_state>
<recent_turns>
[41s] user: get me the red can on the table
[39s] assistant: The table is about four steps ahead, slightly to your right. Turn slightly right and take two steps forward, then say done.
[30s] user: done
[28s] assistant: Hold your hands out at waist height and step forward slowly until you touch the table, about two steps, then say done.
</recent_turns>
<trigger type="user_utterance" intercepted="none">done</trigger>
<frame id="f-203" age_ms="380"/>
```

Variants:
- No task: `<task_state>none</task_state>`
- Haptic result: `<trigger type="haptic_result">{event JSON without session_id and frame}</trigger>`
- Idle: `<trigger type="idle_timeout" checkin="1"/>`
- No image: `<frame>none</frame>`

recent_turns holds user utterances, everything spoken, and bracketed events (task started or closed, reach started, haptic status, intercepts), each with its age in seconds.

## 12. Notes

- Required on speak and reach. The model rewrites the whole field each time. The controller stores it verbatim and shows it every turn. Plain answers, finish_task, and intercepts leave it unchanged.
- Frames aren't kept between turns, so anything the model learned has to survive as text.
- Content: body-relative spatial facts tagged with the step they came from, facts the user confirmed, and a one-line sketch of what remains.
- Facts go stale once the user moves. When the notes and the current frame disagree, the frame wins.

## 13. Interruptions and concurrency

- Triggers are handled one turn at a time, in arrival order.
- speech_start: stop_speech; if the current step's speech was cut off, set heard_fully to false; pause a running reach; stop the idle timer. If no utterance follows within speech_start_timeout_s, or the utterance is empty, restore the pending step.
- An utterance that arrives while a turn is in flight abandons that turn: its model output is discarded and never executed. The new utterance gets the next turn. Effects the abandoned turn already executed stay, and the next turn repairs the state through the section 6 fallbacks.
- Any new step or finish_task cancels a paused or running reach session first.
- haptic_result events for sessions other than the current step's are dropped.

Restore a pending step (after a plain answer, a repeat, an empty utterance, or a speech_start timeout):
1. If the step's speech (speak text or reach say_first) wasn't heard fully, say it again. It now counts as heard fully.
2. Speak step: re-arm the reply wait and restart the idle timer. The last thing said carries the step's expect. If nothing was re-said in 1, the preceding answer is said with the step's expect.
3. Reach step: resume the session, or start it if it never started.

Idle timer: starts when a speak step's playback completes, stops on speech_start, and is frozen during turns. When it expires, an idle_timeout trigger fires with a check-in count. The count resets on any utterance. After max_idle_checkins check-ins in a row, no more idle triggers fire until the next utterance, and the step stays pending.

Inactivity: a task with no user activity for task_inactivity_close_s closes silently as cancelled.

## 14. Agent behavior

This is the content the system prompt must encode.

### 14.1 Setting
- The user is blind and knows nothing about their surroundings unless told.
- The camera is on their glasses and points where their face points. Image left is the user's left.
- The frame was captured when the trigger happened (end of the user's speech, or end of a reach) and may be slightly old. It is the only image. With no frame, make no visual claims and say the camera isn't available.
- During a reach the user feels the glove on their {glove_hand} hand and knows what each vibration means.

### 14.2 Speech
- Plain spoken sentences. No markdown, lists, emojis, or symbols.
- A step is one instruction: at most 2 sentences and step_max_words.
- Plain answers: at most answer_max_words. When reading text the user asked to hear, read it verbatim; if it's long, read the start and offer to continue.

### 14.3 Language for a blind user
- Never ask the user to look, see, spot, or watch. No visual-only references ("over there", "that one").
- Directions are relative to where the face points: ahead, behind, left, right, or clock positions (12 is straight ahead). Heights are relative to the body: knee, waist, chest, head.
- Distances are approximate steps, said with "about".
- Checks use touch, sound, or the user's own report, never sight.

### 14.4 Questions and tasks
- A question about the scene or text gets a plain answer with no tools.
- A request that needs the user to move or act gets start_task and the first step in the same response.
- If such a request is ambiguous, start the task and ask (speak with expect answer).
- With no frame, don't start a task. Explain that the camera isn't available.

### 14.5 Steps and moving
- One step per turn. Each step says what to say next: "then say done" for an action, a direct question for an answer.
- The first step of a task includes a short orientation: where the target is relative to the user.
- Walking: at most 3 steps per instruction. Prefer moves that end on contact ("step forward until your hands touch the table").
- After a turn of the head, tell the user to turn their body to match before walking.
- If the target isn't in view, ask for a bounded turn of head and body ("turn a quarter to your right"), then "say done".
- After every "done", check the new frame and correct with small moves.

### 14.6 Reach
- Call reach only when the target looks within arm's reach in the current frame. Otherwise guide the user closer first.
- If the target is near the bottom edge of the frame, first ask the user to tilt their head down so the target and hand stay in view.
- target: 1 to 4 words naming the object, with color or material if useful ("red soda can", "door handle"). No positions.
- refer: what tells it apart from similar objects: color, shape, material, the surface it's on. Positions are relative to the current frame.
- say_first: tell the user to bring their {glove_hand} hand out in front of them where the camera can see it, keep their head still, and follow the vibrations.
- At most 2 reach attempts per target. After that, ask the user how to proceed or finish as failed.

### 14.7 Haptic results

| Status | Expected response |
|---|---|
| success | Touch check, with the final action if the goal needs one (14.8) |
| target_not_found | Bounded turn of head and body, then "say done"; or reach again with a revised target or refer if the frame shows the object |
| target_lost | Ask the user to pull the hand back toward their chest, then "say done"; reach again after |
| hand_lost | Reach again with say_first stressing where to hold the hand, or a speak step to bring it into view |
| stalled | Ask whether something is in the way (expect answer), or reach again with a hint |
| timeout | closest_approach near: reach again with a short hint. Far: reposition the user with speak steps first |
| stale_link | Say guidance stopped because of a connection problem and reach again once. On a second stale_link, finish as failed |

### 14.8 Success
- After a success, inferred or not, issue one step with expect answer that checks by touch and gives the final action if there is one. Example: "Your hand should be on the can. Close your fingers around it and say yes if you have it."
- Yes: finish_task with outcome success and a short confirmation.
- No or unsure: ask what they feel, then reach again with a refined refer, or reposition.
- Tasks end at touching or grasping plus at most one simple action, such as pressing a handle or lifting.

### 14.9 During a task
- A question gets a plain answer and no step tool. End with a short reminder of the pending instruction when useful.
- A correction to the current task ("no, the blue one") gets a new speak or reach step. Never start_task.
- A clear switch to another task: start_task with replace_active true, plus its first step.
- A clear request to drop the task for a question: finish_task with outcome cancelled and the answer in say.
- If it's unclear whether the user wants to switch, ask (speak with expect answer).
- After a halt: confirm you stopped and ask whether to keep going (speak with expect answer).
- Idle check-in: one short sentence restating the pending instruction.

### 14.10 Notes
The rules in section 12.

### 14.11 Safety
- Name obstacles and hazards in the path in the same sentence as the move.
- Never say a path is clear unless the frame shows it.
- Sharp objects: guide to the handle and warn about the edge. Hot items: warn before the reach.
- No guidance on stairs, across roads, or near drops. Say it isn't safe and suggest sighted help.
- Describe people neutrally. Never identify anyone.

## 15. System prompt structure

In order:
1. 14.1, with assistant_name and glove_hand filled in.
2. 14.2.
3. Protocol: the tools, terminal and non-terminal, one step per turn, start_task together with the first step, and what each trigger type means.
4. 14.3 to 14.6.
5. Status meanings (section 10) and 14.7.
6. 14.8, 14.9, 14.10, 14.11.
7. Compact worked examples: a plain question with no task; a task start; "done" leading to reach; a side question during a pending step; success, touch check, finish; target_lost recovery.

## 16. Logging

One JSON line per turn: turn_id, task_id, trigger (type, text or event), intercept, composed input text and frame id, each model call (raw response, latency, token counts if available), tool calls with validation results, warning and info codes, output calls with timestamps, and task state before and after. Abandoned turns are logged with abandoned: true. Turn latency is measured from trigger arrival to the first output call (say or haptic.start).

## 17. Testing

### 17.1 Controller tests
Deterministic, with a scripted fake model and no API calls. At least one test per rule in sections 6, 7, 8, 9, 12, and 13, including:
- Each intercept class, and the non-matches in section 7.
- start_task alone continues the loop; start_task plus a step in one response executes both in order.
- Each tool error and recovery on the next call.
- Each protocol warning and its runtime behavior, including the turn-cap fallback.
- Restore: speech heard fully or cut off, speak or reach step, reach that never started.
- An utterance abandoning an in-flight turn; a stale haptic_result being dropped; cancel during an in-flight turn.
- Idle timer, check-in limit, reset on utterance; inactivity close.
- haptic.start failure producing stale_link.

All must pass.

### 17.2 Interactive harness
Real model, stubbed interfaces. Type utterances, pick each frame (fixture id or none), fire speech_start and idle_timeout, and choose the haptic_result when a reach starts. Print every output call and the task state after each turn.

### 17.3 Eval runner
Runs the model cases in 17.4 against the real model, 3 runs per case by default. The report shows pass rate per case and per category, warning counts, words per spoken output, and turn latency p50 and p95. Cases with missing fixtures are reported as skipped.

Case format:

```json
{
  "id": "done_reach",
  "category": "task_flow",
  "setup": {
    "task": {"goal": "get the red can", "request": "get me the red can", "current_step": {}, "history": [], "notes": ""},
    "recent_turns": [],
    "last_spoken": ""
  },
  "steps": [
    {
      "trigger": {"type": "user_utterance", "text": "done", "frame": "can_reach"},
      "expect": {
        "terminal": ["reach"],
        "args": {"target": {"contains_any": ["can"]}},
        "spoken": {"contains_all": ["right hand"]},
        "state_after": {"task": "active", "step_type": "reach"},
        "judge": "Tells the user to keep their head still."
      }
    }
  ]
}
```

- Matchers, case-insensitive: equals, in, contains_any, contains_all, not_contains_any, max_words, regex.
- terminal lists the acceptable terminal actions: plain, speak, reach, finish_task.
- spoken applies to whatever is said: plain text, speak text, reach say_first, finish_task say.
- judge is an optional rubric graded by judge_model.
- Multi-step cases script haptic_result triggers.
- Global checks on every spoken output: no banned phrases (default list: "look at", "you can see", "you'll see", "over there", "arm's length"), no markdown symbols, and word limits (a case can turn off answer_max_words).

### 17.4 Model cases (minimum set)

| id | Setup, trigger, frame | Expected |
|---|---|---|
| describe_scene | No task; "what am I looking at"; desk_scene | plain |
| read_text | No task; "what does this say"; sign_text | plain, contains the fixture's expected_text |
| no_frame | No task; "what's in front of me"; none | plain, says the camera isn't available, no scene claims (judge) |
| start_far | No task; "get me the can"; can_far | start_task + speak (done), at most 3 steps of walking |
| start_reach | No task; "get me the can"; can_reach | start_task + reach |
| start_not_visible | No task; "get me the can"; no_can | start_task + speak with a bounded turn |
| start_ambiguous | No task; "get me the can"; two_cans_reach | start_task + speak (answer) asking which |
| hazard | No task; "get me the can"; chair_in_path | start_task + speak that mentions the chair |
| knife | No task; "help me pick up the knife"; knife_reach | start_task + reach; handle in refer or say_first; warns about the blade |
| tilt_down | Pending walk step; "done"; can_reach_low | speak asking the user to tilt their head down |
| done_next | Pending walk step; "done"; can_mid | speak (done) moving closer |
| done_reach | Pending walk step; "done"; can_reach | reach |
| side_q_speak | Pending speak step; "what else is on the table"; can_mid | plain; step unchanged |
| side_q_reach | Running reach; "what color is it"; can_reach | plain; reach resumes |
| correction | Pending step after asking which can; "no, the blue one"; two_cans_reach | speak or reach, blue in refer, no start_task |
| switch_clear | Active can task; "forget the can, find the door"; door_far | start_task (replace_active true) + step |
| switch_unclear | Active can task; "can you find the door"; door_far | speak (answer) asking whether to switch; no start_task |
| drop_for_question | Active can task; "forget it, what does this sign say"; sign_text | finish_task (cancelled) with the text in say |
| idle_checkin | Pending speak step; idle_timeout, check-in 1 | speak restating the step |
| halt | Running reach; "stop"; can_reach | controller cancels the session before the model turn; model returns speak (answer) asking whether to continue |
| success_check | Reach step; success | speak (answer), touch based |
| verify_yes | Pending touch check; "yes, I have it" | finish_task (success) |
| verify_no | Pending touch check; "no, it feels like a wall" | speak or reach; not finish_task success |
| not_found | Reach step; target_not_found; no_can | speak with a bounded turn, or reach with a revised target |
| target_lost | Reach step; target_lost; can_hand_occluding | speak asking to pull the hand back |
| hand_lost | Reach step; hand_lost; can_reach | reach or speak bringing the hand into view |
| stalled | Reach step; stalled; can_reach | speak (answer) or reach with a hint |
| timeout_near | Reach step; timeout, closest near; can_reach | reach |
| timeout_far | Reach step; timeout, closest far; can_mid | speak repositioning |
| stale_link_twice | One earlier stale_link; stale_link again | finish_task (failed) |
| attempt_limit | Two failed reaches for the can; target_lost | no third reach: speak (answer) or finish_task (failed) |
| can_full | The reference flow in 17.6 | each step as listed |
| door_full | "open the door" from door_far through success and "yes" | ends with finish_task (success) |

### 17.5 Fixture frames
Phone photos with the real demo objects, landscape, no zoom, taken from eye level. "Within reach" photos are taken with the head tilted down so the object sits near the middle of the frame, except can_reach_low. Each fixture has a manifest entry: id, file, description, and expected_text for text fixtures. If a photo is missing, keep its manifest entry marked missing and skip the cases that need it.

| id | Content |
|---|---|
| desk_scene | A table with several everyday objects, about 2 m away |
| sign_text | A printed sign with short known text, about 1.5 m away |
| can_far | A red can on a table, about 3 m ahead |
| can_mid | Same, about 1.5 m ahead |
| can_reach | Same, within arm's reach, no hand in view |
| can_reach_low | Same, within reach, taken facing straight ahead so the can is at the bottom edge |
| two_cans_reach | A red can and a blue can side by side, within reach |
| no_can | The same area with no can |
| can_hand_occluding | The gloved hand partly covering the can |
| chair_in_path | A chair between the camera and the table with the can |
| door_far | A closed door about 3 m ahead |
| door_handle_reach | The door handle within arm's reach |
| knife_reach | A kitchen knife on a table within reach |

### 17.6 Reference flow

| # | Trigger (frame) | Expected action |
|---|---|---|
| 1 | "get me the red can on the table" (can_far) | start_task("get the red can") + speak("The table is about four steps ahead, slightly to your right. Turn slightly right and take two steps forward, then say done.", expect done) |
| 2 | "done" (can_mid) | speak("Hold your hands out at waist height and step forward slowly until you touch the table, about two steps, then say done.", expect done) |
| 3 | "done" (can_reach) | reach(target "red soda can", refer "the red can on the table", say_first "Bring your right hand out in front of you, keep your head still, and follow the vibrations.") |
| 4 | haptic_result success | speak("Your hand should be on the can. Close your fingers around it and say yes if you have it.", expect answer) |
| 5 | "yes" | finish_task(success, say "Great, you have the can.") |

Variants: a question during step 3 pauses the reach, gets a plain answer, and the reach resumes. target_lost instead of success at step 4 leads to a speak step asking the user to pull the hand back toward their chest, then a second reach.

Wording is illustrative. The eval checks action types and constraints, not exact text.

## 18. Definition of done

- All controller tests pass.
- The harness runs the reference flow and both variants with stubbed haptics.
- The eval runner runs every case 3 times and writes the report.
- On the final run: at least 90% of model cases pass, protocol warnings occur in under 5% of turns, and there are no global-check violations.

## 19. Decisions (resolved 2026-10-17)

These resolve gaps and contradictions found while planning. Items marked (contract) depend on another component behaving as described; confirm with its owner.

1. Notes are exempt from invalid_args. A missing, empty, or non-string notes field is the missing_notes warning (previous notes kept).
2. Tool errors stop the response. On the first tool error, the remaining calls in that response are not executed and get a `skipped` result. Calls already executed keep their effects. The model is called again.
3. target validation stays at "over 5 words". The prompt asks for 1 to 4.
4. An unknown tool name is invalid_args ("unknown tool").
5. speech_start while a turn is in flight holds that turn: its model call may finish, but nothing executes until the utterance arrives (the turn is abandoned) or the speech turns out empty or times out (the held turn then executes).
6. (contract) The audio layer keeps its reply window open until it delivers a non-empty utterance. Restoring a speak step with nothing to re-say only restarts the idle timer.
7. (contract) say() queues behind current playback, each say reports its own completion, and stop_speech interrupts everything queued.
8. A restored reach resumes (or starts) when the speech said during the restore finishes, the same as the first haptic.start. With nothing to say, it resumes immediately.
9. (contract) haptic_result.frame is a full section 5 Frame, image included. A haptic.start failure is synthesized as stale_link with frame camera.latest_frame(), last_seen null, closest_approach far, elapsed_s 0, inferred false.
10. A reach attempt is a reach step for the same object that ended with a status other than success. stale_link does not count (it has its own retry-once rule). Halted or corrected reaches do not count. The attempt_limit case has one failed reach in history and the current, second reach ending target_lost.
11. say_first is held to step_max_words. finish_task say is held to answer_max_words. When a case turns off answer_max_words, its answer_too_long warnings are excluded from the warning rate.
12. Eval case format additions: `expect.start_task` (true, false, or matchers on its args), `expect.outputs` (`includes`, `excludes`, `order` over output call names, where `model` marks the model call), `setup.camera_frame` (what camera.latest_frame() returns), and `trigger.speech_start` (default true: a speech_start fires before each utterance, like the real audio layer). `expect.args` matchers apply to the executed terminal tool; keys that tool doesn't have are skipped, so one case can accept speak or finish_task. `state_after` keys: task, step_type, step_state, expect, step_n, outcome (of the last closed task), goal. `{fixture:<id>}` in a matcher is replaced by that fixture's expected_text. A run fails on any global-check violation, and a multi-step case stops at its first failed step.
13. Frames for cases the table leaves open: idle_checkin can_mid (latest frame), verify_yes can_hand_occluding, verify_no can_reach, stale_link_twice can_reach, attempt_limit can_hand_occluding, door_full door_far, door_handle_reach, no frame on success, door_handle_reach for "yes".
14. assistant_name is the placeholder Nova. model is gemini-3.8-flash with thinking_level low. temperature stays 0.2.

Smaller defaults:
- Intercept phrases repeated within one class match ("stop stop", "sorry what"). Mixed classes ("stop cancel") go to the model.
- An utterance that abandons a turn takes the next turn ahead of queued triggers. Queued haptic_results are checked for staleness when their turn starts.
- An abandoned turn's model call is allowed to finish and is logged with abandoned: true; nothing it returns executes.
- Text next to a start_task with no step is dropped and logged as text_with_tool.
- A speak step moves from speaking to awaiting_reply when its playback ends, completed or cut off; heard_fully says which.
- last_user_activity_at updates on speech_start and on utterances.
- The idle timer length follows the step's expect: idle_timeout_done_s or idle_timeout_answer_s. Empty utterances do not reset the check-in count.
- recent_turns_max counts lines. The current trigger is not listed. Cut-off speech is marked "(cut off)". Intercepts, task opens and closes, reach starts, and haptic statuses are bracketed lines.
- On haptic success the input carries `<frame>none</frame>` as section 11 says; the prompt explains that this is normal after a success.
- The judge sees the composed input text (task state, recent turns, trigger), the tool calls, and everything spoken. It does not see the image.

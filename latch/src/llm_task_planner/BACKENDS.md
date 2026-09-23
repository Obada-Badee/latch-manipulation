# LLM backends

Two providers, one loop. `llm_planner.py` contains no branch on provider — that
was the plan's swappability constraint, and adding the second backend is what
turned it from a claim into something checked.

```bash
# Claude (default)
export ANTHROPIC_API_KEY=...
ros2 run llm_task_planner llm_planner

# Groq
export GROQ_API_KEY=...
ros2 run llm_task_planner llm_planner --ros-args -p backend:=groq

# Groq, reproducing the predecessor project's exact model
ros2 run llm_task_planner llm_planner --ros-args \
    -p backend:=groq -p model:=llama-3.3-70b-versatile
```

## What the abstraction cost

The loop needed **no** change. What did have to move is the conversation history,
into the backends — because the two providers genuinely disagree about what a
message is:

| | Anthropic | OpenAI / Groq |
|---|---|---|
| assistant content | list of typed blocks | string + separate `tool_calls` array |
| tool results | `tool_result` blocks inside one **user** message | one message each, `role: "tool"` |
| failure signalling | `is_error` flag | no flag — has to be legible in the text |
| schema enforcement | server-side via `strict` | none guaranteed |
| prompt caching | `cache_control` on the frozen prefix | none |

A loop holding `messages` would need a provider branch to append to it. So each
backend keeps its own native history and exposes only neutral types: the loop
asks for a `Decision` and hands back `ToolResult`s, and never sees a message.

## Measured on the target command

*"Put the two yellow cubes in the red bin, then move the leftmost bolt next to
the red cube."* Same scene, same tool schemas, same system prompt. Scored
automatically: correct means both yellow cubes picked, the **leftmost** bolt
identified by largest y, and placement within ~0.12 m of the red cube.

| model | correct | failure mode |
|---|---|---|
| `openai/gpt-oss-120b` | 2/3, then **4/4** after the prompt fix below | — |
| `llama-3.3-70b-versatile` | **0/3** | never emitted a usable tool call |

`llama-3.3-70b-versatile` — the model the predecessor project used — failed two
distinct ways:

1. Malformed call syntax rejected by Groq's own parser with HTTP 400
   `tool_use_failed`:
   ```
   I will pick up the yellow cube at x=+0.242, y=+0.424.
   <function=pick":{"object_id":4}</function>
   ```
   Note the stray quote. Right intent, right object, fumbled syntax.
2. Prose describing what it would do, with no tool call at all.

`GroqBackend` repairs case 1 by handing the failed text back and asking for a
re-emit (`_create_with_repair`) — the closed loop applied to the model's own
output rather than to the robot's. That is deliberately *not* the predecessor's
approach, which gave its model an `error "Command not feasible"` escape hatch for
its own malformed output and moved on.

Case 2 is not repairable, which is why the default here is `gpt-oss-120b` rather
than the inherited model.

## The failure that was mine, not the model's

`gpt-oss-120b`'s one refusal:

> Could not move the leftmost bolt because it is at x=0.158 m, which is outside
> the robot's reachable range (0.25–0.75 m).

Wrong — reach is radial, `sqrt(0.158² + 0.618²) = 0.638 m`, comfortably in range.
But the prompt had said only *"the arm reaches roughly 0.25 m to 0.75 m from its
base"*, never stating the distance is `sqrt(x² + y²)`. A model reading that
naturally applies the band to the coordinate in front of it.

This would have been very easy to file as "the model is bad at spatial
reasoning". It was an ambiguous prompt. `prompts.py` now gives the formula, a
worked example using these exact coordinates, and an instruction not to
pre-screen objects for reach at all — the skill's own error is more reliable than
an estimate, and refusing a reachable object wastes a turn.

## Cost of the switch

* **No prompt caching on Groq.** The frozen system prompt and the tool schemas
  are re-sent and re-billed every turn. On the free tier this is not academic —
  see the rate-limit section below, where it killed three runs.
* **No server-side `strict`.** `_coerce_arguments` re-implements the validation
  locally — required arguments, unknown arguments, enum membership, and the
  numeric coercions models actually get wrong. A bad call becomes a tool *error*
  the model can read and retry, not an exception that ends the run.
* **No refusal stop reason.** A decline arrives as ordinary text with no tool
  call, which the loop already treats as "stopped without calling a tool".

## Rate limits are a design constraint, not an operational nuisance

Groq's free tier enforces two separate caps, and both were hit while bringing up
the memory and affordance layers. They are worth writing down because the first
one changed the design.

**Tokens per minute — 8000.** A live run died at `Requested 9799`. Two causes,
one of them a genuine bug:

1. Memory had committed 43 objects on a table holding 9 — every transient
   detector phantom became a permanent belief and was rendered into the prompt
   forever after. Fixed properly in `memory.py` (`MIN_SIGHTINGS`): one sighting
   is a detection, several is an object.
2. Every tool result carried a full world state, and they accumulated. After
   five steps the context held five tables, four describing positions that had
   since changed. `backends/base.py` now elides superseded scenes when a newer
   one arrives — **1302 characters, ~325 tokens, saved per stale result per
   turn** at a realistic 12-object scene.

The second is a correctness fix that happens to save tokens. The prompt asks the
model to trust the newest world state; removing the stale ones makes that true
by construction rather than by request. It applies to both backends — Anthropic
pays for stale scenes too, just in cache misses rather than rate limits.

After both fixes, requests measured 7083–7553 tokens: inside the limit, with the
system prompt and tools accounting for most of what remains.

**Tokens per day — 200000.** Reached after a day of bring-up testing. Nothing to
engineer around; it just means live LLM-loop testing is a finite daily resource
and the deterministic layers should be verified without the model wherever
possible. `tools/verify_skills.py` and the memory verification do exactly that.

## Adding a third provider

Add a module and one line in `backends/__init__.py`'s registry. Implement
`begin`, `step`, `add_results`. Do not touch the loop; if you find yourself
wanting to, the abstraction has leaked and that is worth fixing instead.

SDKs are imported lazily per backend, so having only one provider's package
installed is fine.

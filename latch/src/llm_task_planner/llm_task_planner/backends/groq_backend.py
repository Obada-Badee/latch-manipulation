"""Llama and friends, via Groq's OpenAI-compatible chat completions API.

Why this exists
---------------
Two reasons, and the second is the interesting one.

1. It is the provider the predecessor project used, so this makes the two
   directly comparable on the same robot, same scene, same skills - the only
   difference being the model doing the reasoning.

2. It proves the swappability claim. The plan says the loop must own nothing but
   sequencing; adding a second provider is the only way to find out whether that
   is true. It was *almost* true: the loop needed no change at all, but the
   message-history handling had to move into the backends (see base.py), because
   the two providers disagree about what a message is.

What is deliberately NOT copied from the old project
----------------------------------------------------
The predecessor asked Groq for free text and split the reply on newlines, then
sent each line over a socket. That is precisely the failure mode this project
exists to remove - it is why that system needed an `error "Command not feasible"`
escape hatch for its own malformed output. Groq's API supports OpenAI-style tool
calling, so tools are used here exactly as with Claude. Same schemas, same
dispatch, same closed loop.

Differences from the Anthropic backend, and what each costs
-----------------------------------------------------------
* **No prompt caching.** The frozen system prompt and tool definitions are
  re-sent and re-billed every turn. Harmless at this scale, but it is why
  prompts.CACHE_NOTE matters less here.
* **No `strict` schema enforcement.** Anthropic validates tool input against the
  schema server-side; the OpenAI-compatible surface does not guarantee it, and
  open-weight models do sometimes emit a malformed or hallucinated call. So this
  backend validates locally - see _coerce_arguments - and turns a bad call into a
  tool *error* the model can read and retry, rather than an exception that kills
  the run. That is the same closed-loop treatment a failed grasp gets.
* **Arguments arrive as a JSON string**, not a parsed object, and models
  occasionally wrap them in markdown fences or emit `"{}"` for a no-argument
  tool. Parsed defensively.
* **No refusal stop reason.** A declined request comes back as ordinary text
  with no tool call, which the loop already handles as "stopped without calling
  a tool".
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Sequence

try:
    from .base import Decision, LlmBackend, ToolCall, ToolResult, strip_scene
except ImportError:                                   # python3 groq_backend.py
    # Every module in this project runs its own self-test as `python3 <file>`,
    # which breaks a relative import. Fall back to an absolute one so the
    # argument-validation tests below stay one command away.
    import os as _os
    import sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    from base import Decision, LlmBackend, ToolCall, ToolResult, strip_scene

#: Default model, chosen by measurement rather than by inheritance.
#:
#: The predecessor project used `llama-3.3-70b-versatile`, and that model is
#: still one `model:=` away - but it is not reliable enough to drive an arm. On
#: the target command ("put the two yellow cubes in the red bin, then move the
#: leftmost bolt next to the red cube"), same scene and same schemas, it failed
#: in two distinct ways across repeated runs:
#:
#:   * emitted `<function=pick":{"object_id":4}</function>` - a stray quote,
#:     rejected by Groq's own parser with HTTP 400 tool_use_failed
#:   * produced prose describing what it would do and no tool call at all
#:
#: gpt-oss-120b completed the same task correctly and first-time, including the
#: two spatial groundings ("leftmost" -> largest y, "next to" -> an 80 mm offset).
#: Both models see identical tools and prompt, so the difference is the model.
DEFAULT_MODEL = "openai/gpt-oss-120b"

#: What the predecessor project used. Kept named here so the comparison is one
#: argument away:  --ros-args -p backend:=groq -p model:=llama-3.3-70b-versatile
LEGACY_MODEL = "llama-3.3-70b-versatile"

MAX_TOKENS = 4096

#: Deterministic by default. The old project used temperature 0.0 for the same
#: reason: a task planner driving a real arm should not roll dice about which
#: object to pick up.
DEFAULT_TEMPERATURE = 0.0

#: Attempts to get a parseable tool call out of the model before giving up.
#: See _create_with_repair - this is not retrying a flaky network, it is asking
#: the model to fix its own syntax, which is a thing open-weight models
#: measurably need.
TOOL_REPAIR_TRIES = 3

#: Waits, in seconds, between successive attempts at ONE request after the host
#: answers 429. Cumulative ~7 minutes, then the error is raised as before.
#:
#: This exists because the alternative - letting the 429 kill the trial and
#: having the benchmark restart the trial - cannot work for a long task. A
#: multi-object trial at 60 tool calls spends more than a minute's token
#: allowance by itself, so restarting it re-spends everything it had already
#: spent and trips the same ceiling again. Three restarts at 75 s lost every
#: `c1` and `c4` trial in a 75-trial run, and no other task - the rate limit
#: selected exactly the hardest cells out of the sample. Waiting here keeps the
#: conversation intact and resumes it, which costs nothing while it waits.
#:
#: The ladder is short at the start because most 429s clear within the minute,
#: and long at the end so a genuinely busy window is not given up on.
RATE_LIMIT_WAITS_S = (20.0, 40.0, 60.0, 90.0, 120.0, 120.0)


def to_openai_tools(schemas: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert tools.TOOL_SCHEMAS (Anthropic shape) to the OpenAI function shape.

    Converted rather than duplicated: one set of schemas, so a tool cannot exist
    for one provider and not the other, and a description edit cannot land in
    only half the system.

    `strict` is dropped - it is an Anthropic key and the OpenAI-compatible
    surface on Groq does not honour it, so leaving it in would imply a guarantee
    that is not there. Validation happens locally instead.
    """
    out: List[Dict[str, Any]] = []
    for t in schemas:
        out.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            },
        })
    return out


class GroqBackend(LlmBackend):
    """An OpenAI-shaped chat-completions backend, with Groq as the default host.

    Everything below the three hooks - `_env_key`, `_make_client` and
    `_bad_request_type` - is provider-neutral: the OpenAI tool schema, the
    message history, the scene elision, the tool-repair loop and the usage
    parsing are identical on any host that speaks `chat.completions`. Those
    hooks exist so a second such host is a subclass and not a 465-line copy,
    which would guarantee the two drift apart at the first bug fix.
    """

    name = "groq"

    #: Environment variable holding the key. Never read a key from source - a
    #: key in source ends up in git history, which is how the predecessor
    #: project's key ended up public.
    _env_key = "GROQ_API_KEY"

    def _make_client(self):
        from groq import Groq                                 # noqa: PLC0415
        return Groq()

    @staticmethod
    def _bad_request_type():
        """The exception class the SDK raises for a rejected request."""
        from groq import BadRequestError                      # noqa: PLC0415
        return BadRequestError

    @staticmethod
    def _rate_limit_type():
        """The exception class the SDK raises for a 429."""
        from groq import RateLimitError                       # noqa: PLC0415
        return RateLimitError

    def __init__(self, model: str, system: str, tools: Sequence[Dict[str, Any]],
                 temperature: float = DEFAULT_TEMPERATURE):
        super().__init__(model or DEFAULT_MODEL, system, tools)
        if not os.environ.get(self._env_key):
            raise RuntimeError(
                f"{self._env_key} is not set. export it, or choose another "
                f"backend with -p backend:=...")
        self._client = self._make_client()
        self.temperature = temperature
        self._openai_tools = to_openai_tools(self.tools)
        self._schema_by_name = {t["name"]: t["input_schema"] for t in self.tools}
        self._messages: List[Dict[str, Any]] = []
        #: How many times the model had to be asked to re-emit a tool call it
        #: had malformed. Worth counting rather than swallowing: it is the
        #: clearest single number separating providers on this task, and it
        #: belongs in the run log next to token usage.
        self.repairs = 0
        self.rate_limit_waits = 0

    # ------------------------------------------------------------------ history

    def begin(self, user_text: str) -> None:
        # The system prompt is an ordinary first message here, not a separate
        # parameter as with Anthropic.
        self._messages = [
            {"role": "system", "content": self.system},
            {"role": "user", "content": user_text},
        ]

    def add_results(self, results: Sequence[ToolResult]) -> None:
        # Drop the scene from every earlier result: this one supersedes them.
        # See base.SCENE_SEPARATOR. It matters more here than on Anthropic -
        # there is no prompt caching, so every stale table is re-billed on every
        # turn, and a live run died on the free tier's 8000 token/minute limit
        # with four of them in context.
        for message in self._messages:
            if message.get("role") == "tool":
                message["content"] = strip_scene(message["content"])

        # One message per result, each keyed by tool_call_id - the OpenAI shape.
        # Anthropic instead nests all results inside a single user message; that
        # difference is the whole reason history lives in the backend.
        for r in results:
            content = r.content
            if r.is_error:
                # No is_error flag exists here, so failure has to be legible in
                # the text itself or the model may read a failure as a success.
                content = f"ERROR: {content}"
            self._messages.append({
                "role": "tool",
                "tool_call_id": r.call.id,
                "content": content,
            })

    # ------------------------------------------------------------------- calling

    def _create_with_repair(self):
        """Call the API, repairing malformed tool syntax by asking again.

        Groq parses tool calls server-side and returns HTTP 400
        `tool_use_failed` when the model's output does not parse - the request
        errors out, so local validation never gets a chance. Observed on
        llama-3.3-70b-versatile, on the very first turn:

            I will pick up the yellow cube at x=+0.242, y=+0.424.
            <function=pick":{"object_id":4}</function>

        - a stray quote after the tool name. The model had the right intent and
        the right object; it fumbled the syntax.

        Ending the run there would be the same capitulation the predecessor
        project made when it gave its model an `error "Command not feasible"`
        escape hatch for its own malformed output. Instead the failed text is
        handed back with an instruction to re-emit, which is just the closed
        loop applied to the model's own output rather than to the robot's.

        Temperature is 0 by default, so a bare retry would reproduce the same
        bytes - the corrective message is what changes the input, and therefore
        the output. Returns None if it still cannot parse after TOOL_REPAIR_TRIES.
        """
        bad_request = self._bad_request_type()

        for attempt in range(TOOL_REPAIR_TRIES):
            try:
                return self._create_waiting_out_rate_limits()
            except bad_request as exc:
                failed = self._failed_generation(exc)
                if failed is None or attempt == TOOL_REPAIR_TRIES - 1:
                    if failed is None:
                        raise
                    return None
                self.repairs += 1
                self._messages.append({
                    "role": "user",
                    "content": (
                        "Your last reply was rejected because its tool call was "
                        "not valid JSON:\n\n"
                        f"{failed}\n\n"
                        "Call the tool again. Emit the call through the tool "
                        "interface only - no prose around it, no XML tags, and "
                        "the arguments as a plain JSON object matching the "
                        "tool's schema exactly."),
                })
        return None

    def _create_waiting_out_rate_limits(self):
        """One completion request, retried in place across per-minute 429s.

        Same messages, same everything: a 429 says nothing about the request
        except that it arrived in a busy minute, so the correct response is to
        wait for a quieter one and send the identical request. The trial's
        conversation is untouched, which is the whole point - see
        RATE_LIMIT_WAITS_S for why restarting the trial instead does not work.

        Counted in `self.rate_limit_waits` so a run can report how much of its
        wall time was spent waiting on the host rather than moving the arm.
        """
        rate_limited = self._rate_limit_type()
        for i, wait in enumerate(RATE_LIMIT_WAITS_S + (None,)):
            try:
                return self._client.chat.completions.create(
                    model=self.model,
                    messages=self._messages,
                    tools=self._openai_tools,
                    tool_choice="auto",
                    temperature=self.temperature,
                    max_tokens=MAX_TOKENS,
                )
            except rate_limited:
                if wait is None:
                    raise
                self.rate_limit_waits += 1
                print(f"    rate limited; waiting {wait:.0f}s and resending "
                      f"({i + 1}/{len(RATE_LIMIT_WAITS_S)})", flush=True)
                time.sleep(wait)
        raise AssertionError("unreachable")               # loop always returns or raises

    @staticmethod
    def _failed_generation(exc) -> Optional[str]:
        """Pull `failed_generation` out of a tool_use_failed 400, if that is what
        this is. Returns None for any other BadRequestError, which should be
        raised rather than retried."""
        body = getattr(exc, "body", None)
        if isinstance(body, dict):
            err = body.get("error") or {}
            if err.get("code") == "tool_use_failed":
                return str(err.get("failed_generation", ""))[:1500]
        return None

    def step(self) -> Decision:
        response = self._create_with_repair()
        if response is None:
            return Decision(
                stop_reason="tool_use_failed",
                text="The model could not produce a well-formed tool call.")

        choice = response.choices[0]
        message = choice.message
        finish = choice.finish_reason or ""

        if finish == "length":
            return Decision(stop_reason="length", usage=str(response.usage),
                            truncated=True)

        # Record the assistant turn verbatim. tool_calls must be echoed back in
        # the history or the follow-up `role: tool` messages have nothing to
        # attach to and the API rejects the next request.
        assistant: Dict[str, Any] = {"role": "assistant",
                                     "content": message.content or ""}
        raw_calls = list(message.tool_calls or [])
        if raw_calls:
            assistant["tool_calls"] = [{
                "id": c.id,
                "type": "function",
                "function": {"name": c.function.name,
                             "arguments": c.function.arguments},
            } for c in raw_calls]
        self._messages.append(assistant)

        calls: List[ToolCall] = []
        for c in raw_calls:
            calls.append(ToolCall(
                id=c.id,
                name=c.function.name,
                input=self._coerce_arguments(c.function.name,
                                             c.function.arguments)))

        return Decision(text=(message.content or "").strip(),
                        tool_calls=calls,
                        stop_reason=finish,
                        usage=f"{response.usage} repairs={self.repairs}")

    # ---------------------------------------------------------------- validation

    def _coerce_arguments(self, name: str, raw: Optional[str]) -> Dict[str, Any]:
        """Parse and sanity-check tool arguments that arrived as a JSON string.

        Anthropic enforces the schema server-side with `strict`. Here nothing
        does, and open-weight models really do emit fenced JSON, empty strings
        for no-argument tools, or a number where a string belongs.

        A problem is reported in a `__error__` key rather than raised. The loop
        turns that into a tool error the model can read and correct on its next
        turn - the same closed-loop treatment a failed grasp gets. Raising would
        end the run over something the model could have fixed itself.
        """
        text = (raw or "").strip()
        if not text:
            return {}
        # Strip a ```json ... ``` fence if the model wrapped its arguments.
        fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.S)
        if fence:
            text = fence.group(1).strip()

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            return {"__error__": f"arguments for {name} were not valid JSON "
                                 f"({exc}). Send them as a JSON object."}
        if not isinstance(parsed, dict):
            return {"__error__": f"arguments for {name} must be a JSON object, "
                                 f"got {type(parsed).__name__}."}

        schema = self._schema_by_name.get(name)
        if schema is None:
            return {"__error__": f"no such tool: {name}"}

        props: Dict[str, Any] = schema.get("properties", {})
        missing = [k for k in schema.get("required", []) if k not in parsed]
        if missing:
            return {"__error__": f"{name} is missing required argument(s): "
                                 f"{', '.join(missing)}."}
        unknown = [k for k in parsed if k not in props]
        if unknown:
            return {"__error__": f"{name} got unexpected argument(s): "
                                 f"{', '.join(unknown)}. Valid ones are: "
                                 f"{', '.join(props) or 'none'}."}

        # Coerce the numeric types models most often get wrong, and reject an
        # out-of-enum value before it reaches the arm.
        for key, value in list(parsed.items()):
            spec = props.get(key, {})
            # A nullable argument is declared `"type": ["integer", "null"]`.
            # Without unwrapping that, `want` is a list, every comparison below
            # is False, and a nullable field silently skips validation - which
            # is how a string "3" would have reached int(object_id) unchecked.
            declared = spec.get("type")
            types = declared if isinstance(declared, list) else [declared]
            nullable = "null" in types
            want = next((t for t in types if t != "null"), None)

            if value is None:
                if not nullable:
                    return {"__error__": f"{name}.{key} may not be null."}
                continue
            try:
                if want == "integer" and not isinstance(value, bool):
                    parsed[key] = int(value)
                elif want == "number" and not isinstance(value, bool):
                    parsed[key] = float(value)
                elif want == "boolean" and isinstance(value, str):
                    parsed[key] = value.strip().lower() in ("true", "yes", "1")
            except (TypeError, ValueError):
                return {"__error__": f"{name}.{key} should be a {want}, got "
                                     f"{value!r}."}
            if "enum" in spec and parsed[key] not in spec["enum"]:
                return {"__error__": f"{name}.{key} must be one of "
                                     f"{spec['enum']}, got {parsed[key]!r}."}
        return parsed


# --------------------------------------------------------------------------
# Self-test:  python3 groq_backend.py       (no API key, no network)
# --------------------------------------------------------------------------

def _self_test() -> int:
    """Exercise the schema conversion and the local validation.

    This is the part that replaces Anthropic's server-side `strict`, so it is
    the part most worth testing - and it can be tested offline, which the
    conversation handling cannot.
    """
    import sys
    sys.path.insert(0, __file__.rsplit("/backends/", 1)[0].rsplit("\\backends\\", 1)[0])
    from tools import TOOL_SCHEMAS

    failures = []

    def check(label, got, want):
        if got != want:
            failures.append(f"{label}: got {got!r}, want {want!r}")

    # 1. Conversion keeps every tool, its description, and its schema, and drops
    #    `strict` (which this API does not honour - leaving it in would imply a
    #    guarantee that is not there).
    converted = to_openai_tools(TOOL_SCHEMAS)
    check("tool count", len(converted), len(TOOL_SCHEMAS))
    for src, dst in zip(TOOL_SCHEMAS, converted):
        check(f"{src['name']} type", dst["type"], "function")
        check(f"{src['name']} name", dst["function"]["name"], src["name"])
        check(f"{src['name']} desc", dst["function"]["description"],
              src["description"])
        check(f"{src['name']} params", dst["function"]["parameters"],
              src["input_schema"])
        if "strict" in dst["function"]:
            failures.append(f"{src['name']}: `strict` leaked into the OpenAI "
                            f"shape, implying enforcement that does not happen")
    print(f"  converted {len(converted)} tools to the OpenAI function shape")

    # 2. Argument coercion. Build a backend without touching __init__ (which
    #    would demand an API key) - only the schema table is needed.
    b = GroqBackend.__new__(GroqBackend)
    b._schema_by_name = {t["name"]: t["input_schema"] for t in TOOL_SCHEMAS}

    cases = [
        # (label,            tool,         raw arguments,          expected)
        ("well formed",      "pick",       '{"object_id": 3}',     {"object_id": 3}),
        ("no-arg tool",      "home",       "",                     {}),
        ("no-arg tool {}",   "home",       "{}",                   {}),
        ("string int",       "pick",       '{"object_id": "3"}',   {"object_id": 3}),
        ("float to number",  "place_at",   '{"x": 1, "y": "0.5"}', {"x": 1.0, "y": 0.5}),
        ("markdown fence",   "pick",       '```json\n{"object_id": 7}\n```',
                                                                   {"object_id": 7}),
        ("valid enum",       "place_in_bin", '{"bin_name": "red_bin"}',
                                                                   {"bin_name": "red_bin"}),
    ]
    for label, tool, raw, want in cases:
        got = b._coerce_arguments(tool, raw)
        check(f"coerce {label}", got, want)

    # 3. Every malformed case must come back as a readable __error__, never an
    #    exception - the model gets to read it and try again.
    bad = [
        ("not json",        "pick",         "object_id=3"),
        ("not an object",   "pick",         "[1, 2, 3]"),
        ("missing required", "pick",        "{}"),
        ("unknown arg",     "pick",         '{"object_id": 1, "colour": "red"}'),
        ("bad enum",        "place_in_bin", '{"bin_name": "green_bin"}'),
        ("uncoercible int", "pick",         '{"object_id": "the red one"}'),
        ("no such tool",    "fly",          '{"how": "fast"}'),
    ]
    for label, tool, raw in bad:
        got = b._coerce_arguments(tool, raw)
        if "__error__" not in got:
            failures.append(f"{label!r} was accepted: {got!r}")
        elif not got["__error__"].strip():
            failures.append(f"{label!r} produced an empty error message")
    print(f"  {len(cases)} valid argument forms accepted, "
          f"{len(bad)} malformed forms rejected with a readable message")
    for label, tool, raw in bad[:3]:
        msg = b._coerce_arguments(tool, raw).get("__error__", "")
        print(f"    {label:<16} -> {msg}")

    print()
    if failures:
        print(f"FAILED ({len(failures)}):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("all groq backend self-tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())

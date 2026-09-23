"""The contract every LLM provider implements, so the planner loop has none.

The plan's swappability constraint says the loop owns nothing but sequencing:
scene knowledge lives in /world_state, robot knowledge in skill_server, schemas
in tools.py, the prompt in prompts.py. This module is what makes that testable
rather than aspirational - there are now two providers behind one interface, and
the loop cannot tell them apart.

WHY THE BACKEND OWNS THE CONVERSATION
-------------------------------------
The obvious split - "the loop keeps `messages`, the backend just posts them" -
does not work, because the two providers disagree about what a message *is*:

  Anthropic   assistant content is a list of typed blocks; tool results go back
              as a `tool_result` block inside a *user* message, keyed by
              tool_use_id, and several results share one message.
  OpenAI/Groq assistant messages carry a separate `tool_calls` array; each result
              goes back as its own message with `role: "tool"`.

A loop that held the history would have to branch on provider to append to it,
which is exactly the coupling this file exists to prevent. So each backend keeps
its own native history and exposes only provider-neutral types: the loop asks for
a Decision and hands back ToolResults, and never sees a message at all.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


#: Separates what HAPPENED in a tool result from the scene as it looked
#: afterwards. The planner joins the two with this; backends use it to drop the
#: scene from results that are no longer current.
#:
#: WHY OLD SCENES ARE REMOVED, and why this is a correctness fix rather than a
#: cost one. Every tool result carries a freshly perceived world state - that is
#: what closes the loop. But they accumulate: after five steps the context holds
#: five tables, four of which describe positions that have since changed, and
#: the model has no structural reason to prefer the newest. The system prompt
#: asks it to ("trust the latest one"), which is a request, not a guarantee.
#: Dropping the stale ones makes it true by construction.
#:
#: It is also what keeps a run inside a rate limit. A live run died at
#: "Requested 9799, limit 8000" on Groq's free tier with four world states in
#: context; each is a few hundred tokens and there is no prompt caching there.
SCENE_SEPARATOR = "\n\n--- scene as it was at that moment ---\n"

#: What replaces a superseded scene. Kept terse - it is paid once per stale
#: result per turn, so a sentence here is a recurring cost, and the model only
#: needs to know that something was removed and where the current version is.
SCENE_ELIDED = "\n\n[scene omitted - see the latest result]"


def strip_scene(content: str) -> str:
    """Drop a superseded world state, keeping what actually happened."""
    head, sep, _ = content.partition(SCENE_SEPARATOR)
    return head + SCENE_ELIDED if sep else content


@dataclass(frozen=True)
class ToolCall:
    """One tool the model wants run. `id` is opaque and provider-specific."""
    id: str
    name: str
    input: Dict[str, Any]


@dataclass(frozen=True)
class ToolResult:
    """What happened when a ToolCall was run."""
    call: ToolCall
    content: str
    is_error: bool


@dataclass
class Decision:
    """One model turn, in provider-neutral form."""
    text: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    stop_reason: str = ""
    usage: str = ""
    #: Set when the provider declined outright. The loop stops rather than
    #: retrying, and reports this verbatim.
    refusal: Optional[str] = None
    #: Set when the response was cut off at the token limit. Any tool call in it
    #: may be half-written, so the loop must not act on it - a truncated call
    #: could carry a partial argument and send the arm somewhere unintended.
    truncated: bool = False


class LlmBackend(abc.ABC):
    """One provider. Owns its own conversation history."""

    #: Short identifier used in logs and in the --backend argument.
    name: str = "base"

    #: Each backend keeps its native history here. Declared in the base only so
    #: nudge() has something shared to append to - nothing else may touch it.
    _messages: List[Dict[str, Any]]

    def __init__(self, model: str, system: str, tools: Sequence[Dict[str, Any]]):
        self.model = model
        self.system = system
        self.tools = list(tools)
        self._messages = []

    @abc.abstractmethod
    def begin(self, user_text: str) -> None:
        """Start a fresh conversation with one user message."""

    @abc.abstractmethod
    def step(self) -> Decision:
        """Call the model once and record its reply in the history."""

    @abc.abstractmethod
    def add_results(self, results: Sequence[ToolResult]) -> None:
        """Feed tool results back, in whatever shape this provider expects.

        All results for one assistant turn arrive together. Providers that want
        them as separate messages split them here; splitting them in the *loop*
        would train the model out of making parallel calls.
        """

    def nudge(self, text: str) -> None:
        """Append a plain user message, outside the tool protocol.

        For one situation only: the model ended a turn with no tool call when
        the task was not over. Both providers take a plain user message the
        same way, so unlike history *shape* this needs no per-provider code.

        WHY THIS EXISTS. gpt-oss-120b completed all three commands of a live run
        correctly and then wrote its `done` call as prose instead of calling it:

            {"success": true, "summary": "Picked up the blue cube (id 8) ..."}

        Right content, right arguments, emitted as text. Every task succeeded
        and every one was recorded as incomplete. That is the same failure
        family as llama-3.3-70b emitting `<function=pick":{...}` - a model
        describing a call rather than making one - and it gets the same
        treatment: say what went wrong and let it correct itself, rather than
        parsing the prose. Parsing it is precisely what the predecessor project
        did, and why it needed an escape hatch for its own output.

        Safe to append here: this is only reached when the previous assistant
        turn contained no tool_use, so no provider is left waiting for a
        tool_result that never comes.
        """
        self._messages.append({"role": "user", "content": text})

    def describe(self) -> str:
        return f"{self.name}:{self.model}"


def _self_test() -> int:
    """python3 base.py - no SDK, no key, no network."""
    failures = []

    outcome = "SUCCESS: picked up object 4. Grasp confirmed, gap 33.6 mm."
    # A realistic table, because the elision notice has a fixed cost and a
    # toy-sized scene makes the substitution look like a loss. Twelve objects is
    # what the live cell actually reports.
    scene = ("World state:\n"
             "  id  class     colour   x       y       width   where  can pick\n"
             + "".join(
                 f"  {i:<3} cube      yellow   +0.242  +0.424    40mm   "
                 f"on table, visible{'':<31}yes\n" for i in range(1, 13)))
    full = outcome + SCENE_SEPARATOR + scene

    stripped = strip_scene(full)
    saved = len(full) - len(stripped)
    print(f"  a 12-object result: {len(full)} chars -> {len(stripped)} "
          f"({saved} saved, ~{saved // 4} tokens per stale result per turn)")
    if outcome not in stripped:
        failures.append("stripping the scene lost the outcome, which is the "
                        "part that must survive")
    if "can pick" in stripped:
        failures.append("the superseded scene is still there")
    if saved <= 0:
        failures.append("stripping did not shorten anything")

    # Idempotent: add_results runs over the whole history every turn, so an
    # already-stripped result must not accumulate a second elision notice.
    twice = strip_scene(stripped)
    if twice != stripped:
        failures.append(f"strip_scene is not idempotent:\n{twice}")

    # A result with no scene at all - recall, or an argument error - is left
    # exactly as it is.
    plain = "recall: object 4 is in red_bin"
    if strip_scene(plain) != plain:
        failures.append("a result carrying no scene was modified anyway")

    print()
    if failures:
        print(f"FAILED ({len(failures)}):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("all base self-tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())

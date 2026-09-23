"""Claude, via the Anthropic SDK. The default backend.

Kept as the reference implementation because it is the one with the features the
closed loop leans on: strict tool schemas, prompt caching over a frozen prefix,
and an explicit refusal stop reason. See groq_backend.py for what has to be
given up when those are absent.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Sequence

from .base import Decision, LlmBackend, ToolCall, ToolResult, strip_scene

#: Opus 5. Thinking is on by default on this model, so max_tokens has to cover
#: thinking and the reply together - a value sized for just the tool call would
#: truncate mid-thought.
DEFAULT_MODEL = "claude-opus-5"
MAX_TOKENS = 16000


class AnthropicBackend(LlmBackend):

    name = "anthropic"

    def __init__(self, model: str, system: str, tools: Sequence[Dict[str, Any]],
                 effort: str = "high"):
        super().__init__(model or DEFAULT_MODEL, system, tools)
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. export it, or run with "
                "--backend groq.")
        import anthropic                                     # noqa: PLC0415
        self._client = anthropic.Anthropic()
        self.effort = effort
        self._messages: List[Dict[str, Any]] = []

    def begin(self, user_text: str) -> None:
        self._messages = [{"role": "user", "content": user_text}]

    def step(self) -> Decision:
        response = self._client.beta.messages.create(
            model=self.model,
            max_tokens=MAX_TOKENS,
            # cache_control on the last (only) system block caches the tool
            # definitions with it - they render first. Both are frozen, so every
            # turn after the first reads them instead of re-billing them.
            system=[{
                "type": "text",
                "text": self.system,
                "cache_control": {"type": "ephemeral"},
            }],
            tools=self.tools,
            output_config={"effort": self.effort},
            # Opus 5's safety classifiers can decline a request outright. Nothing
            # here should trip them, but a declined request otherwise just stops;
            # this re-runs it on the recommended fallback instead.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=self._messages,
        )

        if response.stop_reason == "refusal":
            detail = getattr(response, "stop_details", None)
            return Decision(
                stop_reason="refusal",
                usage=str(response.usage),
                refusal=(getattr(detail, "explanation", None)
                         or "no explanation given"))

        if response.stop_reason == "max_tokens":
            return Decision(stop_reason="max_tokens", usage=str(response.usage),
                            truncated=True)

        self._messages.append({"role": "assistant", "content": response.content})

        text = "\n".join(b.text.strip() for b in response.content
                         if b.type == "text" and b.text.strip())
        calls = [ToolCall(id=b.id, name=b.name, input=dict(b.input))
                 for b in response.content if b.type == "tool_use"]
        return Decision(text=text, tool_calls=calls,
                        stop_reason=str(response.stop_reason),
                        usage=str(response.usage))

    def add_results(self, results: Sequence[ToolResult]) -> None:
        # Every result now in the history describes a scene that this one is
        # about to supersede, so drop those scenes. See base.SCENE_SEPARATOR for
        # why that is a correctness fix and not only a cost one.
        for message in self._messages:
            if message.get("role") != "user" or not isinstance(
                    message.get("content"), list):
                continue
            for block in message["content"]:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    block["content"] = strip_scene(block["content"])

        # All results for one assistant turn go back in a single user message.
        # Splitting them across messages trains the model out of parallel calls.
        self._messages.append({
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": r.call.id,
                "content": r.content,
                "is_error": r.is_error,
            } for r in results],
        })

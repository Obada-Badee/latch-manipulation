"""The same open models, hosted by DeepInfra instead of Groq.

    export DEEPINFRA_API_KEY=...
    ros2 run llm_task_planner llm_planner --ros-args -p backend:=deepinfra

Why a second host for the *same* model rather than a different model: the free
Groq tier caps at 8,000 tokens per minute and 200,000 per day, and a single
benchmark trial averages 11,090 tokens - so one average trial does not fit
inside one minute, and a day's allowance is about eighteen trials. The paid Groq
tier would lift both, but is not currently open to sign up for. DeepInfra serves
`openai/gpt-oss-120b` - byte-identical model id, and the model this project
measured at 4/4 on the target command where llama-3.3-70b scored 0/3 - at
$0.037 per million input and $0.17 per million output tokens. The ~133 trials
still outstanding come to roughly seven cents.

This subclasses `GroqBackend` rather than copying it. Both hosts expose the same
OpenAI `chat.completions` surface, so the tool schema, the message history, the
scene elision, the tool-repair loop and the usage parsing are shared code. Only
three things differ, and they are the three hooks overridden below.

**Recording the host matters for the paper.** `describe()` returns
`deepinfra:openai/gpt-oss-120b`, and that string goes into every run-log record,
so a result can say which serving stack produced it. "gpt-oss-120b" alone is not
a reproducible specification: the same weights served at a different
quantisation are not guaranteed to make the same tool calls.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Sequence

try:
    from llm_task_planner.backends.groq_backend import (DEFAULT_TEMPERATURE,
                                                        GroqBackend)
except ImportError:                                  # direct run, no ROS paths
    import os as _os
    import sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    from groq_backend import DEFAULT_TEMPERATURE, GroqBackend   # type: ignore

#: DeepInfra's OpenAI-compatible endpoint. The `/v1/openai` suffix is required -
#: the bare host serves DeepInfra's own (different) API shape.
BASE_URL = "https://api.deepinfra.com/v1/openai"


class DeepInfraBackend(GroqBackend):

    name = "deepinfra"
    _env_key = "DEEPINFRA_API_KEY"

    def _make_client(self):
        from openai import OpenAI                             # noqa: PLC0415
        return OpenAI(base_url=BASE_URL,
                      api_key=os.environ[self._env_key])

    @staticmethod
    def _bad_request_type():
        from openai import BadRequestError                    # noqa: PLC0415
        return BadRequestError

    @staticmethod
    def _rate_limit_type():
        from openai import RateLimitError                     # noqa: PLC0415
        return RateLimitError

    @staticmethod
    def _failed_generation(exc) -> Optional[str]:
        """Extract the model's malformed output from a rejected request.

        Groq parses tool calls server-side and hands back the offending text in
        a `tool_use_failed` error with a `failed_generation` field, which the
        repair loop feeds to the model so it can fix its own syntax.

        DeepInfra has no such field. Returning None here is not a gap - it is
        the honest answer, and the base class treats None as "not a repairable
        tool-call error" and re-raises. A repair prompt built from an error
        body that does not contain the model's output would be asking it to fix
        something it was never shown.

        The practical consequence is that `self.repairs` stays 0 on this backend,
        so it must not be compared against Groq's repair count as if it were the
        same measurement. It is not: one host counts malformed tool calls, the
        other does not detect them.
        """
        return None

    def __init__(self, model: str, system: str, tools: Sequence[Dict[str, Any]],
                 temperature: float = DEFAULT_TEMPERATURE):
        super().__init__(model=model, system=system, tools=tools,
                         temperature=temperature)


def _self_test() -> int:
    """python3 deepinfra_backend.py - no SDK, no key, no network."""
    failures = []

    def check(label, got, want):
        if got != want:
            failures.append(f"{label}: got {got!r}, want {want!r}")

    check("name", DeepInfraBackend.name, "deepinfra")
    check("env key", DeepInfraBackend._env_key, "DEEPINFRA_API_KEY")
    check("base url ends /v1/openai", BASE_URL.endswith("/v1/openai"), True)

    # The repair hook must return None rather than raising, whatever it is
    # handed - it is called from an except block, and an exception there would
    # replace a recoverable API error with a confusing traceback.
    class _Fake(Exception):
        body = {"error": {"code": "tool_use_failed", "failed_generation": "x"}}

    check("failed_generation is always None",
          DeepInfraBackend._failed_generation(_Fake()), None)

    # Missing key must fail loudly and name the variable, rather than
    # constructing a client that 401s on the first real request halfway through
    # a benchmark trial.
    saved = os.environ.pop("DEEPINFRA_API_KEY", None)
    try:
        DeepInfraBackend(model="", system="s", tools=[])
        failures.append("missing key: constructed anyway")
    except RuntimeError as exc:
        if "DEEPINFRA_API_KEY" not in str(exc):
            failures.append(f"missing key: unhelpful message {exc!r}")
    except Exception as exc:                                   # noqa: BLE001
        failures.append(f"missing key: wrong exception {type(exc).__name__}")
    finally:
        if saved is not None:
            os.environ["DEEPINFRA_API_KEY"] = saved

    if failures:
        for f in failures:
            print(f"FAIL  {f}")
        return 1
    print("all deepinfra backend self-tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())

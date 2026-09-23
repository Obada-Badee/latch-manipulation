#!/usr/bin/env python3
"""Does this provider answer, and can it call a tool? No robot involved.

    python3 tools/backend_smoke.py --backend deepinfra

Deliberately does not need the simulator. When a benchmark trial fails there are
two candidate explanations - the provider and the cell - and they look identical
from the outside: no tool call arrives, nothing moves, the trial scores zero.
Splitting them costs one cheap request here and saves a session of chasing the
wrong one.

Checks, in order:

  1. the key is accepted and the model responds at all
  2. it emits a *tool call*, through the tool interface, not prose about one
  3. it grounds a spatial referring expression - "the leftmost yellow cube" -
     against a fixed scene whose correct answer is known

Test 3 is the one that matters. `llama-3.3-70b` passed 1 and 2 and still scored
0/3 on the real command, so "the API works" is not evidence the provider is
usable for this task.

Costs roughly 700 tokens - about 0.005 cents on gpt-oss-120b via DeepInfra.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional, Sequence

HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(_ROOT, "src", "llm_task_planner"))

#: A scene with one unambiguous answer. "Leftmost" is +y in link0 (stated in the
#: real system prompt and restated below), so among the yellow cubes - ids 2 and
#: 5 - the leftmost is id 5 at y = +0.62. Id 7 is a decoy: further left still,
#: but red. A model that ignores colour picks 7; one that ignores position picks
#: 2; one that ignores both picks 4.
SCENE = """\
Objects on the table (link0 frame, metres). Larger y is further LEFT.

  id 2   yellow cube      x +0.30  y +0.21
  id 4   blue cube        x +0.45  y +0.11
  id 5   yellow cube      x +0.27  y +0.62
  id 7   red cube         x +0.19  y +0.71
  id 9   grey bolt        x +0.38  y +0.35
"""

EXPECTED_ID = 5
SYSTEM = (
    "You control a Franka Panda arm on a tabletop. Positions are in the robot "
    "base frame (link0): larger y is further LEFT, larger x is further FORWARD. "
    "Call exactly one tool to carry out the user's command. Do not describe what "
    "you would do - call the tool."
)
COMMAND = "Pick up the leftmost yellow cube."


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", default="deepinfra")
    ap.add_argument("--model", default="", help="'' = the backend's default")
    args = ap.parse_args(argv)

    from llm_task_planner import tools as tool_defs
    from llm_task_planner.backends import make_backend

    try:
        backend = make_backend(args.backend, args.model, SYSTEM,
                               tool_defs.TOOL_SCHEMAS)
    except RuntimeError as exc:
        print(f"FAILED to construct backend: {exc}", file=sys.stderr)
        return 1

    print(f"backend: {backend.describe()}\n")

    backend.begin(f"{SCENE}\nCommand: {COMMAND}")
    try:
        decision = backend.step()
    except Exception as exc:                                       # noqa: BLE001
        print(f"FAILED: the request raised {type(exc).__name__}: {exc}",
              file=sys.stderr)
        print("\nIf this is a 401, the key is wrong or truncated. If it is a "
              "404, the model id is not served by this provider.",
              file=sys.stderr)
        return 1

    print(f"stop_reason : {decision.stop_reason}")
    print(f"usage       : {decision.usage}")
    if decision.text:
        print(f"text        : {decision.text[:200]}")

    calls = list(decision.tool_calls or [])
    print(f"tool calls  : {len(calls)}")
    for c in calls:
        print(f"   {c.name}({c.input})")

    problems = []
    if not calls:
        problems.append(
            "no tool call - the model answered in prose. This is the failure "
            "mode llama-3.3-70b showed; the planner's nudge path handles it, "
            "but a provider that does it routinely will be slow and unreliable.")
    else:
        first = calls[0]
        if first.name != "pick":
            problems.append(f"called {first.name!r}, expected 'pick'")
        else:
            got = first.input.get("object_id")
            if got != EXPECTED_ID:
                problems.append(
                    f"grounded 'leftmost yellow cube' to id {got}, expected "
                    f"{EXPECTED_ID}. Ids 2 and 5 are the yellow cubes; 5 has the "
                    f"larger y. Picking 7 means colour was ignored, 2 means "
                    f"position was.")

    print()
    if problems:
        for p in problems:
            print(f"PROBLEM: {p}")
        return 1
    print(f"PASS - answered, called pick, and grounded to id {EXPECTED_ID}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Tool schemas exposed to the planner, and the name -> callable dispatch table.

Deliberately separate from llm_planner.py. The plan's swappability constraint is
that the loop owns nothing but sequencing: schemas live here, the system prompt
lives in prompts.py, and all state lives in /world_state and the skill results.
Replacing the loop (with LangGraph, or the SDK's beta tool_runner) then means
rewriting one function and importing these unchanged.

Every tool declares `strict: True`. That is the single biggest departure from the
Webots version this project replaces, where the model emitted text lines that
were split on whitespace - which is exactly why that system needed an
`error "Command not feasible"` escape hatch for its own malformed output. With a
strict JSON schema, malformed tool input stops being a failure mode at all.

`strict` requires `additionalProperties: false` and a `required` list. Rather than
fight that with a union-typed `place(target | x,y)`, place is split into two
tools - `place_in_bin` and `place_at` - which also gives the model clearer,
narrower choices.
"""

from __future__ import annotations

from typing import Any, Dict, List

#: Tool definitions sent with every request. Order is stable and the content is
#: frozen: these render before the system prompt, so any change here invalidates
#: the whole prompt cache (see prompts.CACHE_NOTE).
TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "name": "pick",
        "description": (
            "Pick up one object and lift it. Identify the object by the `id` field "
            "from the world state; those ids are stable for as long as the robot "
            "remembers the object, not just while it is visible. Do not call this "
            "on an object whose `can pick` column says NO - that verdict comes "
            "from actually solving the arm's kinematics for this grasp, so it is "
            "a fact rather than an estimate. Fails if the grasp misses - the "
            "error message includes the measured finger gap so you can tell a "
            "miss from a planning failure. You must be holding nothing."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "object_id": {
                    "type": "integer",
                    "description": "The `id` of the object to pick up, from the world state.",
                },
            },
            "required": ["object_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "place_in_bin",
        "description": (
            "Release the object currently held into a named bin. Call this only "
            "after a successful pick."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "bin_name": {
                    "type": "string",
                    "enum": ["red_bin", "blue_bin"],
                    "description": "Which bin to drop the held object into.",
                },
            },
            "required": ["bin_name"],
            "additionalProperties": False,
        },
    },
    {
        "name": "place_at",
        "description": (
            "Release the object currently held at a free position on the table, "
            "given in robot coordinates. Use this for relative placements such as "
            "'next to the red cube': read that cube's x/y from the world state and "
            "offset by about 0.08 m so the objects do not collide."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "x": {
                    "type": "number",
                    "description": "Target x in metres, panda_link0 frame. Away from the robot is +.",
                },
                "y": {
                    "type": "number",
                    "description": "Target y in metres, panda_link0 frame. The robot's LEFT is +.",
                },
            },
            "required": ["x", "y"],
            "additionalProperties": False,
        },
    },
    {
        "name": "home",
        "description": (
            "Return the arm to its ready pose. Use this to clear the arm out of the "
            "camera's view before re-reading the world state - the arm is mounted "
            "under the camera and occludes the table - and to recover after a "
            "failed grasp."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    },
    {
        "name": "recall",
        "description": (
            "Look up what the robot remembers, without moving anything. Give an "
            "object id for that object's full history - where it was first seen, "
            "every time it was picked up, and where it was put down. Give no "
            "argument for the log of earlier commands in this session. Use this "
            "when a command refers to something that is not on the table now "
            "('the cube you put away', 'put it back where it was'), or when you "
            "are asked what you have done."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "object_id": {
                    "type": ["integer", "null"],
                    "description": (
                        "Which object to recall. Null for the command log "
                        "instead of a single object."
                    ),
                },
            },
            "required": ["object_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "done",
        "description": (
            "Call this when the user's command is fully carried out, or when it "
            "cannot be carried out and you have stopped trying. This ends the run - "
            "say plainly in the summary what you did and, if anything was left "
            "undone, what and why."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": (
                        "What was accomplished, in one or two sentences. State "
                        "explicitly anything that was not completed."
                    ),
                },
                "success": {
                    "type": "boolean",
                    "description": "True only if the whole command was carried out.",
                },
            },
            "required": ["summary", "success"],
            "additionalProperties": False,
        },
    },
]

#: Tools that end the run when called.
TERMINAL_TOOLS = {"done"}


def tool_names() -> List[str]:
    return [t["name"] for t in TOOL_SCHEMAS]

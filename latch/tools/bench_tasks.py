#!/usr/bin/env python3
"""The benchmark task suite, and what counts as success for each task.

    python3 bench_tasks.py            # self-test: checkers on synthetic states

Two things here are load-bearing and worth stating plainly.

**Success is judged from Gazebo, never from the planner.** Every task carries a
`check` that reads the simulator's ground-truth object poses and decides whether
the world ends in the required state. The planner's own `done(success=...)` is
recorded separately and never consulted. This is not pedantry: the predecessor
project reported 96-99% success from a system that could not observe its own
outcome, and this project has already seen the model report success on tasks it
had genuinely completed but log them as incomplete - the self-report and the
world are independent variables and both are interesting. Their *disagreement*
rate is itself a result (a planner that claims success when the table says
otherwise is a specific, reportable failure).

**Success criteria are written down before the runs.** A criterion invented after
seeing the outcome is not a criterion. Each one below is a predicate over
positions with explicit tolerances, so a reviewer can check it, and so no
judgement call happens at scoring time.

Tiers are the supervisor's suggestion and a good one - reporting one aggregate
success rate over a mixed suite hides which capability is failing:

  simple   one object, named directly, one skill
  medium   a spatial or comparative referring expression, or two skills
  complex  several objects, cross-command memory, or an under-specified phrase
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(_ROOT, "src", "llm_task_planner"))

from llm_task_planner import scene                              # noqa: E402

#: link0 z of the table surface, and the height above it that means "held".
TABLE_TOP_Z = scene.TABLE_TOP_Z
LIFTED_Z = 0.10

#: An object counts as in a bin when its centre is inside the usable drop area
#: and below the rim. Both halves are needed: an object dangling above the bin
#: satisfies the first and is not in it, and one beside the bin on the table
#: satisfies the second.
BIN_MARGIN = 0.005

XYZ = Tuple[float, float, float]
Objects = Dict[str, XYZ]                # model name -> link0 position


# --------------------------------------------------------------------------
# Reading the world
# --------------------------------------------------------------------------

def classify(name: str) -> Optional[Tuple[str, str]]:
    """Gazebo model name -> (class, colour). None if it is not a table object.

    Handles both the authored names (`cube_yellow_1`, `bolt2`) and the ones
    scene_manager spawns (`dyn_cube_red_43175`). Anything unrecognised returns
    None and is ignored rather than guessed - a mis-classified object would
    silently corrupt a checker.
    """
    stem = name[4:] if name.startswith("dyn_") else name
    parts = stem.split("_")
    base = parts[0].rstrip("0123456789")
    if base not in ("cube", "cylinder", "bolt"):
        return None
    if base == "bolt":
        return "bolt", "grey"           # the mesh is bare metal, no colour
    colour = parts[1] if len(parts) > 1 else "unknown"
    return base, colour


#: An object is "in play" only inside this link0 slab. The authored world keeps
#: a template model named `bolt` parked at z = -1.02, far under the table, and
#: `classify` cannot tell it from a real bolt by name. Left unfiltered it would
#: sit permanently outside every bin and make "clear the bolts off the table"
#: unpassable - a checker failing on a phantom, which is the same class of bug
#: as scene_manager reading robot links as table objects. Filter by geometry,
#: never by name.
IN_PLAY_Z = (-0.10, 0.60)
IN_PLAY_X = (-0.10, 0.80)
IN_PLAY_Y = (-0.50, 0.90)


def in_play(objects: Objects) -> Objects:
    """Drop parked templates and anything else outside the working volume."""
    keep = {}
    for name, (x, y, z) in objects.items():
        if (IN_PLAY_X[0] <= x <= IN_PLAY_X[1]
                and IN_PLAY_Y[0] <= y <= IN_PLAY_Y[1]
                and IN_PLAY_Z[0] <= z <= IN_PLAY_Z[1]):
            keep[name] = (x, y, z)
    return keep


def manipulable(objects: Objects) -> Objects:
    """Just the things the robot can pick up.

    `in_play` is a spatial filter, so it also returns `arm_robot`, both bins and
    `visual1` - 13 entries for a nine-object scene. That is harmless where it is
    used for checking (a bin does not move, and `select` matches on name), but
    it makes any COUNT of in_play wrong by four, which is exactly the number a
    scene-completeness check depends on.
    """
    keep = {}
    for name, xyz in objects.items():
        stem = name[4:] if name.startswith("dyn_") else name
        if stem.split("_")[0].rstrip("0123456789") in ("cube", "cylinder", "bolt"):
            keep[name] = xyz
    return keep


def select(objects: Objects, class_name: Optional[str] = None,
           colour: Optional[str] = None) -> List[Tuple[str, XYZ]]:
    """Every object matching a class and/or colour, as (name, position)."""
    out = []
    for name, xyz in objects.items():
        got = classify(name)
        if got is None:
            continue
        cls, col = got
        if class_name and cls != class_name:
            continue
        if colour and col != colour:
            continue
        out.append((name, xyz))
    return sorted(out)


def in_bin(xyz: XYZ, bin_name: str) -> bool:
    r = scene.RECEPTACLES[bin_name]
    x, y, z = xyz
    inside = (abs(x - r.x) <= r.inner_half_width + BIN_MARGIN
              and abs(y - r.y) <= r.inner_half_width + BIN_MARGIN)
    return inside and z < r.rim_z


def in_any_bin(xyz: XYZ) -> Optional[str]:
    for name in scene.RECEPTACLES:
        if in_bin(xyz, name):
            return name
    return None


def lifted(xyz: XYZ) -> bool:
    return xyz[2] > LIFTED_Z


# --------------------------------------------------------------------------
# Checkers
# --------------------------------------------------------------------------
# Each returns (passed, human-readable detail). The detail is written to the
# CSV whether the task passed or not, because "failed" without a reason cannot
# be turned into the failure taxonomy later.

def all_of(class_name: Optional[str], colour: Optional[str],
           bin_name: Optional[str] = None) -> Callable[[Objects], Tuple[bool, str]]:
    """Every matching object ends in `bin_name` (or in any bin if None)."""
    def check(objects: Objects) -> Tuple[bool, str]:
        matches = select(objects, class_name, colour)
        if not matches:
            return False, "no matching object was on the table"
        placed, stray = [], []
        for name, xyz in matches:
            where = in_bin(xyz, bin_name) if bin_name else bool(in_any_bin(xyz))
            (placed if where else stray).append(name)
        target = bin_name or "a bin"
        if stray:
            return False, f"{len(placed)}/{len(matches)} in {target}; not: {', '.join(stray)}"
        return True, f"all {len(placed)} in {target}"
    return check


def one_of(class_name: Optional[str], colour: Optional[str],
           bin_name: Optional[str] = None) -> Callable[[Objects], Tuple[bool, str]]:
    """At least one matching object ends in the bin, and none is left held."""
    def check(objects: Objects) -> Tuple[bool, str]:
        matches = select(objects, class_name, colour)
        if not matches:
            return False, "no matching object was on the table"
        for name, xyz in matches:
            if (in_bin(xyz, bin_name) if bin_name else in_any_bin(xyz)):
                return True, f"{name} in {bin_name or in_any_bin(xyz)}"
        return False, f"none of {len(matches)} reached {bin_name or 'a bin'}"
    return check


def extreme_in_bin(class_name: str, colour: Optional[str], axis: str,
                   want_max: bool, bin_name: str
                   ) -> Callable[[Objects], Tuple[bool, str]]:
    """The extreme object along an axis - and *only* it - ends in the bin.

    This is the referring-expression check, so it has to be strict in both
    directions. Putting the right object in the bin is necessary; leaving the
    others alone is what distinguishes "resolved the expression" from "moved
    something that matched the noun". A planner that bins every yellow cube
    scores 0 here, correctly.

    The extreme is computed from the pose the objects had at the START of the
    trial, which the runner passes in as `initial`; using final poses would ask
    which object is leftmost *after* one of them moved into a bin.
    """
    def check(objects: Objects, initial: Optional[Objects] = None
              ) -> Tuple[bool, str]:
        source = initial if initial is not None else objects
        matches = select(source, class_name, colour)
        if len(matches) < 2:
            return False, f"needs >=2 candidates to be a real choice, saw {len(matches)}"
        idx = 0 if axis == "x" else 1
        chosen = (max if want_max else min)(matches, key=lambda m: m[1][idx])
        target_name = chosen[0]
        if target_name not in objects:
            return False, f"{target_name} vanished from the world"
        if not in_bin(objects[target_name], bin_name):
            return False, f"{target_name} was the right object but is not in {bin_name}"
        moved = [n for n, xyz in matches
                 if n != target_name and n in objects and in_any_bin(objects[n])]
        if moved:
            return False, f"{target_name} placed, but also moved: {', '.join(moved)}"
        return True, f"{target_name} in {bin_name}, others untouched"
    return check


def same_bin(colour_a: str, colour_b: str) -> Callable[[Objects], Tuple[bool, str]]:
    """Two named objects end in the same bin - whichever bin that is."""
    def check(objects: Objects) -> Tuple[bool, str]:
        a = select(objects, "cube", colour_a)
        b = select(objects, "cube", colour_b)
        if not a or not b:
            return False, f"need one {colour_a} and one {colour_b} cube"
        bin_a, bin_b = in_any_bin(a[0][1]), in_any_bin(b[0][1])
        if bin_a is None or bin_b is None:
            return False, f"{colour_a} in {bin_a}, {colour_b} in {bin_b}"
        if bin_a != bin_b:
            return False, f"different bins: {colour_a} in {bin_a}, {colour_b} in {bin_b}"
        return True, f"both in {bin_a}"
    return check


def is_lifted(class_name: str, colour: Optional[str]
              ) -> Callable[[Objects], Tuple[bool, str]]:
    def check(objects: Objects) -> Tuple[bool, str]:
        matches = select(objects, class_name, colour)
        if not matches:
            return False, "no matching object"
        for name, xyz in matches:
            if lifted(xyz):
                return True, f"{name} held at z={xyz[2]:.3f}"
        best = max(matches, key=lambda m: m[1][2])
        return False, f"nothing lifted; highest {best[0]} at z={best[1][2]:.3f}"
    return check


def nothing_moved(tol: float = 0.02) -> Callable[[Objects], Tuple[bool, str]]:
    """No object moved. For commands the robot should refuse or no-op."""
    def check(objects: Objects, initial: Optional[Objects] = None
              ) -> Tuple[bool, str]:
        if initial is None:
            return False, "no initial state recorded"
        for name, xyz in objects.items():
            if classify(name) is None or name not in initial:
                continue
            was = initial[name]
            d = sum((a - b) ** 2 for a, b in zip(xyz, was)) ** 0.5
            if d > tol:
                return False, f"{name} moved {d * 1000:.0f} mm"
        return True, "table unchanged, as required"
    return check


# --------------------------------------------------------------------------
# The suite
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Task:
    id: str
    tier: str
    command: str
    check: Callable[..., Tuple[bool, str]]
    #: (class, colour, count) that must be on the table for the task to be
    #: answerable. The runner guarantees these when it resets the scene; a task
    #: run against a scene that cannot satisfy it is not a failure, it is a
    #: void trial, and conflating the two would understate success.
    requires: Tuple[Tuple[str, Optional[str], int], ...] = ()
    note: str = ""


SUITE: List[Task] = [
    # ---------------------------------------------------------------- simple
    Task("s1", "simple", "pick up the red cube",
         is_lifted("cube", "red"),
         (("cube", "red", 1),)),
    Task("s2", "simple", "put the red cube in the red bin",
         one_of("cube", "red", "red_bin"),
         (("cube", "red", 1),)),
    Task("s3", "simple", "put the blue cube in the blue bin",
         one_of("cube", "blue", "blue_bin"),
         (("cube", "blue", 1),)),
    Task("s4", "simple", "put a bolt in the red bin",
         one_of("bolt", None, "red_bin"),
         (("bolt", None, 1),)),
    Task("s5", "simple", "put the green cube in the blue bin",
         one_of("cube", "green", "blue_bin"),
         (("cube", "green", 1),)),

    # ---------------------------------------------------------------- medium
    Task("m1", "medium", "put the leftmost yellow cube in the red bin",
         extreme_in_bin("cube", "yellow", "y", True, "red_bin"),
         (("cube", "yellow", 2),),
         "referring expression: +y is the robot's left"),
    Task("m2", "medium", "put the rightmost yellow cube in the blue bin",
         extreme_in_bin("cube", "yellow", "y", False, "blue_bin"),
         (("cube", "yellow", 2),)),
    Task("m3", "medium", "put the cube nearest to the robot in the red bin",
         extreme_in_bin("cube", None, "x", False, "red_bin"),
         (("cube", None, 3),)),
    Task("m4", "medium", "put both yellow cubes in the blue bin",
         all_of("cube", "yellow", "blue_bin"),
         (("cube", "yellow", 2),)),
    Task("m5", "medium",
         "put the red cube in the blue bin, then put the blue cube in the red bin",
         lambda o: (lambda a, b: (a[0] and b[0], f"{a[1]}; {b[1]}"))(
             one_of("cube", "red", "blue_bin")(o),
             one_of("cube", "blue", "red_bin")(o)),
         (("cube", "red", 1), ("cube", "blue", 1)),
         "two skills, and the bins are deliberately crossed"),

    # --------------------------------------------------------------- complex
    Task("c1", "complex", "put all the cubes in the blue bin",
         all_of("cube", None, "blue_bin"),
         (("cube", None, 3),)),
    Task("c2", "complex", "clear the bolts off the table",
         all_of("bolt", None, None),
         (("bolt", None, 2),)),
    Task("c3", "complex", "put the red one away",
         one_of(None, "red", None),
         (("cube", "red", 1),),
         "under-specified: no class, no destination. Robustness probe."),
    Task("c4", "complex", "tidy up the table",
         all_of("cube", None, None),
         (("cube", None, 3),),
         "maximally vague. Measures whether it acts sensibly, not exactly."),
    Task("c5", "complex", "put the purple cube in the red bin",
         nothing_moved(),
         (("cube", None, 2),),
         "IMPOSSIBLE: no purple cube exists. Success = report it and move "
         "nothing. Tests whether it invents a referent rather than refusing."),
]

#: Multi-command episodes: memory across turns is the thing under test, so the
#: check runs only after the last command. Kept apart from SUITE because the
#: runner has to drive them differently.
EPISODES: List[Tuple[str, str, List[str], Callable[..., Tuple[bool, str]],
                     Tuple[Tuple[str, Optional[str], int], ...]]] = [
    ("e1", "complex",
     ["put the red cube in the red bin",
      "now put the blue cube in the same bin as the red cube"],
     same_bin("red", "blue"),
     (("cube", "red", 1), ("cube", "blue", 1))),
    ("e2", "complex",
     ["put the red cube in the blue bin",
      "where is the red cube now?",
      "put the green cube in the same bin"],
     same_bin("red", "green"),
     (("cube", "red", 1), ("cube", "green", 1))),
]


def _self_test() -> int:
    """Check the checkers against hand-built worlds, including the near misses.

    A checker that returns True for everything passes any run, so these test the
    NEGATIVE cases hardest: right object wrong place, wrong object right place,
    and the dangling-above-the-bin case that a bounds-only test would accept.
    """
    fails = 0

    def expect(label: str, got: Tuple[bool, str], want: bool):
        nonlocal fails
        if got[0] != want:
            print(f"  FAIL {label}: wanted {want}, got {got[0]} ({got[1]})")
            fails += 1
        else:
            print(f"  ok   {label}: {got[1]}")

    red = scene.RECEPTACLES["red_bin"]
    blue = scene.RECEPTACLES["blue_bin"]
    inside_red = (red.x, red.y, 0.02)
    inside_blue = (blue.x, blue.y, 0.02)
    on_table = (0.45, 0.30, 0.015)
    # Directly over the bin but still in the gripper. Bounds say yes, height
    # says no - this is the case that makes the z half of in_bin necessary.
    above_red = (red.x, red.y, 0.25)

    print("in_bin")
    expect("resting in red bin", (in_bin(inside_red, "red_bin"), "-"), True)
    expect("on the table", (in_bin(on_table, "red_bin"), "-"), False)
    expect("dangling above the rim", (in_bin(above_red, "red_bin"), "-"), False)
    expect("in the other bin", (in_bin(inside_blue, "red_bin"), "-"), False)

    print("\nclassify")
    for name, want in (("cube_yellow_1", ("cube", "yellow")),
                       ("dyn_cube_red_43175", ("cube", "red")),
                       ("bolt2", ("bolt", "grey")),
                       ("red_bin", None),
                       ("panda_link0", None)):
        got = classify(name)
        ok = got == want
        fails += 0 if ok else 1
        print(f"  {'ok  ' if ok else 'FAIL'} {name:<22} -> {got}")

    print("\nall_of / one_of")
    world = {"cube_yellow_1": inside_blue, "cube_yellow_2": inside_blue}
    expect("both yellow in blue", all_of("cube", "yellow", "blue_bin")(world), True)
    world2 = {"cube_yellow_1": inside_blue, "cube_yellow_2": on_table}
    expect("one left out", all_of("cube", "yellow", "blue_bin")(world2), False)
    expect("one_of accepts partial", one_of("cube", "yellow", "blue_bin")(world2), True)

    print("\nextreme_in_bin (the referring-expression check)")
    # left = larger y. left cube at y=0.40, right at y=0.20.
    left, right = "cube_yellow_1", "cube_yellow_2"
    initial = {left: (0.45, 0.40, 0.015), right: (0.45, 0.20, 0.015)}
    chk = extreme_in_bin("cube", "yellow", "y", True, "red_bin")
    expect("correct object binned",
           chk({left: inside_red, right: initial[right]}, initial), True)
    expect("wrong object binned",
           chk({left: initial[left], right: inside_red}, initial), False)
    expect("both binned - not a resolution",
           chk({left: inside_red, right: inside_red}, initial), False)
    expect("right object, still on table",
           chk(dict(initial), initial), False)

    print("\nnothing_moved")
    expect("untouched", nothing_moved()(dict(initial), initial), True)
    expect("something moved", nothing_moved()({left: inside_red,
                                               right: initial[right]}, initial), False)

    print("\nin_play (the parked-template filter)")
    mixed = {"bolt": (-0.05, 0.0, -1.02),          # parked template
             "bolt1": (0.386, 0.474, 0.014),       # real
             "cube_red_1": inside_red}
    kept = in_play(mixed)
    expect("parked template dropped", ("bolt" not in kept, str(sorted(kept))), True)
    expect("real bolt kept", ("bolt1" in kept, "-"), True)
    expect("binned cube kept", ("cube_red_1" in kept, "-"), True)
    expect("clear-the-bolts passes once filtered",
           all_of("bolt", None, None)({**kept, "bolt1": inside_red}), True)
    expect("...and fails unfiltered, as the bug would have",
           all_of("bolt", None, None)({**mixed, "bolt1": inside_red}), False)

    print("\nsuite")
    ids = [t.id for t in SUITE]
    if len(ids) != len(set(ids)):
        print("  FAIL duplicate task ids")
        fails += 1
    else:
        print(f"  ok   {len(SUITE)} tasks, unique ids, "
              f"{sum(1 for t in SUITE if t.tier == 'simple')} simple / "
              f"{sum(1 for t in SUITE if t.tier == 'medium')} medium / "
              f"{sum(1 for t in SUITE if t.tier == 'complex')} complex, "
              f"+{len(EPISODES)} multi-command episodes")

    print("\nFAILURES:", fails)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(_self_test())

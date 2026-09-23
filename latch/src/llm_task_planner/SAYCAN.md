# Relationship to SayCan

Written because "make it like SayCan" is easy to answer badly. SayCan is four
years old, and half of what made it work has been absorbed into ordinary tool
calling while the other half is still missing from most LLM-robotics demos. This
file records which half is which, and what this system actually implements.

## What SayCan does

*Do As I Can, Not As I Say* (Ahn et al., 2022) chooses the next skill by

```
argmax_k  p(l_k | i)  ×  p(c_k | s, l_k)
          ^^^^^^^^^^     ^^^^^^^^^^^^^^^
          task           affordance
          grounding      grounding
          "would this    "would this
           help?"         work here?"
```

Every skill in a fixed library is scored on both terms and the product decides.
Then the chosen skill is appended to the prompt and the loop repeats.

## The task half: not reimplemented, deliberately

SayCan computed `p(l_k | i)` as the log-likelihood of a skill's English
description ("pick up the sponge") under a frozen PaLM. That was necessary in
2022: there was no way to make a language model commit to one option from a
fixed set except to score all the options.

Reproducing that here would be a **downgrade**. This planner emits typed tool
calls validated against a JSON schema, which beats ranking sentences on every
axis that matters:

| | SayCan scoring | tool calls |
|---|---|---|
| can it name a skill that does not exist? | no | no |
| can it carry arguments? | no — one string per skill | yes, typed |
| does it scale with objects? | no — the skill set is `skills × objects` | yes |
| does it compose? | no | yes |

The "skills × objects" row is the fatal one. SayCan's library enumerates the
object into the skill string, so a scene with 12 objects and 4 verbs needs 48
scored strings, and every new object re-runs the scoring. `pick(object_id)` does
not care how many objects exist.

So: the task half stays as it is. Claiming otherwise would be re-implementing a
workaround for a limitation that no longer applies.

## The affordance half: this is the part worth having

`p(c_k | s, l_k)` is what stops a language model instructing a robot to pick up
something behind a wall. It is a filter applied **before** acting, and this
project did not have one — the planner chose freely and discovered infeasibility
by failing. That works, but it wastes a turn per mistake and it makes every
failure look alike.

[`affordance.py`](llm_task_planner/affordance.py) is that filter.

### Where the number comes from

| | SayCan | here |
|---|---|---|
| source | learned RL value function, trained on real robot rollouts | solved IK, real collision geometry, measured gripper limits |
| kind of answer | a prediction | the precondition itself |
| covers | anything visible in the rollout data, including friction and lighting | anything expressible in geometry |

This is not a shortcut. The question is "will `pick(4)` succeed?", and
`skill_server` owns the exact IK solver and planning scene that `pick(4)` is
about to use — so it can answer by **asking** rather than by estimating. A
learned value function regresses onto outcomes; a solved IK query is the
outcome's actual precondition.

**Where the learned version is genuinely better**, and this matters if the
project ever leaves simulation: it captures failure modes geometry cannot see. A
slippery surface, a deformable object, a lighting condition that ruins the pose
estimate. SayCan's value function learns all of those from rollouts and nothing
here will ever notice them. On a simulated Panda with exactly known geometry the
gap is small. On hardware it would not be.

### The four gates

Run cheapest-first, short-circuiting, so an object that is too wide never costs
an IK call.

| gate | question | kind |
|---|---|---|
| `width` | is the short axis inside the jaw's travel? | mechanical limit |
| `reach` | is `sqrt(x²+y²)` inside the working band? | cheap screen |
| `ik` | does IK solve at the grasp pose, at the yaw this grasp needs? | **solver** |
| `ik_approach` | and at the hover above it? | **solver** |
| `clutter` | do the fingertips clear the neighbours on the way down? | geometry, soft |

The `ik` gate is the one that earns its place. A pose can sit comfortably inside
the radial reach envelope and still have no solution *at the wrist angle this
particular grasp demands* — and nothing cheaper can tell you that.

The clutter gate is geometric rather than a proximity radius: the fingertips sit
at ±half the open jaw gap along `grasp_yaw`, so a neighbour beside the jaw is
irrelevant while the same neighbour at the same distance along the closing axis
is a collision. A centre-distance test scores those two identically, and they
are the common case on a cluttered table.

### Honesty about the score

`score` is **not** a calibrated probability. Nothing here was fitted to observed
success rates, and saying otherwise would be a fabrication.

* `0.0` — a hard gate failed. This is ground truth, not an estimate.
* `(0, 1)` — solvable, but a fingertip passes close to a neighbour. Interpolated
  on clearance. A heuristic.
* `1.0` — solvable with clearance to spare.

The planner is told to treat `0.0` as a veto and to prefer higher scores when it
has a choice. The ordering is the useful part; the values in between mean no
more than "this one is tighter than that one".

## Memory: not from SayCan

SayCan's "memory" is the running transcript of executed skills appended to the
prompt. That is enough for one instruction and nothing more — it does not
survive a new command, and it holds no belief about objects it cannot see.

[`memory.py`](llm_task_planner/memory.py) is a belief store instead, with two
sources trusted differently:

* **observation** refreshes a pose. Absence of observation is *weak* evidence:
  the arm occludes the table constantly.
* **action** ("I placed it in red_bin") is *strong* evidence and outranks
  observation. A cube at the bottom of a bin may never be detected again, and
  without this ordering every successful place would read as the object
  vanishing.

Getting that ordering right is the whole trick.

It also keeps its own stable ids. `world_model`'s ids are stable only while an
object stays continuously visible — correct for a perception component, useless
as a long-term name. The planner shows memory ids and translates to world ids at
the moment it sends a goal, so `pick` on a remembered-but-invisible object fails
with *"it is in red_bin"* rather than *"no object with id 4"*.

## Verified

On the live cell, not in a unit test:

| claim | evidence |
|---|---|
| affordances are computed, not guessed | 9 objects scored in 1.3 s via real IK + planning-scene collision |
| the layer discriminates | an object moved to reach 0.814 m was vetoed `NO (reach)`; the other 11 stayed feasible |
| clutter is directional | two objects 42 mm apart each vetoed the other; isolated objects scored 1.0 |
| the world is dynamic | objects spawned into the running cell, detected, scored, and removed, with ids assigned and retired |
| memory survives a bin | see the run log referenced in the commit; an object placed in a bin stays listed as `in red_bin` |

### The bug this layer found in itself

The first live run vetoed **all nine** objects with "no IK solution". The cause
was a missing `RobotState.update()` before the collision check — link transforms
were stale, so every IK solution appeared to collide. The probe reported that as
a kinematics failure because both branches returned the same bool, which sent me
looking at frames and tip links.

Splitting the probe's return from `bool` to a reason string fixed the diagnosis
permanently, and `_on_check_affordances` now runs a control probe at a known-good
pose whenever the verdict is unanimously negative — because "nothing can be
picked up" and "the scorer is broken" look identical from outside, and only one
of them is worth acting on.

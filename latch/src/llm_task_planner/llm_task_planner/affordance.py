"""Affordance grounding: can the robot actually do this, before it tries.

--------------------------------------------------------------------------
WHAT THIS TAKES FROM SayCan, AND WHAT IT DELIBERATELY DOES NOT
--------------------------------------------------------------------------
SayCan (Ahn et al., 2022) picks the next skill by

    argmax_k  p(l_k | i) * p(c_k | s, l_k)
              ^^^^^^^^^^   ^^^^^^^^^^^^^^^
              task          affordance
              grounding     grounding
              "does this    "would this
               help?"        work here?"

Two halves, and they have aged very differently.

The TASK half is obsolete. SayCan scored the log-likelihood of each skill's
English description ("pick up the sponge") under a frozen LLM, because in 2022
that was the only way to get a language model to commit to one option out of a
fixed set. Re-implementing that here would be a downgrade: this planner emits
typed tool calls against a JSON schema, which is strictly better than ranking
sentences - it cannot pick a skill that does not exist, it carries arguments,
and it composes. So the task half stays as it is, and is not reimplemented.

The AFFORDANCE half is the part this project was missing, and it is the part
that made SayCan work. Its point is a filter *before* acting: a plan is only
allowed to contain steps the robot can actually execute from the state it is
in. Without it a language model will cheerfully instruct a robot to pick up
something behind a wall.

This module is that half. The difference is where the number comes from:

    SayCan     a learned RL value function, trained from real robot rollouts.
    here       solved IK, real collision geometry, measured gripper limits.

That is not a shortcut, it is the better instrument for this robot. The
affordance question is "will pick(4) succeed?", and this process owns the very
IK solver and planning scene that pick(4) will use - so it can answer by asking
rather than by estimating. A learned value function is a regression onto
outcomes; a solved IK query is the outcome's actual precondition.

Where the learned version is genuinely better, and this is worth stating
plainly: it captures failure modes geometry cannot see. A slippery surface,
a deformable object, a lighting condition that ruins the grasp pose estimate -
SayCan's value function can learn all of those from rollouts, and nothing here
will ever notice them. On a simulated Panda with exactly known geometry that
gap is small. On hardware it would not be.

--------------------------------------------------------------------------
THE SCORE
--------------------------------------------------------------------------
`score` is in 0..1 and is NOT a calibrated probability - claiming that would be
dishonest, since nothing here was fitted to observed success rates. It is:

    0.0            a hard gate failed. This is ground truth, not an estimate:
                   the jaw does not open that wide, or IK has no solution.
    (0, 1)         everything solvable, but a fingertip passes close to a
                   neighbour. Interpolated on clearance - a heuristic.
    1.0            solvable with clearance to spare.

The planner is told to treat 0.0 as a veto and to prefer higher scores when it
has a choice. That is the useful part of the ordering; the exact values in
between are not claimed to mean more than "this one is tighter than that one".

Kept free of ROS and of MoveIt on purpose: the geometry is a pure function, and
IK enters through a callable. That way the interesting half can be self-tested
offline, which is what _self_test() at the bottom does.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

from llm_task_planner import scene

#: Half the footprint of one fingertip in the plane, metres. The Panda pad is
#: about 22 mm across and 15 mm deep, so ~12 mm is a fair circular bound on the
#: part that can strike a neighbour on the way down.
FINGER_RADIUS = 0.012

#: Clearance beyond contact at which a neighbour stops being a concern. Below
#: this a descent is "tight but possible" and the score degrades toward zero.
#:
#: Not arbitrary: perception's worst-case position error is 8.3 mm (measured
#: for a cylinder in tools/synth_dataset/labeler.py). A margin narrower than
#: that error is not a margin at all - the object may simply be somewhere else.
#: 15 mm keeps roughly double the error before the score starts falling.
SAFE_CLEARANCE = 0.015

#: IK is asked for the grasp pose and the hover above it. Both must solve: a
#: grasp you can reach but cannot approach from above is not a grasp.
#:
#: Signature: (x, y, z, yaw) -> '' if the pose is achievable, else a phrase
#: saying why. NOT a bool, and that is not cosmetic: "the wrist cannot reach
#: that angle" and "it can, but the elbow would be through the table" call for
#: completely different responses from the planner, and a bool cannot tell them
#: apart. The first version of this returned a bool and reported every failure
#: as "no IK solution", which sent me looking at the kinematics when the answer
#: was somewhere else entirely.
IkProbe = Callable[[float, float, float, float], str]


@dataclass(frozen=True)
class ObjectGeom:
    """The part of a DetectedObject that affordance scoring needs.

    A plain dataclass rather than the ROS message so this module can be tested
    without a ROS install, and so the scorer cannot quietly start depending on
    fields perception may stop publishing.
    """
    id: int
    class_name: str
    x: float
    y: float
    grasp_yaw: float
    long_axis: float
    short_axis: float

    @classmethod
    def from_msg(cls, obj) -> "ObjectGeom":
        return cls(id=int(obj.id), class_name=str(obj.class_name),
                   x=float(obj.x), y=float(obj.y),
                   grasp_yaw=float(obj.grasp_yaw),
                   long_axis=float(obj.long_axis),
                   short_axis=float(obj.short_axis))


@dataclass(frozen=True)
class Affordance:
    id: int
    feasible: bool
    score: float
    reason: str
    blocker: str        # '' | width | reach | ik | ik_approach | clutter


def fingertip_positions(obj: ObjectGeom,
                        jaw_gap: Optional[float] = None
                        ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Where the two fingertips sit, in link0, at the moment of descent.

    `grasp_yaw` is the direction the jaw closes ALONG (projection.grasp_yaw:
    the hand's Y axis, aligned with the object's short axis), so the tips are
    offset from the object centre by half the open gap in that direction and
    in the opposite one.

    This is what makes the clutter check geometric rather than a radius guess:
    a bolt lying beside a cube is fine if the fingers close along the gap
    between them, and a collision if they close across it. A centre-distance
    test cannot tell those two apart, and they are the common case on a
    cluttered table.
    """
    gap = scene.OPEN_JAW_GAP if jaw_gap is None else jaw_gap
    half = 0.5 * gap
    dx = half * math.cos(obj.grasp_yaw)
    dy = half * math.sin(obj.grasp_yaw)
    return (obj.x + dx, obj.y + dy), (obj.x - dx, obj.y - dy)


def open_gap_for(obj: ObjectGeom) -> float:
    """How wide the jaw will actually be opened before descending on this.

    Delegates, so the claim in that sentence is true. It used to compute its own
    value, which differed from the skill server's by up to 18 mm.
    """
    return scene.open_gap_for(obj.short_axis)


def clutter_clearance(obj: ObjectGeom, neighbours: Iterable[ObjectGeom]
                      ) -> Tuple[float, Optional[int]]:
    """Smallest gap between a fingertip and any neighbour, in metres.

    Negative means the fingertip would be inside that neighbour's footprint.
    Returns (gap, offending_id); the id is None when nothing is near.

    Each neighbour is bounded by a circle of half its LONG axis - conservative
    on purpose. The alternative, an oriented-rectangle test, is more accurate
    than the 7 mm position estimate feeding it, so it would be precision the
    inputs do not support.
    """
    tips = fingertip_positions(obj, open_gap_for(obj))
    worst, culprit = math.inf, None

    for other in neighbours:
        if other.id == obj.id:
            continue
        bound = 0.5 * other.long_axis
        for tx, ty in tips:
            gap = math.hypot(tx - other.x, ty - other.y) - bound - FINGER_RADIUS
            if gap < worst:
                worst, culprit = gap, other.id

    return (worst if culprit is not None else math.inf), culprit


def score_pick(obj: ObjectGeom,
               neighbours: Sequence[ObjectGeom] = (),
               ik_probe: Optional[IkProbe] = None) -> Affordance:
    """p(success | pick, state) for one object.

    Hard gates run cheapest-first and short-circuit, so an object that is too
    wide never costs an IK call. Each gate names itself in `blocker` so the
    planner gets a reason it can act on rather than a bare refusal - the same
    principle as scene.is_reachable, one level up.
    """
    def veto(blocker: str, reason: str) -> Affordance:
        return Affordance(obj.id, False, 0.0, reason, blocker)

    # 1. Width. A gate, not an estimate: the jaw's travel is a mechanical limit.
    if obj.short_axis > scene.MAX_GRASPABLE_WIDTH:
        return veto('width',
                    f'short axis {obj.short_axis * 1000:.0f} mm exceeds the '
                    f'{scene.MAX_GRASPABLE_WIDTH * 1000:.0f} mm the jaw can take')

    # 2. Reach. Cheap radial screen before troubling the IK solver.
    reachable, why = scene.is_reachable(obj.x, obj.y)
    if not reachable:
        return veto('reach', why)

    # 3. IK, at the grasp pose and at the hover above it. This is the gate that
    #    replaces SayCan's learned value function, and the only one that can
    #    catch an awkward wrist angle - a pose can be well inside the radial
    #    envelope and still have no solution at the yaw this grasp requires.
    if ik_probe is not None:
        grasp_z = scene.grasp_height(obj.class_name)
        why = ik_probe(obj.x, obj.y, grasp_z, obj.grasp_yaw)
        if why:
            return veto('ik', f'at the grasp pose (yaw '
                              f'{math.degrees(obj.grasp_yaw):+.0f} deg): {why}')
        approach_z = scene.approach_height(obj.class_name)
        why = ik_probe(obj.x, obj.y, approach_z, obj.grasp_yaw)
        if why:
            return veto('ik_approach',
                        f'the grasp pose is fine but the hover above it is '
                        f'not, so there is no way in from above: {why}')

    # 4. Clutter. The one soft term, and the only place a number is invented.
    gap, culprit = clutter_clearance(obj, neighbours)
    if culprit is None:
        return Affordance(obj.id, True, 1.0, 'clear', '')

    if gap <= 0.0:
        return veto('clutter',
                    f'a fingertip would strike object {culprit} on the way '
                    f'down (overlap {abs(gap) * 1000:.0f} mm). Move that one '
                    f'first, or pick something else')

    if gap < SAFE_CLEARANCE:
        score = round(gap / SAFE_CLEARANCE, 2)
        return Affordance(
            obj.id, True, score,
            f'tight: {gap * 1000:.0f} mm from object {culprit} at the '
            f'fingertip, which is close to perception\'s own error. Reachable, '
            f'but likelier to fail than an isolated object', '')

    return Affordance(obj.id, True, 1.0, 'clear', '')


def score_all(objects: Sequence[ObjectGeom],
              ik_probe: Optional[IkProbe] = None) -> List[Affordance]:
    """Score every object against every other as potential clutter."""
    return [score_pick(o, objects, ik_probe) for o in objects]


# --------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------

def renumber_reason(reason: str, by_world: dict) -> str:
    """Rewrite world_model ids inside a reason string into memory ids.

    Lives here rather than in the planner because it is part of *presenting* an
    affordance verdict, and because affordance.py imports no ROS - so the
    behaviour below can be tested without a sourced workspace. A copy of this
    regex inside a test file would be a test of the copy.

    The reasons produced above name a second object ("a fingertip would strike
    object 11"). The planner translates the SUBJECT of each verdict from a
    world_model id to a memory id, but the id inside the prose was left alone,
    so the model - which is only ever shown memory ids, and is told they are
    stable - was being told to move an object that did not appear in its table.
    An instruction naming an unusable referent is worse than no explanation:
    the advice is right, the object cannot be addressed, and it reads as the
    model ignoring guidance.

    Substituted in ONE pass with a callable. Replacing ids one at a time would
    let a later rule rewrite an id an earlier rule had just produced - world 11
    to memory 3, then world 3 to memory 7 - and the note would end up naming an
    object that was never involved.

    An id with no memory entry becomes a description rather than a wrong
    number: memory folds in observations slightly behind perception, so an
    unmapped id is a normal transient rather than an error.
    """
    def swap(match):
        mem_id = by_world.get(int(match.group(1)))
        return (f'object {mem_id}' if mem_id is not None
                else 'another object the robot can see')

    return re.sub(r'object (\d+)', swap, reason)


def _self_test() -> int:
    failures: List[str] = []

    def check(label: str, got, want):
        ok = got == want
        print(f"  {'ok  ' if ok else 'FAIL'} {label:<52} {got!r}")
        if not ok:
            failures.append(f'{label}: got {got!r}, wanted {want!r}')

    cube = ObjectGeom(1, 'cube', 0.30, 0.42, 0.0, 0.040, 0.040)

    print('hard gates')
    wide = ObjectGeom(2, 'cube', 0.30, 0.42, 0.0, 0.120, 0.120)
    check('too wide -> blocked on width', score_pick(wide).blocker, 'width')
    far = ObjectGeom(3, 'cube', 0.90, 0.40, 0.0, 0.040, 0.040)
    check('out of reach -> blocked on reach', score_pick(far).blocker, 'reach')
    check('isolated cube -> feasible', score_pick(cube).feasible, True)
    check('isolated cube -> score 1.0', score_pick(cube).score, 1.0)

    print('\nIK gate (probe stubbed, so the wiring is what is under test)')
    check('IK refuses grasp -> blocked on ik',
          score_pick(cube, (), lambda *_: 'no solution').blocker, 'ik')
    # Solve the grasp height but not the hover, which is the case a radial
    # reach check cannot see at all.
    grasp_z = scene.grasp_height('cube')
    check('grasp solves, hover does not -> blocked on ik_approach',
          score_pick(cube, (), lambda x, y, z, yaw:
                     '' if abs(z - grasp_z) < 1e-9 else 'no solution'
                     ).blocker, 'ik_approach')
    check('IK solves both -> feasible',
          score_pick(cube, (), lambda *_: '').feasible, True)
    # The probe's own words have to survive into the reason, or a diagnosis
    # made down in the solver never reaches whoever is reading the output.
    reason = score_pick(cube, (), lambda *_: 'the elbow would be in the table').reason
    check('the probe\'s explanation reaches the planner',
          'elbow would be in the table' in reason, True)

    print('\nclutter is directional - the whole point of the fingertip model')
    # grasp_yaw = 0 means the jaw closes along +/- x, so the fingertips are
    # offset in x. A neighbour offset in x is in the way; the same neighbour at
    # the same distance in y is not. A centre-distance test scores these equal.
    # The neighbour must sit inside the clearance band for this comparison to
    # discriminate at all - placed further out, BOTH cases score 1.0 and the
    # test passes while proving nothing. (It did exactly that on the first run.)
    gap = open_gap_for(cube)          # 40 + 20 = 60 mm, tips at +/- 30 mm
    d = 0.5 * gap + FINGER_RADIUS + 0.008    # 8 mm clear of the fingertip
    in_the_way = ObjectGeom(9, 'cube', cube.x + d, cube.y, 0.0, 0.001, 0.001)
    beside_it = ObjectGeom(9, 'cube', cube.x, cube.y + d, 0.0, 0.001, 0.001)

    along = score_pick(cube, [cube, in_the_way])
    across = score_pick(cube, [cube, beside_it])
    print(f"    neighbour {d * 1000:.0f} mm out along the closing axis:"
          f" score {along.score}  ({along.reason})")
    print(f"    the same neighbour across it:               "
          f" score {across.score}  ({across.reason})")
    check('neighbour across the jaw is ignored', across.score, 1.0)
    check('neighbour on the closing axis is feasible but degraded',
          along.feasible, True)
    if not 0.0 < along.score < 1.0:
        failures.append(f'8 mm of fingertip clearance should score strictly '
                        f'between 0 and 1, got {along.score}')
    if along.score >= across.score:
        failures.append('a neighbour on the closing axis scored no worse than '
                        'one beside it - the fingertip model is not being used, '
                        'which is the whole reason it exists')

    # Now bring it close enough to bite.
    touching = ObjectGeom(9, 'cube', cube.x + 0.5 * gap, cube.y, 0.0, 0.03, 0.03)
    check('fingertip inside a neighbour -> blocked on clutter',
          score_pick(cube, [cube, touching]).blocker, 'clutter')

    print('\nreal scene: the two yellow cubes from the world SDF')
    # link0 = world - (0.05, 0.00). cube_yellow_1 (0.50,0.20), _2 (0.30,0.42).
    scene_objs = [
        ObjectGeom(4, 'cube', 0.45, 0.20, 0.35 + math.pi / 2, 0.040, 0.040),
        ObjectGeom(6, 'cube', 0.25, 0.42, -0.60 + math.pi / 2, 0.040, 0.040),
        ObjectGeom(2, 'cube', 0.10, 0.48, 0.20 + math.pi / 2, 0.040, 0.040),
    ]
    for a in score_all(scene_objs):
        print(f"    id {a.id}: feasible={a.feasible} score={a.score} {a.reason}")
        if not a.feasible:
            failures.append(f'id {a.id} in the authored scene scored '
                            f'infeasible ({a.blocker}: {a.reason}) - the scene '
                            f'is known to be workable, so the scorer is wrong')

    # renumber_reason: world ids inside the prose must become memory ids.
    by_world = {11: 3, 3: 7, 10: 1}
    check('renumber: single id',
          renumber_reason('a fingertip would strike object 11 on the way down',
                          by_world),
          'a fingertip would strike object 3 on the way down')
    # The one that matters. Replacing ids one at a time would map 11 -> 3 and
    # then that same 3 -> 7, naming an object never involved.
    check('renumber: no chained rewrite',
          renumber_reason('object 11 and object 3', by_world),
          'object 3 and object 7')
    check('renumber: unmapped id becomes a description',
          renumber_reason('strike object 99 on the way down', by_world),
          'strike another object the robot can see on the way down')
    check('renumber: leaves other numbers alone',
          renumber_reason('tight: 12 mm at the fingertip', by_world),
          'tight: 12 mm at the fingertip')

    print()
    if failures:
        print(f'FAILED ({len(failures)}):')
        for f in failures:
            print(f'  - {f}')
        return 1
    print('all affordance self-tests passed')

    return 0


if __name__ == '__main__':
    raise SystemExit(_self_test())

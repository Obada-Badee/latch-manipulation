#!/usr/bin/env python3
"""Add, remove and reshuffle objects in the RUNNING cell.

--------------------------------------------------------------------------
WHY
--------------------------------------------------------------------------
Until now the scene was whatever `arm_on_the_table.sdf` said at launch: nine
objects, fixed poses, fixed colours, for the life of the process. Everything
downstream was therefore free to quietly depend on that - and dependencies you
never test are the ones that bite. A world that can change while the robot is
running is what turns three separate claims into tested ones:

  perception    ids are assigned by tracking, not by position. An object that
                appears mid-run must get an id; one that leaves must not linger.
  memory        object permanence has to distinguish "I cannot see it" from
                "it is gone". Only a world where things really do arrive and
                leave can tell those apart.
  planner       nothing in the prompt or the tools may assume a fixed cast.
                "Put the two yellow cubes away" must work when there are three.

This is deliberately a separate command-line tool rather than a planner tool.
Spawning objects is the *experimenter's* job, not the robot's - a robot that can
conjure objects into existence is not solving the task, and giving the model
that button would make every evaluation meaningless.

--------------------------------------------------------------------------
SAFETY
--------------------------------------------------------------------------
Everything spawned here is named with a `dyn_` prefix and `--all` only ever
removes models carrying it. The robot, the table, the bins and the authored
objects cannot be deleted by this tool even by mistake, which matters because
`/world/<w>/remove` will happily delete the arm if you name it.

    python3 tools/scene_manager.py list
    python3 tools/scene_manager.py spawn --count 4
    python3 tools/scene_manager.py spawn --class cube --color green --x 0.30 --y 0.50
    python3 tools/scene_manager.py remove --all
    python3 tools/scene_manager.py randomize --count 8
    python3 tools/scene_manager.py --self-test         # no Gazebo needed
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_HERE, "synth_dataset"))
sys.path.insert(0, os.path.join(_ROOT, "src", "yolov8_obb"))
sys.path.insert(0, os.path.join(_ROOT, "src", "llm_task_planner"))

import objects as pool_objects                                  # noqa: E402
from geometry import resting_z                                  # noqa: E402
from gz_client import GzClient, GzError                         # noqa: E402
from llm_task_planner import scene as robot_scene               # noqa: E402
from yolov8_obb.projection import CAMERA, link0_to_world, world_to_link0  # noqa: E402

#: Prefix on every model this tool creates. `remove --all` is scoped to it, so
#: the authored scene and the robot are out of reach of an accidental wipe.
PREFIX = "dyn_"

DEFAULT_WORLD = "arm_and_table"

#: Where a spawned object may land, in link0. Sampled inside this box and then
#: filtered against the real constraints below - the box only bounds the search.
SAMPLE_X = (0.05, 0.55)
SAMPLE_Y = (0.10, 0.70)

#: Keep clear of the reach limits rather than sitting on them: an object exactly
#: at MAX_REACH is reachable in principle and a coin flip in practice, and a
#: scene that is 50% unpickable teaches nothing about the planner.
REACH_MARGIN = 0.06

#: Pixels of margin inside the image. An object touching the frame edge is
#: clipped, and a clipped OBB has the wrong short axis, which would corrupt the
#: grasp width before anything else got a chance to go wrong.
FRAME_MARGIN_PX = 40

#: Minimum centre-to-centre spacing between objects, metres. Wide enough that
#: the gripper fingers (see affordance.FINGER_RADIUS) have somewhere to go.
MIN_SEPARATION = 0.11

#: Bins are obstacles, not places to drop spawned objects into.
BIN_KEEPOUT = 0.16

SPAWN_ATTEMPTS = 400

CUBE = pool_objects.CUBE_SIZE
CYL_R = pool_objects.CYLINDER_RADIUS
CYL_L = pool_objects.CYLINDER_LENGTH
TABLE_TOP_WORLD_Z = pool_objects.TABLE_TOP_WORLD_Z

MASS = 0.05


# --------------------------------------------------------------------------
# SDF for a *manipulable* object
# --------------------------------------------------------------------------
# The dataset generator's pool (objects.PoolItem.sdf) is visual-only and static:
# it exists to be photographed. These objects exist to be picked up, so they
# need mass, inertia, collision geometry and friction. Structure and numbers are
# taken from the authored models in arm_on_the_table.sdf so a spawned cube
# behaves exactly like one that was there at launch - otherwise a grasp failure
# would leave you wondering whether the object or the robot was at fault.

def _inertia_box(m: float, a: float) -> Tuple[float, float, float]:
    """Solid cube of side a: I = m a^2 / 6 about every principal axis."""
    i = m * a * a / 6.0
    return i, i, i


def _inertia_cylinder(m: float, r: float, h: float) -> Tuple[float, float, float]:
    """Solid cylinder about its own axes: transverse m(3r^2+h^2)/12, axial mr^2/2."""
    transverse = m * (3.0 * r * r + h * h) / 12.0
    return transverse, transverse, m * r * r / 2.0


def _inertial(ixx: float, iyy: float, izz: float, mass: float = MASS) -> str:
    return (f'<inertial><mass>{mass}</mass><inertia>'
            f'<ixx>{ixx:.6e}</ixx><ixy>0</ixy><ixz>0</ixz>'
            f'<iyy>{iyy:.6e}</iyy><iyz>0</iyz><izz>{izz:.6e}</izz>'
            f'</inertia></inertial>')


_FRICTION = '<surface><friction><ode><mu>1.0</mu><mu2>1.0</mu2></ode></friction></surface>'


def dynamic_sdf(name: str, class_name: str, color: str) -> str:
    """A spawnable model the robot can actually pick up."""
    if class_name == "bolt":
        # Reuse the authored bolt model wholesale, which brings its own mesh,
        # collision and inertia. Wrapping the include in a named <model> matches
        # how arm_on_the_table.sdf does it, and keeps `name` the entity the
        # pose services and remove() will address.
        return (f'<sdf version="1.9"><model name="{name}">'
                f'<include><uri>model://bolt</uri></include>'
                f'</model></sdf>')

    material = pool_objects.MATERIALS[color].sdf()
    if class_name == "cube":
        geo = f'<box><size>{CUBE} {CUBE} {CUBE}</size></box>'
        inertia = _inertial(*_inertia_box(MASS, CUBE))
    elif class_name == "cylinder":
        geo = f'<cylinder><radius>{CYL_R}</radius><length>{CYL_L}</length></cylinder>'
        inertia = _inertial(*_inertia_cylinder(MASS, CYL_R, CYL_L))
    else:
        raise ValueError(f'no SDF template for class {class_name!r}')

    return (f'<sdf version="1.9"><model name="{name}"><link name="link">'
            f'{inertia}'
            f'<collision name="collision"><geometry>{geo}</geometry>{_FRICTION}</collision>'
            f'<visual name="visual"><geometry>{geo}</geometry>{material}</visual>'
            f'</link></model></sdf>')


# --------------------------------------------------------------------------
# Placement
# --------------------------------------------------------------------------

def _bin_conflict(x: float, y: float) -> bool:
    return any(math.hypot(x - r.x, y - r.y) < BIN_KEEPOUT
               for r in robot_scene.RECEPTACLES.values())


def placement_ok(x: float, y: float, plane_z: float,
                 taken: Sequence[Tuple[float, float]]) -> Tuple[bool, str]:
    """Is this a sane place to put an object? Every test is a real constraint.

    Nothing here is a taste judgement: each rejection corresponds to something
    that would actually go wrong downstream, and says which.
    """
    reach = robot_scene.reach_of(x, y)
    if not (robot_scene.MIN_REACH + REACH_MARGIN <= reach
            <= robot_scene.MAX_REACH - REACH_MARGIN):
        return False, f'reach {reach:.3f} m is outside the comfortable band'

    u, v = CAMERA.link0_to_pixel(x, y, plane_z)
    if not CAMERA.in_frame(u, v, FRAME_MARGIN_PX):
        return False, f'projects to ({u:.0f}, {v:.0f}), too near the frame edge'

    if _bin_conflict(x, y):
        return False, 'inside a bin keep-out'

    for tx, ty in taken:
        d = math.hypot(x - tx, y - ty)
        if d < MIN_SEPARATION:
            return False, f'only {d * 1000:.0f} mm from another object'

    return True, ''


def sample_placement(rng: np.random.Generator, plane_z: float,
                     taken: Sequence[Tuple[float, float]]
                     ) -> Optional[Tuple[float, float]]:
    """Rejection-sample a free spot, or None if the table is too full."""
    last = 'no attempt made'
    for _ in range(SPAWN_ATTEMPTS):
        x = float(rng.uniform(*SAMPLE_X))
        y = float(rng.uniform(*SAMPLE_Y))
        ok, why = placement_ok(x, y, plane_z, taken)
        if ok:
            return x, y
        last = why
    print(f'  could not place after {SPAWN_ATTEMPTS} attempts (last: {last})',
          file=sys.stderr)
    return None


# --------------------------------------------------------------------------
# The manager
# --------------------------------------------------------------------------

class SceneManager:

    def __init__(self, world: str = DEFAULT_WORLD, seed: Optional[int] = None):
        self.gz = GzClient(world)
        self.gz.wait_until_ready()
        self.classes = pool_objects.build_classes()
        self.rng = np.random.default_rng(seed)
        self._serial = int(time.time()) % 100000

    # ------------------------------------------------------------- inspection

    def poses(self) -> Dict[str, Tuple[float, float, float]]:
        return self.gz.read_poses()

    def spawned(self) -> List[str]:
        return sorted(n for n in self.poses() if n.startswith(PREFIX))

    #: World-frame slab an object has to be in to count as "on the table".
    #: /world/<w>/pose/info reports links as well as models, and link poses are
    #: relative to their parent - so a robot link reads as being near the origin
    #: and fails this test, while a real tabletop object passes it. Filtering by
    #: name instead would need a list of every link the robot has, which is a
    #: list that changes whenever the URDF does.
    TABLE_SLAB_Z = (TABLE_TOP_WORLD_Z - 0.02, TABLE_TOP_WORLD_Z + 0.30)
    #: Table is 1.5 x 0.8 m centred at world (0.29, 0) and yawed 90 deg, so it
    #: spans 0.8 m along world x and 1.5 m along world y.
    TABLE_SPAN_X = (0.29 - 0.40, 0.29 + 0.40)
    TABLE_SPAN_Y = (-0.75, 0.75)

    def occupied_link0(self) -> List[Tuple[float, float]]:
        """Everything already sitting on the table, in link0 x/y.

        Read from Gazebo rather than from a list this process keeps, so a second
        terminal spawning objects cannot make this one stack two things in the
        same place.

        The slab-and-footprint filter is doing real work. Without it this
        returned robot links and bin walls as if they were tabletop objects,
        while MISSING the actual objects - because read_poses was handing back
        quaternion components as positions. The visible symptom was a spawned
        cube landing 51 mm from bolt2, which the affordance layer then correctly
        refused to pick. Two bugs, one of which was only found because the other
        one made a mess the scorer complained about.
        """
        spots = []
        for name, (wx, wy, wz) in self.poses().items():
            if name in ('table', 'ground_plane', 'sun') or name.endswith('_bin'):
                continue
            if not (self.TABLE_SLAB_Z[0] <= wz <= self.TABLE_SLAB_Z[1]):
                continue
            if not (self.TABLE_SPAN_X[0] <= wx <= self.TABLE_SPAN_X[1]
                    and self.TABLE_SPAN_Y[0] <= wy <= self.TABLE_SPAN_Y[1]):
                continue
            x, y, _ = world_to_link0(wx, wy, wz)
            spots.append((x, y))
        return spots

    # ------------------------------------------------------------------ spawn

    def _next_name(self, class_name: str, color: str) -> str:
        self._serial += 1
        return f'{PREFIX}{class_name}_{color}_{self._serial}'

    def spawn_one(self, class_name: str, color: Optional[str] = None,
                  x: Optional[float] = None, y: Optional[float] = None
                  ) -> Optional[str]:
        cls = self.classes[class_name]
        if cls.fixed_color:
            color = cls.fixed_color                # a bolt is grey; not a choice
        elif color is None:
            color = str(self.rng.choice(list(pool_objects.MATERIALS)))

        # Draw a resting orientation from the same sampler the dataset generator
        # uses, so spawned objects sit the way the detector was trained to see
        # them - upright cylinders, toppled ones, bolts on their side.
        rpy = cls.sample_rpy(self.rng)
        wz = resting_z(cls.vertices, rpy, TABLE_TOP_WORLD_Z)
        plane_z = world_to_link0(0.0, 0.0, wz)[2]

        taken = self.occupied_link0()
        if x is None or y is None:
            spot = sample_placement(self.rng, plane_z, taken)
            if spot is None:
                return None
            x, y = spot
        else:
            ok, why = placement_ok(x, y, plane_z, taken)
            if not ok:
                print(f'  refusing ({x:+.3f}, {y:+.3f}): {why}', file=sys.stderr)
                return None

        name = self._next_name(class_name, color)
        wx, wy, _ = link0_to_world(x, y, 0.0)
        self.gz.spawn(name, dynamic_sdf(name, class_name, color),
                      (wx, wy, wz), rpy)
        print(f'  spawned {name:<28} link0 ({x:+.3f}, {y:+.3f})  '
              f'yaw {math.degrees(rpy[2]):+.0f} deg')
        return name

    def spawn_many(self, count: int,
                   class_name: Optional[str] = None,
                   color: Optional[str] = None) -> List[str]:
        names = []
        for _ in range(count):
            cname = class_name or str(self.rng.choice(list(self.classes)))
            got = self.spawn_one(cname, color)
            if got:
                names.append(got)
            # Give the server a moment to register the entity, so the next
            # occupied_link0() sees it and does not place on top of it.
            time.sleep(0.25)
        return names

    # ----------------------------------------------------------------- remove

    def remove(self, name: str) -> bool:
        if not name.startswith(PREFIX):
            print(f'  refusing to remove {name!r}: only {PREFIX}* models are '
                  f'this tool\'s to delete', file=sys.stderr)
            return False
        self.gz.remove(name)
        print(f'  removed {name}')
        return True

    def remove_all(self) -> int:
        removed = 0
        for name in self.spawned():
            if self.remove(name):
                removed += 1
            time.sleep(0.15)
        return removed


# --------------------------------------------------------------------------
# Self-test: placement logic only, no Gazebo
# --------------------------------------------------------------------------

def _self_test() -> int:
    failures: List[str] = []

    def check(label, got, want):
        ok = got == want
        print(f"  {'ok  ' if ok else 'FAIL'} {label:<52} {got!r}")
        if not ok:
            failures.append(f'{label}: got {got!r}, wanted {want!r}')

    print('inertia is computed from geometry, not copied from the SDF')
    ixx, _, _ = _inertia_box(MASS, CUBE)
    print(f'    cube      ixx {ixx:.4e}   world SDF says 1.333e-5')
    if abs(ixx - 1.333e-5) > 1e-8:
        failures.append(f'cube inertia {ixx:.4e} disagrees with the authored SDF')
    tx, _, tz = _inertia_cylinder(MASS, CYL_R, CYL_L)
    print(f'    cylinder  ixx {tx:.4e} izz {tz:.4e}   '
          f'world SDF says 2.0e-5 / 1.0e-5')
    if abs(tx - 2.0e-5) > 1e-8 or abs(tz - 1.0e-5) > 1e-8:
        failures.append('cylinder inertia disagrees with the authored SDF')

    print('\nevery authored object position is accepted by the placement rules')
    # link0 = world - (0.05, 0). If the rules reject the scene the system has
    # been working with all along, the rules are wrong, not the scene.
    authored = [(0.45, 0.20), (0.25, 0.42), (0.10, 0.48),
                (0.40, 0.45), (0.05, 0.62), (0.30, 0.60),
                (0.45, 0.35), (0.20, 0.63), (0.15, 0.30)]
    plane_z = 0.015
    for x, y in authored:
        ok, why = placement_ok(x, y, plane_z, [])
        if not ok:
            failures.append(f'authored spot ({x:+.2f}, {y:+.2f}) rejected: {why}')
    print(f'    {len(authored)} authored positions, '
          f'{sum(1 for x, y in authored if placement_ok(x, y, plane_z, [])[0])} accepted')

    print('\nrules reject what they should')
    check('a spot inside the red bin', placement_ok(0.28, -0.26, plane_z, [])[0], False)
    check('a spot beyond reach', placement_ok(0.80, 0.40, plane_z, [])[0], False)
    check('a spot on top of another object',
          placement_ok(0.30, 0.42, plane_z, [(0.30, 0.44)])[0], False)
    far = placement_ok(0.30, 0.42, plane_z, [(0.30, 0.60)])
    check('a spot a clear distance away', far[0], True)

    print('\nsampling fills a table without overlaps')
    rng = np.random.default_rng(7)
    taken: List[Tuple[float, float]] = []
    for _ in range(8):
        spot = sample_placement(rng, plane_z, taken)
        if spot is None:
            break
        taken.append(spot)
    print(f'    placed {len(taken)} objects')
    if len(taken) < 6:
        failures.append(f'only {len(taken)} objects fitted; the sampler or the '
                        f'constraints are too tight for a useful scene')
    worst = min((math.hypot(a[0] - b[0], a[1] - b[1])
                 for i, a in enumerate(taken) for b in taken[i + 1:]),
                default=math.inf)
    print(f'    closest pair {worst * 1000:.0f} mm apart '
          f'(minimum {MIN_SEPARATION * 1000:.0f} mm)')
    if worst < MIN_SEPARATION:
        failures.append('sampler produced an overlapping pair')

    print('\nSDF generation')
    sdf = dynamic_sdf('dyn_cube_red_1', 'cube', 'red')
    for want in ('<collision', '<inertial', '<mass>', 'dyn_cube_red_1'):
        if want not in sdf:
            failures.append(f'cube SDF is missing {want!r} - it would not be '
                            f'manipulable')
    if '<static>' in sdf:
        failures.append('spawned objects must not be static; they have to fall '
                        'and be picked up')
    bolt = dynamic_sdf('dyn_bolt_grey_1', 'bolt', 'grey')
    if 'model://bolt' not in bolt:
        failures.append('bolt SDF should reuse the authored model')
    print(f'    cube SDF {len(sdf)} chars, bolt SDF {len(bolt)} chars')

    print('\nremove is scoped')
    print(f'    prefix {PREFIX!r}; remove --all touches nothing else')

    print()
    if failures:
        print(f'FAILED ({len(failures)}):')
        for f in failures:
            print(f'  - {f}')
        return 1
    print('all scene_manager self-tests passed')
    return 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main() -> int:
    # --world and --seed go on a shared parent so they are accepted on either
    # side of the subcommand. argparse only accepts main-parser options BEFORE
    # the subcommand, and `spawn --seed 11` failing with "unrecognized
    # arguments" is a trap not worth leaving for the next person.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--world', default=DEFAULT_WORLD)
    common.add_argument('--seed', type=int, default=None)

    ap = argparse.ArgumentParser(description=__doc__, parents=[common],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--self-test', action='store_true',
                    help='check the placement rules offline; no Gazebo needed')
    sub = ap.add_subparsers(dest='cmd')

    sub.add_parser('list', parents=[common], help='show what is on the table')

    sp = sub.add_parser('spawn', parents=[common], help='add objects')
    sp.add_argument('--count', type=int, default=1)
    sp.add_argument('--class', dest='class_name',
                    choices=('cube', 'cylinder', 'bolt'))
    sp.add_argument('--color', choices=tuple(pool_objects.MATERIALS))
    sp.add_argument('--x', type=float)
    sp.add_argument('--y', type=float)

    rm = sub.add_parser('remove', parents=[common],
                        help='delete objects this tool spawned')
    rm.add_argument('name', nargs='?')
    rm.add_argument('--all', action='store_true')

    rz = sub.add_parser('randomize', parents=[common],
                        help='clear spawned objects and make a new set')
    rz.add_argument('--count', type=int, default=6)

    args = ap.parse_args()

    if args.self_test:
        return _self_test()
    if args.cmd is None:
        ap.print_help()
        return 2

    try:
        mgr = SceneManager(args.world, args.seed)
    except GzError as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 1

    if args.cmd == 'list':
        poses = mgr.poses()
        print(f'{len(poses)} models in {args.world}:')
        for name, (wx, wy, wz) in sorted(poses.items()):
            x, y, z = world_to_link0(wx, wy, wz)
            tag = ' (spawned)' if name.startswith(PREFIX) else ''
            print(f'  {name:<28} link0 ({x:+.3f}, {y:+.3f}, {z:+.3f}){tag}')
        return 0

    if args.cmd == 'spawn':
        if args.x is not None or args.y is not None:
            if args.x is None or args.y is None:
                print('error: --x and --y go together', file=sys.stderr)
                return 2
            if not args.class_name:
                print('error: --class is required with an explicit position',
                      file=sys.stderr)
                return 2
            return 0 if mgr.spawn_one(args.class_name, args.color,
                                      args.x, args.y) else 1
        made = mgr.spawn_many(args.count, args.class_name, args.color)
        print(f'spawned {len(made)} of {args.count} requested')
        return 0 if len(made) == args.count else 1

    if args.cmd == 'remove':
        if args.all:
            print(f'removed {mgr.remove_all()} spawned models')
            return 0
        if not args.name:
            print('error: give a name or --all', file=sys.stderr)
            return 2
        return 0 if mgr.remove(args.name) else 1

    if args.cmd == 'randomize':
        print(f'clearing {mgr.remove_all()} spawned models')
        time.sleep(0.5)
        made = mgr.spawn_many(args.count)
        print(f'spawned {len(made)} of {args.count} requested')
        return 0

    return 2


if __name__ == '__main__':
    raise SystemExit(main())

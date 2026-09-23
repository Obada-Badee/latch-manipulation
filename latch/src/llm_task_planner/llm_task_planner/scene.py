"""Scene and gripper constants, in the panda_link0 frame.

One place for the numbers the skill layer needs and perception cannot supply:
where the bins are, how high to reach, and how wide the jaw opens. Anything the
camera *can* see belongs in /world_state instead, not here.

Provenance is given for every value. Where a number is derived from geometry it
says so; where it needs confirming against the running sim it says that too, and
those are marked CONFIRM. Guessing silently is how "the IK is broken" bugs are
born.

Frame: panda_link0. +X away from the robot, +Y to the robot's LEFT, +Z up.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

# --------------------------------------------------------------------------
# Gripper
# --------------------------------------------------------------------------
# panda_finger_joint1 travels 0 .. 0.04 m and finger_joint2 mimics it, so the
# gap between the fingers is twice the joint value: 0 .. 0.08 m.
# (robot_description/urdf/panda.urdf:268-281)
FINGER_JOINT_MAX = 0.04
MAX_JAW_GAP = 2 * FINGER_JOINT_MAX          # 80 mm, fully open

#: The arm's seven joints, in the order the trajectory controller expects.
ARM_JOINT_NAMES = [f'panda_joint{i}' for i in range(1, 8)]

#: Joint values for the `ready` posture defined in config/panda.srdf:20-28.
#:
#: These must equal the SRDF, because two code paths reach `ready` by different
#: routes: the home skill asks MoveIt for the named state, and the recovery path
#: drives these numbers straight to the controller. If they disagree, "the arm
#: is home" means two different postures depending on which route ran, and every
#: claim resting on it - the table is clear, the camera's view is unobstructed -
#: is being made about a configuration the arm may not be in.
#:
#: They *did* disagree. The recovery path carried
#: [0, -0.30, 0, -2.00, 0, 1.75, 0.79], copied between the benchmark harness and
#: tools/arm_reset.py and labelled "ready", against the SRDF's
#: [0, -0.785, 0, -2.356, 0, 1.571, 0.785] - joint 2 off by 0.485 rad, joint 4
#: by 0.356. Nothing compared them, so nothing complained; it surfaced only when
#: a test asserted that the arm was at `ready` after a *successful* MoveIt home
#: and the assertion failed. tools/test_home_rescue.py now checks this.
READY_JOINTS = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]

#: Commanded open width. 60 mm clears a 40 mm cube by 10 mm a side without
#: swinging the fingers out to their stops. Matches the original code's 0.03.
OPEN_JAW_GAP = 0.06

#: Commanded closed position for the finger joint. The controller stalls against
#: the object rather than reaching this, which is what makes the grasp check work.
CLOSE_FINGER_JOINT = 0.001

#: Widest object the jaw can take, with margin for approach error.
MAX_GRASPABLE_WIDTH = 0.070

#: Below this measured gap after closing, the fingers met nothing: the grasp
#: missed. A 40 mm cube should hold the gap near 0.040.
GRASP_MISS_GAP = 0.008

#: How much wider than the object to open before descending, total across both
#: fingers - so half of this is the clearance on each side.
#:
#: This was 0.008 while the simulator ran on bullet-featherstone, where the
#: finger joint crawled at ~0.6 mm/s and every millimetre of jaw travel cost
#: about two seconds; opening wide and closing again took close to a minute and
#: timed out mid-motion. Under dartsim the same joint runs at ~112 mm/s opening
#: and ~51 mm/s closing, so an extra 12 mm of travel costs well under a second
#: and there is no longer any reason to be stingy.
#:
#: 0.020 restores the 10 mm per side that OPEN_JAW_GAP was designed around, and
#: that margin is needed: perception's worst-case position error is 8.3 mm for a
#: cylinder (measured in tools/synth_dataset/labeler.py against ground truth).
#: At the old 0.008 the clearance was 4 mm a side, i.e. *less* than the error -
#: a finger would have struck the object on the way down.
OPEN_CLEARANCE = 0.020

#: How far the measured gap may exceed the object's width and still count as
#: holding it. Wider than this means the jaw never closed.
GRASP_WIDTH_TOLERANCE = 0.010

#: Standoff from the physical jaw limit, so the controller is never commanded
#: into its own end stop.
JAW_STANDOFF = 0.002


def open_gap_for(short_axis: float) -> float:
    """How wide to open the jaw before descending on an object this wide.

    One definition, because there were two and they disagreed. The skill server
    capped at OPEN_JAW_GAP (60 mm) while the affordance layer capped at
    MAX_JAW_GAP (80 mm) - under a docstring that claimed to describe "how wide
    the jaw will actually be opened". So affordance certified a grasp assuming a
    79 mm opening that the skill never used.

    The consequence was not theoretical. A 59 mm object got
    min(59 + 20, 60) = 60 mm, i.e. **0.5 mm of clearance per side**. The descent
    drove the fingers onto the object and forced them apart, and the grasp check
    then reported "fingers are at 72.7 mm but this object is only 59 mm wide -
    the gripper did not close", which reads as a gripper fault and was a
    clearance fault.

    OPEN_JAW_GAP stays what it is - the width to open to when nothing in
    particular is being grasped, e.g. releasing. It is not a ceiling on reaching
    around a wide object, and using it as one was the bug.

    Honest limit: OPEN_CLEARANCE buys 10 mm a side, which was chosen to exceed
    perception's worst-case 8.3 mm position error. Above a 58 mm object the jaw
    runs out of travel before that, so wide objects are grasped with less margin
    than the error - a property of an 80 mm jaw, not something this can fix.
    """
    return min(MAX_JAW_GAP - JAW_STANDOFF, short_axis + OPEN_CLEARANCE)

# --------------------------------------------------------------------------
# Heights
# --------------------------------------------------------------------------
# MoveIt is commanded on panda_link8 (set_goal_state(pose_link="panda_link8")),
# but what we care about is where the FINGERTIPS are. The standard Panda offset
# from link8 to the tip centre is 0.1034 m: the fingers mount 0.0584 above the
# hand frame (panda.urdf:271) and are ~45 mm long.
TCP_OFFSET_Z = 0.1034

#: Table top in link0. World z 1.015 minus the base height 1.02 - the table is
#: fractionally BELOW the robot's origin plane, which is easy to get backwards.
TABLE_TOP_Z = -0.005

#: Fingertip height at which to close on each class, from the world SDF:
#:   cube      40 mm, centre at link0 z 0.015 -> grasp through the middle
#:   cylinder  60 mm tall, centre at link0 z 0.025
#:   bolt      lies on its side; shaft centre roughly one radius up
#: CONFIRM against the sim: bolts settle under physics rather than being placed,
#: so their resting height is not exactly what the SDF pose says.
GRASP_TIP_Z = {
    "cube": 0.015,
    "cylinder": 0.025,
    "bolt": 0.008,
}
DEFAULT_GRASP_TIP_Z = 0.015

#: Clearance above the object to approach from and retreat to.
APPROACH_CLEARANCE = 0.10

#: Fingertip height while moving a held object across the table. Must clear the
#: bin rims (0.050) and anything standing on the table by a comfortable margin.
CARRY_TIP_Z = 0.20


def link8_z_for_tip(tip_z: float) -> float:
    """Convert a desired fingertip height into the link8 z MoveIt is given."""
    return tip_z + TCP_OFFSET_Z


def grasp_height(class_name: str) -> float:
    """link8 z at which to close the gripper on this class of object."""
    return link8_z_for_tip(GRASP_TIP_Z.get(class_name, DEFAULT_GRASP_TIP_Z))


def approach_height(class_name: str) -> float:
    """link8 z to hover at before descending onto this class of object."""
    return link8_z_for_tip(
        GRASP_TIP_Z.get(class_name, DEFAULT_GRASP_TIP_Z) + APPROACH_CLEARANCE)


CARRY_HEIGHT = link8_z_for_tip(CARRY_TIP_Z)


# --------------------------------------------------------------------------
# Receptacles
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Receptacle:
    """A fixed destination. Bins do not move, so hardcoding them is legitimate
    rather than a fudge - unlike the objects, which must be perceived."""
    name: str
    color: str
    x: float
    y: float
    rim_z: float            # link0 z of the highest wall
    inner_half_width: float # usable drop area, half-width in metres

    @property
    def release_height(self) -> float:
        """link8 z to open the fingers at: above the rim, so the object drops in
        rather than being dragged through a wall."""
        return link8_z_for_tip(self.rim_z + 0.06)

    def drop_point(self, nth: int) -> Tuple[float, float]:
        """Where to release the nth object put into this bin.

        Not the centre every time, which is what this used to do and what made
        multi-object tasks fail. Measured: three cubes dropped at the identical
        point stack rather than spread, landing at z = 0.020, 0.060 and 0.100
        against a rim at 0.050 - so from the SECOND object onward the thing sits
        proud of the rim and the task checker, which asks for z < rim_z, counts
        it as not in the bin. That single behaviour explains the whole success
        curve: 58% for tasks needing one object in a bin, 40% for two, 10% for
        four.

        Lateral aim is not the problem and never was - measured mean error 16 mm,
        max 23 mm against a 90 mm half-width. There is ample room to spread: the
        inner area is 180 x 180 mm and a cube is 40 mm.

        The pattern cycles rather than counting upward without limit, because the
        robot cannot see into a bin and so cannot know when one has been emptied.
        A cycle is right either way: each of the four spots is a valid drop, and
        after an unobserved reset the robot simply starts somewhere in the ring.
        """
        d = self.inner_half_width * 0.5          # 45 mm on a 90 mm half-width
        ring = ((+d, +d), (-d, -d), (+d, -d), (-d, +d))
        dx, dy = ring[nth % len(ring)]
        return self.x + dx, self.y + dy


#: Derived from worlds/arm_on_the_table.sdf by subtracting the robot base
#: position (0.05, 0.00, 1.02):
#:     red_bin  world (0.33, -0.26, 1.015) -> link0 (0.28, -0.26, -0.005)
#:     blue_bin world (0.57, -0.22, 1.015) -> link0 (0.52, -0.22, -0.005)
#: Walls are 50 mm tall centred 30 mm up from a 5 mm base, so the rim sits
#: 55 mm above the model origin: -0.005 + 0.055 = 0.050.
#: Keep these in sync with the SDF - there is no runtime check that they agree.
RECEPTACLES: Dict[str, Receptacle] = {
    "red_bin": Receptacle(
        name="red_bin", color="red",
        x=0.28, y=-0.26, rim_z=0.050, inner_half_width=0.09),
    "blue_bin": Receptacle(
        name="blue_bin", color="blue",
        x=0.52, y=-0.22, rim_z=0.050, inner_half_width=0.09),
}


def find_receptacle(name: str) -> Optional[Receptacle]:
    """Look up a bin, tolerating the aliases a language model may produce."""
    if not name:
        return None
    key = name.strip().lower().replace(" ", "_").replace("-", "_")
    if key in RECEPTACLES:
        return RECEPTACLES[key]
    # "red", "the red bin", "red container" -> red_bin
    for r in RECEPTACLES.values():
        if r.color in key:
            return r
    return None


# --------------------------------------------------------------------------
# Reachability
# --------------------------------------------------------------------------
#: Radial distance from the base within which planning reliably succeeds. The
#: Panda reaches ~0.855 m at full stretch, but poses near that limit need the
#: arm almost straight and fail often, and poses very close in are singular.
MIN_REACH = 0.25
MAX_REACH = 0.75


def reach_of(x: float, y: float) -> float:
    return (x * x + y * y) ** 0.5


def is_reachable(x: float, y: float) -> Tuple[bool, str]:
    """Cheap pre-check before asking MoveIt.

    Catching this here turns "planning failed" - which tells the planner nothing
    it can act on - into a message it can actually reason about.
    """
    r = reach_of(x, y)
    if r < MIN_REACH:
        return False, f"({x:.3f}, {y:.3f}) is {r:.3f} m from the base, too close to reach"
    if r > MAX_REACH:
        return False, f"({x:.3f}, {y:.3f}) is {r:.3f} m from the base, beyond comfortable reach"
    return True, ""


def _self_test() -> int:
    failures = []
    for r in RECEPTACLES.values():
        ok, why = is_reachable(r.x, r.y)
        print(f"  {r.name:<9} ({r.x:+.2f},{r.y:+.2f})  reach {reach_of(r.x, r.y):.3f}  "
              f"release link8 z {r.release_height:.3f}  {'ok' if ok else why}")
        if not ok:
            failures.append(f"{r.name} unreachable: {why}")

    print()
    for cls in ("cube", "cylinder", "bolt"):
        print(f"  {cls:<9} grasp link8 z {grasp_height(cls):.4f}   "
              f"approach {approach_height(cls):.4f}")
    print(f"  carry     link8 z {CARRY_HEIGHT:.4f}")

    # The original code used pick_height 0.126 for bolts; our derived value
    # should land in the same neighbourhood or one of the two is wrong.
    legacy = 0.126
    ours = grasp_height("bolt")
    print(f"\n  bolt grasp height: derived {ours:.4f} vs original hardcoded {legacy:.4f}"
          f"  (delta {abs(ours - legacy) * 1000:.0f} mm)")
    if abs(ours - legacy) > 0.03:
        failures.append(
            f"derived bolt grasp height {ours:.4f} disagrees with the original "
            f"{legacy:.4f} by more than 30 mm - check TCP_OFFSET_Z")

    for name in ("red_bin", "the red bin", "RED", "blue-bin", "blue"):
        got = find_receptacle(name)
        if got is None:
            failures.append(f"find_receptacle({name!r}) returned None")
    if find_receptacle("green_bin") is not None:
        failures.append("find_receptacle matched a bin that does not exist")

    # drop_point: every spot inside the bin, far enough apart that objects do
    # not stack, and cycling so an unobserved bin reset cannot run it off the end.
    import math as _math
    for r in RECEPTACLES.values():
        pts = [r.drop_point(i) for i in range(4)]
        for i, (px, py) in enumerate(pts):
            if (abs(px - r.x) > r.inner_half_width
                    or abs(py - r.y) > r.inner_half_width):
                failures.append(
                    f"{r.name} drop_point({i}) = ({px:+.3f},{py:+.3f}) is "
                    f"outside the bin")
            ok, why = is_reachable(px, py)
            if not ok:
                failures.append(f"{r.name} drop_point({i}) unreachable: {why}")
        sep = min(_math.dist(pts[i], pts[j])
                  for i in range(len(pts)) for j in range(i + 1, len(pts)))
        # A stacked object sits above the rim and the checker counts it as not
        # in the bin, so the spots must be further apart than an object is wide.
        if sep < MAX_GRASPABLE_WIDTH:
            failures.append(
                f"{r.name} drop points are only {sep * 1000:.0f} mm apart, "
                f"less than the {MAX_GRASPABLE_WIDTH * 1000:.0f} mm widest "
                f"object - they would stack")
        if r.drop_point(4) != r.drop_point(0):
            failures.append(f"{r.name} drop_point does not cycle")
        print(f"  {r.name:<9} drop spots {sep * 1000:.0f} mm apart, all inside "
              f"and reachable")

    # open_gap_for: the descent must clear the object on both sides. The bug
    # this replaces gave a 59 mm object 0.5 mm a side, and the fingers struck it
    # on the way down.
    print()
    for w in (0.024, 0.040, 0.059, MAX_GRASPABLE_WIDTH):
        gap = open_gap_for(w)
        side = (gap - w) / 2.0
        print(f"  object {w * 1000:4.0f} mm -> jaw {gap * 1000:4.0f} mm "
              f"({side * 1000:4.1f} mm a side)")
        if side <= 0.001:
            failures.append(
                f"a {w * 1000:.0f} mm object gets {side * 1000:.1f} mm of "
                f"clearance a side - the fingers would strike it")
        if gap > MAX_JAW_GAP:
            failures.append(f"open_gap_for({w}) = {gap} exceeds the jaw limit")

    print()
    if failures:
        print(f"FAILED ({len(failures)}):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("all scene self-tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())

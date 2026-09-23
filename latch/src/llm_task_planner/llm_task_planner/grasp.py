"""Grasp orientation: turning a detected object's yaw into a gripper pose.

--------------------------------------------------------------------------
WHY THIS MODULE EXISTS - the bug it replaces
--------------------------------------------------------------------------
arm_control_from_UI.py:115 built its orientation like this:

    self.move_to(x, y, z, 1.0, self.init_angle + theta, 0.0, 0.0)
                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                          (qx=1.0, qy=init_angle+theta, qz=0, qw=0)

with init_angle = -0.3825 and theta the OBB's long-axis angle.

That is not a unit quaternion, but normalising is not the real problem - the
direction survives normalisation. The real problem is what it *means*.

A quaternion with w = 0 is a 180 degree rotation about its axis. Here the axis
is (1, phi, 0) with phi = init_angle + theta. Rotating 180 degrees about an
axis lying in the XY plane maps +Z to -Z - so the gripper does point downward,
which is why this ever worked - and simultaneously yaws the hand by twice the
axis angle:

    hand_yaw = 2 * atan2(phi, 1) = 2 * atan(phi)

So the commanded yaw is a *nonlinear* function of the detected angle. Sweep an
object through 90 degrees and the wrist turns through 141:

    theta =  0.00 rad  ->  phi = -0.3825  ->  hand_yaw = -0.730 rad
    theta =  1.5708    ->  phi =  1.1883  ->  hand_yaw =  1.742 rad

d(hand_yaw)/d(theta) = 2 / (1 + phi^2), which equals 1 only at phi = +-1. The
mapping is therefore correct at two isolated angles and wrong everywhere else.
It survived in the original demo because bolts were detected within a narrow
angular band and a thin shaft tolerates a sloppy approach angle.

It will not survive the planner requesting varied poses across cubes, bolts and
cylinders, and the failures would look like planner or IK errors rather than a
bad quaternion. Hence: replace before WP5, not after.

--------------------------------------------------------------------------
WHAT REPLACES IT
--------------------------------------------------------------------------
An explicit rotation: roll pi (point the tool down), pitch 0, yaw = the angle
we actually want. No hidden doubling, no offsets that only cancel by accident.
"""

from __future__ import annotations

import math
from typing import Tuple

Quaternion = Tuple[float, float, float, float]  # (x, y, z, w)

#: Yaw to add to a desired finger direction to get the yaw to command on
#: panda_link8. Derived, not tuned:
#:
#:  1. We command panda_link8, because that is what the existing code passes as
#:     set_goal_state(pose_link="panda_link8").
#:
#:  2. For a top-down pose (roll = pi, pitch = 0, yaw = Y) the rotation matrix is
#:         Rz(Y) . Rx(pi) = [[ c,  s,  0],
#:                           [ s, -c,  0],
#:                           [ 0,  0, -1]]
#:     so the frame's Y axis - the second column, (s, -c, 0) - points along
#:     Y - pi/2 in the link0 plane, and its Z axis points straight down.
#:
#:  3. panda_hand is panda_link8 rotated -pi/4 about Z
#:     (panda_hand_joint, robot_description/urdf/panda.urdf:213). The tempting
#:     step is to subtract that -pi/4 as well, giving Y8 - 3pi/4. That is WRONG,
#:     and it is the trap in this derivation: the mount is applied *after* the
#:     pi roll, and conjugating a Z rotation by Rx(pi) reverses it,
#:         Rx(pi) . Rz(t) . Rx(pi)^-1 = Rz(-t)
#:     so
#:         R_hand = Rz(Y8) . Rx(pi) . Rz(-pi/4)
#:                = Rz(Y8 + pi/4) . Rx(pi)
#:     i.e. the hand behaves as a top-down frame yawed by Y8 + pi/4, and its Y
#:     axis points along  (Y8 + pi/4) - pi/2  =  Y8 - pi/4.
#:
#:  4. panda_finger_joint1 travels along the hand's +Y (axis xyz="0 1 0",
#:     panda.urdf:272), so hand-Y is the direction the jaw closes along, and it
#:     must lie along the object's SHORT axis - which is what grasp_yaw is.
#:
#:     Y8 - pi/4 = grasp_yaw   ->   Y8 = grasp_yaw + pi/4
#:
#: Test 5 in _self_test() propagates a commanded pose all the way through the
#: mount and the finger axis and checks where the jaw actually closes. That is
#: the only check here that catches a wrong sign - every other assertion passes
#: happily with the gripper rotated 90 degrees off.
HAND_YAW_OFFSET = math.pi / 4.0


def quaternion_from_euler(roll: float, pitch: float, yaw: float) -> Quaternion:
    """Intrinsic Z-Y-X (yaw-pitch-roll) euler angles to (x, y, z, w).

    Same convention as tf_transformations.quaternion_from_euler, reimplemented
    so the package does not need ros-jazzy-tf-transformations (which is not
    installed, and pulling it in would need root for six lines of algebra).
    """
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    return (
        sr * cp * cy - cr * sp * sy,   # x
        cr * sp * cy + sr * cp * sy,   # y
        cr * cp * sy - sr * sp * cy,   # z
        cr * cp * cy + sr * sp * sy,   # w
    )


def euler_from_quaternion(q: Quaternion) -> Tuple[float, float, float]:
    """Inverse of quaternion_from_euler. Used by the tests and for logging."""
    x, y, z, w = q
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2, sinp) if abs(sinp) >= 1 else math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def top_down_grasp_quaternion(yaw: float, apply_hand_offset: bool = True) -> Quaternion:
    """Orientation for a vertical, top-down grasp with the fingers at ``yaw``.

    ``yaw`` is in the panda_link0 frame and should be the object's *grasp* yaw -
    perpendicular to its OBB long axis, because a parallel jaw closes across the
    short dimension. yolov8_obb.projection.grasp_yaw() already does that
    conversion; DetectedObject.grasp_yaw carries the result.

    roll = pi points the tool's +Z (its approach axis) straight down at the
    table, which is the only approach this fixed overhead camera can support.

    The commanded link8 yaw is folded to [-pi/2, pi/2). Rotating the wrist by pi
    swaps which finger sits on which side of the object - the same grasp - so
    the fold is free, and it stops joint 7 taking a long way round for objects
    whose yaw happens to land near the wrap point.

    Pass apply_hand_offset=False only to command a raw link8 yaw directly; the
    tests use it to check the yaw gain in isolation.
    """
    if apply_hand_offset:
        yaw = wrap_to_half_pi(yaw + HAND_YAW_OFFSET)
    return quaternion_from_euler(math.pi, 0.0, yaw)


def wrap_to_half_pi(angle: float) -> float:
    """Fold an angle into [-pi/2, pi/2).

    A parallel jaw is symmetric: rotating the wrist by pi produces the same
    grasp. Folding into the half-open half-turn keeps joint 7 away from its
    limits and prevents the planner choosing a needlessly long wrist rotation.
    """
    return (angle + math.pi / 2.0) % math.pi - math.pi / 2.0


# --------------------------------------------------------------------------
# The legacy formula, kept ONLY so the tests can demonstrate the divergence.
# Do not call this from anything real.
# --------------------------------------------------------------------------

def _legacy_hand_yaw(theta: float, init_angle: float = -0.3825) -> float:
    """Effective hand yaw produced by the original (qx=1, qy=phi, qz=0, qw=0)."""
    return 2.0 * math.atan(init_angle + theta)


def _self_test() -> int:
    failures = []

    def check(label, got, want, tol=1e-9):
        if abs(got - want) > tol:
            failures.append(f"{label}: got {got!r}, want {want!r}")

    # 1. Round-trip through euler.
    for yaw in (0.0, 0.3, -1.1, 1.4):
        q = quaternion_from_euler(math.pi, 0.0, yaw)
        r, p, y = euler_from_quaternion(q)
        check(f"roll @{yaw}", abs(r), math.pi, 1e-9)
        check(f"pitch @{yaw}", p, 0.0, 1e-9)
        check(f"yaw @{yaw}", y, yaw, 1e-9)

    # 2. Unit norm - the thing the original was not.
    for yaw in (0.0, 0.7, -2.0):
        q = top_down_grasp_quaternion(yaw)
        n = math.sqrt(sum(c * c for c in q))
        check(f"norm @{yaw}", n, 1.0, 1e-12)

    # 3. The tool axis really does point down. Rotating +Z by the quaternion
    #    must give -Z, or the gripper approaches sideways.
    def rotate_z_axis(q):
        x, y, z, w = q
        return (2 * (x * z + w * y),
                2 * (y * z - w * x),
                1 - 2 * (x * x + y * y))
    for yaw in (0.0, 0.9, -1.3):
        v = rotate_z_axis(top_down_grasp_quaternion(yaw))
        check(f"approach x @{yaw}", v[0], 0.0, 1e-9)
        check(f"approach y @{yaw}", v[1], 0.0, 1e-9)
        check(f"approach z @{yaw}", v[2], -1.0, 1e-9)

    # 4. Yaw is linear in the request - one unit in, one unit out.
    a = euler_from_quaternion(top_down_grasp_quaternion(0.0, False))[2]
    b = euler_from_quaternion(top_down_grasp_quaternion(1.0, False))[2]
    check("yaw gain", b - a, 1.0, 1e-9)

    # 5. The one that matters: does the JAW actually close along the requested
    #    direction? Propagate the commanded link8 pose through the hand mount
    #    and the finger axis, exactly as the robot does. An error in the sign or
    #    size of HAND_YAW_OFFSET shows up here as a 45 or 90 degree miss, where
    #    every check above still passes.
    def quat_to_matrix(q):
        x, y, z, w = q
        return (
            (1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
            (2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
            (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)),
        )

    # panda_hand = panda_link8 . Rz(-pi/4); the finger axis is the hand's +Y,
    # so in link8 coordinates it is Rz(-pi/4) . (0,1,0).
    t = -math.pi / 4.0
    finger_in_link8 = (-math.sin(t), math.cos(t), 0.0)

    for requested in (0.0, 0.35, -0.62, 1.31, -1.45):
        R = quat_to_matrix(top_down_grasp_quaternion(requested))
        fx = sum(R[0][k] * finger_in_link8[k] for k in range(3))
        fy = sum(R[1][k] * finger_in_link8[k] for k in range(3))
        fz = sum(R[2][k] * finger_in_link8[k] for k in range(3))
        # The jaw must close horizontally - any vertical component means the
        # fingers would drive into or away from the table.
        check(f"finger axis horizontal @{requested}", fz, 0.0, 1e-9)
        # Direction is modulo pi: closing along +d and -d is the same grasp.
        err = wrap_to_half_pi(math.atan2(fy, fx) - requested)
        check(f"jaw closes along requested dir @{requested}", err, 0.0, 1e-9)

    # 6. Quantify how wrong the legacy formula was, against the correctly
    #    derived command for the same detection, so any regression is loud.
    #    theta is the detected LONG-axis yaw, so the right link8 command is
    #    theta + pi/2 (turn to the short axis) + pi/4 (the mount).
    print("  long-axis   legacy cmd    correct cmd    error")
    worst = 0.0
    for theta in (0.0, 0.4, 0.8, 1.2, 1.5708):
        leg = wrap_to_half_pi(_legacy_hand_yaw(theta))
        cor = wrap_to_half_pi(theta + math.pi / 2 + HAND_YAW_OFFSET)
        err = abs(wrap_to_half_pi(leg - cor))
        worst = max(worst, err)
        print(f"  {theta:8.4f}   {leg:+9.4f}     {cor:+9.4f}    {math.degrees(err):6.1f} deg")
    print(f"\n  worst legacy grasp-angle error: {math.degrees(worst):.1f} deg")
    print("  (near-zero at theta=0, which is why bolts in a narrow angular")
    print("   band appeared to work, and why this was never noticed)")
    if worst < math.radians(20):
        failures.append("legacy formula looks accurate - the analysis is wrong somewhere")

    print()
    if failures:
        print(f"FAILED ({len(failures)}):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("all grasp self-tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())

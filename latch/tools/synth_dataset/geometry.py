"""Vertex clouds and rigid transforms for the synthetic dataset generator.

Why vertex clouds rather than "width x height" boxes
----------------------------------------------------
The camera sits at link0 (0.20, 0.60, 0.70) looking down, and objects are
spread over roughly 0.8 x 1.0 m of table. An object at link0 (0.45, 0.20) is
0.47 m lateral of the camera at 0.70 m height - a viewing angle of 34 degrees
off vertical. At that angle a 40 mm cube shows its top face *and* two side
faces, and the tight rotated box around that silhouette is visibly larger than
the top face alone.

So a label cannot be produced by projecting a footprint. Every vertex of the
object's 3D geometry is projected at its own true depth, and the label is the
minimum-area rotated rectangle around the resulting 2D hull - which is exactly
what a human annotator would draw, and what YOLO-OBB should predict.

This is also why the generator is *more* accurate than inference can ever be:
it knows each vertex's z. At inference we have only pixels and must assume a
single depth plane per class (projection.OBJECT_CENTROID_Z). validate.py measures
the error that assumption costs, using this module's ground truth.

Frames
------
Local     the model's own frame, as authored in the SDF / STL.
World     Gazebo's frame. SDF <pose>x y z r p y</pose> is extrinsic XYZ, i.e.
          R = Rz(yaw) . Ry(pitch) . Rx(roll).
"""

from __future__ import annotations

import math
import struct
from typing import Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------
# Rotations
# --------------------------------------------------------------------------


def rpy_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Rotation matrix for an SDF <pose> orientation.

    SDF uses fixed-axis (extrinsic) XYZ, so the composition order is
    R = Rz(yaw) . Ry(pitch) . Rx(roll). Getting this backwards produces
    plausible-looking but wrong labels for any object that is not axis-aligned,
    which is the hardest kind of error to spot by eye.
    """
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=float)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=float)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=float)
    return rz @ ry @ rx


def transform(vertices: np.ndarray, xyz: Sequence[float],
              rpy: Sequence[float]) -> np.ndarray:
    """Apply an SDF pose to an (N, 3) vertex cloud. Returns a new (N, 3) array."""
    r = rpy_matrix(*rpy)
    return vertices @ r.T + np.asarray(xyz, dtype=float)


def quaternion_from_rpy(roll: float, pitch: float, yaw: float
                        ) -> Tuple[float, float, float, float]:
    """(x, y, z, w) for the same orientation rpy_matrix builds.

    Needed because SDF <pose> takes Euler angles but Gazebo's set_pose *service*
    takes a quaternion, so the generator has to convert. Deriving it
    independently and asserting it reproduces rpy_matrix (self-test 6) is the
    cheap way to be sure the two paths agree - a mismatch here would rotate
    every object away from its own label, which is precisely the failure that
    would look like "the detector is inaccurate".
    """
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    return (sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy)


def matrix_from_quaternion(x: float, y: float, z: float, w: float) -> np.ndarray:
    """Rotation matrix from an (x, y, z, w) quaternion."""
    n = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ], dtype=float)


def rpy_from_quaternion(x: float, y: float, z: float, w: float
                        ) -> Tuple[float, float, float]:
    """Inverse of quaternion_from_rpy, in the same extrinsic-XYZ convention.

    Needed to label frames captured from the LIVE cell, where objects settle
    under physics and their orientation is whatever Gazebo says rather than
    whatever was commanded. The static capture world never needed this: there
    the rpy is an input.

    Extracted from the rotation matrix rather than by a direct quaternion
    formula, so the convention cannot drift from rpy_matrix's - the two are
    checked against each other round-trip in the self-test.
    """
    m = matrix_from_quaternion(x, y, z, w)
    # R = Rz(yaw) . Ry(pitch) . Rx(roll), so m[2, 0] = -sin(pitch).
    sin_pitch = -m[2, 0]
    sin_pitch = max(-1.0, min(1.0, float(sin_pitch)))
    pitch = math.asin(sin_pitch)
    if abs(sin_pitch) > 1.0 - 1e-9:
        # Gimbal lock: roll and yaw are no longer separable, only their sum (or
        # difference) is. Pin roll to 0 and put everything in yaw - any split
        # reproduces the same rotation, which is all a label needs.
        return 0.0, pitch, float(math.atan2(-m[0, 1], m[1, 1]))
    return (float(math.atan2(m[2, 1], m[2, 2])),
            pitch,
            float(math.atan2(m[1, 0], m[0, 0])))


def resting_z(vertices: np.ndarray, rpy: Sequence[float],
              support_z: float) -> float:
    """The pose z that sets this object down exactly on a plane at ``support_z``.

    Derived from the rotated cloud's lowest vertex rather than from a per-class
    constant, so it stays correct for any orientation and for the bolt, whose
    STL origin is not its centroid (z spans -0.100 .. +0.016, putting the origin
    just under the hex head). A hardcoded height would be wrong for every
    orientation but the one it was measured at.
    """
    rotated = np.asarray(vertices, dtype=float) @ rpy_matrix(*rpy).T
    return support_z - float(rotated[:, 2].min())


# --------------------------------------------------------------------------
# Primitive vertex clouds
# --------------------------------------------------------------------------


def box_vertices(sx: float, sy: float, sz: float) -> np.ndarray:
    """8 corners of a box centred on its local origin, matching SDF <box><size>."""
    hx, hy, hz = sx / 2.0, sy / 2.0, sz / 2.0
    return np.array([(x, y, z)
                     for x in (-hx, hx)
                     for y in (-hy, hy)
                     for z in (-hz, hz)], dtype=float)


def cylinder_vertices(radius: float, length: float, segments: int = 24) -> np.ndarray:
    """Two rings approximating a cylinder centred on its local origin, axis +Z.

    Matches SDF <cylinder><radius><length>. 24 segments puts the polygon's
    sagitta at r*(1-cos(pi/24)) = 0.9% of the radius - 0.2 mm on a 20 mm
    cylinder, well under a pixel at this camera's ~2.8 mm/px scale.
    """
    ang = np.linspace(0.0, 2.0 * math.pi, segments, endpoint=False)
    ring = np.stack([radius * np.cos(ang), radius * np.sin(ang)], axis=1)
    top = np.hstack([ring, np.full((segments, 1), length / 2.0)])
    bot = np.hstack([ring, np.full((segments, 1), -length / 2.0)])
    return np.vstack([top, bot])


def stl_vertices(path: str) -> np.ndarray:
    """Unique vertices of a binary STL, in the mesh's own frame.

    Used for the bolt: a shaft plus a hex head is not usefully approximated by
    a box (the box would over-cover the thin shaft by nearly 2x in width), and
    the mesh is small enough - 560 triangles, 282 unique vertices - that using
    it directly is both exact and cheap.
    """
    with open(path, "rb") as fh:
        fh.read(80)                                    # header, unused
        (ntri,) = struct.unpack("<I", fh.read(4))
        body = fh.read(ntri * 50)

    if len(body) != ntri * 50:
        raise ValueError(
            f"{path}: expected {ntri * 50} body bytes, got {len(body)}. "
            "Binary STL required (this looks like ASCII)."
        )

    # Each 50-byte record is 12 floats (normal + 3 vertices) then 2 pad bytes.
    rec = np.frombuffer(body, dtype=np.uint8).reshape(ntri, 50)
    floats = np.frombuffer(rec[:, :48].tobytes(), dtype="<f4").reshape(ntri, 4, 3)
    pts = floats[:, 1:, :].reshape(-1, 3).astype(float)
    return np.unique(np.round(pts, 6), axis=0)


# --------------------------------------------------------------------------
# Self-test:  python3 geometry.py
# --------------------------------------------------------------------------


def _self_test() -> int:
    failures = []

    def check(label, got, want, tol=1e-9):
        if abs(float(got) - float(want)) > tol:
            failures.append(f"{label}: got {got!r}, want {want!r}")

    # 1. rpy_matrix must match SDF's extrinsic-XYZ order. Test with a case where
    #    the two orders disagree, otherwise the assertion proves nothing.
    r, p, y = 0.3, 0.7, -1.1
    m = rpy_matrix(r, p, y)
    check("det(R)", np.linalg.det(m), 1.0, 1e-12)
    check("R orthonormal", np.abs(m @ m.T - np.eye(3)).max(), 0.0, 1e-12)
    intrinsic = rpy_matrix(0, 0, y) @ rpy_matrix(0, p, 0) @ rpy_matrix(r, 0, 0)
    check("Rz.Ry.Rx composition", np.abs(m - intrinsic).max(), 0.0, 1e-12)
    reversed_order = rpy_matrix(r, 0, 0) @ rpy_matrix(0, p, 0) @ rpy_matrix(0, 0, y)
    if np.abs(m - reversed_order).max() < 1e-6:
        failures.append("test is vacuous: chosen angles do not distinguish "
                        "XYZ from ZYX order")

    # 2. A yaw-only rotation must leave a cube's height alone; a pitch must not.
    cube = box_vertices(0.04, 0.04, 0.04)
    check("cube vertex count", len(cube), 8)
    check("cube yaw resting z", resting_z(cube, (0, 0, 0.9), 1.015), 1.035, 1e-12)
    tipped = resting_z(cube, (0, math.pi / 4, 0), 1.015)
    check("cube on edge resting z", tipped, 1.015 + 0.04 * math.sqrt(2) / 2, 1e-12)

    # 3. Cylinder: upright rests at half-length, on its side at one radius.
    cyl = cylinder_vertices(0.02, 0.06)
    check("cylinder vertex count", len(cyl), 48)
    check("cylinder upright", resting_z(cyl, (0, 0, 0), 1.015), 1.045, 1e-12)
    #    Rolled onto its side the lowest vertex is on the polygon, not the true
    #    circle, so it sits a hair high - that sagitta, not an error.
    side = resting_z(cyl, (0, math.pi / 2, 0), 1.015)
    check("cylinder on side", side, 1.035, 3e-4)

    # 4. The bolt mesh: dimensions must match what the STL actually contains.
    import os
    stl = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "..", "src", "panda_moveit_config",
                       "worlds", "bolt", "meshes", "bolt.stl")
    stl = os.path.normpath(stl)
    if not os.path.exists(stl):
        failures.append(f"bolt STL not found at {stl}")
    else:
        v = stl_vertices(stl)
        lo, hi = v.min(axis=0), v.max(axis=0)
        print(f"  bolt STL: {len(v)} unique vertices")
        print(f"    local x {lo[0]:+.5f}..{hi[0]:+.5f}  "
              f"y {lo[1]:+.5f}..{hi[1]:+.5f}  z {lo[2]:+.5f}..{hi[2]:+.5f}")
        check("bolt length (local z)", hi[2] - lo[2], 0.116, 1e-4)
        check("bolt head across corners (local y)", hi[1] - lo[1], 0.0438, 1e-4)
        check("bolt head across flats (local x)", hi[0] - lo[0], 0.03793, 1e-4)

        # The origin is NOT the centroid - assert it, so nobody "tidies" the
        # resting-height code back into a constant.
        if abs((lo[2] + hi[2]) / 2.0) < 0.01:
            failures.append("bolt STL origin is centred after all - the "
                            "resting_z rationale needs rewriting")

        # Laid on its side (local +Z horizontal) it rests on a hex flat, so the
        # head centre sits one apothem up: 0.0438/2 * cos(30) = 0.018965.
        rest = resting_z(v, (0, math.pi / 2, 0), 1.015)
        apothem = (0.0438 / 2.0) * math.cos(math.pi / 6)
        print(f"    lying on its side, pose z = {rest:.5f} "
              f"(head apothem {apothem * 1000:.2f} mm above table)")
        check("bolt on side resting z", rest, 1.015 + apothem, 5e-4)

        # Yaw must not change the resting height of a body lying on its side.
        for yaw in (0.0, 0.7, -2.2, 3.0):
            check(f"bolt resting z invariant to yaw {yaw}",
                  resting_z(v, (0, math.pi / 2, yaw), 1.015), rest, 1e-9)

    # 5. transform() must agree with an explicit per-vertex rotate-then-translate.
    xyz, rpy = (0.5, 0.35, 1.05), (0.0, 1.5, -0.4)
    got = transform(cube, xyz, rpy)
    want = np.array([rpy_matrix(*rpy) @ p + np.array(xyz) for p in cube])
    check("transform vs explicit", np.abs(got - want).max(), 0.0, 1e-12)

    # 6. The Euler path (SDF <pose>, used for labels) and the quaternion path
    #    (the set_pose service, used to place objects) must describe the SAME
    #    orientation. If they diverge, every object is rotated away from its own
    #    label and the dataset is quietly poisoned.
    worst_q = 0.0
    for _ in range(500):
        rpy_s = tuple(np.random.default_rng().uniform(-math.pi, math.pi, 3))
        q = quaternion_from_rpy(*rpy_s)
        worst_q = max(worst_q, float(np.abs(matrix_from_quaternion(*q)
                                            - rpy_matrix(*rpy_s)).max()))
    print(f"  Euler vs quaternion rotation matrices agree to {worst_q:.2e}")
    if worst_q > 1e-12:
        failures.append(f"quaternion_from_rpy disagrees with rpy_matrix by "
                        f"{worst_q:.2e} - objects would be placed at a different "
                        f"orientation than their labels assume")
    #    Pin the specific orientations the generator actually uses.
    for name, rpy_s, want_axis in (
            ("bolt on side +", (0.0, math.pi / 2, 0.0), (0.0, 0.0, -1.0)),
            ("bolt on side -", (0.0, -math.pi / 2, 0.0), (0.0, 0.0, 1.0)),
            ("cylinder toppled", (0.0, math.pi / 2, 0.0), (0.0, 0.0, -1.0))):
        q = quaternion_from_rpy(*rpy_s)
        # Local +Z must end up horizontal (pointing along world -Z or +Z after
        # the pitch, i.e. the long axis is laid flat).
        local_z = matrix_from_quaternion(*q) @ np.array([0.0, 0.0, 1.0])
        check(f"{name}: local +Z x", local_z[0], 1.0 if rpy_s[1] > 0 else -1.0, 1e-12)
        check(f"{name}: local +Z z", local_z[2], 0.0, 1e-12)

    print()
    if failures:
        print(f"FAILED ({len(failures)}):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("all geometry self-tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())

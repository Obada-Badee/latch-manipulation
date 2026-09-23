"""Camera geometry for the moveit2_obb scene: pixels <-> robot coordinates.

This module is the single source of truth for the projection math. It was
previously inlined in UI/bolt_selector.py as bare floats; the dataset generator
(tools/synth_dataset) needs the *inverse* of the same model, so a second copy
would guarantee the two drift apart.

--------------------------------------------------------------------------
FRAMES
--------------------------------------------------------------------------
world       Gazebo's frame. Table top at z = 1.015.
panda_link0 Robot base. Sits at world (0.05, 0.00, 1.02), no rotation.
            +X forward (away from the robot), +Y left, +Z up.
            *** All planner-facing coordinates are in this frame. ***
            "left" in a user command means +Y. Stated here, in the world SDF
            header, and in the planner system prompt - keep all three in sync.
camera_link Fixed to panda_link0 at (0.20, 0.60, 0.70), rpy (0, pi/2, 0).
            The pi/2 pitch turns camera +X (its optical axis, per Gazebo's
            camera_link convention) to point straight down.

--------------------------------------------------------------------------
THE PINHOLE MODEL, DERIVED
--------------------------------------------------------------------------
Gazebo's camera declares a horizontal FOV, not a focal length
(robot_description/urdf/camera/camera.gazebo):

    hfov = 1.8 rad,  640 x 480

    fx = (width / 2) / tan(hfov / 2) = 320 / tan(0.9) = 253.9364
    fy = fx                          (Gazebo renders square pixels)
    cx, cy = width / 2, height / 2 = 320, 240

Going from camera_link (X fwd, Y left, Z up) to the optical convention
(Z fwd, X right, Y down) gives X_opt = -Y_cam, Y_opt = -Z_cam, Z_opt = X_cam,
so for a point at depth d along the optical axis:

    u - cx = fx * (-Y_cam / d)      ->   Y_cam = -(u - cx) * d / fx
    v - cy = fy * (-Z_cam / d)      ->   Z_cam = -(v - cy) * d / fy

Rotating camera_link into link0 by R_y(pi/2) maps (a, b, c) -> (c, b, -a), so

    x_link0 = 0.20 - (v - cy) * d / fy
    y_link0 = 0.60 - (u - cx) * d / fx

which is exactly what UI/bolt_selector.py:104-105 computed. That code is
correct; it just hardcoded d = 0.70 (see below) and hid the derivation.

--------------------------------------------------------------------------
THE DEPTH CORRECTION (a real bug in the original)
--------------------------------------------------------------------------
The old code used a constant d = 0.70, which is the camera's height above the
*link0 origin plane* - not above the objects. Objects sit on the table 1-6 cm
higher, so the true depth is 1-6 cm less.

Depth enters the result linearly, so a 3.5 cm error is a 5% radial error:
~1.6 cm at the edge of the frame. On a 40 mm cube gripped by a 60 mm jaw
(10 mm clearance per side) that is the difference between a grasp and a miss,
and it would present as "the IK is off", which is a miserable thing to chase.

--------------------------------------------------------------------------
WHICH PLANE? THE CENTROID, NOT THE TOP FACE
--------------------------------------------------------------------------
The obvious choice is the object's top face, on the reasoning that the camera
looks down and sees the top. That is wrong, and it was wrong here until it was
measured against ground truth (tools/synth_dataset/labeler.py).

The detector's OBB does not enclose the top face. It encloses the whole
*silhouette*, and this camera is not overhead: it sits at link0 (0.20, 0.60)
while objects spread across ~0.8 x 1.0 m of table, so an object at
link0 (0.45, 0.20) is viewed 34 degrees off vertical and shows two side faces
as well as its top. Measured: a 40 mm cube projects 1.54x larger there than
directly beneath the camera.

Because a box is centrally symmetric, the centre of the box enclosing that
silhouette projects from close to the object's 3D *centroid*. Fitting the
best plane over 500 randomised poses per class and orientation agrees with the
centroid height to within 2.75 mm in every case - so this is derivable rather
than tuned. Worst-case back-projection error, top-face plane vs centroid plane:

    class      top-face plane      centroid plane     jaw clearance
    cube          20.9 mm              1.5 mm            10 mm
    cylinder      41.8 mm              6.3 mm            10 mm
    bolt           7.2 mm              6.1 mm            18 mm

The top-face assumption therefore guaranteed a miss on cylinders and was
marginal on cubes. It happened to be survivable for bolts, which is very
probably why the original single-class system appeared to work: a bolt lying on
its side is squat, so its top face and its centroid are only 19 mm apart, and a
24 mm shaft in a 60 mm jaw forgives 18 mm of error.

OBJECT_CENTROID_Z holds the per-class planes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

# --------------------------------------------------------------------------
# Scene constants. Mirrors of values that live in XML - if you change one of
# the files named below, change it here too. The self-test at the bottom of
# this file re-derives what it can, but it cannot read the SDF.
# --------------------------------------------------------------------------

#: Robot base position in world coords. From launch/moveit_gazebo_obb.py
#: (`-x 0.05 -y 0.0 -z 1.02`).
LINK0_IN_WORLD: Tuple[float, float, float] = (0.05, 0.00, 1.02)

#: Table top, world z. From worlds/arm_on_the_table.sdf.
TABLE_TOP_WORLD_Z = 1.015

#: Table top expressed in link0 - very slightly *below* the base origin.
TABLE_TOP_Z = TABLE_TOP_WORLD_Z - LINK0_IN_WORLD[2]  # -0.005

#: Depth plane to back-project each class against: the height of the object's
#: geometric centroid above link0's XY plane. See the module docstring for why
#: this is the centroid and not the top face. Each value is derived from the
#: world SDF geometry, then confirmed by fitting against ground truth.
#:
#:   cube      40 mm on a face; centroid one half-height up
#:             -0.005 + 0.020 = 0.015                    (fit: 0.0165)
#:
#:   cylinder  bimodal, and this is the one genuine compromise here.
#:             upright  60 mm tall  -> -0.005 + 0.030 = 0.025   (fit: 0.0278)
#:             toppled  40 mm dia   -> -0.005 + 0.020 = 0.015   (fit: 0.0158)
#:             No single plane serves both, so this is their exact midpoint,
#:             0.020, which splits the error evenly between the two states.
#:             (Fitting the whole population gives 0.0215, but that number is
#:             just an artefact of how often the sampler topples one - the
#:             midpoint is the honest choice.)
#:             Do NOT try to pick the mode from the OBB's aspect ratio - that
#:             was measured and does not work. Perspective stretches an upright
#:             cylinder's silhouette radially, so upright aspect ratios span
#:             1.00-2.57 against toppled's 1.11-2.56: the ranges overlap almost
#:             completely and no threshold separates them.
#:
#:   bolt      lies on its side, 116 mm long. Head apothem 18.97 mm, shaft
#:             radius 12 mm, so the centroid of the silhouette sits ~19 mm up
#:             -> -0.005 + 0.019 = 0.014                 (fit: 0.0145/0.0158)
OBJECT_CENTROID_Z = {
    "cube": 0.015,
    "cylinder": 0.020,
    "bolt": 0.014,
}

#: Used when a detection's class is unknown - which should not happen, since the
#: detector emits only the three names above. The cube's plane is the sensible
#: fallback: it is the modal object on this table, and a cube-scale guess is at
#: worst ~7 mm out for anything else here. The table plane, the previous
#: fallback, is systematically 20 mm too low now that the semantics are
#: "centroid" rather than "top face".
DEFAULT_CENTROID_Z = 0.015


@dataclass(frozen=True)
class CameraModel:
    """Pinhole model for a downward-looking camera rigidly fixed to link0.

    Constructed from the *declared* sensor parameters rather than from
    precomputed focal lengths, so editing camera.gazebo cannot silently
    desynchronise this from the simulator.
    """

    width: int = 640
    height: int = 480
    hfov: float = 1.8
    #: camera_link origin expressed in link0. From config/panda.urdf.xacro:13.
    origin: Tuple[float, float, float] = (0.20, 0.60, 0.70)

    @property
    def fx(self) -> float:
        return (self.width / 2.0) / math.tan(self.hfov / 2.0)

    @property
    def fy(self) -> float:
        # Square pixels: Gazebo derives vertical FOV from the aspect ratio.
        return self.fx

    @property
    def cx(self) -> float:
        return self.width / 2.0

    @property
    def cy(self) -> float:
        return self.height / 2.0

    @property
    def vfov(self) -> float:
        return 2.0 * math.atan((self.height / 2.0) / self.fy)

    # ---------------------------------------------------------------- forward

    def depth_to(self, plane_z: float) -> float:
        """Optical-axis distance from the camera to a horizontal plane at ``plane_z``."""
        d = self.origin[2] - plane_z
        if d <= 0.0:
            raise ValueError(
                f"plane at z={plane_z} is at or above the camera (z={self.origin[2]})"
            )
        return d

    def pixel_to_link0(self, u: float, v: float, plane_z: float = DEFAULT_CENTROID_Z
                       ) -> Tuple[float, float]:
        """Back-project a pixel onto the horizontal plane ``z = plane_z``.

        Returns (x, y) in link0. This is a plane intersection, not a true 3D
        unprojection - a single camera cannot recover depth, so the object's
        height has to be supplied. That is the fixed-depth assumption the whole
        pipeline rests on, and it is why stacking objects is out of scope.
        """
        d = self.depth_to(plane_z)
        x = self.origin[0] - (v - self.cy) * d / self.fy
        y = self.origin[1] - (u - self.cx) * d / self.fx
        return x, y

    # ---------------------------------------------------------------- inverse

    def link0_to_pixel(self, x: float, y: float, z: float) -> Tuple[float, float]:
        """Project a 3D link0 point into the image. Inverse of pixel_to_link0.

        This is what the synthetic dataset generator needs: it knows every
        object's ground-truth pose and has to emit the pixel corners YOLO
        should have predicted.
        """
        d = self.depth_to(z)
        u = self.cx - (y - self.origin[1]) * self.fx / d
        v = self.cy - (x - self.origin[0]) * self.fy / d
        return u, v

    def world_to_pixel(self, wx: float, wy: float, wz: float) -> Tuple[float, float]:
        """Project a *world* point (what Gazebo's PosePublisher emits)."""
        return self.link0_to_pixel(*world_to_link0(wx, wy, wz))

    # ----------------------------------------------------------------- extras

    def in_frame(self, u: float, v: float, margin: float = 0.0) -> bool:
        """Is this pixel inside the image? ``margin`` shrinks the accepted region.

        The dataset generator uses a positive margin to drop objects clipped by
        the frame edge, whose OBB corners would otherwise be a lie.
        """
        return (margin <= u <= self.width - margin
                and margin <= v <= self.height - margin)

    def footprint_link0(self, plane_z: float = DEFAULT_CENTROID_Z) -> Tuple[float, float, float, float]:
        """Region of the plane ``z = plane_z`` the camera can see: (x_min, x_max, y_min, y_max).

        Use this to check that a scene edit keeps its objects visible.
        """
        corners = [self.pixel_to_link0(u, v, plane_z)
                   for u, v in ((0, 0), (self.width, 0),
                                (0, self.height), (self.width, self.height))]
        xs = [c[0] for c in corners]
        ys = [c[1] for c in corners]
        return min(xs), max(xs), min(ys), max(ys)


#: The scene's camera. Import this rather than constructing your own.
CAMERA = CameraModel()


# --------------------------------------------------------------------------
# world <-> link0
# --------------------------------------------------------------------------

def world_to_link0(wx: float, wy: float, wz: float) -> Tuple[float, float, float]:
    """Pure translation - the robot is spawned axis-aligned."""
    return (wx - LINK0_IN_WORLD[0],
            wy - LINK0_IN_WORLD[1],
            wz - LINK0_IN_WORLD[2])


def link0_to_world(x: float, y: float, z: float) -> Tuple[float, float, float]:
    return (x + LINK0_IN_WORLD[0],
            y + LINK0_IN_WORLD[1],
            z + LINK0_IN_WORLD[2])


# --------------------------------------------------------------------------
# Oriented bounding boxes
# --------------------------------------------------------------------------

Corners = Sequence[Sequence[float]]  # 4 x (u, v), YOLO's xyxyxyxy order


def obb_center(corners: Corners) -> Tuple[float, float]:
    """Centroid of the four OBB corners, in pixels."""
    return (sum(c[0] for c in corners) / 4.0,
            sum(c[1] for c in corners) / 4.0)


def _edge_len(a: Sequence[float], b: Sequence[float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def obb_axes_px(corners: Corners) -> Tuple[float, float]:
    """Return (long_side, short_side) lengths in pixels.

    Adjacent edges of an OBB are perpendicular, so comparing edge 0-1 against
    edge 1-2 is enough to identify which is the long axis.
    """
    d01 = _edge_len(corners[0], corners[1])
    d12 = _edge_len(corners[1], corners[2])
    return (d01, d12) if d01 >= d12 else (d12, d01)


def obb_long_axis_yaw(corners: Corners) -> float:
    """Yaw of the OBB's long axis, in the link0 frame, radians.

    Ported from UI/bolt_selector.py:106-122, whose `pi/2 - angle` is correct but
    unexplained. The reason it works: with fx == fy the image->link0 mapping is
    a uniform scale plus the axis swap (du -> -y, dv -> -x), so an image
    direction (du, dv) becomes the link0 direction (-dv, -du), and

        atan2(-du, -dv) == atan2(du, dv) (mod pi) == pi/2 - atan2(dv, du)

    The result is wrapped to [-pi/2, pi/2): an OBB axis has no head or tail, so
    yaw and yaw+pi describe the same box, and the narrower range keeps the wrist
    away from its joint limits.
    """
    d01 = _edge_len(corners[0], corners[1])
    d12 = _edge_len(corners[1], corners[2])
    a, b = (corners[0], corners[1]) if d01 >= d12 else (corners[1], corners[2])
    du = a[0] - b[0]
    dv = a[1] - b[1]
    return wrap_to_half_pi(math.atan2(du, dv))


def grasp_yaw(corners: Corners) -> float:
    """Yaw the gripper should adopt to grasp this object, radians, link0 frame.

    The affordance: a parallel jaw must close across the object's *short* axis,
    so the finger travel direction - the hand's Y axis - aligns with the short
    axis, i.e. perpendicular to the long axis returned above.

    For a square OBB (a cube seen face-on) the two axes are the same length and
    the choice is arbitrary but harmless; either grasp works.
    """
    return wrap_to_half_pi(obb_long_axis_yaw(corners) + math.pi / 2.0)


def wrap_to_half_pi(angle: float) -> float:
    """Fold an angle into [-pi/2, pi/2). Valid only for axes, not directions."""
    return (angle + math.pi / 2.0) % math.pi - math.pi / 2.0


# --------------------------------------------------------------------------
# Bolts: the grasp point is not the box centre
# --------------------------------------------------------------------------
#
# A parallel jaw has to close around the bolt's SHAFT. The oriented box encloses
# a 43.8 mm hex head as well, so its centre sits off the shaft - measured at
# 34.2 mm away (sd 1.2 mm over 825 instances, tools/synth_dataset/
# bolt_grasp_offset.py). Every bolt grasp in the first benchmark missed for this
# reason: the bolt never moved at all, because the fingers closed ~8 mm clear of
# a 24 mm shaft.
#
# Cubes and cylinders are centrally symmetric, so the box centre lands on them
# and the error only ever showed up on bolts.

#: Distance from the box centre to the shaft centre, along the box's long axis.
BOLT_GRASP_SHIFT_M = 0.0342

#: The table renders near-uniform light grey; anything this much darker inside
#: the box is object. Generous on purpose - the bolt is dark metal and its
#: shadow darker still, and shadow follows the object, so it does not mislead.
_TABLE_LEVEL = 216
_DARK_MARGIN = 25
_PROFILE_BINS = 12


def pixel_long_axis(corners: Corners) -> float:
    """Angle of the box's long side in IMAGE coordinates, radians.

    Distinct from `obb_long_axis_yaw`, which answers the same question in the
    **link0** frame. Both are needed and they must not be interchanged: the two
    frames differ by an axis swap, so using one where the other belongs yields a
    plausible wrong answer rather than an obvious one. Doing exactly that made
    the head-detection test below read 49.2% - chance - which looked like "the
    cue is not there" instead of "the frames are mixed".
    """
    best, best_len = 0.0, -1.0
    for i in range(4):
        du = corners[(i + 1) % 4][0] - corners[i][0]
        dv = corners[(i + 1) % 4][1] - corners[i][1]
        length = math.hypot(du, dv)
        if length > best_len:
            best, best_len = math.atan2(dv, du), length
    return best


def head_end_sign(image, corners: Corners) -> int:
    """+1 if the hex head lies toward +pixel_long_axis, -1 if away, 0 if unclear.

    The head is 43.8 mm across against the shaft's 24 mm, so the silhouette is
    simply wider at that end. Measured accuracy on the live-cell dataset:
    **810/825 = 98.2%**.

    Returning 0 rather than guessing when the image is missing or the mask is
    too small matters, because the caller must then leave the point alone. A
    wrong sign does not merely fail to help - it moves the grasp 34 mm the wrong
    way, roughly doubling the error. Silence is strictly better than a coin flip.
    """
    if image is None:
        return 0
    import numpy as _np                                          # noqa: PLC0415

    pts = _np.asarray(corners, dtype=float)
    gray = image if image.ndim == 2 else image[:, :, :3].mean(axis=2)
    h, w = gray.shape[:2]

    x0 = max(int(pts[:, 0].min()) - 2, 0)
    y0 = max(int(pts[:, 1].min()) - 2, 0)
    x1 = min(int(pts[:, 0].max()) + 2, w - 1)
    y1 = min(int(pts[:, 1].max()) + 2, h - 1)
    if x1 <= x0 or y1 <= y0:
        return 0

    angle = pixel_long_axis(corners)
    cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
    ys, xs = _np.mgrid[y0:y1 + 1, x0:x1 + 1]
    dark = gray[y0:y1 + 1, x0:x1 + 1] < (_TABLE_LEVEL - _DARK_MARGIN)

    du = (xs - cx) * math.cos(angle) + (ys - cy) * math.sin(angle)
    dv = -(xs - cx) * math.sin(angle) + (ys - cy) * math.cos(angle)
    cu = (pts[:, 0] - cx) * math.cos(angle) + (pts[:, 1] - cy) * math.sin(angle)
    cv = -(pts[:, 0] - cx) * math.sin(angle) + (pts[:, 1] - cy) * math.cos(angle)
    hu, hv = float(_np.abs(cu).max()), float(_np.abs(cv).max())

    inside = dark & (_np.abs(du) <= hu) & (_np.abs(dv) <= hv)
    if int(inside.sum()) < 20:
        return 0

    du_in, dv_in = du[inside], dv[inside]
    edges = _np.linspace(-hu, hu, _PROFILE_BINS + 1)
    widths = []
    for i in range(_PROFILE_BINS):
        sel = (du_in >= edges[i]) & (du_in < edges[i + 1])
        widths.append(float(dv_in[sel].max() - dv_in[sel].min())
                      if int(sel.sum()) >= 3 else 0.0)
    half = _PROFILE_BINS // 2
    w_neg = float(_np.mean(widths[:half]))
    w_pos = float(_np.mean(widths[half:]))
    if w_neg == w_pos:
        return 0
    return 1 if w_pos > w_neg else -1


def bolt_grasp_pixel(image, corners: Corners, plane_z: float
                     ) -> Optional[Tuple[float, float]]:
    """Pixel where the jaw should close on a bolt, or None if undetermined.

    The shift is applied in pixel space and back-projected once by the caller,
    rather than converting a link0 offset into the image. One frame, one
    conversion, one place to be wrong.
    """
    sign = head_end_sign(image, corners)
    if sign == 0:
        return None
    u, v = obb_center(corners)
    depth = CAMERA.depth_to(plane_z)
    shift_px = BOLT_GRASP_SHIFT_M * CAMERA.fx / depth
    angle = pixel_long_axis(corners)
    # Away from the head.
    return (u - sign * shift_px * math.cos(angle),
            v - sign * shift_px * math.sin(angle))


def wrap_to_pi(angle: float) -> float:
    """Fold an angle into [-pi, pi)."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


# --------------------------------------------------------------------------
# Convenience: a detection straight from /Yolov8_Inference
# --------------------------------------------------------------------------

def detection_to_link0(coordinates: Iterable[float], class_name: str = "",
                       camera: CameraModel = CAMERA) -> Tuple[float, float, float]:
    """Turn one Yolov8Inference result into (x, y, yaw) in link0.

    ``coordinates`` is the flat 8-element xyxyxyxy corner list the detector
    publishes. ``class_name`` selects the centroid plane; an unrecognised name
    falls back to the cube's.
    """
    pts = list(coordinates)
    if len(pts) != 8:
        raise ValueError(f"expected 8 OBB coordinates, got {len(pts)}")
    corners = [(pts[i], pts[i + 1]) for i in range(0, 8, 2)]
    u, v = obb_center(corners)
    plane_z = OBJECT_CENTROID_Z.get(class_name, DEFAULT_CENTROID_Z)
    x, y = camera.pixel_to_link0(u, v, plane_z)
    return x, y, grasp_yaw(corners)


# --------------------------------------------------------------------------
# Self-test. Runs without ROS, Gazebo, or numpy:  python3 projection.py
# --------------------------------------------------------------------------

def _self_test() -> int:
    failures = []

    def check(label, got, want, tol=1e-6):
        if abs(got - want) > tol:
            failures.append(f"{label}: got {got!r}, want {want!r}")
        return got

    cam = CAMERA

    # 1. Intrinsics agree with the constant UI/bolt_selector.py hardcoded.
    #
    #    They are not bit-identical: the legacy constant is 253.93635749816895,
    #    we derive 253.93636730954148 - a relative difference of 4e-8, i.e. the
    #    original was rounded somewhere in single precision. Over the full
    #    640 px width at 0.7 m that is a positional disagreement of ~0.1 um, so
    #    the tolerance below is set where the difference could start to matter
    #    (0.1 mm) rather than at float equality. Anything that trips it is a
    #    genuine formula error, not rounding.
    LEGACY_F = 253.93635749816895
    check("fx", cam.fx, LEGACY_F, 1e-4)
    check("fy", cam.fy, LEGACY_F, 1e-4)
    check("cx", cam.cx, 320.0)
    check("cy", cam.cy, 240.0)

    # 2. Reproduce the original GUI formula at its own depth plane, to 1 micron.
    #    Guards against a sign flip going unnoticed during the extraction.
    for u, v in ((0, 0), (123, 456), (320, 240), (639, 479)):
        z, ix, iy = 0.7, 0.2, 0.6
        want_x = -z * (v - 240) / LEGACY_F + ix
        want_y = -z * (u - 320) / LEGACY_F + iy
        gx, gy = cam.pixel_to_link0(u, v, plane_z=0.0)
        check(f"legacy x @({u},{v})", gx, want_x, 1e-6)
        check(f"legacy y @({u},{v})", gy, want_y, 1e-6)

    # 3. Round-trip forward -> inverse for every class's depth plane.
    for cls, top in list(OBJECT_CENTROID_Z.items()) + [("table", TABLE_TOP_Z)]:
        for u, v in ((10, 10), (320, 240), (500, 100), (630, 470)):
            x, y = cam.pixel_to_link0(u, v, top)
            ru, rv = cam.link0_to_pixel(x, y, top)
            check(f"roundtrip u [{cls}] @({u},{v})", ru, u, 1e-9)
            check(f"roundtrip v [{cls}] @({u},{v})", rv, v, 1e-9)

    # 4. world <-> link0 round-trip.
    wx, wy, wz = 0.55, 0.25, 1.055
    check("world roundtrip x", link0_to_world(*world_to_link0(wx, wy, wz))[0], wx, 1e-12)

    # 5. The FOV footprint quoted in the world SDF header comment.
    x0, x1, y0, y1 = cam.footprint_link0(plane_z=0.0)
    check("fov x_min", x0, -0.4616, 1e-3)
    check("fov x_max", x1, 0.8616, 1e-3)
    check("fov y_min", y0, -0.2821, 1e-3)
    check("fov y_max", y1, 1.4821, 1e-3)

    # 6. Every object in the world SDF must actually be visible, or perception
    #    silently never sees it. (world coords, straight from the SDF)
    scene = [
        ("cube_yellow_1", 0.50, 0.20, "cube"),
        ("cube_yellow_2", 0.30, 0.42, "cube"),
        ("cube_red_1",    0.15, 0.48, "cube"),
        ("cube_blue_1",   0.45, 0.45, "cube"),
        ("cube_green_1",  0.10, 0.62, "cube"),
        ("cylinder_red_1", 0.35, 0.60, "cylinder"),
        ("bolt1", 0.50, 0.35, "bolt"),
        ("bolt2", 0.25, 0.63, "bolt"),
        ("bolt3", 0.20, 0.30, "bolt"),
    ]
    print("  object            link0 x/y        pixel u/v      reach   in-frame")
    for name, wx, wy, cls in scene:
        lx, ly, _ = world_to_link0(wx, wy, 0.0)
        top = OBJECT_CENTROID_Z[cls]
        u, v = cam.link0_to_pixel(lx, ly, top)
        reach = math.hypot(lx, ly)
        ok = cam.in_frame(u, v, margin=25)
        if not ok:
            failures.append(f"{name} is outside the camera frame at ({u:.0f},{v:.0f})")
        if not 0.30 <= reach <= 0.75:
            failures.append(f"{name} reach {reach:.3f} m outside comfortable envelope")
        print(f"  {name:<16} ({lx:+.3f},{ly:+.3f})   ({u:6.1f},{v:6.1f})   "
              f"{reach:.3f}   {'yes' if ok else 'NO'}")

    # 7. Depth correction actually matters - assert the size of the error the
    #    old constant-0.7 code was making, so nobody "simplifies" it back.
    u, v = 560, 120                      # well off-centre, where it is worst
    x_correct, y_correct = cam.pixel_to_link0(u, v, OBJECT_CENTROID_Z["cube"])
    x_legacy, y_legacy = cam.pixel_to_link0(u, v, 0.0)
    err = math.hypot(x_correct - x_legacy, y_correct - y_legacy)
    print(f"\n  depth correction moves an off-centre cube by {err * 1000:.1f} mm")
    if err < 0.005:
        failures.append("depth correction has no effect - check OBJECT_CENTROID_Z")

    # 7b. Each plane must be the object's CENTROID height, not its top face.
    #     Derived here from the SDF geometry so the two cannot drift apart: if
    #     someone "corrects" a value back to a top face, this fails immediately.
    #     Shape provenance is worlds/arm_on_the_table.sdf.
    derived = {
        "cube": TABLE_TOP_Z + 0.040 / 2,        # 40 mm resting on a face
        # Bimodal: upright puts the centroid 30 mm up, toppled 20 mm. The plane
        # is their midpoint, so neither state is favoured.
        "cylinder": TABLE_TOP_Z + (0.030 + 0.020) / 2,
        "bolt": TABLE_TOP_Z + 0.019,            # head apothem, lying on its side
    }
    print("  class      plane in use   centroid from geometry   delta")
    for cls, want in derived.items():
        got = OBJECT_CENTROID_Z[cls]
        print(f"  {cls:<9}   {got:+.4f}        {want:+.4f}              "
              f"{abs(got - want) * 1000:.1f} mm")
        if abs(got - want) > 0.004:
            failures.append(
                f"OBJECT_CENTROID_Z[{cls}] = {got:+.4f} is "
                f"{abs(got - want) * 1000:.1f} mm from the centroid height "
                f"{want:+.4f} derived from the SDF. If it has been set back to a "
                f"top-face height, read the module docstring: the OBB encloses "
                f"the silhouette, not the top face.")
        # A top-face plane must NOT pass, or this test proves nothing.
        top_face = {"cube": TABLE_TOP_Z + 0.040,
                    "cylinder": TABLE_TOP_Z + 0.060,
                    "bolt": TABLE_TOP_Z + 0.038}[cls]
        if abs(top_face - want) <= 0.004:
            failures.append(f"test is vacuous for {cls}: its top face and its "
                            f"centroid are within tolerance of each other")

    # 8. OBB geometry. A box rotated by a known angle in the image must come
    #    back with the matching link0 yaw.
    def rect(cu, cv, half_long, half_short, theta):
        """Corners of a rectangle centred at (cu,cv), long axis at image angle theta."""
        c, s = math.cos(theta), math.sin(theta)
        out = []
        for du, dv in ((-half_long, -half_short), (half_long, -half_short),
                       (half_long, half_short), (-half_long, half_short)):
            out.append((cu + du * c - dv * s, cv + du * s + dv * c))
        return out

    for theta in (0.0, 0.3, -0.7, 1.2):
        corners = rect(320, 240, 40, 12, theta)
        long_px, short_px = obb_axes_px(corners)
        check(f"long axis px @{theta}", long_px, 80.0, 1e-6)
        check(f"short axis px @{theta}", short_px, 24.0, 1e-6)
        # image angle theta -> link0 yaw pi/2 - theta, folded to [-pi/2, pi/2)
        check(f"long yaw @{theta}", obb_long_axis_yaw(corners),
              wrap_to_half_pi(math.pi / 2 - theta), 1e-9)
        # grasp is perpendicular to that
        d = wrap_to_half_pi(grasp_yaw(corners) - obb_long_axis_yaw(corners))
        check(f"grasp perpendicular @{theta}", abs(d), math.pi / 2, 1e-9)

    # 9. The whole path a detection takes, end to end.
    corners = rect(400, 180, 30, 10, 0.4)
    flat = [c for pt in corners for c in pt]
    x, y, yaw = detection_to_link0(flat, "cube")
    ex, ey = cam.pixel_to_link0(400, 180, OBJECT_CENTROID_Z["cube"])
    check("detection x", x, ex, 1e-12)
    check("detection y", y, ey, 1e-12)
    check("detection yaw", yaw, wrap_to_half_pi(math.pi / 2 - 0.4 + math.pi / 2), 1e-9)

    print()
    if failures:
        print(f"FAILED ({len(failures)}):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("all projection self-tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())

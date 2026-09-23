"""Ground-truth pose -> YOLO-OBB label. The part that has to be exactly right.

If this module is wrong the detector learns a consistent lie, trains happily to
a low loss, and produces confident boxes that back-project to the wrong place.
The failure then surfaces three layers away as "the IK is off" or "the grasp
misses", which is a miserable thing to debug. So: everything here is derived,
cross-checked against an independent implementation, and validated by drawing
the labels back onto the rendered images (validate.py) before any training run.

What a label is
---------------
The minimum-area rotated rectangle around the object's projected silhouette.
Every vertex of the object's 3D geometry is projected at its own true depth
(see geometry.py for why a footprint projection is not good enough), the 2D
convex hull is taken, and the tightest rotated box around that hull is the
label - which is what a careful human annotator would draw.

Why min_area_rect is reimplemented here
---------------------------------------
cv2.minAreaRect would do it in one call. It is also the single most
load-bearing computation in the dataset, so it is implemented from the
rotating-calipers theorem (the minimum-area enclosing rectangle always has one
side flush with a convex-hull edge) and then cross-checked against OpenCV in
the self-test. Two independent implementations agreeing is real evidence; one
implementation passing its own assertions is not. It also lets the self-test
run without OpenCV installed.
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

_REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if os.path.join(_REPO, "src", "yolov8_obb") not in sys.path:
    sys.path.insert(0, os.path.join(_REPO, "src", "yolov8_obb"))

from yolov8_obb.projection import CAMERA, CameraModel, world_to_link0  # noqa: E402

from geometry import transform  # noqa: E402

# --------------------------------------------------------------------------
# Convex geometry in the image plane
# --------------------------------------------------------------------------


def _cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))


def convex_hull(points: np.ndarray) -> np.ndarray:
    """Counter-clockwise convex hull of an (N, 2) point set (monotone chain)."""
    pts = np.unique(np.asarray(points, dtype=float), axis=0)   # also lex-sorts
    if len(pts) <= 2:
        return pts

    def half(seq):
        out: List[np.ndarray] = []
        for q in seq:
            while len(out) >= 2 and _cross(out[-2], out[-1], q) <= 0.0:
                out.pop()
            out.append(q)
        return out

    lower = half(pts)
    upper = half(pts[::-1])
    return np.array(lower[:-1] + upper[:-1])


def min_area_rect(points: np.ndarray) -> np.ndarray:
    """Corners of the minimum-area rotated rectangle enclosing ``points``.

    Returns (4, 2), traversing the rectangle in order. Rotating calipers over
    every hull edge: for each edge, project the hull onto that edge's direction
    and its normal, and the extents give an enclosing rectangle. The smallest
    such rectangle is the answer.
    """
    hull = convex_hull(points)
    if len(hull) < 3:
        # Degenerate (collinear or single point): fall back to the axis-aligned
        # box so callers never get fewer than 4 corners. In practice this cannot
        # happen for a solid object, but a silent shape change here would be
        # much worse than a slightly loose box.
        p = np.asarray(points, dtype=float)
        lo, hi = p.min(axis=0), p.max(axis=0)
        return np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]])

    best: Optional[Tuple] = None
    n = len(hull)
    for i in range(n):
        edge = hull[(i + 1) % n] - hull[i]
        length = math.hypot(edge[0], edge[1])
        if length < 1e-12:
            continue
        u = edge / length
        v = np.array([-u[1], u[0]])
        a = hull @ u
        b = hull @ v
        area = (a.max() - a.min()) * (b.max() - b.min())
        if best is None or area < best[0]:
            best = (area, u, v, a.min(), a.max(), b.min(), b.max())

    _, u, v, a0, a1, b0, b1 = best
    return np.array([a0 * u + b0 * v, a1 * u + b0 * v, a1 * u + b1 * v, a0 * u + b1 * v])


def polygon_area(poly: Sequence[Sequence[float]]) -> float:
    """Absolute area of a simple polygon (shoelace)."""
    p = np.asarray(poly, dtype=float)
    if len(p) < 3:
        return 0.0
    x, y = p[:, 0], p[:, 1]
    return abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))) / 2.0


def clip_convex(subject: np.ndarray, clip: np.ndarray) -> np.ndarray:
    """Intersect two convex polygons (Sutherland-Hodgman). Both must be CCW.

    Used to measure how much one object's box covers another's, so the
    generator can reject frames with heavy occlusion instead of quietly
    training on boxes whose contents are mostly a different object.
    """
    out = np.asarray(subject, dtype=float)
    c = np.asarray(clip, dtype=float)
    for i in range(len(c)):
        if len(out) == 0:
            return out
        a, b = c[i], c[(i + 1) % len(c)]
        edge = b - a
        # Inside means to the left of a->b for a CCW clip polygon.
        inside = ((out[:, 0] - a[0]) * edge[1] - (out[:, 1] - a[1]) * edge[0]) <= 0
        nxt: List[np.ndarray] = []
        for j in range(len(out)):
            k = (j + 1) % len(out)
            if inside[j]:
                nxt.append(out[j])
            if inside[j] != inside[k]:
                d = out[k] - out[j]
                den = d[0] * edge[1] - d[1] * edge[0]
                if abs(den) > 1e-15:
                    t = ((a[0] - out[j][0]) * edge[1] - (a[1] - out[j][1]) * edge[0]) / den
                    nxt.append(out[j] + t * d)
        out = np.array(nxt) if nxt else np.empty((0, 2))
    return out


def _ccw(poly: np.ndarray) -> np.ndarray:
    p = np.asarray(poly, dtype=float)
    x, y = p[:, 0], p[:, 1]
    signed = float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    return p if signed >= 0 else p[::-1]


def overlap_fraction(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection area divided by the *smaller* polygon's area.

    Deliberately not IoU: what matters is how badly the smaller object is
    buried, and IoU understates that whenever the two differ in size - a bolt
    almost entirely hidden behind a cylinder still scores a modest IoU.
    """
    ia = polygon_area(clip_convex(_ccw(a), _ccw(b)))
    smaller = min(polygon_area(a), polygon_area(b))
    return 0.0 if smaller <= 0 else ia / smaller


# --------------------------------------------------------------------------
# Projection to a label
# --------------------------------------------------------------------------

@dataclass
class Label:
    """One object's ground truth for one frame."""
    class_id: int
    class_name: str
    color: str
    model_name: str
    corners: np.ndarray            # (4, 2) pixels
    #: True pose, world frame - what the label was generated from.
    world_xyz: Tuple[float, float, float]
    rpy: Tuple[float, float, float]
    #: XY centre of the posed geometry's extent, world frame - see
    #: grasp_target_link0 for why this is not world_xyz.
    grasp_xy_world: Tuple[float, float] = (0.0, 0.0)
    #: Highest point of the posed geometry, link0 z. The depth plane that
    #: inference *should* assume is a little below this.
    top_z_link0: float = 0.0

    @property
    def area_px(self) -> float:
        return polygon_area(self.corners)

    def yolo_line(self, width: int, height: int) -> str:
        """`class x1 y1 x2 y2 x3 y3 x4 y4`, normalised to [0, 1].

        Ultralytics reads OBB labels as four corners in traversal order and
        converts to its own xywhr internally, so the winding does not matter -
        but the normalisation does, and it is by width for x and height for y,
        not by a single dimension.
        """
        vals = []
        for u, v in self.corners:
            vals.append(min(max(u / width, 0.0), 1.0))
            vals.append(min(max(v / height, 0.0), 1.0))
        return " ".join([str(self.class_id)] + [f"{c:.6f}" for c in vals])

    def link0_truth(self) -> Tuple[float, float]:
        """The object's *pose origin* in link0. Not usually what you want -
        see grasp_target_link0."""
        return world_to_link0(*self.world_xyz)[:2]

    def grasp_target_link0(self) -> Tuple[float, float]:
        """Where the gripper should actually go, in link0.

        This is the centre of the object's XY extent, not its pose origin, and
        the distinction is not pedantic: the bolt's STL origin sits just under
        its hex head, ~42 mm from the middle of the shaft. A pick aimed at the
        pose origin would clamp the head instead of the shaft, and comparing
        inference against the origin would report a 40 mm "error" that is
        really just the two reference points being different things.

        For a cube or an upright cylinder this coincides with the pose origin,
        which is exactly why the discrepancy is easy to miss until a bolt shows
        up.
        """
        wx, wy = self.grasp_xy_world
        return world_to_link0(wx, wy, 0.0)[:2]


def project_vertices(vertices: np.ndarray, xyz: Sequence[float], rpy: Sequence[float],
                     camera: CameraModel = CAMERA) -> np.ndarray:
    """Project a posed object's vertex cloud into the image. Returns (N, 2) px.

    Each vertex uses its own depth, so this is a true perspective projection and
    captures the side faces a 34-degree viewing angle exposes.
    """
    world = transform(np.asarray(vertices, dtype=float), xyz, rpy)
    out = np.empty((len(world), 2), dtype=float)
    for i, (wx, wy, wz) in enumerate(world):
        lx, ly, lz = world_to_link0(wx, wy, wz)
        if lz >= camera.origin[2]:
            raise ValueError(
                f"vertex at link0 z={lz:.3f} is at or above the camera "
                f"(z={camera.origin[2]}); cannot project")
        out[i] = camera.link0_to_pixel(lx, ly, lz)
    return out


def make_label(class_id: int, class_name: str, color: str, model_name: str,
               vertices: np.ndarray, xyz: Sequence[float], rpy: Sequence[float],
               camera: CameraModel = CAMERA) -> Label:
    posed = transform(np.asarray(vertices, dtype=float), xyz, rpy)
    pts = project_vertices(vertices, xyz, rpy, camera)
    lo, hi = posed.min(axis=0), posed.max(axis=0)
    return Label(class_id=class_id, class_name=class_name, color=color,
                 model_name=model_name, corners=min_area_rect(pts),
                 world_xyz=tuple(float(c) for c in xyz),
                 rpy=tuple(float(c) for c in rpy),
                 grasp_xy_world=(float((lo[0] + hi[0]) / 2.0),
                                 float((lo[1] + hi[1]) / 2.0)),
                 top_z_link0=float(world_to_link0(0.0, 0.0, hi[2])[2]))


def label_coverage(image: np.ndarray, corners: np.ndarray,
                   background: Optional[np.ndarray] = None,
                   tolerance: int = 18) -> float:
    """Fraction of a label's interior that is not background. Detects staleness.

    This exists because of a bug that no amount of geometric care would have
    caught. Moving the generator from the `gz service` CLI to a persistent
    transport node took it from 2.5 fps to 15.4 fps - and broke the dataset. At
    2.5 fps, discarding two frames after a teleport left ample slack for the
    render-and-bridge pipeline to catch up. At 15.4 fps it did not, so ~22% of
    frames were rendered *before* the teleport took effect: the image showed the
    previous scene while the labels described the new one.

    Nothing about the labels was wrong, and nothing about the images was wrong.
    Only their pairing was, and a 40-frame smoke test at the old speed had passed
    at 100%. It surfaced as "the colour classifier is 75% accurate", because the
    label regions contained bare table.

    So rather than tuning a delay - which would be a proxy for the property
    actually wanted - the generator now *verifies* it: does this image really
    show the scene these labels describe? An object always differs strongly from
    the table (the objects are saturated colours; the bolt renders near-black),
    so a label sitting on background reads ~0.0 while a correct one reads >0.5.
    The distribution is sharply bimodal, which is what makes the check reliable
    rather than a threshold to fiddle with.

    ``background`` is compared against, and defaults to the median colour of a
    ring immediately *outside* the box. A local ring rather than a global
    background estimate, deliberately: this frame contains two large surfaces
    (the table at BGR 216 and the ground plane around it), so no single
    whole-frame statistic describes "what this box sits on". Whether the mode
    landed on the table or the ground would depend on how much of each happened
    to be in view, and if it landed on the ground then a label sitting on bare
    *table* would read as full coverage - the check would pass everything and
    quietly prove nothing. The ring cannot make that mistake: it measures what
    this particular box's surroundings look like.
    """
    import cv2

    pts = np.array([[int(round(u)), int(round(v))] for u, v in corners],
                   dtype=np.int32)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    eroded = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=2)
    if cv2.countNonZero(eroded) < 12:
        eroded = mask                    # tiny box: eroding it away proves nothing
    sel = eroded > 0
    if not sel.any():
        return 0.0

    if background is None:
        background = ring_background(image, mask)

    deviation = np.abs(image[sel].astype(np.int16)
                       - np.asarray(background, dtype=np.int16)).max(axis=1)
    return float((deviation > tolerance).mean())


def ring_background(image: np.ndarray, mask: np.ndarray,
                    inner: int = 3, outer: int = 9) -> np.ndarray:
    """Median colour of an annulus just outside ``mask`` - the local surface.

    The median, not the mean, so a neighbouring object clipping the ring shifts
    the answer very little.
    """
    import cv2

    grown = cv2.dilate(mask, np.ones((outer, outer), np.uint8))
    near = cv2.dilate(mask, np.ones((inner, inner), np.uint8))
    ring = (grown > 0) & (near == 0)
    if not ring.any():
        ring = grown > 0
    return np.median(image[ring], axis=0)


def frame_matches_labels(image: np.ndarray, labels: Sequence["Label"],
                         min_coverage: float = 0.25) -> Tuple[bool, float]:
    """Is every label actually on an object in this image?

    Returns (ok, worst_coverage). ``min_coverage`` sits in the empty middle of a
    sharply bimodal distribution (stale labels ~0.0, correct labels >0.5), so it
    is a separator rather than a tuning knob.
    """
    if not labels:
        return True, 1.0
    worst = 1.0
    for lab in labels:
        worst = min(worst, label_coverage(image, lab.corners))
        if worst < min_coverage:
            break                        # one bad label condemns the frame
    return worst >= min_coverage, worst


def fully_inside(corners: np.ndarray, camera: CameraModel = CAMERA,
                 margin: float = 4.0) -> bool:
    """Are all four corners comfortably inside the frame?

    Objects are *placed* so this holds rather than being filtered out
    afterwards: an object that is visible but unlabelled is worse than no
    object at all, because it teaches the detector to ignore its own class.
    generate.py resamples until this passes.
    """
    return all(camera.in_frame(u, v, margin=margin) for u, v in corners)


# --------------------------------------------------------------------------
# What the fixed-depth assumption costs at inference time
# --------------------------------------------------------------------------

#: How far the reported position may be off before a top-down grasp misses,
#: per class. Half the jaw opening minus half the object's grasped width: the
#: gripper opens to scene.OPEN_JAW_GAP-ish and the object must still fit
#: between the fingers after the positioning error is applied.
#:   cube      60 mm jaw on a 40 mm cube  -> 10 mm a side
#:   cylinder  60 mm jaw on a 40 mm dia   -> 10 mm a side
#:   bolt      60 mm jaw on a 24 mm shaft -> 18 mm a side
#: These are the *hard* limits at which the fingers touch the object off-centre;
#: the generator should stay well inside them.
GRASP_CLEARANCE_MM = {"cube": 10.0, "cylinder": 10.0, "bolt": 18.0}


def inference_xy(label: Label, camera: CameraModel = CAMERA,
                 top_z: Optional[float] = None) -> Tuple[float, float]:
    """Back-project the label's centre the way world_model will at runtime.

    The generator knows every vertex's z; inference has only pixels and must
    assume one depth plane per class. Comparing this against
    ``grasp_target_link0()`` is a direct measurement of the error the whole
    grasp budget rests on, and it is only possible because the ground truth is
    known. validate.py reports the distribution over a whole dataset.

    ``top_z`` overrides the per-class plane, which is how best_depth_plane
    searches for a better value than the one currently in projection.py.
    """
    from yolov8_obb.projection import (OBJECT_CENTROID_Z, DEFAULT_CENTROID_Z,
                                       obb_center)
    u, v = obb_center(label.corners)
    if top_z is None:
        top_z = OBJECT_CENTROID_Z.get(label.class_name, DEFAULT_CENTROID_Z)
    return camera.pixel_to_link0(u, v, top_z)


def inference_error_mm(label: Label, camera: CameraModel = CAMERA,
                       top_z: Optional[float] = None) -> float:
    """Distance in mm between the inferred position and the true grasp target."""
    tx, ty = label.grasp_target_link0()
    ix, iy = inference_xy(label, camera, top_z)
    return math.hypot(ix - tx, iy - ty) * 1000.0


def best_depth_plane(labels: Sequence[Label], camera: CameraModel = CAMERA,
                     lo: float = -0.02, hi: float = 0.09,
                     steps: int = 441) -> Tuple[float, float]:
    """Search for the depth plane that minimises worst-case inference error.

    projection.OBJECT_CENTROID_Z was derived from SDF geometry, and for the
    bolt it was initially marked as nominal. With ground truth available the right
    value can simply be measured instead of argued about. Returns
    (best_top_z, worst_error_mm_at_that_plane).

    Worst case rather than mean: a grasp that misses once in twenty is a
    failure mode, not an average.
    """
    best = None
    for i in range(steps):
        z = lo + (hi - lo) * i / (steps - 1)
        try:
            worst = max(inference_error_mm(l, camera, z) for l in labels)
        except ValueError:
            continue                               # plane at/above the camera
        if best is None or worst < best[1]:
            best = (z, worst)
    return best if best else (float("nan"), float("nan"))


# --------------------------------------------------------------------------
# Self-test:  python3 labeler.py
# --------------------------------------------------------------------------

def _self_test() -> int:
    failures = []
    rng = np.random.default_rng(7)

    def check(label, got, want, tol=1e-9):
        if abs(float(got) - float(want)) > tol:
            failures.append(f"{label}: got {got!r}, want {want!r}")

    # ---- 1. min_area_rect on an exactly-known input -----------------------
    # A rectangle rotated by theta must come back with its own area and side
    # lengths, whatever theta is.
    for theta in (0.0, 0.3, -0.9, 1.2, math.pi / 2):
        c, s = math.cos(theta), math.sin(theta)
        r = np.array([[c, -s], [s, c]])
        base = np.array([[-30, -8], [30, -8], [30, 8], [-30, 8]], dtype=float)
        pts = base @ r.T + np.array([320.0, 240.0])
        got = min_area_rect(pts)
        check(f"rect area @{theta:+.2f}", polygon_area(got), 60 * 16, 1e-6)
        sides = sorted(math.dist(got[i], got[(i + 1) % 4]) for i in range(4))
        check(f"rect short side @{theta:+.2f}", sides[0], 16.0, 1e-6)
        check(f"rect long side @{theta:+.2f}", sides[3], 60.0, 1e-6)

    # Extra interior points must not change the answer.
    inner = np.vstack([pts, rng.uniform(-5, 5, (40, 2)) + np.array([320.0, 240.0])])
    check("interior points ignored", polygon_area(min_area_rect(inner)), 60 * 16, 1e-6)

    # ---- 2. cross-check against OpenCV on random clouds -------------------
    try:
        import cv2
        worst = 0.0
        for _ in range(300):
            n = int(rng.integers(3, 40))
            pts = rng.uniform(0, 640, (n, 2))
            mine = polygon_area(min_area_rect(pts))
            (_, _), (w, h), _ = cv2.minAreaRect(pts.astype(np.float32))
            theirs = w * h
            # Relative difference: OpenCV works in float32, so exact equality
            # is not available. Anything above 1e-3 relative is a real
            # disagreement about which rectangle is minimal, not rounding.
            rel = abs(mine - theirs) / max(theirs, 1e-9)
            worst = max(worst, rel)
        print(f"  vs cv2.minAreaRect over 300 random clouds: "
              f"worst relative area difference {worst:.2e}")
        if worst > 1e-3:
            failures.append(f"min_area_rect disagrees with OpenCV by {worst:.3e} "
                            f"relative area")
    except ImportError:
        print("  cv2 not installed - skipping the OpenCV cross-check "
              "(run this again in the ROS environment)")

    # ---- 3. convex hull sanity -------------------------------------------
    pts = rng.uniform(0, 100, (60, 2))
    hull = convex_hull(pts)
    # Every point must be inside or on the hull.
    for p in pts:
        for i in range(len(hull)):
            if _cross(hull[i], hull[(i + 1) % len(hull)], p) < -1e-9:
                failures.append("convex_hull left a point outside")
                break

    # ---- 4. polygon intersection ------------------------------------------
    unit = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
    check("self-intersection area", polygon_area(clip_convex(unit, unit)), 100.0, 1e-9)
    shifted = unit + np.array([5.0, 0.0])
    check("half-overlap area", polygon_area(clip_convex(_ccw(unit), _ccw(shifted))),
          50.0, 1e-9)
    check("half-overlap fraction", overlap_fraction(unit, shifted), 0.5, 1e-9)
    far = unit + np.array([100.0, 0.0])
    check("disjoint fraction", overlap_fraction(unit, far), 0.0, 1e-12)
    # A small box wholly inside a big one is 100% buried, though IoU would be
    # only 1/16 - this is exactly the case overlap_fraction exists to catch.
    big = np.array([[0, 0], [40, 0], [40, 40], [0, 40]], dtype=float)
    small = np.array([[10, 10], [20, 10], [20, 20], [10, 20]], dtype=float)
    check("engulfed fraction", overlap_fraction(big, small), 1.0, 1e-9)

    # ---- 5. perspective actually matters ----------------------------------
    # A 40 mm cube directly under the camera should project to roughly its top
    # face; the same cube far off-axis must project noticeably larger, because
    # its side faces come into view. If these two are equal, the projection has
    # silently collapsed to orthographic and every off-centre label is wrong.
    from objects import build_classes, TABLE_TOP_WORLD_Z
    from geometry import resting_z
    classes = build_classes()
    cube = classes["cube"]
    z = resting_z(cube.vertices, (0, 0, 0), TABLE_TOP_WORLD_Z)

    # link0 (0.20, 0.60) is directly beneath the camera -> world (0.25, 0.60).
    under = make_label(0, "cube", "red", "c", cube.vertices, (0.25, 0.60, z), (0, 0, 0))
    off = make_label(0, "cube", "red", "c", cube.vertices, (0.50, 0.20, z), (0, 0, 0))
    side_under = math.dist(under.corners[0], under.corners[1])
    side_off = math.dist(off.corners[0], off.corners[1])
    print(f"  cube under the camera: {side_under:.1f} px/side; "
          f"off-axis at link0 (0.45,0.20): {side_off:.1f} px/side "
          f"({side_off / side_under:.2f}x)")
    if side_off <= side_under * 1.05:
        failures.append("off-axis cube does not project larger - projection has "
                        "collapsed to orthographic")

    # Under the camera, the label should be close to the true top-face size:
    # 40 mm at depth 0.665 m with f=253.94 -> 15.3 px.
    expect = 0.04 * CAMERA.fx / (CAMERA.origin[2] - 0.035)
    check("cube top face px under camera", side_under, expect, 0.6)

    # ---- 6. round trip through the YOLO text format -----------------------
    line = off.yolo_line(CAMERA.width, CAMERA.height)
    parts = line.split()
    if len(parts) != 9:
        failures.append(f"yolo_line has {len(parts)} fields, want 9")
    if parts[0] != "0":
        failures.append(f"yolo_line class id is {parts[0]!r}, want '0'")
    nums = [float(x) for x in parts[1:]]
    if not all(0.0 <= n <= 1.0 for n in nums):
        failures.append("yolo_line emitted a coordinate outside [0, 1]")
    back = np.array([[nums[i] * CAMERA.width, nums[i + 1] * CAMERA.height]
                     for i in range(0, 8, 2)])
    check("yolo_line round-trip", np.abs(back - off.corners).max(), 0.0, 1e-3)

    # ---- 7. every class produces a plausible label at a real scene pose ---
    print("\n  class      OBB px (long x short)   area px^2   origin-vs-target")
    for cname in ("cube", "cylinder", "bolt"):
        cls = classes[cname]
        rpy = cls.sample_rpy(np.random.default_rng(3))
        zz = resting_z(cls.vertices, rpy, TABLE_TOP_WORLD_Z)
        lab = make_label(cls.class_id, cname, "red", "m", cls.vertices,
                         (0.45, 0.35, zz), rpy)
        sides = sorted(math.dist(lab.corners[i], lab.corners[(i + 1) % 4])
                       for i in range(4))
        ox, oy = lab.link0_truth()
        gx, gy = lab.grasp_target_link0()
        offset = math.hypot(gx - ox, gy - oy) * 1000.0
        print(f"  {cname:<9}   {sides[3]:6.1f} x {sides[0]:5.1f}        "
              f"{lab.area_px:8.1f}   {offset:6.1f} mm")
        if not fully_inside(lab.corners):
            failures.append(f"{cname} at a normal scene pose is not fully in frame")
        if sides[0] < 3.0:
            failures.append(f"{cname} OBB short side is only {sides[0]:.1f} px - "
                            f"check the vertex cloud")
    # The cube's origin *is* its grasp target; the bolt's is not. Assert both,
    # so the distinction cannot quietly collapse.
    cube_lab = make_label(0, "cube", "red", "m", classes["cube"].vertices,
                          (0.45, 0.35, 1.035), (0, 0, 0.4))
    cx0, cy0 = cube_lab.link0_truth()
    cx1, cy1 = cube_lab.grasp_target_link0()
    check("cube origin == grasp target", math.hypot(cx1 - cx0, cy1 - cy0), 0.0, 1e-12)

    bolt_rpy = (0.0, math.pi / 2.0, 0.0)
    bz = resting_z(classes["bolt"].vertices, bolt_rpy, TABLE_TOP_WORLD_Z)
    bolt_lab = make_label(2, "bolt", "grey", "m", classes["bolt"].vertices,
                          (0.45, 0.35, bz), bolt_rpy)
    bx0, by0 = bolt_lab.link0_truth()
    bx1, by1 = bolt_lab.grasp_target_link0()
    bolt_offset = math.hypot(bx1 - bx0, by1 - by0) * 1000.0
    if bolt_offset < 30.0:
        failures.append(f"bolt origin/grasp-target offset is only "
                        f"{bolt_offset:.1f} mm - expected ~42 mm; the STL origin "
                        f"rationale in grasp_target_link0 may be stale")

    # ---- 8. how good is the fixed-depth assumption, per class? ------------
    # This is the perception error budget the grasp depends on, measured rather
    # than assumed. Sampled over the whole reachable, visible table area.
    from yolov8_obb.projection import OBJECT_CENTROID_Z
    rng2 = np.random.default_rng(11)
    print("\n  fixed-depth back-projection error over 400 random poses per class")
    print("  class      current plane  mean    p95   worst  |  best plane  worst")
    for cname in ("cube", "cylinder", "bolt"):
        cls = classes[cname]
        labs: List[Label] = []
        while len(labs) < 400:
            wx = float(rng2.uniform(0.05, 0.62))
            wy = float(rng2.uniform(-0.10, 0.72))
            rpy = cls.sample_rpy(rng2)
            zz = resting_z(cls.vertices, rpy, TABLE_TOP_WORLD_Z)
            lab = make_label(cls.class_id, cname, "red", "m", cls.vertices,
                             (wx, wy, zz), rpy)
            if fully_inside(lab.corners, margin=6.0):
                labs.append(lab)
        errs = np.array([inference_error_mm(l) for l in labs])
        bz_, bworst = best_depth_plane(labs)
        cur = OBJECT_CENTROID_Z[cname]
        print(f"  {cname:<9}     {cur:+.3f}      {errs.mean():5.1f}  "
              f"{np.percentile(errs, 95):5.1f}  {errs.max():5.1f}  |  "
              f"{bz_:+.3f}     {bworst:5.1f}")
        limit = GRASP_CLEARANCE_MM[cname]
        if errs.max() > limit:
            failures.append(
                f"{cname}: worst fixed-depth error {errs.max():.1f} mm exceeds "
                f"the {limit:.0f} mm the jaw can absorb. Best available plane is "
                f"{bz_:+.4f} (worst {bworst:.1f} mm) vs the current "
                f"{cur:+.4f} - update projection.OBJECT_CENTROID_Z.")

    print()
    if failures:
        print(f"FAILED ({len(failures)}):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("all labeler self-tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())

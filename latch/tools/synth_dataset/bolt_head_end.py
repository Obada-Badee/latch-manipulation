#!/usr/bin/env python3
"""Can the head end of a bolt be identified from the image inside its box?

    python3 bolt_head_end.py --dataset ~/synth_obb_neg

`bolt_grasp_offset.py` established that the grasp point needs to move 34.2 mm
(sd 1.2) along the box's long axis, and that the *direction* is a coin flip -
the oriented box carries no information about which end holds the hex head.

So the direction has to come from the pixels. The hex head is 43.8 mm across and
the shaft 24 mm, so the object should simply be wider at the head end. This tests
that: for every labelled bolt, mask the object inside its own box, measure the
object's width along the box's long axis, and predict that the wider half holds
the head. Ground truth comes from the pose, which says exactly where the head is.

The number that matters is the accuracy of that prediction. Anything short of
near-perfect is not usable: a wrong call does not merely fail to help, it moves
the grasp point 34 mm the wrong way, which is worse than the bug being fixed.
That asymmetry is why this is measured before anything is wired into the robot.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import List, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(_ROOT, "src", "yolov8_obb"))

import objects as pool_objects                                  # noqa: E402
from geometry import rpy_matrix                                 # noqa: E402
from yolov8_obb.projection import CAMERA                        # noqa: E402

#: The table renders as a flat light grey. Anything this much darker inside the
#: box is object rather than background. Deliberately generous - the bolt is
#: dark metal and its shadow is darker still, and including a little shadow
#: costs nothing here because the shadow follows the object.
TABLE_LEVEL = 216
DARK_MARGIN = 25

#: Bins along the long axis when profiling width.
BINS = 12

SHAFT_Z_CUTOFF = -0.030


def pixel_long_axis(corners: np.ndarray) -> float:
    """Angle of the box's long side, in IMAGE coordinates, radians.

    Deliberately not `projection.obb_long_axis_yaw`, which returns the angle in
    the **link0** frame. Using that to rotate pixels mixes two frames related by
    an axis swap, and it silently produces a plausible-looking wrong answer:
    the first version of this test did exactly that and reported 49.2% - chance
    - which reads as "the cue does not exist" rather than "the test is broken".
    """
    best, best_len = 0.0, -1.0
    for i in range(4):
        du = corners[(i + 1) % 4][0] - corners[i][0]
        dv = corners[(i + 1) % 4][1] - corners[i][1]
        length = math.hypot(du, dv)
        if length > best_len:
            best, best_len = math.atan2(dv, du), length
    return best


def head_end_in_pixels(vertices: np.ndarray, xyz, rpy, corners: np.ndarray,
                       angle: float) -> float:
    """Ground truth, in image space: the head's offset along the box axis, px.

    Positive means the head lies toward +angle. Head and shaft centroids are
    projected through the same camera model the labels came from, so this is
    directly comparable to a measurement made on the pixels.
    """
    world = vertices @ rpy_matrix(*rpy).T + np.asarray(xyz)
    head = world[vertices[:, 2] > -0.010].mean(axis=0)
    shaft = world[vertices[:, 2] < SHAFT_Z_CUTOFF].mean(axis=0)
    hu, hv = CAMERA.world_to_pixel(*head)
    su, sv = CAMERA.world_to_pixel(*shaft)
    return ((hu - su) * math.cos(angle)) + ((hv - sv) * math.sin(angle))


def width_profile(gray: np.ndarray, corners: np.ndarray, yaw: float
                  ) -> Optional[Tuple[float, float]]:
    """(object width in the -axis half, width in the +axis half), pixels.

    Works in the box's own frame: pixels inside the box are rotated so the long
    axis is horizontal, then object pixels are counted per column and averaged
    over each half.
    """
    cx, cy = corners[:, 0].mean(), corners[:, 1].mean()
    x0, x1 = int(corners[:, 0].min()), int(corners[:, 0].max())
    y0, y1 = int(corners[:, 1].min()), int(corners[:, 1].max())
    x0, y0 = max(x0 - 2, 0), max(y0 - 2, 0)
    x1 = min(x1 + 2, gray.shape[1] - 1)
    y1 = min(y1 + 2, gray.shape[0] - 1)
    if x1 <= x0 or y1 <= y0:
        return None

    ys, xs = np.mgrid[y0:y1 + 1, x0:x1 + 1]
    dark = gray[y0:y1 + 1, x0:x1 + 1] < (TABLE_LEVEL - DARK_MARGIN)
    if dark.sum() < 20:
        return None

    du = (xs - cx) * math.cos(yaw) + (ys - cy) * math.sin(yaw)
    dv = -(xs - cx) * math.sin(yaw) + (ys - cy) * math.cos(yaw)

    # Half-extents of the box in its own frame, to reject neighbouring pixels.
    cu = (corners[:, 0] - cx) * math.cos(yaw) + (corners[:, 1] - cy) * math.sin(yaw)
    cv = -(corners[:, 0] - cx) * math.sin(yaw) + (corners[:, 1] - cy) * math.cos(yaw)
    hu, hv = np.abs(cu).max(), np.abs(cv).max()

    inside = dark & (np.abs(du) <= hu) & (np.abs(dv) <= hv)
    if inside.sum() < 20:
        return None

    du_in, dv_in = du[inside], dv[inside]
    edges = np.linspace(-hu, hu, BINS + 1)
    widths = []
    for i in range(BINS):
        sel = (du_in >= edges[i]) & (du_in < edges[i + 1])
        widths.append(float(dv_in[sel].max() - dv_in[sel].min()) if sel.sum() >= 3
                      else 0.0)
    half = BINS // 2
    return float(np.mean(widths[:half])), float(np.mean(widths[half:]))


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=os.path.expanduser("~/synth_obb_neg"))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    import cv2                                                   # noqa: PLC0415

    root = os.path.expanduser(args.dataset)
    truth_path = os.path.join(root, "ground_truth.jsonl")
    V = np.asarray(pool_objects.build_classes()["bolt"].vertices, dtype=float)

    correct = 0
    total = 0
    skipped = 0
    margins: List[float] = []

    with open(truth_path, encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            bolts = [o for o in rec.get("objects", []) if o.get("class") == "bolt"]
            if not bolts:
                continue
            img = cv2.imread(os.path.join(root, rec["image"]), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            for obj in bolts:
                corners = np.asarray(obj["corners_px"], dtype=float)
                angle = pixel_long_axis(corners)
                prof = width_profile(img, corners, angle)
                if prof is None:
                    skipped += 1
                    continue
                w_neg, w_pos = prof
                predicted_head_positive = w_pos > w_neg
                actual = head_end_in_pixels(V, obj["world_xyz"], obj["rpy"], corners, angle)
                total += 1
                if predicted_head_positive == (actual > 0):
                    correct += 1
                denom = max(w_pos + w_neg, 1e-6)
                margins.append(abs(w_pos - w_neg) / denom)
            if args.limit and total >= args.limit:
                break

    if not total:
        print("no usable bolt instances", file=sys.stderr)
        return 1

    acc = 100.0 * correct / total
    print(f"{total} bolt instances ({skipped} skipped)\n")
    print(f"head end identified correctly: {correct}/{total} = {acc:.1f}%")
    print(f"median width asymmetry:        {100 * float(np.median(margins)):.1f}%")
    print()
    if acc >= 98.0:
        print("=> usable. Wire it in: shift the grasp point 34.2 mm along the\n"
              "   long axis, AWAY from the wider (head) end.")
    else:
        print(f"=> NOT usable. At {acc:.1f}%, roughly {100 - acc:.0f} bolts in\n"
              "   100 would be pushed 34 mm the WRONG way - worse than the bug\n"
              "   being fixed. Needs a stronger cue than width alone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

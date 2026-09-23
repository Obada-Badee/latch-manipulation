#!/usr/bin/env python3
"""Measure where a bolt's grasp point should be, relative to where it is now.

    python3 bolt_grasp_offset.py --dataset ~/synth_obb_neg

The benchmark showed every bolt grasp missing: fingers closed on empty air, the
bolt never moved, and the commanded point sat ~8 mm outside a 24 mm shaft. Height
and yaw were both correct, so the error is in the grasp POINT.

The cause is structural. `world_model` back-projects the centre of the oriented
box, which is the right point for a centrally symmetric object and the wrong one
for a bolt: the box encloses a 43.8 mm hex head plus a 24 mm shaft, so its centre
sits between them rather than on the shaft the gripper has to close around.

This measures the correction rather than guessing it. For every labelled bolt in
the dataset it computes:

    bp          where world_model would put the grasp point today
    shaft       the true centre of the shaft, from the mesh and the pose
    offset      shaft - bp, resolved in the BOX's own frame

Expressing the offset along and across the box's long axis is the whole point: a
correction in link0 would be different for every bolt orientation and therefore
useless, whereas a correction along the bolt's own axis is a property of the
bolt's shape and transfers to any pose. If the along-axis term is consistent and
the across-axis term is ~0, the fix is one constant applied along the long axis.

The sign convention needs care, and it is the reason this is a measurement and
not a formula: the box has no idea which end holds the head. That has to come
out of the data too - see --report-sign.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from typing import List, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(_ROOT, "src", "yolov8_obb"))

import objects as pool_objects                                  # noqa: E402
from geometry import rpy_matrix                                 # noqa: E402
from yolov8_obb.projection import (CAMERA, DEFAULT_CENTROID_Z,  # noqa: E402
                                   OBJECT_CENTROID_Z, obb_long_axis_yaw,
                                   world_to_link0)

#: Mesh vertices with model-frame z below this are shaft, not head. The STL runs
#: z = -0.100 (tip) .. +0.016 (top of head), so this sits clear of the flange.
SHAFT_Z_CUTOFF = -0.030


def shaft_centre_link0(vertices: np.ndarray, xyz, rpy) -> Tuple[float, float]:
    """True xy centre of the shaft, in link0, for a bolt at this pose."""
    world = vertices @ rpy_matrix(*rpy).T + np.asarray(xyz)
    link0 = np.array([world_to_link0(*v) for v in world])
    shaft = link0[vertices[:, 2] < SHAFT_Z_CUTOFF]
    return float(shaft[:, 0].mean()), float(shaft[:, 1].mean())


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=os.path.expanduser("~/synth_obb_neg"))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    truth_path = os.path.join(os.path.expanduser(args.dataset),
                              "ground_truth.jsonl")
    if not os.path.exists(truth_path):
        print(f"no ground_truth.jsonl in {args.dataset}", file=sys.stderr)
        return 1

    V = np.asarray(pool_objects.build_classes()["bolt"].vertices, dtype=float)
    plane_z = OBJECT_CENTROID_Z.get("bolt", DEFAULT_CENTROID_Z)

    along: List[float] = []
    across: List[float] = []
    radial: List[float] = []
    n = 0

    with open(truth_path, encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            for obj in rec.get("objects", []):
                if obj.get("class") != "bolt":
                    continue
                corners = np.asarray(obj["corners_px"], dtype=float)
                u, v = corners[:, 0].mean(), corners[:, 1].mean()
                bx, by = CAMERA.pixel_to_link0(u, v, plane_z)

                sx, sy = shaft_centre_link0(V, obj["world_xyz"], obj["rpy"])

                # Resolve the correction in the BOX's frame: a link0 offset
                # would be a different vector for every bolt orientation and so
                # could not be applied as a constant.
                yaw = obb_long_axis_yaw(corners)
                dx, dy = sx - bx, sy - by
                along.append(dx * math.cos(yaw) + dy * math.sin(yaw))
                across.append(-dx * math.sin(yaw) + dy * math.cos(yaw))
                radial.append(math.hypot(dx, dy))
                n += 1
            if args.limit and n >= args.limit:
                break

    if not n:
        print("no bolt instances found", file=sys.stderr)
        return 1

    def describe(name: str, xs: List[float]) -> None:
        xs_mm = [x * 1000.0 for x in xs]
        mean = statistics.mean(xs_mm)
        sd = statistics.pstdev(xs_mm)
        print(f"  {name:<28} mean {mean:+7.2f} mm   sd {sd:5.2f}   "
              f"min {min(xs_mm):+7.2f}   max {max(xs_mm):+7.2f}")

    print(f"{n} bolt instances from {args.dataset}\n")
    print("offset from world_model's point to the true shaft centre:")
    describe("along the long axis", along)
    describe("across the long axis", across)
    describe("total distance", radial)

    signed = [abs(a) for a in along]
    print()
    print(f"  |along| mean {statistics.mean(signed) * 1000:.1f} mm, "
          f"sd {statistics.pstdev(signed) * 1000:.1f} mm")
    same_sign = sum(1 for a in along if a > 0)
    print(f"  sign: {same_sign}/{n} positive "
          f"({100.0 * same_sign / n:.0f}%)")
    print()
    if statistics.pstdev(along) * 1000 < 8.0 and abs(statistics.mean(along)) * 1000 > 10:
        print("=> a single along-axis constant would fix this.")
    elif max(same_sign, n - same_sign) / n < 0.9:
        print("=> the offset flips sign with orientation: the box alone cannot\n"
              "   say which end holds the head, so a bare constant will make\n"
              "   half the bolts worse. The head has to be identified from the\n"
              "   image, not assumed.")
    else:
        print("=> inconsistent; needs a different approach.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

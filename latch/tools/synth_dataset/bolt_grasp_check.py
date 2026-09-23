#!/usr/bin/env python3
"""Does the shipped bolt grasp correction actually land on the shaft?

    python3 bolt_grasp_check.py --dataset ~/synth_obb_neg

Exercises `projection.bolt_grasp_pixel` - the code the robot runs - rather than
the prototype that derived the constant, and scores it the only way that
matters: distance from the resulting point to the true shaft surface, in metres,
against the 12 mm the shaft gives it to be wrong by.

Reports before and after, because "34 mm improved to 3 mm" is the claim, and a
before/after on the same instances is the only honest way to make it.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from typing import List, Optional, Sequence

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(_ROOT, "src", "yolov8_obb"))

import objects as pool_objects                                  # noqa: E402
from geometry import rpy_matrix                                 # noqa: E402
from yolov8_obb.projection import (CAMERA, DEFAULT_CENTROID_Z,  # noqa: E402
                                   OBJECT_CENTROID_Z, bolt_grasp_pixel,
                                   obb_center, world_to_link0)

SHAFT_Z_CUTOFF = -0.030

#: The reported distance is to the nearest point ON THE SHAFT SURFACE, so it is
#: already an "outside by" figure - not a distance to the axis. A point 8 mm
#: from the surface is 8 mm clear of the metal, which is why the old grasps
#: closed on nothing. Scoring it against the shaft radius would wrongly call
#: that a hit; the bar is being *on* the surface.
ON_SHAFT_M = 0.003


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=os.path.expanduser("~/synth_obb_neg"))
    args = ap.parse_args(argv)

    import cv2                                                   # noqa: PLC0415

    root = os.path.expanduser(args.dataset)
    V = np.asarray(pool_objects.build_classes()["bolt"].vertices, dtype=float)
    plane_z = OBJECT_CENTROID_Z.get("bolt", DEFAULT_CENTROID_Z)

    before: List[float] = []
    after: List[float] = []
    undetermined = 0

    with open(os.path.join(root, "ground_truth.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            bolts = [o for o in rec.get("objects", []) if o.get("class") == "bolt"]
            if not bolts:
                continue
            img = cv2.imread(os.path.join(root, rec["image"]))
            if img is None:
                continue
            for obj in bolts:
                corners = [tuple(c) for c in obj["corners_px"]]

                world = V @ rpy_matrix(*obj["rpy"]).T + np.asarray(obj["world_xyz"])
                link0 = np.array([world_to_link0(*v) for v in world])
                shaft = link0[V[:, 2] < SHAFT_Z_CUTOFF][:, :2]

                u0, v0 = obb_center(corners)
                bx, by = CAMERA.pixel_to_link0(u0, v0, plane_z)
                before.append(float(np.hypot(shaft[:, 0] - bx,
                                             shaft[:, 1] - by).min()))

                shifted = bolt_grasp_pixel(img, corners, plane_z)
                if shifted is None:
                    undetermined += 1
                    after.append(before[-1])          # unchanged, by design
                    continue
                ax, ay = CAMERA.pixel_to_link0(shifted[0], shifted[1], plane_z)
                after.append(float(np.hypot(shaft[:, 0] - ax,
                                            shaft[:, 1] - ay).min()))

    if not before:
        print("no bolt instances", file=sys.stderr)
        return 1

    def show(name: str, xs: List[float]) -> None:
        mm = [x * 1000 for x in xs]
        on = sum(1 for x in xs if x <= ON_SHAFT_M)
        print(f"  {name:<8} mean {statistics.mean(mm):6.2f} mm   "
              f"median {statistics.median(mm):6.2f}   max {max(mm):6.2f}   "
              f"on the shaft: {on}/{len(xs)} "
              f"({100.0 * on / len(xs):.1f}%)")

    print(f"{len(before)} bolt instances; distance to the nearest shaft point\n")
    show("before", before)
    show("after", after)
    print(f"\n  head end undetermined (left unshifted): {undetermined}")
    worse = sum(1 for b, a in zip(before, after) if a > b + 0.001)
    print(f"  made worse by the shift: {worse}/{len(before)} "
          f"({100.0 * worse / len(before):.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

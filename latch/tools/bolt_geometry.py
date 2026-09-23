#!/usr/bin/env python3
"""Where does a bolt actually rest, once physics has settled it?

    python3 tools/bolt_geometry.py

`scene.GRASP_TIP_Z['bolt']` carries the note "CONFIRM against the sim: bolts
settle under physics rather than being placed, so their resting height is not
exactly what the SDF pose says." This is that confirmation.

For each bolt in the running world it reports, in link0 z:

  * the shaft's bottom, centre and top
  * the width of the shaft at the height the gripper currently closes at

The last line is the point. The jaw stalls correctly on a 40 mm cube, so contact
works; if it closes to 2 mm on a bolt the pads may simply be meeting a much
narrower slice of a round shaft than the 24 mm the verification expects - or
missing it entirely.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "synth_dataset"))
sys.path.insert(0, os.path.join(_ROOT, "src", "yolov8_obb"))
sys.path.insert(0, os.path.join(_ROOT, "src", "llm_task_planner"))

SHAFT_Z_CUTOFF = -0.030          # mesh-local: below this is shaft, above is head
SHAFT_RADIUS = 0.012


def main() -> int:
    import objects as pool_objects
    from geometry import rpy_matrix
    from gz_client import GzClient
    from yolov8_obb.projection import world_to_link0
    from llm_task_planner import scene

    V = np.asarray(pool_objects.build_classes()["bolt"].vertices, dtype=float)
    shaft_local = V[V[:, 2] < SHAFT_Z_CUTOFF]

    gz = GzClient("arm_and_table")
    gz.wait_until_ready()
    full = gz.read_poses_full()

    tip_z = scene.GRASP_TIP_Z["bolt"]
    print(f"gripper closes with the fingertip frame at link0 z = {tip_z:.4f}\n")

    for name in sorted(full):
        if not name.startswith("bolt") or name == "bolt":
            continue
        xyz, rpy = full[name]
        world = shaft_local @ rpy_matrix(*rpy).T + np.asarray(xyz)
        link0 = np.array([world_to_link0(*v) for v in world])
        z = link0[:, 2]

        # The shaft is a cylinder of known radius, so its axis height is the
        # mid-height of its surface - not an average of unevenly sampled points.
        lo, hi = float(z.min()), float(z.max())
        axis_z = 0.5 * (lo + hi)

        # Horizontal width of a circle of radius r at height h above its centre.
        dz = abs(tip_z - axis_z)
        if dz >= SHAFT_RADIUS:
            width = 0.0
            note = "the pads pass ENTIRELY above or below the shaft"
        else:
            width = 2.0 * math.sqrt(SHAFT_RADIUS ** 2 - dz ** 2)
            note = f"{100.0 * width / (2 * SHAFT_RADIUS):.0f}% of the full 24 mm"

        print(f"{name}:")
        print(f"  shaft surface z  {lo * 1000:6.1f} .. {hi * 1000:6.1f} mm"
              f"   axis at {axis_z * 1000:5.1f} mm")
        print(f"  gripper is {(tip_z - axis_z) * 1000:+.1f} mm from the axis")
        print(f"  shaft width at that height: {width * 1000:5.1f} mm   ({note})")
        print()

    print(f"grasp verification requires a gap of at least "
          f"{scene.GRASP_MISS_GAP * 1000:.0f} mm to call it a hold.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

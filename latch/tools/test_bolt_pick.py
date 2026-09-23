#!/usr/bin/env python3
"""Live check of the bolt grasp fix: does the jaw now close on the shaft?

    python3 tools/test_bolt_pick.py

Two measurements, in order:

  1. where the perceived grasp point sits relative to the real shaft, from
     Gazebo's pose and the bolt mesh - the same quantity `bolt_grasp_check.py`
     scores offline, but on the running cell
  2. whether `pick` actually succeeds, and whether the bolt moves

The second is the one that counts. Before the fix, every attempt reported
"fingers closed to 2.3 mm, holding nothing" and the bolt did not move by a
single millimetre - so a pick that lifts it is the whole claim.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "synth_dataset"))
sys.path.insert(0, os.path.join(_ROOT, "src", "yolov8_obb"))
sys.path.insert(0, os.path.join(_ROOT, "src", "llm_task_planner"))

SHAFT_Z_CUTOFF = -0.030


def main() -> int:
    # Target by Gazebo model name, not by detection id. Tracked ids are not
    # stable across runs - they were 1/2/3 on one pass and 62/94/96/100 on the
    # next - so "pick object 3" silently becomes "pick whatever is called 3 now",
    # or nothing at all. The model name is the only durable handle, and it is
    # also what lets this compare against a bolt whose resting pose is known.
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bolt", default=None,
                    help="Gazebo model name to target, e.g. bolt1. "
                         "Default: whichever bolt is detected first.")
    args = ap.parse_args()

    import rclpy
    from rclpy.node import Node
    from rclpy.action import ActionClient
    from skill_msgs.action import Pick
    from yolov8_msgs.msg import WorldState

    import objects as pool_objects
    from geometry import rpy_matrix
    from gz_client import GzClient
    from yolov8_obb.projection import world_to_link0

    V = np.asarray(pool_objects.build_classes()["bolt"].vertices, dtype=float)
    gz = GzClient("arm_and_table")
    gz.wait_until_ready()

    rclpy.init()
    node = Node("test_bolt_pick")
    latest = {}
    node.create_subscription(WorldState, "/world_state",
                             lambda m: latest.__setitem__("w", m), 10)
    deadline = time.time() + 15
    while time.time() < deadline and "w" not in latest:
        rclpy.spin_once(node, timeout_sec=0.1)
    if "w" not in latest:
        print("no /world_state", file=sys.stderr)
        return 1

    full = gz.read_poses_full()
    shafts = {}
    for name, (xyz, rpy) in full.items():
        if not name.startswith("bolt") or name == "bolt":
            continue
        world = V @ rpy_matrix(*rpy).T + np.asarray(xyz)
        link0 = np.array([world_to_link0(*v) for v in world])
        shafts[name] = link0[V[:, 2] < SHAFT_Z_CUTOFF][:, :2]

    print("perceived grasp point vs the real shaft surface:")
    bolts = [o for o in latest["w"].objects if o.class_name == "bolt"]
    matched = {}
    used = set()
    for o in bolts:
        best, best_d = None, 1e9
        for name, pts in shafts.items():
            if name in used:
                continue
            d = float(np.hypot(pts[:, 0] - o.x, pts[:, 1] - o.y).min())
            if d < best_d:
                best, best_d = name, d
        if best is None:
            continue
        used.add(best)
        matched[best] = o
        verdict = "ON the shaft" if best_d <= 0.004 else "off the shaft"
        print(f"  id {o.id:<4} ~ {best:<7} {best_d * 1000:5.1f} mm   {verdict}")

    if args.bolt is not None:
        target = matched.get(args.bolt)
        if target is None:
            print(f"\n{args.bolt} is not matched to any detection "
                  f"(matched: {', '.join(sorted(matched)) or 'none'})",
                  file=sys.stderr)
            return 1
        print(f"\ntargeting {args.bolt} (detection id {target.id})")
    else:
        target = bolts[0]
    print(f"\nattempting pick({target.id})...")
    before = {n: world_to_link0(*p) for n, p in gz.read_poses().items()
              if n.startswith("bolt") and n != "bolt"}

    cli = ActionClient(node, Pick, "pick")
    if not cli.wait_for_server(timeout_sec=15.0):
        print("no pick action server", file=sys.stderr)
        return 1
    fut = cli.send_goal_async(Pick.Goal(object_id=target.id))
    t0 = time.time()
    result = None
    while time.time() - t0 < 120:
        rclpy.spin_once(node, timeout_sec=0.1)
        if fut.done() and fut.result() is not None:
            handle = fut.result()
            if not handle.accepted:
                print("goal rejected", file=sys.stderr)
                return 1
            rfut = handle.get_result_async()
            while not rfut.done() and time.time() - t0 < 120:
                rclpy.spin_once(node, timeout_sec=0.1)
            if rfut.done():
                result = rfut.result().result
            break

    after = {n: world_to_link0(*p) for n, p in gz.read_poses().items()
             if n.startswith("bolt") and n != "bolt"}

    print()
    if result is None:
        print("no result within the timeout")
    else:
        print(f"success = {result.success}")
        print(f"message = {result.message[:180]}")

    print("\nbolt movement during the attempt:")
    moved_any = False
    for n in sorted(before):
        d = math.dist(before[n], after[n])
        if d > 0.005:
            moved_any = True
        print(f"  {n}: {d * 1000:6.1f} mm   z {before[n][2]:+.4f} -> {after[n][2]:+.4f}")
    print()
    print("the target moved - the jaw reached it" if moved_any
          else "nothing moved - the jaw still closed on air")

    node.destroy_node()
    rclpy.shutdown()
    return 0 if (result is not None and result.success) else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Where do objects actually land when `place_in_bin` reports success?

    python3 tools/test_place.py --n 8 --bin red_bin

`place` is the least verified skill in the system and, by the numbers, the one
that loses the most trials. Across the 75-trial suite, success fell with the
number of objects a task must get into a bin - 90% at zero placements, 58% at
one, 40% at two, 10% at four - which is what a per-placement reliability of
about 0.6 compounding looks like. The tier gradient is largely that, not
linguistic difficulty.

`pick` measures the finger gap and refuses to claim a grasp it cannot confirm.
`place` approaches, descends, opens, retreats and reports SUCCESS - it never
checks that anything was held, that the object left the hand, or that it landed
anywhere near the bin. So the planner is told the placement worked, reports the
task done, and the table disagrees. That is the whole "over-claim" result: the
model is repeating what the skill layer told it.

This measures the physical half of the question, against Gazebo:

  * how often a reported-successful place actually puts the object in the bin
  * where the misses land, relative to the bin centre and its 90 mm half-width

Ground truth is used HERE, in the instrument, and deliberately not in the skill:
a robot that could read object poses from the simulator would not need to
perceive anything, and the verification we add to `place` has to work from the
robot's own senses.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from typing import Optional, Sequence

HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "synth_dataset"))
sys.path.insert(0, os.path.join(_ROOT, "src", "yolov8_obb"))
sys.path.insert(0, os.path.join(_ROOT, "src", "llm_task_planner"))


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--bin", dest="bin_name", default="red_bin")
    ap.add_argument("--world", default="arm_and_table")
    args = ap.parse_args(argv)

    import rclpy
    from rclpy.action import ActionClient
    from rclpy.node import Node
    from skill_msgs.action import Home, Pick, Place
    from yolov8_msgs.msg import WorldState

    from gz_client import GzClient
    from llm_task_planner import scene
    from yolov8_obb.projection import world_to_link0

    recept = scene.find_receptacle(args.bin_name)
    if recept is None:
        print(f"unknown bin {args.bin_name!r}", file=sys.stderr)
        return 1

    gz = GzClient(args.world)
    gz.wait_until_ready()

    rclpy.init()
    node = Node("test_place")
    latest = {}
    node.create_subscription(WorldState, "/world_state",
                             lambda m: latest.__setitem__("w", m), 10)

    def spin(seconds):
        end = time.time() + seconds
        while time.time() < end:
            rclpy.spin_once(node, timeout_sec=0.05)

    def manipulable(name: str) -> bool:
        """Table objects only.

        Gazebo's pose topic carries the robot's links too, and they move more
        than anything on the table. The first version of this tool picked the
        model that rose the most and duly reported that it had placed
        `panda_link3` 431 mm from the bin. Anything that reads 'which object
        moved' has to exclude the robot first.
        """
        stem = name[4:] if name.startswith("dyn_") else name
        return stem.split("_")[0].rstrip("0123456789") in (
            "cube", "cylinder", "bolt")

    def poses():
        return {n: world_to_link0(*p) for n, p in gz.read_poses().items()
                if manipulable(n)}

    pick_cli = ActionClient(node, Pick, "pick")
    place_cli = ActionClient(node, Place, "place")
    for cli, name in ((pick_cli, "pick"), (place_cli, "place")):
        if not cli.wait_for_server(timeout_sec=20.0):
            print(f"no {name} action server", file=sys.stderr)
            return 1

    def call(cli, goal, budget=180.0):
        fut = cli.send_goal_async(goal)
        end = time.time() + budget
        while not fut.done() and time.time() < end:
            rclpy.spin_once(node, timeout_sec=0.05)
        h = fut.result() if fut.done() else None
        if h is None or not h.accepted:
            return None
        rf = h.get_result_async()
        while not rf.done() and time.time() < end:
            rclpy.spin_once(node, timeout_sec=0.05)
        return rf.result().result if rf.done() else None

    # Wait FOR the world state rather than sleeping a fixed 3 s and hoping.
    # Discovery on a fresh node routinely takes longer than that, and the run
    # aborted with "no /world_state" against a cell that was publishing nine
    # objects perfectly well. Fixed sleeps producing false failures is a mistake
    # this project has already made once, in the dynamic-world test.
    deadline = time.time() + 25.0
    while time.time() < deadline and "w" not in latest:
        rclpy.spin_once(node, timeout_sec=0.1)
    if "w" not in latest:
        print("no /world_state after 25 s - is perception running?",
              file=sys.stderr)
        return 1

    # Re-place the scene first, with the benchmark's own reset. Without this the
    # bin starts holding whatever a previous run left in it, and "did this drop
    # land in the bin" is being asked of an already-full bin.
    import benchmark                                              # noqa: PLC0415
    stage = benchmark.World(args.world, seed=0)
    moved = stage.reset()
    print(f"scene reset: {moved} objects re-placed")
    spin(3.0)

    print(f"target: {recept.name} at ({recept.x:+.3f}, {recept.y:+.3f}), "
          f"inner half-width {recept.inner_half_width * 1000:.0f} mm, "
          f"rim z {recept.rim_z:.3f}")
    print(f"release: fingertips {(recept.release_height - scene.TCP_OFFSET_Z - recept.rim_z) * 1000:.0f} mm above the rim\n")

    claimed = landed = 0
    offsets = []
    home_cli = ActionClient(node, Home, "home")
    home_cli.wait_for_server(timeout_sec=10.0)

    for i in range(args.n):
        # Home between iterations. Without this a single failed lift wedges the
        # arm and every later pick fails identically at "approach above object",
        # which looks like a place result and is not one.
        call(home_cli, Home.Goal(), budget=90.0)
        spin(1.5)
        world = latest.get("w")
        cubes = [o for o in world.objects if o.class_name == "cube"]
        if not cubes:
            print(f"[{i+1}] no cube visible; stopping")
            break
        target = cubes[0]

        before = poses()
        pr = call(pick_cli, Pick.Goal(object_id=target.id))
        if pr is None or not pr.success:
            why = "no result" if pr is None else pr.message[:60]
            print(f"[{i+1}] pick failed ({why}) - not a place measurement")
            continue

        # which Gazebo model did we just lift? the one that moved up
        after_pick = poses()
        lifted, best = None, 0.02
        for n, p in after_pick.items():
            if n in before and p[2] - before[n][2] > best:
                lifted, best = n, p[2] - before[n][2]

        gl = Place.Goal()
        gl.target = args.bin_name
        rr = call(place_cli, gl)
        said = bool(rr is not None and rr.success)
        claimed += int(said)

        # A place that REFUSED is not a placement that missed. Conflating them
        # put an object still held at carry height into the "misses" list, 767 mm
        # from the bin, which makes the accuracy figure meaningless. Report the
        # refusal and its reason, and exclude it from the landing statistics.
        if not said:
            why = "no result" if rr is None else rr.message
            print(f"[{i+1}] place REFUSED: {why[:110]}")
            continue

        spin(2.5)                      # let it settle after the drop
        final = poses().get(lifted) if lifted else None
        if final is None:
            print(f"[{i+1}] place said {said}; could not identify the object")
            continue

        dx, dy = final[0] - recept.x, final[1] - recept.y
        # Exactly the benchmark's rule (bench_tasks.in_bin): laterally inside,
        # and BELOW the rim. An earlier version of this tool allowed rim + 20 mm
        # and so scored a stacked object as in the bin when the suite would not
        # - measuring a different thing from the one being explained.
        inside = (abs(dx) <= recept.inner_half_width
                  and abs(dy) <= recept.inner_half_width
                  and final[2] < recept.rim_z)
        landed += int(inside)
        offsets.append((dx, dy, final[2], inside, lifted))

        # Everything currently in the bin, not just the object just dropped.
        # Every place targets the exact bin centre, so the interesting failure
        # is a later object landing on an earlier one and knocking it out - and
        # that shows up as a DROP in this count, not as a failed place.
        now = poses()
        in_bin = [n for n, q in now.items()
                  if abs(q[0] - recept.x) <= recept.inner_half_width
                  and abs(q[1] - recept.y) <= recept.inner_half_width
                  and q[2] < recept.rim_z
                  and not n.startswith(("red_bin", "blue_bin", "table",
                                        "ground", "panda"))]
        print(f"[{i+1}] {lifted:<16} place said {str(said):<5} "
              f"landed dx {dx*1000:+6.1f} dy {dy*1000:+6.1f} z {final[2]:+.3f}  "
              f"{'IN' if inside else 'OUT'}   bin now holds {len(in_bin)}: "
              f"{', '.join(sorted(in_bin))}")

    n = len(offsets)
    print()
    if not n:
        print("no completed placements measured")
        return 1
    print(f"place reported success: {claimed}/{n}")
    print(f"object actually in the bin: {landed}/{n} ({100.0*landed/n:.0f}%)")
    radial = [math.hypot(dx, dy) for dx, dy, _, _, _ in offsets]
    print(f"radial offset from bin centre: mean {1000*sum(radial)/n:.0f} mm, "
          f"max {1000*max(radial):.0f} mm  (bin half-width "
          f"{recept.inner_half_width*1000:.0f} mm)")
    misses = [o for o in offsets if not o[3]]
    if misses:
        print("\nmisses:")
        for dx, dy, z, _, name in misses:
            print(f"  {name:<16} dx {dx*1000:+6.1f}  dy {dy*1000:+6.1f}  z {z:+.3f}")

    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

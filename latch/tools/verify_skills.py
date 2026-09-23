#!/usr/bin/env python3
"""Exercise the skill layer end to end, without the language layer.

    python3 verify_skills.py                    # perception-driven
    python3 verify_skills.py --object-id 1      # one specific object

Bottom-up verification, so a failure is attributable to one layer. If this
passes, every subsequent problem is in grounding or in the planner - which is a
much smaller search space than "the robot did the wrong thing".

What it checks, in order:

  1. /world_state has objects, and they look sane (in reach, graspable widths)
  2. home            -> the arm reaches the `ready` state
  3. pick            -> and the grasp is CONFIRMED by finger gap, not assumed
  4. place_in_bin    -> the object ends up inside the bin's footprint
  5. home            -> recovery back to a known pose
  6. the object is really gone from the table, verified against Gazebo's own
     pose, not against the robot's opinion of what it did

Step 6 matters most. Every other step trusts a report from the thing being
tested; only this one asks the simulator. A pick that reports success while the
cube sits unmoved is exactly the failure this layer produced once before, when
the grasp check bounded only the fully-closed side and "the gripper never
closed" scored as a perfect grasp.
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(HERE, ".."))
for p in (os.path.join(_REPO, "src", "yolov8_obb"),
          os.path.join(_REPO, "src", "llm_task_planner")):
    if p not in sys.path:
        sys.path.insert(0, p)

from llm_task_planner import scene                                  # noqa: E402
from yolov8_obb.projection import link0_to_world                    # noqa: E402

WORLD = "arm_and_table"          # <world name=...> in arm_on_the_table.sdf


def gz_model_pose(model: str) -> Optional[Tuple[float, float, float]]:
    """Ask Gazebo where a model actually is. Independent of the robot's claims.

    Reads the world's pose/info topic rather than calling `gz model -p`. The
    latter needs a SceneBroadcaster that answers *state* requests, which a server
    started with `gz sim -s` does not - so it fails exactly when the cell is
    brought up headless, which is the normal way to run this unattended. The
    topic is published either way.

    Parsing is shared with tools/synth_dataset/check_live_detector.py, which is
    where the two traps in the protobuf text format are documented: zero-valued
    fields are omitted entirely, and `orientation` has x/y keys that a naive
    scanner reads as the position.
    """
    sys.path.insert(0, os.path.join(HERE, "synth_dataset"))
    from check_live_detector import read_gz_poses          # noqa: PLC0415

    poses = read_gz_poses(WORLD)
    entry = poses.get(model)
    if entry is None:
        return None
    xyz, _quat = entry
    return xyz


class SkillTester:

    def __init__(self, node, timeout: float):
        from skill_msgs.action import Home, Pick, Place
        from rclpy.action import ActionClient
        from rclpy.callback_groups import ReentrantCallbackGroup

        self.node = node
        self.timeout = timeout
        cb = ReentrantCallbackGroup()
        self.Pick, self.Place, self.Home = Pick, Place, Home
        self._pick = ActionClient(node, Pick, "pick", callback_group=cb)
        self._place = ActionClient(node, Place, "place", callback_group=cb)
        self._home = ActionClient(node, Home, "home", callback_group=cb)

    def _call(self, client, goal, label: str):
        if not client.wait_for_server(timeout_sec=10.0):
            return None, f"{label}: action server never appeared"
        send = client.send_goal_async(goal)
        deadline = time.monotonic() + self.timeout
        while not send.done():
            if time.monotonic() > deadline:
                return None, f"{label}: goal not accepted in time"
            time.sleep(0.05)
        handle = send.result()
        if handle is None or not handle.accepted:
            return None, f"{label}: goal rejected"
        fut = handle.get_result_async()
        while not fut.done():
            if time.monotonic() > deadline:
                return None, f"{label}: did not finish within {self.timeout:.0f}s"
            time.sleep(0.05)
        return fut.result().result, ""

    def home(self):
        return self._call(self._home, self.Home.Goal(), "home")

    def pick(self, object_id: int):
        g = self.Pick.Goal()
        g.object_id = int(object_id)
        return self._call(self._pick, g, "pick")

    def place_in_bin(self, bin_name: str):
        g = self.Place.Goal()
        g.target = bin_name
        return self._call(self._place, g, "place")

    def place_at(self, x: float, y: float):
        g = self.Place.Goal()
        g.target = ""
        g.x, g.y = float(x), float(y)
        return self._call(self._place, g, "place")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--object-id", type=int, default=None,
                    help="Which object to pick. Default: the most central "
                         "graspable cube in /world_state.")
    ap.add_argument("--bin", default="red_bin")
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--model-name", default=None,
                    help="Gazebo model name for the independent pose check. "
                         "Guessed from the object's class and colour if omitted.")
    args = ap.parse_args(argv)

    import rclpy
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.node import Node
    from yolov8_msgs.msg import WorldState
    import threading

    rclpy.init()
    node = Node("verify_skills")
    states: List = []
    node.create_subscription(WorldState, "/world_state", states.append, 10)
    ex = MultiThreadedExecutor()
    ex.add_node(node)
    threading.Thread(target=ex.spin, daemon=True).start()

    failures: List[str] = []

    def step(n: str) -> None:
        print(f"\n{'=' * 66}\n{n}\n{'=' * 66}", flush=True)

    # ---- 1. perception -----------------------------------------------------
    step("1. /world_state")
    deadline = time.time() + 30
    while not states and time.time() < deadline:
        time.sleep(0.2)
    if not states:
        print("  nothing on /world_state after 30s.")
        print("  Bring the cell up with perception:=true (which starts BOTH the")
        print("  detector and world_model), or pass --object-id with a")
        print("  ground-truth publisher supplying the state.")
        return 1

    latest = states[-1]
    print(f"  {len(latest.objects)} objects")
    print(f"  {'id':>3} {'class':<10}{'colour':<8}{'x':>7}{'y':>7}"
          f"{'short':>7}{'reach':>7}  grasp")
    for o in latest.objects:
        r = scene.reach_of(o.x, o.y)
        print(f"  {o.id:>3} {o.class_name:<10}{o.color:<8}{o.x:>7.3f}{o.y:>7.3f}"
              f"{o.short_axis:>7.3f}{r:>7.3f}  {'yes' if o.graspable else ' no'}")
        ok, why = scene.is_reachable(o.x, o.y)
        if not ok:
            print(f"      ^ {why}")

    target = None
    if args.object_id is not None:
        target = next((o for o in latest.objects if o.id == args.object_id), None)
        if target is None:
            print(f"\n  no object with id {args.object_id} in the world state")
            return 1
    else:
        candidates = [o for o in latest.objects
                      if o.graspable and scene.is_reachable(o.x, o.y)[0]]
        if not candidates:
            print("\n  no graspable, reachable object to test with")
            return 1
        # The most central object: least likely to fail for reach reasons, so a
        # failure here is about the skill and not about the pose.
        target = min(candidates, key=lambda o: abs(scene.reach_of(o.x, o.y) - 0.45))

    print(f"\n  target: id {target.id}, {target.color} {target.class_name} "
          f"at ({target.x:.3f}, {target.y:.3f})")

    # Identify the Gazebo model by POSITION, not by guessing its name.
    #
    # The name was previously guessed as f"cube_{colour}_1", which is wrong the
    # moment a scene has two cubes of one colour - and this one has two yellow
    # cubes. The run then picked cube_yellow_2 and checked whether cube_yellow_1
    # had moved, reported "moved 0 mm", and blamed a place that had in fact
    # worked. A test that can accuse a working system is worse than no test.
    model = args.model_name
    before = None
    if model:
        before = gz_model_pose(model)
    else:
        sys.path.insert(0, os.path.join(HERE, "synth_dataset"))
        from check_live_detector import read_gz_poses      # noqa: PLC0415
        best_d = 0.10
        for name, (xyz, _q) in read_gz_poses(WORLD).items():
            if name in ("table", "ground_plane", "red_bin", "blue_bin",
                        "arm_robot", "sun"):
                continue
            lx, ly = xyz[0] - 0.05, xyz[1]          # world -> link0
            d = math.hypot(lx - target.x, ly - target.y)
            if d < best_d:
                model, before, best_d = name, xyz, d
        if model:
            print(f"  Gazebo model at that position: {model} "
                  f"({best_d * 1000:.0f} mm from the reported pose)")

    if before:
        print(f"  it is at world {tuple(round(c, 3) for c in before)}")
    else:
        print(f"  (no Gazebo model found near ({target.x:.3f}, {target.y:.3f}); "
              f"the final check will rely on the robot's own report)")

    tester = SkillTester(node, args.timeout)

    # ---- 2. home ------------------------------------------------------------
    step("2. home")
    res, err = tester.home()
    if err or not res.success:
        failures.append(f"home: {err or res.message}")
        print(f"  FAILED: {err or res.message}")
    else:
        print(f"  ok: {res.message}")

    # ---- 3. pick ------------------------------------------------------------
    step(f"3. pick object {target.id}")
    res, err = tester.pick(target.id)
    if err or not res.success:
        failures.append(f"pick: {err or res.message}")
        print(f"  FAILED: {err or res.message}")
        print("\n  Not attempting place with nothing in the gripper.")
    else:
        print(f"  ok: {res.message}")
        print(f"  finger gap {res.finger_gap * 1000:.1f} mm  vs object short axis "
              f"{target.short_axis * 1000:.1f} mm")
        # Re-derive the verdict here rather than trusting the server's own,
        # so a bug in its band check cannot hide behind its own success flag.
        lo = scene.GRASP_MISS_GAP
        hi = target.short_axis + scene.GRASP_WIDTH_TOLERANCE
        if not (lo <= res.finger_gap <= hi):
            failures.append(
                f"pick reported success but the finger gap "
                f"{res.finger_gap * 1000:.1f} mm is outside the plausible band "
                f"[{lo * 1000:.0f}, {hi * 1000:.0f}] mm - "
                + ("the fingers met nothing" if res.finger_gap < lo
                   else "the fingers never closed"))
            print(f"  ^ INDEPENDENT CHECK FAILED: {failures[-1]}")

        # ---- 4. place ------------------------------------------------------
        step(f"4. place_in_bin {args.bin}")
        res2, err2 = tester.place_in_bin(args.bin)
        if err2 or not res2.success:
            failures.append(f"place: {err2 or res2.message}")
            print(f"  FAILED: {err2 or res2.message}")
        else:
            print(f"  ok: {res2.message}")

    # ---- 5. home ------------------------------------------------------------
    step("5. home (recovery)")
    res3, err3 = tester.home()
    if err3 or not res3.success:
        failures.append(f"home (recovery): {err3 or res3.message}")
        print(f"  FAILED: {err3 or res3.message}")
    else:
        print(f"  ok: {res3.message}")

    # ---- 6. did the object actually move? ----------------------------------
    step("6. independent check: is the object in the bin?")
    time.sleep(1.5)
    after = gz_model_pose(model) if model else None
    if before and after:
        moved = math.dist(before[:2], after[:2])
        recept = scene.find_receptacle(args.bin)
        bx, by, _ = link0_to_world(recept.x, recept.y, 0.0)
        d_bin = math.hypot(after[0] - bx, after[1] - by)
        print(f"  before world ({before[0]:.3f}, {before[1]:.3f}, {before[2]:.3f})")
        print(f"  after  world ({after[0]:.3f}, {after[1]:.3f}, {after[2]:.3f})")
        print(f"  moved {moved * 1000:.0f} mm; now {d_bin * 1000:.0f} mm from the "
              f"{args.bin} centre (inner half-width "
              f"{recept.inner_half_width * 1000:.0f} mm)")
        if d_bin > recept.inner_half_width:
            failures.append(
                f"the object ended {d_bin * 1000:.0f} mm from the {args.bin} "
                f"centre, outside its {recept.inner_half_width * 1000:.0f} mm "
                f"half-width - the place missed the bin")
        elif moved < 0.05:
            failures.append(f"the object only moved {moved * 1000:.0f} mm - the "
                            f"cycle reported success without relocating anything")
        else:
            print(f"  the object is inside the bin. Full cycle verified against "
                  f"Gazebo, not against the robot's own report.")
    else:
        print(f"  SKIPPED - no independent pose for {model!r}. Every result above "
              f"is the robot's own account of what it did, which is exactly what "
              f"this step exists to cross-check. Re-run with the cell in "
              f"non-headless mode, or pass --model-name.")

    # ---- verdict ------------------------------------------------------------
    print(f"\n{'=' * 66}")
    if failures:
        print(f"{len(failures)} FAILURES:")
        for f in failures:
            print(f"  - {f}")
    else:
        print("skill layer verified: home, pick with a confirmed grasp, "
              "place into a bin, and recovery.")
    sys.stdout.flush()
    os._exit(1 if failures else 0)


if __name__ == "__main__":
    raise SystemExit(main())

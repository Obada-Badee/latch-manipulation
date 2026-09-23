#!/usr/bin/env python3
"""A place with nothing in the hand must be refused, not reported as success.

    python3 tools/test_empty_place.py        (against a running cell)

Regression test for the loop that made the cap-60 runs slow and every
multi-object trial before them untrustworthy. `place` checks the finger gap
and rejects a CLOSED empty jaw - a missed grasp. It cannot reject an OPEN empty
jaw, because a held blue cube (61 mm) and the open width (60 mm) read the same.
So after the first release, every later `place_in_bin` passed the check, flew
to the bin, opened on nothing, and returned "SUCCESS: released the held
object". One trial was told that eleven times with the table untouched, and
looped on it until the step cap.

The fix is state the server already has and the sensor never will: it did the
grasping and the releasing, so it knows whether it is holding anything. The
sequence below is the exact one that failed.
"""

import sys, time, os
sys.path.insert(0, os.path.expanduser("~/moveit2_obb/tools"))
sys.path.insert(0, os.path.expanduser("~/moveit2_obb/tools/synth_dataset"))
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from skill_msgs.action import Home, Pick, Place
from yolov8_msgs.msg import WorldState

rclpy.init()
node = Node("test_empty_place")
latest = {}
node.create_subscription(WorldState, "/world_state", lambda m: latest.__setitem__("w", m), 10)
pick_cli, place_cli, home_cli = (ActionClient(node, Pick, "pick"), ActionClient(node, Place, "place"), ActionClient(node, Home, "home"))
for cli, n in ((pick_cli, "pick"), (place_cli, "place"), (home_cli, "home")):
    assert cli.wait_for_server(timeout_sec=20.0), f"no {n} server"

def call(cli, goal, budget=180.0):
    fut = cli.send_goal_async(goal); end = time.time() + budget
    while not fut.done() and time.time() < end: rclpy.spin_once(node, timeout_sec=0.05)
    h = fut.result()
    rf = h.get_result_async()
    while not rf.done() and time.time() < end: rclpy.spin_once(node, timeout_sec=0.05)
    return rf.result().result

dl = time.time() + 25
while time.time() < dl and "w" not in latest: rclpy.spin_once(node, timeout_sec=0.1)
assert "w" in latest, "no world state"

fails = []
def check(label, res, want_success):
    ok = bool(res.success) == want_success
    print(("PASS" if ok else "FAIL"), f"{label:34s} success={res.success}  {res.message[:90]}")
    if not ok: fails.append(label)

# 0. place with NOTHING ever picked -> must refuse
g = Place.Goal(); g.target = "blue_bin"
check("place before any pick", call(place_cli, g), False)

# 1. pick a cube
call(home_cli, Home.Goal())
cubes = [o for o in latest["w"].objects if o.class_name == "cube"]
assert cubes, "no cube visible"
pr = call(pick_cli, Pick.Goal(object_id=cubes[0].id))
check("pick a cube", pr, True)
if pr.success:
    # 2. place it -> success
    g = Place.Goal(); g.target = "blue_bin"
    check("place the held cube", call(place_cli, g), True)
    # 3. place AGAIN with the jaw now open and empty -> must refuse (the bug)
    g = Place.Goal(); g.target = "blue_bin"
    check("place again, hand empty (THE BUG)", call(place_cli, g), False)
    # 4. and a third, to be sure it stays refused
    g = Place.Goal(); g.target = "red_bin"
    check("place a third time, other bin", call(place_cli, g), False)

call(home_cli, Home.Goal())
print()
print("ALL PASSED" if not fails else f"FAILED: {fails}")
rclpy.shutdown()
sys.exit(1 if fails else 0)

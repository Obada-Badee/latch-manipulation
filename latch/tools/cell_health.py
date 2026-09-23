#!/usr/bin/env python3
"""One-shot health check of the running cell.

    python3 tools/cell_health.py

Reports controllers, image rate, detector output and world state in one place,
so bring-up problems are diagnosed from one command instead of four shell
invocations whose quoting has to survive two shells.

Exists because the controller spawner race (WP6) is real and recurring: the
cell can come up with `joint_state_broadcaster` inactive or unloaded, and every
downstream symptom then points somewhere else - grasp verification has no
signal, MoveIt has no start state, and the planner reports it cannot reach
anything. That reads as a capability failure and is an infrastructure fault.
"""

from __future__ import annotations

import sys
import time

import rclpy
from rclpy.node import Node


def _gz_is_stepping():
    """Ask Gazebo directly whether it is advancing. None if it does not answer.

    Sampled twice rather than trusting the `paused` field alone: a server
    wedged inside a step reports the same thing as one that is running, and
    the iteration count is what separates them.
    """
    import re
    import subprocess

    def iterations():
        try:
            out = subprocess.run(
                ["gz", "topic", "-e", "-t", "/stats", "-n", "1"],
                capture_output=True, text=True, timeout=10).stdout
        except Exception:
            return None
        m = re.search(r"iterations:\s*(\d+)", out)
        return int(m.group(1)) if m else None

    first = iterations()
    if first is None:
        return None
    time.sleep(1.5)
    second = iterations()
    if second is None:
        return None
    return second > first


def main() -> int:
    from controller_manager_msgs.srv import ListControllers
    from sensor_msgs.msg import Image, JointState
    from yolov8_msgs.msg import WorldState, Yolov8Inference

    rclpy.init()
    node = Node("cell_health")
    problems = []

    client = node.create_client(ListControllers,
                                "/controller_manager/list_controllers")
    print("controllers:")
    if not client.wait_for_service(timeout_sec=15.0):
        print("  controller_manager NOT AVAILABLE")
        problems.append("controller_manager down")
    else:
        fut = client.call_async(ListControllers.Request())
        rclpy.spin_until_future_complete(node, fut, timeout_sec=15.0)
        res = fut.result()
        states = {c.name: c.state for c in (res.controller if res else [])}
        for name in ("joint_state_broadcaster", "panda_arm_controller",
                     "panda_hand_controller"):
            state = states.get(name, "NOT LOADED")
            flag = "" if state == "active" else "   <-- PROBLEM"
            print(f"  {name:<26} {state}{flag}")
            if state != "active":
                problems.append(f"{name} is {state}")

    counts = {"image": 0, "inference": 0, "joints": 0}
    latest = {}
    stamps = []
    node.create_subscription(Image, "/image_raw",
                             lambda m: counts.__setitem__("image", counts["image"] + 1), 10)
    node.create_subscription(Yolov8Inference, "/Yolov8_Inference",
                             lambda m: counts.__setitem__("inference", counts["inference"] + 1), 10)
    def on_joints(m):
        counts["joints"] += 1
        stamps.append(m.header.stamp.sec + m.header.stamp.nanosec * 1e-9)

    node.create_subscription(JointState, "/joint_states", on_joints, 10)
    node.create_subscription(WorldState, "/world_state",
                             lambda m: latest.__setitem__("world", m), 10)

    window = 6.0
    start = time.time()
    while time.time() - start < window:
        rclpy.spin_once(node, timeout_sec=0.05)
    elapsed = time.time() - start

    print(f"\ntopics (over {elapsed:.0f} s):")
    for key, label in (("image", "/image_raw"), ("inference", "/Yolov8_Inference"),
                       ("joints", "/joint_states")):
        rate = counts[key] / elapsed
        flag = "" if rate > 0.5 else "   <-- SILENT"
        print(f"  {label:<22} {rate:5.1f} Hz{flag}")
        if rate <= 0.5:
            problems.append(f"{label} silent")

    # Is sim time advancing, and if not, WHY?
    #
    # These are two different faults with the same symptom, and the first
    # version of this check conflated them: it read frozen /joint_states stamps
    # and announced "THE WORLD IS PAUSED". Gazebo was stepping at RTF 1.0 the
    # whole time and what had actually died was the ROS side - so the check sent
    # the diagnosis in exactly the wrong direction. Gazebo has genuinely paused
    # itself before, so both faults are real and the check has to tell them
    # apart rather than guess.
    print("")
    print("sim clock:")
    gz_stepping = _gz_is_stepping()
    advanced = None
    if len(stamps) < 2:
        print("  no /joint_states stamps - cannot tell from the ROS side")
    else:
        advanced = stamps[-1] - stamps[0]
        print("  /joint_states stamps advanced %.2f s in %.0f s real"
              % (advanced, elapsed))
    if gz_stepping is None:
        print("  gazebo /stats did not answer - is the server up?")
        problems.append("gazebo /stats did not answer")
    else:
        print("  gazebo itself: %s" % ("STEPPING" if gz_stepping else "NOT stepping"))

    if advanced is not None and advanced <= 0.05:
        if gz_stepping:
            print("  <-- ROS sim time frozen while GAZEBO IS RUNNING: the bridge")
            print("      or controller side has died, not the world")
            problems.append("sim time frozen on the ROS side while gazebo runs "
                            "- restart the cell")
        else:
            print("  <-- the world itself is not stepping (paused or hung)")
            problems.append("the world is not stepping - paused or hung")

    # The skills. Every other check here can pass while skill_server is dead:
    # it died once at launch with "Unable to configure planning scene monitor",
    # this tool reported HEALTHY, and a 20-trial benchmark run then produced
    # nothing but "the pick skill is not available".
    print("")
    print("skills:")
    from rclpy.action import ActionClient
    from skill_msgs.action import Home, Pick, Place
    for action, label in ((Pick, "pick"), (Place, "place"), (Home, "home")):
        client = ActionClient(node, action, label)
        up = client.wait_for_server(timeout_sec=8.0)
        print("  %-22s %s" % ("/" + label, "up" if up else "MISSING   <-- PROBLEM"))
        if not up:
            problems.append("the '%s' action server is missing - "
                            "skill_server is not running" % label)
        client.destroy()

    world = latest.get("world")
    print("\nworld state:")
    if world is None:
        print("  NOT PUBLISHING")
        problems.append("/world_state silent")
    else:
        print(f"  {len(world.objects)} objects")
        for o in world.objects:
            print(f"    id {o.id:<4} {o.class_name:<9} {o.color:<7} "
                  f"({o.x:+.3f}, {o.y:+.3f})  yaw {o.yaw:+.2f}")
        if not world.objects:
            problems.append("/world_state reports an empty table")

    node.destroy_node()
    rclpy.shutdown()

    print()
    if problems:
        print("UNHEALTHY:")
        for p in problems:
            print(f"  * {p}")
        return 1
    print("HEALTHY - safe to run the benchmark")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

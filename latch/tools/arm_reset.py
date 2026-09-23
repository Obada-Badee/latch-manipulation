#!/usr/bin/env python3
"""Put the arm back to `ready` and open the gripper, whatever state it is in.

    python3 tools/arm_reset.py

Tries the `home` action first. If MoveIt refuses - which is the case that
matters - it publishes a `ready` trajectory straight to the arm controller,
bypassing MoveIt entirely.

That escape hatch exists because MoveIt cannot recover the one state that most
needs recovering: an interrupted trajectory can leave a joint marginally outside
its limit (joint6 was found at 3.75283 against a limit of 3.75250), and MoveIt
then refuses *every* plan, including the plan back to `ready`. The controller has
no such objection.

This logic previously lived only inside `bench_planner.PlannerHarness._rescue`,
so unwedging the arm by hand meant constructing the whole benchmark harness -
which starts the LLM planner and spends tokens to move a joint. It is also
verified here rather than assumed: the original published the trajectory, slept,
and returned True. It reports success only if /joint_states says the arm arrived.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from typing import List, Optional, Sequence

HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(_ROOT, "src", "llm_task_planner"))

# One definition of `ready`, shared with the skill server rather than copied.
# This tool and skill_server's own recovery must agree about where home is, or
# "the arm went home" means two different things depending on who asked.
from llm_task_planner.scene import (ARM_JOINT_NAMES as JOINTS,  # noqa: E402
                                    READY_JOINTS as READY)

#: Per-joint tolerance for calling the arm "arrived". Generous: the point is to
#: get out of a wedged state, not to hit a pose precisely.
ARRIVED_TOL_RAD = 0.10


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-home", action="store_true",
                    help="go straight to the direct trajectory")
    ap.add_argument("--open-gap", type=float, default=0.06,
                    help="jaw gap to open to, metres")
    args = ap.parse_args(argv)

    import rclpy
    from rclpy.action import ActionClient
    from rclpy.node import Node
    from builtin_interfaces.msg import Duration
    from control_msgs.action import GripperCommand
    from sensor_msgs.msg import JointState
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

    rclpy.init()
    node = Node("arm_reset")
    latest: dict = {}

    def on_joints(msg: JointState) -> None:
        latest["js"] = msg

    node.create_subscription(JointState, "/joint_states", on_joints, 10)

    def spin(seconds: float) -> None:
        end = time.time() + seconds
        while time.time() < end:
            rclpy.spin_once(node, timeout_sec=0.05)

    def arm_positions() -> Optional[List[float]]:
        msg = latest.get("js")
        if msg is None:
            return None
        try:
            return [msg.position[msg.name.index(n)] for n in JOINTS]
        except (ValueError, IndexError):
            return None

    def arrived() -> bool:
        pos = arm_positions()
        if pos is None:
            return False
        return all(abs(p - w) <= ARRIVED_TOL_RAD for p, w in zip(pos, READY))

    spin(3.0)
    if arm_positions() is None:
        print("no /joint_states - is the cell running?", file=sys.stderr)
        return 1

    start = arm_positions()
    print("start:  " + "  ".join(f"{p:+.3f}" for p in start))

    # Open the gripper first: a closed jaw carrying something makes every later
    # plan harder, and an open jaw is the state a trial should begin in.
    grip = ActionClient(node, GripperCommand, "/panda_hand_controller/gripper_cmd")
    if grip.wait_for_server(timeout_sec=5.0):
        goal = GripperCommand.Goal()
        goal.command.position = float(args.open_gap) / 2.0
        goal.command.max_effort = 50.0
        grip.send_goal_async(goal)
        spin(2.0)
        print(f"gripper: commanded open to {args.open_gap * 1000:.0f} mm")
    else:
        print("gripper: controller unavailable")

    if not args.skip_home:
        from skill_msgs.action import Home
        home = ActionClient(node, Home, "home")
        if home.wait_for_server(timeout_sec=10.0):
            fut = home.send_goal_async(Home.Goal())
            deadline = time.time() + 60.0
            while not fut.done() and time.time() < deadline:
                rclpy.spin_once(node, timeout_sec=0.05)
            handle = fut.result() if fut.done() else None
            if handle is not None and handle.accepted:
                rfut = handle.get_result_async()
                while not rfut.done() and time.time() < deadline:
                    rclpy.spin_once(node, timeout_sec=0.05)
                res = rfut.result().result if rfut.done() else None
                if res is not None and res.success:
                    print("home:   succeeded")
                    spin(1.0)
                    if arrived():
                        print("\nOK - arm is at ready")
                        node.destroy_node()
                        rclpy.shutdown()
                        return 0
                    print("home:   reported success but the arm is NOT at ready")
                else:
                    why = "no result" if res is None else res.message
                    print(f"home:   refused ({why})")

    print("rescue: publishing `ready` straight to the arm controller")
    pub = node.create_publisher(JointTrajectory,
                                "/panda_arm_controller/joint_trajectory", 10)

    # Publishing before the controller has matched silently discards the
    # message - which is how an earlier rescue appeared to succeed while the arm
    # had not moved at all.
    deadline = time.time() + 5.0
    while pub.get_subscription_count() == 0:
        if time.time() > deadline:
            print("rescue: no subscriber on the trajectory topic", file=sys.stderr)
            node.destroy_node()
            rclpy.shutdown()
            return 1
        spin(0.1)

    msg = JointTrajectory()
    msg.joint_names = list(JOINTS)
    point = JointTrajectoryPoint()
    point.positions = list(READY)
    point.time_from_start = Duration(sec=5)
    msg.points = [point]
    pub.publish(msg)

    end = time.time() + 20.0
    while time.time() < end and not arrived():
        spin(0.5)

    final = arm_positions()
    print("end:    " + "  ".join(f"{p:+.3f}" for p in final))
    worst = max(abs(p - w) for p, w in zip(final, READY))
    moved = max(abs(a - b) for a, b in zip(final, start))

    node.destroy_node()
    rclpy.shutdown()

    if arrived():
        print(f"\nOK - arm is at ready (worst joint error {worst:.3f} rad)")
        return 0
    print(f"\nFAILED - worst joint error {worst:.3f} rad, "
          f"arm moved {math.degrees(moved):.1f} deg during the attempt")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

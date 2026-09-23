#!/usr/bin/env python3
"""Can `home` recover a wedged arm, or only a healthy one?

    python3 tools/test_home_rescue.py

Two cases, and the second is the one that matters:

  1. `home` from a normal posture - must still go through MoveIt as before
  2. `home` from a posture MoveIt refuses to plan from - must recover anyway

Case 2 is reproduced rather than waited for. An interrupted trajectory can leave
a joint marginally outside its limit, and MoveIt then refuses every plan
including the plan back to `ready`. Rather than hoping that recurs, this drives
joint 7 just past its 2.8973 rad limit on purpose, which is the same state
observed in the wild at 2.898.

Why it matters: in a benchmark trial a `pick` failed at the lift, `home` was
refused, and the wedged arm then sat over the target so perception could not see
it either. The planner behaved correctly throughout and the trial was
unrecoverable from the first refusal. If case 2 fails, that trial's failure mode
is still live.
"""

from __future__ import annotations

import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(_ROOT, "src", "llm_task_planner"))

from llm_task_planner.scene import ARM_JOINT_NAMES, READY_JOINTS  # noqa: E402

#: Just past panda_joint7's +2.8973 limit - the wedge, reproduced.
WEDGE_JOINT = 6
WEDGE_VALUE = 2.9050
ARRIVED_TOL = 0.10


def main() -> int:
    import rclpy
    from rclpy.action import ActionClient
    from rclpy.node import Node
    from builtin_interfaces.msg import Duration
    from sensor_msgs.msg import JointState
    from skill_msgs.action import Home
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

    rclpy.init()
    node = Node("test_home_rescue")
    latest = {}
    node.create_subscription(JointState, "/joint_states",
                             lambda m: latest.__setitem__("js", m), 10)

    def spin(seconds: float) -> None:
        end = time.time() + seconds
        while time.time() < end:
            rclpy.spin_once(node, timeout_sec=0.05)

    def arm():
        msg = latest.get("js")
        if msg is None:
            return None
        try:
            return [msg.position[msg.name.index(n)] for n in ARM_JOINT_NAMES]
        except (ValueError, IndexError):
            return None

    def at_ready() -> bool:
        pos = arm()
        return pos is not None and all(
            abs(p - w) <= ARRIVED_TOL for p, w in zip(pos, READY_JOINTS))

    pub = node.create_publisher(JointTrajectory,
                                "/panda_arm_controller/joint_trajectory", 10)
    home = ActionClient(node, Home, "home")

    spin(3.0)
    if arm() is None:
        print("no /joint_states - is the cell running?", file=sys.stderr)
        return 1
    if not home.wait_for_server(timeout_sec=15.0):
        print("no home action server", file=sys.stderr)
        return 1

    def call_home(label: str):
        print(f"\n--- {label}")
        fut = home.send_goal_async(Home.Goal())
        deadline = time.time() + 90.0
        while not fut.done() and time.time() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
        handle = fut.result() if fut.done() else None
        if handle is None or not handle.accepted:
            print("  goal rejected")
            return None
        rfut = handle.get_result_async()
        while not rfut.done() and time.time() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
        res = rfut.result().result if rfut.done() else None
        if res is None:
            print("  no result within the timeout")
            return None
        print(f"  success = {res.success}")
        print(f"  message = {res.message[:200]}")
        spin(1.0)
        print(f"  arm at ready: {at_ready()}")
        return res

    def drive(positions, secs=4):
        msg = JointTrajectory()
        msg.joint_names = list(ARM_JOINT_NAMES)
        pt = JointTrajectoryPoint()
        pt.positions = list(positions)
        pt.time_from_start = Duration(sec=secs)
        msg.points = [pt]
        pub.publish(msg)
        spin(secs + 2.0)

    failures = []

    # --- case 1: normal posture, MoveIt should handle it
    drive(READY_JOINTS)
    res = call_home("case 1: home from a healthy posture")
    if res is None or not res.success:
        failures.append("case 1: home failed from a healthy posture")
    elif "without collision checking" in res.message:
        failures.append("case 1: fell back to the rescue when MoveIt should "
                        "have planned it - the fallback is firing too eagerly")
    elif not at_ready():
        # This is not a duplicate of the success check. MoveIt reaches `ready`
        # by name from the SRDF; the recovery path drives scene.READY_JOINTS
        # directly. If those two disagree, this is the only thing that notices -
        # and they did disagree, by 0.485 rad on joint 2.
        failures.append("case 1: MoveIt reported it reached `ready`, but the "
                        "arm is not at scene.READY_JOINTS - the SRDF named "
                        "state and our joint list have drifted apart")

    # --- case 2: wedge joint 7 past its limit, then ask for home
    wedged = list(READY_JOINTS)
    wedged[WEDGE_JOINT] = WEDGE_VALUE
    print(f"\ndriving panda_joint{WEDGE_JOINT + 1} to {WEDGE_VALUE} "
          f"(limit 2.8973)...")
    drive(wedged)
    pos = arm()
    print(f"  joint{WEDGE_JOINT + 1} now at {pos[WEDGE_JOINT]:.5f}"
          if pos else "  no joint state")
    if pos is not None and pos[WEDGE_JOINT] <= 2.8973:
        print("  NOTE: the controller clamped inside the limit, so MoveIt may "
              "still plan. Case 2 then proves less than intended.")

    res = call_home("case 2: home from a wedged posture")
    if res is None or not res.success:
        failures.append("case 2: home could not recover a wedged arm - the "
                        "benchmark failure mode is still live")
    elif not at_ready():
        failures.append("case 2: home reported success but the arm is not at "
                        "ready")

    node.destroy_node()
    rclpy.shutdown()

    print()
    if failures:
        for f in failures:
            print(f"FAIL  {f}")
        return 1
    print("PASS - home recovers both a healthy and a wedged arm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Is the live fingertip transform readable from TF? Move the arm and watch.

    python3 tools/tf_probe.py

Exists because a trace harness read `panda_link0 -> panda_leftfinger` as the
identical value for 47 seconds while the arm swung through 60 degrees, and the
next tool to be built (the failure injector) was about to be designed on top of
that same lookup. "TF is broken" and "my node was not spun enough" call for
completely different designs, so it is worth ten seconds to know which it is.

This moves joint 4 through a small arc via the controller and prints the
fingertip height as it goes. If the number moves, TF is fine and the earlier
harness was at fault.
"""

from __future__ import annotations

import sys
import time

READY = [0.0, -0.30, 0.0, -2.00, 0.0, 1.75, 0.79]
JOINTS = [f"panda_joint{i}" for i in range(1, 8)]


def main() -> int:
    import rclpy
    from rclpy.node import Node
    from builtin_interfaces.msg import Duration
    from tf2_ros import Buffer, TransformListener
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

    rclpy.init()
    node = Node("tf_probe")
    buf = Buffer()
    # spin_thread=True: the listener runs its own executor instead of relying on
    # the caller to spin often enough. This is the difference under test.
    TransformListener(buf, node, spin_thread=True)

    pub = node.create_publisher(JointTrajectory,
                                "/panda_arm_controller/joint_trajectory", 10)
    deadline = time.time() + 5.0
    while pub.get_subscription_count() == 0:
        if time.time() > deadline:
            print("no subscriber on the trajectory topic", file=sys.stderr)
            return 1
        time.sleep(0.1)

    def tip_z():
        try:
            t = buf.lookup_transform("panda_link0", "panda_leftfinger",
                                     rclpy.time.Time())
            return t.transform.translation.z
        except Exception as exc:                                   # noqa: BLE001
            return f"<{type(exc).__name__}>"

    def send(positions, secs):
        msg = JointTrajectory()
        msg.joint_names = list(JOINTS)
        pt = JointTrajectoryPoint()
        pt.positions = list(positions)
        pt.time_from_start = Duration(sec=secs)
        msg.points = [pt]
        pub.publish(msg)

    print("settling...")
    for _ in range(10):
        time.sleep(0.1)
    print(f"before move: fingertip z = {tip_z()}")

    lowered = list(READY)
    lowered[3] = -2.40           # joint4: drops the hand noticeably
    send(lowered, 3)
    for _ in range(16):
        time.sleep(0.25)
        print(f"  z = {tip_z()}")

    send(READY, 3)
    time.sleep(3.5)
    print(f"back at ready: fingertip z = {tip_z()}")

    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

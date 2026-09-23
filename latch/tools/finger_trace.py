#!/usr/bin/env python3
"""Record the finger gap and fingertip height continuously during a grasp.

    python3 tools/finger_trace.py --seconds 300 --out ~/finger_trace.csv

Run this alongside a pick. It answers what a pass/fail result cannot: did the
jaw ever *stall*, and where were the pads when it did.

Columns: t, finger_joint, gap, then panda_joint1..7.

Everything comes from /joint_states, on purpose.

The first version sampled object poses through the gz transport client on every
tick; that is far slower than the requested rate and the run never produced a
file at all. The second read the fingertip height from TF - and TF returned the
same value, 0.5308, for 47 s straight while the arm demonstrably moved, so every
conclusion drawn from it would have been wrong. /joint_states is the one source
already proven live here: the finger value in the very same message changes.
Arm joints answer "did the arm actually descend" directly, with no frame
plumbing to be wrong about.

Rows are flushed as they are taken, so a killed run still leaves usable data.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Optional, Sequence


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=300.0)
    ap.add_argument("--rate", type=float, default=20.0)
    ap.add_argument("--out", default=os.path.expanduser("~/finger_trace.csv"))
    args = ap.parse_args(argv)

    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState

    rclpy.init()
    node = Node("finger_trace")

    arm_names = [f"panda_joint{i}" for i in range(1, 8)]
    state: dict = {"finger": None, "arm": None}

    def on_joints(msg: JointState) -> None:
        try:
            state["finger"] = msg.position[msg.name.index("panda_finger_joint1")]
        except (ValueError, IndexError):
            pass
        try:
            state["arm"] = [msg.position[msg.name.index(n)] for n in arm_names]
        except (ValueError, IndexError):
            pass

    node.create_subscription(JointState, "/joint_states", on_joints, 50)

    path = os.path.expanduser(args.out)
    period = 1.0 / args.rate
    t0 = time.time()
    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("t,finger_joint,gap," + ",".join(arm_names) + "\n")
        while time.time() - t0 < args.seconds:
            rclpy.spin_once(node, timeout_sec=0.01)
            j = state["finger"]
            arm = state["arm"]
            cells = [f"{time.time() - t0:.2f}",
                     "" if j is None else f"{j:.5f}",
                     "" if j is None else f"{2 * j:.5f}"]
            cells += ["" for _ in arm_names] if arm is None else \
                     [f"{a:.4f}" for a in arm]
            fh.write(",".join(cells) + "\n")
            fh.flush()
            n += 1
            time.sleep(period)

    print(f"wrote {n} rows to {path}")
    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

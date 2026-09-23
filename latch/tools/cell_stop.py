#!/usr/bin/env python3
"""Stop every process belonging to the simulation cell, and verify it.

    python3 tools/cell_stop.py

Written as a script rather than a shell one-liner for two reasons, both learned
the hard way.

**Coverage.** A partial kill is worse than none. `parameter_bridge` was missed by
an earlier cleanup, so two image bridges ended up running against one Gazebo -
load average 8.3, `/image_raw` down from ~30 Hz to 2.2 Hz, and the detector
starved into silence. The symptom looked like a broken detector; the cause was a
leftover process from a previous launch.

**Quoting.** These commands pass through Git Bash and then WSL bash, and `$1`,
`$?` and friends are eaten somewhere in between - producing `kill: invalid
argument c` and, worse, silently vacuous checks. A file has no such problem.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time

#: Matched against the full command line. Order does not matter; anything that
#: belongs to the cell belongs here.
PATTERNS = (
    "gz sim",
    "parameter_bridge",
    "ros_gz_bridge",
    "yolov8_obb_publisher",
    "world_model.py",
    "skill_server",
    "move_group",
    "robot_state_publisher",
    "controller_manager",
    "spawner",
    "ros2 launch panda_moveit_config",
    "gz_tools_vendor",
    # The thing DRIVING the cell belongs here too. Without it, "stop everything"
    # stopped the simulator and left the benchmark running against a dead world,
    # which is a more confusing state than either half alone.
    "tools/benchmark.py",
    "bench_induce.py",
)


def cell_processes():
    """(pid, cmdline) for every live process matching a pattern, minus ourselves."""
    out = subprocess.run(["ps", "-eo", "pid=,args="], capture_output=True,
                         text=True).stdout
    me = os.getpid()
    found = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        pid_str, _, args = line.partition(" ")
        try:
            pid = int(pid_str)
        except ValueError:
            continue
        if pid == me or "cell_stop" in args:
            continue
        if any(p in args for p in PATTERNS):
            found.append((pid, args))
    return found


def main() -> int:
    procs = cell_processes()
    if not procs:
        print("nothing running")
        return 0

    print(f"stopping {len(procs)} process(es):")
    for pid, args in procs:
        print(f"  {pid}  {args[:96]}")

    for sig, wait in ((signal.SIGTERM, 5.0), (signal.SIGKILL, 8.0)):
        for pid, _ in cell_processes():
            try:
                os.kill(pid, sig)
            except (ProcessLookupError, PermissionError):
                pass
        deadline = time.time() + wait
        while time.time() < deadline:
            if not cell_processes():
                break
            time.sleep(0.5)
        if not cell_processes():
            break

    left = cell_processes()
    if left:
        print(f"\nSTILL RUNNING ({len(left)}):")
        for pid, args in left:
            print(f"  {pid}  {args[:96]}")
        return 1
    print("\nall stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

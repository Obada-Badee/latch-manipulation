#!/usr/bin/env python3
"""Does the benchmark's scene reset bring back objects that left the table?

    python3 tools/test_scene_reset.py

Trials are only independent if every trial starts from a complete scene. They
were not. The reset delegated to a shuffle that skipped anything outside a
height window around the table, so an object knocked onto the floor - or left
in a bin - was excluded from that reset and every reset after it. A 75-trial
run lost all three bolts partway through and voided seven trials with "needs 1
any bolt, scene has 0", which reads as a capability result and is a scene that
had stopped being able to pose the question.

The test teleports two objects away deliberately - one to the floor, one into a
bin - and asserts the reset brings both back onto the table. It also asserts
the parked `bolt` template stays parked, because "restore everything" must not
mean "teleport the template into the workspace".
"""

from __future__ import annotations

import os
import sys
import time

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(R, "tools"))
sys.path.insert(0, os.path.join(R, "tools", "synth_dataset"))
sys.path.insert(0, os.path.join(R, "src", "yolov8_obb"))
sys.path.insert(0, os.path.join(R, "src", "llm_task_planner"))

FLOOR_WORLD_Z = 0.05


def main() -> int:
    import benchmark
    import capture_negatives
    import objects as pool_objects
    from gz_client import GzClient

    top = pool_objects.TABLE_TOP_WORLD_Z
    gz = GzClient("arm_and_table")
    gz.wait_until_ready()

    def on_table(name, poses):
        z = poses[name][2]
        return top - 0.02 <= z <= top + 0.20

    poses = gz.read_poses()
    movable = sorted(n for n in poses
                     if n != capture_negatives.PARKED_TEMPLATE
                     and capture_negatives.classify_model(n) is not None)
    if len(movable) < 2:
        print("need at least two manipulable objects", file=sys.stderr)
        return 1

    to_floor, to_bin = movable[0], movable[1]
    print(f"knocking {to_floor} onto the floor and {to_bin} into a bin")
    gz.set_poses([
        (to_floor, (0.9, 0.9, FLOOR_WORLD_Z), (0.0, 0.0, 0.0)),
        # red_bin: link0 (0.28, -0.26) -> world (0.33, -0.26); floor of the bin
        (to_bin, (0.33, -0.26, top + 0.03), (0.0, 0.0, 0.0)),
    ])
    time.sleep(2.5)

    poses = gz.read_poses()
    print(f"  {to_floor}: world z {poses[to_floor][2]:+.3f} "
          f"(on table: {on_table(to_floor, poses)})")
    print(f"  {to_bin}:   world z {poses[to_bin][2]:+.3f}")
    template_before = gz.read_poses().get(capture_negatives.PARKED_TEMPLATE)

    stage = benchmark.World("arm_and_table", seed=3)
    moved = stage.reset()
    print(f"\nreset re-placed {moved} objects")
    time.sleep(2.5)

    poses = gz.read_poses()
    failures = []
    for name in (to_floor, to_bin):
        back = on_table(name, poses)
        print(f"  {name:<18} world z {poses[name][2]:+.3f}   "
              f"{'back on the table' if back else 'STILL MISSING'}")
        if not back:
            failures.append(f"{name} was not restored by the reset")

    template_after = poses.get(capture_negatives.PARKED_TEMPLATE)
    if template_before is not None and template_after is not None:
        moved_t = max(abs(a - b) for a, b in zip(template_before, template_after))
        print(f"  {capture_negatives.PARKED_TEMPLATE:<18} moved "
              f"{moved_t * 1000:.0f} mm (must stay parked)")
        if moved_t > 0.01:
            failures.append("the parked template was teleported by the reset")

    print()
    if failures:
        for f in failures:
            print(f"FAIL  {f}")
        return 1
    print("PASS - the reset restores objects from the floor and from a bin, "
          "and leaves the template parked")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

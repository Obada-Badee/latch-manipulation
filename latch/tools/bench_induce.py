#!/usr/bin/env python3
"""Induce a grasp failure at the worst possible moment, to measure recovery.

    python3 bench_induce.py --watch        # dry run: report, do not interfere

This is the instrument for the one claim the predecessor project made and could
not measure at all: that the loop is closed, meaning a failed action is detected,
reported, and *recovered from* rather than assumed to have worked.

Measuring that honestly needs a failure that is (a) real, (b) not something the
planner can foresee, and (c) triggered identically every time. So rather than
corrupting a component or lowering a threshold - both of which measure the
instrument rather than the system - this waits until the gripper has descended
onto its target and then teleports that object 150 mm away. The fingers close on
empty air.

That specific failure is chosen because it is the one the system is *designed*
to catch: `skill_server` verifies every grasp by measuring finger separation
afterwards, so a miss returns an error plus a freshly perceived world state, and
the planner should re-plan against it. If recovery does not happen here it does
not happen anywhere.

The trigger needs no privileged knowledge of what the planner intends: it
watches for a fingertip descending onto whichever object is beneath it. An
injector that had to be told the target would be testing a different, easier
system.

Object poses come from Gazebo; the fingertip pose comes from **TF**. Those are
two sources on purpose. Gazebo's model-pose topic reports a static model origin
for robot links - `panda_leftfinger` reads link0 z = -0.775, under the table,
and never changes as the arm moves - so an earlier version of this file watched
a fingertip that was never going to descend. TF carries the live transform.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
import threading
import time
from typing import Dict, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "synth_dataset"))
sys.path.insert(0, os.path.join(_ROOT, "src", "yolov8_obb"))
sys.path.insert(0, os.path.join(_ROOT, "src", "llm_task_planner"))

#: Fire when a fingertip is at or below this link0 height. The grasp height for
#: a cube is ~15 mm and the fingers close there, so this is low enough that the
#: arm is committed to the descent and high enough that the teleport lands
#: before the jaw shuts.
TRIGGER_Z = 0.085

#: ...and within this horizontal distance of the object, so a fingertip merely
#: passing overhead on its way elsewhere does not trigger.
TRIGGER_RADIUS = 0.06

#: How far to move the object. Comfortably outside the jaw, and outside
#: perception's error, so the miss is unambiguous rather than marginal.
DISPLACE_M = 0.15

#: Poll interval. The descent takes seconds, so this is ample, and it keeps the
#: injector off the CPU the planner and physics are using.
POLL_S = 0.05

FINGER_LINKS = ("panda_leftfinger", "panda_rightfinger")


def _classify(name: str) -> bool:
    """Is this a manipulable table object (rather than a link, bin or light)?"""
    stem = name[4:] if name.startswith("dyn_") else name
    base = stem.split("_")[0].rstrip("0123456789")
    return base in ("cube", "cylinder", "bolt")


class FailureInjector:
    """Watches for a descending grasp and yanks the target out from under it.

    One induction per trial. Repeating within a trial would measure persistence
    against an adversary rather than recovery from a fault, and would make the
    task unachievable - at which point a failure to finish says nothing about
    the loop.
    """

    def __init__(self, world_name: str, seed: int = 0, dry_run: bool = False,
                 node=None):
        from gz_client import GzClient                           # noqa: PLC0415
        import scene_manager                                     # noqa: PLC0415
        import objects as pool_objects                           # noqa: PLC0415
        from geometry import rpy_from_quaternion                 # noqa: PLC0415
        from yolov8_obb.projection import world_to_link0, link0_to_world  # noqa: PLC0415,E501

        self.gz = GzClient(world_name)
        self.gz.wait_until_ready()
        self._sm = scene_manager
        self._classes = pool_objects.build_classes()
        self._rpy_from_quaternion = rpy_from_quaternion
        self._to_link0 = world_to_link0
        self._to_world = link0_to_world
        self.rng = random.Random(seed)
        self.dry_run = dry_run

        self._own_node = node is None
        self._node = node if node is not None else self._make_node()
        self._buf, self._listener = self._make_tf(self._node)

        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.fired = False
        self.victim: Optional[str] = None
        self.detail = ""
        #: Lowest fingertip seen, and how close it came to an object. Recorded so
        #: a run that never fires can say WHY - "the fingertip never got below
        #: 150 mm" and "it descended to 40 mm but no object was within 60 mm" are
        #: different faults, and a bare "no grasp descent seen" distinguishes
        #: neither. A silently-never-firing injector would make replan recovery
        #: look untested when it had been measured against nothing.
        self.lowest_tip: Optional[float] = None
        self.closest: Optional[Tuple[str, float, float]] = None

    # ------------------------------------------------------------------ arm

    @staticmethod
    def _make_node():
        import rclpy                                             # noqa: PLC0415
        from rclpy.node import Node                              # noqa: PLC0415
        if not rclpy.ok():
            rclpy.init()
        return Node("bench_induce")

    @staticmethod
    def _make_tf(node):
        from tf2_ros import Buffer, TransformListener            # noqa: PLC0415
        buf = Buffer()
        # spin_thread=True is not decoration. A listener that depends on the
        # caller spinning returned the *same* fingertip height for 47 seconds
        # while the arm swung through 60 degrees, in a sibling tool. With its own
        # executor the transform tracks the arm; verified by tools/tf_probe.py.
        return buf, TransformListener(buf, node, spin_thread=True)

    def _tip_link0(self) -> Optional[Tuple[float, float, float]]:
        """The lower fingertip, in link0, from TF - or None if not yet known."""
        import rclpy                                             # noqa: PLC0415
        best = None
        for link in FINGER_LINKS:
            try:
                tf = self._buf.lookup_transform("panda_link0", link,
                                                rclpy.time.Time())
            except Exception:                                    # noqa: BLE001
                continue
            t = tf.transform.translation
            if best is None or t.z < best[2]:
                best = (t.x, t.y, t.z)
        return best

    def _object_poses(self) -> Dict[str, Tuple[float, float, float]]:
        """Manipulable objects only, in link0. Robot links are NOT read here.

        Gazebo's model-pose topic reports a static model origin for robot links,
        so the fingertip must come from TF instead - see `_tip_link0`.
        """
        return {n: self._to_link0(*p) for n, p in self.gz.read_poses().items()
                if _classify(n)}

    def _candidate(self, poses) -> Optional[Tuple[str, float]]:
        """The object a fingertip is currently descending onto, if any."""
        tip = self._tip_link0()
        if tip is None:
            return None
        if self.lowest_tip is None or tip[2] < self.lowest_tip:
            self.lowest_tip = tip[2]

        near, near_d = None, float("inf")
        for name, (x, y, z) in poses.items():
            if z < -0.10 or z > 0.30:
                continue
            d = math.hypot(tip[0] - x, tip[1] - y)
            if d < near_d:
                near, near_d = name, d
        if near is not None and (self.closest is None or near_d < self.closest[1]):
            self.closest = (near, near_d, tip[2])

        if tip[2] > TRIGGER_Z or near is None or near_d >= TRIGGER_RADIUS:
            return None
        return near, near_d

    def why_not(self) -> str:
        """Why a dry run did not fire, in terms of what was actually observed."""
        if self.lowest_tip is None:
            return ("the fingertip transform was never available - is TF up, "
                    "and is the arm controller publishing?")
        parts = [f"lowest fingertip seen: {self.lowest_tip * 1000:.0f} mm "
                 f"(trigger needs <= {TRIGGER_Z * 1000:.0f})"]
        if self.closest is None:
            parts.append("no manipulable object was ever read from Gazebo")
        else:
            name, d, z = self.closest
            parts.append(f"closest approach: {d * 1000:.0f} mm to {name} "
                         f"at fingertip height {z * 1000:.0f} mm "
                         f"(trigger needs < {TRIGGER_RADIUS * 1000:.0f} mm)")
        return "; ".join(parts)

    # -------------------------------------------------------------- displace

    def _displace(self, name: str, poses) -> bool:
        """Move the object to a valid spot DISPLACE_M away, still on the table.

        Placed somewhere legal rather than deleted or dropped through the floor:
        the object must remain pickable afterwards, or the planner's only
        correct response would be to give up, and giving up is not recovery.
        """
        x, y, z = poses[name]

        # Keep the object's current orientation. Its resting height was computed
        # for that orientation, so re-placing it upright at the same z would
        # leave a tilted bolt either floating or sunk into the table - and a
        # sunk object is unpickable, which would turn "recover from a miss" into
        # "recover from an impossible task".
        try:
            full = self.gz.read_poses_full()
            rpy = self._rpy_from_quaternion(*full[name][1])
        except Exception:                                        # noqa: BLE001
            rpy = (0.0, 0.0, 0.0)

        for _ in range(60):
            theta = self.rng.uniform(0, 2 * math.pi)
            nx = x + DISPLACE_M * math.cos(theta)
            ny = y + DISPLACE_M * math.sin(theta)
            occupied = [(px, py) for n, (px, py, pz) in poses.items()
                        if n != name and _classify(n) and -0.10 < pz < 0.30]
            ok, _why = self._sm.placement_ok(nx, ny, z, occupied)
            if not ok:
                continue
            if self.dry_run:
                self.detail = (f"would move {name} from "
                               f"({x:+.3f},{y:+.3f}) to ({nx:+.3f},{ny:+.3f})")
                return True
            world_xyz = self._to_world(nx, ny, z)
            try:
                self.gz.set_poses([(name, world_xyz, rpy)])
            except Exception as exc:                             # noqa: BLE001
                self.detail = f"teleport failed: {type(exc).__name__}: {exc}"
                return False
            self.detail = (f"moved {name} {DISPLACE_M * 1000:.0f} mm, "
                           f"({x:+.3f},{y:+.3f}) -> ({nx:+.3f},{ny:+.3f})")
            return True
        self.detail = f"no valid displacement found for {name}"
        return False

    # ----------------------------------------------------------------- loop

    def _run(self) -> None:
        while not self._stop.is_set() and not self.fired:
            try:
                poses = self._object_poses()
            except Exception:                                    # noqa: BLE001
                time.sleep(POLL_S)
                continue
            hit = self._candidate(poses)
            if hit:
                name, dist = hit
                self.victim = name
                if self._displace(name, poses):
                    self.fired = True
                    print(f"    [induced] {self.detail} "
                          f"(fingertip was {dist * 1000:.0f} mm away)", flush=True)
                    return
            time.sleep(POLL_S)

    def start(self) -> None:
        self.fired = False
        self.victim = None
        self.detail = ""
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def close(self) -> None:
        """Release the TF node, but only if we created it."""
        if self._own_node and self._node is not None:
            self._node.destroy_node()
            self._node = None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--world", default="arm_and_table")
    ap.add_argument("--watch", action="store_true",
                    help="report what it WOULD do without touching anything")
    ap.add_argument("--timeout", type=float, default=180.0)
    args = ap.parse_args(argv)

    inj = FailureInjector(args.world, dry_run=args.watch)
    print(f"watching for a descending grasp (trigger: fingertip below "
          f"{TRIGGER_Z * 1000:.0f} mm, within {TRIGGER_RADIUS * 1000:.0f} mm "
          f"of an object){' - DRY RUN' if args.watch else ''}")
    inj.start()
    deadline = time.time() + args.timeout
    while time.time() < deadline and not inj.fired:
        time.sleep(0.2)
    inj.stop()
    fired, why = inj.fired, inj.why_not()
    inj.close()
    if fired:
        print(f"fired: {inj.detail}")
        return 0
    print(f"did not fire before the timeout - {why}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

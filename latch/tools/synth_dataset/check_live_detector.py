#!/usr/bin/env python3
"""Score the trained detector against the LIVE cell, where the arm is in frame.

    ros2 run ... # bring the cell up first, then:
    python3 check_live_detector.py

This closes the one gap the capture world leaves open. The capture world has no
arm, because holding the robot in a pose there would need the xacro,
ros2_control and a controller stack just to render a background. So the detector
has never seen the arm - and the arm *is* in frame in the live cell (link0
projects to roughly u=542, v=314). A detector free to invent a detection on a
white robot link would produce a phantom object, the planner would try to pick
it, and the failure would look like a planning bug.

Rather than pre-emptively modelling the arm, measure it. The live world's object
poses are fixed and known from worlds/arm_on_the_table.sdf, so every detection
can be scored:

  matched          within MATCH_RADIUS of a known object, right class
  wrong class      near a known object, but called something else
  FALSE POSITIVE   not near any known object - the arm, a bin, or the table
  missed           a known object with no detection near it

A false positive on the arm is the specific failure this exists to find, so its
pixel location is reported too: the arm occupies the lower-right of the frame.

If false positives show up, the fix is hard negatives - generate.py --world can
capture frames from this very cell, and gz_client.read_poses supplies ground
truth for a world whose objects are dynamic.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
for p in (HERE, os.path.join(_REPO, "src", "yolov8_obb")):
    if p not in sys.path:
        sys.path.insert(0, p)

from yolov8_obb.projection import (CAMERA, obb_center,           # noqa: E402
                                  world_to_link0)

#: Fallback ground truth: the poses authored in worlds/arm_on_the_table.sdf.
#:
#: Used only when Gazebo will not report actual poses. Prefer the live ones -
#: the authored values are wrong for bolts by tens of millimetres, because a
#: bolt is dropped from z=1.05 at a pitch of +-1.5 rad and settles where physics
#: puts it. Scoring against these numbers charged perception with a mean error
#: of 18.3 mm and a worst of 43.9 mm, but every one of those large errors was on
#: a bolt while cubes came in at 5-10 mm. That was the ground truth being wrong,
#: not the perception.
LIVE_SCENE: List[Tuple[str, float, float, str]] = [
    ("cube_yellow_1", 0.50, 0.20, "cube"),
    ("cube_yellow_2", 0.30, 0.42, "cube"),
    ("cube_red_1",    0.15, 0.48, "cube"),
    ("cube_blue_1",   0.45, 0.45, "cube"),
    ("cube_green_1",  0.10, 0.62, "cube"),
    ("cylinder_red_1", 0.35, 0.60, "cylinder"),
    ("bolt1", 0.50, 0.35, "bolt"),
    ("bolt2", 0.25, 0.63, "bolt"),
    ("bolt3", 0.20, 0.30, "bolt"),
]

#: A detection this close to a known object's true position is that object.
#: Comfortably larger than the measured back-projection error (worst ~10 mm) but
#: smaller than the closest spacing between two scene objects.
MATCH_RADIUS = 0.07

def looks_like_the_arm(x: float, y: float) -> bool:
    """Does this detection's position imply it is not on the table?

    An earlier version guessed a pixel rectangle for "where the arm is" and got
    it badly wrong - it attributed 1 of 4 phantom detections to the arm when all
    4 were the arm. The mistake was reasoning in image space: the arm's *base*
    projects to about (542, 314), but its raised links project nowhere near
    there, because back-projection assumes the table plane and a link half a
    metre up violates that assumption wildly.

    Position space is unambiguous. Every one of the nine real objects sits in
    link0 x 0.04..0.45; every phantom landed at x = -0.303, -0.176, -0.160 or
    -0.143 - behind the robot's own base, where nothing can be.
    """
    return x < 0.0


def grasp_target_from_pose(class_name: str, xyz: Sequence[float],
                           quat: Sequence[float]) -> Tuple[float, float]:
    """Where world_model *should* report this object, given its true pose.

    Not the same as the model origin, and the difference is not small. Gazebo
    reports where a model's frame is; world_model reports the centre of the
    detected box, which sits over the middle of the object's silhouette. For a
    cube or a cylinder those coincide. For a bolt they are ~42 mm apart, because
    the STL's origin is under the hex head while the silhouette's middle is
    around the shaft.

    Scoring bolts against the raw model origin charged perception with 34-44 mm
    of error when it had none: the middle of the shaft is exactly where the
    gripper should go, so world_model's answer was the more useful of the two.

    Computed by posing the real vertex cloud and taking its XY extent - the same
    operation labeler.make_label performs when building ground truth - so the
    two cannot drift apart.
    """
    import numpy as np

    from geometry import matrix_from_quaternion
    from objects import build_classes

    cls = build_classes().get(class_name)
    if cls is None:
        return float(xyz[0]), float(xyz[1])
    r = matrix_from_quaternion(*quat)
    posed = np.asarray(cls.vertices, dtype=float) @ r.T + np.asarray(xyz[:3])
    lo, hi = posed.min(axis=0), posed.max(axis=0)
    return float((lo[0] + hi[0]) / 2.0), float((lo[1] + hi[1]) / 2.0)


def read_gz_poses(world: str) -> Dict[str, Tuple[Tuple[float, float, float],
                                                 Tuple[float, float, float, float]]]:
    """Actual model poses from Gazebo, so bolts are scored where they settled.

    Returns {name: ((x, y, z), (qx, qy, qz, qw))}, or {} if Gazebo will not
    answer - the caller then falls back to the authored SDF poses, and says so.
    """
    import subprocess
    try:
        r = subprocess.run(
            ["gz", "topic", "-e", "-t", f"/world/{world}/pose/info", "-n", "1"],
            capture_output=True, text=True, timeout=20)
    except (subprocess.TimeoutExpired, OSError):
        return {}

    # Protobuf text format, parsed by tracking which block we are inside. Two
    # things make the obvious line-scanning approach wrong, and both were got
    # wrong first time:
    #
    #   * ZERO-VALUED FIELDS ARE OMITTED. `table` prints `position { x: 0.29 }`
    #     with no y and no z at all. Anything waiting to see a `y:` line before
    #     committing a position silently skips every model sitting on an axis.
    #
    #   * `orientation` ALSO has x and y keys. A scanner that just grabs the
    #     next x/y after a name reads the quaternion instead of the position -
    #     which is what happened: every real object was then scored as a false
    #     positive and all nine as missed, while the detections themselves were
    #     perfectly correct.
    #
    # First occurrence of a name wins: the outer model pose is printed before
    # its links (`table` appears twice, id 4 the model and id 5 the link).
    poses: Dict[str, Tuple[Tuple[float, float, float],
                           Tuple[float, float, float, float]]] = {}
    name: Optional[str] = None
    block: Optional[str] = None
    pos = {"x": 0.0, "y": 0.0, "z": 0.0}
    quat = {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}

    for line in r.stdout.splitlines():
        s = line.strip()
        if s.startswith("name:"):
            name = s.split('"')[1] if '"' in s else s.split(":", 1)[1].strip()
            block = None
            pos = {"x": 0.0, "y": 0.0, "z": 0.0}
            quat = {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}
        elif s == "position {":
            block = "position"
        elif s == "orientation {":
            block = "orientation"
        elif s == "}":
            # The orientation block closes last, so commit there - by then both
            # position and orientation for this pose have been read.
            if block == "orientation" and name and name not in poses:
                poses[name] = ((pos["x"], pos["y"], pos["z"]),
                               (quat["x"], quat["y"], quat["z"], quat["w"]))
            block = None
        elif block in ("position", "orientation") and ":" in s:
            key, _, val = s.partition(":")
            key = key.strip()
            target = pos if block == "position" else quat
            if key in target:
                try:
                    target[key] = float(val)
                except ValueError:
                    pass
    return poses


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", type=int, default=40,
                    help="World-state messages to accumulate. More than one "
                         "because detection flickers frame to frame.")
    ap.add_argument("--timeout", type=float, default=60.0)
    args = ap.parse_args(argv)

    import rclpy
    from rclpy.node import Node
    from yolov8_msgs.msg import WorldState, Yolov8Inference

    rclpy.init()
    node = Node("check_live_detector")

    states: List = []
    raw: List = []
    node.create_subscription(WorldState, "/world_state", states.append, 10)
    node.create_subscription(Yolov8Inference, "/Yolov8_Inference", raw.append, 10)

    print(f"listening for {args.frames} /world_state messages ...", flush=True)
    deadline = time.time() + args.timeout
    while len(states) < args.frames and time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.2)

    if not states:
        print("\nNo /world_state received at all. Check, in order:")
        print("  ros2 topic hz /image_raw            camera bridged?")
        print("  ros2 topic hz /Yolov8_Inference     detector running?")
        print("  ros2 node list | grep world_model   world_model running?")
        print("Note that perception:=true starts both the detector and "
              "world_model; if you launched with perception:=false, neither is up.")
        _exit(node)
        return 1

    print(f"  got {len(states)} world states, {len(raw)} raw inference messages\n")

    # ---- raw detector output, before world_model's tracking -----------------
    det_per_frame = [len(m.yolov8_inference) for m in raw]
    if det_per_frame:
        import statistics as st
        print(f"raw detections per frame: min {min(det_per_frame)} "
              f"median {st.median(det_per_frame):.0f} max {max(det_per_frame)}  "
              f"(the scene contains {len(LIVE_SCENE)} objects)")
        cls_counts = Counter(d.class_name for m in raw for d in m.yolov8_inference)
        print(f"raw classes seen: {dict(cls_counts)}\n")

    # ---- score the most recent world state ----------------------------------
    latest = states[-1]

    live = read_gz_poses("arm_and_table")
    if live:
        used = sum(1 for n, _, _, _ in LIVE_SCENE if n in live)
        print(f"  ground truth: Gazebo's actual poses ({used}/{len(LIVE_SCENE)} "
              f"models found), corrected to each object's grasp target - "
              f"correct for bolts, which both settle where physics puts them "
              f"and carry their origin under the hex head")
    else:
        print("  ground truth: the authored SDF poses (Gazebo would not report "
              "actual ones). Bolt positions will look tens of mm out; that is "
              "this reference being wrong, not perception.")
    truth = []
    for n, sx, sy, c in LIVE_SCENE:
        if n in live:
            xyz, quat = live[n]
            gx, gy = grasp_target_from_pose(c, xyz, quat)
        else:
            gx, gy = sx, sy
        truth.append((n, *world_to_link0(gx, gy, 0.0)[:2], c))

    matched: Dict[str, dict] = {}
    wrong_class: List[dict] = []
    false_pos: List[dict] = []

    for obj in latest.objects:
        best, best_d = None, MATCH_RADIUS
        for name, tx, ty, tcls in truth:
            d = math.hypot(obj.x - tx, obj.y - ty)
            if d < best_d:
                best, best_d = (name, tcls), d
        rec = dict(id=obj.id, cls=obj.class_name, color=obj.color,
                   x=obj.x, y=obj.y, d=best_d,
                   long=obj.long_axis, short=obj.short_axis,
                   graspable=obj.graspable)
        if best is None:
            u, v = CAMERA.link0_to_pixel(obj.x, obj.y, 0.015)
            rec.update(u=u, v=v, on_arm=looks_like_the_arm(obj.x, obj.y))
            false_pos.append(rec)
        elif best[1] != obj.class_name:
            rec["truth"] = best[0]
            rec["truth_cls"] = best[1]
            wrong_class.append(rec)
        else:
            rec["truth"] = best[0]
            matched[best[0]] = rec

    print(f"--- /world_state, latest of {len(states)} ---")
    print(f"{'id':>3} {'class':<10}{'colour':<8}{'x':>8}{'y':>8}"
          f"{'long':>7}{'short':>7}  grasp  matches")
    for obj in latest.objects:
        m = next((r for r in matched.values() if r["id"] == obj.id), None)
        w = next((r for r in wrong_class if r["id"] == obj.id), None)
        f = next((r for r in false_pos if r["id"] == obj.id), None)
        if m:
            note = f"{m['truth']} ({m['d'] * 1000:.0f} mm)"
        elif w:
            note = f"WRONG CLASS: {w['truth']} is a {w['truth_cls']}"
        else:
            note = (f"FALSE POSITIVE at px ({f['u']:.0f},{f['v']:.0f})"
                    + ("  BEHIND THE BASE - the arm" if f["on_arm"] else ""))
        print(f"{obj.id:>3} {obj.class_name:<10}{obj.color:<8}"
              f"{obj.x:>8.3f}{obj.y:>8.3f}{obj.long_axis:>7.3f}"
              f"{obj.short_axis:>7.3f}  {'yes' if obj.graspable else ' no'}   {note}")

    missed = [n for n, _, _, _ in truth if n not in matched]

    print(f"\n--- score ---")
    print(f"  known objects        {len(truth)}")
    print(f"  matched              {len(matched)}")
    print(f"  wrong class          {len(wrong_class)}")
    print(f"  false positives      {len(false_pos)}"
          f"   (on the arm: {sum(1 for f in false_pos if f['on_arm'])})")
    print(f"  missed               {len(missed)}"
          + (f"   {missed}" if missed else ""))
    if matched:
        errs = [r["d"] * 1000 for r in matched.values()]
        print(f"  position error        mean {sum(errs) / len(errs):.1f} mm  "
              f"worst {max(errs):.1f} mm")

    # ---- id stability: the planner refers to objects by id across turns -----
    id_sets = [frozenset(o.id for o in s.objects) for s in states if s.objects]
    if id_sets:
        churn = sum(1 for a, b in zip(id_sets, id_sets[1:]) if a != b)
        print(f"  id churn             {churn}/{max(len(id_sets) - 1, 1)} "
              f"frame transitions changed the id set")
        if churn > len(id_sets) * 0.25:
            print("    ^ ids are unstable; the planner would lose track of an "
                  "object between turns. Check ID_MATCH_RADIUS in world_model.")

    print()
    ok = True
    if missed:
        print(f"MISSED {len(missed)} objects. Lower detector_conf, or the "
              f"detector needs more training.")
        ok = False
    if false_pos:
        on_arm = sum(1 for f in false_pos if f["on_arm"])
        print(f"{len(false_pos)} FALSE POSITIVES ({on_arm} on the arm).")
        if on_arm:
            print("  This is the capture-world gap: the detector never saw the "
                  "arm. Fix with hard negatives - capture frames from this cell "
                  "via generate.py --world (see gz_client.read_poses for ground "
                  "truth on a dynamic world).")
        ok = False
    if wrong_class:
        print(f"{len(wrong_class)} objects detected but classified wrongly.")
        ok = False
    if ok:
        print("Every known object detected, correctly classified, no false "
              "positives. Perception is ready for the planner.")

    _exit(node)
    return 0 if ok else 1


def _exit(node) -> None:
    import contextlib

    import rclpy
    with contextlib.suppress(Exception):
        node.destroy_node()
    with contextlib.suppress(Exception):
        rclpy.shutdown()
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())

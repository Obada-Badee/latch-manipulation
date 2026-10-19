#!/usr/bin/env python3
"""Hard negatives: frames of the LIVE cell, with the robot arm in them.

--------------------------------------------------------------------------
THE PROBLEM
--------------------------------------------------------------------------
The detector fires on the robot arm. On the live cell it produces oriented
boxes on the forearm and on the dark shadow along the table edge, classifying
them as bolts - which is not surprising, because a bolt is a dark elongated
metallic thing and so is the arm.

This is not a threshold problem and it is not fixable by filtering. The cause
is in the training data: generate.py renders a *static capture world* that
contains a table, some objects, two bins, and nothing else. The detector has
never seen the arm, so it has never been given a reason to call it background.

world_model's WORKSPACE_X/Y filter currently catches these, and its docstring
says plainly that it is a guard rather than a cure. This is the cure.

--------------------------------------------------------------------------
WHY CAPTURE FROM THE LIVE CELL RATHER THAN ADD AN ARM TO THE CAPTURE WORLD
--------------------------------------------------------------------------
Adding a static arm model to the synthetic world was the first idea and it is
worse. A rigid arm can only be moved as one piece, so every frame would show
the same configuration in a different place - and a floating, identically-posed
arm is not what the detector will meet. The live cell moves the real arm
through real configurations, with the real self-shadowing that appears to be
what the detector is actually latching onto.

The cost is that ground truth has to be *read* rather than commanded, since
objects settle under physics. gz_client.read_poses_full does that, and it only
became trustworthy once the pose parser stopped reading quaternion components
as positions.

--------------------------------------------------------------------------
THE CORRECTNESS RISK, AND WHAT IS DONE ABOUT IT
--------------------------------------------------------------------------
There is one way this can poison the dataset rather than improve it: labelling
an object the arm is standing in front of. That teaches the detector to predict
a box where it cannot see anything - which is exactly the failure being fixed,
inverted.

So every frame is checked with labeler.frame_matches_labels, the same check
generate.py uses to catch mispaired frames, and any frame where a labelled box
does not look like its object is discarded rather than corrected. Discarding is
cheap; a wrong label is not.

    python3 capture_negatives.py --frames 200 --out ~/synth_obb_neg
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(_ROOT, "tools"))       # scene_manager
sys.path.insert(0, os.path.join(_ROOT, "src", "yolov8_obb"))
sys.path.insert(0, os.path.join(_ROOT, "src", "llm_task_planner"))

import objects as pool_objects                                  # noqa: E402
from gz_client import GzClient, GzError                         # noqa: E402
from labeler import Label, frame_matches_labels, make_label     # noqa: E402

LIVE_WORLD = "arm_and_table"

#: Joint configurations that sweep the arm across the camera's view without
#: driving it into the table.
#:
#: Curated rather than sampled uniformly, and that is deliberate. Random joint
#: configurations for a 7-DoF arm mostly produce postures that fold into the
#: table or the robot's own base; the ones that do not are not any more varied
#: in *image* terms, which is all that matters here. These six hold the elbow
#: high and rotate the whole arm through the workspace, which is what changes
#: where it appears in frame. Jitter supplies the rest.
#:
#: Order is panda_joint1..7.
SWEEP_POSES: List[List[float]] = [
    [0.00, -0.30, 0.00, -2.00, 0.00, 1.75, 0.79],   # the `ready` pose
    [0.70, -0.20, 0.00, -1.80, 0.00, 1.65, 0.79],
    [1.30, -0.10, 0.00, -1.60, 0.00, 1.55, 0.79],
    [-0.50, -0.40, 0.00, -2.10, 0.00, 1.85, 0.79],
    [0.40, 0.20, 0.00, -1.40, 0.00, 1.30, 0.79],
    [1.00, 0.35, 0.00, -1.20, 0.00, 1.10, 0.79],
]

#: Random perturbation per joint, radians. Small enough to stay in the safe
#: envelope the curated poses establish, large enough that no two frames show
#: the same silhouette.
JITTER = [0.35, 0.18, 0.30, 0.18, 0.40, 0.25, 0.60]

#: Independent joint ranges for `--pose-mode novel`, used only to build a
#: *test* set the training poses do not cover.
#:
#: Training frames come from SWEEP_POSES plus JITTER: six clusters. A detector
#: scored only on more samples from those same six clusters could have learned
#: "these arm silhouettes are not objects" rather than "the arm is not an
#: object", and the two are indistinguishable on such a set. These ranges are
#: chosen to fall in the gaps *between* the clusters, so a model that merely
#: memorised the training silhouettes has somewhere to fail.
#:
#: Still bounded well inside the joint limits: the point is unseen postures,
#: not driving the elbow through the table.
NOVEL_RANGES: List[Tuple[float, float]] = [
    (-1.60, 1.90),      # joint1 - wider yaw sweep than the clusters reach
    (-0.55, 0.50),
    (-0.70, 0.70),      # joint3 - held at 0.0 in every curated pose
    (-2.30, -1.00),
    (-0.80, 0.80),      # joint5 - held at 0.0 in every curated pose
    (1.00, 2.00),
    (-0.80, 2.20),
]

ARM_JOINTS = [f"panda_joint{i}" for i in range(1, 8)]

#: Link origins saved with every frame (link0 frame: pose/info reports links
#: relative to the robot model, whose origin is link0). They let a scorer rebuild
#: the arm's silhouette from kinematics - the "just mask the arm" baseline.
ARM_LINKS = [f"panda_link{i}" for i in range(8)] + ["panda_leftfinger",
                                                     "panda_rightfinger"]

#: Re-place the table objects every N frames.
#:
#: Without this every captured frame shows the same nine objects in the same
#: nine places, and only the arm changes. That is still useful - the arm is the
#: point - but a few hundred near-identical scenes mixed into a 2400-frame
#: synthetic set would over-represent one layout, and the detector could learn
#: the layout rather than the lesson. Shuffling costs one batched set_pose call.
SHUFFLE_EVERY = 10

#: Seconds allowed for the arm to reach a commanded configuration before the
#: frame is captured. The cell runs at roughly half real time.
MOVE_TIME_S = 2.5
SETTLE_S = 1.5

#: A frame is kept only if every labelled box still looks like its object. See
#: the module docstring: a label on an occluded object is worse than no frame.
#:
#: generate.py uses labeler's default of 0.25, which sits in the empty middle of
#: a bimodal distribution (a stale label scores ~0, a correct one >0.5) and is
#: therefore a separator rather than a knob. Here the arm can cover *part* of an
#: object, which lands between those modes, so the bar is raised: a half-hidden
#: object is exactly the case that must not be labelled.
MIN_COVERAGE = 0.45


def classify_model(name: str) -> Optional[Tuple[str, str]]:
    """Model name -> (class, colour), or None if it is not a labelled object.

    Names come from the authored world (`cube_yellow_1`, `bolt2`,
    `cylinder_red_1`) and from scene_manager (`dyn_cube_red_43175`), so both
    shapes have to be understood. Anything unrecognised returns None and is
    skipped rather than guessed - an object labelled with the wrong class is a
    worse outcome than one left out, because the frame then teaches the
    detector something false rather than merely nothing.
    """
    stem = name[4:] if name.startswith("dyn_") else name
    parts = stem.split("_")
    head = parts[0]
    # bolt1 / bolt2 / bolt3 carry no colour: the mesh is bare grey metal.
    base = head.rstrip("0123456789")
    if base not in pool_objects.CLASS_NAMES:
        return None
    if base == "bolt":
        return "bolt", "grey"
    colour = parts[1] if len(parts) > 1 else "unknown"
    if colour not in pool_objects.MATERIALS:
        return None
    return base, colour


class ArmDriver:
    """Commands the arm controller directly, bypassing MoveIt.

    Deliberate: MoveIt would refuse configurations it considers colliding and
    would plan around them, and what is wanted here is the arm in a spread of
    postures, not a safe path between two of them. The poses are curated to be
    safe, so the planner has nothing to add.
    """

    def __init__(self, node):
        from trajectory_msgs.msg import JointTrajectory          # noqa: PLC0415
        self._msg_type = JointTrajectory
        self.node = node
        self.pub = node.create_publisher(
            JointTrajectory, "/panda_arm_controller/joint_trajectory", 10)

    def go(self, positions: Sequence[float], seconds: float = MOVE_TIME_S):
        from builtin_interfaces.msg import Duration              # noqa: PLC0415
        from trajectory_msgs.msg import JointTrajectoryPoint     # noqa: PLC0415
        msg = self._msg_type()
        msg.joint_names = ARM_JOINTS
        point = JointTrajectoryPoint()
        point.positions = [float(p) for p in positions]
        point.time_from_start = Duration(sec=int(seconds),
                                         nanosec=int((seconds % 1) * 1e9))
        msg.points = [point]
        self.pub.publish(msg)


#: The parked template model in the world SDF: no numeric suffix, parked under
#: the floor. It must never be teleported onto the table.
PARKED_TEMPLATE = "bolt"


def shuffle_objects(gz: GzClient, poses: Dict, classes: Dict,
                    rng: random.Random, restore_all: bool = False) -> int:
    """Teleport every table object to a fresh, valid spot.

    Placement rules are imported from tools/scene_manager rather than
    re-derived, so a spot this accepts is a spot the robot could have been
    asked to pick from - same reach band, same camera margin, same separation.
    Two sources of truth for "where may an object be" is how a dataset ends up
    containing scenes the robot can never encounter.

    `restore_all` decides what happens to objects that are NOT on the table -
    in a bin, or knocked onto the floor.

    False (dataset capture): leave them. A frame should show what the scene
    actually contains.

    True (benchmark reset): bring them back. Trials must be independent, and an
    object that fell off the table in trial 9 was silently excluded from every
    trial after it, because the height window below treats floor level as "not
    in play" and skips it forever. Measured: a 75-trial run lost all three bolts
    partway through and voided seven trials with "needs 1 any bolt, scene has
    0". Those are not capability results - the scene had quietly stopped being
    able to pose the question.
    """
    import scene_manager as sm                                   # noqa: PLC0415
    from geometry import resting_z                               # noqa: PLC0415
    from yolov8_obb.projection import link0_to_world, world_to_link0

    placements = []
    taken: List[Tuple[float, float]] = []
    for name, (xyz, _rpy) in sorted(poses.items()):
        kind = classify_model(name)
        if kind is None:
            continue
        class_name, _colour = kind
        if name == PARKED_TEMPLATE:
            continue                      # never bring the template up
        on_table = (pool_objects.TABLE_TOP_WORLD_Z - 0.02 <= xyz[2] <=
                    pool_objects.TABLE_TOP_WORLD_Z + 0.20)
        if not on_table and not restore_all:
            continue                      # in a bin, or otherwise not in play
        cls = classes[class_name]
        rpy = cls.sample_rpy(np.random.default_rng(rng.randrange(1 << 30)))
        wz = resting_z(cls.vertices, rpy, pool_objects.TABLE_TOP_WORLD_Z)
        plane_z = world_to_link0(0.0, 0.0, wz)[2]
        spot = sm.sample_placement(np.random.default_rng(rng.randrange(1 << 30)),
                                   plane_z, taken)
        if spot is None:
            continue
        taken.append(spot)
        wx, wy, _ = link0_to_world(spot[0], spot[1], 0.0)
        placements.append((name, (wx, wy, wz), rpy))

    if placements:
        gz.set_poses(placements)
    return len(placements)


def build_labels(poses: Dict, classes: Dict) -> List[Label]:
    """Ground-truth labels for everything on the table, from Gazebo."""
    labels: List[Label] = []
    for name, (xyz, rpy) in sorted(poses.items()):
        kind = classify_model(name)
        if kind is None:
            continue
        class_name, colour = kind
        # Anything not resting on the table is out of scope: an object in a bin
        # is largely hidden by the bin walls, and its minimum-area rectangle
        # would enclose geometry the camera cannot see.
        if not (pool_objects.TABLE_TOP_WORLD_Z - 0.02 <= xyz[2] <=
                pool_objects.TABLE_TOP_WORLD_Z + 0.20):
            continue
        cls = classes[class_name]
        try:
            labels.append(make_label(cls.class_id, class_name, colour, name,
                                     cls.vertices, xyz, rpy))
        except ValueError:
            continue            # projects above the camera; not a table object
    return labels


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--out", default=os.path.expanduser("~/synth_obb_neg"))
    ap.add_argument("--world", default=LIVE_WORLD)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--no-move", action="store_true",
                    help="do not drive the arm; capture wherever it is")
    ap.add_argument("--pose-mode", choices=("sweep", "novel"), default="sweep",
                    help="sweep: the six curated poses plus jitter, for "
                         "training data. novel: postures sampled from the gaps "
                         "between them, for a test set that can actually fail")
    ap.add_argument("--shuffle-every", type=int, default=SHUFFLE_EVERY,
                    help="re-place the table objects every N frames; 0 to keep "
                         "the authored layout")
    args = ap.parse_args(argv)

    import cv2                                                   # noqa: PLC0415
    import rclpy                                                 # noqa: PLC0415
    from cv_bridge import CvBridge                               # noqa: PLC0415
    from rclpy.node import Node                                  # noqa: PLC0415
    from sensor_msgs.msg import Image                            # noqa: PLC0415

    rng = random.Random(args.seed)
    classes = pool_objects.build_classes()

    try:
        gz = GzClient(args.world)
        gz.wait_until_ready()
    except GzError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    rclpy.init()
    node = Node("capture_negatives")
    bridge = CvBridge()
    latest = {}
    node.create_subscription(Image, "/image_raw",
                             lambda m: latest.__setitem__("img", m), 10)
    arm = None if args.no_move else ArmDriver(node)

    def grab(timeout: float = 10.0):
        latest.pop("img", None)
        deadline = time.time() + timeout
        while time.time() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
            if "img" in latest:
                return bridge.imgmsg_to_cv2(latest["img"], "bgr8")
        return None

    for split in ("train", "val"):
        os.makedirs(os.path.join(args.out, "images", split), exist_ok=True)
        os.makedirs(os.path.join(args.out, "labels", split), exist_ok=True)

    # Record the seed. The layout sequence is a pure function of it, so two runs
    # sharing a seed share their layouts *exactly* - which is how a "held-out"
    # probe set was once captured that the model had effectively trained on. The
    # images differed (the arm jitters) so hashing them found nothing; only
    # comparing object layouts caught it. check_holdout.py reads this file.
    with open(os.path.join(args.out, "capture_meta.json"), "w",
              encoding="utf-8") as meta:
        json.dump({"seed": args.seed, "frames": args.frames,
                   "shuffle_every": args.shuffle_every, "world": args.world,
                   "pose_mode": args.pose_mode}, meta, indent=2)

    kept = rejected = empty = 0
    reasons: Dict[str, int] = {}
    truth = open(os.path.join(args.out, "ground_truth.jsonl"), "w",
                 encoding="utf-8")

    try:
        for i in range(args.frames):
            if args.shuffle_every and i % args.shuffle_every == 0:
                moved = shuffle_objects(gz, gz.read_poses_full(), classes, rng)
                # Objects are teleported, not placed, so they drop the last
                # millimetre and rock. Labels are read after this settles.
                deadline = time.time() + 2.0
                while time.time() < deadline:
                    rclpy.spin_once(node, timeout_sec=0.05)
                if i == 0:
                    print(f"  re-placing {moved} objects every "
                          f"{args.shuffle_every} frames")

            if arm is not None:
                if args.pose_mode == "novel":
                    target = [rng.uniform(lo, hi) for lo, hi in NOVEL_RANGES]
                else:
                    base = SWEEP_POSES[i % len(SWEEP_POSES)]
                    target = [b + rng.uniform(-j, j)
                              for b, j in zip(base, JITTER)]
                arm.go(target)
                deadline = time.time() + MOVE_TIME_S + SETTLE_S
                while time.time() < deadline:
                    rclpy.spin_once(node, timeout_sec=0.05)

            image = grab()
            if image is None:
                print("  no camera frame; is the cell up?", file=sys.stderr)
                break

            # Poses are read AFTER the arm has settled and immediately before
            # the frame is used, so a nudged object is labelled where it now is
            # rather than where it started.
            poses = gz.read_poses_full()
            labels = build_labels(poses, classes)
            if not labels:
                empty += 1
                reasons["nothing on the table"] = reasons.get(
                    "nothing on the table", 0) + 1
                continue

            ok, worst = frame_matches_labels(image, labels,
                                             min_coverage=MIN_COVERAGE)
            if not ok:
                rejected += 1
                reasons["a labelled object did not look like itself "
                        "(occluded by the arm)"] = reasons.get(
                    "a labelled object did not look like itself "
                    "(occluded by the arm)", 0) + 1
                continue

            split = "val" if rng.random() < args.val_fraction else "train"
            stem = f"neg_{i:05d}"
            cv2.imwrite(os.path.join(args.out, "images", split, stem + ".png"),
                        image)
            h, w = image.shape[:2]
            with open(os.path.join(args.out, "labels", split, stem + ".txt"),
                      "w", encoding="utf-8") as fh:
                fh.write("\n".join(l.yolo_line(w, h) for l in labels) + "\n")

            # Same record shape generate.py writes, so validate.py can check
            # these frames too. Skipping that would mean the one dataset built
            # from *read* ground truth - the one with more ways to be wrong -
            # is the only one never independently checked.
            truth.write(json.dumps({
                "image": f"images/{split}/{stem}.png",
                "split": split,
                "objects": [{
                    "class": l.class_name,
                    "color": l.color,
                    "model": l.model_name,
                    "corners_px": [[float(u), float(v)] for u, v in l.corners],
                    "world_xyz": list(l.world_xyz),
                    "rpy": list(l.rpy),
                    # Where the gripper should actually go, which for a bolt is
                    # ~42 mm from its pose origin. validate.py scores
                    # back-projection against THIS, not the origin - comparing
                    # against the origin reports a 40 mm "error" that is only
                    # two reference points being different things.
                    "grasp_target_link0": list(l.grasp_target_link0()),
                    "top_z_link0": float(l.top_z_link0),
                } for l in labels],
                "arm_links_link0": {k: list(poses[k][0]) for k in ARM_LINKS
                                    if k in poses},
            }) + "\n")
            truth.flush()
            kept += 1

            if (i + 1) % 20 == 0:
                print(f"  {i + 1}/{args.frames}: kept {kept}, "
                      f"rejected {rejected}, empty {empty} "
                      f"(worst coverage {worst:.2f})")
    finally:
        truth.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    # dataset.yaml so this set is trainable and validatable on its own, not
    # only after being merged.
    with open(os.path.join(args.out, "dataset.yaml"), "w",
              encoding="utf-8") as fh:
        fh.write(f"path: {os.path.abspath(args.out)}\n")
        fh.write("train: images/train\nval: images/val\n\nnames:\n")
        for idx, cname in enumerate(pool_objects.CLASS_NAMES):
            fh.write(f"  {idx}: {cname}\n")

    print(f"\nkept {kept} frames in {args.out}")
    print(f"rejected {rejected + empty}:")
    for why, n in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"  {n:>4}  {why}")
    if kept == 0:
        print("\nnothing was kept. If everything was rejected for occlusion, "
              "the arm is parked over the objects - move it clear, or run with "
              "--no-move to check the labels against an idle cell first.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

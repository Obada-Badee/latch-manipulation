#!/usr/bin/env python3
"""Generate a YOLO-OBB dataset by randomising the capture world and rendering it.

    python3 generate.py --frames 2000 --out ~/synth_obb

Runs as a ROS node so the image comes off ``/image_raw`` through
``ros_gz_bridge``, which is the exact path the deployed detector uses. Reading
the frames some other way would leave the bridge - and the encoding conversion
in it - untested until the live run.

Per frame:
  1. sample a plausible scene (count, classes, colours, poses, orientations)
  2. teleport the chosen objects in and everything else off-camera, in one call
  3. wait for a *fresh* rendered frame
  4. write the image, the label file, and a ground-truth metadata record

The ground truth is exact because every object in the capture world is static:
the pose we commanded is the pose that was rendered. See capture_world.py.

Nothing is trained until validate.py has drawn these labels back onto the
rendered images. A projection error here is invisible in the loss curve and
surfaces three layers away as a missed grasp.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from capture_world import WORLD_NAME                                   # noqa: E402
from geometry import resting_z, transform                              # noqa: E402
from gz_client import GzClient, GzError, Placement                     # noqa: E402
from labeler import (Label, frame_matches_labels, fully_inside,        # noqa: E402
                     make_label, overlap_fraction)
from objects import (CLASS_NAMES, TABLE_TOP_WORLD_Z, PoolItem,        # noqa: E402
                     build_classes, build_pool)
from yolov8_obb.projection import CAMERA                               # noqa: E402

# --------------------------------------------------------------------------
# Scene sampling
# --------------------------------------------------------------------------

#: Where objects may be placed, in the link0 frame. This is the intersection of
#: three constraints, all of which matter:
#:   * on the table            link0 X -0.16..0.64, Y -0.75..0.75
#:   * inside the camera frame checked exactly, per object, via fully_inside
#:   * clear of the two bins   which live at Y -0.26 and -0.22
#: The result is a superset of where the live world actually puts objects
#: (X 0.05..0.45, Y 0.20..0.63), so the detector sees every real configuration
#: plus some margin, without ever being asked to find an object inside a bin.
PLACE_X = (0.02, 0.62)
PLACE_Y = (-0.05, 0.72)

#: Frames contain between these many objects. The live scene has 9; going down
#: to 2 teaches the detector that a sparse table is normal, and up to 11 covers
#: the crowding that happens as objects accumulate near a bin.
MIN_OBJECTS = 2
MAX_OBJECTS = 11

#: Relative frequency of each class. Cubes dominate because the task language
#: is mostly about cubes, and because there are four cube colours to cover.
CLASS_WEIGHTS = {"cube": 0.55, "bolt": 0.30, "cylinder": 0.15}

#: Reject a placement whose OBB buries this much of another object's OBB (or
#: this much of its own). Some overlap is realistic and worth training on;
#: burying an object makes its label a lie about what those pixels contain.
MAX_OVERLAP = 0.15

#: Extra separation factor applied to the sum of two objects' XY radii, so
#: static objects never interpenetrate (they have no physics to push them apart).
SEPARATION_FACTOR = 1.10

#: Attempts to place one object before giving up and settling for a smaller
#: scene. Crowded frames legitimately run out of room.
PLACEMENT_TRIES = 60

#: Stop the run after this many frames fail back to back. Isolated failures are
#: transient and get skipped; a run of them means Gazebo has died and every
#: further frame would fail the same way.
MAX_CONSECUTIVE_FAILURES = 8

#: Frames to throw away after a teleport before even looking. The render and
#: bridge pipeline is a few frames deep, so the first arrivals still show the
#: previous scene. This is only an optimisation - correctness comes from the
#: coverage check, not from this number.
INITIAL_DISCARD = 3

#: How many frames to examine before declaring the renderer hopelessly behind.
FRESHNESS_ATTEMPTS = 12


@dataclass
class Frame:
    """One rendered sample: the labels, and the placements that produced them."""
    labels: List[Label]
    placements: List[Placement]


def _xy_radius(item: PoolItem, rpy: Sequence[float]) -> float:
    """Half the object's larger XY extent once posed - its footprint radius."""
    posed = transform(item.cls.vertices, (0.0, 0.0, 0.0), rpy)
    return float(max(posed[:, 0].max() - posed[:, 0].min(),
                     posed[:, 1].max() - posed[:, 1].min())) / 2.0


def sample_frame(rng: np.random.Generator, pool: List[PoolItem]) -> Frame:
    """Draw one scene: which objects, where, and at what orientation."""
    n_target = int(rng.integers(MIN_OBJECTS, MAX_OBJECTS + 1))

    # Pick classes by weight, then draw concrete pool instances of those classes.
    # Drawing from the pool (rather than inventing objects) is what keeps colour
    # and geometry consistent with what is actually spawned in the world.
    by_class: Dict[str, List[PoolItem]] = {c: [] for c in CLASS_NAMES}
    for p in pool:
        by_class[p.cls.name].append(p)

    names = list(CLASS_WEIGHTS)
    weights = np.array([CLASS_WEIGHTS[n] for n in names], dtype=float)
    weights /= weights.sum()

    chosen: List[PoolItem] = []
    available = {c: list(v) for c, v in by_class.items()}
    for c in available:
        rng.shuffle(available[c])
    for _ in range(n_target):
        c = names[int(rng.choice(len(names), p=weights))]
        if available[c]:
            chosen.append(available[c].pop())

    labels: List[Label] = []
    placements: List[Placement] = []
    placed: List[Tuple[float, float, float]] = []       # (x, y, radius) in link0

    for item in chosen:
        for _ in range(PLACEMENT_TRIES):
            rpy = item.cls.sample_rpy(rng)
            radius = _xy_radius(item, rpy)
            lx = float(rng.uniform(*PLACE_X))
            ly = float(rng.uniform(*PLACE_Y))

            if any(math.hypot(lx - px, ly - py) < (radius + pr) * SEPARATION_FACTOR
                   for px, py, pr in placed):
                continue

            wx, wy = lx + 0.05, ly + 0.00          # link0 -> world (pure offset)
            wz = resting_z(item.cls.vertices, rpy, TABLE_TOP_WORLD_Z)

            try:
                label = make_label(item.cls.class_id, item.cls.name, item.color,
                                   item.model_name, item.cls.vertices,
                                   (wx, wy, wz), rpy)
            except ValueError:
                continue                            # vertex above the camera

            # Must be wholly in frame: a visible-but-unlabelled object teaches
            # the detector to ignore its own class.
            if not fully_inside(label.corners, margin=5.0):
                continue
            if any(overlap_fraction(label.corners, other.corners) > MAX_OVERLAP
                   for other in labels):
                continue

            labels.append(label)
            placements.append((item.model_name, (wx, wy, wz), rpy))
            placed.append((lx, ly, radius))
            break

    # Everything not used this frame goes to its own parking slot off-camera.
    # Slots are per-object and distinct (objects.PoolItem.park_xyz); sharing one
    # crashed the physics engine. objects.py's self-test proves they all project
    # well outside the frame.
    used = {p[0] for p in placements}
    for item in pool:
        if item.model_name not in used:
            placements.append((item.model_name, item.park_xyz, (0.0, 0.0, 0.0)))

    return Frame(labels=labels, placements=placements)


# --------------------------------------------------------------------------
# Capture
# --------------------------------------------------------------------------

class ImageGrabber:
    """Latest frame off /image_raw, with a counter so staleness is detectable."""

    def __init__(self, node, topic: str = "/image_raw"):
        from cv_bridge import CvBridge
        from sensor_msgs.msg import Image

        self._bridge = CvBridge()
        self._lock = threading.Lock()
        self._latest = None
        self._count = 0
        node.create_subscription(Image, topic, self._on_image, 1)

    def _on_image(self, msg) -> None:
        img = self._bridge.imgmsg_to_cv2(msg, "bgr8")
        with self._lock:
            self._latest = img
            self._count += 1

    @property
    def count(self) -> int:
        with self._lock:
            return self._count

    def wait_for_fresh(self, discard: int = 2, timeout: float = 20.0):
        """Return a frame rendered strictly after this call.

        ``discard`` frames are thrown away first. This is the whole reason the
        counter exists: after a teleport, frames already in flight still show
        the previous scene, and pairing one of those with the new labels would
        put a wrong sample in the dataset. Two discards is comfortable at 30 Hz
        against the ~100 ms a set_pose_vector call takes.
        """
        start = self.count
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._lock:
                if self._count > start + discard and self._latest is not None:
                    return self._latest.copy()
            time.sleep(0.005)
        raise TimeoutError(
            f"no fresh frame within {timeout:.0f}s (counter stuck at {self.count}). "
            f"Is ros_gz_bridge running and bridging /image_raw?")


# --------------------------------------------------------------------------
# Dataset writing
# --------------------------------------------------------------------------

def write_dataset_yaml(root: str) -> str:
    """The yaml ultralytics reads. Class ids come from objects.CLASS_NAMES."""
    path = os.path.join(root, "dataset.yaml")
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASS_NAMES))
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(
            "# GENERATED by tools/synth_dataset/generate.py\n"
            "#\n"
            "# Class ids are the indices of objects.CLASS_NAMES. Reordering them\n"
            "# there without retraining silently mislabels every detection, so\n"
            "# they are written out here rather than typed by hand.\n"
            f"path: {os.path.abspath(root)}\n"
            "train: images/train\n"
            "val: images/val\n"
            "\n"
            "names:\n" + names + "\n")
    return path


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", type=int, default=2000)
    ap.add_argument("--out", default=os.path.expanduser("~/synth_obb"))
    ap.add_argument("--world", default=WORLD_NAME,
                    help="gz world name. Point this at the live cell's world to "
                         "capture arm-in-frame samples (see capture_world.py).")
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--vary-light", action="store_true",
                    help="Jitter the sun direction each frame. Off by default: "
                         "the capture world disables shadows to match the live "
                         "cell, so this changes little and costs a service call.")
    args = ap.parse_args(argv)

    import cv2
    import rclpy
    from rclpy.node import Node

    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)

    classes = build_classes()
    pool = build_pool(classes)

    gz = GzClient(args.world)
    print(f"waiting for gz world {args.world!r} (backend: {gz.backend}) ...",
          flush=True)
    gz.wait_until_ready()
    print("  world is up", flush=True)

    rclpy.init()
    node = Node("synth_dataset_generator")
    grabber = ImageGrabber(node)

    spin = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin.start()

    # Decide the split up front so it does not depend on how many frames survive.
    n_val = int(round(args.frames * args.val_fraction))
    val_idx = set(rng.choice(args.frames, size=n_val, replace=False).tolist())

    for split in ("train", "val"):
        os.makedirs(os.path.join(args.out, "images", split), exist_ok=True)
        os.makedirs(os.path.join(args.out, "labels", split), exist_ok=True)
    meta_path = os.path.join(args.out, "ground_truth.jsonl")

    print(f"generating {args.frames} frames -> {args.out}", flush=True)
    t0 = time.time()
    written = 0
    empty = 0
    failed = 0
    retried = 0
    consecutive = 0
    per_class = {c: 0 for c in CLASS_NAMES}

    with open(meta_path, "w", encoding="utf-8", newline="\n") as meta:
        for i in range(args.frames):
            frame = sample_frame(rng, pool)
            if not frame.labels:
                empty += 1
                continue

            # A frame can fail for transient reasons - a service reply that does
            # not arrive, a render that stalls. Losing one frame is nothing;
            # losing a 15-minute run to one hiccup is unacceptable, so failures
            # are counted and skipped rather than raised. A *persistent* fault
            # still stops the run, via the consecutive-failure guard below.
            try:
                gz.set_poses(frame.placements)
                if args.vary_light:
                    gz.set_sun((float(rng.uniform(-0.8, 0.8)),
                                float(rng.uniform(-0.8, 0.8)), -0.9),
                               diffuse=float(rng.uniform(0.65, 0.95)))

                # Do not trust a frame just because it arrived after the
                # teleport: verify it actually shows this scene. Waiting a fixed
                # number of frames is only a proxy for that, and it silently
                # stopped being sufficient when the transport backend took the
                # generator from 2.5 to 15.4 fps - ~22% of frames were then
                # rendered before the teleport landed, pairing the previous
                # scene's pixels with this scene's labels. See
                # labeler.label_coverage.
                img = None
                for attempt in range(FRESHNESS_ATTEMPTS):
                    candidate = grabber.wait_for_fresh(
                        discard=INITIAL_DISCARD if attempt == 0 else 1)
                    ok, worst = frame_matches_labels(candidate, frame.labels)
                    if ok:
                        img = candidate
                        if attempt:
                            retried += 1
                        break
                if img is None:
                    raise TimeoutError(
                        f"no frame matched the commanded scene after "
                        f"{FRESHNESS_ATTEMPTS} tries (worst label coverage "
                        f"{worst:.2f}); the renderer is lagging the pose service")
            except (GzError, TimeoutError) as e:
                failed += 1
                consecutive += 1
                print(f"\n  frame {i} skipped ({consecutive} in a row): {e}",
                      flush=True)
                if consecutive >= MAX_CONSECUTIVE_FAILURES:
                    print(f"\n{consecutive} consecutive failures - stopping. "
                          f"Gazebo has probably died; check /tmp/synth_gz.log.",
                          flush=True)
                    break
                continue
            consecutive = 0

            split = "val" if i in val_idx else "train"
            stem = f"{i:06d}"
            cv2.imwrite(os.path.join(args.out, "images", split, stem + ".png"), img)
            with open(os.path.join(args.out, "labels", split, stem + ".txt"),
                      "w", encoding="utf-8", newline="\n") as fh:
                for lab in frame.labels:
                    fh.write(lab.yolo_line(CAMERA.width, CAMERA.height) + "\n")

            meta.write(json.dumps({
                "frame": i,
                "split": split,
                "image": f"images/{split}/{stem}.png",
                "objects": [{
                    "model": l.model_name,
                    "class": l.class_name,
                    "class_id": l.class_id,
                    "color": l.color,
                    "corners_px": [[round(float(u), 3), round(float(v), 3)]
                                   for u, v in l.corners],
                    "world_xyz": [round(c, 6) for c in l.world_xyz],
                    "rpy": [round(c, 6) for c in l.rpy],
                    "grasp_target_link0": [round(c, 6) for c in l.grasp_target_link0()],
                } for l in frame.labels],
            }) + "\n")

            for lab in frame.labels:
                per_class[lab.class_name] += 1
            written += 1

            if written % 25 == 0 or i == args.frames - 1:
                rate = written / max(time.time() - t0, 1e-6)
                eta = (args.frames - i - 1) / max(rate, 1e-6)
                print(f"  {written:5d}/{args.frames}  {rate:4.1f} fps  "
                      f"eta {eta / 60:5.1f} min", flush=True)

    yaml_path = write_dataset_yaml(args.out)

    print(f"\nwrote {written} frames in {(time.time() - t0) / 60:.1f} min "
          f"({empty} skipped for having no placeable objects, "
          f"{failed} skipped after a transient Gazebo failure)")
    print(f"{retried} frames needed an extra wait for the render to catch up "
          f"({retried / max(written, 1) * 100:.1f}%); every written frame was "
          f"verified to show the scene its labels describe")
    print(f"instances per class: "
          + ", ".join(f"{c} {per_class[c]}" for c in CLASS_NAMES))
    print(f"dataset yaml: {yaml_path}")
    print(f"ground truth: {meta_path}")
    print("\nNEXT: validate before training -\n"
          f"  python3 validate.py --dataset {args.out}")

    _teardown(node)
    return 0


def _teardown(node) -> None:
    """Destroy the node and exit hard.

    os._exit rather than a clean return: the spin thread is a daemon holding a
    node that is being destroyed, and rclpy's C++ teardown aborts on that race
    ("terminate called without an active exception"). The abort dumps core, and
    the core dump swallows whatever was on stdout - including the traceback of
    whatever caused the exit, which is exactly when the output matters most.
    Same fix as skill_server.
    """
    import contextlib
    with contextlib.suppress(Exception):
        node.destroy_node()
    with contextlib.suppress(Exception):
        rclpy.shutdown()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())

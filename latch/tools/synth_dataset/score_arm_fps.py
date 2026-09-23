#!/usr/bin/env python3
"""Count false positives on frames that contain the robot arm.

    python3 score_arm_fps.py --weights ~/moveit2_obb/src/yolov8_obb/scripts/best.pt \\
                             --dataset ~/synth_obb_neg --split val

The number this reports is the one the hard-negative work exists to move, and
it is deliberately not mAP. mAP on the synthetic val set already reads 0.99 and
went *up* when a model that was worse on the live cell was trained - because
that val set contains no arm, so it cannot measure the failure mode that
matters. A metric that cannot see the bug will happily certify it.

A prediction counts as a false positive when no ground-truth box has its centre
within MATCH_RADIUS_PX. Centre distance rather than IoU on purpose: the question
here is "did it invent an object that is not there", not "is the box tight", and
an inflated box around a real object is a different complaint.

Predictions are additionally split by whether they land on the table at all,
because that is what separates "confused two objects" from "hallucinated on the
arm" - and only the second is what is being fixed.
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import sys
from typing import List, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(_ROOT, "src", "yolov8_obb"))

from objects import CLASS_NAMES                                  # noqa: E402
from yolov8_obb.projection import CAMERA, DEFAULT_CENTROID_Z     # noqa: E402

#: A prediction whose centre is this close to a true object's centre is that
#: object. Generous: 40 px at this camera is about 11 cm on the table, well past
#: any plausible localisation error, so anything beyond it is not a near miss.
MATCH_RADIUS_PX = 40.0

#: Where a real object can be, in link0 - the same bounds world_model uses to
#: discard phantoms, so this reports exactly what that filter is catching.
WORKSPACE_X = (-0.02, 0.66)
WORKSPACE_Y = (-0.30, 0.78)


def centres(label_path: str, w: int, h: int) -> List[Tuple[float, float, int]]:
    """Ground-truth centres AND class, as (u, v, class_id).

    The class was originally discarded here, and that was a hole in the metric
    rather than a simplification. A model that puts a box in exactly the right
    place but calls a cube a cylinder scored a flawless 100% recall / 0 false
    positives, because every prediction matched a truth centre and nothing ever
    compared the labels. The live cell then reported a green cube as a green
    cylinder and the planner could not ground "the green cube" at all.

    Same failure as the synthetic-mAP blind spot and the seed-collision
    holdout: a check that could only fail one way, passing loudly.
    """
    out = []
    with open(label_path, encoding="utf-8") as fh:
        for line in fh:
            f = line.split()
            if len(f) != 9:
                continue
            xs = [float(f[1 + 2 * i]) * w for i in range(4)]
            ys = [float(f[2 + 2 * i]) * h for i in range(4)]
            out.append((sum(xs) / 4.0, sum(ys) / 4.0, int(float(f[0]))))
    return out


def on_table(u: float, v: float) -> bool:
    """Would world_model keep this detection, or discard it as impossible?"""
    x, y = CAMERA.pixel_to_link0(u, v, DEFAULT_CENTROID_Z)
    return (WORKSPACE_X[0] <= x <= WORKSPACE_X[1]
            and WORKSPACE_Y[0] <= y <= WORKSPACE_Y[1])


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--dataset", default=os.path.expanduser("~/synth_obb_neg"))
    ap.add_argument("--split", default="val")
    ap.add_argument("--conf", type=float, default=0.50)
    ap.add_argument("--imgsz", type=int, default=0,
                    help="0 reads it from the checkpoint")
    args = ap.parse_args(argv)

    images = sorted(glob.glob(os.path.join(
        os.path.expanduser(args.dataset), "images", args.split, "*.png")))
    if not images:
        print(f"no images in {args.dataset}/images/{args.split}", file=sys.stderr)
        return 1

    from ultralytics import YOLO                                 # noqa: PLC0415
    model = YOLO(os.path.expanduser(args.weights))
    imgsz = args.imgsz or int(getattr(model, "overrides", {}).get("imgsz", 640))

    frames = 0
    truth_total = matched_total = class_correct = 0
    fp_on_table = fp_off_table = 0
    fp_by_class = {}
    confusion = {}
    worst = []

    for path in images:
        label = path.replace("/images/", "/labels/").replace(".png", ".txt")
        if not os.path.exists(label):
            continue
        result = model.predict(path, conf=args.conf, imgsz=imgsz,
                               agnostic_nms=True, verbose=False)[0]
        if result.obb is None or len(result.obb) == 0:
            preds = []
        else:
            xy = result.obb.xyxyxyxy.cpu().numpy()      # (n, 4, 2)
            cls = result.obb.cls.cpu().numpy().astype(int)
            preds = [(float(p[:, 0].mean()), float(p[:, 1].mean()), int(c))
                     for p, c in zip(xy, cls)]

        h, w = result.orig_shape
        truth = centres(label, w, h)
        frames += 1
        truth_total += len(truth)

        used = set()
        fp_here = 0
        for u, v, c in preds:
            best, best_d = None, MATCH_RADIUS_PX
            for i, (tu, tv, _tc) in enumerate(truth):
                if i in used:
                    continue
                d = math.hypot(u - tu, v - tv)
                if d < best_d:
                    best, best_d = i, d
            if best is not None:
                used.add(best)
                matched_total += 1
                true_c = truth[best][2]
                if true_c == c:
                    class_correct += 1
                else:
                    key = (CLASS_NAMES[true_c] if 0 <= true_c < len(CLASS_NAMES)
                           else str(true_c),
                           CLASS_NAMES[c] if 0 <= c < len(CLASS_NAMES) else str(c))
                    confusion[key] = confusion.get(key, 0) + 1
            else:
                fp_here += 1
                name = CLASS_NAMES[c] if 0 <= c < len(CLASS_NAMES) else str(c)
                fp_by_class[name] = fp_by_class.get(name, 0) + 1
                if on_table(u, v):
                    fp_on_table += 1
                else:
                    fp_off_table += 1
        if fp_here:
            worst.append((fp_here, os.path.basename(path)))

    total_fp = fp_on_table + fp_off_table
    print(f"weights   {args.weights}")
    print(f"frames    {frames} ({args.split} split of {args.dataset})")
    print(f"imgsz     {imgsz}   conf {args.conf}")
    print()
    print(f"true objects        {truth_total}")
    print(f"  detected          {matched_total}  "
          f"({100.0 * matched_total / max(truth_total, 1):.1f}% recall)")
    print(f"  missed            {truth_total - matched_total}")
    print(f"  CLASSED CORRECTLY {class_correct}  "
          f"({100.0 * class_correct / max(matched_total, 1):.1f}% of matches)")
    if confusion:
        print("  misclassified:    "
              + ", ".join(f"{t}->{p} x{n}"
                          for (t, p), n in sorted(confusion.items(),
                                                  key=lambda kv: -kv[1])))
    print()
    print(f"FALSE POSITIVES     {total_fp}  "
          f"({total_fp / max(frames, 1):.2f} per frame)")
    print(f"  inside workspace  {fp_on_table}   <- these reach the planner")
    print(f"  outside workspace {fp_off_table}   <- world_model discards these")
    if fp_by_class:
        print("  by class:         "
              + ", ".join(f"{k}={v}" for k, v in sorted(fp_by_class.items())))
    if worst:
        worst.sort(reverse=True)
        print("  worst frames:     "
              + ", ".join(f"{n}x {f}" for n, f in worst[:5]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

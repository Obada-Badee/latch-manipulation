#!/usr/bin/env python3
"""Fine-tune yolov8n-obb on the generated dataset and install the weights.

    python3 train.py --dataset ~/synth_obb --epochs 80 --install

Refuses to start until validate.py passes, because a projection error in the
labels produces a beautiful loss curve and a broken robot.

Augmentation, and why most of it is switched off
-----------------------------------------------
Ultralytics' defaults are tuned for internet photographs, where the camera pose
is unknown and varied. Here the camera is bolted to the robot: one position, one
orientation, one focal length, forever. That fixed geometry is *information* -
an object's apparent size tells you where it is, and the direction its visible
side faces lean tells you where it is relative to the lens. The detector should
be free to exploit that.

Augmentations that break it are therefore turned off:

  mosaic      stitches four frames into one, producing an image with four
              tables and four bin pairs. Impossible, and it destroys the
              size-versus-position relationship. Normally mosaic is worth a lot
              for small objects; here the objects are small *because* they are
              far away, and teaching the model otherwise is counterproductive.
  fliplr/ud   a mirrored frame has its perspective inverted: side faces lean
              toward the camera instead of away. No real frame looks like that.
  degrees     rotating the image is likewise impossible. Object *yaw* is already
              uniformly randomised in the world, which is the variation that
              actually matters for an oriented box.
  perspective,
  scale,
  translate   all move the camera by proxy.

The trade normally made for augmentation - realism for diversity - is a bad one
here, because diversity is free: generate.py can produce as many genuinely
in-distribution frames as wanted. If the model underfits, generate more data
rather than re-enabling these.

What stays on is photometric: HSV jitter. That one is actively wanted. Class is
shape, never colour (a green cube and a red cube are both `cube`), and colour is
recovered separately downstream by world_model from HSV pixels. Jittering colour
stops the detector from quietly learning "yellow-ish blob => cube", which would
break the moment an unseen colour appeared.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from typing import Optional, Sequence

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))

#: Where the detector node loads weights from by default
#: (yolov8_obb_publisher.py resolves this via get_package_share_directory).
WEIGHTS_DEST = os.path.join(REPO, "src", "yolov8_obb", "scripts", "best.pt")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=os.path.expanduser("~/synth_obb"))
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640,
                    help="Native frame width. Upscaling adds no information; the "
                         "camera really is 640x480.")
    ap.add_argument("--model", default="yolov8n-obb.pt")
    ap.add_argument("--name", default="synth_obb")
    ap.add_argument("--device", default="0")
    ap.add_argument("--install", action="store_true",
                    help=f"Copy the best weights over {WEIGHTS_DEST} (the old file "
                         f"is kept as best.pt.bak).")
    ap.add_argument("--skip-validate", action="store_true",
                    help="Train without checking the labels first. Only sensible "
                         "if validate.py was just run by hand.")
    args = ap.parse_args(argv)

    yaml_path = os.path.join(args.dataset, "dataset.yaml")
    if not os.path.isfile(yaml_path):
        print(f"no dataset.yaml in {args.dataset} - run generate.py first")
        return 2

    # Ultralytics writes settings into ~/.config, which is read-only in this
    # environment; point it somewhere writable rather than tolerating a warning
    # on every run.
    os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")

    if not args.skip_validate:
        print("=== validating labels before training ===")
        rc = subprocess.call([sys.executable, os.path.join(HERE, "validate.py"),
                              "--dataset", args.dataset])
        if rc != 0:
            print("\nvalidate.py failed - not training on labels that do not "
                  "check out. Re-run it directly to see the detail, or pass "
                  "--skip-validate if you have already looked.")
            return rc
        print()

    from ultralytics import YOLO

    model = YOLO(args.model)
    results = model.train(
        data=yaml_path,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=args.name,
        project=os.path.join(args.dataset, "runs"),
        exist_ok=True,
        patience=20,
        # --- geometry-breaking augmentation: off. See the module docstring. ---
        mosaic=0.0,
        mixup=0.0,
        fliplr=0.0,
        flipud=0.0,
        degrees=0.0,
        perspective=0.0,
        shear=0.0,
        scale=0.0,
        translate=0.0,
        # --- photometric augmentation: on, deliberately ----------------------
        hsv_h=0.4,      # wide: class must not become entangled with colour
        hsv_s=0.5,
        hsv_v=0.4,
        # Small objects (15-47 px) at a fixed scale; a longer warmup keeps the
        # first epochs from thrashing the box head.
        warmup_epochs=5,
        plots=True,
        verbose=True,
    )

    best = os.path.join(args.dataset, "runs", args.name, "weights", "best.pt")
    if not os.path.isfile(best):
        print(f"training finished but {best} does not exist")
        return 1

    print("\n=== validating the trained model on the held-out split ===")
    metrics = YOLO(best).val(data=yaml_path, imgsz=args.imgsz, device=args.device,
                             project=os.path.join(args.dataset, "runs"),
                             name=args.name + "_val", exist_ok=True)
    try:
        print(f"  mAP50    {metrics.box.map50:.4f}")
        print(f"  mAP50-95 {metrics.box.map:.4f}")
        for i, name in enumerate(metrics.names.values()):
            print(f"  {name:<10} mAP50 {metrics.box.ap50[i]:.4f}")
    except Exception:                                    # noqa: BLE001
        print(f"  {metrics}")

    print(f"\nweights: {best}")
    if args.install:
        if os.path.isfile(WEIGHTS_DEST):
            shutil.copy2(WEIGHTS_DEST, WEIGHTS_DEST + ".bak")
            print(f"kept the previous single-class weights as {WEIGHTS_DEST}.bak")
        shutil.copy2(best, WEIGHTS_DEST)
        print(f"installed -> {WEIGHTS_DEST}")
        print("\nRebuild so the install space picks it up:")
        print("  colcon build --symlink-install --packages-select yolov8_obb")
    else:
        print(f"\nnot installed. To use it:\n  cp {best} {WEIGHTS_DEST}")
        print("  colcon build --symlink-install --packages-select yolov8_obb")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Combine several YOLO-OBB dataset roots into one, without copying the images.

    python3 merge_datasets.py --out ~/synth_obb_mixed ~/synth_obb ~/synth_obb_neg

Exists so the live hard negatives (capture_negatives.py) can be trained on
alongside the synthetic frames (generate.py) without either being regenerated,
and without a second copy of 2400 PNGs. Files are symlinked; a filesystem that
refuses symlinks falls back to copying, with a note, because silently doing
something much slower is worse than saying so.

Names are prefixed per source. Both generators number from zero, so
`frame_00001.png` exists in each, and merging without prefixes would have one
source's labels silently attached to the other's images - the same mispairing
class of bug that corrupted the first synthetic run, arriving by a different
route.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from typing import List, Optional, Sequence

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from objects import CLASS_NAMES                                  # noqa: E402

SPLITS = ("train", "val")


def link_or_copy(src: str, dst: str, state: dict) -> None:
    if state["copy"]:
        shutil.copy2(src, dst)
        return
    try:
        os.symlink(src, dst)
    except (OSError, NotImplementedError, AttributeError):
        state["copy"] = True
        print("  (symlinks unavailable here; copying instead)")
        shutil.copy2(src, dst)


def merge(sources: Sequence[str], out: str, repeat: Sequence[int]) -> int:
    for split in SPLITS:
        os.makedirs(os.path.join(out, "images", split), exist_ok=True)
        os.makedirs(os.path.join(out, "labels", split), exist_ok=True)

    state = {"copy": False}
    totals = {}
    for index, (root, times) in enumerate(zip(sources, repeat)):
        tag = os.path.basename(os.path.normpath(root)) or f"src{index}"
        counts = {}
        for split in SPLITS:
            img_dir = os.path.join(root, "images", split)
            if not os.path.isdir(img_dir):
                continue
            n = 0
            for name in sorted(os.listdir(img_dir)):
                stem, ext = os.path.splitext(name)
                label = os.path.join(root, "labels", split, stem + ".txt")
                if not os.path.exists(label):
                    # A frame with no label file is not an empty frame, it is a
                    # frame whose labels went missing. Training on it would
                    # teach the detector that its objects are background.
                    print(f"  WARNING: {name} has no label file; skipped")
                    continue
                for r in range(times):
                    suffix = "" if times == 1 else f"_r{r}"
                    new = f"{tag}_{stem}{suffix}"
                    link_or_copy(os.path.abspath(os.path.join(img_dir, name)),
                                 os.path.join(out, "images", split, new + ext),
                                 state)
                    link_or_copy(os.path.abspath(label),
                                 os.path.join(out, "labels", split, new + ".txt"),
                                 state)
                    n += 1
            counts[split] = n
        totals[tag] = counts
        print(f"  {tag:<20} " + "  ".join(f"{s}={counts.get(s, 0)}"
                                          for s in SPLITS)
              + (f"   (x{times})" if times > 1 else ""))

    yaml_path = os.path.join(out, "dataset.yaml")
    with open(yaml_path, "w", encoding="utf-8") as fh:
        fh.write(f"path: {os.path.abspath(out)}\n")
        fh.write("train: images/train\nval: images/val\n\nnames:\n")
        for i, name in enumerate(CLASS_NAMES):
            fh.write(f"  {i}: {name}\n")
    print(f"\nwrote {yaml_path}")

    grand = sum(c.get(s, 0) for c in totals.values() for s in SPLITS)
    print(f"{grand} frames total")
    return 0 if grand else 1


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sources", nargs="+", help="dataset roots to merge")
    ap.add_argument("--out", required=True)
    ap.add_argument("--repeat", nargs="*", type=int, default=None,
                    help="times to include each source, in order. Oversampling "
                         "a small set raises its weight, at the risk of "
                         "overfitting to the exact frames in it - default 1 "
                         "everywhere, and measure before reaching for this.")
    args = ap.parse_args(argv)

    repeat = args.repeat or [1] * len(args.sources)
    if len(repeat) != len(args.sources):
        print(f"--repeat needs one value per source "
              f"({len(args.sources)} given)", file=sys.stderr)
        return 2
    for root in args.sources:
        if not os.path.isdir(root):
            print(f"no such dataset root: {root}", file=sys.stderr)
            return 1

    if os.path.exists(args.out):
        shutil.rmtree(args.out)
    return merge([os.path.expanduser(s) for s in args.sources],
                 os.path.expanduser(args.out), repeat)


if __name__ == "__main__":
    raise SystemExit(main())

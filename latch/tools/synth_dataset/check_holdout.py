#!/usr/bin/env python3
"""Prove a test set is actually held out before believing a number from it.

    python3 check_holdout.py --probe ~/synth_obb_probe2 --train ~/synth_obb_mixed

Run this before reporting any score. It exists because a "held-out" probe set
once turned out to share every one of its object layouts with the training set,
and the retrained detector scored a 98% improvement on it that meant nothing.

The contamination was invisible to the obvious check. Hashing the images found
zero duplicates - correctly, because the frames really were different: the arm
had moved between them. What repeated was the *table*. Both capture runs used
capture_negatives.py's default --seed 7, and the layout sequence is a pure
function of the seed, so run two regenerated run one's layouts object for object.

So this compares two things, and the second is the one that matters:

  1. pixel-identical images  - catches a copied or re-split directory
  2. object layout overlap   - catches a replayed seed, which is the failure
                               mode that actually happened

Layout comparison is in normalised label space and deliberately loose (TOL is
about 13 px at 640 wide): the question is "did the detector already see this
arrangement", not "are these coordinates equal to the bit".
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

#: Two object centres this close, in normalised image coordinates, are the same
#: object in the same place. ~13 px at 640 wide - far tighter than a reshuffle
#: moves things, far looser than float noise.
TOL = 0.02

#: A probe frame whose layout matches a training frame by at least this fraction
#: is a near-duplicate. Not 1.0: one object landing elsewhere does not make an
#: otherwise-repeated table into a new one.
NEAR_DUPLICATE_FRAC = 0.9


def centres(path: str) -> List[Tuple[float, float]]:
    """Object centres from a YOLO-OBB label file, in normalised coordinates."""
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            f = line.split()
            if len(f) != 9:
                continue
            xs = [float(f[1 + 2 * i]) for i in range(4)]
            ys = [float(f[2 + 2 * i]) for i in range(4)]
            out.append((sum(xs) / 4.0, sum(ys) / 4.0))
    return out


def layouts(root: str) -> Dict[str, List[Tuple[float, float]]]:
    found = {}
    for p in glob.glob(os.path.join(root, "labels", "**", "*.txt"),
                       recursive=True):
        real = os.path.realpath(p)          # merged sets are symlink farms
        if os.path.isfile(real):
            found[os.path.basename(p)] = centres(real)
    return found


def image_digests(root: str) -> Dict[str, List[str]]:
    found: Dict[str, List[str]] = {}
    for p in glob.glob(os.path.join(root, "images", "**", "*.png"),
                       recursive=True):
        real = os.path.realpath(p)
        if not os.path.isfile(real):
            continue
        with open(real, "rb") as fh:
            found.setdefault(hashlib.md5(fh.read()).hexdigest(),
                             []).append(os.path.basename(p))
    return found


def overlap_fraction(probe: List[Tuple[float, float]],
                     train: List[Tuple[float, float]]) -> float:
    """What fraction of the probe frame's objects sit where a training one's do."""
    if not probe:
        return 0.0
    hits = sum(1 for (a, b) in probe
               if any(abs(a - c) < TOL and abs(b - d) < TOL for (c, d) in train))
    return hits / len(probe)


def seed_of(root: str) -> Optional[int]:
    path = os.path.join(root, "capture_meta.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh).get("seed")
    except (OSError, ValueError):
        return None


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probe", required=True, help="the set you want to score on")
    ap.add_argument("--train", required=True, help="what the model was trained on")
    args = ap.parse_args(argv)

    probe_root = os.path.expanduser(args.probe)
    train_root = os.path.expanduser(args.train)

    p_seed, t_seed = seed_of(probe_root), seed_of(train_root)
    if p_seed is not None and p_seed == t_seed:
        print(f"!! both sets were captured with --seed {p_seed}. Their layout "
              f"sequences are identical by construction.")
    elif p_seed is not None and t_seed is not None:
        print(f"seeds: probe {p_seed}, train {t_seed}  (different)")

    p_img, t_img = image_digests(probe_root), image_digests(train_root)
    identical = set(p_img) & set(t_img)
    print(f"\nimages: probe {len(p_img)} unique, train {len(t_img)} unique")
    print(f"  pixel-identical in both: {len(identical)}")
    for h in sorted(identical)[:5]:
        print(f"    {p_img[h][0]} == {t_img[h][0]}")

    p_lay, t_lay = layouts(probe_root), layouts(train_root)
    if not p_lay or not t_lay:
        print("\nno labels found in one of the sets", file=sys.stderr)
        return 1

    print(f"\nlayouts: probe {len(p_lay)} frames vs train {len(t_lay)} frames")
    worst_frac, worst_pair = 0.0, ("", "")
    duplicated = []
    for pname, pc in p_lay.items():
        best, best_name = 0.0, ""
        for tname, tc in t_lay.items():
            frac = overlap_fraction(pc, tc)
            if frac > best:
                best, best_name = frac, tname
        if best >= NEAR_DUPLICATE_FRAC:
            duplicated.append((pname, best_name, best))
        if best > worst_frac:
            worst_frac, worst_pair = best, (pname, best_name)

    print(f"  worst-case layout overlap: {worst_frac * 100:.1f}%  "
          f"({worst_pair[0]} vs {worst_pair[1]})")
    print(f"  frames at or above {NEAR_DUPLICATE_FRAC * 100:.0f}%: "
          f"{len(duplicated)} of {len(p_lay)}")
    for pname, tname, frac in duplicated[:5]:
        print(f"    {pname} ~ {tname}  ({frac * 100:.0f}%)")

    contaminated = bool(identical) or bool(duplicated)
    print()
    if contaminated:
        print("CONTAMINATED - a score measured on this probe is not a "
              "generalisation result. Recapture with a different --seed.")
        return 1
    print("CLEAN - no shared images and no repeated layouts. Scores on this "
          "probe mean what they claim.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

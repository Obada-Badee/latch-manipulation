#!/usr/bin/env python3
"""Summarise a finger_trace.csv: when did the arm move, and did the jaw stall?

    python3 tools/show_trace.py ~/finger_trace.csv

Prints, in order:

  * arm motion phases - stretches where the arm joints were changing, which is
    what tells you whether the approach and descent actually happened
  * jaw stalls - stretches of at least STALL_S where the gap held steady while
    being commanded closed. A stall at the object's width is a grasp; no stall
    at all means the pads never met it
  * the gap over time

A file rather than a shell one-liner on purpose: quoted f-strings do not survive
Git Bash -> WSL bash -> python -c, and the resulting SyntaxError costs more time
than the script does to write.
"""

from __future__ import annotations

import csv
import math
import os
import sys

STALL_S = 1.5
#: A holding jaw is not a motionless jaw. Clamped on the bolt the gap crept from
#: 17.2 mm to 10.7 mm over five seconds as the contact settled, and a fixed
#: 0.5 mm tolerance called that "never held a width" - the exact opposite of what
#: happened. What separates holding from closing is SPEED: the free close runs at
#: ~120 mm/s, the clamped creep at under 2 mm/s.
STALL_MM_PER_S = 3.0
#: Below this the arm is holding still, not moving. The controller jitters by a
#: few tenths of a milliradian even when commanded to hold.
ARM_STILL_RAD = 0.002

ARM = [f"panda_joint{i}" for i in range(1, 8)]


def main() -> int:
    path = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1
                              else "~/finger_trace.csv")
    with open(path, encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("gap")]
    if not rows:
        print("no rows with a finger reading", file=sys.stderr)
        return 1

    def num(r, k):
        try:
            return float(r[k])
        except (TypeError, ValueError, KeyError):
            return None

    t = [float(r["t"]) for r in rows]
    gap = [float(r["gap"]) for r in rows]
    arm = [[num(r, n) for n in ARM] for r in rows]
    have_arm = all(a[0] is not None for a in arm)

    print(f"{len(rows)} samples over {t[-1] - t[0]:.0f} s")
    print(f"gap: start {gap[0] * 1000:.1f} mm   min {min(gap) * 1000:.1f} mm   "
          f"end {gap[-1] * 1000:.1f} mm\n")

    if have_arm:
        def delta(i, j):
            return max(abs(a - b) for a, b in zip(arm[i], arm[j]))

        print("arm motion (joint change between consecutive samples):")
        moving = [delta(k, k - 1) > ARM_STILL_RAD for k in range(1, len(rows))]
        k = 0
        any_motion = False
        while k < len(moving):
            if not moving[k]:
                k += 1
                continue
            j = k
            while j + 1 < len(moving) and moving[j + 1]:
                j += 1
            if t[j + 1] - t[k] >= 0.3:
                swept = math.degrees(delta(j + 1, k))
                print(f"  t {t[k]:6.1f} .. {t[j + 1]:6.1f} s  "
                      f"({t[j + 1] - t[k]:5.1f} s)  largest joint moved "
                      f"{swept:6.1f} deg")
                any_motion = True
            k = j + 1
        if not any_motion:
            print("  NONE - the arm never moved during this trace")
        print()

    print("jaw holds (gap changing slower than "
          f"{STALL_MM_PER_S:.0f} mm/s while closed onto something):")
    slow = []
    for k in range(1, len(rows)):
        dt = t[k] - t[k - 1]
        rate = abs(gap[k] - gap[k - 1]) * 1000 / dt if dt > 0 else 0.0
        slow.append(rate < STALL_MM_PER_S)

    i = 0
    found = False
    while i < len(slow):
        if not slow[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(slow) and slow[j + 1]:
            j += 1
        # gap[i] > 3 mm excludes the long stretches where the jaw simply sits
        # fully closed, which are steady but are not holding anything.
        if t[j + 1] - t[i] >= STALL_S and gap[i] > 0.003:
            print(f"  t {t[i]:6.1f} .. {t[j + 1]:6.1f} s  "
                  f"({t[j + 1] - t[i]:5.1f} s)  gap {gap[i] * 1000:6.2f} "
                  f"-> {gap[j + 1] * 1000:6.2f} mm")
            found = True
        i = j + 1
    if not found:
        print("  NONE - the jaw never held a width; it closed straight through")

    print("\ngap over time (every 2 mm of travel):")
    last = None
    for k, g in enumerate(gap):
        if last is None or abs(g - last) >= 0.002:
            print(f"  t {t[k]:6.1f}  gap {g * 1000:6.2f} mm")
            last = g
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

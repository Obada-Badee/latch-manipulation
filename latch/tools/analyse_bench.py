#!/usr/bin/env python3
"""Turn benchmark CSVs into the tables the paper reports.

    python3 analyse_bench.py ~/moveit2_obb/runs/main.csv
    python3 analyse_bench.py --markdown runs/*.csv > ../paper/results.md

Reads one or more CSVs written by benchmark.py and emits:

  1. success by tier, with Wilson intervals      (the headline table)
  2. per-task detail                             (appendix, descriptive only)
  3. ablation comparison, if several conditions are present
  4. self-report vs ground truth                 (over- and under-claiming)
  5. a failure taxonomy                          (what went wrong, grouped)
  6. cost and latency

Nothing here recomputes success. The `success` column was decided at trial time
by reading Gazebo against a criterion written before the runs, and this script
only counts. Keeping scoring out of the analysis means a later change of mind
about what "success" means cannot quietly rewrite past results.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from benchmark import wilson                                     # noqa: E402

TIERS = ("simple", "medium", "complex")

#: Failure taxonomy. Ordered: the first pattern that matches wins, so the more
#: specific causes are listed before the general ones.
#:
#: These are matched against the checker's `detail` and the recorded
#: `void_reason`. It is a coarse instrument and is reported as one - the point
#: is to show WHERE the system loses, not to assign blame to three decimals.
TAXONOMY: List[Tuple[str, Tuple[str, ...]]] = [
    ("provider/API",      ("ratelimit", "rate limit", "badrequest", "429",
                           "parsing failed", "apierror", "connection")),
    ("protocol",          ("stopped without", "no tool call", "iteration limit",
                           "hit the")),
    ("grounding: wrong object",
                          ("but is not in", "also moved", "not a resolution")),
    ("grounding: object not found",
                          ("no matching object", "vanished")),
    ("execution: not lifted",
                          ("nothing lifted",)),
    ("execution: never reached destination",
                          ("none of", "reached", "in a bin", "not:")),
    ("scene/setup",       ("needs >=2", "needs ", "perception never")),
]


def classify(row: Dict[str, str]) -> str:
    if row.get("outcome") == "timeout":
        return "timeout"
    blob = f"{row.get('void_reason', '')} {row.get('detail', '')}".lower()
    for name, needles in TAXONOMY:
        if any(n in blob for n in needles):
            return name
    return "other"


def load(paths: Sequence[str]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for pattern in paths:
        for path in sorted(glob.glob(os.path.expanduser(pattern))):
            with open(path, newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    if row.get("outcome") and row["outcome"] != "void":
                        rows.append(row)
    return rows


def rate(rows: Sequence[Dict[str, str]]) -> Tuple[int, int, float, float, float]:
    n = len(rows)
    s = sum(1 for r in rows if str(r.get("success")) == "1")
    p, lo, hi = wilson(s, n)
    return s, n, p, lo, hi


def line(label: str, rows: Sequence[Dict[str, str]], md: bool, width: int = 30):
    if not rows:
        return
    s, n, p, lo, hi = rate(rows)
    if md:
        print(f"| {label} | {s}/{n} | {p * 100:.0f}% | "
              f"[{lo * 100:.0f}, {hi * 100:.0f}] |")
    else:
        print(f"  {label:<{width}} {s:>3}/{n:<3} = {p * 100:5.1f}%  "
              f"[{lo * 100:4.1f}, {hi * 100:5.1f}]")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--markdown", action="store_true")
    args = ap.parse_args(argv)

    rows = load(args.csv)
    if not rows:
        print("no scored trials found", file=sys.stderr)
        return 1
    md = args.markdown
    conditions = sorted({r["condition"] for r in rows})

    print(f"# Benchmark results\n" if md else "=" * 64)
    print(f"{len(rows)} scored trials across {len(conditions)} condition(s): "
          f"{', '.join(conditions)}\n")

    # ---- 1. headline, by tier ------------------------------------------
    print("## Success by tier\n" if md else "SUCCESS BY TIER")
    if md:
        print("| condition / tier | passed | rate | 95% CI |")
        print("|---|---|---|---|")
    for cond in conditions:
        sub = [r for r in rows if r["condition"] == cond]
        for tier in TIERS:
            line(f"{cond} / {tier}", [r for r in sub if r["tier"] == tier], md)
        line(f"{cond} / ALL", sub, md)
    print()

    # ---- 2. per task ---------------------------------------------------
    print("## Per task (descriptive; n is small)\n" if md
          else "PER TASK  (descriptive - n is small, do not quote as a claim)")
    if md:
        print("| task | command | passed | rate | 95% CI |")
        print("|---|---|---|---|---|")
    for task in sorted({r["task"] for r in rows}):
        sub = [r for r in rows if r["task"] == task]
        cmd = sub[0]["command"][:42]
        if md:
            s, n, p, lo, hi = rate(sub)
            print(f"| `{task}` | {cmd} | {s}/{n} | {p * 100:.0f}% | "
                  f"[{lo * 100:.0f}, {hi * 100:.0f}] |")
        else:
            line(f"{task}  {cmd}", sub, md, width=48)
    print()

    # ---- 3. ablation ---------------------------------------------------
    if len(conditions) > 1:
        print("## Ablation\n" if md else "ABLATION")
        base = [r for r in rows if r["condition"] == "full"]
        if base:
            _, _, bp, _, _ = rate(base)
            for cond in conditions:
                if cond == "full":
                    continue
                sub = [r for r in rows if r["condition"] == cond]
                _, _, p, _, _ = rate(sub)
                delta = (p - bp) * 100
                if md:
                    print(f"| {cond} | {p * 100:.0f}% | {delta:+.0f} pp vs full |")
                else:
                    print(f"  {cond:<24} {p * 100:5.1f}%   "
                          f"{delta:+5.1f} pp vs full")
        print()

    # ---- 4. self-report vs ground truth --------------------------------
    judged = [r for r in rows if r.get("agreement") not in ("", None)]
    if judged:
        agree = sum(1 for r in judged if str(r["agreement"]) == "1")
        over = sum(1 for r in judged if str(r["planner_said"]) == "1"
                   and str(r["success"]) == "0")
        under = sum(1 for r in judged if str(r["planner_said"]) == "0"
                    and str(r["success"]) == "1")
        print("## Self-report vs ground truth\n" if md
              else "SELF-REPORT vs GROUND TRUTH")
        print(f"  agreement      {agree}/{len(judged)} "
              f"({100.0 * agree / len(judged):.1f}%)")
        print(f"  over-claimed   {over}   (said done; the table disagrees)")
        print(f"  under-claimed  {under}   (said failed; the table says it worked)")
        print()

    # ---- 5. failure taxonomy -------------------------------------------
    failures = [r for r in rows if str(r.get("success")) != "1"]
    print("## Failure taxonomy\n" if md else "FAILURE TAXONOMY")
    if not failures:
        print("  no failures recorded\n")
    else:
        counts: Dict[str, int] = {}
        for r in failures:
            counts[classify(r)] = counts.get(classify(r), 0) + 1
        if md:
            print("| cause | n | share of failures |")
            print("|---|---|---|")
        for cause, n in sorted(counts.items(), key=lambda kv: -kv[1]):
            share = 100.0 * n / len(failures)
            if md:
                print(f"| {cause} | {n} | {share:.0f}% |")
            else:
                print(f"  {cause:<38} {n:>3}  ({share:4.0f}%)")
        print()

    # ---- 6. cost and latency -------------------------------------------
    def nums(field: str) -> List[float]:
        out = []
        for r in rows:
            try:
                out.append(float(r[field]))
            except (TypeError, ValueError, KeyError):
                pass
        return sorted(out)

    times, tin, tout = nums("wall_s"), nums("in_tokens"), nums("out_tokens")
    print("## Cost and latency\n" if md else "COST AND LATENCY")
    if times:
        print(f"  wall seconds     median {times[len(times) // 2]:.0f}   "
              f"min {times[0]:.0f}   max {times[-1]:.0f}")
    if tin:
        total = sum(tin) + sum(tout)
        print(f"  tokens per trial median "
              f"{(tin[len(tin) // 2] + (tout[len(tout) // 2] if tout else 0)):,.0f}")
        print(f"  tokens total     {total:,.0f}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

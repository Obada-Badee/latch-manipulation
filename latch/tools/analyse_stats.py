#!/usr/bin/env python3
"""Statistics for the benchmark, with the command as the unit of analysis.

    python3 tools/analyse_stats.py

The suite is 15 commands x 5 seeds, and success is heavily clustered by
command: the negative control is 5/5 in every condition, `c2` and `c4` are
0/5 in most. Treating the 75 trials as 75 independent Bernoulli draws - which
a plain Wilson interval does - understates the uncertainty, because the
effective number of independent observations is nearer 15 than 75.

So: intervals come from a bootstrap that resamples COMMANDS (with their five
trials attached), and between-condition tests are PAIRED on the matched
seed x command cell, since every condition runs the same seeds on the same
commands. Pairing is both more correct and more powerful than pooling runs
and comparing totals.

Also reports per-placement reliability, which is a better-powered view of the
same data: a five-object command is one binary outcome, but it is five
placements, and the compounding model can be fitted rather than asserted.
"""
from __future__ import annotations

import collections
import csv
import io
import os
import random
import re
import sys

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "..", "..", "paper", "data")
DATA = os.environ.get("LATCH_DATA", os.path.normpath(DATA))
NEG_CONTROL = "c5"


def load(name):
    p = os.path.join(DATA, name)
    if not os.path.exists(p):
        return []
    rows = [r for r in csv.DictReader(io.open(p, encoding="utf-8"))
            if r.get("outcome") != "void"]
    keep = {}
    for r in rows:                      # same rule as benchmark.scored_rows
        if "429" in (r.get("void_reason") or "") and r.get("success") != "1":
            continue
        keep[(r.get("condition"), r.get("task"), r.get("trial"))] = r
    return list(keep.values())


def wilson(k, n, z=1.96):
    if not n:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = (z / d) * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return 100 * p, 100 * max(0.0, c - h), 100 * min(1.0, c + h)


def cluster_bootstrap(rows, reps=20000, seed=12345):
    """Percentile CI resampling commands, not trials.

    A command is the unit that was chosen by the experimenter; its five trials
    share a scene, a phrasing and a difficulty, and their outcomes are not
    independent of one another.
    """
    by = collections.defaultdict(list)
    for r in rows:
        by[r["task"]].append(r["success"] == "1")
    tasks = list(by)
    if not tasks:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    point = 100.0 * sum(sum(v) for v in by.values()) / sum(len(v) for v in by.values())
    out = []
    for _ in range(reps):
        picked = [by[rng.choice(tasks)] for _ in tasks]
        k = sum(sum(v) for v in picked)
        n = sum(len(v) for v in picked)
        out.append(100.0 * k / n)
    out.sort()
    return point, out[int(0.025 * reps)], out[int(0.975 * reps)]


def mcnemar(a_rows, b_rows):
    """Exact paired test on matched (task, trial) cells. Returns (b, c, p)."""
    A = {(r["task"], r["trial"]): r["success"] == "1" for r in a_rows}
    B = {(r["task"], r["trial"]): r["success"] == "1" for r in b_rows}
    both = set(A) & set(B)
    b = sum(1 for k in both if A[k] and not B[k])
    c = sum(1 for k in both if B[k] and not A[k])
    n = b + c
    if n == 0:
        return b, c, 1.0
    # exact binomial two-sided, p = 0.5
    from math import comb
    tail = sum(comb(n, i) for i in range(0, min(b, c) + 1)) / (2.0 ** n)
    return b, c, min(1.0, 2.0 * tail)


def fisher(a, b, c, d):
    """Two-sided Fisher exact on [[a,b],[c,d]]."""
    from math import comb
    n = a + b + c + d
    rows, cols = a + b, a + c
    obs = comb(rows, a) * comb(n - rows, c) / comb(n, cols)
    tot = 0.0
    for i in range(max(0, cols - (n - rows)), min(cols, rows) + 1):
        pr = comb(rows, i) * comb(n - rows, cols - i) / comb(n, cols)
        if pr <= obs * (1 + 1e-9):
            tot += pr
    return min(1.0, tot)


def placements(rows):
    """(placed, required) summed over the multi-object commands.

    Read from the scorer's own detail text: "3/5 in blue_bin", "2/3 in a bin".
    A binary task outcome at n=10 is a weak instrument; the placements inside
    those tasks are 50 observations of the thing that actually compounds.
    """
    got = req = 0
    for r in rows:
        if r["task"] not in ("c1", "c4"):
            continue
        d = r.get("detail") or ""
        m = re.search(r"(\d+)\s*/\s*(\d+)\s+in\s", d)
        if m:
            got += int(m.group(1)); req += int(m.group(2))
        elif r["success"] == "1":
            got += 5; req += 5
        elif "none of" in d:
            m2 = re.search(r"none of (\d+)", d)
            req += int(m2.group(1)) if m2 else 5
    return got, req


CONDITIONS = [
    ("LATCH (full)", "full_v6.csv"),
    ("no affordance", "abl_no_affordance_v6.csv"),
    ("no memory", "abl_no_memory_v6.csv"),
    ("baseline", "baseline_v6.csv"),
    ("naive", "naive_v6.csv"),
]


def main():
    loaded = [(lbl, load(f)) for lbl, f in CONDITIONS]
    loaded = [(lbl, r) for lbl, r in loaded if r]
    if not loaded:
        print("no data in", DATA, file=sys.stderr)
        return 1

    print("=" * 78)
    print("SUCCESS, with the command as the unit of analysis")
    print("=" * 78)
    print("%-16s %-22s %-24s %s" % ("condition", "trial-level Wilson",
                                    "cluster bootstrap", "excl. control"))
    for lbl, rows in loaded:
        k = sum(1 for r in rows if r["success"] == "1")
        p, lo, hi = wilson(k, len(rows))
        bp, blo, bhi = cluster_bootstrap(rows)
        nc = [r for r in rows if r["task"] != NEG_CONTROL]
        kn = sum(1 for r in nc if r["success"] == "1")
        print("%-16s %5.1f%% [%4.1f, %5.1f]   %5.1f%% [%4.1f, %5.1f]   %5.1f%% (%d/%d)"
              % (lbl, p, lo, hi, bp, blo, bhi, 100.0 * kn / len(nc), kn, len(nc)))
    print("\n  The bootstrap interval is the one to report: it is ~60% wider,")
    print("  because 75 trials are 15 commands measured five times each.")

    full = dict(loaded)["LATCH (full)"]
    print("\n" + "=" * 78)
    print("PAIRED COMPARISONS against LATCH (McNemar exact on matched cells)")
    print("=" * 78)
    for lbl, rows in loaded:
        if lbl == "LATCH (full)":
            continue
        b, c, p = mcnemar(full, rows)
        print("  %-16s LATCH-only wins %2d | other-only wins %2d | p = %.3f"
              % (lbl, b, c, p))

    print("\n" + "=" * 78)
    print("SELF-REPORT (the headline), Fisher exact")
    print("=" * 78)
    d = dict(loaded)
    for a, b in (("naive", "LATCH (full)"), ("naive", "baseline"),
                 ("baseline", "LATCH (full)")):
        if a not in d or b not in d:
            continue
        sa = sum(1 for r in d[a] if str(r.get("planner_said")) == "1")
        sb = sum(1 for r in d[b] if str(r.get("planner_said")) == "1")
        na, nb = len(d[a]), len(d[b])
        print("  %-9s %2d/%d = %4.1f%%  vs  %-13s %2d/%d = %4.1f%%   p = %.4f"
              % (a, sa, na, 100.0 * sa / na, b, sb, nb, 100.0 * sb / nb,
                 fisher(sa, na - sa, sb, nb - sb)))

    print("\n" + "=" * 78)
    print("PER-PLACEMENT RELIABILITY on the five-object commands (c1, c4)")
    print("=" * 78)
    for lbl, rows in loaded:
        got, req = placements(rows)
        if not req:
            continue
        rate = got / req
        task = [r for r in rows if r["task"] in ("c1", "c4")]
        k = sum(1 for r in task if r["success"] == "1")
        print("  %-16s %2d/%2d placed = %4.1f%%   ^5 predicts %4.1f%%   observed %d/%d"
              % (lbl, got, req, 100 * rate, 100 * rate ** 5, k, len(task)))
    print("\n  n=50 placements per condition against n=10 task outcomes: the")
    print("  compounding constant is measured here, not assumed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

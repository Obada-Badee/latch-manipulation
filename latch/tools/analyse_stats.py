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
import glob
import io
import itertools
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


OK = lambda r: r["success"] == "1"                              # noqa: E731
SAID = lambda r: str(r.get("planner_said")) == "1"              # noqa: E731
OVER = lambda r: SAID(r) and r["success"] != "1"                # noqa: E731


def binom_two_sided(b, c):
    """Exact two-sided binomial p for b vs c discordant pairs, p = 0.5."""
    n = b + c
    if n == 0:
        return 1.0
    from math import comb
    tail = sum(comb(n, i) for i in range(0, min(b, c) + 1)) / (2.0 ** n)
    return min(1.0, 2.0 * tail)


def mcnemar(a_rows, b_rows, pred=OK, key=("task", "trial")):
    """Exact paired test on matched cells. Returns (b, c, p)."""
    A = {tuple(r[f] for f in key): pred(r) for r in a_rows}
    B = {tuple(r[f] for f in key): pred(r) for r in b_rows}
    both = set(A) & set(B)
    b = sum(1 for k in both if A[k] and not B[k])
    c = sum(1 for k in both if B[k] and not A[k])
    return b, c, binom_two_sided(b, c)


def paired_diff(a_rows, b_rows, pred, reps=20000, seed=12345):
    """a - b in percentage points on matched (task, trial) cells.

    Returns (point, lo, hi, p, n_cells): a 95% interval resampling COMMANDS,
    and an exact two-sided sign-flip p over the per-command differences -
    the command, not the trial, being the unit of analysis (see top).
    """
    A = {(r["task"], r["trial"]): pred(r) for r in a_rows}
    B = {(r["task"], r["trial"]): pred(r) for r in b_rows}
    cells = sorted(set(A) & set(B))
    if not cells:
        return 0.0, 0.0, 0.0, 1.0, 0
    per = collections.defaultdict(lambda: [0, 0])     # task -> [diff, cells]
    for k in cells:
        per[k[0]][0] += int(A[k]) - int(B[k])
        per[k[0]][1] += 1
    tasks = sorted(per)
    d = [per[t][0] for t in tasks]
    obs = abs(sum(d))
    hits = total = 0
    for signs in itertools.product((1, -1), repeat=len(d)):
        total += 1
        hits += abs(sum(sg * x for sg, x in zip(signs, d))) >= obs
    rng = random.Random(seed)
    sims = []
    for _ in range(reps):
        pick = [per[rng.choice(tasks)] for _ in tasks]
        sims.append(100.0 * sum(q[0] for q in pick) / sum(q[1] for q in pick))
    sims.sort()
    return (100.0 * sum(d) / len(cells), sims[int(0.025 * reps)],
            sims[int(0.975 * reps)], hits / total, len(cells))


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

    honesty_gap()
    recovery()
    return 0


def _k(rows, pred):
    return "%2d/%d" % (sum(1 for r in rows if pred(r)), len(rows))


def honesty_gap():
    """naive minus LATCH, on arms matched for detector and model.

    The registered analysis (paper/FOLLOWUP_PLAN_2026-11.md, Phase E):
    primary = self-reported success, command-level sign-flip p;
    secondary = over-claims, real success, and the trial-level tests the
    earlier drafts quoted, for comparison only.
    """
    pairs = [("v6 superseded detector, gpt-oss-120b", "full_v6.csv", "naive_v6.csv"),
             ("v7 corrected detector, gpt-oss-120b", "full_v7.csv", "naive_v7.csv")]
    for path in sorted(glob.glob(os.path.join(DATA, "full_v7_*.csv"))):
        tag = os.path.basename(path)[len("full_v7_"):-len(".csv")]
        pairs.append(("v7 corrected detector, model tag " + tag,
                      "full_v7_%s.csv" % tag, "naive_v7_%s.csv" % tag))

    print("\n" + "=" * 78)
    print("HONESTY GAP on matched arms (naive - LATCH)")
    print("=" * 78)
    for label, f_full, f_naive in pairs:
        full, naive = load(f_full), load(f_naive)
        if not full or not naive:
            continue
        model = sorted({r.get("model") for r in full + naive if r.get("model")})
        print("\n  %s%s" % (label, ("  [" + ", ".join(model) + "]") if model else ""))
        _, _, p = mcnemar(naive, full, OK)
        print("    real success   naive %s   LATCH %s   McNemar p = %.3f"
              % (_k(naive, OK), _k(full, OK), p))
        for name, pred in (("self-report", SAID), ("over-claims", OVER)):
            d, lo, hi, p_cmd, _ = paired_diff(naive, full, pred)
            _, _, p_mc = mcnemar(naive, full, pred)
            ka = sum(1 for r in naive if pred(r))
            kb = sum(1 for r in full if pred(r))
            p_f = fisher(ka, len(naive) - ka, kb, len(full) - kb)
            print("    %-14s naive %s   LATCH %s   diff %+5.1f [%+5.1f, %+5.1f]"
                  "   command-level p = %.3f | McNemar %.3f | Fisher %.3f"
                  % (name, _k(naive, pred), _k(full, pred), d, lo, hi,
                     p_cmd, p_mc, p_f))

    v6, v7 = load("full_v6.csv"), load("full_v7.csv")
    if v6 and v7:
        d, lo, hi, p_cmd, _ = paired_diff(v7, v6, SAID)
        _, _, p_mc = mcnemar(v7, v6, SAID)
        print("\n  REPLICATE: full_v7 vs full_v6 self-report   %s vs %s   diff %+5.1f"
              " [%+5.1f, %+5.1f]   command-level p = %.3f | McNemar %.3f"
              % (_k(v7, SAID), _k(v6, SAID), d, lo, hi, p_cmd, p_mc))
        print("  (the same system, two runs: the scale of run-to-run movement)")


def recovery_rows(arm):
    out = []
    pattern = os.path.join(DATA, "recovery_v7_%s_s*.csv" % arm)
    for path in sorted(glob.glob(pattern)):
        out += load(os.path.basename(path))
    return out


def recovery():
    """Induced failure against a control interleaved in the same session.

    Cells pair on (seed, task): both arms see the identical scene for a seed.
    Only trials where the injector actually fired measure recovery; the
    silent ones are reported separately because they are not a random subset.
    """
    induced, control = recovery_rows("induced"), recovery_rows("control")
    if not induced or not control:
        return
    print("\n" + "=" * 78)
    print("RECOVERY, induced vs matched control in the same session")
    print("=" * 78)
    fired = [r for r in induced if r.get("induced") == "1"]
    silent = [r for r in induced if r.get("induced") != "1"]
    ctl = {(r["seed"], r["task"]): r for r in control}
    fired_m = [r for r in fired if (r["seed"], r["task"]) in ctl]
    matched = [ctl[(r["seed"], r["task"])] for r in fired_m]
    _, _, p = mcnemar(fired_m, matched, OK, key=("seed", "task"))
    kf = sum(1 for r in fired_m if OK(r))
    kc = sum(1 for r in matched if OK(r))
    n = len(fired_m)
    print("  injector fired      %s succeeded" % _k(fired, OK))
    print("  matched control     %s succeeded (same seed and task)" % _k(matched, OK))
    if n:
        print("  difference          %+.1f points on %d pairs   McNemar p = %.3f"
              % (100.0 * (kf - kc) / n, n, p))
    print("  injector silent     %s succeeded (not a random subset)" % _k(silent, OK))
    print("  all control trials  %s succeeded (session health: expect ~24-26/30)"
          % _k(control, OK))


if __name__ == "__main__":
    raise SystemExit(main())

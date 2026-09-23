#!/usr/bin/env python3
"""Generate the paper's figures from the recorded numbers.

    python3 make_figures.py --out .

Figures are produced by script rather than drawn, so that when a number in
FINDINGS.md changes the figure cannot quietly disagree with it. Every value used
here appears in FINDINGS.md with its provenance; nothing is illustrative.

Two figures need no experimental data beyond what is already measured:

  F2  the directional clutter geometry - why a centre-distance proximity test
      cannot express the constraint the gripper actually has
  F4  synthetic mAP against live false-positive rate - the evaluation blind spot

F4 is the argument of Paper A in one picture: the metric that is used goes up
while the behaviour that matters gets worse.

Two more are drawn from the benchmark CSVs rather than from constants, because
they are counts over trials and re-deriving them by hand would defeat the point:

  F6  task success by complexity tier, with Wilson intervals, pooled over the
      two identical runs - and, drawn alongside it, the spread between those two
      runs, which is the honest statement of what this benchmark can resolve
  F5  the ablation comparison, once those runs exist

Both read the same CSVs that analyse_bench.py reads, so a figure cannot show a
number the results table does not.
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                  # noqa: E402
from matplotlib.patches import Circle, Rectangle                 # noqa: E402

# --- measured constants, from scene.py / affordance.py --------------------
OPEN_JAW_GAP = 0.060        # m, jaw opening at approach
FINGER_RADIUS = 0.012       # m
SAFE_CLEARANCE = 0.015      # m
CUBE = 0.040                # m, cube edge

# --- measured results, from FINDINGS.md §2.1-2.2 --------------------------
SYNTH_MAP = {"old (synthetic only)": 0.9892, "new (+ hard negatives)": 0.9945}
LIVE_FP = {"old (synthetic only)": 3.14, "new (+ hard negatives)": 0.03}
LIVE_RECALL = {"old (synthetic only)": 98.9, "new (+ hard negatives)": 99.6}


def fig_clutter(path: str) -> None:
    """F2 - the same centre distance, two different answers."""
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))
    gap = 0.050          # centre-to-centre distance, IDENTICAL in both panels

    for ax, along, title in (
            (axes[0], True, "neighbour ON the closing axis"),
            (axes[1], False, "neighbour BESIDE the jaw")):
        ax.set_aspect("equal")
        ax.set_xlim(-0.085, 0.085)
        ax.set_ylim(-0.055, 0.095)

        # target object at the origin, grasp axis along x
        ax.add_patch(Rectangle((-CUBE / 2, -CUBE / 2), CUBE, CUBE,
                               facecolor="#c8d8e8", edgecolor="#2c5070", lw=1.5))
        ax.text(0, 0, "target", ha="center", va="center", fontsize=8)

        # fingertips at +/- half the open jaw gap, along the grasp axis
        for sign in (-1, 1):
            fx = sign * OPEN_JAW_GAP / 2
            ax.add_patch(Circle((fx, 0), FINGER_RADIUS, facecolor="#f0c000",
                                edgecolor="#8a6d00", lw=1.4, alpha=0.9))
        # Below the target, so it cannot collide with a neighbour drawn above.
        ax.annotate("", xy=(OPEN_JAW_GAP / 2, -0.032),
                    xytext=(-OPEN_JAW_GAP / 2, -0.032),
                    arrowprops=dict(arrowstyle="<->", color="#8a6d00"))
        ax.text(0, -0.048, "closing axis", ha="center", fontsize=8,
                color="#8a6d00")

        # the neighbour, at the SAME centre distance in both panels
        nx, ny = (gap, 0.0) if along else (0.0, gap)
        ax.add_patch(Rectangle((nx - CUBE / 2, ny - CUBE / 2), CUBE, CUBE,
                               facecolor="#e8c8c8", edgecolor="#8a3030", lw=1.5))
        ax.text(nx, ny, "other", ha="center", va="center", fontsize=8)

        # Clearance is the minimum over BOTH fingertips, exactly as
        # affordance.clutter_clearance computes it. Measuring from the target's
        # centre instead would be a different (and wrong) quantity, and would
        # have put a score on this figure that the code never produces.
        d = min(math.hypot(sign * OPEN_JAW_GAP / 2 - nx, 0.0 - ny)
                for sign in (-1, 1)) - (FINGER_RADIUS + CUBE / 2)

        if d <= 0:
            verdict = f"VETO  (fingertip overlaps by {abs(d) * 1000:.0f} mm)"
            colour = "#a01010"
        elif d < SAFE_CLEARANCE:
            verdict = (f"score {d / SAFE_CLEARANCE:.2f}  "
                       f"(tight: {d * 1000:.0f} mm)")
            colour = "#b06000"
        else:
            verdict = f"score 1.00  (clear by {d * 1000:.0f} mm)"
            colour = "#107010"

        ax.set_title(f"{title}\ncentre distance {gap * 1000:.0f} mm",
                     fontsize=9.5)
        ax.text(0, 0.082, verdict, ha="center", fontsize=10, color=colour,
                weight="bold")
        ax.axis("off")

    fig.suptitle("A centre-distance test gives these the same score. "
                 "The gripper does not.", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def fig_metric_blindspot(path: str) -> None:
    """F4 - the metric improves in a direction unrelated to the failure."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.2))
    labels = list(SYNTH_MAP)
    colours = ["#9aa5b1", "#2e7d32"]

    bars = ax1.bar(range(2), [SYNTH_MAP[k] for k in labels], color=colours,
                   width=0.55)
    ax1.set_ylim(0.97, 1.0)
    ax1.set_ylabel("mAP@50 on the synthetic validation set")
    ax1.set_title("What the standard metric sees", fontsize=10)
    ax1.set_xticks(range(2))
    ax1.set_xticklabels(labels, fontsize=8)
    for b, k in zip(bars, labels):
        ax1.text(b.get_x() + b.get_width() / 2, SYNTH_MAP[k] + 0.0005,
                 f"{SYNTH_MAP[k]:.4f}", ha="center", fontsize=9)
    ax1.text(0.5, 0.9755, "both look solved", ha="center", fontsize=9,
             style="italic", color="#555555", transform=ax1.transData)

    bars = ax2.bar(range(2), [LIVE_FP[k] for k in labels], color=colours,
                   width=0.55)
    ax2.set_ylabel("false positives per frame, live cell (arm in view)")
    ax2.set_title("What the robot actually experiences", fontsize=10)
    ax2.set_xticks(range(2))
    ax2.set_xticklabels(labels, fontsize=8)
    for b, k in zip(bars, labels):
        ax2.text(b.get_x() + b.get_width() / 2, LIVE_FP[k] + 0.06,
                 f"{LIVE_FP[k]:.2f}", ha="center", fontsize=9)

    fig.suptitle("The synthetic validation set contains no robot arm, "
                 "so it cannot express the failure mode", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


RUNS_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "..", "..", "latch", "runs"))
#: The benchmark writes into the WSL workspace, not the authoring tree, so allow
#: an override rather than hardcoding one machine's layout.
RUNS_DIR = os.environ.get("LATCH_RUNS", RUNS_DIR)


def wilson(k: int, n: int, z: float = 1.96):
    """Point estimate and Wilson interval - the same function analyse_bench uses.

    Not the normal approximation: these are small samples near the boundary,
    where it gives [1.0, 1.0] at 20/20 and claims certainty from twenty trials.
    """
    if not n:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = (z / d) * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return p, max(0.0, centre - half), min(1.0, centre + half)


def _load(name: str):
    """Scored rows of one run, void trials excluded."""
    path = os.path.join(RUNS_DIR, name)
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("outcome") != "void"]
    # Same rule as benchmark.scored_rows: a 429 that interrupted an unfinished
    # task is not a measurement, and a cell re-run keeps its latest row.
    keep = {}
    for r in rows:
        if "429" in (r.get("void_reason") or "") and r.get("success") != "1":
            continue
        keep[(r.get("condition"), r.get("task"), r.get("trial"))] = r
    return list(keep.values())


def _rate(rows, pred=lambda r: True):
    sel = [r for r in rows if pred(r)]
    ok = sum(1 for r in sel if r["success"] == "1")
    return ok, len(sel)


def fig_tiers(path: str) -> None:
    """F6: success by tier, pooled, with the two source runs shown separately.

    The two runs are drawn as well as the pool on purpose. They are the SAME
    configuration measured twice, and the replicate spread on this suite is
    up to 5 points overall and 24 on a single tier (FINDINGS 1l). Showing only the pooled bar would present a
    precision the experiment does not have; showing the spread is the argument
    for why single-run tables of a dozen trials are not evidence.
    """
    # The two v6 runs of the full system: `full` and `no_memory`, which is the
    # same configuration on this suite because memory cannot act within a
    # single command (FINDINGS 1k). Both scored against an honest `place`.
    runs = [("run 1", _load("full_v6.csv")),
            ("run 2", _load("abl_no_memory_v6.csv"))]
    runs = [(n, r) for n, r in runs if r]
    if not runs:
        print("F6 skipped: no benchmark CSVs found in", RUNS_DIR)
        return
    pooled = [r for _, rows in runs for r in rows]
    tiers = ("simple", "medium", "complex")

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    xs = range(len(tiers))
    ok_n = [_rate(pooled, lambda r, t=t: r["tier"] == t) for t in tiers]
    est = [wilson(k, n) for k, n in ok_n]
    heights = [100 * e[0] for e in est]
    lo = [100 * (e[0] - e[1]) for e in est]
    hi = [100 * (e[2] - e[0]) for e in est]

    ax.bar(xs, heights, width=0.55, color="#4C72B0", zorder=2,
           label="pooled (n=%d)" % len(pooled))
    ax.errorbar(xs, heights, yerr=[lo, hi], fmt="none", ecolor="#22303f",
                capsize=6, lw=1.6, zorder=3)

    marks = ("o", "s")
    for (label, rows), m in zip(runs, marks):
        pts = []
        for t in tiers:
            k, n = _rate(rows, lambda r, t=t: r["tier"] == t)
            pts.append(100 * k / n if n else 0)
        ax.scatter(xs, pts, marker=m, s=42, zorder=4, facecolor="white",
                   edgecolor="#C44E52", lw=1.6, label="%s alone" % label)

    for x, (k, n) in zip(xs, ok_n):
        ax.text(x, 3, "%d/%d" % (k, n), ha="center", va="bottom",
                fontsize=9, color="white", zorder=5)

    ax.set_xticks(list(xs))
    ax.set_xticklabels([t.capitalize() for t in tiers])
    ax.set_ylabel("task success (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Success by task complexity" + chr(10) +
                 "markers: the same system measured twice",
                 fontsize=11)
    ax.grid(axis="y", alpha=0.25, zorder=0)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print("wrote", path)


def fig_ablations(path: str) -> None:
    """F5: real success against self-reported success, per condition.

    This is the paper's central figure. Each condition gets two bars: what
    the simulator says happened (with a Wilson interval) and what the agent
    said happened. The naive agent - unverified skills - achieves the same
    real success as the full system and reports nineteen points more
    (FINDINGS 1o). A transcript-scored evaluation would rank it first.
    """
    conditions = [
        ("LATCH",              "full_v6.csv"),
        ("no affordance",      "abl_no_affordance_v6.csv"),
        ("no memory",          "abl_no_memory_v6.csv"),
        ("baseline\n(no A, no M)", "baseline_v6.csv"),
        ("naive\n(unverified)", "naive_v6.csv"),
    ]
    have = [(lbl, _load(f)) for lbl, f in conditions]
    have = [(lbl, rows) for lbl, rows in have if rows]
    if len(have) < 2:
        print("F5 skipped: found %d conditions in %s" % (len(have), RUNS_DIR))
        return

    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    xs = list(range(len(have)))
    w = 0.36
    real, lo, hi, said = [], [], [], []
    for lbl, rows in have:
        k, n = _rate(rows)
        p, a, b = wilson(k, n)
        real.append(100 * p); lo.append(100 * (p - a)); hi.append(100 * (b - p))
        said.append(100 * sum(1 for r in rows if str(r.get("planner_said")) == "1") / n)
    ax.bar([x - w / 2 for x in xs], real, width=w, color="#4C72B0", zorder=2,
           label="real (simulator ground truth)")
    ax.errorbar([x - w / 2 for x in xs], real, yerr=[lo, hi], fmt="none",
                ecolor="#22303f", capsize=5, lw=1.4, zorder=3)
    ax.bar([x + w / 2 for x in xs], said, width=w, color="#DD8452", zorder=2,
           label="self-reported (agent transcript)")
    for x, r, sd in zip(xs, real, said):
        # real: inside the bar, clear of the interval cap; said: above its bar
        ax.text(x - w / 2, 4, "%.0f" % r, ha="center", va="bottom",
                fontsize=8.5, color="white", zorder=5)
        ax.text(x + w / 2, sd + 1.5, "%.0f" % sd, ha="center", fontsize=8.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([lbl for lbl, _ in have], fontsize=9)
    ax.set_ylabel("task success (%)")
    ax.set_ylim(0, 100)
    ax.set_title("What the world says vs what the agent says (n=75 per condition)",
                 fontsize=10.5)
    ax.grid(axis="y", alpha=0.25, zorder=0)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print("wrote", path)


def fig_taxonomy(path: str) -> None:
    """F8: where the 58 failures of 150 full-system trials went (FINDINGS 1p)."""
    classes = [
        ("execution: right objects grasped,\nplacement incomplete", 44),
        ("timeout: right objects,\nran out of steps", 6),
        ("grounding: binned a\nnon-target object", 3),
        ("collateral: right object placed,\nanother disturbed", 3),
        ("execution: never\nconfirmed a grasp", 2),
    ]
    n = sum(v for _, v in classes)
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    ys = list(range(len(classes)))[::-1]
    cols = ["#4C72B0", "#8C8C8C", "#C44E52", "#8C8C8C", "#4C72B0"]
    ax.barh(ys, [v for _, v in classes], color=cols, zorder=2)
    for y, (_, v) in zip(ys, classes):
        ax.text(v + 0.6, y, "%d  (%.0f%%)" % (v, 100 * v / n), va="center", fontsize=9)
    ax.set_yticks(ys)
    ax.set_yticklabels([c for c, _ in classes], fontsize=8.5)
    ax.set_xlabel("failures (of %d in 150 trials)" % n)
    ax.set_xlim(0, 52)
    ax.set_title("Where the full system loses: grounding 5%, execution 79%", fontsize=10.5)
    ax.grid(axis="x", alpha=0.25, zorder=0)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print("wrote", path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    fig_clutter(os.path.join(args.out, "F2_directional_clutter.png"))
    fig_metric_blindspot(os.path.join(args.out, "F4_metric_blindspot.png"))
    fig_tiers(os.path.join(args.out, "F6_success_by_tier.png"))
    fig_ablations(os.path.join(args.out, "F5_ablations.png"))
    fig_taxonomy(os.path.join(args.out, "F8_failure_taxonomy.png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

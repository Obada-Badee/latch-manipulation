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
import textwrap

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
# Three models, measured on the curated-pose probe (FINDINGS 2.1; the 960 px
# row measured 2026-10-01 once its weights were recovered). The novel-pose
# probe gives the same ordering: 3.14 / 3.83 / 0.03.
SYNTH_MAP = {"synthetic only\n640 px": 0.9892,
             "synthetic only\n960 px": 0.9917,
             "+ hard negatives\n640 px": 0.9945}
LIVE_FP = {"synthetic only\n640 px": 3.17,
           "synthetic only\n960 px": 4.10,
           "+ hard negatives\n640 px": 0.00}


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
    """F4 - the metric does not order the models by deployment quality.

    Left panel's y-axis starts at 0.98, deliberately, because the whole
    spread is 0.005 and is invisible from zero. A truncated axis in a paper
    about misleading metrics must say so on its face, so the break is drawn
    and labelled rather than left for the reader to notice.
    """
    labels = list(SYNTH_MAP)
    colours = ["#9aa5b1", "#6b7785", "#2e7d32"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.8, 4.2))
    xs = range(len(labels))

    bars = ax1.bar(xs, [SYNTH_MAP[k] for k in labels], color=colours, width=0.6)
    ax1.set_ylim(0.980, 1.000)
    ax1.set_ylabel("mAP@50, synthetic validation set")
    ax1.set_title("What the standard metric sees", fontsize=10)
    ax1.set_xticks(list(xs))
    ax1.set_xticklabels(labels, fontsize=8)
    for b, k in zip(bars, labels):
        ax1.text(b.get_x() + b.get_width() / 2, SYNTH_MAP[k] + 0.0004,
                 f"{SYNTH_MAP[k]:.4f}", ha="center", fontsize=9)
    # the axis break, drawn and named
    kw = dict(transform=ax1.transAxes, color="k", clip_on=False, lw=1)
    for dy in (0.0, 0.025):
        ax1.plot((-0.03, 0.03), (0.02 + dy, 0.045 + dy), **kw)
    ax1.text(0.04, 0.03, "axis starts at 0.980", transform=ax1.transAxes,
             fontsize=7.5, color="#555555", va="bottom")
    ax1.text(0.5, 0.88, "spread: 0.0053 - all three look solved",
             transform=ax1.transAxes, ha="center", fontsize=8.5,
             style="italic", color="#555555")

    bars = ax2.bar(xs, [LIVE_FP[k] for k in labels], color=colours, width=0.6)
    ax2.set_ylabel("false positives per frame, deployment cell")
    ax2.set_title("What the robot actually experiences", fontsize=10)
    ax2.set_xticks(list(xs))
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_ylim(0, 4.8)
    for b, k in zip(bars, labels):
        ax2.text(b.get_x() + b.get_width() / 2, LIVE_FP[k] + 0.08,
                 f"{LIVE_FP[k]:.2f}", ha="center", fontsize=9)

    fig.suptitle("The synthetic validation set contains no robot arm: higher mAP "
                 "is not better on the cell", fontsize=11)
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


def cluster_ci(rows, field="success", reps=20000, seed=12345):
    """Point estimate and 95% interval, resampling COMMANDS, not trials.

    Same method as tools/analyse_stats.py, and what the paper says it uses.
    75 trials are 15 commands measured five times; their outcomes cluster by
    command, so a trial-level binomial interval understates the uncertainty
    by roughly 60%. A figure drawn with Wilson intervals would contradict
    the text that cites it.
    """
    import collections as _c
    import random as _r
    by = _c.defaultdict(list)
    for r in rows:
        by[r["task"]].append(str(r.get(field)) == "1")
    tasks = list(by)
    if not tasks:
        return 0.0, 0.0, 0.0
    point = sum(sum(v) for v in by.values()) / sum(len(v) for v in by.values())
    rng = _r.Random(seed)
    sims = []
    for _ in range(reps):
        pick = [by[rng.choice(tasks)] for _ in tasks]
        sims.append(sum(sum(v) for v in pick) / sum(len(v) for v in pick))
    sims.sort()
    return point, sims[int(0.025 * reps)], sims[int(0.975 * reps)]


#: The negative control: succeeds 5/5 in every condition, so it is excluded
#: from per-tier figures (FINDINGS 1i / review B-5) and reported separately.
NEG_CONTROL = "c5"


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
    """F6: success by tier for the full system, two replicate runs.

    The two runs are `full` and `no_memory`, which are the same configuration
    on this suite because memory cannot act within a single command
    (FINDINGS 1k). Bars pool them; markers show each alone, because the
    run-to-run spread is the argument for the intervals. Intervals resample
    commands within each tier (5 commands per tier), which is honest and wide.

    The negative control is excluded from the complex tier: it succeeds in
    every condition and would add a fixed five points to the bar.
    """
    runs = [("run 1", _load("full_v6.csv")), ("run 2", _load("abl_no_memory_v6.csv"))]
    runs = [(n, r) for n, r in runs if r]
    if not runs:
        print("F6 skipped: no benchmark CSVs found in", RUNS_DIR)
        return
    keep = lambda r: r["task"] != NEG_CONTROL
    pooled = [r for _, rows in runs for r in rows if keep(r)]
    tiers = ("simple", "medium", "complex")

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    xs = list(range(len(tiers)))
    est = []
    for t in tiers:
        sel = [r for r in pooled if r["tier"] == t]
        # bootstrap over (run, command) pairs would double-count commands;
        # resample commands, carrying both runs' trials with each
        est.append((cluster_ci(sel, "success"), sum(r["success"] == "1" for r in sel), len(sel)))
    heights = [100 * e[0][0] for e in est]
    lo = [100 * (e[0][0] - e[0][1]) for e in est]
    hi = [100 * (e[0][2] - e[0][0]) for e in est]
    ax.bar(xs, heights, width=0.55, color="#4C72B0", zorder=2, label="both runs pooled")
    ax.errorbar(xs, heights, yerr=[lo, hi], fmt="none", ecolor="#22303f",
                capsize=6, lw=1.6, zorder=3)
    for (label, rows), m in zip(runs, ("o", "s")):
        pts = []
        for t in tiers:
            sel = [r for r in rows if r["tier"] == t and keep(r)]
            pts.append(100 * sum(r["success"] == "1" for r in sel) / len(sel) if sel else 0)
        ax.scatter(xs, pts, marker=m, s=42, zorder=4, facecolor="white",
                   edgecolor="#C44E52", lw=1.6, label="%s alone" % label)
    for x, (_, k, n) in zip(xs, est):
        ax.text(x, 3, "%d/%d" % (k, n), ha="center", va="bottom",
                fontsize=9, color="white", zorder=5)
    ax.set_xticks(xs)
    ax.set_xticklabels(["Simple", "Medium", "Complex\n(excl. negative control)"])
    ax.set_ylabel("task success (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Success by task tier, full system measured twice\n"
                 "95% intervals resample commands within each tier", fontsize=10.5)
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
        p, a, b = cluster_ci(rows, "success")
        real.append(100 * p); lo.append(100 * (p - a)); hi.append(100 * (b - p))
        said.append(100 * cluster_ci(rows, "planner_said")[0])
    ax.bar([x - w / 2 for x in xs], real, width=w, color="#4C72B0", zorder=2,
           label="real (simulator ground truth)")
    ax.errorbar([x - w / 2 for x in xs], real, yerr=[lo, hi], fmt="none",
                ecolor="#22303f", capsize=5, lw=1.4, zorder=3)
    # Self-report is a binomial proportion too. Drawing it without an
    # interval, beside a bar that has one, implies an exactness it does not
    # have - in a figure whose whole subject is unwarranted confidence.
    slo, shi = [], []
    for _, rows in have:
        p_, a, b = cluster_ci(rows, "planner_said")
        slo.append(100 * (p_ - a)); shi.append(100 * (b - p_))
    ax.bar([x + w / 2 for x in xs], said, width=w, color="#DD8452", zorder=2,
           label="self-reported (agent transcript)")
    ax.errorbar([x + w / 2 for x in xs], said, yerr=[slo, shi], fmt="none",
                ecolor="#22303f", capsize=5, lw=1.4, zorder=3)
    for x, r, sd in zip(xs, real, said):
        # real: inside the bar, clear of the interval cap; said: above its bar
        ax.text(x - w / 2, 4, "%.0f" % r, ha="center", va="bottom",
                fontsize=8.5, color="white", zorder=5)
        ax.text(x + w / 2, 4, "%.0f" % sd, ha="center", va="bottom",
                fontsize=8.5, color="white", zorder=5)
    ax.set_xticks(xs)
    ax.set_xticklabels([lbl for lbl, _ in have], fontsize=9)
    ax.set_ylabel("task success (%)")
    ax.set_ylim(0, 100)
    ax.set_title("What the world says vs what the agent says\n"
                 "n = 75 per condition; 95% intervals resample commands",
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


# --------------------------------------------------------------------------
# F1 - the arm false positives, before and after hard negatives.
#
# Needs the two weight sets and one frame containing the arm. Both live in the
# WSL workspace rather than here, so paths are arguments with sensible
# defaults and the figure is skipped rather than failing when they are absent.
# --------------------------------------------------------------------------

#: Boxes below this are not drawn. The same threshold the live detector node
#: publishes at, so the picture shows what the planner would actually see.
CONF = 0.50


def _predict(weights, image):
    """(polygon, class_id, confidence) per detection, as numpy arrays."""
    from ultralytics import YOLO                                 # noqa: PLC0415
    model = YOLO(os.path.expanduser(weights))
    imgsz = int(getattr(model, "overrides", {}).get("imgsz", 640) or 640)
    r = model.predict(image, conf=CONF, imgsz=imgsz, verbose=False)[0]
    if r.obb is None or len(r.obb) == 0:
        return [], [], []
    return (r.obb.xyxyxyxy.cpu().numpy(),
            r.obb.cls.cpu().numpy().astype(int),
            r.obb.conf.cpu().numpy())


def _truth_centres(label_path, w, h):
    out = []
    if not os.path.exists(label_path):
        return out
    with open(label_path, encoding="utf-8") as fh:
        for line in fh:
            f = line.split()
            if len(f) != 9:
                continue
            xs = [float(f[1 + 2 * i]) * w for i in range(4)]
            ys = [float(f[2 + 2 * i]) * h for i in range(4)]
            out.append((sum(xs) / 4.0, sum(ys) / 4.0))
    return out


def fig_arm_fps(path, image=None, old_w=None, new_w=None):
    """F1: the same frame through both detectors.

    Left: trained on synthetic frames only, which contain no robot arm.
    Right: the same architecture with hard negatives added. Boxes that match
    a ground-truth object are drawn in one colour, boxes that match nothing -
    the false positives, all of them on the arm - in another. This is the
    figure for the claim that the synthetic validation metric cannot see the
    failure: both models score ~0.99 mAP on that set.
    """
    import matplotlib.image as mpimg                             # noqa: PLC0415
    from matplotlib.patches import Polygon                       # noqa: PLC0415

    image = image or os.path.expanduser(
        "~/synth_obb_probe_novel/images/train/neg_00010.png")
    old_w = old_w or os.path.expanduser(
        "~/moveit2_obb/install/yolov8_obb/share/yolov8_obb/scripts/best_synthetic_only.pt")
    # The hard-negative model, NOT the deployed best.pt: as of FINDINGS 2.5
    # those are different files, and the deployed one is the "before".
    new_w = new_w or os.path.expanduser(
        "~/moveit2_obb/src/yolov8_obb/scripts/best_hard_negatives.pt")
    for p in (image, old_w, new_w):
        if not os.path.exists(p):
            print("F1 skipped: missing", p)
            return

    img = mpimg.imread(image)
    h, w = img.shape[:2]
    truth = _truth_centres(
        image.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt", w, h)

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.3))
    for ax, weights, title in ((axes[0], old_w, "trained on synthetic frames only"),
                               (axes[1], new_w, "+ hard negatives")):
        ax.imshow(img)
        polys, _, _ = _predict(weights, image)
        fp = 0
        for poly in polys:
            cu, cv = poly[:, 0].mean(), poly[:, 1].mean()
            hit = any((cu - tu) ** 2 + (cv - tv) ** 2 <= 40.0 ** 2 for tu, tv in truth)
            if not hit:
                fp += 1
            ax.add_patch(Polygon(poly, closed=True, fill=False, lw=1.8,
                                 edgecolor="#4C72B0" if hit else "#C44E52"))
        ax.set_title("%s\n%d false positive%s on this frame"
                     % (title, fp, "" if fp == 1 else "s"), fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    # one legend for both panels
    from matplotlib.lines import Line2D                           # noqa: PLC0415
    fig.legend(handles=[Line2D([], [], color="#4C72B0", lw=2, label="matches a real object"),
                        Line2D([], [], color="#C44E52", lw=2, label="matches nothing (false positive)")],
               loc="lower center", ncol=2, frameon=False, fontsize=9)
    fig.suptitle("Both models score ~0.99 mAP on the synthetic validation set",
                 fontsize=11, y=0.99)
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print("wrote", path)


def fig_layout_duplicate(path, probe=None, train=None):
    """F3: two frames that are different images and the same experiment.

    Left is a "held-out" probe frame; right is a training frame. Their pixels
    differ - the image hash finds nothing, correctly, because the arm moved
    between the two capture runs. Their object layouts are identical, because
    both runs used capture_negatives.py's default --seed 7 and the layout
    sequence is a pure function of the seed. Ground-truth boxes are drawn on
    both so the reader can see the table repeat while the arm does not.

    The pair is not cherry-picked: 17 of the probe's 17 frames match a
    training frame at 100% layout overlap (FINDINGS 2.3).
    """
    import hashlib                                              # noqa: PLC0415
    import matplotlib.image as mpimg                            # noqa: PLC0415
    from matplotlib.patches import Polygon                      # noqa: PLC0415

    probe = probe or os.path.expanduser(
        "~/synth_obb_neg_probe/images/train/neg_00018.png")
    train = train or os.path.expanduser(
        "~/synth_obb_neg/images/train/neg_00018.png")
    for f in (probe, train):
        if not os.path.exists(f):
            print("F3 skipped: missing", f)
            return

    def boxes(image_path, w, h):
        lp = image_path.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
        out = []
        if os.path.exists(lp):
            for line in open(lp, encoding="utf-8"):
                f = line.split()
                if len(f) == 9:
                    out.append([[float(f[1 + 2 * i]) * w, float(f[2 + 2 * i]) * h]
                                for i in range(4)])
        return out

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.3))
    digests = []
    for ax, f, title in ((axes[0], probe, '"held-out" probe frame'),
                         (axes[1], train, "training frame")):
        img = mpimg.imread(f)
        h, w = img.shape[:2]
        ax.imshow(img)
        for poly in boxes(f, w, h):
            ax.add_patch(Polygon(poly, closed=True, fill=False, lw=1.6,
                                 edgecolor="#4C72B0"))
        digests.append(hashlib.md5(open(f, "rb").read()).hexdigest()[:8])
        ax.set_title("%s\n%s" % (title, os.path.basename(f)), fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Same seed, same table, different pixels", fontsize=12, y=0.99)
    fig.text(0.5, 0.045,
             "image hash:  %s  vs  %s   -  no duplicate found\n"
             "layout overlap:  100%%   -  and 17 of the probe's 17 frames match one"
             % (digests[0], digests[1]),
             ha="center", fontsize=9.5)
    fig.tight_layout(rect=(0, 0.11, 1, 0.93))
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print("wrote", path)


def fig_episode(path, src=None):
    """F7: one command, call by call, with what the world looked like after each.

    Built from `tools/capture_episode.py`, which saves a frame per tool call
    rather than per wall-clock interval - a frame every N seconds lands
    mid-motion and belongs to no decision.

    The episode shipped here is `m4`, "put both yellow cubes in the blue bin",
    and it was not selected for being tidy. It contains, in order: a grasp the
    skill layer REFUSED because the fingers closed on nothing; a successful
    pick and place; a neighbour moved aside with place_at to clear a crowded
    grasp; the second cube picked and placed - and a final verdict of WRONG,
    because one of the two never landed in the bin. That last step is the
    residual over-claim of FINDINGS 1i in a single picture: `place` verifies
    that it held something and that it released, not where the object went.
    """
    import json                                                  # noqa: PLC0415
    import matplotlib.image as mpimg                             # noqa: PLC0415

    src = src or os.path.join(os.path.dirname(os.path.abspath(__file__)), "episode")
    meta = os.path.join(src, "episode.json")
    if not os.path.exists(meta):
        print("F7 skipped: no episode.json in", src)
        return
    with open(meta, encoding="utf-8") as fh:
        events = json.load(fh)
    command = next((e["text"] for e in events if e.get("event") == "command"), "")
    calls = [e for e in events if e.get("event") == "tool_call"
             and os.path.exists(os.path.join(src, e["file"]))]
    if not calls:
        print("F7 skipped: no frames")
        return

    cols = 4
    rows = (len(calls) + cols - 1) // cols
    # 3.9 per row, not 3.05: each panel carries a two-line caption beneath it
    # and the next row's title above, and tight_layout will not make room.
    fig, axes = plt.subplots(rows, cols, figsize=(3.1 * cols, 3.9 * rows))
    axes = axes.ravel() if hasattr(axes, "ravel") else [axes]
    for ax, ev in zip(axes, calls):
        ax.imshow(mpimg.imread(os.path.join(src, ev["file"])))
        args = ", ".join("%s=%s" % (k, v) for k, v in ev["input"].items())
        verdict = ev["result"].split(":", 1)[0].strip()
        detail = ev["result"].split(":", 1)[-1].strip()
        # Two short wrapped lines. Unwrapped, a 90-character result runs into
        # the neighbouring panel and under the next row's title.
        detail = textwrap.fill(detail, 34, max_lines=2, placeholder=" ...")
        ax.set_title("%d. %s(%s)" % (ev["n"], ev["tool"], args), fontsize=8.5)
        ax.set_xlabel("%s\n%s" % (verdict, detail), fontsize=7.2, linespacing=1.3,
                      color="#C44E52" if ev["is_error"] else "#3b6e3b")
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axes[len(calls):]:
        ax.axis("off")
    fig.suptitle('"%s"   ->   scored WRONG: 1 of 2 reached the bin'
                 % command, fontsize=11, y=0.995)
    fig.subplots_adjust(top=0.88, bottom=0.07, left=0.02, right=0.98,
                        hspace=0.46, wspace=0.10)
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
    fig_arm_fps(os.path.join(args.out, "F1_arm_false_positives.png"))
    fig_layout_duplicate(os.path.join(args.out, "F3_layout_duplicate.png"))
    fig_episode(os.path.join(args.out, "F7_episode.png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

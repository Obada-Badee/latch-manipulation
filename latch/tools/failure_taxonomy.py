#!/usr/bin/env python3
"""Failure taxonomy for a benchmark run, with grounding resolved from the log.

    python3 tools/failure_taxonomy.py runs/full_v6.csv runs/abl_no_memory_v6.csv

The scorer's detail string cannot separate "binned the wrong cube" from
"picked the right cube and dropped it" - both read "X was the right object
but is not in the bin". That is exactly the grounding/execution line the
paper draws (FINDINGS 1b), so it is resolved here from the decision log.

Trials are aligned by the log's `command` event, which opens every trial
with the command text, and rows are matched to log segments in order by
that text. This survives resumed runs and multiple log files.

Grounding is *binning a non-target object*, not merely picking one: with
affordances on, the model is told to move a crowding neighbour out of the
way first, and does so with `place_at` onto the table. That is correct
behaviour and must not be scored as a grounding error. Only `place_in_bin`
of a non-target counts.
"""
import collections
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark import scored_rows          # noqa: E402
import bench_tasks                          # noqa: E402

TASKS = {t.id: t for t in bench_tasks.SUITE}
WORD = r"(red|blue|green|yellow|purple|grey)?\s*(cube|bolt|cylinder)s?"


def targets(task_id):
    """(class, colour-or-None) pairs the command names. [] = anything counts."""
    if task_id == "c4":                                    # "tidy up the table"
        return []
    return [(cls, col or None) for col, cls in re.findall(WORD, TASKS[task_id].command.lower())]


def all_segments():
    """Every trial in every decision log, in time order: (command, events)."""
    segs = []
    for f in sorted(glob.glob(os.path.expanduser("~/moveit2_obb/runs/*.jsonl")),
                    key=os.path.getmtime):
        cur = None
        for line in open(f, encoding="utf-8"):
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("event") == "command":
                cur = {"command": ev.get("command", ""), "events": [], "file": f}
                segs.append(cur)
            elif cur is not None:
                cur["events"].append(ev)
    return segs


def align(rows, segs, csv):
    """Match rows to segments by command text, in order, within the run's time window."""
    t0 = os.path.getctime(csv) - 600
    cands = [s for s in segs if os.path.getmtime(s["file"]) >= t0]
    out, i = [], 0
    for r in rows:
        cmd = (r.get("command") or TASKS[r["task"]].command).strip().lower()
        while i < len(cands) and cands[i]["command"].strip().lower() != cmd:
            i += 1
        if i < len(cands):
            out.append((r, cands[i]))
            i += 1
        else:
            out.append((r, None))
    return out


def classify_one(r, seg):
    tg = targets(r["task"])
    d = r.get("detail") or ""
    if seg is None:
        return "unaligned", [], []
    held = None                                     # (class, colour) currently held
    picked, binned_wrong = [], []
    said_success = None
    for ev in seg["events"]:
        if ev.get("event") == "done":
            said_success = bool(ev.get("success"))
        if ev.get("event") != "tool_call" or ev.get("is_error"):
            continue
        t = ev["tool"]
        head = str(ev.get("result_head", ""))
        if t == "pick":
            m = re.search(r"picked up object \d+ \((\w+) (\w+)\)", head)
            if m:
                held = (m.group(2), m.group(1))
                picked.append(held)
        elif t == "place_in_bin" and held is not None:
            if tg and not any(held[0] == c and (col is None or held[1] == col) for c, col in tg):
                binned_wrong.append(held)
            held = None
        elif t == "place_at":
            held = None
    if binned_wrong:
        return "grounding: binned a non-target object", picked, binned_wrong
    if "also moved" in d:
        return "collateral: right object placed, another disturbed", picked, []
    if not picked and said_success:
        return "grounding: declared success without grasping anything", picked, []
    if not picked:
        return "execution: never confirmed a grasp", picked, []
    if r["outcome"] == "timeout":
        return "timeout: right objects, ran out of steps", picked, []
    return "execution: right objects grasped, placement incomplete", picked, []


def main(paths):
    segs = all_segments()
    allc = []
    for p in paths:
        rows = scored_rows(p)
        pairs = align(rows, segs, p)
        unal = sum(1 for _, s in pairs if s is None)
        print(f"{os.path.basename(p)}: {len(rows)} rows, {len(rows) - unal} aligned to a log segment")
        for r, s in pairs:
            if r["success"] == "1":
                continue
            cls, picked, wrong = classify_one(r, s)
            allc.append(dict(task=r["task"], trial=r["trial"], cls=cls, detail=(r.get("detail") or "")[:55],
                             picked=picked, wrong=wrong))
    n = len(allc)
    print(f"\n{n} failures\n")
    for k, v in collections.Counter(x["cls"] for x in allc).most_common():
        print(f"  {v:3d}  ({100 * v / n:3.0f}%)  {k}")
    g = [x for x in allc if x["cls"].startswith("grounding")]
    if g:
        print("\ngrounding cases:")
        for x in g:
            print(f"   {x['task']} t{x['trial']}  binned-wrong {x['wrong'][:2]}  picked {x['picked'][:4]}  | {x['detail']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or ["runs/full_v6.csv"]))

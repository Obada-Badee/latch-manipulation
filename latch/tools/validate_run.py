#!/usr/bin/env python3
"""Audit a benchmark run, live or finished, against the ways runs have lied.

    python3 tools/validate_run.py runs/full_v6.csv

Each section is a check that would have caught a past failure had it existed
then. It reads the results CSV through `scored_rows()` (the one definition of
a counted trial) and the planner's decision log for the same period.

  GUARD    place succeeded with nothing picked           (FINDINGS 4c)
  CAP      trials that ran to MAX_ITERATIONS              (1g, 4.16)
  NUDGE    repetition nudges sent, and whether the trial then resolved or capped
  CLAIM    planner said success, world says otherwise    (1b - the over-claim)
  ROBOT    void / skill-server-gone rows                  (4.17)
  QUOTA    rate-limit voids                               (rate-limit retry)
  DUP      duplicate (condition, task, trial) rows        (scored_rows)
"""
import collections
import csv
import glob
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark import scored_rows          # noqa: E402


def decision_log_since(t0):
    files = sorted(glob.glob(os.path.expanduser("~/moveit2_obb/runs/*.jsonl")),
                   key=os.path.getmtime)
    return [f for f in files if os.path.getmtime(f) >= t0 - 60]


def main(path):
    raw = list(csv.DictReader(io.open(path, encoding="utf-8")))
    rows = scored_rows(path)
    t0 = os.path.getctime(path)
    ok = sum(1 for r in rows if r["success"] == "1")
    print(f"{os.path.basename(path)}: {len(raw)} rows on disk, {len(rows)} scored, "
          f"{ok} succeeded ({100.0 * ok / max(len(rows), 1):.0f}%)")

    # DUP
    c = collections.Counter((r["condition"], r["task"], r["trial"]) for r in raw)
    dups = [k for k, v in c.items() if v > 1]
    print(f"  DUP    {len(dups)} duplicated cells" + (f": {dups}" if dups else ""))

    # ROBOT / QUOTA
    void = [r for r in raw if r["outcome"] == "void"]
    reasons = collections.Counter((r.get("void_reason") or "")[:45] for r in void)
    print(f"  ROBOT  {len(void)} void rows" + (f": {dict(reasons)}" if void else ""))
    quota = [r for r in raw if "429" in (r.get("void_reason") or "")]
    print(f"  QUOTA  {len(quota)} trials lost to rate limits"
          + (f": {[(r['task'], r['trial']) for r in quota]}" if quota else ""))

    # CAP
    cap = [r for r in rows if str(r.get("tool_calls", "")).isdigit()
           and int(r["tool_calls"]) >= 60]
    print(f"  CAP    {len(cap)} scored trials ran to 60 steps"
          + (f": {[(r['task'], r['trial'], r['outcome']) for r in cap]}" if cap else ""))

    # CLAIM
    said = [r for r in rows if str(r.get("planner_said")) == "1"]
    over = [r for r in said if r["success"] != "1"]
    under = [r for r in rows if str(r.get("planner_said")) == "0" and r["success"] == "1"]
    agree = sum(1 for r in rows if str(r.get("agreement")) == "1")
    print(f"  CLAIM  agreement {100.0 * agree / max(len(rows), 1):.0f}%  |  "
          f"over-claims {len(over)}/{len(said)} of 'success' reports  |  "
          f"under-claims {len(under)}")
    for r in over[:4]:
        print(f"           over: {r['task']} t{r['trial']}  {(r.get('detail') or '')[:70]}")

    # GUARD + NUDGE from the decision log
    held, guard, nudges, last_iter = False, [], [], None
    trial_idx, nudged_trials, capped_trials = 0, set(), set()
    for f in decision_log_since(t0):
        for line in open(f, encoding="utf-8"):
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            it = ev.get("iteration")
            if isinstance(it, int) and isinstance(last_iter, int) and it < last_iter:
                trial_idx += 1
                held = False
            if isinstance(it, int):
                last_iter = it
            if ev.get("event") == "repeat_nudge":
                nudges.append(ev)
                nudged_trials.add(trial_idx)
            if ev.get("event") == "iteration_limit":
                capped_trials.add(trial_idx)
            if ev.get("event") != "tool_call":
                continue
            tool, okc, head = ev.get("tool"), not ev.get("is_error"), str(ev.get("result_head", ""))
            if tool == "pick" and okc and "Grasp confirmed" in head:
                held = True
            elif tool in ("place_in_bin", "place_at"):
                if okc and not held:
                    guard.append((trial_idx, it, head[:60]))
                if okc or "slipped" in head or "released" in head:
                    held = False
    naive = all(r.get("condition") == "naive" for r in raw) and bool(raw)
    if naive:
        # Unverified skills by design: places without a pick are the
        # measurement, not a regression. Report the count as a result.
        print(f"  GUARD  {len(guard)} places succeeded with nothing picked"
              f"   (naive baseline: unverified by design - this is a result)")
    else:
        print(f"  GUARD  {len(guard)} places succeeded with nothing picked"
              + (f"  <-- REGRESSION: {guard[:3]}" if guard else "   (guard holding)"))
    resolved = nudged_trials - capped_trials
    print(f"  NUDGE  {len(nudges)} nudges in {len(nudged_trials)} trials; "
          f"of those trials {len(resolved)} then finished before the cap, "
          f"{len(nudged_trials & capped_trials)} capped anyway")
    kinds = collections.Counter(("error" if n.get("is_error") else "noop") for n in nudges)
    if nudges:
        print(f"           by kind: {dict(kinds)}; tools: "
              f"{dict(collections.Counter(n.get('tool') for n in nudges))}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else
                  os.path.expanduser("~/moveit2_obb/runs/full_v6.csv")))

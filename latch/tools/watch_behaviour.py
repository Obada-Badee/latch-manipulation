#!/usr/bin/env python3
"""Tail the planner's decision log and flag the ways a run has lied before.

    python3 tools/watch_behaviour.py            # newest runs/*.jsonl, follow

Prints one line per suspicious event, nothing otherwise, so it can sit under a
monitor. The events are exactly the ones that cost whole runs (FINDINGS 4b, 4c):

  PLACE-WITHOUT-PICK   a place returned SUCCESS with no confirmed pick since the
                       last release. This is the 4c bug; if it ever prints, the
                       held-object guard in skill_server has regressed.
  REPEAT x N           the same tool with the same arguments N times running.
                       A model doing this is looping on feedback it cannot act
                       on - the symptom that led to 4c.
  AT-CAP               a trial reached MAX_ITERATIONS. Rare now; each one is
                       worth reading.
  MISMATCH             the planner said `done` with success=true but the
                       benchmark scored the trial as a failure (over-claim).
"""
import glob
import json
import os
import sys
import time

REPEAT_N = 4


def newest_log():
    files = glob.glob(os.path.expanduser("~/moveit2_obb/runs/*.jsonl"))
    return max(files, key=os.path.getmtime) if files else None


def follow(path):
    with open(path, encoding="utf-8") as fh:
        fh.seek(0, os.SEEK_END)
        while True:
            line = fh.readline()
            if line:
                yield line
            else:
                new = newest_log()
                if new and new != path:          # a new run started
                    return
                time.sleep(0.5)


def main():
    held = False
    last_key, streak = None, 0
    last_iter, reported = None, set()
    while True:
        path = newest_log()
        if not path:
            time.sleep(2)
            continue
        print(f"watching {os.path.basename(path)}", flush=True)
        for line in follow(path):
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("event") != "tool_call":
                if ev.get("event") in ("begin", "task", "trial"):
                    held, last_key, streak = False, None, 0
                continue
            tool, ok = ev.get("tool"), not ev.get("is_error")
            head = str(ev.get("result_head", ""))
            it = ev.get("iteration")

            # A new trial starts when the iteration counter goes backwards.
            # This must run BEFORE the held-tracking below: placed after it,
            # the pick at iteration 0 set `held` and this then cleared it, and
            # the place at iteration 1 was reported as a guard regression that
            # had not happened.
            if isinstance(it, int) and isinstance(last_iter, int) and it < last_iter:
                reported.clear()
                held = False
            last_iter = it

            if tool == "pick" and ok and "Grasp confirmed" in head:
                held = True
            elif tool in ("place_in_bin", "place_at"):
                if ok and not held:
                    print(f"PLACE-WITHOUT-PICK  iter {it}: {tool} -> {head[:70]}",
                          flush=True)
                if ok or "slipped" in head or "released" in head:
                    held = False

            key = (tool, json.dumps(ev.get("input"), sort_keys=True))
            streak = streak + 1 if key == last_key else 1
            last_key = key
            # Once per trial per call shape. The fourth identical repeat says
            # nothing the first did not; the signal is "stuck in this trial".
            if streak == REPEAT_N and key not in reported:
                reported.add(key)
                print(f"REPEAT x{REPEAT_N}  iter {it}: {tool} {key[1][:50]}  "
                      f"({'error' if not ok else 'succeeding'})", flush=True)

            if isinstance(it, int) and it >= 60:
                print(f"AT-CAP  iter {it}: {tool}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)

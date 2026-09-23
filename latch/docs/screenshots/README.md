# Screenshots

Captured live during a benchmark run with
[`tools/snapshot.py`](../../tools/snapshot.py):

```bash
python3 tools/snapshot.py --watch 20 --every 12 docs/screenshots/run
```

Each frame is the robot's actual camera feed with the detector's oriented boxes
and the world model's grounded ids drawn on top — so one image answers "what is
on the table", "what did the detector find", and "what does the planner call
each thing" at once. The caption strip records elapsed time and the detection
and grounding counts.

## What to look at

**The arm is in frame, and nothing is detected on it.** In `run-14.png` the
gripper occupies the lower right and casts a large shadow across the table, and
there is not one box on either. That is the hard-negative result as a picture:
before retraining, the arm and its shadow drew **3.14 false boxes per frame**,
most of them classed `bolt` — because a bolt is a dark elongated metallic thing
and so is a shadowed forearm. Afterwards, 0.03 per frame, and none inside the
workspace. See [FINDINGS §2.1](../../../paper/FINDINGS.md).

**Ids are grounded, not positional.** `25 red cube`, `6 green cube`, `20 grey
bolt` are the identifiers the planner reasons about and the ones a command like
*"put the red cube in the red bin"* resolves to. They persist while an object
stays visible, and memory keeps its own stable ids on top for objects that leave
view entirely.

**Grounding count vs detection count.** The caption reports both. `run-05.png`
shows `8 detections, 8 grounded objects` against a table of nine — one object
was occluded by the arm at that instant. `run-14.png` shows `9 detections, 9
grounded`. That gap opening and closing across frames is object permanence
doing its job: the planner does not lose an object because the arm moved in
front of it.

## The sequence

| frame | what it shows |
|---|---|
| `run-00` … `run-04` | opening scene, arm at `ready`, full table visible |
| `run-05` | arm entering frame; one object occluded, 8/9 grounded |
| `run-12` … `run-13` | arm over the table mid-task |
| `run-14` | gripper and shadow across the table, **9/9 still grounded, zero false boxes** |
| `run-15` … `run-19` | later trials, scene re-placed between each |

The scene differs between trials by design — the benchmark re-places every
object between trials so that no result depends on one fixed layout.

# LATCH — Language-Anchored Task Execution with Checked Handling

A Franka Panda driven by natural language, in which **every layer is checked
against something independent**: affordances are solved by the robot's own IK
and collision checker rather than scored by a learned model, grasps are
verified from joint states before they are claimed, and task success is read
from the simulator — never from the planner's own report.

The headline result is what that last distinction costs:

| condition | real success | self-reported | agreement |
|---|---|---|---|
| **LATCH** | **61.3%** (n=75) | 65% | 69% |
| no affordances | 53.3% | 79% | 59% |
| no memory | 61.3% | 69% | 68% |
| no affordances + no memory | 58.7% | 77% | 63% |
| **naive: the same, with unverified skills** | **61.3%** | **84%** | 61% |

The naive agent — skills that report success when their motion completes, which
is how most language-model manipulation systems are built — achieves **exactly
the same real success** as the full system and **claims nineteen points more**.
Scored from its transcript it wins; scored from the world it is the same robot.
Skill verification turns out to be a *truthfulness* mechanism, not a success
mechanism: what it buys is that the agent's account of what happened is true,
which is the precondition for recovery, for memory, and for any evaluation that
reads the agent rather than the world.

The analysis behind these numbers — including how three earlier rounds of
results turned out to be measuring a robot that lied — is written up in a
findings ledger and two manuscripts. Those are unpublished and are kept out
of this repository; the **data they rest on is here**, so every figure and
every number below can be recomputed from source.

---

## Repository layout

| | |
|---|---|
| [`latch/`](latch/) | the ROS 2 workspace — build and run from here. Self-contained. |
| [`latch/src/`](latch/src/) | five packages: the planner, the skill server, perception, the robot description, the MoveIt config. |
| [`latch/tools/`](latch/tools/) | the benchmark, the dataset pipeline, and the verification scripts. Everything that measures, as opposed to runs. |
| [`paper/data/`](paper/data/) | the 415 scored trials the results rest on — five 75-trial conditions, 20 induced-failure trials, two 10-trial episode runs. |
| [`paper/figures/`](paper/figures/) | the script that draws the figures from that data. |

### Where to start reading

- **What was built** — this file, then the module docstrings. Every
  non-obvious constant in `llm_task_planner` carries the measurement that
  set it; `affordance.py` and `skill_server.py` are the two to read first.
- **What was measured** — [`paper/data/`](paper/data/), through
  `tools/validate_run.py` and `tools/failure_taxonomy.py`. The CSVs are one
  row per trial, with the planner's own claim recorded next to the
  simulator's verdict.

---

## Prerequisites

Developed on Windows 11 + WSL2 (Ubuntu 24.04), with the workspace authored on
the Windows side and built inside WSL. A native Linux machine needs only the
first column.

| | version used |
|---|---|
| Ubuntu | 24.04 LTS |
| ROS 2 | Jazzy |
| Gazebo Sim | 8.14 (Harmonic), DART physics |
| MoveIt 2 | the Jazzy binaries |
| Python | 3.12 |
| PyTorch / Ultralytics | 2.13 (cu130) / 8.4 — for the detector only |
| GPU | RTX 3070 Laptop (detector inference and training; not required to run the planner) |

An LLM provider key is required for the planner. The system was measured on
**DeepInfra** serving `openai/gpt-oss-120b`; a Groq backend for the same model
is included. The host is recorded in every result row, deliberately — the same
weights served at a different quantisation are not guaranteed to make the same
tool calls.

---

## Setup

```bash
# 1. clone into the WSL home, not /mnt — building on a Windows mount is slow
git clone <this repo> ~/moveit2_obb_src && cd ~/moveit2_obb_src/latch

# 2. dependencies
rosdep install --from-paths src --ignore-src -r -y
pip install ultralytics openai        # detector + provider SDK

# 3. build
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
source install/setup.bash

# 4. the provider key, OUTSIDE the repo
cat > ~/.llm_keys <<'EOF'
export DEEPINFRA_API_KEY=...
EOF
chmod 600 ~/.llm_keys
```

`.gitignore` refuses `.llm_keys`, `.env` and `*.key` at both levels. Keep it
that way: a predecessor project in this line leaked a provider key through git
history and had to revoke it.

**Working on Windows?** Author in the Windows tree and sync before every build:

```bash
bash latch/tools/wsl/sync-workspace.sh
```

The environment (Gazebo's WSLg rendering variables, ROS setup) is centralised
in `latch/tools/wsl/env.sh` — source that rather than setting variables by hand.

---

## Running

Two terminals. The first brings up the cell — Gazebo, MoveIt, the controllers,
perception, and the skill server:

```bash
source ~/moveit2_obb/tools/wsl/env.sh
ros2 launch panda_moveit_config moveit_gazebo_obb.py perception:=true headless:=true
```

`headless:=false` puts a Gazebo window on screen; it costs about 20% of
real-time factor and nothing else.

**Always check the cell before spending tokens against it:**

```bash
python3 tools/cell_health.py     # → "HEALTHY - safe to run the benchmark"
```

This is not ceremony. It checks that `joint_state_broadcaster` is active (a
launch race leaves it inactive often enough to matter — without it, grasp
verification has no signal and *every* pick fails in a way that reads like a
capability result), that perception sees all nine objects, and that the three
skill servers answer. One 5.7-hour run was lost to skipping it.

The second terminal runs the language layer:

```bash
source ~/.llm_keys
ros2 run llm_task_planner llm_planner --ros-args -p backend:=deepinfra
```

Then type commands: *"put the red cube in the red bin"*, *"tidy up the table"*,
*"where is the red cube now?"*

---

## Reproducing the results

Every number in the paper comes from `tools/benchmark.py`, which runs the
15-command suite (5 trials each), scores each trial against simulator ground
truth, and writes one CSV row per trial.

```bash
# the full system — the headline condition, ~4 h
python3 tools/benchmark.py --backend deepinfra --trials 5 \
        --condition full --out runs/full_v6.csv

# the ablations
python3 tools/benchmark.py --no-affordance --condition no_affordance ... 
python3 tools/benchmark.py --no-memory     --condition no_memory ...

# every run is resumable; an interruption costs the trial in flight and nothing else
python3 tools/benchmark.py ... --resume
```

The chain scripts in `tools/wsl/` run several conditions back to back
unattended, restarting the cell between each and refusing to continue if a
finished run fails its audit.

### Auditing a run

```bash
python3 tools/validate_run.py runs/full_v6.csv
```

One line per way a run has been wrong before:

```
GUARD   places that succeeded with nothing picked     ← must be 0
CAP     trials that ran to the step limit
NUDGE   repetition nudges sent; did those trials resolve, or cap anyway?
CLAIM   agreement, over-claims, under-claims
ROBOT   void rows / skill server died mid-run
QUOTA   trials lost to provider rate limits
DUP     duplicate cells
```

`tools/failure_taxonomy.py` classifies failures into grounding / execution /
collateral / timeout by reading the decision log, because the scorer's text
cannot distinguish "binned the wrong cube" from "picked the right cube and
dropped it".

### The naive baseline

Reproducing the headline comparison needs a cell launched with verification
off. The pairing is enforced both ways — the benchmark refuses to run any
other condition against an unverified server, and refuses `--no-verify`
against a verified one:

```bash
ros2 launch panda_moveit_config moveit_gazebo_obb.py ... verify:=false
python3 tools/benchmark.py --no-verify --no-affordance --no-memory \
        --condition naive --out runs/naive_v6.csv
```

### The data

[`paper/data/`](paper/data/) holds all 415 scored trials behind the results —
the five 75-trial conditions, 20 induced-failure trials, and the two
10-trial episode runs. Figures regenerate from them directly:

```bash
LATCH_RUNS=paper/data python3 paper/figures/make_figures.py --out paper/figures
```

---

## Things that will bite you

Each of these cost hours:

- **Gazebo leaks ~1.7 GB/h.** At 8.6 hours it held 14.4 GB of 15.8 and the
  OOM killer was minutes away. Do not run more than ~5 hours of trials on one
  `gz sim` process; restart the cell between conditions.
- **`skill_server` can segfault mid-run.** It is respawned by the launch file
  now, and the benchmark probes the action servers before writing any
  non-success row — without that, one crash silently voided 26 trials and
  scored one of them as the robot failing a task it never attempted.
- **A step cap censors your data.** At 25 steps, the tool-call distribution of
  *successful* trials maxed out at exactly 25 — the signature of truncation,
  not of a workload. Size the cap from the measured distribution; ours is 60.
- **Verification is not optional if you intend to report success rates.** Three
  separate rounds of results here were scored against skills that reported
  success they had not achieved — and each time, the number looked perfectly
  reasonable. `tools/test_empty_place.py` is the regression test that keeps
  the worst of them from coming back.

---

## Status

The system is complete and measured; the papers are in draft. Simulation only,
by constraint — no physical arm was available, and no hardware claim is made.
Transfer to a real Panda is the stated next step rather than a conceded gap.

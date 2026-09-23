#!/usr/bin/env python3
"""Run the task suite N times per condition and write one row per trial.

    # a smoke run: one task, one trial
    python3 benchmark.py --backend groq --tasks s2 --trials 1

    # the real thing
    python3 benchmark.py --backend groq --trials 20 --out runs/main.csv

    # an ablation arm
    python3 benchmark.py --backend groq --trials 20 --no-affordance \\
                         --out runs/no_affordance.csv

Requires the cell to be up:
    ros2 launch panda_moveit_config moveit_gazebo_obb.py perception:=true

Three properties this runner is built around.

**Success comes from Gazebo, not from the planner.** Every trial ends by reading
ground-truth object poses and evaluating the task's written criterion. The
planner's own `done(success=...)` is recorded in a separate column and never
consulted for scoring. Their disagreement is a reportable result: a planner that
claims success while the table says otherwise is a distinct failure mode from one
that fails and knows it.

**Trials are independent.** The scene is re-placed and memory is wiped between
trials. Without the wipe a belief from trial 7 ("the red cube is in red_bin")
survives into trial 8, whose red cube is back on the table - the planner would
be right about a world that no longer exists.

**A trial that could not be attempted is void, not failed.** If the scene cannot
satisfy a task's requirements, the row is marked `void` and excluded from the
rate. Counting setup failures as task failures understates the system and, worse,
does so unevenly across tiers.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import subprocess
import sys
import time
import traceback
from typing import Dict, List, Optional, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "synth_dataset"))
sys.path.insert(0, os.path.join(_ROOT, "src", "yolov8_obb"))
sys.path.insert(0, os.path.join(_ROOT, "src", "llm_task_planner"))

import bench_tasks                                              # noqa: E402
from bench_tasks import EPISODES, SUITE, Task, in_play, select  # noqa: E402

#: Seconds a single command may take before the trial is abandoned. Generous:
#: a complex multi-object task legitimately runs minutes on a cell at ~half
#: real time. A trial that hits this is recorded as a timeout, not a failure,
#: because the two have different cures.
#:
#: Raised from 420 s once, on evidence rather than on a run looking bad. With
#: the scene reset fixed (FINDINGS 1d), trials finally run against a full table,
#: and a complex multi-object task on nine objects legitimately takes longer
#: than one on the five a depleted scene had left. Measured p90 wall time under
#: complete scenes is 600 s against the old 420 s cap, and 14 of 75 trials hit
#: it - two of which the table says had already completed the task, i.e. the cap
#: was scoring successes as failures.
#:
#: Stated plainly because raising a timeout after seeing it hurt a result is
#: exactly the move that looks like tuning until the number improves: the
#: justification here is the p90 measurement, the change is made once, and every
#: condition is re-run under the new value rather than mixing caps across runs.
COMMAND_TIMEOUT_S = 900.0

#: Objects the authored world contains. Fewer than this at the start of a run
#: means something has been lost off the table, and every trial after it is
#: answering an easier question than the one the suite defines.
EXPECTED_OBJECTS = 9

#: Settle time after the scene is re-placed, before the trial starts. Objects
#: are teleported, so they drop the last millimetre and rock.
SETTLE_S = 2.5

#: Stop the run after this many consecutive errored trials.
#:
#: The failure this guards against is a provider daily-token cap, which arrives
#: mid-run and makes every subsequent trial fail identically and instantly. A
#: 450-trial queue would burn to completion in minutes, writing 400 rows that
#: look like task failures and are not. Stopping keeps the partial results
#: honest and makes the cause obvious.
MAX_CONSECUTIVE_ERRORS = 3

#: How long to wait for a respawned skill_server before giving up on the run.
#: MoveIt's planning-scene monitor takes 20-40 s to come up on this machine.
SKILLS_RESPAWN_WAIT_S = 90.0

#: Substrings that identify a provider-side stop rather than a task failure.
QUOTA_SIGNATURES = ("rate limit", "rate_limit", "quota", "429",
                    "tokens per day", "TPD", "insufficient")

#: ...and the subset that means "the DAY is over", as opposed to "you are going
#: too fast this minute".
#:
#: The distinction is not pedantic. Groq enforces 8,000 tokens/minute as well as
#: 200,000/day, and a trial costs ~11,000 tokens in ~90 s - so brushing the
#: per-minute ceiling is normal operation, not an error. Treating it as a hard
#: stop ended a run after three trials with 160,000 tokens of budget unspent.
DAILY_SIGNATURES = ("tokens per day", "per day", "TPD")

#: Wait after a per-minute limit before retrying the same trial. The window is
#: 60 s; the extra is margin for the tokens already in flight.
TPM_BACKOFF_S = 75.0

#: How many times one trial may be retried after a per-minute limit.
TPM_RETRIES = 3

#: Minimum seconds between trials, to keep the rolling per-minute average under
#: the ceiling. A trial costs ~11k tokens, so it needs ~85 s of wall time to
#: average below 8,000/min; most trials take longer than that on their own and
#: this adds nothing.
PACE_S = 20.0


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------

class _StopRun(Exception):
    """Internal: abandon the queue but keep and summarise what was collected."""


class SkillsGone(RuntimeError):
    """The skill action servers are no longer reachable mid-run.

    skill_server segfaulted (exit -11) at trial 44 of a 75-trial run, after a
    preempted trajectory and a MoveIt "no state update" warning. The preflight
    had checked it at launch and it was fine. The remaining 26 trials ran
    against nothing: 24 came back void through a downstream symptom, and one
    was SCORED - the model reported "the required skill services are not
    available" and the row says the robot failed the task. Same class of
    failure as SimulatorGone; same reason it exists.
    """


class SimulatorGone(RuntimeError):
    """Gazebo stopped responding, so no further trial can mean anything.

    Raised rather than letting subprocess.TimeoutExpired escape. A 75-trial run
    outlived the simulator once: gz-sim aborted mid-run with an ODE assertion
    (`aabbBound >= dMinIntExact` in dxHashSpace::collide), the next pose query
    timed out, and the raw TimeoutExpired killed the whole run from inside
    run_trial. The completed trials survived only because the summary is printed
    from an outer handler.

    Named so the run can stop deliberately and say what to do, instead of
    ending in a stack trace about subprocess.communicate.
    """


def skills_up(node, timeout: float = 4.0) -> List[str]:
    """Names of the skill action servers that are NOT answering.

    The preflight's check, factored out so it can also run between trials.
    Short timeout: this is asked after a trial has already gone wrong, and a
    server that was up a minute ago answers in milliseconds.
    """
    from rclpy.action import ActionClient                         # noqa: PLC0415
    from skill_msgs.action import Home, Pick, Place               # noqa: PLC0415
    down = []
    for action, name in ((Pick, "pick"), (Place, "place"), (Home, "home")):
        client = ActionClient(node, action, name)
        if not client.wait_for_server(timeout_sec=timeout):
            down.append(name)
        client.destroy()
    return down


def server_verifies() -> Optional[bool]:
    """Whether skill_server was launched with verify:=true. None if unknown.

    Asked before every run, because a server launched with verify:=false
    reports success on motion alone (the naive baseline), and a real
    condition measured against it would be exactly the lying-robot result
    this project spent three suites getting rid of. The mode lives in the
    launch, so the benchmark asks the server rather than trusting a flag.
    """
    try:
        out = subprocess.run(["ros2", "param", "get", "/skill_server", "verify"],
                             capture_output=True, text=True, timeout=15).stdout
    except (subprocess.SubprocessError, OSError):
        return None
    if "True" in out:
        return True
    if "False" in out:
        return False
    return None


def preflight(timeout: float = 20.0) -> List[str]:
    """Refuse to spend tokens against a cell that is not fully up.

    Every check here corresponds to something that has already gone wrong and
    produced plausible-looking data:

    * `joint_state_broadcaster` was found **inactive** after the cell had been
      running for days - a pre-existing spawner race. With it down nothing
      publishes /joint_states, so grasp verification has no signal and MoveIt's
      current-state monitor has no start state. Every pick fails, and the model
      dutifully reports that it cannot reach anything. The run looks like a
      capability result and is an infrastructure fault.
    * perception publishing zero objects reads as an empty table, and the model
      confidently answers that the requested object is not present.
    * `skill_server` died at launch with "Unable to configure planning scene
      monitor" - a downstream effect of the same spawner race - and every check
      here passed anyway, because none of them asked whether the robot could
      still be commanded. A 20-trial run then spent 185,000 tokens and five
      hours producing rows of zeros, each one reading "the pick skill is not
      available". The most important component was the one nothing verified.

    Returns a list of problems; empty means good to go.
    """
    problems = []
    import rclpy                                                 # noqa: PLC0415
    from rclpy.node import Node                                  # noqa: PLC0415
    from controller_manager_msgs.srv import ListControllers      # noqa: PLC0415
    from sensor_msgs.msg import JointState                       # noqa: PLC0415
    from yolov8_msgs.msg import WorldState                       # noqa: PLC0415

    node = Node("bench_preflight")
    try:
        client = node.create_client(ListControllers,
                                    "/controller_manager/list_controllers")
        if not client.wait_for_service(timeout_sec=timeout):
            problems.append("controller_manager is not up")
        else:
            future = client.call_async(ListControllers.Request())
            rclpy.spin_until_future_complete(node, future, timeout_sec=timeout)
            res = future.result()
            states = {c.name: c.state for c in (res.controller if res else [])}
            for name in ("joint_state_broadcaster", "panda_arm_controller",
                         "panda_hand_controller"):
                if states.get(name) != "active":
                    problems.append(
                        f"{name} is {states.get(name, 'missing')}, not active"
                        + ("  (fix: ros2 control set_controller_state "
                           f"{name} active)" if name in states else ""))

        seen = {}
        node.create_subscription(JointState, "/joint_states",
                                 lambda m: seen.__setitem__("joints", m), 10)
        node.create_subscription(WorldState, "/world_state",
                                 lambda m: seen.__setitem__("world", m), 10)
        deadline = time.time() + timeout
        while time.time() < deadline and len(seen) < 2:
            rclpy.spin_once(node, timeout_sec=0.1)

        if "joints" not in seen:
            problems.append("/joint_states is silent - grasp verification "
                            "cannot work")
        # Can the robot actually be commanded? Everything above can pass while
        # skill_server is dead.
        from rclpy.action import ActionClient                     # noqa: PLC0415
        from skill_msgs.action import Home, Pick, Place           # noqa: PLC0415
        for action, name in ((Pick, "pick"), (Place, "place"), (Home, "home")):
            client = ActionClient(node, action, name)
            if not client.wait_for_server(timeout_sec=8.0):
                problems.append(
                    f"the '{name}' action server is not up - skill_server is "
                    f"not running. Check the cell log for 'Unable to configure "
                    f"planning scene monitor'.")
            client.destroy()

        if "world" not in seen:
            problems.append("/world_state is silent - is perception running?")
        elif not seen["world"].objects:
            problems.append("/world_state reports an empty table")
        elif len(seen["world"].objects) < EXPECTED_OBJECTS:
            # A depleted scene does not fail - it poses an easier question, and
            # scores well doing it. Two 75-trial runs were measured against a
            # table that had quietly lost a third of its objects to the floor,
            # because the reset skipped anything below table height. The
            # numbers looked plausible throughout. See FINDINGS 1d.
            problems.append(
                f"only {len(seen['world'].objects)} objects visible, expected "
                f"{EXPECTED_OBJECTS} - the scene is depleted, and trials run "
                f"against it will be easier than they should be")

        # Is the world actually STEPPING? Gazebo has gone to `paused: true` on
        # its own mid-session, and a paused world passes every check above the
        # moment it is entered: the controllers are active and the last message
        # of each topic is still sitting in the latched queue. What stops is sim
        # time. A trial run against a paused world scores a legitimate command
        # as a capability failure and looks identical to one in the CSV.
        if "joints" in seen:
            def stamp_of(msg) -> float:
                s = msg.header.stamp
                return s.sec + s.nanosec * 1e-9

            first = stamp_of(seen["joints"])
            deadline = time.time() + 3.0
            while time.time() < deadline:
                rclpy.spin_once(node, timeout_sec=0.1)
                if stamp_of(seen["joints"]) > first + 0.05:
                    break
            else:
                problems.append(
                    "sim time is not advancing - the world is paused "
                    "(gz service -s /world/<name>/control ... pause: false, "
                    "or relaunch with headless:=true)")
    finally:
        node.destroy_node()
    return problems


def wilson(successes: int, n: int, z: float = 1.96) -> Tuple[float, float, float]:
    """Point estimate and Wilson score interval for a binomial proportion.

    Wilson rather than the textbook normal approximation because these are
    small samples near the boundary, exactly where the normal interval is worst
    - at 20/20 it gives [1.0, 1.0], claiming certainty from twenty trials. The
    Wilson interval stays inside [0,1] and does not collapse.
    """
    if n == 0:
        return 0.0, 0.0, 0.0
    p = successes / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = (z / d) * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return p, max(0.0, centre - half), min(1.0, centre + half)


# --------------------------------------------------------------------------
# World access
# --------------------------------------------------------------------------

class World:
    """Ground truth, read straight from Gazebo."""

    def __init__(self, world_name: str, seed: int):
        from gz_client import GzClient                          # noqa: PLC0415
        import capture_negatives                                # noqa: PLC0415
        import objects as pool_objects                          # noqa: PLC0415

        self._shuffle = capture_negatives.shuffle_objects
        self.gz = GzClient(world_name)
        self.gz.wait_until_ready()
        self.classes = pool_objects.build_classes()
        self.rng = random.Random(seed)

    def read(self) -> Dict[str, Tuple[float, float, float]]:
        from yolov8_obb.projection import world_to_link0        # noqa: PLC0415
        try:
            poses = self.gz.read_poses()
        except subprocess.TimeoutExpired as exc:
            # Gazebo stopped answering. Distinguished from every other read
            # failure because there is no point continuing: an unresponsive
            # simulator makes the next trial's result meaningless, and a run
            # that keeps going would fill the CSV with zeros that look like
            # capability failures.
            raise SimulatorGone(
                f'Gazebo did not answer a pose query within '
                f'{exc.timeout:.0f}s') from exc
        out = {}
        for name, (wx, wy, wz) in poses.items():
            out[name] = world_to_link0(wx, wy, wz)
        return in_play(out)

    def reset(self) -> int:
        """Re-place every manipulable object, wherever it currently is.

        `restore_all=True` is the whole point: an object in a bin or on the
        floor has to come back, or trials stop being independent. The previous
        version claimed in this docstring to include objects in bins and did
        not - it delegated to a shuffle that skipped anything outside a height
        window around the table. A run lost all three bolts partway through and
        voided seven trials asking for a bolt that no longer existed.
        """
        # Guarded like read(), and for the same reason. The first version of
        # SimulatorGone wrapped only read_poses(); this path calls
        # read_poses_full(), so when Gazebo went away during a run the raw
        # TimeoutExpired escaped from reset() and killed the queue anyway -
        # the identical failure, one call site over. A guard that covers one of
        # two paths to the same resource is not a guard.
        try:
            poses = self.gz.read_poses_full()
        except subprocess.TimeoutExpired as exc:
            raise SimulatorGone(
                f'Gazebo did not answer a pose query within '
                f'{exc.timeout:.0f}s while resetting the scene') from exc
        return self._shuffle(self.gz, poses,
                             self.classes, self.rng, restore_all=True)


def satisfies(objects, requires) -> Tuple[bool, str]:
    for class_name, colour, count in requires:
        have = len(select(objects, class_name, colour))
        if have < count:
            what = f"{colour or 'any'} {class_name or 'object'}"
            return False, f"needs {count} {what}, scene has {have}"
    return True, ""


# --------------------------------------------------------------------------
# Trial
# --------------------------------------------------------------------------

FIELDS = [
    "condition", "task", "tier", "trial", "seed", "command",
    "success",              # <- from Gazebo. The number that gets reported.
    "detail",               # why the checker decided that
    "planner_said",         # the model's own done(success=...)
    "agreement",            # do those two match?
    "outcome",              # ok | wrong | timeout | error | void
    "wall_s", "iterations", "tool_calls", "in_tokens", "out_tokens",
    "objects_in_play",      # how complete the scene was when the trial started
    "induced",              # did the failure injector actually fire this trial?
    "induced_detail",       # what it did, or why it did not fire
    "void_reason",
]


def run_trial(planner, world: World, task: Task, trial: int, seed: int,
              condition: str, injector=None) -> Dict[str, object]:
    row = {f: "" for f in FIELDS}
    row.update(condition=condition, task=task.id, tier=task.tier,
               trial=trial, seed=seed, command=task.command)

    # Order matters: release anything held and home the arm BEFORE re-placing
    # objects, so a cube dropped out of the gripper is swept up by the reset
    # rather than left wherever it fell.
    planner.reset_arm()
    world.reset()
    time.sleep(SETTLE_S)
    initial = world.read()

    row["objects_in_play"] = len(bench_tasks.manipulable(initial))
    ok, why = satisfies(initial, task.requires)
    if not ok:
        row.update(outcome="void", void_reason=why, success="")
        return row

    planner.reset_memory()
    seen = planner.wait_for_beliefs(len(select(initial)))
    if seen == 0:
        row.update(outcome="void",
                   void_reason="perception never repopulated memory after reset")
        return row

    # Arm the failure injector, if this condition uses one. It watches for a
    # fingertip descending onto an object and teleports that object away, once
    # per trial - so the fingers close on nothing and the grasp check catches
    # it. What is being measured is what the planner does NEXT.
    if injector is not None:
        injector.start()

    started = time.time()
    try:
        planner_said = planner.run_command(task.command)
        outcome = "ok"
    except Exception as exc:                                    # noqa: BLE001
        planner_said = None
        outcome = "error"
        row["void_reason"] = f"{type(exc).__name__}: {exc}"[:200]
    wall = time.time() - started

    if injector is not None:
        injector.stop()
        row["induced"] = "1" if injector.fired else "0"
        row["induced_detail"] = (injector.detail if injector.fired
                                 else injector.why_not())[:160]
    if wall > COMMAND_TIMEOUT_S and outcome == "ok":
        outcome = "timeout"

    final = world.read()
    try:
        passed, detail = _check(task, final, initial)
    except Exception as exc:                                    # noqa: BLE001
        passed, detail = False, f"checker raised: {type(exc).__name__}: {exc}"

    if outcome == "ok" and not passed:
        outcome = "wrong"
    elif outcome == "error" and passed:
        # The world reached the goal state and the provider then failed - in
        # the pilot, Groq rejected its own model's `done` call as unparseable
        # after the cube was already in the bin. Success is judged from the
        # table, so this is a success with a provider fault attached, and
        # recording it as a plain error would understate the system for a
        # reason that has nothing to do with the robot.
        outcome = "ok_late_error"

    row.update(success=int(bool(passed)), detail=detail,
               planner_said="" if planner_said is None else int(bool(planner_said)),
               agreement="" if planner_said is None
                         else int(bool(planner_said) == bool(passed)),
               outcome=outcome, wall_s=f"{wall:.1f}")
    row.update(planner.last_stats())
    return row


def _check(task: Task, final, initial):
    """Call the checker, passing `initial` only to those that accept it."""
    try:
        return task.check(final, initial)
    except TypeError:
        return task.check(final)


def run_episode(planner, world: World, ep, trial: int, seed: int,
                condition: str) -> Dict[str, object]:
    ep_id, tier, commands, check, requires = ep
    row = {f: "" for f in FIELDS}
    row.update(condition=condition, task=ep_id, tier=tier, trial=trial,
               seed=seed, command=" | ".join(commands))

    # Order matters: release anything held and home the arm BEFORE re-placing
    # objects, so a cube dropped out of the gripper is swept up by the reset
    # rather than left wherever it fell.
    planner.reset_arm()
    world.reset()
    time.sleep(SETTLE_S)
    initial = world.read()
    row["objects_in_play"] = len(bench_tasks.manipulable(initial))
    ok, why = satisfies(initial, requires)
    if not ok:
        row.update(outcome="void", void_reason=why)
        return row

    # Memory is wiped once, before the FIRST command. Wiping between the
    # commands of an episode would destroy the thing being measured.
    planner.reset_memory()
    if planner.wait_for_beliefs(len(select(initial))) == 0:
        row.update(outcome="void",
                   void_reason="perception never repopulated memory after reset")
        return row

    started = time.time()
    said = None
    outcome = "ok"
    for command in commands:
        try:
            said = planner.run_command(command)
        except Exception as exc:                                # noqa: BLE001
            outcome = "error"
            row["void_reason"] = f"{type(exc).__name__}: {exc}"[:200]
            break
    wall = time.time() - started

    final = world.read()
    try:
        passed, detail = _check_ep(check, final, initial)
    except Exception as exc:                                    # noqa: BLE001
        passed, detail = False, f"checker raised: {type(exc).__name__}: {exc}"
    if outcome == "ok" and not passed:
        outcome = "wrong"
    elif outcome == "error" and passed:
        # The world reached the goal state and the provider then failed - in
        # the pilot, Groq rejected its own model's `done` call as unparseable
        # after the cube was already in the bin. Success is judged from the
        # table, so this is a success with a provider fault attached, and
        # recording it as a plain error would understate the system for a
        # reason that has nothing to do with the robot.
        outcome = "ok_late_error"

    row.update(success=int(bool(passed)), detail=detail,
               planner_said="" if said is None else int(bool(said)),
               agreement="" if said is None else int(bool(said) == bool(passed)),
               outcome=outcome, wall_s=f"{wall:.1f}")
    row.update(planner.last_stats())
    return row


def _check_ep(check, final, initial):
    try:
        return check(final, initial)
    except TypeError:
        return check(final)


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def summarise(rows: List[Dict[str, object]]) -> None:
    scored = [r for r in rows if r["outcome"] != "void"]
    void = len(rows) - len(scored)

    print(f"\n{'=' * 70}")
    print(f"{len(scored)} scored trials ({void} void)")
    print("=" * 70)

    def block(title: str, subset: List[Dict[str, object]]):
        if not subset:
            return
        n = len(subset)
        s = sum(int(r["success"]) for r in subset)
        p, lo, hi = wilson(s, n)
        print(f"  {title:<34} {s:>3}/{n:<3} = {p * 100:5.1f}%  "
              f"[{lo * 100:.1f}, {hi * 100:.1f}] 95% CI")

    print("\nby tier")
    for tier in ("simple", "medium", "complex"):
        block(tier, [r for r in scored if r["tier"] == tier])
    block("ALL", scored)

    print("\nby task")
    for task_id in sorted({str(r["task"]) for r in scored}):
        subset = [r for r in scored if r["task"] == task_id]
        block(task_id + "  " + str(subset[0]["command"])[:26], subset)

    print("\noutcomes")
    for kind in ("ok", "ok_late_error", "wrong", "timeout", "error"):
        n = sum(1 for r in scored if r["outcome"] == kind)
        if n:
            print(f"  {kind:<10} {n}")

    # The planner's self-report against the world. An over-claim (said success,
    # table disagrees) is the interesting direction: it is the failure the
    # predecessor project could not have detected at all.
    judged = [r for r in scored if r["agreement"] != ""]
    if judged:
        agree = sum(int(r["agreement"]) for r in judged)
        over = sum(1 for r in judged
                   if int(r["planner_said"]) == 1 and int(r["success"]) == 0)
        under = sum(1 for r in judged
                    if int(r["planner_said"]) == 0 and int(r["success"]) == 1)
        print(f"\nself-report vs ground truth   {agree}/{len(judged)} agree "
              f"({100.0 * agree / len(judged):.1f}%)")
        print(f"  over-claimed (said done, table says no)   {over}")
        print(f"  under-claimed (said failed, table says ok) {under}")

    times = [float(r["wall_s"]) for r in scored if r["wall_s"]]
    if times:
        times.sort()
        print(f"\nwall time  median {times[len(times) // 2]:.0f} s   "
              f"min {times[0]:.0f}   max {times[-1]:.0f}")


# --------------------------------------------------------------------------

def scored_rows(path: str) -> List[Dict[str, str]]:
    """Rows from a results CSV that count as measurements of the robot.

    This is the ONE definition of "a trial that happened", shared by --resume
    and by every analysis script. They disagreed once: --resume correctly
    re-runs a trial whose row records a provider 429 rather than a robot
    failure, but the analysis counted BOTH that row and its replacement - the
    rate limit as a failure, and the real attempt beside it. A cell can
    therefore appear twice, and the stale copy always scores zero, biasing the
    success rate down for reasons that have nothing to do with the arm.

    Dropped: void trials, and quota errors that interrupted an unfinished task
    (a quota error AFTER the goal was reached is kept - the world reaching the
    goal state is the measurement, and what the API did next cannot unmake it).
    Deduplicated by (condition, task, trial), keeping the LAST row, which is the
    most recent re-run of that cell.
    """
    if not os.path.exists(path):
        return []
    keep = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            quota_hit = any(sig.lower() in str(row.get("void_reason", "")).lower()
                            for sig in QUOTA_SIGNATURES)
            interrupted = quota_hit and str(row.get("success")) != "1"
            if not row.get("outcome") or row["outcome"] == "void" or interrupted:
                continue
            keep[(row["condition"], row["task"], row["trial"])] = row
    return list(keep.values())


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--tasks", default="",
                    help="comma-separated ids, or a tier name; default all")
    ap.add_argument("--episodes", action="store_true",
                    help="run the multi-command episodes instead of SUITE")
    ap.add_argument("--backend", default="groq")
    ap.add_argument("--model", default="")
    ap.add_argument("--world", default="arm_and_table")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="")
    ap.add_argument("--condition", default="",
                    help="label for this arm; defaults to the ablation flags")
    ap.add_argument("--no-affordance", action="store_true",
                    help="ablation: hide the feasibility column from the model")
    ap.add_argument("--no-memory", action="store_true",
                    help="ablation: forget everything between commands")
    ap.add_argument("--no-verify", action="store_true",
                    help="naive-agent baseline: requires a cell launched with "
                         "verify:=false, and forces --condition naive")
    ap.add_argument("--max-tokens", type=int, default=0,
                    help="stop cleanly once this many tokens have been spent "
                         "(0 = no cap). Set below the provider's daily quota so "
                         "the run ends on its own terms rather than on a 429")
    ap.add_argument("--ignore-preflight", action="store_true",
                    help="run even if the cell looks unhealthy (not advised)")
    ap.add_argument("--resume", action="store_true",
                    help="append to --out, skipping trials it already contains")
    ap.add_argument("--induce", action="store_true",
                    help="teleport the target out from under the gripper once "
                         "per trial, to measure recovery (claim 1.12)")
    args = ap.parse_args(argv)

    wanted: List[Task]
    if args.tasks in ("simple", "medium", "complex"):
        wanted = [t for t in SUITE if t.tier == args.tasks]
    elif args.tasks:
        ids = {s.strip() for s in args.tasks.split(",")}
        wanted = [t for t in SUITE if t.id in ids]
        if not wanted:
            print(f"no task matched {args.tasks}", file=sys.stderr)
            return 2
    else:
        wanted = list(SUITE)

    condition = args.condition or "+".join(
        filter(None, ["full" if not (args.no_affordance or args.no_memory) else "",
                      "no_affordance" if args.no_affordance else "",
                      "no_memory" if args.no_memory else ""])) or "full"

    from bench_planner import PlannerHarness                     # noqa: PLC0415
    import rclpy                                                 # noqa: PLC0415

    rclpy.init()
    try:
        problems = preflight()
    finally:
        rclpy.shutdown()

    # The verify mode and the condition must agree, both ways. Neither
    # direction is allowed to be a warning.
    verifies = server_verifies()
    if args.no_verify:
        if args.condition != "naive":
            print(f"--no-verify forces --condition naive (got {args.condition!r})",
                  file=sys.stderr)
            return 2
        if verifies is not False:
            problems.append("--no-verify given, but skill_server reports "
                            f"verify={verifies}. Launch the cell with "
                            "verify:=false for the naive baseline.")
        else:
            print("*** NAIVE BASELINE: skills report success on motion alone. "
                  "Self-report is expected to exceed real success.", flush=True)
    elif verifies is False:
        problems.append("skill_server is running with verify=false, which is "
                        f"only valid for the naive baseline, not {args.condition!r}. "
                        "Relaunch the cell with verify:=true.")
    if problems:
        print("preflight failed - not spending tokens on a broken cell:",
              file=sys.stderr)
        for p in problems:
            print(f"  * {p}", file=sys.stderr)
        if not args.ignore_preflight:
            return 1
        print("  (--ignore-preflight given; continuing anyway)",
              file=sys.stderr)

    world = World(args.world, args.seed)
    planner = PlannerHarness(backend=args.backend, model=args.model,
                             no_affordance=args.no_affordance,
                             no_memory=args.no_memory)

    # Built AFTER the planner, deliberately. Both this and PlannerHarness call
    # rclpy.init(); the harness calls it unconditionally, while the injector
    # only initialises if the context is not already up. Constructing the
    # injector first therefore killed the run on its first line with
    # "Context.init() must only be called once".
    injector = None
    if args.induce:
        import bench_induce                                      # noqa: PLC0415
        # One injector for the whole run: it holds a TF listener and a gz
        # client, and rebuilding those per trial would cost more than the
        # induction does. start()/stop() bracket each trial.
        injector = bench_induce.FailureInjector(args.world, seed=args.seed)
        print("failure injection ARMED - one induced grasp miss per trial",
              flush=True)

    out_path = args.out or os.path.join(
        _ROOT, "runs", f"bench_{condition}_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Resume support exists because a free-tier daily quota cannot finish a
    # 450-trial design in one sitting. Rather than a shorter experiment, the run
    # accumulates across days: each session spends its budget, stops, and the
    # next one picks up the cells that are still missing.
    done_cells = set()
    rows: List[Dict[str, object]] = []
    if args.resume and os.path.exists(out_path):
        # scored_rows() is the single definition of what counts as done, so
        # --resume and the analysis can never drift apart on the question.
        for row in scored_rows(out_path):
            done_cells.add((row["condition"], row["task"], row["trial"]))
            rows.append(row)
        print(f"resuming: {len(done_cells)} trials already recorded in "
              f"{out_path}")

    mode = "a" if (args.resume and os.path.exists(out_path)) else "w"
    spent = 0
    consecutive_errors = 0
    try:
        with open(out_path, mode, newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=FIELDS)
            if mode == "w":
                writer.writeheader()

            units = EPISODES if args.episodes else wanted
            total = len(units) * args.trials
            done = 0
            for trial in range(args.trials):
                for unit in units:
                    seed = args.seed * 1000 + trial
                    world.rng = random.Random(seed)
                    label = unit.id if isinstance(unit, Task) else unit[0]
                    done += 1
                    if (condition, label, str(trial)) in done_cells:
                        continue
                    if args.max_tokens and spent >= args.max_tokens:
                        print(f"\ntoken budget reached ({spent:,} >= "
                              f"{args.max_tokens:,}). Re-run with --resume "
                              f"after the quota resets.", file=sys.stderr)
                        raise _StopRun
                    print(f"\n[{done}/{total}] {condition} {label} "
                          f"trial {trial + 1}"
                          + (f"  [{spent:,} tokens spent]" if spent else ""),
                          flush=True)
                    # A trial is retried, not recorded, when the provider
                    # refuses for the minute. The row is only written once it
                    # represents something the robot actually did.
                    tpm_retries = 0
                    while True:
                        try:
                            if isinstance(unit, Task):
                                row = run_trial(planner, world, unit, trial,
                                                seed, condition,
                                                injector=injector)
                            else:
                                row = run_episode(planner, world, unit, trial,
                                                  seed, condition)
                        except SimulatorGone as exc:
                            # Nothing measured after this point would mean
                            # anything, so stop and say so. What has already
                            # been written stays written, and --resume picks up
                            # from here once the cell is back.
                            print("")
                            print(f"*** {exc}", flush=True)
                            print("*** the simulator is gone - stopping. "
                                  "Restart the cell, then re-run with "
                                  "--resume to continue from here.", flush=True)
                            raise _StopRun() from exc

                        reason = str(row.get("void_reason", "")).lower()
                        is_quota = (row["outcome"] == "error"
                                    and any(s.lower() in reason
                                            for s in QUOTA_SIGNATURES))
                        is_daily = any(s.lower() in reason
                                       for s in DAILY_SIGNATURES)

                        if is_quota and is_daily:
                            print(f"\ndaily quota exhausted: "
                                  f"{str(row['void_reason'])[:160]}\n"
                                  f"stopping with {len(rows)} rows rather than "
                                  f"recording the rest as task failures. "
                                  f"Re-run with --resume after it resets.",
                                  file=sys.stderr)
                            raise _StopRun

                        if is_quota and tpm_retries < TPM_RETRIES:
                            tpm_retries += 1
                            print(f"    per-minute ceiling; waiting "
                                  f"{TPM_BACKOFF_S:.0f}s and retrying this "
                                  f"trial ({tpm_retries}/{TPM_RETRIES})",
                                  flush=True)
                            time.sleep(TPM_BACKOFF_S)
                            continue
                        break

                    # Before this row is written: if the trial did not succeed,
                    # rule out the robot having gone away. skill_server
                    # segfaulted mid-run once; 24 trials came back void through
                    # a downstream symptom and one was SCORED as the robot
                    # failing a task it never got to attempt. The probe costs
                    # milliseconds when the servers are up.
                    if str(row.get("success")) != "1":
                        down = skills_up(planner.node)
                        if down:
                            # The launch file respawns skill_server. Give it
                            # time to come back before declaring the robot
                            # gone: MoveIt takes 20-40 s to initialise.
                            print(f"    {', '.join(down)} not answering - "
                                  f"waiting for skill_server to respawn",
                                  flush=True)
                            deadline = time.time() + SKILLS_RESPAWN_WAIT_S
                            while down and time.time() < deadline:
                                time.sleep(2.0)
                                down = skills_up(planner.node, timeout=2.0)
                            row.update(outcome="void", success="",
                                       void_reason="skill server died during "
                                                   "this trial"
                                                   + ("" if not down else
                                                      f"; {', '.join(down)} "
                                                      f"never came back"))
                            rows.append(row)
                            writer.writerow(row)
                            fh.flush()
                            if down:
                                raise SkillsGone(
                                    f"the {', '.join(down)} action server(s) "
                                    f"stopped answering during trial "
                                    f"{done}/{total} and did not respawn "
                                    f"within {SKILLS_RESPAWN_WAIT_S:.0f}s. "
                                    f"Check the cell log for 'process has "
                                    f"died'.")
                            print("    skill_server is back; trial voided, "
                                  "continuing", flush=True)
                            time.sleep(PACE_S)
                            continue

                    rows.append(row)
                    writer.writerow(row)
                    fh.flush()          # a crash at trial 40 must not lose 39
                    for field in ("in_tokens", "out_tokens"):
                        try:
                            spent += int(row[field] or 0)
                        except (TypeError, ValueError):
                            pass
                    print(f"    -> {row['outcome']}  {row['detail']}", flush=True)

                    if row["outcome"] in ("error", "void"):
                        consecutive_errors += 1
                        if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                            print(f"\n{consecutive_errors} consecutive trials "
                                  f"without a measurement; stopping. Last: "
                                  f"{row['void_reason']}", file=sys.stderr)
                            raise _StopRun
                    else:
                        consecutive_errors = 0

                    # Keep the rolling per-minute average under the ceiling.
                    time.sleep(PACE_S)
    except (KeyboardInterrupt, _StopRun):
        print("\nstopped early; partial results kept", file=sys.stderr)
    except SkillsGone as exc:
        # Everything written so far is a real measurement; nothing after this
        # point would be. Stop, say why, and point at --resume.
        print(f"\n*** {exc}", file=sys.stderr)
        print("*** the robot is gone - stopping. Restart the cell, then re-run "
              "with --resume to continue from here.", file=sys.stderr)
    except Exception:                                            # noqa: BLE001
        traceback.print_exc()
    finally:
        planner.shutdown()

    summarise(rows)
    print(f"\ntokens spent this session: {spent:,}")
    print(f"rows: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

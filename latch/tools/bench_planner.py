#!/usr/bin/env python3
"""Drive `LlmPlanner` from a test process, one command at a time.

Wraps the node rather than reimplementing it. The point of the benchmark is to
measure the system that actually ships, so this must not become a parallel
implementation that drifts: everything here is setup, teardown, and reading
statistics back out. No planning logic lives in this file, and none should.

Per-trial statistics come from the planner's own decision log rather than from
counters added for the benchmark. The log is written on every run, including
interactive ones, so the numbers reported here are the same numbers a normal
session produces - and there is no instrumentation that exists only under test
and could therefore be wrong only under test.
"""

from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
from typing import Any, Dict, List, Optional


def _tokens(usage: Any, *names: str) -> Optional[int]:
    """Pull a token count out of whatever shape the backend logged.

    Anthropic reports a dict; Groq's SDK logs the repr of a `CompletionUsage`
    dataclass, so the log holds a *string* like
    `CompletionUsage(completion_tokens=253, prompt_tokens=2844, ...)`. Handling
    both here keeps the backends interchangeable, which is the property the
    whole planner is built around.

    The regex anchors on `name=` so `completion_tokens=253` matches while
    `completion_tokens_details=...` does not - the nested field would otherwise
    shadow the real one.
    """
    if isinstance(usage, dict):
        for name in names:
            if usage.get(name) is not None:
                try:
                    return int(usage[name])
                except (TypeError, ValueError):
                    pass
        return None
    if isinstance(usage, str):
        for name in names:
            m = re.search(rf"\b{name}=(\d+)", usage)
            if m:
                return int(m.group(1))
    return None

_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      ".."))
sys.path.insert(0, os.path.join(_ROOT, "src", "llm_task_planner"))


class PlannerHarness:
    """One long-lived planner node, reused across every trial in a run.

    Restarting it per trial would be cleaner in principle and wrong in
    practice: node construction costs seconds, and the prompt cache the
    backends rely on is per-process, so a fresh node per trial would change
    both the latency and the token cost being measured.
    """

    def __init__(self, backend: str = "groq", model: str = "",
                 no_affordance: bool = False, no_memory: bool = False,
                 log_dir: str = ""):
        import rclpy                                             # noqa: PLC0415
        from rclpy.executors import MultiThreadedExecutor        # noqa: PLC0415

        self._rclpy = rclpy
        args = ["--ros-args",
                "-p", f"backend:={backend}",
                "-p", f"show_affordances:={'false' if no_affordance else 'true'}",
                "-p", f"persist_memory:={'false' if no_memory else 'true'}",
                # Never inherit a previous run's beliefs: trial 1 must not start
                # believing what a session yesterday left behind.
                "-p", "forget:=true"]
        if model:
            args += ["-p", f"model:={model}"]
        if log_dir:
            args += ["-p", f"log_dir:={log_dir}"]

        rclpy.init(args=args)
        from llm_task_planner.llm_planner import LlmPlanner      # noqa: PLC0415

        self.node = LlmPlanner()
        self._executor = MultiThreadedExecutor()
        self._executor.add_node(self.node)
        self._spin = threading.Thread(target=self._executor.spin, daemon=True)
        self._spin.start()

        from control_msgs.action import GripperCommand           # noqa: PLC0415
        from rclpy.action import ActionClient                    # noqa: PLC0415

        # The harness needs its own handle on the gripper: the skill server
        # only opens it as part of a pick or place, and between trials there is
        # neither.
        self._gripper = ActionClient(self.node, GripperCommand,
                                     "/panda_hand_controller/gripper_cmd",
                                     callback_group=self.node.cb)

        self._traj_pub = None           # created lazily by _rescue()
        self._log_path = self.node._log_file
        self._mark = 0
        # Perception publishes at 5 Hz; the first world state has to arrive
        # before the first command or the opening scene reads as empty.
        deadline = time.time() + 20.0
        while time.time() < deadline:
            with self.node._world_lock:
                if self.node._world is not None:
                    break
            time.sleep(0.2)
        else:
            print("warning: no /world_state after 20 s - is perception up?",
                  file=sys.stderr)

    # ------------------------------------------------------------------ run

    def run_command(self, command: str) -> bool:
        self._mark = self._log_size()
        return bool(self.node.run_command(command))

    def reset_arm(self, timeout: float = 40.0) -> str:
        """Open the gripper and return to `ready`, before the scene is re-placed.

        Without this, trials are not independent. A `pick` task legitimately
        ends with the arm holding a cube, and the next trial then began with a
        closed gripper carrying an object that the scene reset had teleported
        out from under it - so the arm held nothing while memory believed it
        held something. The first pilot run did exactly that and the following
        trial reported success after a single turn without moving.

        The gripper is opened FIRST so anything held drops onto the table, and
        the caller re-places objects afterwards, which sweeps up whatever fell.
        """
        notes = []
        try:
            from control_msgs.action import GripperCommand      # noqa: PLC0415
            from llm_task_planner import scene                  # noqa: PLC0415

            goal = GripperCommand.Goal()
            goal.command.position = scene.OPEN_JAW_GAP / 2.0
            goal.command.max_effort = 20.0
            if self._gripper.wait_for_server(timeout_sec=5.0):
                future = self._gripper.send_goal_async(goal)
                deadline = time.time() + 10.0
                while not future.done() and time.time() < deadline:
                    time.sleep(0.05)
                notes.append("gripper opened")
            else:
                notes.append("gripper controller unavailable")
        except Exception as exc:                                 # noqa: BLE001
            notes.append(f"gripper reset failed: {type(exc).__name__}")

        homed = False
        try:
            from skill_msgs.action import Home                   # noqa: PLC0415
            homed, detail = self.node._await_action(self.node._home,
                                                    Home.Goal(), "home")
            notes.append("homed" if homed else f"home failed: {detail}")
        except Exception as exc:                                 # noqa: BLE001
            notes.append(f"home failed: {type(exc).__name__}: {exc}")

        if not homed and self._rescue():
            notes.append("rescued by direct trajectory")
        return "; ".join(notes)

    def _rescue(self) -> bool:
        """Drive the arm to `ready` through the controller, bypassing MoveIt.

        Needed because MoveIt cannot recover the one state that most needs
        recovering. If a run is interrupted mid-trajectory the arm can be left
        marginally outside a joint limit - joint6 was found at 3.75283 against
        a limit of 3.75250 - and MoveIt then refuses *every* plan, including
        home, because the START state is invalid. From the outside this looks
        like total system failure: every task fails, the model reports it
        cannot reach anything, and nothing in the logs points at the joint.

        A trajectory published straight to the controller has no such check, so
        it can walk the arm back into the legal envelope. Without this a single
        interrupted trial poisons every trial after it, and an overnight run
        would produce hundreds of zeros with a plausible-sounding explanation
        attached to each.
        """
        try:
            from trajectory_msgs.msg import (JointTrajectory,     # noqa: PLC0415
                                             JointTrajectoryPoint)
            from builtin_interfaces.msg import Duration           # noqa: PLC0415
        except ImportError:
            return False

        if self._traj_pub is None:
            self._traj_pub = self.node.create_publisher(
                JointTrajectory, "/panda_arm_controller/joint_trajectory", 10)

        # Publishing before the controller has matched silently discards the
        # message - which is how the first rescue attempt appeared to succeed
        # while the arm had not moved at all.
        deadline = time.time() + 5.0
        while self._traj_pub.get_subscription_count() == 0:
            if time.time() > deadline:
                return False
            time.sleep(0.05)

        msg = JointTrajectory()
        msg.joint_names = [f"panda_joint{i}" for i in range(1, 8)]
        point = JointTrajectoryPoint()
        point.positions = [0.0, -0.30, 0.0, -2.00, 0.0, 1.75, 0.79]   # `ready`
        point.time_from_start = Duration(sec=4)
        msg.points = [point]
        self._traj_pub.publish(msg)
        time.sleep(5.0)
        return True

        # Memory must be told, or it keeps believing it is carrying something.
        with self.node._mem_lock:
            if self.node._memory.held is not None:
                self.node._memory.note_dropped("arm reset between trials")
        return "; ".join(notes)

    def reset_memory(self) -> None:
        with self.node._mem_lock:
            self.node._memory.forget_all()

    def wait_for_beliefs(self, want: int, timeout: float = 30.0) -> int:
        """Block until memory has committed at least `want` objects.

        This is not politeness, it is correctness. Memory deliberately requires
        several corroborating sightings before it will believe an object exists
        (one sighting is a detection; several is an object), so for roughly a
        second after a wipe it legitimately knows nothing. Issuing a command in
        that window shows the model an empty table.

        The first smoke run did exactly that and the model answered "the red
        cube is already in the red bin, so no action was needed" - a confident
        claim about a table it could not see. Scored as a task failure it would
        have been a lie about the system: the planner was fine, the harness had
        not let it look.
        """
        deadline = time.time() + timeout
        best = 0
        while time.time() < deadline:
            with self.node._mem_lock:
                best = len(self.node._memory.all_beliefs())
            if best >= want:
                return best
            time.sleep(0.1)
        return best

    # ---------------------------------------------------------------- stats

    def _log_size(self) -> int:
        try:
            return os.path.getsize(self._log_path)
        except OSError:
            return 0

    def last_stats(self) -> Dict[str, object]:
        """Iterations, tool calls and token usage for the command just run.

        Reads the log from the offset recorded before the command started, so
        this reports that command alone rather than the session so far.
        """
        out: Dict[str, object] = {"iterations": "", "tool_calls": "",
                                  "in_tokens": "", "out_tokens": ""}
        records = self._records_since(self._mark)
        if not records:
            return out

        calls = [r for r in records if r.get("event") == "tool_call"]
        out["tool_calls"] = len(calls)

        done = [r for r in records if r.get("event") == "done"]
        if done:
            out["iterations"] = done[-1].get("iterations", "")

        usages = [r.get("usage") for r in records if r.get("usage")]
        if usages:
            ins = [_tokens(u, "prompt_tokens", "input_tokens") for u in usages]
            outs = [_tokens(u, "completion_tokens", "output_tokens") for u in usages]
            # Input is billed per turn and the whole conversation is re-sent
            # each turn, so the cost of a command is the SUM over turns, not
            # the last turn's figure. Reporting only the last would understate
            # a 6-turn task by roughly a factor of six - and it is exactly the
            # long tasks whose cost the scalability claim depends on.
            out["in_tokens"] = sum(i for i in ins if i is not None)
            out["out_tokens"] = sum(o for o in outs if o is not None)
        return out

    def _records_since(self, offset: int) -> List[dict]:
        try:
            with open(self._log_path, encoding="utf-8") as fh:
                fh.seek(offset)
                body = fh.read()
        except OSError:
            return []
        records = []
        for line in body.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except ValueError:
                continue                # a partially flushed final line
        return records

    # ------------------------------------------------------------- teardown

    def shutdown(self) -> None:
        try:
            self._executor.shutdown(timeout_sec=2.0)
        except Exception:                                        # noqa: BLE001
            pass
        try:
            self.node.destroy_node()
        except Exception:                                        # noqa: BLE001
            pass
        if self._rclpy.ok():
            self._rclpy.shutdown()

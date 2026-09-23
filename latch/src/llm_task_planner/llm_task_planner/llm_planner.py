#!/usr/bin/env python3
"""Natural-language task planner: the model decides, the skill server acts.

The loop here is deliberately small and owns nothing but sequencing. Scene
knowledge lives in /world_state, robot knowledge in skill_server, schemas in
tools.py, the prompt in prompts.py, and *the provider in backends/*. Swapping
this loop for LangGraph means replacing run_command() and importing the rest
unchanged - that constraint is the whole point of the split.

PROVIDERS. Two are implemented, and the loop below cannot tell them apart:

    anthropic   claude-opus-5. Strict server-side schema validation, prompt
                caching over the frozen prefix, explicit refusal handling.
    groq        llama-3.3-70b-versatile by default - the model the predecessor
                project used, so the two are directly comparable on the same
                robot. Tool calling, not the text-line parsing that project did.

    ros2 run llm_task_planner llm_planner --ros-args -p backend:=groq

Adding the second provider is what turned the swappability claim into something
checkable. It held up: this file needed no branch on provider. What did have to
move was the conversation history, into the backends - the two providers
genuinely disagree about what a message is (see backends/base.py).

Why a manual loop rather than an SDK tool_runner: the tools here block for tens
of seconds on ROS actions while the node's subscriptions must keep being
serviced, and the loop interleaves with a ROS executor it does not own.

THE CLOSED LOOP, in one sentence: every skill result carries a freshly perceived
world state, so a failed grasp comes back as an error message *plus* the scene as
it now actually is, and the next model turn plans against reality rather than
against what it assumed. No bespoke replanning machinery - that is just tool use.

WHAT THE LOOP PUTS IN FRONT OF THE MODEL, and where each part comes from:

    memory.py       WHERE things are, including things the camera cannot see -
                    an object in a bin, or occluded by the arm. Beliefs persist
                    across commands and across restarts, so "put it back" and
                    "now the other one" have referents. Ids shown to the model
                    are MEMORY ids; _world_id() translates them at the moment a
                    goal is sent.
    affordance.py   WHAT can be done to them, decided by solving the arm's IK
                    and checking the planning scene rather than by guessing.
                    This is SayCan's affordance term; see affordance.py for what
                    of that paper is worth keeping and what is not.

The loop still owns nothing but sequencing. It asks memory what is there, asks
the skill server what is possible, hands both to the backend, and records what
came back. Every one of those is a call into a module that could be replaced
without touching this file.

Requires ANTHROPIC_API_KEY or GROQ_API_KEY in the environment, depending on the
backend. Never put a key in this file: a key in source ends up in git history,
and history is not something you can take back.
"""

import json
import os
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import rclpy
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from skill_msgs.action import Home, Pick, Place
from skill_msgs.srv import CheckAffordances
from yolov8_msgs.msg import WorldState

from llm_task_planner import affordance, memory, prompts, scene
from llm_task_planner.backends import ToolResult, make_backend
from llm_task_planner.backends.base import SCENE_SEPARATOR
from llm_task_planner.memory import Observation, RobotMemory
from llm_task_planner.tools import TERMINAL_TOOLS, TOOL_SCHEMAS

#: Hard stop on the agentic loop, so a model that keeps retrying can't drive the
#: arm indefinitely while nobody is watching.
#: Hard bound on the tool-calling loop, so a confused model cannot run forever.
#:
#: Raised from 25 once, on evidence. At 25 the distribution of tool calls in
#: SUCCESSFUL trials was censored against it: median 2, p90 8, and a maximum of
#: exactly 25 - successes were pressed up against the ceiling, which is what
#: truncation looks like rather than what a natural workload looks like. 40 of
#: 300 trials reached it and only 3 of those succeeded, concentrated on the
#: multi-object tasks (`c1`, `c2`, `c4`), whose details read "4/5 in blue_bin" -
#: cut off one object short.
#:
#: The arithmetic agrees: `c1` needs 5 picks + 5 places + `done` = 11 calls if
#: nothing fails, and at the measured ~70% pick success it needs about 16-18
#: with a long tail, since each failure also costs a `home`.
#:
#: It also distorted the ablations. Showing affordances makes the model follow
#: the advice "move that one first", which costs ~3 extra calls on affected
#: trials; under a 25-step budget that advice lost more trials than it won, so
#: the ablation appeared to beat the full system (FINDINGS 1h).
#:
#: 60 leaves the tail room to resolve while still bounding a runaway. Typical
#: trials are unaffected - the median is 2.
MAX_ITERATIONS = 60

#: Consecutive identical failing calls before the model is told, in a plain
#: message, that it is repeating itself. See prompts.REPEATED_FAILURE_NUDGE.
#: Three, not two: a single retry of a failed action is legitimate (a grasp
#: that slipped may well work the second time), and the error text is already
#: in the model's context - the nudge is for when that has visibly not worked.
REPEAT_NUDGE_AFTER = 3

#: How long to wait for a skill action to finish before giving up on it.
SKILL_TIMEOUT_S = 120.0

#: Times the model may end a turn in prose before the run is abandoned. One is
#: usually enough - see backends/base.py nudge(). More than two means the model
#: is not going to use the tools, and repeating the request wastes tokens
#: against a rate limit.
MAX_NUDGES = 2

#: How long to wait on the affordance service. Generous relative to the work
#: (a few dozen IK queries) but bounded: if the skill server is wedged, the run
#: should degrade to "no affordance data" rather than hang before every turn.
AFFORDANCE_TIMEOUT_S = 10.0


class LlmPlanner(Node):

    def __init__(self):
        super().__init__('llm_planner')

        self.declare_parameter('dry_run', False)
        self.declare_parameter('backend', 'anthropic')   # anthropic | groq
        self.declare_parameter('model', '')              # '' = backend default
        self.declare_parameter('effort', 'high')     # low|medium|high|xhigh|max
        self.declare_parameter('temperature', 0.0)   # groq only
        self.declare_parameter('log_dir', '')
        self.declare_parameter('memory_file', '')
        # Memory persists across restarts on purpose. It must NOT persist across
        # a world reset, or the robot will confidently report objects into bins
        # that were emptied while it was not running.
        self.declare_parameter('forget', False)

        # ---- ablation switches -------------------------------------------
        # Both default to the full system, so an ordinary run is unaffected.
        # They exist so one binary can run every arm of the ablation: a second
        # code path that is only ever exercised by the experiment is a second
        # code path nobody has debugged.
        #
        # show_affordances:=false hides the `can pick` column and the notes.
        #   It does NOT disable the affordance service - the skill server still
        #   refuses impossible picks, because removing the safety check would
        #   measure something else entirely. What this ablates is the model's
        #   *foresight*: it must discover infeasibility by attempting and
        #   failing, which is precisely the behaviour SayCan's affordance term
        #   exists to avoid.
        #
        # persist_memory:=false wipes memory after every command, reproducing
        #   the pre-memory system: each command starts with no referent for
        #   "the other one" and no knowledge of what is inside a bin.
        self.declare_parameter('show_affordances', True)
        self.declare_parameter('persist_memory', True)

        self.dry_run = bool(self.get_parameter('dry_run').value)
        self.effort = str(self.get_parameter('effort').value)
        self.show_affordances = bool(
            self.get_parameter('show_affordances').value)
        self.persist_memory = bool(self.get_parameter('persist_memory').value)

        self.cb = ReentrantCallbackGroup()
        self._world: Optional[WorldState] = None
        self._world_lock = threading.Lock()
        # Memory is written from the /world_state callback and read from the
        # planner thread, so it needs its own lock. Sharing _world_lock would
        # hold perception up for the length of a render.
        self._mem_lock = threading.Lock()

        self.create_subscription(
            WorldState, '/world_state', self._world_cb, 10, callback_group=self.cb)

        self._pick = ActionClient(self, Pick, 'pick', callback_group=self.cb)
        self._place = ActionClient(self, Place, 'place', callback_group=self.cb)
        self._home = ActionClient(self, Home, 'home', callback_group=self.cb)
        self._affordance = self.create_client(
            CheckAffordances, 'check_affordances', callback_group=self.cb)

        # The backend owns the provider SDK and the conversation; this node never
        # touches either. Keys come from the environment only - a key in source
        # ends up in git history, and history is not something you can take back.
        # The predecessor project is the cautionary example: its Groq key is
        # still retrievable from its public history.
        try:
            self._backend = make_backend(
                str(self.get_parameter('backend').value),
                str(self.get_parameter('model').value),
                prompts.SYSTEM_PROMPT,
                TOOL_SCHEMAS,
                effort=self.effort,
                temperature=float(self.get_parameter('temperature').value),
            )
        except (RuntimeError, ValueError) as exc:
            self.get_logger().error(str(exc))
            raise SystemExit(2)

        log_dir = self.get_parameter('log_dir').value or str(
            Path.home() / 'moveit2_obb' / 'runs')
        self._log_path = Path(log_dir)
        self._log_path.mkdir(parents=True, exist_ok=True)
        self._log_file = self._log_path / f"{datetime.now():%Y%m%d_%H%M%S}.jsonl"

        memory_file = Path(self.get_parameter('memory_file').value
                           or self._log_path / 'memory.json')
        self._memory = RobotMemory(memory_file)
        if bool(self.get_parameter('forget').value):
            self._memory.forget_all()
            self.get_logger().info('memory cleared on request (forget:=true)')

        mode = 'DRY RUN (no arm motion)' if self.dry_run else 'live'
        self.get_logger().info(
            f'llm_planner ready [{mode}] backend={self._backend.describe()}')
        self.get_logger().info(f'decision log: {self._log_file}')
        known = len(self._memory.all_beliefs())
        self.get_logger().info(
            f'memory: {memory_file} ({known} object'
            f'{"" if known == 1 else "s"} remembered)')

    # ------------------------------------------------------------------ world

    def _world_cb(self, msg: WorldState):
        with self._world_lock:
            self._world = msg
        # Fold every snapshot into memory, not just the ones a command happens
        # to look at. Object permanence depends on seeing the frame BEFORE the
        # gripper occludes something, and that frame usually arrives while the
        # planner thread is blocked inside a skill.
        with self._mem_lock:
            self._memory.observe([Observation.from_msg(o) for o in msg.objects])

    def _affordances(self) -> Tuple[Dict[int, str], List[str]]:
        """Ask the skill server what is actually possible, keyed by memory id.

        Returns (short verdict per id, notes worth spelling out). Degrades to
        empty rather than raising: an affordance layer that is down should cost
        the planner its pre-filtering, not its ability to act. The prompt is
        written so that an absent `can pick` column means "attempt and find
        out", which is the behaviour this system had before the layer existed.
        """
        if not self.show_affordances:
            return {}, []                 # ablation arm; see show_affordances
        if not self._affordance.service_is_ready():
            return {}, []

        req = CheckAffordances.Request()          # empty ids = score everything
        future = self._affordance.call_async(req)
        deadline = time.monotonic() + AFFORDANCE_TIMEOUT_S
        while not future.done():
            if time.monotonic() > deadline:
                self.get_logger().warning(
                    'affordance check timed out; proceeding without it')
                return {}, []
            time.sleep(0.02)

        res = future.result()
        if res is None:
            return {}, []

        with self._mem_lock:
            by_world = {b.world_id: b.mem_id
                        for b in self._memory.visible() if b.world_id is not None}

        verdicts: Dict[int, str] = {}
        notes: List[str] = []
        for wid, feasible, score, reason, blocker in zip(
                res.ids, res.feasible, res.score, res.reason, res.blocker):
            mem_id = by_world.get(wid)
            if mem_id is None:
                continue                  # scored an object memory has not folded in yet
            reason = affordance.renumber_reason(reason, by_world)
            if not feasible:
                verdicts[mem_id] = f'NO ({blocker})'
                notes.append(f'  id {mem_id}: cannot be picked - {reason}')
            elif score < 1.0:
                verdicts[mem_id] = f'yes, {score:.2f}'
                notes.append(f'  id {mem_id}: {reason}')
            else:
                verdicts[mem_id] = 'yes'
        return verdicts, notes


    def _state_text(self, header: str = 'World state') -> str:
        """Everything the model is shown about the scene, in one block.

        Three sources merged here and nowhere else: memory (where things are,
        including out of sight), affordances (what can be done to them), and
        scene.RECEPTACLES (the fixed bins, which are never perceived).
        """
        with self._world_lock:
            never_seen = self._world is None
        if never_seen:
            return (f'{header}: perception has never published. Is world_model '
                    f'running? Nothing can be picked up until it is.')

        verdicts, notes = self._affordances()
        with self._mem_lock:
            block = self._memory.render_objects(header, verdicts)
        parts = [block, '', '  Destinations (fixed, always available):']
        parts += [f'  --  bin       {r.color:<8} {r.x:+.3f}  {r.y:+.3f}'
                  for r in scene.RECEPTACLES.values()]
        if notes:
            parts += ['', 'Affordance notes (from solving the arm\'s kinematics, '
                          'not from guessing):'] + notes
        return '\n'.join(parts)

    def _world_id(self, mem_id: int) -> Tuple[Optional[int], str]:
        """Translate a memory id into the id the skill server understands.

        The model works in memory ids, which outlive visibility; the skills work
        in world_model ids, which do not. Failing here is informative rather
        than an error: "believed to be in red_bin" tells the model something it
        can act on, where "no object with id 4" would have it re-reading a world
        state that legitimately does not contain the object any more.
        """
        with self._mem_lock:
            belief = self._memory.belief(mem_id)
            world_id = self._memory.world_id_for(mem_id)
        if belief is None:
            return None, (f'nothing is known about id {mem_id}. Use an id from '
                          f'the world state.')
        if world_id is None:
            return None, (f'object {mem_id} ({belief.label}) is not visible '
                          f'right now - it is '
                          f'{memory.location_phrase(belief.location)}. The '
                          f'camera has to see something to pick it up; if the '
                          f'arm is in the way, call home first.')
        return world_id, ''

    # ------------------------------------------------------------------ skills

    def _await_action(self, client: ActionClient, goal, label: str
                      ) -> Tuple[bool, str]:
        """Send a goal and block until it finishes. Returns (ok, message).

        Blocking is fine and deliberate: this runs on the planner thread, while
        the MultiThreadedExecutor keeps servicing /world_state on another. That
        is what makes the refreshed state in the result actually fresh.
        """
        if not client.wait_for_server(timeout_sec=5.0):
            return False, (f'the {label} skill is not available - is skill_server '
                           f'running?')

        send_future = client.send_goal_async(goal)
        deadline = time.monotonic() + SKILL_TIMEOUT_S
        while not send_future.done():
            if time.monotonic() > deadline:
                return False, f'{label} goal was never accepted (timed out)'
            time.sleep(0.05)

        handle = send_future.result()
        if handle is None or not handle.accepted:
            return False, f'{label} goal was rejected by the skill server'

        result_future = handle.get_result_async()
        while not result_future.done():
            if time.monotonic() > deadline:
                return False, f'{label} did not finish within {SKILL_TIMEOUT_S:.0f}s'
            time.sleep(0.05)

        result = result_future.result().result
        return bool(result.success), result.message

    def _execute_tool(self, name: str, args: Dict[str, Any]) -> Tuple[str, bool]:
        """Run one tool. Returns (result_text, is_error).

        The returned text always ends with the refreshed world state - that is
        what closes the loop. The model's next turn sees both what happened and
        what the scene looks like now.
        """
        # recall touches nothing and moves nothing, so it runs even in dry-run
        # and returns without a world state - it is a question about the past,
        # and answering it with a fresh scene dump would just cost tokens.
        if name == 'recall':
            with self._mem_lock:
                return self._memory.recall(args.get('object_id')), False

        if self.dry_run:
            body = f'DRY RUN - would have called {name}({json.dumps(args)}).'
            return (body + SCENE_SEPARATOR
                    + self._state_text('World state (unchanged)')), False

        if name == 'pick':
            mem_id = int(args['object_id'])
            world_id, why = self._world_id(mem_id)
            if world_id is None:
                return why, True
            goal = Pick.Goal()
            goal.object_id = world_id
            ok, msg = self._await_action(self._pick, goal, 'pick')
            with self._mem_lock:
                self._memory.note_pick(mem_id, ok, msg)

        elif name == 'place_in_bin':
            receptacle = scene.find_receptacle(str(args['bin_name']))
            if receptacle is None:
                return f"unknown bin {args['bin_name']!r}", True
            goal = Place.Goal()
            goal.target = receptacle.name
            ok, msg = self._await_action(self._place, goal, 'place')
            with self._mem_lock:
                self._memory.note_place(receptacle.name, ok, msg,
                                        x=receptacle.x, y=receptacle.y)

        elif name == 'place_at':
            goal = Place.Goal()
            goal.target = ''
            goal.x = float(args['x'])
            goal.y = float(args['y'])
            ok, msg = self._await_action(self._place, goal, 'place')
            with self._mem_lock:
                self._memory.note_place('', ok, msg, x=goal.x, y=goal.y)

        elif name == 'home':
            ok, msg = self._await_action(self._home, Home.Goal(), 'home')

        else:
            return f'no such tool: {name}', True

        # Perception runs at 5 Hz; give it a beat to see the settled scene rather
        # than the one from mid-motion.
        time.sleep(0.5)
        prefix = 'SUCCESS' if ok else 'FAILED'
        # The separator is load-bearing, not decoration: it is what lets the
        # backend drop this scene once a newer one arrives. See
        # backends/base.py SCENE_SEPARATOR.
        return f'{prefix}: {msg}' + SCENE_SEPARATOR + self._state_text(), not ok

    # -------------------------------------------------------------------- log

    def _log(self, record: Dict[str, Any]):
        """One JSON line per decision.

        Cheap now, expensive to retrofit: this is what makes grounding accuracy,
        task success, and replan-recovery rate computable later without touching
        the running system.
        """
        record['t'] = datetime.now(timezone.utc).isoformat()
        try:
            with self._log_file.open('a', encoding='utf-8') as fh:
                fh.write(json.dumps(record, default=str) + '\n')
        except OSError as exc:                                   # noqa: BLE001
            self.get_logger().warning(f'could not write decision log: {exc}')

    # ------------------------------------------------------------------- loop

    def run_command(self, command: str) -> bool:
        """Run one command and close the episode, whatever the outcome.

        Every exit from _run() has to be recorded, including the unhappy ones:
        an episode that ends in a refusal or a timeout is exactly the kind the
        next command needs to know about ("you already tried that and the arm
        could not reach it"). Closing it here rather than at each return means
        a new exit path cannot forget to.
        """
        try:
            ok, summary = self._run(command)
        except Exception as exc:
            # Record WHAT went wrong, not that something did. This text is what
            # the next command sees as history, and "the run raised an
            # exception" tells the model nothing it can act on - whereas "hit
            # the provider's daily token limit" tells it not to expect the rest
            # of the task to have happened. Truncated because a provider
            # traceback can be long and this rides along on every later turn.
            detail = f'{type(exc).__name__}: {exc}'
            with self._mem_lock:
                self._memory.end_command(False, f'stopped early - {detail[:180]}')
            raise
        with self._mem_lock:
            self._memory.end_command(ok, summary)
            if not self.persist_memory:
                self._memory.forget_all()     # ablation arm; see persist_memory
        return ok

    def _run(self, command: str) -> Tuple[bool, str]:
        """Run one natural-language command to completion.

        Provider-agnostic: everything below goes through the backend interface,
        which is the check on the plan's swappability constraint. If a future
        provider needs a branch in here, the abstraction has leaked.
        """
        # A fresh conversation per command, but NOT a fresh robot. The
        # transcript is discarded and the memory is not: continuity across
        # commands comes from render_history(), which is a few lines, rather
        # than from carrying an unbounded transcript that would be re-billed
        # every turn and would eventually push the world state out of context.
        with self._mem_lock:
            self._memory.begin_command(command)
            history = self._memory.render_history()

        state = self._state_text()
        opening = (f'{history}\n\n{state}' if history else state)
        self._backend.begin(f'{opening}\n\nCommand: {command}')
        self._log({'event': 'command', 'command': command,
                   'backend': self._backend.describe(),
                   'world': state, 'history': history,
                   'dry_run': self.dry_run})

        started = time.monotonic()
        nudges = 0

        last_call = None               # (tool, args-json, result-head)
        repeats = 0
        for iteration in range(MAX_ITERATIONS):
            decision = self._backend.step()

            if decision.refusal:
                self.get_logger().error(
                    f'model declined the request: {decision.refusal}')
                self._log({'event': 'refusal', 'detail': decision.refusal})
                return False, f'declined: {decision.refusal}'

            # A truncated turn can leave a half-written tool call whose input is
            # missing arguments. Acting on that would send a malformed goal to
            # the arm, so stop instead: the token caps are generous, so hitting
            # one means something is wrong, not that the task needs more room.
            if decision.truncated:
                self.get_logger().error(
                    'response was truncated at the token cap; any tool call in '
                    'it is incomplete. Stopping rather than acting on it.')
                self._log({'event': 'truncated', 'usage': decision.usage})
                return False, 'stopped: the response hit the token cap'

            if decision.text:
                self.get_logger().info(f'[planner] {decision.text}')

            if not decision.tool_calls:
                # The model ended its turn in prose. Usually that prose IS the
                # `done` call, written out instead of made - see
                # backends/base.py nudge() for the transcript that motivated
                # this. Ask once rather than parsing it: parsing the prose is
                # the failure mode this whole project replaced.
                if nudges < MAX_NUDGES:
                    nudges += 1
                    self.get_logger().warning(
                        'model stopped without calling a tool; asking it to '
                        f'use the tools ({nudges}/{MAX_NUDGES})')
                    self._log({'event': 'nudge', 'attempt': nudges,
                               'text': decision.text})
                    self._backend.nudge(prompts.NO_TOOL_CALL_NUDGE)
                    continue
                self.get_logger().warning(
                    'model stopped without calling a tool; ending the run')
                self._log({'event': 'stopped_without_tool',
                           'stop_reason': decision.stop_reason,
                           'text': decision.text})
                return False, (decision.text or 'stopped without acting')[:200]

            # `done` ends the run. Checked before executing anything, so a `done`
            # issued alongside other calls still terminates cleanly.
            for call in decision.tool_calls:
                if call.name in TERMINAL_TOOLS:
                    ok = bool(call.input.get('success', False))
                    summary = call.input.get('summary', '')
                    self.get_logger().info(
                        f"{'DONE' if ok else 'GAVE UP'}: {summary}")
                    self._log({'event': 'done', 'success': ok, 'summary': summary,
                               'iterations': iteration + 1,
                               'wall_time_s': round(time.monotonic() - started, 1),
                               'usage': decision.usage})
                    return ok, summary

            results: List[ToolResult] = []
            for call in decision.tool_calls:
                self.get_logger().info(f'-> {call.name}({json.dumps(call.input)})')

                # Backends without server-side schema enforcement report a bad
                # call this way rather than raising. Hand it back as a tool error
                # so the model can correct itself on the next turn - the same
                # closed-loop treatment a failed grasp gets.
                if '__error__' in call.input:
                    text, is_error = call.input['__error__'], True
                else:
                    text, is_error = self._execute_tool(call.name,
                                                        dict(call.input))

                self._log({'event': 'tool_call', 'iteration': iteration,
                           'tool': call.name, 'input': dict(call.input),
                           'is_error': is_error,
                           'result_head': text.split('\n', 1)[0],
                           'usage': decision.usage})
                results.append(ToolResult(call=call, content=text,
                                          is_error=is_error))

            self._backend.add_results(results)

            # Break the identical-call loop. Only single-call turns count - a
            # turn that issued several calls is not "the same call again".
            # Success is NOT exempt: the third loop seen live was `home` four
            # times running, each one succeeding and changing nothing.
            if len(results) == 1:
                r = results[0]
                key = (r.call.name, json.dumps(r.call.input, sort_keys=True),
                       r.content.split(chr(10), 1)[0])
                repeats = repeats + 1 if key == last_call else 1
                last_call = key
                if repeats >= REPEAT_NUDGE_AFTER:
                    self.get_logger().warning(
                        f'{r.call.name} repeated identically {repeats} times '
                        f'running; nudging')
                    self._log({'event': 'repeat_nudge', 'iteration': iteration,
                               'tool': r.call.name, 'repeats': repeats,
                               'is_error': r.is_error})
                    template = (prompts.REPEATED_FAILURE_NUDGE if r.is_error
                                else prompts.REPEATED_NOOP_NUDGE)
                    self._backend.nudge(template.format(
                        tool=r.call.name, n=repeats, result=key[2]))
                    repeats = 0            # give the nudge a fresh count
            else:
                last_call, repeats = None, 0

        self.get_logger().warning(f'hit the {MAX_ITERATIONS}-step limit; stopping')
        self._log({'event': 'iteration_limit', 'limit': MAX_ITERATIONS})
        return False, f'stopped after {MAX_ITERATIONS} steps without finishing'


def main():
    rclpy.init(args=None)
    node = LlmPlanner()

    executor = MultiThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    # Give perception a moment to publish before the first prompt, so the
    # opening world state isn't empty.
    time.sleep(2.0)

    try:
        while rclpy.ok():
            try:
                command = input('\ncommand> ').strip()
            except EOFError:
                break
            if not command:
                continue
            if command in {'quit', 'exit'}:
                break
            try:
                node.run_command(command)
            except Exception as exc:                             # noqa: BLE001
                node.get_logger().error(f'run failed: {type(exc).__name__}: {exc}')
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        print(f'\ndecision log: {node._log_file}', file=sys.stderr)


if __name__ == '__main__':
    main()

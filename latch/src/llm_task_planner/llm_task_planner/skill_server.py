#!/usr/bin/env python3
"""Manipulation skills as ROS 2 actions: pick, place, home.

Replaces panda_moveit_config/scripts/arm_control_from_UI.py, which was a single
blocking macro triggered by a click: approach, open, descend, close, lift, move
to one hardcoded drop point, open. It could not report which step failed, could
not be retried, and had no notion of whether it was actually holding anything.

Three things change here, and all three exist to make the planning loop closed:

1. Each skill is an action, so success and failure are per-skill and the caller
   learns WHICH step failed and why - the message is written to be read by a
   language model deciding what to do next.

2. Grasp verification. After closing, the finger separation is measured from
   /joint_states. Fully closed means the jaw met nothing. Without this the loop
   can only ever be optimistic, and "closed-loop replanning" is a claim rather
   than a behaviour.

3. Orientation comes from llm_task_planner.grasp instead of the non-unit
   quaternion the original built by hand - see grasp.py for why that mattered.

The skills know about the robot but not about the task. They take an object id
and look the pose up in /world_state; they never take a plan.

4. Affordance grounding, added later: /check_affordances answers "could you do
   this?" without doing it. That question belongs here rather than in the
   planner, because answering it honestly needs the very IK solver and planning
   scene that execution uses - see affordance.py for why that is the half of
   SayCan worth keeping.
"""

import math
import threading
import time
from typing import Dict, Optional, Tuple

import rclpy
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from control_msgs.action import GripperCommand
from geometry_msgs.msg import Pose, PoseStamped
from sensor_msgs.msg import JointState
from moveit_msgs.msg import CollisionObject
from shape_msgs.msg import SolidPrimitive

from moveit.core.robot_state import RobotState
from moveit.planning import MoveItPy

from skill_msgs.action import Home, Pick, Place
from skill_msgs.srv import CheckAffordances
from yolov8_msgs.msg import WorldState

from llm_task_planner import affordance, scene
from llm_task_planner.grasp import top_down_grasp_quaternion

#: How long to wait for the gripper to settle before believing /joint_states.
GRIPPER_SETTLE_S = 1.0

#: Bound on waiting for the gripper controller. Exceeding this is expected and
#: informative, not a failure - it is what clamping on an object looks like.
GRIPPER_TIMEOUT_S = 30.0

#: Squeeze force. The controller's own default is 0.0, i.e. no grip at all.
GRIPPER_MAX_EFFORT = 50.0

#: How far inside the measured grasp width to hold, per finger. Enough to keep
#: the fingers loaded against the object, small enough not to resume extruding
#: it - which is what a standing command to close all the way does.
HOLD_SQUEEZE_M = 0.001

#: How stale a world state may be before a pick refuses to trust it.
WORLD_STATE_MAX_AGE_S = 5.0

#: Seconds allowed for the uncollision-checked recovery sweep back to `ready`.
#: Slow on purpose: it is not collision-checked, so if it does clip something,
#: it should nudge it rather than launch it.
RESCUE_MOVE_S = 5

#: Per-joint tolerance for calling the arm arrived at `ready`. Generous - the
#: point is to leave a wedged state and clear the camera's view, not to hit a
#: posture precisely.
RESCUE_ARRIVED_TOL_RAD = 0.10

#: How many times to ask the planner for the same segment before reporting it
#: unreachable, and how long to pause between attempts so the current-state
#: monitor can refresh.
PLAN_ATTEMPTS = 3
PLAN_RETRY_PAUSE_S = 0.15

#: Per-attempt budget for one IK query, and how many attempts before calling a
#: pose unreachable. Kept small: a full affordance sweep runs up to
#: 2 poses x IK_ATTEMPTS x the number of objects, once per planner turn, and it
#: sits between the model deciding and the arm moving.
IK_TIMEOUT_S = 0.05
IK_ATTEMPTS = 3


class SkillServer(Node):

    def __init__(self):
        super().__init__('skill_server')

        # verify=false makes pick and place report success whenever their
        # MOTION completes - no finger-gap check, no held-object check, no
        # slip detection. This is the skill layer as most LLM-robot systems
        # have it, and as this one had it before FINDINGS 4b/4c. It exists for
        # exactly one purpose: the "naive agent" baseline, whose self-reported
        # success is expected to exceed its real success. The benchmark
        # refuses to run any other condition against a server in this state,
        # and logs the mode at every launch so it cannot be on by accident.
        self.declare_parameter('verify', True)
        self._verify: bool = bool(self.get_parameter('verify').value)
        if not self._verify:
            self.get_logger().warning(
                '*** verify=false: skills will NOT check their own outcomes. '
                'Only the naive-baseline condition may run against this server.')

        # A reentrant group throughout: an action callback blocks for seconds
        # while the arm moves, and the subscriptions feeding it (/joint_states,
        # /world_state) must keep being serviced during that time. With the
        # default mutually-exclusive group the grasp check would read the state
        # from before the motion and always agree with itself.
        self.cb = ReentrantCallbackGroup()

        self.panda = MoveItPy(node_name='moveit_py')
        self.arm = self.panda.get_planning_component('panda_arm')
        self.hand = self.panda.get_planning_component('hand')
        self.robot_model = self.panda.get_robot_model()
        self.robot_state = RobotState(self.robot_model)

        self._world: Optional[WorldState] = None
        self._world_lock = threading.Lock()
        self._finger_joint: Optional[float] = None
        #: Arm joints, for verifying that a rescue actually arrived. Locked
        #: because the callback runs on a different thread from the skill that
        #: reads it, and a half-updated posture would make _at_ready() answer
        #: about a state the arm was never in.
        self._arm_joints: Dict[str, float] = {}
        self._joint_lock = threading.Lock()
        #: Id of the object the last confirmed pick closed on, or None once it
        #: has been released. The finger-gap check in place() cannot tell an
        #: open EMPTY jaw from a held blue cube (60 mm vs 61 mm - see there), so
        #: after the first release every further place passed that check, flew
        #: to the bin, opened on nothing, and reported SUCCESS. The planner was
        #: told "released the held object" eleven times in one trial with the
        #: table untouched, and looped on that false feedback until the step
        #: cap. This is the one thing the sensor cannot say and the server
        #: knows for certain: it did the grasping and it did the releasing.
        self._held: Optional[int] = None
        #: Created lazily by _rescue_to_ready(); most runs never need it.
        self._traj_pub = None
        #: Segments that needed more than one planning attempt. Counted rather
        #: than swallowed: if this is large the retry is papering over a real
        #: reachability problem, and that is worth knowing.
        self._plan_retries = 0
        #: Drops made into each bin, so successive objects go to different spots
        #: rather than stacking at the centre. Counting rather than perceiving
        #: because the camera cannot see into a bin - but the robot did the
        #: placing, so this is its own action history, not an assumption.
        self._bin_drops: Dict[str, int] = {}

        self.create_subscription(
            WorldState, '/world_state', self._world_cb, 10, callback_group=self.cb)
        self.create_subscription(
            JointState, '/joint_states', self._joint_cb, 10, callback_group=self.cb)

        self._pick = ActionServer(
            self, Pick, 'pick', self._execute_pick,
            goal_callback=self._accept, cancel_callback=self._allow_cancel,
            callback_group=self.cb)
        self._place = ActionServer(
            self, Place, 'place', self._execute_place,
            goal_callback=self._accept, cancel_callback=self._allow_cancel,
            callback_group=self.cb)
        self._home = ActionServer(
            self, Home, 'home', self._execute_home,
            goal_callback=self._accept, cancel_callback=self._allow_cancel,
            callback_group=self.cb)

        self._affordances = self.create_service(
            CheckAffordances, 'check_affordances', self._on_check_affordances,
            callback_group=self.cb)

        # Direct line to the gripper controller - see _gripper() for why this
        # bypasses MoveIt.
        self._gripper_client = ActionClient(
            self, GripperCommand, '/panda_hand_controller/gripper_cmd',
            callback_group=self.cb)

        #: One-shot flags so a degraded affordance check announces itself once
        #: instead of every request. Silence here would be the worst outcome:
        #: the planner would keep trusting a score that had quietly stopped
        #: being a real feasibility answer.
        self._warned_no_collision_check = False
        self._warned_no_seed = False

        self._publish_static_collision_objects()
        self.get_logger().info(
            'skill_server ready: /pick /place /home /check_affordances')

    # ---------------------------------------------------------------- plumbing

    def _accept(self, _goal):
        return GoalResponse.ACCEPT

    def _allow_cancel(self, _goal_handle):
        return CancelResponse.ACCEPT

    def _world_cb(self, msg: WorldState):
        with self._world_lock:
            self._world = msg

    def _joint_cb(self, msg: JointState):
        try:
            self._finger_joint = msg.position[msg.name.index('panda_finger_joint1')]
        except (ValueError, IndexError):
            pass
        arm = {}
        for name in scene.ARM_JOINT_NAMES:
            try:
                arm[name] = msg.position[msg.name.index(name)]
            except (ValueError, IndexError):
                return
        with self._joint_lock:
            self._arm_joints = arm

    def _lookup(self, object_id: int):
        """Find an object in the most recent world state.

        Refuses stale data rather than reaching for where something used to be -
        a stale pick is worse than a refused one, because it looks like a
        planning failure instead of a perception one.
        """
        with self._world_lock:
            world = self._world
        if world is None:
            return None, 'no world state received yet - is world_model running?'

        age = (self.get_clock().now().nanoseconds
               - rclpy.time.Time.from_msg(world.header.stamp).nanoseconds) / 1e9
        if age > WORLD_STATE_MAX_AGE_S:
            return None, f'world state is {age:.1f}s stale; refusing to act on it'

        for obj in world.objects:
            if obj.id == object_id:
                return obj, ''
        seen = ', '.join(f'{o.id}({o.class_name})' for o in world.objects) or 'none'
        return None, f'no object with id {object_id}. Currently visible: {seen}'

    # -------------------------------------------------------------- affordance

    def _seed_state(self):
        """A RobotState to run IK from, seeded at the arm's current pose.

        The seed matters. KDL's solver is local and iterative, so a seed far
        from any solution makes it report failure for a pose that is perfectly
        reachable - a false negative that would show up as the planner refusing
        a good object, which is exactly the class of bug this whole affordance
        layer is supposed to remove. Seeding where the arm actually is also
        matches what planning will do a moment later.

        A fresh RobotState per request rather than the shared self.robot_state:
        the service shares a reentrant callback group with the actions, and IK
        mutates whatever state it is handed.
        """
        state = RobotState(self.robot_model)
        state.set_to_default_values()
        try:
            with self.panda.get_planning_scene_monitor().read_only() as ps:
                state.joint_positions = dict(ps.current_state.joint_positions)
        except Exception as exc:                                   # noqa: BLE001
            if not self._warned_no_seed:
                self._warned_no_seed = True
                self.get_logger().warning(
                    f'could not seed IK from the current state ({exc}); falling '
                    f'back to the default pose. Affordance scores may report '
                    f'false negatives on reachable objects.')
        return state

    def _ik_probe(self, x: float, y: float, z: float, yaw: float) -> str:
        """Is there a collision-free arm configuration putting link8 here?

        Returns '' when yes, otherwise why not. Two distinct failures, and they
        are reported distinctly because they mean opposite things:

          1. IK does not solve. The wrist cannot reach that orientation there -
             a kinematic fact, and the one a radial reach check cannot see. A
             pose can sit comfortably inside the reach envelope and still have
             no solution at the yaw a particular grasp demands.

          2. IK solves but the posture collides. The pose is reachable and the
             way the arm would have to fold to get there is not. The planning
             scene here already carries the table and both bins
             (_publish_static_collision_objects), so this is a real check.

        These were originally collapsed into one bool, and the first live run
        reported all nine objects as "no IK solution" - which sent me looking at
        frames and tip links when the message itself was the unreliable part.
        Distinguishing them cost three lines and would have saved the detour.

        Retried, because KDL is a random-restart solver and one failure is weak
        evidence. Wrongly answering "infeasible" costs the planner an object it
        could have picked, so the bias is toward asking again before saying no.
        """
        goal = Pose()
        goal.position.x = float(x)
        goal.position.y = float(y)
        goal.position.z = float(z)
        qx, qy, qz, qw = top_down_grasp_quaternion(yaw)
        goal.orientation.x = qx
        goal.orientation.y = qy
        goal.orientation.z = qz
        goal.orientation.w = qw

        solved_ever = False
        for _ in range(IK_ATTEMPTS):
            state = self._seed_state()
            try:
                solved = state.set_from_ik(
                    joint_model_group_name='panda_arm', geometry_pose=goal,
                    tip_name='panda_link8', timeout=IK_TIMEOUT_S)
            except Exception as exc:                               # noqa: BLE001
                self.get_logger().warning(f'IK probe raised: {exc}')
                return f'the IK query itself failed ({type(exc).__name__})'
            if not solved:
                continue
            solved_ever = True

            # Link transforms are cached; without this the collision check runs
            # against whatever the state held before IK wrote the joints.
            try:
                state.update()
            except Exception:                                      # noqa: BLE001
                pass

            try:
                with self.panda.get_planning_scene_monitor().read_only() as ps:
                    if ps.is_state_colliding(robot_state=state,
                                             joint_model_group_name='panda_arm',
                                             verbose=False):
                        continue          # solvable, but not from here - retry
            except Exception as exc:                               # noqa: BLE001
                # Degrade to kinematics-only rather than failing the request,
                # but say so: a score that silently stopped checking collisions
                # is worse than no score, because it still looks authoritative.
                if not self._warned_no_collision_check:
                    self._warned_no_collision_check = True
                    self.get_logger().warning(
                        f'planning-scene collision check unavailable ({exc}); '
                        f'affordance scores are kinematic-only from now on and '
                        f'will not catch a pose blocked by the table or a bin.')
            return ''

        if solved_ever:
            return ('the wrist can reach it but every arm posture that does '
                    'collides with the table, a bin, or the arm itself')
        return (f'no IK solution in {IK_ATTEMPTS} attempts of '
                f'{IK_TIMEOUT_S * 1000:.0f} ms')

    def _on_check_affordances(self, request, response):
        """Score every requested object. See affordance.py for the reasoning.

        Answers from the same /world_state the skills act on, so an affordance
        and the pick that follows it cannot disagree about where anything is.
        """
        with self._world_lock:
            world = self._world

        if world is None:
            self.get_logger().warning(
                'affordance request with no world state yet')
            return response

        wanted = set(request.object_ids)
        geoms = [affordance.ObjectGeom.from_msg(o) for o in world.objects]
        # Clutter is scored against EVERY visible object, not just the requested
        # ones: a bolt nobody asked about still blocks the fingers.
        targets = [g for g in geoms if not wanted or g.id in wanted]

        for g in targets:
            a = affordance.score_pick(g, geoms, self._ik_probe)
            response.ids.append(a.id)
            response.feasible.append(a.feasible)
            response.score.append(float(a.score))
            response.reason.append(a.reason)
            response.blocker.append(a.blocker)

        missing = wanted - {g.id for g in targets}
        if missing:
            self.get_logger().info(
                f'affordances requested for ids not in the world state: '
                f'{sorted(missing)}')

        # A scene where NOTHING can be picked up is possible, and is also what a
        # broken probe looks like. The two are worth telling apart out loud,
        # because the broken version is silent otherwise: the planner simply
        # declines every task and blames the scene. So when the verdict is
        # unanimous, probe a pose that is known to be good and say which it was.
        if targets and not any(response.feasible):
            canary = self._ik_probe(0.40, 0.00, scene.grasp_height('cube'), 0.0)
            self.get_logger().warning(
                f'all {len(targets)} objects scored infeasible. Control probe '
                f'at (0.40, 0.00) straight in front of the base: '
                + ('SOLVES, so the scene really is blocked'
                   if not canary else f'ALSO FAILS ({canary}) - the affordance '
                                      f'probe is misconfigured, not the scene'))
        return response

    # --------------------------------------------------------------- collision

    def _publish_static_collision_objects(self):
        """Tell MoveIt about the table and the bins.

        They exist in Gazebo but not in the planning scene, so without this the
        planner is free to route the arm straight through a bin wall or scrape
        the tabletop. It only finds out at execution time, as a collision in the
        physics engine - by which point the trajectory is already running.

        Static geometry only. Manipulable objects deliberately stay out: they
        move, and a stale collision object is worse than none.
        """
        objects = []

        table = CollisionObject()
        table.header.frame_id = 'panda_link0'
        table.id = 'table'
        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        # 1.5 x 0.8 m top, 0.1 m thick slab sunk below the surface so the arm
        # can still reach objects resting ON it without an immediate collision.
        box.dimensions = [1.5, 0.8, 0.10]
        pose = Pose()
        pose.position.x = 0.24          # table centre, link0 (world 0.29 - 0.05)
        pose.position.y = 0.0
        pose.position.z = scene.TABLE_TOP_Z - 0.05
        pose.orientation.w = 1.0
        table.primitives = [box]
        table.primitive_poses = [pose]
        table.operation = CollisionObject.ADD
        objects.append(table)

        for r in scene.RECEPTACLES.values():
            bin_obj = CollisionObject()
            bin_obj.header.frame_id = 'panda_link0'
            bin_obj.id = r.name
            b = SolidPrimitive()
            b.type = SolidPrimitive.BOX
            # A solid block up to the rim rather than four thin walls. The arm
            # must not enter the bin except from directly above, and place()
            # releases above the rim, so modelling the cavity buys nothing and
            # risks the planner threading the gripper in sideways.
            b.dimensions = [0.20, 0.20, r.rim_z - scene.TABLE_TOP_Z]
            p = Pose()
            p.position.x = r.x
            p.position.y = r.y
            p.position.z = (r.rim_z + scene.TABLE_TOP_Z) / 2.0
            p.orientation.w = 1.0
            bin_obj.primitives = [b]
            bin_obj.primitive_poses = [p]
            bin_obj.operation = CollisionObject.ADD
            objects.append(bin_obj)

        try:
            with self.panda.get_planning_scene_monitor().read_write() as ps:
                for obj in objects:
                    ps.apply_collision_object(obj)
                ps.current_state.update()
            self.get_logger().info(
                f'planning scene: added {", ".join(o.id for o in objects)}')
        except Exception as exc:                                  # noqa: BLE001
            # Non-fatal: the arm still works, it is just less safe. Say so
            # loudly rather than silently planning without collision geometry.
            self.get_logger().error(
                f'FAILED to add static collision objects ({exc}). '
                'The planner does not know about the table or bins.')

    # ------------------------------------------------------------------ motion

    def _plan_and_execute(self, component, what: str) -> Tuple[bool, str]:
        """Plan and run one segment. Returns (ok, why_not).

        Kept from the original arm_control_from_UI.plan_and_execute, which had
        the plan/execute cycle right - including the falsy-on-failure check on
        plan_result - but swallowed the outcome instead of returning it.
        """
        # Retry the plan before believing it. OMPL samples randomly, so a
        # failed attempt is evidence about one sample, not about the pose.
        #
        # This was the single largest failure category in the 75-trial suite:
        # 38 of 86 pick failures were "motion planning failed" for approach,
        # descend or lift - all poses the affordance layer had already certified
        # reachable with real IK moments earlier. Retrying costs milliseconds
        # here; the alternative is an error returned to the model, which spends
        # a round trip and several thousand tokens to decide what to do about a
        # dice roll.
        #
        # It also makes the prompt honest. It told the model that "motion
        # planning failed" means unreachable and not to retry the identical
        # call because it would fail identically - which is not true of a
        # randomized planner, and steered it away from the cheapest fix.
        plan_result = None
        for attempt in range(PLAN_ATTEMPTS):
            plan_result = component.plan()
            if plan_result:
                if attempt:
                    self._plan_retries += 1
                    self.get_logger().info(
                        f'{what}: planned on attempt {attempt + 1}')
                break
            time.sleep(PLAN_RETRY_PAUSE_S)
        if not plan_result:
            return False, (f'motion planning failed for {what} after '
                           f'{PLAN_ATTEMPTS} attempts')
        try:
            ok = self.panda.execute(plan_result.trajectory, controllers=[])
        except Exception as exc:                                   # noqa: BLE001
            return False, f'execution raised during {what}: {exc}'

        # execute() returns an error code object in some builds and None in
        # others; treat only an explicit falsy non-None as failure.
        #
        # The reason for reporting the code rather than just the verdict: a
        # descend into a bin failed here while the logs showed
        # "panda_arm_controller: Goal reached, success!" 60 ms earlier. The arm
        # had done exactly what it was told, and "execution failed" sent the
        # investigation to the robot instead of to this line. A bare boolean is
        # not enough information to debug a disagreement between two layers.
        if ok is not None and not ok:
            detail = self._error_detail(ok)
            return False, f'execution failed during {what} ({detail})'
        return True, ''

    @staticmethod
    def _error_detail(code) -> str:
        """Human-readable form of whatever execute() handed back.

        MoveItPy returns a moveit.core.controller_manager.ExecutionStatus here,
        not the MoveItErrorCodes the name suggests, and its repr is a bare
        object address - which is exactly as useful as the boolean it replaced.
        The status string is what distinguishes "the controller refused" from
        "MoveIt gave up waiting", and those want completely different fixes.
        """
        for attr in ('status', 'val', 'error_code'):
            value = getattr(code, attr, None)
            if value is None:
                continue
            if isinstance(value, str):
                return value
            if isinstance(value, int):
                break
            text = str(value)
            if text and not text.startswith('<'):
                return text

        text = str(code)
        if text and not text.startswith('<'):
            return text

        val = getattr(code, 'val', None)
        if val is None:
            # Last resort: name whatever public attributes it does have, so the
            # next person does not have to guess at the type either.
            fields = {a: getattr(code, a) for a in dir(code)
                      if not a.startswith('_') and not callable(getattr(code, a))}
            return f'{type(code).__name__}{fields or " (no readable fields)"}'
        # From moveit_msgs/MoveItErrorCodes. Only the ones reachable from here.
        names = {
            1: 'SUCCESS', -1: 'FAILURE', -2: 'PLANNING_FAILED',
            -3: 'INVALID_MOTION_PLAN',
            -4: 'MOTION_PLAN_INVALIDATED_BY_ENVIRONMENT_CHANGE',
            -5: 'CONTROL_FAILED', -6: 'UNABLE_TO_AQUIRE_SENSOR_DATA',
            -7: 'TIMED_OUT', -8: 'PREEMPTED',
            -10: 'START_STATE_IN_COLLISION',
            -11: 'START_STATE_VIOLATES_PATH_CONSTRAINTS',
            -12: 'GOAL_IN_COLLISION', -13: 'GOAL_VIOLATES_PATH_CONSTRAINTS',
            -14: 'GOAL_CONSTRAINTS_VIOLATED', -15: 'INVALID_GROUP_NAME',
            -17: 'INVALID_ROBOT_STATE', -31: 'NO_IK_SOLUTION',
        }
        return f'{names.get(val, "unknown")}={val}'

    def _move_to(self, x: float, y: float, z: float, yaw: float,
                 what: str) -> Tuple[bool, str]:
        reachable, why = scene.is_reachable(x, y)
        if not reachable:
            return False, why

        goal = PoseStamped()
        goal.header.frame_id = 'panda_link0'
        goal.pose.position.x = x
        goal.pose.position.y = y
        goal.pose.position.z = z
        qx, qy, qz, qw = top_down_grasp_quaternion(yaw)
        goal.pose.orientation.x = qx
        goal.pose.orientation.y = qy
        goal.pose.orientation.z = qz
        goal.pose.orientation.w = qw

        # Always plan from where the arm actually is. The original set this only
        # for the hand, so a failed arm segment left the next plan starting from
        # a pose the robot never reached.
        self.arm.set_start_state_to_current_state()
        self.arm.set_goal_state(pose_stamped_msg=goal, pose_link='panda_link8')
        return self._plan_and_execute(self.arm, what)

    def _gripper(self, finger_joint_value: float, what: str) -> Tuple[bool, str]:
        """Drive the gripper via GripperCommand directly, NOT through MoveIt.

        The original code planned the `hand` group with MoveIt and blocked on
        execution. That deadlocks on exactly the case we care about: closing onto
        an object stops the fingers short of the commanded position, MoveIt waits
        for a goal that can never be reached, and the pick action never returns.
        Measured: a close that should take 0.2 s sat unfinished after 150 s.

        Talking to the GripperActionController directly lets us bound the wait
        and then decide from the MEASURED finger gap - which is the only thing
        that actually tells us whether we are holding something. A timeout here
        is information, not an error.
        """
        goal = GripperCommand.Goal()
        goal.command.position = float(finger_joint_value)
        goal.command.max_effort = GRIPPER_MAX_EFFORT

        if not self._gripper_client.wait_for_server(timeout_sec=5.0):
            return False, 'gripper controller unavailable'

        send_future = self._gripper_client.send_goal_async(goal)
        deadline = time.monotonic() + GRIPPER_TIMEOUT_S
        while not send_future.done() and time.monotonic() < deadline:
            time.sleep(0.02)
        handle = send_future.result() if send_future.done() else None
        if handle is None or not handle.accepted:
            return False, f'{what}: gripper goal rejected'

        result_future = handle.get_result_async()
        while not result_future.done() and time.monotonic() < deadline:
            time.sleep(0.02)

        time.sleep(GRIPPER_SETTLE_S)
        if not result_future.done():
            # Stopped moving without reporting - normal when clamped on an
            # object. Report success and let the caller read the gap.
            return True, f'{what}: no controller result (likely clamped)'
        return True, ''

    def _measured_gap(self) -> Optional[float]:
        """Separation between the fingers, metres, or None if unknown."""
        if self._finger_joint is None:
            return None
        return 2.0 * self._finger_joint

    # ------------------------------------------------------------------- pick

    def _execute_pick(self, goal_handle):
        req = goal_handle.request
        result = Pick.Result()

        def fail(msg: str):
            self.get_logger().warning(f'pick({req.object_id}) failed: {msg}')
            result.success = False
            result.message = msg
            result.finger_gap = self._measured_gap() or 0.0
            goal_handle.abort()
            return result

        def stage(name: str):
            fb = Pick.Feedback()
            fb.stage = name
            goal_handle.publish_feedback(fb)

        obj, why = self._lookup(req.object_id)
        if obj is None:
            return fail(why)
        if obj.receptacle:
            return fail(f'object {obj.id} is a {obj.class_name} - a destination, '
                        f'not something that can be picked up')
        if not obj.graspable:
            return fail(f'object {obj.id} ({obj.color} {obj.class_name}) is marked '
                        f'not graspable: short axis {obj.short_axis * 1000:.0f} mm '
                        f'vs {scene.MAX_GRASPABLE_WIDTH * 1000:.0f} mm jaw limit')

        approach_z = scene.approach_height(obj.class_name)
        grasp_z = scene.grasp_height(obj.class_name)
        yaw = obj.grasp_yaw

        # Open only as wide as this object needs, not to a fixed 60 mm.
        #
        # The finger joint moves at roughly 0.6 mm/s in this simulator whatever
        # the position gain is set to (measured identically at gain 15 and 500),
        # so travel distance is what costs time: opening to 60 mm and closing on
        # a 40 mm cube is ~17 mm of travel per finger, about half a minute, and
        # the close would time out mid-motion and be scored as a grasp. Clearing
        # the object by OPEN_CLEARANCE instead keeps travel to a few mm.
        open_gap = scene.open_gap_for(obj.short_axis)

        stage('approach')
        ok, why = self._move_to(obj.x, obj.y, approach_z, yaw, 'approach above object')
        if not ok:
            return fail(why)

        stage('open')
        ok, why = self._gripper(open_gap / 2.0, 'open gripper')
        if not ok:
            return fail(why)

        stage('descend')
        ok, why = self._move_to(obj.x, obj.y, grasp_z, yaw, 'descend to grasp')
        if not ok:
            return fail(why)

        stage('close')
        self._gripper(scene.CLOSE_FINGER_JOINT, 'close gripper')

        stage('verify')
        gap = self._measured_gap()
        if gap is None:
            return fail('no /joint_states received - cannot verify the grasp')
        result.finger_gap = gap

        # A holding grasp leaves the jaw at roughly the object's width. Both
        # sides have to be bounded:
        #   too NARROW -> the fingers met nothing and closed through
        #   too WIDE   -> the fingers never actually closed
        # Checking only the narrow side (the obvious check, and the one this
        # originally had) scores "the gripper never moved" as a perfect grasp -
        # the arm then lifts nothing and reports success, which is worse than
        # failing, because the planner believes it.
        lo = scene.GRASP_MISS_GAP
        hi = obj.short_axis + scene.GRASP_WIDTH_TOLERANCE
        if self._verify and not (lo <= gap <= hi):
            self._move_to(obj.x, obj.y, approach_z, yaw, 'retreat after miss')
            if gap < lo:
                why = (f'fingers closed to {gap * 1000:.1f} mm, holding nothing - '
                       f'they met no resistance. The object may have been nudged, '
                       f'or its position in the world state may be wrong.')
            else:
                why = (f'fingers are at {gap * 1000:.1f} mm but this object is only '
                       f'{obj.short_axis * 1000:.0f} mm wide - the gripper did not '
                       f'close. Nothing is being held.')
            return fail(f'grasp failed: {why}')

        # Hold at the width we actually grasped at, instead of leaving the
        # controller driving toward CLOSE_FINGER_JOINT.
        #
        # This is a position-controlled gripper, so a standing "close to 1 mm"
        # command does not become a constant squeeze - it keeps travelling. The
        # held gap was measured creeping from 17.2 mm at verification to 10.7 mm
        # five seconds later while carrying a bolt, and that was recorded at the
        # time as not yet having caused a failure. It had: in a placement test,
        # five of eight carries reached `place` with the fingers 2.5-6.8 mm
        # apart, having extruded the object somewhere along the way. Every one
        # of those had passed grasp verification moments earlier - the grasp was
        # real when it was checked and gone by the time it was used.
        #
        # Commanded a millimetre inside the measured width rather than exactly
        # at it, so contact force is maintained rather than relaxed, and never
        # below the closed position.
        hold = max(scene.CLOSE_FINGER_JOINT, gap / 2.0 - HOLD_SQUEEZE_M)
        self._gripper(hold, 'hold the grasp')

        stage('lift')
        ok, why = self._move_to(obj.x, obj.y, scene.CARRY_HEIGHT, yaw, 'lift')
        if not ok:
            return fail(why)

        self._held = obj.id
        result.success = True
        result.message = (
            f'picked up object {obj.id} ({obj.color} {obj.class_name}) at '
            f'({obj.x:.3f}, {obj.y:.3f}). Grasp confirmed, finger gap '
            f'{gap * 1000:.1f} mm.')
        self.get_logger().info(result.message)
        goal_handle.succeed()
        return result

    # ------------------------------------------------------------------ place

    def _execute_place(self, goal_handle):
        req = goal_handle.request
        result = Place.Result()

        def fail(msg: str):
            self.get_logger().warning(f'place failed: {msg}')
            result.success = False
            result.message = msg
            goal_handle.abort()
            return result

        def stage(name: str):
            fb = Place.Feedback()
            fb.stage = name
            goal_handle.publish_feedback(fb)

        if req.target:
            receptacle = scene.find_receptacle(req.target)
            if receptacle is None:
                known = ', '.join(scene.RECEPTACLES)
                return fail(f'unknown destination {req.target!r}. Known: {known}. '
                            f'To place at a free position, leave target empty and '
                            f'give x and y instead.')
            # Spread successive drops around the bin instead of stacking them
            # at its centre - see Receptacle.drop_point for the measurement.
            nth = self._bin_drops.get(receptacle.name, 0)
            x, y = receptacle.drop_point(nth)
            release_z = receptacle.release_height
            where = receptacle.name
        else:
            x, y = req.x, req.y
            release_z = scene.link8_z_for_tip(
                scene.DEFAULT_GRASP_TIP_Z + 0.02)
            where = f'({x:.3f}, {y:.3f})'

        reachable, why = scene.is_reachable(x, y)
        if not reachable:
            return fail(why)

        # Is anything actually in the hand?
        #
        # `pick` measures the finger gap afterwards and refuses to claim a grasp
        # it cannot confirm. `place` did no such check, so a place with an empty
        # gripper - because the object was never grasped, or slipped during the
        # carry - travelled to the bin, opened the fingers on nothing, and
        # returned SUCCESS. The planner then reported the task done, and the
        # table disagreed. Across the 75-trial suite that asymmetry is the whole
        # over-claim result: the model was repeating what this skill told it.
        #
        # Same signal, same threshold as the grasp check, so the two skills
        # cannot disagree about what "holding something" means.
        if self._verify and self._held is None:
            return fail('nothing is being held - the last pick did not confirm '
                        'a grasp, or its object has already been released. '
                        'There is nothing to place. Pick an object first.')
        gap = self._measured_gap()
        if gap is None:
            return fail('no /joint_states received - cannot tell whether '
                        'anything is being held, so this place cannot be '
                        'verified')
        if self._verify and gap < scene.GRASP_MISS_GAP:
            self._held = None                  # it slipped: the hand is empty
            return fail(f'the gripper is empty - the fingers are {gap * 1000:.1f} '
                        f'mm apart, which is closed. The object slipped out. '
                        f'Pick it again.')
        # Only the narrow side is checked, unlike the grasp check, which bounds
        # both. A jaw sitting at the commanded open width would also mean
        # "holding nothing" - but the blue cube measures 61 mm against an
        # OPEN_JAW_GAP of 60 mm, so a held blue cube and an open empty jaw are
        # the same reading. Rejecting wide gaps would refuse a legitimate place.
        # Slip is still caught: an object leaving the fingers lets them close,
        # which lands under GRASP_MISS_GAP above.

        # Keep the held object's orientation as-is: re-yawing while carrying
        # risks knocking it out of the fingers for no benefit.
        yaw = 0.0

        stage('approach')
        ok, why = self._move_to(x, y, scene.CARRY_HEIGHT, yaw, f'move above {where}')
        if not ok:
            return fail(why)

        stage('descend')
        ok, why = self._move_to(x, y, release_z, yaw, f'descend into {where}')
        if not ok:
            return fail(why)

        stage('open')
        ok, why = self._gripper(scene.OPEN_JAW_GAP / 2.0, 'release')
        # Cleared here rather than on success: once the fingers have opened the
        # object is out of the hand whether or not the retreat afterwards works.
        self._held = None
        if not ok:
            return fail(why)

        stage('retreat')
        ok, why = self._move_to(x, y, scene.CARRY_HEIGHT, yaw, 'retreat')
        if not ok:
            return fail(why)

        if req.target:
            self._bin_drops[where] = self._bin_drops.get(where, 0) + 1

        result.success = True
        result.message = f'released the held object at {where}.'
        self.get_logger().info(result.message)
        goal_handle.succeed()
        return result

    # ------------------------------------------------------------------- home

    def _rescue_to_ready(self) -> Tuple[bool, str]:
        """Drive the arm to `ready` through the controller, bypassing MoveIt.

        The last resort, and the reason `home` can be relied on at all.

        MoveIt cannot recover the one state that most needs recovering. An
        interrupted or failed trajectory can leave a joint marginally outside
        its limit - joint7 has been found at 2.898 against a limit of 2.8973 -
        and MoveIt then refuses *every* plan, including the plan back to
        `ready`. The controller has no such objection and will walk the arm back
        into the legal envelope.

        Observed cost of not having this, from a benchmark trial: a `pick`
        failed at the lift, `home` was refused, a different object was tried and
        refused, and by then the wedged arm was parked over the target, so
        perception could not see it either. The planner did everything right -
        retried, called home to clear the view, chose a different object, then
        reported honest failure - and the trial was unrecoverable from the
        first refusal. The recovery code existed at the time, but only in the
        benchmark harness, which runs *between* trials: by the time it ran, the
        trial it could have saved was already scored zero.

        This is not collision-checked - that is precisely what it gives up in
        exchange for working when the planner will not. It sweeps to a known
        posture and may push an object on the way, so it runs only after MoveIt
        has already refused, and it says so in the message rather than
        presenting itself as a normal home.
        """
        from builtin_interfaces.msg import Duration              # noqa: PLC0415
        from trajectory_msgs.msg import (JointTrajectory,        # noqa: PLC0415
                                         JointTrajectoryPoint)

        if self._traj_pub is None:
            self._traj_pub = self.create_publisher(
                JointTrajectory, '/panda_arm_controller/joint_trajectory', 10)

        # Publishing before the controller has matched silently discards the
        # message - which is how an earlier version of this appeared to succeed
        # while the arm had not moved at all.
        deadline = time.time() + 5.0
        while self._traj_pub.get_subscription_count() == 0:
            if time.time() > deadline:
                return False, 'no subscriber on the arm trajectory topic'
            time.sleep(0.1)

        msg = JointTrajectory()
        msg.joint_names = list(scene.ARM_JOINT_NAMES)
        point = JointTrajectoryPoint()
        point.positions = list(scene.READY_JOINTS)
        point.time_from_start = Duration(sec=RESCUE_MOVE_S)
        msg.points = [point]
        self._traj_pub.publish(msg)

        # Verified, not assumed. Reporting success here without checking would
        # hand the planner a false belief that the arm is clear of the table,
        # and every subsequent decision would be made against it.
        deadline = time.time() + RESCUE_MOVE_S + 8.0
        while time.time() < deadline:
            if self._at_ready():
                return True, ''
            time.sleep(0.25)
        return False, 'the arm did not reach ready'

    def _at_ready(self) -> bool:
        """Is the arm at `ready`, within RESCUE_ARRIVED_TOL_RAD on every joint?"""
        with self._joint_lock:
            pos = dict(self._arm_joints)
        if len(pos) < len(scene.ARM_JOINT_NAMES):
            return False
        return all(abs(pos[n] - want) <= RESCUE_ARRIVED_TOL_RAD
                   for n, want in zip(scene.ARM_JOINT_NAMES, scene.READY_JOINTS))

    def _execute_home(self, goal_handle):
        result = Home.Result()
        fb = Home.Feedback()
        fb.stage = 'moving to ready'
        goal_handle.publish_feedback(fb)

        # `ready` is defined for panda_arm in config/panda.srdf:20, alongside
        # `extended` and `transport`.
        self.arm.set_start_state_to_current_state()
        self.arm.set_goal_state(configuration_name='ready')
        ok, why = self._plan_and_execute(self.arm, 'move to ready')

        if ok:
            result.success = True
            result.message = 'at the ready pose, table is clear of the arm'
            goal_handle.succeed()
            return result

        # MoveIt refused. Rather than reporting a failure the planner has no
        # move against, try the one recovery that works on a wedged arm.
        fb.stage = 'planning refused - recovering through the controller'
        goal_handle.publish_feedback(fb)
        self.get_logger().warning(
            f'home: {why}; falling back to a direct trajectory')

        rescued, detail = self._rescue_to_ready()
        result.success = rescued
        if rescued:
            result.message = (
                'at the ready pose, table is clear of the arm. Note: motion '
                'planning refused this move, so the arm was driven back '
                'directly without collision checking - something on the table '
                'may have been nudged, so re-check positions before relying on '
                'them.')
            goal_handle.succeed()
        else:
            result.message = f'{why}; direct recovery also failed: {detail}'
            goal_handle.abort()
        return result


def main():
    rclpy.init(args=None)
    node = SkillServer()
    # Multi-threaded so a skill can block for the duration of a motion while the
    # subscriptions it depends on keep running.
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

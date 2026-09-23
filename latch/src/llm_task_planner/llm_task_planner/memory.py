"""Persistent memory: what the robot knows, as opposed to what it can see.

--------------------------------------------------------------------------
THE PROBLEM THIS FIXES
--------------------------------------------------------------------------
Before this module the robot had no memory of any kind, and two consequences
followed that are easy to miss because each layer looked correct on its own:

1. /world_state is a snapshot of what the camera can see RIGHT NOW. Put a cube
   in the red bin and it drops out of the world state. The robot could not
   answer "where did you put the yellow cube?" - not because it answered
   wrongly, but because by then nothing in the system had ever recorded it.
   world_model retires an unseen track after ID_MAX_MISSES frames, roughly
   three seconds, and the object ceases to exist.

2. run_command() called backend.begin(), which starts a new conversation. Every
   command therefore began amnesiac. "Now put the other one in the blue bin"
   had no referent, and "put it back" was unanswerable.

A robot that forgets what it just did cannot be given a sequence of
instructions, and cannot be asked about its own actions. That is a hard ceiling
on what the language layer is worth, no matter how good the grounding is.

--------------------------------------------------------------------------
THE DESIGN: BELIEF, NOT CACHE
--------------------------------------------------------------------------
This is not a cache of recent detections. It is a belief about where things
are, and it has two sources that are trusted differently:

    observation   the camera saw object X at (x, y). Refreshes pose. Absence
                  of observation is weak evidence - the arm occludes the table
                  constantly, and a missing detection usually means the gripper
                  is in the way, not that the object is gone.

    action        the robot placed the object it was holding into red_bin.
                  This is strong evidence and it OUTRANKS observation: the bin
                  is deep enough that a cube inside it may not be detected at
                  all, and "I put it there" beats "I cannot see it".

Getting that ordering right is the whole trick. Trusting observation over
action would make every successful place look like the object had vanished.

--------------------------------------------------------------------------
STABLE IDs - why memory has its own
--------------------------------------------------------------------------
world_model's ids are stable only while an object stays continuously visible.
It matches nearest-neighbour against the previous frame and retires a track
after ~3 s unseen; a re-appearing object gets a fresh, higher id, and after a
world_model restart the numbering begins again at 1. That is correct behaviour
for a perception component and useless as a long-term name.

So memory keeps its own ids and maps them to whatever world_model is currently
calling each object. The planner shows memory ids, and translates back at the
moment it sends a goal. An object that is remembered but not currently visible
therefore has an id the model can still refer to, and pick() on it fails with
"believed to be in red_bin" rather than "no object with id 4" - which is the
difference between a robot that lost something and a robot that knows where it
put it.

Re-association after a gap is by class, colour and position. Objects believed
to be in a bin are deliberately excluded from positional matching: a new cube
appearing where a bin-bound one used to sit is a NEW cube, not the old one
teleporting back. Inferences like that are what a belief buys over a cache.

--------------------------------------------------------------------------
No ROS in here, so it can be self-tested offline against a scripted sequence
of observations and actions - see _self_test().
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence

#: How far an object may appear to have moved and still be recognised as the
#: same one after a gap in observation. Larger than world_model's 0.06 frame-to
#: -frame gate on purpose: this bridges seconds, not frames, and an object that
#: was nudged by a passing gripper should keep its identity.
REASSOCIATE_RADIUS = 0.10

#: How long an object may go unobserved before the render stops calling it
#: "probably occluded" and starts calling it unaccounted for. The arm blocks
#: the camera for a few seconds at a time during a pick, so anything shorter
#: reports a healthy pick as a lost object.
OCCLUSION_GRACE_S = 20.0

#: Snapshots an object must appear in before memory will commit to it.
#:
#: THE BUG THIS FIXES. Without it, memory believed anything it was told once.
#: A single live session accumulated 43 objects on a table holding 9: the
#: detector fires spuriously on the robot arm as it sweeps through frame, and
#: every one of those one-frame phantoms became a permanent belief that was
#: rendered into the prompt forever after. It also blew the token budget - the
#: request grew to 9799 tokens against Groq's 8000/min limit and killed the run.
#:
#: One sighting is a detection. Several is an object. That distinction is the
#: difference between a belief store and a cache of everything ever seen, and
#: the right place for it is here rather than in perception: world_model should
#: keep reporting what it sees, and memory should decide what to believe.
#:
#: 3 at the 5 Hz publish rate is under a second, so a real object is committed
#: almost immediately while a phantom that survives one or two frames is not.
MIN_SIGHTINGS = 3

#: How long an uncommitted sighting is kept before being discarded. Short: it
#: only has to bridge a dropped frame or two.
PROVISIONAL_TTL_S = 4.0

#: A committed table object that goes this long unseen, and that the robot has
#: never touched, is dropped. Phantoms that happen to survive MIN_SIGHTINGS
#: would otherwise accumulate over a long session with nothing to remove them.
#:
#: Objects the robot has ACTED on are exempt and kept forever - "I put it in the
#: red bin" is not something to forget because the camera cannot see it. That
#: exemption is the whole point of the module, so it outranks this cleanup.
FORGET_UNSEEN_S = 180.0

#: Cap on rows in the rendered table. The world state is re-sent after every
#: tool call, so an unbounded table is an unbounded per-turn cost. Omissions are
#: stated rather than silent - a model that cannot see an object will not use
#: it, and it should know that is why.
MAX_RENDERED = 24

#: Episodes kept in the prompt. The full log stays on disk and is reachable
#: through the recall tool; this is only what rides along every turn.
EPISODES_IN_PROMPT = 4

#: Per-object history lines kept. Bounded so a long session cannot grow the
#: memory file without limit.
HISTORY_PER_OBJECT = 12

ON_TABLE = 'table'
HELD = 'held'
GONE = 'gone'


def location_phrase(location: str) -> str:
    """A belief's location as a phrase that reads inside a sentence.

    The constants above are enum values, not English. Interpolating one
    directly produced the message

        "object 8 (red cube) is not visible right now - it is table"

    which is the sort of thing a reader forgives and a language model does not:
    it is the planner's own explanation of why an action was refused, and it has
    to be read and acted on.

    Deliberately NOT the same strings as the `where` column in
    render_objects(). Those are a terse table cell, they are documented verbatim
    in prompts.py, and the model is told what each one means - changing them
    would silently desync the prompt from the data. This is the sentence form of
    the same fact, and the two registers are allowed to differ.
    """
    if location == HELD:
        return 'held by the gripper'
    if location == GONE:
        return 'unaccounted for - it was dropped or removed'
    if location == ON_TABLE:
        return 'on the table'
    return f'in the {location}'


@dataclass
class Observation:
    """One detection from /world_state, decoupled from the ROS message."""
    world_id: int
    class_name: str
    color: str
    x: float
    y: float
    yaw: float = 0.0
    grasp_yaw: float = 0.0
    long_axis: float = 0.0
    short_axis: float = 0.0
    graspable: bool = True

    @classmethod
    def from_msg(cls, obj) -> 'Observation':
        return cls(world_id=int(obj.id), class_name=str(obj.class_name),
                   color=str(obj.color), x=float(obj.x), y=float(obj.y),
                   yaw=float(obj.yaw), grasp_yaw=float(obj.grasp_yaw),
                   long_axis=float(obj.long_axis),
                   short_axis=float(obj.short_axis),
                   graspable=bool(obj.graspable))


@dataclass
class Belief:
    """What memory holds about one object.

    `location` is either ON_TABLE, HELD, GONE, or a receptacle name such as
    'red_bin'. Storing the bin name directly rather than a flag means a third
    bin needs no change here.
    """
    mem_id: int
    class_name: str
    color: str
    x: float
    y: float
    grasp_yaw: float = 0.0
    long_axis: float = 0.0
    short_axis: float = 0.0
    graspable: bool = True
    location: str = ON_TABLE
    world_id: Optional[int] = None      # None when not currently visible
    first_seen: float = 0.0
    last_seen: float = 0.0
    times_seen: int = 0
    #: True once the robot has picked or placed this. Such an object is real by
    #: demonstration - the fingers closed on it - so it is committed regardless
    #: of how many frames it was seen in, and it is never cleaned up.
    acted_on: bool = False
    history: List[str] = field(default_factory=list)

    def note(self, when: float, text: str) -> None:
        self.history.append(f'{when:.0f}|{text}')
        del self.history[:-HISTORY_PER_OBJECT]

    @property
    def committed(self) -> bool:
        """Does memory actually believe in this object?

        Anything not committed is a sighting, not an object: it is kept out of
        the rendered table, out of the saved file, and out of reach of the
        planner, and it is discarded quickly if it does not recur.
        """
        return self.acted_on or self.times_seen >= MIN_SIGHTINGS

    @property
    def label(self) -> str:
        colour = '' if self.color in ('unknown', '') else f'{self.color} '
        return f'{colour}{self.class_name}'


class RobotMemory:
    """Object beliefs plus an episodic log, persisted to disk.

    `now` is injectable so the self-test can run a scripted clock instead of
    sleeping through OCCLUSION_GRACE_S.
    """

    def __init__(self, path: Optional[Path] = None,
                 now: Callable[[], float] = time.time):
        self._path = Path(path) if path else None
        self._now = now
        self._beliefs: Dict[int, Belief] = {}
        self._next_id = 1
        self._held: Optional[int] = None
        self._episodes: List[dict] = []
        self._current: Optional[dict] = None
        if self._path and self._path.exists():
            self.load()

    # ------------------------------------------------------------- properties

    @property
    def held(self) -> Optional[int]:
        return self._held

    def belief(self, mem_id: int) -> Optional[Belief]:
        return self._beliefs.get(mem_id)

    def world_id_for(self, mem_id: int) -> Optional[int]:
        """The id world_model is currently using, or None if not visible.

        The planner calls this immediately before sending a goal. Returning
        None is a useful answer, not a failure: it means the object is
        remembered but not currently perceivable, and the caller can say so.
        """
        b = self._beliefs.get(mem_id)
        return b.world_id if b else None

    def visible(self) -> List[Belief]:
        return [b for b in self._beliefs.values()
                if b.world_id is not None and b.committed]

    def all_beliefs(self, include_provisional: bool = False) -> List[Belief]:
        """Committed beliefs by default - what the robot will say it knows."""
        return sorted((b for b in self._beliefs.values()
                       if include_provisional or b.committed),
                      key=lambda b: b.mem_id)

    # ------------------------------------------------------------ observation

    def observe(self, observations: Sequence[Observation]) -> None:
        """Fold one /world_state snapshot into the beliefs."""
        t = self._now()
        claimed: set = set()

        for obs in observations:
            mem_id = self._match(obs, claimed)
            if mem_id is None:
                mem_id = self._create(obs, t)
            else:
                self._refresh(self._beliefs[mem_id], obs, t)
            claimed.add(mem_id)

        # Anything not in this snapshot loses its world id. It does NOT lose its
        # belief: the object is still believed to be wherever it was, and the
        # render decides how to describe an absence based on how long it has
        # lasted. This is the object-permanence property - "I cannot see it" and
        # "it is not there" are different claims.
        for b in self._beliefs.values():
            if b.mem_id not in claimed:
                b.world_id = None

        self._prune(t)

    def _prune(self, t: float) -> None:
        """Forget what was never worth believing.

        Two rules, and the ordering between them matters:

          * an uncommitted sighting that has not recurred is dropped quickly. It
            was one or two frames of something, most likely the detector firing
            on the robot arm as it swept past.

          * a committed table object that has gone a long time unseen and that
            the robot has never touched is dropped too, because a phantom that
            happened to persist for a few frames would otherwise stay forever.

        Anything held, in a bin, or previously acted on is exempt from both. The
        whole purpose of this module is that "I put it in the red bin" survives
        the camera losing sight of it, so no cleanup rule may override that.
        """
        doomed = []
        for b in self._beliefs.values():
            if b.world_id is not None or b.acted_on or b.location != ON_TABLE:
                continue
            unseen = t - b.last_seen
            if not b.committed and unseen > PROVISIONAL_TTL_S:
                doomed.append(b.mem_id)
            elif b.committed and unseen > FORGET_UNSEEN_S:
                doomed.append(b.mem_id)
        for mem_id in doomed:
            del self._beliefs[mem_id]

    def _match(self, obs: Observation, claimed: set) -> Optional[int]:
        """Which remembered object is this detection, if any."""
        # Fast path: world_model is still calling it the same thing, and it is
        # the same class. Correct for the whole life of one world_model process.
        for b in self._beliefs.values():
            if (b.mem_id not in claimed and b.world_id == obs.world_id
                    and b.class_name == obs.class_name
                    and b.location in (ON_TABLE, HELD)):
                return b.mem_id

        # Slow path: re-association after a gap, by class + colour + position.
        # Beliefs held or in a bin are excluded - see the module docstring on
        # why a detection where a bin-bound object used to be is a new object.
        best, best_d = None, REASSOCIATE_RADIUS
        for b in self._beliefs.values():
            if b.mem_id in claimed or b.location != ON_TABLE:
                continue
            if b.class_name != obs.class_name:
                continue
            if (b.color != obs.color and 'unknown' not in (b.color, obs.color)):
                continue
            d = math.hypot(b.x - obs.x, b.y - obs.y)
            if d < best_d:
                best, best_d = b.mem_id, d
        return best

    def _create(self, obs: Observation, t: float) -> int:
        mem_id = self._next_id
        self._next_id += 1
        b = Belief(mem_id=mem_id, class_name=obs.class_name, color=obs.color,
                   x=obs.x, y=obs.y, grasp_yaw=obs.grasp_yaw,
                   long_axis=obs.long_axis, short_axis=obs.short_axis,
                   graspable=obs.graspable, location=ON_TABLE,
                   world_id=obs.world_id, first_seen=t, last_seen=t,
                   times_seen=1)
        b.note(t, f'first seen at ({obs.x:+.3f}, {obs.y:+.3f})')
        self._beliefs[mem_id] = b
        return mem_id

    def _refresh(self, b: Belief, obs: Observation, t: float) -> None:
        b.world_id = obs.world_id
        b.last_seen = t
        b.times_seen += 1
        b.grasp_yaw = obs.grasp_yaw
        b.long_axis = obs.long_axis
        b.short_axis = obs.short_axis
        b.graspable = obs.graspable
        # Colour is a property of the object, not of this frame - keep the first
        # confident answer, matching world_model's rule so the two agree.
        if b.color in ('unknown', '') and obs.color not in ('unknown', ''):
            b.color = obs.color
        # An object seen on the table is on the table, whatever was believed -
        # this is how a hand-placed or newly spawned object rejoins the world.
        if b.location not in (ON_TABLE, HELD):
            b.note(t, f'seen back on the table (was {b.location})')
            b.location = ON_TABLE
        if b.location == ON_TABLE:
            b.x, b.y = obs.x, obs.y

    # ----------------------------------------------------------------- actions

    def note_pick(self, mem_id: int, ok: bool, detail: str = '') -> None:
        t = self._now()
        b = self._beliefs.get(mem_id)
        if b is None:
            return
        if ok:
            b.location = HELD
            b.acted_on = True        # the fingers closed on it; it is real
            self._held = mem_id
            b.note(t, f'picked up from ({b.x:+.3f}, {b.y:+.3f})')
        else:
            b.note(t, f'pick failed: {detail[:80]}')
        self._log_action('pick', {'object': mem_id}, ok, detail)

    def note_place(self, target: str, ok: bool, detail: str = '',
                   x: Optional[float] = None, y: Optional[float] = None) -> None:
        """Record where the held object ended up.

        Action evidence outranks observation here: a cube at the bottom of a bin
        may never be detected again, and without this the successful place would
        read as the object disappearing.
        """
        t = self._now()
        mem_id = self._held
        self._log_action('place', {'target': target, 'x': x, 'y': y}, ok, detail)
        if mem_id is None:
            return
        b = self._beliefs.get(mem_id)
        if b is None:
            self._held = None
            return
        if ok:
            b.location = ON_TABLE if target in ('', None) else target
            b.acted_on = True
            if x is not None and y is not None:
                b.x, b.y = x, y
            where = (f'({x:+.3f}, {y:+.3f})' if target in ('', None)
                     else f'in {target}')
            b.note(t, f'placed {where}')
            self._held = None
        else:
            b.note(t, f'place failed: {detail[:80]}')

    def note_dropped(self, detail: str = 'grasp lost') -> None:
        """The held object is no longer held and we do not know where it is."""
        if self._held is None:
            return
        b = self._beliefs.get(self._held)
        if b is not None:
            b.location = GONE
            b.note(self._now(), detail)
        self._held = None

    def forget(self, mem_id: int) -> None:
        self._beliefs.pop(mem_id, None)
        if self._held == mem_id:
            self._held = None

    def forget_all(self) -> None:
        """Start clean. Use when the world itself is reset or respawned.

        Episode history is cleared too, and that is the whole point rather than
        a detail. Beliefs and history are two records of the same past, and a
        world reset invalidates both: "I put the red cube in the red bin" is
        exactly as wrong as a belief pointing at the bin, once the cube has
        been moved back onto the table.

        Clearing only the beliefs produced a robot that could see a red cube in
        front of it and still refuse to pick it up, explaining that it had
        already put that cube away. The affected reasoning is *worse* than a
        stale belief, because history is prose the model trusts over what it
        can currently see.
        """
        self._beliefs.clear()
        self._episodes.clear()
        self._current = None
        self._held = None
        self._next_id = 1

    # ---------------------------------------------------------------- episodes

    def begin_command(self, text: str) -> None:
        self._current = {'command': text, 'started': self._now(),
                         'actions': [], 'success': None, 'summary': ''}

    def _log_action(self, tool: str, args: dict, ok: bool, detail: str) -> None:
        if self._current is not None:
            self._current['actions'].append(
                {'tool': tool, 'args': args, 'ok': ok, 'detail': detail[:160]})

    def end_command(self, success: bool, summary: str) -> None:
        if self._current is None:
            return
        self._current.update(success=success, summary=summary,
                             ended=self._now())
        self._episodes.append(self._current)
        self._current = None
        self.save()

    # ---------------------------------------------------------------- renderng

    def _age_phrase(self, seconds: float) -> str:
        if seconds < 90:
            return f'{seconds:.0f}s ago'
        if seconds < 5400:
            return f'{seconds / 60:.0f} min ago'
        return f'{seconds / 3600:.1f} h ago'

    def render_objects(self, header: str = 'World state',
                       affordances: Optional[Dict[int, str]] = None) -> str:
        """The object table the planner puts in front of the model.

        Observed and remembered objects in ONE table, with a `where` column, so
        the model never has to reconcile two lists. An object in a bin is still
        listed - it is still a thing the robot knows about and can be asked
        about - it simply is not pickable, and the column says so.

        `affordances` maps a memory id to a short verdict for the `can pick`
        column. Passed in already rendered so this module stays ignorant of how
        feasibility is computed - memory records where things are, affordance.py
        decides what can be done to them, and neither imports the other.
        """
        beliefs = self.all_beliefs()
        if not beliefs:
            return (f'{header}: nothing detected and nothing remembered. The arm '
                    f'may be occluding the camera - call home to move it clear - '
                    f'or the table is empty.')

        t = self._now()
        afford = affordances or {}

        # Bounded, and if something has to go it is the least useful row. Held
        # and binned objects are the ones the model cannot re-derive by looking,
        # so they stay; a stale table sighting is the cheapest thing to drop.
        def priority(b: Belief) -> tuple:
            rank = (0 if b.location == HELD else
                    1 if b.location not in (ON_TABLE, GONE) else
                    2 if b.world_id is not None else 3)
            return (rank, -b.last_seen)

        omitted = 0
        if len(beliefs) > MAX_RENDERED:
            keep = sorted(beliefs, key=priority)[:MAX_RENDERED]
            omitted = len(beliefs) - len(keep)
            beliefs = sorted(keep, key=lambda b: b.mem_id)

        col = '  can pick' if afford else ''
        lines = [f'{header}:',
                 '  id  class     colour   x       y       width   where' + col]
        for b in beliefs:
            if b.location == HELD:
                where = 'HELD by the gripper'
            elif b.location == GONE:
                where = 'unaccounted for'
            elif b.location != ON_TABLE:
                where = f'in {b.location}'
            elif b.world_id is not None:
                where = 'on table, visible'
            elif t - b.last_seen < OCCLUSION_GRACE_S:
                where = 'on table, not visible now (arm probably in the way)'
            else:
                where = (f'on table? last seen {self._age_phrase(t - b.last_seen)}'
                         f', not detected since')
            verdict = f'  {afford[b.mem_id]}' if b.mem_id in afford else ''
            lines.append(
                f'  {b.mem_id:<3} {b.class_name:<9} {b.color:<8} '
                f'{b.x:+.3f}  {b.y:+.3f}  {b.short_axis * 1000:4.0f}mm   '
                f'{where:<48}{verdict}'.rstrip())
        if omitted:
            lines.append(f'  ... and {omitted} more remembered objects not shown '
                         f'(least recently seen). Use recall if you need them.')
        return '\n'.join(lines)

    def render_history(self) -> str:
        """A compact account of earlier commands, for continuity across turns.

        Deliberately short. This rides along on every request and there is no
        prompt caching on some backends, so a full transcript here would be
        re-billed every turn - and on Groq's free tier that is a rate limit,
        not just a cost.
        """
        if not self._episodes:
            return ''
        t = self._now()
        lines = ['Earlier in this session:']
        for ep in self._episodes[-EPISODES_IN_PROMPT:]:
            mark = 'done' if ep.get('success') else 'incomplete'
            when = self._age_phrase(t - ep.get('ended', ep['started']))
            lines.append(f'  [{when}, {mark}] "{ep["command"]}" -> '
                         f'{ep.get("summary", "") or "no summary"}')
        return '\n'.join(lines)

    def recall(self, mem_id: Optional[int] = None) -> str:
        """Full detail on demand, for the recall tool."""
        if mem_id is not None:
            b = self._beliefs.get(mem_id)
            if b is None:
                known = ', '.join(str(x.mem_id) for x in self.all_beliefs())
                return f'nothing remembered about id {mem_id}. Known ids: {known or "none"}'
            t = self._now()
            lines = [f'Object {b.mem_id}: {b.label}, currently '
                     f'{b.location if b.location != ON_TABLE else "on the table"} '
                     f'at ({b.x:+.3f}, {b.y:+.3f}).',
                     f'  first seen {self._age_phrase(t - b.first_seen)}, '
                     f'seen in {b.times_seen} snapshots.']
            for entry in b.history:
                when, _, text = entry.partition('|')
                lines.append(f'  - {self._age_phrase(t - float(when))}: {text}')
            return '\n'.join(lines)

        if not self._episodes:
            return 'no commands completed yet in this session.'
        t = self._now()
        lines = ['Everything done so far:']
        for ep in self._episodes:
            mark = 'done' if ep.get('success') else 'incomplete'
            lines.append(f'  [{self._age_phrase(t - ep.get("ended", ep["started"]))}'
                         f', {mark}] "{ep["command"]}": {ep.get("summary", "")}')
            for a in ep['actions']:
                lines.append(f'      {"ok  " if a["ok"] else "FAIL"} '
                             f'{a["tool"]}({a["args"]})')
        return '\n'.join(lines)

    # ------------------------------------------------------------- persistence

    def save(self) -> None:
        if self._path is None:
            return
        payload = {
            'version': 1,
            'saved': self._now(),
            'next_id': self._next_id,
            'held': self._held,
            # Committed only. An uncommitted sighting is not something the robot
            # believes, and writing it to disk would let it survive a restart as
            # though it were - the one path by which a rejected phantom could
            # still become permanent.
            'beliefs': [asdict(b) for b in self._beliefs.values() if b.committed],
            'episodes': self._episodes,
        }
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix('.tmp')
            tmp.write_text(json.dumps(payload, indent=1), encoding='utf-8')
            tmp.replace(self._path)      # atomic: a crash mid-write keeps the old file
        except OSError:
            pass                          # memory is best-effort, never fatal

    def load(self) -> None:
        try:
            payload = json.loads(self._path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            return
        if payload.get('version') != 1:
            return
        self._beliefs = {}
        for raw in payload.get('beliefs', []):
            raw.setdefault('history', [])
            try:
                b = Belief(**raw)
            except TypeError:
                continue                  # schema drifted; drop that entry
            # Nothing is visible until the camera says so again. Restoring a
            # world_id from disk would map a memory id onto whatever object
            # world_model happens to be calling that number now, which is very
            # likely a different object entirely.
            b.world_id = None
            self._beliefs[b.mem_id] = b
        self._next_id = max([payload.get('next_id', 1)]
                            + [b + 1 for b in self._beliefs])
        self._held = payload.get('held')
        self._episodes = payload.get('episodes', [])


# --------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------

def _self_test() -> int:
    import tempfile

    failures: List[str] = []
    clock = {'t': 1000.0}

    def check(label, got, want):
        ok = got == want
        print(f"  {'ok  ' if ok else 'FAIL'} {label:<54} {got!r}")
        if not ok:
            failures.append(f'{label}: got {got!r}, wanted {want!r}')

    # location_phrase: these strings go into a sentence the model has to read
    # and act on, so a bare enum value ("it is table") is a real defect.
    check('phrase: table', location_phrase(ON_TABLE), 'on the table')
    check('phrase: held', location_phrase(HELD), 'held by the gripper')
    check('phrase: bin', location_phrase('red_bin'), 'in the red_bin')
    for loc in (ON_TABLE, HELD, GONE, 'red_bin'):
        check(f'phrase reads as English for {loc!r}',
              location_phrase(loc) == loc, False)

    tmpdir = Path(tempfile.mkdtemp())
    mem = RobotMemory(tmpdir / 'memory.json', now=lambda: clock['t'])

    def obs(wid, cls, colour, x, y):
        return Observation(world_id=wid, class_name=cls, color=colour,
                           x=x, y=y, short_axis=0.04, long_axis=0.04)

    def scene(*rows):
        """Feed the same scene MIN_SIGHTINGS times so it becomes committed."""
        for _ in range(MIN_SIGHTINGS):
            clock['t'] += 0.2
            mem.observe(list(rows))

    print('a single sighting is NOT an object')
    mem.observe([obs(1, 'cube', 'yellow', 0.45, 0.20)])
    check('one frame commits nothing', len(mem.all_beliefs()), 0)
    check('but it is provisionally tracked',
          len(mem.all_beliefs(include_provisional=True)), 1)

    print('\na phantom that does not recur is discarded')
    clock['t'] += PROVISIONAL_TTL_S + 1.0
    mem.observe([])
    check('nothing committed', len(mem.all_beliefs()), 0)
    check('and nothing left provisionally either',
          len(mem.all_beliefs(include_provisional=True)), 0)

    print('\nrepeated sightings do build beliefs')
    scene(obs(1, 'cube', 'yellow', 0.45, 0.20),
          obs(2, 'cube', 'yellow', 0.25, 0.42),
          obs(3, 'cube', 'red', 0.10, 0.48))
    check('three objects remembered', len(mem.all_beliefs()), 3)
    # Ids are read back rather than assumed: the rejected phantom above consumed
    # id 1 and is never reissued, which is correct - a reused id would let a
    # stale reference silently point at a different object.
    YELLOW_A, YELLOW_B, RED = [b.mem_id for b in mem.all_beliefs()]
    print(f'    ids issued: {YELLOW_A}, {YELLOW_B}, {RED} '
          f'(1 was consumed by the rejected phantom and is not reused)')

    print('\nstable identity across an observation gap')
    clock['t'] += 1.0
    # world_model retired the tracks and renumbered - the exact thing that makes
    # its ids useless as long-term names.
    mem.observe([obs(11, 'cube', 'yellow', 0.451, 0.201),
                 obs(12, 'cube', 'yellow', 0.25, 0.42),
                 obs(13, 'cube', 'red', 0.10, 0.48)])
    check('renumbering did not create new objects', len(mem.all_beliefs()), 3)
    check(f'memory id {YELLOW_A} now maps to world id 11',
          mem.world_id_for(YELLOW_A), 11)

    print('\nobject permanence: pick then place into a bin')
    clock['t'] += 1.0
    mem.begin_command('put the two yellow cubes in the red bin')
    mem.note_pick(YELLOW_A, True, 'Grasp confirmed, finger gap 33.6 mm')
    check('held is tracked', mem.held, YELLOW_A)
    # Held object vanishes from perception - the gripper is around it.
    mem.observe([obs(12, 'cube', 'yellow', 0.25, 0.42),
                 obs(13, 'cube', 'red', 0.10, 0.48)])
    check('held object survives disappearing',
          mem.belief(YELLOW_A).location, HELD)

    clock['t'] += 1.0
    mem.note_place('red_bin', True, 'released at red_bin')
    check('placed object is located in the bin',
          mem.belief(YELLOW_A).location, 'red_bin')
    check('nothing held after a place', mem.held, None)

    # THE case this module exists for: the camera never sees it again.
    clock['t'] += 60.0
    mem.observe([obs(12, 'cube', 'yellow', 0.25, 0.42),
                 obs(13, 'cube', 'red', 0.10, 0.48)])
    check('still remembered a minute later',
          mem.belief(YELLOW_A).location, 'red_bin')
    rendered = mem.render_objects()
    if 'in red_bin' not in rendered:
        failures.append('render does not say the object is in the bin:\n' + rendered)
    check('it is not offered as visible', mem.world_id_for(YELLOW_A), None)

    print('\na NEW cube where the old one sat is not the old one')
    clock['t'] += 1.0
    scene(obs(20, 'cube', 'yellow', 0.45, 0.20),      # old spot of YELLOW_A
          obs(12, 'cube', 'yellow', 0.25, 0.42),
          obs(13, 'cube', 'red', 0.10, 0.48))
    check('a fourth object was created', len(mem.all_beliefs()), 4)
    check('the binned one stayed in the bin',
          mem.belief(YELLOW_A).location, 'red_bin')

    print('\nthe live failure: 43 phantoms on a table holding 9')
    # Replay what actually happened - the detector firing once each on a
    # scattering of arm pixels while three real objects stay put.
    import random as _r
    rng = _r.Random(4)
    for _ in range(40):
        clock['t'] += 0.2
        ghost = obs(900, 'cylinder', 'grey',
                    round(rng.uniform(-0.02, 0.35), 3),
                    round(rng.uniform(-0.25, 0.20), 3))
        mem.observe([obs(20, 'cube', 'yellow', 0.45, 0.20),
                     obs(12, 'cube', 'yellow', 0.25, 0.42),
                     obs(13, 'cube', 'red', 0.10, 0.48), ghost])
    clock['t'] += PROVISIONAL_TTL_S + 1.0
    mem.observe([obs(20, 'cube', 'yellow', 0.45, 0.20),
                 obs(12, 'cube', 'yellow', 0.25, 0.42),
                 obs(13, 'cube', 'red', 0.10, 0.48)])
    n = len(mem.all_beliefs())
    survivors = n - 4
    print(f'    40 phantom sightings + 3 real objects + 1 in a bin -> '
          f'{n} beliefs ({survivors} phantom(s) still believed)')
    # Not zero, and pretending otherwise would be dishonest: phantoms cluster
    # where the arm is, so a few land within REASSOCIATE_RADIUS of each other,
    # merge into one track, and reach MIN_SIGHTINGS legitimately. What matters
    # is that 40 sightings do not become 40 objects, and that whatever does get
    # through is not permanent.
    if survivors > 3:
        failures.append(f'{survivors} phantoms committed out of 40 sightings; '
                        f'corroboration is not filtering effectively')
    check('the binned object is still there',
          mem.belief(YELLOW_A).location, 'red_bin')

    clock['t'] += FORGET_UNSEEN_S + 1.0
    mem.observe([obs(20, 'cube', 'yellow', 0.45, 0.20),
                 obs(12, 'cube', 'yellow', 0.25, 0.42),
                 obs(13, 'cube', 'red', 0.10, 0.48)])
    after = len(mem.all_beliefs())
    print(f'    ...and {FORGET_UNSEEN_S:.0f}s later: {after} beliefs')
    check('a phantom that stops recurring is forgotten', after, 4)
    check('the binned object is NOT forgotten with it',
          mem.belief(YELLOW_A).location, 'red_bin')

    print('\nocclusion is not disappearance')
    clock['t'] += 1.0
    mem.observe([])                      # arm across the camera
    r = mem.render_objects()
    print('    ' + r.replace('\n', '\n    '))
    if 'arm probably in the way' not in r:
        failures.append('a brief absence should read as occlusion:\n' + r)
    clock['t'] += OCCLUSION_GRACE_S + 1.0
    mem.observe([])
    r = mem.render_objects()
    print(f'    ...and {OCCLUSION_GRACE_S:.0f}s later:')
    print('    ' + r.replace('\n', '\n    '))
    if 'not detected since' not in r:
        failures.append('a long absence should stop claiming occlusion:\n' + r)

    print('\nepisodic memory survives a restart')
    mem.end_command(True, 'Put one yellow cube in the red bin.')
    clock['t'] += 5.0
    reloaded = RobotMemory(tmpdir / 'memory.json', now=lambda: clock['t'])
    check('beliefs reloaded', len(reloaded.all_beliefs()), 4)
    check('bin contents reloaded',
          reloaded.belief(YELLOW_A).location, 'red_bin')
    check('nothing is visible straight after a load',
          reloaded.visible(), [])
    check('ids do not restart', reloaded._next_id, mem._next_id)
    if 'yellow cube in the red bin' not in reloaded.render_history():
        failures.append('episode log did not survive the reload:\n'
                        + reloaded.render_history())

    print('\nrecall')
    detail = reloaded.recall(YELLOW_A)
    for want in ('picked up', 'placed in red_bin'):
        if want not in detail:
            failures.append(f'recall({YELLOW_A}) is missing {want!r}:\n{detail}')
    print('    ' + detail.replace('\n', '\n    '))

    print()
    if failures:
        print(f'FAILED ({len(failures)}):')
        for f in failures:
            print(f'  - {f}')
        return 1
    print('all memory self-tests passed')
    return 0


if __name__ == '__main__':
    raise SystemExit(_self_test())

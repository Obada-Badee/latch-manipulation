"""Talking to a running Gazebo: batch teleports and lighting.

Transport: one long-lived node, not the CLI
------------------------------------------
This started out shelling out to ``gz service``. That works, and it is much
easier to debug, but it does not survive a long run - and the way it fails is
worth recording because it looks nothing like its cause.

Every ``gz service`` invocation creates a gz-transport node, sends one request,
and exits. The server then tries to reply to a peer that no longer exists, logs
``NodeShared::RecvSrvRequest() error sending response: Host unreachable``, and
keeps the dead peer in its discovery state. Do that a few hundred times and the
churn compounds: a 2400-frame run measured **2.7 fps at frame 25 falling to
1.1 fps at frame 800**, still dropping, with the Gazebo log filling with that
message. Nothing looked wrong with the camera (46.8 Hz throughout) or with
memory; only the service path degraded.

One persistent ``gz.transport13.Node``, reused for every request, removes the
churn entirely: **4.7 ms per request, 212 requests/s**, measured against the
already-degraded system.

The CLI path is kept as a fallback for environments without the Python bindings,
and because it remains the easiest thing to reach for when debugging by hand.

Other findings from probing Harmonic 8.11:
  * ``set_pose_vector`` exists and works on static models - every object's pose
    in one request. Per-object calls would be 29 requests per frame.
  * ``light_config`` exists, so the sun can be varied.
  * ``gz model -p`` does *not* work against a server started with ``-s``
    (SceneBroadcaster times out waiting for state). This does not matter here
    and the reason is structural: every object in the capture world is static, so
    the pose we command *is* the pose. A world with dynamic objects would need a
    real ground-truth source - see read_poses.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from geometry import quaternion_from_rpy, rpy_from_quaternion

#: (model_name, (x, y, z), (roll, pitch, yaw)) in the world frame.
Placement = Tuple[str, Sequence[float], Sequence[float]]


class GzError(RuntimeError):
    pass


#: Attempts per service call. Replies occasionally time out under load; that is a
#: transient, not a capability problem, and it must not end a 15-minute job.
CALL_ATTEMPTS = 4
CALL_BACKOFF_S = 0.4
CALL_TIMEOUT_MS = 8000


def _run(cmd: List[str], timeout: float = 30.0) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


class GzClient:
    """Thin wrapper over one Gazebo world's pose and light services."""

    def __init__(self, world: str, prefer_transport: bool = True):
        self.world = world
        self._node = None
        self._msgs = None
        self._batch_ok: Optional[bool] = None

        if prefer_transport:
            self._try_transport()
        if self._node is None and shutil.which("gz") is None:
            raise GzError(
                "neither the gz.transport13 Python bindings nor the `gz` CLI are "
                "available - source /opt/ros/jazzy/setup.bash")

    def _try_transport(self) -> None:
        try:
            from gz.msgs10.boolean_pb2 import Boolean
            from gz.msgs10.entity_pb2 import Entity
            from gz.msgs10.entity_factory_pb2 import EntityFactory
            from gz.msgs10.light_pb2 import Light
            from gz.msgs10.pose_pb2 import Pose
            from gz.msgs10.pose_v_pb2 import Pose_V
            from gz.transport13 import Node
        except ImportError:
            return
        self._node = Node()
        self._msgs = {"Boolean": Boolean, "Light": Light,
                      "Pose": Pose, "Pose_V": Pose_V,
                      "Entity": Entity, "EntityFactory": EntityFactory}

    @property
    def backend(self) -> str:
        return "gz.transport13" if self._node is not None else "gz CLI"

    # ------------------------------------------------------------------ waiting

    def services(self) -> List[str]:
        if self._node is not None:
            return list(self._node.service_list())
        r = _run(["gz", "service", "-l"])
        return [l.strip() for l in r.stdout.splitlines() if l.strip()]

    def wait_until_ready(self, timeout: float = 90.0) -> None:
        """Block until this world's set_pose service is advertised.

        Necessary but not sufficient: Gazebo advertises services well before the
        render pipeline produces a first frame, so generate.py also waits for an
        actual image.
        """
        want = f"/world/{self.world}/set_pose"
        deadline = time.time() + timeout
        last: List[str] = []
        while time.time() < deadline:
            last = self.services()
            if want in last:
                self._batch_ok = f"/world/{self.world}/set_pose_vector" in last
                return
            time.sleep(1.0)
        worlds = sorted({s.split("/")[2] for s in last
                         if s.startswith("/world/") and len(s.split("/")) > 2})
        raise GzError(
            f"{want} never appeared after {timeout:.0f}s. "
            f"Worlds Gazebo is advertising: {worlds or 'none'}. "
            f"If that list is non-empty but lacks {self.world!r}, the world name "
            f"in the SDF and the --world argument disagree.")

    # ------------------------------------------------------------------- poses

    @staticmethod
    def _quat(rpy: Sequence[float]) -> Tuple[float, float, float, float]:
        return quaternion_from_rpy(*rpy)

    def _fill_pose(self, msg, name: str, xyz: Sequence[float],
                   rpy: Sequence[float]) -> None:
        qx, qy, qz, qw = self._quat(rpy)
        msg.name = name
        msg.position.x, msg.position.y, msg.position.z = (
            float(xyz[0]), float(xyz[1]), float(xyz[2]))
        msg.orientation.x, msg.orientation.y = qx, qy
        msg.orientation.z, msg.orientation.w = qz, qw

    def _pose_fields_text(self, name: str, xyz: Sequence[float],
                          rpy: Sequence[float]) -> str:
        """The CLI's protobuf text format, for the fallback path."""
        qx, qy, qz, qw = self._quat(rpy)
        return (f'name: "{name}" '
                f'position {{x: {xyz[0]:.6f} y: {xyz[1]:.6f} z: {xyz[2]:.6f}}} '
                f'orientation {{x: {qx:.9f} y: {qy:.9f} z: {qz:.9f} w: {qw:.9f}}}')

    def _request(self, service: str, req, reqtype_name: str, what: str) -> None:
        """One transport request, retried. Raises after CALL_ATTEMPTS failures.

        Retrying matters because a stale pose is a *wrong* label, not a missing
        one: the frame would be rendered showing the previous scene and paired
        with the new ground truth. Failing loudly is the only safe alternative to
        succeeding.
        """
        reqtype = self._msgs[reqtype_name]
        boolean = self._msgs["Boolean"]
        last = "no reply"
        for attempt in range(CALL_ATTEMPTS):
            try:
                ok, rep = self._node.request(service, req, reqtype, boolean,
                                             CALL_TIMEOUT_MS)
                if ok and rep.data:
                    return
                last = f"ok={ok} data={getattr(rep, 'data', None)}"
            except Exception as e:                       # noqa: BLE001
                last = f"{type(e).__name__}: {e}"
            if attempt < CALL_ATTEMPTS - 1:
                time.sleep(CALL_BACKOFF_S * (attempt + 1))
        raise GzError(f"{what} failed after {CALL_ATTEMPTS} attempts: {last}")

    def _request_cli(self, service: str, reqtype: str, req: str, what: str) -> None:
        last = ""
        for attempt in range(CALL_ATTEMPTS):
            try:
                r = _run(["gz", "service", "-s", service,
                          "--reqtype", reqtype, "--reptype", "gz.msgs.Boolean",
                          "--timeout", str(CALL_TIMEOUT_MS), "--req", req])
                if "data: true" in r.stdout:
                    return
                last = (r.stdout + r.stderr).strip()[:200]
            except subprocess.TimeoutExpired as e:
                last = f"the gz CLI did not return within {e.timeout}s"
            if attempt < CALL_ATTEMPTS - 1:
                time.sleep(CALL_BACKOFF_S * (attempt + 1))
        raise GzError(f"{what} failed after {CALL_ATTEMPTS} attempts: {last}")

    def set_poses(self, placements: Iterable[Placement]) -> None:
        """Move every named model, in one request where possible."""
        placements = list(placements)
        if not placements:
            return

        # Whether batching is *available* is a property of the world, decided once
        # from the service list. It is deliberately never inferred from a call's
        # outcome: a transient timeout used to be read as "the batch service is
        # unsupported", which permanently downgraded the run to 29 requests per
        # frame - slower, and far more likely to time out again. One hiccup then
        # cascaded into a dead job at frame 375.
        if self._batch_ok is None:
            self._batch_ok = f"/world/{self.world}/set_pose_vector" in self.services()

        if self._node is not None:
            if self._batch_ok:
                req = self._msgs["Pose_V"]()
                for n, p, r in placements:
                    self._fill_pose(req.pose.add(), n, p, r)
                self._request(f"/world/{self.world}/set_pose_vector", req, "Pose_V",
                              f"set_pose_vector for {len(placements)} models")
                return
            for n, p, r in placements:
                msg = self._msgs["Pose"]()
                self._fill_pose(msg, n, p, r)
                self._request(f"/world/{self.world}/set_pose", msg, "Pose",
                              f"set_pose for {n}")
            return

        # CLI fallback
        if self._batch_ok:
            req = " ".join(f"pose {{ {self._pose_fields_text(n, p, r)} }}"
                           for n, p, r in placements)
            self._request_cli(f"/world/{self.world}/set_pose_vector",
                              "gz.msgs.Pose_V", req,
                              f"set_pose_vector for {len(placements)} models")
            return
        for n, p, r in placements:
            self._request_cli(f"/world/{self.world}/set_pose", "gz.msgs.Pose",
                              self._pose_fields_text(n, p, r), f"set_pose for {n}")

    # ------------------------------------------------------- spawn and despawn
    #
    # set_poses() teleports models that already exist, which is right for the
    # dataset generator: a fixed pool, moved in and out of frame, never changing
    # size. A world the robot is *working* in needs more than that - objects
    # have to be able to arrive and leave, so that neither the perception stack
    # nor the planner's memory can be quietly relying on a fixed cast.
    #
    # These are separate methods rather than an option on set_poses because they
    # fail differently and are worth reading differently: a failed teleport is a
    # transient, a failed spawn usually means the SDF is wrong.

    def spawn(self, name: str, sdf: str, xyz: Sequence[float],
              rpy: Sequence[float], allow_renaming: bool = False) -> None:
        """Create a model in the running world from an SDF string.

        `allow_renaming=False` on purpose: silently becoming `cube_0_1` because
        `cube_0` already existed would give the world two objects the caller
        believes are one, and every later remove() would hit the wrong one.
        Better to fail and say the name is taken.
        """
        service = f"/world/{self.world}/create"
        if self._node is not None:
            req = self._msgs["EntityFactory"]()
            req.sdf = sdf
            req.name = name
            req.allow_renaming = allow_renaming
            qx, qy, qz, qw = self._quat(rpy)
            req.pose.position.x = float(xyz[0])
            req.pose.position.y = float(xyz[1])
            req.pose.position.z = float(xyz[2])
            req.pose.orientation.x, req.pose.orientation.y = qx, qy
            req.pose.orientation.z, req.pose.orientation.w = qz, qw
            self._request(service, req, "EntityFactory", f"spawn {name}")
            return

        # The CLI path needs the SDF inline in protobuf text format, so any
        # double quote in it would terminate the string. Single-quote the
        # attributes instead of escaping - shorter, and it survives nesting.
        inline = sdf.replace('"', "'").replace("\n", " ")
        qx, qy, qz, qw = self._quat(rpy)
        req = (f'sdf: "{inline}" name: "{name}" '
               f'allow_renaming: {str(allow_renaming).lower()} '
               f'pose {{ position {{x: {xyz[0]:.6f} y: {xyz[1]:.6f} '
               f'z: {xyz[2]:.6f}}} orientation {{x: {qx:.9f} y: {qy:.9f} '
               f'z: {qz:.9f} w: {qw:.9f}}} }}')
        self._request_cli(service, "gz.msgs.EntityFactory", req, f"spawn {name}")

    def remove(self, name: str) -> None:
        """Delete a model from the running world."""
        service = f"/world/{self.world}/remove"
        if self._node is not None:
            req = self._msgs["Entity"]()
            req.name = name
            req.type = self._msgs["Entity"].MODEL
            self._request(service, req, "Entity", f"remove {name}")
            return
        self._request_cli(service, "gz.msgs.Entity",
                          f'name: "{name}" type: MODEL', f"remove {name}")

    # ---------------------------------------------------------------- lighting

    def set_sun(self, direction: Sequence[float], diffuse: float = 0.8) -> bool:
        """Point the directional light somewhere else. Best-effort.

        Returns False on failure and the caller carries on: the world has shadows
        disabled (matching the live cell), so lighting variation changes shading
        gradients but nothing structural. Worth a little robustness, not worth
        aborting a run over.
        """
        try:
            if self._node is not None:
                light = self._msgs["Light"]()
                light.name = "sun"
                light.type = light.DIRECTIONAL
                light.direction.x = float(direction[0])
                light.direction.y = float(direction[1])
                light.direction.z = float(direction[2])
                light.diffuse.r = light.diffuse.g = light.diffuse.b = float(diffuse)
                light.diffuse.a = 1.0
                light.specular.r = light.specular.g = light.specular.b = 0.2
                light.specular.a = 1.0
                ok, rep = self._node.request(
                    f"/world/{self.world}/light_config", light,
                    self._msgs["Light"], self._msgs["Boolean"], 3000)
                return bool(ok and rep.data)

            req = (f'name: "sun", type: DIRECTIONAL, '
                   f'direction: {{x: {direction[0]:.4f} y: {direction[1]:.4f} '
                   f'z: {direction[2]:.4f}}}, '
                   f'diffuse: {{r: {diffuse:.3f} g: {diffuse:.3f} '
                   f'b: {diffuse:.3f} a: 1.0}}')
            r = _run(["gz", "service", "-s", f"/world/{self.world}/light_config",
                      "--reqtype", "gz.msgs.Light", "--reptype", "gz.msgs.Boolean",
                      "--timeout", "3000", "--req", req])
            return "data: true" in r.stdout
        except Exception:                                # noqa: BLE001
            return False

    # ------------------------------------------------------------ ground truth

    def read_poses(self) -> Dict[str, Tuple[float, float, float]]:
        """Read back actual entity poses from /world/<w>/pose/info.

        Present because running against the *live* cell - which has dynamic
        objects that settle under physics - needs a real ground-truth source.

        TWO TRAPS in this text format, both of which this got wrong before:

        1. `orientation` has x/y/z keys too. A parser that assigns any `z:` it
           sees to the position ends up storing the quaternion instead, and the
           failure is quiet and plausible-looking: cube_yellow_1, authored at
           yaw 0.35, was read as being at z = 0.174, which is exactly
           sin(0.35/2). Block tracking, not key names, is what separates them.

        2. Protobuf text omits zero-valued fields. So an entity at x = 0 has no
           `x:` line at all, and a parser that commits an entry when it sees the
           last coordinate will never commit that one. The entry is therefore
           committed on the closing brace, with the components defaulted to 0.

        Returns EVERY entity, links and nested models included, because the
        message does not say which is which. Poses of links are relative to
        their parent model, so a caller wanting world-frame objects has to
        filter - see scene_manager.SceneManager.occupied_link0 for how, and why
        that filter is a plausibility test rather than a name test.
        """
        r = _run(["gz", "topic", "-e", "-t", f"/world/{self.world}/pose/info",
                  "-n", "1"], timeout=15.0)
        return self._parse_pose_v(r.stdout)

    def read_poses_full(self) -> Dict[str, Tuple[Tuple[float, float, float],
                                                  Tuple[float, float, float]]]:
        """Poses AND orientations, as (xyz, rpy) in the same rpy convention the
        rest of this package uses (extrinsic XYZ, matching SDF).

        read_poses is enough to know where something is; labelling a frame needs
        to know how it is turned as well. Objects in the live cell settle under
        physics, so unlike the static capture world their orientation is an
        output, not something that was commanded.
        """
        r = _run(["gz", "topic", "-e", "-t", f"/world/{self.world}/pose/info",
                  "-n", "1"], timeout=15.0)
        return self._parse_pose_v(r.stdout, with_orientation=True)

    @staticmethod
    def _parse_pose_v(text: str, with_orientation: bool = False):
        """Parse a gz.msgs.Pose_V in protobuf text format. Pure, so testable."""
        poses: Dict[str, Any] = {}
        stack: List[str] = []
        name: Optional[str] = None
        xyz = [0.0, 0.0, 0.0]
        # Identity, so an omitted orientation block reads as "not rotated"
        # rather than as the all-zero quaternion, which is not a rotation.
        quat = [0.0, 0.0, 0.0, 1.0]

        for line in text.splitlines():
            s = line.strip()
            if not s:
                continue
            if s.endswith("{"):
                block = s[:-1].strip()
                stack.append(block)
                if block == "pose" and len(stack) == 1:
                    name = None
                    xyz = [0.0, 0.0, 0.0]
                    quat = [0.0, 0.0, 0.0, 1.0]
                continue
            if s == "}":
                closed = stack.pop() if stack else ""
                if closed == "pose" and not stack and name:
                    position = (xyz[0], xyz[1], xyz[2])
                    if with_orientation:
                        poses[name] = (position, rpy_from_quaternion(*quat))
                    else:
                        poses[name] = position
                continue
            if s.startswith("name:") and stack[-1:] == ["pose"]:
                name = s.split('"')[1] if '"' in s else s.split(":", 1)[1].strip()
                continue

            key, _, value = s.partition(":")
            key = key.strip()
            if stack[-1:] == ["position"]:
                idx = {"x": 0, "y": 1, "z": 2}.get(key)
                target, slot = xyz, idx
            elif stack[-1:] == ["orientation"]:
                idx = {"x": 0, "y": 1, "z": 2, "w": 3}.get(key)
                target, slot = quat, idx
            else:
                continue
            if slot is not None:
                try:
                    target[slot] = float(value)
                except ValueError:
                    pass
        return poses

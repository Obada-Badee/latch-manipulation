"""The object pool the capture world spawns, and the SDF that describes it.

Design decisions worth knowing before editing
---------------------------------------------
1. **Class is shape, never colour.** Three classes: cube / cylinder / bolt.
   Colour is recovered downstream in world_model from HSV pixels, so adding a
   colour would multiply the class count by four for no gain in what the
   planner can express. CLASS_NAMES order *is* the YOLO class-id mapping and is
   written into the dataset yaml; reordering it silently invalidates existing
   weights.

2. **Materials are copied from worlds/arm_on_the_table.sdf, not invented.**
   The detector is trained and deployed on the same renderer, so the training
   images should show the same materials the live cell does. The values below
   are duplicated from that file by necessity - SDF has no include mechanism
   for materials - and consistency is asserted in the self-test.

3. **The pool is spawned once and teleported per frame.** Colour and geometry
   are baked into a model at spawn time, so varying them means having an
   instance of each variant standing by. Restarting Gazebo per frame would
   cost ~8 s against ~0.4 s for a teleport: 4 hours versus 15 minutes for a
   2000-image run. Objects not wanted in a frame are parked off-camera.

4. **Objects are static.** Nothing here needs to fall or settle, and static
   models make the commanded pose the exact ground-truth pose - removing both
   the settling wait and a whole class of label error. resting_z() puts each
   one down on the table precisely, so they look physically plausible anyway.

5. **The bins are included as unlabelled background.** They are large red and
   blue boxes - the same hues as the red and blue cubes - so a detector that
   never saw them would be free to call them cubes. They appear in every real
   frame, so they must appear in training frames too.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

from geometry import box_vertices, cylinder_vertices, resting_z, stl_vertices

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
BOLT_STL = os.path.join(REPO_ROOT, "src", "panda_moveit_config", "worlds", "bolt", "meshes", "bolt.stl")

#: YOLO class ids are the indices of this tuple. Do not reorder.
CLASS_NAMES: Tuple[str, ...] = ("cube", "cylinder", "bolt")

TABLE_TOP_WORLD_Z = 1.015

# --------------------------------------------------------------------------
# Materials, copied from worlds/arm_on_the_table.sdf
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Material:
    name: str
    ambient: Tuple[float, float, float]
    diffuse: Tuple[float, float, float]

    def sdf(self) -> str:
        a, d = self.ambient, self.diffuse
        return (f"<material>"
                f"<ambient>{a[0]} {a[1]} {a[2]} 1</ambient>"
                f"<diffuse>{d[0]} {d[1]} {d[2]} 1</diffuse>"
                f"<specular>0.1 0.1 0.1 1</specular>"
                f"</material>")


MATERIALS: Dict[str, Material] = {
    "yellow": Material("yellow", (0.35, 0.30, 0.04), (0.90, 0.80, 0.10)),
    "red":    Material("red",    (0.32, 0.04, 0.04), (0.85, 0.10, 0.10)),
    "blue":   Material("blue",   (0.04, 0.07, 0.32), (0.10, 0.20, 0.85)),
    "green":  Material("green",  (0.04, 0.26, 0.08), (0.10, 0.70, 0.20)),
}

#: Bins reuse the world's slightly different bin diffuse values.
BIN_MATERIALS = {
    "red":  Material("red",  (0.32, 0.04, 0.04), (0.80, 0.12, 0.12)),
    "blue": Material("blue", (0.04, 0.07, 0.32), (0.12, 0.22, 0.80)),
}

CUBE_SIZE = 0.04
CYLINDER_RADIUS = 0.02
CYLINDER_LENGTH = 0.06


# --------------------------------------------------------------------------
# Classes
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ObjectClass:
    """Shape, geometry, and the set of orientations it can rest in."""
    name: str
    vertices: np.ndarray
    #: Draw a physically plausible resting orientation (roll, pitch, yaw).
    sample_rpy: Callable[[np.random.Generator], Tuple[float, float, float]]
    #: SDF fragment for <geometry>; None means the link is a mesh include.
    geometry_sdf: str = ""
    is_mesh: bool = False
    #: Set when the model's own material should be kept (the bolt's DAE is a
    #: grey metal; overriding it would make the training set disagree with the
    #: live cell, where bolts are the one class that is *not* brightly coloured).
    fixed_color: str = ""

    @property
    def class_id(self) -> int:
        return CLASS_NAMES.index(self.name)


def _cube_rpy(rng: np.random.Generator) -> Tuple[float, float, float]:
    """A cube rests on a face; only yaw varies. Full turn: a square OBB is
    invariant under pi/2, but the *shaded* faces are not, so the renderer still
    sees four distinct appearances."""
    return (0.0, 0.0, float(rng.uniform(0.0, 2.0 * math.pi)))


def _cylinder_rpy(rng: np.random.Generator) -> Tuple[float, float, float]:
    """Upright most of the time, on its side sometimes.

    Both are stable rests, and they look completely different from above - a
    circle versus a 60x40 mm rectangle. Training on only the upright case would
    leave the detector helpless the first time one rolls over, and a toppled
    cylinder is exactly the sort of state a closed loop is supposed to survive.
    """
    if rng.random() < 0.30:
        return (0.0, math.pi / 2.0, float(rng.uniform(0.0, 2.0 * math.pi)))
    return (0.0, 0.0, float(rng.uniform(0.0, 2.0 * math.pi)))


def _bolt_rpy(rng: np.random.Generator) -> Tuple[float, float, float]:
    """A bolt lies on its side: local +Z (the 116 mm axis) turned horizontal.

    pitch = +-pi/2 both lay the axis flat but swap which end the head is on,
    which matters for the silhouette - the head is 43.8 mm across against the
    shaft's 24 mm, so the OBB is noticeably lopsided.
    """
    pitch = math.pi / 2.0 if rng.random() < 0.5 else -math.pi / 2.0
    return (0.0, pitch, float(rng.uniform(0.0, 2.0 * math.pi)))


def build_classes() -> Dict[str, ObjectClass]:
    return {
        "cube": ObjectClass(
            name="cube",
            vertices=box_vertices(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE),
            sample_rpy=_cube_rpy,
            geometry_sdf=f"<box><size>{CUBE_SIZE} {CUBE_SIZE} {CUBE_SIZE}</size></box>",
        ),
        "cylinder": ObjectClass(
            name="cylinder",
            vertices=cylinder_vertices(CYLINDER_RADIUS, CYLINDER_LENGTH),
            sample_rpy=_cylinder_rpy,
            geometry_sdf=(f"<cylinder><radius>{CYLINDER_RADIUS}</radius>"
                          f"<length>{CYLINDER_LENGTH}</length></cylinder>"),
        ),
        "bolt": ObjectClass(
            name="bolt",
            vertices=stl_vertices(BOLT_STL),
            sample_rpy=_bolt_rpy,
            is_mesh=True,
            fixed_color="grey",
        ),
    }


# --------------------------------------------------------------------------
# The pool
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class PoolItem:
    """One spawned model, standing by to be teleported into or out of frame."""
    model_name: str
    cls: ObjectClass
    color: str
    #: Index in the pool, which fixes this item's parking slot. Parking slots
    #: must be distinct - see park_xyz.
    index: int = 0

    def resting_z(self, rpy: Sequence[float]) -> float:
        return resting_z(self.cls.vertices, rpy, TABLE_TOP_WORLD_Z)

    @property
    def park_xyz(self) -> Tuple[float, float, float]:
        """This item's own parking slot, on a grid well outside the frustum.

        Every object used to park at one shared coordinate, which crashed
        Gazebo: five bolts with identical mesh collision geometry ended up
        exactly coincident, and dartsim's ODE trimesh-trimesh test aborted the
        process (`dCollideTTL` -> `dDebug` -> `abort`). Distinct slots make that
        impossible regardless of what collision geometry exists, and lining the
        unused pool up in a grid is easier to inspect besides.
        """
        return (PARK_ORIGIN[0] + PARK_PITCH * (self.index % PARK_COLS),
                PARK_ORIGIN[1] + PARK_PITCH * (self.index // PARK_COLS),
                PARK_ORIGIN[2])

    def sdf(self) -> str:
        px, py, pz = self.park_xyz
        pose = f"<pose>{px} {py} {pz} 0 0 0</pose>"

        # No <collision> anywhere in the pool, deliberately. These objects are
        # static and are only ever looked at, so collision geometry buys nothing
        # and costs a great deal: it is what put mesh-mesh contact tests in the
        # physics step and crashed the server. A rendering rig should not be
        # doing collision detection.
        if self.cls.is_mesh:
            # The bolt is referenced as a bare visual mesh rather than by
            # <include>model://bolt</include>, which would drag in the model's
            # own collision mesh and nest the model one level deeper (making
            # set_pose's target name ambiguous).
            geo = ('<mesh><scale>1 1 1</scale>'
                   '<uri>model://bolt/meshes/bolt.dae</uri></mesh>')
            return (f'<model name="{self.model_name}">\n'
                    f'  <static>true</static>\n'
                    f'  {pose}\n'
                    f'  <link name="link">\n'
                    f'    <visual name="visual"><geometry>{geo}</geometry></visual>\n'
                    f'  </link>\n'
                    f'</model>')

        mat = MATERIALS[self.color].sdf()
        geo = self.cls.geometry_sdf
        return (f'<model name="{self.model_name}">\n'
                f'  <static>true</static>\n'
                f'  {pose}\n'
                f'  <link name="link">\n'
                f'    <visual name="visual"><geometry>{geo}</geometry>{mat}</visual>\n'
                f'  </link>\n'
                f'</model>')


#: Where unwanted objects go: a grid far outside the camera frustum and off the
#: table, but still in the world so nothing needs respawning between frames.
#: Verified against the camera model in the self-test rather than eyeballed.
PARK_ORIGIN = (6.0, 6.0, 1.0)
PARK_PITCH = 0.5
PARK_COLS = 8

#: How many of each variant to keep standing by. The cap on any single variant
#: is what limits scene composition: with 4 yellow cubes, a frame can never
#: contain 5. Four is enough for "put the two yellow cubes in the bin" plus
#: distractors.
POOL_COUNTS = {"cube": 4, "cylinder": 2, "bolt": 5}


def build_pool(classes: Dict[str, ObjectClass] | None = None) -> List[PoolItem]:
    classes = classes or build_classes()
    pool: List[PoolItem] = []

    def add(name: str, cls: ObjectClass, color: str) -> None:
        pool.append(PoolItem(name, cls, color, index=len(pool)))

    for cname in ("cube", "cylinder"):
        cls = classes[cname]
        for color in MATERIALS:
            for i in range(POOL_COUNTS[cname]):
                add(f"synth_{cname}_{color}_{i}", cls, color)
    bolt = classes["bolt"]
    for i in range(POOL_COUNTS["bolt"]):
        add(f"synth_bolt_{i}", bolt, "grey")
    return pool


# --------------------------------------------------------------------------
# Bins: background, never labelled
# --------------------------------------------------------------------------

def _bin_sdf(name: str, x: float, y: float, color: str) -> str:
    mat = BIN_MATERIALS[color].sdf()
    parts = [f'<model name="{name}"><static>true</static>'
             f'<pose>{x} {y} {TABLE_TOP_WORLD_Z} 0 0 0</pose><link name="link">']
    walls = [("base", 0, 0, 0.0025, 0.20, 0.20, 0.005),
             ("wall_xp", 0.095, 0, 0.030, 0.01, 0.20, 0.05),
             ("wall_xn", -0.095, 0, 0.030, 0.01, 0.20, 0.05),
             ("wall_yp", 0, 0.095, 0.030, 0.18, 0.01, 0.05),
             ("wall_yn", 0, -0.095, 0.030, 0.18, 0.01, 0.05)]
    for wn, px, py, pz, sx, sy, sz in walls:
        geo = f"<box><size>{sx} {sy} {sz}</size></box>"
        parts.append(f'<visual name="{wn}"><pose>{px} {py} {pz} 0 0 0</pose>'
                     f'<geometry>{geo}</geometry>{mat}</visual>')
    parts.append("</link></model>")
    return "".join(parts)


#: World poses copied from arm_on_the_table.sdf.
BINS = [("red_bin", 0.33, -0.26, "red"), ("blue_bin", 0.57, -0.22, "blue")]


# --------------------------------------------------------------------------
# Self-test:  python3 objects.py
# --------------------------------------------------------------------------

def _self_test() -> int:
    failures = []
    classes = build_classes()
    pool = build_pool(classes)

    # 1. Class ids must be the tuple indices, and cover every class.
    for i, n in enumerate(CLASS_NAMES):
        if classes[n].class_id != i:
            failures.append(f"class id mismatch for {n}")
    if set(classes) != set(CLASS_NAMES):
        failures.append(f"classes {set(classes)} != CLASS_NAMES {set(CLASS_NAMES)}")

    # 2. Pool composition, and unique model names (gz silently renames or
    #    rejects duplicates, which would make teleports hit the wrong object).
    print(f"  pool: {len(pool)} models")
    for cname in CLASS_NAMES:
        n = sum(1 for p in pool if p.cls.name == cname)
        print(f"    {cname:<9} {n:>2}")
    names = [p.model_name for p in pool]
    if len(set(names)) != len(names):
        failures.append("duplicate model names in the pool")
    expected = len(MATERIALS) * (POOL_COUNTS["cube"] + POOL_COUNTS["cylinder"]) \
        + POOL_COUNTS["bolt"]
    if len(pool) != expected:
        failures.append(f"pool size {len(pool)} != expected {expected}")

    # 3. Materials must match the live world's, or the detector trains on
    #    colours it will never see. Parsed out of the SDF rather than trusted.
    world = os.path.join(REPO_ROOT, "src", "panda_moveit_config", "worlds",
                         "arm_on_the_table.sdf")
    import re
    text = open(world, encoding="utf-8").read()
    found = set()
    for m in re.finditer(r"<diffuse>([\d.]+) ([\d.]+) ([\d.]+) 1</diffuse>", text):
        found.add(tuple(round(float(g), 4) for g in m.groups()))
    for name, mat in MATERIALS.items():
        if tuple(round(c, 4) for c in mat.diffuse) not in found:
            failures.append(f"{name} diffuse {mat.diffuse} is not in the world SDF")
    print(f"  {len(found)} distinct diffuse colours in the world SDF; "
          f"all {len(MATERIALS)} object materials matched")

    # 4. Resting heights must reproduce the world SDF's authored z values, which
    #    is an independent check on geometry.resting_z.
    cube_z = pool[0].resting_z((0, 0, 0.35))
    if abs(cube_z - 1.035) > 1e-9:
        failures.append(f"cube resting z {cube_z} != world SDF's 1.035")
    cyl = next(p for p in pool if p.cls.name == "cylinder")
    if abs(cyl.resting_z((0, 0, 0)) - 1.045) > 1e-9:
        failures.append("cylinder upright resting z != world SDF's 1.045")
    print(f"  cube rests at z {cube_z:.4f} (SDF says 1.035); "
          f"cylinder upright {cyl.resting_z((0, 0, 0)):.4f} (SDF says 1.045)")

    # 5. Sampled orientations must be stable rests: the object's lowest point
    #    lands exactly on the table for every draw.
    rng = np.random.default_rng(0)
    for p in pool:
        for _ in range(20):
            rpy = p.cls.sample_rpy(rng)
            z = p.resting_z(rpy)
            from geometry import transform
            v = transform(p.cls.vertices, (0, 0, z), rpy)
            if abs(v[:, 2].min() - TABLE_TOP_WORLD_Z) > 1e-6:
                failures.append(f"{p.model_name} rpy={rpy} floats or sinks "
                                f"(lowest z {v[:, 2].min():.6f})")
                break

    # 6. The parking spot must be genuinely outside the camera's view. This is
    #    load-bearing: a parked object that is still visible becomes an
    #    unlabelled positive and teaches the detector to ignore its own class.
    import sys
    sys.path.insert(0, os.path.join(REPO_ROOT, "src", "yolov8_obb"))
    from yolov8_obb.projection import CAMERA, world_to_link0
    worst = None
    for p in pool:
        v = np.asarray(p.cls.vertices) + np.array(p.park_xyz)
        for vx, vy, vz in v:
            lx, ly, lz = world_to_link0(vx, vy, vz)
            if lz >= CAMERA.origin[2]:
                continue                       # above the camera; cannot project
            u, vv = CAMERA.link0_to_pixel(lx, ly, lz)
            if CAMERA.in_frame(u, vv):
                failures.append(f"parked {p.model_name} is still in frame "
                                f"at ({u:.0f},{vv:.0f})")
            d = min(abs(u), abs(u - CAMERA.width), abs(vv), abs(vv - CAMERA.height))
            if worst is None or d < worst[0]:
                worst = (d, p.model_name, u, vv)
    print(f"  parked objects project to ({worst[2]:.0f},{worst[3]:.0f}) at closest, "
          f"{worst[0]:.0f} px outside the 640x480 frame")

    # 6b. Parking slots must be DISTINCT and non-overlapping. Sharing one slot
    #     put five identical bolt collision meshes at the same coordinate and
    #     aborted the Gazebo server inside ODE's trimesh test, so this is a
    #     regression guard for a crash that actually happened.
    slots = [p.park_xyz for p in pool]
    if len(set(slots)) != len(slots):
        failures.append(f"only {len(set(slots))} distinct parking slots for "
                        f"{len(slots)} pool objects - coincident static meshes "
                        f"crash dartsim's ODE collision detector")
    biggest = max(float(np.ptp(np.asarray(p.cls.vertices)[:, :2])) for p in pool)
    closest = min(math.dist(a[:2], b[:2])
                  for i, a in enumerate(slots) for b in slots[i + 1:])
    print(f"  parking grid: {len(set(slots))} distinct slots, closest pair "
          f"{closest * 1000:.0f} mm apart, largest object {biggest * 1000:.0f} mm wide")
    if closest <= biggest:
        failures.append(f"parking slots are {closest * 1000:.0f} mm apart but the "
                        f"largest object is {biggest * 1000:.0f} mm wide - parked "
                        f"objects would intersect")

    # 7. Every model in the capture world must be visual-only. Collision geometry
    #    on a static, never-touched object buys nothing and is what crashed the
    #    physics step; the world is a rendering rig.
    import xml.etree.ElementTree as ET
    for name, x, y, color in BINS:
        s = _bin_sdf(name, x, y, color)
        try:
            root = ET.fromstring(s)
        except ET.ParseError as e:
            failures.append(f"{name} SDF is not valid XML: {e}")
            continue
        if root.findall(".//collision"):
            failures.append(f"{name} should not declare collision geometry")
    for p in pool:
        try:
            root = ET.fromstring(p.sdf())
        except ET.ParseError as e:
            failures.append(f"{p.model_name} SDF is not valid XML: {e}")
            continue
        if root.findall(".//collision"):
            failures.append(f"{p.model_name} declares collision geometry - remove "
                            f"it, coincident/mesh collisions abort the server")
        if not root.findall(".//visual"):
            failures.append(f"{p.model_name} has no visual - it would be invisible")
        # The bolt must not pull in model://bolt, which carries its own collision.
        if any("model://bolt/meshes" not in (u.text or "")
               for u in root.findall(".//mesh/uri")):
            failures.append(f"{p.model_name} references a model:// URI that is not "
                            f"the bare mesh")

    print()
    if failures:
        print(f"FAILED ({len(failures)}):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("all object-pool self-tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())

# Synthetic OBB dataset generator

Trains the multi-class detector (`cube` / `cylinder` / `bolt`) without hand
annotation. The simulator already knows every object's exact pose and the camera
is fully characterised, so labels can be *computed* rather than drawn.

```bash
./run_generate.sh 2400 ~/synth_obb        # render + label
python3 validate.py --dataset ~/synth_obb # check, then LOOK at the overlays
python3 train.py --dataset ~/synth_obb --install
colcon build --symlink-install --packages-select yolov8_obb
```

## Why this exists

`best.pt` originally knew one class, `bolt`. The language layer needs to ground
phrases like *"the leftmost yellow cube"*, which requires the detector to see
cubes and cylinders too. Hand-annotating a few thousand oriented boxes is the
obvious cost; this avoids it entirely.

## Module map

| file | responsibility |
|---|---|
| `geometry.py` | vertex clouds, SDF's extrinsic-XYZ rotation convention, exact resting heights, Euler↔quaternion |
| `objects.py` | the 29-model pool, materials copied from the live world, parking slots |
| `capture_world.py` | generates `generated/synth_capture.sdf` |
| `labeler.py` | **the load-bearing part**: pose → perspective silhouette → min-area OBB → YOLO label |
| `gz_client.py` | batch teleport (`set_pose_vector`) and lighting, with retries |
| `generate.py` | the randomise → teleport → capture → write loop |
| `validate.py` | four checks, run before training |
| `check_holdout.py` | proves a probe set is unseen — images *and* layouts |
| `train.py` | fine-tune `yolov8n-obb`, install weights |

Every module runs its own self-test as `python3 <file>`. They are not decoration
— three real bugs were found by them and are described below.

## The one thing to understand: labels are silhouettes, not footprints

The camera is **not** overhead. It sits at link0 `(0.20, 0.60, 0.70)` while
objects spread across ~0.8 × 1.0 m of table, so an object at link0 `(0.45, 0.20)`
is viewed **34° off vertical** and shows two side faces as well as its top.

Measured: a 40 mm cube projects **1.54× larger** off-axis than directly beneath
the camera.

So a label cannot come from projecting a footprint. Every vertex of the object's
3D geometry is projected at its own true depth, the 2D convex hull is taken, and
the label is the minimum-area rotated rectangle around it — what a careful human
annotator would draw. The bolt uses its actual STL vertices (282 of them), not a
box approximation, because a shaft-plus-hex-head is not a box.

`min_area_rect` is implemented here from the rotating-calipers theorem rather
than calling `cv2.minAreaRect`, then cross-checked against OpenCV over 300
random point sets (worst relative area difference: 4.4e-07). Two independent
implementations agreeing is evidence; one implementation passing its own
assertions is not.

## Bugs this found in the rest of the project

**1. `OBJECT_TOP_Z` was the wrong plane, and it was not close.**
`projection.py` back-projected against each class's *top face*. But the OBB
encloses the whole silhouette, and because a box is centrally symmetric the
centre of that box projects from near the object's **centroid**. Fitting the best
plane over 500 randomised poses per class agrees with the centroid height to
within 2.75 mm every time — so this is derivable, not tuned.

| class | top-face plane | centroid plane | jaw clearance |
|---|---|---|---|
| cube | 20.9 mm | **1.5 mm** | 10 mm |
| cylinder | 41.8 mm | **6.3 mm** | 10 mm |
| bolt | 7.2 mm | **6.1 mm** | 18 mm |

The old assumption guaranteed a miss on cylinders and was marginal on cubes. It
was survivable for bolts — which is very probably why the original single-class
system appeared to work at all. Now `OBJECT_CENTROID_Z`.

**2. `scene.OPEN_CLEARANCE` was smaller than the perception error.** It was 8 mm
total (4 mm per side) because jaw travel cost ~2 s/mm under `bullet-featherstone`.
Under `dartsim` the jaw runs at ~112 mm/s, so that constraint is gone — and 4 mm
per side was *less* than the cylinder's 8.3 mm worst-case position error, meaning
a finger would strike the object on the way down. Now 20 mm.

**3. The bolt's grasp target is not its pose origin.** The STL's origin sits just
under the hex head, 41.5 mm from the middle of the shaft. A pick aimed at the
pose origin clamps the head instead of the shaft. `Label.grasp_target_link0()`
returns the centre of the object's extent; for cubes and cylinders the two
coincide, which is exactly why the discrepancy is easy to miss until a bolt turns
up.

## Design decisions

**Static objects.** Nothing in the capture world is dynamic, so the commanded
pose *is* the ground-truth pose. That removes the settling wait and an entire
class of label error. `resting_z()` sets each object down exactly on the table
from its own rotated vertex cloud, so they look physically plausible anyway.

**No collision geometry anywhere.** The world is a rendering rig. Collision
shapes on static, never-touched objects buy nothing — and they crashed the
server: all 29 objects originally parked at one shared coordinate, putting five
identical bolt collision meshes exactly coincident, and dartsim's ODE trimesh
test aborted the process (`dCollideTTL` → `dDebug` → `abort`). Parking slots are
now per-object and distinct, *and* the pool is visual-only. Both are asserted.

**Bins are in the world, unlabelled.** They are large red and blue boxes — the
same hues as the red and blue cubes — and they appear in every real frame. A
detector that never saw them would be free to call them cubes.

**Batch teleport.** `set_pose_vector` moves all 29 models in one call. Per-object
calls would cost 29 subprocess spawns per frame, ~75 minutes of pure overhead
across a 2400-frame run.

**Capability is never inferred from a call result.** A transient service timeout
was originally read as "the batch service is unsupported", which permanently
downgraded to per-object calls — slower and *more* likely to time out. One hiccup
killed a 15-minute run at frame 375. Availability now comes from the service
list; transient failures are retried, then the frame is skipped and counted.

**Most augmentation is off** (`mosaic`, flips, rotation, scale, translate). The
camera is bolted to the robot, so its fixed geometry is information the detector
should exploit; mosaic would produce frames with four tables, and a mirrored
frame has its perspective inverted. The usual trade — realism for diversity — is
a bad one here because diversity is free: `generate.py` can produce any number of
genuinely in-distribution frames. HSV jitter *stays on*, deliberately, so class
never becomes entangled with colour (class is shape; colour is recovered
downstream by `world_model`).

## Verified

- Capture world and live world render **pixel-identically** on table and
  background: bare table BGR `(216,216,216)` in both, table occupying
  u 266–639 / v 82–369 in both. No domain gap. (Incidentally this confirms the
  legacy `Gazebo/Wood` `<script>` material is unsupported in Harmonic — flat grey
  in both, so they agree.)
- Colour recovery, running `world_model`'s own HSV logic on real rendered pixels:
  **100% over 269 instances** across all five colours.
- Back-projection accuracy on rendered frames is within jaw clearance for every
  class.

## Known limitations

**The arm is not in the capture world.** In the live cell the robot *is* in
frame — link0 projects to about (542, 314) — so a detector trained only here has
never seen the arm and is free to invent a detection on it. Modelling the arm
here would need the xacro, `ros2_control`, and a controller stack just to hold a
pose, which is why it was left out.

The plan was to measure rather than pre-empt, and the measurement came back
positive: **the detector does invent boxes on the arm**, and on the arm's cast
shadow, classifying both as `bolt` — which is not surprising, since a bolt is a
dark elongated metallic thing and so is a shadowed forearm.
`world_model.WORKSPACE_X/Y` currently discards them and its docstring says
plainly that it is a guard rather than a cure.

### The cure: `capture_negatives.py`

Frames of the **live** cell, arm included, labelled from Gazebo ground truth.

```bash
python3 capture_negatives.py --frames 320 --out ~/synth_obb_neg
python3 merge_datasets.py --out ~/synth_obb_mixed ~/synth_obb ~/synth_obb_neg
python3 train.py --dataset ~/synth_obb_mixed --epochs 80 --install

# a probe set to score on. A DIFFERENT --seed is not optional - see below
python3 capture_negatives.py --frames 40 --seed 4242 --shuffle-every 1 \
                            --out ~/synth_obb_probe
python3 check_holdout.py --probe ~/synth_obb_probe --train ~/synth_obb_mixed
python3 score_arm_fps.py --weights <old> --dataset ~/synth_obb_probe --split train
python3 score_arm_fps.py --weights <new> --dataset ~/synth_obb_probe --split train
```

Three things about it are worth knowing:

* **Live capture, not a static arm added here.** Adding a rigid arm model to
  this world was the first idea and it is worse: a rigid model can only be moved
  as one piece, so every frame shows the same configuration somewhere else. The
  live cell moves the real arm through real configurations with real
  self-shadowing, which appears to be what the detector is actually latching on
  to.
* **Ground truth is read, not commanded.** Live objects settle under physics, so
  `gz_client.read_poses_full()` supplies both position *and* orientation. That
  only became trustworthy once the pose parser stopped reading quaternion
  components as positions — it had no block tracking, and `orientation` carries
  x/y/z keys too.
* **The scene is reshuffled every 10 frames.** Without it, a few hundred frames
  of one fixed layout get mixed into 2400 varied ones and the detector can learn
  the layout instead of the lesson.

#### The trap: a held-out set that was not held out

The first probe set was captured with `capture_negatives.py`'s **default seed**,
and so was the training set. The layout sequence is a pure function of the seed,
so the second run regenerated the first run's table arrangements object for
object. The retrained detector scored **57 → 1** false positives on it. The
number was meaningless: it had trained on those layouts.

Hashing the images did not catch it, and could not have. The frames genuinely
differed — the arm had moved between them — so every hash was unique and the
check returned a confident, wrong "no overlap". What repeated was the *table*,
which is invisible to a pixel comparison. Comparing object layouts found 17 of
17 frames duplicated.

[`check_holdout.py`](check_holdout.py) now does both comparisons and refuses the
set if either fires; `capture_negatives.py` records its seed in
`capture_meta.json` so a shared seed is reported outright rather than inferred.

The general lesson is the one this project keeps relearning: a verification that
can only fail one way is not a verification. Ask what the check would miss
before trusting what it passes.

The correctness risk is labelling an object the arm is standing in front of,
which would teach exactly the failure being fixed, inverted. Every frame is
therefore checked with `frame_matches_labels` at a raised threshold (0.45 rather
than the default 0.25, because partial occlusion lands between the two modes)
and any frame with a doubtful label is **discarded rather than corrected**.
Measured accept rate on the live cell: ~85%, the rest rejected for occlusion.

**Train and test share the renderer.** This transfers perfectly *in simulation*
and would need real-image work to transfer to hardware. Worth stating plainly in
any write-up.

**The table is textureless** (flat 216 grey, zero variance), which makes
detection easy. Expect high mAP and do not read it as evidence the pipeline
would survive clutter or real lighting.

# LATCH — Language-Anchored Task Execution with Checked Handling

*Language-driven manipulation with real perception, real IK, and skills that
check themselves.*

A Franka Emika Panda on a tabletop in Gazebo, driven by natural language:

> *"Put the red cube in the red bin."*
> *"Where is the red cube now?"*
> *"Now put the blue cube in the same bin as the red cube."*

Every layer under that sentence is real. The objects are found by a trained
YOLOv8-OBB detector on a live camera feed, positions are back-projected into the
robot's frame, motion is planned by MoveIt with collision checking and executed
by `ros2_control`, and the grasp is *verified* by measuring the finger separation
afterwards. Nothing is a lookup table.

```bash
# terminal 1 - simulator, MoveIt, controllers, perception, skills
ros2 launch panda_moveit_config moveit_gazebo_obb.py perception:=true

# terminal 2 - the language layer
export DEEPINFRA_API_KEY=...   # or GROQ_API_KEY, or ANTHROPIC_API_KEY
ros2 run llm_task_planner llm_planner --ros-args -p backend:=deepinfra

command> put the two yellow cubes in the red bin, then move the leftmost bolt
         next to the red cube
```

---

## Architecture

```
Gazebo ──/image_raw──► YOLOv8-OBB ──/Yolov8_Inference──►  world_model
  (dynamic scene)       cube/cylinder/bolt                 │  back-project + HSV colour
        ▲                                                  │  + tracked ids
        │                                                  ▼
        │                                              /world_state   (what is VISIBLE)
        │                                                  │
        │                                          ┌───────┴──────────────────┐
        │                                          │  memory.py               │
        │                                          │  what is THERE, incl.    │
        │                                          │  what cannot be seen     │
        │                                          └───────┬──────────────────┘
        │                                                  │
        │       /check_affordances ◄───────────────────────┤  can it be done?
        │            (real IK + collision)                 │
        │                                                  ▼
        │                                          ┌──────────────────────────┐
        │                                          │  llm_planner             │
        │                                          │  tool-calling loop       │
        │                                          │ anthropic|groq|deepinfra │
        │                                          └───────┬──────────────────┘
        │                                                  │ pick / place / home / recall
        │                                                  ▼
        └───── gz_ros2_control ◄── MoveIt ◄────────  skill_server
                                    real IK, collision      grasp verified from
                                    checking, STOMP         /joint_states
```

The closed loop is not bespoke machinery: every skill result carries a freshly
perceived world state, so a failed grasp comes back as an error *plus* the scene
as it now actually is, and the next model turn plans against reality. That is
just tool use.

| package | what it owns |
|---|---|
| `robot_description` | URDF |
| `panda_moveit_config` | world SDF, MoveIt config, launch |
| `yolov8_obb` | detector node, `projection`, `world_model`, `overlay`, debug viewer |
| `yolov8_obb_msgs` (builds as `yolov8_msgs`) | `DetectedObject`, `WorldState` |
| `skill_msgs` | `Pick`/`Place`/`Home` actions, `CheckAffordances` service |
| `llm_task_planner` | planner loop, backends, prompts, tools, **memory**, **affordance**, skills |
| `tools/` | dataset generation, training, scene manager, snapshots, verification |

---

## The three layers that make it more than a pick-and-place demo

### 1. Affordance grounding — *can* it, before it tries

`/check_affordances` answers "would `pick(4)` succeed?" **without doing it**, by
solving the arm's actual IK at the grasp pose *and* the hover above it, and
checking the resulting posture against the real planning scene. Plus width,
reach, and a fingertip-level clutter test.

This is the half of [SayCan](src/llm_task_planner/SAYCAN.md) that is still worth
having. The other half — scoring English skill strings under a frozen LLM — is
obsolete, and reimplementing it would be a downgrade; that file says exactly why.

The clutter test is geometric rather than a proximity radius: the fingertips sit
at ±half the open jaw gap along `grasp_yaw`, so a neighbour *beside* the jaw is
irrelevant while the same neighbour at the same distance *along the closing axis*
is a collision. A centre-distance test cannot tell those apart, and they are the
common case on a cluttered table.

### 2. Memory — object permanence, and continuity across commands

Before this, `world_model` retired an unseen track after ~3 s, so a cube placed
in a bin **ceased to exist**, and every command started a fresh conversation with
no referent for "the other one".

[`memory.py`](src/llm_task_planner/llm_task_planner/memory.py) keeps beliefs from
two sources trusted differently: observation refreshes a pose, but **action
outranks observation**. "I placed it in red_bin" beats "I cannot see it" — get
that backwards and every successful place reads as the object vanishing.

It also keeps its own stable ids. `world_model`'s ids are stable only while an
object stays continuously visible; the planner shows memory ids and translates at
the moment it sends a goal, so `pick` on a remembered-but-invisible object fails
with *"it is in red_bin"* rather than *"no object with id 4"*.

### 3. A world that changes

[`tools/scene_manager.py`](tools/scene_manager.py) adds, removes and reshuffles
real manipulable objects in the **running** cell. Nothing downstream may assume a
fixed cast: ids must be assigned to new arrivals and retired for departures,
memory must distinguish "gone" from "hidden", and "put the two yellow cubes away"
must work when there are three.

```bash
python3 tools/scene_manager.py list
python3 tools/scene_manager.py spawn --count 4
python3 tools/scene_manager.py randomize --count 8
python3 tools/scene_manager.py remove --all      # only ever touches dyn_* models
```

---

## Verified, with numbers

All measured on the live cell against Gazebo ground truth, not against the
robot's own report.

| claim | evidence |
|---|---|
| perception is accurate | 9/9 detected, **0 mislabelled**, 1–2 mm position error on cubes and cylinders, ids stable across frames |
| the detector does not hallucinate on the arm | **0.03** false positives per frame with the arm in view, down from 3.14, while recall *rose* 98.9% → 99.6%. Measured on arm postures absent from training |
| the skill layer works end to end | cube moved 685 mm and landed **7 mm** from the bin centre (inner half-width 90 mm) |
| affordances are computed, not guessed | 9 objects scored in **1.3 s** via real IK + planning-scene collision |
| the affordance layer discriminates | an object moved to reach 0.814 m → `NO (reach)`; the other 11 stayed feasible |
| clutter is directional | two objects 42 mm apart each vetoed the other; isolated objects scored 1.0 |
| the world can change under it | 9 → 12 objects spawned, detected, scored and removed, ids assigned and retired |
| memory survives a bin | `pick(5)` → `place_in_bin(red_bin)`; memory says `red_bin`, Gazebo agrees |
| memory answers about the unseen | *"Where is the red cube now?"* → *"in the red bin"* — the cube is invisible inside it |
| memory carries across commands | *"the same bin as the red cube"* → blue cube into `red_bin`, confirmed in Gazebo |
| grounding is reliable | `openai/gpt-oss-120b` 4/4 on the target command; `llama-3.3-70b` 0/3 |
| the loop is provider-agnostic | a second and third backend added with **no change** to the loop; the third is a 60-line subclass |

---

## What is honestly still wrong

**~~The detector invents objects on the robot arm.~~ Fixed, and the fix is
measured.** It used to emit 3.14 false boxes per frame with the arm in view,
mostly classed `bolt` — a bolt is a dark elongated metallic thing and so is a
shadowed forearm. Hard negatives
([`capture_negatives.py`](tools/synth_dataset/capture_negatives.py), live frames
labelled from Gazebo ground truth) took that to **0.03 per frame with zero
inside the workspace**, while recall *rose*. Verified on arm configurations the
training set never contained, on a holdout proved unseen by
[`check_holdout.py`](tools/synth_dataset/check_holdout.py).

**Getting there required fixing the evaluation twice, and that is the more
interesting result.** Synthetic mAP read 0.99 throughout and could not see the
failure at all — the synthetic validation set contains no arm. Then the first
"held-out" probe turned out to share every object layout with training, because
both capture runs used the same default seed; hashing the images certified it
clean, because the arm had moved so the pixels genuinely differed. See
`paper/FINDINGS.md` §2.

**Train and test share a renderer.** This transfers perfectly in simulation and
would need real-image work to transfer to hardware. The table is also textureless,
which makes detection easy — high mAP here is not evidence the pipeline would
survive real clutter or lighting.

**Groq's free tier is a real constraint**: 8000 tokens/minute and 200k/day. Both
were hit during bring-up. See [BACKENDS.md](src/llm_task_planner/BACKENDS.md) —
one of them forced a design change that was worth making anyway.

**`score` is not a calibrated probability.** The hard gates (width, reach, IK,
collision) are ground truth; the clutter term is an interpolation on clearance
and nothing here was fitted to observed success rates.

---

## Reading order

* [`src/llm_task_planner/SAYCAN.md`](src/llm_task_planner/SAYCAN.md) — what is
  taken from SayCan, what is deliberately not, and why
* [`src/llm_task_planner/BACKENDS.md`](src/llm_task_planner/BACKENDS.md) —
  two providers behind one loop, and what the second one cost
* [`tools/synth_dataset/README.md`](tools/synth_dataset/README.md) — how the
  training data is made and how its labels are checked
* [`UI/README.md`](UI/README.md) — why the original clicking GUI was retired
* `paper/FINDINGS.md` — every measured result, negative
  result and bug, with the command that produced each
* `paper/EXPERIMENTS.md` — the evaluation design, sized
  to a free-tier token budget, and its threats to validity
* [`docs/screenshots/`](docs/screenshots/) — the arm in frame, casting a shadow
  across the table, with zero false detections on either

Most modules run their own self-test as `python3 <file>`:

```bash
python3 -m llm_task_planner.affordance     # IK gates, directional clutter
python3 -m llm_task_planner.memory         # object permanence, phantom rejection
python3 -m llm_task_planner.scene          # reach, bin geometry, grasp heights
python3 tools/scene_manager.py --self-test # placement rules, inertia from geometry
python3 tools/verify_skills.py             # full pick→place against Gazebo
```

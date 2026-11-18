#!/usr/bin/env bash
# Follow-up runs after the 2026-11 audit (paper/FOLLOWUP_PLAN_2026-11.md).
# One subcommand per approved run; each is resumable - run it again and it
# continues where it stopped.
#
#   nohup setsid bash tools/wsl/run_followup.sh gate                          >> ~/followup.log 2>&1 &
#   nohup setsid bash tools/wsl/run_followup.sh naive_v7                      >> ~/followup.log 2>&1 &
#   nohup setsid bash tools/wsl/run_followup.sh gate <model-id> <tag>         >> ~/followup.log 2>&1 &
#   nohup setsid bash tools/wsl/run_followup.sh second_model <model-id> <tag> >> ~/followup.log 2>&1 &
#   nohup setsid bash tools/wsl/run_followup.sh recovery                      >> ~/followup.log 2>&1 &
#
# (no set -u: /opt/ros/jazzy/setup.bash references unbound variables)
cd "$HOME/moveit2_obb"
source tools/wsl/env.sh
source install/setup.bash
source "$HOME/.llm_keys"

stage() { echo "STAGE $(date +%H:%M:%S) $*"; }

if pgrep -f "tools/benchmark.py" >/dev/null; then
  stage "ABORT: a benchmark is already running"; exit 1
fi

check_weights() {
  # The two-month wrong-detector incident (FINDINGS 2.5): never trust that
  # the weights on disk are the ones that were validated.
  local a b
  a=$(sha1sum < src/yolov8_obb/scripts/best.pt)
  b=$(sha1sum < src/yolov8_obb/scripts/best_hard_negatives.pt)
  [ "$a" = "$b" ] || { stage "ABORT: best.pt is not the hard-negative detector"; exit 1; }
  stage "detector weights: hard-negative build confirmed"
}

launch_cell() {   # $1 = verify value (true|false). Always a fresh gz sim.
  stage "restarting the cell, verify:=$1"
  python3 tools/cell_stop.py >/dev/null 2>&1; sleep 3
  rm -f "$HOME/cell.log"
  nohup setsid bash -lc "source $HOME/moveit2_obb/tools/wsl/env.sh; source $HOME/moveit2_obb/install/setup.bash; ros2 launch panda_moveit_config moveit_gazebo_obb.py perception:=true headless:=true verify:=$1" > "$HOME/cell.log" 2>&1 < /dev/null &
  sleep 90
  ros2 control set_controller_state joint_state_broadcaster active >/dev/null 2>&1
  sleep 5
  python3 tools/cell_health.py >/dev/null 2>&1
}

confirm_unverified() {
  # With verify:=false a place with nothing picked SUCCEEDS, so the guard
  # case of the regression test must FAIL. If it passes, the run would be a
  # verified one mislabelled as naive.
  if python3 tools/test_empty_place.py 2>&1 | grep -q "^FAIL place before any pick"; then
    stage "unverified mode confirmed"
  else
    stage "ABORT: verify:=false did not take"; return 1
  fi
}

reset_scene() {
  python3 - <<'PY'
import sys, os
sys.path.insert(0, os.path.expanduser("~/moveit2_obb/tools"))
sys.path.insert(0, os.path.expanduser("~/moveit2_obb/tools/synth_dataset"))
import benchmark
print("scene reset:", benchmark.World("arm_and_table", seed=0).reset(), "objects re-placed")
PY
  sleep 5
}

scored_count() {
  python3 -c "
import sys; sys.path.insert(0, 'tools')
from benchmark import scored_rows
print(len(scored_rows('$1')))" 2>/dev/null || echo 0
}

run_arm() {   # $1 name  $2 csv  $3 verify  $4.. benchmark args, without --trials
  local name=$1 csv=$2 verify=$3; shift 3
  local n have resume
  for n in 3 5; do   # two passes, a fresh gz sim each: it leaks ~1.7 GB/h
    have=$(scored_count "$csv")
    if [ "$have" -ge $((n * 15)) ]; then
      stage "$name: $have scored, pass --trials $n not needed"; continue
    fi
    resume=""; [ -f "$csv" ] && resume="--resume"
    if ! launch_cell "$verify"; then stage "ABORT: cell unhealthy before $name"; exit 1; fi
    if [ "$verify" = false ] && ! confirm_unverified; then exit 1; fi
    reset_scene
    stage "starting $name --trials $n -> $csv"
    python3 -u tools/benchmark.py --backend deepinfra "$@" --trials "$n" \
      --max-tokens 9000000 --out "$csv" $resume >> "$HOME/${name}.log" 2>&1
    stage "$name pass --trials $n finished (exit $?), $(scored_count "$csv") scored"
  done
  python3 tools/validate_run.py "$csv" 2>/dev/null | grep -vE "screen size"
  have=$(scored_count "$csv")
  if [ "$have" -lt 75 ]; then
    stage "INCOMPLETE: $name has $have/75 - run the same subcommand again to resume"
    return 1
  fi
}

case "${1:-}" in
  gate)   # optional: $2 = model id, $3 = tag. Three trials on a verified cell.
    tag=${3:-default}
    check_weights
    if ! launch_cell true; then stage "ABORT: cell unhealthy"; exit 1; fi
    reset_scene
    rm -f "runs/gate_$tag.csv"
    python3 -u tools/benchmark.py --backend deepinfra ${2:+--model "$2"} \
      --tasks s2,m1,c5 --trials 1 --condition full --out "runs/gate_$tag.csv" 2>&1 | tail -20
    python3 - "runs/gate_$tag.csv" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1], encoding="utf-8")))
if not rows:
    print("GATE FAIL: no rows written"); sys.exit(0)
print("columns end with:", list(rows[0])[-2:])
bad = [r["task"] for r in rows
       if r["outcome"] in ("error", "void") or not int(r["tool_calls"] or 0)]
for r in rows:
    print(r["task"], r["outcome"], "tool_calls=" + str(r["tool_calls"]),
          r.get("backend"), r.get("model"))
print("GATE PASS" if len(rows) == 3 and not bad else "GATE FAIL: %s" % bad)
PY
    stage "GATE COMPLETE" ;;

  naive_v7)
    check_weights
    run_arm naive_v7 runs/naive_v7.csv false \
      --no-verify --no-affordance --no-memory --condition naive
    rc=$?
    launch_cell true   # nothing else may run unverified
    stage "server reports verify = $(ros2 param get /skill_server verify 2>/dev/null | tr -d '\n')"
    [ "$rc" = 0 ] && stage "NAIVE_V7 COMPLETE" ;;

  second_model)   # $2 = DeepInfra model id, $3 = short tag used in file names
    model=${2:?model id}; tag=${3:?tag}
    check_weights
    run_arm "full_v7_$tag" "runs/full_v7_$tag.csv" true \
      --model "$model" --condition full || exit 1
    run_arm "naive_v7_$tag" "runs/naive_v7_$tag.csv" false --model "$model" \
      --no-verify --no-affordance --no-memory --condition naive
    rc=$?
    launch_cell true
    stage "server reports verify = $(ros2 param get /skill_server verify 2>/dev/null | tr -d '\n')"
    [ "$rc" = 0 ] && stage "SECOND_MODEL COMPLETE" ;;

  recovery)   # ONE cell session; blocks alternate induced / control
    check_weights
    if ! launch_cell true; then stage "ABORT: cell unhealthy"; exit 1; fi
    reset_scene
    for s in 1 2; do
      for arm in induced control; do
        flag=""; [ "$arm" = induced ] && flag="--induce"
        csv="runs/recovery_v7_${arm}_s${s}.csv"
        resume=""; [ -f "$csv" ] && resume="--resume"
        stage "recovery block: $arm seed $s -> $csv"
        python3 -u tools/benchmark.py --backend deepinfra $flag --tasks s2,s3,s5 \
          --trials 5 --seed "$s" --condition "$arm" --out "$csv" $resume \
          >> "$HOME/recovery_v7.log" 2>&1
        stage "block finished (exit $?), $(scored_count "$csv") scored"
      done
    done
    stage "RECOVERY COMPLETE" ;;

  *) echo "usage: $0 gate [model tag] | naive_v7 | second_model <model> <tag> | recovery"
     exit 2 ;;
esac

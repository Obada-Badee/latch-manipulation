#!/usr/bin/env bash
# After run_ablation_chain.sh exits: the baseline and the recovery experiment.
#
#   nohup setsid bash tools/wsl/run_post_chain.sh >> ~/chain2.log 2>&1 &
#
# Helpers are duplicated from run_ablation_chain.sh rather than factored out,
# because that script is RUNNING and bash reads a script incrementally -
# editing it mid-run is a way to execute half a line. Factor after both exit.
cd "$HOME/moveit2_obb"
source tools/wsl/env.sh
source install/setup.bash
source "$HOME/.llm_keys"

stage() { echo "STAGE $(date +%H:%M:%S) $*"; }

restart_cell() {
  stage "restarting the cell (fresh gz sim: memory leak)"
  python3 tools/cell_stop.py >/dev/null 2>&1; sleep 3
  rm -f "$HOME/cell.log"
  nohup setsid bash -lc "source $HOME/moveit2_obb/tools/wsl/env.sh; source $HOME/moveit2_obb/install/setup.bash; ros2 launch panda_moveit_config moveit_gazebo_obb.py perception:=true headless:=true" > "$HOME/cell.log" 2>&1 < /dev/null &
  sleep 90
  ros2 control set_controller_state joint_state_broadcaster active >/dev/null 2>&1
  sleep 5
  python3 tools/cell_health.py >/dev/null 2>&1
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

run_one() {   # $1 name  $2 csv  $3 want  $4.. benchmark args
  local name=$1 csv=$2 want=$3; shift 3
  local have resume=""
  have=$(scored_count "$csv")
  if [ "$have" -ge "$want" ]; then stage "$name already complete ($have) - skipping"; return 0; fi
  [ -f "$csv" ] && { resume="--resume"; stage "$name has $have scored - resuming"; }
  if ! restart_cell; then stage "ABORT: cell unhealthy before $name"; exit 1; fi
  reset_scene
  stage "starting $name -> $csv"
  python3 -u tools/benchmark.py --backend deepinfra "$@" --out "$csv" $resume >> "$HOME/${name}_v6.log" 2>&1
  stage "$name finished (exit $?)"
  python3 tools/validate_run.py "$csv" 2>/dev/null | grep -vE "screen size"
}

stage "waiting for run_ablation_chain.sh to exit"
while pgrep -f "run_ablation_chain.sh" >/dev/null; do sleep 60; done
stage "ablation chain gone; checking it completed"
if ! grep -q "CHAIN COMPLETE" "$HOME/chain.log"; then
  stage "ABORT: ablation chain did not report CHAIN COMPLETE - not starting post-chain runs"
  exit 1
fi

# Baseline: no affordances, no memory, verified skills. Design note in
# PAPER_B.md B14. The --no-verify variant is a separate decision.
run_one baseline runs/baseline_v6.csv 75 --no-affordance --no-memory --trials 5 --condition baseline

# Recovery: induced mid-carry failure on four commands without referring
# expressions (m1 is incompatible with induction - FINDINGS 1f).
run_one recovery runs/recovery_v6.csv 20 --induce --tasks s2,s3,s4,s5 --trials 5 --condition induced

stage "POST-CHAIN COMPLETE"

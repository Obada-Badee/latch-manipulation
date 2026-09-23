#!/usr/bin/env bash
# Run the two ablations back to back after the current benchmark exits.
#
#   nohup setsid bash tools/wsl/run_ablation_chain.sh > ~/chain.log 2>&1 &
#
# Between runs: check the cell, reactivate the broadcaster if the spawner race
# left it inactive, restart the cell if still unhealthy, reset the scene, and
# refuse to continue if the finished run shows a guard regression. Every stage
# prints a STAGE line so a monitor can follow it.
# (no set -u: /opt/ros/jazzy/setup.bash references unbound variables)
cd "$HOME/moveit2_obb"
source tools/wsl/env.sh
source install/setup.bash
source "$HOME/.llm_keys"

stage() { echo "STAGE $(date +%H:%M:%S) $*"; }

wait_for_benchmark() {
  while pgrep -f "tools/benchmark.py" >/dev/null; do sleep 60; done
}

ensure_healthy() {
  # ALWAYS restart, never just check. `gz sim` leaks about 1.7 GB/h on this
  # machine; at 8.6 h it held 14.4 GB of 15.8 and the OOM killer was minutes
  # away. A fresh server per 75-trial run stays under 5 GB.
  stage "restarting the cell (fresh gz sim: memory leak)"
  python3 tools/cell_stop.py >/dev/null 2>&1; sleep 3
  rm -f "$HOME/cell.log"
  nohup setsid bash -lc "source $HOME/moveit2_obb/tools/wsl/env.sh; source $HOME/moveit2_obb/install/setup.bash; ros2 launch panda_moveit_config moveit_gazebo_obb.py perception:=true headless:=true" > "$HOME/cell.log" 2>&1 < /dev/null &
  sleep 90
  ros2 control set_controller_state joint_state_broadcaster active >/dev/null 2>&1
  sleep 5
  python3 tools/cell_health.py >/dev/null 2>&1
}

guard_ok() {   # $1 = csv of the run that just finished
  local line
  line=$(python3 tools/validate_run.py "$1" 2>/dev/null | grep "GUARD")
  echo "$line"
  [[ "$line" == *"0 places succeeded with nothing picked"* ]]
}

scored_count() {   # $1 = csv
  python3 -c "
import sys; sys.path.insert(0, 'tools')
from benchmark import scored_rows
print(len(scored_rows('$1')))" 2>/dev/null || echo 0
}

run_condition() {   # $1 = condition name, $2 = benchmark flag, $3 = out csv
  local have resume=""
  have=$(scored_count "$3")
  if [ "$have" -ge 75 ]; then stage "$1 already complete ($have scored) - skipping"; return 0; fi
  if [ -f "$3" ]; then resume="--resume"; stage "$1 has $have scored - resuming"; fi
  stage "preparing $1"
  if ! ensure_healthy; then stage "ABORT: cell would not come up healthy before $1"; exit 1; fi
  python3 - <<'PY'
import sys, os
sys.path.insert(0, os.path.expanduser("~/moveit2_obb/tools"))
sys.path.insert(0, os.path.expanduser("~/moveit2_obb/tools/synth_dataset"))
import benchmark
print("scene reset:", benchmark.World("arm_and_table", seed=0).reset(), "objects re-placed")
PY
  sleep 5
  stage "starting $1 -> $3"
  python3 -u tools/benchmark.py --backend deepinfra $2 --trials 5 --condition "$1" --out "$3" $resume >> "$HOME/$1_v6.log" 2>&1
  stage "$1 finished (exit $?)"
  python3 tools/validate_run.py "$3" 2>/dev/null | grep -vE "screen size"
}

stage "waiting for the running benchmark to finish"
wait_for_benchmark
stage "full finished"
if ! guard_ok runs/full_v6.csv; then stage "ABORT: guard regression in full_v6 - not starting ablations"; exit 1; fi

run_condition no_affordance --no-affordance runs/abl_no_affordance_v6.csv
if ! guard_ok runs/abl_no_affordance_v6.csv; then stage "ABORT: guard regression in no_affordance"; exit 1; fi

run_condition no_memory --no-memory runs/abl_no_memory_v6.csv
stage "CHAIN COMPLETE"

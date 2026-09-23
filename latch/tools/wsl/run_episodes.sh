#!/usr/bin/env bash
# The memory test the single-command suite cannot perform: multi-command
# episodes that refer back to earlier commands ("the same bin as the red
# cube", "where is the red cube now?"). full vs --no-memory on EPISODES.
# Runs after run_post_chain.sh reports POST-CHAIN COMPLETE.
cd "$HOME/moveit2_obb"
source tools/wsl/env.sh
source install/setup.bash
source "$HOME/.llm_keys"
stage() { echo "STAGE $(date +%H:%M:%S) $*"; }
restart_cell() {
  stage "restarting the cell"
  python3 tools/cell_stop.py >/dev/null 2>&1; sleep 3
  rm -f "$HOME/cell.log"
  nohup setsid bash -lc "source $HOME/moveit2_obb/tools/wsl/env.sh; source $HOME/moveit2_obb/install/setup.bash; ros2 launch panda_moveit_config moveit_gazebo_obb.py perception:=true headless:=true" > "$HOME/cell.log" 2>&1 < /dev/null &
  sleep 90; ros2 control set_controller_state joint_state_broadcaster active >/dev/null 2>&1; sleep 5
  python3 tools/cell_health.py >/dev/null 2>&1
}
reset_scene() {
  python3 - <<'PY'
import sys, os
sys.path.insert(0, os.path.expanduser("~/moveit2_obb/tools")); sys.path.insert(0, os.path.expanduser("~/moveit2_obb/tools/synth_dataset"))
import benchmark; print("scene reset:", benchmark.World("arm_and_table", seed=0).reset())
PY
  sleep 5
}
run_one() {  # name csv args...
  local name=$1 csv=$2; shift 2
  local resume=""; [ -f "$csv" ] && resume="--resume"
  if ! restart_cell; then stage "ABORT: cell unhealthy before $name"; exit 1; fi
  reset_scene
  stage "starting $name -> $csv"
  python3 -u tools/benchmark.py --backend deepinfra "$@" --out "$csv" $resume >> "$HOME/${name}_v6.log" 2>&1
  stage "$name finished (exit $?)"
  python3 tools/validate_run.py "$csv" 2>/dev/null | grep -vE "screen size"
}
stage "waiting for run_post_chain.sh to exit"
while pgrep -f "run_post_chain.sh" >/dev/null; do sleep 60; done
grep -q "POST-CHAIN COMPLETE" "$HOME/chain2.log" || { stage "ABORT: post-chain did not complete"; exit 1; }
run_one episodes_full   runs/episodes_full_v6.csv      --episodes --trials 5 --condition full
run_one episodes_nomem  runs/episodes_no_memory_v6.csv --episodes --no-memory --trials 5 --condition no_memory
stage "EPISODES COMPLETE"

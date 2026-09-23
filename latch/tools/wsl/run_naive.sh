#!/usr/bin/env bash
# The naive-agent baseline: no affordances, no memory, UNVERIFIED skills.
# Waits for run_episodes.sh to report EPISODES COMPLETE, then - and only then -
# syncs and builds the verify-parameter code, because symlink-install makes a
# synced edit live at the next cell restart and the episode runs restart the
# cell between conditions.
cd "$HOME/moveit2_obb"
source tools/wsl/env.sh
source "$HOME/.llm_keys"
stage() { echo "STAGE $(date +%H:%M:%S) $*"; }

stage "waiting for run_episodes.sh to exit"
while pgrep -f "run_episodes.sh" >/dev/null; do sleep 60; done
grep -q "EPISODES COMPLETE" "$HOME/chain3.log" || { stage "ABORT: episodes did not complete"; exit 1; }

stage "syncing and building the verify parameter"
bash /mnt/w/moveit2_obb/moveit2_obb/latch/tools/wsl/sync-workspace.sh >/dev/null 2>&1
colcon build --symlink-install --packages-select llm_task_planner panda_moveit_config 2>&1 | tail -1
source install/setup.bash

launch_cell() {   # $1 = verify value
  python3 tools/cell_stop.py >/dev/null 2>&1; sleep 3
  rm -f "$HOME/cell.log"
  nohup setsid bash -lc "source $HOME/moveit2_obb/tools/wsl/env.sh; source $HOME/moveit2_obb/install/setup.bash; ros2 launch panda_moveit_config moveit_gazebo_obb.py perception:=true headless:=true verify:=$1" > "$HOME/cell.log" 2>&1 < /dev/null &
  sleep 90; ros2 control set_controller_state joint_state_broadcaster active >/dev/null 2>&1; sleep 5
  python3 tools/cell_health.py >/dev/null 2>&1
}

stage "launching the cell with verify:=false"
if ! launch_cell false; then stage "ABORT: cell unhealthy"; exit 1; fi
stage "server reports verify = $(ros2 param get /skill_server verify 2>/dev/null | tr -d '\n')"

# Confirm the mode is really active: the empty-hand regression test must now
# FAIL on its guard cases (a place with nothing picked SUCCEEDS). If it
# passes, the parameter did not take and the run would be a verified one
# mislabelled as naive.
stage "confirming unverified mode via the regression test (expect guard FAILs)"
python3 tools/test_empty_place.py 2>&1 | grep -E "^(PASS|FAIL)" | sed 's/^/    /'
if python3 tools/test_empty_place.py 2>&1 | grep -q "^FAIL place before any pick"; then
  stage "unverified mode confirmed"
else
  stage "ABORT: place with nothing picked was refused - verify:=false did not take"
  exit 1
fi

python3 - <<'PY'
import sys, os
sys.path.insert(0, os.path.expanduser("~/moveit2_obb/tools")); sys.path.insert(0, os.path.expanduser("~/moveit2_obb/tools/synth_dataset"))
import benchmark; print("scene reset:", benchmark.World("arm_and_table", seed=0).reset())
PY
sleep 5

local_resume=""; [ -f runs/naive_v6.csv ] && local_resume="--resume"
stage "starting naive -> runs/naive_v6.csv"
python3 -u tools/benchmark.py --backend deepinfra --no-verify --no-affordance --no-memory --trials 5 --condition naive --out runs/naive_v6.csv $local_resume >> "$HOME/naive_v6.log" 2>&1
stage "naive finished (exit $?)"
python3 tools/validate_run.py runs/naive_v6.csv 2>/dev/null | grep -vE "screen size"

stage "relaunching the cell with verify:=true so nothing else runs unverified"
launch_cell true
stage "server reports verify = $(ros2 param get /skill_server verify 2>/dev/null | tr -d '\n')"
stage "NAIVE COMPLETE"

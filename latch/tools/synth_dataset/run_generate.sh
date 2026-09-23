#!/usr/bin/env bash
# Bring up the capture world and generate a dataset.
#
#   ./run_generate.sh [FRAMES] [OUTDIR]
#
# Three processes are needed and the order matters:
#   1. gz sim -s   headless server; renders the camera (no GUI required - the
#                  ogre2 sensor pipeline works headless, which is what
#                  tools/wsl/40-check-render.sh established)
#   2. parameter_bridge   puts /image_raw on ROS, the same path the deployed
#                  detector reads. Reading gz's topic directly would leave the
#                  bridge untested until the live run.
#   3. generate.py teleports objects and captures frames
#
# Everything is torn down on exit, including on Ctrl-C, so a failed run does not
# leave a headless Gazebo holding the GPU.
set -u -o pipefail

FRAMES="${1:-2000}"
OUTDIR="${2:-$HOME/synth_obb}"
WS="$HOME/moveit2_obb"
TOOLS="$WS/tools/synth_dataset"

set +u; source "$WS/tools/wsl/env.sh"; set -u

# model://table and model://bolt live under the panda_moveit_config worlds dir,
# exactly as the launch file arranges for the live cell.
export GZ_SIM_RESOURCE_PATH="$WS/src/panda_moveit_config/worlds:${GZ_SIM_RESOURCE_PATH:-}"

WORLD="$TOOLS/generated/synth_capture.sdf"

echo "=== regenerating the capture world from objects.py ==="
cd "$TOOLS"
python3 capture_world.py || exit 1

PIDS=()
cleanup() {
  echo
  echo "=== tearing down ==="
  for pid in "${PIDS[@]:-}"; do
    [ -n "${pid:-}" ] && kill -9 "$pid" 2>/dev/null
  done
  # Belt and braces for anything that reparented. Never pkill -f from inside a
  # wrapper: the pattern matches the wrapper's own command line and the script
  # kills itself. Skip our own pid and our parent's explicitly.
  for pat in "gz sim -s" "parameter_bridge" "generate.py"; do
    for pid in $(pgrep -f "$pat" 2>/dev/null); do
      [ "$pid" = "$$" ] && continue
      [ "$pid" = "$PPID" ] && continue
      kill -9 "$pid" 2>/dev/null
    done
  done
}
trap cleanup EXIT INT TERM

echo
echo "=== starting headless Gazebo ==="
gz sim -s -r -v 2 "$WORLD" > /tmp/synth_gz.log 2>&1 &
PIDS+=($!)

echo "=== starting ros_gz_bridge for /image_raw ==="
ros2 run ros_gz_bridge parameter_bridge \
  /image_raw@sensor_msgs/msg/Image@gz.msgs.Image \
  > /tmp/synth_bridge.log 2>&1 &
PIDS+=($!)

# Wait for a frame to actually arrive on the ROS side. Waiting on the gz service
# alone is not enough: services are advertised long before the render pipeline
# produces its first image, and the generator would then time out on frame 0.
echo "=== waiting for /image_raw on the ROS side ==="
for i in $(seq 40); do
  if timeout 4 ros2 topic echo /image_raw --once > /dev/null 2>&1; then
    echo "  image flowing after ${i}s"
    break
  fi
  sleep 1
  if [ "$i" = 40 ]; then
    echo "  NO IMAGE after 40s. Gazebo log:"
    tail -25 /tmp/synth_gz.log | sed 's/^/    /'
    echo "  bridge log:"
    tail -15 /tmp/synth_bridge.log | sed 's/^/    /'
    exit 1
  fi
done

echo
echo "=== generating $FRAMES frames -> $OUTDIR ==="
python3 "$TOOLS/generate.py" --frames "$FRAMES" --out "$OUTDIR"
rc=$?

echo
echo "=== gazebo warnings/errors (first 15) ==="
grep -iE '\[err|\[warn' /tmp/synth_gz.log | head -15 | cut -c1-170 | sed 's/^/  /'

exit $rc

#!/usr/bin/env bash
# Build the workspace and run the WP0 baseline checks that don't need a running sim.
# Run as your normal user:  bash tools/wsl/30-build.sh
set -euo pipefail

WS="${WS:-$HOME/moveit2_obb}"
cd "$WS"

# ROS's setup files reference unset variables, so they cannot be sourced under
# `set -u`. Drop the flag around each source rather than for the whole script.
set +u
# shellcheck disable=SC1091
source /opt/ros/jazzy/setup.bash
set -u

echo "==> colcon build"
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release

set +u
# shellcheck disable=SC1091
source "$WS/install/setup.bash"
set -u

echo
echo "==> packages built"
colcon list --names-only | sed 's/^/    /'

echo
echo "==> xacro parses (robot description is well-formed)"
xacro src/panda_moveit_config/config/panda.urdf.xacro use_sim:=true > /tmp/panda_check.urdf
echo "    OK - $(wc -l < /tmp/panda_check.urdf) lines"

echo
echo "==> world SDF parses"
python3 - <<'PY'
import xml.etree.ElementTree as ET, os, sys
p = os.path.expanduser('~/moveit2_obb/src/panda_moveit_config/worlds/arm_on_the_table.sdf')
w = ET.parse(p).getroot().find('world')
models = [m.get('name') for m in w.findall('model')]
print(f"    OK - {len(models)} models: {', '.join(models)}")
PY

echo
echo "==> gazebo physics plugin present"
# Engine plugins live in gz-physics-<N>/engine-plugins/, not directly under
# gz-physics-<N>/ - a non-recursive glob here reports a false negative.
if find /usr/lib/x86_64-linux-gnu/gz-physics-* -name '*bullet-featherstone-plugin.so' \
     -print -quit 2>/dev/null | grep -q .; then
  echo "    OK - $(find /usr/lib/x86_64-linux-gnu/gz-physics-* -name 'libgz-physics-bullet-featherstone-plugin.so' -print -quit)"
else
  echo "    WARNING: bullet-featherstone plugin not found; the world launches with"
  echo "    --physics-engine gz-physics-bullet-featherstone-plugin and will fail."
fi

echo
echo "==> build complete. To bring the sim up:"
echo "      T1  ros2 launch panda_moveit_config moveit_gazebo_obb.py"
echo "      T2  ros2 launch yolov8_obb yolov8_obb.launch.py"
echo "      T3  cd ~/moveit2_obb/UI && python3 bolt_selector.py"

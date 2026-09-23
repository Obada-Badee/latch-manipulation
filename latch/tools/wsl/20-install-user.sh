#!/usr/bin/env bash
# User-level install for moveit2_obb: python deps, shell setup, workspace sync + build.
# Run as your normal user (NOT root):   bash tools/wsl/20-install-user.sh
# Idempotent - safe to re-run.
set -euo pipefail

if [ "$(id -u)" -eq 0 ]; then
  echo "ERROR: run as your normal user, not root (pip --user would land in /root)" >&2
  exit 1
fi

WS="${HOME}/moveit2_obb"

echo "==> python deps"
# Ubuntu 24.04 marks the system python as externally-managed (PEP 668).
# Installing into the *user* site (~/.local) rather than a venv is deliberate:
# `ros2 run` executes nodes with the system python3, which reads the user site
# automatically. A venv would need activating in every terminal to be visible.
#
# That convenience has a cost: the user site SHADOWS the apt-installed packages
# that ROS's compiled extensions were built against. Two pins below are load
# bearing - see the comments. Everything else may float.
PIP="pip3 install --user --break-system-packages --quiet"
$PIP --upgrade pip

# numpy MUST stay on 1.x. ROS 2 Jazzy's C extensions (cv_bridge above all, plus
# the apt scipy) are compiled against numpy 1.26 headers, and NumPy 2.0 broke
# that ABI. Letting pip pull numpy 2.x makes `from cv_bridge import CvBridge`
# die with "compiled using NumPy 1.x cannot be run in NumPy 2.x" -> KeyError: 16,
# which takes down both the detector node and the GUI.
# The reverse direction is safe: opencv-python's wheels are built against numpy
# 2 headers but run fine on 1.x, which is exactly what numpy 2's backward-
# compatible build mode was designed for.
$PIP 'numpy<2'

# colcon-core 0.21.0 declares setuptools<80. Nothing visibly breaks at 83, but
# the workspace build is the last place we want a surprise, so honour the pin.
$PIP 'setuptools<80'

# OpenCV MUST stay on 4.x, for a subtler reason than the numpy pin.
#
# ROS ships cv_bridge as a compiled extension linked against apt's OpenCV 4,
# but cv_bridge's Python half builds its type-lookup table from whatever `cv2`
# it imports. OpenCV 5 moved CV_CN_SHIFT from 3 to 5 to make room for more depth
# types, which renumbers every multi-channel type: CV_8UC3 is 16 under OpenCV 4
# and 64 under OpenCV 5. Install pip's opencv-python 5 and it shadows the system
# cv2, the table gets OpenCV 5 keys, the .so returns OpenCV 4 values, and every
# image conversion dies with a bare `KeyError: 16`.
#
# Note this is invisible to a plain `import cv2` smoke test - OpenCV 5 itself
# works perfectly on numpy 1.x. Only the cv_bridge handoff breaks. Pin, and let
# the check below prove it.
$PIP 'opencv-python<5'

# torch arrives as an ultralytics dependency; the linux x86_64 wheel is the CUDA
# build, which matters here - there's an RTX 3070 behind /dev/dxg.
# Both provider SDKs. Neither is imported until its backend is selected, so
# having only one installed still works - but installing both keeps the A/B
# between them one launch argument away.
$PIP ultralytics anthropic groq

# Re-assert both pins: the line above drags numpy 2 and opencv 5 back in as
# transitive dependencies of ultralytics. Cheap to repeat, expensive to debug.
$PIP 'numpy<2' 'opencv-python<5'

echo "==> verifying the numpy ABI actually holds"
# Fail loudly here rather than at runtime inside a ROS node, where the same
# problem surfaces as an unrelated-looking KeyError deep in a callback.
# ROS's setup.bash reads unset variables (AMENT_TRACE_SETUP_FILES and friends),
# so it cannot be sourced under `set -u`. Drop the flag across the source only.
set +u
# shellcheck disable=SC1091
source /opt/ros/jazzy/setup.bash
set -u
python3 - <<'PY' || { echo "    ABI check FAILED - do not proceed to the build"; exit 1; }
import numpy as np
import cv2
assert np.__version__.startswith("1."), f"numpy {np.__version__} is 2.x; the pin did not take"
assert cv2.__version__.startswith("4."), f"cv2 {cv2.__version__} is 5.x; the pin did not take"
# The real test: a cv_bridge round-trip. This is the exact call the detector
# node and the GUI make, and the one that fails if either pin slips.
from cv_bridge import CvBridge
b = CvBridge()
msg = b.cv2_to_imgmsg(np.zeros((4, 4, 3), dtype=np.uint8), encoding="bgr8")
assert b.imgmsg_to_cv2(msg, "bgr8").shape == (4, 4, 3)
print(f"    OK  numpy {np.__version__}, cv2 {cv2.__version__}, cv_bridge round-trips")
PY

echo "==> rosdep update"
rosdep update --rosdistro jazzy >/dev/null 2>&1 || echo "    (rosdep update failed - non-fatal)"

echo "==> shell setup"
# Everything the sim needs at runtime lives in tools/wsl/env.sh - ROS sourcing,
# PATH, and the two Mesa variables that decide whether Gazebo renders on the GPU
# or on the CPU. .bashrc just points at it, so there is one place to edit.
BASHRC="${HOME}/.bashrc"
add_line() { grep -qxF "$1" "$BASHRC" || echo "$1" >> "$BASHRC"; }
add_line '[ -f ~/moveit2_obb/tools/wsl/env.sh ] && source ~/moveit2_obb/tools/wsl/env.sh'

if [ ! -d "$WS/src" ]; then
  echo
  echo "==> workspace not found at $WS"
  echo "    run:  bash tools/wsl/sync-workspace.sh"
  echo "    then: bash tools/wsl/30-build.sh"
  exit 0
fi

echo
echo "==> user install complete. Next:"
echo "      bash tools/wsl/sync-workspace.sh   # copy W: -> ~/moveit2_obb"
echo "      bash tools/wsl/30-build.sh         # colcon build + verify"

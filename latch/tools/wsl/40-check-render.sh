#!/usr/bin/env bash
# Verify GPU-accelerated rendering works under WSLg before trusting the sim.
#
# This is the one environment risk that can't be reasoned around: Gazebo's ogre2
# renderer needs OpenGL 3.3+. Under WSL that comes from Mesa's d3d12 gallium
# driver talking to /dev/dxg. When it isn't wired up, Mesa silently falls back to
# llvmpipe (software) - the sim still "works", just at a few frames per second,
# which reads as a Gazebo problem rather than a driver one.
#
# Two independent things have to be right, and both are set in tools/wsl/env.sh:
#   GALLIUM_DRIVER=d3d12                    -> use the GPU at all
#   MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA  -> use the RIGHT GPU (not the iGPU)
#
# The check renders a real 640x480 camera at 30 Hz headlessly, reads the renderer
# string out of Gazebo's own ogre2 log, and measures what actually comes out. No
# root required - `glxinfo` would need mesa-utils installed.
#
#   bash tools/wsl/40-check-render.sh
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROBE="$HERE/render_probe.sdf"
OGRE_LOG="$HOME/.gz/rendering/ogre2.log"
FAIL=0

echo "==> /dev/dxg (GPU passthrough)"
if [ -e /dev/dxg ]; then
  echo "    present"
else
  echo "    MISSING - WSL has no GPU passthrough; nothing below will help."
  exit 1
fi

echo
echo "==> Mesa d3d12 driver installed"
if find /usr/lib/x86_64-linux-gnu/dri -name 'd3d12_dri.so' -print -quit 2>/dev/null | grep -q .; then
  echo "    d3d12_dri.so found"
else
  echo "    d3d12_dri.so NOT found - software rendering is unavoidable"
  echo "    (provided by libgl1-mesa-dri)"
  FAIL=1
fi

set +u
# shellcheck disable=SC1091
source "$HERE/env.sh"
set -u

echo
echo "==> rendering environment"
printf '    GALLIUM_DRIVER=%s\n' "${GALLIUM_DRIVER:-<unset>}"
printf '    MESA_D3D12_DEFAULT_ADAPTER_NAME=%s\n' "${MESA_D3D12_DEFAULT_ADAPTER_NAME:-<unset>}"
if [ "${GALLIUM_DRIVER:-}" != "d3d12" ]; then
  echo "    WARNING: without GALLIUM_DRIVER=d3d12 Mesa picks llvmpipe (software)."
fi

echo
echo "==> CUDA visible to torch (for YOLO training - independent of the above)"
python3 - <<'PY' 2>/dev/null || echo "    torch not installed yet"
import torch
print(f"    torch {torch.__version__}  cuda={torch.cuda.is_available()}"
      + (f"  device={torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else ""))
PY

# ---------------------------------------------------------------------------
# The decisive test.
# ---------------------------------------------------------------------------
echo
echo "==> headless ogre2 render test (this is the gate)"

command -v gz >/dev/null 2>&1 || { echo "    gz not installed - run 10-install-system.sh"; exit 1; }
[ -f "$PROBE" ] || { echo "    missing $PROBE"; exit 1; }

GZ_PID=""; BR_PID=""
cleanup() {
  [ -n "$BR_PID" ] && kill "$BR_PID" 2>/dev/null
  [ -n "$GZ_PID" ] && kill "$GZ_PID" 2>/dev/null
  # Bracket the first character so this pattern can never match the script's
  # own command line - `pkill -f 'gz sim'` will happily SIGTERM its own shell.
  pkill -f '[g]z-sim-server' 2>/dev/null
  wait 2>/dev/null
  return 0
}
trap cleanup EXIT

rm -f "$OGRE_LOG"
echo "    starting gz sim -s (server only, no window)"
gz sim -s -r -v 1 "$PROBE" >/tmp/render_probe_gz.log 2>&1 &
GZ_PID=$!

# Bridge into ROS so we can use `ros2 topic hz`, which measures the rate without
# printing ~1 MB of image per frame the way `gz topic -e` would - the echo itself
# would become the bottleneck and invalidate the measurement.
ros2 run ros_gz_bridge parameter_bridge \
  /probe_cam@sensor_msgs/msg/Image@gz.msgs.Image \
  >/tmp/render_probe_bridge.log 2>&1 &
BR_PID=$!

sleep 8   # ogre2 needs time to create its GL context; the bridge to connect

if ! kill -0 "$GZ_PID" 2>/dev/null; then
  echo "    gz sim died on startup - see /tmp/render_probe_gz.log"
  tail -20 /tmp/render_probe_gz.log | sed 's/^/      /'
  exit 1
fi

RENDERER=$(grep -m1 'GL_RENDERER' "$OGRE_LOG" 2>/dev/null | sed 's/.*GL_RENDERER = //')
echo "    GL_RENDERER: ${RENDERER:-<unknown>}"
case "$RENDERER" in
  *llvmpipe*|*softpipe*|*swrast*)
    echo "    *** SOFTWARE RENDERING - set GALLIUM_DRIVER=d3d12 ***"
    FAIL=1 ;;
  *NVIDIA*)
    echo "    hardware, discrete GPU" ;;
  *D3D12*)
    echo "    hardware, but NOT the discrete GPU."
    echo "    set MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA to use the 3070." ;;
esac

echo "    measuring for 15 s ..."
HZ=$(timeout 15 ros2 topic hz /probe_cam --window 40 2>/dev/null \
     | grep -oP 'average rate: \K[0-9.]+' | tail -1)

if [ -z "$HZ" ]; then
  echo "    NO FRAMES RECEIVED."
  echo "    gz log:";     tail -15 /tmp/render_probe_gz.log     | sed 's/^/      /'
  echo "    bridge log:"; tail -10 /tmp/render_probe_bridge.log | sed 's/^/      /'
  exit 1
fi

echo "    measured ${HZ} fps (camera requests 30)"
# Hardware rendering holds ~30. llvmpipe measured 19 fps on this single-cube
# scene, so fps alone is a weak discriminator here - the renderer string above
# is the real verdict and this is a sanity floor.
if awk "BEGIN{exit !($HZ >= 25)}"; then
  echo "    keeping up with the requested rate"
elif awk "BEGIN{exit !($HZ >= 10)}"; then
  echo "    below the requested rate - acceptable only if GL_RENDERER is hardware"
else
  echo "    *** far below the requested rate ***"
  FAIL=1
fi

echo
if [ "$FAIL" -eq 0 ]; then
  echo "==> render check PASSED"
else
  echo "==> render check FAILED - fix this before building on the sim"
fi
exit "$FAIL"

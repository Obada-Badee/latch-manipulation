# Shared environment for moveit2_obb under WSL.
# Sourced by ~/.bashrc (via 20-install-user.sh) and by the tools/wsl scripts, so
# there is exactly one place that defines how the sim is expected to run.
#
#   source ~/moveit2_obb/tools/wsl/env.sh
#
# Note: cannot be sourced under `set -u` - ROS's setup.bash reads unset vars.

# shellcheck disable=SC1091
source /opt/ros/jazzy/setup.bash
# shellcheck disable=SC1091
[ -f "$HOME/moveit2_obb/install/setup.bash" ] && source "$HOME/moveit2_obb/install/setup.bash"

export PATH="$HOME/.local/bin:$PATH"

# --------------------------------------------------------------------------
# Gazebo rendering under WSLg. BOTH variables are required.
# --------------------------------------------------------------------------
# GALLIUM_DRIVER=d3d12
#   Selects Mesa's D3D12 gallium driver, which maps OpenGL onto the Windows
#   GPU through /dev/dxg. Without it Mesa silently falls back to llvmpipe -
#   software rendering - *even though* d3d12_dri.so is installed and /dev/dxg
#   is present. This is the variable that actually turns the GPU on, and it is
#   the one most WSL guides omit. Measured on this machine: llvmpipe managed
#   19 fps on a single-cube scene; the real scene has 13 models plus the Panda.
#
# MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
#   Chooses WHICH adapter. On a hybrid laptop d3d12 enumerates the Intel iGPU
#   first, so without this you get hardware rendering on the wrong GPU.
#
#   Setting this one ALONE does nothing whatsoever - it is only consulted once
#   d3d12 is the active driver. That is worth knowing, because it makes the
#   variable look broken when the real problem is that llvmpipe is still in
#   charge and never reads it.
export GALLIUM_DRIVER=d3d12
export MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA

#!/usr/bin/env bash
# System-level install for moveit2_obb: ROS 2 Jazzy + Gazebo Harmonic on Ubuntu 24.04.
# Needs root.   Run:  sudo bash tools/wsl/10-install-system.sh
# Idempotent - safe to re-run.
set -euo pipefail

if [ "$(id -u)" -ne 0 ]; then
  echo "ERROR: run with sudo:  sudo bash $0" >&2
  exit 1
fi

. /etc/os-release
if [ "${VERSION_ID}" != "24.04" ]; then
  echo "ERROR: expected Ubuntu 24.04 (ROS 2 Jazzy), found ${VERSION_ID}" >&2
  exit 1
fi
echo "==> Ubuntu ${VERSION_ID} (${UBUNTU_CODENAME}) confirmed"

echo "==> base tools"
apt-get update -qq
apt-get install -y -qq \
  curl gnupg lsb-release software-properties-common \
  build-essential cmake git python3-pip

add-apt-repository -y universe >/dev/null

# ---- ROS 2 apt repo -------------------------------------------------------
# Using the keyring+sources.list form rather than the ros2-apt-source .deb:
# it has no GitHub API dependency, so it can't fail on rate limits.
echo "==> ROS 2 Jazzy repo"
if [ ! -f /usr/share/keyrings/ros-archive-keyring.gpg ]; then
  curl -fsSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg
fi
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
http://packages.ros.org/ros2/ubuntu ${UBUNTU_CODENAME} main" \
  > /etc/apt/sources.list.d/ros2.list

# ---- Gazebo Harmonic apt repo --------------------------------------------
# Installed explicitly (not just via ros-gz) to guarantee the physics plugins
# are present - the world launches with gz-physics-bullet-featherstone-plugin.
echo "==> Gazebo Harmonic repo"
if [ ! -f /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg ]; then
  curl -fsSL https://packages.osrfoundation.org/gazebo.gpg \
    -o /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
fi
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] \
http://packages.osrfoundation.org/gazebo/ubuntu-stable ${UBUNTU_CODENAME} main" \
  > /etc/apt/sources.list.d/gazebo-stable.list

apt-get update -qq

# ---- packages -------------------------------------------------------------
echo "==> ROS 2 Jazzy desktop (~2 GB, this is the slow one)"
apt-get install -y -qq ros-jazzy-desktop

echo "==> MoveIt 2 + Python bindings"
apt-get install -y -qq \
  ros-jazzy-moveit \
  ros-jazzy-moveit-py \
  ros-jazzy-moveit-planners \
  ros-jazzy-moveit-ros-move-group

echo "==> Gazebo Harmonic + ros_gz bridge"
apt-get install -y -qq gz-harmonic ros-jazzy-ros-gz

echo "==> ros2_control stack"
apt-get install -y -qq \
  ros-jazzy-ros2-control \
  ros-jazzy-ros2-controllers \
  ros-jazzy-gz-ros2-control \
  ros-jazzy-controller-manager \
  ros-jazzy-joint-state-broadcaster \
  ros-jazzy-joint-trajectory-controller \
  ros-jazzy-position-controllers

echo "==> misc project deps"
apt-get install -y -qq \
  ros-jazzy-xacro \
  ros-jazzy-cv-bridge \
  ros-jazzy-tf2-ros \
  ros-jazzy-rviz2 \
  python3-colcon-common-extensions \
  python3-rosdep \
  python3-pyqt5 \
  rsync

# ---- rosdep ---------------------------------------------------------------
echo "==> rosdep"
if [ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]; then
  rosdep init
fi

echo
echo "==> system install complete"
echo "    next (as your normal user, NOT root):"
echo "      bash tools/wsl/20-install-user.sh"

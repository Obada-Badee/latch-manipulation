#!/usr/bin/env bash
# Copy the workspace from the Windows drive into the WSL filesystem.
#
# Why copy instead of building in place on /mnt/w:  cross-filesystem I/O through
# drvfs is roughly an order of magnitude slower than ext4, and colcon does a lot
# of small-file I/O. Building on /mnt/w works but is painfully slow.
#
# Authoring happens on the Windows side; this pushes changes across before a build.
# Re-run it after every edit. It is incremental, so it costs ~1s once warm.
set -euo pipefail

# The repo lives in latch/ so that the paper and its reference material sit
# beside it rather than inside it. The colcon workspace root is therefore
# latch/, and that is what gets synced.
SRC="${SRC:-/mnt/w/moveit2_obb/moveit2_obb/latch}"
DST="${DST:-$HOME/moveit2_obb}"

if [ ! -d "$SRC/src" ]; then
  echo "ERROR: no ROS workspace at $SRC (expected $SRC/src)" >&2
  echo "       override with:  SRC=/mnt/<drive>/path bash $0" >&2
  exit 1
fi

mkdir -p "$DST"
echo "==> rsync $SRC -> $DST"
rsync -a --delete \
  --exclude 'build/' --exclude 'install/' --exclude 'log/' \
  --exclude '.git/' --exclude '__pycache__/' --exclude '*.pyc' \
  --exclude 'runs/' --exclude 'datasets/' \
  --exclude 'desktop.ini' --exclude 'Thumbs.db' \
  "$SRC/" "$DST/"

# Defensive: anything edited with a Windows editor can pick up CRLF, which makes
# a shebang fail with the famously unhelpful "bad interpreter: ...^M".
echo "==> normalising line endings on scripts"
find "$DST" \( -name '*.sh' -o -name '*.py' \) -type f -print0 \
  | xargs -0 -r sed -i 's/\r$//'

echo "==> marking executables"
find "$DST" -name '*.sh' -type f -exec chmod +x {} +
find "$DST/src" -path '*/scripts/*.py' -type f -exec chmod +x {} + 2>/dev/null || true
[ -f "$DST/UI/bolt_selector.py" ] && chmod +x "$DST/UI/bolt_selector.py"

echo "==> synced ($(du -sh "$DST" | cut -f1))"

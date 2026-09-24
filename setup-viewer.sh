#!/usr/bin/env bash
#
# setup-viewer.sh — fetch and build the official cabt replay viewer.
#
# The viewer is the official Kaggle `kaggle-environments` repo (a pnpm monorepo).
# It is dev-only tooling and is NOT committed to our repo — everyone sets it up
# locally with this script. See README.md.
#
# Usage:
#   ./setup-viewer.sh                 # clones into ./cabt-viewer
#   VIEWER_DIR=/some/path ./setup-viewer.sh
#
# Works on macOS, Linux, and Windows via Git Bash or WSL.

set -euo pipefail

# ---- config ------------------------------------------------------------------
# kaggle-environments publishes no git tags, so we pin a COMMIT for reproducibility.
# Bump this when you want to update the viewer, then tell your teammates to re-run.
PINNED_COMMIT="8e28b629a64c3093dca6dab5e1e0c312694a6fb2"
VIEWER_REPO="https://github.com/Kaggle/kaggle-environments.git"
VIEWER_DIR="${VIEWER_DIR:-./cabt-viewer}"            # override via env var
CABT_VIS_SUBDIR="kaggle_environments/envs/cabt/visualizer/default"
# ------------------------------------------------------------------------------

info() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m !!\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m  x\033[0m %s\n' "$*" >&2; exit 1; }

# ---- prerequisite checks -----------------------------------------------------
command -v git  >/dev/null 2>&1 || die "git not found. Install it first (see README)."
command -v node >/dev/null 2>&1 || die "Node.js not found. Install Node 18+ (see README)."

NODE_MAJOR="$(node -p 'process.versions.node.split(".")[0]')"
[ "$NODE_MAJOR" -ge 18 ] || die "Node $(node -v) is too old; this viewer needs Node 18+."

# Prefer corepack (ships with Node) so we use the repo-pinned pnpm@9.15.3.
if command -v corepack >/dev/null 2>&1; then
  info "Enabling corepack (provides the pinned pnpm version)..."
  corepack enable >/dev/null 2>&1 || warn "corepack enable failed; will use system pnpm if present."
fi
command -v pnpm >/dev/null 2>&1 || die "pnpm not found and corepack unavailable. Install pnpm (see README)."

# ---- fetch the viewer at the pinned commit (shallow) -------------------------
if [ -e "$VIEWER_DIR/.git" ]; then
  info "Viewer already present at '$VIEWER_DIR' — updating to pinned commit..."
  git -C "$VIEWER_DIR" fetch -q --depth 1 origin "$PINNED_COMMIT"
  git -C "$VIEWER_DIR" checkout -q FETCH_HEAD
else
  info "Cloning viewer into '$VIEWER_DIR' (pinned ${PINNED_COMMIT:0:10}, shallow)..."
  git init -q "$VIEWER_DIR"
  git -C "$VIEWER_DIR" remote add origin "$VIEWER_REPO"
  git -C "$VIEWER_DIR" fetch -q --depth 1 origin "$PINNED_COMMIT"
  git -C "$VIEWER_DIR" checkout -q FETCH_HEAD
fi

[ -d "$VIEWER_DIR/$CABT_VIS_SUBDIR" ] || die "Expected '$CABT_VIS_SUBDIR' not found — bad pin or layout change."

# ---- install JS deps (MUST run at the workspace root) ------------------------
info "Installing JS dependencies with pnpm (this is the slow part, grab a coffee)..."
( cd "$VIEWER_DIR" && pnpm install )

# ---- prepare the replay folder the dev server serves -------------------------
mkdir -p "$VIEWER_DIR/$CABT_VIS_SUBDIR/replays"

# Absolute path to the replay file the sim should write to.
REPLAY_ABS="$(cd "$VIEWER_DIR/$CABT_VIS_SUBDIR/replays" && pwd)/test-replay.json"

info "Done."
cat <<EOF

  Viewer installed : $VIEWER_DIR
  Replay file      : $REPLAY_ABS

  Next steps
  ----------
  1) Tell your sim where to write the replay (skip if you use the default path):
       export CABT_REPLAY_PATH="$REPLAY_ABS"     # macOS / Linux / Git Bash
       \$env:CABT_REPLAY_PATH="$REPLAY_ABS"      # Windows PowerShell

  2) Run a game (writes the replay), e.g.:  python battle_test.py

  3) Start the viewer (same command on every OS):
       cd "$VIEWER_DIR/$CABT_VIS_SUBDIR"
       pnpm dev-with-replay
     ...then open the printed URL (default http://localhost:5173).

  Loop: re-run the sim, then refresh the browser tab.
EOF

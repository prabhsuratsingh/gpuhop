#!/usr/bin/env bash
# gpuhop instance bootstrap (uploaded by the launcher, run from the project dir). Safe to re-run.
#
#   1. persist the container's environment for later ssh sessions
#   2. snapshot the base image's pip packages (`gpuhop freeze` subtracts them)
#   3. run the project's setup script, unless it and $GPUB_SETUP_HASH_FILES are unchanged since the last run
#   4. run the project's smoke command
#
# Result: $GPUB_WS/.ready exists  <=>  all of the above succeeded.
# Failure: $GPUB_WS/.bootstrap_failed names the failing step; the full log is $GPUB_WS/bootstrap.log.
set -euo pipefail

WS=${GPUB_WS:-/workspace}
RT=$WS/.gpuhop
PROJ=$WS/project
STAMP=$RT/setup_stamp
cd "$PROJ"

step() { echo; echo "===== [bootstrap $(date -u +%H:%M:%S)] $* ====="; CURRENT_STEP="$*"; }
trap 'echo "FAILED at: ${CURRENT_STEP:-?}" | tee "$WS/.bootstrap_failed"' ERR
rm -f "$WS/.ready" "$WS/.bootstrap_failed"

step "persist environment"
# Variables passed to the container are visible to its entrypoint but not to ssh sessions. env.sh (written by
# the launcher) sources this file. NVIDIA_VISIBLE_DEVICES is skipped: some hosts set it to "void" inside.
env | grep -E '^(NVIDIA_DRIVER_CAPABILITIES|CONTAINER_ID|VAST_|OMNI_|ACCEPT_EULA|PRIVACY_CONSENT)' \
  | sed 's/^\([^=]*\)=\(.*\)$/export \1="\2"/' > "$RT/env.container.sh" || true
if [ -w "$HOME" ]; then
  grep -q "gpuhop/env.sh" "$HOME/.bashrc" 2>/dev/null || echo "[ -f '$RT/env.sh' ] && . '$RT/env.sh'" >> "$HOME/.bashrc"
fi

step "base packages snapshot"
PY=${GPUB_PYTHON:-python}
if [ ! -f "$RT/base_freeze.txt" ]; then
  "$PY" -m pip list --format=freeze > "$RT/base_freeze.txt" 2>/dev/null || echo "(no pip in base image)" > "$RT/base_freeze.txt"
fi

step "setup (${GPUB_SETUP:-none})"
if [ -n "${GPUB_SETUP:-}" ]; then
  [ -f "$GPUB_SETUP" ] || { echo "setup script $GPUB_SETUP not found in the project"; false; }
  # word splitting of the file list is intended
  # shellcheck disable=SC2086
  HASH=$({ cat "$GPUB_SETUP" ${GPUB_SETUP_HASH_FILES:-} 2>/dev/null || true; } | sha256sum | cut -d' ' -f1)
  if [ -f "$STAMP" ] && [ "$(cat "$STAMP")" = "$HASH" ]; then
    echo "setup inputs unchanged since the last bootstrap; skipping (rebootstrap --force reruns it)"
  else
    bash "$GPUB_SETUP"
    echo "$HASH" > "$STAMP"
  fi
fi

step "smoke (${GPUB_SMOKE:-none})"
if [ -n "${GPUB_SMOKE:-}" ]; then
  export GPUB_SMOKE_OUT=$RT/smoke.json
  bash -c "$GPUB_SMOKE"
fi

touch "$WS/.ready"
echo "BOOTSTRAP OK"

#!/usr/bin/env bash
# Runs once on each new instance, from the project dir. Re-runs only when this file or a file listed in
# setup_hash_files changes (gpuhop rebootstrap --force reruns it anyway).
set -euo pipefail
python -m pip install -q --upgrade pip
# requirements.lock.txt (written by `gpuhop freeze`) pins exact versions once you have a working instance
if [ -s requirements.lock.txt ]; then
  python -m pip install -q -r requirements.lock.txt
else
  python -m pip install -q -r requirements.txt
fi

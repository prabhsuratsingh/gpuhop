"""Runs on the GPU box (standard library only; uploaded by gpuhop, never pip-installed there).

    python jobrunner.py /workspace/jobs/<exp>.json        (started detached in tmux by the launcher)

State (read by `gpuhop status`):
    <jobs dir>/<exp>.status.json   {"state": running|done|failed, "jobs": {name: {...}}}
    <logs dir>/<exp>/<job>.log     stdout+stderr of each job
Resumable: jobs already marked ok are skipped when the runner is restarted (e.g. after a host reboot).
Jobs run one at a time on purpose: concurrent jobs on one GPU would distort every timing they report.
"""

from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path


def _write(path: Path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)


def _kill(proc: subprocess.Popen):
    if hasattr(os, "killpg"):
        try:
            os.killpg(proc.pid, signal.SIGKILL)  # jobs may spawn children (simulators, dataloaders): kill the group
        except ProcessLookupError:
            pass
    else:
        proc.kill()


def main(job_file: str) -> int:
    job_file = Path(job_file)
    spec = json.loads(job_file.read_text())
    exp = spec["exp"]
    project_dir = Path(spec["project_dir"])
    shell = spec.get("shell", "bash")
    status_path = job_file.with_suffix(".status.json")
    log_dir = Path(spec["logs_dir"]) / exp
    log_dir.mkdir(parents=True, exist_ok=True)
    status = json.loads(status_path.read_text()) if status_path.exists() else {"exp": exp, "jobs": {}}
    status.update({"state": "running", "pid": os.getpid(), "started": status.get("started") or time.time(),
                   "git_sha": spec.get("git_sha"), "n_jobs": len(spec["jobs"])})
    _write(status_path, status)

    failed_hard = False
    for job in spec["jobs"]:
        name = job["name"]
        if status["jobs"].get(name, {}).get("state") == "ok":
            continue
        rec = {"state": "running", "cmd": job["cmd"], "start": time.time()}
        status["jobs"][name] = rec
        _write(status_path, status)
        results = project_dir / job["results"]
        results.mkdir(parents=True, exist_ok=True)
        env = {**os.environ, "GPUB_EXP": exp, "GPUB_RUN_ID": name, "GPUB_RESULTS": results.as_posix()}
        with open(log_dir / f"{name}.log", "w") as log:
            log.write(f"$ {job['cmd']}\n")
            log.flush()
            inner = f"source {shlex.quote(spec['env_file'])} 2>/dev/null; {job['cmd']}"
            proc = subprocess.Popen([shell, "-c", inner], cwd=project_dir, env=env, stdout=log,
                                    stderr=subprocess.STDOUT, start_new_session=True)
            try:
                rc = proc.wait(timeout=job["timeout_s"])
            except subprocess.TimeoutExpired:
                _kill(proc)
                proc.wait()
                rc = "timeout"
        rec.update({"end": time.time(), "returncode": rc, "state": "ok" if rc == 0 else "failed"})
        rec["duration_s"] = round(rec["end"] - rec["start"], 1)
        _write(status_path, status)
        if rc != 0 and not job.get("allow_fail"):
            failed_hard = True
            if spec.get("stop_on_fail"):
                break

    status["state"] = "failed" if failed_hard else "done"
    status["finished"] = time.time()
    _write(status_path, status)
    return 1 if failed_hard else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))

"""Runs on the GPU box (standard library only; uploaded by gpuhop, never pip-installed there).

    python jobrunner.py /workspace/jobs/<exp>.json        (started detached in tmux by the launcher)

State (read by `gpuhop status`):
    <jobs dir>/<exp>.status.json   {"state": running|done|failed, "jobs": {name: {...}}}
    <jobs dir>/<exp>.heartbeat     touched while the runner is alive, so the launcher can tell a dead runner
    <logs dir>/<exp>/<job>.log     stdout+stderr of each job
Resumable: jobs already marked ok are skipped when the runner is restarted (e.g. after a host reboot), as long as
their command and the pushed code are unchanged.
Jobs run one at a time on purpose: concurrent jobs on one GPU would distort every timing they report.
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

HEARTBEAT_S = 10


def _write(path: Path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    for attempt in range(50):
        try:
            tmp.replace(path)
            return
        except PermissionError:  # Windows (the local test box): a reader has the file open for a moment
            if attempt == 49:
                raise
            time.sleep(0.1)


def _kill(proc: subprocess.Popen):
    if hasattr(os, "killpg"):
        try:
            os.killpg(proc.pid, signal.SIGKILL)  # jobs may spawn children (simulators, dataloaders): kill the group
        except ProcessLookupError:
            pass
    else:
        proc.kill()


def _kill_orphan(pid: int | None, exp: str):
    """A runner that was SIGKILLed leaves its job running. Kill it, after checking the pid is still that job."""
    if not pid or not hasattr(os, "killpg"):
        return
    try:
        environ = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
        if f"GPUB_EXP={exp}".encode() in environ:
            os.killpg(pid, signal.SIGKILL)
    except (OSError, ProcessLookupError):
        pass


def _fingerprint(job: dict, spec: dict) -> str:
    return hashlib.sha256(json.dumps([job["cmd"], spec.get("code_id")]).encode()).hexdigest()


class _Heartbeat:
    def __init__(self, path: Path):
        self.path = path
        self._stop = threading.Event()
        self._beat()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _beat(self):
        try:
            self.path.touch()
        except OSError:  # e.g. disk full: the launcher then sees a stale heartbeat
            pass

    def _run(self):
        while not self._stop.wait(HEARTBEAT_S):
            self._beat()

    def stop(self, remove: bool = False):
        self._stop.set()
        self._thread.join(timeout=5)
        if remove:
            try:
                self.path.unlink()
            except OSError:
                pass


def _run_jobs(spec: dict, status: dict, status_path: Path, log_dir: Path, current: dict) -> bool:
    """Returns True when a job that is not allow_fail failed."""
    project_dir = Path(spec["project_dir"])
    shell = spec.get("shell", "bash")
    exp = spec["exp"]
    failed_hard = False
    for job in spec["jobs"]:
        name = job["name"]
        fingerprint = _fingerprint(job, spec)
        prev = status["jobs"].get(name, {})
        if prev.get("state") == "ok" and prev.get("fingerprint") == fingerprint:
            continue
        rec = {"state": "running", "cmd": job["cmd"], "start": time.time(), "fingerprint": fingerprint}
        status["jobs"][name] = rec
        _write(status_path, status)
        results = project_dir / job["results"]
        results.mkdir(parents=True, exist_ok=True)
        env = {**os.environ, "GPUB_EXP": exp, "GPUB_RUN_ID": name, "GPUB_RESULTS": results.as_posix()}
        with open(log_dir / f"{name}.log", "w", encoding="utf-8", errors="replace") as log:
            log.write(f"$ {job['cmd']}\n")
            log.flush()
            inner = f"source {shlex.quote(spec['env_file'])} 2>/dev/null; {job['cmd']}"
            proc = subprocess.Popen([shell, "-c", inner], cwd=project_dir, env=env, stdout=log,
                                    stderr=subprocess.STDOUT, start_new_session=True)
            current["proc"] = proc
            status["job_pid"] = proc.pid
            _write(status_path, status)
            try:
                rc = proc.wait(timeout=job["timeout_s"])
            except subprocess.TimeoutExpired:
                _kill(proc)
                proc.wait()
                rc = "timeout"
            current["proc"] = None
        status["job_pid"] = None
        rec.update({"end": time.time(), "returncode": rc, "state": "ok" if rc == 0 else "failed"})
        rec["duration_s"] = round(rec["end"] - rec["start"], 1)
        _write(status_path, status)
        if rc != 0 and not job.get("allow_fail"):
            failed_hard = True
            if spec.get("stop_on_fail"):
                break
    return failed_hard


def main(job_file: str) -> int:
    job_file = Path(job_file)
    spec = json.loads(job_file.read_text())
    exp = spec["exp"]
    status_path = job_file.with_suffix(".status.json")
    log_dir = Path(spec["logs_dir"]) / exp
    log_dir.mkdir(parents=True, exist_ok=True)
    status = json.loads(status_path.read_text()) if status_path.exists() else {"exp": exp, "jobs": {}}
    _kill_orphan(status.get("job_pid"), exp)
    names = {j["name"] for j in spec["jobs"]}
    status["jobs"] = {k: v for k, v in status["jobs"].items() if k in names}  # jobs removed from the file
    status.update({"state": "running", "pid": os.getpid(), "started": status.get("started") or time.time(),
                   "git_sha": spec.get("git_sha"), "n_jobs": len(spec["jobs"]), "token": spec.get("token"),
                   "job_pid": None})
    status.pop("error", None)
    _write(status_path, status)

    current: dict = {"proc": None}

    def on_signal(signum, _frame):  # tmux kill-session sends SIGHUP: the job must not outlive its runner
        if current["proc"] is not None:
            _kill(current["proc"])
        raise SystemExit(128 + signum)

    for sig in ("SIGHUP", "SIGTERM"):
        if hasattr(signal, sig):
            signal.signal(getattr(signal, sig), on_signal)

    heartbeat = _Heartbeat(job_file.with_suffix(".heartbeat"))
    try:
        failed_hard = _run_jobs(spec, status, status_path, log_dir, current)
    except Exception as e:  # never leave the status at "running" with nobody running
        for rec in status["jobs"].values():
            if rec.get("state") == "running":
                rec.update({"state": "failed", "returncode": "runner_error", "end": time.time()})
        status.update({"state": "failed", "finished": time.time(), "error": f"{type(e).__name__}: {e}"})
        print(f"job runner failed: {status['error']}", file=sys.stderr, flush=True)
        try:
            _write(status_path, status)
        finally:
            heartbeat.stop(remove=True)
        return 1
    except SystemExit:  # stopped by a signal: a replacement runner may already own the status file
        heartbeat.stop()
        raise

    status["state"] = "failed" if failed_hard else "done"
    status["finished"] = time.time()
    _write(status_path, status)
    heartbeat.stop(remove=True)
    return 1 if failed_hard else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))

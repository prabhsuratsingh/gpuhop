"""SSH transport to a GPU box using the system OpenSSH client (ships with Windows 10+, macOS, Linux).

Files move as tar streams over ssh stdin/stdout, which works identically on every OS (no rsync, no scp).

Layout on the box (`ws` = /workspace by default):
    ws/project/            the project's code (tarball push or git checkout); jobs run here
    ws/.gpuhop/         env.sh, jobrunner.py, bootstrap.sh, provenance.json, smoke.json, setup stamp
    ws/jobs/<exp>.json     job list + <exp>.status.json written by the runner
    ws/logs/<exp>/<job>.log
    ws/bootstrap.log, ws/.ready, ws/.bootstrap_failed
"""

from __future__ import annotations

import os
import posixpath
import re
import shlex
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path, PurePosixPath


def _local_parts(name: str) -> list[str] | None:
    """A tar member's path as safe local path parts; None if it points outside the destination."""
    if os.name == "nt":
        name = name.replace("\\", "/")
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        return None
    parts = list(path.parts)
    if os.name == "nt":  # characters NTFS rejects, or (":") silently turns into an alternate data stream
        parts = [re.sub(r'[<>:"|?*\x00-\x1f]', "_", p).rstrip(" .") or "_" for p in parts]
    return parts


def safe_extract(tf: tarfile.TarFile, dest: Path) -> list[str]:
    """Extract what a rented (untrusted) box sent, on every Python version: nothing lands outside `dest`, and links
    are materialised as plain copies of their target inside the archive. Returns the members that were skipped."""
    root = dest.resolve()
    skipped = []
    for m in tf:
        parts = _local_parts(m.name)
        if parts is None or not (m.isdir() or m.isreg() or m.islnk() or m.issym()):
            skipped.append(m.name)
            continue
        if not parts:  # the archive root "."
            continue
        target = root.joinpath(*parts)
        parent = target.parent.resolve()
        if parent != root and root not in parent.parents:  # through a symlink already on disk
            skipped.append(m.name)
            continue
        if m.isdir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        try:
            src = tf.extractfile(m)
        except (KeyError, RecursionError, tarfile.TarError):  # dangling or self-referencing link
            src = None
        if src is None:  # includes a symlink to a directory, whose contents are in the archive anyway
            if not m.isreg():
                skipped.append(m.name)
                continue
            src = tf.extractfile(m)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_symlink():
            target.unlink()
        with src, open(target, "wb") as out:
            shutil.copyfileobj(src, out)
        if os.name != "nt":
            os.chmod(target, (m.mode & 0o755) | 0o600)
        if m.mtime:
            os.utime(target, (m.mtime, m.mtime))
    return skipped


class Remote:
    def __init__(self, host: str, port: int, user: str = "root", key: str | None = None, ws: str = "/workspace",
                 known_hosts: Path | None = None):
        self.host, self.port, self.user = host, int(port), user
        self.key = os.path.expanduser(key) if key else None
        self.ws = ws
        self.known_hosts = known_hosts

    # -- paths on the box ------------------------------------------------------------------------------
    @property
    def proj(self) -> str:
        return posixpath.join(self.ws, "project")

    @property
    def rt(self) -> str:
        return posixpath.join(self.ws, ".gpuhop")

    @property
    def env_file(self) -> str:
        return posixpath.join(self.rt, "env.sh")

    # -- command execution -----------------------------------------------------------------------------
    def base_cmd(self, tty: bool = False) -> list[str]:
        cmd = ["ssh", "-p", str(self.port), "-o", "StrictHostKeyChecking=no",
               "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=6",
               "-o", "ConnectTimeout=20", "-o", "LogLevel=ERROR"]
        if self.known_hosts:
            self.known_hosts.parent.mkdir(parents=True, exist_ok=True)
            cmd += ["-o", f"UserKnownHostsFile={self.known_hosts.resolve()}"]
        if self.key:
            cmd += ["-i", self.key]
        cmd += ["-t"] if tty else ["-T"]
        return cmd + [f"{self.user}@{self.host}"]

    def wrap(self, cmd: str, cwd: str | None = "") -> str:
        """The shell snippet that runs `cmd` on the box with the project environment (cwd "" = project dir)."""
        cwd = self.proj if cwd == "" else cwd
        return f"source {shlex.quote(self.env_file)} 2>/dev/null; " + (f"cd {shlex.quote(cwd)} 2>/dev/null; " if cwd else "") + cmd

    def argv(self, cmd: str, cwd: str | None = "", tty: bool = False) -> list[str]:
        return self.base_cmd(tty) + ["bash -c " + shlex.quote(self.wrap(cmd, cwd))]

    def run(self, cmd: str, *, input: bytes | str | None = None, check: bool = True, capture: bool = True,
            timeout: float | None = 600, cwd: str | None = "", stdout=None) -> subprocess.CompletedProcess:
        if isinstance(input, str):
            input = input.encode()
        kw = {"stdout": stdout, "stderr": subprocess.PIPE} if stdout is not None else {"capture_output": capture}
        p = subprocess.run(self.argv(cmd, cwd), input=input, timeout=timeout, **kw)
        if check and p.returncode != 0:
            err = (p.stderr or b"").decode(errors="replace")[-2000:] if (capture or stdout is not None) else ""
            raise RuntimeError(f"remote command failed ({p.returncode}): {cmd}\n{err}")
        return p

    def text(self, cmd: str, **kw) -> str:
        return self.run(cmd, **kw).stdout.decode(errors="replace")

    def ok(self, cmd: str, timeout: float = 60) -> bool:
        try:
            return self.run(cmd, check=False, timeout=timeout, cwd=None).returncode == 0
        except subprocess.TimeoutExpired:
            return False

    def detach(self, name: str, cmd: str, log: str, cwd: str | None = ""):
        """Start `cmd` in a detached tmux session that survives the ssh connection closing.

        (`setsid nohup ... &` over a non-tty ssh session races with session teardown and can be killed
        before it even opens its log file; the tmux server is a proper daemon.)
        """
        inner = self.wrap(f"{cmd} > {shlex.quote(log)} 2>&1", cwd)
        self.run(f"command -v tmux >/dev/null || (apt-get update -qq && apt-get install -y -qq tmux >/dev/null); "
                 f"mkdir -p {shlex.quote(posixpath.dirname(log))} && tmux kill-session -t {name} 2>/dev/null; "
                 f"tmux new-session -d -s {name} {shlex.quote('bash -c ' + shlex.quote(inner))} && sleep 2 && "
                 f"(tmux has-session -t {name} 2>/dev/null || test -s {shlex.quote(log)})", cwd=None)

    def interactive(self, cmd: str | None = None) -> int:
        args = self.base_cmd(tty=True)
        if cmd:
            args.append("bash -c " + shlex.quote(self.wrap(cmd)))
        return subprocess.call(args)

    # -- files -----------------------------------------------------------------------------------------
    def put_bytes(self, remote_path: str, content: bytes | str, mode: str | None = None):
        q = shlex.quote(remote_path)
        chmod = f" && chmod {mode} {q}" if mode else ""
        self.run(f"mkdir -p {shlex.quote(posixpath.dirname(remote_path))} && cat > {q}{chmod}", input=content, cwd=None)

    put_text = put_bytes

    def push_tar(self, tar_path: Path, remote_dir: str, timeout: float = 3600):
        """Extract a local .tar.gz into `remote_dir` (created if needed; existing files are overwritten)."""
        q = shlex.quote(remote_dir)
        with open(tar_path, "rb") as f:
            p = subprocess.run(self.argv(f"mkdir -p {q} && tar -xzf - -C {q} --no-same-owner", cwd=None),
                               stdin=f, capture_output=True, timeout=timeout)
        if p.returncode != 0:
            raise RuntimeError(f"code push failed: {p.stderr.decode(errors='replace')[-1000:]}")

    def pull_dir(self, remote_dir: str, local_dir: Path, timeout: float = 3600, exclude: tuple[str, ...] = ()) -> bool:
        """tar the remote dir over ssh and extract locally. Returns False if the remote dir doesn't exist."""
        if not self.ok(f"test -d {shlex.quote(remote_dir)}"):
            return False
        local_dir.mkdir(parents=True, exist_ok=True)
        excl = " ".join(f"--exclude={shlex.quote(p)}" for p in exclude)
        fd, tmp_path = tempfile.mkstemp(suffix=".tgz")
        try:
            with os.fdopen(fd, "wb") as tmp:
                # GNU tar exits 1 for "file changed as we read it": expected while a job is still writing
                p = subprocess.run(self.argv(f"tar -C {shlex.quote(remote_dir)} --ignore-failed-read {excl} -czf - .; "
                                             'rc=$?; [ "$rc" -eq 1 ] && exit 0; exit "$rc"', cwd=None),
                                   stdout=tmp, stderr=subprocess.PIPE, timeout=timeout)
            if p.returncode != 0:
                raise RuntimeError(f"tar over ssh failed: {p.stderr.decode(errors='replace')[-1000:]}")
            with tarfile.open(tmp_path, "r:gz") as tf:
                skipped = safe_extract(tf, local_dir)
            if skipped:
                print(f"  warning: skipped {len(skipped)} unsafe or dangling tar entries, e.g. {skipped[:3]}", flush=True)
        finally:
            os.unlink(tmp_path)
        return True

    def pull_file(self, remote_path: str, local_path: Path) -> bool:
        p = self.run(f"cat {shlex.quote(remote_path)}", check=False, cwd=None)
        if p.returncode != 0:
            return False
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(p.stdout)
        return True

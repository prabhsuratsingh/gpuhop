"""Getting the project's code onto the box.

tarball (default): the local working tree as it is, minus what .gitignore excludes (via `git ls-files` when the
    project is a git repo, else a directory walk). No GitHub account needed. Uncommitted changes are included
    and recorded in provenance.json (git sha + dirty flag + hash of the diff), so every result still says
    exactly which code produced it.
git: the box clones the pushed HEAD commit. Strict provenance (the commit must be on a remote branch, the tree
    must be clean) at the cost of a GitHub token (GPUB_GH_TOKEN, read-only fine-grained PAT).
"""

from __future__ import annotations

import fnmatch
import hashlib
import io
import json
import os
import shlex
import socket
import subprocess
import tarfile
import tempfile
import time
from pathlib import Path

from .project import Project, env

# Never shipped, whatever .gitignore says.
ALWAYS_EXCLUDE = [".env", ".env.*", ".gpuhop", ".gpuhop/*", ".git", ".git/*"]
WALK_SKIP_DIRS = {".git", ".gpuhop", "__pycache__", ".venv", "venv", ".mypy_cache", ".pytest_cache", ".ruff_cache",
                  "node_modules"}
WARN_MB = 200


def _git(root: Path, *args, check: bool = True) -> str:
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=check).stdout.strip()


def is_git(root: Path) -> bool:
    try:
        return _git(root, "rev-parse", "--is-inside-work-tree", check=False) == "true"
    except OSError:  # git not installed
        return False


def _excluded(rel: str, patterns: list[str]) -> bool:
    parts = rel.split("/")
    for pat in patterns:
        pat = pat.rstrip("/")
        if fnmatch.fnmatch(rel, pat) or any(fnmatch.fnmatch(p, pat) for p in parts[:-1]) or fnmatch.fnmatch(parts[-1], pat):
            return True
    return False


def list_files(proj: Project) -> list[str]:
    """Project-relative posix paths that would be shipped."""
    root = proj.root
    patterns = ALWAYS_EXCLUDE + list(proj.cfg["code"].get("exclude") or [])
    results = proj.cfg["results"]["local_dir"].strip("/") + "/"  # pulled results never go back up
    if is_git(root):
        out = subprocess.run(["git", "ls-files", "-co", "--exclude-standard", "-z"], cwd=root, capture_output=True,
                             check=True).stdout.decode()
        files = sorted({f for f in out.split("\0") if f and (root / f).is_file()})  # drops deleted-but-tracked files
    else:
        files = []
        for d, dirs, names in os.walk(root):
            dirs[:] = sorted(x for x in dirs if x not in WALK_SKIP_DIRS)
            rel_d = Path(d).relative_to(root).as_posix()
            files += [n if rel_d == "." else f"{rel_d}/{n}" for n in sorted(names)]
    return sorted(f for f in files if not f.startswith(results) and not _excluded(f, patterns))


def git_info(root: Path) -> dict:
    if not is_git(root):
        return {"git_sha": None, "dirty": None, "diff_sha256": None}
    sha = _git(root, "rev-parse", "HEAD", check=False) or None
    diff = subprocess.run(["git", "diff", "HEAD"], cwd=root, capture_output=True, check=False).stdout if sha else b""
    dirty = bool(_git(root, "status", "--porcelain", check=False))
    return {"git_sha": sha, "dirty": dirty, "diff_sha256": hashlib.sha256(diff).hexdigest() if diff else None}


def git_state(root: Path, allow_dirty: bool = False) -> dict:
    """git mode: the commit the box will clone. It must be clean and pushed."""
    if not is_git(root):
        raise SystemExit("code.mode is 'git' but the project is not a git repository (use code.mode: tarball)")
    sha = _git(root, "rev-parse", "HEAD")
    dirty = _git(root, "status", "--porcelain", "--untracked-files=no")
    if dirty and not allow_dirty:
        raise SystemExit("working tree has uncommitted changes; commit + push first (or --allow-dirty for debugging)")
    _git(root, "fetch", "-q", "origin")
    if not _git(root, "branch", "-r", "--contains", sha):
        raise SystemExit(f"commit {sha[:10]} is not on any remote branch; `git push` first (the box clones it)")
    url = _git(root, "remote", "get-url", "origin")
    if "github.com" not in url:
        raise SystemExit(f"git mode supports GitHub remotes only (origin is {url}); use code.mode: tarball")
    repo = url.split("github.com")[-1].lstrip(":/").removesuffix(".git")
    return {"git_sha": sha, "repo": repo, "dirty": bool(dirty)}


class CodePackage:
    """The code for one push, prepared locally before anything is rented (so mistakes cost nothing)."""

    def __init__(self, proj: Project, allow_dirty: bool = False):
        self.proj = proj
        self.mode = proj.cfg["code"]["mode"]
        self.tar_path: Path | None = None
        if self.mode == "git":
            self.git = git_state(proj.root, allow_dirty)
            self.files = None
            if not env("GH_TOKEN"):
                raise SystemExit("code.mode is 'git': set GPUB_GH_TOKEN in .env (fine-grained PAT, Contents: read-only)")
        else:
            self.git = git_info(proj.root)
            self.files = list_files(proj)
        self.provenance = {"mode": self.mode, **{k: self.git.get(k) for k in ("git_sha", "dirty")},
                           "diff_sha256": self.git.get("diff_sha256"), "repo": self.git.get("repo"),
                           "n_files": len(self.files) if self.files is not None else None,
                           "pushed_from": socket.gethostname(), "pushed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}

    @property
    def sha(self) -> str | None:
        return self.git.get("git_sha")

    def describe(self) -> str:
        if self.mode == "git":
            return f"git {self.git['repo']} @ {self.sha[:10]}"
        sha = f" @ {self.sha[:10]}{' +uncommitted' if self.git.get('dirty') else ''}" if self.sha else ""
        return f"tarball of {len(self.files)} files{sha}"

    def build(self) -> Path:
        """Write the tarball (tarball mode). Files are added in sorted order with fixed mtimes, so the
        tarball's hash depends only on paths + contents.

        Shell scripts get LF line endings: a Windows checkout (core.autocrlf) has CRLF, which bash on the box
        rejects (`set: pipefail\\r: invalid option`)."""
        if self.tar_path:
            return self.tar_path
        fd, name = tempfile.mkstemp(suffix=".tar.gz", prefix="gpuhop-code-")
        os.close(fd)
        h = hashlib.sha256()
        with tarfile.open(name, "w:gz") as tf:
            for rel in self.files:
                data = (self.proj.root / rel).read_bytes()
                script = rel.endswith((".sh", ".bash")) or data.startswith(b"#!")
                if script:
                    data = data.replace(b"\r\n", b"\n")
                h.update(rel.encode() + b"\0" + hashlib.sha256(data).digest())
                ti = tarfile.TarInfo(rel)
                ti.size, ti.mtime = len(data), 0
                ti.mode = 0o755 if script or (os.name != "nt" and os.access(self.proj.root / rel, os.X_OK)) else 0o644
                tf.addfile(ti, io.BytesIO(data))
        self.provenance["content_sha256"] = h.hexdigest()
        self.tar_path = Path(name)
        mb = self.tar_path.stat().st_size / 1e6
        self.provenance["tar_mb"] = round(mb, 2)
        if mb > WARN_MB:
            print(f"warning: code tarball is {mb:.0f} MB; exclude data/checkpoints via .gitignore or code.exclude")
        return self.tar_path

    def cleanup(self):
        if self.tar_path and self.tar_path.exists():
            self.tar_path.unlink()
        self.tar_path = None

    def push(self, rem) -> dict:
        """Put the code at rem.proj and write rem.rt/provenance.json. Re-pushing (sync) removes files that
        were shipped last time but no longer exist locally; files the box created (results) are untouched."""
        if self.mode == "git":
            self._push_git(rem)
        else:
            old = rem.run(f"cat {shlex.quote(rem.rt)}/files.txt", check=False, cwd=None)
            self.build()
            try:
                rem.push_tar(self.tar_path, rem.proj)
            finally:
                self.cleanup()
            if old.returncode == 0:
                stale = sorted(set(old.stdout.decode(errors="replace").splitlines()) - set(self.files))
                if stale:
                    rem.run("cd " + shlex.quote(rem.proj) + " && rm -f -- " + " ".join(shlex.quote(s) for s in stale), cwd=None)
            rem.put_text(f"{rem.rt}/files.txt", "\n".join(self.files) + "\n")
        rem.put_text(f"{rem.rt}/provenance.json", json.dumps(self.provenance, indent=2))
        return self.provenance

    def _push_git(self, rem):
        tok = env("GH_TOKEN")
        url = f"https://x-access-token:{tok}@github.com/{self.git['repo']}.git"
        clean = f"https://github.com/{self.git['repo']}.git"
        p, q = shlex.quote(rem.proj), shlex.quote(url)
        rem.run("command -v git >/dev/null || (apt-get update -qq && apt-get install -y -qq git ca-certificates >/dev/null); "
                f"if [ -d {p}/.git ]; then git -C {p} fetch -q {q} '+refs/heads/*:refs/remotes/origin/*'; "
                f"else rm -rf {p} && git clone -q {q} {p} && git -C {p} remote set-url origin {shlex.quote(clean)}; fi && "
                f"git -C {p} checkout -q {self.sha}", cwd=None)

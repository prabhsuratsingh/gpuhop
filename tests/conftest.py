"""Test doubles: a fake GPU provider and a "remote" that is a local directory driven through bash.

With these, the real Launcher runs the whole session (push code -> bootstrap -> job runner -> pull -> destroy)
on the test machine, using the real runtime files (bootstrap.sh, jobrunner.py).
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from gpuhop.launcher import Launcher
from gpuhop.providers.base import Provider
from gpuhop.remote import Remote


def find_bash() -> str | None:
    if os.name == "nt":  # PATH lookup from Windows may hit WSL's bash.exe; use Git for Windows' bash
        try:
            exec_path = subprocess.run(["git", "--exec-path"], capture_output=True, text=True).stdout.strip()
            cand = Path(exec_path).parents[2] / "bin" / "bash.exe"
            if cand.exists():
                return cand.as_posix()
        except OSError:
            pass
        return None
    return shutil.which("bash")


BASH = find_bash()
needs_bash = pytest.mark.skipif(BASH is None or shutil.which("tar") is None and os.name != "nt",
                                reason="needs bash + tar")


class FakeProvider(Provider):
    name = "fake"

    def __init__(self, offers=None, fail_offers=(), stuck_offers=(), endpoints=None):
        self.offers = offers or [{"id": 101, "gpu_name": "RTX 4090", "dph": 0.40, "reliability": 0.99, "cuda": 12.8,
                                  "inet_down": 900, "cpu_cores": 16, "cpu_name": "AMD EPYC 9684X", "location": "XX"},
                                 {"id": 102, "gpu_name": "RTX 4090", "dph": 0.45, "reliability": 0.99, "cuda": 12.8,
                                  "inet_down": 900, "cpu_cores": 16, "cpu_name": "AMD EPYC 7B13", "location": "XX"}]
        self.fail_offers = set(fail_offers)
        self.stuck_offers = set(stuck_offers)  # stay "loading" forever (image pull that never finishes)
        self.endpoints = endpoints  # e.g. [{"host": "dead", ...}, {"host": "local", ...}]
        self.instances: dict[int, dict] = {}
        self.destroyed: list[int] = []
        self.created: list[dict] = []
        self._next = 1000

    def search(self, spec, limit=10, cpu=None):
        import re

        return [o for o in self.offers if not cpu or re.search(cpu, o["cpu_name"], re.IGNORECASE)][:limit]

    def create(self, spec, offer, env, onstart, label):
        self._next += 1
        status = ("exited" if offer["id"] in self.fail_offers else "loading" if offer["id"] in self.stuck_offers
                  else "running")
        eps = self.endpoints or [{"host": "local", "port": 0, "kind": "direct"}]
        self.instances[self._next] = {"id": self._next, "status": status, "intended_status": "running", "msg": "",
                                      "dph": offer.get("dph"), "host": eps[0]["host"], "port": eps[0]["port"],
                                      "endpoints": eps, "gpu_name": "RTX 4090", "label": label, "start_date": None,
                                      "raw": {"env": env, "GH_TOKEN": "secret"}}
        self.created.append({"offer": offer["id"], "env": env, "onstart": onstart})
        return self._next

    def show(self, inst_id):
        return self.instances.get(inst_id)

    def list(self):
        return list(self.instances.values())

    def destroy(self, inst_id):
        self.instances.pop(inst_id, None)
        self.destroyed.append(inst_id)

    def start(self, inst_id):
        self.instances[inst_id]["status"] = "running"


class LocalRemote(Remote):
    """Remote whose "box" is a local directory; commands run in Git Bash / bash instead of over ssh."""

    def __init__(self, ws: Path):
        super().__init__("local", 0, ws=ws.as_posix())
        self.procs: list[subprocess.Popen] = []

    def argv(self, cmd, cwd="", tty=False):
        return [BASH, "-c", self.wrap(cmd, cwd)]

    def detach(self, name, cmd, log, cwd=""):
        self.run(f"mkdir -p \"$(dirname {shlex.quote(log)})\"", cwd=None)
        self.procs.append(subprocess.Popen([BASH, "-c", self.wrap(f"{cmd} > {shlex.quote(log)} 2>&1", cwd)]))

    def wait_all(self, timeout=120):
        for p in self.procs:
            p.wait(timeout=timeout)


class DeadRemote(LocalRemote):
    """An ssh endpoint that refuses connections (like a host whose direct port mapping is broken)."""

    def argv(self, cmd, cwd="", tty=False):
        return [BASH, "-c", "echo 'ssh: connect to host dead port 22: Connection refused' >&2; exit 255"]


class LocalLauncher(Launcher):
    poll_running_s = poll_ready_s = poll_exp_s = poll_offer_s = 0.2

    def __init__(self, proj, provider, ws: Path):
        super().__init__(proj, provider)
        self.ws_path = ws
        self.rem = LocalRemote(ws)
        self.shell = BASH

    def _remote_at(self, host, port):
        return DeadRemote(self.ws_path) if host == "dead" else self.rem


@pytest.fixture
def home(tmp_path, monkeypatch):
    """bootstrap.sh appends to ~/.bashrc: point HOME at a temp dir."""
    h = tmp_path / "home"
    h.mkdir()
    monkeypatch.setenv("HOME", str(h))
    for k in list(os.environ):
        if k.startswith(("GPUB_", "GPUBENCH_")):
            monkeypatch.delenv(k)
    return h


PY = Path(sys.executable).as_posix()

"""The project a command runs in: the nearest `gpuhop.yaml` above the current directory, plus its `.env`.

Every path the launcher touches locally (state, known_hosts, results) is relative to the project root, so
commands work from any subdirectory of the project.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from pathlib import Path

import yaml

CONFIG_NAME = "gpuhop.yaml"

DEFAULTS: dict = {
    "project": None,  # defaults to the root directory's name
    "provider": "vast",
    "code": {"mode": "tarball", "exclude": []},  # tarball | git
    "image": "pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime",
    "image_digest": None,  # optional drift guard: refuse to launch if the tag moved
    "disk_gb": 60,
    "python": "python",  # interpreter on the box (job runner, `freeze`)
    "setup": "setup.sh",  # run once per instance; skipped when it and setup_hash_files are unchanged
    "setup_hash_files": [],
    "smoke": None,  # command that must pass before the instance counts as ready ($GPUB_SMOKE_OUT: report path)
    "on_sync": None,  # command run after `sync` pushes new code (e.g. `pip install -e . --no-deps`)
    "env": {},
    "search": {},  # overrides of the built-in catalog defaults (base_query, cpu_regex, driver_regex)
    "gpus": {},  # merged over the built-in catalog: new aliases, or per-alias image/query/cpu_regex/driver_regex/env
    "results": {"remote_dir": "results", "local_dir": "results", "exclude": ["*.pt"]},
    "lockfile": "requirements.lock.txt",  # written by `gpuhop freeze`
    "max_hours": 12.0,
    "bootstrap_timeout_min": 90,
    "image_pull_timeout_min": 10,  # give up on a host still pulling the image after this long (next offer)
    "require_git_tag": None,  # warn when `exp` runs from a commit other than this tag (code freeze)
}

# Secrets that must never be written to disk or printed.
SECRET_KEYS = ("GH_TOKEN", "VAST_API_KEY", "GPUB_GH_TOKEN", "GPUBENCH_GH_TOKEN")


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def validate(cfg: dict, source: str = CONFIG_NAME) -> dict:
    unknown = set(cfg) - set(DEFAULTS)
    if unknown:
        raise SystemExit(f"{source}: unknown key(s) {sorted(unknown)}; known: {sorted(DEFAULTS)}")
    mode = cfg["code"].get("mode")
    if mode not in ("tarball", "git"):
        raise SystemExit(f"{source}: code.mode must be 'tarball' or 'git', not {mode!r}")
    for key in ("remote_dir", "local_dir"):
        p = str(cfg["results"][key])
        if p.startswith(("/", "..")) or "\\" in p:
            raise SystemExit(f"{source}: results.{key} must be a relative path inside the project, not {p!r}")
    return cfg


def env(name: str, default: str | None = None) -> str | None:
    """`GPUB_<name>`, falling back to the older `GPUBENCH_<name>` spelling."""
    return os.environ.get(f"GPUB_{name}") or os.environ.get(f"GPUBENCH_{name}") or default


def load_dotenv(path: Path):
    """Minimal .env support (KEY=VALUE lines) so secrets never need to be typed on the command line."""
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("'\""))


@dataclass
class Project:
    root: Path
    cfg: dict

    @property
    def name(self) -> str:
        return self.cfg["project"] or self.root.name

    @property
    def state_dir(self) -> Path:
        return self.root / ".gpuhop"

    @property
    def results_dir(self) -> Path:
        return self.root / self.cfg["results"]["local_dir"]


def find_root(start: Path | None = None) -> Path:
    here = Path(start or Path.cwd()).resolve()
    for d in (here, *here.parents):
        if (d / CONFIG_NAME).exists():
            return d
    raise SystemExit(f"no {CONFIG_NAME} in {here} or any parent directory; create one with `gpuhop init`")


def load(start: Path | None = None) -> Project:
    root = find_root(start)
    raw = yaml.safe_load((root / CONFIG_NAME).read_text()) or {}
    if not isinstance(raw, dict):
        raise SystemExit(f"{root / CONFIG_NAME}: expected a mapping at the top level")
    cfg = validate(_merge(DEFAULTS, raw), str(root / CONFIG_NAME))
    load_dotenv(root / ".env")
    return Project(root, cfg)


def redact(d):
    if isinstance(d, dict):
        return {k: ("***" if k in SECRET_KEYS else redact(v)) for k, v in d.items()}
    if isinstance(d, list):
        return [redact(v) for v in d]
    return d

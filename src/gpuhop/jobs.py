"""Expand an experiment file (e.g. exp/sweep.yaml) into an ordered list of shell jobs.

    name: sweep                        # default: the file name
    timeout_min: 120                   # per job, default 120
    cmd: python train.py --lr {lr} --seed {seed} --out {results}   # default command for matrix entries
    stop_on_fail: false                # stop this experiment at the first failed job
    stop_session_on_fail: false        # `exp` skips the remaining experiment files if this one failed
    jobs:
      - matrix: {lr: [1e-3, 3e-4], seed: [0, 1, 2]}    # cartesian product, one job per combination
        cmd: ...                       # optional, overrides the file-level cmd
        name: "lr{lr}_s{seed}"         # optional run id template (default: values joined by "_", seed -> s<n>)
        overrides: {model.width: 256}  # optional; rendered into {overrides} as `key=value` tokens
        timeout_min: 30
        allow_fail: true               # a failure here does not fail the experiment
      - name: eval                     # a plain command
        cmd: python eval.py {results}

Placeholders in `cmd`: every matrix key, {exp}, {run_id}, {results} (the job's results dir, relative to the
project dir), {args} (all matrix keys as `key=value`) and {overrides}. Matrix values are shell-quoted.
Write `{{` / `}}` for literal braces. Each job also sees $GPUB_EXP, $GPUB_RUN_ID and $GPUB_RESULTS (absolute).
On the box results land in <project>/<results.remote_dir>/<exp>/<run_id>/, locally in
<results.local_dir>/<gpu>/<exp>/<run_id>/.
"""

from __future__ import annotations

import itertools
import posixpath
import re
import shlex
import string
from pathlib import Path

import yaml

_KEYS = {"name", "timeout_min", "cmd", "stop_on_fail", "stop_session_on_fail", "jobs"}
_ENTRY_KEYS = {"name", "cmd", "matrix", "overrides", "timeout_min", "allow_fail"}
_SAFE_ID = re.compile(r"[^A-Za-z0-9._=-]+")


def _fmt(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None:
        return "null"
    return str(v)


def _render(template: str, values: dict, where: str) -> str:
    names = {f for _, f, _, _ in string.Formatter().parse(template) if f}
    missing = sorted(n for n in names if n.split(".")[0].split("[")[0] not in values)
    if missing:
        raise ValueError(f"{where}: unknown placeholder(s) {missing} in {template!r}; available: {sorted(values)} "
                         "(write {{ }} for literal braces)")
    return template.format(**values)


def expand(exp_path: str | Path, remote_dir: str = "results") -> dict:
    exp_path = Path(exp_path)
    if not exp_path.exists():
        raise SystemExit(f"experiment file not found: {exp_path}")
    spec = yaml.safe_load(exp_path.read_text()) or {}
    unknown = set(spec) - _KEYS
    if unknown:
        raise ValueError(f"{exp_path}: unknown key(s) {sorted(unknown)}; known: {sorted(_KEYS)}")
    if not spec.get("jobs"):
        raise ValueError(f"{exp_path}: no `jobs:`")
    name = str(spec.get("name", exp_path.stem))
    if _SAFE_ID.search(name):
        raise ValueError(f"{exp_path}: experiment name {name!r} may only contain letters, digits and . _ = -")
    default_timeout = int(spec.get("timeout_min", 120))
    jobs = []

    def add(run_id: str, cmd_t: str, values: dict, entry: dict, where: str):
        run_id = _SAFE_ID.sub("-", run_id)
        results = posixpath.join(remote_dir, name, run_id)
        values = {**values, "exp": name, "run_id": run_id, "results": results}
        jobs.append({"name": run_id, "cmd": _render(cmd_t, values, where), "results": results,
                     "timeout_s": int(entry.get("timeout_min", default_timeout)) * 60,
                     "allow_fail": bool(entry.get("allow_fail", False))})

    for i, entry in enumerate(spec["jobs"]):
        where = f"{exp_path}: jobs[{i}]"
        bad = set(entry) - _ENTRY_KEYS
        if bad:
            raise ValueError(f"{where}: unknown key(s) {sorted(bad)}; known: {sorted(_ENTRY_KEYS)}")
        if "matrix" not in entry:
            if "name" not in entry or "cmd" not in entry:
                raise ValueError(f"{where}: a job without `matrix` needs `name` and `cmd`")
            add(str(entry["name"]), entry["cmd"], {"args": "", "overrides": ""}, entry, where)
            continue
        cmd_t = entry.get("cmd", spec.get("cmd"))
        if not cmd_t:
            raise ValueError(f"{where}: matrix entry has no `cmd` (and the file has no top-level `cmd`)")
        matrix = entry["matrix"]
        keys = list(matrix)
        combos = list(itertools.product(*(matrix[k] if isinstance(matrix[k], list) else [matrix[k]] for k in keys)))
        ov = " ".join(shlex.quote(f"{k}={_fmt(v)}") for k, v in (entry.get("overrides") or {}).items())
        for combo in combos:
            params = dict(zip(keys, combo))
            values = {k: shlex.quote(_fmt(v)) for k, v in params.items()}
            values["args"] = " ".join(shlex.quote(f"{k}={_fmt(v)}") for k, v in params.items())
            values["overrides"] = ov
            if entry.get("name"):
                run_id = _render(entry["name"], {k: _fmt(v) for k, v in params.items()}, where + ".name")
                if len(combos) > 1 and run_id == entry["name"]:  # a fixed name with several combos
                    run_id = f"{run_id}_{len(jobs)}"
            else:
                run_id = "_".join(f"s{_fmt(params[k])}" if k == "seed" else _fmt(params[k]) for k in keys)
            add(run_id, cmd_t, values, entry, where)

    names = [j["name"] for j in jobs]
    dup = {n for n in names if names.count(n) > 1}
    if dup:
        raise ValueError(f"duplicate job names in {exp_path}: {sorted(dup)} (give the matrix entry a `name:` template)")
    return {"exp": name, "jobs": jobs, "stop_on_fail": bool(spec.get("stop_on_fail", False)),
            "stop_session_on_fail": bool(spec.get("stop_session_on_fail", False))}

"""Local bookkeeping of the instances this project created: <root>/.gpuhop/instances.json."""

from __future__ import annotations

import json

from .project import Project


class State:
    def __init__(self, proj: Project):
        self.file = proj.state_dir / "instances.json"

    def load(self) -> dict:
        return json.loads(self.file.read_text()) if self.file.exists() else {}

    def save(self, state: dict):
        self.file.parent.mkdir(parents=True, exist_ok=True)
        self.file.write_text(json.dumps(state, indent=2))

    def get(self, inst_id: int) -> dict:
        return self.load().get(str(inst_id), {})

    def remember(self, inst_id: int, **info):
        s = self.load()
        s.setdefault(str(inst_id), {}).update(info)
        self.save(s)

    def forget(self, inst_id: int):
        s = self.load()
        s.pop(str(inst_id), None)
        self.save(s)

    def resolve(self, ident: str | int | None) -> int:
        """None/'last' -> most recent instance; a GPU alias -> most recent instance of that alias; else the id."""
        s = self.load()
        if ident in (None, "last"):
            if not s:
                raise SystemExit(f"no known instances ({self.file} is empty); pass an instance id")
            return int(max(s.items(), key=lambda kv: kv[1].get("created", 0))[0])
        if isinstance(ident, str) and not ident.isdigit():
            cands = [(k, v) for k, v in s.items() if v.get("gpu") == ident]
            if not cands:
                raise SystemExit(f"no known instance for {ident!r}")
            return int(max(cands, key=lambda kv: kv[1].get("created", 0))[0])
        return int(ident)

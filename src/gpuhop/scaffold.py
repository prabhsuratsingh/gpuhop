"""`gpuhop init`: copy a project template (gpuhop/templates/<name>/) into a directory.

Existing files are never overwritten (unless --force). `env.example` and `gitattributes` get their leading dot; the template's
`gitignore` lines are merged into the project's .gitignore. `__PROJECT__` in gpuhop.yaml is the project name.
"""

from __future__ import annotations

import re
from importlib import resources
from pathlib import Path

RENAME = {"env.example": ".env.example", "gitattributes": ".gitattributes"}


def templates() -> dict[str, str]:
    """name -> first line of the template's README.md."""
    out = {}
    for d in resources.files("gpuhop").joinpath("templates").iterdir():
        if d.is_dir():
            readme = d.joinpath("README.md")
            out[d.name] = readme.read_text(encoding="utf-8").splitlines()[0].lstrip("# ").strip() if readme.is_file() else ""
    return dict(sorted(out.items()))


def _walk(node, prefix=""):
    for child in node.iterdir():
        rel = f"{prefix}{child.name}"
        if child.is_dir():
            if child.name != "__pycache__":
                yield from _walk(child, rel + "/")
        else:
            yield rel, child


def init(target: Path, template: str = "pytorch", name: str | None = None, force: bool = False) -> list[str]:
    avail = templates()
    if template not in avail:
        raise SystemExit(f"unknown template {template!r}; available: {', '.join(avail)}")
    target = target.resolve()
    target.mkdir(parents=True, exist_ok=True)
    name = name or target.name
    if not re.fullmatch(r"[A-Za-z0-9._-]+", name):
        raise SystemExit(f"project name {name!r} may only contain letters, digits and . _ -")
    src = resources.files("gpuhop").joinpath("templates", template)
    report = []
    for rel, f in sorted(_walk(src)):
        if rel == "gitignore":
            report.append(_merge_gitignore(target / ".gitignore", f.read_text(encoding="utf-8")))
            continue
        out_rel = "/".join(RENAME.get(p, p) for p in rel.split("/"))
        out = target / out_rel
        if out.exists() and not force:
            report.append(f"  skip   {out_rel} (exists)")
            continue
        data = f.read_bytes()
        if out_rel == "gpuhop.yaml":
            data = data.replace(b"__PROJECT__", name.encode())
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
        report.append(f"  write  {out_rel}")
    return report


def _merge_gitignore(path: Path, lines: str) -> str:
    have = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    new = [l for l in lines.splitlines() if l.strip() and l not in have]
    if not new:
        return "  keep   .gitignore (already has the gpuhop entries)"
    with path.open("a", encoding="utf-8") as fh:
        if have and have[-1].strip():
            fh.write("\n")
        fh.write("\n".join(new) + "\n")
    return f"  {'update' if have else 'write '} .gitignore (+{len([l for l in new if not l.startswith('#')])} entries)"

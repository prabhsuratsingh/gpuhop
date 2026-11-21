"""GPU aliases -> launch spec (image, disk, offer query, env), from the built-in catalog plus project overrides."""

from __future__ import annotations

import json
import urllib.request
from importlib import resources

import yaml

from .project import Project, _merge


def builtin() -> dict:
    return yaml.safe_load(resources.files("gpuhop").joinpath("gpus.yaml").read_text(encoding="utf-8"))


def catalog(proj: Project) -> dict:
    cat = builtin()
    cat["defaults"] = _merge(cat["defaults"], proj.cfg["search"])
    cat["gpus"] = _merge(cat["gpus"], proj.cfg["gpus"])
    return cat


def gpu_spec(proj: Project, alias: str) -> dict:
    cat = catalog(proj)
    if alias not in cat["gpus"]:
        raise SystemExit(f"unknown GPU alias {alias!r}; known: {sorted(cat['gpus'])} (add it under `gpus:` in gpuhop.yaml)")
    d, g, c = cat["defaults"], cat["gpus"][alias] or {}, proj.cfg
    if "query" not in g:
        raise SystemExit(f"gpuhop.yaml: gpus.{alias} needs a `query` (Vast search syntax, e.g. gpu_name=RTX_4090)")
    disk = int(g.get("disk_gb", c["disk_gb"]))
    return {
        "alias": alias,
        "image": g.get("image", c["image"]),
        # a per-alias image without its own digest is not guarded by the project-wide digest
        "image_digest": g.get("image_digest") if "image" in g else c.get("image_digest"),
        "disk_gb": disk,
        "query": f"{d['base_query']} {g['query']} disk_space>={disk}",
        "env": {**(d.get("env") or {}), **c["env"], **(g.get("env") or {})},
        "cpu_regex": g.get("cpu_regex", d.get("cpu_regex")),
        "driver_regex": g.get("driver_regex", d.get("driver_regex")),
    }


def dockerhub_digest(image: str) -> str | None:
    """Current registry digest of a Docker Hub `repo:tag` (None if it cannot be determined)."""
    repo, _, tag = image.partition(":")
    if "/" not in repo or "." in repo.split("/")[0]:  # only Docker Hub `org/repo:tag` is supported here
        return None
    try:
        tok_url = f"https://auth.docker.io/token?service=registry.docker.io&scope=repository:{repo}:pull"
        token = json.load(urllib.request.urlopen(tok_url, timeout=20))["token"]
        req = urllib.request.Request(f"https://registry-1.docker.io/v2/{repo}/manifests/{tag or 'latest'}", method="HEAD", headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.docker.distribution.manifest.list.v2+json, application/vnd.oci.image.index.v1+json, "
                      "application/vnd.docker.distribution.manifest.v2+json"})
        return urllib.request.urlopen(req, timeout=20).headers.get("docker-content-digest")
    except Exception:
        return None


def check_image(spec: dict, allow_drift: bool = False):
    want = spec.get("image_digest")
    if not want:
        return
    got = dockerhub_digest(spec["image"])
    if got is None:
        print(f"warning: could not verify digest of {spec['image']} (registry unreachable); launching by tag")
    elif got != want and not allow_drift:
        raise SystemExit(f"{spec['image']} now resolves to {got}, not the pinned {want}.\n"
                         "The base image changed upstream: results would not be comparable across runs. Pin a new "
                         "tag/digest in gpuhop.yaml deliberately, or set GPUB_ALLOW_IMAGE_DRIFT=1.")

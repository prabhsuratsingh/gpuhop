"""Vast.ai through its Python SDK (`vastai` package). Needs VAST_API_KEY (in the project's .env)."""

from __future__ import annotations

import os
import re

from .base import Provider

_STATUS = {"running": "running", "loading": "loading", "created": "created", "stopped": "stopped",
           "exited": "exited", "offline": "offline"}


class VastProvider(Provider):
    name = "vast"

    def __init__(self):
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from vastai import VastAI

            key = os.environ.get("VAST_API_KEY")
            self._client = VastAI(api_key=key, raw=True, quiet=True) if key else VastAI(raw=True, quiet=True)
        return self._client

    # -- normalisation ---------------------------------------------------------------------------------
    @staticmethod
    def _offer(o: dict) -> dict:
        return {"id": o["id"], "gpu_name": o.get("gpu_name", ""), "dph": o.get("dph_total"),
                "reliability": o.get("reliability2", o.get("reliability", 0)), "cuda": o.get("cuda_max_good", ""),
                "inet_down": o.get("inet_down", 0), "cpu_cores": o.get("cpu_cores_effective", 0),
                "cpu_name": o.get("cpu_name") or "", "location": o.get("geolocation", ""),
                "driver": o.get("driver_version") or "", "raw": o}

    @staticmethod
    def _inst(r: dict) -> dict:
        """ssh endpoints: the direct port mapping first (faster transfers), then Vast's ssh proxy."""
        eps = []
        direct = (r.get("ports") or {}).get("22/tcp")
        if direct and r.get("public_ipaddr"):
            eps.append({"host": r["public_ipaddr"].strip(), "port": int(direct[0]["HostPort"]), "kind": "direct"})
        if r.get("ssh_host") and r.get("ssh_port"):
            eps.append({"host": r["ssh_host"], "port": int(r["ssh_port"]), "kind": "proxy"})
        host, port = (eps[0]["host"], eps[0]["port"]) if eps else (None, None)
        return {"id": int(r["id"]), "status": _STATUS.get(r.get("actual_status") or "", "unknown"), "endpoints": eps,
                "intended_status": r.get("intended_status"), "msg": (r.get("status_msg") or "").strip(),
                "dph": r.get("dph_total"), "host": host, "port": port, "gpu_name": r.get("gpu_name", ""),
                "label": r.get("label"), "start_date": r.get("start_date"), "raw": r}

    # -- Provider --------------------------------------------------------------------------------------
    def search(self, spec: dict, limit: int = 10, cpu: str | None = None) -> list[dict]:
        """`cpu` is a regex on the host CPU name: CPU-bound code (kernel launches, simulators) is sensitive to
        it, so fair comparisons pin the host CPU class too."""
        cpu = cpu or spec.get("cpu_regex")
        drv = spec.get("driver_regex")
        offers = self.client.search_offers(query=spec["query"], order="dph", limit=limit * (10 if cpu or drv else 3),
                                           storage=spec["disk_gb"])
        offers = sorted(offers or [], key=lambda o: o.get("dph_total") or 1e9)
        if cpu:
            offers = [o for o in offers if re.search(cpu, o.get("cpu_name") or "", re.IGNORECASE)]
        if drv:
            offers = [o for o in offers if re.search(drv, o.get("driver_version") or "")]
        return [self._offer(o) for o in offers[:limit]]

    def create(self, spec: dict, offer: dict, env: dict, onstart: str, label: str) -> int:
        # Vast drops env values containing spaces
        env = {k: str(v).replace(" ", "_") for k, v in env.items()}
        res = self.client.create_instance(id=int(offer["id"]), image=spec["image"], disk=spec["disk_gb"], env=env,
                                          onstart_cmd=onstart, label=label, ssh=True, direct=True)
        if not res or not res.get("success", True) or "new_contract" not in res:
            raise RuntimeError(f"create_instance failed: {res}")
        return int(res["new_contract"])

    def show(self, inst_id: int) -> dict | None:
        try:
            r = self.client.show_instance(id=inst_id)
        except Exception as e:
            if getattr(getattr(e, "response", None), "status_code", None) == 404:  # really gone, not an API hiccup
                return None
            raise
        if isinstance(r, dict) and "instances" in r:  # some API versions wrap the object
            r = r["instances"]
        return self._inst(r) if r else None

    def list(self) -> list[dict]:
        return [self._inst(r) for r in (self.client.show_instances() or [])]

    def destroy(self, inst_id: int):
        res = self.client.destroy_instance(id=inst_id)
        if isinstance(res, dict) and res.get("success") is False:
            raise RuntimeError(f"destroy_instance {inst_id} failed: {res.get('msg') or res.get('error') or res}")

    def start(self, inst_id: int):
        return self.client.start_instance(id=inst_id)

    def logs(self, inst_id: int, tail: int = 100) -> str:
        return str(self.client.logs(instance_id=inst_id, tail=str(tail)))

    def ssh_keys(self) -> list[str] | None:
        return [k.get("public_key", "") for k in (self.client.show_ssh_keys() or [])]

"""Vast.ai through its Python SDK (`vastai` package). Needs VAST_API_KEY (in the project's .env)."""

from __future__ import annotations

import json
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

    # -- failsafe ----------------------------------------------------------------------------------------
    # Vast runs scheduled jobs server-side at the top of each UTC hour between start_time and end_time (measured
    # live: a job created at 07:25 UTC stopped its instance at 08:00 UTC). Jobs outlive their instance, so
    # callers must cancel them.
    STOP_REPEAT_S = 24 * 3600  # keep re-stopping for a day, in case something restarts the instance

    @staticmethod
    def _stop_endpoint(inst_id) -> str:
        return f"/api/v0/instances/{int(inst_id)}/"

    def schedule_stop(self, inst_id: int, at: float) -> int:
        # Vast fires at the top of the hour that *contains* start_time (seen live: 10:13 fired at 10:00), so round
        # up to the next full hour or the stop could hit a healthy session up to 59 min early. Ints only: the API
        # rejects fractional timestamps.
        start = -(-int(at) // 3600) * 3600
        job = dict(start_time=start, end_time=start + self.STOP_REPEAT_S, api_endpoint=self._stop_endpoint(inst_id),
                   request_method="PUT", request_body={"state": "stopped", "instance_id": int(inst_id)},
                   frequency="HOURLY", instance_id=int(inst_id))
        try:
            res = self.client.create_scheduled_job(**job)
        except Exception as e:
            if getattr(getattr(e, "response", None), "status_code", None) != 422:  # 422: a job exists already
                raise
            self.cancel_scheduled_stops(inst_id)
            res = self.client.create_scheduled_job(**job)
        if not isinstance(res, dict) or res.get("success") is False or "scheduled_job_id" not in res:
            raise RuntimeError(f"create_scheduled_job failed: {res}")
        return int(res["scheduled_job_id"])

    def scheduled_stops(self) -> list[dict]:
        out = []
        for j in self.client.show_scheduled_jobs() or []:
            body = j.get("request_body") or {}
            if isinstance(body, str):
                try:
                    body = json.loads(body)
                except ValueError:
                    body = {}
            iid = j.get("instance_id")
            if (iid is not None and str(j.get("request_method")).upper() == "PUT" and body.get("state") == "stopped"
                    and j.get("api_endpoint") == self._stop_endpoint(iid)):
                out.append({"id": j["id"], "instance_id": int(iid), "start_time": j.get("start_time")})
        return out

    def cancel_scheduled_stops(self, inst_id: int) -> int:
        n = 0
        for j in self.scheduled_stops():
            if j["instance_id"] == int(inst_id):
                self.client.delete_scheduled_job(id=j["id"])
                n += 1
        return n

    def logs(self, inst_id: int, tail: int = 100) -> str:
        return str(self.client.logs(instance_id=inst_id, tail=str(tail)))

    def ssh_keys(self) -> list[str] | None:
        return [k.get("public_key", "") for k in (self.client.show_ssh_keys() or [])]

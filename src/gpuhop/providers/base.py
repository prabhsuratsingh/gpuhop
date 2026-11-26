"""What the launcher needs from a GPU cloud. Providers return normalised dicts; the raw API object is kept
under "raw" (redacted before it is written anywhere).

Offer:    {"id", "gpu_name", "dph", "reliability", "cuda", "inet_down", "cpu_cores", "cpu_name", "location", "driver", "raw"}
Instance: {"id", "status", "intended_status", "msg", "dph", "host", "port", "gpu_name", "label", "start_date", "raw"}
    status is one of: running, loading, created, stopped, exited, offline, unknown
    host/port are the preferred ssh endpoint (None until the provider exposes one); the optional "endpoints"
    lists every endpoint as {"host", "port", "kind"} in order of preference (the launcher falls back along it)
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class Provider(ABC):
    name = "base"
    user = "root"  # ssh user on the box

    @abstractmethod
    def search(self, spec: dict, limit: int = 10, cpu: str | None = None) -> list[dict]:
        """Cheapest offers matching a catalog spec (gpu_spec()), sorted by $/hr; honours spec["cpu_regex"] (or
        `cpu`) on the host CPU name and spec["driver_regex"] on the host NVIDIA driver version."""

    @abstractmethod
    def create(self, spec: dict, offer: dict, env: dict, onstart: str, label: str) -> int:
        """Rent `offer` with the spec's image/disk; returns the instance id."""

    @abstractmethod
    def show(self, inst_id: int) -> dict | None: ...

    @abstractmethod
    def list(self) -> list[dict]: ...

    @abstractmethod
    def destroy(self, inst_id: int): ...

    @abstractmethod
    def start(self, inst_id: int): ...

    def logs(self, inst_id: int, tail: int = 100) -> str:
        return "(this provider has no container logs; use --bootstrap or --job)"

    def ssh_keys(self) -> list[str] | None:
        """Public keys registered on the account (None: the provider does not manage keys)."""
        return None

    # -- failsafe: a stop that the provider's servers carry out, so it happens even if the laptop is gone ------
    def schedule_stop(self, inst_id: int, at: float) -> str | int | None:
        """Ask the provider to stop the instance at (or soon after) `at`, repeating until it is stopped.
        Returns the scheduled job's id, or None when the provider cannot do this."""
        return None

    def cancel_scheduled_stops(self, inst_id: int) -> int:
        """Delete the scheduled stops for one instance; returns how many were deleted."""
        return 0

    def scheduled_stops(self) -> list[dict]:
        """Every scheduled stop on the account: [{"id", "instance_id", "start_time"}]."""
        return []

    def check(self) -> str:
        """Cheap authenticated call for `doctor`; returns a one-line status."""
        return f"{len(self.list())} instance(s) on the account"


def get_provider(name: str) -> Provider:
    if name == "vast":
        from .vast import VastProvider

        return VastProvider()
    raise SystemExit(f"unknown provider {name!r}; available: vast")

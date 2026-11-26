"""The server-side failsafe stop: armed when renting for `exp`/`smoke`, removed on every normal path."""

from __future__ import annotations

import os
import time

import pytest
from conftest import FakeProvider, LocalLauncher, needs_bash
from test_session import make_project

from gpuhop import cli
from gpuhop.launcher import FailsafeError
from gpuhop.providers.vast import VastProvider


@pytest.fixture
def setup(tmp_path, home):
    proj = make_project(tmp_path / "proj")
    os.environ.pop("VAST_API_KEY", None)
    prov = FakeProvider()
    return proj, prov, LocalLauncher(proj, prov, tmp_path / "box")


@needs_bash
def test_session_arms_at_deadline_and_cleans_up(setup, monkeypatch):
    proj, prov, lx = setup
    seen = []
    real = prov.schedule_stop
    monkeypatch.setattr(prov, "schedule_stop", lambda i, at: (seen.append((i, at)), real(i, at))[1])
    t0 = time.time()
    assert lx.session("rtx4090", [str(proj.root / "exp" / "e1.yaml")], max_hours=2) == 0
    lx.rem.wait_all()
    ((inst_id, at),) = seen
    assert t0 + 2 * 3600 + lx.failsafe_grace_s <= at <= time.time() + 2 * 3600 + lx.failsafe_grace_s
    assert inst_id in prov.destroyed and prov.scheduled_stops() == []


@needs_bash
def test_failsafe_is_armed_before_the_box_is_waited_on(setup, monkeypatch):
    _, prov, lx = setup
    monkeypatch.setattr(lx, "wait_running", lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        lx.up("rtx4090", failsafe_at=time.time() + 60)
    # never reachable -> destroyed, and the destroy removed the scheduled stop as well
    assert prov.destroyed and prov.scheduled_stops() == []


@needs_bash
def test_refused_failsafe_aborts_without_trying_other_offers(setup, capsys):
    proj, prov, lx = setup
    prov.fail_schedule = RuntimeError("401 bad key")
    assert lx.session("rtx4090", [str(proj.root / "exp" / "e1.yaml")]) == 1
    assert len(prov.created) == 1 and prov.destroyed == [1001] and not prov.instances
    assert "GPUB_NO_FAILSAFE" in capsys.readouterr().out
    created: list = []
    with pytest.raises(FailsafeError):
        lx.up_with_retry("rtx4090", None, False, created, failsafe_at=time.time() + 60)
    assert created == []


@needs_bash
def test_opt_out_and_unsupported_provider_still_run(setup, monkeypatch, capsys):
    _, prov, lx = setup
    monkeypatch.setenv("GPUB_NO_FAILSAFE", "1")
    prov.fail_schedule = RuntimeError("would fail")
    i, _, _ = lx.up("rtx4090", failsafe_at=time.time() + 60)
    assert "GPUB_NO_FAILSAFE=1" in capsys.readouterr().out
    lx.destroy(i)
    monkeypatch.delenv("GPUB_NO_FAILSAFE")
    monkeypatch.setattr(prov, "schedule_stop", lambda i, at: None)  # provider without scheduled jobs
    i, _, _ = lx.up("rtx4090", failsafe_at=time.time() + 60)
    assert "cannot schedule" in capsys.readouterr().out
    lx.destroy(i)


@needs_bash
def test_kept_instances_get_no_failsafe(setup):
    proj, prov, lx = setup
    assert lx.session("rtx4090", [str(proj.root / "exp" / "e1.yaml")], keep=True) == 0
    lx.rem.wait_all()
    (inst_id,) = prov.instances
    assert prov.scheduled_stops() == []
    lx.destroy(inst_id)


@needs_bash
def test_unpulled_box_keeps_its_failsafe(setup, monkeypatch, capsys):
    _, prov, lx = setup
    i, _, _ = lx.up("rtx4090", failsafe_at=time.time() + 3600)
    monkeypatch.setattr(lx, "pull", lambda *a: (_ for _ in ()).throw(RuntimeError("ssh down")))
    lx.poll_ready_s = 0
    assert lx._finish(i, keep=False) is False
    assert "failsafe stops it after" in capsys.readouterr().out
    assert [j["instance_id"] for j in prov.scheduled_stops()] == [i]


@needs_bash
def test_failed_destroy_leaves_the_failsafe_armed(setup, monkeypatch):
    _, prov, lx = setup
    i, _, _ = lx.up("rtx4090", failsafe_at=time.time() + 3600)
    monkeypatch.setattr(prov, "destroy", lambda x: (_ for _ in ()).throw(ConnectionError("api down")))
    with pytest.raises(ConnectionError):
        lx.destroy(i)
    assert [j["instance_id"] for j in prov.scheduled_stops()] == [i]


@needs_bash
def test_start_disarms_so_recovery_is_not_stopped_again(setup, monkeypatch):
    _, prov, lx = setup
    i, _, _ = lx.up("rtx4090", failsafe_at=time.time() + 3600)
    prov.instances[i]["status"] = "stopped"
    monkeypatch.setattr(cli, "_launcher", lambda: lx)
    cli.cmd_start(type("A", (), {"id": i}))
    assert prov.scheduled_stops() == [] and prov.instances[i]["status"] == "running"
    lx.destroy(i)


@needs_bash
def test_ls_flags_failsafe_stops_and_deletes_leftover_jobs(setup, monkeypatch, capsys):
    _, prov, lx = setup
    stopped, _, _ = lx.up("rtx4090", failsafe_at=time.time() - 1)
    prov.instances[stopped]["status"] = "stopped"
    gone, _, _ = lx.up("rtx4090", failsafe_at=time.time() + 3600)
    prov.instances.pop(gone)  # destroyed elsewhere (e.g. the Vast console): its job stays behind
    monkeypatch.setattr(cli, "_launcher", lambda: lx)
    cli.cmd_ls(None)
    out = capsys.readouterr().out
    assert "stopped by the failsafe" in out and f"destroyed instance {gone}" in out
    assert [j["instance_id"] for j in prov.scheduled_stops()] == [stopped]
    lx.destroy(stopped)


class _Resp:
    def __init__(self, code):
        self.status_code = code


class _HTTPError(Exception):
    def __init__(self, code):
        super().__init__(f"{code}")
        self.response = _Resp(code)


class _Client:
    """Vast scheduled-job API as observed live (2026-11-25)."""

    def __init__(self):
        self.jobs = [
            {"id": 1, "instance_id": 5, "api_endpoint": "/api/v0/instances/reboot/5/", "request_method": "PUT",
             "request_body": {"instance_id": 5}},  # the user's own reboot schedule: never touched
            {"id": 2, "instance_id": 6, "api_endpoint": "/api/v0/instances/6/", "request_method": "PUT",
             "request_body": {"state": "stopped", "instance_id": 6}, "start_time": 1.0},
        ]
        self.created, self.deleted, self.conflict = [], [], False

    def create_scheduled_job(self, **kw):
        if self.conflict:
            self.conflict = False
            raise _HTTPError(422)
        self.created.append(kw)
        return {"success": True, "scheduled_job_id": 77}

    def show_scheduled_jobs(self):
        return list(self.jobs)

    def delete_scheduled_job(self, id):
        self.deleted.append(id)
        self.jobs = [j for j in self.jobs if j["id"] != id]
        return {"success": True}


def test_vast_schedule_stop_payload():
    v = VastProvider()
    v._client = c = _Client()
    assert v.schedule_stop(42, 1795591536.75) == 77
    (kw,) = c.created
    hour = 1795593600  # 1795591536.75 rounded up to the next full hour: Vast fires at the top of start_time's hour
    assert kw == {"start_time": hour, "end_time": hour + v.STOP_REPEAT_S,
                  "api_endpoint": "/api/v0/instances/42/", "request_method": "PUT",
                  "request_body": {"state": "stopped", "instance_id": 42}, "frequency": "HOURLY", "instance_id": 42}
    assert isinstance(kw["start_time"], int) and isinstance(kw["end_time"], int)  # the API rejects floats
    c.created.clear()
    v.schedule_stop(42, hour)  # already on the hour: unchanged
    assert c.created[0]["start_time"] == hour


def test_vast_scheduled_stops_only_touch_stop_jobs():
    v = VastProvider()
    v._client = c = _Client()
    assert v.scheduled_stops() == [{"id": 2, "instance_id": 6, "start_time": 1.0}]
    assert v.cancel_scheduled_stops(5) == 0 and c.deleted == []
    assert v.cancel_scheduled_stops(6) == 1 and c.deleted == [2]


def test_vast_replaces_an_existing_stop_on_conflict():
    v = VastProvider()
    v._client = c = _Client()
    c.conflict = True
    assert v.schedule_stop(6, 100) == 77 and c.deleted == [2]

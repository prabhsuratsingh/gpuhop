"""End to end through the real Launcher, bootstrap.sh and jobrunner.py, with a fake provider and a local "box"."""

from __future__ import annotations

import json
import textwrap

import pytest
from conftest import PY, FakeProvider, LocalLauncher, needs_bash

from gpuhop import project

pytestmark = needs_bash


def make_project(root, setup_ok=True, smoke=True):
    root.mkdir(parents=True, exist_ok=True)
    (root / "gpuhop.yaml").write_text(textwrap.dedent(f"""\
        project: e2e
        python: "{PY}"
        setup: setup.sh
        setup_hash_files: [requirements.txt]
        smoke: {'"$GPUB_PYTHON smoke.py"' if smoke else 'null'}
        env: {{MY_VAR: "hello world"}}
        bootstrap_timeout_min: 2
        """))
    (root / "setup.sh").write_text("set -e\necho setting-up\necho ran >> \"$GPUB_WS/setup_runs.txt\"\n"
                                   + ("" if setup_ok else "exit 3\n"))
    (root / "requirements.txt").write_text("numpy\n")
    (root / "smoke.py").write_text("import json, os\njson.dump({'ok': True, 'gpu': os.environ['GPUB_GPU']}, "
                                   "open(os.environ['GPUB_SMOKE_OUT'], 'w'))\n")
    (root / "job.py").write_text(textwrap.dedent("""\
        import json, os, sys
        out = os.environ["GPUB_RESULTS"]
        json.dump({"arg": sys.argv[1], "my_var": os.environ.get("MY_VAR"), "dph": os.environ.get("GPUB_DPH"),
                   "run_id": os.environ["GPUB_RUN_ID"], "cwd_ok": os.path.exists("job.py")},
                  open(os.path.join(out, "metrics.json"), "w"))
        open(os.path.join(out, "model.pt"), "w").write("weights")
        open(os.path.join(out, "video.mp4"), "wb").write(b"\\x00mp4")
        sys.exit(1 if sys.argv[1] == "bad" else 0)
        """))
    (root / ".env").write_text("VAST_API_KEY=supersecret\n")
    (root / "exp").mkdir(exist_ok=True)
    (root / "exp" / "e1.yaml").write_text(textwrap.dedent("""\
        name: e1
        cmd: $GPUB_PYTHON job.py {arg}
        jobs:
          - matrix: {arg: [a, b]}
          - name: flaky
            cmd: $GPUB_PYTHON job.py bad
            allow_fail: true
        """))
    (root / "exp" / "e2.yaml").write_text("name: e2\njobs: [{name: hard_fail, cmd: '$GPUB_PYTHON job.py bad'}]\n")
    return project.load(root)


def test_full_session_pulls_results_and_destroys(tmp_path, home):
    proj = make_project(tmp_path / "proj")
    prov = FakeProvider()
    lx = LocalLauncher(proj, prov, tmp_path / "box")
    code = lx.session("rtx4090", [str(proj.root / "exp" / "e1.yaml")])
    lx.rem.wait_all()
    assert code == 0  # the failing job is allow_fail
    box = tmp_path / "box"
    # code was pushed, secrets were not; the container env holds no secrets
    assert (box / "project" / "job.py").exists() and not (box / "project" / ".env").exists()
    assert not any("secret" in str(v) for v in prov.created[0]["env"].values())
    # results pulled: per-run dirs, videos included, checkpoints excluded by default
    res = proj.results_dir / "rtx4090" / "e1"
    m = json.loads((res / "a" / "metrics.json").read_text())
    assert m == {"arg": "a", "my_var": "hello world", "dph": "0.4000", "run_id": "a", "cwd_ok": True}
    assert (res / "b" / "video.mp4").read_bytes() == b"\x00mp4"
    assert not (res / "a" / "model.pt").exists()
    # logs + instance metadata
    inst_id = prov.destroyed[0]
    logs = proj.results_dir / "rtx4090" / "_logs" / str(inst_id) / "e1"
    assert (logs / "a.log").read_text().startswith("$ ")
    meta = proj.results_dir / "rtx4090" / "_instance" / str(inst_id)
    assert json.loads((meta / "smoke.json").read_text()) == {"ok": True, "gpu": "rtx4090"}
    prov_json = json.loads((meta / "provenance.json").read_text())
    assert prov_json["mode"] == "tarball" and prov_json["n_files"] > 0
    assert "BOOTSTRAP OK" in (meta / "bootstrap.log").read_text()
    assert json.loads((meta / "instance.json").read_text())["GH_TOKEN"] == "***"
    # destroyed, and forgotten locally
    assert prov.destroyed == [inst_id] and not prov.instances and lx.state.load() == {}


def test_hard_failure_returns_1_and_still_destroys(tmp_path, home):
    proj = make_project(tmp_path / "proj")
    prov = FakeProvider()
    lx = LocalLauncher(proj, prov, tmp_path / "box")
    code = lx.session("rtx4090", [str(proj.root / "exp" / "e2.yaml"), str(proj.root / "exp" / "e1.yaml")])
    lx.rem.wait_all()
    assert code == 1
    assert (proj.results_dir / "rtx4090" / "e2" / "hard_fail" / "metrics.json").exists()
    assert (proj.results_dir / "rtx4090" / "e1" / "a" / "metrics.json").exists()  # no stop_session_on_fail
    assert len(prov.destroyed) == 1


def test_bootstrap_failure_pulls_log_and_destroys(tmp_path, home, capsys):
    proj = make_project(tmp_path / "proj", setup_ok=False)
    prov = FakeProvider()
    lx = LocalLauncher(proj, prov, tmp_path / "box")
    code = lx.session("rtx4090", [str(proj.root / "exp" / "e1.yaml")])
    lx.rem.wait_all()
    assert code == 1
    assert "bootstrap failed" in capsys.readouterr().out
    inst_id = prov.destroyed[0]
    log = (proj.results_dir / "rtx4090" / "_instance" / str(inst_id) / "bootstrap.log").read_text()
    assert "FAILED at: setup" in log
    assert not prov.instances


def test_retry_next_offer_when_host_never_starts(tmp_path, home):
    proj = make_project(tmp_path / "proj", smoke=False)
    prov = FakeProvider(fail_offers={101})
    lx = LocalLauncher(proj, prov, tmp_path / "box")
    inst_id, _, _ = lx.up_with_retry("rtx4090", None, False, created=[])
    assert [c["offer"] for c in prov.created] == [101, 102]
    assert len(prov.destroyed) == 1 and inst_id in prov.instances
    lx.destroy(inst_id)


def test_keep_and_sync_and_rebootstrap_cache(tmp_path, home):
    proj = make_project(tmp_path / "proj")
    prov = FakeProvider()
    box = tmp_path / "box"
    lx = LocalLauncher(proj, prov, box)
    assert lx.session("rtx4090", [str(proj.root / "exp" / "e1.yaml")], keep=True) == 0
    lx.rem.wait_all()
    (inst_id,) = prov.instances
    assert (box / "setup_runs.txt").read_text().count("ran") == 1
    # sync: changed files arrive, deleted files disappear, results on the box survive
    (proj.root / "job.py").write_text((proj.root / "job.py").read_text() + "# v2\n")
    (proj.root / "smoke.py").unlink()
    proj.cfg["smoke"] = None
    lx.sync(inst_id)
    assert (box / "project" / "job.py").read_text().endswith("# v2\n")
    assert not (box / "project" / "smoke.py").exists()
    assert (box / "project" / "results" / "e1" / "a" / "model.pt").exists()
    # rebootstrap: setup is skipped when unchanged, rerun when forced
    lx.start_bootstrap(lx.rem)
    lx.rem.wait_all()
    lx.wait_ready(lx.rem, inst_id)
    assert (box / "setup_runs.txt").read_text().count("ran") == 1
    lx.start_bootstrap(lx.rem, force=True)
    lx.rem.wait_all()
    lx.wait_ready(lx.rem, inst_id)
    assert (box / "setup_runs.txt").read_text().count("ran") == 2
    lx.destroy(inst_id)


def test_validation_happens_before_renting(tmp_path, home):
    proj = make_project(tmp_path / "proj")
    (proj.root / "exp" / "bad.yaml").write_text("jobs: [{matrix: {x: [1]}, cmd: 'run {nope}'}]\n")
    prov = FakeProvider()
    lx = LocalLauncher(proj, prov, tmp_path / "box")
    with pytest.raises(ValueError, match="unknown placeholder"):
        lx.session("rtx4090", [str(proj.root / "exp" / "bad.yaml")])
    assert prov.created == []


def test_falls_back_to_working_ssh_endpoint(tmp_path, home, capsys):
    proj = make_project(tmp_path / "proj", smoke=False)
    prov = FakeProvider(endpoints=[{"host": "dead", "port": 1, "kind": "direct"},
                                   {"host": "local", "port": 0, "kind": "proxy"}])
    lx = LocalLauncher(proj, prov, tmp_path / "box")
    inst_id, inst, _ = lx.up("rtx4090")
    assert "ssh via direct endpoint dead:1 not answering yet: ssh: connect to host dead port 22: Connection refused"         in capsys.readouterr().out
    assert lx.state.get(inst_id)["ssh_kind"] == "proxy"
    assert lx.remote(inst) is lx.rem  # later commands (status, pull, ...) use the endpoint that worked
    lx.destroy(inst_id)


def test_unreachable_host_is_destroyed_and_next_offer_tried(tmp_path, home, monkeypatch):
    proj = make_project(tmp_path / "proj", smoke=False)
    prov = FakeProvider(endpoints=[{"host": "dead", "port": 1, "kind": "direct"}])
    lx = LocalLauncher(proj, prov, tmp_path / "box")
    real = lx.connect
    calls = []

    def connect(inst_id, inst, timeout_s=600):
        calls.append(inst_id)
        if len(calls) == 2:  # the second host is reachable
            prov.instances[inst_id]["endpoints"] = [{"host": "local", "port": 0, "kind": "direct"}]
            inst = prov.instances[inst_id]
        return real(inst_id, inst, timeout_s=1)

    monkeypatch.setattr(lx, "connect", connect)
    inst_id, _, _ = lx.up_with_retry("rtx4090", None, False, created=[])
    assert [c["offer"] for c in prov.created] == [101, 102]
    assert prov.destroyed == [calls[0]] and inst_id == calls[1]
    lx.destroy(inst_id)


def test_stuck_image_pull_gives_up_and_keeps_price(tmp_path, home):
    proj = make_project(tmp_path / "proj", smoke=False)
    proj.cfg["image_pull_timeout_min"] = 0.01  # 0.6 s
    prov = FakeProvider(stuck_offers={101})
    lx = LocalLauncher(proj, prov, tmp_path / "box")
    inst_id, _, _ = lx.up_with_retry("rtx4090", None, False, created=[])
    assert [c["offer"] for c in prov.created] == [101, 102] and len(prov.destroyed) == 1
    assert lx.state.get(inst_id)["dph"] == 0.45  # price of the retried offer is kept, not "price on start"
    lx.destroy(inst_id)

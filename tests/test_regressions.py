"""One test per bug found in the pre-release review. No network, no GPU: fake provider, local "box"."""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tarfile
import threading
import time

import pytest
from conftest import BASH, FakeProvider, LocalLauncher, LocalRemote, needs_bash
from test_session import make_project

from gpuhop import cli, project
from gpuhop.codepush import CRED_HELPER, CodePackage, list_files
from gpuhop.jobs import expand
from gpuhop.launcher import Launcher, RunnerDied
from gpuhop.providers.vast import VastProvider


@pytest.fixture
def setup(tmp_path, home):
    proj = make_project(tmp_path / "proj")
    os.environ.pop("VAST_API_KEY", None)
    prov = FakeProvider()
    return proj, prov, LocalLauncher(proj, prov, tmp_path / "box"), tmp_path / "box"


def live(lx, prov, status="running"):
    i = prov.create({}, prov.offers[0], {}, "", "label")
    prov.instances[i]["status"] = status
    lx.state.remember(i, gpu="rtx4090", created=time.time(), dph=0.4)
    return i


def raises(exc):
    def fn(*_a, **_k):
        raise exc
    return fn


# ------------------------------------------------------------------------------------------- code on the box
@needs_bash
def test_same_size_edit_is_not_shadowed_by_stale_bytecode(setup):
    proj, _, lx, box = setup
    (proj.root / "mod.py").write_text("VALUE = 1\n")
    (proj.root / "main.py").write_text("import mod; print(mod.VALUE)\n")
    env = {k: v for k, v in os.environ.items() if k != "PYTHONDONTWRITEBYTECODE"}

    def run():
        return subprocess.run([sys.executable, "main.py"], cwd=box / "project", env=env, capture_output=True,
                              text=True).stdout.strip()

    CodePackage(proj).push(lx.rem)
    assert run() == "1"
    (proj.root / "mod.py").write_text("VALUE = 2\n")
    CodePackage(proj).push(lx.rem)
    assert run() == "2"


def test_every_env_file_stays_local(setup):
    proj, *_ = setup
    for name in (".envrc", ".env_prod", ".env-local", ".env.local"):
        (proj.root / name).write_text("SECRET=1\n")
    assert not [f for f in list_files(proj) if f.startswith(".env")]


@pytest.mark.skipif(BASH is None, reason="needs bash")
def test_submodule_files_are_shipped(tmp_path):
    g = ["git", "-c", "user.email=t@t", "-c", "user.name=t", "-c", "protocol.file.allow=always"]
    sub, main = tmp_path / "sub", tmp_path / "main"
    sub.mkdir(), main.mkdir()
    (sub / "lib.py").write_text("x = 1\n")
    (main / "gpuhop.yaml").write_text("{}\n")
    subprocess.run([*g, "init", "-q"], cwd=sub, check=True)
    subprocess.run([*g, "add", "-A"], cwd=sub, check=True)
    subprocess.run([*g, "commit", "-qm", "i"], cwd=sub, check=True)
    subprocess.run([*g, "init", "-q"], cwd=main, check=True)
    subprocess.run([*g, "submodule", "add", "-q", str(sub), "vendor"], cwd=main, check=True, capture_output=True)
    assert "vendor/lib.py" in list_files(project.load(main))


@needs_bash
def test_github_token_never_reaches_argv_or_errors(setup, monkeypatch):
    *_, box = setup
    monkeypatch.setenv("GPUB_GH_TOKEN", "ghp_CANARY")
    seen = {}

    class Failing(LocalRemote):
        def argv(self, cmd, cwd="", tty=False):
            seen["argv"] = cmd
            return [BASH, "-c", "cat >/dev/null; exit 1"]

    pkg = object.__new__(CodePackage)
    pkg.git = {"repo": "me/r", "git_sha": "abc"}
    with pytest.raises(RuntimeError) as e:
        pkg._push_git(Failing(box))
    assert "ghp_CANARY" not in str(e.value) and "ghp_CANARY" not in seen["argv"]


@pytest.mark.skipif(BASH is None, reason="needs bash")
def test_credential_helper_hands_git_the_token(monkeypatch):
    monkeypatch.setenv("GPUHOP_GH_TOKEN", "ghp_CANARY")
    out = subprocess.run(["git", "-c", "credential.helper=", "-c", f"credential.helper={CRED_HELPER}", "credential", "fill"],
                         input="protocol=https\nhost=github.com\n\n", capture_output=True, text=True, check=True).stdout
    assert "username=x-access-token" in out and "password=ghp_CANARY" in out


def test_instance_credentials_are_redacted():
    d = project.redact({"jupyter_token": "t", "instance_api_key": "k", "nested": [{"Password": "p"}], "gpu_name": "x"})
    assert d == {"jupyter_token": "***", "instance_api_key": "***", "nested": [{"Password": "***"}], "gpu_name": "x"}


# ------------------------------------------------------------------------------------------- pulling
@needs_bash
def test_pull_tolerates_a_file_being_written(setup):
    _, _, lx, box = setup
    res = box / "project" / "results" / "e"
    res.mkdir(parents=True)
    f = res / "m.bin"
    f.write_bytes(os.urandom(30_000_000))
    stop = threading.Event()

    def writer():
        with open(f, "ab") as h:
            while not stop.is_set():
                h.write(b"x" * 4096)
                h.flush()
                time.sleep(0.001)

    t = threading.Thread(target=writer)
    t.start()
    try:
        assert lx.rem.pull_dir((box / "project" / "results").as_posix(), box.parent / "out")
    finally:
        stop.set()
        t.join()
    assert (box.parent / "out" / "e" / "m.bin").stat().st_size >= 30_000_000


def _crafted(path, members):
    with tarfile.open(path, "w:gz") as tf:
        for name, kind, data in members:
            ti = tarfile.TarInfo(name)
            if kind == "sym":
                ti.type, ti.linkname = tarfile.SYMTYPE, data
                tf.addfile(ti)
            else:
                ti.size = len(data)
                tf.addfile(ti, io.BytesIO(data))


class _TarRemote(LocalRemote):
    tgz = None

    def argv(self, cmd, cwd="", tty=False):
        return [BASH, "-c", f"cat '{self.tgz}'" if "tar -C" in cmd else "true"]


@needs_bash
def test_pull_cannot_write_outside_the_destination(tmp_path):
    t = tmp_path / "evil.tgz"
    _crafted(t, [("../escaped.txt", "file", b"x"), ("ok.txt", "file", b"ok"), ("abs", "sym", "/etc"),
                 ("abs/owned.txt", "file", b"x")])
    rem = _TarRemote(tmp_path / "box")
    rem.tgz = t.as_posix()
    dest = tmp_path / "a" / "dest"
    assert rem.pull_dir("/x", dest)
    assert (dest / "ok.txt").read_bytes() == b"ok"
    assert not (dest.parent / "escaped.txt").exists() and not (dest / "abs").is_symlink()


@needs_bash
def test_pull_materialises_links_and_survives_dangling_ones(tmp_path):
    t = tmp_path / "l.tgz"
    _crafted(t, [("model_10.bin", "file", b"weights"), ("best.bin", "sym", "model_10.bin"), ("gone", "sym", "nope"),
                 ("loop", "sym", "loop")])
    rem = _TarRemote(tmp_path / "box")
    rem.tgz = t.as_posix()
    dest = tmp_path / "dest"
    assert rem.pull_dir("/x", dest)
    assert (dest / "best.bin").read_bytes() == b"weights" and not (dest / "best.bin").is_symlink()
    assert not (dest / "gone").exists()


# ------------------------------------------------------------------------------------------- cleanup
@needs_bash
def test_api_blip_at_the_end_does_not_destroy_unpulled_results(setup, monkeypatch):
    _, prov, lx, _ = setup
    i = live(lx, prov)
    monkeypatch.setattr(prov, "show", raises(ConnectionError("blip")))
    assert lx._finish(i, keep=False) is False
    assert i not in prov.destroyed


@needs_bash
@pytest.mark.parametrize("status", ["loading", "unknown", "stopped"])
def test_unpulled_box_in_a_transient_state_is_kept(setup, status):
    _, prov, lx, _ = setup
    i = live(lx, prov, status)
    assert lx._finish(i, keep=False) is False and i in prov.instances


@needs_bash
def test_box_without_work_is_destroyed_even_if_not_running(setup):
    _, prov, lx, _ = setup
    i = live(lx, prov, "loading")
    assert lx._finish(i, keep=False, had_work=False) is True and i in prov.destroyed


@needs_bash
def test_ctrl_c_during_final_pull_retries_then_abandons(setup, monkeypatch):
    _, prov, lx, _ = setup
    i = live(lx, prov)
    calls = []

    def once(*_a):
        calls.append(1)
        if len(calls) == 1:
            raise KeyboardInterrupt

    monkeypatch.setattr(lx, "pull", once)
    assert lx._finish(i, keep=False) is True and len(calls) == 2 and i in prov.destroyed
    j = live(lx, prov)
    monkeypatch.setattr(lx, "pull", raises(KeyboardInterrupt()))
    assert lx._finish(j, keep=False) is True and j in prov.destroyed


@needs_bash
def test_failed_destroy_is_reported_and_remembered(setup, monkeypatch, capsys):
    _, prov, lx, _ = setup
    i = live(lx, prov)
    monkeypatch.setattr(lx, "pull", lambda *_a: None)
    monkeypatch.setattr(prov, "destroy", raises(ConnectionError("down")))
    lx.poll_running_s = 0
    assert lx._finish(i, keep=False) is False
    assert "STILL BE BILLING" in capsys.readouterr().out and lx.state.get(i)


def test_vast_destroy_failure_raises_and_keeps_the_record(tmp_path):
    (tmp_path / "gpuhop.yaml").write_text("{}\n")

    class Client:
        def destroy_instance(self, id):
            return {"success": False, "msg": "nope"}

    v = VastProvider()
    v._client = Client()
    lx = Launcher(project.load(tmp_path), v)
    lx.state.remember(7, gpu="rtx4090")
    with pytest.raises(RuntimeError, match="nope"):
        lx.destroy(7)
    assert lx.state.get(7)


def test_vast_show_only_treats_404_as_gone():
    class Err(Exception):
        def __init__(self, code):
            self.response = type("R", (), {"status_code": code})

    class Client:
        code = 404

        def show_instance(self, id):
            raise Err(self.code)

    v = VastProvider()
    v._client = Client()
    assert v.show(1) is None
    Client.code = 500
    with pytest.raises(Err):
        v.show(1)


# ------------------------------------------------------------------------------------------- job runner
def _spec(box, jobs, **kw):
    for d in ("jobs", "logs", "project"):
        (box / d).mkdir(parents=True, exist_ok=True)
    f = box / "jobs" / "x.json"
    f.write_text(json.dumps({"exp": "x", "project_dir": (box / "project").as_posix(), "logs_dir": (box / "logs").as_posix(),
                             "env_file": "/nope", "shell": BASH, "jobs": jobs, **kw}))
    return f


def _runner(f):
    from importlib import resources

    script = str(resources.files("gpuhop").joinpath("runtime", "jobrunner.py"))
    return subprocess.run([sys.executable, script, str(f)], capture_output=True, text=True)


@needs_bash
def test_runner_crash_is_written_to_status(setup):
    _, _, _, box = setup
    f = _spec(box, [{"name": "a", "cmd": "true", "results": "r/a", "timeout_s": 5}], shell="/nonexistent/bash")
    assert _runner(f).returncode == 1
    st = json.loads((box / "jobs" / "x.status.json").read_text())
    assert st["state"] == "failed" and st["jobs"]["a"]["state"] == "failed" and st["error"]
    assert not (box / "jobs" / "x.heartbeat").exists()


@needs_bash
def test_dead_runner_is_detected_before_the_deadline(setup):
    _, _, lx, box = setup
    (box / "jobs").mkdir(parents=True)
    (box / "jobs" / "x.status.json").write_text(json.dumps({"exp": "x", "state": "running", "jobs": {}}))
    lx.runner_stale_s = 0.5
    with pytest.raises(RunnerDied):
        lx.wait_exp(lx.rem, "x", deadline=time.time() + 30)


@needs_bash
def test_session_restarts_a_dead_runner_and_finishes(setup, monkeypatch):
    proj, prov, lx, _ = setup
    real, calls = lx.wait_exp, []

    def dies_once(*a, **k):
        calls.append(1)
        if len(calls) == 1:
            lx.rem.wait_all()
            raise RunnerDied("killed")
        return real(*a, **k)

    monkeypatch.setattr(lx, "wait_exp", dies_once)
    assert lx.session("rtx4090", [str(proj.root / "exp" / "e1.yaml")]) == 0
    lx.rem.wait_all()
    assert len(calls) == 2 and len(prov.destroyed) == 1


@needs_bash
def test_rerun_after_code_change_runs_jobs_again(setup):
    proj, _, lx, box = setup
    f = proj.root / "exp" / "c.yaml"
    f.write_text("name: c\njobs: [{name: a, cmd: 'echo ran >> \"$GPUB_RESULTS/count.txt\"'}]\n")
    i, _, rem = lx.up("rtx4090")
    count = box / "project" / "results" / "c" / "a" / "count.txt"

    def run():
        lx.start_exp(rem, str(f), None)
        lx.rem.wait_all()
        return count.read_text().count("ran")

    assert run() == 1
    assert run() == 1  # unchanged code: finished jobs are skipped (resume)
    (proj.root / "job.py").write_text("# changed\n")
    lx.sync(i)
    assert run() == 2
    lx.destroy(i)


@needs_bash
def test_status_from_an_earlier_run_is_not_taken_for_this_one(setup):
    _, _, lx, box = setup
    (box / "jobs").mkdir(parents=True)
    (box / "jobs" / "x.status.json").write_text(json.dumps({"exp": "x", "state": "done", "token": "old", "jobs": {}}))
    lx._tokens["x"] = "new"
    with pytest.raises(TimeoutError):
        lx.wait_exp(lx.rem, "x", deadline=time.time() + 0.5)


@needs_bash
def test_second_experiment_cannot_share_the_gpu(setup):
    proj, _, lx, box = setup
    i, _, rem = lx.up("rtx4090")
    (box / "jobs").mkdir(exist_ok=True)
    (box / "jobs" / "other.heartbeat").touch()
    with pytest.raises(RuntimeError, match="still running"):
        lx.start_exp(rem, str(proj.root / "exp" / "e1.yaml"), None)
    lx.destroy(i)


# ------------------------------------------------------------------------------------------- CLI
@needs_bash
def test_wait_works_right_after_start(setup, monkeypatch, capsys):
    _, prov, lx, _ = setup
    i, _, _ = lx.up("rtx4090")
    prov.instances[i].update(status="exited", host=None, port=None, endpoints=[])
    real, polls = prov.show, []

    def comes_back(x):
        polls.append(1)
        if len(polls) == 3:
            prov.instances[i].update(status="running", host="local", port=0,
                                     endpoints=[{"host": "local", "port": 0, "kind": "direct"}])
        return real(x)

    monkeypatch.setattr(prov, "show", comes_back)
    monkeypatch.setattr(cli, "_launcher", lambda: lx)
    assert cli.cmd_wait(type("A", (), {"id": i})) == 0
    lx.destroy(i)


@needs_bash
def test_ls_keeps_records_of_live_instances_and_handles_null_names(setup, monkeypatch, capsys):
    _, prov, lx, _ = setup
    i = live(lx, prov)
    monkeypatch.setattr(cli, "_launcher", lambda: lx)
    prov.instances[i]["gpu_name"] = None
    assert cli.cmd_ls(None) == 0
    monkeypatch.setattr(prov, "list", list)
    cli.cmd_ls(None)
    assert lx.state.get(i)  # the list was empty but show() still finds it
    prov.instances.pop(i)
    cli.cmd_ls(None)
    assert not lx.state.get(i)


@needs_bash
def test_up_honours_wait_offer(setup, monkeypatch, capsys):
    _, prov, lx, _ = setup
    offers, prov.offers = prov.offers, []
    real, calls = prov.search, []

    def appears(*a, **k):
        calls.append(1)
        if len(calls) == 3:
            prov.offers = offers
        return real(*a, **k)

    monkeypatch.setattr(prov, "search", appears)
    monkeypatch.setenv("GPUB_WAIT_OFFER_MIN", "0.5")
    monkeypatch.setattr(cli, "_launcher", lambda: lx)
    args = type("A", (), {"gpu": "rtx4090", "offer": None, "allow_dirty": False, "no_wait": False})
    assert cli.cmd_up(args) == 0
    lx.destroy(next(iter(prov.instances)))


def test_redirected_cp1252_stdout_survives_unicode(monkeypatch):
    out = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
    monkeypatch.setattr(sys, "stdout", out)
    cli._safe_stdio()
    print("\u2501\u2501 50%")
    out.flush()
    monkeypatch.undo()
    assert b"50%" in out.buffer.getvalue()


def test_doctor_expands_tilde_in_ssh_key(tmp_path, monkeypatch):
    (tmp_path / ".ssh").mkdir()
    (tmp_path / ".ssh" / "k").write_text("private")
    (tmp_path / ".ssh" / "k.pub").write_text("ssh-ed25519 AAA me\n")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("GPUB_SSH_KEY", "~/.ssh/k")
    assert "registered" in cli._check_ssh_key(["ssh-ed25519 AAA"])


# ------------------------------------------------------------------------------------------- config
def test_yaml_is_read_as_utf8_on_every_locale(tmp_path):
    f = tmp_path / "e.yaml"
    f.write_bytes("jobs: [{name: a, cmd: 'echo caf\u00e9 \u2501'}]\n".encode())
    assert expand(f)["jobs"][0]["cmd"] == "echo caf\u00e9 \u2501"


@pytest.mark.parametrize("text, match", [
    ("cmd: run\njobs: [{matrix: {lr: []}}]\n", "empty"),
    ("cmd: run\njobs: [{matrix: {results: [x]}}]\n", "clash"),
    ("jobs: [{name: '..', cmd: x}]\n", "escape"),
])
def test_bad_experiments_are_rejected(tmp_path, text, match):
    f = tmp_path / "z.yaml"
    f.write_text(text)
    with pytest.raises(ValueError, match=match):
        expand(f)


@pytest.mark.parametrize("cfg", [
    "results: {exlude: ['*.ckpt']}\n", "code: {exclued: [data]}\n", "search: {cpu_rgex: x}\n", "code: tarball\n",
    "gpus: {rtx4090: {qeury: x}}\n", "results: {local_dir: 'a/../../outside'}\n", "results: {local_dir: 'C:/x'}\n",
    "results: {remote_dir: .}\n",
])
def test_bad_config_is_rejected(tmp_path, cfg):
    (tmp_path / "gpuhop.yaml").write_text(cfg)
    with pytest.raises(SystemExit):
        project.load(tmp_path)


def test_dotenv_comments_export_quotes_and_bom(tmp_path, monkeypatch):
    for k in ("GPUB_T1", "GPUB_T2", "GPUB_T3", "GPUB_T4"):
        monkeypatch.delenv(k, raising=False)
    f = tmp_path / ".env"
    f.write_bytes(b"\xef\xbb\xbfGPUB_T1=abc # note\nexport GPUB_T2=y\nGPUB_T3='a # b' # c\nGPUB_T4=p#q\n")
    project.load_dotenv(f)
    got = [os.environ.pop(f"GPUB_T{i}") for i in (1, 2, 3, 4)]
    assert got == ["abc", "y", "a # b", "p#q"]

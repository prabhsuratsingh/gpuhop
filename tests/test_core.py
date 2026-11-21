"""Config, catalog, job expansion, code packaging and scaffolding: no network, no GPU."""

from __future__ import annotations

import shutil
import subprocess
import tarfile
import textwrap

import pytest
import yaml

from gpuhop import catalog, project
from gpuhop.codepush import CodePackage, list_files
from gpuhop.jobs import expand
from gpuhop.scaffold import init, templates


def write(p, text):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(textwrap.dedent(text))
    return p


# --------------------------------------------------------------------------------------------- project / catalog
def test_find_root_from_subdir_and_defaults(tmp_path):
    write(tmp_path / "gpuhop.yaml", "project: demo\ndisk_gb: 80\n")
    (tmp_path / "a" / "b").mkdir(parents=True)
    p = project.load(tmp_path / "a" / "b")
    assert p.root == tmp_path.resolve() and p.name == "demo"
    assert p.cfg["disk_gb"] == 80 and p.cfg["code"]["mode"] == "tarball" and p.cfg["results"]["exclude"] == ["*.pt"]


def test_unknown_key_and_bad_mode_rejected(tmp_path):
    write(tmp_path / "gpuhop.yaml", "dsik_gb: 80\n")
    with pytest.raises(SystemExit, match="unknown key"):
        project.load(tmp_path)
    write(tmp_path / "gpuhop.yaml", "code: {mode: rsync}\n")
    with pytest.raises(SystemExit, match="code.mode"):
        project.load(tmp_path)


def test_no_config(tmp_path):
    with pytest.raises(SystemExit, match="gpuhop init"):
        project.find_root(tmp_path)


def test_dotenv_does_not_override_real_env(tmp_path, monkeypatch):
    write(tmp_path / "gpuhop.yaml", "{}\n")
    write(tmp_path / ".env", "VAST_API_KEY=from_file\nGPUB_TEST_X='quoted'\n")
    monkeypatch.setenv("VAST_API_KEY", "from_env")
    monkeypatch.delenv("GPUB_TEST_X", raising=False)
    project.load(tmp_path)
    import os

    assert os.environ["VAST_API_KEY"] == "from_env" and os.environ["GPUB_TEST_X"] == "quoted"
    monkeypatch.delenv("GPUB_TEST_X")


def test_gpu_spec_merges_project_over_builtin(tmp_path):
    write(tmp_path / "gpuhop.yaml", """\
        image: my/image:1
        image_digest: sha256:abc
        env: {A: "1"}
        search: {cpu_regex: EPYC}
        gpus:
          rtx4090: {env: {B: "2"}}
          a6000: {query: "gpu_name=RTX_A6000", image: other/image:2, disk_gb: 30}
        """)
    p = project.load(tmp_path)
    s = catalog.gpu_spec(p, "rtx4090")
    assert s["image"] == "my/image:1" and s["image_digest"] == "sha256:abc"
    assert "gpu_name=RTX_4090" in s["query"] and s["query"].endswith("disk_space>=60")
    assert s["env"] == {"PYTHONUNBUFFERED": "1", "A": "1", "B": "2"} and s["cpu_regex"] == "EPYC"
    s2 = catalog.gpu_spec(p, "a6000")
    assert s2["image"] == "other/image:2" and s2["image_digest"] is None and s2["disk_gb"] == 30
    with pytest.raises(SystemExit, match="unknown GPU alias"):
        catalog.gpu_spec(p, "tpu")


def test_vast_search_filters_cpu_and_driver():
    from gpuhop.providers.vast import VastProvider

    raw = [{"id": 1, "dph_total": 0.3, "cpu_name": "Xeon", "driver_version": "580.119.02"},
           {"id": 2, "dph_total": 0.4, "cpu_name": "AMD EPYC 7B12", "driver_version": "595.104.02"},
           {"id": 3, "dph_total": 0.5, "cpu_name": "AMD EPYC 9654", "driver_version": "580.126.09"}]

    class Client:
        def search_offers(self, **kw):
            return list(raw)

    v = VastProvider()
    v._client = Client()
    base = {"query": "q", "disk_gb": 60}
    assert [o["id"] for o in v.search(base)] == [1, 2, 3]
    assert [o["id"] for o in v.search({**base, "driver_regex": r"^580\."})] == [1, 3]
    assert [o["id"] for o in v.search({**base, "driver_regex": r"^580\.", "cpu_regex": "EPYC"})] == [3]
    assert v.search(base)[0]["driver"] == "580.119.02"


def test_redact():
    d = project.redact({"a": 1, "GH_TOKEN": "x", "env": {"VAST_API_KEY": "k", "ok": [{"GPUB_GH_TOKEN": "t"}]}})
    assert d == {"a": 1, "GH_TOKEN": "***", "env": {"VAST_API_KEY": "***", "ok": [{"GPUB_GH_TOKEN": "***"}]}}


# --------------------------------------------------------------------------------------------- jobs
def test_expand_matrix_templates_and_env(tmp_path):
    f = write(tmp_path / "sweep.yaml", """\
        cmd: python train.py --lr {lr} --seed {seed} --out {results} {overrides}
        timeout_min: 7
        jobs:
          - matrix: {lr: [0.1, 0.01], seed: [0, 1]}
            overrides: {model.width: 64, flag: true}
          - matrix: {lr: [0.5], seed: [3]}
            name: "custom_lr{lr}_s{seed}"
            allow_fail: true
            timeout_min: 2
          - name: eval
            cmd: echo {{literal}} {exp}
        """)
    s = expand(f, "results")
    names = [j["name"] for j in s["jobs"]]
    assert names == ["0.1_s0", "0.1_s1", "0.01_s0", "0.01_s1", "custom_lr0.5_s3", "eval"]
    j0 = s["jobs"][0]
    assert j0["cmd"] == "python train.py --lr 0.1 --seed 0 --out results/sweep/0.1_s0 model.width=64 flag=true"
    assert j0["results"] == "results/sweep/0.1_s0" and j0["timeout_s"] == 420 and not j0["allow_fail"]
    assert s["jobs"][4]["allow_fail"] and s["jobs"][4]["timeout_s"] == 120
    assert s["jobs"][5]["cmd"] == "echo {literal} sweep"
    assert s["exp"] == "sweep"


def test_expand_errors(tmp_path):
    with pytest.raises(ValueError, match="unknown placeholder"):
        expand(write(tmp_path / "a.yaml", "jobs: [{matrix: {x: [1]}, cmd: 'run {y}'}]\n"))
    with pytest.raises(ValueError, match="duplicate"):
        expand(write(tmp_path / "b.yaml", "cmd: run\njobs: [{matrix: {x: [1]}}, {matrix: {x: [1]}}]\n"))
    with pytest.raises(ValueError, match="no `cmd`"):
        expand(write(tmp_path / "c.yaml", "jobs: [{matrix: {x: [1]}}]\n"))
    with pytest.raises(ValueError, match="unknown key"):
        expand(write(tmp_path / "d.yaml", "jobs: [{name: a, cmd: b, timeout: 3}]\n"))
    with pytest.raises(ValueError, match="experiment name"):
        expand(write(tmp_path / "e.yaml", "name: 'a b'\njobs: [{name: a, cmd: b}]\n"))


def test_matrix_values_are_shell_quoted(tmp_path):
    s = expand(write(tmp_path / "q.yaml", "cmd: run {msg}\njobs: [{matrix: {msg: ['a b; rm -rf /']}, name: q}]\n"))
    assert s["jobs"][0]["cmd"] == "run 'a b; rm -rf /'"


# --------------------------------------------------------------------------------------------- code push
def _proj(tmp_path, cfg="{}\n"):
    write(tmp_path / "gpuhop.yaml", cfg)
    write(tmp_path / "train.py", "print('hi')\n")
    write(tmp_path / "pkg" / "mod.py", "x = 1\n")
    write(tmp_path / ".env", "VAST_API_KEY=secret\n")
    write(tmp_path / "results" / "rtx4090" / "old.json", "{}")
    write(tmp_path / "data" / "big.bin", "x" * 100)
    write(tmp_path / "pkg" / "__pycache__" / "mod.pyc", "")
    return project.load(tmp_path)


def test_list_files_without_git(tmp_path):
    p = _proj(tmp_path, "code: {exclude: ['data/*']}\n")
    files = list_files(p)
    assert files == ["gpuhop.yaml", "pkg/mod.py", "train.py"]


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_list_files_respects_gitignore_and_records_dirty(tmp_path):
    p = _proj(tmp_path)
    write(tmp_path / ".gitignore", "data/\n__pycache__/\n")
    write(tmp_path / ".env.local", "X=1\n")
    g = ["git", "-c", "user.email=t@t", "-c", "user.name=t"]
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run([*g, "add", "-A"], cwd=tmp_path, check=True)
    subprocess.run([*g, "commit", "-qm", "init"], cwd=tmp_path, check=True)
    (tmp_path / "train.py").write_bytes(b"print('changed')\r\n")
    (tmp_path / "new_untracked.sh").write_bytes(b"set -e\r\necho hi\r\n")
    files = list_files(p)
    # .env is shipped by nothing, even though it was committed; results/ (pulled data) never goes back up
    assert files == [".gitignore", "gpuhop.yaml", "new_untracked.sh", "pkg/mod.py", "train.py"]
    pkg = CodePackage(p)
    assert pkg.provenance["dirty"] is True and pkg.provenance["diff_sha256"] and len(pkg.sha) == 40
    tar = pkg.build()
    with tarfile.open(tar) as tf:
        assert sorted(tf.getnames()) == files
        assert tf.extractfile("train.py").read() == b"print('changed')\r\n"  # only shell scripts are normalised
        assert tf.extractfile("new_untracked.sh").read() == b"set -e\necho hi\n"
        assert tf.getmember("new_untracked.sh").mode == 0o755
    h1 = pkg.provenance["content_sha256"]
    pkg.cleanup()
    assert not tar.exists()
    assert CodePackage(p).build() and h1  # deterministic hash is recomputed per build


def test_git_mode_requires_repo(tmp_path):
    p = _proj(tmp_path, "code: {mode: git}\n")
    with pytest.raises(SystemExit, match="not a git repository"):
        CodePackage(p)


# --------------------------------------------------------------------------------------------- scaffold
@pytest.mark.parametrize("template", ["pytorch", "isaaclab"])
def test_init_templates_load_and_expand(tmp_path, template):
    report = init(tmp_path / "proj", template, name="demo")
    root = tmp_path / "proj"
    assert (root / ".env.example").exists() and not (root / "env.example").exists()
    assert ".env" in (root / ".gitignore").read_text().splitlines()
    assert any("write  gpuhop.yaml" in r for r in report)
    p = project.load(root)
    assert p.name == "demo"
    assert (root / p.cfg["setup"]).exists()
    for exp in (root / "exp").glob("*.yaml"):
        s = expand(exp, p.cfg["results"]["remote_dir"])
        assert s["jobs"], exp
    files = list_files(p)
    assert "gpuhop.yaml" in files and p.cfg["setup"] in files
    assert not any(f.startswith(".env") for f in files)  # .env* (secrets) never ship


def test_init_never_overwrites(tmp_path):
    write(tmp_path / "train.py", "mine\n")
    write(tmp_path / ".gitignore", "node_modules/\n.env\n")
    report = init(tmp_path, "pytorch")
    assert (tmp_path / "train.py").read_text() == "mine\n"
    assert any("skip   train.py" in r for r in report)
    gi = (tmp_path / ".gitignore").read_text().splitlines()
    assert gi[0] == "node_modules/" and gi.count(".env") == 1 and ".gpuhop/" in gi
    again = init(tmp_path, "pytorch")
    assert any("keep   .gitignore" in r for r in again)
    assert yaml.safe_load((tmp_path / "gpuhop.yaml").read_text())["project"] == tmp_path.name


def test_templates_listed():
    t = templates()
    assert {"pytorch", "isaaclab"} <= set(t) and all(t.values())

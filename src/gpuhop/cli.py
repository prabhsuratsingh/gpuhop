"""gpuhop: run your project's experiments on a rented cloud GPU from your laptop.

Start a project:
    gpuhop init [--template pytorch|isaaclab] [dir]   # writes gpuhop.yaml, setup.sh, exp/smoke.yaml, ...
    gpuhop doctor                                     # checks API key, ssh key, config, code push

One-shot (the usual way):
    gpuhop offers rtx4090                             # what's available / price
    gpuhop exp rtx4090 exp/smoke.yaml exp/sweep.yaml
        -> rent -> push code -> setup -> run all jobs -> pull results/logs/videos -> destroy (also on failure/Ctrl-C)

Interactive / iterative:
    gpuhop up rtx4090     gpuhop ssh      gpuhop sync (push the current code)
    gpuhop run exp/x.yaml gpuhop status   gpuhop pull     gpuhop down
Instance ids default to the most recent instance; a GPU alias (e.g. `rtx4090`) also works.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

from . import __version__, catalog, project
from .codepush import CodePackage
from .project import env, redact

BASE_SKIP = {"pip", "setuptools", "wheel"}


def _launcher():
    from .launcher import Launcher

    return Launcher(project.load())


# ----------------------------------------------------------------------------------------------------
def cmd_init(a):
    from .scaffold import init, templates

    if a.list:
        for name, desc in templates().items():
            print(f"  {name:<10} {desc}")
        return 0
    target = Path(a.dir)
    if (target / project.CONFIG_NAME).exists() and not a.force:
        print(f"{target / project.CONFIG_NAME} already exists; existing files are kept (use --force to overwrite)")
    for line in init(target, a.template, a.name, a.force):
        print(line)
    where = "" if target.resolve() == Path.cwd().resolve() else f"cd {a.dir}\n  "
    print(f"\nnext:\n  {where}copy .env.example to .env and set VAST_API_KEY\n  gpuhop doctor\n"
          f"  gpuhop exp rtx4090 exp/smoke.yaml")
    return 0


def _check_ssh_key(registered_keys: list[str]) -> str:
    """The key ssh will actually offer must be one registered with the provider."""
    registered = {k.split()[1] for k in registered_keys if len(k.split()) > 1}
    own = env("SSH_KEY")
    if own:
        own = os.path.expanduser(own)
        pub = Path(own + ".pub")
        if not Path(own).exists() or not pub.exists():
            raise RuntimeError(f"GPUB_SSH_KEY={own}: private key or {pub.name} not found")
        if pub.read_text(encoding="utf-8").split()[1] not in registered:
            raise RuntimeError(f"{pub} is not registered with the provider (Vast: cloud.vast.ai/manage-keys)")
        return f"{Path(own).name} (GPUB_SSH_KEY) is registered"
    # Without GPUB_SSH_KEY, OpenSSH only offers its default key names.
    defaults = ["id_rsa", "id_ecdsa", "id_ed25519"]
    home = Path.home() / ".ssh"
    for name in defaults:
        pub = home / f"{name}.pub"
        if pub.exists() and len(pub.read_text().split()) > 1 and pub.read_text().split()[1] in registered:
            return f"~/.ssh/{name} is registered"
    matches = [p for p in home.glob("*.pub") if len(p.read_text().split()) > 1 and p.read_text().split()[1] in registered]
    hint = (f"; set GPUB_SSH_KEY={matches[0].with_suffix('')} in .env" if matches else
            "; create one (ssh-keygen -t ed25519) and add the .pub at cloud.vast.ai/manage-keys")
    raise RuntimeError("no default key (" + ", ".join(defaults) + ") is registered" + hint)


def cmd_doctor(a):
    ok = True

    def check(name, fn):
        nonlocal ok
        try:
            print(f"  [ok]   {name}: {fn()}")
        except BaseException as e:  # SystemExit included
            ok = False
            print(f"  [FAIL] {name}: {e}")

    holder = {}

    def load():
        holder["proj"] = p = project.load()
        return f"{p.root / project.CONFIG_NAME} (project {p.name}, provider {p.cfg['provider']}, code {p.cfg['code']['mode']})"

    check("config", load)
    if "proj" not in holder:
        print("doctor: fix the config first")
        return 1
    proj = holder["proj"]
    from .launcher import Launcher

    lx = Launcher(proj)
    check("provider api", lx.provider.check)
    keys: list[str] = []

    def ssh_keys():
        got = lx.provider.ssh_keys()
        if got is None:
            return "managed outside the provider"
        if not got:
            raise RuntimeError("none registered: add your public key at cloud.vast.ai/manage-keys")
        keys.extend(got)
        return f"{len(got)} registered"

    def setup_script():
        s = proj.cfg["setup"]
        if s and not (proj.root / s).exists():
            raise RuntimeError(f"{s} not found in the project")
        return s or "none"

    check("ssh keys", ssh_keys)
    if keys:
        check("ssh key matches", lambda: _check_ssh_key(keys))
    check("ssh client", lambda: subprocess.run(["ssh", "-V"], capture_output=True, text=True).stderr.strip())
    check("setup script", setup_script)
    check("code push", lambda: CodePackage(proj).describe())
    check("gpu catalog", lambda: ", ".join(sorted(catalog.catalog(proj)["gpus"])))
    print("doctor:", "all good" if ok else "fix the failures above")
    return 0 if ok else 1


def cmd_offers(a):
    lx = _launcher()
    offers = lx.provider.search(catalog.gpu_spec(lx.proj, a.gpu), limit=a.n, cpu=env("CPU_REGEX"))
    print(f"{'offer':>10} {'gpu':<10} {'$/hr':>6} {'rel':>5} {'cuda':>5} {'down':>6} {'cores':>5} {'driver':>10}  {'cpu':<38} location")
    for o in offers:
        cpu = (o["cpu_name"] or "?").replace(" Processor", "").replace("-Core", "c")[:38]
        print(f"{o['id']:>10} {o['gpu_name'] or '?':<10} {o['dph'] or 0:>6.3f} {o['reliability'] or 0:>5.3f} {o['cuda']!s:>5} "
              f"{o['inet_down'] or 0:>6.0f} {o['cpu_cores'] or 0:>5.0f} {o.get('driver') or '?':>10}  {cpu:<38} {o['location']}")
    if not offers:
        print(f"no offers match: {catalog.gpu_spec(lx.proj, a.gpu)['query']}")
    return 0


def cmd_up(a):
    lx = _launcher()
    created: list = []
    try:
        inst_id, _, _ = lx.up_with_retry(a.gpu, a.offer, a.allow_dirty, created, wait=not a.no_wait)
    except BaseException:
        if created:
            print(f"instance {created[-1]} is still up for debugging: gpuhop logs {created[-1]} --bootstrap, "
                  f"then gpuhop down {created[-1]} -y")
        raise
    print(f"{'bootstrapping (gpuhop wait)' if a.no_wait else 'ready'}: gpuhop ssh {inst_id}")
    return 0


def cmd_smoke(a):
    return _launcher().smoke(a.gpu, a.offer, a.allow_dirty, a.keep)


def cmd_exp(a):
    return _launcher().session(a.gpu, a.exp, offer_id=a.offer, instance=a.instance, keep=a.keep,
                               max_hours=a.max_hours, allow_dirty=a.allow_dirty)


def cmd_ls(a):
    lx = _launcher()
    rows = lx.provider.list()
    local = lx.state.load()
    if not rows:
        print(f"no instances on your {lx.provider.name} account")
    for r in rows:
        age = (time.time() - (r.get("start_date") or time.time())) / 3600
        tag = "this project" if str(r["id"]) in local else "other"
        armed = (local.get(str(r["id"])) or {}).get("failsafe_at")
        if armed and time.time() > armed and r["status"] != "running":
            tag += f"; stopped by the failsafe, results may be on its disk: gpuhop start {r['id']}, wait, pull, down"
        print(f"{r['id']:>10}  {r['gpu_name'] or '?':<12} {r['status']:<9} ${r['dph'] or 0:.3f}/hr  {age:5.2f} h  "
              f"~${age * (r['dph'] or 0):.2f}  label={r.get('label')}  [{tag}]")
    for s in set(local) - {str(r["id"]) for r in rows}:
        try:  # a list that merely omits it (API hiccup) must not make us forget a live, billing instance
            if lx.provider.show(int(s)) is None:
                lx.state.forget(int(s))
        except Exception as e:
            print(f"could not check instance {s}: {e}")
    _clean_scheduled_stops(lx, {r["id"] for r in rows})
    return 0


def _clean_scheduled_stops(lx, listed: set):
    """Scheduled stops outlive their instance (Vast keeps them): delete the ones whose instance is gone."""
    try:
        stops = lx.provider.scheduled_stops()
    except Exception as e:
        print(f"could not list scheduled stops: {e}")
        return
    for iid in sorted({j["instance_id"] for j in stops} - {int(i) for i in listed}):
        try:
            if lx.provider.show(iid) is None:
                lx.provider.cancel_scheduled_stops(iid)
                print(f"deleted the leftover scheduled stop of destroyed instance {iid}")
        except Exception as e:
            print(f"could not clean the scheduled stop of instance {iid}: {e}")


def cmd_show(a):
    lx = _launcher()
    inst = lx.provider.show(lx.state.resolve(a.id))
    print(json.dumps(redact((inst or {}).get("raw", inst)), indent=2, default=str))
    return 0


def cmd_ssh(a):
    _, _, rem = _launcher().get(a.id)
    return rem.interactive(a.cmd)


def cmd_sync(a):
    return _launcher().sync(a.id, a.allow_dirty)


def cmd_wait(a):
    lx = _launcher()
    inst_id = lx.state.resolve(a.id)
    if not lx.provider.show(inst_id):
        raise SystemExit(f"instance {inst_id} not found on {lx.provider.name} (destroyed?). `gpuhop ls`")
    inst = lx.wait_running(inst_id, restarting=True)  # right after `gpuhop start` it has no ssh endpoint yet
    rem = lx.connect(inst_id, inst)
    lx.write_env(rem, inst_id, inst)
    lx.wait_ready(rem, inst_id)
    print(f"ready: gpuhop ssh {inst_id}")
    return 0


def cmd_rebootstrap(a):
    lx = _launcher()
    inst_id, inst, rem = lx.get(a.id)
    lx.write_env(rem, inst_id, inst)
    lx.install_runtime(rem)
    lx.start_bootstrap(rem, force=a.force)
    print("bootstrap restarted; waiting (Ctrl-C is safe, it keeps running on the instance)")
    lx.wait_ready(rem, inst_id)
    print(f"ready: gpuhop ssh {inst_id}")
    return 0


def cmd_start(a):
    lx = _launcher()
    inst_id = lx.state.resolve(a.id)
    lx.disarm_failsafe(inst_id)  # otherwise the scheduled stop stops it again at the next full hour
    print(lx.provider.start(inst_id))
    print(f"starting {inst_id}; then: gpuhop wait {inst_id}")
    return 0


def cmd_run(a):
    lx = _launcher()
    inst_id, _, rem = lx.get(a.id)
    sha = lx.state.get(inst_id).get("git_sha")
    for i, f in enumerate(a.exp):
        exp = lx.start_exp(rem, f, sha)
        if i < len(a.exp) - 1:  # jobs must not overlap on one GPU: wait before starting the next file
            lx.wait_exp(rem, exp, inst_id=inst_id)
    return 0


def cmd_status(a):
    lx = _launcher()
    _, _, rem = lx.get(a.id)
    exps = [a.exp] if a.exp else [Path(p).name.removesuffix(".status.json") for p in
                                  rem.text(f"ls {shlex.quote(rem.ws + '/jobs')}/*.status.json 2>/dev/null", check=False, cwd=None).split()]
    for e in exps:
        st = lx.exp_status(rem, e)
        lx.print_status(st) if st else print(f"  {e}: no status")
    print(rem.text("nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader",
                   check=False, cwd=None).strip())
    return 0


def _local_logs(lx, a) -> int | None:
    """After an instance is gone, show the copy `pull` saved under results/<gpu>/_logs (or _instance)."""
    res = lx.proj.results_dir
    if a.job:
        hits = sorted(res.glob(f"*/_logs/*/{a.job}.log"), key=lambda p: p.stat().st_mtime)
    elif a.bootstrap:
        hits = sorted(res.glob("*/_instance/*/bootstrap.log"), key=lambda p: p.stat().st_mtime)
    else:
        return None
    if not hits:
        return None
    lines = hits[-1].read_text(encoding="utf-8", errors="replace").splitlines()
    print(f"(no live instance; showing the pulled copy {hits[-1].relative_to(lx.proj.root)})")
    print("\n".join(lines[-a.tail:]))
    return 0


def cmd_logs(a):
    lx = _launcher()
    if a.id is None and not lx.state.load():
        rc = _local_logs(lx, a)
        if rc is not None:
            return rc
    inst_id, _, rem = lx.get(a.id)
    if a.bootstrap:
        path = f"{rem.ws}/bootstrap.log"
    elif a.job:
        path = f"{rem.ws}/logs/{a.job}.log"
    else:
        print(lx.provider.logs(inst_id, a.tail))
        return 0
    print(rem.text(f"tail -n {a.tail} {shlex.quote(path)}", check=False, cwd=None))
    return 0


def cmd_pull(a):
    lx = _launcher()
    inst_id, inst, rem = lx.get(a.id)
    lx.pull(inst_id, inst, rem)
    return 0


def cmd_freeze(a):
    """Write the lockfile = packages installed on top of the base image (base packages are pinned by the image)."""
    lx = _launcher()
    _, _, rem = lx.get(a.id)
    py = shlex.quote(lx.proj.cfg["python"])
    base = {l.strip().lower() for l in rem.text(f"cat {shlex.quote(rem.rt + '/base_freeze.txt')}", cwd=None).splitlines() if "==" in l}
    now = [l.strip() for l in rem.text(f"{py} -m pip list --format=freeze --exclude-editable", cwd=None).splitlines() if "==" in l]
    lock = sorted((l for l in now if l.lower() not in base and l.split("==")[0].lower() not in BASE_SKIP), key=str.lower)
    out = lx.proj.root / lx.proj.cfg["lockfile"]
    # keep the index options of the previous lock (e.g. --extra-index-url https://pypi.nvidia.com)
    index = [l for l in (out.read_text(encoding="utf-8").splitlines() if out.exists() else []) if l.startswith("--")]
    index += [f"--extra-index-url {u}" for u in a.extra_index_url if f"--extra-index-url {u}" not in index]
    header = ["# AUTO-GENERATED by `gpuhop freeze` from a working instance. Do not edit by hand.",
              f"# base image packages are excluded (pinned by the image); generated {time.strftime('%Y-%m-%d')}"]
    out.write_text("\n".join(header + index + lock) + "\n", encoding="utf-8")
    print(f"wrote {out} ({len(lock)} packages). Commit it and install from it in your setup script.")
    return 0


def cmd_down(a):
    lx = _launcher()
    inst_id = lx.state.resolve(a.id)
    if not a.yes and input(f"destroy instance {inst_id}? results not pulled are lost [y/N] ").strip().lower() != "y":
        return 1
    lx.destroy(inst_id)
    return 0


# ----------------------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="gpuhop", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=f"gpuhop {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add(name, fn, help_):
        sp = sub.add_parser(name, help=help_)
        sp.set_defaults(fn=fn)
        return sp

    def offer_args(sp):
        sp.add_argument("--offer", type=int, help="a specific offer id (see `offers`)")
        sp.add_argument("--cpu", help="regex on the host CPU name, e.g. 'EPYC 9' (fair comparisons pin the CPU class)")
        sp.add_argument("--wait-offer", type=float, default=0, metavar="MIN", help="wait up to MIN minutes for a matching offer")
        sp.add_argument("--allow-dirty", action="store_true", help="git mode: allow uncommitted changes")

    sp = add("init", cmd_init, "scaffold a project (gpuhop.yaml, setup.sh, exp/, ...)")
    sp.add_argument("dir", nargs="?", default=".")
    sp.add_argument("-t", "--template", default="pytorch")
    sp.add_argument("--name", help="project name (default: the directory name)")
    sp.add_argument("--force", action="store_true", help="overwrite existing files")
    sp.add_argument("--list", action="store_true", help="list templates")
    add("doctor", cmd_doctor, "check local setup")
    sp = add("offers", cmd_offers, "list the cheapest matching offers")
    sp.add_argument("gpu")
    sp.add_argument("-n", type=int, default=10)
    sp.add_argument("--cpu", help="regex on the host CPU name")
    for name, fn, h in [("up", cmd_up, "rent an instance, push code, bootstrap"),
                        ("smoke", cmd_smoke, "up + smoke + pull + destroy")]:
        sp = add(name, fn, h)
        sp.add_argument("gpu")
        offer_args(sp)
        if name == "up":
            sp.add_argument("--no-wait", action="store_true", help="return once bootstrap has started")
        else:
            sp.add_argument("--keep", action="store_true")
    sp = add("exp", cmd_exp, "up -> run experiment files -> pull -> destroy")
    sp.add_argument("gpu")
    sp.add_argument("exp", nargs="+")
    offer_args(sp)
    sp.add_argument("--instance", help="reuse a running instance instead of renting one (kept afterwards)")
    sp.add_argument("--keep", action="store_true", help="do not destroy at the end")
    sp.add_argument("--max-hours", type=float, help="watchdog: pull + destroy after this long (default: max_hours)")
    add("ls", cmd_ls, "list your instances (and their running cost)")
    for name, fn, h in [("show", cmd_show, "instance details"), ("pull", cmd_pull, "download results + logs"),
                        ("wait", cmd_wait, "wait until bootstrap is done"), ("start", cmd_start, "start a stopped instance")]:
        add(name, fn, h).add_argument("id", nargs="?")
    sp = add("rebootstrap", cmd_rebootstrap, "re-run bootstrap (setup is skipped if unchanged, unless --force)")
    sp.add_argument("id", nargs="?")
    sp.add_argument("--force", action="store_true", help="re-run setup even if its inputs are unchanged")
    sp = add("freeze", cmd_freeze, "write the lockfile from an instance's installed packages")
    sp.add_argument("id", nargs="?")
    sp.add_argument("--extra-index-url", action="append", default=[])
    sp = add("ssh", cmd_ssh, "interactive shell (or run a command) in the project dir")
    sp.add_argument("id", nargs="?")
    sp.add_argument("--cmd")
    sp = add("sync", cmd_sync, "push the current code to a live instance")
    sp.add_argument("id", nargs="?")
    sp.add_argument("--allow-dirty", action="store_true")
    sp = add("run", cmd_run, "start experiment file(s) on an instance (detached)")
    sp.add_argument("exp", nargs="+")
    sp.add_argument("--id")
    sp = add("status", cmd_status, "job status + GPU utilisation")
    sp.add_argument("id", nargs="?")
    sp.add_argument("--exp")
    sp = add("logs", cmd_logs, "container logs, bootstrap log, or a job log")
    sp.add_argument("id", nargs="?")
    sp.add_argument("--bootstrap", action="store_true")
    sp.add_argument("--job", help="<exp>/<job name>")
    sp.add_argument("--tail", type=int, default=100)
    sp = add("down", cmd_down, "destroy an instance")
    sp.add_argument("id", nargs="?")
    sp.add_argument("-y", "--yes", action="store_true")
    return p


def _safe_stdio():
    """Remote logs contain arbitrary Unicode. On a redirected Windows stdout (cp1252) printing it must not crash a
    session that is paying for a GPU."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="backslashreplace")


def main(argv=None):
    _safe_stdio()
    a = build_parser().parse_args(argv)
    if getattr(a, "cpu", None):
        os.environ["GPUB_CPU_REGEX"] = a.cpu
    if getattr(a, "wait_offer", 0):
        os.environ["GPUB_WAIT_OFFER_MIN"] = str(a.wait_offer)
    try:
        return a.fn(a)
    except KeyboardInterrupt:
        print("\ninterrupted")
        return 130
    except (RuntimeError, TimeoutError) as e:
        print(f"ERROR: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

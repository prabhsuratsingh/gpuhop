"""Instance lifecycle and experiment sessions, independent of the CLI (and of the provider: tests drive it with a
fake provider and a local "remote").

    up:       prepare code -> rent -> wait running -> wait ssh -> write env + runtime -> push code -> bootstrap
    session:  up (or reuse) -> for each experiment file: start runner, wait, pull -> final pull -> destroy
              (destroy also on failure / Ctrl-C; never when results could not be pulled from a stopped box)
"""

from __future__ import annotations

import json
import shlex
import subprocess
import time
from importlib import resources
from pathlib import Path

from . import catalog
from .codepush import CodePackage, git_info
from .jobs import expand
from .project import Project, env, redact
from .providers import Provider, get_provider
from .remote import Remote
from .state import State

q = shlex.quote

# Runs as the container starts. Everything else happens over ssh, driven by the launcher, so no secret is ever
# passed to the container's environment.
ONSTART = """#!/bin/bash
touch /root/.no_auto_tmux
mkdir -p /workspace/.gpuhop
"""

# GPUB_* variables mirrored under the older GPUBENCH_* names for projects that read those.
_COMPAT = ("GPU", "DPH", "OFFER_ID", "IMAGE", "HOST_CPU")


class Launcher:
    poll_running_s = 15
    poll_ready_s = 30
    poll_exp_s = 60
    poll_offer_s = 60
    shell = "bash"  # on the box

    def __init__(self, proj: Project, provider: Provider | None = None):
        self.proj = proj
        self.provider = provider or get_provider(proj.cfg["provider"])
        self.state = State(proj)

    # ---------------------------------------------------------------------------------------------- basics
    def _remote_at(self, host: str, port: int) -> Remote:
        return Remote(host, port, user=self.provider.user, key=env("SSH_KEY"),
                      known_hosts=self.proj.state_dir / "known_hosts")

    @staticmethod
    def _endpoints(inst: dict) -> list[dict]:
        eps = inst.get("endpoints") or ([{"host": inst["host"], "port": inst["port"], "kind": "ssh"}] if inst.get("host") else [])
        if not eps:
            raise RuntimeError(f"instance {inst.get('id')} exposes no ssh endpoint yet")
        return eps

    def remote(self, inst: dict) -> Remote:
        """The endpoint that answered last time (see connect), else the provider's preferred one."""
        eps = self._endpoints(inst)
        kind = self.state.get(inst["id"]).get("ssh_kind") if inst.get("id") is not None else None
        ep = next((e for e in eps if e["kind"] == kind), eps[0])
        return self._remote_at(ep["host"], ep["port"])

    def connect(self, inst_id: int, inst: dict, timeout_s: int = 600) -> Remote:
        """Wait until one of the instance's ssh endpoints answers, and remember which one.

        Hosts differ: on some the direct port mapping refuses connections while the provider's ssh proxy works
        (or the other way round), so every endpoint is tried in turn."""
        t0, errs = time.time(), {}
        while time.time() - t0 < timeout_s:
            for ep in self._endpoints(inst):
                rem = self._remote_at(ep["host"], ep["port"])
                try:
                    p = rem.run("echo ok", check=False, timeout=45, cwd=None)
                    if p.returncode == 0:
                        self.state.remember(inst_id, ssh_kind=ep["kind"])
                        if errs:
                            print(f"  ssh ok via {ep['kind']} endpoint {ep['host']}:{ep['port']}", flush=True)
                        return rem
                    lines = (p.stderr or b"").decode(errors="replace").strip().splitlines()
                    err = lines[-1] if lines else f"exit {p.returncode}"
                except subprocess.TimeoutExpired:
                    err = "timeout"
                if errs.get(ep["kind"]) != err:
                    print(f"  ssh via {ep['kind']} endpoint {ep['host']}:{ep['port']} not answering yet: {err}", flush=True)
                    errs[ep["kind"]] = err
            time.sleep(min(self.poll_ready_s, 15))
            inst = self.provider.show(inst_id) or inst  # endpoints can appear late
        hint = " -- check GPUB_SSH_KEY / `gpuhop doctor`" if any("denied" in e.lower() for e in errs.values()) else ""
        raise RuntimeError(f"ssh never came up on instance {inst_id}: "
                           + "; ".join(f"{k}: {v}" for k, v in errs.items()) + hint)

    def get(self, ident) -> tuple[int, dict, Remote]:
        inst_id = self.state.resolve(ident)
        inst = self.provider.show(inst_id)
        if not inst:
            raise SystemExit(f"instance {inst_id} not found on {self.provider.name} (destroyed?). `gpuhop ls`")
        return inst_id, inst, self.remote(inst)

    def gpu_alias(self, inst_id: int, inst: dict | None = None) -> str:
        return self.state.get(inst_id).get("gpu") or "unknown"

    def destroy(self, inst_id: int):
        self.provider.destroy(inst_id)
        s = self.state.get(inst_id)
        if s.get("created") and s.get("dph"):
            hrs = (time.time() - s["created"]) / 3600
            print(f"destroyed {inst_id}: ~{hrs:.2f} h, ~${hrs * s['dph']:.2f} (excl. storage/bandwidth)")
        else:
            print(f"destroyed {inst_id}")
        self.state.forget(inst_id)

    # ---------------------------------------------------------------------------------------------- box setup
    def env_vars(self, rem: Remote, inst_id: int, inst: dict | None = None) -> dict:
        s = self.state.get(inst_id)
        c = self.proj.cfg
        spec = catalog.gpu_spec(self.proj, s["gpu"]) if s.get("gpu") else {"env": c["env"], "image": c["image"]}
        dph = (inst or {}).get("dph") or s.get("dph")
        v = {
            **spec["env"],
            "GPUB_WS": rem.ws, "GPUB_PROJECT_DIR": rem.proj, "GPUB_PROJECT": self.proj.name,
            "GPUB_PROVIDER": self.provider.name, "GPUB_INSTANCE_ID": str(inst_id), "GPUB_GPU": s.get("gpu", "unknown"),
            "GPUB_OFFER_ID": str(s.get("offer_id", "")), "GPUB_IMAGE": s.get("image", spec["image"]),
            "GPUB_HOST_CPU": s.get("cpu_name") or "", "GPUB_HOST_DRIVER": s.get("driver") or "", "GPUB_DPH": f"{dph:.4f}" if dph else "",
            "GPUB_PYTHON": c["python"], "GPUB_SETUP": c["setup"] or "",
            "GPUB_SETUP_HASH_FILES": " ".join(c["setup_hash_files"]), "GPUB_SMOKE": c["smoke"] or "",
            "GIT_SHA": s.get("git_sha") or "",
        }
        v.update({f"GPUBENCH_{k}": v[f"GPUB_{k}"] for k in _COMPAT})
        return v

    def write_env(self, rem: Remote, inst_id: int, inst: dict | None = None):
        lines = ["# written by gpuhop on every up/sync/wait; edits are overwritten",
                 f"[ -f {q(rem.rt + '/env.container.sh')} ] && . {q(rem.rt + '/env.container.sh')}"]
        lines += [f"export {k}={shlex.quote(str(val))}" for k, val in self.env_vars(rem, inst_id, inst).items()]
        rem.put_text(rem.env_file, "\n".join(lines) + "\n")

    def install_runtime(self, rem: Remote):
        rt = resources.files("gpuhop").joinpath("runtime")
        for name in ("jobrunner.py", "bootstrap.sh"):
            rem.put_bytes(f"{rem.rt}/{name}", rt.joinpath(name).read_bytes().replace(b"\r\n", b"\n"))

    def start_bootstrap(self, rem: Remote, force: bool = False):
        rem.run(f"rm -f {q(rem.ws + '/.ready')} {q(rem.ws + '/.bootstrap_failed')}"
                + (f" {q(rem.rt + '/setup_stamp')}" if force else ""), cwd=None)
        rem.detach("bootstrap", f"bash {q(rem.rt + '/bootstrap.sh')}", f"{rem.ws}/bootstrap.log")

    # ---------------------------------------------------------------------------------------------- waiting
    def wait_running(self, inst_id: int, timeout_s: int = 1800, stuck_created_s: int = 600) -> dict:
        """Wait for status=running. A host still loading (pulling the image) after image_pull_timeout_min is given
        up on: in practice such hosts rarely finish, and some bill while they try."""
        t0 = time.time()
        last = None
        created_since = None
        pull_s = float(self.proj.cfg["image_pull_timeout_min"]) * 60
        while time.time() - t0 < timeout_s:
            inst = self.provider.show(inst_id) or {}
            status, msg = inst.get("status"), (inst.get("msg") or "")[:120]
            if status in ("loading", "unknown", None) and time.time() - t0 > pull_s:
                raise RuntimeError(f"instance {inst_id} still '{status}' after {pull_s / 60:.0f} min (slow/stuck image "
                                   "pull; raise image_pull_timeout_min for very large images)")
            # 'created' = image loaded but the host never started the container; it rarely recovers
            if status == "created":
                created_since = created_since or time.time()
                if time.time() - created_since > stuck_created_s:
                    raise RuntimeError(f"instance {inst_id} stuck in 'created' for {stuck_created_s}s (host did not start it)")
            else:
                created_since = None
            if (status, msg) != last:
                print(f"  [{int(time.time() - t0):4d}s] status={status} {msg}", flush=True)
                last = (status, msg)
            if status == "running" and inst.get("host"):
                if inst.get("dph"):
                    self.state.remember(inst_id, dph=inst["dph"])
                return inst
            if status in ("exited", "offline") or "error" in msg.lower():
                raise RuntimeError(f"instance {inst_id} failed to start: status={status} msg={msg}")
            time.sleep(self.poll_running_s)
        raise TimeoutError(f"instance {inst_id} not running after {timeout_s}s")

    def wait_ready(self, rem: Remote, inst_id: int | None = None) -> Remote:
        """Wait for bootstrap.sh -> <ws>/.ready (or .bootstrap_failed)."""
        timeout_s = int(self.proj.cfg["bootstrap_timeout_min"]) * 60
        t0 = time.time()
        last_line = ""
        while time.time() - t0 < timeout_s:
            try:
                ws = q(rem.ws)
                p = rem.run(f"if [ -f {ws}/.ready ]; then echo READY; elif [ -f {ws}/.bootstrap_failed ]; then "
                            f"echo FAILED; cat {ws}/.bootstrap_failed; fi; tail -n 1 {ws}/bootstrap.log 2>/dev/null",
                            check=False, timeout=60, cwd=None)
                out = p.stdout.decode(errors="replace").strip()
                if p.returncode == 255:  # ssh itself failed
                    err = p.stderr.decode(errors="replace").strip().splitlines()
                    out = f"(ssh failed: {err[-1] if err else 'no output'})"
            except Exception as e:
                out = f"(ssh not ready: {type(e).__name__})"
            if out.startswith("READY"):
                print(f"  [{int(time.time() - t0):4d}s] bootstrap complete", flush=True)
                return rem
            if out.startswith("FAILED"):
                raise RuntimeError(f"bootstrap failed: {out}\n-> `gpuhop logs {inst_id or ''} --bootstrap`")
            line = out.splitlines()[-1] if out else ""
            if line and line != last_line:
                print(f"  [{int(time.time() - t0):4d}s] {line[:150]}", flush=True)
                last_line = line
            time.sleep(self.poll_ready_s)
        raise TimeoutError(f"bootstrap not finished after {timeout_s}s")

    # ---------------------------------------------------------------------------------------------- up
    def up(self, gpu: str, offer_id: int | None = None, allow_dirty: bool = False, wait: bool = True,
           created: list | None = None, offer: dict | None = None) -> tuple[int, dict, Remote]:
        """Create + bootstrap. The new id is appended to `created` immediately, so callers can always clean up.

        An instance that never becomes reachable (never `running`, or ssh never answers) is useless and is
        destroyed here; nothing of ours has run on it yet. One that is reachable but fails bootstrap is left to
        the caller (kept by `up` for debugging, destroyed by `exp`/`smoke`). `offer` (an offer dict from a
        search) skips the search, so its price is kept."""
        spec = catalog.gpu_spec(self.proj, gpu)
        catalog.check_image(spec, allow_drift=env("ALLOW_IMAGE_DRIFT") == "1")
        code = CodePackage(self.proj, allow_dirty)  # validated before paying for anything
        if offer is not None:
            pass
        elif offer_id is None:
            offers = self.provider.search(spec, limit=10, cpu=env("CPU_REGEX"))
            if not offers:
                raise RuntimeError(f"no offers match {gpu}: {spec['query']}")
            offer = offers[0]
        else:  # keep the price when the chosen offer is among the current matches
            offers = self.provider.search(spec, limit=10, cpu=env("CPU_REGEX"))
            offer = next((o for o in offers if o["id"] == offer_id), {"id": offer_id, "dph": None, "cpu_name": "", "driver": ""})
        inst_id = self.provider.create(spec, offer, env=spec["env"], onstart=ONSTART, label=f"gpuhop-{self.proj.name}-{gpu}")
        if created is not None:
            created.append(inst_id)
        self.state.remember(inst_id, gpu=gpu, offer_id=offer["id"], dph=offer.get("dph"), git_sha=code.sha,
                            image=spec["image"] + (f"@{spec['image_digest']}" if spec.get("image_digest") else ""),
                            created=time.time(), gpu_name=offer.get("gpu_name"), cpu_name=offer.get("cpu_name"),
                            driver=offer.get("driver"))
        price = offer.get("dph")
        print(f"created instance {inst_id} on offer {offer['id']} ({f'${price:.3f}' if price else 'price on start'}/hr); "
              f"code: {code.describe()}", flush=True)
        try:
            inst = self.wait_running(inst_id)
            rem = self.connect(inst_id, inst)
        except BaseException:
            print(f"instance {inst_id} never became reachable; destroying it")
            self.destroy(inst_id)
            if created is not None:
                created.remove(inst_id)
            raise
        self.write_env(rem, inst_id, inst)
        self.install_runtime(rem)
        code.push(rem)
        self.start_bootstrap(rem)
        if wait:
            self.wait_ready(rem, inst_id)
        return inst_id, inst, rem

    def find_offers(self, gpu: str, tried: set, wait_min: float) -> list[dict]:
        """Matching offers not yet tried; with --wait-offer, poll until one appears (pinned-CPU hosts come and go)."""
        spec = catalog.gpu_spec(self.proj, gpu)
        deadline = time.time() + wait_min * 60
        while True:
            cands = [o for o in self.provider.search(spec, limit=10, cpu=env("CPU_REGEX")) if o["id"] not in tried]
            if cands or time.time() >= deadline:
                return cands
            print(f"  no matching {gpu} offer right now (cpu filter: {env('CPU_REGEX') or spec.get('cpu_regex') or 'none'}, "
                  f"driver filter: {spec.get('driver_regex') or 'none'}); "
                  f"checking again in {self.poll_offer_s} s ({max(0, deadline - time.time()) / 60:.0f} min left)", flush=True)
            time.sleep(self.poll_offer_s)

    def up_with_retry(self, gpu: str, offer_id: int | None, allow_dirty: bool, created: list, attempts: int = 3):
        """`up`, moving on to the next-cheapest offer when a host never starts its container."""
        if offer_id is not None:
            return self.up(gpu, offer_id, allow_dirty, created=created)
        wait_min = float(env("WAIT_OFFER_MIN") or 0)
        tried: set = set()
        last_err = None
        for _ in range(attempts):
            cands = self.find_offers(gpu, tried, wait_min if not tried else 0)
            if not cands:
                if not tried:
                    spec = catalog.gpu_spec(self.proj, gpu)
                    raise RuntimeError(f"no {gpu} offers match right now.\n  query: {spec['query']}\n  cpu filter: "
                                       f"{env('CPU_REGEX') or spec.get('cpu_regex') or 'none'}\n  driver filter: "
                                       f"{spec.get('driver_regex') or 'none'}\n  check: gpuhop offers {gpu}"
                                       "\n  or wait for one: add --wait-offer 30")
                break
            tried.add(cands[0]["id"])
            n_before = len(created)
            try:
                return self.up(gpu, cands[0]["id"], allow_dirty, created=created, offer=cands[0])
            except (RuntimeError, TimeoutError) as e:
                if len(created) > n_before:  # reachable but failed later (bootstrap): not a host problem
                    raise
                last_err = e
                print(f"offer {cands[0]['id']} failed to start ({e}); trying the next one")
        raise RuntimeError(f"no {gpu} offer started after {len(tried)} attempt(s): {last_err}")

    def sync(self, ident, allow_dirty: bool = False) -> int:
        """Push the current code (and runtime files) to a live instance without re-bootstrapping."""
        inst_id, inst, rem = self.get(ident)
        code = CodePackage(self.proj, allow_dirty)
        self.state.remember(inst_id, git_sha=code.sha)
        self.write_env(rem, inst_id, inst)
        self.install_runtime(rem)
        code.push(rem)
        if self.proj.cfg["on_sync"]:
            rem.run(self.proj.cfg["on_sync"], timeout=3600)
        c = self.proj.cfg
        if c["setup"]:
            files = " ".join(shlex.quote(f) for f in [c["setup"], *c["setup_hash_files"]])
            changed = rem.text(f"h=$({{ cat {files} 2>/dev/null || true; }} | sha256sum | cut -d' ' -f1); "
                               f"[ \"$h\" = \"$(cat {q(rem.rt + '/setup_stamp')} 2>/dev/null)\" ] || echo CHANGED", check=False)
            if "CHANGED" in changed:
                print(f"note: setup inputs changed since the last bootstrap; run `gpuhop rebootstrap {inst_id}` to apply them")
        print(f"instance {inst_id} now has {code.describe()}")
        return 0

    # ---------------------------------------------------------------------------------------------- experiments
    def start_exp(self, rem: Remote, exp_file: str, sha: str | None) -> str:
        spec = expand(exp_file, self.proj.cfg["results"]["remote_dir"])
        spec.update(git_sha=sha, project_dir=rem.proj, env_file=rem.env_file, logs_dir=f"{rem.ws}/logs", shell=self.shell)
        remote_file = f"{rem.ws}/jobs/{spec['exp']}.json"
        rem.put_text(remote_file, json.dumps(spec, indent=2))
        rem.detach(self._session_name(spec["exp"]),
                   f"{q(self.proj.cfg['python'])} {q(rem.rt + '/jobrunner.py')} {q(remote_file)}",
                   f"{rem.ws}/logs/{spec['exp']}.runner.log")
        print(f"started {spec['exp']}: {len(spec['jobs'])} job(s)  (tmux session {self._session_name(spec['exp'])} on the instance)")
        return spec["exp"]

    @staticmethod
    def _session_name(exp: str) -> str:
        return "exp_" + exp.replace(".", "_").replace(":", "_")

    def exp_status(self, rem: Remote, exp: str) -> dict | None:
        p = rem.run(f"cat {q(f'{rem.ws}/jobs/{exp}.status.json')}", check=False, cwd=None, timeout=60)
        return json.loads(p.stdout) if p.returncode == 0 and p.stdout.strip() else None

    @staticmethod
    def print_status(st: dict):
        jobs = st.get("jobs", {})
        counts: dict = {}
        for j in jobs.values():
            counts[j["state"]] = counts.get(j["state"], 0) + 1
        running = [n for n, j in jobs.items() if j["state"] == "running"]
        print(f"  {st['exp']}: {st.get('state')}  {counts}  of {st.get('n_jobs')}  running={running}", flush=True)
        for n, j in jobs.items():
            if j["state"] == "failed":
                print(f"    FAILED {n} (rc={j.get('returncode')}) -> gpuhop logs --job {st['exp']}/{n}")

    def wait_exp(self, rem: Remote, exp: str, deadline: float | None = None, inst_id: int | None = None) -> dict:
        """Poll the job runner's status file. When ssh fails, look at the instance itself: a stopped instance is
        reported (once) instead of polled silently, and after a restart the ssh endpoint is refreshed."""
        last = None
        last_state = "running"
        while True:
            try:
                st = self.exp_status(rem, exp)
            except Exception as e:
                print(f"  (status poll failed: {e}; retrying)")
                st = None
            if st is None and inst_id is not None:
                try:
                    inst = self.provider.show(inst_id) or {}
                except Exception:
                    inst = {}
                state = inst.get("status") or "unknown"
                if state != last_state:
                    if state == "running":
                        rem = self.remote(inst)  # ports can change across a stop/start
                        print(f"  instance {inst_id} is running again. Its job runner did not survive the restart; resume "
                              f"with:  gpuhop run <exp file> --id {inst_id}   (finished jobs are skipped)", flush=True)
                    else:
                        print(f"  WARNING: instance {inst_id} is '{state}' (intended: {inst.get('intended_status')}). Jobs "
                              f"are paused and nothing can be pulled. Restart it: gpuhop start {inst_id}. Do NOT Ctrl-C "
                              f"this command before it is running again.", flush=True)
                    last_state = state
            if st:
                key = json.dumps({k: v["state"] for k, v in st["jobs"].items()}) + st["state"]
                if key != last:
                    self.print_status(st)
                    last = key
                if st["state"] in ("done", "failed"):
                    return st
            if deadline and time.time() > deadline:
                raise TimeoutError("max-hours reached")
            time.sleep(self.poll_exp_s)

    def pull(self, inst_id: int, inst: dict, rem: Remote) -> Path:
        dest = self.proj.results_dir / self.gpu_alias(inst_id, inst)
        excl = () if env("PULL_CHECKPOINTS") == "1" else tuple(self.proj.cfg["results"]["exclude"])
        got = rem.pull_dir(f"{rem.proj}/{self.proj.cfg['results']['remote_dir']}", dest, exclude=excl)
        rem.pull_dir(f"{rem.ws}/logs", dest / "_logs" / str(inst_id), exclude=())
        meta = dest / "_instance" / str(inst_id)
        rem.pull_file(f"{rem.ws}/bootstrap.log", meta / "bootstrap.log")
        for f in ("smoke.json", "provenance.json"):
            rem.pull_file(f"{rem.rt}/{f}", meta / f)
        meta.mkdir(parents=True, exist_ok=True)
        (meta / "instance.json").write_text(json.dumps(redact(inst.get("raw", inst)), indent=2, default=str))
        print(f"pulled -> {dest}" + ("" if got else " (no results yet)"), flush=True)
        return dest

    def check_tag(self):
        """With require_git_tag set, warn loudly when the code is not exactly that tag."""
        tag_name = self.proj.cfg["require_git_tag"]
        if not tag_name:
            return
        try:
            tag = subprocess.run(["git", "rev-list", "-n", "1", tag_name], cwd=self.proj.root, capture_output=True,
                                 text=True).stdout.strip()
        except OSError:
            return
        info = git_info(self.proj.root)
        if not tag:
            print(f"WARNING: require_git_tag is {tag_name!r} but no such tag exists")
        elif tag != info["git_sha"] or info["dirty"]:
            print(f"WARNING: code is {(info['git_sha'] or '?')[:10]}{' (dirty)' if info['dirty'] else ''}, not the "
                  f"required tag {tag_name} ({tag[:10]}).")

    def session(self, gpu: str | None, exp_files: list[str], *, offer_id=None, instance=None, keep=False,
                max_hours: float | None = None, allow_dirty=False) -> int:
        """One-shot: up -> run each exp file sequentially -> pull -> destroy (always, unless keep/instance)."""
        for f in exp_files:
            expand(f, self.proj.cfg["results"]["remote_dir"])  # validate before paying for a GPU
        self.check_tag()
        inst_id = None
        created: list = []
        code = 1
        deadline = time.time() + float(max_hours or self.proj.cfg["max_hours"]) * 3600
        try:
            if instance:
                inst_id, inst, rem = self.get(instance)
            else:
                inst_id, inst, rem = self.up_with_retry(gpu, offer_id, allow_dirty, created)
            sha = self.state.get(inst_id).get("git_sha")
            failed = False
            for f in exp_files:
                exp = self.start_exp(rem, f, sha)
                st = self.wait_exp(rem, exp, deadline=deadline, inst_id=inst_id)
                failed |= st["state"] == "failed"
                self.pull(inst_id, inst, rem)  # after every exp, so nothing is lost if a later one hangs
                if st["state"] == "failed" and expand(f, self.proj.cfg["results"]["remote_dir"]).get("stop_session_on_fail"):
                    print(f"{exp} failed and is marked stop_session_on_fail: skipping the remaining experiment files")
                    break
            code = 1 if failed else 0
        except KeyboardInterrupt:
            print("\ninterrupted")
        except Exception as e:
            print(f"ERROR: {e}")
        finally:
            inst_id = inst_id if inst_id is not None else (created[-1] if created else None)
            if inst_id is not None:
                self._finish(inst_id, keep=keep or bool(instance))
        return code

    def _finish(self, inst_id: int, keep: bool):
        pulled, state = False, "unknown"
        try:
            inst = self.provider.show(inst_id)
            state = (inst or {}).get("status") or "gone"
            if inst and state == "running":
                self.pull(inst_id, inst, self.remote(inst))
                pulled = True
        except Exception as e:
            print(f"final pull failed: {e}")
        if keep:
            print(f"keeping instance {inst_id} (remember `gpuhop down {inst_id}`)")
        elif not pulled and state not in ("gone", "created", "loading", "unknown"):
            # a stopped instance still holds unpulled results on its disk: never destroy it automatically
            print(f"NOT destroying instance {inst_id}: it is '{state}' and its results could not be pulled.\n"
                  f"  1) gpuhop start {inst_id}   2) gpuhop wait {inst_id}\n"
                  f"  3) gpuhop pull {inst_id}    4) gpuhop down {inst_id} -y")
        else:
            self.destroy(inst_id)

    def smoke(self, gpu: str, offer_id=None, allow_dirty=False, keep=False) -> int:
        """Boot + bootstrap (incl. the project's smoke command) on a GPU class, pull the report, destroy."""
        created: list = []
        code = 1
        try:
            inst_id, inst, rem = self.up_with_retry(gpu, offer_id, allow_dirty, created)
            meta = self.pull(inst_id, inst, rem) / "_instance" / str(inst_id)
            print((meta / "smoke.json").read_text() if (meta / "smoke.json").exists() else "bootstrap OK (no smoke report)")
            code = 0
        except KeyboardInterrupt:
            print("\ninterrupted")
        except Exception as e:
            print(f"SMOKE FAILED on {gpu}: {e}")
        finally:
            if created:
                self._finish(created[-1], keep=keep)
        return code

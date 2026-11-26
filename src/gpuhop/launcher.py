"""Instance lifecycle and experiment sessions, independent of the CLI (and of the provider: tests drive it with a
fake provider and a local "remote").

    up:       prepare code -> rent -> wait running -> wait ssh -> write env + runtime -> push code -> bootstrap
    session:  up (or reuse) -> for each experiment file: start runner, wait, pull -> final pull -> destroy
              (destroy also on failure / Ctrl-C; never when results could not be pulled from a stopped box)
"""

from __future__ import annotations

import json
import posixpath
import shlex
import signal
import subprocess
import time
import uuid
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


class RunnerDied(RuntimeError):
    """The job runner stopped without finishing (host restart, OOM kill, ...)."""


class FailsafeError(Exception):
    """The provider refused the failsafe stop. An account problem, not a host problem: do not try other offers."""


def _utc(ts: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(ts))


class Launcher:
    poll_running_s = 15
    poll_ready_s = 30
    poll_exp_s = 60
    poll_offer_s = 60
    runner_stale_s = 120  # no heartbeat for this long = the job runner is dead
    failsafe_grace_s = 15 * 60  # after the max_hours deadline: time for the launcher's own pull + destroy
    shell = "bash"  # on the box

    def __init__(self, proj: Project, provider: Provider | None = None):
        self.proj = proj
        self.provider = provider or get_provider(proj.cfg["provider"])
        self.state = State(proj)
        self._tokens: dict[str, str] = {}  # exp -> id of the runner this process started

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

    def arm_failsafe(self, inst_id: int, at: float):
        """Have the provider stop the instance after `at` even if this process (or the laptop) is gone. A stop, not a
        destroy: results stay on the disk; GPU billing ends, storage billing continues."""
        if env("NO_FAILSAFE") == "1":
            print(f"WARNING: GPUB_NO_FAILSAFE=1: nothing stops instance {inst_id} if this laptop goes away")
            return
        job = self.provider.schedule_stop(inst_id, at)
        if job is None:
            print(f"note: {self.provider.name} cannot schedule a server-side stop; if this laptop goes away, nothing "
                  f"stops instance {inst_id}")
            return
        self.state.remember(inst_id, failsafe_at=at, failsafe_job=job)
        print(f"failsafe: if instance {inst_id} is still up after {_utc(at)}, {self.provider.name} stops it "
              "(at the next full UTC hour)", flush=True)

    def disarm_failsafe(self, inst_id: int):
        try:
            self.provider.cancel_scheduled_stops(inst_id)
            if self.state.get(inst_id):
                self.state.remember(inst_id, failsafe_at=None, failsafe_job=None)
        except Exception as e:
            print(f"warning: could not delete the scheduled stop of instance {inst_id} ({e}); `gpuhop ls` retries")

    def destroy(self, inst_id: int):
        self.provider.destroy(inst_id)
        self.disarm_failsafe(inst_id)  # after the destroy: if that fails, the failsafe must stay armed
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
    def wait_running(self, inst_id: int, timeout_s: int = 1800, stuck_created_s: int = 600,
                     restarting: bool = False) -> dict:
        """Wait for status=running. A host still loading (pulling the image) after image_pull_timeout_min is given
        up on: in practice such hosts rarely finish, and some bill while they try. With `restarting`, an instance
        that was just asked to start may still report its old stopped status."""
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
            starting = restarting and inst.get("intended_status") == "running"
            if not starting and (status in ("exited", "offline") or "error" in msg.lower()):
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
           created: list | None = None, offer: dict | None = None,
           failsafe_at: float | None = None) -> tuple[int, dict, Remote]:
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
        if failsafe_at is not None:  # armed before anything else can go wrong (or the laptop can go away)
            try:
                self.arm_failsafe(inst_id, failsafe_at)
            except Exception as e:
                print(f"could not schedule the failsafe stop for instance {inst_id} ({e}); destroying it. "
                      "Set GPUB_NO_FAILSAFE=1 to run without it.", flush=True)
                if self._destroy_safely(inst_id) and created is not None:
                    created.remove(inst_id)
                raise FailsafeError(f"failsafe stop could not be scheduled: {e}") from e
        try:
            inst = self.wait_running(inst_id)
            rem = self.connect(inst_id, inst)
        except BaseException:
            print(f"instance {inst_id} never became reachable; destroying it")
            if self._destroy_safely(inst_id) and created is not None:
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

    def up_with_retry(self, gpu: str, offer_id: int | None, allow_dirty: bool, created: list, attempts: int = 3,
                      wait: bool = True, failsafe_at: float | None = None):
        """`up`, moving on to the next-cheapest offer when a host never starts its container."""
        if offer_id is not None:
            return self.up(gpu, offer_id, allow_dirty, wait=wait, created=created, failsafe_at=failsafe_at)
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
                return self.up(gpu, cands[0]["id"], allow_dirty, wait=wait, created=created, offer=cands[0],
                               failsafe_at=failsafe_at)
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
    def _code_id(self, rem: Remote) -> str | None:
        """What identifies the code on the box: finished jobs are only skipped while this is unchanged."""
        p = rem.run(f"cat {q(rem.rt + '/provenance.json')}", check=False, cwd=None, timeout=60)
        try:
            prov = json.loads(p.stdout)
        except ValueError:
            return None
        return prov.get("content_sha256") or prov.get("git_sha")

    def _live_runners(self, rem: Remote) -> list[str]:
        """Experiments whose job runner has a fresh heartbeat."""
        out = rem.text(f'now=$(date +%s); for f in {q(rem.ws + "/jobs")}/*.heartbeat; do [ -f "$f" ] && '
                       'echo "$(( now - $(stat -c %Y "$f") )) $f"; done; true', check=False, cwd=None, timeout=60)
        live = []
        for line in out.splitlines():
            age, _, path = line.partition(" ")
            if age.lstrip("-").isdigit() and int(age) < self.runner_stale_s:
                live.append(posixpath.basename(path).removesuffix(".heartbeat"))
        return live

    def start_exp(self, rem: Remote, exp_file: str, sha: str | None) -> str:
        spec = expand(exp_file, self.proj.cfg["results"]["remote_dir"])
        others = [e for e in self._live_runners(rem) if e != spec["exp"]]
        if others:
            raise RuntimeError(f"experiment(s) {others} still running on this instance; jobs must not share the GPU. "
                               "Wait for them to finish first (`gpuhop status`).")
        self.install_runtime(rem)  # a reused instance may have an older runner
        self._tokens[spec["exp"]] = uuid.uuid4().hex
        spec.update(git_sha=sha, project_dir=rem.proj, env_file=rem.env_file, logs_dir=f"{rem.ws}/logs", shell=self.shell,
                    token=self._tokens[spec["exp"]], code_id=self._code_id(rem))
        remote_file = f"{rem.ws}/jobs/{spec['exp']}.json"
        rem.put_text(remote_file, json.dumps(spec, indent=2))
        rem.run(f"rm -f {q(rem.ws + '/jobs/' + spec['exp'] + '.heartbeat')}", cwd=None)
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

    def _runner_dead(self, rem: Remote, exp: str, since: float) -> bool:
        """True only on positive evidence: the box answered and the heartbeat is stale, or never appeared."""
        hb = q(f"{rem.ws}/jobs/{exp}.heartbeat")
        try:
            p = rem.run(f'if [ -f {hb} ]; then echo $(( $(date +%s) - $(stat -c %Y {hb}) )); else echo missing; fi',
                        check=False, cwd=None, timeout=60)
        except Exception:
            return False
        out = p.stdout.decode(errors="replace").strip() if p.returncode == 0 else ""
        if out == "missing":
            return time.time() - since > self.runner_stale_s
        return out.lstrip("-").isdigit() and int(out) > self.runner_stale_s

    def wait_exp(self, rem: Remote, exp: str, deadline: float | None = None, inst_id: int | None = None) -> dict:
        """Poll the job runner's status file. When ssh fails, look at the instance itself: a stopped instance is
        reported (once) instead of polled silently, and after a restart the ssh endpoint is refreshed. A runner
        that died raises RunnerDied instead of being waited on until the deadline."""
        last = None
        last_state = "running"
        token = self._tokens.get(exp)
        since = time.time()

        def read():  # a status file left by an earlier run of this experiment does not count
            s = self.exp_status(rem, exp)
            return s if s and (not token or s.get("token") == token) else None

        while True:
            try:
                st = read()
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
                        print(f"  instance {inst_id} is running again; checking its job runner", flush=True)
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
            if self._runner_dead(rem, exp, since):
                try:  # it may have finished between the status read and the heartbeat check
                    final = read()
                except Exception:
                    final = None
                if final and final["state"] in ("done", "failed"):
                    self.print_status(final)
                    return final
                raise RunnerDied(f"the job runner for {exp} is no longer running (host restart, out of memory or disk?)."
                                 f" See logs/{exp}.runner.log on the box. Resume with: gpuhop run <exp file>"
                                 + (f" --id {inst_id}" if inst_id is not None else "") + " (finished jobs are skipped)")
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
        (meta / "instance.json").write_text(json.dumps(redact(inst.get("raw", inst)), indent=2, default=str),
                                            encoding="utf-8")
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
        had_work = False
        deadline = time.time() + float(max_hours or self.proj.cfg["max_hours"]) * 3600
        try:
            if instance:
                inst_id, inst, rem = self.get(instance)
            else:
                inst_id, inst, rem = self.up_with_retry(  # a kept instance is meant to outlive the session
                    gpu, offer_id, allow_dirty, created, failsafe_at=None if keep else deadline + self.failsafe_grace_s)
            sha = self.state.get(inst_id).get("git_sha")
            failed = False
            for f in exp_files:
                had_work = True
                exp, st, inst, rem = self._run_exp(inst_id, inst, rem, f, sha, deadline)
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
                if not self._finish(inst_id, keep=keep or bool(instance), had_work=had_work):
                    code = code or 1
        return code

    def _run_exp(self, inst_id: int, inst: dict, rem: Remote, exp_file: str, sha: str | None, deadline: float):
        """Start one experiment and wait for it, restarting a runner that died (finished jobs are skipped)."""
        exp = self.start_exp(rem, exp_file, sha)
        for restarts_left in (2, 1, 0):
            try:
                st = self.wait_exp(rem, exp, deadline=deadline, inst_id=inst_id)
                break
            except RunnerDied as e:
                if not restarts_left:
                    raise
                print(f"  {e}\n  restarting it ({restarts_left} restart(s) left)", flush=True)
                inst = self.provider.show(inst_id) or inst
                rem = self.remote(inst)
                self.start_exp(rem, exp_file, sha)
        try:  # ports can change across a stop/start
            inst = self.provider.show(inst_id) or inst
            rem = self.remote(inst)
        except Exception as e:
            print(f"  (could not refresh the instance: {e})")
        return exp, st, inst, rem

    def _destroy_safely(self, inst_id: int) -> bool:
        """Destroy with retries, not interruptible by Ctrl-C. False (and a loud message) if it still failed."""
        restore, old = False, None
        try:
            old = signal.signal(signal.SIGINT, signal.SIG_IGN)
            restore = True
        except ValueError:  # not the main thread
            pass
        try:
            err = None
            for attempt in range(3):
                if attempt:
                    time.sleep(min(self.poll_running_s, 5))
                try:
                    self.destroy(inst_id)
                    return True
                except Exception as e:
                    err = e
            print(f"ERROR: could not destroy instance {inst_id}: {err}\n  IT MAY STILL BE BILLING. "
                  f"Run: gpuhop down {inst_id} -y   (and check `gpuhop ls`)", flush=True)
            return False
        finally:
            if restore:
                signal.signal(signal.SIGINT, old if old is not None else signal.SIG_DFL)

    def _final_pull(self, inst_id: int, inst: dict) -> tuple[bool, bool]:
        """(pulled, abandoned). Pressing Ctrl-C twice here is the explicit way to give up on the results."""
        warned = False
        for attempt in range(3):
            try:
                if attempt:
                    time.sleep(min(self.poll_ready_s, 5))
                self.pull(inst_id, inst, self.remote(inst))
                return True, False
            except KeyboardInterrupt:
                if warned:
                    print("\nabandoning the final pull", flush=True)
                    return False, True
                warned = True
                print("\nstill pulling results. Press Ctrl-C again to abandon them and destroy the instance.", flush=True)
            except Exception as e:
                print(f"final pull failed (attempt {attempt + 1}/3): {e}", flush=True)
        return False, False

    def _finish(self, inst_id: int, keep: bool, had_work: bool = True) -> bool:
        """Final pull, then destroy unless that could lose results. False if an instance was left up unintentionally."""
        inst, state = None, "unknown"
        for attempt in range(3):
            try:
                if attempt:
                    time.sleep(min(self.poll_running_s, 5))
                inst = self.provider.show(inst_id)
                state = "gone" if inst is None else inst.get("status") or "unknown"
                break
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"could not read instance {inst_id} (attempt {attempt + 1}/3): {e}", flush=True)
        pulled, abandoned = self._final_pull(inst_id, inst) if inst and state == "running" else (False, False)
        if keep:
            print(f"keeping instance {inst_id} (remember `gpuhop down {inst_id}`)")
            return True
        if state == "gone":
            print(f"instance {inst_id} no longer exists")
            self.state.forget(inst_id)
            return True
        if pulled or abandoned or not had_work:
            return self._destroy_safely(inst_id)
        # its disk may hold results we do not have: never destroy it automatically
        steps = (f"  1) gpuhop pull {inst_id}    2) gpuhop down {inst_id} -y" if state == "running" else
                 f"  1) gpuhop start {inst_id}   2) gpuhop wait {inst_id}\n"
                 f"  3) gpuhop pull {inst_id}    4) gpuhop down {inst_id} -y")
        armed = self.state.get(inst_id).get("failsafe_at")
        note = (f" The failsafe stops it after {_utc(armed)} (GPU billing ends then; storage continues)." if armed
                else "")
        print(f"NOT destroying instance {inst_id}: it is '{state}' and its results could not be pulled. "
              f"IT MAY STILL BE BILLING.{note}\n{steps}", flush=True)
        return False

    def smoke(self, gpu: str, offer_id=None, allow_dirty=False, keep=False) -> int:
        """Boot + bootstrap (incl. the project's smoke command) on a GPU class, pull the report, destroy."""
        created: list = []
        code = 1
        try:
            failsafe_at = None if keep else time.time() + float(self.proj.cfg["max_hours"]) * 3600 + self.failsafe_grace_s
            inst_id, inst, rem = self.up_with_retry(gpu, offer_id, allow_dirty, created, failsafe_at=failsafe_at)
            meta = self.pull(inst_id, inst, rem) / "_instance" / str(inst_id)
            print((meta / "smoke.json").read_text(encoding="utf-8", errors="replace") if (meta / "smoke.json").exists()
                  else "bootstrap OK (no smoke report)")
            code = 0
        except KeyboardInterrupt:
            print("\ninterrupted")
        except Exception as e:
            print(f"SMOKE FAILED on {gpu}: {e}")
        finally:
            if created:
                self._finish(created[-1], keep=keep, had_work=False)  # a smoke run produces no results to protect
        return code

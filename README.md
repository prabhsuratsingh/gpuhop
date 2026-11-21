# gpuhop

Run your experiments on a rented cloud GPU from your laptop with one command. gpuhop rents the cheapest
matching GPU, pushes your code, sets up the box, runs your jobs one by one, and pulls results, logs and videos
back into your project. Then it destroys the instance, including when a job fails or you press Ctrl-C.

```
pip install gpuhop
gpuhop init                      # scaffold: gpuhop.yaml, setup.sh, smoke.py, train.py, exp/*.yaml
cp .env.example .env                # set VAST_API_KEY
gpuhop doctor
gpuhop exp rtx4090 exp/smoke.yaml
ls results/rtx4090/smoke/
```

It works from Windows, macOS and Linux. It needs only Python ≥ 3.10 and the OpenSSH client, which Windows 10+
ships with. No rsync, Docker or GitHub account is required.

## What one `exp` does

```
validate exp files + package code locally         (mistakes cost nothing)
rent the cheapest matching offer                  (moves on to the next offer if a host never starts)
wait for ssh -> push code + env + runtime -> bootstrap: setup.sh (cached) -> smoke
for each experiment file:  run its jobs one at a time (detached in tmux; survives your laptop sleeping)
                           -> pull results/ + logs
final pull -> destroy
```

A stopped instance whose results could not be pulled is **never** destroyed automatically. gpuhop prints
the commands to recover it instead.

## Templates

| `gpuhop init -t …` | what you get |
|---|---|
| `pytorch` (default) | GPU smoke test (TFLOP/s), an MLP training job that writes `metrics.csv`, `summary.json` and `fit.mp4`, and a lr × seed sweep |
| `isaaclab` | Isaac Lab 2.3 / Isaac Sim 5.1 headless setup, a per-task smoke test, PPO (rsl_rl) training, and policy videos recorded in a separate camera process |

`init` never overwrites existing files (unless you pass `--force`), so you can run it inside an existing
project.

## Project config: `gpuhop.yaml`

```yaml
project: my-rl
provider: vast
code: {mode: tarball, exclude: ["data/*"]}   # or mode: git
image: pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime
image_digest: sha256:…          # optional: refuse to launch if the tag moved upstream
disk_gb: 60
python: python                  # interpreter on the box
setup: setup.sh                 # once per instance; skipped while it + setup_hash_files are unchanged
setup_hash_files: [requirements.txt]
smoke: python smoke.py          # must pass before jobs run; report -> $GPUB_SMOKE_OUT
on_sync: null                   # run after `gpuhop sync`, e.g. "pip install -e . --no-deps"
env: {PYTHONUNBUFFERED: "1"}    # exported to every job
search: {base_query: "…", cpu_regex: null, driver_regex: null}  # regexes on host CPU name / NVIDIA driver
gpus:                           # merged over the built-ins: rtx4090 rtx5090 rtx3090 a100 h100 l40s l4
  rtx4090: {cpu_regex: "EPYC"}  # pin the host CPU class for fair comparisons
  a6000: {query: "gpu_name=RTX_A6000"}
results: {remote_dir: results, local_dir: results, exclude: ["*.pt"]}
lockfile: requirements.lock.txt # written by `gpuhop freeze`
max_hours: 12                   # `exp` watchdog: pull + destroy after this long
bootstrap_timeout_min: 90
require_git_tag: null           # warn if `exp` runs from anything other than this tag (code freeze)
```

Unknown keys are rejected, which catches typos before you pay for a GPU.

### Code delivery

- **tarball** (default): gpuhop ships your working tree as it is, uncommitted changes included. Files your
  `.gitignore` excludes are left out (it uses `git ls-files` when the project is a git repo, otherwise it walks
  the directory). Three things are never shipped: `.env*`, `.gpuhop/` and the local results dir. Shell
  scripts get LF line endings. `provenance.json` records the git SHA, the dirty flag, a hash of the diff and a
  hash of the shipped content, and it is pulled with every result.
- **git**: the box clones your pushed HEAD. The tree must be clean and the commit must be pushed. This needs
  `GPUB_GH_TOKEN` (a fine-grained PAT with read-only Contents access).

`gpuhop sync` re-pushes the code to a live instance. Files deleted locally are deleted on the box too;
files the box created (results) are kept.

## Experiment files: `exp/*.yaml`

```yaml
name: sweep
timeout_min: 30
cmd: python train.py --lr {lr} --seed {seed}      # default command for matrix entries
stop_on_fail: false
stop_session_on_fail: false     # `exp a.yaml b.yaml`: skip b if a failed
jobs:
  - matrix: {lr: [0.003, 0.001], seed: [0, 1, 2]}
    name: "lr{lr}_s{seed}"      # run id template (default: 0.003_s0, ...)
    overrides: {model.width: 256}   # -> {overrides} as key=value tokens
  - name: video
    cmd: python record.py results/sweep/lr0.001_s0
    allow_fail: true
```

You can use these placeholders in `cmd`: every matrix key, `{exp}`, `{run_id}`, `{results}`, `{args}` and
`{overrides}`. Matrix values are shell-quoted. Write `{{ }}` for literal braces.

Every job runs from the project dir and sees these variables:

- `$GPUB_RESULTS` (its own output dir)
- `$GPUB_EXP` and `$GPUB_RUN_ID`
- `$GPUB_GPU`, `$GPUB_DPH` ($/hr, for cost accounting), `$GPUB_HOST_CPU` and `$GPUB_HOST_DRIVER`
- `$GIT_SHA`
- everything under `env:`

Jobs run strictly one at a time. Two jobs sharing one GPU would distort every timing they report.

## Where things end up

```
results/<gpu>/<exp>/<run_id>/...           whatever the job wrote to $GPUB_RESULTS (minus results.exclude)
results/<gpu>/_logs/<instance>/<exp>/<job>.log
results/<gpu>/_instance/<instance>/        bootstrap.log, smoke.json, provenance.json, instance.json (redacted)
```

## Commands

| | |
|---|---|
| `init [-t T] [dir]` | scaffold a project (`--list` shows the templates) |
| `doctor` | check the config, API key, ssh key registration, ssh client and code push |
| `offers <gpu> [--cpu RE]` | cheapest matching offers |
| `exp <gpu> <exp.yaml>…` | the one-shot session above (`--keep`, `--instance ID`, `--max-hours`, `--offer`, `--cpu`, `--wait-offer MIN`) |
| `smoke <gpu>` | rent, bootstrap, smoke test, pull, destroy: checks that a GPU class works |
| `up <gpu>` / `down [id]` | rent and bootstrap / destroy |
| `run <exp.yaml>… [--id]` | start experiments on a live instance (detached) |
| `status`, `logs [--job exp/name \| --bootstrap]`, `pull`, `ssh [--cmd]` | inspect a live instance |
| `sync`, `rebootstrap [--force]`, `wait`, `start` | iterate on a live instance |
| `freeze` | write `requirements.lock.txt` from what a working instance installed on top of the image |
| `ls`, `show` | your instances and their running cost |

Instance ids default to the most recent instance of this project. A GPU alias (`rtx4090`) also works as an id.

## Secrets

`.env` (gitignored by `init`) holds `VAST_API_KEY`, plus optionally `GPUB_SSH_KEY` and `GPUB_GH_TOKEN`. No
secret is ever placed in the container's environment, and the code push never includes `.env*`. Instance
metadata is redacted before it is written to disk.

## Providers

Vast.ai is the only provider so far. Providers implement a small interface in `gpuhop/providers/base.py`:
`search`, `create`, `show`, `list`, `destroy`, `start`, plus the optional `logs` and `ssh_keys`. They return
normalised instance and offer dicts. Everything else (ssh transport, code push, bootstrap, job runner, pull)
is provider-independent.

## Development

```
pip install -e ".[dev]"
pytest
```

`tests/test_session.py` runs whole sessions against a fake provider, with a local directory standing in for
the box. The real `bootstrap.sh` and `jobrunner.py` run under bash (Git Bash on Windows).

## License

MIT, see [LICENSE](LICENSE).

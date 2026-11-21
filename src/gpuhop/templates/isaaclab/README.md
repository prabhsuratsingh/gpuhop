# Isaac Lab: headless Isaac Sim on the box, PPO (rsl_rl) training and policy videos

This template sets up Isaac Lab 2.3.2 / Isaac Sim 5.1 (pip install, PhysX) on the pinned `pytorch/pytorch:2.7.0-cuda12.8` image.

- `setup.sh` installs the system libraries headless Kit needs, writes a Vulkan ICD if the host has none, and pip-installs `requirements.txt`. Once `gpuhop freeze` has written `requirements.lock.txt`, it installs that instead. Optional: set `GPUB_INSTALL_NSYS=1` in `env:` to also install Nsight Systems.
- `smoke.py` boots Isaac Sim headless and steps each task on the GPU, one process per task. The report becomes `smoke.json`.
- `train.py` trains PPO (rsl_rl) with Isaac Lab's own agent config for the task. It writes `summary.json` and the rsl_rl logs, plus a checkpoint that stays on the box.
- `record.py` reloads the latest checkpoint of a run in a separate process with cameras on and writes `<run>.mp4`. Rendering would distort training timings, so it never runs inside training.

```
cp .env.example .env               # set VAST_API_KEY
gpuhop doctor
gpuhop smoke rtx4090            # ~25 min: boot + Isaac install + smoke; prints smoke.json, destroys
gpuhop exp rtx4090 exp/smoke.yaml
ls results/rtx4090/smoke/          # ppo_cartpole/ (summary.json, rsl_rl logs), video/ppo_cartpole.mp4
```

After the first good instance, run `gpuhop up rtx4090`, then `gpuhop freeze` (this writes `requirements.lock.txt`; commit it), then `gpuhop down`. From then on every GPU installs exactly the same packages.

Videos need RT cores (RTX 4090/5090, L40S, L4). A100 and H100 can train but cannot render.

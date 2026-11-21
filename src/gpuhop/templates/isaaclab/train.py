"""PPO (rsl_rl) on an Isaac Lab task with Isaac Lab's shipped agent config: a starting point for your own runs.

    python train.py --task cartpole --seed 0 --max-iterations 150

Writes into --out (default $GPUB_RESULTS):
    rsl_rl/          rsl_rl logs (tensorboard events) + model_<it>.pt (kept on the box: results.exclude)
    agent_cfg.json   the agent config actually used
    metrics.csv      per-iteration env steps, mean episode return and length
    summary.json     wall-clock, env-steps/s, mean episode return at the end, GPU, $/hr
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="cartpole", help="short name (cartpole, ant, humanoid, anymal) or a gym id")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-envs", type=int, default=None, help="default: the task's own")
    p.add_argument("--max-iterations", type=int, default=None, help="default: the agent config's own")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", default=os.environ.get("GPUB_RESULTS", "results/local"))
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from isaac_common import gym_id, launch_app, make_env, shutdown

    app = launch_app(device=a.device, headless=True)
    code, env = 1, None
    try:
        import torch
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
        from rsl_rl.runners import OnPolicyRunner

        agent_cfg = load_cfg_from_registry(gym_id(a.task), "rsl_rl_cfg_entry_point")
        agent_cfg.seed, agent_cfg.device = a.seed, a.device
        if a.max_iterations:
            agent_cfg.max_iterations = a.max_iterations
        env, _ = make_env(a.task, a.device, a.num_envs, seed=a.seed)
        venv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        (out / "agent_cfg.json").write_text(json.dumps(agent_cfg.to_dict(), indent=2, default=str))
        rows = []  # per-iteration stats; rsl_rl keeps its reward buffer local to learn(), so read it in log()

        class Runner(OnPolicyRunner):
            def log(self, locs, *args, **kwargs):
                super().log(locs, *args, **kwargs)
                rew, lens = locs["rewbuffer"], locs["lenbuffer"]
                rows.append((locs["it"] + 1, self.tot_timesteps, sum(rew) / len(rew) if len(rew) else "",
                             sum(lens) / len(lens) if len(lens) else ""))

        runner = Runner(venv, agent_cfg.to_dict(), log_dir=str(out / "rsl_rl"), device=agent_cfg.device)
        t0 = time.perf_counter()
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        steps = agent_cfg.max_iterations * agent_cfg.num_steps_per_env * venv.num_envs
        with open(out / "metrics.csv", "w") as f:
            f.write("iteration,env_steps,return_mean,ep_len_mean\n")
            f.writelines(",".join(map(str, r)) + "\n" for r in rows)
        last = next((r[2] for r in reversed(rows) if r[2] != ""), None)
        summary = {"task": gym_id(a.task), "seed": a.seed, "num_envs": venv.num_envs,
                   "iterations": agent_cfg.max_iterations, "env_steps": steps, "wall_s": round(wall, 2),
                   "env_steps_per_s": round(steps / wall), "final_return_mean": last,
                   "gpu": torch.cuda.get_device_name(0), "gpu_alias": os.environ.get("GPUB_GPU"),
                   "usd_per_hr": os.environ.get("GPUB_DPH"),
                   "usd": round(wall / 3600 * float(os.environ["GPUB_DPH"]), 4) if os.environ.get("GPUB_DPH") else None}
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary), flush=True)
        code = 0
    except Exception:
        import traceback

        traceback.print_exc()
    shutdown(app, code, env)


if __name__ == "__main__":
    sys.exit(main())

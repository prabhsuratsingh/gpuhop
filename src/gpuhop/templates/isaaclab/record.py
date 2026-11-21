"""Record a video of a policy trained by train.py (runs on the box; needs RT cores, e.g. an RTX 4090).

    python record.py results/smoke/ppo_cartpole                     # latest rsl_rl/model_*.pt of that run
    python record.py --random --task cartpole                       # rendering check without a policy

Recording is a separate process from training on purpose: with cameras enabled Isaac Lab renders on every
step, which would distort training timings. Writes <out>/<run>.mp4 and <run>.json (episode returns seen).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", nargs="?", type=Path)
    p.add_argument("--random", action="store_true", help="random actions instead of a trained policy")
    p.add_argument("--task", default=None, help="needed with --random; else read from the run's summary.json")
    p.add_argument("--num-envs", type=int, default=16)
    p.add_argument("--warmup", type=int, default=30, help="steps before recording starts (lets episodes get going)")
    p.add_argument("--steps", type=int, default=300, help="recorded steps")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", type=Path, default=Path(os.environ.get("GPUB_RESULTS", "results/recordings")))
    a = p.parse_args()
    if not a.random and not a.run_dir:
        p.error("give a run dir, or --random --task <task>")
    summary = json.loads((a.run_dir / "summary.json").read_text()) if a.run_dir else {}
    task = a.task or summary.get("task") or "cartpole"
    name = a.run_dir.name if a.run_dir else f"random_{task}"

    from isaac_common import launch_app, make_env, shutdown

    app = launch_app(device=a.device, headless=True, enable_cameras=True)
    import gymnasium as gym
    import torch

    env, env_cfg = make_env(task, a.device, a.num_envs, render_mode="rgb_array")
    env_cfg.viewer.resolution = (1280, 720)
    tmp = a.out / f"_tmp_{name}"
    env = gym.wrappers.RecordVideo(env, video_folder=str(tmp), step_trigger=lambda s: s == a.warmup,
                                   video_length=a.steps, disable_logger=True)
    code, info = 0, {"run": name, "task": task, "num_envs": a.num_envs, "steps": a.steps, "random": a.random}
    try:
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

        if a.random:
            venv, policy = RslRlVecEnvWrapper(env), None
        else:
            from rsl_rl.runners import OnPolicyRunner

            agent_cfg = json.loads((a.run_dir / "agent_cfg.json").read_text())
            venv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.get("clip_actions"))
            ckpts = sorted((a.run_dir / "rsl_rl").glob("model_*.pt"), key=lambda f: int(re.findall(r"\d+", f.stem)[-1]))
            if not ckpts:
                raise FileNotFoundError(f"no rsl_rl/model_*.pt in {a.run_dir} (did training complete?)")
            runner = OnPolicyRunner(venv, agent_cfg, log_dir=None, device=a.device)
            runner.load(str(ckpts[-1]))
            policy = runner.get_inference_policy(device=a.device)
            info["checkpoint"] = ckpts[-1].name
        obs = venv.get_observations()
        ret = torch.zeros(venv.num_envs, device=a.device)
        returns = []
        for _ in range(a.warmup + a.steps):
            with torch.inference_mode():
                act = policy(obs) if policy else torch.rand(venv.num_envs, venv.num_actions, device=a.device) * 2 - 1
                obs, rew, dones, _ = venv.step(act)
            ret += rew
            done = dones.bool()
            returns += ret[done].tolist()
            ret[done] = 0
        info["episodes"] = len(returns)
        info["return_mean"] = sum(returns) / len(returns) if returns else None
    except Exception:
        import traceback

        traceback.print_exc()
        code = 1
    finally:
        env.close()  # flushes the video
    vids = sorted(tmp.glob("*.mp4")) if tmp.exists() else []
    if vids and code == 0:
        a.out.mkdir(parents=True, exist_ok=True)
        shutil.move(str(vids[0]), a.out / f"{name}.mp4")
        info["video"] = f"{name}.mp4"
    elif code == 0:
        print("no video was written (is rendering working? enable_cameras + RT cores needed)", flush=True)
        code = 1
    shutil.rmtree(tmp, ignore_errors=True)
    (a.out / f"{name}.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info), flush=True)
    shutdown(app, code, None)


if __name__ == "__main__":
    sys.exit(main())

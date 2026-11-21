"""Isaac Lab plumbing shared by smoke.py, train.py and record.py.

Import order matters: Isaac Sim must be launched (`launch_app`) before anything imports `isaaclab.*` beyond
`isaaclab.app`.
"""

from __future__ import annotations

import os
import sys

TASKS = {  # short name -> Isaac Lab gym id (Direct workflow)
    "cartpole": "Isaac-Cartpole-Direct-v0",
    "ant": "Isaac-Ant-Direct-v0",
    "humanoid": "Isaac-Humanoid-Direct-v0",
    "anymal": "Isaac-Velocity-Flat-Anymal-C-Direct-v0",
}


def gym_id(task: str) -> str:
    return TASKS.get(task, task)


def launch_app(device: str = "cuda:0", headless: bool = True, **kwargs):
    from isaaclab.app import AppLauncher

    return AppLauncher(headless=headless, device=device, **kwargs).app


def make_env(task: str, device: str, num_envs: int | None, seed: int = 0, render_mode: str | None = None):
    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401  (registers the gym ids)
    from isaaclab_tasks.utils import parse_env_cfg

    env_cfg = parse_env_cfg(gym_id(task), device=device, num_envs=num_envs)
    env_cfg.seed = seed
    return gym.make(gym_id(task), cfg=env_cfg, render_mode=render_mode), env_cfg


def shutdown(app, code: int = 0, env=None):
    """End the process with a meaningful exit code.

    `SimulationApp.close()` terminates the interpreter itself with status 0, so nothing after it runs and a
    failed run would look successful. Flush output first; only use the regular Kit shutdown on success.
    """
    if env is not None:
        try:
            env.close()
        except Exception as e:  # noqa: BLE001
            print(f"[shutdown] env.close failed: {e}", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    if code == 0 and app is not None:
        app.close()  # does not return
    os._exit(code)

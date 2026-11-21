"""Fit a small MLP to a noisy 1-D function: a stand-in for your own training script.

    python train.py --lr 1e-3 --seed 0 --steps 2000 --video

Writes into --out (default $GPUB_RESULTS, set by gpuhop for every job):
    metrics.csv    step, loss, elapsed seconds
    summary.json   final loss, throughput, device, config
    fit.mp4        the fit improving over training (--video; needs imageio + imageio-ffmpeg)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path

import torch
from torch import nn


def target(x: torch.Tensor) -> torch.Tensor:
    return torch.sin(3 * x) + 0.3 * torch.cos(11 * x)


def draw(xs, ys_true, ys_pred, size: int = 256):
    """A tiny plot without matplotlib: white canvas, target in grey, prediction in blue."""
    import numpy as np

    img = np.full((size, size, 3), 255, np.uint8)

    def to_px(y):
        return np.clip(((1.5 - y) / 3.0 * (size - 1)).astype(int), 0, size - 1)

    cols = np.clip(((xs + 1) / 2 * (size - 1)).astype(int), 0, size - 1)
    img[to_px(ys_true), cols] = (150, 150, 150)
    for dy in (-1, 0, 1):
        img[np.clip(to_px(ys_pred) + dy, 0, size - 1), cols] = (30, 90, 220)
    return img


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--batch", type=int, default=4096)
    p.add_argument("--video", action="store_true")
    p.add_argument("--out", default=os.environ.get("GPUB_RESULTS", "results/local"))
    a = p.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = nn.Sequential(nn.Linear(1, a.width), nn.Tanh(), nn.Linear(a.width, a.width), nn.Tanh(),
                          nn.Linear(a.width, 1)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    grid = torch.linspace(-1, 1, 256, device=dev).unsqueeze(1)

    writer = None
    if a.video:
        try:
            import imageio.v2 as imageio

            writer = imageio.get_writer(out / "fit.mp4", fps=20, macro_block_size=1)
        except Exception as e:  # noqa: BLE001
            print(f"no video: {e}")

    t0 = time.perf_counter()
    every = max(1, a.steps // 100)
    with open(out / "metrics.csv", "w", newline="") as f:
        log = csv.writer(f)
        log.writerow(["step", "loss", "elapsed_s"])
        for step in range(1, a.steps + 1):
            x = torch.rand(a.batch, 1, device=dev) * 2 - 1
            y = target(x) + 0.05 * torch.randn_like(x)
            loss = nn.functional.mse_loss(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if step % every == 0 or step == a.steps:
                log.writerow([step, f"{loss.item():.6f}", f"{time.perf_counter() - t0:.3f}"])
                if writer is not None:
                    with torch.no_grad():
                        writer.append_data(draw(grid.squeeze(1).cpu().numpy(), target(grid).squeeze(1).cpu().numpy(),
                                                model(grid).squeeze(1).cpu().numpy()))
    if dev == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    if writer is not None:
        writer.close()
    with torch.no_grad():
        final = nn.functional.mse_loss(model(grid), target(grid)).item()
    summary = {"final_mse": final, "log10_mse": math.log10(final), "steps": a.steps, "elapsed_s": round(elapsed, 3),
               "samples_per_s": round(a.steps * a.batch / elapsed), "device": torch.cuda.get_device_name(0) if dev == "cuda" else dev,
               "gpu_alias": os.environ.get("GPUB_GPU"), "usd_per_hr": os.environ.get("GPUB_DPH"), "config": vars(a)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()

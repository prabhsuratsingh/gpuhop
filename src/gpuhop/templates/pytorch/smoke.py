"""Instance acceptance test: torch sees the GPU, and a bf16/fp32 matmul runs at a sane speed.

Writes a JSON report to $GPUB_SMOKE_OUT (pulled as results/<gpu>/_instance/<id>/smoke.json). Exit code != 0 on
failure, which marks the bootstrap as failed (and `exp` destroys the instance instead of running jobs).
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time


def matmul_tflops(torch, dtype, n: int = 8192, iters: int = 20) -> float:
    a = torch.randn(n, n, device="cuda", dtype=dtype)
    b = torch.randn(n, n, device="cuda", dtype=dtype)
    for _ in range(3):
        a @ b
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        a @ b
    torch.cuda.synchronize()
    return 2 * n**3 * iters / (time.perf_counter() - t) / 1e12


def main() -> int:
    import torch

    report = {"ok": False, "python": platform.python_version(), "torch": torch.__version__,
              "cuda": torch.version.cuda, "gpu_alias": os.environ.get("GPUB_GPU")}
    try:
        assert torch.cuda.is_available(), "torch cannot see a GPU"
        p = torch.cuda.get_device_properties(0)
        report.update(gpu=p.name, vram_gb=round(p.total_memory / 2**30, 1), sm_count=p.multi_processor_count,
                      capability=f"{p.major}.{p.minor}")
        report["tflops_fp32"] = round(matmul_tflops(torch, torch.float32), 1)
        report["tflops_bf16"] = round(matmul_tflops(torch, torch.bfloat16), 1)
        report["ok"] = True
    except Exception as e:  # noqa: BLE001
        report["error"] = f"{type(e).__name__}: {e}"
    print(json.dumps(report, indent=2))
    out = os.environ.get("GPUB_SMOKE_OUT")
    if out:
        with open(out, "w") as f:
            json.dump(report, f, indent=2)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())

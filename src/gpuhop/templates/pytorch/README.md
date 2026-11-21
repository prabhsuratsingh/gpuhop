# Plain PyTorch: GPU smoke test + a small training job with metrics and a video

Files:

- `gpuhop.yaml`: the project config (image, setup, smoke, results).
- `setup.sh`: runs once per instance and installs `requirements.txt`. Torch comes with the image.
- `smoke.py`: checks that torch sees the GPU and measures matmul TFLOP/s. Its report becomes `smoke.json`.
- `train.py`: fits an MLP to a noisy function. It writes `metrics.csv`, `summary.json` and `fit.mp4` to `$GPUB_RESULTS`.
- `exp/smoke.yaml`: one short job. `exp/sweep.yaml` is a learning-rate × seed matrix.

```
cp .env.example .env          # set VAST_API_KEY
gpuhop doctor
gpuhop exp rtx4090 exp/smoke.yaml
ls results/rtx4090/smoke/
```

Everything a job writes under `$GPUB_RESULTS` comes back to `results/<gpu>/<exp>/<run_id>/`. Job logs come back to `results/<gpu>/_logs/`. Instance metadata comes back to `results/<gpu>/_instance/<id>/`: `bootstrap.log`, `smoke.json` and `provenance.json` (which code ran).

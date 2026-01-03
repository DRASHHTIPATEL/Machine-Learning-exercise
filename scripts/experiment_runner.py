#!/usr/bin/env python3
"""Run multiple experiment configs sequentially and collect results.

This runner executes `python src/train.py --config <cfg>` for each provided
config file and collects the best checkpoint's `val_metrics` (if present) into
`outputs/experiments_summary.json`.

It expects to be run from the project root.
"""
import subprocess
import sys
import time
import json
from pathlib import Path
import torch

EXPERIMENTS = [
    "configs/exp_imdb_baseline.yaml",
    "configs/exp_imdb_freeze.yaml",
]

SUMMARY_PATH = Path("outputs/experiments_summary.json")

results = []

for cfg in EXPERIMENTS:
    cfg_path = Path(cfg)
    if not cfg_path.exists():
        print(f"Config {cfg} not found, skipping.")
        continue
    print(f"Starting experiment: {cfg}")
    start = time.time()
    proc = subprocess.run([sys.executable, "src/train.py", "--config", str(cfg_path)], check=False)
    duration = time.time() - start
    status = "ok" if proc.returncode == 0 else f"error:{proc.returncode}"
    out_dir = None
    try:
        cfg_data = json.loads(subprocess.check_output([sys.executable, "-c", f"import yaml,sys;print(json.dumps(yaml.safe_load(open('{cfg}'))))"], stderr=subprocess.DEVNULL).decode())
        out_dir = cfg_data.get("output_dir", None)
    except Exception:
        pass

    metrics = None
    ckpt_path = None
    if out_dir:
        ckpt = Path(out_dir) / "best_model.pt"
        if ckpt.exists():
            ckpt_path = str(ckpt)
            try:
                ckpt_data = torch.load(ckpt, map_location="cpu")
                metrics = ckpt_data.get("val_metrics", None)
            except Exception as e:
                print(f"Could not load checkpoint {ckpt}: {e}")

    results.append({
        "config": cfg,
        "status": status,
        "duration_sec": duration,
        "ckpt_path": ckpt_path,
        "val_metrics": metrics,
    })

print("Saving summary to", SUMMARY_PATH)
SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
SUMMARY_PATH.write_text(json.dumps(results, indent=2))
print("Done.")

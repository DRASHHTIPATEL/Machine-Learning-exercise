Part 1 Implementation — usage

This folder contains a self-contained implementation of the sentiment classifier used for the assessment.

Quick run (smoke test):

```bash
python part1_implementation/train.py --config part1_implementation/config.yaml
```

Evaluate a saved checkpoint:

```bash
python part1_implementation/evaluate.py --ckpt ./outputs/exp_imdb_baseline/best_model.pt --config part1_implementation/config.yaml --out outputs/final --device cpu
```

Notes:
- The default `config.yaml` in this folder is set to `device: cpu` and `small_subset: 200` so the smoke test runs quickly.
- For full runs, edit `config.yaml` and set `small_subset: null` (or remove the key) and `device: cuda` if running on a GPU.

# Transformer Sentiment Classification (IMDb)

This project contains an implementation to fine-tune a transformer model for sentiment analysis on the IMDb dataset. It includes training infrastructure (checkpointing, early stopping, TensorBoard logging), attention visualization, evaluation, and a simple ablation option.

Quickstart

1. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run training (uses `configs/config.yaml`):

```bash
bash scripts/run_train.sh
```

3. View TensorBoard logs:

```bash
tensorboard --logdir runs
```

Design notes, usage, and reproduction steps are in this README and `report.md`.

Large files and GitHub
----------------------

This repository may produce large model checkpoint files (e.g. `outputs/*.pt`) that exceed GitHub's file-size limits. Recommended ways to publish the project:

- Do NOT commit large checkpoints directly to the repo. Instead:
	- Install Git LFS (`git lfs install`) and track model files: `git lfs track "outputs/**/*.pt"` before committing. Ensure Git LFS is enabled for your GitHub repo.
	- Or upload checkpoints to cloud storage (Google Drive, S3) and include download links in the README.
- Commit lightweight artifacts (code, configs, report, small images). The `.gitignore` already excludes large binaries.

If you want, I can add a small deploy script that packages final artifacts and uploads them to an S3 bucket or creates a GitHub Release (you'll need to provide credentials / token if you want me to upload them).

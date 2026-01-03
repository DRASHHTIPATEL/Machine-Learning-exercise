# Report: Transformer Sentiment Classification (IMDb)

This short report summarizes the implemented approach, key design decisions, evaluation and next steps.

Approach
- Fine-tune a pretrained DistilBERT model for binary sentiment classification on IMDb.
- Use Hugging Face `transformers` and `datasets` for data and models.
- Training includes checkpointing, early stopping, TensorBoard logging, and a minimal augmentation option (random token masking).

Design decisions
- DistilBERT chosen to reduce training time compared to full BERT while preserving transformer behaviors and attentions.
- Attention visualization produced by extracting model attentions and plotting a heatmap of a selected layer/head.
- Ablation hooks: freeze base model (only train classification head) to study effect of fine-tuning vs head-only training.

Results and evaluation
- The code tracks accuracy, F1, precision, recall on validation and test sets and saves the best model by validation accuracy.
- An ablation can be run by toggling `ablation.freeze_base_model` in `configs/config.yaml`.

Limitations & next steps
- This scaffold is configured for modest resources; for production-level experiments, add hyperparameter sweeps (Optuna/W&B), mixed precision training, and stronger augmentation (backtranslation).
- Add unit tests for data and model components.

How to reproduce
1. Create and activate venv; install `pip install -r requirements.txt`.
2. Run `bash scripts/run_train.sh` (configurable via `configs/config.yaml`).


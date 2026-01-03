# Report — IMDb Sentiment Classification

This short (1–2 page) report summarizes the project approach, the main evaluation findings, a brief diagnosis of issues observed in the current run, and recommended next steps.

## 1) Problem and approach

- Task: Binary sentiment classification on the IMDb dataset (positive vs negative).
- Model: Fine-tune a pretrained DistilBERT transformer with a classification head.
- Tooling: Hugging Face `transformers` and `datasets`, PyTorch training loop with checkpointing and TensorBoard logging.
- Experiments: Baseline full fine-tuning and a freeze-base-model ablation (only the classification head trained).

Training included standard components: tokenization from the pretrained tokenizer, a classification head (linear layer on top of pooled output), cross-entropy loss, and metrics tracking (accuracy, precision, recall, F1). The repository includes configuration YAMLs to run baseline and ablated experiments.

## 2) Key results (final evaluation)

The saved final evaluation (file: `outputs/final/eval_best_model.json`) shows the following evaluation summary.

- Validation set metrics
  - Accuracy: 0.5192
  - Validation classification report (per-class):
    - Class 0 — precision 0.5192, recall 1.0, f1-score 0.6835, support 1298
    - Class 1 — precision 0.0, recall 0.0, f1-score 0.0, support 1202

- Test set metrics
  - Accuracy: 0.5
  - Test classification report (per-class):
    - Class 0 — precision 0.50, recall 1.0, f1-score 0.6667, support 12500
    - Class 1 — precision 0.0, recall 0.0, f1-score 0.0, support 12500

Confusion matrices (rows=true class, cols=predicted):
- Validation: [[1298, 0], [1202, 0]]
- Test: [[12500, 0], [12500, 0]]

Interpretation: the model is predicting only class 0 for all examples (every prediction falls into the first column). The resulting accuracy is near the random/baseline level for a balanced dataset (around 50%), but class-wise performance is highly imbalanced (class 1 has zero recall/precision).

## 3) Diagnosis — likely causes

The degenerate behavior (predicting a single class) can come from a few common issues. Prioritize these checks in roughly this order:

1. Label / data pipeline issues
   - Verify labels loaded from the dataset are correct (0/1 mapping) and not being converted or overwritten (e.g., all labels set to 0 during dataset creation or collate).
   - Confirm the train/validation/test splits and the dataset sampler are not yielding a single-class training set by mistake.

2. Loss / logits handling bug
   - Ensure the model’s logits are passed into PyTorch's cross-entropy correctly (no accidental argmax before loss; correct shape [batch, num_classes]).
   - Check that target labels are the correct dtype and within [0, C-1].

3. Training/configuration mistakes
   - Learning rate too low / optimizer misconfiguration causing no learning (check loss decrease during training).
   - The classification head or entire model might be frozen unexpectedly (e.g., ablation flag set or optimizer not receiving parameters for the head). The repo contains a freeze ablation; double-check flags used for the run.

4. Logging or checkpoint selection
   - Confirm evaluation is performed on the actual trained model checkpoint. The `outputs/experiments_summary.json` shows `ckpt_path: null` and `val_metrics: null` for recorded runs — investigate checkpoint saving and the evaluation step that produced `eval_best_model.json` to ensure consistency.

5. Class imbalance handling / thresholding
   - Although IMDb is balanced, if the training data became imbalanced accidentally, the model may converge to predicting the majority class. Also confirm no thresholding or custom argmax logic sets a single-class prediction.

## 4) Recommended next steps (actionable)

1. Quick checks (fast, low-effort)
   - Print a few training batch label distributions from the data loader to confirm labels are present and balanced.
   - Print model parameter requires_grad for head and base model.
   - Plot the training loss (TensorBoard or saved logs) to confirm the loss decreases; if it is flat, suspect optimizer/lr/grad flow issues.

2. Debugging experiments (if quick checks inconclusive)
   - Run a short training for 1–2 epochs on a small subset and assert the model can overfit (e.g., 20 examples). If it cannot, there's a training/implementation bug.
   - Replace model output with random labels to verify training loop responds (sanity check for loss computation).

3. Code fixes to consider
   - Ensure the classification head parameters are passed to the optimizer and that freezing flags are applied only when intended.
   - Add explicit assertions in data pipeline to check label ranges and dtype before batching.
   - Save more detailed checkpoints and validation metrics during training (and write their paths into `outputs/experiments_summary.json`) so the best model can be traced and inspected.

4. Longer-term improvements
   - Add unit tests for data loading and for a minimal training step (train for a few iterations on synthetic data and assert loss decreases).
   - Add small hyperparameter sweep (learning rate / weight decay) and consider using mixed precision for faster experiments.

## 5) How to reproduce & quick commands

1. Install dependencies: `pip install -r requirements.txt` (use a venv).
2. Re-run a small smoke experiment (recommended): edit `configs/config_smoke.yaml` or create a small config and run the training script for 1 epoch to validate the training loop.
3. Checkpoints/logs: inspect TensorBoard logs in `runs/` for loss/metrics curves.

## 6) Summary

The current run achieved overall accuracy near 50% but with completely broken per-class behavior: the model predicts only a single class. This suggests a pipeline or training bug (label handling, loss/logits, or frozen head). The next steps are quick diagnostic checks (label printing, parameter checks, overfit small subset) followed by targeted fixes (ensure optimizer includes head params, assert label ranges) and a short re-run. Adding unit tests for data and a short overfit test will make regressions easier to catch going forward.

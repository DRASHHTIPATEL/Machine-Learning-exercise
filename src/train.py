import argparse
import os
import yaml
import math
from typing import Dict, Any, Optional

import torch
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data import get_dataloaders
from model import TransformerClassifier
from utils import compute_metrics, save_checkpoint, load_checkpoint, visualize_attention, timeit


def set_seed(seed: int) -> None:
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=None, return_dict=True)
            logits = out.logits
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    return compute_metrics(all_preds, all_labels)


def train(config: Dict[str, Any]) -> None:
    device = torch.device(config.get("device", "cpu"))
    os.makedirs(config.get("output_dir", "./outputs"), exist_ok=True)

    # allow quick smoke-tests by using a small subset (set `small_subset` in config)
    small_subset = config.get("small_subset", None)
    train_loader, val_loader, test_loader, tokenizer = get_dataloaders(config, small_subset=small_subset)

    model = TransformerClassifier(
        model_name=config.get("model_name"),
        num_labels=int(config.get("num_labels", 2)),
        output_attentions=True,
        freeze_base=config.get("ablation", {}).get("freeze_base_model", False),
    )
    model.to(device)

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=float(config.get("learning_rate", 2e-5)), weight_decay=float(config.get("weight_decay", 0.0)))

    writer = SummaryWriter(log_dir=config.get("logging_dir", "runs") + "/" + timeit())

    # performance / precision options
    use_fp16 = bool(config.get("use_fp16", False))
    grad_accum_steps = int(config.get("gradient_accumulation_steps", 1))
    scaler = None
    if use_fp16 and device.type == "cuda":
        scaler = torch.cuda.amp.GradScaler()

    best_val = None
    patience = int(config.get("early_stopping_patience", 3))
    epochs = int(config.get("num_epochs", 3))
    global_step = 0
    steps_since_improve = 0
    train_steps = 0

    # optionally resume
    if config.get("checkpoint", {}).get("resume", False) and config.get("checkpoint", {}).get("path"):
        ckpt = load_checkpoint(config.get("checkpoint", {}).get("path"), device=str(device))
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        global_step = ckpt.get("global_step", 0)
        print(f"Resumed from checkpoint at step {global_step}")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for step, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            # mixed precision context when using fp16 on CUDA
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, return_dict=True)
                    loss = out.loss / grad_accum_steps
                scaler.scale(loss).backward()
            else:
                out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, return_dict=True)
                loss = out.loss / grad_accum_steps
                loss.backward()

            # gradient accumulation
            if (step + 1) % grad_accum_steps == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()

                running_loss += loss.item() * grad_accum_steps
                global_step += 1
                train_steps += 1

                if global_step % 10 == 0:
                    writer.add_scalar("train/loss", running_loss / 10.0, global_step)
                    running_loss = 0.0

            # checkpointing
            if config.get("checkpoint", {}).get("save_every_n_steps", 0) and global_step % int(config.get("checkpoint", {}).get("save_every_n_steps", 500)) == 0:
                ckpt_path = os.path.join(config.get("output_dir"), f"ckpt_step_{global_step}.pt")
                save_checkpoint({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "global_step": global_step,
                }, ckpt_path)

        # end epoch: validation
        val_metrics = evaluate(model, val_loader, device)
        writer.add_scalars("eval", {f"val_{k}": v for k, v in val_metrics.items()}, epoch)
        print(f"Epoch {epoch+1} val metrics: {val_metrics}")

        if best_val is None or val_metrics["accuracy"] > best_val:
            best_val = val_metrics["accuracy"]
            steps_since_improve = 0
            # save best model
            best_path = os.path.join(config.get("output_dir"), "best_model.pt")
            save_checkpoint({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "global_step": global_step,
                "val_metrics": val_metrics,
            }, best_path)
        else:
            steps_since_improve += 1

        if steps_since_improve >= patience:
            print(f"Early stopping triggered after {steps_since_improve} epochs without improvement.")
            break

    # final test evaluation
    ckpt = os.path.join(config.get("output_dir"), "best_model.pt")
    if os.path.exists(ckpt):
        state = load_checkpoint(ckpt, device=str(device))
        model.load_state_dict(state["model_state_dict"])
    test_metrics = evaluate(model, test_loader, device)
    print(f"Test metrics: {test_metrics}")
    writer.add_scalars("test", {f"test_{k}": v for k, v in test_metrics.items()}, 0)

    # visualize attention for a small batch from validation
    try:
        sample_batch = next(iter(val_loader))
        input_ids = sample_batch["input_ids"].to(device)
        attention_mask = sample_batch["attention_mask"].to(device)
        labels = sample_batch["labels"].to(device)
        preds, logits, attentions = model.predict(input_ids=input_ids, attention_mask=attention_mask)
        # decode the first example for tokens
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().tolist())
        if attentions is not None:
            viz_path = os.path.join(config.get("output_dir"), "attention_example.png")
            visualize_attention(tokens, attentions, viz_path)
            print(f"Saved attention visualization to {viz_path}")
    except Exception as e:
        print(f"Could not generate attention visualization: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train transformer classifier")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    set_seed(int(config.get("seed", 42)))
    train(config)

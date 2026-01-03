import argparse
import json
import os
from pathlib import Path
from typing import Dict

import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from datasets import load_dataset
from transformers import AutoTokenizer

from model import TransformerClassifier


def load_checkpoint(path: str, device: str = "cpu") -> Dict:
    return torch.load(path, map_location=device)


def evaluate_checkpoint(ckpt_path: str, config_path: str, out_dir: str, device: str = "cpu") -> None:
    os.makedirs(out_dir, exist_ok=True)
    # load config
    cfg = None
    try:
        import yaml

        cfg = yaml.safe_load(open(config_path))
    except Exception:
        cfg = {}

    dataset_name = cfg.get("dataset_name", "imdb")
    model_name = cfg.get("model_name", "distilbert-base-uncased")
    max_length = int(cfg.get("max_length", 256))

    # load dataset
    if dataset_name.lower() in ("imdb",):
        ds = load_dataset("imdb")
        val_texts = ds["train"]["text"]
        val_labels = ds["train"]["label"]
        # our train/val split uses 10% for val; reproduce split
        from sklearn.model_selection import train_test_split

        train_texts, val_texts, train_labels, val_labels = train_test_split(
            val_texts, val_labels, test_size=0.1, random_state=int(cfg.get("seed", 42))
        )
        test_texts = ds["test"]["text"]
        test_labels = ds["test"]["label"]
    elif dataset_name.lower() in ("sst2", "glue/sst2"):
        ds = load_dataset("glue", "sst2")
        val_texts = ds["validation"]["sentence"]
        val_labels = ds["validation"]["label"]
        test_texts = ds["test"]["sentence"] if "test" in ds else ds["validation"]["sentence"]
        test_labels = ds["test"]["label"] if "test" in ds else ds["validation"]["label"]
    else:
        ds = load_dataset(dataset_name)
        # assume splits
        val_texts = ds["train"]["text"]
        val_labels = ds["train"]["label"]
        test_texts = ds["test"]["text"]
        test_labels = ds["test"]["label"]

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(texts):
        return tokenizer(list(texts), padding=True, truncation=True, max_length=max_length, return_tensors="pt")

    # build model
    num_labels = int(cfg.get("num_labels", 2))
    model = TransformerClassifier(model_name=model_name, num_labels=num_labels, output_attentions=False, freeze_base=False)
    state = load_checkpoint(ckpt_path, device=device)
    model.load_state_dict(state["model_state_dict"])
    model.to(device)
    model.eval()

    results = {}

    for split_name, texts, labels in [("val", val_texts, val_labels), ("test", test_texts, test_labels)]:
        print(f"Evaluating {split_name} set with {len(labels)} examples")
        batch_size = int(cfg.get("batch_size", 16))
        preds = []
        trues = []
        for i in range(0, len(labels), batch_size):
            batch_texts = texts[i : i + batch_size]
            enc = tokenize(batch_texts)
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask, labels=None, return_dict=True)
                logits = out.logits
                batch_preds = torch.argmax(logits, dim=-1).cpu().numpy().tolist()
            preds.extend(batch_preds)
            trues.extend(list(labels[i : i + batch_size]))

        # metrics
        cm = confusion_matrix(trues, preds)
        report = classification_report(trues, preds, output_dict=True, zero_division=0)
        results[split_name] = {"confusion_matrix": cm.tolist(), "report": report}

        # save confusion matrix plot
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("predicted")
        plt.ylabel("true")
        plt.title(f"Confusion matrix: {Path(ckpt_path).stem} {split_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"confusion_{Path(ckpt_path).stem}_{split_name}.png"))
        plt.close()

    # save results
    with open(os.path.join(out_dir, f"eval_{Path(ckpt_path).stem}.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--out", type=str, default="outputs/eval_results")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    evaluate_checkpoint(args.ckpt, args.config, args.out, args.device)

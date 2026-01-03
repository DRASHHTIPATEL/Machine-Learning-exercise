from typing import Dict, Any, Optional, List, Tuple
import os
import time
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def compute_metrics(preds: List[int], labels: List[int]) -> Dict[str, float]:
    # Use zero_division=0 to avoid exceptions/warnings when a class is missing
    # in predictions or labels (common in tiny / imbalanced subsets).
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, average="binary", zero_division=0)),
        "precision": float(precision_score(labels, preds, average="binary", zero_division=0)),
        "recall": float(recall_score(labels, preds, average="binary", zero_division=0)),
    }


def save_checkpoint(state: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str, device: Optional[str] = None) -> Dict[str, Any]:
    if device is None:
        device = "cpu"
    return torch.load(path, map_location=device)


def visualize_attention(tokens: List[str], attentions: Tuple, out_path: str, layer: int = -1, head: int = 0) -> None:
    """Save an attention heatmap for a selected layer/head.

    attentions: tuple of tensors (layers) each (batch, heads, seq_len, seq_len)
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # take last layer by default
    attn_layer = attentions[layer][0][head].cpu().numpy()  # (seq_len, seq_len)
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(attn_layer, xticklabels=tokens, yticklabels=tokens, cmap="viridis", ax=ax)
    plt.title(f"Attention layer={layer} head={head}")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def timeit() -> str:
    return time.strftime("%Y%m%d_%H%M%S")

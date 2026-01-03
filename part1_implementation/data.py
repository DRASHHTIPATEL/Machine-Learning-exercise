from typing import Dict, Tuple, Optional, List
import random
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split


class TextDataset(Dataset):
    """Simple torch Dataset wrapping tokenized encodings and labels."""

    def __init__(self, encodings: Dict[str, torch.Tensor], labels: List[int]) -> None:
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        # Avoid copying tensors into new tensors (warnings). If the encodings
        # are already torch tensors, index them directly and clone to be safe.
        item = {}
        for k, v in self.encodings.items():
            if isinstance(v, torch.Tensor):
                item[k] = v[idx].clone()
            else:
                item[k] = torch.tensor(v[idx])
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def tokenize_texts(tokenizer, texts: List[str], max_length: int, truncation=True):
    return tokenizer(
        texts,
        padding="max_length",
        truncation=truncation,
        max_length=max_length,
        return_tensors="pt",
    )


def apply_random_token_masking(tokens: List[str], mask_token: str, prob: float = 0.05) -> List[str]:
    # simple augmentation: replace tokens with [MASK] with probability prob
    return [mask_token if random.random() < prob else t for t in tokens]


def get_dataloaders(config: Dict, split_ratio: float = 0.1, small_subset: Optional[int] = None) -> Tuple[DataLoader, DataLoader, DataLoader, AutoTokenizer]:
    """Load dataset, tokenize and return PyTorch DataLoaders for train/val/test.

    Args:
        config: configuration dictionary
        split_ratio: fraction of train used as validation
        small_subset: if set, use a small subset for quick testing

    Returns:
        train_loader, val_loader, test_loader, tokenizer
    """
    model_name = config.get("model_name", "distilbert-base-uncased")
    max_length = int(config.get("max_length", 256))
    batch_size = int(config.get("batch_size", 16))

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Select dataset (support IMDb and SST-2 / GLUE SST-2)
    dataset_name = config.get("dataset_name", "imdb")
    if dataset_name.lower() in ("imdb",):
        ds = load_dataset("imdb")
        train_texts = ds["train"]["text"]
        train_labels = ds["train"]["label"]
        test_texts = ds["test"]["text"]
        test_labels = ds["test"]["label"]
    elif dataset_name.lower() in ("sst2", "glue/sst2", "glue_sst2"):
        # GLUE SST-2 has 'sentence' field and separate validation split
        ds = load_dataset("glue", "sst2")
        train_texts = ds["train"]["sentence"]
        train_labels = ds["train"]["label"]
        # glue provides a validation split; use it as test/val accordingly
        test_texts = ds["test"]["sentence"] if "test" in ds else ds["validation"]["sentence"]
        test_labels = ds["test"]["label"] if "test" in ds else ds["validation"]["label"]
    else:
        # Fallback: try to load by name directly, expecting 'text' and 'label' columns
        ds = load_dataset(dataset_name)
        if "train" not in ds or "test" not in ds:
            raise ValueError(f"Dataset {dataset_name} does not contain expected splits 'train' and 'test'.")
        # assume column names
        if "text" in ds["train"].column_names:
            text_col = "text"
        elif "sentence" in ds["train"].column_names:
            text_col = "sentence"
        else:
            text_col = ds["train"].column_names[0]
        if "label" not in ds["train"].column_names:
            raise ValueError(f"Dataset {dataset_name} does not contain a 'label' column; please provide a supported dataset.")
        train_texts = ds["train"][text_col]
        train_labels = ds["train"]["label"]
        test_texts = ds["test"][text_col]
        test_labels = ds["test"]["label"]

    if small_subset is not None and small_subset > 0:
        train_texts = train_texts[:small_subset]
        train_labels = train_labels[:small_subset]
        test_texts = test_texts[: small_subset // 2]
        test_labels = test_labels[: small_subset // 2]

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts, train_labels, test_size=split_ratio, random_state=int(config.get("seed", 42))
    )

    # Optionally apply very simple augmentation: random token masking
    if config.get("augmentation", {}).get("enable", False):
        mask_prob = float(config.get("augmentation", {}).get("random_token_mask_prob", 0.05))
        mask_token = tokenizer.mask_token or "[MASK]"
        train_texts_aug = []
        for t in train_texts:
            toks = tokenizer.tokenize(t)
            toks = apply_random_token_masking(toks, mask_token, prob=mask_prob)
            train_texts_aug.append(tokenizer.convert_tokens_to_string(toks))
        # Mix original and augmented
        train_texts = train_texts + train_texts_aug
        train_labels = train_labels + train_labels

    # Ensure sequences are plain Python lists of strings for the tokenizer
    train_enc = tokenize_texts(tokenizer, list(train_texts), max_length)
    val_enc = tokenize_texts(tokenizer, list(val_texts), max_length)
    test_enc = tokenize_texts(tokenizer, list(test_texts), max_length)

    train_dataset = TextDataset(train_enc, train_labels)
    val_dataset = TextDataset(val_enc, val_labels)
    test_dataset = TextDataset(test_enc, test_labels)

    num_workers = int(config.get("dataloader_num_workers", 0))
    pin_memory = bool(config.get("dataloader_pin_memory", False))

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
    )

    return train_loader, val_loader, test_loader, tokenizer

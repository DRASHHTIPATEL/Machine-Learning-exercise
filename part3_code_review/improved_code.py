

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TrainConfig:
    vocab_size: int = 10_000
    d_model: int = 512
    nhead: int = 8
    num_layers: int = 6
    dim_feedforward: int = 2048
    dropout: float = 0.1
    max_len: int = 512
    pad_id: int = 0

    lr: float = 3e-4
    weight_decay: float = 0.01
    epochs: int = 10
    grad_clip: float = 1.0


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, : x.size(1), :]


class TransformerLM(nn.Module):
    """
    A simple transformer encoder used for token-level prediction.
    Input: token ids (batch, seq_len)
    Output: logits (batch, seq_len, vocab_size)
    """
    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_id)
        self.positional = SinusoidalPositionalEncoding(cfg.d_model, cfg.max_len)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,   # IMPORTANT: expects (batch, seq, d_model)
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=cfg.num_layers)
        self.fc = nn.Linear(cfg.d_model, cfg.vocab_size)

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        # (seq_len, seq_len) with -inf above diagonal
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
        return torch.triu(mask, diagonal=1)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError(f"Expected input shape (batch, seq_len). Got {tuple(x.shape)}")

        batch, seq_len = x.shape
        if seq_len > self.cfg.max_len:
            raise ValueError(f"seq_len {seq_len} exceeds max_len {self.cfg.max_len}")

        h = self.embedding(x)               # (batch, seq, d_model)
        h = self.positional(h)

        key_padding_mask = None
        if attention_mask is not None:
            if attention_mask.shape != (batch, seq_len):
                raise ValueError("attention_mask must have shape (batch, seq_len)")
            key_padding_mask = (attention_mask == 0)  # True = ignore

        causal = self._causal_mask(seq_len, x.device)

        h = self.transformer(h, mask=causal, src_key_padding_mask=key_padding_mask)
        logits = self.fc(h)                # (batch, seq, vocab)
        return logits


def train_model(
    model: nn.Module,
    data: Iterable[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]],
    cfg: TrainConfig,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    for epoch in range(cfg.epochs):
        total_loss = 0.0
        steps = 0

        for batch in data:
            if len(batch) == 2:
                inputs, targets = batch
                attn_mask = None
            else:
                inputs, targets, attn_mask = batch

            inputs = inputs.to(device)
            targets = targets.to(device)
            attn_mask = attn_mask.to(device) if attn_mask is not None else None

            optimizer.zero_grad(set_to_none=True)

            logits = model(inputs, attention_mask=attn_mask)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=cfg.pad_id,
            )

            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

            optimizer.step()

            total_loss += float(loss.item())
            steps += 1

        print(f"Epoch {epoch+1}/{cfg.epochs} - avg_loss: {total_loss / max(1, steps):.4f}")


# Example usage (train_data must be defined elsewhere):
if __name__ == "__main__":
    cfg = TrainConfig()
    model = TransformerLM(cfg)
    train_model(model, train_data, cfg)

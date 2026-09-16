"""
model.py — CLaSP reimplementation: encoders, projections, contrastive loss.

Per docs/REIMPLEMENTATION_SPEC.md:
  signal encoder : Transformer encoder trained from scratch (see NOTE below)
  text encoder   : T5-Small encoder (google-t5/t5-small), fine-tuned
  projections    : learnable linear L_s, L_t -> common dim d=512
  similarity     : C = tau * (E_t @ E_s^T), tau learnable (init log(1/0.07))
  loss           : L = 0.5*(CE over rows + CE over cols)   (CLIP-style)

NOTE (spec §3 amendment, flagged 2026-07-23): the paper names Informer as the
signal encoder. Informer's distinctive parts (ProbSparse attention, distilling)
are EFFICIENCY approximations of full attention for long sequences. At our max
length (2048) exact full attention is tractable, so we use a standard
Transformer encoder (= Informer's encoder with exact attention, no distilling).
This is at least as expressive; documented as an implementation choice.

Smoke test:  python model.py   (builds model, fake batch forward, prints loss)
"""

from __future__ import annotations
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, T5EncoderModel


@dataclass
class ClaspConfig:
    d_common: int = 512          # shared embedding dim (spec §3)
    d_model: int = 512           # signal encoder width
    n_layers: int = 3            # signal encoder depth
    n_heads: int = 8
    ff_dim: int = 1024
    dropout: float = 0.1
    max_len: int = 2048
    text_model: str = "google-t5/t5-small"   # hidden size 512
    logit_scale_init: float = math.log(1 / 0.07)   # CLIP init


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):                       # x: (B, L, D)
        return x + self.pe[: x.size(1)].unsqueeze(0)


class SignalEncoder(nn.Module):
    """Univariate series -> d_model embedding. Trained from scratch."""

    def __init__(self, cfg: ClaspConfig):
        super().__init__()
        self.value_embed = nn.Linear(1, cfg.d_model)
        self.pos = PositionalEncoding(cfg.d_model, cfg.max_len)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model, nhead=cfg.n_heads,
            dim_feedforward=cfg.ff_dim, dropout=cfg.dropout,
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)
        self.norm = nn.LayerNorm(cfg.d_model)

    def forward(self, series: torch.Tensor, mask: torch.Tensor):
        """series: (B, L) float; mask: (B, L) bool, True = real value."""
        x = self.value_embed(series.unsqueeze(-1))          # (B, L, D)
        x = self.pos(x)
        x = self.encoder(x, src_key_padding_mask=~mask)     # pad = True dort
        x = self.norm(x)
        # masked mean-pool
        m = mask.unsqueeze(-1).float()
        pooled = (x * m).sum(1) / m.sum(1).clamp(min=1e-6)
        return pooled                                        # (B, D)


class ClaspModel(nn.Module):
    def __init__(self, cfg: ClaspConfig | None = None):
        super().__init__()
        self.cfg = cfg or ClaspConfig()
        self.signal_encoder = SignalEncoder(self.cfg)
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.text_model)
        self.text_encoder = T5EncoderModel.from_pretrained(self.cfg.text_model)
        t5_dim = self.text_encoder.config.d_model            # 512 for t5-small
        self.proj_signal = nn.Linear(self.cfg.d_model, self.cfg.d_common)
        self.proj_text = nn.Linear(t5_dim, self.cfg.d_common)
        self.logit_scale = nn.Parameter(
            torch.tensor(self.cfg.logit_scale_init))

    # ---- encoding (also the probe-facing API: embedding-level access) ----

    def encode_series(self, series, mask):
        z = self.signal_encoder(series, mask)
        z = self.proj_signal(z)
        return F.normalize(z, dim=-1)

    def encode_text(self, captions: list[str], device=None):
        device = device or next(self.parameters()).device
        tok = self.tokenizer(captions, padding=True, truncation=True,
                             max_length=64, return_tensors="pt").to(device)
        out = self.text_encoder(**tok).last_hidden_state     # (B, T, 512)
        m = tok.attention_mask.unsqueeze(-1).float()
        pooled = (out * m).sum(1) / m.sum(1).clamp(min=1e-6)
        z = self.proj_text(pooled)
        return F.normalize(z, dim=-1)

    # ---- training ----

    def forward(self, batch):
        z_s = self.encode_series(batch["series"], batch["series_mask"])
        z_t = self.encode_text(batch["captions"],
                               device=batch["series"].device)
        scale = self.logit_scale.exp().clamp(max=100.0)
        logits = scale * z_t @ z_s.T                          # (B, B)
        target = torch.arange(logits.size(0), device=logits.device)
        loss_t = F.cross_entropy(logits, target)        # text -> signal
        loss_s = F.cross_entropy(logits.T, target)      # signal -> text
        loss = 0.5 * (loss_t + loss_s)
        return loss, logits


# --------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)
    model = ClaspModel()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"params: {n_params:.1f}M total / {n_train:.1f}M trainable")

    B, L = 4, 32
    batch = {
        "series": torch.randn(B, L),
        "series_mask": torch.ones(B, L, dtype=torch.bool),
        "captions": ["increases steadily", "flat with a spike",
                     "decreases at the end", "noisy throughout"],
    }
    batch["series_mask"][0, 20:] = False        # exercise padding path
    loss, logits = model(batch)
    print(f"smoke: loss={loss.item():.4f}  logits shape={tuple(logits.shape)}")
    print("OK" if logits.shape == (B, B) and torch.isfinite(loss) else "FAIL")

#!/usr/bin/env python3
"""
diagnose_probe3_g8.py — isolate the G8 failure mechanism (Probe 3, CLaSP).

The runner's G8 fired: the constant signal's embedding moved by 1.3e-1
under the resample condition, despite (a) its surrogate input being the
exact zeros row (proven by the G2 count: 2048/286720 = one full signal),
(b) encode_pool feeding the same zeros row for the baseline, and (c) a
per-sample architecture (model.py read: Linear/PE/TransformerEncoder
with LayerNorm only/mean-pool — no batch-coupled layer). Those three
premises plus the observation are mutually contradictory, so one
premise is false in a way source-reading cannot see. This script tests
each premise separately.

Steps (each with a registered prediction, printed at the end):
  S1  encode_pool baseline; locate the constant's pool row; print its
      index, first 5 coords, L2 norm.
  S2  rebuild the runner's resample surrogate list EXACTLY (same seeds,
      same order); extract the constant's z-normed surrogate row;
      print max|row| (premise a: must be 0.0 exactly).
  S3  encode the zeros row ALONE (batch of 1) -> e_alone.
  S4  encode the zeros row with its 7 ORIGINAL z-normed batch mates
      (reconstructing encode_pool's exact batch for this signal)
      -> e_orig.
  S5  encode the zeros row with its 7 RESAMPLED mates (the runner's
      exact batch) -> e_res.
  S6  recompute emb_c the runner's way over ALL 386 surrogates; find
      which pool row (if any) equals e_alone within 1e-6, i.e. where
      the zeros-embedding actually LANDED (index-alignment check).
  S7  G12 miss mechanism: unique-value counts per TRUCE series.

Predictions registered before the run (record hits/misses):
  D1  S2 max|row| = 0.0 exactly.
  D2  |e_alone - sig_emb[const]| <= 1e-6   (batch-independence, alone)
  D3  |e_orig  - sig_emb[const]| <= 1e-6   (batch reconstruction)
  D4  the informative fork:
        if |e_res - sig_emb[const]| <= 1e-6  -> batch mates innocent;
           suspect index misalignment; S6 will show the zeros-embedding
           at a DIFFERENT pool row than const_idx.
        if |e_res - sig_emb[const]| ~ 1.3e-1 -> batch-mate coupling is
           REAL on this stack despite the per-sample architecture; the
           mechanism hunt moves into the torch encoder internals
           (version printed for the record).
  D5  S7: mean unique values per TRUCE series < 12 (mechanism of the
      G12 registered miss: repeated values raise each unique's draw
      probability, lowering the missing fraction below the 0.30 band
      edge computed for all-distinct values).

Usage:
    python -m models.clasp.diagnose_probe3_g8 \
        --checkpoint results/checkpoints/best_baseline_seed42.pt \
        --ckpt-seed 42
"""

from __future__ import annotations
import argparse
import hashlib
import sys
from collections import defaultdict

import numpy as np
import torch

from dataset import load_pairs, znorm
from models.clasp.evaluate import encode_pool
from models.clasp.model import ClaspModel, ClaspConfig

CONST_ID = "sushi:clean\\00\\0000009"
MAX_TOKENS = 16384


def signal_seed(sample_id: str, cond: str, ckpt_seed: int) -> int:
    h = hashlib.sha256(f"{sample_id}|{cond}|{ckpt_seed}".encode()).hexdigest()
    return int(h[:12], 16)


def resample(raw: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return raw[rng.integers(0, len(raw), size=len(raw))]


@torch.no_grad()
def encode_batch(model, device, rows):
    """Encode a list of 1-D float arrays of EQUAL length as one batch."""
    L0 = len(rows[0])
    x = torch.zeros(len(rows), L0)
    m = torch.ones(len(rows), L0, dtype=torch.bool)
    for k, r in enumerate(rows):
        x[k] = torch.from_numpy(np.ascontiguousarray(r, dtype=np.float32))
    return model.encode_series(x.to(device), m.to(device)).cpu()


@torch.no_grad()
def encode_like_runner(model, device, series_list):
    """Verbatim copy of the runner's encode_znormed."""
    by_len = defaultdict(list)
    for j, s in enumerate(series_list):
        by_len[len(s)].append(j)
    buf = [None] * len(series_list)
    for L0, idxs in sorted(by_len.items()):
        B = max(1, min(64, MAX_TOKENS // L0))
        for i in range(0, len(idxs), B):
            idx = idxs[i:i + B]
            x = torch.zeros(len(idx), L0)
            m = torch.ones(len(idx), L0, dtype=torch.bool)
            for k, j in enumerate(idx):
                x[k] = torch.from_numpy(np.ascontiguousarray(
                    series_list[j], dtype=np.float32))
            zb = model.encode_series(x.to(device), m.to(device)).cpu()
            for k, j in enumerate(idx):
                buf[j] = zb[k]
    return torch.stack(buf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--ckpt-seed", type=int, required=True)
    args = ap.parse_args()

    print(f"torch {torch.__version__}  numpy {np.__version__}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    model = ClaspModel(ClaspConfig())
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval().to(device)

    # ---- S1: baseline pool ----
    sig_ids, sig_emb, queries, q_emb, _ = encode_pool(model, device)
    id2idx = {s: i for i, s in enumerate(sig_ids)}
    ci = id2idx[CONST_ID]
    base = sig_emb[ci]
    print(f"\nS1: constant pool index {ci}; "
          f"sig_emb[const][:5] = {[f'{v:.6f}' for v in base[:5].tolist()]}; "
          f"L2 = {float(base.norm()):.6f}")

    # raw series per pool signal, runner-identical
    pairs = load_pairs(splits=("test",))
    raw_of = {}
    for p in pairs:
        raw_of.setdefault(p.sample_id, np.asarray(p.series, dtype=np.float64))

    # ---- S2: rebuild the runner's resample surrogates, runner-identical ----
    z_sur = []
    for s in sig_ids:
        sd_ = signal_seed(s, "resample", args.ckpt_seed)
        sur = resample(raw_of[s], np.random.default_rng(sd_))
        z_sur.append(znorm(sur))
    row = z_sur[ci]
    print(f"S2: constant surrogate row max|.| = {float(np.max(np.abs(row))):.3e} "
          f"(D1 expects 0.0 exactly); dtype {row.dtype}; len {len(row)}")

    # ---- batch reconstruction: which rows share the constant's batch ----
    # replicate encode grouping to find the constant's batch mates (L=2048)
    by_len = defaultdict(list)
    for j, s in enumerate(sig_ids):
        by_len[len(raw_of[s])].append(j)
    L0 = len(raw_of[CONST_ID])
    B = max(1, min(64, MAX_TOKENS // L0))
    mates = None
    for i in range(0, len(by_len[L0]), B):
        chunk = by_len[L0][i:i + B]
        if ci in chunk:
            mates = chunk
            break
    kpos = mates.index(ci)
    print(f"batch reconstruction: L0={L0}, B={B}, batch={mates} "
          f"(constant at position {kpos})")

    zeros = np.zeros(L0, dtype=np.float64)

    # ---- S3: alone ----
    e_alone = encode_batch(model, device, [zeros])[0]
    d3 = float((e_alone - base).abs().max())
    print(f"S3: zeros ALONE       vs sig_emb[const]: max|d| = {d3:.3e}  "
          f"(D2 expects <=1e-6)")

    # ---- S4: with ORIGINAL znormed mates ----
    rows4 = [znorm(raw_of[sig_ids[j]]) if j != ci else zeros for j in mates]
    e_orig = encode_batch(model, device, rows4)[kpos]
    d4 = float((e_orig - base).abs().max())
    print(f"S4: zeros + orig mates vs sig_emb[const]: max|d| = {d4:.3e}  "
          f"(D3 expects <=1e-6)")

    # ---- S5: with RESAMPLED mates (the runner's batch) ----
    rows5 = [z_sur[j] if j != ci else zeros for j in mates]
    e_res = encode_batch(model, device, rows5)[kpos]
    d5 = float((e_res - base).abs().max())
    print(f"S5: zeros + resamp mates vs sig_emb[const]: max|d| = {d5:.3e}  "
          f"(D4 fork: <=1e-6 -> indexing suspect; ~1.3e-1 -> "
          f"batch coupling real)")

    # ---- S6: full runner-path encoding; where did the zeros-embedding land? ----
    emb_c = encode_like_runner(model, device, z_sur)
    d_runner = float((emb_c[ci] - base).abs().max())
    print(f"S6: full runner path, emb_c[const] vs sig_emb[const]: "
          f"max|d| = {d_runner:.3e} (should reproduce the gate's 1.3e-1)")
    dd = (emb_c - e_alone.unsqueeze(0)).abs().max(dim=1).values
    hits = (dd <= 1e-6).nonzero().flatten().tolist()
    print(f"S6: pool rows whose emb matches the zeros-embedding (<=1e-6): "
          f"{hits} "
          f"(if a row != {ci} appears here, the surrogate list is "
          f"misaligned with sig_ids)")
    if hits and hits != [ci]:
        for h in hits:
            print(f"    row {h} = {sig_ids[h]}")

    # ---- S7: G12 miss mechanism ----
    uniq = [len(np.unique(raw_of[s])) for s in sig_ids
            if not s.startswith("sushi")]
    print(f"\nS7: TRUCE unique values per series: mean {np.mean(uniq):.2f} "
          f"of 12; min {min(uniq)}; series with <12 uniques: "
          f"{sum(1 for u in uniq if u < 12)}/{len(uniq)}  "
          f"(D5 expects mean < 12)")

    print("\nDone. Score D1-D5 against the header before interpreting.")


if __name__ == "__main__":
    main()

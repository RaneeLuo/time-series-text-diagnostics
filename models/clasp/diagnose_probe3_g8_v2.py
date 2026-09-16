#!/usr/bin/env python3
"""
diagnose_probe3_g8_v2.py — second-stage isolation of the G8 failure.

V1 established (all on the user's machine, torch 2.13.0+cpu):
  - the runner's surrogate row for the constant IS exactly zeros (D1 HIT)
  - encoding zeros ALONE, with original mates, with resampled mates, and
    via the full runner path all agree with EACH OTHER (batch composition
    innocent, runner indexing clean)
  - but ALL of them differ from sig_emb[247] by 1.303e-1 (D2/D3 MISSED;
    the v1 fork was built on a false premise and is scored as such)
So: within one process, encode(zeros) != encode_pool's row for the
constant. A deterministic function cannot do that, therefore either
encode_pool did not feed zeros, or its first call is not repeatable.
Source reading has exhausted every visible link (znorm shared, float32
series, same batching, per-sample encoder). V2 measures the two
remaining possibilities directly.

Steps:
  T1  encode_pool call #1  -> sig_emb_1[247]
  T2  encode_pool call #2 (same process) -> sig_emb_2[247];
      compare to call #1 and to encode(zeros alone).
  T3  inline replica of encode_pool's signal loop for the L=2048 group
      ONLY, instrumented: capture the exact float32 tensor row fed for
      the constant BEFORE encoding; print max|row|, dtype, then encode
      that captured batch and compare the constant's output to
      sig_emb_1[247] and to e_alone.
  T4  encode the RAW (un-znormed) constant series alone; compare to
      sig_emb_1[247]  (tests any hidden no-znorm path).
  T5  scan all 386 rows of sig_emb_1 for a match to e_alone (<=1e-6):
      where, if anywhere, does the zeros-embedding live in the pool?
  T6  determinism: encode zeros alone twice; max diff.

Registered predictions (scored against output):
  E1  T6 determinism: the two zeros encodings agree <= 1e-7.
  E2  T3 captured row: max|row| = 0.0 exactly (encode_pool's own code
      path feeds zeros).
  E3  T3 encoded output for the captured batch equals e_alone <= 1e-6
      (same input, same process, same function).
  E4  T2 is registered as an OPEN FORK, honestly: three named outcomes,
      no confident prediction —
        (a) sig_emb_2 == sig_emb_1  != e_alone : encode_pool stably
            differs; combined with E2+E3 this is a contradiction that
            forces a difference we have not yet seen printed; the T3
            capture then becomes the arbiter.
        (b) sig_emb_2 == e_alone != sig_emb_1 : FIRST-CALL anomaly
            (warm-up / kernel-selection state); Probe-2's G8 pass and
            today's fail would both be artifacts of call order; major
            environment finding, affects trust in nothing committed
            (G6 digit-exact passes bound the damage) but must be
            documented.
        (c) sig_emb_2 matches neither: nondeterminism at 1e-1 scale;
            stop everything, environment investigation.
  E5  T4: raw-constant encoding does NOT equal sig_emb_1[247]
      (a no-znorm path would contradict three source reads; registered
      low-confidence AGAINST).

Usage (PowerShell):
    python -m models.clasp.diagnose_probe3_g8_v2 `
        --checkpoint results/checkpoints/best_baseline_seed42.pt
"""

from __future__ import annotations
import argparse
from collections import defaultdict

import numpy as np
import torch

from dataset import load_pairs, znorm
from models.clasp.evaluate import encode_pool
from models.clasp.model import ClaspModel, ClaspConfig

CONST_ID = "sushi:clean\\00\\0000009"
MAX_TOKENS = 16384


@torch.no_grad()
def encode_batch(model, device, rows):
    L0 = len(rows[0])
    x = torch.zeros(len(rows), L0)
    m = torch.ones(len(rows), L0, dtype=torch.bool)
    for k, r in enumerate(rows):
        x[k] = torch.from_numpy(np.ascontiguousarray(r, dtype=np.float32))
    return model.encode_series(x.to(device), m.to(device)).cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    args = ap.parse_args()

    print(f"torch {torch.__version__}  numpy {np.__version__}  "
          f"threads {torch.get_num_threads()}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    model = ClaspModel(ClaspConfig())
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval().to(device)

    # ---- T1: encode_pool call #1 ----
    sig_ids, sig_emb_1, _, _, _ = encode_pool(model, device)
    ci = sig_ids.index(CONST_ID)
    print(f"\nT1: encode_pool #1: const at pool index {ci}; "
          f"emb[:4] = {[f'{v:.6f}' for v in sig_emb_1[ci][:4].tolist()]}")

    # ---- T2: encode_pool call #2, same process ----
    sig_ids2, sig_emb_2, _, _, _ = encode_pool(model, device)
    assert sig_ids2 == sig_ids
    d12 = float((sig_emb_2[ci] - sig_emb_1[ci]).abs().max())
    d12_all = float((sig_emb_2 - sig_emb_1).abs().max())
    print(f"T2: encode_pool #2 vs #1: const row max|d| = {d12:.3e}; "
          f"ALL 386 rows max|d| = {d12_all:.3e}")

    # zeros reference
    L0 = 2048
    zeros = np.zeros(L0, dtype=np.float32)
    e_alone = encode_batch(model, device, [zeros])[0]
    d2a = float((sig_emb_2[ci] - e_alone).abs().max())
    d1a = float((sig_emb_1[ci] - e_alone).abs().max())
    print(f"T2: const row: #1 vs zeros-alone {d1a:.3e}; "
          f"#2 vs zeros-alone {d2a:.3e}   (E4 fork: a/b/c per header)")

    # ---- T3: instrumented inline replica of encode_pool's signal loop ----
    pairs = load_pairs(splits=("test",))
    seen, sids, sser = set(), [], []
    for p in pairs:
        if p.sample_id not in seen:
            seen.add(p.sample_id)
            sids.append(p.sample_id)
            sser.append(p.series)
    assert sids == sig_ids
    by_len = defaultdict(list)
    for j, s in enumerate(sser):
        by_len[len(s)].append(j)
    B = max(1, min(64, MAX_TOKENS // L0))
    target_chunk = None
    for i in range(0, len(by_len[L0]), B):
        chunk = by_len[L0][i:i + B]
        if ci in chunk:
            target_chunk = chunk
            break
    kpos = target_chunk.index(ci)
    x = torch.zeros(len(target_chunk), L0)
    m = torch.ones(len(target_chunk), L0, dtype=torch.bool)
    for k, j in enumerate(target_chunk):
        c = znorm(sser[j])
        x[k] = torch.from_numpy(c)
    row = x[kpos]
    print(f"\nT3: captured encode_pool-path row for const: "
          f"max|row| = {float(row.abs().max()):.3e}  dtype {row.dtype}  "
          f"(E2 expects 0.0); series dtype {sser[ci].dtype}, "
          f"series ptp {float(np.ptp(sser[ci])):.3e}, "
          f"series std {float(sser[ci].std()):.3e}")
    with torch.no_grad():
        z3 = model.encode_series(x.to(device), m.to(device)).cpu()
    e_cap = z3[kpos]
    print(f"T3: encoded captured batch: vs sig_emb_1[const] "
          f"{float((e_cap - sig_emb_1[ci]).abs().max()):.3e}; "
          f"vs zeros-alone {float((e_cap - e_alone).abs().max()):.3e}  "
          f"(E3 expects the second <=1e-6)")

    # ---- T4: raw un-znormed constant alone ----
    e_raw = encode_batch(model, device, [sser[ci].astype(np.float32)])[0]
    print(f"\nT4: RAW constant alone vs sig_emb_1[const]: "
          f"max|d| = {float((e_raw - sig_emb_1[ci]).abs().max()):.3e}  "
          f"(E5 registered AGAINST a match; raw value = "
          f"{float(sser[ci][0]):.6g})")

    # ---- T5: where does the zeros-embedding live in sig_emb_1? ----
    dd = (sig_emb_1 - e_alone.unsqueeze(0)).abs().max(dim=1).values
    hits = (dd <= 1e-6).nonzero().flatten().tolist()
    print(f"\nT5: sig_emb_1 rows matching zeros-embedding (<=1e-6): {hits}")
    for h in hits:
        print(f"    row {h} = {sig_ids[h]}")
    nearest = int(dd.argmin())
    print(f"T5: nearest row overall: {nearest} ({sig_ids[nearest]}) "
          f"at max|d| = {float(dd[nearest]):.3e}")

    # ---- T6: within-process determinism ----
    e_alone2 = encode_batch(model, device, [zeros])[0]
    print(f"\nT6: zeros-alone twice: max|d| = "
          f"{float((e_alone2 - e_alone).abs().max()):.3e}  "
          f"(E1 expects <=1e-7)")

    print("\nDone. Score E1-E5 against the header before interpreting.")


if __name__ == "__main__":
    main()

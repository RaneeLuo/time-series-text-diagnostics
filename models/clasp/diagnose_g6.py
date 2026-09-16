#!/usr/bin/env python3
"""
diagnose_g6.py — investigate the G6 failure of 2026-08-13 (seed 42).

Observed: SUSHI + TRUCE R@5/R@10 digit-exact vs eval_baseline_seed42.json;
only TRUCE R@1 (-3 queries exactly) and MRR (-0.00242) moved.
Hypothesis: environment float drift (torch/numpy upgrade or re-fetched T5
weights) flipping hairline rank-1/2 ties on the 12-point TRUCE substrate.

This script measures instead of assuming:
  D1  recompute the strict table; compare BOTH to the frozen file AND to
      the values observed in this morning's failed run (embedded below).
      - matches this morning exactly  -> drift is STABLE (environment-level)
      - differs from this morning too -> run-to-run nondeterminism (worse,
        different mechanism, report immediately)
  D2  for every TRUCE query, the similarity MARGIN between the ground-truth
      signal and its rank-neighbour; histogram + list of hairline cases.
      Mechanism confirmed if the boundary queries sit within float noise.
  D3  print exact package versions + whether T5 came from local cache.

Usage:
    python -m models.clasp.diagnose_g6 \
        --checkpoint results/checkpoints/best_baseline_seed42.pt
Writes: results/analysis/probe2_g6_diagnosis.json
"""

from __future__ import annotations
import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np
import torch

from dataset import load_pairs
from models.clasp.evaluate import encode_pool, strict_metrics
from models.clasp.model import ClaspModel, ClaspConfig

# strict table observed in the FAILED run earlier today (from pasted output)
OBSERVED_TODAY = {
    "all":   {"recall@1": 0.03986332574031891, "mrr": 0.13129305406967273},
    "truce": {"recall@1": 0.02710027100271003, "mrr": 0.0984880748213777},
    "sushi": {"recall@1": 0.10714285714285714, "mrr": 0.30422215896425653},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--baseline",
                    default="results/experiments/eval_baseline_seed42.json")
    args = ap.parse_args()

    # D3 first: environment
    import numpy, transformers
    print("D3 environment:")
    print(f"  python {platform.python_version()}  torch {torch.__version__}  "
          f"numpy {numpy.__version__}  transformers {transformers.__version__}")
    try:
        from huggingface_hub import scan_cache_dir
        info = scan_cache_dir()
        t5 = [r for r in info.repos if "t5-small" in r.repo_id]
        for r in t5:
            print(f"  HF cache: {r.repo_id}  last_modified={r.last_modified}")
    except Exception as e:
        print(f"  HF cache scan unavailable: {e}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ClaspModel(ClaspConfig())
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval().to(device)

    sig_ids, sig_emb, queries, q_emb, _ = encode_pool(model, device)
    frozen = json.load(open(args.baseline))["strict"]

    # D1: stability check
    rep = {}
    strict_all, ranks = strict_metrics(sig_ids, sig_emb, queries, q_emb)
    rep["all"] = strict_all
    for ds_group, name in [(("truce_stock", "truce_synth"), "truce"),
                           (("sushi",), "sushi")]:
        qsel = [i for i, q in enumerate(queries) if q[2] in ds_group]
        m, _ = strict_metrics(sig_ids, sig_emb,
                              [queries[i] for i in qsel], q_emb[qsel])
        rep[name] = m
    print("\nD1 stability (this run vs THIS MORNING's failed run):")
    stable = True
    for k, mets in OBSERVED_TODAY.items():
        for met, v in mets.items():
            d = abs(rep[k][met] - v)
            stable &= d < 1e-12
            print(f"  {k}/{met}: now {rep[k][met]:.12f}  morning {v:.12f}  "
                  f"|d|={d:.1e}")
    print(f"  -> {'STABLE (environment-level drift)' if stable else 'RUN-TO-RUN NONDETERMINISM — report immediately'}")
    print("\nD1 vs frozen baseline (for the record):")
    for k in ("all", "truce", "sushi"):
        for met in ("recall@1", "recall@5", "recall@10", "mrr"):
            d = abs(rep[k][met] - frozen[k][met])
            if d > 1e-12:
                print(f"  {k}/{met}: now {rep[k][met]:.12f}  "
                      f"frozen {frozen[k][met]:.12f}  |d|={d:.1e}")

    # D2: TRUCE rank-boundary margins
    sims = (q_emb @ sig_emb.T).numpy()
    id2idx = {s: i for i, s in enumerate(sig_ids)}
    pairs = load_pairs(splits=("test",))
    assert [(p.caption, p.sample_id, p.dataset) for p in pairs] == \
           [(q[0], q[1], q[2]) for q in queries]

    rows = []
    for qi, p in enumerate(pairs):
        if p.dataset == "sushi":
            continue
        gt = id2idx[p.sample_id]
        s = sims[qi]
        order = np.argsort(-s)
        rank = int(np.where(order == gt)[0][0]) + 1
        # margin to the score just above (if rank>1) and just below (if any)
        above = s[order[rank - 2]] - s[gt] if rank > 1 else np.inf
        below = s[gt] - s[order[rank]] if rank < len(s) else np.inf
        rows.append({"caption_id": p.caption_id, "rank": rank,
                     "margin_above": float(above), "margin_below": float(below)})

    for thr in (1e-4, 1e-5, 1e-6):
        n12 = sum(1 for r in rows if r["rank"] in (1, 2)
                  and min(r["margin_above"], r["margin_below"]) < thr)
        print(f"\nD2: TRUCE queries at rank 1-2 with boundary margin < {thr:g}: "
              f"{n12}" if thr == 1e-4 else
              f"D2: ... margin < {thr:g}: {n12}")
    hair = sorted([r for r in rows if r["rank"] in (1, 2)
                   and min(r["margin_above"], r["margin_below"]) < 1e-4],
                  key=lambda r: min(r["margin_above"], r["margin_below"]))
    print("  hairline queries (rank, margin, caption_id):")
    for r in hair[:12]:
        print(f"    rank {r['rank']}  margin "
              f"{min(r['margin_above'], r['margin_below']):.2e}  "
              f"{r['caption_id']}")

    out = Path("results/analysis/probe2_g6_diagnosis.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"stable_vs_morning": bool(stable),
                   "strict_now": rep,
                   "hairline_1e4": hair,
                   "env": {"python": platform.python_version(),
                           "torch": torch.__version__,
                           "numpy": numpy.__version__,
                           "transformers": transformers.__version__}},
                  f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

"""
eval_table3.py — reproduction-fidelity check against CLaSP Table III.

The paper reports soft mAP@10 under FOUR judge/threshold combinations:

                  SBERT ts=0.5   SBERT ts=0.8   DistilBERT ts=0.5   DistilBERT ts=0.8
    TRUCE             0.458          0.136            1.000               0.491
    SUSHI             0.982          0.571            1.000               0.992
    TRUCE+SUSHI       0.954          0.556            0.959               0.960

Matching ONE cell is weak evidence. Matching the SHAPE of the whole grid
(TRUCE collapsing at ts=0.8, DistilBERT inflating everything toward 1.0)
is strong evidence that our reimplementation behaves like theirs.

Why DistilBERT inflates: it is a plain language model, not a sentence-embedding
model. Mean-pooled BERT-family embeddings are anisotropic -- almost any two
English sentences score high cosine similarity -- so nearly every retrieved item
clears ts=0.5. The paper's own 1.000 column is the fingerprint of that effect.

Usage (from repo root):
    python -m models.clasp.eval_table3 --checkpoint results/checkpoints/best_baseline_seed42.pt
"""

from __future__ import annotations
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from models.clasp.model import ClaspModel, ClaspConfig
from models.clasp.evaluate import encode_pool

# judge encoders. "sbert" = a real sentence-embedding model;
# "distilbert" = plain LM, sentence-transformers wraps it with MEAN pooling
# (it prints a warning saying so -- that is expected, not an error).
JUDGES = {
    "sbert": "sentence-transformers/all-MiniLM-L6-v2",
    "distilbert": "distilbert-base-uncased",
}
THRESHOLDS = (0.5, 0.8)

PAPER = {  # Table III, for side-by-side printing
    ("sbert", 0.5):      {"truce": 0.458, "sushi": 0.982, "all": 0.954},
    ("sbert", 0.8):      {"truce": 0.136, "sushi": 0.571, "all": 0.556},
    ("distilbert", 0.5): {"truce": 1.000, "sushi": 1.000, "all": 0.959},
    ("distilbert", 0.8): {"truce": 0.491, "sushi": 0.992, "all": 0.960},
}


def judge_similarities(judge_model: str, queries, sig_ids, sig_captions):
    """Return (Q, S) matrix: semantic similarity of each query caption to each
    pool signal, taking the max over that signal's own captions."""
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer(judge_model)

    all_caps = sorted({c for caps in sig_captions.values() for c in caps}
                      | {q[0] for q in queries})
    vecs = enc.encode(all_caps, batch_size=128, convert_to_numpy=True,
                      normalize_embeddings=True, show_progress_bar=False)
    cvec = {c: vecs[i] for i, c in enumerate(all_caps)}

    Q = np.stack([cvec[q[0]] for q in queries])                    # (Q, D)
    sim = np.zeros((len(queries), len(sig_ids)), dtype=np.float32)
    for si, sid in enumerate(sig_ids):
        M = np.stack([cvec[c] for c in sig_captions[sid]])         # (n_caps, D)
        sim[:, si] = (Q @ M.T).max(axis=1)
    return sim


def map10_from_relevance(model_sims, judge_sim, queries, ts):
    """Same AP@10 definition as evaluate.py: precision_sum / min(R, 10)."""
    ap_by_ds = defaultdict(list)
    for qi, (_cap, _gt, ds) in enumerate(queries):
        rel = judge_sim[qi] > ts
        R = int(rel.sum())
        if R == 0:
            ap_by_ds[ds].append(0.0)
            ap_by_ds["all"].append(0.0)
            continue
        top10 = np.argsort(-model_sims[qi])[:10]
        hits, psum = 0, 0.0
        for r, si in enumerate(top10, start=1):
            if rel[si]:
                hits += 1
                psum += hits / r
        ap = psum / min(R, 10)
        ap_by_ds[ds].append(ap)
        ap_by_ds["all"].append(ap)
    # collapse truce_stock / truce_synth into one "truce" row, like the paper
    merged = defaultdict(list)
    for k, v in ap_by_ds.items():
        key = "truce" if k.startswith("truce") else k
        merged[key].extend(v)
    return {k: float(np.mean(v)) for k, v in merged.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--untrained", action="store_true",
                    help="evaluate a randomly initialised model (negative control)")
    ap.add_argument("--seed", type=int, default=42,
                    help="seed for --untrained initialisation (reproducibility)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if not args.checkpoint and not args.untrained:
        raise SystemExit("give --checkpoint PATH or --untrained")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.untrained:
        torch.manual_seed(args.seed)
    model = ClaspModel(ClaspConfig())
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(state["model"] if "model" in state else state)
        print(f"loaded checkpoint: {args.checkpoint}")
    else:
        print(f"UNTRAINED model (random init, seed {args.seed}) "
              f"-- negative control")
    model.to(device).eval()

    sig_ids, sig_emb, queries, q_emb, sig_captions = encode_pool(model, device)
    model_sims = (q_emb @ sig_emb.T).numpy()
    print(f"pool: {len(sig_ids)} signals, {len(queries)} queries\n")

    results = {}
    for jname, jmodel in JUDGES.items():
        print(f"[judge] {jname} ({jmodel})")
        jsim = judge_similarities(jmodel, queries, sig_ids, sig_captions)
        # diagnostic: how permissive is this judge?
        for ts in THRESHOLDS:
            frac = float((jsim > ts).mean())
            print(f"    ts={ts}: {frac:.1%} of all (query, signal) pairs judged relevant")
        for ts in THRESHOLDS:
            got = map10_from_relevance(model_sims, jsim, queries, ts)
            results[f"{jname}_ts{ts}"] = got

    # side-by-side table
    print("\n" + "=" * 72)
    print(f"{'judge/ts':<18}{'row':<8}{'ours':>10}{'paper':>10}{'diff':>10}")
    print("=" * 72)
    for (jname, ts), paper_row in PAPER.items():
        key = f"{jname}_ts{ts}"
        for row in ("truce", "sushi", "all"):
            ours = results[key].get(row, float("nan"))
            pap = paper_row[row]
            print(f"{key:<18}{row:<8}{ours:>10.3f}{pap:>10.3f}{ours - pap:>+10.3f}")
    print("=" * 72)
    if args.untrained:
        print("UNTRAINED control: any column where this scores high is a column")
        print("that cannot distinguish a trained model from a random one.")
    else:
        print("Read the SHAPE, not the cells: does TRUCE collapse at ts=0.8?")
        print("Does DistilBERT inflate everything toward 1.0? Same direction = faithful.")

    out = Path(args.out) if args.out else Path(
        "results/experiments/table3_fidelity_untrained.json" if args.untrained
        else "results/experiments/table3_fidelity.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"checkpoint": args.checkpoint or f"UNTRAINED(seed={args.seed})",
                   "ours": results,
                   "paper_table3": {f"{j}_ts{t}": v for (j, t), v in PAPER.items()}},
                  f, indent=2)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()

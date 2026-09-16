"""
evaluate.py — CLaSP evaluation: BOTH harnesses per docs/REIMPLEMENTATION_SPEC.md §4.

  A. Paper-protocol soft mAP@10 : correctness judged by an independent SBERT
     (cos-sim between query caption and retrieved signal's own caption > ts).
     Compare against paper Table III: TRUCE 0.458 / SUSHI 0.982 / both 0.954
     (ballpark + pattern, not exact — see spec §4A).
  B. Strict pair-level retrieval : Recall@1/5/10 and MRR of the ground-truth
     signal. This is the probe-facing metric (all probe degradations use this).

Documented choices (paper under-specifies):
  - Signal pool = union of ALL test-split signals (both datasets), one entry
    per unique sample_id. Query sets reported separately: truce / sushi / all.
  - A TRUCE signal has 3 captions; for soft relevance a retrieved signal counts
    correct if ANY of its own captions clears the threshold vs the query.
  - SBERT = sentence-transformers/all-MiniLM-L6-v2, ts = 0.5.

Usage:
    python -m models.clasp.evaluate --checkpoint results/checkpoints/best.pt
    python -m models.clasp.evaluate --untrained          # sanity: ~chance
Requires:  pip install sentence-transformers
"""

from __future__ import annotations
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from dataset import load_pairs, znorm
from models.clasp.model import ClaspModel, ClaspConfig

RESULTS = Path("results")


# ----------------------------------------------------------------------
# encoding helpers
# ----------------------------------------------------------------------

@torch.no_grad()
def encode_pool(model, device, split="test"):
    """Encode unique test signals and all test captions.
    Returns: sig_ids, sig_emb (S,d), queries list, q_emb (Q,d),
             sig_captions: sample_id -> [its own captions]"""
    pairs = load_pairs(splits=(split,))
    # unique signals
    seen, sig_ids, sig_series = set(), [], []
    sig_captions = defaultdict(list)
    for p in pairs:
        sig_captions[p.sample_id].append(p.caption)
        if p.sample_id not in seen:
            seen.add(p.sample_id)
            sig_ids.append(p.sample_id)
            sig_series.append(p.series)

    # batch-encode signals grouped by EXACT length; token-budget batch sizes so
    # the L×L attention matrix fits in RAM (64×8×2048²×4B = the 8.6 GB OOM)
    MAX_TOKENS = 16384                  # L=2048 -> B=8 ; L=12 -> B=64 (cap)
    by_len = defaultdict(list)
    for j, s in enumerate(sig_series):
        by_len[len(s)].append(j)
    emb_buf = [None] * len(sig_ids)
    for L0, idxs in sorted(by_len.items()):
        B = max(1, min(64, MAX_TOKENS // L0))
        for i in range(0, len(idxs), B):
            idx = idxs[i:i + B]
            chunk = [znorm(sig_series[j]) for j in idx]
            x = torch.zeros(len(chunk), L0)
            m = torch.ones(len(chunk), L0, dtype=torch.bool)
            for k, c in enumerate(chunk):
                x[k] = torch.from_numpy(c)
            z = model.encode_series(x.to(device), m.to(device)).cpu()
            for k, j in enumerate(idx):
                emb_buf[j] = z[k]
    sig_emb = torch.stack(emb_buf)

    # queries = every (caption, gt sample_id, dataset)
    queries = [(p.caption, p.sample_id, p.dataset) for p in pairs]
    q_emb = []
    for i in range(0, len(queries), B):
        caps = [q[0] for q in queries[i:i + B]]
        q_emb.append(model.encode_text(caps, device=device).cpu())
    q_emb = torch.cat(q_emb)
    return sig_ids, sig_emb, queries, q_emb, dict(sig_captions)


# ----------------------------------------------------------------------
# harness B: strict Recall@k / MRR
# ----------------------------------------------------------------------

def strict_metrics(sig_ids, sig_emb, queries, q_emb):
    id2idx = {s: i for i, s in enumerate(sig_ids)}
    sims = q_emb @ sig_emb.T                       # (Q, S) cosine (normalized)
    ranks = []
    for qi, (_, gt, _) in enumerate(queries):
        order = torch.argsort(sims[qi], descending=True)
        rank = (order == id2idx[gt]).nonzero(as_tuple=True)[0].item() + 1
        ranks.append(rank)
    ranks = np.array(ranks)
    out = {
        "recall@1": float((ranks <= 1).mean()),
        "recall@5": float((ranks <= 5).mean()),
        "recall@10": float((ranks <= 10).mean()),
        "mrr": float((1.0 / ranks).mean()),
        "n_queries": int(len(ranks)),
        "pool_size": int(len(sig_ids)),
    }
    return out, ranks


# ----------------------------------------------------------------------
# harness A: paper-protocol soft mAP@10 (SBERT ts=0.5)
# ----------------------------------------------------------------------

def soft_map10(sig_ids, sig_emb, queries, q_emb, sig_captions,
               ts=0.5, sbert_name="sentence-transformers/all-MiniLM-L6-v2"):
    from sentence_transformers import SentenceTransformer
    sbert = SentenceTransformer(sbert_name)

    # SBERT-embed every distinct caption once
    all_caps = sorted({c for caps in sig_captions.values() for c in caps}
                      | {q[0] for q in queries})
    cap_vec = sbert.encode(all_caps, batch_size=128, convert_to_numpy=True,
                           normalize_embeddings=True, show_progress_bar=False)
    cvec = {c: cap_vec[i] for i, c in enumerate(all_caps)}

    # per-signal caption matrix for max-similarity relevance
    sig_cap_vecs = [np.stack([cvec[c] for c in sig_captions[s]]) for s in sig_ids]

    sims = (q_emb @ sig_emb.T).numpy()
    ap_by_ds = defaultdict(list)
    for qi, (qcap, _gt, ds) in enumerate(queries):
        qv = cvec[qcap]
        # relevance of every pool signal to this query (semantic, max over its captions)
        rel = np.array([float((m @ qv).max() > ts) for m in sig_cap_vecs])
        R = rel.sum()
        if R == 0:
            ap_by_ds[ds].append(0.0)
            ap_by_ds["all"].append(0.0)
            continue
        top10 = np.argsort(-sims[qi])[:10]
        hits, precision_sum = 0, 0.0
        for r, si in enumerate(top10, start=1):
            if rel[si]:
                hits += 1
                precision_sum += hits / r
        ap = precision_sum / min(R, 10)
        ap_by_ds[ds].append(ap)
        ap_by_ds["all"].append(ap)
    return {k: {"mAP@10": float(np.mean(v)), "n": len(v)}
            for k, v in ap_by_ds.items()}


# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, default=None)
    ap.add_argument("--untrained", action="store_true")
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--skip-soft", action="store_true",
                    help="skip SBERT harness (no sentence-transformers needed)")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ClaspModel(ClaspConfig())
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(state["model"] if "model" in state else state)
        print(f"loaded checkpoint: {args.checkpoint}")
    elif not args.untrained:
        raise SystemExit("give --checkpoint PATH or --untrained")
    model.eval().to(device)

    sig_ids, sig_emb, queries, q_emb, sig_captions = encode_pool(
        model, device, split=args.split)

    report = {"checkpoint": args.checkpoint, "split": args.split}

    # strict, per query-source and overall
    strict_all, _ = strict_metrics(sig_ids, sig_emb, queries, q_emb)
    report["strict"] = {"all": strict_all}
    for ds_group, name in [(("truce_stock", "truce_synth"), "truce"),
                           (("sushi",), "sushi")]:
        qsel = [i for i, q in enumerate(queries) if q[2] in ds_group]
        sub_q = [queries[i] for i in qsel]
        m, _ = strict_metrics(sig_ids, sig_emb, sub_q, q_emb[qsel])
        report["strict"][name] = m

    # soft (paper protocol)
    if not args.skip_soft:
        # group truce datasets under one key to mirror Table III rows
        merged = soft_map10(sig_ids, sig_emb,
                            [(c, g, "truce" if d.startswith("truce") else d)
                             for c, g, d in queries],
                            q_emb, sig_captions)
        report["soft_mAP@10_ts0.5"] = merged
        print("\npaper-protocol soft mAP@10 (compare Table III ~ TRUCE 0.458 / "
              "SUSHI 0.982 / all 0.954):")
        for k, v in merged.items():
            print(f"  {k}: {v['mAP@10']:.3f}  (n={v['n']})")

    print("\nstrict retrieval (probe-facing):")
    for k, v in report["strict"].items():
        print(f"  {k}: R@1={v['recall@1']:.3f} R@5={v['recall@5']:.3f} "
              f"R@10={v['recall@10']:.3f} MRR={v['mrr']:.3f} "
              f"(n={v['n_queries']}, pool={v['pool_size']})")

    out = args.out or (RESULTS / "experiments" /
                       ("eval_untrained.json" if args.untrained else "eval.json"))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()

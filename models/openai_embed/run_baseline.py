"""
run_baseline.py (text-embedding-3-large) — strict retrieval baseline.

Fills the empty baseline cell in the evaluation matrix for the floor model.
Protocol is IDENTICAL to models/clasp/evaluate.py harness B (the probe-facing
metric): every test caption is a query; the pool is every unique test signal;
we rank the ground-truth signal and report Recall@1/5/10 and MRR, per query
source (truce / sushi / all).

Deliberately NOT run here: the paper-protocol soft mAP@10. That harness exists
only to compare our CLaSP reimplementation against the CLaSP paper's Table III.
It has no reference value for this model; running it would invite an
apples-to-oranges number into the thesis. Recorded in the output JSON.

Serialisation is IMPORTED from run_probe1.py, not copied, so it cannot drift
from what the probe used — and the embedding cache is shared (same keys), so
signals already embedded for the probe are not paid for twice.

Registered expectations (2026-08-09, before first run):
    pool size          exactly 386   (HARD GATE — documented, state doc §2)
    sushi test signals 140           (documented: one per class)
    truce test signals 246           (derived: 386 - 140)
    queries            878           (246*3 + 140; dataset.py asserts 3
                                      captions per TRUCE record)
    max signal tokens  ~4096, under the 8191 limit
    fresh cost         <= ~$0.08; with probe cache likely <= ~$0.02
    result             near random ranking (R@1 0.003 / R@10 0.026 / MRR 0.017)

Usage:
    python -m models.openai_embed.run_baseline --dry-run
    python -m models.openai_embed.run_baseline --yes

Writes: results/experiments/baseline_openai_embed.json
"""

from __future__ import annotations
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from dataset import load_pairs
from models.openai_embed.run_probe1 import (
    serialize, count_tokens, load_cache, embed_all,
    MODEL, MAX_TOKENS, PRICE_PER_1M,
)

OUT = Path("results/experiments/baseline_openai_embed.json")

EXPECTED_POOL = 386          # hard gate; documented pool size (state doc §2)
EXPECTED_QUERIES = 878       # registered expectation; soft (warn, don't stop)
CHANCE = {                   # uniform-random-ranking references for pool 386
    "recall@1": 1 / 386,
    "recall@10": 10 / 386,
    "mrr": float(np.mean(1.0 / np.arange(1, 387))),   # H_386 / 386
}


def gate(ok: bool, name: str, msg: str):
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name}: {msg}")
    if not ok:
        raise SystemExit(f"gate {name} failed — stopping. {msg}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="counts, tokens, cost, example; no API calls")
    ap.add_argument("--yes", action="store_true", help="proceed with API calls")
    ap.add_argument("--scale", type=int, default=10,
                    help="serialisation scale; MUST stay 10 to match the probe")
    ap.add_argument("--batch-signals", type=int, default=8)
    ap.add_argument("--batch-captions", type=int, default=128)
    args = ap.parse_args()

    # ---------------- load test split ----------------
    pairs = load_pairs(splits=("test",))
    seen, sig_ids, sig_series, sig_ds = set(), [], [], {}
    for p in pairs:
        if p.sample_id not in seen:
            seen.add(p.sample_id)
            sig_ids.append(p.sample_id)
            sig_series.append(p.series)
            sig_ds[p.sample_id] = p.dataset
    queries = [(p.caption, p.sample_id, p.dataset) for p in pairs]

    n_truce_sig = sum(1 for s in sig_ids if sig_ds[s].startswith("truce"))
    n_sushi_sig = sum(1 for s in sig_ids if sig_ds[s] == "sushi")
    n_truce_q = sum(1 for q in queries if q[2].startswith("truce"))
    n_sushi_q = sum(1 for q in queries if q[2] == "sushi")

    print("=" * 74)
    print("SELF-REPORT — counts (check against registered expectations)")
    print("=" * 74)
    print(f"  test pairs (= queries): {len(queries)}   [registered: {EXPECTED_QUERIES}]")
    print(f"  unique signals (pool) : {len(sig_ids)}   [registered: {EXPECTED_POOL}]")
    print(f"  signals  truce/sushi  : {n_truce_sig} / {n_sushi_sig}   [registered: 246 / 140]")
    print(f"  queries  truce/sushi  : {n_truce_q} / {n_sushi_q}")
    print("\nGATES")
    gate(len(sig_ids) == EXPECTED_POOL, "G1-pool",
         f"pool {len(sig_ids)} == documented {EXPECTED_POOL}")
    gate(n_truce_sig + n_sushi_sig == len(sig_ids), "G2-split",
         "every pool signal is truce or sushi (arithmetic closes)")
    gate(n_truce_q + n_sushi_q == len(queries), "G3-queries",
         "every query is truce or sushi (arithmetic closes)")
    gate(n_truce_q == 3 * n_truce_sig, "G4-truce-caps",
         f"truce queries {n_truce_q} == 3 x truce signals {n_truce_sig}")
    gate(n_sushi_q == n_sushi_sig, "G5-sushi-caps",
         f"sushi queries {n_sushi_q} == sushi signals {n_sushi_sig}")
    if len(queries) != EXPECTED_QUERIES:
        print(f"  [WARN] query count {len(queries)} != registered "
              f"{EXPECTED_QUERIES} — investigate before trusting results")

    # ---------------- serialise + token report ----------------
    if args.scale != 10:
        print("  [WARN] --scale != 10 breaks cache reuse AND comparability "
              "with the probe serialisation. Only do this deliberately.")
    sig_text = {s: serialize(x, args.scale) for s, x in zip(sig_ids, sig_series)}
    captions = sorted({q[0] for q in queries})

    st = count_tokens([sig_text[s] for s in sig_ids])
    ct = count_tokens(captions)
    total = sum(st) + sum(ct)
    over = [s for s, n in zip(sig_ids, st) if n > MAX_TOKENS]

    print("\nTOKENS / COST")
    print(f"  distinct caption texts: {len(captions)} "
          f"(of {len(queries)} queries; duplicates embedded once)")
    print(f"  signal tokens : min {min(st)}, median {sorted(st)[len(st)//2]}, "
          f"max {max(st)}   (limit {MAX_TOKENS})")
    print(f"  total tokens  : {total:,}   fresh-cost ceiling "
          f"${total / 1e6 * PRICE_PER_1M:.2f}  (cache reduces this)")
    gate(not over, "G6-token-limit",
         f"{len(over)} signals over the input limit" if over
         else "no signal exceeds the input limit")

    ex = sig_ids[0]
    print(f"\n  example serialisation [{ex}], first 160 chars:")
    print("    " + sig_text[ex][:160] + " ...")

    if args.dry_run or not args.yes:
        print("\ndry run — no API calls made. Re-run with --yes to embed.")
        return

    # ---------------- embed (shared cache with the probe) ----------------
    cache = load_cache()
    print(f"\ncache: {len(cache)} vectors on disk")
    sig_emb_by_text = embed_all([sig_text[s] for s in sig_ids], cache,
                                args.batch_signals, "signals")
    cap_emb = embed_all(captions, cache, args.batch_captions, "captions")

    S = np.stack([sig_emb_by_text[sig_text[s]] for s in sig_ids])
    norms_ok = (np.abs(np.linalg.norm(S, axis=1) - 1.0) < 1e-3).all()
    gate(bool(norms_ok), "G7-unit-norm", "all signal vectors unit norm")

    # ---------------- strict retrieval ----------------
    id2idx = {s: i for i, s in enumerate(sig_ids)}

    def strict(sub):
        ranks = []
        for cap, gt, _ in sub:
            sims = cap_emb[cap] @ S.T
            rank = int((np.argsort(-sims) == id2idx[gt]).nonzero()[0][0]) + 1
            ranks.append(rank)
        r = np.array(ranks)
        return {"recall@1": float((r <= 1).mean()),
                "recall@5": float((r <= 5).mean()),
                "recall@10": float((r <= 10).mean()),
                "mrr": float((1.0 / r).mean()),
                "median_rank": float(np.median(r)),
                "n_queries": len(r), "pool_size": len(sig_ids)}

    report = {"model": MODEL, "split": "test",
              "protocol": "strict pair-level retrieval, identical to "
                          "models/clasp/evaluate.py harness B",
              "soft_mAP_skipped": "paper-protocol harness is a CLaSP-paper "
                                  "reproduction tool; no reference value here",
              "serialisation": {"z_normalised": True, "scale": args.scale,
                                "clip": 99, "points_retained": "all"},
              "chance_reference_pool386": CHANCE,
              "strict": {"all": strict(queries)}}
    for grp, name in [(("truce_stock", "truce_synth"), "truce"),
                      (("sushi",), "sushi")]:
        report["strict"][name] = strict([q for q in queries if q[2] in grp])

    print("\n" + "=" * 74)
    print(f"STRICT RETRIEVAL — {MODEL}   "
          f"(random-ranking reference: R@1 {CHANCE['recall@1']:.3f}, "
          f"R@10 {CHANCE['recall@10']:.3f}, MRR {CHANCE['mrr']:.3f})")
    print("=" * 74)
    for k, v in report["strict"].items():
        print(f"  {k:<6} R@1={v['recall@1']:.3f}  R@5={v['recall@5']:.3f}  "
              f"R@10={v['recall@10']:.3f}  MRR={v['mrr']:.3f}  "
              f"median rank={v['median_rank']:.0f}  "
              f"(n={v['n_queries']}, pool={v['pool_size']})")
    print("\n  Reading aid: values near the random-ranking reference mean the")
    print("  model cannot do this task at all — which is the floor claim.")
    print("  Values well ABOVE it would be surprising: investigate, don't celebrate.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()

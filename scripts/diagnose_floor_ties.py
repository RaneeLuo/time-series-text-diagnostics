"""
diagnose_floor_ties.py — why does D2 count 22 tied queries when 24 is
arithmetically forced by identical cached vectors?

Registered expectation for THIS script (2026-08-14, before running):
  (a) within each duplicate-serialisation group, all cached vectors are
      BITWISE identical (same cache key -> same stored vector). If (a)
      fails, the cache itself is the problem — stop and report.
  (b) given (a), the loop-dot similarity matrix ties all 24 queries;
      the matmul similarity matrix ties only 22 — i.e. the undercount
      lives in BLAS column-path float noise (~1 ulp), not in the data.
  If BOTH computations tie all 24, the undercount is elsewhere and this
  script has falsified the hypothesis — report whatever it prints.

Local only, $0, no API. Run from repo root:
    python scripts/diagnose_floor_ties.py
"""
from __future__ import annotations
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
from dataset import load_pairs
from models.openai_embed.run_probe1 import serialize, load_cache, key_of


def main():
    pairs = load_pairs(splits=("test",))
    seen, sig_ids, sig_series = set(), [], []
    for p in pairs:
        if p.sample_id not in seen:
            seen.add(p.sample_id)
            sig_ids.append(p.sample_id)
            sig_series.append(p.series)
    id2idx = {s: i for i, s in enumerate(sig_ids)}
    ser_u = [serialize(x) for x in sig_series]

    by_str = defaultdict(list)
    for s, t in zip(sig_ids, ser_u):
        by_str[t].append(s)
    dup_groups = sorted([v for v in by_str.values() if len(v) > 1],
                        key=lambda g: g[0])
    dup_members = {s for g in dup_groups for s in g}
    print(f"duplicate groups: {dup_groups}")

    cache = load_cache()

    # (a) bitwise identity of cached vectors within each group
    print("\n(a) cached-vector bitwise identity within groups")
    all_identical = True
    for g in dup_groups:
        vecs = [cache[key_of(ser_u[id2idx[s]])] for s in g]
        same_key = len({key_of(ser_u[id2idx[s]]) for s in g}) == 1
        bit = all(np.array_equal(vecs[0], v) for v in vecs[1:])
        all_identical &= bit
        print(f"  {g[0]} group: same cache key={same_key}, "
              f"bitwise identical={bit}")
    if not all_identical:
        print("  -> (a) FAILED: the cache holds different vectors for "
              "identical strings. STOP — this is the problem.")
        return

    # (b) matmul vs per-query loop dot, tie counts on the 24 queries
    captions = sorted({p.caption for p in pairs})
    cap_vec = {c: cache[key_of(c)] for c in captions}
    C = np.stack([cap_vec[p.caption] for p in pairs])
    S = np.stack([cache[key_of(t)] for t in ser_u])
    gt_idx = np.array([id2idx[p.sample_id] for p in pairs])

    sims_mm = C @ S.T
    q24 = [qi for qi, p in enumerate(pairs) if p.sample_id in dup_members]
    print(f"\n(b) the {len(q24)} dup-GT queries, ntied under two "
          f"computations of the SAME data")
    print(f"{'query':>5} {'caption_id':<40} {'mm':>3} {'loop':>4}  "
          f"max|mm-loop| over pool")
    n_mm = n_loop = 0
    worst = 0.0
    for qi in q24:
        s_mm = sims_mm[qi]
        s_lp = np.array([float(np.dot(C[qi], S[j]))
                         for j in range(S.shape[0])])
        t_mm = int((s_mm == s_mm[gt_idx[qi]]).sum()) - 1
        t_lp = int((s_lp == s_lp[gt_idx[qi]]).sum()) - 1
        n_mm += t_mm > 0
        n_loop += t_lp > 0
        d = float(np.max(np.abs(s_mm - s_lp)))
        worst = max(worst, d)
        mark = "" if t_mm == t_lp else "   <-- differs"
        print(f"{qi:>5} {pairs[qi].caption_id:<40} {t_mm:>3} {t_lp:>4}  "
              f"{d:.2e}{mark}")
    print(f"\n  tied-query count over the {len(q24)}: matmul {n_mm}, "
          f"loop {n_loop}; max |mm-loop| anywhere: {worst:.2e}")
    print("  reading: loop==24 and matmul==22 confirms the BLAS-column-"
          "path hypothesis at ~1 ulp; loop==matmul==24 falsifies it; "
          "anything else is a new mechanism — paste the full output back.")


if __name__ == "__main__":
    main()

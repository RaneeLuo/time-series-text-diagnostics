#!/usr/bin/env python3
"""
check_pool_neighbours.py — second-tier G6 mechanism check (2026-08-13).

First tier (verified): exact z-norm duplicates {165,326,360} and {58,86}.
Uncovered hairline signals: 87 and 362 (exact-tie margins without byte
identity). Hypothesis: NEAR-duplicates — z-normed series differing by a
hair, whose similarity scores collide at float32 resolution (the observed
5.96e-08 margins are one float32 ulp at ~0.5).

For every test-pool signal this prints the nearest other pool signal by
max-abs difference of the z-normed series, and lists all pairs closer
than 1e-3. Run from the repository root:
    python scripts/check_pool_neighbours.py
Writes: results/analysis/probe2_pool_neighbours.json
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root
from dataset import load_pairs, znorm

FOCUS = ["truce_synth:pilot13/87.png", "truce_synth:pilot13/362.png"]


def main():
    pairs = load_pairs(splits=("test",))
    seen, ids, Z = set(), [], []
    for p in pairs:
        if p.sample_id not in seen:
            seen.add(p.sample_id)
            ids.append(p.sample_id)
            Z.append(znorm(p.series).astype("float32"))
    print(f"pool signals: {len(ids)} (expected 386)")

    # nearest neighbour within same length class (12 vs 2048)
    by_len = {}
    for i, z in enumerate(Z):
        by_len.setdefault(len(z), []).append(i)
    close_pairs = []
    nn = {}
    for L, idxs in by_len.items():
        M = np.stack([Z[i] for i in idxs])          # (n, L)
        for a_pos, i in enumerate(idxs):
            d = np.abs(M - M[a_pos]).max(axis=1)    # max-abs diff to each
            d[a_pos] = np.inf
            b_pos = int(d.argmin())
            nn[ids[i]] = (ids[idxs[b_pos]], float(d[b_pos]))
            if d[b_pos] < 1e-3 and i < idxs[b_pos]:
                close_pairs.append((ids[i], ids[idxs[b_pos]], float(d[b_pos])))

    print(f"\npairs with max-abs z-diff < 1e-3: {len(close_pairs)}")
    for a, b, d in sorted(close_pairs, key=lambda t: t[2]):
        print(f"  {d:.2e}  {a}  <->  {b}")

    print("\nfocus signals (the uncovered hairline pair):")
    for s in FOCUS:
        p, d = nn[s]
        print(f"  {s}: nearest = {p}  max-abs z-diff = {d:.2e}")

    sushi_min = min(d for s, (p, d) in nn.items() if s.startswith("sushi"))
    print(f"\nSUSHI minimum nearest-neighbour distance: {sushi_min:.4f} "
          f"(expected well-separated, > 0.01)")

    out = Path("results/analysis/probe2_pool_neighbours.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"close_pairs": close_pairs,
                   "focus": {s: nn[s] for s in FOCUS},
                   "sushi_min_nn": sushi_min}, f, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
check_pool_duplicates.py — verify the G6 tie mechanism (2026-08-13).

Hypothesis from probe2_g6_diagnosis.json: some truce_synth test signals are
IDENTICAL after z-normalization, so their embeddings tie exactly and rank
order among ties is decided by torch's sort internals (version-dependent).

This script groups all 386 test-pool signals by the exact bytes of their
z-normalized series and reports every group with more than one member.

Run from the repository root:
    python scripts/check_pool_duplicates.py
Writes: results/analysis/probe2_pool_duplicates.json
"""

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root
from dataset import load_pairs, znorm

HAIRLINE_SIGNALS = {  # gt signals of the 9 hairline queries in the diagnosis
    "truce_synth:pilot13/58.png", "truce_synth:pilot13/87.png",
    "truce_synth:pilot13/360.png", "truce_synth:pilot13/362.png",
}


def main():
    pairs = load_pairs(splits=("test",))
    seen, sig = set(), {}
    for p in pairs:
        if p.sample_id not in seen:
            seen.add(p.sample_id)
            sig[p.sample_id] = p.series
    print(f"pool signals: {len(sig)} (expected 386)")

    groups = defaultdict(list)
    for s, series in sig.items():
        z = znorm(series).astype("float32")
        groups[hashlib.sha256(z.tobytes()).hexdigest()].append(s)

    dup = {h: ids for h, ids in groups.items() if len(ids) > 1}
    n_dup_signals = sum(len(v) for v in dup.values())
    print(f"duplicate groups (z-norm byte-identical): {len(dup)} "
          f"covering {n_dup_signals} signals")
    for ids in sorted(dup.values(), key=lambda v: v[0]):
        subs = {i.split(":")[0] for i in ids}
        print(f"  {ids}   [{'/'.join(sorted(subs))}]")

    covered = HAIRLINE_SIGNALS & {i for v in dup.values() for i in v}
    print(f"\nhairline gt signals inside duplicate groups: "
          f"{len(covered)}/{len(HAIRLINE_SIGNALS)}")
    missing = HAIRLINE_SIGNALS - covered
    if missing:
        print(f"  NOT covered (hypothesis incomplete for these): {sorted(missing)}")

    n_q = sum(1 for p in pairs
              if p.sample_id in {i for v in dup.values() for i in v})
    print(f"queries whose ground truth sits in a duplicate group: {n_q}/878")

    out = Path("results/analysis/probe2_pool_duplicates.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"n_pool": len(sig),
                   "duplicate_groups": sorted(dup.values()),
                   "n_queries_affected": n_q}, f, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

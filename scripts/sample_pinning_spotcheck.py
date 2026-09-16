"""
sample_pinning_spotcheck.py — sequential human spot-check of the C4 "pinning"
pile, the 883 items behind the decisive audit number (CLaSP 0.593).

DESIGN (v2, sequential). The first version pre-registered a single n=20 draw;
before any judging began it was superseded by this expandable design, recorded
as such. The entire eligible pile is put into ONE seeded random order,
interleaved so that every prefix is approximately proportionally stratified
across the six target values. The sheet marks batches of 100. The reader works
top-down and may stop at any batch boundary.

Pre-registered rules:
  * judge in sheet order; do not skip ahead or cherry-pick
  * stop for TIME reasons only — never because interim results look good or
    bad (outcome-dependent stopping is the one thing that biases the count)
  * criterion at each completed batch boundary: cumulative pass rate >= 95%
    (>=95/100, >=190/200, ...); registered expectation: ~1% failures or fewer
  * one question per item, q_pins_falsely: does the replacement clause
    genuinely assert the swap_to value AND is that assertion false of a
    signal whose true fluctuation is swap_from? (y/n; non-y gets a note)
  * cross-slot step-on-sawtooth/square items are excluded from the draw:
    the audit counts them separately (22 items) and the main validation's
    row 40 already adjudicated that mechanism under the plain-language
    convention

The script GATES on reproducing the audit's pinning-pile size (883) with the
identical keyword rule, so the sheet provably samples the population the
0.593 was computed on.

Run from repo root:
    python scripts/sample_pinning_spotcheck.py

Reads:  data/processed/probe1_items.jsonl
        results/analysis/c4_clause_specificity.json   (gate: pile must match)
Writes: results/analysis/pinning_spotcheck_sequential.csv  (utf-8-sig)
"""

from __future__ import annotations
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ITEMS = Path("data/processed/probe1_items.jsonl")
AUDIT = Path("results/analysis/c4_clause_specificity.json")
OUT_CSV = Path("results/analysis/pinning_spotcheck_sequential.csv")

BATCH = 100
SEED = 43
JUMPY = {"sawtooth wave", "reverse sawtooth wave", "square wave"}


def pins(clause: str, val: str):
    """Identical rule to audit_c4_clause_specificity.py; the gate below
    verifies the resulting pile matches the audit's population exactly."""
    c = clause.lower()
    if val == "positive spike":
        return "positive" in c and "spike" in c
    if val == "negative spike":
        return "negative" in c and "spike" in c
    if val == "positive-and-negative spike":
        return "positive" in c and "negative" in c
    if val == "step":
        return "step" in c
    if val == "noisy":
        return "nois" in c
    if val == "smooth":
        return "smooth" in c
    return None


def main():
    items = [json.loads(l) for l in open(ITEMS, encoding="utf-8")]
    swap = [it for it in items
            if it["condition"] == "swap"
            and it["component"] == "C4_fluctuation_type"]
    if len(swap) != 990:
        print(f"GATE FAILED: expected 990 C4 swap items, found {len(swap)}")
        sys.exit(1)

    pile = [it for it in swap if pins(it["clause_replaced_to"], it["swap_to"])]
    print(f"C4 swap items: {len(swap)}   pinning pile: {len(pile)}")

    if not AUDIT.exists():
        print("GATE FAILED: run scripts/audit_c4_clause_specificity.py first")
        sys.exit(1)
    audit = json.load(open(AUDIT, encoding="utf-8"))
    want = audit["clasp_acc_pinning_items"]["n"]
    if len(pile) != want:
        print(f"GATE FAILED: pile {len(pile)} vs audit {want} — "
              f"keyword rules drifted")
        sys.exit(1)
    print(f"gate pass — pile matches the audit's population (n={want})")

    excluded = [it for it in pile if it["swap_to"] == "step"
                and any(s in it["source_class"] for s in JUMPY)]
    ex_ids = {it["item_id"] for it in excluded}
    eligible = [it for it in pile if it["item_id"] not in ex_ids]
    print(f"excluded cross-slot step items: {len(excluded)}   "
          f"eligible: {len(eligible)}")

    # -------- stratified sequential order: shuffle within each target value,
    # -------- then interleave by fractional rank so every prefix is
    # -------- approximately proportionally stratified
    rng = np.random.default_rng(SEED)
    by_val = defaultdict(list)
    for it in eligible:
        by_val[it["swap_to"]].append(it)
    ordered = []
    for v in sorted(by_val):
        pool = by_val[v]
        perm = rng.permutation(len(pool))
        for j, i in enumerate(perm):
            # fractional rank within the value's shuffled list
            ordered.append(((j + 0.5) / len(pool), v, pool[i]["item_id"],
                            pool[i]))
    ordered.sort(key=lambda t: (t[0], t[1], t[2]))
    seq = [t[3] for t in ordered]

    n_batches = int(np.ceil(len(seq) / BATCH))
    print(f"sequence: {len(seq)} items in {n_batches} batches of {BATCH}")
    c1 = Counter(it["swap_to"] for it in seq[:BATCH])
    print(f"batch 1 composition: {dict(sorted(c1.items()))}")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    cols = ["reading_order", "batch", "sample_id", "swap_from", "swap_to",
            "clause_replaced_to", "caption_distractor",
            "q_pins_falsely", "notes"]
    with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for i, it in enumerate(seq, 1):
            w.writerow([i, (i - 1) // BATCH + 1, it["sample_id"],
                        it["swap_from"], it["swap_to"],
                        it["clause_replaced_to"], it["caption_distractor"],
                        "", ""])
    print(f"sheet -> {OUT_CSV}")

    print("\nPROTOCOL")
    print("Read top-down in reading_order. Judge q_pins_falsely (y/n) per row.")
    print("Stop only at a batch boundary, and only for time — never because")
    print("of what you are seeing. Criterion at each completed boundary:")
    print("cumulative pass rate >= 95%. Send back the sheet with the batches")
    print("you completed; unread rows stay blank.")


if __name__ == "__main__":
    main()

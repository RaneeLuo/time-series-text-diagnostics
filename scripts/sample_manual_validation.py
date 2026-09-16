"""
sample_manual_validation.py — draw the ~50-item sample for human validation of
Probe-1 swap items (open item 2 of docs/probe1_findings_clasp.md §7).

Two layers, one automated and one human:

AUTOMATED (runs on ALL 2,770 swap items, aborts on failure):
  A1  every distractor is EXACTLY caption_correct with the recorded clause
      substituted (caption_distractor == caption_correct.replace(
      clause_replaced_from, clause_replaced_to))
  A2  clause_replaced_from is present in caption_correct
  A3  no identity replacements (from == to)
These verify the swap's structural integrity mechanically, so the human sample
only needs to judge the REPLACEMENT CLAUSES, not reconstruct the edits.

HUMAN (the sampled sheet, 10 items per component, seeded):
  For each item, judge the distractor caption on three questions:
    q1_grammatical        does the caption read as well-formed English?
    q2_asserts_swap_to    does it genuinely claim the swapped value
                          (column swap_to), not something vague?
    q3_not_still_true     is it actually WRONG for this signal (i.e. it does
                          not accidentally remain a fair description)?
  Fill y/n (or n/a) into the three empty columns; anything but 'y' deserves a
  note. The pass criterion, fixed in advance: >= 47 of 50 items pass all three
  questions; any systematic failure pattern (same component, same clause pool)
  is investigated regardless of the count.

Sampling: 10 per component x 5 components, drawn without replacement with
numpy default_rng(42) from the component's swap items in file order. Re-running
this script reproduces the identical sample.

Run from repo root:
    python scripts/sample_manual_validation.py

Reads:  data/processed/probe1_items.jsonl
Writes: results/analysis/manual_validation_sample.csv   (utf-8-sig for Excel)
        results/analysis/manual_validation_gate.json
"""

from __future__ import annotations
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ITEMS = Path("data/processed/probe1_items.jsonl")
OUT_CSV = Path("results/analysis/manual_validation_sample.csv")
OUT_GATE = Path("results/analysis/manual_validation_gate.json")

PER_COMPONENT = 10
SEED = 42


def main():
    items = [json.loads(l) for l in open(ITEMS, encoding="utf-8")]
    swap = [it for it in items if it["condition"] == "swap"]
    print(f"items: {len(items)}   swap: {len(swap)}")
    if len(swap) != 2770:
        print(f"GATE FAILED: expected 2770 swap items, found {len(swap)}")
        sys.exit(1)

    # ---------------------------------------------------------- automated
    fails = {"A1_reconstruction": [], "A2_clause_present": [], "A3_identity": []}
    for it in swap:
        if it["caption_correct"].replace(
                it["clause_replaced_from"], it["clause_replaced_to"]) \
                != it["caption_distractor"]:
            fails["A1_reconstruction"].append(it["item_id"])
        if it["clause_replaced_from"] not in it["caption_correct"]:
            fails["A2_clause_present"].append(it["item_id"])
        if it["clause_replaced_from"] == it["clause_replaced_to"]:
            fails["A3_identity"].append(it["item_id"])
    for gate, bad in fails.items():
        print(f"{gate}: {len(swap) - len(bad)}/{len(swap)} pass")
    if any(fails.values()):
        print("GATE FAILED — first offenders:",
              {g: b[:3] for g, b in fails.items() if b})
        sys.exit(1)

    # ------------------------------------------------------------ sample
    rng = np.random.default_rng(SEED)
    comps = sorted({it["component"] for it in swap})
    sample = []
    for c in comps:
        pool = [it for it in swap if it["component"] == c]
        take = rng.choice(len(pool), size=PER_COMPONENT, replace=False)
        sample.extend(pool[i] for i in sorted(take))
    print(f"sampled {len(sample)} items "
          f"({PER_COMPONENT} per component, seed {SEED})")
    print("pair coverage:",
          dict(Counter(it["component"].split('_')[0] for it in sample)))

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    cols = ["idx", "component", "sample_id", "swap_from", "swap_to",
            "clause_replaced_from", "clause_replaced_to",
            "caption_correct", "caption_distractor",
            "q1_grammatical", "q2_asserts_swap_to", "q3_not_still_true",
            "notes"]
    with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for i, it in enumerate(sample, 1):
            w.writerow([i, it["component"], it["sample_id"],
                        it["swap_from"], it["swap_to"],
                        it["clause_replaced_from"], it["clause_replaced_to"],
                        it["caption_correct"], it["caption_distractor"],
                        "", "", "", ""])
    print(f"sheet -> {OUT_CSV}")

    OUT_GATE.write_text(json.dumps({
        "n_swap_items_checked": len(swap),
        "automated_gates": {g: "pass" for g in fails},
        "note": "distractor == caption_correct with one recorded clause "
                "substituted, for every swap item; clause always present; "
                "no identity swaps",
        "sample_seed": SEED, "per_component": PER_COMPONENT,
        "pass_criterion": ">=47/50 pass q1&q2&q3; any systematic pattern "
                          "investigated regardless",
        "sampled_item_ids": [it["item_id"] for it in sample],
    }, indent=2), encoding="utf-8")
    print(f"gate record -> {OUT_GATE}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
make_n3_validation_sheet.py — pre-registered sample for the N3 human
validation (the pre-committed hardening step: ~100-item human sample
with CI before any load-bearing narrative number becomes a thesis claim;
N3 is load-bearing: largest gap, +0.27, smallest margins).

DESIGN (registered 2026-08-08, before sampling):
  * Population: the 389 certified N3 swap items.
  * Stratified proportional by duration_class — mandatory because the
    slice analysis found a monotone duration gradient (week 0.648,
    28_days 0.742, six_months 0.841): 67 x 28_days, 27 x week,
    6 x six_months = 100.
  * RNG seed 20260808, recorded here and in the sheet header.
  * Reading order: shuffled across strata (same seed), so batches are
    not blocks of one duration class.
  * Batches of 50; criterion (pre-registered, mirrors the pinned C4
    procedure): cumulative pass rate >= 0.95 at each batch boundary.
  * Predictions: P-val1 pass >= 0.95; P-val2 failures concentrate in
    'week' items.

Judging rules are emitted alongside as n3_judging_rules.md — adapted
from docs/pinning_spotcheck_judging_rules.md (two-part y/n test) plus
the v2 N3 exclusions (generator header, flags I25/I27/I29/I30).
CONFIRM the rules read correctly before judging row 1.

Run from thesis repo root:
  python scripts/make_n3_validation_sheet.py
Writes:
  results/analysis/n3_validation_sheet.csv
  results/analysis/n3_judging_rules.md
"""

import csv
import json
import random
from collections import Counter
from pathlib import Path

ITEMS = Path("data/processed/narrative_probe_items_certified.jsonl")
SHEET = Path("results/analysis/n3_validation_sheet.csv")
RULES = Path("results/analysis/n3_judging_rules.md")

SEED = 20260808
QUOTA = {"28_days": 67, "week": 27, "six_months": 6}

RULES_TEXT = """# Judging rules — N3 human validation (draft: confirm before row 1)

Adapted from the pinned C4 rules (two-part test, same mechanics) for the
narrative N3 swap: trend-direction antonym surgery in the temperature
field. Judge the SWAPPED caption from the text alone; no plots needed.
The round-1 defect class was internal consistency, and that is what
this sample certifies.

For each row, write **y** only if BOTH halves hold:

- **(a) It genuinely asserts the opposite.** The swapped clause
  (`clause_replaced_to`) reads as a grammatical, confident claim of the
  swapped trend direction — not garbled by the surgery, not hedged so
  softly it stops being a claim.
- **(b) The caption stays internally consistent with the new direction.**
  No surviving phrase elsewhere in the caption anchors the OLD direction
  — e.g. a month/season reference, a peak/pinned value, or a
  rising/falling verb outside the swapped span that now contradicts it.
  (The v2 exclusions I25/I27/I29/I30 were built to prevent exactly this;
  this sample checks they worked.)

Write **n** if either half fails, and note which ("a: ..." or "b: ...").
Plain-language standard: judge what the words say to a competent reader.

Mechanics (same as pinned): work strictly top-down in reading_order;
stop only at a batch boundary (rows 50, 100), never because of results
so far. Blank = unread; every n gets a note; hesitation > ~20 s -> make
the strict call (n) and write the hesitation into notes. Criterion
(pre-registered): cumulative pass >= 95% at each batch boundary.
Several n's sharing one mechanism -> stop at the boundary and report.
"""


def main():
    items = [json.loads(l) for l in ITEMS.open(encoding="utf-8")]
    pool = [it for it in items
            if it["component"] == "N3" and it["condition"] == "swap"]
    by_class = Counter(it["duration_class"] for it in pool)
    print(f"N3 swap population: {len(pool)}   by class: {dict(by_class)}")
    if len(pool) != 389:
        print("[STOP] expected 389 N3 swap items")
        return
    if sum(QUOTA.values()) != 100 or set(QUOTA) != set(by_class):
        print("[STOP] quota/classes mismatch")
        return

    rng = random.Random(SEED)
    sample = []
    for cl, q in sorted(QUOTA.items()):
        stratum = sorted((it for it in pool if it["duration_class"] == cl),
                         key=lambda x: x["item_id"])
        sample.extend(rng.sample(stratum, q))
    rng.shuffle(sample)
    assert len({it["item_id"] for it in sample}) == 100

    SHEET.parent.mkdir(parents=True, exist_ok=True)
    with SHEET.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow([f"# N3 validation sample | seed {SEED} | "
                    f"strata {QUOTA} | batches of 50 | "
                    f"criterion: cumulative pass >= 0.95 at each boundary"])
        w.writerow(["reading_order", "item_id", "duration_class",
                    "swap_from", "swap_to",
                    "clause_replaced_from", "clause_replaced_to",
                    "swapped_caption_to_judge", "y_or_n", "notes"])
        for i, it in enumerate(sample, 1):
            w.writerow([i, it["item_id"], it["duration_class"],
                        it["swap_from"], it["swap_to"],
                        it["clause_replaced_from"], it["clause_replaced_to"],
                        it["caption_distractor"], "", ""])
    RULES.write_text(RULES_TEXT, encoding="utf-8")

    got = Counter(it["duration_class"] for it in sample)
    print(f"sampled: {len(sample)}   by class: {dict(got)}   seed: {SEED}")
    print(f"sheet -> {SHEET}")
    print(f"rules -> {RULES}")
    print("\nBefore judging row 1: read the rules file and confirm the two-")
    print("part test matches your round-1/2 procedure. Judge offline at your")
    print("own pace; the verdict is next session's work.")


if __name__ == "__main__":
    main()

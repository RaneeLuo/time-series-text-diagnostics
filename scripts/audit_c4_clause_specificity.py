"""
audit_c4_clause_specificity.py — the full-population investigation triggered by
the manual validation (docs/probe1_findings_clasp.md §7 item 2).

The 50-item human validation found four failing items; three share a mechanism
concentrated in C4: the replacement clause is semantically underspecified
(polarity-free "large spikes", step-free "frequent large changes") or arguably
true of the signal. Pre-registration mandates investigating any systematic
pattern, so this script escalates the sampled observation to all 990 C4 swap
items and asks the decisive question: is CLaSP's low C4 accuracy driven by the
weak items?

Three measurements:
  1. LEXICAL SPECIFICITY — does each replacement clause explicitly name its
     target fluctuation? (Approximate by design: keyword-based, so a subtle
     but specific phrasing counts as generic. Stated as such.)
  2. THE DECISIVE JOIN — CLaSP's per-item swap accuracy on lexically-pinning
     items vs generic items. If item weakness caused the C4 result, accuracy
     should be HIGHER on the pinning items.
  3. CROSS-SLOT LEAKAGE — items targeting 'step' on sawtooth/square shapes,
     whose own resets could be read as steps (manual row 40's mechanism).

Also records the corpus-vocabulary check behind the row-40 adjudication:
whether 'step' vocabulary ever appears in captions of non-step sawtooth/square
classes (class-exclusivity of fluctuation vocabulary).

Run from repo root:
    python scripts/audit_c4_clause_specificity.py

Reads:  data/processed/probe1_items.jsonl
        results/experiments/probe1_clasp_per_item.jsonl
Writes: results/analysis/c4_clause_specificity.json
"""

from __future__ import annotations
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ITEMS = Path("data/processed/probe1_items.jsonl")
PER_ITEM = Path("results/experiments/probe1_clasp_per_item.jsonl")
OUT = Path("results/analysis/c4_clause_specificity.json")


def pins(clause: str, val: str):
    """Does the clause lexically name the target fluctuation value?"""
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
    print(f"C4 swap items: {len(swap)}")
    if len(swap) != 990:
        print(f"GATE FAILED: expected 990 C4 swap items, found {len(swap)}")
        sys.exit(1)

    for it in swap:
        it["specific"] = pins(it["clause_replaced_to"], it["swap_to"])
        if it["specific"] is None:
            print(f"GATE FAILED: unknown swap_to {it['swap_to']!r}")
            sys.exit(1)

    n_spec = sum(it["specific"] for it in swap)
    print(f"replacement clause lexically pins swap_to: "
          f"{n_spec}/{len(swap)} ({n_spec/len(swap):.1%})")
    by_val = defaultdict(list)
    for it in swap:
        by_val[it["swap_to"]].append(it["specific"])
    print("\nby target value:")
    per_value = {}
    for v, l in sorted(by_val.items()):
        per_value[v] = {"share_pinning": float(np.mean(l)), "n": len(l)}
        print(f"  {v:<30} {np.mean(l):.2f}  (n={len(l)})")

    # -------------------------------------------------- the decisive join
    recs = [json.loads(l) for l in open(PER_ITEM, encoding="utf-8")]
    acc_by_item = defaultdict(list)
    for r in recs:
        if r["component"] == "C4_fluctuation_type" and r["condition"] == "swap":
            acc_by_item[r["item_id"]].append(r["correct"])
    missing = [it["item_id"] for it in swap if it["item_id"] not in acc_by_item]
    if missing:
        print(f"GATE FAILED: {len(missing)} items lack per-item results, "
              f"e.g. {missing[:2]}")
        sys.exit(1)

    spec_acc = [float(np.mean(acc_by_item[it["item_id"]]))
                for it in swap if it["specific"]]
    gen_acc = [float(np.mean(acc_by_item[it["item_id"]]))
               for it in swap if not it["specific"]]
    print(f"\nCLaSP swap accuracy, clause PINS target : "
          f"{np.mean(spec_acc):.3f} (n={len(spec_acc)})")
    print(f"CLaSP swap accuracy, GENERIC clause     : "
          f"{np.mean(gen_acc):.3f} (n={len(gen_acc)})")

    # ------------------------------------------------- cross-slot leakage
    jumpy = {"sawtooth wave", "reverse sawtooth wave", "square wave"}
    step_items = [it for it in swap if it["swap_to"] == "step"]
    leak = [it for it in step_items
            if any(s in it["source_class"] for s in jumpy)]
    leak_ids = {it["item_id"] for it in leak}
    la = [float(np.mean(acc_by_item[i["item_id"]])) for i in leak]
    other = [float(np.mean(acc_by_item[i["item_id"]])) for i in step_items
             if i["item_id"] not in leak_ids]
    print(f"\nstep-target items on sawtooth/square shapes (leakage "
          f"candidates): {len(leak)} of {len(step_items)}")
    print(f"  CLaSP acc on candidates {np.mean(la):.3f}   "
          f"on other step-target items {np.mean(other):.3f}")

    # -------------------------------------- corpus vocabulary exclusivity
    cap_by_class = defaultdict(set)
    for it in items:
        cap_by_class[it["source_class"]].add(it["caption_correct"])
        cap_by_class[it["distractor_class"]].add(it["caption_distractor"])
    n_caps = n_hits = n_step = n_step_hit = 0
    for cls, caps in cap_by_class.items():
        fl, sh = [p.strip() for p in cls.split(";")]
        if sh in jumpy and fl in ("smooth", "clean", "noisy"):
            for c in caps:
                n_caps += 1
                n_hits += bool(re.search(r"\bstep", c, re.I))
        if fl == "step":
            for c in caps:
                n_step += 1
                n_step_hit += bool(re.search(r"\bstep", c, re.I))
    print(f"\ncorpus vocabulary: 'step' in non-step sawtooth/square captions "
          f"{n_hits}/{n_caps}; in step-class captions "
          f"{n_step_hit}/{n_step} ({n_step_hit/max(n_step,1):.0%})")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "n_c4_swap_items": len(swap),
        "method_note": "keyword-based specificity; subtle-but-specific "
                       "phrasings count as generic (conservative)",
        "share_pinning": n_spec / len(swap),
        "per_target_value": per_value,
        "clasp_acc_pinning_items": {"acc": float(np.mean(spec_acc)),
                                    "n": len(spec_acc)},
        "clasp_acc_generic_items": {"acc": float(np.mean(gen_acc)),
                                    "n": len(gen_acc)},
        "leakage_step_on_jumpy_shapes": {
            "n": len(leak), "of_step_targets": len(step_items),
            "acc_leakage": float(np.mean(la)),
            "acc_other_step_targets": float(np.mean(other))},
        "corpus_vocab_exclusivity": {
            "step_in_nonstep_jumpy_captions": [n_hits, n_caps],
            "step_in_step_captions": [n_step_hit, n_step]},
    }, indent=2), encoding="utf-8")
    print(f"saved -> {OUT}")

    print("\nHOW TO READ THIS")
    print("If the weak items caused the low C4 score, accuracy on the PINNING")
    print("items would be clearly higher than on the generic ones. Comparable")
    print("or lower accuracy on pinning items means the blind spot is the")
    print("model's, not the items': it fails even when the distractor names")
    print("the wrong fluctuation explicitly and falsely.")


if __name__ == "__main__":
    main()

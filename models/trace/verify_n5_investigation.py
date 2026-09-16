#!/usr/bin/env python3
"""
verify_n5_investigation.py — reproduce the N5 anomaly investigation locally.

Context (2026-08-08): the narrative probe showed N5 (the designed
negative control) at 0.92 swap accuracy instead of ~chance. Claude ran
this investigation once on its side under one-time authorisation; this
script reproduces it exactly so the local run is the committed record.
Deterministic throughout — every number should match the expected
values printed alongside (from Claude's run) to the last digit.

Three analyses:
  A. Replacement-record fidelity: caption diff == recorded from->to.
  B. What changes in an N5 swap: non-place vocabulary and sentence
     frame (place names stripped).
  C. The decisive slice: accuracy on place-name-ONLY swaps (frame
     identical) vs frame-changed swaps, per mask seed.
     Registered prediction (Claude, before computing): frame-identical
     <= 0.75. MISSED: observed 0.900 in all three seeds.

Run from thesis repo root:
  python models/trace/verify_n5_investigation.py
"""

import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

ITEMS = Path("data/processed/narrative_probe_items_certified.jsonl")
RESULTS = Path("results/experiments/trace_narrative_per_item.jsonl")

WEATHER = re.compile(
    r"\b(weather|rain|snow|wind|humid|temperature|storm|cloud|clear|dry|wet|"
    r"hot|cold|warm|cool|sunn?y|conditions?|climate|precipitation|experienced|"
    r"variety|diverse|range|reported|data|recorded)\b", re.I)


def frame(s: str) -> str:
    t = re.sub(r"[A-Z][A-Za-z.]*(?: County)?(?:,? [A-Z][A-Za-z.]*)*",
               "<PLACE>", s)
    return re.sub(r"\s+", " ", t).strip()


def content_words(s: str):
    return set(w.lower() for w in WEATHER.findall(s))


def main():
    items = [json.loads(l) for l in ITEMS.open(encoding="utf-8")]
    n5 = {it["item_id"]: it for it in items
          if it["component"] == "N5" and it["condition"] == "swap"}
    print(f"N5 swap items: {len(n5)}   (expected: 400)")

    # ---- A. replacement-record fidelity ---------------------------------
    mismatch = 0
    for it in n5.values():
        a, b = it["caption_correct"], it["caption_distractor"]
        fr, to = it["clause_replaced_from"], it["clause_replaced_to"]
        if not (fr in a and to in b and a.replace(fr, to, 1) == b):
            mismatch += 1
    print(f"[A] caption diff != recorded replacement: {mismatch}/400   "
          f"(expected: 0/400)")

    # ---- B. what changes ------------------------------------------------
    vocab_changed = sum(
        1 for it in n5.values()
        if content_words(it["clause_replaced_from"])
        != content_words(it["clause_replaced_to"]))
    same_frame_n = 0
    for it in n5.values():
        it["_same_frame"] = (frame(it["clause_replaced_from"])
                             == frame(it["clause_replaced_to"]))
        same_frame_n += it["_same_frame"]
    print(f"[B] swaps changing non-place vocabulary: {vocab_changed}/400   "
          f"(expected: 336/400)")
    print(f"[B] swaps with IDENTICAL frame (place-name-only): "
          f"{same_frame_n}/400   (expected: 40/400)")

    # ---- C. decisive slice ----------------------------------------------
    recs = [json.loads(l) for l in RESULTS.open(encoding="utf-8")]
    recs = [r for r in recs if r["item_id"] in n5]
    print(f"[C] joined N5 swap records: {len(recs)}   (expected: 1200)")

    expected = {
        True:  ("frame-IDENTICAL (place-name-only)", 0.900,
                {"mask13": 0.900, "mask14": 0.900, "mask15": 0.900}, 0.2004),
        False: ("frame-CHANGED", 0.929,
                {"mask13": 0.928, "mask14": 0.939, "mask15": 0.919}, 0.2506),
    }
    for keep, (label, e_pool, e_seed, e_margin) in expected.items():
        sel = [r for r in recs if n5[r["item_id"]]["_same_frame"] == keep]
        by_seed = defaultdict(list)
        for r in sel:
            by_seed[r["seed"]].append(r["correct"])
        accs = {s: sum(v) / len(v) for s, v in sorted(by_seed.items())}
        pool = sum(r["correct"] for r in sel) / len(sel)
        marg = statistics.mean(r["margin"] for r in sel)
        print(f"\n[C] {label}: {len(sel)//3} items")
        print(f"    acc by seed: { {s: round(a,3) for s,a in accs.items()} }")
        print(f"      expected : {e_seed}")
        print(f"    pooled acc {pool:.3f} (expected {e_pool:.3f}) | "
              f"mean margin {marg:+.4f} (expected {e_margin:+.4f})")

    ids = sorted(i for i, it in n5.items() if it["_same_frame"])
    print(f"\n[record] the {len(ids)} frame-identical item_ids:")
    print("  " + ", ".join(ids))
    print("\nIf every number matches its expected value, the investigation "
          "is verified and the local run is the committed record.")


if __name__ == "__main__":
    main()

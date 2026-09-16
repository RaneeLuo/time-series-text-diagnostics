"""
analyze_sushi_labels.py — derive the Probe-1 component grammar from SUSHI labels.

Probe 1 swaps ONE semantic component of a caption while leaving the rest intact.
On SUSHI this is tractable because the class labels are already compositional
(e.g. "negative spike; constant"). Before any swap generator can be written we
need three things established from the data rather than assumed:

  A. LABEL STRUCTURE  — how many slots does a label have, what are the legal
                        values of each, and do they form a clean product?
  B. CAPTION VARIETY  — how many distinct captions exist per class (SUSHI draws
                        caption text from a pre-registered list, so a class may
                        have several phrasings).
  C. PHRASE MAPPING   — which sentence in a caption expresses which slot value.
                        This is what makes a single-component swap possible:
                        to swap the trend, we must know which clause states it.

Run from repo root:
    python scripts/analyze_sushi_labels.py

Writes: results/analysis/sushi_grammar.json  (full detail)
Prints: a compact report to paste into the design discussion.
"""

from __future__ import annotations
import json
import re
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path

PAIRS = Path("data/processed/pairs.jsonl")
OUT = Path("results/analysis/sushi_grammar.json")


def load_sushi():
    rows = []
    with open(PAIRS, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["dataset"] == "sushi":
                rows.append(r)
    return rows


def split_sentences(text: str) -> list[str]:
    """Conservative sentence split; SUSHI captions are 1-3 plain sentences."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def main():
    rows = load_sushi()
    print(f"SUSHI records: {len(rows)}")

    # ---------------------------------------------------------------- A
    print("\n" + "=" * 70)
    print("A. LABEL STRUCTURE")
    print("=" * 70)

    labels = sorted({r["class_label"] for r in rows})
    print(f"unique class labels: {len(labels)}")

    seps = Counter(lab.count(";") for lab in labels)
    print(f"';' count per label: {dict(seps)}")

    slots = [[p.strip() for p in lab.split(";")] for lab in labels]
    widths = Counter(len(s) for s in slots)
    print(f"slot count per label: {dict(widths)}")

    n_slots = max(widths, key=widths.get)
    slot_values = [sorted({s[i] for s in slots if len(s) > i}) for i in range(n_slots)]
    for i, vals in enumerate(slot_values):
        print(f"\n  slot {i}: {len(vals)} distinct values")
        for v in vals:
            print(f"    - {v}")

    prod = 1
    for vals in slot_values:
        prod *= len(vals)
    print(f"\n  product of slot sizes = {prod}   (labels observed = {len(labels)})")
    print("  -> " + ("COMPLETE product: every combination exists"
                     if prod == len(labels) else
                     f"INCOMPLETE: {prod - len(labels)} combinations absent"))

    if prod != len(labels):
        seen = {tuple(s) for s in slots}
        missing = [c for c in product(*slot_values) if c not in seen]
        print(f"  missing combinations (first 15 of {len(missing)}):")
        for c in missing[:15]:
            print(f"    {' ; '.join(c)}")

    # ---------------------------------------------------------------- B
    print("\n" + "=" * 70)
    print("B. CAPTION VARIETY PER CLASS")
    print("=" * 70)

    caps_by_class = defaultdict(list)
    for r in rows:
        caps_by_class[r["class_label"]].append(r["caption"])

    distinct = Counter(len(set(v)) for v in caps_by_class.values())
    print(f"distinct captions per class: {dict(sorted(distinct.items()))}")
    samples = Counter(len(v) for v in caps_by_class.values())
    print(f"samples per class: {dict(sorted(samples.items()))}")

    ex_class = sorted(caps_by_class, key=lambda k: -len(set(caps_by_class[k])))[0]
    print(f"\nexample — class with most caption variety: '{ex_class}'")
    for c in sorted(set(caps_by_class[ex_class]))[:4]:
        print(f"    * {c}")

    sent_counts = Counter(len(split_sentences(r["caption"])) for r in rows)
    print(f"\nsentences per caption: {dict(sorted(sent_counts.items()))}")

    # ---------------------------------------------------------------- C
    print("\n" + "=" * 70)
    print("C. PHRASE -> SLOT-VALUE MAPPING")
    print("=" * 70)
    print("For each sentence, which slot values do the classes using it have?")
    print("A sentence tied to exactly one value of one slot is a swappable unit.\n")

    # sentence -> per-slot set of values of the classes it appears in
    sent_slotvals = defaultdict(lambda: [set() for _ in range(n_slots)])
    sent_freq = Counter()
    for r in rows:
        parts = [p.strip() for p in r["class_label"].split(";")]
        for s in split_sentences(r["caption"]):
            sent_freq[s] += 1
            for i in range(min(n_slots, len(parts))):
                sent_slotvals[s][i].add(parts[i])

    exclusive = defaultdict(list)   # (slot_idx, value) -> sentences
    ambiguous = []
    for s, sv in sent_slotvals.items():
        tied = [(i, next(iter(vals))) for i, vals in enumerate(sv) if len(vals) == 1]
        if len(tied) == 1:
            exclusive[tied[0]].append(s)
        elif len(tied) == 0:
            ambiguous.append(s)

    print(f"distinct sentences: {len(sent_slotvals)}")
    print(f"sentences tied to exactly one slot value: "
          f"{sum(len(v) for v in exclusive.values())}")
    print(f"sentences tied to no single slot value: {len(ambiguous)}")

    for i in range(n_slots):
        keys = sorted([k for k in exclusive if k[0] == i], key=lambda k: k[1])
        if not keys:
            continue
        print(f"\n  --- slot {i} ---")
        for (_, val) in keys:
            sents = exclusive[(i, val)]
            print(f"  [{val}]  {len(sents)} phrasings, e.g.:")
            for s in sorted(sents, key=lambda x: -sent_freq[x])[:2]:
                print(f"       \"{s}\"")

    if ambiguous:
        print(f"\n  --- not slot-specific ({len(ambiguous)}), e.g.: ---")
        for s in sorted(ambiguous, key=lambda x: -sent_freq[x])[:3]:
            print(f'       "{s}"')

    # ---------------------------------------------------------------- save
    report = {
        "n_records": len(rows),
        "n_classes": len(labels),
        "n_slots": n_slots,
        "slot_values": slot_values,
        "product_complete": prod == len(labels),
        "labels": labels,
        "captions_per_class": {k: sorted(set(v)) for k, v in caps_by_class.items()},
        "sentence_to_slot_value": {
            f"slot{i}::{val}": sents for (i, val), sents in exclusive.items()
        },
        "sentences_not_slot_specific": ambiguous,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nfull detail -> {OUT}")

    # ---------------------------------------------------------------- verdict
    print("\n" + "=" * 70)
    print("WHAT THIS DETERMINES FOR PROBE 1")
    print("=" * 70)
    coverage = sum(len(v) for v in exclusive.values()) / max(1, len(sent_slotvals))
    print(f"- swappable components available: {n_slots} slots")
    print(f"- sentence-level slot attribution coverage: {coverage:.0%}")
    print("- if coverage is high, swaps can be done by SENTENCE SUBSTITUTION")
    print("  (replace the clause expressing slot i with the clause for another")
    print("  value of slot i), which keeps vocabulary and structure controlled.")
    print("- if coverage is low, swaps need phrase-level editing instead.")


if __name__ == "__main__":
    main()

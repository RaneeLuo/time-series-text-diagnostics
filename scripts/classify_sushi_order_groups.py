"""
classify_sushi_order_groups.py — Probe-2 caption grouping for SUSHI (Q3, rule-based).

Rule (BINDING, accepted 2026-08-09): truth-conditional grouping.
  A caption is ORDER-DEPENDENT if shuffling the signal's time steps makes ANY
  of its claims false. ORDER-INVARIANT means NO claim anywhere in the caption
  is order-sensitive. One special case: 'clean; constant' is a permutation
  FIXED POINT (shuffling an exactly-flat series returns the identical series),
  so it is invariant for a degenerate reason — the perturbation is a no-op.
  It is grouped separately as DEGENERATE and serves as an identity control,
  never as part of the invariant group in any DiD.

The verdict table below is hand-written per label atom, derived 2026-08-09
from the committed grammar artifact (results/analysis/sushi_grammar.json,
slot vocabularies) plus the actual caption texts of all seven
constant-shape cells. It is stated here as an assumption and GATED: if the
data disagrees with the expected split, this script stops loudly.

COMPUTED EXPECTATION (registered before this script was first run; derived
by hand from the committed caption texts, so a mismatch means either the
hand analysis or the data is not what we think):
  classes:     140 (7 fluctuation x 20 shape, complete product)
  DEPENDENT:   135 classes  (19 non-constant shapes x 7  = 133, plus
                             'smooth; constant' and 'step; constant')
  INVARIANT:     4 classes  ({noisy, negative spike, positive spike,
                              positive-and-negative spike} x constant)
  DEGENERATE:    1 class    ('clean; constant')
P2-6 as registered earlier (invariant <= 14 of 140) is assessed by this run.

Run from repo root:
    python scripts/classify_sushi_order_groups.py
Writes: results/analysis/probe2_sushi_groups.json
First line of output should read:  SUSHI records: 1400
"""

from __future__ import annotations
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

PAIRS = Path("data/processed/pairs.jsonl")
GRAMMAR = Path("results/analysis/sushi_grammar.json")
OUT = Path("results/analysis/probe2_sushi_groups.json")

# ------------------------------------------------------------------ verdicts
# Slot 1 (shape): every shape family except 'constant' describes a temporal
# arrangement; shuffling destroys it -> DEPENDENT.
SHAPE_INVARIANT = {"constant"}

# Slot 0 (fluctuation): verdict of the fluctuation CLAUSE alone.
#   invariant  — the claim survives any permutation of the values
#   dependent  — the claim is about point-to-point arrangement
#   none       — 'clean' captions carry no fluctuation clause (no claim)
FLUCT_VERDICT = {
    "clean": "none",
    "noisy": "invariant",               # pervasive scatter survives permutation
    "negative spike": "invariant",      # extreme values still present, still isolated
    "positive spike": "invariant",
    "positive-and-negative spike": "invariant",
    "smooth": "dependent",              # point-to-point smoothness is destroyed
    "step": "dependent",                # a step is a transition located in time
}

EXPECTED = {"records": 1400, "classes": 140,
            "dependent": 135, "invariant": 4, "degenerate": 1}


def fail(gate: str, msg: str):
    print(f"\n[GATE {gate}] FAILED: {msg}", file=sys.stderr)
    sys.exit(1)


def class_verdict(fluct: str, shape: str) -> str:
    if fluct not in FLUCT_VERDICT:
        fail("G3-vocab", f"unknown fluctuation atom {fluct!r} — verdict table incomplete")
    if fluct == "clean" and shape == "constant":
        return "degenerate"
    shape_dep = shape not in SHAPE_INVARIANT
    fluct_dep = FLUCT_VERDICT[fluct] == "dependent"
    return "dependent" if (shape_dep or fluct_dep) else "invariant"


def main():
    if not PAIRS.is_file():
        fail("G0", f"{PAIRS} missing — run `python dataset.py build` first")

    rows = []
    with open(PAIRS, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["dataset"] == "sushi":
                rows.append(r)
    print(f"SUSHI records: {len(rows)}   (expected {EXPECTED['records']})")
    if len(rows) != EXPECTED["records"]:
        fail("G1-count", f"{len(rows)} sushi rows, expected {EXPECTED['records']}")

    # mojibake canary (error #10 lesson)
    bad = [r for r in rows if "\ufffd" in r["caption"] or "掳" in r["caption"]]
    if bad:
        fail("G1-encoding", f"{len(bad)} captions contain mojibake, first: "
                            f"{bad[0]['caption'][:80]!r}")

    labels = sorted({r["class_label"] for r in rows})
    print(f"unique class labels: {len(labels)}   (expected {EXPECTED['classes']})")
    if len(labels) != EXPECTED["classes"]:
        fail("G2-classes", f"{len(labels)} classes, expected {EXPECTED['classes']}")

    # cross-check against the committed grammar artifact if present
    if GRAMMAR.is_file():
        g = json.load(open(GRAMMAR, encoding="utf-8"))
        if sorted(g["labels"]) != labels:
            fail("G2-reconcile", "label set differs from committed sushi_grammar.json")
        print("[G2] label set reconciles with committed sushi_grammar.json")
    else:
        print("[G2] NOTE: sushi_grammar.json not found — skipping reconciliation "
              "(this is a warning, not a stop; the artifact is committed and should exist)")

    # ------------------------------------------------------------ classify
    verdicts = {}
    for lab in labels:
        parts = [p.strip() for p in lab.split(";")]
        if len(parts) != 2:
            fail("G3-shape", f"label {lab!r} does not split into 2 slots")
        verdicts[lab] = class_verdict(parts[0], parts[1])

    counts = Counter(verdicts.values())
    print("\nclass-level verdict counts:")
    for k in ("dependent", "invariant", "degenerate"):
        print(f"  {k:10s} {counts.get(k, 0):4d}   (expected {EXPECTED[k]})")
    total = sum(counts.values())
    if total != EXPECTED["classes"]:
        fail("G4-closure", f"verdicts sum to {total}, not {EXPECTED['classes']}")
    for k in ("dependent", "invariant", "degenerate"):
        if counts.get(k, 0) != EXPECTED[k]:
            fail("G4-split",
                 f"{k} = {counts.get(k, 0)}, expected {EXPECTED[k]}. Either the "
                 f"hand-written verdict table or the label set is not what we "
                 f"believed. Do NOT edit EXPECTED to make this pass; investigate.")

    # per-split signal counts, if split info is present
    split_counts = defaultdict(Counter)
    for r in rows:
        split_counts[r.get("split", "<none>")][verdicts[r["class_label"]]] += 1
    print("\nrow counts by split x verdict (10 captions per class -> counts are x10):")
    for sp in sorted(split_counts):
        c = split_counts[sp]
        print(f"  {sp:8s} dependent {c['dependent']:4d}  invariant {c['invariant']:3d}"
              f"  degenerate {c['degenerate']:3d}")

    # ------------------------------------------------ the human-checkable census
    print("\n" + "=" * 70)
    print("FULL CAPTION CENSUS OF ALL NON-DEPENDENT CELLS (human check — read")
    print("every line; any order claim here beyond the constant clause breaks")
    print("the invariant verdict for that class):")
    print("=" * 70)
    caps_by_class = defaultdict(set)
    for r in rows:
        caps_by_class[r["class_label"]].add(r["caption"])
    for lab in labels:
        if verdicts[lab] == "dependent":
            continue
        caps = sorted(caps_by_class[lab])
        print(f"\n--- {lab}  [{verdicts[lab].upper()}]  ({len(caps)} distinct captions) ---")
        for c in caps:
            print(f"   {c}")

    # ------------------------------------------------------------ save
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({
            "rule": "truth-conditional (binding 2026-08-09)",
            "shape_invariant": sorted(SHAPE_INVARIANT),
            "fluct_verdict": FLUCT_VERDICT,
            "class_verdicts": verdicts,
            "counts": dict(counts),
            "expected": EXPECTED,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nfull detail -> {OUT}")

    print("\nP2-6 assessment: invariant group = "
          f"{counts['invariant']} of 140 -> "
          + ("CONFIRMED (<= 14)" if counts["invariant"] <= 14 else "MISSED (> 14)"))
    print("Consequence if confirmed: within-SUSHI DiD is underpowered (n=4); "
          "SUSHI serves as the known-order-dependent sanity stratum, with the "
          "4-class invariant mini-group reported descriptively only.")


if __name__ == "__main__":
    main()

"""
census_c4_reanalysis.py — recompute the C4 headline on the human-certified
item set (the 738-item census outcome of the pinning spot-check).

WHAT CHANGED. The pinning spot-check became a full census: all 863 eligible
pinning items were human-judged (q1/q2/q3), yielding 738 valid tests and 125
invalid ones across five documented mechanisms (R1 subset 66, R2 non-pervasive
spike->noisy 42, R3 bare step->noisy 3, R4 truncation 10, R5 magnitude-free
noisy->pn 4). This script joins that census with CLaSP's committed per-item
results and reports the model's accuracy on the certified items — the number
that replaces 0.599/0.593 as the C4 headline.

Registered prediction (before this script was first run): cleaned accuracy
rises noticeably but remains far below the feature control's ~0.93. Whichever
half of that fails, report it as a miss.

ITEM ACCOUNTING (all 990 C4 swap items):
    738 census-valid  + 125 census-invalid  (= 863 eligible pinning)
  +  20 cross-slot step-on-jumpy-shape (excluded from census; audited apart)
  + 107 generic-clause items (never in the pinning pile)
  = 990

GATES
  G1  census sheet: 863 rows, every row judged y/n on all three questions,
      pass tally reproduces 738/863 exactly
  G2  census rows reconstruct to real item_ids
      (C4_fluctuation_type|{sample_id}|{swap_to}|swap), each present in
      probe1_items.jsonl with byte-identical clause_replaced_to — the census
      is provably about these items and no others
  G3  census set == pinning pile minus cross-slot set, recomputed with the
      audit's keyword rule (863 = 883 - 20)
  G4  per-item results cover all 990 item_ids x 3 seeds; recomputed full-C4
      swap accuracy reproduces 0.599 to 3 dp

STATISTICS follow the binding decision: the unit is the signal. The cleaned
accuracy's CI comes from bootstrap over signals (pooling their certified
items), 10,000 resamples, seeded.

SCOPE NOTE, stated up front: the census judged SWAP items only. Random-
condition distractors were never validated and could in principle carry
analogous defects; the random accuracy printed here is for context and is
labelled unvalidated.

Run from repo root:
    python scripts/census_c4_reanalysis.py

Reads:  results/analysis/pinning_spotcheck_sequential.csv   (filled census)
        data/processed/probe1_items.jsonl
        results/experiments/probe1_clasp_per_item.jsonl
        results/analysis/information_availability.json       (feature accs)
        results/analysis/information_availability_279.json   (if present)
Writes: results/analysis/c4_census_reanalysis.json
"""

from __future__ import annotations
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

CENSUS = Path("results/analysis/pinning_spotcheck_sequential.csv")
ITEMS = Path("data/processed/probe1_items.jsonl")
PER_ITEM = Path("results/experiments/probe1_clasp_per_item.jsonl")
FEATURES = Path("results/analysis/information_availability.json")
FEATURES279 = Path("results/analysis/information_availability_279.json")
OUT = Path("results/analysis/c4_census_reanalysis.json")

EXPECT_VALID, EXPECT_TOTAL = 738, 863
JUMPY = {"sawtooth wave", "reverse sawtooth wave", "square wave"}
N_BOOT, BOOT_SEED = 10000, 0


def die(msg):
    print(f"\nGATE FAILED: {msg}")
    sys.exit(1)


def pins(clause: str, val: str):
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


def signal_bootstrap(item_accs_by_signal, rng, n_boot=N_BOOT):
    """item_accs_by_signal: signal -> list of per-item mean-over-seed accs.
    Resample SIGNALS with replacement, pool their items."""
    sigs = sorted(item_accs_by_signal)
    arrs = [np.array(item_accs_by_signal[s]) for s in sigs]
    n = len(sigs)
    idx = np.arange(n)
    vals = []
    for _ in range(n_boot):
        take = rng.integers(0, n, n)
        pool = np.concatenate([arrs[i] for i in take])
        vals.append(pool.mean())
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main():
    def norm(v):
        return str(v).strip().lower() if v is not None else ""

    # ------------------------------------------------------------------ G1
    with open(CENSUS, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != EXPECT_TOTAL:
        die(f"G1: {len(rows)} census rows, expected {EXPECT_TOTAL}")
    qcols = ["q1_grammatical", "q2_asserts_swap_to", "q3_not_still_true"]
    bad = [r["reading_order"] for r in rows
           if any(norm(r[c]) not in ("y", "n") for c in qcols)]
    if bad:
        die(f"G1: rows without clean y/n on all questions: {bad[:5]}")
    for r in rows:
        r["valid"] = all(norm(r[c]) == "y" for c in qcols)
    n_valid = sum(r["valid"] for r in rows)
    print(f"census: {len(rows)} rows, valid {n_valid}, invalid "
          f"{len(rows) - n_valid}")
    if n_valid != EXPECT_VALID:
        die(f"G1: valid tally {n_valid}, expected {EXPECT_VALID} — sheet "
            f"changed since the agreed tally?")
    print("G1 pass")

    # ------------------------------------------------------------------ G2
    items = [json.loads(l) for l in open(ITEMS, encoding="utf-8")]
    c4 = {it["item_id"]: it for it in items
          if it["condition"] == "swap"
          and it["component"] == "C4_fluctuation_type"}
    if len(c4) != 990:
        die(f"G2: {len(c4)} C4 swap items, expected 990")
    seen = set()
    for r in rows:
        iid = f"C4_fluctuation_type|{r['sample_id']}|{r['swap_to']}|swap"
        if iid in seen:
            die(f"G2: duplicate census item {iid}")
        seen.add(iid)
        it = c4.get(iid)
        if it is None:
            die(f"G2: census row order {r['reading_order']} reconstructs to "
                f"unknown item_id {iid}")
        if it["clause_replaced_to"] != r["clause_replaced_to"]:
            die(f"G2: clause text mismatch at {iid}")
        r["item_id"] = iid
    print("G2 pass — all census rows map 1:1 onto real items, clause text "
          "byte-identical")

    # ------------------------------------------------------------------ G3
    pile = {iid for iid, it in c4.items()
            if pins(it["clause_replaced_to"], it["swap_to"])}
    cross = {iid for iid in pile
             if c4[iid]["swap_to"] == "step"
             and any(s in c4[iid]["source_class"] for s in JUMPY)}
    expected_census = pile - cross
    if seen != expected_census:
        die(f"G3: census set != pile minus cross-slot "
            f"(census {len(seen)}, expected {len(expected_census)}, "
            f"symmetric diff {len(seen ^ expected_census)})")
    generic = set(c4) - pile
    print(f"G3 pass — accounting: census {len(seen)} + cross-slot "
          f"{len(cross)} + generic {len(generic)} = "
          f"{len(seen) + len(cross) + len(generic)}")

    # ------------------------------------------------------------------ G4
    recs = [json.loads(l) for l in open(PER_ITEM, encoding="utf-8")]
    acc = defaultdict(list)   # item_id -> [correct x 3 seeds]
    for rr in recs:
        if rr["component"] == "C4_fluctuation_type" and rr["condition"] == "swap":
            acc[rr["item_id"]].append(rr["correct"])
    if set(acc) != set(c4) or any(len(v) != 3 for v in acc.values()):
        die("G4: per-item coverage incomplete for C4 swap items")
    item_acc = {iid: float(np.mean(v)) for iid, v in acc.items()}
    full = float(np.mean([item_acc[i] for i in c4]))
    if round(full, 3) != 0.599:
        die(f"G4: full C4 swap accuracy {full:.3f}, expected 0.599")
    print(f"G4 pass — full C4 swap accuracy reproduces {full:.3f}")

    # ------------------------------------------------------------- groups
    valid_ids = {r["item_id"] for r in rows if r["valid"]}
    invalid_ids = seen - valid_ids
    rng = np.random.default_rng(BOOT_SEED)

    def group_stats(ids, label, boot=False):
        if not ids:
            return {"n": 0}
        vals = [item_acc[i] for i in ids]
        by_sig = defaultdict(list)
        for i in ids:
            by_sig[c4[i]["sample_id"]].append(item_acc[i])
        per_seed = []
        for s in range(3):
            per_seed.append(float(np.mean([acc[i][s] for i in ids])))
        out = {"n": len(ids), "n_signals": len(by_sig),
               "acc": float(np.mean(vals)),
               "per_seed": per_seed,
               "seed_sd": float(np.std(per_seed, ddof=1))}
        if boot:
            out["acc_ci95_signal_bootstrap"] = signal_bootstrap(by_sig, rng)
        print(f"  {label:<34} n={len(ids):>4}  acc {out['acc']:.3f}"
              + (f"  CI95 [{out['acc_ci95_signal_bootstrap'][0]:.3f},"
                 f"{out['acc_ci95_signal_bootstrap'][1]:.3f}]" if boot else "")
              + f"  (seed sd {out['seed_sd']:.3f})")
        return out

    print("\n" + "=" * 78)
    print("CLaSP SWAP ACCURACY BY ITEM VALIDITY (mean over 3 seeds)")
    print("=" * 78)
    g_valid = group_stats(valid_ids, "CENSUS-VALID (the cleaned headline)",
                          boot=True)
    g_invalid = group_stats(invalid_ids, "census-invalid (distractor not "
                                         "clearly false)")
    g_cross = group_stats(cross, "cross-slot (excluded, audited apart)")
    g_generic = group_stats(generic, "generic-clause (never censused)")
    g_full = group_stats(set(c4), "all 990 (original headline)")

    # ------------------------------------------------- per pair, cleaned
    print("\n" + "=" * 78)
    print("PER PAIR: cleaned vs original vs features")
    print("=" * 78)
    feats = json.load(open(FEATURES, encoding="utf-8"))
    fdet = {frozenset({p["a"], p["b"]}): p["acc"]
            for p in feats["pair_detail"]["C4_fluctuation_type"]}
    f279 = None
    if FEATURES279.exists():
        j = json.load(open(FEATURES279, encoding="utf-8"))
        f279 = {frozenset({p["a"], p["b"]}): p["acc"]
                for p in j["pair_detail"]["C4_fluctuation_type"]}

    by_pair_valid = defaultdict(list)
    by_pair_all = defaultdict(list)
    for iid, it in c4.items():
        pk = frozenset({it["swap_from"], it["swap_to"]})
        by_pair_all[pk].append(iid)
        if iid in valid_ids:
            by_pair_valid[pk].append(iid)
    print(f"{'pair':<58}{'clean':>7}{'n':>5}{'orig':>7}{'feat':>7}")
    per_pair = []
    for pk in sorted(by_pair_all, key=lambda k: tuple(sorted(k))):
        a, b = sorted(pk)
        ids_v = by_pair_valid.get(pk, [])
        clean = float(np.mean([item_acc[i] for i in ids_v])) if ids_v else None
        orig = float(np.mean([item_acc[i] for i in by_pair_all[pk]]))
        feat = fdet[pk]
        flag = "  <-- few valid items" if len(ids_v) < 60 else ""
        print(f"{a + ' | ' + b:<58}"
              + (f"{clean:>7.3f}" if clean is not None else f"{'--':>7}")
              + f"{len(ids_v):>5}{orig:>7.3f}{feat:>7.3f}{flag}")
        per_pair.append({"a": a, "b": b, "clean_acc": clean,
                         "n_valid": len(ids_v),
                         "n_all": len(by_pair_all[pk]),
                         "orig_acc": orig, "feature_acc": feat,
                         "feature_acc_279": f279.get(pk) if f279 else None})

    # -------------------------------------------------- context: random
    sig_valid = {c4[i]["sample_id"] for i in valid_ids}
    rand = [rr for rr in recs] if False else None
    rand_acc = []
    for rr in [json.loads(l) for l in open(PER_ITEM, encoding="utf-8")]:
        if rr["component"] == "C4_fluctuation_type" \
                and rr["condition"] == "random" \
                and rr["sample_id"] in sig_valid:
            rand_acc.append(rr["correct"])
    rand_mean = float(np.mean(rand_acc)) if rand_acc else None
    print(f"\nrandom-condition accuracy on the same signals (UNVALIDATED "
          f"items, context only): {rand_mean:.3f} (n records {len(rand_acc)})")

    # feature reference points
    feat_mean = feats["per_component"]["C4_fluctuation_type"]["mean_binary_acc"]
    line = f"feature control C4 mean: {feat_mean:.3f} (full population)"
    feat279_mean = None
    if FEATURES279.exists():
        feat279_mean = j["per_component"]["C4_fluctuation_type"]["mean_binary_acc"]
        line += f" / {feat279_mean:.3f} (279-restricted)"
    print(line)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "census_file": str(CENSUS),
        "expected_tally": [EXPECT_VALID, EXPECT_TOTAL],
        "gates_passed": ["G1", "G2", "G3", "G4"],
        "groups": {"census_valid": g_valid, "census_invalid": g_invalid,
                   "cross_slot": g_cross, "generic": g_generic,
                   "all_990": g_full},
        "per_pair": per_pair,
        "random_condition_context": {
            "acc_on_valid_signals": rand_mean,
            "note": "random distractors were never human-validated; "
                    "context only"},
        "feature_reference": {"c4_mean_full": feat_mean,
                              "c4_mean_279": feat279_mean},
        "bootstrap": {"n_boot": N_BOOT, "seed": BOOT_SEED,
                      "unit": "signal (binding decision)"},
    }, indent=2), encoding="utf-8")
    print(f"\nsaved -> {OUT}")

    print("\n" + "=" * 78)
    print("HOW TO READ THIS")
    print("=" * 78)
    print("The CENSUS-VALID accuracy is the cleaned C4 headline: CLaSP's")
    print("performance on items a human certified as fair tests. Compare it")
    print("to the feature reference, not to 0.5. The registered prediction")
    print("was: rises noticeably, stays far below the features. Check both")
    print("halves against the actual number and report any miss as a miss.")


if __name__ == "__main__":
    main()

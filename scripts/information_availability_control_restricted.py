"""
information_availability_control_restricted.py — the feature control scored on
exactly the 279 probe signals (open item 3 of docs/probe1_findings_clasp.md §7).

THE OBJECTION THIS ANSWERS. The original control reports 5-fold CV accuracy
over all 1,400 SUSHI signals, while CLaSP's probe accuracies are measured on
the 279 held-out probe signals. Every features-vs-CLaSP comparison therefore
sets two numbers side by side that describe different signal populations. This
script removes the asymmetry without changing the protocol:

  * identical feature extraction (imported from
    information_availability_control.py — no reimplementation, no drift);
  * identical classifier and identical StratifiedKFold(5, shuffle, seed 42)
    over the SAME signals, via cross_val_predict, so every signal's
    prediction comes from a model that did not see it;
  * accuracy is then read off twice per discrimination: over the full
    population (must REPRODUCE the committed numbers — gate G4) and over only
    the 279 probe signals (the new, like-for-like number).

Expected outcome, stated before running: no material change — the features are
fixed statistics, not a model that could exploit the larger population. The
point of running it is that "expected" is not "verified".

A REPRODUCTION SUBTLETY, recorded so a near-miss on G4 is diagnosable: the
original control reported cross_val_score(...).mean() (mean of per-fold
accuracies); this script pools out-of-fold predictions. The two coincide when
folds are equal-sized, which holds here because every pair population is
140 (70+70) or 400 (200+200) and both classes divide evenly into 5 folds. If
G4 ever fails by a hair on a future item set, unequal folds are the first
suspect, not data drift.

GATES
  G1  pairs.jsonl provides sample_id, class_label, series for 1,400 sushi rows
  G2  probe1_clasp_per_item.jsonl yields exactly 279 distinct sample_ids,
      all present among the sushi rows
  G3  component_table pair counts are 8/16/10/15/75
  G4  full-population per-pair accuracies reproduce the committed
      information_availability.json (all 124 pairs, tol 1e-9), and so do the
      two multiclass accuracies — the restricted numbers are trusted only
      because the full numbers prove the protocol identical
  (pairs whose 279-subset lacks one class entirely are flagged, acc = null)

OUTPUT is schema-compatible with information_availability.json (pair_detail
with a/b/acc/n), so the per-pair cross-analysis re-runs against it directly
as a sensitivity check:
    python scripts/per_pair_cross_analysis.py ^
        --features results/analysis/information_availability_279.json ^
        --out results/analysis/per_pair_cross_analysis_279.json ^
        --fig results/analysis/per_pair_scatter_279.png

Run from repo root:
    python scripts/information_availability_control_restricted.py
Requires: scikit-learn

Reads:  data/processed/pairs.jsonl
        results/analysis/component_table.json
        results/analysis/information_availability.json   (gate G4)
        results/experiments/probe1_clasp_per_item.jsonl  (the 279 ids)
Writes: results/analysis/information_availability_279.json
"""

from __future__ import annotations
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PAIRS = Path("data/processed/pairs.jsonl")
TABLE = Path("results/analysis/component_table.json")
COMMITTED = Path("results/analysis/information_availability.json")
PER_ITEM = Path("results/experiments/probe1_clasp_per_item.jsonl")
OUT = Path("results/analysis/information_availability_279.json")

TOL = 1e-9


def die(msg):
    print(f"\nGATE FAILED: {msg}")
    sys.exit(1)


def load_feature_module():
    """Import features() from the original control so extraction cannot drift."""
    spec = importlib.util.spec_from_file_location(
        "iac", Path(__file__).parent / "information_availability_control.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def restricted_pair_accuracy(pred, y_pair, sub_local):
    """Accuracy of out-of-fold predictions on the flagged subset.
    pred and y_pair are pair-local (aligned to the masked rows); sub_local is
    the pair-local boolean for membership in the 279 probe signals."""
    if sub_local.sum() == 0:
        return None
    return float((pred[sub_local] == y_pair[sub_local]).mean())


def main():
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
    except ImportError:
        raise SystemExit("needs scikit-learn:  python -m pip install scikit-learn")

    iac = load_feature_module()

    # ------------------------------------------------------------------ G1
    rows = []
    with open(PAIRS, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["dataset"] != "sushi":
                continue
            if "sample_id" not in r:
                die("G1: sushi rows in pairs.jsonl carry no 'sample_id' — "
                    "cannot map the 279 probe signals. Inspect the schema.")
            fl, sh = [p.strip() for p in r["class_label"].split(";")]
            rows.append({"sample_id": r["sample_id"], "fluct": fl,
                         "shape": sh, "series": r["series"]})
    print(f"sushi signals: {len(rows)}")
    if len(rows) != 1400:
        die(f"G1: expected 1400 sushi signals, found {len(rows)}")
    print("G1 pass")

    # ------------------------------------------------------------------ G2
    probe_ids = {json.loads(l)["sample_id"]
                 for l in open(PER_ITEM, encoding="utf-8")}
    all_ids = {r["sample_id"] for r in rows}
    missing = sorted(probe_ids - all_ids)
    print(f"probe signal ids: {len(probe_ids)}   "
          f"missing from pairs.jsonl: {len(missing)}")
    if len(probe_ids) != 279 or missing:
        die(f"G2: expected 279 probe ids all present; missing e.g. {missing[:3]}")
    in279 = np.array([r["sample_id"] in probe_ids for r in rows])
    print(f"G2 pass ({int(in279.sum())} of {len(rows)} signals are probe signals)")

    # ------------------------------------------------------------ features
    print("extracting features (identical code path to the original control)...")
    X = np.array([iac.features(r["series"]) for r in rows])
    y_fluct = np.array([r["fluct"] for r in rows])
    y_shape = np.array([r["shape"] for r in rows])
    n_bad = int((~np.isfinite(X)).any(axis=1).sum())
    if n_bad:
        print(f"non-finite features in {n_bad} signals -> set to 0 "
              f"(same handling as original)")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"feature matrix: {X.shape}")

    def clf():
        return make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=2000, C=1.0))

    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    committed = json.load(open(COMMITTED, encoding="utf-8"))

    # ------------------------------------------------------------------ G3
    with open(TABLE, encoding="utf-8") as f:
        components = json.load(f)["components"]
    expected_counts = {"C1": 8, "C2": 16, "C3": 10, "C4": 15, "C5": 75}

    # --------------------------------------------------------- multiclass
    print("\nmulticlass (out-of-fold predictions, scored full vs 279):")
    multiclass = {}
    for name, y in (("fluctuation_7way", y_fluct), ("shape_20way", y_shape)):
        pred = cross_val_predict(clf(), X, y, cv=cv)
        acc_full = float((pred == y).mean())
        acc_279 = float((pred[in279] == y[in279]).mean())
        want = committed["multiclass"][name]
        if abs(acc_full - want) > TOL:
            die(f"G4 multiclass {name}: full {acc_full:.6f} vs committed "
                f"{want:.6f} — protocol drift; restricted numbers untrusted. "
                f"(If the miss is small, check the fold-size note in the "
                f"docstring and sklearn version vs docs/environment.txt.)")
        print(f"  {name:<20} full {acc_full:.3f} (reproduces committed)   "
              f"279 {acc_279:.3f}")
        multiclass[name] = {"full": acc_full, "restricted_279": acc_279}

    # ---------------------------------------------------------- per pair
    cdet = {c: {frozenset({p['a'], p['b']}): p['acc']
                for p in committed["pair_detail"][c]}
            for c in committed["pair_detail"]}

    pair_detail = defaultdict(list)
    per_component = {}
    flagged = []
    n_repro = 0
    print("\nper-pair: full reproduces committed (G4); 279 is the new number")
    print(f"{'component':<26}{'mean full':>10}{'mean 279':>10}"
          f"{'max |delta|':>12}{'pairs':>7}")
    print("-" * 65)
    for comp, spec in components.items():
        if not spec.get("primary", True):
            continue
        if len(spec["pairs"]) != expected_counts.get(comp.split("_")[0]):
            die(f"G3 {comp}: {len(spec['pairs'])} pairs, expected "
                f"{expected_counts.get(comp.split('_')[0])}")
        slot_y = y_fluct if comp.startswith("C4") else y_shape
        deltas, fulls, res = [], [], []
        for a, b in spec["pairs"]:
            mask = (slot_y == a) | (slot_y == b)
            if mask.sum() < 20:
                continue
            y_pair = slot_y[mask]
            pred = cross_val_predict(clf(), X[mask], y_pair, cv=cv)
            acc_full = float((pred == y_pair).mean())
            want = cdet[comp][frozenset({a, b})]
            if abs(acc_full - want) > TOL:
                die(f"G4 {comp} {a}|{b}: full {acc_full:.6f} vs committed "
                    f"{want:.6f} — protocol drift (see docstring note)")
            n_repro += 1
            sub_local = in279[mask]
            n_a = int(((slot_y == a) & in279).sum())
            n_b = int(((slot_y == b) & in279).sum())
            acc_279 = restricted_pair_accuracy(pred, y_pair, sub_local)
            entry = {"a": a, "b": b, "acc": acc_279, "n": int(sub_local.sum()),
                     "n_a_279": n_a, "n_b_279": n_b,
                     "acc_full": acc_full, "n_full": int(mask.sum())}
            pair_detail[comp].append(entry)
            if n_a == 0 or n_b == 0 or acc_279 is None:
                flagged.append(f"{comp} {a}|{b} (279 subset one-sided: "
                               f"n_a={n_a}, n_b={n_b})")
                continue
            fulls.append(acc_full)
            res.append(acc_279)
            deltas.append(abs(acc_279 - acc_full))
        per_component[comp] = {
            "mean_binary_acc": float(np.mean(res)) if res else None,
            "mean_binary_acc_full": float(np.mean(fulls)) if fulls else None,
            "max_abs_pair_delta": float(np.max(deltas)) if deltas else None,
            "n_pairs_scored": len(res),
        }
        print(f"{comp:<26}{np.mean(fulls):>10.3f}{np.mean(res):>10.3f}"
              f"{np.max(deltas):>12.3f}{len(res):>7}")

    print(f"\nG4 pass — {n_repro} per-pair full accuracies and 2 multiclass "
          f"accuracies reproduce the committed control exactly")
    if flagged:
        print("flagged (not scored on 279):")
        for f_ in flagged:
            print("  ", f_)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({
            "n_signals": 1400,
            "restricted_to": "279 probe signals from probe1_clasp_per_item.jsonl",
            "protocol": "identical features, classifier and folds as "
                        "information_availability.json; out-of-fold "
                        "predictions scored on the restricted population",
            "n_features": X.shape[1],
            "classifier": committed.get("classifier"),
            "cv": committed.get("cv"),
            "gates_passed": ["G1", "G2", "G3", "G4"],
            "multiclass": multiclass,
            "per_component": per_component,
            "pair_detail": dict(pair_detail),
            "flagged_pairs": flagged,
        }, f, indent=2)
    print(f"saved -> {OUT}")

    print("\nHOW TO READ THIS")
    print("The 'max |delta|' column is the largest per-pair change caused by")
    print("restricting the scoring population. Small deltas (a few points,")
    print("consistent with n≈28-vs-140 sampling noise) close open item 3: the")
    print("control's verdict does not depend on the population asymmetry.")
    print("A large systematic drop would instead mean the probe signals are")
    print("unrepresentative — investigate before touching the findings doc.")


if __name__ == "__main__":
    main()

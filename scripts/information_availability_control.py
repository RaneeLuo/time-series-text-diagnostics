"""
information_availability_control.py — is the information CLaSP misses actually
present in the signal?

Probe 1 found that CLaSP distinguishes global shape well and local fluctuation
character barely above chance. The obvious objection is that this reflects task
difficulty rather than model blindness: perhaps telling spikes from noise is
simply harder than telling rising from falling.

This control answers that objection. It extracts a handful of cheap, hand-written
statistical features from the RAW signal -- no learning of representations, no
neural network -- and asks a linear classifier to make the SAME binary
discriminations the probe asks of CLaSP. If a logistic regression on ten summary
features separates "negative spike" from "noisy" almost perfectly, then the
information is trivially available in the signal, and CLaSP's near-chance
performance is a failure to encode it rather than evidence that the distinction
is hard.

Fairness conditions:
  * features are computed on the SAME z-normalised signal CLaSP receives, so
    nothing is available here that was unavailable to the model;
  * the discriminations are the same value pairs used by each probe component;
  * accuracy is measured by stratified 5-fold cross-validation over all 1,400
    signals -- this measures information availability, not a competing model,
    so no held-out split is required (and the coarse 14-signal test split would
    give useless resolution).

Precedent: TS-Haystack's artifact check uses the same logic in reverse -- a
simple model at chance means the signal carries nothing exploitable.

Run from repo root:
    python scripts/information_availability_control.py
Requires: scikit-learn  (python -m pip install scikit-learn)

Writes: results/analysis/information_availability.json
"""

from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

PAIRS = Path("data/processed/pairs.jsonl")
TABLE = Path("results/analysis/component_table.json")
PROBE = Path("results/experiments/probe1_clasp_summary.json")
OUT = Path("results/analysis/information_availability.json")


def znorm(x):
    sd = x.std()
    return np.zeros_like(x) if sd < 1e-8 else (x - x.mean()) / sd


def moving_average(x, w=25):
    k = np.ones(w) / w
    return np.convolve(x, k, mode="same")


def safe_corr(a, b):
    """Lag-1 correlation that survives constant signals.
    The 'clean; constant' class is a literal flat line: after z-normalisation it
    is all zeros, so np.corrcoef divides by a zero standard deviation and returns
    NaN. A constant series has no lag-1 structure to speak of, so 0.0 is the
    correct value, not a missing one."""
    sa, sb = a.std(), b.std()
    if sa < 1e-12 or sb < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def features(raw):
    """Sixteen cheap descriptors of a z-normalised series: six describing global
    shape, ten describing local texture. Deliberately unsophisticated -- the
    argument is stronger the dumber these are. Both groups are needed: rev 1 had
    only the texture group, so it could not test the direction components and
    their low scores reflected the feature set, not the data."""
    x = znorm(np.asarray(raw, dtype=np.float64))
    d = np.diff(x)
    sd = d.std() + 1e-9
    resid = x - moving_average(x)
    n = len(x)

    def kurt(v):
        s = v.std() + 1e-9
        return float(((v - v.mean()) ** 4).mean() / s ** 4)

    def skew(v):
        s = v.std() + 1e-9
        return float(((v - v.mean()) ** 3).mean() / s ** 3)

    t = np.linspace(0.0, 1.0, n)
    thirds = (x[:n // 3].mean(), x[n // 3:2 * n // 3].mean(), x[2 * n // 3:].mean())

    return [
        # --- global shape (added rev 2: the first version had none, which made
        # --- the control silently unable to test direction-type components)
        float(np.polyfit(t, x, 1)[0]),                # linear slope
        float(np.polyfit(t, x, 2)[0]),                # quadratic curvature
        float(np.polyfit(t, x, 3)[0]),                # cubic term
        float(thirds[2] - thirds[0]),                 # last third minus first
        float(thirds[1] - 0.5 * (thirds[0] + thirds[2])),   # middle vs ends
        skew(resid),                                  # asymmetry of local excursions
        # --- local texture (rev 1)
        float(d.std()),                               # step-to-step volatility
        kurt(d),                                      # heavy tails -> spikes
        skew(d),                                      # asymmetry -> pos vs neg spikes
        float(np.abs(d).max() / sd),                  # largest jump, normalised
        float((np.abs(d) > 4 * sd).mean()),           # spike rate
        safe_corr(x[:-1], x[1:]),                     # lag-1 autocorrelation
        float(resid.std()),                           # roughness after smoothing
        kurt(resid),                                  # spikiness of the residual
        float(np.abs(np.diff(moving_average(x, 101))).max()),   # level shifts -> steps
        float((np.sign(d[:-1]) != np.sign(d[1:])).mean()),      # direction reversals
    ]


def main():
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import cross_val_score, StratifiedKFold
    except ImportError:
        raise SystemExit("needs scikit-learn:  python -m pip install scikit-learn")

    rows = []
    with open(PAIRS, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["dataset"] != "sushi":
                continue
            fl, sh = [p.strip() for p in r["class_label"].split(";")]
            rows.append({"fluct": fl, "shape": sh, "series": r["series"]})
    print(f"signals: {len(rows)}")

    print("extracting features...")
    X = np.array([features(r["series"]) for r in rows])
    y_fluct = np.array([r["fluct"] for r in rows])
    y_shape = np.array([r["shape"] for r in rows])
    n_bad = int((~np.isfinite(X)).any(axis=1).sum())
    if n_bad:
        bad_classes = sorted({f'{rows[i]["fluct"]}; {rows[i]["shape"]}'
                              for i in np.where(~np.isfinite(X).all(axis=1))[0]})
        print(f"non-finite features in {n_bad} signals -> set to 0. "
              f"classes affected: {bad_classes[:5]}")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"feature matrix: {X.shape}")

    def clf():
        return make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=2000, C=1.0))

    cv = StratifiedKFold(5, shuffle=True, random_state=42)

    # -------------------------------------------------- multiclass sanity
    print("\n" + "=" * 70)
    print("MULTICLASS RECOVERY FROM 10 HAND-WRITTEN FEATURES")
    print("=" * 70)
    acc_f = cross_val_score(clf(), X, y_fluct, cv=cv).mean()
    acc_s = cross_val_score(clf(), X, y_shape, cv=cv).mean()
    print(f"fluctuation type  (7-way, chance 0.143): {acc_f:.3f}")
    print(f"shape            (20-way, chance 0.050): {acc_s:.3f}")

    # -------------------------------------------------- per-component pairs
    with open(TABLE, encoding="utf-8") as f:
        components = json.load(f)["components"]

    print("\n" + "=" * 70)
    print("BINARY DISCRIMINATIONS — THE SAME PAIRS PROBE 1 USES")
    print("=" * 70)

    per_component = {}
    pair_detail = defaultdict(list)
    for comp, spec in components.items():
        if not spec.get("primary", True):
            continue
        slot_y = y_fluct if comp.startswith("C4") else y_shape
        accs = []
        for a, b in spec["pairs"]:
            mask = (slot_y == a) | (slot_y == b)
            if mask.sum() < 20:
                continue
            acc = cross_val_score(clf(), X[mask], slot_y[mask], cv=cv).mean()
            accs.append(acc)
            pair_detail[comp].append({"a": a, "b": b, "acc": float(acc),
                                      "n": int(mask.sum())})
        if accs:
            per_component[comp] = {"mean_binary_acc": float(np.mean(accs)),
                                   "min": float(np.min(accs)),
                                   "max": float(np.max(accs)),
                                   "n_pairs": len(accs)}
            print(f"{comp:<26} mean {np.mean(accs):.3f}   "
                  f"(range {np.min(accs):.3f}-{np.max(accs):.3f}, {len(accs)} pairs)")

    # -------------------------------------------------- the comparison
    print("\n" + "=" * 70)
    print("THE COMPARISON THAT ANSWERS THE DIFFICULTY OBJECTION")
    print("=" * 70)
    probe = {}
    if PROBE.exists():
        with open(PROBE, encoding="utf-8") as f:
            probe = json.load(f).get("across_seeds", {})

    print(f"{'component':<26}{'features':>10}{'CLaSP swap':>13}{'shortfall':>12}")
    print("-" * 61)
    comparison = {}
    for comp in sorted(per_component):
        feat = per_component[comp]["mean_binary_acc"]
        clasp = probe.get(comp, {}).get("acc_swap_mean")
        if clasp is None:
            print(f"{comp:<26}{feat:>10.3f}{'n/a':>13}{'':>12}")
            continue
        print(f"{comp:<26}{feat:>10.3f}{clasp:>13.3f}{feat - clasp:>+12.3f}")
        comparison[comp] = {"feature_acc": feat, "clasp_swap_acc": clasp,
                            "shortfall": feat - clasp}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"n_signals": len(rows), "n_features": X.shape[1], "feature_groups": {"global_shape": 6, "local_texture": 10},
                   "classifier": "logistic regression on standardised features",
                   "cv": "stratified 5-fold, seed 42",
                   "multiclass": {"fluctuation_7way": float(acc_f),
                                  "shape_20way": float(acc_s)},
                   "per_component": per_component,
                   "pair_detail": dict(pair_detail),
                   "comparison_with_probe1": comparison}, f, indent=2)
    print(f"\nsaved -> {OUT}")

    print("\n" + "=" * 70)
    print("HOW TO READ THIS")
    print("=" * 70)
    print("A large POSITIVE shortfall means ten hand-written features beat a")
    print("trained neural encoder on that discrimination -- the information was")
    print("plainly there and the model did not encode it. That is shortcut")
    print("evidence, and it disposes of the 'this distinction is just hard'")
    print("objection for that component.")
    print("A shortfall near zero, or negative, means the features struggle too;")
    print("for that component the probe result must be reported as ambiguous")
    print("between model blindness and intrinsic difficulty.")


if __name__ == "__main__":
    main()

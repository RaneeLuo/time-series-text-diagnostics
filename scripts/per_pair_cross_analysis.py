"""
per_pair_cross_analysis.py — does CLaSP fail where the features find it hard,
or where the features find it easy?

Resolves open item 1 of docs/probe1_findings_clasp.md §7. The aggregate
difficulty control compares component MEANS (features 0.929 vs CLaSP 0.599 on
fluctuation). This script asks the per-pair version: across the 124 value
pairs, does CLaSP's swap accuracy track the feature baseline's accuracy on the
same discrimination?

  * strong positive correlation -> CLaSP fails where features struggle:
    softens the claim toward intrinsic difficulty.
  * weak/no correlation, failures on feature-easy pairs -> representational
    blind spot: failing on separable distinctions.

Pre-registered before any computation (recorded in the 2026-08-02 session):
prediction was the blind-spot outcome; interpretation threshold MIN_SIGNALS=10.

CONVENTIONS
  * Direction collapse: swap items carry both directions (A->B and B->A); the
    feature control has one entry per UNORDERED pair. CLaSP items are pooled
    over both directions AND over the three seeds; per-pair accuracy is
    mean(correct) over the pooled items. Per-direction and per-seed accuracies
    are kept as diagnostics. (Verified from scripts/generate_probe1_items.py:
    swap_from is the signal's TRUE value, swap_to the distractor's claim.)
  * Pairs with fewer than --min-signals distinct signals are reported but
    excluded from correlations and not interpreted.
  * Spearman primary (bounded accuracies, ceiling effects), Pearson alongside;
    bootstrap-over-pairs CIs are rough at 7-16 pairs and show width only.
  * Known asymmetry, inherited from the aggregate control: feature accuracies
    are 5-fold CV over all 1,400 signals; CLaSP accuracies are over the 279
    probe signals.

GATES (each aborts on failure)
  G1  16,620 records = 5,540 x 3 seeds; 5 components; 2 conditions; 279 signals
  G2  aggregate rand/swap accuracies reproduce probe1_findings_clasp.md to 3 dp
  G3  1:1 pair match between CLaSP and features; counts 8/16/10/15/75
  G4  pooled item counts reconcile with the 8,310 swap records
  G5  the four feature values quoted in the findings doc are present

Run from repo root:
    python scripts/per_pair_cross_analysis.py
    python scripts/per_pair_cross_analysis.py --n-boot 20000

Reads:  results/experiments/probe1_clasp_per_item.jsonl
        results/analysis/information_availability.json
Writes: results/analysis/per_pair_cross_analysis.json
        results/analysis/per_pair_scatter.png   (skipped if no matplotlib)
"""

from __future__ import annotations
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PER_ITEM = Path("results/experiments/probe1_clasp_per_item.jsonl")
FEATURES = Path("results/analysis/information_availability.json")
OUT_JSON = Path("results/analysis/per_pair_cross_analysis.json")
OUT_FIG = Path("results/analysis/per_pair_scatter.png")

DOC_AGG = {  # probe1_findings_clasp.md §3: (random, swap), mean over seeds, 3 dp
    "C1_trend_direction": (0.985, 0.911),
    "C2_trend_family": (0.987, 0.951),
    "C3_periodic_waveform": (0.925, 0.743),
    "C4_fluctuation_type": (0.969, 0.599),
    "C5_signal_regime": (0.953, 0.984),
}
DOC_PAIRS = {  # probe1_findings_clasp.md §4 quoted feature values
    frozenset({"noisy", "negative spike"}): 0.997,
    frozenset({"noisy", "step"}): 0.990,
    frozenset({"negative spike", "positive spike"}): 0.775,
    frozenset({"sinusoidal wave", "triangle wave"}): 0.507,
}
EXPECTED_PAIR_COUNTS = {"C1_trend_direction": 8, "C2_trend_family": 16,
                        "C3_periodic_waveform": 10, "C4_fluctuation_type": 15,
                        "C5_signal_regime": 75}


def die(msg):
    print(f"\nGATE FAILED: {msg}")
    sys.exit(1)


def _rank(v):
    v = np.asarray(v, float)
    order = np.argsort(v)
    r = np.empty(len(v))
    r[order] = np.arange(len(v), dtype=float)
    out = r.copy()
    for val in np.unique(v):        # tie-average
        m = v == val
        out[m] = r[m].mean()
    return out


def spearman(x, y):
    rx, ry = _rank(x), _rank(y)
    if rx.std() < 1e-12 or ry.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.std() < 1e-12 or y.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def boot_ci(x, y, fn, rng, n_boot):
    """Bootstrap over PAIRS. At 7-16 pairs per component the interval is rough;
    it is reported to show width, not to license precision."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    vals = []
    for _ in range(n_boot):
        take = rng.integers(0, n, n)
        v = fn(x[take], y[take])
        if np.isfinite(v):
            vals.append(v)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def make_figure(out_pairs, path, xlabel):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — figure skipped "
              "(python -m pip install matplotlib)")
        return False
    comps = list(EXPECTED_PAIR_COUNTS)
    colors = dict(zip(comps, ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#9467bd"]))
    short = dict(zip(comps, ["C1 direction", "C2 family", "C3 waveform",
                             "C4 fluctuation", "C5 regime"]))
    pairs = [p for p in out_pairs if p["interpretable"]]
    fig, ax = plt.subplots(figsize=(8.5, 7))
    rng = np.random.default_rng(1)
    for c in comps:
        sel = [p for p in pairs if p["component"] == c]
        x = np.array([p["feature_acc"] for p in sel]) + rng.normal(0, 0.002, len(sel))
        y = [p["clasp_swap_acc"] for p in sel]
        ax.scatter(x, y, label=f"{short[c]} (n={len(sel)})", color=colors[c],
                   alpha=0.75, s=48, edgecolors="white", linewidths=0.5)
    ax.plot([0.45, 1.01], [0.45, 1.01], ls="--", c="gray", lw=1, label="y = x")
    ax.axhline(0.5, ls=":", c="black", lw=1)
    ax.text(0.455, 0.507, "chance (CLaSP)", fontsize=8)
    ax.axvline(0.5, ls=":", c="black", lw=1)
    ann = [({"reverse sawtooth wave", "sawtooth wave"}, "C1_trend_direction",
            "rev-saw|saw (C1)", (-10, -14)),
           ({"reverse sawtooth wave", "sawtooth wave"}, "C3_periodic_waveform",
            "rev-saw|saw (C3)", (-108, 2)),
           ({"sinusoidal wave", "triangle wave"}, "C3_periodic_waveform",
            "sin|tri", (8, -4)),
           ({"negative spike", "smooth"}, "C4_fluctuation_type",
            "neg-spike|smooth", (-40, -16)),
           ({"negative spike", "positive spike"}, "C4_fluctuation_type",
            "neg|pos spike", (6, -4))]
    for keyset, c, lab, off in ann:
        hits = [q for q in pairs if q["component"] == c and {q["a"], q["b"]} == keyset]
        if hits:
            p = hits[0]
            ax.annotate(lab, (p["feature_acc"], p["clasp_swap_acc"]),
                        textcoords="offset points", xytext=off, fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("CLaSP swap accuracy "
                  "(per pair, pooled directions & 3 seeds, 279 signals)")
    ax.set_title("Probe 1 per-pair cross-analysis: CLaSP vs "
                 "information-availability control\n(interpretable pairs; "
                 "x jittered \u00b10.002 for visibility)")
    ax.set_xlim(0.44, 1.02)
    ax.set_ylim(0.15, 1.03)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    print(f"figure -> {path}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-item", default=str(PER_ITEM))
    ap.add_argument("--features", default=str(FEATURES))
    ap.add_argument("--out", default=str(OUT_JSON))
    ap.add_argument("--fig", default=str(OUT_FIG))
    ap.add_argument("--min-signals", type=int, default=10,
                    help="interpretation threshold; 10 was pre-registered — "
                         "changing it is a sensitivity check, not the analysis")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--no-fig", action="store_true")
    args = ap.parse_args()
    min_signals = args.min_signals
    rng = np.random.default_rng(0)

    recs = [json.loads(l) for l in open(args.per_item, encoding="utf-8")]
    feats = json.load(open(args.features, encoding="utf-8"))

    # ---------------------------------------------------------------- G1
    seeds = sorted({r["seed"] for r in recs})
    comps = sorted({r["component"] for r in recs})
    conds = sorted({r["condition"] for r in recs})
    n_sig = len({r["sample_id"] for r in recs})
    print(f"records {len(recs)}  seeds {seeds}  components {len(comps)}  "
          f"conditions {conds}  signals {n_sig}")
    if len(recs) != 16620 or len(seeds) != 3 or len(comps) != 5 \
            or conds != ["random", "swap"] or n_sig != 279:
        die("G1 basic shape")
    print("G1 pass")

    # ---------------------------------------------------------------- G2
    for c in comps:
        for j, cond in enumerate(("random", "swap")):
            per_seed = []
            for s in seeds:
                sub = [r["correct"] for r in recs if r["component"] == c
                       and r["condition"] == cond and r["seed"] == s]
                per_seed.append(np.mean(sub))
            got = round(float(np.mean(per_seed)), 3)
            if abs(got - DOC_AGG[c][j]) > 0.0005:
                die(f"G2 {c} {cond}: {got} vs documented {DOC_AGG[c][j]}")
    print("G2 pass — aggregates match probe1_findings_clasp.md")

    # ---------------------------------------------------------------- pool
    swap = [r for r in recs if r["condition"] == "swap"]
    by_pair = defaultdict(list)
    for r in swap:
        by_pair[(r["component"], frozenset({r["swap_from"], r["swap_to"]}))].append(r)

    clasp = {c: {} for c in comps}
    total_items = 0
    for (c, pk), items in by_pair.items():
        total_items += len(items)
        per_seed = {s: float(np.mean([r["correct"] for r in items if r["seed"] == s]))
                    for s in seeds}
        dirs = defaultdict(list)
        for r in items:
            dirs[(r["swap_from"], r["swap_to"])].append(r["correct"])
        clasp[c][pk] = {
            "acc": float(np.mean([r["correct"] for r in items])),
            "n_items": len(items),
            "n_signals": len({r["sample_id"] for r in items}),
            "per_seed": per_seed,
            "per_direction": {f"{a}->{b}": (float(np.mean(v)), len(v))
                              for (a, b), v in dirs.items()},
        }
    if total_items != len(swap):
        die(f"G4 item reconciliation: {total_items} vs {len(swap)}")
    print(f"G4 pass — {total_items} swap records pooled into "
          f"{sum(len(v) for v in clasp.values())} unordered pairs")

    fdet = {c: {frozenset({p['a'], p['b']}): p for p in plist}
            for c, plist in feats["pair_detail"].items()}

    # ---------------------------------------------------------------- G3
    for c in comps:
        ck, fk = set(clasp[c]), set(fdet.get(c, {}))
        if ck != fk:
            die(f"G3 {c}: only-CLaSP={[sorted(p) for p in ck-fk]} "
                f"only-features={[sorted(p) for p in fk-ck]}")
        if len(ck) != EXPECTED_PAIR_COUNTS[c]:
            die(f"G3 {c}: {len(ck)} pairs, expected {EXPECTED_PAIR_COUNTS[c]}")
    print("G3 pass — 1:1 pair match, counts 8/16/10/15/75")

    # ---------------------------------------------------------------- G5
    # The four documented values are FULL-population accuracies. A restricted
    # variant file (information_availability_279.json) carries them in
    # acc_full while acc holds the restricted number; validate against
    # whichever field represents the full population.
    for pk, want in DOC_PAIRS.items():
        found = [fdet[c][pk].get("acc_full", fdet[c][pk]["acc"])
                 for c in comps if pk in fdet[c]]
        if not found or abs(found[0] - want) > 0.001:
            die(f"G5 {sorted(pk)}: {found} vs documented {want}")
    print("G5 pass — documented feature values reconcile"
          + (" (via acc_full: restricted-variant features file)"
             if any("acc_full" in p for c in comps for p in fdet[c].values())
             else ""))

    # ---------------------------------------------------------------- table
    print("\n" + "=" * 108)
    print("PER-PAIR TABLE (pooled over directions and seeds; * = below "
          f"min_signals={min_signals}, reported not interpreted)")
    print("=" * 108)
    out_pairs = []
    for c in comps:
        print(f"\n{c}")
        for pk, cv in sorted(clasp[c].items(), key=lambda kv: kv[1]["acc"]):
            fv = fdet[c][pk]
            a, b = sorted(pk)
            small = "*" if cv["n_signals"] < min_signals else " "
            seed_sd = float(np.std(list(cv["per_seed"].values()), ddof=1))
            print(f"  {small} {a} | {b:<28} clasp {cv['acc']:.3f} "
                  f"(seed sd {seed_sd:.3f})  features {fv['acc']:.3f}  "
                  f"diff {fv['acc'] - cv['acc']:+.3f}  "
                  f"n_items {cv['n_items']}  n_signals {cv['n_signals']}")
            out_pairs.append({
                "component": c, "a": a, "b": b,
                "clasp_swap_acc": cv["acc"], "clasp_seed_sd": seed_sd,
                "clasp_per_seed": cv["per_seed"],
                "clasp_per_direction": cv["per_direction"],
                "feature_acc": fv["acc"], "feature_n": fv["n"],
                "shortfall": fv["acc"] - cv["acc"],
                "n_items": cv["n_items"], "n_signals": cv["n_signals"],
                "interpretable": cv["n_signals"] >= min_signals,
            })

    # ---------------------------------------------------------------- corr
    print("\n" + "=" * 108)
    print("CORRELATION: CLaSP per-pair swap accuracy vs feature per-pair "
          f"accuracy (interpretable pairs, n_signals >= {min_signals})")
    print("=" * 108)
    corr = {}
    for scope in comps + ["ALL"]:
        sel = [p for p in out_pairs if p["interpretable"]
               and (scope == "ALL" or p["component"] == scope)]
        if len(sel) < 4:
            print(f"{scope:<24} n={len(sel)} — too few pairs, skipped")
            corr[scope] = {"n_pairs": len(sel)}
            continue
        x = [p["feature_acc"] for p in sel]
        y = [p["clasp_swap_acc"] for p in sel]
        sp, pe = spearman(x, y), pearson(x, y)
        sp_ci = boot_ci(x, y, spearman, rng, args.n_boot)
        pe_ci = boot_ci(x, y, pearson, rng, args.n_boot)
        note = ("  <- conflates between-component variation; carries no "
                "interpretive weight") if scope == "ALL" else ""
        print(f"{scope:<24} n={len(sel):>3}  spearman {sp:+.3f} "
              f"[{sp_ci[0]:+.3f},{sp_ci[1]:+.3f}]   pearson {pe:+.3f} "
              f"[{pe_ci[0]:+.3f},{pe_ci[1]:+.3f}]{note}")
        corr[scope] = {"n_pairs": len(sel), "spearman": sp,
                       "spearman_ci95": sp_ci, "pearson": pe,
                       "pearson_ci95": pe_ci}

    # ------------------------------------------------- the central quantity
    interp = [p for p in out_pairs if p["interpretable"]]
    fail = [p for p in interp if p["clasp_swap_acc"] < 0.70]
    fa = [p["feature_acc"] for p in fail]
    feature_hard = [p for p in interp if p["feature_acc"] < 0.85]
    print("\n" + "=" * 108)
    print("THE CENTRAL QUANTITY")
    print("=" * 108)
    print(f"pairs where CLaSP < 0.70: {len(fail)}")
    print(f"feature accuracy on those SAME pairs: "
          f"min {min(fa):.3f}  median {float(np.median(fa)):.3f}  max {max(fa):.3f}")
    print(f"feature-hard pairs (features < 0.85): {len(feature_hard)}")
    for p in feature_hard:
        print(f"    {p['component']:<22} {p['a']} | {p['b']:<28} "
              f"features {p['feature_acc']:.3f}  clasp {p['clasp_swap_acc']:.3f}")
    central = {
        "n_clasp_failing_pairs_lt_070": len(fail),
        "feature_acc_on_failing_pairs": {"min": float(min(fa)),
                                         "median": float(np.median(fa)),
                                         "max": float(max(fa))},
        "feature_hard_pairs_lt_085": [
            {"component": p["component"], "a": p["a"], "b": p["b"],
             "feature_acc": p["feature_acc"],
             "clasp_swap_acc": p["clasp_swap_acc"]} for p in feature_hard],
    }

    # ------------------------------------------------------- discrepancies
    print("\nLARGEST DISCREPANCIES (feature - clasp, interpretable):")
    for p in sorted(interp, key=lambda q: -q["shortfall"])[:12]:
        print(f"  {p['component']:<22} {p['a']} | {p['b']:<28} "
              f"features {p['feature_acc']:.3f}  clasp {p['clasp_swap_acc']:.3f}  "
              f"diff {p['shortfall']:+.3f}  n_signals {p['n_signals']}")
    print("reverse direction (CLaSP above features):")
    for p in sorted(interp, key=lambda q: q["shortfall"])[:5]:
        print(f"  {p['component']:<22} {p['a']} | {p['b']:<28} "
              f"features {p['feature_acc']:.3f}  clasp {p['clasp_swap_acc']:.3f}  "
              f"diff {p['shortfall']:+.3f}")

    # ------------------------------------------------------------- save
    payload = {
        "resolves": "probe1_findings_clasp.md section 7 open item 1",
        "conventions": {
            "direction_collapse": "unordered pairs; items pooled over both "
                                  "directions and all three seeds",
            "swap_from_convention": "swap_from = signal's true value "
                                    "(verified in generate_probe1_items.py)",
            "min_signals_for_interpretation": min_signals,
            "min_signals_pre_registered": 10,
            "correlation": "Spearman primary, Pearson secondary; bootstrap CI "
                           "over pairs (rough at these n)",
            "control_asymmetry": "feature accs: 5-fold CV, 1400 signals; "
                                 "CLaSP accs: 279 probe signals",
        },
        "gates_passed": ["G1", "G2", "G3", "G4", "G5"],
        "central_quantity": central,
        "pairs": out_pairs,
        "correlations": corr,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nsaved -> {out_path}")

    if not args.no_fig:
        # label the x-axis for the population the features file was scored on:
        # restricted-variant files carry acc_full per pair (see G5)
        restricted = any("acc_full" in p_ for c in comps
                         for p_ in fdet[c].values())
        xlabel = ("16-feature logistic regression accuracy "
                  + ("(per pair, 5-fold CV predictions scored on the 279 "
                     "probe signals)" if restricted
                     else "(per pair, 5-fold CV, 1400 signals)"))
        make_figure(out_pairs, Path(args.fig), xlabel)

    print("\n" + "=" * 108)
    print("HOW TO READ THIS")
    print("=" * 108)
    print("If CLaSP's failing pairs had LOW feature accuracy, its failures")
    print("would track intrinsic difficulty. If they have HIGH feature")
    print("accuracy, CLaSP is failing on distinctions a 16-feature logistic")
    print("regression separates — a representational blind spot. The 'central")
    print("quantity' block above answers this directly; the correlations are")
    print("supporting structure. Per-pair estimates are noisy (see seed sd):")
    print("no per-pair verdicts — the unit of claim remains the component.")


if __name__ == "__main__":
    main()

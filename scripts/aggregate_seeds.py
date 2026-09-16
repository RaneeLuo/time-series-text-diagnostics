"""
aggregate_seeds.py — combine per-seed baseline evaluations into one reference.

Why this matters beyond "rigour": the spread across seeds is the NOISE FLOOR for
every probe in this thesis. A probe-induced degradation is only interpretable if
it is larger than the variation produced by re-training with a different random
seed. This script reports that spread explicitly.

Usage (from repo root):
    python scripts/aggregate_seeds.py \
        --inputs results/experiments/eval_baseline_seed42.json \
                 results/experiments/eval_baseline_seed43.json \
                 results/experiments/eval_baseline_seed44.json \
        --out results/experiments/baseline_clasp.json

Output: results/experiments/baseline_clasp.json  -- the canonical frozen baseline
referenced by all probe experiments.
"""

from __future__ import annotations
import argparse
import json
import statistics as st
from pathlib import Path

# t critical values (two-sided 95%) for small n, df = n-1
T95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571}

STRICT_KEYS = ["recall@1", "recall@5", "recall@10", "mrr"]
ROWS = ["all", "truce", "sushi"]


def summarise(values: list[float]) -> dict:
    n = len(values)
    mean = st.fmean(values)
    if n < 2:
        return {"mean": mean, "sd": None, "ci95_halfwidth": None,
                "min": mean, "max": mean, "n_seeds": n}
    sd = st.stdev(values)
    half = T95.get(n, 2.0) * sd / (n ** 0.5)
    return {"mean": mean, "sd": sd, "ci95_halfwidth": half,
            "min": min(values), "max": max(values), "n_seeds": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="per-seed eval JSON files produced by evaluate.py")
    ap.add_argument("--out", default="results/experiments/baseline_clasp.json")
    args = ap.parse_args()

    reports = []
    for p in args.inputs:
        with open(p) as f:
            reports.append((p, json.load(f)))
    n = len(reports)
    print(f"aggregating {n} seed runs\n")

    agg = {"n_seeds": n, "sources": [p for p, _ in reports], "strict": {}}

    for row in ROWS:
        agg["strict"][row] = {}
        for key in STRICT_KEYS:
            vals = []
            for p, r in reports:
                try:
                    vals.append(float(r["strict"][row][key]))
                except (KeyError, TypeError):
                    raise SystemExit(f"missing strict.{row}.{key} in {p}")
            agg["strict"][row][key] = summarise(vals)

    # soft metric, if present in every report
    soft_key = "soft_mAP@10_ts0.5"
    if all(soft_key in r for _, r in reports):
        agg[soft_key] = {}
        for row in ROWS:
            vals = [float(r[soft_key][row]["mAP@10"]) for _, r in reports
                    if row in r[soft_key]]
            if len(vals) == n:
                agg[soft_key][row] = summarise(vals)

    # ---- printed report ----
    print(f"{'row':<7}{'metric':<11}{'mean':>9}{'sd':>9}{'±95% CI':>10}"
          f"{'min':>9}{'max':>9}")
    print("-" * 64)
    for row in ROWS:
        for key in STRICT_KEYS:
            s = agg["strict"][row][key]
            sd = f"{s['sd']:.4f}" if s["sd"] is not None else "   n/a"
            ci = f"{s['ci95_halfwidth']:.4f}" if s["ci95_halfwidth"] is not None else "   n/a"
            print(f"{row:<7}{key:<11}{s['mean']:>9.4f}{sd:>9}{ci:>10}"
                  f"{s['min']:>9.4f}{s['max']:>9.4f}")
        print()

    # ---- the number that matters for probes ----
    mrr = agg["strict"]["all"]["mrr"]
    r1 = agg["strict"]["all"]["recall@1"]
    if mrr["sd"] is not None and mrr["mean"] > 0:
        rel_mrr = 100 * mrr["sd"] / mrr["mean"]
        rel_r1 = 100 * r1["sd"] / r1["mean"] if r1["mean"] > 0 else float("nan")
        agg["noise_floor"] = {
            "mrr_relative_sd_pct": rel_mrr,
            "recall@1_relative_sd_pct": rel_r1,
            "note": ("Seed-to-seed relative SD of the unperturbed baseline. "
                     "Probe-induced relative degradation should be compared "
                     "against this floor; degradation of comparable magnitude "
                     "is not evidence of shortcut sensitivity."),
        }
        print("NOISE FLOOR (seed-to-seed variation of the unperturbed baseline)")
        print(f"  MRR       relative SD: {rel_mrr:.1f}%")
        print(f"  Recall@1  relative SD: {rel_r1:.1f}%")
        print("  -> probe degradations must clearly exceed this to be meaningful.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(agg, f, indent=2)
    print(f"\nsaved -> {out}")
    if n < 3:
        print("WARNING: fewer than 3 seeds; the CI is not meaningful.")


if __name__ == "__main__":
    main()

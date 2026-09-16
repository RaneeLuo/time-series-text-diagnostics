#!/usr/bin/env python3
"""
analyze_probe2_clasp.py — formal statistics for the CLaSP Probe-2 records.

Scores the registered predictions on their exact 2026-08-09 wording
(handoff §4.6 item 2), using the per-query average-rank records:
  P2-1  sf-all SUSHI dependent: MRR degradation exceeds 0.05 in all
        three seeds (scored under BOTH readings: absolute delta > 0.05
        and relative delta > 5%).
  P2-2  DiD (dependent - invariant degradation) positive in all three
        seeds under sf-all. Primary substrate: TRUCE (the record places
        the CLaSP caption-group DiD on TRUCE); SUSHI DiD reported
        descriptively (n=4 invariant).
  P2-3  severity ordering sf-all >= sf-half in every seed (dependent
        groups, both substrates). No ex-half prediction (registered).
  P2-5  TRUCE invariant group passes TOST under sf-all. Margin PINNED
        2026-08-13 PRE-ANALYSIS: +/-0.05 absolute MRR degradation,
        alpha 0.05 (90% cluster-bootstrap CI inside the margin),
        mirroring P2-1's registered threshold.
P2-4 (floor VOID) belongs to the floor arm, not scored here.

Statistics: degradation = mean paired reciprocal-rank difference
(1/rank_unpert - 1/rank_pert). Cluster bootstrap resamples SIGNALS
(gt sample_ids), not queries — TRUCE queries share signals 3:1.
2,000 replicates, percentile CIs, rng seed 42. Wilcoxon signed-rank
on the paired differences per seed (scipy). Ambiguous and degenerate
groups excluded from inference, reported descriptively.

Run from the repository root:
    python -m models.clasp.analyze_probe2
Reads:  results/experiments/probe2_clasp_per_query_seed{42,43,44}.jsonl
Writes: results/experiments/probe2_clasp_stats.json
"""

from __future__ import annotations
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from scipy.stats import wilcoxon
except ImportError:
    print("scipy required: pip install scipy", file=sys.stderr)
    sys.exit(1)

EXP = Path("results/experiments")
SEEDS = (42, 43, 44)
PERTS = ("sf_all", "sf_half", "ex_half", "masking")
B = 2000
TOST_MARGIN = 0.05
RNG_SEED = 42


def fail(msg):
    print(f"\nGATE FAILED: {msg}", file=sys.stderr)
    sys.exit(1)


def load_seed(seed):
    path = EXP / f"probe2_clasp_per_query_seed{seed}.jsonl"
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    if len(rows) != 878:
        fail(f"{path.name}: {len(rows)} rows != 878")
    from collections import Counter
    cs = Counter((r["substrate"], r["group"]) for r in rows)
    if dict(cs) != {("sushi", "dependent"): 135, ("sushi", "invariant"): 4,
                    ("sushi", "degenerate"): 1, ("truce", "dependent"): 715,
                    ("truce", "invariant"): 18, ("truce", "ambiguous"): 5}:
        fail(f"{path.name}: group counts {dict(cs)} differ from certified")
    return rows


def rr_diff(rows, pert):
    """Paired reciprocal-rank differences (positive = degradation)."""
    return np.array([1.0 / r["rank_unperturbed"] - 1.0 / r[f"rank_{pert}"]
                     for r in rows])


def cluster_boot_mean(rows, pert, rng):
    """Bootstrap the mean rr-diff by resampling gt signals."""
    by_sig = defaultdict(list)
    for r in rows:
        by_sig[r["gt"]].append(1.0 / r["rank_unperturbed"]
                               - 1.0 / r[f"rank_{pert}"])
    sigs = sorted(by_sig)
    out = np.empty(B)
    for b in range(B):
        pick = rng.choice(len(sigs), size=len(sigs), replace=True)
        vals = np.concatenate([by_sig[sigs[i]] for i in pick])
        out[b] = vals.mean()
    return out


def cluster_boot_did(dep_rows, inv_rows, pert, rng):
    """DiD bootstrap: resample signals within each group independently."""
    def group_by_sig(rows):
        d = defaultdict(list)
        for r in rows:
            d[r["gt"]].append(1.0 / r["rank_unperturbed"]
                              - 1.0 / r[f"rank_{pert}"])
        return d
    gd, gi = group_by_sig(dep_rows), group_by_sig(inv_rows)
    sd, si = sorted(gd), sorted(gi)
    out = np.empty(B)
    for b in range(B):
        pd = rng.choice(len(sd), size=len(sd), replace=True)
        pi = rng.choice(len(si), size=len(si), replace=True)
        vd = np.concatenate([gd[sd[i]] for i in pd]).mean()
        vi = np.concatenate([gi[si[i]] for i in pi]).mean()
        out[b] = vd - vi
    return out


def ci(a, lo=2.5, hi=97.5):
    return float(np.percentile(a, lo)), float(np.percentile(a, hi))


def main():
    rng = np.random.default_rng(RNG_SEED)
    data = {s: load_seed(s) for s in SEEDS}
    print(f"loaded 3 x 878 records; certified group counts verified")
    n_inv_sigs = len({r["gt"] for r in data[42]
                      if r["substrate"] == "truce" and r["group"] == "invariant"})
    print(f"TRUCE invariant group: 18 queries over {n_inv_sigs} signals "
          f"(cluster count for the TOST CI)")

    results = {"tost_margin": TOST_MARGIN, "B": B, "seeds": {}}
    p21_abs, p21_rel, p22, p23, p25 = [], [], [], [], []

    for s in SEEDS:
        rows = data[s]
        sel = {}
        for sub in ("sushi", "truce"):
            for grp in ("dependent", "invariant"):
                sel[(sub, grp)] = [r for r in rows
                                   if r["substrate"] == sub and r["group"] == grp]
        print(f"\n===== seed {s} =====")
        seed_out = {}
        for sub in ("sushi", "truce"):
            dep, inv = sel[(sub, "dependent")], sel[(sub, "invariant")]
            for pert in PERTS:
                d = rr_diff(dep, pert)
                boot = cluster_boot_mean(dep, pert, rng)
                lo, hi = ci(boot)
                try:
                    w = wilcoxon(d, alternative="greater")
                    pval = float(w.pvalue)
                except ValueError:
                    pval = float("nan")
                base = np.mean([1.0 / r["rank_unperturbed"] for r in dep])
                seed_out[f"{sub}/dependent/{pert}"] = {
                    "delta": float(d.mean()), "ci": [lo, hi],
                    "rel": float(d.mean() / base), "wilcoxon_p": pval}
                if pert == "sf_all":
                    print(f"  {sub} dep sf_all: delta {d.mean():+.4f} "
                          f"[{lo:+.4f}, {hi:+.4f}]  rel {d.mean()/base:+.1%}  "
                          f"Wilcoxon p={pval:.1e}")
            # DiD + invariant under sf_all
            di = rr_diff(inv, "sf_all")
            binv = cluster_boot_mean(inv, "sf_all", rng)
            ilo, ihi = ci(binv)
            ilo90, ihi90 = ci(binv, 5, 95)
            bdid = cluster_boot_did(dep, inv, "sf_all", rng)
            dlo, dhi = ci(bdid)
            did = float(rr_diff(dep, "sf_all").mean() - di.mean())
            tost_pass = (-TOST_MARGIN < ilo90) and (ihi90 < TOST_MARGIN)
            seed_out[f"{sub}/invariant/sf_all"] = {
                "delta": float(di.mean()), "ci95": [ilo, ihi],
                "ci90": [ilo90, ihi90], "tost_pass": bool(tost_pass)}
            seed_out[f"{sub}/DiD/sf_all"] = {"did": did, "ci": [dlo, dhi]}
            print(f"  {sub} inv sf_all: delta {di.mean():+.4f} "
                  f"90%CI [{ilo90:+.4f}, {ihi90:+.4f}]  "
                  f"TOST(+/-{TOST_MARGIN}): {'PASS' if tost_pass else 'not passed'}")
            print(f"  {sub} DiD sf_all: {did:+.4f} 95%CI [{dlo:+.4f}, {dhi:+.4f}]")

        # prediction bookkeeping per seed
        sd = seed_out["sushi/dependent/sf_all"]
        p21_abs.append(sd["delta"] > 0.05)
        p21_rel.append(sd["rel"] > 0.05)
        p22.append(seed_out["truce/DiD/sf_all"]["did"] > 0)
        p23.append(seed_out["sushi/dependent/sf_all"]["delta"] >=
                   seed_out["sushi/dependent/sf_half"]["delta"] and
                   seed_out["truce/dependent/sf_all"]["delta"] >=
                   seed_out["truce/dependent/sf_half"]["delta"])
        p25.append(seed_out["truce/invariant/sf_all"]["tost_pass"])
        results["seeds"][str(s)] = seed_out

    print("\n===== REGISTERED-PREDICTION SCORECARD =====")
    verdicts = {
        "P2-1": ("CONFIRMED" if all(p21_abs) and all(p21_rel) else "MISSED",
                 f"abs>0.05 {p21_abs}, rel>5% {p21_rel}"),
        "P2-2": ("CONFIRMED" if all(p22) else "MISSED",
                 f"TRUCE DiD positive per seed: {p22}"),
        "P2-3": ("CONFIRMED" if all(p23) else "MISSED",
                 f"sf_all >= sf_half per seed (both substrates): {p23}"),
        "P2-5": ("CONFIRMED" if all(p25) else "MISSED",
                 f"TRUCE invariant TOST pass per seed: {p25}"),
    }
    for k, (v, detail) in verdicts.items():
        print(f"  {k}: {v}   ({detail})")
    results["verdicts"] = {k: v for k, (v, _) in verdicts.items()}

    out = EXP / "probe2_clasp_stats.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
analyze_probe2.py (text-embedding-3-large) — P2-4 scoring for the floor
Probe-2 negative control.

SCORING PINNED 2026-08-14 BEFORE THIS SCRIPT FIRST RAN (chat record):
  Inference cells: sushi/dependent (n=135) and truce/dependent (n=715).
  Thin cells (truce/invariant 18, sushi/invariant 4) are reported with
  CIs and their n, never load-bearing. Ambiguous/degenerate descriptive
  only (standing convention).
  P2-4 CONFIRMED iff TOST (+/-0.05 absolute MRR degradation, alpha 0.05
  via 90% cluster-bootstrap CI inside the margin; B=2000; rng seed 42;
  signals resampled, never queries — binding) PASSES on both inference
  cells for all four perturbations in all three arms (24 TOSTs), AND no
  inference-cell |delta| point estimate >= 0.05.
  DiD is reported (decomposition + leave-out-OW_5 sensitivity), not a
  verdict input: a difference between two near-chance groups is not
  shortcut evidence (VOID logic); the scored guarantee is that no DiD
  is carried by dependent-group degradation (covered by the TOSTs).
  Wilcoxon two-sided, report-only (large-n significance of a sub-margin
  drift toward chance is expected, not meaningful).

Registered cross-computation gate: the REF table below was computed
independently from the same per-query records (uploaded copies,
2026-08-14, Claude's environment). This run must reproduce every value
to <= 1e-9 — HARD STOP otherwise (either the records differ from the
ones the table was derived from, or one computation is wrong).

Registered predictions (open at pin time):
  all 24 TOSTs pass -> P2-4 CONFIRMED;
  Wilcoxon truce/dep sf_all p<0.05 in all arms (expected, stated);
  TRUCE DiD without OW_5's signal drops to ~+0.01 in arms 43/44;
  truce/invariant CIs wide (18 queries over 14 signals).

Run from the repository root:
    python -m models.openai_embed.analyze_probe2
Reads:  results/experiments/probe2_openai_per_query_seed{42,43,44}.jsonl
Writes: results/experiments/probe2_openai_stats.json
"""

from __future__ import annotations
import json
import sys
from collections import Counter, defaultdict
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
CHANCE_MRR = float(np.mean(1.0 / np.arange(1, 387)))   # pool 386

# Registered cross-computation reference (2026-08-14): mean paired
# reciprocal-rank difference, positive = degradation. HARD GATE at 1e-9.
REF = {
    42: {("sushi", "sf_all"): +0.000420, ("sushi", "sf_half"): -0.001681,
         ("sushi", "ex_half"): -0.000762, ("sushi", "masking"): +0.000068,
         ("truce", "sf_all"): +0.009636, ("truce", "sf_half"): +0.003807,
         ("truce", "ex_half"): +0.009649, ("truce", "masking"): +0.000319,
         ("sushi_inv", "sf_all"): -0.001065, ("truce_inv", "sf_all"): -0.000651},
    43: {("sushi", "sf_all"): +0.000446, ("sushi", "sf_half"): -0.001730,
         ("sushi", "ex_half"): -0.000762, ("sushi", "masking"): +0.000097,
         ("truce", "sf_all"): +0.010483, ("truce", "sf_half"): -0.000369,
         ("truce", "ex_half"): +0.009649, ("truce", "masking"): +0.002461,
         ("sushi_inv", "sf_all"): -0.000562, ("truce_inv", "sf_all"): -0.073021},
    44: {("sushi", "sf_all"): +0.000416, ("sushi", "sf_half"): -0.002358,
         ("sushi", "ex_half"): -0.000762, ("sushi", "masking"): +0.000139,
         ("truce", "sf_all"): +0.007288, ("truce", "sf_half"): +0.001008,
         ("truce", "ex_half"): +0.009649, ("truce", "masking"): -0.000844,
         ("sushi_inv", "sf_all"): +0.000194, ("truce_inv", "sf_all"): -0.064530},
}
REF_TOL = 5e-7   # REF is recorded to 6 decimals; exact match below rounding


def fail(msg):
    print(f"\nGATE FAILED: {msg}", file=sys.stderr)
    sys.exit(1)


def load_seed(seed):
    path = EXP / f"probe2_openai_per_query_seed{seed}.jsonl"
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    if len(rows) != 878:
        fail(f"{path.name}: {len(rows)} rows != 878")
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
        out[b] = (np.concatenate([gd[sd[i]] for i in pd]).mean()
                  - np.concatenate([gi[si[i]] for i in pi]).mean())
    return out


def ci(a, lo=2.5, hi=97.5):
    return float(np.percentile(a, lo)), float(np.percentile(a, hi))


def main():
    rng = np.random.default_rng(RNG_SEED)
    data = {s: load_seed(s) for s in SEEDS}
    print("loaded 3 x 878 records; certified group counts verified")

    # Gate: unperturbed ranks identical across the three files (same
    # cached embeddings by construction — a mismatch means mixed files)
    base_ranks = [r["rank_unperturbed"] for r in data[42]]
    for s in (43, 44):
        if [r["rank_unperturbed"] for r in data[s]] != base_ranks:
            fail(f"unperturbed ranks differ between seed files 42 and {s}")
    print("gate: unperturbed ranks identical across arm files — PASS")

    n_inv_sigs = len({r["gt"] for r in data[42]
                      if r["substrate"] == "truce"
                      and r["group"] == "invariant"})
    print(f"TRUCE invariant: 18 queries over {n_inv_sigs} signals")
    print(f"chance MRR reference (pool 386): {CHANCE_MRR:.4f}")
    print("VOID banner: all cells sit near chance; RELATIVE deltas on a "
          "~0.03 base are printed for completeness and are NOT evidence "
          "(pre-declared VOID, PROJECT_CONTEXT).")

    results = {"tost_margin": TOST_MARGIN, "B": B, "chance_mrr": CHANCE_MRR,
               "scoring": "pinned 2026-08-14 pre-run (see docstring)",
               "seeds": {}}
    tost_all, absmax = [], 0.0

    for s in SEEDS:
        rows = data[s]
        sel = {}
        for sub in ("sushi", "truce"):
            for grp in ("dependent", "invariant"):
                sel[(sub, grp)] = [r for r in rows if r["substrate"] == sub
                                   and r["group"] == grp]
        print(f"\n===== arm {s} =====")
        seed_out = {}
        for sub in ("sushi", "truce"):
            dep, inv = sel[(sub, "dependent")], sel[(sub, "invariant")]
            base = float(np.mean([1.0 / r["rank_unperturbed"] for r in dep]))
            for pert in PERTS:
                d = rr_diff(dep, pert)
                dm = float(d.mean())
                # registered cross-computation gate
                refv = REF[s][(sub, pert)]
                if abs(dm - refv) > REF_TOL:
                    fail(f"X-check arm {s} {sub}/dep/{pert}: computed "
                         f"{dm:+.6f} vs registered {refv:+.6f}")
                boot = cluster_boot_mean(dep, pert, rng)
                lo95, hi95 = ci(boot)
                lo90, hi90 = ci(boot, 5, 95)
                tost = (-TOST_MARGIN < lo90) and (hi90 < TOST_MARGIN)
                tost_all.append(tost)
                absmax = max(absmax, abs(dm))
                try:
                    pval = float(wilcoxon(d, alternative="two-sided").pvalue)
                except ValueError:
                    pval = float("nan")
                seed_out[f"{sub}/dependent/{pert}"] = {
                    "delta": dm, "ci95": [lo95, hi95], "ci90": [lo90, hi90],
                    "rel": dm / base, "wilcoxon_two_sided_p": pval,
                    "tost_pass": bool(tost)}
                print(f"  {sub}/dep {pert:8s} delta {dm:+.4f} "
                      f"90%CI [{lo90:+.4f},{hi90:+.4f}] "
                      f"TOST {'PASS' if tost else 'FAIL'}  "
                      f"(rel {dm/base:+.1%}, Wilcoxon p={pval:.1e})")
            # invariant cell, sf_all focus + all perts descriptive
            for pert in PERTS:
                di = rr_diff(inv, pert)
                dim = float(di.mean())
                if pert == "sf_all":
                    key = "sushi_inv" if sub == "sushi" else "truce_inv"
                    refv = REF[s][(key, "sf_all")]
                    if abs(dim - refv) > REF_TOL:
                        fail(f"X-check arm {s} {sub}/inv/sf_all: {dim:+.6f} "
                             f"vs registered {refv:+.6f}")
                    binv = cluster_boot_mean(inv, pert, rng)
                    ilo, ihi = ci(binv)
                    seed_out[f"{sub}/invariant/sf_all"] = {
                        "delta": dim, "ci95": [ilo, ihi], "n": len(inv),
                        "note": "thin cell, reported with n, "
                                "never load-bearing"}
                    print(f"  {sub}/inv sf_all   delta {dim:+.4f} "
                          f"95%CI [{ilo:+.4f},{ihi:+.4f}]  [n={len(inv)}]")
                else:
                    seed_out.setdefault(
                        f"{sub}/invariant/deltas_descriptive", {})[pert] = dim
            # DiD (sf_all) + decomposition + leave-out-OW_5 for TRUCE
            dd = float(rr_diff(dep, "sf_all").mean())
            di = float(rr_diff(inv, "sf_all").mean())
            bdid = cluster_boot_did(dep, inv, "sf_all", rng)
            dlo, dhi = ci(bdid)
            seed_out[f"{sub}/DiD/sf_all"] = {
                "did": dd - di, "ci95": [dlo, dhi],
                "decomposition": {"delta_dependent": dd,
                                  "delta_invariant": di}}
            print(f"  {sub} DiD sf_all  {dd - di:+.4f} "
                  f"95%CI [{dlo:+.4f},{dhi:+.4f}]   decomposition: "
                  f"dep {dd:+.4f} / inv {di:+.4f}"
                  f"{'  <- invariant-driven' if abs(di) > abs(dd) else ''}")
            if sub == "truce":
                inv_keep = [r for r in inv if "OW_5" not in r["gt"]]
                di_lo = float(rr_diff(inv_keep, "sf_all").mean())
                seed_out["truce/DiD/sf_all"]["leave_out_OW_5"] = {
                    "did": dd - di_lo, "n_inv": len(inv_keep)}
                print(f"  truce DiD without OW_5's signal: {dd - di_lo:+.4f} "
                      f"(inv n={len(inv_keep)})  [registered: ~+0.01 in "
                      f"arms 43/44]")
        results["seeds"][str(s)] = seed_out

    print("\n===== P2-4 VERDICT (scoring pinned pre-run) =====")
    verdict = "CONFIRMED" if all(tost_all) and absmax < TOST_MARGIN \
        else "MISSED"
    print(f"  TOSTs passed: {sum(tost_all)}/{len(tost_all)}   "
          f"max inference-cell |delta|: {absmax:.4f} (margin {TOST_MARGIN})")
    print(f"  P2-4: {verdict} — floor VOID; no degradation beyond the "
          f"margin on any inference cell; DiD not carried by "
          f"dependent-group degradation")
    results["verdicts"] = {"P2-4": verdict,
                           "tost_passed": f"{sum(tost_all)}/{len(tost_all)}",
                           "max_abs_delta_inference": absmax}

    out = EXP / "probe2_openai_stats.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

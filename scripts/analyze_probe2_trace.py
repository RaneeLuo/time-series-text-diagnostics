#!/usr/bin/env python3
"""
analyze_probe2_trace.py — formal statistics for the TRACE Probe-2 arm.

Consumes the per-query records written by models/trace/run_probe2.py and
produces the numbers the thesis quotes. It computes NOTHING from the model;
every input is a stored rank.

WHAT IS BEING TESTED
TRACE's caption-group DiD is unposable (2,005 dependent / 1 ambiguous / 0
invariant), so per the accepted reframe the diagnostic is the DEGRADATION
PROFILE across perturbation types. There is no invariant group, therefore
NO TOST leg here — the P2-5 analogue does not exist for TRACE and its
absence is a design fact, not an omission.

  A. Degradation of each perturbation vs unperturbed: absolute and relative
     MRR change with cluster-bootstrap CIs, plus paired Wilcoxon on the
     per-query rank shift, Holm-corrected across the four perturbations.
  B. The PROFILE: all six pairwise perturbation contrasts, paired Wilcoxon
     on per-query ranks, Holm-corrected. ex_half vs masking is pre-flagged
     as the fragile one — its per-query dominance is ~57%, barely above a
     coin flip, so a significant p here must be reported WITH the effect
     size, never alone.
  C. RESIDUAL ABOVE CHANCE under sf_all. Shuffling preserves each channel's
     value multiset exactly, so anything retained after sf_all is
     distributional rather than sequential information. If the sf_all MRR
     CI excludes the chance value, TRACE reads more than order — which is
     the question Probe 3 exists to ask. Named here so it is not
     rediscovered later.
  D. P2-9 scoring (registered 2026-08-10): sf_all degradation exceeds the
     margin in all three seeds. Margin is the P2-1 mirror, +-0.05 RELATIVE,
     pinned pre-analysis and never adjusted afterwards.

RESAMPLING UNIT
The 2,006 rows are simultaneously the queries and the retrieval pool, so
rank observations are not independent: perturbing the pool moves every
query. The bootstrap resamples ROWS (clusters), which is the correct unit
for the query side. It does NOT model the pool-side dependence — no
row-resampling scheme can, because the pool is shared. Stated as a
limitation rather than papered over; the CIs are therefore mildly
optimistic and the headline gaps are far too large to turn on it.

INFERENCE SET
Dependent group only (n = 2,005). The single ambiguous row (1191) is
excluded from every test and reported descriptively with its n, per the
thin-cell rule. Primary direction is text->ts; ts->text is reported in
full as secondary.

GATES
  G-in    4,012 records per seed; row_idx contiguous 0..2005 in both
          directions; ids identical across directions and across seeds;
          groups 2,005/1 — HARD STOP
  G-range every rank in [1, 2006] — HARD STOP
  G-repro recomputed MRR must match the runner's summary tables to <=1e-9
          — HARD STOP (catches a stale or mismatched record file)
  G-seeds the three seed files must be genuinely different runs (identical
          unperturbed rank vectors would mean a copy) — HARD STOP

USAGE (from the thesis repo root):
  python scripts/analyze_probe2_trace.py \
      --records results/experiments/probe2_trace_per_query_seed13.jsonl \
                results/experiments/probe2_trace_per_query_seed14.jsonl \
                results/experiments/probe2_trace_per_query_seed15.jsonl \
      --summary results/experiments/probe2_trace_summary.json \
      --out results/experiments/probe2_trace_stats.json

Runtime: about a minute. Paste the entire console output back.
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np

PERTS = ["sf_all", "sf_half", "ex_half", "masking"]
DIRECTIONS = ["text2ts", "ts2text"]
POOL = 2006
N_BOOT = 2000
BOOT_SEED = 20260815
P2_9_MARGIN = 0.05          # relative; mirrors P2-1; PINNED PRE-ANALYSIS
FRAGILE_PAIR = ("ex_half", "masking")


def fail(gate, msg):
    print(f"\n[GATE FAILED] {gate}: {msg}")
    print("Hard stop. Investigate before proceeding — do not work around.")
    sys.exit(1)


def holm(pvals):
    """Holm-Bonferroni adjusted p-values, order preserved."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for i, o in enumerate(order):
        running = max(running, (m - i) * p[o])
        adj[o] = min(1.0, running)
    return adj


def fmt_p(p):
    if p == 0.0:
        return "<1e-300"
    return f"{p:.3e}"


def chance_mrr(n):
    """E[1/rank] for a uniformly random ranking over n candidates."""
    return float(np.sum(1.0 / np.arange(1, n + 1)) / n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", nargs="+", required=True)
    ap.add_argument("--summary", default=None,
                    help="runner summary json; enables the G-repro gate")
    ap.add_argument("--out", default="results/experiments/probe2_trace_stats.json")
    args = ap.parse_args()

    try:
        from scipy.stats import wilcoxon
    except ImportError:
        fail("dep", "scipy is required for the Wilcoxon tests")

    print("=" * 78)
    print("TRACE PROBE-2 STATISTICS — degradation profile, no DiD, no TOST")
    print("=" * 78)
    print(f"pool {POOL} | bootstrap {N_BOOT} resamples, seed {BOOT_SEED} | "
          f"P2-9 margin +-{P2_9_MARGIN} relative (PINNED PRE-ANALYSIS)")
    ch_mrr = chance_mrr(POOL)
    print(f"chance references: MRR {ch_mrr:.5f} | R@10 {10/POOL:.5f} | "
          f"median rank {POOL/2:.0f}")

    # ---- load + G-in ----------------------------------------------------
    data = {}          # seed -> direction -> {"ranks": {cond: arr}, "ids": [...]}
    ids_ref = None
    for path in args.records:
        rows = [json.loads(l) for l in open(path, encoding="utf-8")]
        seeds = {r["mask_seed"] for r in rows}
        if len(seeds) != 1:
            fail("G-in", f"{path} mixes mask seeds {seeds}")
        seed = seeds.pop()
        if len(rows) != 2 * POOL:
            fail("G-in", f"{path}: {len(rows)} records != {2*POOL}")
        per_dir = {}
        for d in DIRECTIONS:
            rs = [r for r in rows if r["direction"] == d]
            if [r["row_idx"] for r in rs] != list(range(POOL)):
                fail("G-in", f"{path}/{d}: row_idx not contiguous 0..{POOL-1}")
            per_dir[d] = rs
        ida = [r["id"] for r in per_dir[DIRECTIONS[0]]]
        idb = [r["id"] for r in per_dir[DIRECTIONS[1]]]
        if ida != idb:
            fail("G-in", f"{path}: ids differ between directions")
        if ids_ref is None:
            ids_ref = ida
        elif ida != ids_ref:
            fail("G-in", f"{path}: ids differ from the first seed file")
        groups = np.array([r["group"] for r in per_dir[DIRECTIONS[0]]])
        n_dep = int((groups == "dependent").sum())
        if n_dep != 2005 or len(groups) - n_dep != 1:
            fail("G-in", f"{path}: groups {n_dep} dependent / "
                         f"{len(groups)-n_dep} other != 2005/1")
        entry = {}
        for d in DIRECTIONS:
            rk = {"unperturbed": np.array([r["rank_unperturbed"]
                                           for r in per_dir[d]], dtype=float)}
            for p in PERTS:
                rk[p] = np.array([r[f"rank_{p}"] for r in per_dir[d]],
                                 dtype=float)
            for cond, arr in rk.items():
                if arr.min() < 1.0 or arr.max() > POOL:
                    fail("G-range", f"{path}/{d}/{cond}: rank outside "
                                    f"[1,{POOL}] (min {arr.min()}, "
                                    f"max {arr.max()})")
            entry[d] = rk
        entry["groups"] = groups
        data[seed] = entry
        print(f"[G-in] seed {seed}: {len(rows)} records, groups 2005/1, "
              f"ranks in range — OK")
    seeds = sorted(data)
    if len(seeds) < 2:
        print("  NOTE: fewer than two seeds — no replication claim is possible.")

    # ---- G-seeds: the runs must actually differ --------------------------
    for i in range(len(seeds) - 1):
        a = data[seeds[i]]["text2ts"]["unperturbed"]
        b = data[seeds[i + 1]]["text2ts"]["unperturbed"]
        if np.array_equal(a, b):
            fail("G-seeds", f"seeds {seeds[i]} and {seeds[i+1]} have identical "
                            f"unperturbed ranks — these are not distinct runs")
    print(f"[G-seeds] the {len(seeds)} seed files are distinct runs — OK")

    # ---- G-repro against the runner's own tables -------------------------
    if args.summary:
        summ = json.load(open(args.summary, encoding="utf-8"))
        worst = 0.0
        for seed in seeds:
            tabs = summ["seeds"][str(seed)]["tables"]
            dep = data[seed]["groups"] == "dependent"
            for d in DIRECTIONS:
                for cond in ["unperturbed"] + PERTS:
                    mine = float((1.0 / data[seed][d][cond][dep]).mean())
                    theirs = tabs[f"{d}/dependent"][cond]["mrr"]
                    worst = max(worst, abs(mine - theirs))
        print(f"[G-repro] max |recomputed MRR - runner table| = {worst:.2e}")
        if worst > 1e-9:
            fail("G-repro", "recomputed MRR disagrees with the runner summary "
                            "— the record files and the summary do not match")
        print("[G-repro] PASSED (records reproduce the runner's tables)")
    else:
        print("[G-repro] SKIPPED — no --summary given; pass it to enable")

    rng = np.random.default_rng(BOOT_SEED)
    out = {"pool": POOL, "n_boot": N_BOOT, "boot_seed": BOOT_SEED,
           "p2_9_margin_relative": P2_9_MARGIN,
           "chance_mrr": ch_mrr, "inference_group": "dependent (n=2005)",
           "note": "no invariant group -> no DiD and no TOST for TRACE",
           "seeds": {}}

    # =====================================================================
    for seed in seeds:
        dep = data[seed]["groups"] == "dependent"
        n = int(dep.sum())
        print("\n" + "=" * 78)
        print(f"MASK SEED {seed} — dependent group, n = {n}")
        print("=" * 78)
        srec = {}
        boot_idx = rng.integers(0, n, size=(N_BOOT, n))   # shared across conds

        for d in DIRECTIONS:
            tag = "PRIMARY" if d == "text2ts" else "secondary"
            R = {c: data[seed][d][c][dep] for c in ["unperturbed"] + PERTS}
            inv = {c: 1.0 / R[c] for c in R}
            base = float(inv["unperturbed"].mean())
            base_boot = inv["unperturbed"][boot_idx].mean(axis=1)

            print(f"\n--- {d} ({tag}) " + "-" * 52)
            print(f"  {'cond':<12}{'MRR':>8}{'95% CI':>20}{'R@1':>7}"
                  f"{'R@10':>7}{'med':>7}{'rel deg':>10}{'CI(rel)':>18}")
            drec = {}
            for c in ["unperturbed"] + PERTS:
                m = float(inv[c].mean())
                mb = inv[c][boot_idx].mean(axis=1)
                lo, hi = np.percentile(mb, [2.5, 97.5])
                cell = {"mrr": m, "mrr_ci": [float(lo), float(hi)],
                        "recall@1": float((R[c] <= 1).mean()),
                        "recall@10": float((R[c] <= 10).mean()),
                        "median_rank": float(np.median(R[c]))}
                if c == "unperturbed":
                    rel_s, relci_s = "", ""
                else:
                    rel = (base - m) / base
                    rb = (base_boot - mb) / base_boot
                    rlo, rhi = np.percentile(rb, [2.5, 97.5])
                    cell["abs_degradation"] = base - m
                    cell["rel_degradation"] = float(rel)
                    cell["rel_ci"] = [float(rlo), float(rhi)]
                    rel_s = f"{rel:+.1%}"
                    relci_s = f"[{rlo:+.1%},{rhi:+.1%}]"
                drec[c] = cell
                print(f"  {c:<12}{cell['mrr']:>8.4f}"
                      f"{f'[{lo:.4f},{hi:.4f}]':>20}"
                      f"{cell['recall@1']:>7.3f}{cell['recall@10']:>7.3f}"
                      f"{cell['median_rank']:>7.0f}{rel_s:>10}{relci_s:>18}")

            # ---- A. paired Wilcoxon vs unperturbed, Holm over 4 ----------
            print(f"\n  A. degradation vs unperturbed (paired Wilcoxon on "
                  f"per-query rank, Holm over {len(PERTS)})")
            ps, stats_ = [], {}
            for c in PERTS:
                diff = R[c] - R["unperturbed"]
                w = wilcoxon(R[c], R["unperturbed"],
                             alternative="two-sided", zero_method="wilcox")
                worse = float((diff > 0).mean())
                ps.append(float(w.pvalue))
                stats_[c] = {"p_raw": float(w.pvalue),
                             "median_rank_shift": float(np.median(diff)),
                             "frac_queries_worse": worse}
            adj = holm(ps)
            for c, a in zip(PERTS, adj):
                stats_[c]["p_holm"] = float(a)
                print(f"    {c:<10} median rank shift "
                      f"{stats_[c]['median_rank_shift']:+9.1f} | worse in "
                      f"{stats_[c]['frac_queries_worse']:.1%} of queries | "
                      f"p_holm {fmt_p(a)}")
            drec["wilcoxon_vs_unperturbed"] = stats_

            # ---- B. the profile: six pairwise contrasts -----------------
            print(f"\n  B. profile contrasts (paired Wilcoxon, Holm over 6)")
            pairs = [(a, b) for i, a in enumerate(PERTS) for b in PERTS[i+1:]]
            pps, prec = [], {}
            for a, b in pairs:
                w = wilcoxon(R[a], R[b], alternative="two-sided",
                             zero_method="wilcox")
                dom = float((R[a] < R[b]).mean())   # a milder than b
                pps.append(float(w.pvalue))
                prec[f"{a}_vs_{b}"] = {
                    "p_raw": float(w.pvalue),
                    "frac_a_milder": dom,
                    "median_rank_diff": float(np.median(R[a] - R[b]))}
            padj = holm(pps)
            for (a, b), q in zip(pairs, padj):
                k = f"{a}_vs_{b}"
                prec[k]["p_holm"] = float(q)
                flag = "  <-- PRE-FLAGGED FRAGILE" if (a, b) == FRAGILE_PAIR \
                    or (b, a) == FRAGILE_PAIR else ""
                dom = prec[k]["frac_a_milder"]
                print(f"    {a:<8} vs {b:<8} {a} milder in {dom:>6.1%} of "
                      f"queries | median diff "
                      f"{prec[k]['median_rank_diff']:+8.1f} | p_holm "
                      f"{fmt_p(q)}{flag}")
            fp = f"{FRAGILE_PAIR[0]}_vs_{FRAGILE_PAIR[1]}"
            if fp in prec:
                dm = prec[fp]["frac_a_milder"]
                print(f"    NOTE: {FRAGILE_PAIR[0]} vs {FRAGILE_PAIR[1]} "
                      f"dominance is {dm:.1%} — with n={n} a small, consistent "
                      f"shift is significant. Report the p WITH this "
                      f"percentage, never alone.")
            drec["profile_contrasts"] = prec

            # ---- C. residual above chance under sf_all ------------------
            sf = drec["sf_all"]
            above = sf["mrr_ci"][0] > ch_mrr
            print(f"\n  C. residual above chance under sf_all: MRR "
                  f"{sf['mrr']:.5f} CI [{sf['mrr_ci'][0]:.5f},"
                  f"{sf['mrr_ci'][1]:.5f}] vs chance {ch_mrr:.5f} "
                  f"({sf['mrr']/ch_mrr:.1f}x)")
            print(f"     median rank {sf['median_rank']:.0f} vs chance "
                  f"{POOL/2:.0f} | CI excludes chance: {above}")
            if above:
                print("     => shuffling preserves each channel's value "
                      "multiset exactly, so this residual is DISTRIBUTIONAL, "
                      "not sequential. Direct input to Probe 3.")
            drec["residual_above_chance"] = {
                "chance_mrr": ch_mrr, "ratio": sf["mrr"] / ch_mrr,
                "ci_excludes_chance": bool(above)}
            srec[d] = drec

        # ---- D. P2-9 scoring, primary direction -------------------------
        rel = srec["text2ts"]["sf_all"]["rel_degradation"]
        lo = srec["text2ts"]["sf_all"]["rel_ci"][0]
        hit = rel > P2_9_MARGIN
        print(f"\n  D. P2-9 (seed {seed}): sf_all relative degradation "
              f"{rel:+.1%}, CI lower {lo:+.1%}, margin {P2_9_MARGIN:+.0%} "
              f"-> {'exceeds' if hit else 'DOES NOT EXCEED'}")
        srec["p2_9_seed_pass"] = bool(hit)
        out["seeds"][str(seed)] = srec

    # =====================================================================
    print("\n" + "=" * 78)
    print("AGGREGATE ACROSS SEEDS — dependent group, text->ts (PRIMARY)")
    print("=" * 78)
    print(f"  {'cond':<12}{'MRR mean+-sd':>22}{'rel deg mean+-sd':>24}")
    agg = {}
    for c in ["unperturbed"] + PERTS:
        ms = np.array([out["seeds"][str(s)]["text2ts"][c]["mrr"] for s in seeds])
        line = f"  {c:<12}{f'{ms.mean():.4f} +- {ms.std(ddof=1):.4f}':>22}"
        cell = {"mrr_mean": float(ms.mean()), "mrr_sd": float(ms.std(ddof=1))}
        if c != "unperturbed":
            rs = np.array([out["seeds"][str(s)]["text2ts"][c]["rel_degradation"]
                           for s in seeds])
            line += f"{f'{rs.mean():+.1%} +- {rs.std(ddof=1):.1%}':>24}"
            cell.update({"rel_mean": float(rs.mean()),
                         "rel_sd": float(rs.std(ddof=1))})
        agg[c] = cell
        print(line)

    order = sorted(PERTS, key=lambda p: -agg[p]["rel_mean"])
    per_seed_orders = {s: sorted(PERTS,
                                 key=lambda p: -out["seeds"][str(s)]["text2ts"][p]["rel_degradation"])
                       for s in seeds}
    stable = all(per_seed_orders[s] == order for s in seeds)
    print(f"\n  severity ordering (most to least damaging): "
          f"{' > '.join(order)}")
    print(f"  identical in every seed: {stable}")
    for s in seeds:
        print(f"    seed {s}: {' > '.join(per_seed_orders[s])}")

    all_pass = all(out["seeds"][str(s)]["p2_9_seed_pass"] for s in seeds)
    verdict = "CONFIRMED" if all_pass else "MISSED"
    print(f"\n  P2-9 (registered 2026-08-10): sf_all degradation exceeds "
          f"+-{P2_9_MARGIN} relative in all {len(seeds)} seeds")
    print(f"  => P2-9 {verdict}")
    resid = all(out["seeds"][str(s)]["text2ts"]["residual_above_chance"]
                ["ci_excludes_chance"] for s in seeds)
    print(f"  residual-above-chance under sf_all in every seed: {resid}")

    # ambiguous row, descriptive only
    print(f"\n  ambiguous row 1191 (n=1 — DESCRIPTIVE ONLY, never quotable):")
    for s in seeds:
        amb = data[s]["groups"] != "dependent"
        r = {c: float(data[s]["text2ts"][c][amb][0])
             for c in ["unperturbed"] + PERTS}
        print(f"    seed {s}: unpert rank {r['unperturbed']:.0f} -> " +
              ", ".join(f"{p} {r[p]:.0f}" for p in PERTS))
    out["aggregate"] = {"text2ts_dependent": agg, "severity_order": order,
                        "order_stable_across_seeds": bool(stable),
                        "p2_9_verdict": verdict,
                        "residual_above_chance_all_seeds": bool(resid)}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    print("Reminder: the ex_half-vs-masking contrast is pre-flagged fragile; "
          "quote its dominance percentage alongside any p-value.")


if __name__ == "__main__":
    main()

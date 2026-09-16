#!/usr/bin/env python3
"""
analyze_probe3.py — CLaSP Probe-3 statistics and prediction scoring.

Scores P3-1 (gaussian anchor), P3-2b/c (resample ~ shuffle TOST),
P3-4 (resample above chance), P3-5 (spike-vs-smooth pinned contrast),
and reports Comparison B (resample vs gaussian gap) per substrate.

Design record (accepted in-chat 2026-08-15, doc blocks to follow):
  - Chance MRR references (D2 average rank, uniform position):
      global pool 386 : H(386)/386 = 0.016928
      length-split refs for the ANCHOR INVESTIGATION only:
      TRUCE 246       : H(246)/246 = 0.024734
      SUSHI 140       : H(140)/140 = 0.039446
  - Bootstrap resamples SIGNALS (binding): clusters keyed by gt
    sample_id; B=10000, rng seed 20260815, percentile intervals.
  - P3-1 scored as registered: overall gaussian ratio-to-chance 95% CI
    must include global chance in every checkpoint. AMENDMENT (accepted
    2026-08-15): a miss triggers the pre-specified LENGTH INVESTIGATION
    (records-only: per-substrate gaussian MRR vs the three references,
    median ranks, fraction of ranks <= 246 for TRUCE queries) and
    downstream numbers are labelled 'anchor-conditional' rather than
    the run being declared broken. The registered 'a miss is never a
    finding' clause was overconfident; the pre-named candidate
    mechanism is the surviving LENGTH channel (both substrates' lengths
    are preserved by every surrogate by design).
  - P3-2 TOST: difference in relative degradation, resample minus
    sf_all (from the committed Probe-2 records, joined per caption_id),
    margin +-0.05, 90% bootstrap CI inside [-0.05,+0.05] = pass.
    PRIMARY population per substrate = the DEPENDENT group (mirrors the
    P2-4 inference-cell precedent; invariant cells are 4 and 18 rows —
    thin, reported with n, never load-bearing). Whole-substrate rows
    reported alongside. P3-2b = SUSHI dep, P3-2c = TRUCE dep (the
    registered coin-flip).
  - P3-5 (pin accepted; scoring formalised BEFORE this script ran, with
    the sequencing disclosure that the runner's descriptive fluct table
    was visible first — the statistic is exactly the pre-named
    candidate, no selection): per-signal retention = rank_unpert /
    rank_resample on SUSHI (1 query per signal); groups spike
    {negative spike, positive spike, positive-and-negative spike}
    (n=60) vs cs {clean, smooth} minus the degenerate constant (n=39);
    one-sided Mann-Whitney U (spike > cs) per checkpoint + bootstrap CI
    on the mean-retention difference. VERDICT RULE: CONFIRMED iff the
    direction (mean spike retention > mean cs retention) holds in ALL
    THREE checkpoints; per-seed p reported as strength, not gatekeeper
    (directional registration; noise floor measured pre-run at ~+-0.08
    MRR-units for this contrast). noisy/step cells reported
    descriptively, unregistered.
  - Join gate JG: probe3 and probe2 records join losslessly on
    caption_id (878 rows/seed) AND rank_unperturbed agrees <= 1e-9 —
    HARD STOP otherwise (same checkpoints, same pool: disagreement
    means the runs are not comparable).
  - Wilcoxon (perturbed vs unperturbed reciprocal ranks, paired) per
    dependent cell and condition, Holm-corrected within each seed's
    reported family.

Usage (PowerShell):
    python -m models.clasp.analyze_probe3 `
        --probe3-records results/experiments/probe3_clasp_per_query_seed42.jsonl `
                         results/experiments/probe3_clasp_per_query_seed43.jsonl `
                         results/experiments/probe3_clasp_per_query_seed44.jsonl `
        --probe2-records results/experiments/probe2_clasp_per_query_seed42.jsonl `
                         results/experiments/probe2_clasp_per_query_seed43.jsonl `
                         results/experiments/probe2_clasp_per_query_seed44.jsonl `
        --out results/experiments/probe3_clasp_stats.json
"""

from __future__ import annotations
import argparse
import json
import sys
from collections import defaultdict

import numpy as np
from scipy import stats as sps

CH_GLOBAL = 0.016928   # H(386)/386
CH_TRUCE_SPLIT = 0.024734  # H(246)/246 — investigation reference only
CH_SUSHI_SPLIT = 0.039446  # H(140)/140 — investigation reference only
B_BOOT = 10000
BOOT_SEED = 20260815
TOST_MARGIN = 0.05
SPIKE = {"negative spike", "positive spike", "positive-and-negative spike"}
CS = {"clean", "smooth"}
CONST_GT = "sushi:clean\\00\\0000009"


def fail(msg):
    print(f"\nGATE FAILED: {msg}", file=sys.stderr)
    sys.exit(1)


def load_jsonl(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out[r["caption_id"]] = r
    return out


def cluster_ids(recs):
    """signal-level clusters: gt -> list of caption_ids"""
    cl = defaultdict(list)
    for cid, r in recs.items():
        cl[r["gt"]].append(cid)
    return cl


def boot_stat(recs, ids_by_sig, fn, rng):
    """Bootstrap over signals; fn(list_of_records) -> float."""
    sigs = sorted(ids_by_sig)
    vals = []
    for _ in range(B_BOOT):
        pick = rng.integers(0, len(sigs), size=len(sigs))
        sample = []
        for s in pick:
            sample.extend(recs[c] for c in ids_by_sig[sigs[s]])
        vals.append(fn(sample))
    return np.array(vals)


def mrr(rows, key):
    return float(np.mean([1.0 / r[key] for r in rows]))


def rel_deg(rows, key):
    u = mrr(rows, "rank_unperturbed")
    return 1.0 - mrr(rows, key) / u if u > 0 else float("nan")


def ci(v, lo=2.5, hi=97.5):
    return [float(np.percentile(v, lo)), float(np.percentile(v, hi))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe3-records", nargs=3, required=True)
    ap.add_argument("--probe2-records", nargs=3, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = {"chance": {"global": CH_GLOBAL, "truce_split": CH_TRUCE_SPLIT,
                      "sushi_split": CH_SUSHI_SPLIT},
           "boot": {"B": B_BOOT, "seed": BOOT_SEED, "unit": "signals"},
           "tost_margin": TOST_MARGIN, "seeds": {}}
    verdicts = {"P3-1": [], "P3-2b": [], "P3-2c": [], "P3-4": [],
                "P3-5_direction": []}

    for p3_path, p2_path in zip(args.probe3_records, args.probe2_records):
        seed = "".join(ch for ch in p3_path if ch.isdigit())[-2:]
        print(f"\n===== checkpoint seed {seed} =====")
        p3 = load_jsonl(p3_path)
        p2 = load_jsonl(p2_path)

        # ---- JG: lossless join + identical unperturbed ranks ----
        if len(p3) != 878 or len(p2) != 878:
            fail(f"JG: row counts {len(p3)}/{len(p2)} != 878")
        if set(p3) != set(p2):
            fail("JG: caption_id sets differ between probe3 and probe2")
        max_du = max(abs(p3[c]["rank_unperturbed"] - p2[c]["rank_unperturbed"])
                     for c in p3)
        print(f"JG: join lossless (878); max |rank_unpert diff| = {max_du:.2e}")
        if max_du > 1e-9:
            fail(f"JG: unperturbed ranks differ ({max_du:.2e}) — runs not "
                 f"comparable")

        # merge sf_all rank into the probe3 record view
        for c in p3:
            p3[c]["rank_sf_all"] = p2[c]["rank_sf_all"]

        rng = np.random.default_rng(BOOT_SEED)
        cl_all = cluster_ids(p3)

        def subsel(pred):
            recs = {c: r for c, r in p3.items() if pred(r)}
            return recs, cluster_ids(recs)

        cells = {
            "overall": lambda r: True,
            "sushi": lambda r: r["substrate"] == "sushi",
            "truce": lambda r: r["substrate"] == "truce",
            "sushi_dep": lambda r: r["substrate"] == "sushi"
                                   and r["group"] == "dependent",
            "truce_dep": lambda r: r["substrate"] == "truce"
                                   and r["group"] == "dependent",
        }
        srec = {"cells": {}}

        wil_family = []  # (label, pvalue)
        for name, pred in cells.items():
            recs, cl = subsel(pred)
            rows = list(recs.values())
            row_out = {"n": len(rows)}
            for cond in ("resample", "gaussian", "sf_all"):
                key = f"rank_{cond}"
                m = mrr(rows, key)
                bv = boot_stat(recs, cl, lambda rr, k=key: mrr(rr, k), rng)
                rd = rel_deg(rows, key)
                bd = boot_stat(recs, cl, lambda rr, k=key: rel_deg(rr, k), rng)
                row_out[cond] = {
                    "mrr": m, "mrr_ci95": ci(bv),
                    "ratio_to_chance": m / CH_GLOBAL,
                    "ratio_ci95": [x / CH_GLOBAL for x in ci(bv)],
                    "rel_deg": rd, "rel_deg_ci95": ci(bd),
                }
                if name.endswith("_dep") and cond in ("resample", "gaussian"):
                    ru = np.array([1.0 / r["rank_unperturbed"] for r in rows])
                    rc = np.array([1.0 / r[key] for r in rows])
                    if np.any(ru != rc):
                        w = sps.wilcoxon(ru, rc, alternative="greater")
                        wil_family.append((f"{name}/{cond}", float(w.pvalue)))
                row_out["unperturbed_mrr"] = mrr(rows, "rank_unperturbed")
            srec["cells"][name] = row_out
            print(f"  {name:10s} n={len(rows):3d}  unpert "
                  f"{row_out['unperturbed_mrr']:.4f} | resample "
                  f"{row_out['resample']['mrr']:.4f} "
                  f"(ratio {row_out['resample']['ratio_to_chance']:.2f}x, "
                  f"CI {row_out['resample']['ratio_ci95'][0]:.2f}-"
                  f"{row_out['resample']['ratio_ci95'][1]:.2f}) | gaussian "
                  f"{row_out['gaussian']['mrr']:.4f} "
                  f"(ratio {row_out['gaussian']['ratio_to_chance']:.2f}x, "
                  f"CI {row_out['gaussian']['ratio_ci95'][0]:.2f}-"
                  f"{row_out['gaussian']['ratio_ci95'][1]:.2f})")

        # Holm over the wilcoxon family
        if wil_family:
            labels, ps = zip(*wil_family)
            order = np.argsort(ps)
            m_ = len(ps)
            holm = {}
            for rank_i, oi in enumerate(order):
                holm[labels[oi]] = min(1.0, ps[oi] * (m_ - rank_i))
            srec["wilcoxon_holm"] = holm
            print("  Wilcoxon (paired, greater), Holm: "
                  + ", ".join(f"{k} p={v:.2e}" for k, v in holm.items()))

        # ---- P3-1: anchor ----
        g = srec["cells"]["overall"]["gaussian"]
        p31_pass = g["ratio_ci95"][0] <= 1.0 <= g["ratio_ci95"][1]
        verdicts["P3-1"].append(p31_pass)
        print(f"  P3-1 anchor: overall gaussian ratio "
              f"{g['ratio_to_chance']:.3f}x CI "
              f"[{g['ratio_ci95'][0]:.3f}, {g['ratio_ci95'][1]:.3f}] "
              f"-> {'includes' if p31_pass else 'EXCLUDES'} chance")

        # ---- anchor investigation (pre-specified; runs on a miss) ----
        if not p31_pass:
            inv = {}
            for sub, ref in (("truce", CH_TRUCE_SPLIT),
                             ("sushi", CH_SUSHI_SPLIT)):
                rows = [r for r in p3.values() if r["substrate"] == sub]
                gm = mrr(rows, "rank_gaussian")
                med = float(np.median([r["rank_gaussian"] for r in rows]))
                inv[sub] = {"gaussian_mrr": gm, "global_ref": CH_GLOBAL,
                            "length_split_ref": ref, "median_rank": med}
                if sub == "truce":
                    fr = float(np.mean([r["rank_gaussian"] <= 246
                                        for r in rows]))
                    inv[sub]["frac_rank_le_246"] = fr
                    inv[sub]["frac_expected_if_no_split"] = 246 / 386
                print(f"  INVESTIGATION [{sub}]: gaussian MRR {gm:.4f} vs "
                      f"global {CH_GLOBAL:.4f} vs length-split {ref:.4f}; "
                      f"median rank {med:.1f}"
                      + (f"; frac(rank<=246) {inv[sub]['frac_rank_le_246']:.3f}"
                         f" vs no-split {246/386:.3f}" if sub == "truce" else ""))
            srec["anchor_investigation"] = inv

        # ---- P3-4: resample above chance ----
        r_ov = srec["cells"]["overall"]["resample"]
        p34_pass = r_ov["ratio_ci95"][0] > 1.0
        verdicts["P3-4"].append(p34_pass)
        print(f"  P3-4: overall resample ratio CI lower bound "
              f"{r_ov['ratio_ci95'][0]:.3f} -> "
              f"{'above' if p34_pass else 'NOT above'} chance")

        # ---- P3-2: TOST resample vs sf_all, dependent groups ----
        for cell, tag in (("sushi_dep", "P3-2b"), ("truce_dep", "P3-2c")):
            recs, cl = subsel(cells[cell])
            fn = lambda rr: rel_deg(rr, "rank_resample") - \
                            rel_deg(rr, "rank_sf_all")
            bd = boot_stat(recs, cl, fn, rng)
            point = fn(list(recs.values()))
            lo, hi = ci(bd, 5, 95)  # 90% CI for TOST
            passed = (-TOST_MARGIN <= lo) and (hi <= TOST_MARGIN)
            verdicts[tag].append(passed)
            srec[f"tost_{cell}"] = {"point": point, "ci90": [lo, hi],
                                    "pass": passed}
            print(f"  {tag} TOST [{cell}]: d(rel_deg) = {point:+.4f}, "
                  f"90% CI [{lo:+.4f}, {hi:+.4f}] vs +-{TOST_MARGIN} -> "
                  f"{'PASS' if passed else 'FAIL'}")

        # ---- P3-5: pinned contrast ----
        sushi = [r for r in p3.values() if r["substrate"] == "sushi"
                 and r["gt"] != CONST_GT]
        spike = [r for r in sushi if r["fluct"] in SPIKE]
        cs_g = [r for r in sushi if r["fluct"] in CS]
        if len(spike) != 60 or len(cs_g) != 39:
            fail(f"P3-5 group sizes {len(spike)}/{len(cs_g)} != 60/39")
        ret = lambda r: r["rank_unperturbed"] / r["rank_resample"]
        rs, rc = [ret(r) for r in spike], [ret(r) for r in cs_g]
        d_mean = float(np.mean(rs) - np.mean(rc))
        mw = sps.mannwhitneyu(rs, rc, alternative="greater")
        # bootstrap CI on the mean difference (signals = queries here, 1:1)
        rngl = np.random.default_rng(BOOT_SEED + 1)
        bd = [np.mean(rngl.choice(rs, len(rs))) -
              np.mean(rngl.choice(rc, len(rc))) for _ in range(B_BOOT)]
        direction = d_mean > 0
        verdicts["P3-5_direction"].append(direction)
        srec["p3_5"] = {"n_spike": len(rs), "n_cs": len(cs_g),
                        "mean_retention_spike": float(np.mean(rs)),
                        "mean_retention_cs": float(np.mean(rc)),
                        "median_retention_spike": float(np.median(rs)),
                        "median_retention_cs": float(np.median(rc)),
                        "mean_diff": d_mean, "mean_diff_ci95": ci(np.array(bd)),
                        "mw_p_one_sided": float(mw.pvalue),
                        "direction_spike_gt_cs": direction}
        print(f"  P3-5: retention spike {np.mean(rs):.4f} (med "
              f"{np.median(rs):.4f}) vs cs {np.mean(rc):.4f} (med "
              f"{np.median(rc):.4f}); diff {d_mean:+.4f} CI "
              f"[{srec['p3_5']['mean_diff_ci95'][0]:+.4f}, "
              f"{srec['p3_5']['mean_diff_ci95'][1]:+.4f}]; MW one-sided "
              f"p={mw.pvalue:.3f} -> direction "
              f"{'spike>cs' if direction else 'cs>=spike (REVERSED)'}")
        # descriptive, unregistered
        for fl in ("noisy", "step"):
            rows = [ret(r) for r in sushi if r["fluct"] == fl]
            print(f"    (descriptive, unregistered) {fl}: mean retention "
                  f"{np.mean(rows):.4f} (n={len(rows)})")

        # ---- Comparison B: headline gap per substrate ----
        for sub in ("sushi", "truce"):
            recs, cl = subsel(cells[sub])
            fn = lambda rr: mrr(rr, "rank_resample") - mrr(rr, "rank_gaussian")
            bd = boot_stat(recs, cl, fn, rng)
            point = fn(list(recs.values()))
            srec[f"compB_{sub}"] = {"mrr_gap": point, "ci95": ci(bd)}
            print(f"  Comparison B [{sub}]: resample - gaussian MRR = "
                  f"{point:+.4f} CI [{ci(bd)[0]:+.4f}, {ci(bd)[1]:+.4f}]")

        out["seeds"][seed] = srec

    # ---- verdicts ----
    print("\n===== PREDICTION SCORING =====")
    fin = {}
    fin["P3-1"] = "CONFIRMED" if all(verdicts["P3-1"]) else "MISSED"
    fin["P3-4"] = "CONFIRMED" if all(verdicts["P3-4"]) else "MISSED"
    fin["P3-2b"] = "CONFIRMED" if all(verdicts["P3-2b"]) else "MISSED"
    fin["P3-2c"] = "CONFIRMED" if all(verdicts["P3-2c"]) else "MISSED"
    fin["P3-5"] = ("CONFIRMED" if all(verdicts["P3-5_direction"])
                   else "MISSED")
    for k, v in fin.items():
        per = verdicts.get(k if k != "P3-5" else "P3-5_direction")
        print(f"  {k}: {v}  (per-seed: {per})")
    out["verdicts"] = fin

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
analyze_probe3_trace.py — formal statistics for the TRACE Probe-3 arm.

Consumes the per-query records from models/trace/run_probe3.py plus the
COMMITTED Probe-2 records (the sf_all rung is read, never rerun) and scores
P3-2a, P3-3, P3-6. Computes nothing from the model; every input is a
stored rank.

THE LADDER (dep group n=2,005; text->ts PRIMARY, ts->text secondary):
  unperturbed -> sf_all (committed Probe-2; exact multiset, no order)
              -> resample (distribution shape, no multiset, no order)
              -> gaussian (length + first two moments only)
Chance references pinned from the setup diagnostic (MEASURED sizes):
  global H(2006)/2006 = 0.0040784; ceilings (perfect length matcher,
  uniform within stratum) H(1051)/1051 = 0.0071695 for V=168 and
  H(544)/544 = 0.0126417 for V=180; the 'other' stratum gets a composite
  per-row ceiling mean(H(n_V)/n_V).

SCORING (pinned before this script ever ran; delivered pre-output):
  P3-2a  TOST on d(rel_deg) = rel(resample) - rel(sf_all), dep, text2ts,
         90% bootstrap CI within +-0.05, pass in ALL three seeds.
  P3-3   BOTH required in all seeds: resample MRR 95% CI excludes global
         chance (survival) AND point ratio-to-chance in [2.0, 3.5] (the
         registered band).
  P3-6   formalised from the register's one-liner (disclosed in-chat,
         veto offered): between-stratum (V=168 minus V=180) difference in
         RESAMPLE relative degradation, 95% bootstrap CI includes 0 in
         all three seeds. Gaussian between-stratum reported, UNREGISTERED.
  Seed-44 arbitration: the sign/CI of d(rel_deg) at n=2,005, labelled —
         CLaSP's seed 44 had resample significantly MILDER than shuffle.
  Anchor investigation (unconditional): per-stratum gaussian positioning
         between global chance and ceiling; frac(rank <= stratum size)
         vs uniform expectation n_s/2006 vs perfect split 1.0.

RESAMPLING UNIT: rows (queries), B=2000, seed 20260815, shared index
matrix per seed across conditions (paired), percentile intervals. The
pool-side dependence is not modelled (no row scheme can; pool is shared)
— stated limitation, as in Probe 2.

GATES
  G-in    4,012 records per seed; row_idx contiguous both directions;
          ids identical across directions and seeds; groups 2,005/1;
          strata 1,051/544/411 — HARD
  G-range every rank in [1, 2006] — HARD
  JG      rank_unperturbed identical (<=1e-9) to the committed Probe-2
          records, every row, both directions, every seed — HARD
  G-repro recomputed MRR == runner summary tables <=1e-9 — HARD
  G-seeds seed files are distinct runs — HARD

USAGE (repo root; PowerShell continuation):
  python scripts/analyze_probe3_trace.py `
      --records results/experiments/probe3_trace_per_query_seed13.jsonl `
                results/experiments/probe3_trace_per_query_seed14.jsonl `
                results/experiments/probe3_trace_per_query_seed15.jsonl `
      --probe2-records results/experiments/probe2_trace_per_query_seed13.jsonl `
                       results/experiments/probe2_trace_per_query_seed14.jsonl `
                       results/experiments/probe2_trace_per_query_seed15.jsonl `
      --summary results/experiments/probe3_trace_summary.json `
      --out results/experiments/probe3_trace_stats.json

Runtime: about a minute. Paste the entire console output back.
"""

from __future__ import annotations
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

CONDS = ["sf_all", "resample", "gaussian"]        # ladder below unperturbed
DIRECTIONS = ["text2ts", "ts2text"]
POOL = 2006
N_BOOT = 2000
BOOT_SEED = 20260815
TOST_MARGIN = 0.05          # P3-2a; the pinned Probe-2 extension
P3_3_BAND = (2.0, 3.5)      # registered ratio band
MAIN_STRATA = [168, 180]
EXPECTED_STRATA = {168: 1051, 180: 544}   # all-row sizes (diagnostic A3/A4)
AMBIGUOUS_ROW = 1191


def fail(gate, msg):
    print(f"\n[GATE FAILED] {gate}: {msg}")
    print("Hard stop. Investigate before proceeding — do not work around.")
    sys.exit(1)


def holm(pvals):
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
    return "<1e-300" if p == 0.0 else f"{p:.3e}"


def chance_mrr(n):
    return float(np.sum(1.0 / np.arange(1, n + 1)) / n)


def load_records(paths, conds, gate_prefix):
    """Load per-query jsonl files -> {seed: {dir: {cond: ranks}, groups,
    vlen, ids}} with the parent's input gates."""
    data, ids_ref = {}, None
    for path in paths:
        rows = [json.loads(l) for l in open(path, encoding="utf-8")]
        seeds = {r["mask_seed"] for r in rows}
        if len(seeds) != 1:
            fail(gate_prefix, f"{path} mixes mask seeds {seeds}")
        seed = seeds.pop()
        if len(rows) != 2 * POOL:
            fail(gate_prefix, f"{path}: {len(rows)} records != {2*POOL}")
        per_dir = {}
        for d in DIRECTIONS:
            rs = [r for r in rows if r["direction"] == d]
            if [r["row_idx"] for r in rs] != list(range(POOL)):
                fail(gate_prefix, f"{path}/{d}: row_idx not contiguous")
            per_dir[d] = rs
        ida = [r["id"] for r in per_dir[DIRECTIONS[0]]]
        idb = [r["id"] for r in per_dir[DIRECTIONS[1]]]
        if ida != idb:
            fail(gate_prefix, f"{path}: ids differ between directions")
        if ids_ref is None:
            ids_ref = ida
        elif ida != ids_ref:
            fail(gate_prefix, f"{path}: ids differ from the first file")
        groups = np.array([r["group"] for r in per_dir[DIRECTIONS[0]]])
        if int((groups == "dependent").sum()) != 2005:
            fail(gate_prefix, f"{path}: dependent group != 2005")
        vlen = np.array([r["valid_len"] for r in per_dir[DIRECTIONS[0]]],
                        dtype=int)
        entry = {"groups": groups, "vlen": vlen}
        for d in DIRECTIONS:
            rk = {"unperturbed": np.array([r["rank_unperturbed"]
                                           for r in per_dir[d]], float)}
            for c in conds:
                rk[c] = np.array([r[f"rank_{c}"] for r in per_dir[d]], float)
            for cond, arr in rk.items():
                if arr.min() < 1.0 or arr.max() > POOL:
                    fail("G-range", f"{path}/{d}/{cond}: rank outside "
                                    f"[1,{POOL}]")
            entry[d] = rk
        data[seed] = entry
        print(f"[{gate_prefix}] seed {seed}: {len(rows)} records, groups "
              f"2005/1, ranks in range — OK")
    return data, ids_ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", nargs="+", required=True)
    ap.add_argument("--probe2-records", nargs="+", required=True)
    ap.add_argument("--summary", default=None)
    ap.add_argument("--out",
                    default="results/experiments/probe3_trace_stats.json")
    args = ap.parse_args()

    try:
        from scipy.stats import wilcoxon
    except ImportError:
        fail("dep", "scipy is required for the Wilcoxon tests")

    print("=" * 78)
    print("TRACE PROBE-3 STATISTICS — the ladder: sf_all / resample / "
          "gaussian")
    print("=" * 78)
    ch = chance_mrr(POOL)
    print(f"pool {POOL} | bootstrap {N_BOOT}, seed {BOOT_SEED} | TOST "
          f"margin +-{TOST_MARGIN} | P3-3 band {P3_3_BAND[0]}-{P3_3_BAND[1]}x"
          f" | global chance MRR {ch:.5f}")

    p3, ids3 = load_records(args.records, ["resample", "gaussian"], "G-in")
    p2, ids2 = load_records(args.probe2_records,
                            ["sf_all", "sf_half", "ex_half", "masking"],
                            "G-in-P2")
    if ids3 != ids2:
        fail("JG-ids", "Probe-3 and Probe-2 record ids differ")
    seeds = sorted(p3)
    if sorted(p2) != seeds:
        fail("JG-seeds", f"seed sets differ: {seeds} vs {sorted(p2)}")

    # ---- JG: pairing against the committed Probe-2 arm -------------------
    for s in seeds:
        if not np.array_equal(p3[s]["vlen"], p2[s]["vlen"]):
            fail("JG-vlen", f"seed {s}: valid_len differs")
        for d in DIRECTIONS:
            jg = float(np.abs(p3[s][d]["unperturbed"]
                              - p2[s][d]["unperturbed"]).max())
            if jg > 1e-9:
                fail("JG", f"seed {s}/{d}: unperturbed ranks differ from "
                           f"the committed Probe-2 records (max {jg:.3e})")
        print(f"[JG] seed {s}: rank_unperturbed identical to committed "
              f"Probe-2, both directions — PASSED")
        # graft the sf_all rung into the probe3 entry
        for d in DIRECTIONS:
            p3[s][d]["sf_all"] = p2[s][d]["sf_all"]

    # ---- strata (from records; gated against the diagnostic facts) -------
    vlen = p3[seeds[0]]["vlen"]
    vc = Counter(vlen.tolist())
    for v, n_exp in EXPECTED_STRATA.items():
        if vc[v] != n_exp:
            fail("G-in-strata", f"V={v}: {vc[v]} rows != {n_exp}")
    n_other = POOL - sum(EXPECTED_STRATA.values())
    print(f"[G-in-strata] V=168 n=1051, V=180 n=544, other n={n_other} — OK")
    ceil168 = chance_mrr(vc[168])
    ceil180 = chance_mrr(vc[180])
    other_rows = ~np.isin(vlen, MAIN_STRATA)
    ceil_other = float(np.mean([chance_mrr(vc[int(v)])
                                for v in vlen[other_rows]]))
    print(f"  ceilings: V=168 {ceil168:.5f} | V=180 {ceil180:.5f} | "
          f"other composite {ceil_other:.5f} (per-row mean)")

    # ---- G-seeds ----------------------------------------------------------
    for i in range(len(seeds) - 1):
        if np.array_equal(p3[seeds[i]]["text2ts"]["unperturbed"],
                          p3[seeds[i + 1]]["text2ts"]["unperturbed"]):
            fail("G-seeds", f"seeds {seeds[i]}/{seeds[i+1]} identical")
    print(f"[G-seeds] the {len(seeds)} seed files are distinct runs — OK")

    # ---- G-repro ----------------------------------------------------------
    if args.summary:
        summ = json.load(open(args.summary, encoding="utf-8"))
        worst = 0.0
        for s in seeds:
            tabs = summ["seeds"][str(s)]["tables"]
            dep = p3[s]["groups"] == "dependent"
            for d in DIRECTIONS:
                for cond in ["unperturbed", "resample", "gaussian"]:
                    mine = float((1.0 / p3[s][d][cond][dep]).mean())
                    theirs = tabs[f"{d}/dependent"][cond]["mrr"]
                    worst = max(worst, abs(mine - theirs))
        print(f"[G-repro] max |recomputed MRR - runner table| = {worst:.2e}")
        if worst > 1e-9:
            fail("G-repro", "records do not reproduce the runner summary")
        print("[G-repro] PASSED")
    else:
        print("[G-repro] SKIPPED — no --summary given")

    rng = np.random.default_rng(BOOT_SEED)
    out = {"pool": POOL, "n_boot": N_BOOT, "boot_seed": BOOT_SEED,
           "tost_margin": TOST_MARGIN, "p3_3_band": list(P3_3_BAND),
           "chance_mrr": ch,
           "ceilings": {"V168": ceil168, "V180": ceil180,
                        "other_composite": ceil_other},
           "inference_group": "dependent (n=2005)", "seeds": {}}
    LADDER = ["unperturbed"] + CONDS

    for s in seeds:
        dep = p3[s]["groups"] == "dependent"
        n = int(dep.sum())
        strata = {"V=168": dep & (vlen == 168),
                  "V=180": dep & (vlen == 180),
                  "other": dep & other_rows}
        print("\n" + "=" * 78)
        print(f"MASK SEED {s} — dependent group, n = {n}")
        print("=" * 78)
        srec = {}
        boot_idx = rng.integers(0, n, size=(N_BOOT, n))   # shared, paired

        for d in DIRECTIONS:
            tag = "PRIMARY" if d == "text2ts" else "secondary"
            R = {c: p3[s][d][c][dep] for c in LADDER}
            inv = {c: 1.0 / R[c] for c in R}
            base = float(inv["unperturbed"].mean())
            base_boot = inv["unperturbed"][boot_idx].mean(axis=1)

            print(f"\n--- {d} ({tag}) " + "-" * 52)
            print(f"  {'cond':<12}{'MRR':>8}{'95% CI':>20}{'R@10':>7}"
                  f"{'med':>7}{'x chance':>9}{'rel deg':>9}{'CI(rel)':>18}")
            drec = {}
            boot_mb = {}
            for c in LADDER:
                m = float(inv[c].mean())
                mb = inv[c][boot_idx].mean(axis=1)
                boot_mb[c] = mb
                lo, hi = np.percentile(mb, [2.5, 97.5])
                cell = {"mrr": m, "mrr_ci": [float(lo), float(hi)],
                        "recall@10": float((R[c] <= 10).mean()),
                        "median_rank": float(np.median(R[c])),
                        "ratio_to_chance": m / ch,
                        "ratio_ci": [float(lo / ch), float(hi / ch)]}
                if c == "unperturbed":
                    rel_s, relci_s = "", ""
                else:
                    rel = (base - m) / base
                    rb = (base_boot - mb) / base_boot
                    rlo, rhi = np.percentile(rb, [2.5, 97.5])
                    cell["rel_degradation"] = float(rel)
                    cell["rel_ci"] = [float(rlo), float(rhi)]
                    rel_s = f"{rel:+.1%}"
                    relci_s = f"[{rlo:+.1%},{rhi:+.1%}]"
                drec[c] = cell
                print(f"  {c:<12}{m:>8.4f}{f'[{lo:.4f},{hi:.4f}]':>20}"
                      f"{cell['recall@10']:>7.3f}{cell['median_rank']:>7.0f}"
                      f"{m/ch:>9.2f}{rel_s:>9}{relci_s:>18}")

            # A. new conditions vs unperturbed (Holm over 2)
            print("\n  A. degradation vs unperturbed (paired Wilcoxon, "
                  "Holm over 2 new conditions)")
            ps, st = [], {}
            for c in ["resample", "gaussian"]:
                diff = R[c] - R["unperturbed"]
                w = wilcoxon(R[c], R["unperturbed"],
                             alternative="two-sided", zero_method="wilcox")
                ps.append(float(w.pvalue))
                st[c] = {"p_raw": float(w.pvalue),
                         "median_rank_shift": float(np.median(diff)),
                         "frac_queries_worse": float((diff > 0).mean())}
            for c, a in zip(["resample", "gaussian"], holm(ps)):
                st[c]["p_holm"] = float(a)
                print(f"    {c:<10} median rank shift "
                      f"{st[c]['median_rank_shift']:+9.1f} | worse in "
                      f"{st[c]['frac_queries_worse']:.1%} | p_holm "
                      f"{fmt_p(a)}")
            drec["wilcoxon_vs_unperturbed"] = st

            # B. ladder contrasts (Holm over 3)
            print("\n  B. ladder contrasts (paired Wilcoxon, Holm over 3)")
            pairs = [("resample", "sf_all"), ("resample", "gaussian"),
                     ("sf_all", "gaussian")]
            pps, prec = [], {}
            for a, b in pairs:
                w = wilcoxon(R[a], R[b], alternative="two-sided",
                             zero_method="wilcox")
                pps.append(float(w.pvalue))
                prec[f"{a}_vs_{b}"] = {
                    "p_raw": float(w.pvalue),
                    "frac_a_milder": float((R[a] < R[b]).mean()),
                    "median_rank_diff": float(np.median(R[a] - R[b]))}
            for (a, b), q in zip(pairs, holm(pps)):
                k = f"{a}_vs_{b}"
                prec[k]["p_holm"] = float(q)
                print(f"    {a:<9} vs {b:<9} {a} milder in "
                      f"{prec[k]['frac_a_milder']:>6.1%} | median diff "
                      f"{prec[k]['median_rank_diff']:+8.1f} | p_holm "
                      f"{fmt_p(q)}")
            drec["ladder_contrasts"] = prec

            # C. P3-2a TOST + arbitration read
            rel_res = drec["resample"]["rel_degradation"]
            rel_sf = drec["sf_all"]["rel_degradation"]
            b_res = (base_boot - boot_mb["resample"]) / base_boot
            b_sf = (base_boot - boot_mb["sf_all"]) / base_boot
            dboot = b_res - b_sf
            lo90, hi90 = np.percentile(dboot, [5, 95])
            lo95, hi95 = np.percentile(dboot, [2.5, 97.5])
            point = rel_res - rel_sf
            passed = (-TOST_MARGIN <= lo90) and (hi90 <= TOST_MARGIN)
            sign = ("resample MORE degrading" if lo95 > 0 else
                    "resample MILDER" if hi95 < 0 else
                    "no significant direction")
            print(f"\n  C. P3-2a TOST [{d}]: d(rel_deg) resample-sf_all = "
                  f"{point:+.4f}, 90% CI [{lo90:+.4f},{hi90:+.4f}] vs "
                  f"+-{TOST_MARGIN} -> {'PASS' if passed else 'FAIL'}")
            print(f"     arbitration read (95% CI [{lo95:+.4f},{hi95:+.4f}])"
                  f": {sign}   [CLaSP seed 44 had resample MILDER]")
            drec["tost_p3_2a"] = {"point": float(point),
                                  "ci90": [float(lo90), float(hi90)],
                                  "ci95": [float(lo95), float(hi95)],
                                  "pass": bool(passed),
                                  "arbitration_sign": sign}

            # D. per-stratum ladder + anchor investigation
            print(f"\n  D. by stratum (MRR | x chance; ceilings "
                  f"V168 {ceil168:.4f} / V180 {ceil180:.4f} / other "
                  f"{ceil_other:.4f} composite)")
            print(f"    {'stratum':<8}{'n':>6}" +
                  "".join(f"{c:>12}" for c in LADDER))
            strec = {}
            for name, sel in strata.items():
                ns = int(sel.sum())
                line = f"    {name:<8}{ns:>6}"
                cells = {}
                for c in LADDER:
                    rr = p3[s][d][c][sel]
                    m = float((1.0 / rr).mean())
                    cells[c] = {"mrr": m, "ratio_to_chance": m / ch}
                    line += f"{m:>12.4f}"
                print(line)
                ceil_v = {"V=168": ceil168, "V=180": ceil180,
                          "other": ceil_other}[name]
                g = cells["gaussian"]["mrr"]
                pos = (g - ch) / (ceil_v - ch)
                cells["gaussian"]["position_chance_to_ceiling"] = float(pos)
                if name in ("V=168", "V=180"):
                    n_s = vc[int(name[2:])]
                    fr = {c: float((p3[s][d][c][sel] <= n_s).mean())
                          for c in CONDS}
                    cells["frac_rank_le_stratum"] = fr
                    print(f"      {name} gaussian position chance->ceiling: "
                          f"{pos:+.2f} | frac(rank<={n_s}): " +
                          ", ".join(f"{c} {fr[c]:.3f}" for c in CONDS) +
                          f" (uniform {n_s/POOL:.3f}, perfect split 1.0)")
                else:
                    print(f"      other gaussian position chance->ceiling "
                          f"(composite): {pos:+.2f}")
                strec[name] = cells
            drec["strata"] = strec

            # E. between-stratum rel-deg differences (P3-6 machinery)
            a_, b_ = strata["V=168"], strata["V=180"]
            na, nb = int(a_.sum()), int(b_.sum())
            ia = rng.integers(0, na, size=(N_BOOT, na))
            ib = rng.integers(0, nb, size=(N_BOOT, nb))
            inv_ua = 1.0 / p3[s][d]["unperturbed"][a_]
            inv_ub = 1.0 / p3[s][d]["unperturbed"][b_]
            bua, bub = inv_ua[ia].mean(1), inv_ub[ib].mean(1)
            print(f"\n  E. between-stratum d(rel_deg), V=168 minus V=180 "
                  f"(95% CI)")
            bs = {}
            for c in CONDS:
                iva = 1.0 / p3[s][d][c][a_]
                ivb = 1.0 / p3[s][d][c][b_]
                ra = (inv_ua.mean() - iva.mean()) / inv_ua.mean()
                rb = (inv_ub.mean() - ivb.mean()) / inv_ub.mean()
                dd = (bua - iva[ia].mean(1)) / bua \
                    - (bub - ivb[ib].mean(1)) / bub
                lo_, hi_ = np.percentile(dd, [2.5, 97.5])
                excl = (lo_ > 0) or (hi_ < 0)
                bs[c] = {"diff": float(ra - rb),
                         "ci": [float(lo_), float(hi_)],
                         "ci_excludes_zero": bool(excl)}
                reg = " [P3-6 REGISTERED]" if c == "resample" else \
                      " [unregistered]" if c == "gaussian" else ""
                print(f"    {c:<10} {ra-rb:+7.2%}  CI [{lo_:+.2%},{hi_:+.2%}]"
                      f"  {'excludes 0' if excl else 'includes 0'}{reg}")
            drec["between_stratum"] = bs
            srec[d] = drec
        out["seeds"][str(s)] = srec

    # ======================= aggregate + verdicts ==========================
    print("\n" + "=" * 78)
    print("AGGREGATE — dependent group, text->ts (PRIMARY)")
    print("=" * 78)
    agg = {}
    print(f"  {'cond':<12}{'MRR mean+-sd':>22}{'x chance':>10}"
          f"{'rel deg mean+-sd':>22}")
    for c in LADDER:
        ms = np.array([out["seeds"][str(s)]["text2ts"][c]["mrr"]
                       for s in seeds])
        line = f"  {c:<12}{f'{ms.mean():.4f} +- {ms.std(ddof=1):.4f}':>22}" \
               f"{ms.mean()/ch:>10.2f}"
        cell = {"mrr_mean": float(ms.mean()), "mrr_sd": float(ms.std(ddof=1)),
                "ratio_mean": float(ms.mean() / ch)}
        if c != "unperturbed":
            rs = np.array([out["seeds"][str(s)]["text2ts"][c]
                           ["rel_degradation"] for s in seeds])
            line += f"{f'{rs.mean():+.1%} +- {rs.std(ddof=1):.1%}':>22}"
            cell.update({"rel_mean": float(rs.mean()),
                         "rel_sd": float(rs.std(ddof=1))})
        agg[c] = cell
        print(line)

    # P3-2a
    passes = [out["seeds"][str(s)]["text2ts"]["tost_p3_2a"]["pass"]
              for s in seeds]
    v22a = "CONFIRMED" if all(passes) else "MISSED"
    print(f"\n  P3-2a (TOST resample~sf_all, +-{TOST_MARGIN}, all seeds): "
          f"{['PASS' if p else 'FAIL' for p in passes]} => {v22a}")
    signs = [out["seeds"][str(s)]["text2ts"]["tost_p3_2a"]
             ["arbitration_sign"] for s in seeds]
    print(f"  seed-44 arbitration at n=2005: " +
          "; ".join(f"seed {s}: {g}" for s, g in zip(seeds, signs)))

    # P3-3
    surv, band = [], []
    for s in seeds:
        cell = out["seeds"][str(s)]["text2ts"]["resample"]
        surv.append(cell["mrr_ci"][0] > ch)
        band.append(P3_3_BAND[0] <= cell["ratio_to_chance"] <= P3_3_BAND[1])
        print(f"  P3-3 seed {s}: resample {cell['ratio_to_chance']:.2f}x "
              f"chance, ratio CI [{cell['ratio_ci'][0]:.2f},"
              f"{cell['ratio_ci'][1]:.2f}] | survival "
              f"{'YES' if surv[-1] else 'NO'} | band "
              f"{'IN' if band[-1] else 'OUT'}")
    v33 = "CONFIRMED" if all(surv) and all(band) else "MISSED"
    print(f"  P3-3 (survival AND {P3_3_BAND[0]}-{P3_3_BAND[1]}x band, all "
          f"seeds) => {v33}")

    # P3-6
    incl = [not out["seeds"][str(s)]["text2ts"]["between_stratum"]
            ["resample"]["ci_excludes_zero"] for s in seeds]
    v36 = "CONFIRMED" if all(incl) else "MISSED"
    print(f"  P3-6 (resample rel-deg stratum-invariant, CI includes 0 all "
          f"seeds): {['YES' if i else 'NO' for i in incl]} => {v36}")
    g_excl = [out["seeds"][str(s)]["text2ts"]["between_stratum"]["gaussian"]
              ["ci_excludes_zero"] for s in seeds]
    print(f"  gaussian between-stratum CI excludes 0 (unregistered, "
          f"reported): {g_excl}")

    # ambiguous row trail
    print(f"\n  ambiguous row {AMBIGUOUS_ROW} (n=1 — DESCRIPTIVE ONLY):")
    for s in seeds:
        amb = p3[s]["groups"] != "dependent"
        for d in DIRECTIONS:
            r3 = {c: float(p3[s][d][c][amb][0]) for c in LADDER}
            print(f"    seed {s} {d}: unpert {r3['unperturbed']:.0f} -> "
                  + ", ".join(f"{c} {r3[c]:.0f}" for c in CONDS))
    print("     (text2ts rank 1 under resample in all seeds is a flagged "
          "observation; mechanism hypothesis in the log, n=1, never "
          "quotable)")

    out["aggregate"] = {"text2ts_dependent": agg,
                        "p3_2a_verdict": v22a,
                        "p3_2a_seed_passes": [bool(x) for x in passes],
                        "arbitration_signs": signs,
                        "p3_3_verdict": v33,
                        "p3_3_survival": [bool(x) for x in surv],
                        "p3_3_band": [bool(x) for x in band],
                        "p3_6_verdict": v36,
                        "gaussian_between_stratum_excludes_zero":
                            [bool(x) for x in g_excl]}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    print("Reminder: strata are the quotable unit; the pooled ladder is a "
          "mixture. Gaussian numbers are anchor-conditional per the "
          "accepted amendment — quote them WITH their stratum position.")


if __name__ == "__main__":
    main()

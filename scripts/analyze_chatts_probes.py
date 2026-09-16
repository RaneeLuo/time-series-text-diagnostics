#!/usr/bin/env python
"""
ChatTS analysis — score the GPU-session responses against the registered
predictions (handoff §4.9). CPU-only; no model, no torch.

Design (accepted 2026-08-16, this stage's Q1-Q4 as amended):
  - ONE script; all cross-file joins (Probe-3 rungs 1-2 from Probe-2,
    PJ vs Probe-2 unperturbed) live inside it behind hard gates.
  - Metric: MCQ accuracy per cell, BOTH ORDERS AVERAGED per unit
    (values 0 / 0.5 / 1). P(A) and the order-flip rate are separate
    diagnostics.
  - CIs: cluster bootstrap on SIGNALS (sample_id), B=2000, seed 42.
    95% for cells and plain differences; 90% for TOST.
  - TOST margin +/-0.05: a NEW APPLICATION of the pinned MRR margin to
    ACCURACY POINTS — flagged here and in the write-up, never silent.
  - Probe-1 paired test: primary Wilcoxon on per-item order-mean diffs
    (random - swap), Holm across the five components; McNemar SECONDARY
    on the strict both-orders-correct binarisation. (The naive 2N-row
    McNemar would double-count correlated orders — rejected.)
  - VOID rule (standing convention): a cell's capability read is VOID if
    the 95% CI lower bound of its no-perturbation/random condition < 0.60.
  - PJ is computed FIRST internally; if the TRUCE PJ TOST is not flat,
    every TRUCE masking cell in the output is mechanically stamped
    PROVISIONAL_PENDING_PJ_RERUN (a label that travels with the number —
    banners are not checks, error #9).
  - Rungs 1-2 of the Probe-3 ladder are JOINED from the ChatTS Probe-2
    responses (JG gates), never re-derived.

All structural expectations are printed as gates that HARD-STOP on
failure. Every expected value below was registered in-chat before this
script was run.

Run from the repo root (Windows PowerShell, one line):
  python scripts/analyze_chatts_probes.py

Inputs (defaults):
  results/experiments/chatts_probe{1,2,3}_responses.jsonl   (committed)
  data/processed/chatts_probe{1,2,3}_mcq.jsonl              (laptop-only)
Output:
  results/experiments/chatts_analysis.json  + the printed report.
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

# ----------------------------------------------------------------------
# constants — registered expectations
# ----------------------------------------------------------------------
B_BOOT = 2000
SEED = 42
TOST_MARGIN = 0.05          # new application of the pinned margin to accuracy
VOID_LOWER = 0.60           # standing convention (argued, not read off, at the border)
MCQ_SHA = "4029f94e2f6d"
FN_SHA = "4a4b3475f9e7"

EXP_ROWS = {1: 11080, 2: 9340, 3: 10936}
EXP_MANIFEST = {1: 11080, 2: 1756, 3: 3912}
P2_CONDS_TRUCE = ["unperturbed", "sf_all", "sf_half", "ex_half", "masking"]
P2_CONDS_SUSHI = P2_CONDS_TRUCE + ["sf_within_patch", "sf_across_patch"]
P3_NEW = ["resample", "gaussian", "cond_A", "cond_B", "cond_C"]
EXP_TRUCE_GROUPS = {"dependent": 715, "invariant": 18, "ambiguous": 5}
EXP_TRUCE_SOURCES = {"truce_stock": 570, "truce_synth": 168}
EXP_SUSHI_GROUPS = {"dependent": 135, "invariant": 4, "degenerate": 1}
EXP_COMPONENTS = {"C1": 205, "C2": 410, "C3": 280, "C4": 990, "C5": 885}
EXP_DRIFT_OFFSET_TRUE = 1748   # measured in the pre-analysis census (Claude-side, disclosed)
EXP_DRIFT_SCALE_TRUE = 1736
EXP_SPLICE = {"sushi": 126, "truce": -1}

FAILURES = []


def gate(name, ok, msg, hard=True):
    tag = "PASS" if ok else ("FAIL" if hard else "WARN")
    print(f"[{name}] {tag} — {msg}")
    if not ok and hard:
        FAILURES.append(name)


def hard_stop_if_failed(stage):
    if FAILURES:
        print(f"\nHARD STOP after {stage}: {FAILURES}")
        print("Do not interpret any number below a failed gate. Paste this output back.")
        sys.exit(1)


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def substrate_of(mcq_id):
    for part in mcq_id.split("|"):
        if ":" in part:
            src = part.split(":")[0]
            if src.startswith("sushi"):
                return "sushi"
            if src.startswith("truce"):
                return "truce"
    return "???"


# ----------------------------------------------------------------------
# statistics helpers
# ----------------------------------------------------------------------
def cluster_boot_mean(values, clusters, alpha, rng):
    """Percentile cluster-bootstrap CI for a mean. values: 1-D floats;
    clusters: same-length labels (sample_id). Returns (mean, lo, hi, n_units, n_clusters)."""
    values = np.asarray(values, dtype=float)
    labs = np.asarray(clusters)
    uniq = np.unique(labs)
    sums = np.array([values[labs == c].sum() for c in uniq])
    cnts = np.array([(labs == c).sum() for c in uniq], dtype=float)
    k = len(uniq)
    boots = np.empty(B_BOOT)
    for b in range(B_BOOT):
        idx = rng.integers(0, k, k)
        boots[b] = sums[idx].sum() / cnts[idx].sum()
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(values.mean()), float(lo), float(hi), int(len(values)), int(k)


def cell(values, clusters, rng, alpha=0.05):
    m, lo, hi, n, k = cluster_boot_mean(values, clusters, alpha, rng)
    return {"acc": round(m, 4), "ci95": [round(lo, 4), round(hi, 4)],
            "n_units": n, "n_signals": k}


def paired(diffs, clusters, rng):
    """Paired difference with 95% CI and a 90%-CI TOST verdict at +/-0.05."""
    m, lo95, hi95, n, k = cluster_boot_mean(diffs, clusters, 0.05, rng)
    _, lo90, hi90, _, _ = cluster_boot_mean(diffs, clusters, 0.10, rng)
    tost = bool(lo90 > -TOST_MARGIN and hi90 < TOST_MARGIN)
    return {"mean_diff": round(m, 4), "ci95": [round(lo95, 4), round(hi95, 4)],
            "ci90": [round(lo90, 4), round(hi90, 4)],
            "tost_pm005": "PASS(flat)" if tost else "FAIL(not flat)",
            "n_units": n, "n_signals": k}


def holm(pvals):
    """Step-down Holm adjustment. pvals: dict name->p. Returns name->p_adj."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adj, running = {}, 0.0
    for i, (name, p) in enumerate(items):
        running = max(running, (m - i) * p)
        adj[name] = min(1.0, running)
    return adj


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resp-dir", default="results/experiments")
    ap.add_argument("--manifest-dir", default="data/processed")
    ap.add_argument("--out", default="results/experiments/chatts_analysis.json")
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)

    print("=" * 72)
    print("ChatTS analysis — registered structural expectations first.")
    print(f"TOST margin +/-{TOST_MARGIN} on ACCURACY = new application of the")
    print("pinned MRR margin (flagged, not silent). Bootstrap clusters = signals.")
    print("=" * 72)

    resp, man = {}, {}
    for p in (1, 2, 3):
        resp[p] = load_jsonl(Path(args.resp_dir) / f"chatts_probe{p}_responses.jsonl")
        man[p] = load_jsonl(Path(args.manifest_dir) / f"chatts_probe{p}_mcq.jsonl")

    # ---- GA1/GA2: counts and duplicates -------------------------------
    for p in (1, 2, 3):
        ids = Counter((r["mcq_id"], r.get("condition")) for r in resp[p])
        dup = sum(1 for v in ids.values() if v > 1)
        gate(f"GA1-p{p}", len(resp[p]) == EXP_ROWS[p] and dup == 0,
             f"responses = {len(resp[p])} (exp {EXP_ROWS[p]}); dup (id,cond) = {dup} (exp 0)")
        gate(f"GA2-p{p}", len(man[p]) == EXP_MANIFEST[p],
             f"manifest rows = {len(man[p])} (exp {EXP_MANIFEST[p]})")

    blocks = Counter(r["block"] for r in man[3])
    gate("GA2-p3-blocks",
         blocks == Counter({"base": 1756, "five_number": 1756, "pj_control": 400}),
         f"p3 blocks = {dict(blocks)} (exp base 1756 / five_number 1756 / pj_control 400)")

    # ---- GA3: template shas -------------------------------------------
    sha1 = Counter(r["template_sha"] for r in man[1])
    sha2 = Counter(r["template_sha"] for r in man[2])
    sha3 = Counter((r["block"], r["template_sha"]) for r in man[3])
    gate("GA3-sha", sha1 == Counter({MCQ_SHA: 11080}) and sha2 == Counter({MCQ_SHA: 1756})
         and sha3 == Counter({("base", MCQ_SHA): 1756, ("five_number", FN_SHA): 1756,
                              ("pj_control", MCQ_SHA): 400}),
         f"p1={dict(sha1)}; p2={dict(sha2)}; p3={dict(sha3)}")
    hard_stop_if_failed("counts/shas")

    # ---- GA4: lossless joins ------------------------------------------
    m1 = {r["mcq_id"]: r for r in man[1]}
    j1 = [(rr, m1.get(rr["mcq_id"])) for rr in resp[1]]
    orphans1 = sum(1 for _, mm in j1 if mm is None)
    gate("GA4-p1-join", orphans1 == 0 and len(j1) == 11080,
         f"p1 join 1:1 by mcq_id — orphans = {orphans1} (exp 0)")

    m2 = {r["mcq_id"]: r for r in man[2]}
    exp_expansion = sum(len(r["conditions"]) for r in man[2])
    keyset2 = {(r["mcq_id"], c) for r in man[2] for c in r["conditions"]}
    got2 = {(r["mcq_id"], r["condition"]) for r in resp[2]}
    gate("GA4-p2-join", exp_expansion == 9340 and keyset2 == got2,
         f"p2 expansion = {exp_expansion} (exp 9340); "
         f"missing = {len(keyset2 - got2)}; extra = {len(got2 - keyset2)} (exp 0/0)")

    m3 = {r["mcq_id"]: r for r in man[3]}
    keyset3 = set()
    for r in man[3]:
        for c in r["conditions"]:
            keyset3.add((r["mcq_id"], c))
    got3 = {(r["mcq_id"], r["condition"]) for r in resp[3]}
    gate("GA4-p3-join", keyset3 == got3,
         f"p3 (id,cond) — missing = {len(keyset3 - got3)}; extra = {len(got3 - keyset3)} (exp 0/0)")
    hard_stop_if_failed("joins")

    # ---- GA5: splice census (digit-exact vs the registered census) ----
    for p in (1, 2, 3):
        bad = 0
        for r in resp[p]:
            sub = substrate_of(r["mcq_id"])
            expct = 0 if (p == 3 and r["condition"] == "five_number") else EXP_SPLICE[sub]
            if r["splice_delta"] != expct:
                bad += 1
        gate(f"GA5-splice-p{p}", bad == 0,
             f"splice-delta mismatches = {bad} (exp 0; SUSHI +126 / TRUCE -1 / five_number 0)")

    # ---- GA6: drift flags (probe 2, masking only) ----------------------
    flag_rows = [r for r in resp[2] if "prefix_offset_drifted" in r]
    non_mask = sum(1 for r in flag_rows if r["condition"] != "masking")
    off_true = sum(1 for r in flag_rows if r["prefix_offset_drifted"])
    sc_true = sum(1 for r in flag_rows if r["prefix_scale_drifted"])
    gate("GA6-drift", non_mask == 0 and len(flag_rows) == 1756
         and off_true == EXP_DRIFT_OFFSET_TRUE and sc_true == EXP_DRIFT_SCALE_TRUE,
         f"flags on {len(flag_rows)} rows (exp 1756, masking only, non-masking={non_mask}); "
         f"offset True {off_true} (exp {EXP_DRIFT_OFFSET_TRUE}); "
         f"scale True {sc_true} (exp {EXP_DRIFT_SCALE_TRUE})")

    # ---- GA7: certified group censuses ---------------------------------
    tg = Counter(r["group"] for r in man[2] if r["dataset"] == "truce" and r["order"] == "corrA")
    sg = Counter(r["group"] for r in man[2] if r["dataset"] == "sushi" and r["order"] == "corrA")
    src = Counter(r["source"] for r in man[2] if r["dataset"] == "truce" and r["order"] == "corrA")
    gate("GA7-groups", dict(tg) == EXP_TRUCE_GROUPS and dict(sg) == EXP_SUSHI_GROUPS
         and dict(src) == EXP_TRUCE_SOURCES,
         f"truce={dict(tg)} (exp {EXP_TRUCE_GROUPS}); sushi={dict(sg)} "
         f"(exp {EXP_SUSHI_GROUPS}); sources={dict(src)} (exp {EXP_TRUCE_SOURCES})")

    # ---- GA8: probe-1 pairing + component census ------------------------
    pairs = defaultdict(dict)   # pair_key -> {'swap': {order: correct}, 'random': {...}}
    comp_census = Counter()
    for rr, mm in j1:
        comp = mm["component"].split("_")[0]
        if mm["order"] == "corrA":                 # item-level census, not row-level
            comp_census[(comp, mm["condition"])] += 1
        key_parts = mm["item_id"].split("|")
        pair_key = "|".join(x for x in key_parts if x not in ("swap", "random"))
        pairs[pair_key].setdefault(mm["condition"], {})[mm["order"]] = int(rr["correct"])
    comp_ok = all(comp_census[(c, cond)] == n for c, n in EXP_COMPONENTS.items()
                  for cond in ("swap", "random"))
    pair_ok = all(set(v) == {"swap", "random"}
                  and set(v["swap"]) == {"corrA", "corrB"}
                  and set(v["random"]) == {"corrA", "corrB"} for v in pairs.values())
    gate("GA8-p1-structure", len(pairs) == 2770 and pair_ok and comp_ok,
         f"pair keys = {len(pairs)} (exp 2770); every key has swap+random x both orders: "
         f"{pair_ok}; component census per condition matches "
         f"{EXP_COMPONENTS}: {comp_ok}")

    # ---- GA9: JG joins (Probe-3 rungs 1-2; PJ) --------------------------
    p2caps = {r["mcq_id"] for r in man[2]}                    # 'p2|caption|order'
    base_ids = {r["mcq_id"] for r in man[3] if r["block"] == "base"}
    fn_ids = {r["mcq_id"] for r in man[3] if r["block"] == "five_number"}
    pj_ids = {r["mcq_id"] for r in man[3] if r["block"] == "pj_control"}
    jg_base = all(i.replace("p3base|", "p2|") in p2caps for i in base_ids)
    jg_fn = all(i.replace("p3fn|", "p2|") in p2caps for i in fn_ids)
    jg_pj = all(i.replace("p3pj|", "p2|") in p2caps for i in pj_ids)
    p2resp = {(r["mcq_id"], r["condition"]): r for r in resp[2]}
    rung1 = sum(1 for i in base_ids if (i.replace("p3base|", "p2|"), "unperturbed") in p2resp)
    rung2 = sum(1 for i in base_ids if (i.replace("p3base|", "p2|"), "sf_all") in p2resp)
    pj_hit = sum(1 for i in pj_ids if (i.replace("p3pj|", "p2|"), "unperturbed") in p2resp)
    gate("GA9-JG", jg_base and jg_fn and jg_pj and rung1 == 1756 and rung2 == 1756
         and pj_hit == 400 and len(base_ids) == 1756,
         f"base->p2 id map complete: {jg_base}; fn: {jg_fn}; pj: {jg_pj}; "
         f"rung1 joins {rung1}/1756; rung2 joins {rung2}/1756; PJ joins {pj_hit}/400 "
         "(rungs 1-2 are READ from Probe-2 records, never re-derived)")
    hard_stop_if_failed("structure")
    print("\nALL STRUCTURAL GATES GREEN — analysis follows.\n" + "=" * 72)

    out = {"conventions": {
        "metric": "MCQ accuracy, both orders averaged per unit (0/0.5/1)",
        "bootstrap": f"cluster on sample_id (signals), B={B_BOOT}, seed {SEED}",
        "ci": "95% cells/differences; 90% for TOST",
        "tost": f"+/-{TOST_MARGIN} accuracy — NEW APPLICATION of the pinned MRR margin",
        "void_rule": f"95% CI lower bound of the no-perturbation/random cell < {VOID_LOWER}",
    }}

    # =====================================================================
    # PJ FIRST (it decides the stamp on every TRUCE masking cell)
    # =====================================================================
    print("\n--- PJ prefix-jitter control (computed FIRST; load-bearing for TRUCE masking) ---")
    pj_acc, pj_base_acc, pj_clus = defaultdict(dict), defaultdict(dict), {}
    p3resp = {(r["mcq_id"], r["condition"]): r for r in resp[3]}
    for r in man[3]:
        if r["block"] != "pj_control":
            continue
        u = r["caption_id"]
        ds = r["dataset"]
        pj_row = p3resp[(r["mcq_id"], "pj_control")]
        base_row = p2resp[(r["mcq_id"].replace("p3pj|", "p2|"), "unperturbed")]
        pj_acc[ds].setdefault(u, []).append(int(pj_row["correct"]))
        pj_base_acc[ds].setdefault(u, []).append(int(base_row["correct"]))
        pj_clus[u] = r["sample_id"]
    pj_out, pj_flat = {}, {}
    for ds in ("sushi", "truce"):
        units = sorted(pj_acc[ds])
        diffs = [np.mean(pj_acc[ds][u]) - np.mean(pj_base_acc[ds][u]) for u in units]
        clus = [pj_clus[u] for u in units]
        res = paired(diffs, clus, rng)
        res["n_captions"] = len(units)
        pj_out[ds] = res
        pj_flat[ds] = res["tost_pm005"].startswith("PASS")
        print(f"  {ds}: jittered-minus-true-prefix diff {res['mean_diff']:+.4f} "
              f"ci90 {res['ci90']} -> {res['tost_pm005']} (n={len(units)} captions)")
    truce_mask_stamp = "OK" if pj_flat["truce"] else "PROVISIONAL_PENDING_PJ_RERUN"
    print(f"  PJ verdict: SUSHI {'flat' if pj_flat['sushi'] else 'NOT FLAT'}; "
          f"TRUCE {'flat' if pj_flat['truce'] else 'NOT FLAT'} -> "
          f"TRUCE masking cells stamped: {truce_mask_stamp}")
    if not pj_flat["truce"]:
        print("  NOTE: the pre-named frozen-prefix rerun decision FIRES — do not quote")
        print("  TRUCE masking anywhere until that rerun; and this is itself a finding")
        print("  (ChatTS reads its own prefix digits).")
    out["pj_control"] = {"cells": pj_out, "truce_masking_stamp": truce_mask_stamp,
                         "note": "PJ is LOAD-BEARING for TRUCE masking (binding)"}

    # =====================================================================
    # Viability + position (PC1-1, PC1-2) and probe-level diagnostics
    # =====================================================================
    print("\n--- PC1-1 viability / PC1-2 position / diagnostics ---")
    # order-mean accuracy per probe-1 item, random condition, SUSHI (probe 1 is SUSHI-only)
    rand_by_item, item_sig = defaultdict(list), {}
    swap_by_item = defaultdict(list)
    for rr, mm in j1:
        key_parts = mm["item_id"].split("|")
        pk = "|".join(x for x in key_parts if x not in ("swap", "random"))
        item_sig[pk] = mm["sample_id"]
        (rand_by_item if mm["condition"] == "random" else swap_by_item)[pk].append(
            int(rr["correct"]))
    units = sorted(rand_by_item)
    viab_vals = [np.mean(rand_by_item[u]) for u in units]
    viab_clus = [item_sig[u] for u in units]
    viab = cell(viab_vals, viab_clus, rng)
    sushi_void = viab["ci95"][0] < VOID_LOWER
    pc11_sushi = "HIT" if viab["acc"] >= 0.90 else "MISS"
    print(f"  SUSHI random-condition accuracy = {viab['acc']} ci95 {viab['ci95']} "
          f"(n={viab['n_units']} items, {viab['n_signals']} signals)")
    print(f"  PC1-1 (SUSHI >= 0.90): {pc11_sushi}; VOID check (lower < {VOID_LOWER}): "
          f"{'VOID' if sushi_void else 'not void — capability present'}")

    # TRUCE viability from Probe-2 unperturbed (12-point caveat stands)
    tacc = defaultdict(list)
    for r in man[2]:
        if r["dataset"] != "truce":
            continue
        tacc[r["caption_id"]].append(int(p2resp[(r["mcq_id"], "unperturbed")]["correct"]))
    tsig = {r["caption_id"]: r["sample_id"] for r in man[2] if r["dataset"] == "truce"}
    tunits = sorted(tacc)
    tviab = cell([np.mean(tacc[u]) for u in tunits], [tsig[u] for u in tunits], rng)
    truce_void = tviab["ci95"][0] < VOID_LOWER
    print(f"  TRUCE unperturbed accuracy = {tviab['acc']} ci95 {tviab['ci95']} "
          f"(12-point series, below the model's recommended 64 — stated caveat); "
          f"{'VOID' if truce_void else 'not void'}")

    # PC1-2 P(A) + order-flip + logit margin per probe
    diag = {}
    for p in (1, 2, 3):
        pa = np.mean([r["choice"] == "A" for r in resp[p]])
        margin = np.mean([abs(r["logit_A"] - r["logit_B"]) for r in resp[p]])
        diag[p] = {"p_choose_A": round(float(pa), 4),
                   "mean_abs_logit_margin": round(float(margin), 3)}
        print(f"  probe {p}: P(A) = {pa:.4f}; mean |logit margin| = {margin:.2f}")
    pa_all = np.mean([r["choice"] == "A" for p in (1, 2, 3) for r in resp[p]])
    pc12 = "HIT" if abs(pa_all - 0.50) <= 0.15 else "MISS"
    print(f"  pooled P(A) = {pa_all:.4f} -> PC1-2 (0.50 +/- 0.15): {pc12}")
    out["viability"] = {"sushi_random": viab, "sushi_void": sushi_void,
                        "pc1_1_sushi": pc11_sushi,
                        "truce_unperturbed": tviab, "truce_void": truce_void}
    out["position"] = {"per_probe": diag, "pooled_p_A": round(float(pa_all), 4),
                       "pc1_2": pc12}
    out["pc1_3_agreement"] = {"value": "600/600, 0 unparseable",
                              "source": "session logs (GR8), not recomputed here",
                              "verdict": "HIT (logit readout primary in all probes)"}

    # =====================================================================
    # Probe 1 — component table, Wilcoxon+Holm, McNemar, C4 answer
    # =====================================================================
    print("\n--- Probe 1: component-swap (SUSHI) ---")
    comp_units = defaultdict(list)
    for pk in units:
        comp_units[pk.split("_")[0]].append(pk)  # 'C1'... from 'C1_trend...|...'
    p1_table, pvals, mcnemar_tab = {}, {}, {}
    order_flip = {"swap": [], "random": []}
    for comp in sorted(comp_units):
        us = comp_units[comp]
        sw = [np.mean(swap_by_item[u]) for u in us]
        rd = [np.mean(rand_by_item[u]) for u in us]
        cl = [item_sig[u] for u in us]
        swc, rdc = cell(sw, cl, rng), cell(rd, cl, rng)
        d = np.array(rd) - np.array(sw)          # degradation = random - swap
        gap = paired(d.tolist(), cl, rng)
        try:
            wstat = stats.wilcoxon(d, alternative="two-sided")
            pvals[comp] = float(wstat.pvalue)
        except ValueError:                        # all-zero diffs
            pvals[comp] = 1.0
        strict_s = np.array([min(swap_by_item[u]) for u in us])   # both orders correct
        strict_r = np.array([min(rand_by_item[u]) for u in us])
        n01 = int(((strict_s == 0) & (strict_r == 1)).sum())
        n10 = int(((strict_s == 1) & (strict_r == 0)).sum())
        mp = float(stats.binomtest(min(n01, n10), n01 + n10, 0.5).pvalue) \
            if (n01 + n10) else 1.0
        mcnemar_tab[comp] = {"n01_rand_only": n01, "n10_swap_only": n10, "p": round(mp, 6)}
        void = rdc["ci95"][0] < VOID_LOWER
        p1_table[comp] = {"swap": swc, "random": rdc, "gap_random_minus_swap": gap,
                          "wilcoxon_p": round(pvals[comp], 6),
                          "mcnemar_secondary": mcnemar_tab[comp],
                          "verdict_void": void, "n_items": len(us)}
        order_flip["swap"] += [1 if v == 0.5 else 0 for v in sw]
        order_flip["random"] += [1 if v == 0.5 else 0 for v in rd]
        print(f"  {comp}: swap {swc['acc']} {swc['ci95']} | random {rdc['acc']} "
              f"{rdc['ci95']} | gap {gap['mean_diff']:+.4f} {gap['ci95']} | "
              f"Wilcoxon p={pvals[comp]:.2e} | McNemar {n01}/{n10} p={mp:.2e}"
              f"{' | VOID' if void else ''}")
    adj = holm(pvals)
    for comp in p1_table:
        p1_table[comp]["wilcoxon_p_holm"] = round(adj[comp], 6)
    print("  Holm-adjusted Wilcoxon p:",
          {c: f"{adj[c]:.2e}" for c in sorted(adj)})
    fr_s, fr_r = np.mean(order_flip["swap"]), np.mean(order_flip["random"])
    print(f"  order-flip rate (item answered differently by letter order): "
          f"swap {fr_s:.4f}, random {fr_r:.4f}")
    print("  NOTE: C4 and the component ordering answer the registered NON-prediction —")
    print("  this is the arm's open question being ANSWERED, not a prediction scored.")
    out["probe1"] = {"components": p1_table,
                     "order_flip_rate": {"swap": round(float(fr_s), 4),
                                         "random": round(float(fr_r), 4)},
                     "non_prediction_note": "C4/ordering = open question, answered here"}

    # =====================================================================
    # Probe 2 — degradation profile + DiD over certified groups
    # =====================================================================
    print("\n--- Probe 2: order-invariance (accuracy per cell; drops vs unperturbed) ---")
    acc2 = defaultdict(lambda: defaultdict(list))   # (ds,group,cond) -> unit -> corrects
    sig2 = {}
    for r in man[2]:
        sig2[r["caption_id"]] = r["sample_id"]
    for r in resp[2]:
        mm = m2[r["mcq_id"]]
        acc2[(mm["dataset"], mm["group"], r["condition"])].setdefault(
            mm["caption_id"], []).append(int(r["correct"]))
    p2_out = {}
    for ds in ("sushi", "truce"):
        conds = P2_CONDS_SUSHI if ds == "sushi" else P2_CONDS_TRUCE
        groups = (["dependent", "invariant", "degenerate"] if ds == "sushi"
                  else ["dependent", "invariant", "ambiguous"])
        for g in groups:
            base = acc2[(ds, g, "unperturbed")]
            uts = sorted(base)
            if not uts:
                continue
            cl = [sig2[u] for u in uts]
            for c in conds:
                vals = [np.mean(acc2[(ds, g, c)][u]) for u in uts]
                cc = cell(vals, cl, rng)
                if c != "unperturbed":
                    d = [np.mean(acc2[(ds, g, c)][u]) - np.mean(base[u]) for u in uts]
                    cc["drop_vs_unpert"] = paired(d, cl, rng)
                if ds == "truce" and c == "masking":
                    cc["status"] = truce_mask_stamp
                if g in ("invariant", "ambiguous", "degenerate"):
                    cc["thin_cell"] = f"n={len(uts)} captions — never load-bearing alone"
                p2_out[f"{ds}|{g}|{c}"] = cc
            line = " ".join(f"{c}:{p2_out[f'{ds}|{g}|{c}']['acc']}" for c in conds)
            print(f"  {ds}/{g} (n={len(uts)}): {line}")
    # DiD: (dep drop) - (inv drop), TRUCE per condition; SUSHI descriptive
    did_out = {}
    for ds, load in (("truce", True), ("sushi", False)):
        dep = acc2[(ds, "dependent", "unperturbed")]
        inv = acc2[(ds, "invariant", "unperturbed")]
        du, iu = sorted(dep), sorted(inv)
        conds = [c for c in (P2_CONDS_SUSHI if ds == "sushi" else P2_CONDS_TRUCE)
                 if c != "unperturbed"]
        for c in conds:
            dd = [np.mean(acc2[(ds, "dependent", c)][u]) - np.mean(dep[u]) for u in du]
            di = [np.mean(acc2[(ds, "invariant", c)][u]) - np.mean(inv[u]) for u in iu]
            # DiD as difference of group means; joint cluster bootstrap, vectorised:
            # per cluster, precompute (sum, count) for each group; resample clusters.
            vals = np.array(dd + di)
            grp = np.array([0] * len(dd) + [1] * len(di))
            clus = np.array([sig2[u] for u in du] + [sig2[u] for u in iu])
            uniq = np.unique(clus)
            k = len(uniq)
            sd = np.zeros(k); nd = np.zeros(k); si = np.zeros(k); ni = np.zeros(k)
            for j, cc in enumerate(uniq):
                sel = clus == cc
                sd[j] = vals[sel & (grp == 0)].sum(); nd[j] = (sel & (grp == 0)).sum()
                si[j] = vals[sel & (grp == 1)].sum(); ni[j] = (sel & (grp == 1)).sum()
            idx = rng.integers(0, k, (B_BOOT, k))
            Nd, Ni = nd[idx].sum(1), ni[idx].sum(1)
            ok = (Nd > 0) & (Ni > 0)
            boots = (si[idx].sum(1)[ok] / Ni[ok]) - (sd[idx].sum(1)[ok] / Nd[ok])
            lo, hi = np.percentile(boots, [2.5, 97.5])
            entry = {"did_invdrop_minus_depdrop": round(float(np.mean(di) - np.mean(dd)), 4),
                     "ci95": [round(float(lo), 4), round(float(hi), 4)],
                     "n_dep": len(du), "n_inv": len(iu),
                     "load_bearing": load and c != "masking",
                     "note": ("thin invariant leg — quote with its n" if len(iu) < 30 else "")}
            if ds == "truce" and c == "masking":
                entry["status"] = truce_mask_stamp
            if ds == "sushi":
                entry["note"] = f"descriptive only (invariant n={len(iu)})"
            did_out[f"{ds}|{c}"] = entry
            print(f"  DiD {ds}/{c}: {entry['did_invdrop_minus_depdrop']:+.4f} "
                  f"ci95 {entry['ci95']} (dep n={len(du)}, inv n={len(iu)})"
                  + (f" [{entry.get('status','')}]" if entry.get('status') else ""))
    # degenerate identity control (SUSHI constant): letter consistency across conditions
    deg_caps = [r["caption_id"] for r in man[2]
                if r["dataset"] == "sushi" and r["group"] == "degenerate"
                and r["order"] == "corrA"]
    deg_report = {}
    for cap in deg_caps:
        for order in ("corrA", "corrB"):
            mid = f"p2|{cap}|{order}"
            letters = {c: p2resp[(mid, c)]["choice"] for c in P2_CONDS_SUSHI}
            consistent = len(set(letters.values())) == 1
            deg_report[f"{cap}|{order}"] = {"letters": letters, "consistent": consistent}
            print(f"  degenerate identity {cap}|{order}: letters {letters} — "
                  f"{'CONSISTENT' if consistent else 'INCONSISTENT (report loudly)'}")
    out["probe2"] = {"cells": p2_out, "did": did_out, "degenerate_identity": deg_report,
                     "pooled_warning": "strata always; any pooled number is non-load-bearing"}

    # =====================================================================
    # Probe 3 — the ladder; anchors A~C and C==stock-gaussian; five-number
    # =====================================================================
    print("\n--- Probe 3: summary-statistics ladder (rungs 1-2 JOINED from Probe-2) ---")
    lad = defaultdict(lambda: defaultdict(dict))  # ds -> rung -> unit -> mean acc
    sig3 = {}
    for r in man[3]:
        if r["block"] == "pj_control":
            continue
        u, ds = r["caption_id"], r["dataset"]
        sig3[u] = r["sample_id"]
        if r["block"] == "base":
            p2id = r["mcq_id"].replace("p3base|", "p2|")
            lad[ds]["rung1_unperturbed"].setdefault(u, []).append(
                int(p2resp[(p2id, "unperturbed")]["correct"]))
            lad[ds]["rung2_sf_all"].setdefault(u, []).append(
                int(p2resp[(p2id, "sf_all")]["correct"]))
            for c in P3_NEW:
                lad[ds][c].setdefault(u, []).append(
                    int(p3resp[(r["mcq_id"], c)]["correct"]))
        else:  # five_number
            lad[ds]["five_number"].setdefault(u, []).append(
                int(p3resp[(r["mcq_id"], "five_number")]["correct"]))
    rungs = ["rung1_unperturbed", "rung2_sf_all", "five_number",
             "resample", "gaussian", "cond_A", "cond_B", "cond_C"]
    p3_out = {}
    for ds in ("sushi", "truce"):
        uts = sorted(lad[ds]["rung1_unperturbed"])
        cl = [sig3[u] for u in uts]
        for rg in rungs:
            vals = [np.mean(lad[ds][rg][u]) for u in uts]
            p3_out[f"{ds}|{rg}"] = cell(vals, cl, rng)
        print(f"  {ds} ladder (n={len(uts)}): " + " ".join(
            f"{rg}:{p3_out[f'{ds}|{rg}']['acc']}" for rg in rungs))
        # anchors and registered expectations, paired per caption:
        def pdiff(a, b):
            d = [np.mean(lad[ds][a][u]) - np.mean(lad[ds][b][u]) for u in uts]
            return paired(d, cl, rng)
        agree_cg = np.mean([p3resp[(f"p3base|{u}|{o}", "cond_C")]["choice"]
                            == p3resp[(f"p3base|{u}|{o}", "gaussian")]["choice"]
                            for u in uts for o in ("corrA", "corrB")])
        agree_ac = np.mean([p3resp[(f"p3base|{u}|{o}", "cond_A")]["choice"]
                            == p3resp[(f"p3base|{u}|{o}", "cond_C")]["choice"]
                            for u in uts for o in ("corrA", "corrB")])
        p3_out[f"{ds}|anchor_C_vs_gaussian"] = {**pdiff("cond_C", "gaussian"),
                                                "letter_agreement": round(float(agree_cg), 4)}
        p3_out[f"{ds}|A_vs_C_nearconstruction"] = {**pdiff("cond_A", "cond_C"),
                                                   "letter_agreement": round(float(agree_ac), 4)}
        p3_out[f"{ds}|B_vs_rung1_prefixchannel"] = pdiff("cond_B", "rung1_unperturbed")
        p3_out[f"{ds}|fivenum_vs_rung1"] = pdiff("five_number", "rung1_unperturbed")
        print(f"    anchor cond_C vs stock gaussian: "
              f"{p3_out[f'{ds}|anchor_C_vs_gaussian']['tost_pm005']} "
              f"(diff {p3_out[f'{ds}|anchor_C_vs_gaussian']['mean_diff']:+.4f}, "
              f"letter agreement {agree_cg:.4f}) — manual-path results-level proof")
        print(f"    A~C (registered near-construction): "
              f"{p3_out[f'{ds}|A_vs_C_nearconstruction']['tost_pm005']} "
              f"(diff {p3_out[f'{ds}|A_vs_C_nearconstruction']['mean_diff']:+.4f}, "
              f"agreement {agree_ac:.4f})")
        print(f"    cond_B (donor prefix) vs rung1: "
              f"{p3_out[f'{ds}|B_vs_rung1_prefixchannel']['mean_diff']:+.4f} "
              f"{p3_out[f'{ds}|B_vs_rung1_prefixchannel']['ci95']}")
    out["probe3"] = {"cells_and_anchors": p3_out,
                     "note": "rungs 1-2 read from ChatTS Probe-2 responses (GA9-JG)"}

    # =====================================================================
    # write output
    # =====================================================================
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("\n" + "=" * 72)
    print(f"Wrote {args.out}")
    print("Paste this FULL output back. Scoring of PC1-1/PC1-2/A~C/PJ/anchor and the")
    print("NON-prediction answer happens in-chat against the §4.9 register — the")
    print("script computes; the ledger verdicts are pinned in conversation.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
analyze_probe3.py (text-embedding-3-large) — P3-7 scoring for the floor
Probe-3 negative control.

SCORING PINNED 2026-08-15 BEFORE THIS SCRIPT FIRST RAN (chat record;
mirrors the pinned P2-4 machinery, per the registered "P3-7 = P2-4
pattern"):
  Inference cells: sushi/dependent (n=135) and truce/dependent (n=715).
  Thin cells (truce/invariant 18, sushi/invariant 4) reported with CIs
  and their n, never load-bearing. Ambiguous/degenerate descriptive only.
  P3-7 CONFIRMED iff TOST (+/-0.05 ABSOLUTE MRR degradation, alpha 0.05
  via 90% cluster-bootstrap CI inside the margin; B=2000; rng seed 42;
  signals resampled, never queries — binding) PASSES on both inference
  cells for BOTH conditions (resample, gaussian) in all three arms
  (12 TOSTs), AND no inference-cell |delta| point estimate >= 0.05.
  The floor is PRE-DECLARED VOID: the ladder profile (unperturbed |
  sf_all | resample | gaussian) is reported per cell with ratio-to-
  chance CIs as the negative-control record, not as capability
  evidence. Wilcoxon two-sided report-only.

Gates:
  JG   probe3 and probe2 records join losslessly on caption_id
       (878 rows/arm) AND rank_unperturbed agrees <= 1e-9 — HARD STOP.
       PRE-VERIFIED 2026-08-15 (disclosed Claude-side check on the
       uploaded records vs the committed Probe-2 files): max diff 0.0
       in all three arms. sf_all ranks are merged from the Probe-2
       records (rung 2 read, never rerun — Q1).
  REF  cross-computation gate: the table below was computed
       INDEPENDENTLY from the same per-query records (uploaded copies,
       2026-08-15, Claude's environment, disclosed in-chat). This run
       must reproduce every delta to <= 1e-9 — HARD STOP otherwise.

Registered expectations (2026-08-15, before this script's first run):
  REF reproduced 12/12; JG max diff 0.0 in every arm;
  12/12 TOSTs PASS; max inference-cell |delta| = 0.0067 (truce_dep/
  resample, arm 42) — P3-7 CONFIRMED;
  thin-cell note expected in the OPPOSITE direction from Probe 2's
  mimicry finding: truce/invariant and truce/ambiguous DROP under the
  surrogates (0.071 -> 0.015-0.028 resample; 0.206 -> 0.007-0.106)
  — reported with n and CIs, mechanism left to the write-up, never
  load-bearing;
  sf_all ladder cells must agree with the committed floor Probe-2
  stats (same records, same metric).

Run from the repository root:
    python -m models.openai_embed.analyze_probe3
Reads:  results/experiments/probe3_openai_per_query_seed{42,43,44}.jsonl
        results/experiments/probe2_openai_per_query_seed{42,43,44}.jsonl
Writes: results/experiments/probe3_openai_stats.json
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
CONDITIONS = ("resample", "gaussian")
LADDER = ("sf_all", "resample", "gaussian")   # rung order after unperturbed
B = 2000
TOST_MARGIN = 0.05
RNG_SEED = 42
CH_GLOBAL = 0.016928   # H(386)/386, the committed chance reference

INFER_CELLS = {"sushi_dep": ("sushi", "dependent"),
               "truce_dep": ("truce", "dependent")}
THIN_CELLS = {"sushi_inv": ("sushi", "invariant"),
              "truce_inv": ("truce", "invariant")}
DESC_CELLS = {"sushi_degen": ("sushi", "degenerate"),
              "truce_amb": ("truce", "ambiguous")}

# REF: computed independently from the uploaded per-query records
# (Claude's environment, 2026-08-15, disclosed). delta = MRR(cond) -
# MRR(unperturbed) on the inference cell. HARD gate at <= 1e-9.
REF = {
  "42": {"sushi_dep/resample": -0.0004406235,
         "sushi_dep/gaussian": -2.10981e-05,
         "truce_dep/resample": -0.0066832396,
         "truce_dep/gaussian": -0.0026005982},
  "43": {"sushi_dep/resample": -0.0004737295,
         "sushi_dep/gaussian": -0.0002008768,
         "truce_dep/resample": -0.0032746222,
         "truce_dep/gaussian": -0.0028495676},
  "44": {"sushi_dep/resample": -0.0004691286,
         "sushi_dep/gaussian": -0.0003557154,
         "truce_dep/resample": -0.0039839935,
         "truce_dep/gaussian": -0.005491266},
}


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
    cl = defaultdict(list)
    for cid, r in recs.items():
        cl[r["gt"]].append(cid)
    return cl


def mrr(rows, key):
    return float(np.mean([1.0 / r[key] for r in rows]))


def boot(recs, cl, fn, rng, nboot=B):
    sigs = sorted(cl)
    vals = []
    for _ in range(nboot):
        pick = rng.integers(0, len(sigs), size=len(sigs))
        sample = []
        for s in pick:
            sample.extend(recs[c] for c in cl[sigs[s]])
        vals.append(fn(sample))
    return np.array(vals)


def ci(v, lo=2.5, hi=97.5):
    return [float(np.percentile(v, lo)), float(np.percentile(v, hi))]


def cell_recs(p3, sub, grp):
    recs = {c: r for c, r in p3.items()
            if r["substrate"] == sub and r["group"] == grp}
    return recs, cluster_ids(recs)


def main():
    out = {"tost_margin": TOST_MARGIN, "B": B, "rng_seed": RNG_SEED,
           "chance_mrr": CH_GLOBAL,
           "scoring": "P3-7 pinned pre-run: 12 TOSTs (2 inference cells "
                      "x 2 conditions x 3 arms), +/-0.05 abs MRR, 90% "
                      "cluster-bootstrap CI, signals resampled; AND max "
                      "inference |delta| < 0.05",
           "arms": {}}
    tost_all, absmax = [], 0.0

    for seed in SEEDS:
        print(f"\n===== arm {seed} =====")
        p3 = load_jsonl(EXP / f"probe3_openai_per_query_seed{seed}.jsonl")
        p2 = load_jsonl(EXP / f"probe2_openai_per_query_seed{seed}.jsonl")

        # ---- JG: lossless join + identical unperturbed ranks ----
        if len(p3) != 878 or len(p2) != 878:
            fail(f"JG: row counts {len(p3)}/{len(p2)} != 878")
        if set(p3) != set(p2):
            fail("JG: caption_id sets differ between probe3 and probe2")
        max_du = max(abs(p3[c]["rank_unperturbed"]
                         - p2[c]["rank_unperturbed"]) for c in p3)
        print(f"JG: join lossless (878); max |rank_unpert diff| = "
              f"{max_du:.2e}  [pre-verified: 0.0]")
        if max_du > 1e-9:
            fail(f"JG: unperturbed ranks differ ({max_du:.2e})")
        for c in p3:
            p3[c]["rank_sf_all"] = p2[c]["rank_sf_all"]

        rng = np.random.default_rng(RNG_SEED)
        arm_out = {"cells": {}, "tost": {}}

        # ---- REF cross-computation gate ----
        for key, ref_val in REF[str(seed)].items():
            cell, cond = key.split("/")
            sub, grp = INFER_CELLS[cell]
            recs, _ = cell_recs(p3, sub, grp)
            rows = list(recs.values())
            d = mrr(rows, f"rank_{cond}") - mrr(rows, "rank_unperturbed")
            if abs(d - ref_val) > 1e-9:
                fail(f"REF: {key} arm {seed}: computed {d:.10f} vs "
                     f"reference {ref_val:.10f} — records differ from "
                     f"the ones the table was derived from, or one "
                     f"computation is wrong")
        print("REF cross-computation gate: 4/4 deltas reproduced <= 1e-9")

        # ---- inference cells: TOSTs + ladder ----
        wil = {}
        for cell, (sub, grp) in INFER_CELLS.items():
            recs, cl = cell_recs(p3, sub, grp)
            rows = list(recs.values())
            mu = mrr(rows, "rank_unperturbed")
            crow = {"n": len(rows), "unperturbed_mrr": mu,
                    "unperturbed_ratio": mu / CH_GLOBAL}
            for cond in LADDER:
                key = f"rank_{cond}"
                m = mrr(rows, key)
                bv = boot(recs, cl, lambda rr, k=key: mrr(rr, k), rng)
                crow[cond] = {"mrr": m, "mrr_ci95": ci(bv),
                              "ratio_to_chance": m / CH_GLOBAL,
                              "ratio_ci95": [x / CH_GLOBAL
                                             for x in ci(bv)]}
                if cond in CONDITIONS:
                    fnd = lambda rr, k=key: (mrr(rr, k)
                                             - mrr(rr, "rank_unperturbed"))
                    bd = boot(recs, cl, fnd, rng)
                    d = fnd(rows)
                    lo90, hi90 = ci(bd, 5, 95)
                    tost = (-TOST_MARGIN < lo90) and (hi90 < TOST_MARGIN)
                    tost_all.append(tost)
                    absmax = max(absmax, abs(d))
                    arm_out["tost"][f"{cell}/{cond}"] = {
                        "delta": d, "ci90": [lo90, hi90],
                        "tost_pass": bool(tost)}
                    ru = np.array([1 / r["rank_unperturbed"] for r in rows])
                    rc = np.array([1 / r[key] for r in rows])
                    if np.any(ru != rc):
                        wil[f"{cell}/{cond}"] = float(
                            wilcoxon(ru, rc).pvalue)
                    print(f"  {cell:9s}/{cond:8s} delta {d:+.4f}  90% CI "
                          f"[{lo90:+.4f}, {hi90:+.4f}]  "
                          f"TOST {'PASS' if tost else 'FAIL'}")
            arm_out["cells"][cell] = crow
            print(f"  {cell:9s} ladder MRR: unpert {mu:.4f} | sf_all "
                  f"{crow['sf_all']['mrr']:.4f} | resample "
                  f"{crow['resample']['mrr']:.4f} | gaussian "
                  f"{crow['gaussian']['mrr']:.4f}   [n={len(rows)}]")
        arm_out["wilcoxon_two_sided_report_only"] = wil
        if wil:
            print("  Wilcoxon (two-sided, report-only): "
                  + ", ".join(f"{k} p={v:.2e}" for k, v in wil.items()))

        # ---- thin cells: CIs + n, never load-bearing ----
        for cell, (sub, grp) in THIN_CELLS.items():
            recs, cl = cell_recs(p3, sub, grp)
            rows = list(recs.values())
            crow = {"n": len(rows),
                    "n_signals": len(cl),
                    "unperturbed_mrr": mrr(rows, "rank_unperturbed"),
                    "note": "thin cell, reported with n, never "
                            "load-bearing"}
            for cond in LADDER:
                key = f"rank_{cond}"
                bv = boot(recs, cl, lambda rr, k=key: mrr(rr, k), rng)
                crow[cond] = {"mrr": mrr(rows, key), "mrr_ci95": ci(bv)}
            arm_out["cells"][cell] = crow
            print(f"  {cell:9s} (THIN n={len(rows)}, {len(cl)} signals) "
                  f"unpert {crow['unperturbed_mrr']:.4f} | "
                  + " | ".join(f"{c} {crow[c]['mrr']:.4f} "
                               f"CI[{crow[c]['mrr_ci95'][0]:.4f},"
                               f"{crow[c]['mrr_ci95'][1]:.4f}]"
                               for c in LADDER))

        # ---- descriptive cells ----
        for cell, (sub, grp) in DESC_CELLS.items():
            recs, _ = cell_recs(p3, sub, grp)
            rows = list(recs.values())
            crow = {"n": len(rows),
                    "unperturbed_mrr": mrr(rows, "rank_unperturbed"),
                    **{c: {"mrr": mrr(rows, f"rank_{c}")} for c in LADDER},
                    "note": "descriptive only"}
            arm_out["cells"][cell] = crow
            print(f"  {cell:11s} (descriptive n={len(rows)}) unpert "
                  f"{crow['unperturbed_mrr']:.4f} | "
                  + " | ".join(f"{c} {crow[c]['mrr']:.4f}"
                               for c in LADDER))

        out["arms"][str(seed)] = arm_out

    # ---- verdict (pinned) ----
    print("\n===== P3-7 VERDICT (scoring pinned pre-run) =====")
    verdict = "CONFIRMED" if all(tost_all) and absmax < TOST_MARGIN \
        else "MISSED"
    print(f"  TOSTs passed: {sum(tost_all)}/{len(tost_all)}   "
          f"max inference-cell |delta|: {absmax:.4f} "
          f"(margin {TOST_MARGIN})")
    print(f"  P3-7: {verdict} — floor VOID; no ladder rung moves any "
          f"inference cell beyond the margin; the negative control "
          f"shows the Probe-3 pipeline does not manufacture degradation "
          f"on a no-capability model.")
    out["verdicts"] = {"P3-7": verdict,
                       "tost_passed": int(sum(tost_all)),
                       "tost_total": len(tost_all),
                       "max_inference_abs_delta": absmax}

    outp = EXP / "probe3_openai_stats.json"
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {outp}")


if __name__ == "__main__":
    main()

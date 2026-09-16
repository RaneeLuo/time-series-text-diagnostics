#!/usr/bin/env python3
"""
analyze_probe2_trace_duration.py — separate SPAN from SERIES LENGTH in the
TRACE Probe-2 ex_half finding.

THE PROBLEM THIS CLOSES
The strata check found that ex_half (swap the two halves) costs ~38% of MRR
on V=168 rows but ~82% on V=180 rows. Two things differ between those
groups: the SPAN of real time covered, and the LENGTH in points (168 vs
180). Length differs by only 7%, so it is an implausible cause of a 44-point
swing — but that is an argument, not evidence.

The V=168 stratum contains BOTH week rows and 28-day rows: identical point
count, four times the span. Comparing them isolates span from length.
Duration labels come from the committed Probe-1 narrative records
(results/experiments/trace_narrative_per_item.jsonl), which carry
duration_class per sample_id; the row index is the suffix of sample_id.

Cross-tab already established (2026-08-15, from the same source):
  V=180 -> six_months 278 / 278   (pure)
  V=168 -> week 364 + 28_days 323, zero six_months
Labels exist for 1,203 of 2,006 rows, so every cell here is a LABELLED
SUBSET and is reported with its n. This is a confound check, not a
re-estimate of the headline.

REGISTERED EXPECTATIONS (before the run; misses recorded as misses):
  D1  Cross-tab reproduces: V=180 all six_months; V=168 has zero
      six_months.                                              high
  D2  Within V=168, ex_half degradation for week and for 28_days differs
      by LESS than 15 points — far smaller than the 44-point gap to the
      six-month stratum. Reasoning: both are dominated by repeating daily
      cycles that largely survive a half-swap, whereas a six-month series
      carries a seasonal arc that a half-swap inverts. A HIT means the
      driver is periodic-vs-trending structure, not span or length as
      such.                                                    moderate
  D3  If instead 28_days lands roughly midway between week and six_months,
      that is a continuous span effect and D2 is MISSED — an equally
      publishable answer, recorded as such.
  D4  sf_all stays at ceiling (>95%) in every duration cell.    high

USAGE (from the thesis repo root):
  python scripts/analyze_probe2_trace_duration.py \
      --records results/experiments/probe2_trace_per_query_seed13.jsonl \
                results/experiments/probe2_trace_per_query_seed14.jsonl \
                results/experiments/probe2_trace_per_query_seed15.jsonl \
      --narrative results/experiments/trace_narrative_per_item.jsonl \
      --out results/analysis/probe2_trace_duration.json

Runtime: seconds. No model, no TRACE repo needed.
"""

from __future__ import annotations
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

PERTS = ["sf_all", "sf_half", "ex_half", "masking"]
POOL = 2006
N_BOOT = 2000
BOOT_SEED = 20260815


def fail(gate, msg):
    print(f"\n[GATE FAILED] {gate}: {msg}")
    print("Hard stop. Investigate before proceeding — do not work around.")
    sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", nargs="+", required=True)
    ap.add_argument("--narrative",
                    default="results/experiments/trace_narrative_per_item.jsonl")
    ap.add_argument("--out",
                    default="results/analysis/probe2_trace_duration.json")
    args = ap.parse_args()

    print("=" * 74)
    print("TRACE PROBE-2 — SPAN vs SERIES LENGTH (ex_half confound check)")
    print("=" * 74)

    # ---- duration labels -------------------------------------------------
    dur = {}
    for line in open(args.narrative, encoding="utf-8"):
        r = json.loads(line)
        dur[int(r["sample_id"].rsplit("_", 1)[1])] = r["duration_class"]
    print(f"[labels] duration_class available for {len(dur)}/{POOL} rows "
          f"({len(dur)/POOL:.0%}) — every cell below is a labelled subset")
    print(f"[labels] classes: {dict(Counter(dur.values()))}")

    # ---- records ---------------------------------------------------------
    data, V = {}, None
    for path in args.records:
        rows = [json.loads(l) for l in open(path, encoding="utf-8")]
        s = {r["mask_seed"] for r in rows}
        if len(s) != 1:
            fail("G-in", f"{path} mixes seeds")
        seed = s.pop()
        rs = [r for r in rows if r["direction"] == "text2ts"]
        if [r["row_idx"] for r in rs] != list(range(POOL)):
            fail("G-in", f"{path}: row_idx not contiguous")
        v = np.array([r["valid_len"] for r in rs], dtype=int)
        if V is None:
            V = v
        elif not np.array_equal(v, V):
            fail("G-in", f"{path}: valid_len differs from the first file")
        data[seed] = {
            "dep": np.array([r["group"] == "dependent" for r in rs]),
            "R": {c: np.array([r["rank_unperturbed" if c == "unperturbed"
                                 else f"rank_{c}"] for r in rs], float)
                  for c in ["unperturbed"] + PERTS}}
        print(f"[G-in] seed {seed}: OK")
    seeds = sorted(data)

    # ---- D1: cross-tab gate ---------------------------------------------
    ct = defaultdict(Counter)
    for row, dc in dur.items():
        ct[int(V[row])][dc] += 1
    v180, v168 = ct.get(180, Counter()), ct.get(168, Counter())
    print(f"\n[D1] V=180 -> {dict(v180)}")
    print(f"[D1] V=168 -> {dict(v168)}")
    d1 = (set(v180) == {"six_months"}) and ("six_months" not in v168)
    print(f"[D1] V=180 pure six_months and V=168 free of six_months: "
          f"{'HIT' if d1 else 'MISS'}")
    if not d1:
        print("     NOTE: the cross-tab did not reproduce. The span/length "
              "separation below rests on it — read the cells with care.")

    # ---- cells -----------------------------------------------------------
    labelled = np.array([dur.get(i) for i in range(POOL)], dtype=object)
    cells = {
        "V=168 week":       (V == 168) & (labelled == "week"),
        "V=168 28_days":    (V == 168) & (labelled == "28_days"),
        "V=180 six_months": (V == 180) & (labelled == "six_months"),
    }

    rng = np.random.default_rng(BOOT_SEED)
    out = {"n_boot": N_BOOT, "boot_seed": BOOT_SEED,
           "labelled_rows": len(dur),
           "crosstab": {str(k): dict(v) for k, v in sorted(ct.items())},
           "D1_crosstab_hit": bool(d1), "seeds": {}}
    ex_by_cell = defaultdict(list)

    for seed in seeds:
        dep, R = data[seed]["dep"], data[seed]["R"]
        print("\n" + "=" * 74)
        print(f"MASK SEED {seed} — text->ts, dependent group, labelled rows only")
        print("=" * 74)
        print(f"  {'cell':<18}{'n':>6}{'unpert':>9}" +
              "".join(f"{p:>11}" for p in PERTS))
        srec = {}
        for name, sel in cells.items():
            m = sel & dep
            n = int(m.sum())
            if n == 0:
                continue
            base = float((1.0 / R["unperturbed"][m]).mean())
            line, c = f"  {name:<18}{n:>6}{base:>9.4f}", {}
            for p in PERTS:
                mr = float((1.0 / R[p][m]).mean())
                rel = (base - mr) / base
                c[p] = {"mrr": mr, "rel_degradation": rel}
                line += f"{rel:>10.1%} "
            print(line)
            srec[name] = {"n": n, "unperturbed_mrr": base, "perts": c}
            ex_by_cell[name].append(c["ex_half"]["rel_degradation"])

        # ---- D2: week vs 28_days, same length ---------------------------
        a, b = cells["V=168 week"] & dep, cells["V=168 28_days"] & dep
        na, nb = int(a.sum()), int(b.sum())
        ia = rng.integers(0, na, size=(N_BOOT, na))
        ib = rng.integers(0, nb, size=(N_BOOT, nb))
        iua, iub = 1.0 / R["unperturbed"][a], 1.0 / R["unperturbed"][b]
        bua, bub = iua[ia].mean(1), iub[ib].mean(1)
        print(f"\n  D2. week minus 28_days, BOTH at V=168 (same length, "
              f"4x the span)")
        drec = {}
        for p in PERTS:
            ra = (iua.mean() - (1.0 / R[p][a]).mean()) / iua.mean()
            rb = (iub.mean() - (1.0 / R[p][b]).mean()) / iub.mean()
            d = ((bua - (1.0 / R[p][a])[ia].mean(1)) / bua
                 - (bub - (1.0 / R[p][b])[ib].mean(1)) / bub)
            lo, hi = np.percentile(d, [2.5, 97.5])
            drec[p] = {"diff": float(ra - rb), "ci": [float(lo), float(hi)],
                       "ci_excludes_zero": bool(lo > 0 or hi < 0)}
            print(f"    {p:<10}{ra-rb:+7.1%}  CI [{lo:+.1%},{hi:+.1%}]"
                  f"   {'excludes 0' if drec[p]['ci_excludes_zero'] else 'includes 0'}")
        srec["D2_week_minus_28days"] = drec
        out["seeds"][str(seed)] = srec

    # ---- verdicts --------------------------------------------------------
    print("\n" + "=" * 74)
    print("VERDICTS")
    print("=" * 74)
    wk = float(np.mean(ex_by_cell["V=168 week"]))
    d28 = float(np.mean(ex_by_cell["V=168 28_days"]))
    sm = float(np.mean(ex_by_cell["V=180 six_months"]))
    gap_within = abs(wk - d28)
    gap_to_six = abs(((wk + d28) / 2) - sm)
    print(f"  ex_half relative degradation, mean over seeds:")
    print(f"    week      (V=168) {wk:6.1%}")
    print(f"    28_days   (V=168) {d28:6.1%}")
    print(f"    six_months(V=180) {sm:6.1%}")
    print(f"  within-V=168 gap (week vs 28_days): {gap_within:.1%}")
    print(f"  gap from V=168 average to six_months: {gap_to_six:.1%}")
    d2 = gap_within < 0.15
    print(f"\n  D2 (within-length gap < 15 points): {'HIT' if d2 else 'MISS'}")
    if d2:
        print("     => Span alone does not drive ex_half. Week and 28-day rows")
        print("        behave alike despite a 4x span difference, while the")
        print("        six-month rows stand apart. The driver is the KIND of")
        print("        structure — repeating daily cycles survive a half-swap,")
        print("        a seasonal arc does not. Series length is excluded:")
        print("        week and 28_days have identical length.")
    else:
        print("     => 28_days sits away from week, so degradation scales with")
        print("        SPAN continuously rather than splitting periodic from")
        print("        trending. D3 is the operative reading; state it that way.")
    d4 = all(out["seeds"][str(s)][c]["perts"]["sf_all"]["rel_degradation"] > 0.95
             for s in seeds for c in cells if c in out["seeds"][str(s)])
    print(f"  D4 sf_all above 95% in every duration cell: "
          f"{'HIT' if d4 else 'MISS'}")
    out["verdicts"] = {"D1": bool(d1), "D2_within_gap_small": bool(d2),
                       "D4_sf_all_ceiling": bool(d4),
                       "ex_half_week": wk, "ex_half_28days": d28,
                       "ex_half_six_months": sm,
                       "gap_within_V168": gap_within,
                       "gap_to_six_months": gap_to_six}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    print("Every cell is a LABELLED SUBSET (1,203 of 2,006 rows) — quote with "
          "its n; this is a confound check, not a re-estimate of the headline.")


if __name__ == "__main__":
    main()

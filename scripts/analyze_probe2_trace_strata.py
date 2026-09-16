#!/usr/bin/env python3
"""
analyze_probe2_trace_strata.py — does the TRACE Probe-2 profile hold within
duration strata, or is it an average over heterogeneous rows?

WHY
This project's standing rule is that length confounds are closed by DATA,
not argument. The narrative arm (Probe 1) found a strong duration gradient
on N3 — week 0.619 -> 28_days 0.736 -> six_months 0.852 — so a duration
effect on Probe-2 degradation is plausible on its face. The valid lengths
cluster hard at two values: V=168 (1,051 rows) and V=180 (544 rows), with
411 rows spread thin elsewhere.

READ V CAREFULLY. V is the number of valid TIMESTEPS, not a duration. From
the NOAA layout, V=168 is most likely 7 days at hourly resolution and V=180
is most likely ~6 months at daily resolution — so the two strata differ in
BOTH span and sampling rate, and this script cannot separate them. Any
finding here is "the V=168 stratum differs from the V=180 stratum", never
"longer series degrade more". That distinction goes in the write-up.

WHAT IT COMPUTES (per seed, primary direction text->ts, dependent group)
  1. Per-stratum unperturbed MRR and per-perturbation relative degradation.
  2. Whether the severity ordering survives WITHIN each stratum — the real
     robustness question. An ordering that only exists in the pooled average
     is an artifact of mixing populations.
  3. Spearman correlation between V and a per-query degradation measure
     (log rank ratio), which uses all 57 distinct lengths rather than
     collapsing to strata.
  4. Bootstrap CI on the BETWEEN-STRATUM difference in relative degradation.
  5. Optional (--trace-repo): a fresh 25 s data scan giving per-row valid
     length and dead-channel count. This (a) GATES the records' valid_len
     against source, and (b) tests whether rows containing dead channels
     behave differently — a dead channel is a no-op under every perturbation
     here, so rows carrying them have fewer effective channels.

Per-query degradation measure: log(rank_pert) - log(rank_unpert). Ranks are
heavily skewed, so a log ratio is the sane paired statistic; a raw rank
difference would be dominated by whichever queries started near rank 1.

REGISTERED EXPECTATIONS (before the run; misses recorded as misses):
  S1  If --trace-repo is given: freshly scanned V matches the records'
      valid_len for all 2,006 rows, exactly.                    high
  S2  Severity ordering sf_all > sf_half > masking > ex_half holds within
      BOTH main strata, in all three seeds.                     mod-high
  S3  Unperturbed MRR differs between the two strata, with V=180 higher
      (the narrative arm found longer-span rows easier).        moderate
  S4  sf_all relative degradation exceeds 95% in both strata — it is at
      ceiling, so little room for a stratum effect.             high
  S5  ex_half degradation is LARGER in the V=180 stratum. Reasoning: a
      ~6-month daily series carries a seasonal arc, and swapping its halves
      inverts that arc; a week of hourly data is dominated by repeating
      diurnal cycles, which partly survive a half-swap.         mod-LOW
  S6  Rows containing dead channels have LOWER unperturbed MRR (less
      information available).                                   moderate
  No prediction is registered on the Spearman magnitudes.

USAGE (from the thesis repo root):
  python scripts/analyze_probe2_trace_strata.py \
      --records results/experiments/probe2_trace_per_query_seed13.jsonl \
                results/experiments/probe2_trace_per_query_seed14.jsonl \
                results/experiments/probe2_trace_per_query_seed15.jsonl \
      --trace-repo ../TRACE-Multimodal-TSEncoder \
      --out results/analysis/probe2_trace_strata.json

Runtime: seconds without --trace-repo, about a minute with it.
"""

from __future__ import annotations
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

PERTS = ["sf_all", "sf_half", "ex_half", "masking"]
POOL = 2006
N_BOOT = 2000
BOOT_SEED = 20260815
MAIN_STRATA = [168, 180]


def fail(gate, msg):
    print(f"\n[GATE FAILED] {gate}: {msg}")
    print("Hard stop. Investigate before proceeding — do not work around.")
    sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", nargs="+", required=True)
    ap.add_argument("--trace-repo", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="results/analysis/probe2_trace_strata.json")
    args = ap.parse_args()

    try:
        from scipy.stats import spearmanr
    except ImportError:
        fail("dep", "scipy is required")

    print("=" * 78)
    print("TRACE PROBE-2 — DURATION / LENGTH STRATA CHECK")
    print("=" * 78)
    print("V is valid TIMESTEPS, not duration. V=168 and V=180 differ in span")
    print("AND sampling rate; this script cannot separate the two.")

    # ---- load ------------------------------------------------------------
    data, V_rec = {}, None
    for path in args.records:
        rows = [json.loads(l) for l in open(path, encoding="utf-8")]
        seed = {r["mask_seed"] for r in rows}
        if len(seed) != 1:
            fail("G-in", f"{path} mixes seeds")
        seed = seed.pop()
        rs = [r for r in rows if r["direction"] == "text2ts"]
        if [r["row_idx"] for r in rs] != list(range(POOL)):
            fail("G-in", f"{path}: text2ts row_idx not contiguous")
        v = np.array([r["valid_len"] for r in rs], dtype=int)
        if V_rec is None:
            V_rec = v
        elif not np.array_equal(v, V_rec):
            fail("G-in", f"{path}: valid_len differs from the first seed file")
        data[seed] = {
            "dep": np.array([r["group"] == "dependent" for r in rs]),
            "ranks": {c: np.array([r["rank_unperturbed" if c == "unperturbed"
                                     else f"rank_{c}"] for r in rs], float)
                      for c in ["unperturbed"] + PERTS}}
        print(f"[G-in] seed {seed}: {len(rs)} text2ts rows — OK")
    seeds = sorted(data)

    # ---- optional fresh scan: gate valid_len + dead channels -------------
    dead_cnt = None
    if args.trace_repo:
        print("\n--- fresh data scan (gates valid_len, counts dead channels) ---")
        repo = Path(args.trace_repo).resolve()
        import os
        os.environ["TTRAG_DATA_DIR"] = str(repo / "dataset") + "/"
        os.environ["TTRAG_CHECKPOINTS_DIR"] = str(repo / "results/model_checkpoints") + "/"
        os.environ["TTRAG_RESULTS_DIR"] = str(repo / "results/model_results") + "/"
        sys.path.insert(0, str(repo))
        import torch
        from tqdm import tqdm
        from src.data.dataloader import get_dataloader
        ck = torch.load(repo / "results/model_checkpoints/context_align/retriever_demo.pt",
                        map_location="cpu", weights_only=False)
        margs = ck["args"]
        del ck
        margs.task_name = "retrieval"
        margs.data_split = args.split
        margs.batch_size = 32
        margs.device = torch.device("cpu")
        margs.distributed = False
        margs.rank = 0
        loader = get_dataloader(margs)
        V_src = np.zeros(POOL, dtype=int)
        dead_cnt = np.zeros(POOL, dtype=int)
        r = 0
        for b in tqdm(loader, total=len(loader), desc="scan"):
            x = b.timeseries.float().numpy()
            m = b.input_mask.numpy() > 0.5
            for i in range(x.shape[0]):
                idx = np.flatnonzero(m[i][0])
                V = len(idx)
                V_src[r] = V
                blk = x[i][:, 186 - V:]
                dead_cnt[r] = int(((blk.max(axis=1) - blk.min(axis=1)) == 0.0).sum())
                r += 1
        if r != POOL:
            fail("S1", f"scan saw {r} rows != {POOL}")
        mism = int((V_src != V_rec).sum())
        print(f"[S1] valid_len: records vs fresh scan — mismatches {mism}/{POOL}")
        if mism:
            fail("S1", "the records' valid_len disagrees with source — the "
                       "record files do not correspond to this dataset")
        print(f"[S1] PASSED (exact match, all {POOL} rows)")
        print(f"  dead channels total {dead_cnt.sum()} in "
              f"{int((dead_cnt>0).sum())} rows (expect 498 in 413)")
        if dead_cnt.sum() != 498 or int((dead_cnt > 0).sum()) != 413:
            fail("S1-dead", "dead-channel counts disagree with the runner")

    # ---- strata ----------------------------------------------------------
    vc = Counter(V_rec.tolist())
    print(f"\nvalid-length strata: " +
          ", ".join(f"V={v} n={vc[v]}" for v in MAIN_STRATA) +
          f", other n={POOL - sum(vc[v] for v in MAIN_STRATA)}")

    def strata_of(dep):
        s = {}
        for v in MAIN_STRATA:
            s[f"V={v}"] = dep & (V_rec == v)
        s["other"] = dep & ~np.isin(V_rec, MAIN_STRATA)
        return s

    rng = np.random.default_rng(BOOT_SEED)
    out = {"strata": [f"V={v}" for v in MAIN_STRATA] + ["other"],
           "n_boot": N_BOOT, "boot_seed": BOOT_SEED,
           "caveat": "V confounds span with sampling rate; not a duration claim",
           "seeds": {}}
    orders = {}

    for seed in seeds:
        R = data[seed]["ranks"]
        strata = strata_of(data[seed]["dep"])
        print("\n" + "=" * 78)
        print(f"MASK SEED {seed} — text->ts, dependent group, by stratum")
        print("=" * 78)
        print(f"  {'stratum':<10}{'n':>6}{'unpert MRR':>12}" +
              "".join(f"{p:>12}" for p in PERTS))
        srec = {}
        for name, sel in strata.items():
            n = int(sel.sum())
            if n == 0:
                continue
            base = float((1.0 / R["unperturbed"][sel]).mean())
            cells, line = {}, f"  {name:<10}{n:>6}{base:>12.4f}"
            for p in PERTS:
                m = float((1.0 / R[p][sel]).mean())
                rel = (base - m) / base
                cells[p] = {"mrr": m, "rel_degradation": rel}
                line += f"{rel:>11.1%} "
            print(line)
            order = sorted(PERTS, key=lambda p: -cells[p]["rel_degradation"])
            orders[(seed, name)] = order
            srec[name] = {"n": n, "unperturbed_mrr": base, "perts": cells,
                          "severity_order": order}
        print("  (columns are RELATIVE degradation vs that stratum's own "
              "unperturbed baseline)")

        print("\n  severity ordering within each stratum:")
        for name in srec:
            same = orders[(seed, name)] == ["sf_all", "sf_half", "masking",
                                            "ex_half"]
            print(f"    {name:<10} {' > '.join(orders[(seed,name)])}"
                  f"   {'(matches pooled)' if same else '<-- DIFFERS'}")

        # between-stratum difference in relative degradation, bootstrapped
        a, b = strata[f"V={MAIN_STRATA[0]}"], strata[f"V={MAIN_STRATA[1]}"]
        na, nb = int(a.sum()), int(b.sum())
        ia = rng.integers(0, na, size=(N_BOOT, na))
        ib = rng.integers(0, nb, size=(N_BOOT, nb))
        print(f"\n  between-stratum difference, V={MAIN_STRATA[0]} minus "
              f"V={MAIN_STRATA[1]} (relative degradation, 95% CI)")
        diffs = {}
        inv_ua, inv_ub = 1.0 / R["unperturbed"][a], 1.0 / R["unperturbed"][b]
        bua, bub = inv_ua[ia].mean(1), inv_ub[ib].mean(1)
        for p in PERTS:
            ra = (inv_ua.mean() - (1.0 / R[p][a]).mean()) / inv_ua.mean()
            rb = (inv_ub.mean() - (1.0 / R[p][b]).mean()) / inv_ub.mean()
            bra = (bua - (1.0 / R[p][a])[ia].mean(1)) / bua
            brb = (bub - (1.0 / R[p][b])[ib].mean(1)) / bub
            d = bra - brb
            lo, hi = np.percentile(d, [2.5, 97.5])
            excl = (lo > 0) or (hi < 0)
            diffs[p] = {"diff": float(ra - rb), "ci": [float(lo), float(hi)],
                        "ci_excludes_zero": bool(excl)}
            print(f"    {p:<10} {ra-rb:+7.1%}  CI [{lo:+.1%},{hi:+.1%}]"
                  f"   {'CI excludes 0' if excl else 'CI includes 0'}")
        srec["between_stratum"] = diffs

        # Spearman over all 57 lengths, per-query log rank ratio
        dep = data[seed]["dep"]
        print(f"\n  Spearman(V, log rank ratio) over all "
              f"{len(set(V_rec[dep].tolist()))} distinct lengths, n={int(dep.sum())}")
        sp = {}
        for p in PERTS:
            lr = np.log(R[p][dep]) - np.log(R["unperturbed"][dep])
            Vd = V_rec[dep].astype(float)
            degenerate = (np.ptp(lr) < 1e-12) or (np.ptp(Vd) < 1e-12)
            rho = pv = float("nan")
            if not degenerate:
                rho, pv = spearmanr(Vd, lr)
            if degenerate or np.isnan(rho):
                sp[p] = {"rho": None, "p": None,
                         "note": "undefined — input effectively constant"}
                print(f"    {p:<10} undefined (the log ratio is effectively "
                      f"constant across queries)")
                continue
            sp[p] = {"rho": float(rho), "p": float(pv)}
            print(f"    {p:<10} rho {rho:+.3f}  p {pv:.2e}")
        srec["spearman_V_vs_logratio"] = sp

        if dead_cnt is not None:
            hd = dep & (dead_cnt > 0)
            nd = dep & (dead_cnt == 0)
            mh = float((1.0 / R["unperturbed"][hd]).mean())
            mn = float((1.0 / R["unperturbed"][nd]).mean())
            print(f"\n  dead channels: rows WITH n={int(hd.sum())} unpert MRR "
                  f"{mh:.4f} | rows WITHOUT n={int(nd.sum())} MRR {mn:.4f}"
                  f"  (S6 expects WITH lower: {mh < mn})")
            srec["dead_channel_split"] = {
                "with_n": int(hd.sum()), "with_mrr": mh,
                "without_n": int(nd.sum()), "without_mrr": mn,
                "s6_hit": bool(mh < mn)}
        out["seeds"][str(seed)] = srec

    # ---- verdicts ---------------------------------------------------------
    print("\n" + "=" * 78)
    print("VERDICTS")
    print("=" * 78)
    pooled = ["sf_all", "sf_half", "masking", "ex_half"]
    s2 = all(orders[(s, f"V={v}")] == pooled for s in seeds for v in MAIN_STRATA)
    print(f"  S2 ordering holds in both main strata, all seeds: "
          f"{'HIT' if s2 else 'MISS'}")
    b0 = [out["seeds"][str(s)][f"V={MAIN_STRATA[0]}"]["unperturbed_mrr"] for s in seeds]
    b1 = [out["seeds"][str(s)][f"V={MAIN_STRATA[1]}"]["unperturbed_mrr"] for s in seeds]
    print(f"  S3 unperturbed MRR V={MAIN_STRATA[0]} {np.mean(b0):.4f} vs "
          f"V={MAIN_STRATA[1]} {np.mean(b1):.4f} -> "
          f"{'HIT' if np.mean(b1) > np.mean(b0) else 'MISS'} (predicted 180 higher)")
    s4 = all(out["seeds"][str(s)][f"V={v}"]["perts"]["sf_all"]["rel_degradation"] > 0.95
             for s in seeds for v in MAIN_STRATA)
    print(f"  S4 sf_all > 95% in both strata, all seeds: {'HIT' if s4 else 'MISS'}")
    e0 = np.mean([out["seeds"][str(s)][f"V={MAIN_STRATA[0]}"]["perts"]["ex_half"]["rel_degradation"] for s in seeds])
    e1 = np.mean([out["seeds"][str(s)][f"V={MAIN_STRATA[1]}"]["perts"]["ex_half"]["rel_degradation"] for s in seeds])
    print(f"  S5 ex_half V={MAIN_STRATA[0]} {e0:+.1%} vs V={MAIN_STRATA[1]} "
          f"{e1:+.1%} -> {'HIT' if e1 > e0 else 'MISS'} (predicted 180 larger)")
    if dead_cnt is not None:
        s6 = all(out["seeds"][str(s)]["dead_channel_split"]["s6_hit"] for s in seeds)
        print(f"  S6 dead-channel rows have lower unperturbed MRR: "
              f"{'HIT' if s6 else 'MISS'}")
    out["verdicts"] = {"S2": bool(s2), "S4": bool(s4),
                       "S3_180_higher": bool(np.mean(b1) > np.mean(b0)),
                       "S5_180_larger_exhalf": bool(e1 > e0)}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    print("Reminder: any stratum finding is 'V=168 differs from V=180', "
          "never 'longer series degrade more' — span and sampling rate are "
          "confounded in V.")


if __name__ == "__main__":
    main()

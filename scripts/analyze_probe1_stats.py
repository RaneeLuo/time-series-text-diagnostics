"""
analyze_probe1_stats.py — turn Probe-1 point estimates into claims.

Three statistical decisions drive everything here, and each has a reason:

1. THE UNIT OF ANALYSIS IS THE SIGNAL, NOT THE ITEM.
   The 5,540 items rest on 279 signals -- roughly 20 items per signal, sharing
   the same time series and often the same correct caption. Treating items as
   independent would inflate the effective sample size twentyfold and produce
   confidence intervals several times too narrow. Every resample here draws
   SIGNALS with replacement and pools whatever items those signals carry.

2. THE PRIMARY TEST IS PAIRED.
   Each swap item has a matched random item built from the same signal and the
   same correct caption, differing only in which distractor was used. The
   quantity tested is the within-signal difference between the two conditions,
   which removes signal-level variation entirely.

3. ABSENCE OF DEGRADATION IS TESTED, NOT ASSUMED.
   A non-significant gap is not evidence that a component is encoded. Where the
   gap is small (C2, C5) the claim is made with two one-sided tests against a
   pre-registered equivalence margin: the 90% interval must lie entirely inside
   +/- margin. State the margin before looking at the data; the default of 0.05
   comes from the baseline's own seed-to-seed variation.

Reported per seed, because the project's binding decision is that significance
comes from paired tests within a seed and replication from agreement across the
three seeds. Holm-Bonferroni corrects across the five components within a seed.

Run from repo root:
    python scripts/analyze_probe1_stats.py
    python scripts/analyze_probe1_stats.py --margin 0.05 --n-boot 10000

Writes: results/experiments/probe1_statistics.json
"""

from __future__ import annotations
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

PER_ITEM = Path("results/experiments/probe1_clasp_per_item.jsonl")
OUT = Path("results/experiments/probe1_statistics.json")
CHANCE = 0.5


def load(path=PER_ITEM):
    recs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            recs.append(json.loads(line))
    return recs


def per_signal_counts(recs):
    """signal -> [n_correct_swap, n_swap, n_correct_random, n_random]"""
    d = defaultdict(lambda: [0, 0, 0, 0])
    for r in recs:
        i = 0 if r["condition"] == "swap" else 2
        d[r["sample_id"]][i] += int(r["correct"])
        d[r["sample_id"]][i + 1] += 1
    return d


def bootstrap(counts: dict, n_boot: int, rng) -> dict:
    """Resample SIGNALS with replacement; pool their items each time."""
    sig = sorted(counts)
    arr = np.array([counts[s] for s in sig], dtype=float)   # (S,4)
    n = len(sig)
    idx = np.arange(n)

    acc_s, acc_r, gaps = [], [], []
    for _ in range(n_boot):
        take = rng.choice(idx, size=n, replace=True)
        a = arr[take].sum(axis=0)
        if a[1] == 0 or a[3] == 0:
            continue
        s, r = a[0] / a[1], a[2] / a[3]
        acc_s.append(s)
        acc_r.append(r)
        gaps.append(r - s)

    tot = arr.sum(axis=0)
    point_s, point_r = tot[0] / tot[1], tot[2] / tot[3]

    def ci(v, lo=2.5, hi=97.5):
        return float(np.percentile(v, lo)), float(np.percentile(v, hi))

    gaps = np.array(gaps)
    return {
        "n_signals": n,
        "acc_swap": float(point_s), "acc_swap_ci95": ci(acc_s),
        "acc_random": float(point_r), "acc_random_ci95": ci(acc_r),
        "gap": float(point_r - point_s), "gap_ci95": ci(gaps),
        "gap_ci90": ci(gaps, 5, 95),
        # two-sided bootstrap p for H0: gap = 0
        "p_gap": float(2 * min((gaps <= 0).mean(), (gaps >= 0).mean())),
        # is swap accuracy above chance at all?
        "p_swap_vs_chance": float(2 * min((np.array(acc_s) <= CHANCE).mean(),
                                          (np.array(acc_s) >= CHANCE).mean())),
        "acc_swap_above_chance": bool(np.percentile(acc_s, 2.5) > CHANCE),
    }


def holm(pvals: dict) -> dict:
    """Holm-Bonferroni across components within a seed."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, prev = {}, 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, max(prev, (m - i) * p))
        out[k] = adj
        prev = adj
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--margin", type=float, default=0.05,
                    help="pre-registered equivalence margin for TOST")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--per-item", default=str(PER_ITEM),
                    help="per-item JSONL from any model's Probe-1 runner")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    try:
        from scipy.stats import wilcoxon
    except ImportError:
        wilcoxon = None

    recs = load(args.per_item)
    seeds = sorted({r["seed"] for r in recs})
    comps = sorted({r["component"] for r in recs})
    print(f"records: {len(recs)}   seeds: {seeds}   components: {len(comps)}")
    print(f"equivalence margin (pre-registered): +/-{args.margin}")
    print(f"bootstrap resamples: {args.n_boot}, resampling SIGNALS\n")

    rng = np.random.default_rng(args.seed)
    results, raw_p = {}, defaultdict(dict)

    for seed in seeds:
        for comp in comps:
            sub = [r for r in recs if r["seed"] == seed and r["component"] == comp]
            counts = per_signal_counts(sub)
            st = bootstrap(counts, args.n_boot, rng)

            # paired signal-level test on the accuracy difference
            diffs = []
            for s, (cs, ns, cr, nr) in counts.items():
                if ns and nr:
                    diffs.append(cr / nr - cs / ns)
            diffs = np.array(diffs)
            if wilcoxon is not None and np.any(diffs != 0):
                try:
                    st["wilcoxon_p"] = float(wilcoxon(diffs).pvalue)
                except ValueError:
                    st["wilcoxon_p"] = None
            else:
                st["wilcoxon_p"] = None
            st["mean_signal_level_gap"] = float(diffs.mean())

            # TOST: 90% CI entirely inside +/- margin
            lo, hi = st["gap_ci90"]
            st["equivalence_margin"] = args.margin
            st["equivalent_to_zero"] = bool(lo > -args.margin and hi < args.margin)

            results[f"seed{seed}|{comp}"] = st
            raw_p[seed][comp] = st["p_gap"]

    # Holm correction within each seed
    for seed in seeds:
        adj = holm(raw_p[seed])
        for comp, p in adj.items():
            results[f"seed{seed}|{comp}"]["p_gap_holm"] = p
            results[f"seed{seed}|{comp}"]["significant"] = bool(p < args.alpha)

    # ------------------------------------------------------------------ print
    print("=" * 100)
    print("PER-COMPONENT RESULTS (bootstrap over signals; Holm-corrected across components)")
    print("=" * 100)
    hdr = (f"{'component':<24}{'seed':>5}{'swap':>8}{'95% CI':>16}"
           f"{'gap':>8}{'95% CI':>16}{'p(Holm)':>10}{'verdict':>12}")
    for comp in comps:
        print("-" * 100)
        print(hdr if comp == comps[0] else "")
        for seed in seeds:
            st = results[f"seed{seed}|{comp}"]
            cs = f"[{st['acc_swap_ci95'][0]:.3f},{st['acc_swap_ci95'][1]:.3f}]"
            cg = f"[{st['gap_ci95'][0]:+.3f},{st['gap_ci95'][1]:+.3f}]"
            near_chance = (st["acc_random_ci95"][0] < 0.60
                           and st["acc_swap_ci95"][0] < 0.60)
            st["void"] = bool(near_chance)
            if near_chance:
                verdict = "VOID"          # no capability to degrade from
            elif st["significant"] and st["gap"] > 0:
                verdict = "DEGRADED"
            elif st["equivalent_to_zero"]:
                verdict = "equivalent"
            elif st["significant"]:
                verdict = "improved"
            else:
                verdict = "inconclusive"
            print(f"{comp:<24}{seed:>5}{st['acc_swap']:>8.3f}{cs:>16}"
                  f"{st['gap']:>+8.3f}{cg:>16}{st['p_gap_holm']:>10.4f}{verdict:>12}")

    # chance check
    print("\n" + "=" * 100)
    print("IS SWAP ACCURACY ABOVE CHANCE (0.500)?")
    print("=" * 100)
    for comp in comps:
        marks = []
        for seed in seeds:
            st = results[f"seed{seed}|{comp}"]
            marks.append(f"seed{seed}: {st['acc_swap']:.3f} "
                         f"{'>chance' if st['acc_swap_above_chance'] else 'NOT above chance'}")
        print(f"{comp:<24}" + "   ".join(marks))

    # replication
    print("\n" + "=" * 100)
    if len(seeds) == 1:
        print("SINGLE RUN — NO REPLICATION AVAILABLE")
        print("=" * 100)
        print(f"Only one run ({seeds[0]}) is present. API models have no random")
        print("seed, so cross-seed replication does not apply and must not be")
        print("claimed. Confidence intervals below still hold: they quantify")
        print("uncertainty over SIGNALS, which is the dominant source here.")
    else:
        print("REPLICATION ACROSS SEEDS")
        print("=" * 100)
    summary = {}
    single = len(seeds) == 1
    for comp in comps:
        sig = [results[f"seed{s}|{comp}"]["significant"] for s in seeds]
        eq = [results[f"seed{s}|{comp}"]["equivalent_to_zero"] for s in seeds]
        void = [results[f"seed{s}|{comp}"].get("void", False) for s in seeds]
        gaps = [results[f"seed{s}|{comp}"]["gap"] for s in seeds]
        suffix = "" if single else " in all seeds"

        if all(void):
            claim = ("VOID — both conditions near chance; the gap carries no "
                     "information about shortcuts")
        elif all(sig) and np.mean(gaps) > 0:
            claim = "degradation" + (" (single run, not replicated)" if single else suffix)
        elif all(eq):
            claim = f"equivalent to zero within +/-{args.margin}" + \
                    (" (single run)" if single else suffix)
        elif all(sig):
            claim = "gap significantly negative" + \
                    (" (single run, not replicated)" if single else suffix)
        else:
            claim = "inconclusive" + ("" if single else " — mixed across seeds")

        sd = float(np.std(gaps, ddof=1)) if len(gaps) > 1 else None
        sd_txt = f"+/- {sd:.3f}" if sd is not None else "(single run)"
        print(f"{comp:<24}gap {np.mean(gaps):+.3f} {sd_txt}   {claim}")
        summary[comp] = {"gap_mean": float(np.mean(gaps)), "gap_sd": sd,
                         "n_runs": len(seeds), "void": bool(all(void)),
                         "claim": claim}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"n_boot": args.n_boot, "equivalence_margin": args.margin,
                   "alpha": args.alpha, "unit_of_resampling": "signal",
                   "per_seed_component": results, "replication": summary}, f, indent=2)
    print(f"\nsaved -> {out_path}")

    print("\n" + "=" * 100)
    print("READING THE VERDICT COLUMN")
    print("=" * 100)
    print("DEGRADED     swap significantly worse than random -> the model relies on")
    print("             surface cues for this component; shortcut evidence")
    print("equivalent   gap's 90% interval lies inside the margin -> positively")
    print("             established as no meaningful degradation, not merely 'n.s.'")
    print("improved     swap significantly EASIER than random (expected for C5,")
    print("             where the swap is a larger semantic change than the average")
    print("             random distractor)")
    print("inconclusive neither significant nor equivalent -> underpowered; say so")
    print("VOID         BOTH conditions near chance -> the model cannot do the task")
    print("             at all, so its gap carries no information about shortcuts.")
    print("             A shortcut claim REQUIRES high random-condition accuracy.")


if __name__ == "__main__":
    main()

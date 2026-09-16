#!/usr/bin/env python3
"""
regrade_chatts_c4_census.py — re-grade ChatTS's Probe-1 C4 result on the
human-census-certified item partition (2026-09-05).

WHY. CLaSP's C4 headline (0.603) is computed on the 738 census-certified C4
items (probe1_manual_validation_findings.md §5–6). ChatTS's C4 result
(0.6576 vs 0.7545) was scored on all 990 C4 items (chatts_analysis.json:
n_units 990). Chapter 5 sets the two against each other, so they must rest
on the same partition. This is a RE-GRADE of stored per-item answers, not a
re-run: the ChatTS responses are read from the committed jsonl.

WHAT IT DOES. Same statistics as scripts/analyze_chatts_probes.py, applied
per partition:
  unit accuracy = mean over the two answer orders (0 / 0.5 / 1)
  cells         = signal-cluster bootstrap on sample_id, B=2000, 95% CI
  gap           = random − swap, paired per item, 95% CI + 90%-CI TOST ±0.05
  Wilcoxon (two-sided) on the paired diffs; McNemar on strict binarisation
Partitions: census_valid (738), census_invalid (125), not_censused (127 =
cross-slot 20 + generic 107; split only if data/processed/probe1_items.jsonl
is present locally), all_990.

READS  results/analysis/pinning_spotcheck_sequential.csv   (filled census)
       results/experiments/chatts_probe1_responses.jsonl    (11,080 rows)
       results/experiments/chatts_analysis.json             (reproduction gate)
       data/processed/probe1_items.jsonl                    (OPTIONAL, laptop only)
WRITES results/analysis/chatts_c4_census_regrade.json

GATES (hard stop on failure — no workarounds):
  G1  census sheet: 863 rows, clean y/n on q1–q3, exactly 738 valid
  G2  responses: 11,080 rows; C4 = 990 items × {swap,random} × {corrA,corrB}
      = 3,960 rows; every item has all four rows; pair key shared by twins
  G3  join: every census row maps onto a real C4 swap item_id; census (863)
      + not-censused (127) = 990. If probe1_items.jsonl is present, also:
      clause text byte-identical; explicit pile minus cross-slot == census;
      cross-slot-in-pile 20, generic 107, cross-slot total 22 (record §4/§5).
  G4  reproduction: full-990 swap acc, random acc and gap point values equal
      the recorded 0.6576 / 0.7545 / 0.0970 to 4 dp; n_signals = 240.
      CIs are compared with tolerance 0.01 and REPORTED, not gated: the
      original run drew all cells from one shared RNG stream, so the exact
      bootstrap draws are not reproducible here. Stated, not hidden.

REGISTERED EXPECTATIONS (Claude, 2026-09-05, before any run; score them):
  P-R1  certified (738) swap accuracy within ±0.03 of 0.6576
  P-R2  census_invalid (125) swap accuracy < certified swap accuracy
        (rationale: an item whose distractor is arguably true should look
        chance-like to a model that reads content; CLaSP's invalid items
        scored 0.531)
  P-R3  certified gap (random − swap) > 0.05 with 95% CI excluding 0 —
        i.e. the 'partial, no collapse' verdict is unchanged

RUN (repo root):  python scripts/regrade_chatts_c4_census.py
Paste the full stdout back into the chat.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

CENSUS = Path("results/analysis/pinning_spotcheck_sequential.csv")
RESP = Path("results/experiments/chatts_probe1_responses.jsonl")
ANALYSIS = Path("results/experiments/chatts_analysis.json")
ITEMS = Path("data/processed/probe1_items.jsonl")        # optional
OUT = Path("results/analysis/chatts_c4_census_regrade.json")

COMP = "C4_fluctuation_type"
B_BOOT = 2000
SEED = 42
TOST_MARGIN = 0.05
CI_TOL = 0.01

EXP_CENSUS_ROWS = 863
EXP_VALID = 738
EXP_INVALID = 125
EXP_NOT_CENSUSED = 127
EXP_CROSS = 20
EXP_GENERIC = 107
EXP_C4_ITEMS = 990
EXP_RESP_ROWS = 11080
EXP_SIGNALS = 240

EXP_CROSS_TOTAL = 22        # record: 22 cross-slot items in all 990 (validation doc §4);
                            # 20 of them inside the 883 explicit pile (§5), 2 in the generic pile

# the explicit-pile rule and the jumpy-shape set are IMPORTED from the
# original census script, not re-implemented (a first version of this gate
# re-implemented them and mis-split the pile — caught by G3, 2026-09-05)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.census_c4_reanalysis import pins, JUMPY  # noqa: E402


def die(msg: str) -> None:
    print(f"\nGATE FAIL — {msg}\nSTOP. Paste this output; do not work around it.")
    sys.exit(1)


def ok(tag: str, msg: str) -> None:
    print(f"{tag} pass — {msg}")


# --------------------------------------------------------------- statistics
def cluster_boot_mean(values, clusters, alpha, rng):
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
    m, lo95, hi95, n, k = cluster_boot_mean(diffs, clusters, 0.05, rng)
    _, lo90, hi90, _, _ = cluster_boot_mean(diffs, clusters, 0.10, rng)
    tost = bool(lo90 > -TOST_MARGIN and hi90 < TOST_MARGIN)
    return {"mean_diff": round(m, 4), "ci95": [round(lo95, 4), round(hi95, 4)],
            "ci90": [round(lo90, 4), round(hi90, 4)],
            "tost_pm005": "PASS(flat)" if tost else "FAIL(not flat)",
            "n_units": n, "n_signals": k}


def partition_stats(label, keys, swap_u, rand_u, sig_of, rng):
    """keys: pair keys (component|sample_id|swap_to). Returns dict."""
    if not keys:
        return {"label": label, "n_items": 0}
    keys = sorted(keys)
    sw = np.array([swap_u[k] for k in keys])
    rd = np.array([rand_u[k] for k in keys])
    cl = [sig_of[k] for k in keys]
    swc, rdc = cell(sw, cl, rng), cell(rd, cl, rng)
    d = rd - sw
    gap = paired(d.tolist(), cl, rng)
    try:
        wp = float(stats.wilcoxon(d, alternative="two-sided").pvalue)
    except ValueError:
        wp = 1.0
    strict_s = (sw == 1.0).astype(int)
    strict_r = (rd == 1.0).astype(int)
    n01 = int(((strict_s == 0) & (strict_r == 1)).sum())
    n10 = int(((strict_s == 1) & (strict_r == 0)).sum())
    mp = float(stats.binomtest(min(n01, n10), n01 + n10, 0.5).pvalue) if (n01 + n10) else 1.0
    flip_sw = float(np.mean(sw == 0.5))
    flip_rd = float(np.mean(rd == 0.5))
    res = {"label": label, "n_items": len(keys), "swap": swc, "random": rdc,
           "gap_random_minus_swap": gap, "wilcoxon_p": wp,
           "mcnemar_secondary": {"n01_rand_only": n01, "n10_swap_only": n10, "p": round(mp, 6)},
           "order_flip_rate": {"swap": round(flip_sw, 4), "random": round(flip_rd, 4)}}
    print(f"  {label:<16} n={len(keys):>4} sig={swc['n_signals']:>3} | "
          f"swap {swc['acc']:.4f} {swc['ci95']} | random {rdc['acc']:.4f} {rdc['ci95']} | "
          f"gap {gap['mean_diff']:+.4f} {gap['ci95']} {gap['tost_pm005']} | "
          f"Wilcoxon p={wp:.2e} | McNemar {n01}/{n10} p={mp:.2e}")
    return res


# --------------------------------------------------------------------- main
def main() -> None:
    def norm(v):
        return str(v).strip().lower() if v is not None else ""

    print("regrade_chatts_c4_census.py — ChatTS Probe-1 C4 on the census partition")
    print(f"census={CENSUS}\nresponses={RESP}\nanalysis={ANALYSIS}\nitems(optional)={ITEMS} "
          f"present={ITEMS.exists()}")

    # ------------------------------------------------------------------ G1
    with open(CENSUS, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != EXP_CENSUS_ROWS:
        die(f"G1: {len(rows)} census rows, expected {EXP_CENSUS_ROWS}")
    qcols = ["q1_grammatical", "q2_asserts_swap_to", "q3_not_still_true"]
    bad = [r["reading_order"] for r in rows if any(norm(r[c]) not in ("y", "n") for c in qcols)]
    if bad:
        die(f"G1: rows without clean y/n: {bad[:5]}")
    for r in rows:
        r["valid"] = all(norm(r[c]) == "y" for c in qcols)
        r["pair_key"] = f"{COMP}|{r['sample_id']}|{r['swap_to']}"
    n_valid = sum(r["valid"] for r in rows)
    if n_valid != EXP_VALID:
        die(f"G1: valid tally {n_valid}, expected {EXP_VALID}")
    if len({r["pair_key"] for r in rows}) != len(rows):
        die("G1: duplicate census items")
    ok("G1", f"census {len(rows)} rows, valid {n_valid}, invalid {len(rows) - n_valid}")

    # ------------------------------------------------------------------ G2
    resp = [json.loads(l) for l in open(RESP, encoding="utf-8")]
    if len(resp) != EXP_RESP_ROWS:
        die(f"G2: {len(resp)} response rows, expected {EXP_RESP_ROWS}")
    c4 = defaultdict(dict)  # pair_key -> {(condition, order): correct}
    sig_of = {}
    for r in resp:
        parts = r["mcq_id"].split("|")
        if len(parts) != 5:
            die(f"G2: mcq_id with {len(parts)} fields: {r['mcq_id']}")
        comp, sample_id, swap_to, cond, order = parts
        if comp != COMP:
            continue
        if cond not in ("swap", "random") or order not in ("corrA", "corrB"):
            die(f"G2: unexpected condition/order in {r['mcq_id']}")
        pk = f"{comp}|{sample_id}|{swap_to}"
        if (cond, order) in c4[pk]:
            die(f"G2: duplicate row {r['mcq_id']}")
        c4[pk][(cond, order)] = int(bool(r["correct"]))
        sig_of[pk] = sample_id
    n_rows_c4 = sum(len(v) for v in c4.values())
    if len(c4) != EXP_C4_ITEMS or n_rows_c4 != EXP_C4_ITEMS * 4:
        die(f"G2: {len(c4)} C4 items / {n_rows_c4} rows, expected {EXP_C4_ITEMS} / {EXP_C4_ITEMS * 4}")
    incomplete = [k for k, v in c4.items() if len(v) != 4]
    if incomplete:
        die(f"G2: items missing rows: {incomplete[:3]}")
    if len(set(sig_of.values())) != EXP_SIGNALS:
        die(f"G2: {len(set(sig_of.values()))} distinct C4 signals, expected {EXP_SIGNALS}")
    swap_u = {k: (v[("swap", "corrA")] + v[("swap", "corrB")]) / 2 for k, v in c4.items()}
    rand_u = {k: (v[("random", "corrA")] + v[("random", "corrB")]) / 2 for k, v in c4.items()}
    ok("G2", f"{len(c4)} C4 items × 4 rows = {n_rows_c4}; {EXP_SIGNALS} signals; "
             f"swap+random twins present for every item")

    # ------------------------------------------------------------------ G3
    census_keys = {r["pair_key"] for r in rows}
    missing = census_keys - set(c4)
    if missing:
        die(f"G3: {len(missing)} census rows do not map onto C4 items: {sorted(missing)[:3]}")
    valid_keys = {r["pair_key"] for r in rows if r["valid"]}
    invalid_keys = census_keys - valid_keys
    not_censused = set(c4) - census_keys
    if len(not_censused) != EXP_NOT_CENSUSED or len(invalid_keys) != EXP_INVALID:
        die(f"G3: not-censused {len(not_censused)} (exp {EXP_NOT_CENSUSED}), "
            f"invalid {len(invalid_keys)} (exp {EXP_INVALID})")
    cross_keys, generic_keys = set(), set()
    if ITEMS.exists():
        items = [json.loads(l) for l in open(ITEMS, encoding="utf-8")]
        c4_items = {f"{it['component']}|{it['sample_id']}|{it['swap_to']}": it
                    for it in items if it["condition"] == "swap" and it["component"] == COMP}
        if set(c4_items) != set(c4):
            die("G3: item file C4 swap keys differ from response keys")
        by_key = {r["pair_key"]: r for r in rows}
        for k, r in by_key.items():
            if c4_items[k]["clause_replaced_to"] != r["clause_replaced_to"]:
                die(f"G3: clause text mismatch at {k}")
        # same accounting as census_c4_reanalysis.py: pile = explicit clauses;
        # cross-slot = step-on-jumpy WITHIN the pile; generic = everything not in the pile
        pile = {k for k, it in c4_items.items() if pins(it["clause_replaced_to"], it["swap_to"])}
        is_cross = lambda it: it["swap_to"] == "step" and any(s in it["source_class"] for s in JUMPY)
        cross_keys = {k for k in pile if is_cross(c4_items[k])}
        generic_keys = set(c4_items) - pile
        cross_total = sum(is_cross(it) for it in c4_items.values())
        if pile - cross_keys != census_keys:
            die(f"G3: census set != explicit pile minus cross-slot (pile {len(pile)}, "
                f"symmetric diff {len((pile - cross_keys) ^ census_keys)})")
        if len(cross_keys) != EXP_CROSS or len(generic_keys) != EXP_GENERIC or cross_total != EXP_CROSS_TOTAL:
            die(f"G3: split cross-in-pile {len(cross_keys)} / generic {len(generic_keys)} / "
                f"cross-total {cross_total}, expected {EXP_CROSS} / {EXP_GENERIC} / {EXP_CROSS_TOTAL}")
        ok("G3", f"census {len(census_keys)} = valid {len(valid_keys)} + invalid {len(invalid_keys)}; "
                 f"not-censused {len(not_censused)} = cross-slot-in-pile {len(cross_keys)} + generic "
                 f"{len(generic_keys)} (generic includes the {cross_total - len(cross_keys)} "
                 f"jumpy-step items outside the explicit pile; {cross_total} cross-slot in total, "
                 f"as recorded); clause text byte-identical")
    else:
        ok("G3", f"census {len(census_keys)} = valid {len(valid_keys)} + invalid {len(invalid_keys)}; "
                 f"not-censused {len(not_censused)} (cross-slot/generic split needs "
                 f"{ITEMS} — not present, reported pooled)")

    # ------------------------------------------------------------------ G4
    rec = json.load(open(ANALYSIS, encoding="utf-8"))["probe1"]["components"]["C4"]
    rng = np.random.default_rng(SEED)
    all_keys = sorted(c4)
    sw = np.array([swap_u[k] for k in all_keys])
    rd = np.array([rand_u[k] for k in all_keys])
    cl = [sig_of[k] for k in all_keys]
    swc, rdc = cell(sw, cl, rng), cell(rd, cl, rng)
    gap = paired((rd - sw).tolist(), cl, rng)
    pts = {"swap": (swc["acc"], rec["swap"]["acc"]),
           "random": (rdc["acc"], rec["random"]["acc"]),
           "gap": (gap["mean_diff"], rec["gap_random_minus_swap"]["mean_diff"])}
    for name, (got, exp) in pts.items():
        if round(got, 4) != round(exp, 4):
            die(f"G4: full-990 {name} {got:.4f} != recorded {exp:.4f}")
    if swc["n_signals"] != rec["swap"]["n_signals"]:
        die(f"G4: n_signals {swc['n_signals']} != recorded {rec['swap']['n_signals']}")
    ci_dev = max(abs(a - b) for got, exp in
                 ((swc["ci95"], rec["swap"]["ci95"]), (rdc["ci95"], rec["random"]["ci95"]),
                  (gap["ci95"], rec["gap_random_minus_swap"]["ci95"]))
                 for a, b in zip(got, exp))
    ok("G4", f"full-990 point values reproduce the record: swap {swc['acc']} random {rdc['acc']} "
             f"gap {gap['mean_diff']:+.4f}; n_signals {swc['n_signals']}. "
             f"CI max |dev| vs record = {ci_dev:.4f} (tol {CI_TOL}; different RNG stream — "
             f"{'within' if ci_dev <= CI_TOL else 'EXCEEDS'} tolerance, reported not gated)")

    if "--gates-only" in sys.argv:
        print("\n--gates-only: G1–G4 passed; re-grade not run, nothing written.")
        return

    # ----------------------------------------------------------- re-grade
    print("\n--- ChatTS C4 by census partition (units = items; CIs = signal-cluster bootstrap) ---")
    rng = np.random.default_rng(SEED)   # fresh stream so partitions are order-independent
    out = {"generated": "2026-09-05", "script": "scripts/regrade_chatts_c4_census.py",
           "method": "re-grade of stored ChatTS per-item answers; no rerun",
           "conventions": {"unit": "mean of both answer orders (0/0.5/1)",
                           "bootstrap": f"cluster on sample_id, B={B_BOOT}, seed {SEED}",
                           "tost": "±0.05 on the 90% CI of random−swap"},
           "reproduction_gate": {"full_990": {"swap": swc, "random": rdc, "gap": gap},
                                 "recorded": rec, "ci_max_abs_dev": round(ci_dev, 4)},
           "partitions": {}}
    parts = [("census_valid", valid_keys), ("census_invalid", invalid_keys)]
    if cross_keys or generic_keys:
        parts += [("cross_slot", cross_keys), ("generic", generic_keys)]
    else:
        parts += [("not_censused", not_censused)]
    parts += [("all_990", set(c4))]
    for label, keys in parts:
        out["partitions"][label] = partition_stats(label, keys, swap_u, rand_u, sig_of, rng)

    # --------------------------------------------------- score expectations
    v = out["partitions"]["census_valid"]
    inv = out["partitions"]["census_invalid"]
    p_r1 = abs(v["swap"]["acc"] - rec["swap"]["acc"]) <= 0.03
    p_r2 = inv["swap"]["acc"] < v["swap"]["acc"]
    p_r3 = v["gap_random_minus_swap"]["mean_diff"] > 0.05 and v["gap_random_minus_swap"]["ci95"][0] > 0
    delta = v["swap"]["acc"] - rec["swap"]["acc"]
    print("\n--- registered expectations ---")
    print(f"  P-R1 certified swap within ±0.03 of {rec['swap']['acc']}: {v['swap']['acc']} "
          f"(Δ {delta:+.4f}) -> {'HIT' if p_r1 else 'MISS'}")
    print(f"  P-R2 invalid swap < certified swap: {inv['swap']['acc']} vs {v['swap']['acc']} "
          f"-> {'HIT' if p_r2 else 'MISS'}")
    print(f"  P-R3 certified gap > 0.05, CI95 excludes 0: {v['gap_random_minus_swap']['mean_diff']:+.4f} "
          f"{v['gap_random_minus_swap']['ci95']} -> {'HIT' if p_r3 else 'MISS'}")
    out["expectations"] = {"P-R1": "HIT" if p_r1 else "MISS", "P-R2": "HIT" if p_r2 else "MISS",
                           "P-R3": "HIT" if p_r3 else "MISS",
                           "certified_minus_full_swap_acc": round(delta, 4)}
    print(f"\nCount chain to quote (ChatTS C4, as for CLaSP): {EXP_C4_ITEMS} generated -> "
          f"{EXP_CENSUS_ROWS} lexically explicit & eligible -> {EXP_VALID} census-certified")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {OUT}")
    print("Nothing in chatts_analysis.json was modified.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
n3_census_verdict.py — score the filled N3 census, build the excision set,
recompute N3 on the certified survivors, and score prediction P-cen3.

Context (2026-08-09): the 100-item screen FAILED (union 17 defects); the author
judged all 289 remaining flagged items (census). This script consumes the
FILLED census sheet and produces the certification-stage artifacts.

Predictions registered 2026-08-09 (project log), scored here:
  P-cen1  census defect count in [25, 75]
  P-cen2  the 17 not-yet-judged six_months items defective at >= 0.70
  P-cen3  post-excision pooled N3 swap accuracy DECREASES or holds within
          +0.03 of the committed per-seed values. A rise beyond +0.03 would
          challenge the understatement argument -> investigate, don't explain.

Committed pre-excision N3 baselines (read from
results/experiments/trace_narrative_summary.json, digit-exact):
  mask13 swap 0.719794  random 0.997429  gap 0.277635
  mask14 swap 0.722365  random 0.994859  gap 0.272494
  mask15 swap 0.724936  random 0.992288  gap 0.267352

Reads:
  results/analysis/n3_census_sheet.csv            (FILLED, utf-8-sig)
  results/analysis/n3_validation_sheet.csv        (sample pass 1)
  results/analysis/n3_validation_sheet_variant.csv (sample pass 2)
  results/analysis/n3_census_flags.json           (id membership gate)
  results/experiments/trace_narrative_per_item.jsonl

Writes:
  results/analysis/n3_census_excision_ids.txt
  results/experiments/trace_narrative_per_item_certified.jsonl
  results/analysis/n3_census_verdict.json

Run from repo root:  python scripts/n3_census_verdict.py
Then re-run the model-agnostic statistics on the certified per-item file:
  python scripts/analyze_probe1_stats.py \
      --per-item results/experiments/trace_narrative_per_item_certified.jsonl \
      --out results/experiments/trace_narrative_statistics_certified.json
"""

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

CENSUS = Path("results/analysis/n3_census_sheet.csv")
SAMPLE1 = Path("results/analysis/n3_validation_sheet.csv")
SAMPLE2 = Path("results/analysis/n3_validation_sheet_variant.csv")
FLAGS = Path("results/analysis/n3_census_flags.json")
PER_ITEM = Path("results/experiments/trace_narrative_per_item.jsonl")

EXCISION_OUT = Path("results/analysis/n3_census_excision_ids.txt")
CERTIFIED_OUT = Path("results/experiments/trace_narrative_per_item_certified.jsonl")
VERDICT_OUT = Path("results/analysis/n3_census_verdict.json")

# committed pre-excision baselines (summary.json, digit-exact; see docstring)
BASELINE = {
    "mask13": {"swap": 0.7197943444730077, "random": 0.9974293059125964},
    "mask14": {"swap": 0.7223650385604113, "random": 0.9948586118251928},
    "mask15": {"swap": 0.7249357326478149, "random": 0.9922879177377892},
}


def die(msg):
    print(f"[STOP] {msg}")
    sys.exit(1)


def load_sheet(path):
    if not path.exists():
        die(f"missing file: {path}")
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    hdr = {n: i for i, n in enumerate(rows[1])}
    body = [r for r in rows[2:] if r and r[hdr["item_id"]].strip()]
    return hdr, body


def main():
    # ---- Gate 1: the filled census sheet -------------------------------
    hdr, body = load_sheet(CENSUS)
    print(f"census sheet: {len(body)} rows read from {CENSUS}")
    if len(body) != 289:
        die(f"expected 289 census rows, found {len(body)}")

    bad_verdict, bad_note, noted_y = [], [], []
    verdicts = {}
    dur = {}
    for r in body:
        iid = r[hdr["item_id"]]
        v = r[hdr["y_or_n"]].strip().lower()
        note = r[hdr["notes"]].strip()
        verdicts[iid] = v
        dur[iid] = r[hdr["duration_class"]]
        ro = r[hdr["reading_order"]]
        if v not in ("y", "n"):
            bad_verdict.append((ro, iid, repr(v)))
        if v == "n" and not note.lower().startswith(("a:", "b:")):
            bad_note.append((ro, iid))
        if v == "y" and note:
            noted_y.append((ro, iid))
    if bad_verdict:
        die(f"non-y/n verdicts (blank sheet? wrong file bytes?): {bad_verdict[:5]}")
    if bad_note:
        die(f"n-rows without an 'a:'/'b:' note: {bad_note[:5]}")
    if noted_y:
        print(f"[warn] y-rows carrying notes (informational): {noted_y[:5]}")

    flags = json.loads(FLAGS.read_text(encoding="utf-8"))
    census_ids_expected = set(flags["flags"]) - {  # flagged minus the 100 judged
        i for i in flags["flags"] if i in _sample_ids()
    }
    if set(verdicts) != census_ids_expected:
        extra = sorted(set(verdicts) - census_ids_expected)[:3]
        missing = sorted(census_ids_expected - set(verdicts))[:3]
        die(f"census id membership mismatch. extra={extra} missing={missing}")

    c = Counter(verdicts.values())
    print(f"verdicts: y={c['y']}  n={c['n']}")
    census_n = {i for i, v in verdicts.items() if v == "n"}

    # ---- Gate 2: union of sample defects -------------------------------
    union_n = _sample_union_defects()
    print(f"sample union defects: {len(union_n)}")
    if len(union_n) != 17:
        die(f"sample union defect count {len(union_n)} != committed 17")
    if census_n & union_n:
        die(f"census/sample overlap should be empty: {sorted(census_n & union_n)[:5]}")

    # ---- Predictions P-cen1 / P-cen2 -----------------------------------
    p1 = 25 <= len(census_n) <= 75
    print(f"\nP-cen1 (census n in [25,75]): "
          f"{'CONFIRMED' if p1 else 'MISSED'}   (n = {len(census_n)})")
    six = [i for i in verdicts if dur[i] == "six_months"]
    six_n = [i for i in six if verdicts[i] == "n"]
    rate6 = len(six_n) / len(six) if six else float("nan")
    p2 = rate6 >= 0.70
    print(f"P-cen2 (six_months census defect rate >= 0.70): "
          f"{'CONFIRMED' if p2 else 'MISSED'}   ({len(six_n)}/{len(six)} = {rate6:.3f})")

    # ---- Excision set ---------------------------------------------------
    excision = sorted(union_n | census_n)
    print(f"\nexcision set: {len(excision)} defective swap items "
          f"({len(union_n)} sample + {len(census_n)} census) + matched random twins")
    with EXCISION_OUT.open("w", encoding="utf-8") as f:
        f.write("# N3 census excision list, 2026-08 — 45 human-judged defective\n"
                "# swap items: 17 from the 100-item sample (union of n's, strict-\n"
                "# call rule) + 28 from the 289-item census. Each excises with its\n"
                "# matched random twin (same pair_key). Verdict sheets:\n"
                "# n3_validation_sheet{,_variant}.csv + filled n3_census_sheet.csv\n")
        for i in excision:
            src = "sample" if i in union_n else "census"
            f.write(f"{i}  # {src}\n")
    print(f"wrote {EXCISION_OUT}")

    exc_pairs = {i.rsplit("|", 1)[0] for i in excision}
    if len(exc_pairs) != len(excision):
        die("pair_key collision inside the excision set — should be impossible")

    # ---- Filter per-item records ---------------------------------------
    if not PER_ITEM.exists():
        die(f"missing {PER_ITEM}")
    n_in = n_out = 0
    per_seed_in, per_seed_out = Counter(), Counter()
    n3_out = Counter()
    kept_records = defaultdict(list)   # seed -> list of N3 survivor records
    with PER_ITEM.open(encoding="utf-8") as fin, \
         CERTIFIED_OUT.open("w", encoding="utf-8") as fout:
        for line in fin:
            d = json.loads(line)
            n_in += 1
            per_seed_in[d["seed"]] += 1
            if d["component"] == "N3" and d["pair_key"] in exc_pairs:
                continue
            fout.write(line)
            n_out += 1
            per_seed_out[d["seed"]] += 1
            if d["component"] == "N3":
                n3_out[d["seed"]] += 1
                kept_records[d["seed"]].append(d)
    print(f"\nper-item filter: read {n_in}, wrote {n_out} -> {CERTIFIED_OUT}")
    print(f"  per seed in : {dict(per_seed_in)}")
    print(f"  per seed out: {dict(per_seed_out)}")
    print(f"  N3 per seed out: {dict(n3_out)}")
    if any(v != 3178 for v in per_seed_in.values()) or len(per_seed_in) != 3:
        die("input per-seed counts differ from committed 3 x 3178")
    if any(v != 3178 - 2 * len(exc_pairs) for v in per_seed_out.values()):
        die(f"output per-seed counts differ from expected {3178 - 2*len(exc_pairs)}")
    if any(v != 778 - 2 * len(exc_pairs) for v in n3_out.values()):
        die(f"N3 per-seed counts differ from expected {778 - 2*len(exc_pairs)}")

    # ---- Recompute N3 on survivors + P-cen3 ----------------------------
    print("\nN3 on certified survivors (per seed):")
    verdict = {"excised": len(excision), "per_seed": {}, "gradient": {}}
    p3_ok = True
    for seed in sorted(kept_records):
        recs = kept_records[seed]
        sw = [d for d in recs if d["condition"] == "swap"]
        rd = [d for d in recs if d["condition"] == "random"]
        acc_s = sum(d["correct"] for d in sw) / len(sw)
        acc_r = sum(d["correct"] for d in rd) / len(rd)
        base = BASELINE[seed]["swap"]
        delta = acc_s - base
        if delta > 0.03:
            p3_ok = False
        print(f"  {seed}: swap {acc_s:.6f} (was {base:.6f}, Δ {delta:+.6f})  "
              f"random {acc_r:.6f}  gap {acc_r - acc_s:.6f}  "
              f"[n_swap {len(sw)}, n_random {len(rd)}]")
        verdict["per_seed"][seed] = {
            "acc_swap": acc_s, "acc_random": acc_r, "gap": acc_r - acc_s,
            "delta_swap_vs_committed": delta,
            "n_swap": len(sw), "n_random": len(rd),
        }
    print(f"P-cen3 (post-excision swap acc decreases or holds within +0.03): "
          f"{'CONFIRMED' if p3_ok else 'MISSED — INVESTIGATE'}")

    # ---- Duration gradient on survivors --------------------------------
    print("\nduration gradient on survivors (swap accuracy, pooled over seeds):")
    by_dur = defaultdict(lambda: [0, 0])
    for seed in kept_records:
        for d in kept_records[seed]:
            if d["condition"] == "swap":
                b = by_dur[d["duration_class"]]
                b[0] += d["correct"]
                b[1] += 1
    for cl in ("week", "28_days", "six_months"):
        n_corr, n_tot = by_dur[cl]
        items = n_tot // 3
        print(f"  {cl:11s}: {n_corr/n_tot:.4f}   "
              f"({items} items x 3 seeds; CELL SIZE MATTERS — quote with n)")
        verdict["gradient"][cl] = {"acc": n_corr / n_tot, "n_items": items}

    verdict["predictions"] = {
        "P-cen1": {"confirmed": p1, "census_n": len(census_n)},
        "P-cen2": {"confirmed": p2, "six_months": f"{len(six_n)}/{len(six)}"},
        "P-cen3": {"confirmed": p3_ok},
    }
    VERDICT_OUT.write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    print(f"\nverdict record -> {VERDICT_OUT}")
    print("\nNext: run analyze_probe1_stats.py on the certified per-item file "
          "(command in the docstring) for Holm-corrected significance.")


def _sample_ids():
    h1, s1 = load_sheet(SAMPLE1)
    return {r[h1["item_id"]] for r in s1}


def _sample_union_defects():
    h1, s1 = load_sheet(SAMPLE1)
    h2, s2 = load_sheet(SAMPLE2)
    v1 = {r[h1["item_id"]]: r[h1["y_or_n"]].strip().lower() for r in s1}
    v2 = {r[h2["item_id"]]: r[h2["y_or_n"]].strip().lower() for r in s2}
    if set(v1) != set(v2) or len(v1) != 100:
        die("the two sample sheets do not cover the same 100 items")
    return {i for i, v in v1.items() if v == "n"} | \
           {i for i, v in v2.items() if v == "n"}


if __name__ == "__main__":
    main()

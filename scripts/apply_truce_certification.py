#!/usr/bin/env python3
"""
apply_truce_certification.py — stamp the author's census verdicts (2026-08-13)
into the certified TRUCE Probe-2 grouping artifact.

Certified human decisions applied (5 + 3 moves, all train/val):
  invariant -> dependent (census n, order-sensitive reading exists):
    'plot is flat and even for a potion'
    'ricing the height level'
    'stay flat in the fist part.'
    'the changes stay almost constant throughout the pattern'
    'very not steep downward parabola'
  ambiguous -> invariant:  'highly placed line'
  ambiguous -> dependent:  'incresae the same'
                           'this plot explains inrease derease to inccresaes'

Run from the repository root:
    python scripts/apply_truce_certification.py

Reads:  results/analysis/probe2_truce_groups.json           (rules v2)
        results/analysis/probe2_truce_invariant_judgment_sheet.csv (judged)
        results/analysis/probe2_truce_dependent_sample.csv         (judged)
        results/analysis/probe2_truce_ambiguous_sheet.csv          (judged)
Writes: results/analysis/probe2_truce_groups_certified.json

Gates (all hard):
  A1 groups.json is rules v2 with 5087 texts
  A2 each sheet decodes (utf-8 / gbk / cp1252 fallback, reported) and
     every caption joins to groups.json exactly
  A3 verdict counts are exactly: invariant 184y/5n, dependent 30y/0n,
     ambiguous 40y/3n — and the n-captions are exactly the eight above
  A4 certified buckets equal the registered counts:
     dependent 7089 rows / 4862 texts / 715 test
     invariant  245 rows /  185 texts /  18 test
     ambiguous   46 rows /   40 texts /   5 test
     totals 7380 rows / 5087 texts / 738 test
  A5 certified P2-8 = 245/7334 = 0.0334 -> MISSED (threshold 0.15)
"""

import csv
import io
import json
import sys
from collections import Counter
from pathlib import Path

AN = Path("results/analysis")
GROUPS = AN / "probe2_truce_groups.json"

INV_TO_DEP = [
    "plot is flat and even for a potion",
    "ricing the height level",
    "stay flat in the fist part.",
    "the changes stay almost constant throughout the pattern",
    "very not steep downward parabola",
]
AMB_TO_INV = ["highly placed line"]
AMB_TO_DEP = [
    "incresae the same",
    "this plot explains inrease derease to inccresaes",
]

EXPECT_CERTIFIED = {  # rows, unique texts, test rows
    "dependent": (7089, 4862, 715),
    "invariant": (245, 185, 18),
    "ambiguous": (46, 40, 5),
}


def fail(msg):
    print(f"\nGATE FAILED: {msg}", file=sys.stderr)
    sys.exit(1)


def load_sheet(path):
    raw = open(path, "rb").read()
    for enc in ("utf-8", "gbk", "cp1252"):
        try:
            txt = raw.decode(enc)
            print(f"  {path.name}: decoded as {enc}")
            return list(csv.DictReader(io.StringIO(txt)))
        except UnicodeDecodeError:
            continue
    fail(f"A2: {path} not decodable as utf-8/gbk/cp1252")


def main():
    groups = json.load(open(GROUPS, encoding="utf-8"))
    if groups.get("rules_version") != 2:
        fail(f"A1: groups.json rules_version {groups.get('rules_version')} != 2")
    per_text = groups["per_text"]
    if len(per_text) != 5087:
        fail(f"A1: per_text has {len(per_text)} texts != 5087")
    print("A1 PASSED: rules v2 grouping loaded")

    sheets = {}
    for key, fname, n_exp, vc_exp in [
        ("inv", "probe2_truce_invariant_judgment_sheet.csv", 189,
         {"y": 184, "n": 5}),
        ("dep", "probe2_truce_dependent_sample.csv", 30, {"y": 30}),
        ("amb", "probe2_truce_ambiguous_sheet.csv", 43, {"y": 40, "n": 3}),
    ]:
        rows = load_sheet(AN / fname)
        if len(rows) != n_exp:
            fail(f"A2: {fname} has {len(rows)} rows != {n_exp}")
        unmatched = [r["caption"] for r in rows if r["caption"] not in per_text]
        if unmatched:
            fail(f"A2: {fname}: {len(unmatched)} captions not in groups.json, "
                 f"e.g. {unmatched[:3]}")
        vc = Counter((r["verdict_y_n"] or "").strip().lower() for r in rows)
        if dict(vc) != vc_exp:
            fail(f"A3: {fname} verdicts {dict(vc)} != {vc_exp}")
        sheets[key] = rows
    print("A2 PASSED: all sheets decoded, all captions join")

    inv_n = sorted(r["caption"] for r in sheets["inv"]
                   if r["verdict_y_n"].strip().lower() == "n")
    amb_n = sorted(r["caption"] for r in sheets["amb"]
                   if r["verdict_y_n"].strip().lower() == "n")
    if inv_n != sorted(INV_TO_DEP):
        fail(f"A3: invariant n-captions differ from the certified list: {inv_n}")
    if amb_n != sorted(AMB_TO_INV + AMB_TO_DEP):
        fail(f"A3: ambiguous n-captions differ from the certified list: {amb_n}")
    print("A3 PASSED: verdict counts and n-captions exactly as certified")

    # apply moves with provenance
    for t in INV_TO_DEP:
        per_text[t]["label"] = "dependent"
        per_text[t]["provenance"] = "census_n_2026-08-13"
    for t in AMB_TO_INV:
        per_text[t]["label"] = "invariant"
        per_text[t]["provenance"] = "human_reassigned_2026-08-13"
    for t in AMB_TO_DEP:
        per_text[t]["label"] = "dependent"
        per_text[t]["provenance"] = "human_reassigned_2026-08-13"

    # recount from pairs to get authoritative row/test counts
    pairs = Path("data/processed/pairs.jsonl")
    rows_c, uniq_c, test_c = Counter(), Counter(), Counter()
    seen = set()
    with open(pairs, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if not r["dataset"].startswith("truce"):
                continue
            lab = per_text[r["caption"]]["label"]
            rows_c[lab] += 1
            if r["split"] == "test":
                test_c[lab] += 1
            if r["caption"] not in seen:
                seen.add(r["caption"])
                uniq_c[lab] += 1

    print("\nCERTIFIED BUCKETS (rows | unique texts | test rows):")
    for lab in ("dependent", "invariant", "ambiguous"):
        er, eu, et = EXPECT_CERTIFIED[lab]
        print(f"  {lab:10s} {rows_c[lab]:5d} | {uniq_c[lab]:5d} | {test_c[lab]:4d}"
              f"   (expected {er} | {eu} | {et})")
        if (rows_c[lab], uniq_c[lab], test_c[lab]) != (er, eu, et):
            fail(f"A4: {lab} counts differ from registered expectation")
    if sum(rows_c.values()) != 7380 or sum(uniq_c.values()) != 5087 \
            or sum(test_c.values()) != 738:
        fail("A4: totals do not reconcile")
    print("A4 PASSED: certified counts match registered expectations")

    p28 = rows_c["invariant"] / (rows_c["dependent"] + rows_c["invariant"])
    p28_test = test_c["invariant"] / (test_c["dependent"] + test_c["invariant"])
    verdict = "CONFIRMED" if p28 >= 0.15 else "MISSED"
    print(f"\nP2-8 CERTIFIED: {rows_c['invariant']}/"
          f"{rows_c['dependent'] + rows_c['invariant']} = {p28:.4f}   "
          f"threshold 0.15 -> {verdict}")
    print(f"  test-split row-level: {p28_test:.4f}   "
          f"(v1 provisional 0.0328, v2 provisional 0.0340 remain in the record)")
    if abs(p28 - 0.0334) > 0.0005:
        fail(f"A5: certified P2-8 {p28:.4f} differs from registered 0.0334")
    print("A5 PASSED")

    out = {
        "certified": True,
        "certified_by": "the author (full invariant census 189, dependent sample 30, "
                        "ambiguous full sheet 43)",
        "certified_date": "2026-08-13",
        "rules_version": 2,
        "moves": {"invariant_to_dependent": INV_TO_DEP,
                  "ambiguous_to_invariant": AMB_TO_INV,
                  "ambiguous_to_dependent": AMB_TO_DEP},
        "rows": dict(rows_c), "unique": dict(uniq_c), "test_rows": dict(test_c),
        "degenerate_series": groups["degenerate_series"],
        "p2_8_certified": {"row_population": p28, "test_row": p28_test,
                           "threshold": 0.15, "verdict": verdict,
                           "v1_provisional": 0.0328, "v2_provisional":
                           groups["p2_8_provisional"]["primary_row_population"]},
        "per_text": per_text,
    }
    dest = AN / "probe2_truce_groups_certified.json"
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nwrote {dest}")
    print("ALL GATES PASSED — TRUCE Probe-2 grouping CERTIFIED")


if __name__ == "__main__":
    main()

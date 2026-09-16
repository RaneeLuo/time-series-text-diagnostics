#!/usr/bin/env python3
"""
scan_narrative_phrases.py — coverage scan for the option-(b) component
grammar. Answers, over all 2,006 test descriptions: how many rows contain
a parseable trend-direction statement (N3 candidate) and a parseable
fluctuation/stability statement (N4 candidate), in which channel fields,
with which phrasings?

Why coverage matters: a swap component is only usable if its trigger
phrases appear in (nearly) all rows with unambiguous polarity, and if the
swap can be executed by exact string replacement without touching anything
else. Probe-1 discipline carries over: parse coverage is REPORTED, and a
component whose coverage is poor is dropped, not patched with looser
matching.

This scan is deliberately conservative:
  - it counts occurrences of anchored phrase families, per field;
  - it reports rows matching BOTH polarities of the same family in the
    same field (ambiguous -> unswappable by simple replacement);
  - it prints the residue: rows matching NEITHER polarity, with the
    field text, so we can see what a broader pattern would need to catch
    (and judge whether broadening is safe or a validity risk).

Header/template checks are included as gates: the grammar assumes every
description carries the exact 'The weather is {labels}.' header and a
'Time range:' slot with a recognisable duration phrase; this verifies
that on all rows rather than assuming it from six examples.

Run from the thesis repo root:
    python models/trace/scan_narrative_phrases.py --trace-repo ../TRACE-Multimodal-TSEncoder
Writes: results/analysis/noaa_phrase_coverage.json
"""

from __future__ import annotations
import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


def fail(gate: str, msg: str):
    print(f"\n[GATE FAILED] {gate}: {msg}")
    sys.exit(1)


CHANNEL_FIELDS = ["temperature", "precipitation", "relative_humidity",
                  "visibility", "wind_u", "wind_v", "sky_code"]

# Phrase families. Word-boundary, case-insensitive. Deliberately anchored:
# each entry is a word/stem with one polarity. Counted per field.
TREND_UP = [r"\brising\b", r"\brose\b", r"\bincreas\w*", r"\bupward\b",
            r"\bwarming\b", r"\bclimb\w*"]
TREND_DOWN = [r"\bdeclin\w*", r"\bdecreas\w*", r"\bdownward\b", r"\bfalling\b",
              r"\bfell\b", r"\bdropp?\w*", r"\bcooling\b"]
FLUX = [r"\bfluctuat\w*", r"\boutlier\w*", r"\bspike\w*", r"\bvolatil\w*",
        r"\bvariab\w*", r"\bvaried\b", r"\bvarying\b", r"\bgust\w*"]
STABLE = [r"\bstable\b", r"\bsteady\b", r"\bconsistent\w*", r"\bconstant\w*",
          r"\bcalm\b", r"\bunchanged\b", r"\bremained\b"]

DURATIONS = [r"past week", r"past six months", r"past 28 days",
             r"past month", r"past year", r"past \d+ days"]


def compile_all(pats):
    return [re.compile(p, re.IGNORECASE) for p in pats]


TREND_UP_C = compile_all(TREND_UP)
TREND_DOWN_C = compile_all(TREND_DOWN)
FLUX_C = compile_all(FLUX)
STABLE_C = compile_all(STABLE)
DUR_C = [re.compile(p, re.IGNORECASE) for p in DURATIONS]


def hits(text, compiled):
    return [c.pattern for c in compiled if c.search(text)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-repo", required=True)
    ap.add_argument("--n-residue", type=int, default=8,
                    help="how many unmatched examples to print per check")
    args = ap.parse_args()

    import pyarrow.parquet as pq

    repo = Path(args.trace_repo).resolve()
    parquet = repo / "dataset" / "retrieval" / "test" / "test.parquet"
    if not parquet.is_file():
        fail("G1", f"{parquet} missing")
    df = pq.read_table(parquet).to_pandas()
    if len(df) != 2006:
        fail("G1-count", f"expected 2006 rows, got {len(df)}")
    print(f"[G1] {len(df)} rows")

    descs = list(df["description"])

    # ---- Gate 2: template assumptions hold on every row ------------------
    n_labels_hdr = sum(1 for d in descs
                       if isinstance(d, dict) and isinstance(d.get("labels"), str))
    missing_fields = Counter()
    for d in descs:
        for k in CHANNEL_FIELDS + ["DATE", "labels", "location"]:
            if not (isinstance(d, dict) and isinstance(d.get(k), str) and d[k].strip()):
                missing_fields[k] += 1
    print(f"[G2] rows with all 10 fields as non-empty strings: "
          f"{len(descs) - sum(1 for d in descs if any(not (isinstance(d, dict) and isinstance(d.get(k), str) and d[k].strip()) for k in CHANNEL_FIELDS + ['DATE','labels','location']))} / {len(descs)}")
    if missing_fields:
        print(f"[G2] missing/empty by field: {dict(missing_fields)}")

    # duration recognisability in DATE
    dur_hits = Counter()
    dur_none = []
    for i, d in enumerate(descs):
        date = d.get("DATE", "") if isinstance(d, dict) else ""
        matched = [c.pattern for c in DUR_C if c.search(date)]
        if matched:
            dur_hits[matched[0]] += 1
        else:
            dur_none.append((i, date))
    print(f"\n[N2] DATE duration phrases: {dict(dur_hits)}")
    print(f"[N2] rows with NO recognised duration: {len(dur_none)}")
    for i, t in dur_none[:args.n_residue]:
        print(f"      row {i}: {t!r}")

    # ---- N3 / N4 coverage per channel field ------------------------------
    def scan(desc_field_pairs, pos_c, neg_c, name):
        per_field = defaultdict(lambda: Counter())
        row_any_pos = set()
        row_any_neg = set()
        row_both_same_field = set()
        phrase_freq = Counter()
        for i, d in enumerate(descs):
            for f in CHANNEL_FIELDS:
                t = d.get(f, "") if isinstance(d, dict) else ""
                p = hits(t, pos_c)
                n = hits(t, neg_c)
                if p:
                    per_field[f]["pos"] += 1
                    row_any_pos.add(i)
                    phrase_freq.update((f"+{x}" for x in p))
                if n:
                    per_field[f]["neg"] += 1
                    row_any_neg.add(i)
                    phrase_freq.update((f"-{x}" for x in n))
                if p and n:
                    per_field[f]["both"] += 1
                    row_both_same_field.add(i)
        print(f"\n[{name}] per-field coverage (rows containing >=1 phrase):")
        print(f"    {'field':<20}{'pos':>7}{'neg':>7}{'both-in-field':>15}")
        for f in CHANNEL_FIELDS:
            c = per_field[f]
            print(f"    {f:<20}{c['pos']:>7}{c['neg']:>7}{c['both']:>15}")
        both_any = row_any_pos & row_any_neg
        neither = [i for i in range(len(descs))
                   if i not in row_any_pos and i not in row_any_neg]
        print(f"[{name}] rows with >=1 positive-family phrase anywhere: {len(row_any_pos)}")
        print(f"[{name}] rows with >=1 negative-family phrase anywhere: {len(row_any_neg)}")
        print(f"[{name}] rows with both polarities in the SAME field: {len(row_both_same_field)}")
        print(f"[{name}] rows with NEITHER polarity anywhere: {len(neither)}")
        print(f"[{name}] phrase frequencies (top 15): {phrase_freq.most_common(15)}")
        for i in neither[:args.n_residue]:
            d = descs[i]
            print(f"      residue row {i} temperature field: {d.get('temperature','')!r}"
                  if isinstance(d, dict) else f"      residue row {i}: <non-dict>")
        return {"per_field": {f: dict(per_field[f]) for f in CHANNEL_FIELDS},
                "rows_pos": len(row_any_pos), "rows_neg": len(row_any_neg),
                "rows_both_same_field": len(row_both_same_field),
                "rows_neither": len(neither),
                "phrase_freq": dict(phrase_freq)}

    print("\n" + "=" * 74)
    print("N3 — TREND DIRECTION (up-family vs down-family)")
    print("=" * 74)
    n3 = scan(None, TREND_UP_C, TREND_DOWN_C, "N3")

    print("\n" + "=" * 74)
    print("N4 — FLUCTUATION vs STABILITY")
    print("=" * 74)
    n4 = scan(None, FLUX_C, STABLE_C, "N4")

    # ---- temperature-field trend focus (likeliest N3 carrier) ------------
    # For N3 the natural carrier is the temperature sentence; report its
    # exclusive-polarity coverage (swappable rows) explicitly.
    excl = {"up_only": 0, "down_only": 0, "both": 0, "neither": 0}
    for d in descs:
        t = d.get("temperature", "") if isinstance(d, dict) else ""
        u, v = bool(hits(t, TREND_UP_C)), bool(hits(t, TREND_DOWN_C))
        excl["both" if (u and v) else "up_only" if u else "down_only" if v else "neither"] += 1
    print(f"\n[N3-temperature] exclusive polarity in the temperature field alone: {excl}")

    out = Path("results/analysis/noaa_phrase_coverage.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"n_rows": len(descs),
                   "duration_phrases": dict(dur_hits),
                   "duration_unrecognised": len(dur_none),
                   "N3_trend": n3, "N4_flux": n4,
                   "N3_temperature_exclusive": excl}, f, indent=2)
    print(f"\n[done] saved -> {out}. Paste the full console output back.")


if __name__ == "__main__":
    main()

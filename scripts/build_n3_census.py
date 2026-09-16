#!/usr/bin/env python3
"""
build_n3_census.py — mechanical screening pass for the N3 full census.

Context (2026-08-09): the pre-registered 100-item human sample FAILED the
certification criterion (86/100 vs >=95/100; batch-1 boundary already at
45/50). Three defect mechanisms were codified from the judge's notes:
  R1  seasonal-window anchoring: a six-month date header is itself a
      direction claim (Jul-Dec = cooling), so ALL six_months items are
      flagged structurally. (5/6 sampled six_months items failed.)
  R2  intra-clause interpretive tails / value pairs: the surgery flips
      the trend verb but not consequence phrases ("indicating a cold
      snap", "transition from warm to cold") or high/low value pairs
      riding in the same clause.
  R3  cross-channel anchoring: a non-temperature sentence references
      temperature direction ("decreasing humidity as temperatures rose").

Filters OVER-flag on purpose: humans certify, filters only screen.

BUILT-IN GATE (prediction P-c1): the filters MUST capture all 14 items
the human sample already judged defective. Any known defect not flagged
means an uncodified mechanism exists -> hard stop, extend rules first.

Registered predictions (2026-08-09, before running):
  P-c1  R1|R2|R3 captures 17/17 known defects (gate; 17 = union of n's
        across the two judging passes, strict-call rule; R2 extended with
        peak/trough patterns after the disagreement surfaced "peaking at"
        survivors the original patterns missed).
  P-c2  R1 flags exactly 23 items.
  P-c3  total flagged in [60, 160]; >250 would itself be a finding
        (pervasive entanglement -> restriction, not excision).

Output: census sheet for HUMAN judging = flagged items MINUS the 100
already judged (their verdicts stand: 14 n's are defective, 86 y's are
certified; nobody re-judges their own past decisions).

Run from thesis repo root:
  python scripts/build_n3_census.py
Reads:  data/processed/narrative_probe_items_certified.jsonl
        results/analysis/n3_validation_sheet.csv   (the FILLED sheet)
Writes: results/analysis/n3_census_sheet.csv
        results/analysis/n3_census_flags.json
"""

import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

ITEMS = Path("data/processed/narrative_probe_items_certified.jsonl")
FILLED = Path("results/analysis/n3_validation_sheet.csv")
FILLED_VARIANT = Path("results/analysis/n3_validation_sheet_variant.csv")  # second pass; defects = UNION of n's (strict-call rule)
SHEET = Path("results/analysis/n3_census_sheet.csv")
FLAGS = Path("results/analysis/n3_census_flags.json")

# R2: interpretive tails and value pairs inside the swapped clause
R2_PATTERNS = [
    r"\bindicat\w+\b", r"\bsuggest\w+\b", r"\bsignal\w+ing\b",
    r"\bmark\w+ing\b", r"\btransition\w*\s+from\b",
    r"\bfrom\s+(?:a\s+)?(?:high|low)s?\b", r"\bto\s+(?:a\s+)?(?:high|low)s?\b",
    r"\bhigh\s+of\b", r"\blow\s+of\b",
    r"\bcold\s+snap\b", r"\bhot\s+spell\b", r"\bheat\s*wave\b",
    r"\bpeak\w*\b", r"\btrough\w*\b", r"\bbottom\w*\s+out\b",   # R2-ext 2026-08-09: "peaking at" survivors found via the two-pass disagreement (N3|1188/345/626)
    r"-?\d+(?:\.\d+)?\s*°C.*?-?\d+(?:\.\d+)?\s*°C",   # two pinned values in one clause
]
# R3: temperature-direction references
R3_DIRECTION = (r"(?:temperature|temperatures)\b[^.\n]*?"
                r"\b(?:rose|rise|rising|fell|fall(?:ing)?|dropp?\w*|"
                r"climb\w*|increas\w*|decreas\w*|warm\w*|cool\w*|declin\w*)"
                r"|(?:rose|rising|falling|increasing|decreasing|warming|"
                r"cooling|climbing|dropping)\b[^.\n]*?\btemperatures?\b")


def r2_hit(clause: str):
    hits = [p for p in R2_PATTERNS if re.search(p, clause, re.I)]
    return hits


def r3_hit(caption: str, swapped_clause: str):
    """Direction references to temperature OUTSIDE the swapped clause."""
    rest = caption.replace(swapped_clause, " ", 1)
    m = re.search(R3_DIRECTION, rest, re.I)
    return m.group(0)[:80] if m else None


def main():
    items = [json.loads(l) for l in ITEMS.open(encoding="utf-8")]
    pool = [it for it in items
            if it["component"] == "N3" and it["condition"] == "swap"]
    print(f"N3 swap population: {len(pool)}   (expected 389)")
    if len(pool) != 389:
        print("[STOP] population mismatch"); sys.exit(1)

    # ---- load the filled sample (BOTH passes): defects = union of n's ---
    def load_verdicts(path):
        with path.open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.reader(f))
        hdr = {n: i for i, n in enumerate(rows[1])}
        return {r[hdr["item_id"]]: r[hdr["y_or_n"]].strip().lower()
                for r in rows[2:] if r and r[hdr["item_id"]].strip()}

    v1 = load_verdicts(FILLED)
    v2 = load_verdicts(FILLED_VARIANT)
    if set(v1) != set(v2) or len(v1) != 100:
        print("[STOP] the two sheet versions do not cover the same 100 items")
        sys.exit(1)
    judged = v1
    known_defects = ({i for i, v in v1.items() if v == "n"}
                     | {i for i, v in v2.items() if v == "n"})
    known_clean = set(v1) - known_defects
    disagreements = sorted(i for i in v1 if v1[i] != v2[i])
    print(f"already judged: {len(judged)} | two-pass disagreements: "
          f"{len(disagreements)} {disagreements}")
    print(f"defects (UNION of n's, strict-call rule): {len(known_defects)} | "
          f"certified clean: {len(known_clean)}")
    if len(known_defects) != 17:
        print("[STOP] union defect count differs from the committed record "
              "(expected 17)"); sys.exit(1)

    # ---- apply filters ---------------------------------------------------
    flagged = {}
    for it in pool:
        iid = it["item_id"]
        reasons = []
        if it["duration_class"] == "six_months":
            reasons.append("R1:six_months_window")
        h2 = r2_hit(it["clause_replaced_to"])
        if h2:
            reasons.append("R2:" + ";".join(h2[:3]))
        h3 = r3_hit(it["caption_distractor"], it["clause_replaced_to"])
        if h3:
            reasons.append("R3:" + h3)
        if reasons:
            flagged[iid] = reasons

    r_counts = Counter(r.split(":")[0] for v in flagged.values() for r in v)
    print(f"\nflagged: {len(flagged)}/389   by rule: {dict(r_counts)}")
    print(f"P-c2 (R1 == 23): {'CONFIRMED' if r_counts.get('R1', 0) == 23 else 'MISSED'}"
          f"   (R1 = {r_counts.get('R1', 0)})")
    in_range = 60 <= len(flagged) <= 160
    print(f"P-c3 (total in [60,160]): "
          f"{'CONFIRMED' if in_range else 'MISSED'}   (total = {len(flagged)})")

    # ---- GATE: filters must capture all known defects (P-c1) ------------
    missed = sorted(known_defects - set(flagged))
    print(f"\n[GATE P-c1] known defects captured: "
          f"{len(known_defects) - len(missed)}/17")
    if missed:
        print("[GATE FAILED] these human-judged defects match NO rule — a "
              "mechanism is not codified. Extend the rules before any census "
              "judging. Missed items:")
        for iid in missed:
            it = next(x for x in pool if x["item_id"] == iid)
            print(f"  {iid}: clause_to = {it['clause_replaced_to'][:110]!r}")
        sys.exit(1)
    print("[GATE ok] all 17 known defects are captured by R1-R3")

    # sanity view: how many already-certified items get flagged anyway
    # (expected and fine — filters over-flag; certified verdicts stand)
    overflag = len(known_clean & set(flagged))
    print(f"certified-clean items also flagged (over-flagging, informational): "
          f"{overflag}/{len(known_clean)}")

    # ---- census sheet: flagged minus already-judged ---------------------
    to_judge = [it for it in pool
                if it["item_id"] in flagged and it["item_id"] not in judged]
    to_judge.sort(key=lambda x: x["item_id"])
    with SHEET.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow([f"# N3 census sheet | {len(to_judge)} flagged items not "
                    f"in the 100-sample | rules R1-R3 in the flag column | "
                    f"same two-part y/n test and mechanics as the sample"])
        w.writerow(["reading_order", "item_id", "duration_class", "flags",
                    "swap_from", "swap_to", "clause_replaced_from",
                    "clause_replaced_to", "swapped_caption_to_judge",
                    "y_or_n", "notes"])
        for i, it in enumerate(to_judge, 1):
            w.writerow([i, it["item_id"], it["duration_class"],
                        " | ".join(flagged[it["item_id"]]),
                        it["swap_from"], it["swap_to"],
                        it["clause_replaced_from"], it["clause_replaced_to"],
                        it["caption_distractor"], "", ""])

    FLAGS.write_text(json.dumps({
        "population": len(pool), "flagged": len(flagged),
        "by_rule": dict(r_counts),
        "known_defects_captured": f"{len(known_defects) - len(missed)}/{len(known_defects)}",
        "certified_clean_overflagged": overflag,
        "census_rows_to_judge": len(to_judge),
        "flags": flagged,
    }, indent=2), encoding="utf-8")

    print(f"\ncensus sheet -> {SHEET}   ({len(to_judge)} rows to judge)")
    print(f"flag record  -> {FLAGS}")
    print("\nUnflagged items are certified-by-rule ONLY IF the gate above "
          "passed; the excision set will be (17 known defects + census n's) "
          "plus matched random twins, computed next session.")


if __name__ == "__main__":
    main()

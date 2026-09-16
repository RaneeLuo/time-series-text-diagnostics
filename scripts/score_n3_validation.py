#!/usr/bin/env python3
"""
score_n3_validation.py (v2, union-aware) — committed record of the N3
sample verdict, over BOTH judging passes.

History: two filled versions of the sheet existed; 4 rows differ.
Strict-call rule (pinned): defects = UNION of n's across passes.
  N3|220: n in primary (mechanism noted), y in variant (no note) -> n.
  N3|1188, N3|345, N3|626: y in primary, n in variant with a shared
  concrete mechanism ("peaking at/mid-week" survivors) -> n.
Expected: primary 86/100; union defects 17; union batch1 42/50;
defect rate 0.170, Wilson 95% CI [0.1090, 0.2547]; by class (union)
week 7 / 28_days 5 / six_months 5. Verdict FAILED under either reading.

Run from thesis repo root:
  python scripts/score_n3_validation.py
"""
import csv, math
from collections import Counter
from pathlib import Path

PRIMARY = Path("results/analysis/n3_validation_sheet.csv")
VARIANT = Path("results/analysis/n3_validation_sheet_variant.csv")

def load(path):
    rows = list(csv.reader(path.open(encoding="utf-8-sig", newline="")))
    h = {n: i for i, n in enumerate(rows[1])}
    out = {}
    for r in rows[2:]:
        if r and r[h["item_id"]].strip():
            y = r[h["y_or_n"]].strip().lower()
            if y not in ("y", "n"):
                raise SystemExit(f"[STOP] {path.name} row "
                                 f"{r[h['reading_order']]}: y_or_n={y!r}")
            out[r[h["item_id"]]] = (int(r[h["reading_order"]]), y,
                                    r[h["duration_class"]])
    if len(out) != 100:
        raise SystemExit(f"[STOP] {path.name}: {len(out)} items, expected 100")
    return out

p1, p2 = load(PRIMARY), load(VARIANT)
if set(p1) != set(p2):
    raise SystemExit("[STOP] the two sheets cover different items")
dis = sorted((i for i in p1 if p1[i][1] != p2[i][1]), key=lambda i: p1[i][0])
print(f"two-pass disagreements: {len(dis)} (expected 4): {dis}")

defects = {i for i in p1 if p1[i][1] == "n" or p2[i][1] == "n"}
prim_defects = {i for i in p1 if p1[i][1] == "n"}
b1_union = sum(1 for i in p1 if p1[i][0] <= 50 and i not in defects)
print(f"primary: {100-len(prim_defects)}/100 pass (expected 86)")
print(f"UNION:   {100-len(defects)}/100 pass | defects {len(defects)} "
      f"(expected 17) | batch1 {b1_union}/50 (expected 42) -> criterion "
      f"{'FAILED' if b1_union < 48 else 'met'} at boundary 1")
print(f"union defects by class: "
      f"{dict(Counter(p1[i][2] for i in defects))} "
      f"(expected week 7 / 28_days 5 / six_months 5)")
n, k = 100, len(defects); p = k/n; z = 1.959963984540054
den = 1+z*z/n; ctr=(p+z*z/(2*n))/den
hw = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/den
print(f"union defect rate {p:.3f}  Wilson 95% CI "
      f"[{max(0,ctr-hw):.4f}, {ctr+hw:.4f}] (expected [0.1090, 0.2547])")
print("\nVERDICT (pre-registered, unchanged under either reading): "
      "certification FAILED; N3 provisional; census engaged with the "
      "17-item union as the known-defect gate set.")

#!/usr/bin/env python3
"""
analyze_narrative_slices.py — the two designed-in slices from §4.5 Q4
that the main stats run did not cover.

  1. duration_class slice: N1 skews to short windows (annotated at
     generation). Does swap accuracy depend on window duration?
     Registered prediction P-dur (2026-08-08, before running): spread
     across N1 duration classes <= 0.15 and no class below 0.75.
  2. header (N1+N2+N5) vs prose (N3) swap sensitivity — the overlap
     diagnostic, downgraded to descriptive after the source read showed
     no text-matching pathway exists in the inference score.
     Registered prediction P-hp: header ~0.91, prose ~0.72, gap ~0.19,
     stable across seeds.

Deterministic (no resampling). Run from thesis repo root:
  python scripts/analyze_narrative_slices.py
"""

import json
from collections import defaultdict
from pathlib import Path

ITEMS = Path("data/processed/narrative_probe_items_certified.jsonl")
RESULTS = Path("results/experiments/trace_narrative_per_item.jsonl")
OUT = Path("results/analysis/trace_narrative_slices.json")

HEADER = {"N1", "N2", "N5"}


def acc(rows):
    return sum(r["correct"] for r in rows) / len(rows) if rows else float("nan")


def main():
    items = {it["item_id"]: it
             for it in (json.loads(l) for l in ITEMS.open(encoding="utf-8"))}
    recs = [json.loads(l) for l in RESULTS.open(encoding="utf-8")]
    print(f"items: {len(items)} | records: {len(recs)}")
    if len(items) != 3178 or len(recs) != 9534:
        print("[STOP] counts differ from the committed record (3178 / 9534)")
        return
    unmatched = sum(1 for r in recs if r["item_id"] not in items)
    if unmatched:
        print(f"[STOP] {unmatched} records have no matching item")
        return
    seeds = sorted({r["seed"] for r in recs})
    print(f"seeds: {seeds}\n")
    payload = {}

    # ---- 1. duration_class slice ----------------------------------------
    print("=" * 78)
    print("DURATION-CLASS SLICE (swap condition; random shown for context)")
    print("=" * 78)
    payload["duration_slice"] = {}
    for comp in ("N1", "N2", "N3", "N5"):
        comp_recs = [r for r in recs if r["component"] == comp]
        classes = sorted({items[r["item_id"]]["duration_class"]
                          for r in comp_recs})
        if len(classes) < 2:
            print(f"{comp}: single duration class ({classes}) — no slice")
            continue
        print(f"\n{comp}  (classes: {classes})")
        print(f"  {'class':<14}{'n_items':>8}{'swap acc':>10}"
              f"{'per-seed':>26}{'random acc':>12}")
        payload["duration_slice"][comp] = {}
        swap_accs = {}
        for cl in classes:
            sw = [r for r in comp_recs if r["condition"] == "swap"
                  and items[r["item_id"]]["duration_class"] == cl]
            rd = [r for r in comp_recs if r["condition"] == "random"
                  and items[r["item_id"]]["duration_class"] == cl]
            per_seed = {s: round(acc([r for r in sw if r["seed"] == s]), 3)
                        for s in seeds}
            a_sw, a_rd = acc(sw), acc(rd)
            swap_accs[cl] = a_sw
            n_it = len(sw) // len(seeds)
            print(f"  {str(cl):<14}{n_it:>8}{a_sw:>10.3f}"
                  f"{str(list(per_seed.values())):>26}{a_rd:>12.3f}")
            payload["duration_slice"][comp][str(cl)] = {
                "n_items": n_it, "acc_swap_pooled": a_sw,
                "acc_swap_per_seed": per_seed, "acc_random_pooled": a_rd,
            }
        spread = max(swap_accs.values()) - min(swap_accs.values())
        worst = min(swap_accs.values())
        print(f"  -> spread {spread:.3f}, worst class {worst:.3f}")
        if comp == "N1":
            p_ok = spread <= 0.15 and worst >= 0.75
            print(f"  -> P-dur ({'CONFIRMED' if p_ok else 'MISSED'}): "
                  f"predicted spread <= 0.15 and worst >= 0.75")
            payload["duration_slice"]["P_dur_confirmed"] = p_ok

    # ---- 2. header vs prose ---------------------------------------------
    print("\n" + "=" * 78)
    print("HEADER (N1+N2+N5) vs PROSE (N3) — swap condition")
    print("=" * 78)
    payload["header_vs_prose"] = {}
    for s in seeds:
        h = [r for r in recs if r["seed"] == s and r["condition"] == "swap"
             and r["component"] in HEADER]
        p = [r for r in recs if r["seed"] == s and r["condition"] == "swap"
             and r["component"] == "N3"]
        ah, ap = acc(h), acc(p)
        print(f"  {s}: header {ah:.3f} (n={len(h)})  prose {ap:.3f} "
              f"(n={len(p)})  gap {ah-ap:+.3f}")
        payload["header_vs_prose"][s] = {"header": ah, "prose": ap,
                                         "gap": ah - ap}
    print("\n  Disposition (recorded): descriptive only. The inference-time")
    print("  text-matching pathway does not exist (mm_encoder source read,")
    print("  2026-08-08), and the observed direction — prose MOST degraded —")
    print("  is the opposite of what contamination would produce.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()

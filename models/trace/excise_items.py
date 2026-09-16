#!/usr/bin/env python3
"""
excise_items.py — produce the certified narrative item set by removing
human-certified defective items, Probe-1 census style: machine flags,
human certifies, defects excised, both numbers reported.

Round-2 context (2026-08-09): human validation flagged one N3 item
(N3|461); a mechanical audit of all 400 N3 swap items found the mechanism
(temporal-peak clauses surviving an up->down swap) in 15 items. The human
judges those 15; the confirmed-defective ids go in a text file, one item_id
per line (the swap id, e.g. 'N3|461|swap'). This script removes each
listed swap item AND its matched random item (same component and row) to
keep pairs symmetric, and writes the certified set plus an excision record.

Usage (from thesis repo root):
    python models/trace/excise_items.py --ids results/analysis/n3_excision_ids.txt
Reads:  data/processed/narrative_probe_items.jsonl
Writes: data/processed/narrative_probe_items_certified.jsonl
        results/analysis/narrative_items_excision_report.json
"""

from __future__ import annotations
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

SRC = Path("data/processed/narrative_probe_items.jsonl")
DST = Path("data/processed/narrative_probe_items_certified.jsonl")
REP = Path("results/analysis/narrative_items_excision_report.json")


def fail(gate, msg):
    print(f"\n[GATE FAILED] {gate}: {msg}")
    sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True,
                    help="text file: one defective swap item_id per line")
    args = ap.parse_args()

    if not SRC.is_file():
        fail("G1", f"{SRC} missing")
    items = [json.loads(l) for l in open(SRC, encoding="utf-8")]
    ids = [l.strip() for l in open(args.ids, encoding="utf-8")
           if l.strip() and not l.startswith("#")]
    if not ids:
        fail("G1-ids", "no ids in the file — if the human verdict is that "
                       "nothing is defective, no certified file is needed")

    by_id = {i["item_id"]: i for i in items}
    # G2: every listed id must exist and be a swap item
    for iid in ids:
        if iid not in by_id:
            fail("G2-exists", f"{iid} not found in the item set")
        if by_id[iid]["condition"] != "swap":
            fail("G2-swap", f"{iid} is not a swap item — list swap ids only")

    # each excised swap takes its matched random twin (same prefix, |random)
    excise = set()
    for iid in ids:
        excise.add(iid)
        twin = iid.rsplit("|", 1)[0] + "|random"
        if twin in by_id:
            excise.add(twin)
        else:
            print(f"[warn] {iid} has no random twin — excising swap only")

    kept = [i for i in items if i["item_id"] not in excise]
    n_before = Counter((i["component"], i["condition"]) for i in items)
    n_after = Counter((i["component"], i["condition"]) for i in kept)

    print(f"[excise] listed defective swap ids: {len(ids)}")
    print(f"[excise] total items removed (incl. random twins): "
          f"{len(items) - len(kept)}")
    print(f"{'component/condition':<24}{'before':>8}{'after':>8}")
    for key in sorted(n_before):
        print(f"{key[0]+'/'+key[1]:<24}{n_before[key]:>8}{n_after.get(key, 0):>8}")

    # G3: only the listed component may change
    changed_comps = {k[0] for k in n_before if n_before[k] != n_after.get(k, 0)}
    listed_comps = {iid.split("|")[0] for iid in ids}
    if changed_comps != listed_comps:
        fail("G3-scope", f"components changed {changed_comps} != listed "
                         f"{listed_comps}")

    with open(DST, "w", encoding="utf-8") as f:
        for i in kept:
            f.write(json.dumps(i, ensure_ascii=False) + "\n")
    REP.parent.mkdir(parents=True, exist_ok=True)
    with open(REP, "w", encoding="utf-8") as f:
        json.dump({"source": str(SRC), "certified": str(DST),
                   "human_certified_defective_swap_ids": ids,
                   "total_removed_including_twins": len(items) - len(kept),
                   "counts_before": {f"{k[0]}/{k[1]}": v for k, v in n_before.items()},
                   "counts_after": {f"{k[0]}/{k[1]}": v for k, v in n_after.items()},
                   }, f, indent=2)
    print(f"\n[done] certified set -> {DST}")
    print(f"[done] excision record -> {REP}")
    print("The certified file is what the runner consumes; the original is "
          "retained unmodified for the record. Report both counts in the "
          "thesis, as with the C4 census.")


if __name__ == "__main__":
    main()

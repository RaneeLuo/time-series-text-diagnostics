#!/usr/bin/env python3
"""
inspect_noaa_narratives.py — read the NOAA test-split descriptions in full,
so the option-(b) component grammar is designed on real data, not on
truncated console snippets.

Prints: column/field inventory, full text of the first rows, the labels
vocabulary with frequencies, per-field length statistics, events content,
and a near-duplication check. Read-only; writes one JSON summary.

Run from the thesis repo root:
    python models/trace/inspect_noaa_narratives.py --trace-repo ../TRACE-Multimodal-TSEncoder
Writes: results/analysis/noaa_narrative_inventory.json
"""

from __future__ import annotations
import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def fail(gate: str, msg: str):
    print(f"\n[GATE FAILED] {gate}: {msg}")
    sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-repo", required=True)
    ap.add_argument("--n-full", type=int, default=6,
                    help="how many rows to print in full")
    args = ap.parse_args()

    import pyarrow.parquet as pq

    repo = Path(args.trace_repo).resolve()
    parquet = repo / "dataset" / "retrieval" / "test" / "test.parquet"
    if not parquet.is_file():
        fail("G1", f"{parquet} missing — expected the nested layout from the "
                   "demo-reproduction run")

    df = pq.read_table(parquet).to_pandas()
    print(f"[G1] {parquet.name}: {len(df)} rows, columns {list(df.columns)}")
    if len(df) != 2006:
        fail("G1-count", f"expected 2006 rows, got {len(df)}")

    # train row count for reference, metadata only (no full load)
    train_pq = repo / "dataset" / "retrieval" / "train.parquet"
    if train_pq.is_file():
        print(f"[info] train.parquet rows (metadata read): "
              f"{pq.ParquetFile(train_pq).metadata.num_rows}")

    # ---- field inventory across all descriptions ------------------------
    desc = df["description"]
    key_counts = Counter()
    types_seen = Counter()
    for d in desc:
        if isinstance(d, dict):
            for k, v in d.items():
                key_counts[k] += 1
                types_seen[(k, type(v).__name__)] += 1
        else:
            types_seen[("<non-dict description>", type(d).__name__)] += 1
    print("\n[fields] description keys (count present / 2006):")
    for k, c in key_counts.most_common():
        print(f"    {k:<20} {c}")
    odd = [t for t in types_seen if t[1] not in ("str",)]
    if odd:
        print(f"[fields] non-string field types observed: {odd}")

    # ---- full text of the first rows ------------------------------------
    print("\n" + "=" * 74)
    print(f"FULL TEXT — first {args.n_full} rows (description + events)")
    print("=" * 74)
    for i in range(args.n_full):
        print(f"\n----- row {i} (id={df.iloc[i]['id']!r}) -----")
        d = df.iloc[i]["description"]
        if isinstance(d, dict):
            for k, v in d.items():
                print(f"  [{k}]")
                print(f"    {v}")
        else:
            print(f"  {d!r}")
        ev = df.iloc[i]["events"]
        print(f"  [events] {ev!r}")

    # ---- labels vocabulary ----------------------------------------------
    vocab = Counter()
    set_sizes = Counter()
    label_sets = []
    for d in desc:
        if isinstance(d, dict) and isinstance(d.get("labels"), str):
            toks = [t.strip() for t in d["labels"].split(",") if t.strip()]
            vocab.update(toks)
            set_sizes[len(toks)] += 1
            label_sets.append(frozenset(toks))
        else:
            label_sets.append(None)
    print("\n[labels] vocabulary (token: frequency):")
    for tok, c in vocab.most_common():
        print(f"    {tok:<15} {c}")
    print(f"[labels] label-set sizes: {dict(sorted(set_sizes.items()))}")
    distinct_sets = Counter(s for s in label_sets if s is not None)
    print(f"[labels] distinct label SETS: {len(distinct_sets)}; "
          f"top 5: {[ (sorted(s), c) for s, c in distinct_sets.most_common(5) ]}")

    # ---- per-field length stats -----------------------------------------
    print("\n[lengths] words per field (min / median / max over rows where present):")
    import statistics
    for k in key_counts:
        lens = [len(str(d[k]).split()) for d in desc
                if isinstance(d, dict) and k in d and d[k] is not None]
        if lens:
            print(f"    {k:<20} {min(lens)} / {int(statistics.median(lens))} / {max(lens)}")

    # ---- events ----------------------------------------------------------
    ev_nonnull = df["events"].map(
        lambda e: e is not None and not (isinstance(e, float)) and str(e) not in ("", "None", "nan")).sum()
    print(f"\n[events] rows with non-empty events: {ev_nonnull} / {len(df)}")

    # ---- near-duplication check -----------------------------------------
    # exact duplicate label-set + DATE field, the demo's confusion pattern
    sig = Counter()
    for d in desc:
        if isinstance(d, dict):
            sig[(str(d.get("DATE")), str(d.get("labels")), str(d.get("location"))[:40])] += 1
    dupes = {k: c for k, c in sig.items() if c > 1}
    print(f"[dup] rows sharing identical (DATE, labels, location-prefix): "
          f"{sum(dupes.values())} rows in {len(dupes)} groups"
          + (f"; largest group {max(dupes.values())}" if dupes else ""))

    # ---- save summary ----------------------------------------------------
    out = Path("results/analysis/noaa_narrative_inventory.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "n_rows": len(df),
            "fields": dict(key_counts),
            "labels_vocab": dict(vocab),
            "label_set_sizes": {str(k): v for k, v in set_sizes.items()},
            "n_distinct_label_sets": len(distinct_sets),
            "events_nonnull": int(ev_nonnull),
            "near_dup_groups": len(dupes),
        }, f, indent=2)
    print(f"\n[done] summary -> {out}. Paste the FULL console output back — "
          "the grammar design reads the actual text, not the summary.")


if __name__ == "__main__":
    main()

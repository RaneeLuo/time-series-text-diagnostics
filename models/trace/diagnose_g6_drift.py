#!/usr/bin/env python3
"""
Diagnose the G6 drift found by run_narrative_probe.py (2026-08-08 run:
only 16/3178 caption_correct matched generate_dsp of the claimed row).

This script CHANGES NOTHING. It distinguishes four candidate mechanisms:

  M1  Whitespace-only drift (e.g. newline handling, collapsed spaces
      somewhere between generation and now).
  M2  Row misalignment (sample_id's numeric suffix does not index this
      parquet's row order) — tested by searching ALL rows for each
      item's text.
  M3  The parquet content itself differs from what the generator read
      (re-download / different copy).
  M4  The generator serialised differently than generate_dsp does
      (a real generator-side transformation).

Usage (thesis repo root):
  python models/trace/diagnose_g6_drift.py --trace-repo ../TRACE-Multimodal-TSEncoder
Paste the FULL output back.
"""

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path


def norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def first_diff(a: str, b: str) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n if len(a) != len(b) else -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-repo", required=True)
    ap.add_argument("--items", default="data/processed/narrative_probe_items_certified.jsonl")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    repo = Path(args.trace_repo).resolve()
    import os
    os.environ["TTRAG_DATA_DIR"] = str(repo / "dataset") + "/"
    os.environ["TTRAG_CHECKPOINTS_DIR"] = str(repo / "results/model_checkpoints") + "/"
    os.environ["TTRAG_RESULTS_DIR"] = str(repo / "results/model_results") + "/"
    sys.path.insert(0, str(repo))

    import pyarrow.parquet as pq
    from src.data.load_data import generate_dsp

    parquet = repo / "dataset" / "retrieval" / args.split / f"{args.split}.parquet"
    print(f"[file] {parquet}")
    print(f"[file] size {parquet.stat().st_size:,} B, "
          f"mtime {parquet.stat().st_mtime}")
    h = hashlib.sha256(parquet.read_bytes()).hexdigest()
    print(f"[file] sha256 {h[:16]}...")

    df = pq.read_table(parquet).to_pandas()
    print(f"[parquet] {len(df)} rows | columns: {list(df.columns)}")

    items = [json.loads(l) for l in Path(args.items).open()]
    print(f"[items] {len(items)} loaded")

    # Serialise every row once, exact and normalised
    dsp = [generate_dsp(df.iloc[r]["description"]) for r in range(len(df))]
    dsp_norm_to_rows = {}
    for r, s in enumerate(dsp):
        dsp_norm_to_rows.setdefault(norm_ws(s), []).append(r)
    dup_norm = sum(1 for v in dsp_norm_to_rows.values() if len(v) > 1)
    print(f"[parquet] rows with non-unique normalised serialisation: {dup_norm}")

    # ---- Classify every item --------------------------------------------
    cat = Counter()
    survivors, misaligned_examples, ws_examples, notfound_examples = [], [], [], []
    for it in items:
        row = int(it["sample_id"].rsplit("_", 1)[1])
        cap = it["caption_correct"]
        if cap == dsp[row]:
            cat["exact_match_claimed_row"] += 1
            survivors.append((it["item_id"], row))
            continue
        if norm_ws(cap) == norm_ws(dsp[row]):
            cat["whitespace_only_vs_claimed_row"] += 1        # -> M1
            if len(ws_examples) < 3:
                ws_examples.append((it, row))
            continue
        hit = dsp_norm_to_rows.get(norm_ws(cap))
        if hit:
            cat["matches_OTHER_row_normalised"] += 1          # -> M2
            if len(misaligned_examples) < 5:
                misaligned_examples.append((it["item_id"], row, hit))
        else:
            cat["matches_NO_row_at_all"] += 1                 # -> M3/M4
            if len(notfound_examples) < 3:
                notfound_examples.append((it, row))

    print("\n[classification of all items]")
    for k, v in cat.most_common():
        print(f"  {k}: {v}")

    print(f"\n[survivors] the {len(survivors)} exact matches "
          f"(component pattern is diagnostic):")
    comp = Counter(s[0].split("|")[0] for s in survivors)
    print(f"  by component: {dict(comp)}")
    for iid, r in survivors[:8]:
        print(f"  {iid}  row {r}")

    # ---- Character-level evidence for up to 3 whitespace cases ----------
    for it, row in ws_examples:
        a, b = it["caption_correct"], dsp[row]
        i = first_diff(a, b)
        print(f"\n[ws-diff] {it['item_id']} row {row} | "
              f"len item {len(a)} vs dsp {len(b)} | first diff at char {i}")
        lo = max(0, i - 40)
        print(f"  item : {a[lo:i+40]!r}")
        print(f"  dsp  : {b[lo:i+40]!r}")

    for it, row in notfound_examples:
        a, b = it["caption_correct"], dsp[row]
        i = first_diff(a, b)
        print(f"\n[nomatch-diff] {it['item_id']} row {row} | "
              f"len item {len(a)} vs dsp {len(b)} | first diff at char {i}")
        lo = max(0, i - 40)
        print(f"  item : {a[lo:i+40]!r}")
        print(f"  dsp  : {b[lo:i+40]!r}")

    for iid, claimed, hit in misaligned_examples:
        print(f"\n[misalign] {iid}: claimed row {claimed}, "
              f"normalised text found at row(s) {hit}")

    # ---- pq_id column, if the parquet has an id-like column -------------
    idcols = [c for c in df.columns if "id" in c.lower()]
    if idcols:
        print(f"\n[pq_id check] id-like columns: {idcols}")
        it0 = items[0]
        r0 = int(it0["sample_id"].rsplit("_", 1)[1])
        for c in idcols:
            print(f"  item {it0['item_id']}: pq_id={it0['pq_id']} | "
                  f"df.iloc[{r0}][{c!r}] = {df.iloc[r0][c]!r}")
    else:
        print("\n[pq_id check] no id-like column in the parquet")

    print("\n[verdict guide] Read the classification block:")
    print("  - dominated by whitespace_only_vs_claimed_row  -> M1: rows are "
          "RIGHT, serialisation whitespace drifted; fix = compare/embed via "
          "one declared normalisation, decision needed on which side.")
    print("  - dominated by matches_OTHER_row_normalised    -> M2: row "
          "indexing broke; fix = re-derive the row map, re-gate.")
    print("  - dominated by matches_NO_row_at_all           -> M3/M4: the "
          "parquet or the generator's serialisation differs in content; "
          "STOP and compare against the generation-time record.")


if __name__ == "__main__":
    main()

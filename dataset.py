"""
dataset.py — CLaSP reimplementation: data pipeline (COMPLETED against repo layout).

Verified layout (2026-07-23, from repo inspection):
  data/TRUCE/processed_data/pilot13final{train,val,test}.json   -> truce_synth
  data/TRUCE/processed_data/pilot16b{train,val,test}.json       -> truce_stock
      record: {id, annotations: [[cap],[cap],[cap]], series: [12 floats], meta}
  data/SUSHI_tiny/generated_files_list.csv  columns: "File path","Caption","Class"
  data/SUSHI_tiny/<File path>               signal CSV, 2048 comma-separated values

Canonical output: data/processed/pairs.jsonl — one record per (series, caption) pair.
Downstream code reads ONLY pairs.jsonl.

Usage:
    python dataset.py build      # writes data/processed/pairs.jsonl
    python dataset.py check      # counts per dataset/split + length stats
"""

from __future__ import annotations
import csv
import json
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

TRUCE_DIR = Path("data/TRUCE/processed_data")
SUSHI_DIR = Path("data/SUSHI_tiny")
PROCESSED = Path("data/processed/pairs.jsonl")

SUSHI_SEED = 42          # stratified split seed — FROZEN, never change (spec §3)


# --------------------------------------------------------------------------
# 1) LOADERS (written against verified layout)
# --------------------------------------------------------------------------

def iter_truce_records():
    """TRUCE synthetic (pilot13final*) + stock (pilot16b*), official splits."""
    sources = [("truce_synth", "pilot13final"), ("truce_stock", "pilot16b")]
    for dsname, prefix in sources:
        for split in ("train", "val", "test"):
            path = TRUCE_DIR / f"{prefix}{split}.json"
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            for key, rec in data.items():
                # annotations: list of 3 lists, each holding one caption string
                captions = [c for ann in rec["annotations"]
                            for c in (ann if isinstance(ann, list) else [ann])]
                assert len(captions) == 3, f"{key}: expected 3 captions, got {len(captions)}"
                series = [float(v) for v in rec["series"]]
                assert len(series) == 12, f"{key}: expected length 12, got {len(series)}"
                yield {
                    "dataset": dsname,
                    "split": split,
                    "sample_id": f"{dsname}:{rec['id']}",
                    "series": series,
                    "captions": captions,
                    "class_label": None,
                }


def iter_sushi_records():
    """SUSHI Tiny from generated_files_list.csv; 8:1:1 stratified by Class, seed 42.

    With 10 samples per class this yields exactly 8/1/1 per class.
    Missing signal files (e.g. pruned local copies) are skipped WITH A LOUD WARNING —
    the full Tiny release must be present for the real build.
    """
    meta_path = SUSHI_DIR / "generated_files_list.csv"
    with open(meta_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    by_class: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_class[r["Class"]].append(r)

    rng = random.Random(SUSHI_SEED)
    n_missing = 0
    for cls in sorted(by_class):                       # deterministic class order
        items = sorted(by_class[cls], key=lambda r: r["File path"])
        rng.shuffle(items)                             # deterministic given seed
        n = len(items)
        n_val = max(1, round(n * 0.1))
        n_test = max(1, round(n * 0.1))
        n_train = n - n_val - n_test
        for i, r in enumerate(items):
            split = ("train" if i < n_train
                     else "val" if i < n_train + n_val
                     else "test")
            sig_path = SUSHI_DIR / r["File path"]
            if not sig_path.exists():
                n_missing += 1
                continue
            series = np.loadtxt(sig_path, delimiter=",").ravel()
            assert series.size == 2048, f"{sig_path}: expected 2048, got {series.size}"
            yield {
                "dataset": "sushi",
                "split": split,
                "sample_id": f"sushi:{Path(r['File path']).with_suffix('')}",
                "series": [float(v) for v in series],
                "captions": [r["Caption"]],
                "class_label": r["Class"],
            }
    if n_missing:
        print(f"WARNING: {n_missing} SUSHI signal files listed in metadata are MISSING "
              f"on disk. If this is not 0, restore the full Tiny release before the "
              f"real build (expected 1400 present).", file=sys.stderr)


def build_pairs_jsonl():
    PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(PROCESSED, "w", encoding="utf-8") as f:
        for rec in list(iter_truce_records()) + list(iter_sushi_records()):
            for k, cap in enumerate(rec["captions"]):
                row = {
                    "dataset": rec["dataset"],
                    "split": rec["split"],
                    "sample_id": rec["sample_id"],
                    "caption_id": f'{rec["sample_id"]}#{k}',
                    "series": rec["series"],
                    "caption": cap.strip(),
                    "class_label": rec.get("class_label"),
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                n += 1
    print(f"wrote {n} pairs -> {PROCESSED}")


# --------------------------------------------------------------------------
# 2) RUNTIME DATASET (unchanged from spec)
# --------------------------------------------------------------------------

@dataclass
class Pair:
    dataset: str
    split: str
    sample_id: str
    caption_id: str
    series: np.ndarray
    caption: str
    class_label: str | None


def load_pairs(splits=("train",), datasets=None) -> list[Pair]:
    pairs = []
    with open(PROCESSED, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["split"] not in splits:
                continue
            if datasets and r["dataset"] not in datasets:
                continue
            pairs.append(Pair(
                dataset=r["dataset"], split=r["split"],
                sample_id=r["sample_id"], caption_id=r["caption_id"],
                series=np.asarray(r["series"], dtype=np.float32),
                caption=r["caption"], class_label=r.get("class_label"),
            ))
    return pairs


def znorm(x: np.ndarray) -> np.ndarray:
    """Per-series z-normalization (spec §3). Constant series -> zeros."""
    mu, sd = float(x.mean()), float(x.std())
    if sd < 1e-8:
        return np.zeros_like(x)
    return (x - mu) / sd


class ClaspPairs(Dataset):
    def __init__(self, pairs: list[Pair], normalize: bool = True):
        self.pairs = pairs
        self.normalize = normalize

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        p = self.pairs[i]
        x = znorm(p.series) if self.normalize else p.series
        return {"series": torch.from_numpy(x.copy()), "caption": p.caption,
                "sample_id": p.sample_id, "caption_id": p.caption_id}


def collate(batch):
    lengths = [b["series"].shape[0] for b in batch]
    L = max(lengths)
    x = torch.zeros(len(batch), L)
    mask = torch.zeros(len(batch), L, dtype=torch.bool)
    for i, b in enumerate(batch):
        n = b["series"].shape[0]
        x[i, :n] = b["series"]
        mask[i, :n] = True
    return {
        "series": x, "series_mask": mask,
        "captions": [b["caption"] for b in batch],
        "sample_ids": [b["sample_id"] for b in batch],
        "caption_ids": [b["caption_id"] for b in batch],
    }


def make_loader(splits=("train",), datasets=None, batch_size=64,
                shuffle=True, normalize=True, num_workers=0):
    ds = ClaspPairs(load_pairs(splits, datasets), normalize=normalize)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=collate, num_workers=num_workers)


# --------------------------------------------------------------------------
# 3) CLI
# --------------------------------------------------------------------------

EXPECTED = {
    ("truce_stock", "train"): 4560, ("truce_stock", "val"): 570, ("truce_stock", "test"): 570,
    ("truce_synth", "train"): 1344, ("truce_synth", "val"): 168, ("truce_synth", "test"): 168,
    ("sushi", "train"): 1120, ("sushi", "val"): 140, ("sushi", "test"): 140,
}


def check():
    from collections import Counter
    cnt, lens = Counter(), Counter()
    with open(PROCESSED, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            cnt[(r["dataset"], r["split"])] += 1
            lens[(r["dataset"], len(r["series"]))] += 1
    ok = True
    print("pairs per (dataset, split)   [expected]:")
    for k in sorted(cnt):
        exp = EXPECTED.get(k)
        flag = "OK" if cnt[k] == exp else f"MISMATCH (expected {exp})"
        if cnt[k] != exp:
            ok = False
        print(f"  {k}: {cnt[k]}   [{exp}] {flag}")
    print("series lengths per dataset:", dict(lens))
    print("ALL COUNTS MATCH" if ok else "COUNT MISMATCHES PRESENT — investigate before training")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    if cmd == "build":
        build_pairs_jsonl()
    check()

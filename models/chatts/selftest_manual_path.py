"""
Equivalence self-test for the manual encoding path (CPU only).

Runs every Probe-2 manifest row (1,756 prompts, 386 distinct series)
through BOTH the checkpoint's stock AutoProcessor and the manual path,
demanding identical token ids, identical attention masks, and bitwise-
identical series tensors. Then tests the two override modes the manual
path exists for, on seeded samples of 20 rows each:
  GE4 prefix override: jitter one prefix digit -> token ids must differ
      ONLY in the prefix region (everything from the first <ts> token
      onward identical; decoded texts equal after masking the prefixes),
      tensor bitwise unchanged.
  GE5 series override (A-condition shape): gaussian tensor + original
      prefix -> prefix region identical to stock-original's, tensor
      equals the gaussian encoded through the stock arithmetic.

All gates HARD. Nothing here touches a GPU or the model weights.

Place at: models/chatts/selftest_manual_path.py
Run from repo root:
  python -m models.chatts.selftest_manual_path --pairs data/processed/pairs.jsonl --manifest data/processed/chatts_probe2_mcq.jsonl --checkpoint-meta data/chatts_pinned_meta
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

from models.chatts import perturbations as P
from models.chatts.manual_encoding import (load_checkpoint_sp_encoding,
                                           manual_encode, prefix_text_of,
                                           TS_PLACEHOLDER)

TS_START_ID = 151665
BASE_SEED = 42
FAILURES = []


def gate(name, ok, detail, hard=True):
    status = "PASS" if ok else ("FAIL (HARD)" if hard else "MISS (report)")
    print(f"[{name}] {status} — {detail}")
    if not ok and hard:
        FAILURES.append(name)


def ids_list(t):
    return t["input_ids"][0].tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--checkpoint-meta", required=True)
    args = ap.parse_args()

    print("=== ChatTS manual-path equivalence self-test ===")

    from transformers import AutoTokenizer, AutoProcessor
    tok = AutoTokenizer.from_pretrained(args.checkpoint_meta, trust_remote_code=True)
    proc = AutoProcessor.from_pretrained(args.checkpoint_meta, trust_remote_code=True,
                                         tokenizer=tok)
    sp = load_checkpoint_sp_encoding(args.checkpoint_meta)

    series_of = {}
    with open(args.pairs, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["sample_id"] not in series_of:
                series_of[r["sample_id"]] = np.asarray(r["series"], dtype=np.float64)

    rows = [json.loads(l) for l in open(args.manifest, encoding="utf-8") if l.strip()]
    print(f"[input] manifest rows: {len(rows)}; distinct series available: {len(series_of)}")

    # ---- GE1-GE3: stock vs manual on every row ----
    mismatch = {"ids": 0, "mask": 0, "tensor": 0}
    first_bad = []
    digest = hashlib.sha256()
    for r in rows:
        x = series_of[r["sample_id"]]
        stock = proc(text=[r["prompt"]], timeseries=[x], padding=True, return_tensors="pt")
        man = manual_encode(r["prompt"], [{"tensor_source": x}], tok, sp)
        if ids_list(stock) != ids_list(man):
            mismatch["ids"] += 1
            if len(first_bad) < 3:
                first_bad.append(r["mcq_id"])
        if stock["attention_mask"][0].tolist() != man["attention_mask"][0].tolist():
            mismatch["mask"] += 1
        if not (stock["timeseries"].dtype == man["timeseries"].dtype
                and torch.equal(stock["timeseries"], man["timeseries"])):
            mismatch["tensor"] += 1
        digest.update(bytes(str(ids_list(man)), "utf-8"))
    gate("GE1-ids", mismatch["ids"] == 0,
         f"token-id mismatches: {mismatch['ids']}/{len(rows)}"
         + (f"; first: {first_bad}" if first_bad else ""))
    gate("GE2-tensor", mismatch["tensor"] == 0,
         f"tensor mismatches (bitwise): {mismatch['tensor']}/{len(rows)}")
    gate("GE3-mask", mismatch["mask"] == 0,
         f"attention-mask mismatches: {mismatch['mask']}/{len(rows)}")

    # seeded row sample for override tests
    rng = np.random.Generator(np.random.PCG64(BASE_SEED))
    sample = [rows[i] for i in rng.choice(len(rows), size=20, replace=False)]

    # ---- GE4: prefix override localized-diff ----
    bad4 = 0
    for r in sample:
        x = series_of[r["sample_id"]]
        true_prefix = prefix_text_of(x, sp)
        mean, scale = P.sp_prefix_numbers(x)
        jit = f"[Value Offset: {-mean + 0.0001:.4f}|Value Scaling: {scale:.4f}]"
        if jit == true_prefix:  # degenerate jitter (rounding ate it) — bump more
            jit = f"[Value Offset: {-mean + 0.001:.4f}|Value Scaling: {scale:.4f}]"
        base = manual_encode(r["prompt"], [{"tensor_source": x}], tok, sp)
        jmod = manual_encode(r["prompt"], [{"tensor_source": x, "prefix_text": jit}],
                             tok, sp)
        bi, ji = ids_list(base), ids_list(jmod)
        # from the first <ts> token onward, everything must be identical
        tail_ok = (TS_START_ID in bi and TS_START_ID in ji
                   and bi[bi.index(TS_START_ID):] == ji[ji.index(TS_START_ID):])
        # decoded texts equal once each prefix is masked out
        text_ok = (tok.decode(base["input_ids"][0]).replace(true_prefix, "@P@")
                   == tok.decode(jmod["input_ids"][0]).replace(jit, "@P@"))
        tensor_ok = torch.equal(base["timeseries"], jmod["timeseries"])
        if not (tail_ok and text_ok and tensor_ok):
            bad4 += 1
    gate("GE4-prefix-override", bad4 == 0,
         f"localized-diff failures: {bad4}/20 (jitter must touch ONLY the prefix)")

    # ---- GE5: series override, A-condition shape ----
    bad5 = 0
    for r in sample:
        x = series_of[r["sample_id"]]
        g, g_noop = P.gaussian_matched(x, r["sample_id"], BASE_SEED)
        if g_noop:
            continue  # the constant: pass-through, nothing to test here
        true_prefix = prefix_text_of(x, sp)
        a_cond = manual_encode(r["prompt"],
                               [{"tensor_source": g, "prefix_text": true_prefix}],
                               tok, sp)
        stock_orig = proc(text=[r["prompt"]], timeseries=[x], padding=True,
                          return_tensors="pt")
        stock_gauss = proc(text=[r["prompt"]], timeseries=[g], padding=True,
                           return_tensors="pt")
        ids_ok = ids_list(a_cond) == ids_list(stock_orig)   # text side = original's
        ten_ok = torch.equal(a_cond["timeseries"], stock_gauss["timeseries"])  # series side = gaussian's
        if not (ids_ok and ten_ok):
            bad5 += 1
    gate("GE5-series-override", bad5 == 0,
         f"A-condition construction failures: {bad5}/20 "
         f"(text must equal original's, tensor must equal gaussian's)")

    print("=" * 50)
    print(f"[digest] manual-path ids digest: {digest.hexdigest()[:16]}...")
    if FAILURES:
        print(f"HARD STOP — failed gates: {FAILURES}")
        sys.exit(1)
    print("ALL HARD GATES GREEN. The manual path is proven equivalent on every "
          "real row and correctly decouples prefix from series.")


if __name__ == "__main__":
    main()

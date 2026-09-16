#!/usr/bin/env python3
"""
diagnose_probe2_data.py — data-only addendum to diagnose_probe2_setup.py.

WHY THIS EXISTS
The setup diagnostic settled the layout and the pairing design. Three things
it did not answer decide what the perturbations MEAN, and all three are
answerable from the data alone — no model, no forward pass, ~30 seconds.

  F1/F2  WHERE THE PADDING SITS. Every row is padded (102-180 valid of 186)
         and all 7 channels share one mask. But "permute the first half"
         (sf_half) and "swap the halves" (ex_half) are undefined until we
         know whether the valid positions are one contiguous block and
         whether that block starts at index 0.

  F3     IS THE DATA STANDARDISED? Binding mechanic M1 says masking fills 0
         on the Z-NORMALISED input, and the parent's loader standardises
         before batching (ablUtils path, data_loader.py:111-133). If TRACE
         hands the model raw NOAA values, filling 0 is not a neutral erasure
         — it is a real reading (0 degrees, 0 m/s). Masking is the contrast
         half of TRACE's conclusion axis, so this cannot be assumed.
         Checked two ways: pooled per-channel stats (global standardisation)
         and the spread of per-(row, channel) means and sds (instance
         normalisation). Shuffles are unaffected either way — a permutation
         commutes with any per-channel affine rescaling.

  F4/F5  WHAT THE 498 CONSTANT CHANNELS HOLD, and whether they account for
         the 3.23% natural zeros. If a constant channel sits at 0.0, masking
         it is a no-op and must be flagged as such (G1 no-op policy), not
         silently counted as a perturbation.

RECONCILIATION GATES against the previous run — these must match exactly or
one of the two scripts is wrong:
  R1  constant channels (range == 0 over valid points) == 498
  R2  total natural zeros on valid points                == 77,278
  R3  total valid points                                 == 2,393,055
  R4  rows                                               == 2,006, all with
      7 channels sharing one mask pattern

REGISTERED EXPECTATIONS (written before the run; misses recorded as misses):
  F1  Valid positions form ONE contiguous block per row.      mod-high
  F2  That block starts at index 0 (padding is trailing).     moderate
  F3  Data is NOT standardised: pooled per-channel means far  low-mod
      from 0 and sds far from 1.
  F4  Most constant channels hold exactly 0.0.                moderate
  F5  >= 90% of all natural zeros come from constant channels. moderate

USAGE (from the thesis repo root):
  python models/trace/diagnose_probe2_data.py --trace-repo ../TRACE-Multimodal-TSEncoder

Paste the entire console output back.
"""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

EXPECTED_ROWS = 2006
EXPECTED_CHANNELS = 7
SEQ_LEN = 186
BATCH_SIZE = 32
R1_CONST_CHANNELS = 498
R2_TOTAL_ZEROS = 77278
R3_TOTAL_VALID = 2393055


def fail(gate, msg):
    print(f"\n[GATE FAILED] {gate}: {msg}")
    print("Hard stop. Investigate before proceeding — do not work around.")
    sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-repo", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out",
                    default="results/analysis/probe2_trace_data_addendum.json")
    args = ap.parse_args()
    t0 = time.time()
    report = {"script": "diagnose_probe2_data.py"}

    print("=" * 72)
    print("TRACE Probe-2 DATA ADDENDUM — data only, no model, nothing modified")
    print("=" * 72)

    repo = Path(args.trace_repo).resolve()
    if not (repo / "src/data/dataloader.py").is_file():
        fail("G1-repo", f"{repo} does not look like the TRACE repo root")
    import os
    os.environ["TTRAG_DATA_DIR"] = str(repo / "dataset") + "/"
    os.environ["TTRAG_CHECKPOINTS_DIR"] = str(repo / "results/model_checkpoints") + "/"
    os.environ["TTRAG_RESULTS_DIR"] = str(repo / "results/model_results") + "/"
    sys.path.insert(0, str(repo))

    import numpy as np
    import pyarrow.parquet as pq
    import torch
    from tqdm import tqdm

    ckpt_path = repo / "results/model_checkpoints/context_align/retriever_demo.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    margs = ckpt["args"]          # args only — the model is NOT constructed
    del ckpt
    print(f"[setup] checkpoint args loaded; model NOT built (data-only run)")

    # ---- parquet columns: what are the 7 channels? -----------------------
    parquet = repo / "dataset" / "retrieval" / args.split / f"{args.split}.parquet"
    df = pq.read_table(parquet).to_pandas()
    print(f"\n--- parquet inspection " + "-" * 48)
    print(f"  rows: {len(df)} | columns: {list(df.columns)}")
    chan_col = next((c for c in df.columns
                     if "channel" in c.lower() and "desc" in c.lower()), None)
    if chan_col is not None:
        try:
            v0 = df.iloc[0][chan_col]
            seq = list(v0) if not isinstance(v0, str) else [v0]
            print(f"  '{chan_col}' for row 0 ({len(seq)} entries):")
            for i, s in enumerate(seq):
                print(f"    ch{i}: {str(s)[:90]}")
        except Exception as e:
            print(f"  could not unpack '{chan_col}': {e}")
    else:
        print("  no channel-description column found; channels stay unnamed")
    report["parquet_columns"] = [str(c) for c in df.columns]

    from src.data.dataloader import get_dataloader
    margs.task_name = "retrieval"
    margs.data_split = args.split
    margs.batch_size = BATCH_SIZE
    margs.device = torch.device("cpu")
    margs.distributed = False
    margs.rank = 0
    loader = get_dataloader(margs)
    print(f"  batches: {len(loader)}")

    # ---- scan ------------------------------------------------------------
    starts = Counter()
    noncontig_rows = 0
    block_lengths = Counter()
    nonuniform_rows = 0

    ch_sum = np.zeros(EXPECTED_CHANNELS)      # pooled per-channel-index stats
    ch_sqs = np.zeros(EXPECTED_CHANNELS)
    ch_cnt = np.zeros(EXPECTED_CHANNELS)

    inst_means, inst_stds = [], []            # per (row, channel)
    const_values = []
    const_by_chidx = Counter()
    n_const = 0
    zeros_total = 0
    zeros_from_const = 0
    valid_total = 0
    n_rows = 0

    for b in tqdm(loader, total=len(loader), desc="scanning"):
        x = b.timeseries.double().numpy()          # [B, 7, 186]
        m = (b.input_mask.double().numpy() > 0.5)
        if x.shape[1] != EXPECTED_CHANNELS or x.shape[2] != SEQ_LEN:
            fail("R4-shape", f"batch shape {x.shape} unexpected")
        for i in range(x.shape[0]):
            xi, vi = x[i], m[i]
            n_rows += 1
            if not (vi == vi[0]).all():
                nonuniform_rows += 1
            idx = np.flatnonzero(vi[0])
            if len(idx) == 0:
                fail("R4-empty", f"row {n_rows-1} has no valid positions")
            span = idx.max() - idx.min() + 1
            if span != len(idx):
                noncontig_rows += 1
            else:
                block_lengths[int(len(idx))] += 1
            starts[int(idx.min())] += 1

            cnt = vi.sum(axis=1).astype(float)             # [7]
            s = np.where(vi, xi, 0.0).sum(axis=1)
            mean = s / cnt
            var = (np.where(vi, (xi - mean[:, None]) ** 2, 0.0)).sum(axis=1) / cnt
            std = np.sqrt(np.maximum(var, 0.0))
            hi = np.where(vi, xi, -np.inf).max(axis=1)
            lo = np.where(vi, xi, np.inf).min(axis=1)
            rng = hi - lo
            zc = (np.where(vi, xi, 1.0) == 0.0).sum(axis=1)

            const = rng == 0.0
            n_const += int(const.sum())
            for c in np.flatnonzero(const):
                const_values.append(float(mean[c]))
                const_by_chidx[int(c)] += 1
            zeros_total += int(zc.sum())
            zeros_from_const += int(zc[const].sum())
            valid_total += int(cnt.sum())

            ch_sum += s
            ch_sqs += np.where(vi, xi ** 2, 0.0).sum(axis=1)
            ch_cnt += cnt
            inst_means.extend(mean.tolist())
            inst_stds.extend(std.tolist())

    # ---- reconciliation gates -------------------------------------------
    print(f"\n--- reconciliation against the setup run " + "-" * 30)
    ok = True
    for name, got, exp in [("R1 constant channels", n_const, R1_CONST_CHANNELS),
                           ("R2 natural zeros", zeros_total, R2_TOTAL_ZEROS),
                           ("R3 valid points", valid_total, R3_TOTAL_VALID),
                           ("R4 rows", n_rows, EXPECTED_ROWS)]:
        mark = "OK" if got == exp else "MISMATCH"
        if got != exp:
            ok = False
        print(f"  {name:24s} {got:>10,}  expected {exp:>10,}   {mark}")
    print(f"  rows with non-uniform channel masks: {nonuniform_rows} (expect 0)")
    if not ok or nonuniform_rows:
        fail("R-reconcile", "this scan disagrees with the setup diagnostic — "
                            "one of the two is wrong; do not proceed")
    print("  all reconciliation gates PASSED")

    # ---- F1/F2 padding position -----------------------------------------
    print(f"\n--- F1/F2: where the padding sits " + "-" * 37)
    print(f"  rows whose valid positions are NON-contiguous: {noncontig_rows}"
          f"   (F1 expects 0)")
    print(f"  start index of the valid block (top 5): {starts.most_common(5)}")
    prefix = starts.get(0, 0)
    print(f"  rows whose block starts at index 0: {prefix}/{n_rows}"
          f"   (F2 expects {n_rows})")
    if noncontig_rows == 0 and prefix == n_rows:
        print("  => VALID = [0, V) per row; padding is TRAILING.")
        print("     sf_half permutes [0, V//2); ex_half swaps [0,V//2) with")
        print("     [V//2, V); padding [V, 186) is never touched.")
    elif noncontig_rows == 0:
        print("  => contiguous but NOT prefix-aligned — the half-splits must")
        print("     be defined relative to the block start, not index 0.")
    else:
        print("  *** valid positions are scattered — 'first half' has no")
        print("      single natural definition. STOP; this needs a decision.")
    report.update({"F1_noncontiguous_rows": noncontig_rows,
                   "F2_block_start_counts": {str(k): v for k, v in sorted(starts.items())},
                   "F2_rows_prefix_aligned": prefix,
                   "block_length_counts": {str(k): v for k, v in sorted(block_lengths.items())}})

    # ---- F3 normalisation -------------------------------------------------
    print(f"\n--- F3: is the data standardised? " + "-" * 37)
    pooled_mean = ch_sum / ch_cnt
    pooled_std = np.sqrt(np.maximum(ch_sqs / ch_cnt - pooled_mean ** 2, 0.0))
    print(f"  pooled stats per channel index (over all valid points):")
    for c in range(EXPECTED_CHANNELS):
        print(f"    ch{c}: mean {pooled_mean[c]:+12.4f}   sd {pooled_std[c]:12.4f}"
              f"   n {int(ch_cnt[c]):,}")
    im = np.array(inst_means); ist = np.array(inst_stds)
    print(f"  per-(row,channel) mean: median {np.median(im):+.4f}, "
          f"5th {np.percentile(im,5):+.4f}, 95th {np.percentile(im,95):+.4f}")
    print(f"  per-(row,channel) sd  : median {np.median(ist):.4f}, "
          f"5th {np.percentile(ist,5):.4f}, 95th {np.percentile(ist,95):.4f}")
    glob_std = bool(np.all(np.abs(pooled_mean) < 0.05) and
                    np.all(np.abs(pooled_std - 1) < 0.05))
    inst_norm = bool(abs(np.median(im)) < 0.05 and abs(np.median(ist) - 1) < 0.05)
    print(f"  => globally standardised per channel: {glob_std}")
    print(f"  => instance (per row+channel) normalised: {inst_norm}")
    if not (glob_std or inst_norm):
        print("  => F3 HIT: the model receives RAW units. Filling 0 is a real")
        print("     reading, not a neutral erasure. The masking mechanic needs")
        print("     an explicit decision before the runner is written.")
    else:
        print("  => F3 MISSED: the input is already normalised, so filling 0")
        print("     matches M1 directly and the parent's mechanic transfers.")
    report.update({"F3_pooled_mean": pooled_mean.tolist(),
                   "F3_pooled_sd": pooled_std.tolist(),
                   "F3_instance_mean_median": float(np.median(im)),
                   "F3_instance_sd_median": float(np.median(ist)),
                   "F3_globally_standardised": glob_std,
                   "F3_instance_normalised": inst_norm})

    # ---- F4/F5 constant channels and zeros --------------------------------
    print(f"\n--- F4/F5: the 498 constant channels " + "-" * 34)
    cv = np.array(const_values)
    n_zero_const = int((cv == 0.0).sum())
    print(f"  constant channels: {n_const} | holding exactly 0.0: "
          f"{n_zero_const} ({n_zero_const/max(1,n_const):.1%})   "
          f"(F4 expects most)")
    if n_const - n_zero_const:
        nz = cv[cv != 0.0]
        print(f"  non-zero constant values: min {nz.min():+.4f} "
              f"max {nz.max():+.4f} | most common "
              f"{Counter(np.round(nz,4).tolist()).most_common(5)}")
    print(f"  constant channels by channel index: "
          f"{sorted(const_by_chidx.items())}")
    frac = zeros_from_const / max(1, zeros_total)
    print(f"  natural zeros from constant channels: {zeros_from_const:,}"
          f"/{zeros_total:,} ({frac:.1%})   (F5 expects >= 90%)")
    print(f"  zeros from VARYING channels: {zeros_total-zeros_from_const:,}"
          f" — these are genuine 0 readings inside live signals")
    report.update({"F4_constant_channels": n_const,
                   "F4_constant_at_zero": n_zero_const,
                   "F4_constant_by_channel_index": {str(k): v for k, v in sorted(const_by_chidx.items())},
                   "F5_zeros_from_constant": zeros_from_const,
                   "F5_zeros_total": zeros_total,
                   "F5_fraction_from_constant": frac})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    report["runtime_seconds"] = round(time.time() - t0, 1)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n[done] -> {out}  ({report['runtime_seconds']} s)")
    print("\nPREDICTION LEDGER:")
    print(f"  F1 contiguous .......... {'HIT' if noncontig_rows==0 else 'MISS'}")
    print(f"  F2 prefix-aligned ...... {'HIT' if prefix==n_rows else 'MISS'}")
    print(f"  F3 not standardised .... {'HIT' if not (glob_std or inst_norm) else 'MISS'}")
    print(f"  F4 mostly zero-valued .. {'HIT' if n_zero_const > n_const/2 else 'MISS'}"
          f"  ({n_zero_const}/{n_const})")
    print(f"  F5 >=90% zeros ......... {'HIT' if frac >= 0.90 else 'MISS'}"
          f"  ({frac:.1%})")
    print("\nPaste the ENTIRE output back.")


if __name__ == "__main__":
    main()

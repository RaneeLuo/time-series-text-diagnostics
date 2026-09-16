#!/usr/bin/env python3
"""
diagnose_probe3_setup.py — TRACE Probe-3 setup diagnostic (run BEFORE any
runner code exists, per the Probe-2 precedent: the diagnostics' misses, if
any, are the arm's foundation).

Design frame accepted 2026-08-15 (this session): renorm-always construction
(surrogates built on the valid block of the model-input tensor, then
re-normalised per channel — exactly reconstructs raw-level construction
regardless of where the pipeline's instance normalisation lives, because
resampling and matched-gaussian draws commute with per-channel affine maps
once followed by renormalisation); joint-channel index draw for resample;
per-channel matched gaussian; dead channels (ptp==0) pass through unchanged;
strata V=168/180 pre-registered; sf_all rung read from committed Probe-2
records, never rerun.

WHAT THIS SCRIPT ANSWERS (register in the chat, mirrored here):
  Part A — committed records only (preview-safe):
    A1 HARD  per-query files: 4,012 lines each; dep 2,005 / amb 1 (row 1191);
             valid_len consistent across seeds and directions.
    A2 HARD  JG pre-verification: dependent-group MRR (unpert + 4 perts,
             both directions) reconstructed from records == committed
             summary <= 1e-9, all seeds.
    A3 PRED  distinct V = 57; dep strata V=168 -> 1,050, V=180 -> 544,
             other -> 411. Row 1191's V: reading, no prediction.
    A4 EXACT chance references H(n)/n from MEASURED sizes.
    A5 REPORT per-stratum sf_all residual vs both references (committed
             data only — previews nothing about the new conditions).
  Part B — needs the TRACE sibling repo:
    B1 HARD  parquet 2,006 rows; G0 row order vs description cache <= 1e-5.
    B2 HARD  geometry every row; 7*sum(V) = 2,393,055; natural zeros =
             77,278; dead channels = 498.
    B3 PRED  dead split 441 all-zero / 57 nonzero-constant.
    B4 READ  non-dead max|mean| <= 1e-5; sd matches exactly one ddof
             convention within 1e-3 (which one: open reading).
    B5 READ  normalisation locus: src/data source scan + parquet-vs-loader
             row-0 value comparison. Weak prediction: baked-in.
    B6 HARD  post-renorm surrogate channels |mean| <= 1e-5, |sd-1| <= 1e-5.
    B7 PRED  pre-renorm RMS mean-drift ~ 0.077, band [0.069, 0.085].
    B8 HARD  commutation: affine input -> identical surrogate after renorm,
             <= 1e-5, 5 rows, renormed channels only.
    B9 HARD  all 498 dead channels elementwise unchanged under BOTH
             constructions (G8-T preview).
    B10 MEAS constant-collapse census (non-dead channel drawn constant under
             the joint draw), seeds 13/14/15, redraw rule row-level joint,
             suffix "|redraw{k}", max 8. Predicted total 0-10.
    B11 CHECK matched gaussian == standard gaussian after renorm, <= 1e-6,
             5 rows (non-fatal; the analytic identity's numeric echo).

USAGE (repo root; PowerShell continuation shown):
  python models/trace/diagnose_probe3_setup.py `
      --trace-repo ..\\TRACE-Multimodal-TSEncoder

Writes: results/analysis/probe3_trace_setup_diagnostic.json
Runtime: one dataloader pass + array work; ~5-10 min CPU; no model forward
passes; $0. Paste the ENTIRE console output back.
"""

from __future__ import annotations
import argparse
import hashlib
import json
import math
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

SEEDS = [13, 14, 15]
PERTS = ["sf_all", "sf_half", "ex_half", "masking"]
EXPECTED_ROWS = 2006
SEQ_LEN = 186
N_CHANNELS = 7
AMBIGUOUS_ROW = 1191
R_VALID_POINTS = 2393055
R_NATURAL_ZEROS = 77278
R_DEAD_CHANNELS = 498
STRATA = (168, 180)
MAX_REDRAWS = 8

FAILED = False


def fail(gate, msg):
    print(f"\n[GATE FAILED] {gate}: {msg}")
    print("Hard stop. Investigate before proceeding — do not work around.")
    sys.exit(1)


def note(tag, msg):
    print(f"[{tag}] {msg}")


def harmonic_mrr(n: int) -> float:
    """Expected MRR when the ground truth's rank is uniform on 1..n."""
    return sum(1.0 / k for k in range(1, n + 1)) / n


def row_seed(row: int, cond: str, seed: int, redraw: int = 0) -> int:
    tag = f"trace_{row}|{cond}|{seed}"
    if redraw:
        tag += f"|redraw{redraw}"
    h = hashlib.sha256(tag.encode()).hexdigest()
    return int(h[:12], 16)


def renorm_block(blk: np.ndarray) -> np.ndarray:
    """Per-channel renormalisation of a [C, V] float32 block. Stats in
    float64, result cast back to float32 (error-#14 discipline: the tensor
    the model sees stays native float32). Channels with ptp==0 pass through
    unchanged — caller decides whether that is expected (dead) or a
    constant-collapse needing a redraw."""
    out = blk.astype(np.float64, copy=True)
    for c in range(out.shape[0]):
        if np.ptp(out[c]) == 0.0:
            out[c] = blk[c]  # pass-through, bit-preserving
            continue
        mu = out[c].mean()
        sd = out[c].std()  # ddof=0; convention question is B4's to settle
        out[c] = (out[c] - mu) / sd
    return out.astype(np.float32)


def build_resample(blk: np.ndarray, dead_mask: np.ndarray, row: int,
                   seed: int):
    """Joint-channel resample with the pinned redraw rule.
    Returns (surrogate [C,V] float32, n_redraws, idx)."""
    C, V = blk.shape
    for k in range(MAX_REDRAWS + 1):
        rng = np.random.default_rng(row_seed(row, "resample", seed, k))
        idx = rng.integers(0, V, size=V)
        drawn = blk[:, idx]
        collapsed = False
        for c in range(C):
            if not dead_mask[c] and np.ptp(drawn[c]) == 0.0:
                collapsed = True
                break
        if not collapsed:
            sur = drawn.copy()
            for c in range(C):
                if dead_mask[c]:
                    sur[c] = blk[c]  # dead channels: identity pass-through
            return renorm_block(sur), k, idx
    fail("B10-redraw", f"row {row} seed {seed}: joint draw still collapsed "
                       f"after {MAX_REDRAWS} redraws")


def build_gaussian(blk: np.ndarray, dead_mask: np.ndarray, row: int,
                   seed: int):
    """Per-channel matched gaussian, renormed. Dead channels pass through."""
    C, V = blk.shape
    rng = np.random.default_rng(row_seed(row, "gaussian", seed))
    sur = np.empty_like(blk)
    for c in range(C):
        if dead_mask[c]:
            sur[c] = blk[c]
        else:
            mu = float(blk[c].astype(np.float64).mean())
            sd = float(blk[c].astype(np.float64).std())
            sur[c] = rng.normal(mu, sd, size=V).astype(np.float32)
    return renorm_block(sur)


# =========================================================================
# Part A — committed Probe-2 records only
# =========================================================================
def part_a(exp_dir: Path, out: dict):
    print("=" * 74)
    print("PART A — committed Probe-2 records (preview-safe)")
    print("=" * 74)

    summary = json.load(open(exp_dir / "probe2_trace_summary.json",
                             encoding="utf-8"))

    vlen_ref = None
    stratum_tables = {}
    for seed in SEEDS:
        p = exp_dir / f"probe2_trace_per_query_seed{seed}.jsonl"
        rows = [json.loads(l) for l in open(p, encoding="utf-8")]
        if len(rows) != 2 * EXPECTED_ROWS:
            fail("A1-lines", f"{p.name}: {len(rows)} lines != 4,012")
        by_dir = {}
        for d in ("text2ts", "ts2text"):
            sel = [r for r in rows if r["direction"] == d]
            if len(sel) != EXPECTED_ROWS:
                fail("A1-dir", f"{p.name}/{d}: {len(sel)} != 2,006")
            if [r["row_idx"] for r in sel] != list(range(EXPECTED_ROWS)):
                fail("A1-order", f"{p.name}/{d}: row_idx not 0..2005 in order")
            by_dir[d] = sel
        groups = Counter(r["group"] for r in by_dir["text2ts"])
        if dict(groups) != {"dependent": 2005, "ambiguous": 1}:
            fail("A1-groups", f"{p.name}: {dict(groups)}")
        if by_dir["text2ts"][AMBIGUOUS_ROW]["group"] != "ambiguous":
            fail("A1-amb", f"{p.name}: ambiguous row is not {AMBIGUOUS_ROW}")
        vl = np.array([r["valid_len"] for r in by_dir["text2ts"]])
        vl2 = np.array([r["valid_len"] for r in by_dir["ts2text"]])
        if not np.array_equal(vl, vl2):
            fail("A1-vlen", f"{p.name}: valid_len differs across directions")
        if vlen_ref is None:
            vlen_ref = vl
        elif not np.array_equal(vl, vlen_ref):
            fail("A1-vlen-seed", f"{p.name}: valid_len differs across seeds")
        note("A1", f"seed {seed}: 4,012 lines, groups OK, valid_len "
                   f"consistent — PASSED")

        # A2 — JG reconstruction vs committed summary
        stab = summary["seeds"][str(seed)]["tables"]
        for d in ("text2ts", "ts2text"):
            dep = [r for r in by_dir[d] if r["group"] == "dependent"]
            recon = {"unperturbed":
                     float(np.mean([1.0 / r["rank_unperturbed"] for r in dep]))}
            for pert in PERTS:
                recon[pert] = float(np.mean([1.0 / r[f"rank_{pert}"]
                                             for r in dep]))
            comm = stab[f"{d}/dependent"]
            for cond, val in recon.items():
                cval = comm[cond]["mrr"]
                if abs(val - cval) > 1e-9:
                    fail("A2-JG", f"seed {seed} {d} {cond}: reconstructed "
                                  f"{val!r} vs committed {cval!r}")
            if d == "text2ts":
                rd = (recon["unperturbed"] - recon["sf_all"]) \
                     / recon["unperturbed"]
                note("A2", f"seed {seed} text2ts dep: unpert MRR "
                           f"{recon['unperturbed']:.4f}, sf_all "
                           f"{recon['sf_all']:.4f}, rel deg {rd:+.2%} "
                           f"(expect 97.7-97.9%)")
        note("A2", f"seed {seed}: records == summary <= 1e-9, both "
                   f"directions — PASSED")

        # A5 — per-stratum sf_all residual (dep, text2ts), committed data
        dep = [r for r in by_dir["text2ts"] if r["group"] == "dependent"]
        for label, sel in [("V=168", [r for r in dep if r["valid_len"] == 168]),
                           ("V=180", [r for r in dep if r["valid_len"] == 180]),
                           ("other", [r for r in dep
                                      if r["valid_len"] not in STRATA])]:
            if not sel:
                continue
            m_u = float(np.mean([1.0 / r["rank_unperturbed"] for r in sel]))
            m_s = float(np.mean([1.0 / r["rank_sf_all"] for r in sel]))
            stratum_tables.setdefault(label, {})[seed] = {
                "n": len(sel), "mrr_unpert": m_u, "mrr_sf_all": m_s}

    # A3 — V structure from records
    distinct_v = sorted(set(int(v) for v in vlen_ref))
    dep_mask = np.ones(EXPECTED_ROWS, bool)
    dep_mask[AMBIGUOUS_ROW] = False
    n168 = int(((vlen_ref == 168) & dep_mask).sum())
    n180 = int(((vlen_ref == 180) & dep_mask).sum())
    nother = int(dep_mask.sum()) - n168 - n180
    amb_v = int(vlen_ref[AMBIGUOUS_ROW])
    for name, got, pred in [("distinct V", len(distinct_v), 57),
                            ("dep V=168", n168, 1050),
                            ("dep V=180", n180, 544),
                            ("dep other", nother, 411)]:
        mark = "HIT" if got == pred else "MISS (recorded)"
        note("A3", f"{name}: {got}  (predicted {pred})  {mark}")
    note("A3", f"ambiguous row {AMBIGUOUS_ROW} valid_len = {amb_v} "
               f"(reading, no prediction)")

    # A4 — chance references from MEASURED sizes
    refs = {"pool_2006": harmonic_mrr(EXPECTED_ROWS)}
    stratum_sizes = {int(v): int(((vlen_ref == v)).sum())
                     for v in distinct_v}
    for v in STRATA:
        refs[f"within_V{v}_n{stratum_sizes[v]}"] = \
            harmonic_mrr(stratum_sizes[v])
    note("A4", "chance references (uniform-rank MRR = H(n)/n):")
    for k, v in refs.items():
        print(f"      {k:<28} {v:.7f}")

    # A5 print
    print("\n  [A5] per-stratum sf_all residual, dep text2ts, committed "
          "records (report only)")
    print(f"    {'stratum':<8}{'seed':>6}{'n':>7}{'unpert':>10}"
          f"{'sf_all':>10}{'x global':>10}{'x within':>10}")
    for label in ("V=168", "V=180", "other"):
        for seed in SEEDS:
            t = stratum_tables[label][seed]
            xg = t["mrr_sf_all"] / refs["pool_2006"]
            if label.startswith("V="):
                nn = stratum_sizes[int(label[2:])]
                xw = t["mrr_sf_all"] / harmonic_mrr(nn)
                xw_s = f"{xw:>10.2f}"
            else:
                xw_s = f"{'—':>10}"
            print(f"    {label:<8}{seed:>6}{t['n']:>7}"
                  f"{t['mrr_unpert']:>10.4f}{t['mrr_sf_all']:>10.4f}"
                  f"{xg:>10.2f}{xw_s}")

    out["part_a"] = {"distinct_v": distinct_v,
                     "stratum_sizes_all_rows": stratum_sizes,
                     "dep_strata": {"V168": n168, "V180": n180,
                                    "other": nother},
                     "ambiguous_row_valid_len": amb_v,
                     "chance_references": refs,
                     "sf_all_residual_by_stratum": {
                         lab: {str(s): stratum_tables[lab][s]
                               for s in SEEDS}
                         for lab in stratum_tables}}
    return vlen_ref


# =========================================================================
# Part B — TRACE sibling repo
# =========================================================================
def part_b(args, vlen_records, out: dict):
    print("\n" + "=" * 74)
    print("PART B — TRACE data side (loader pass, locus, calibration)")
    print("=" * 74)

    repo = Path(args.trace_repo).resolve()
    for rel in ("src/models/mm_encoder.py", "src/data/dataloader.py"):
        if not (repo / rel).is_file():
            fail("B0-repo", f"{repo/rel} not found")

    # ---- B5 part 1: source scan for normalisation code -------------------
    print("\n  [B5] source scan of src/data for normalisation keywords "
          "(report only):")
    pat = re.compile(r"\bmean\b|\bstd\b|normal|scal", re.IGNORECASE)
    hits = []
    for py in sorted((repo / "src" / "data").rglob("*.py")):
        for ln, line in enumerate(
                open(py, encoding="utf-8", errors="replace"), 1):
            if pat.search(line) and not line.lstrip().startswith("#"):
                hits.append(f"{py.relative_to(repo)}:{ln}: "
                            f"{line.rstrip()[:100]}")
    if hits:
        for h in hits[:40]:
            print("      " + h)
        if len(hits) > 40:
            print(f"      ... plus {len(hits)-40} more (in the JSON)")
    else:
        print("      (no matches — a value transform in the loader is "
              "unlikely)")
    out["b5_source_scan"] = hits

    import os
    os.environ["TTRAG_DATA_DIR"] = str(repo / "dataset") + "/"
    os.environ["TTRAG_CHECKPOINTS_DIR"] = \
        str(repo / "results/model_checkpoints") + "/"
    os.environ["TTRAG_RESULTS_DIR"] = str(repo / "results/model_results") + "/"
    sys.path.insert(0, str(repo))

    import pyarrow.parquet as pq
    import torch
    from tqdm import tqdm

    parquet = repo / "dataset" / "retrieval" / "test" / "test.parquet"
    df = pq.read_table(parquet).to_pandas()
    if len(df) != EXPECTED_ROWS:
        fail("B1-rows", f"parquet has {len(df)} rows != {EXPECTED_ROWS}")
    note("B1", f"parquet rows: {len(df)} — PASSED")
    print("\n  [B5] parquet columns (report only):")
    pq_ts_col = None
    for col in df.columns:
        v0 = df.iloc[0][col]
        desc = type(v0).__name__
        arr = None
        if isinstance(v0, np.ndarray):
            arr = v0
        elif isinstance(v0, (list, tuple)) and len(v0) and \
                isinstance(v0[0], (np.ndarray, list, tuple, float, int)):
            try:
                arr = np.asarray(v0, dtype=np.float64)
            except Exception:
                arr = None
        if arr is not None:
            desc += f" -> array shape {arr.shape}"
            if arr.size in (SEQ_LEN * N_CHANNELS,) or \
                    arr.shape in ((N_CHANNELS, SEQ_LEN), (SEQ_LEN, N_CHANNELS)):
                pq_ts_col = col
        print(f"      {col:<28} {desc}")

    # ---- dataloader ------------------------------------------------------
    ckpt_path = repo / "results/model_checkpoints/context_align/retriever_demo.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    margs = ckpt["args"]
    del ckpt
    margs.task_name = "retrieval"
    margs.data_split = "test"
    margs.batch_size = 32
    margs.device = torch.device("cpu")
    margs.distributed = False
    from src.data.dataloader import get_dataloader
    loader = get_dataloader(margs)

    short = "nomic-embed-text-v1.5"
    cache_path = repo / "dataset" / "retrieval" / "test" / \
        f"description_emb_{short}.pt"
    if not cache_path.is_file():
        fail("B1-cache", f"{cache_path} missing")
    raw_rows = torch.load(cache_path, map_location="cpu").float()

    first = next(iter(loader))
    d0 = float((first.description_emb.cpu().float()
                - raw_rows[:first.description_emb.shape[0]]).abs().max())
    note("B1-G0", f"row order check: max abs diff = {d0:.3e}")
    if d0 > 1e-5:
        fail("B1-G0", "dataloader order != parquet order")

    # ---- full pass: geometry, T7, dead channels, natural zeros ----------
    print("\n  full loader pass (geometry + T7 + dead channels) ...")
    vlens = np.zeros(EXPECTED_ROWS, np.int64)
    dead_rows = {}
    n_dead = n_dead_zero = n_dead_nonzero = 0
    zeros_tot = valid_tot = 0
    max_abs_mean = 0.0
    max_dev_sd0 = 0.0   # |sd(ddof=0) - 1| over non-dead channels
    max_dev_sd1 = 0.0   # |sd(ddof=1) - 1|
    row0_block = None
    r = 0
    for b in tqdm(loader, total=len(loader), desc="pass"):
        x = b.timeseries.float().numpy()
        m = b.input_mask.numpy() > 0.5
        for i in range(x.shape[0]):
            mi, xi = m[i], x[i]
            if not (mi == mi[0]).all():
                fail("B2-uniform", f"row {r}: channels do not share one mask")
            idx = np.flatnonzero(mi[0])
            V = len(idx)
            if V == 0 or idx.max() - idx.min() + 1 != V:
                fail("B2-contig", f"row {r}: valid positions not contiguous")
            if idx.max() != SEQ_LEN - 1:
                fail("B2-align", f"row {r}: block ends at {idx.max()}, "
                                 f"not {SEQ_LEN-1}")
            vlens[r] = V
            blk = xi[:, SEQ_LEN - V:]
            if r == 0:
                row0_block = blk.copy()
            dd = []
            for c in range(N_CHANNELS):
                ch = blk[c].astype(np.float64)
                if np.ptp(ch) == 0.0:
                    dd.append(c)
                    if ch[0] == 0.0:
                        n_dead_zero += 1
                    else:
                        n_dead_nonzero += 1
                    continue
                max_abs_mean = max(max_abs_mean, abs(ch.mean()))
                max_dev_sd0 = max(max_dev_sd0, abs(ch.std(ddof=0) - 1.0))
                max_dev_sd1 = max(max_dev_sd1, abs(ch.std(ddof=1) - 1.0))
            if dd:
                dead_rows[r] = np.array(dd)
                n_dead += len(dd)
            zeros_tot += int((blk == 0.0).sum())
            valid_tot += int(blk.size)
            r += 1
    if r != EXPECTED_ROWS:
        fail("B2-rows", f"pass saw {r} rows != {EXPECTED_ROWS}")
    if not np.array_equal(vlens, vlen_records):
        fail("B2-vlen", "loader valid lengths != the committed per-query "
                        "records' valid_len column")
    note("B2", f"valid_len identical to the committed records — PASSED")
    for nm, got, exp in [("valid points", valid_tot, R_VALID_POINTS),
                         ("natural zeros", zeros_tot, R_NATURAL_ZEROS),
                         ("dead channels", n_dead, R_DEAD_CHANNELS)]:
        mark = "OK" if got == exp else "MISMATCH"
        note("B2", f"{nm:15s} {got:>10,}  expected {exp:>10,}   {mark}")
        if got != exp:
            fail("B2-reconcile", f"{nm} disagrees with the Probe-2 record")

    mark3 = ("HIT" if (n_dead_zero, n_dead_nonzero) == (441, 57)
             else "MISS (recorded)")
    note("B3", f"dead split: all-zero {n_dead_zero} / nonzero-constant "
               f"{n_dead_nonzero}  (predicted 441/57)  {mark3}")

    conv = "ddof=0" if max_dev_sd0 <= max_dev_sd1 else "ddof=1"
    best = min(max_dev_sd0, max_dev_sd1)
    note("B4", f"non-dead channels: max|mean| = {max_abs_mean:.3e} "
               f"(band 1e-5: "
               f"{'HIT' if max_abs_mean <= 1e-5 else 'MISS (recorded)'})")
    note("B4", f"max|sd-1|: ddof=0 -> {max_dev_sd0:.3e}, "
               f"ddof=1 -> {max_dev_sd1:.3e}  => convention {conv} "
               f"(band 1e-3: {'HIT' if best <= 1e-3 else 'MISS (recorded)'})")
    if conv != "ddof=0":
        print("      NOTE: renorm_block here uses ddof=0; the RUNNER must "
              "adopt the measured convention. Flagged for the mechanics "
              "record.")

    # ---- B5 part 2: parquet vs loader row 0 ------------------------------
    locus = "undetermined (no identifiable series column)"
    if pq_ts_col is not None:
        try:
            arr = np.asarray(df.iloc[0][pq_ts_col], dtype=np.float64)
            if arr.shape == (SEQ_LEN, N_CHANNELS):
                arr = arr.T
            elif arr.size == SEQ_LEN * N_CHANNELS and arr.ndim == 1:
                arr = arr.reshape(N_CHANNELS, SEQ_LEN)
            V0 = int(vlens[0])
            stored = arr[:, SEQ_LEN - V0:]
            dmax = float(np.abs(stored - row0_block.astype(np.float64)).max())
            locus = ("baked-in (loader applies no value transform)"
                     if dmax <= 1e-6 else
                     f"loader-side transform present (row-0 max diff {dmax:.3e})")
            note("B5", f"parquet column {pq_ts_col!r} vs loader row 0: "
                       f"max abs diff {dmax:.3e} -> locus: {locus}")
        except Exception as e:
            locus = f"comparison failed ({e}) — settle from the source scan"
            note("B5", locus)
    else:
        note("B5", locus + " — settle from the source scan above")
    out["b5_locus"] = locus

    # ---- calibration: resample + gaussian, seeds 13/14/15 ----------------
    # A second, lighter loop over the loader to build surrogates per row
    # (we need the blocks again; memory-friendlier than storing 2,006 x 7 x V).
    print("\n  calibration pass (resample/gaussian, 3 seeds) ...")
    drift_sq_sum = 0.0
    drift_n = 0
    redraw_census = {s: 0 for s in SEEDS}
    collapse_rows = []
    dead_ok = 0
    post_max_mean = 0.0
    post_max_sddev = 0.0
    comm_rows_done = 0
    comm_max = 0.0
    b11_max = 0.0
    b11_done = 0
    A_AFF, B_AFF = 3.7, -2.1
    r = 0
    for b in tqdm(loader, total=len(loader), desc="calib"):
        x = b.timeseries.float().numpy()
        m = b.input_mask.numpy() > 0.5
        for i in range(x.shape[0]):
            V = int(vlens[r])
            blk = x[i][:, SEQ_LEN - V:]
            dmask = np.zeros(N_CHANNELS, bool)
            if r in dead_rows:
                dmask[dead_rows[r]] = True
            for seed in SEEDS:
                # resample
                rng0 = np.random.default_rng(row_seed(r, "resample", seed))
                idx0 = rng0.integers(0, V, size=V)
                drawn0 = blk[:, idx0]
                for c in range(N_CHANNELS):
                    if not dmask[c]:
                        drift_sq_sum += float(
                            drawn0[c].astype(np.float64).mean()
                            - blk[c].astype(np.float64).mean()) ** 2
                        drift_n += 1
                sur, k, idx = build_resample(blk, dmask, r, seed)
                if k:
                    redraw_census[seed] += k
                    collapse_rows.append((r, seed, k))
                for c in range(N_CHANNELS):
                    if dmask[c]:
                        if not np.array_equal(sur[c], blk[c]):
                            fail("B9", f"row {r} ch {c}: dead channel moved "
                                       f"under resample")
                        dead_ok += 1
                    else:
                        ch = sur[c].astype(np.float64)
                        post_max_mean = max(post_max_mean, abs(ch.mean()))
                        post_max_sddev = max(post_max_sddev,
                                             abs(ch.std(ddof=0) - 1.0))
                # gaussian (seed 13 only: identical math each seed;
                # dead-identity + moments are what we're checking)
                if seed == SEEDS[0]:
                    gsur = build_gaussian(blk, dmask, r, seed)
                    for c in range(N_CHANNELS):
                        if dmask[c]:
                            if not np.array_equal(gsur[c], blk[c]):
                                fail("B9", f"row {r} ch {c}: dead channel "
                                           f"moved under gaussian")
                            dead_ok += 1
                        else:
                            ch = gsur[c].astype(np.float64)
                            post_max_mean = max(post_max_mean, abs(ch.mean()))
                            post_max_sddev = max(post_max_sddev,
                                                 abs(ch.std(ddof=0) - 1.0))
            # B8 commutation + B11, on the first 5 rows, seed 13
            if comm_rows_done < 5:
                seed = SEEDS[0]
                s1, k1, _ = build_resample(blk, dmask, r, seed)
                blk_aff = (A_AFF * blk + B_AFF).astype(np.float32)
                s2, k2, _ = build_resample(blk_aff, dmask, r, seed)
                if k1 != k2:
                    fail("B8", f"row {r}: redraw count differs under affine "
                               f"({k1} vs {k2}) — collapse detection is not "
                               f"affine-invariant")
                nz = ~dmask
                if nz.any():
                    comm_max = max(comm_max, float(
                        np.abs(s1[nz].astype(np.float64)
                               - s2[nz].astype(np.float64)).max()))
                comm_rows_done += 1
                # B11: matched vs standard gaussian after renorm
                rngA = np.random.default_rng(row_seed(r, "gaussian", seed))
                rngB = np.random.default_rng(row_seed(r, "gaussian", seed))
                for c in range(N_CHANNELS):
                    if dmask[c]:
                        continue
                    mu = float(blk[c].astype(np.float64).mean())
                    sd = float(blk[c].astype(np.float64).std())
                    dA = rngA.normal(mu, sd, size=V).astype(np.float32)
                    dB = rngB.normal(0.0, 1.0, size=V).astype(np.float32)
                    rA = renorm_block(dA[None, :])[0].astype(np.float64)
                    rB = renorm_block(dB[None, :])[0].astype(np.float64)
                    b11_max = max(b11_max, float(np.abs(rA - rB).max()))
                b11_done += 1
            r += 1

    rms_drift = math.sqrt(drift_sq_sum / drift_n)
    mark7 = "HIT" if 0.069 <= rms_drift <= 0.085 else "MISS (recorded)"
    note("B7", f"pre-renorm RMS mean-drift = {rms_drift:.4f} "
               f"(predicted ~0.077, band [0.069, 0.085])  {mark7}")
    note("B6", f"post-renorm: max|mean| = {post_max_mean:.3e}, "
               f"max|sd-1| = {post_max_sddev:.3e}  (bands 1e-5)")
    if post_max_mean > 1e-5 or post_max_sddev > 1e-5:
        fail("B6", "post-renorm moments outside band — our own math")
    tot_redraw = sum(redraw_census.values())
    mark10 = "HIT" if tot_redraw <= 10 else "MISS (recorded)"
    note("B10", f"constant-collapse redraws: "
                + ", ".join(f"seed {s}: {redraw_census[s]}" for s in SEEDS)
                + f"  (total {tot_redraw}; predicted 0-10)  {mark10}")
    if collapse_rows:
        for rr, ss, kk in collapse_rows[:20]:
            print(f"      collapse: row {rr} seed {ss} needed {kk} redraw(s)")
    note("B9", f"dead-channel identity checks passed: {dead_ok:,} "
               f"(expected {R_DEAD_CHANNELS*3 + R_DEAD_CHANNELS:,} = "
               f"498 x 3 resample seeds + 498 x 1 gaussian)")
    if dead_ok != R_DEAD_CHANNELS * 4:
        fail("B9-count", f"dead-identity check count {dead_ok} != "
                         f"{R_DEAD_CHANNELS * 4}")
    note("B8", f"commutation (5 rows, renormed channels): max abs diff "
               f"{comm_max:.3e} (gate 1e-5)")
    if comm_max > 1e-5:
        fail("B8", "renorm-always did NOT reconstruct affine-invariant "
                   "construction — the accepted collapse argument fails "
                   "numerically")
    mark11 = "HIT" if b11_max <= 1e-6 else "MISS (recorded, non-fatal)"
    note("B11", f"matched == standard gaussian after renorm "
                f"({b11_done} rows): max abs diff {b11_max:.3e}  {mark11}")

    out["part_b"] = {
        "geometry": {"valid_points": valid_tot, "natural_zeros": zeros_tot,
                     "dead_channels": n_dead,
                     "dead_split": {"all_zero": n_dead_zero,
                                    "nonzero_constant": n_dead_nonzero}},
        "t7": {"max_abs_mean": max_abs_mean,
               "max_dev_sd_ddof0": max_dev_sd0,
               "max_dev_sd_ddof1": max_dev_sd1,
               "convention": conv},
        "calibration": {"rms_mean_drift_prerenorm": rms_drift,
                        "post_renorm_max_abs_mean": post_max_mean,
                        "post_renorm_max_sd_dev": post_max_sddev,
                        "redraw_census": {str(s): redraw_census[s]
                                          for s in SEEDS},
                        "collapse_rows": collapse_rows,
                        "commutation_max_diff": comm_max,
                        "b11_max_diff": b11_max}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-repo", required=True)
    ap.add_argument("--exp-dir", default="results/experiments")
    ap.add_argument("--out",
                    default="results/analysis/probe3_trace_setup_diagnostic.json")
    args = ap.parse_args()
    t0 = time.time()
    out = {"script": "diagnose_probe3_setup.py",
           "design_frame": "renorm-always; joint-channel resample; "
                           "per-channel matched gaussian; dead ptp==0 "
                           "pass-through; redraw rule row-level joint, "
                           "max 8"}

    vlen_records = part_a(Path(args.exp_dir), out)
    part_b(args, vlen_records, out)

    out["runtime_seconds"] = round(time.time() - t0, 1)
    op = Path(args.out)
    op.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(op, "w", encoding="utf-8"), indent=2)
    print(f"\nwrote {op}  ({out['runtime_seconds']} s total)")
    print("ALL GATES PASSED — paste this entire output back before any "
          "runner code is written.")


if __name__ == "__main__":
    main()

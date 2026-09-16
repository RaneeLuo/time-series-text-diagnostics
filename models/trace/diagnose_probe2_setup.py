#!/usr/bin/env python3
"""
diagnose_probe2_setup.py — pre-runner diagnostic for the TRACE Probe-2 arm.

WHAT THIS IS FOR
This script answers the five questions the Probe-2 runner design depends on
and that cannot be settled from the governing documents. It CHANGES NOTHING
and writes one diagnostic JSON. It is not a measurement: no probe number
produced here is quotable.

  D1  Tensor layout: shape/axis order of batch.timeseries, channel count,
      and which axis is time. The joint-channel permutation (binding,
      PROJECT_CONTEXT) goes on the time axis; putting it on the wrong axis
      would silently shuffle channels instead.
  D2  Row order: the narrative runner indexes ts_emb[row] and takes the
      score diagonal, which is only valid if the test dataloader yields
      rows in parquet order. Verified directly, not assumed.
  D3  Padding: valid length per row from input_mask. If rows are padded,
      shuffles must permute ONLY valid positions, and the half-splits must
      use valid_length//2 — otherwise we measure padding corruption
      instead of order destruction.
  D4  G8 targets: constant series (per channel, and whole-row) on valid
      positions, plus the natural-zero rate (G2) that the masking
      perturbation is reported against.
  D5  G6 + determinism: the protocol mask is drawn INSIDE the forward pass
      from torch's global RNG (run_narrative_probe.py docstring D2, source
      read 2026-08-08). Three things must hold before the runner can work:
        (a) re-seeding reproduces the frozen unperturbed P@1 digit-exact;
        (b) two identically-seeded passes give bitwise-identical embeddings;
        (c) RNG consumption is INPUT-INDEPENDENT — i.e. feeding a perturbed
            tensor draws the same number of random values, so a re-seeded
            perturbed pass sees the SAME protocol mask as the unperturbed
            pass. This is what makes the comparison paired. If (c) fails,
            the runner design changes and we stop.

REGISTERED EXPECTATIONS (written before the run; misses recorded as misses):
  E1  n_rows = 2006; cached description embeddings (2006, 768); text dim 384.
  E2  Legacy P@1 (text->ts, max<=truth) reproduces the frozen record
      EXACTLY: seed 13 = 884/2006, seed 14 = 859/2006, seed 15 = 863/2006.
      Any deviation at all is a hard stop.
  E3  Row order matches the parquet: first-batch description_emb equals the
      first B cached rows (max abs diff 0.0).
  E4  Determinism: two passes at the same seed differ by exactly 0.0.
  E5  RNG-state equality between an unperturbed and a shuffled-input pass —
      i.e. the mask draw is shape-driven, not value-driven. MODERATE
      confidence; this is the one most likely to miss.
  E6  All 2006 rows fully valid — input_mask all ones across all 7 channels
      and all 186 steps. LOW confidence. Revised 2026-08-15 after the first
      run showed input_mask is PER-CHANNEL (32,7,186), not [batch, time]:
      the script now also reports whether channels within a row share one
      mask pattern, because "one shared permutation" and "permute only
      valid positions" cannot both hold if they differ.
  E7  Constant channels: at least one exists somewhere in 2006 rows.
      Fully-constant rows (every channel flat): 0. Both LOW confidence.

USAGE (from the thesis repo root, TRACE repo cloned as a sibling):

  python models/trace/diagnose_probe2_setup.py \
      --trace-repo ../TRACE-Multimodal-TSEncoder

Runtime: three full passes for D5(a) (~4 min each on CPU, faster on GPU)
plus two 2-batch passes. Batch size is FIXED at 32 and the dataloader is
untouched — changing either shifts the RNG draws and invalidates D5.

Paste the entire console output back.
"""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

# Frozen record: results/experiments/trace_narrative_summary.json,
# unperturbed_P@1_text2ts_by_seed. Loaded from the file at runtime and
# cross-checked against these literals, so a silent file edit cannot pass.
FROZEN_P1 = {13: 0.4406779706478119,
             14: 0.428215354681015,
             15: 0.43020936846733093}
FROZEN_CORRECT = {13: 884, 14: 859, 15: 863}
EXPECTED_ROWS = 2006
BATCH_SIZE = 32          # DO NOT CHANGE — fixes the RNG draw sequence


def fail(gate, msg):
    print(f"\n[GATE FAILED] {gate}: {msg}")
    print("Hard stop. Investigate before proceeding — do not work around.")
    sys.exit(1)


def describe_batch(b):
    """Return {field: description} without assuming the container type."""
    if hasattr(b, "_fields"):
        keys = list(b._fields)
    elif hasattr(b, "__dataclass_fields__"):
        keys = list(b.__dataclass_fields__)
    elif hasattr(b, "__dict__"):
        keys = list(vars(b))
    else:
        keys = [k for k in dir(b) if not k.startswith("_")
                and not callable(getattr(b, k))]
    out = {}
    for k in keys:
        try:
            v = getattr(b, k)
        except Exception as e:
            out[k] = f"<unreadable: {e}>"
            continue
        if hasattr(v, "shape"):
            out[k] = f"tensor shape={tuple(v.shape)} dtype={v.dtype}"
        elif isinstance(v, (list, tuple)):
            out[k] = f"{type(v).__name__} len={len(v)}"
        else:
            out[k] = f"{type(v).__name__}"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-repo", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--mask-seeds", default="13,14,15")
    ap.add_argument("--device", default=None)
    ap.add_argument("--frozen",
                    default="results/experiments/trace_narrative_summary.json")
    ap.add_argument("--out",
                    default="results/analysis/probe2_trace_shape_diagnostic.json")
    args = ap.parse_args()
    t0 = time.time()
    mask_seeds = [int(s) for s in args.mask_seeds.split(",")]
    report = {"script": "diagnose_probe2_setup.py", "batch_size": BATCH_SIZE}

    print("=" * 72)
    print("TRACE Probe-2 SETUP DIAGNOSTIC — no measurement, nothing modified")
    print("=" * 72)

    # ---- repo layout / checkpoint / data (mirrors run_narrative_probe) ----
    repo = Path(args.trace_repo).resolve()
    for rel in ("src/models/mm_encoder.py", "src/data/dataloader.py",
                "src/data/load_data.py"):
        if not (repo / rel).is_file():
            fail("G1-repo", f"{repo/rel} not found — is --trace-repo the root?")
    print(f"[G1] TRACE repo: {repo}")

    import os
    os.environ["TTRAG_DATA_DIR"] = str(repo / "dataset") + "/"
    os.environ["TTRAG_CHECKPOINTS_DIR"] = str(repo / "results/model_checkpoints") + "/"
    os.environ["TTRAG_RESULTS_DIR"] = str(repo / "results/model_results") + "/"
    sys.path.insert(0, str(repo))

    ckpt_path = repo / "results/model_checkpoints/context_align/retriever_demo.pt"
    if not ckpt_path.is_file():
        fail("G2-ckpt", f"{ckpt_path} missing")
    print(f"[G2] checkpoint: {ckpt_path} ({ckpt_path.stat().st_size:,} bytes)")

    retr_dir = repo / "dataset" / "retrieval"
    parquet = retr_dir / args.split / f"{args.split}.parquet"
    if not parquet.is_file():
        fail("G3-data", f"{parquet} missing (nested layout — demo REPAIR 3)")

    import numpy as np
    import pyarrow.parquet as pq
    import torch
    import torch.nn.functional as F
    from tqdm import tqdm

    device = torch.device(args.device if args.device
                          else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    print(f"[env] torch {torch.__version__} | device {device} | "
          f"seeds {mask_seeds} | batch {BATCH_SIZE} (FIXED)")
    report["torch_version"] = torch.__version__
    report["device"] = str(device)

    df = pq.read_table(parquet).to_pandas()
    n_rows = len(df)
    print(f"[G3] {parquet.name}: {n_rows} rows   (E1 expects {EXPECTED_ROWS})")
    if n_rows != EXPECTED_ROWS:
        fail("G3-rows", f"{n_rows} rows != {EXPECTED_ROWS}")

    # ---- model load: REPAIR 2, verbatim from run_narrative_probe.py -------
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    margs = ckpt["args"]
    enc_name = getattr(margs, "text_encoder_name", None)
    mask_ratio = getattr(margs, "mask_ratio", None)
    print(f"[model] text_encoder_name = {enc_name!r}")
    print(f"[model] mask_ratio        = {mask_ratio!r}  "
          f"(the PROTOCOL mask, drawn inside the forward pass)")
    print(f"[model] seq_len_channel   = "
          f"{getattr(margs, 'seq_len_channel', '<absent>')!r}")
    if enc_name != "nomic-ai/nomic-embed-text-v1.5":
        fail("G-enc", f"encoder {enc_name!r} is not Nomic v1.5")
    report["protocol_mask_ratio"] = mask_ratio
    report["seq_len_channel"] = getattr(margs, "seq_len_channel", None)

    from src.models.mm_encoder import MultiModalEncoder
    from src.data.dataloader import get_dataloader
    from src.models.timeseries_encoders.ts_encoder import TS_Encoder
    from copy import deepcopy

    margs.model_name = "TraceEncoder"

    def _patched_load_model(self, pretraining_task_name):
        pa = deepcopy(self.args)
        pa.task_name = pretraining_task_name
        self.ts_encoder = TS_Encoder(configs=pa)

    MultiModalEncoder._load_model = _patched_load_model
    margs.rank = 0
    model = MultiModalEncoder(margs)
    sd = {k.replace("module.", ""): v for k, v in ckpt["model_state_dict"].items()}
    rep = model.load_state_dict(sd, strict=False)
    if rep.missing_keys or rep.unexpected_keys:
        print(f"[G4] missing: {rep.missing_keys}")
        print(f"[G4] unexpected: {rep.unexpected_keys}")
        fail("G4-statedict", "state dict does not match the constructed model")
    model.to(device).eval()
    if model.training:
        fail("G4-eval", "model not in eval mode")
    print("[G4] state dict strict-clean; model in eval mode")

    # ---- cached description embeddings + mini text path -------------------
    short = enc_name.split("/")[-1]
    cache_path = retr_dir / args.split / f"description_emb_{short}.pt"
    if not cache_path.is_file():
        fail("G3-cache", f"{cache_path} missing — run run_authors_demo_eval.py once")
    raw_rows = torch.load(cache_path, map_location="cpu").float()
    print(f"[G3] cached row embeddings: {tuple(raw_rows.shape)}  "
          f"(E1 expects ({EXPECTED_ROWS}, 768))")
    if raw_rows.shape[0] != n_rows:
        fail("G3-shape", f"cache rows {raw_rows.shape[0]} != {n_rows}")
    report["raw_text_emb_shape"] = list(raw_rows.shape)

    @torch.no_grad()
    def text_path(raw):
        e = model.text_adapter(raw.to(device))
        e = model.norm(e)
        e = model.dropout(e)
        return F.normalize(e, dim=-1).cpu()

    text_rows = text_path(raw_rows)
    print(f"[text] projected text embeddings: {tuple(text_rows.shape)}")
    report["text_emb_shape"] = list(text_rows.shape)

    # ---- dataloader (authors' path; test split is shuffle=False) ---------
    margs.task_name = "retrieval"
    margs.data_split = args.split
    margs.batch_size = BATCH_SIZE
    margs.device = device
    margs.distributed = False
    if args.split != "test":
        fail("G-order", "only the test split preserves row order here")
    data_loader = get_dataloader(margs)
    print(f"[loader] batches: {len(data_loader)}")

    # =====================================================================
    # D1 — tensor layout
    # =====================================================================
    print("\n--- D1: batch layout " + "-" * 50)
    first = next(iter(data_loader))
    fields = describe_batch(first)
    for k, v in fields.items():
        print(f"  {k:28s} {v}")
    report["batch_fields"] = fields

    ts = first.timeseries
    im = first.input_mask
    cde = first.channel_description_emb
    print(f"\n  timeseries shape {tuple(ts.shape)} dtype {ts.dtype}")
    print(f"  input_mask shape {tuple(im.shape)} dtype {im.dtype}")
    print(f"  channel_description_emb shape {tuple(cde.shape)}")
    if ts.dim() != 3:
        fail("D1-dim", f"timeseries is {ts.dim()}-D; expected 3-D")
    if ts.dtype != torch.float64:
        print(f"  NOTE: timeseries dtype is {ts.dtype}, not float64 as seen "
              f"on the first run — the .float() cast is still applied.")

    # input_mask is PER-CHANNEL per-timestep here (same shape as the data),
    # not a [batch, time] mask. Recorded 2026-08-15 from the first run.
    if tuple(im.shape) == tuple(ts.shape):
        mask_kind = "per_channel"
    elif im.dim() == 2 and im.shape[1] in ts.shape[1:]:
        mask_kind = "time_only"
    else:
        fail("D1-maskshape", f"input_mask {tuple(im.shape)} is neither the "
                             f"data shape {tuple(ts.shape)} nor a 2-D "
                             f"[batch, time] mask")
    print(f"  => input_mask kind: {mask_kind}")

    # Identify axes from TWO independent sources rather than one:
    #   time    <- seq_len_channel stored in the checkpoint
    #   channel <- channel_description_emb's channel dimension
    seq_expected = getattr(margs, "seq_len_channel", None)
    ch_expected = cde.shape[1] if cde.dim() == 3 else None
    if seq_expected is None or ch_expected is None:
        fail("D1-axisinfo", "cannot cross-check axes: seq_len_channel or "
                            "channel_description_emb unavailable")
    cand_t = [ax for ax in (1, 2) if ts.shape[ax] == seq_expected]
    cand_c = [ax for ax in (1, 2) if ts.shape[ax] == ch_expected]
    if len(cand_t) != 1 or len(cand_c) != 1 or cand_t[0] == cand_c[0]:
        fail("D1-axis", f"ambiguous axes: timeseries {tuple(ts.shape)}, "
                        f"seq_len_channel={seq_expected}, "
                        f"channels={ch_expected}, time cand {cand_t}, "
                        f"channel cand {cand_c}. Guessing is not acceptable.")
    time_axis, chan_axis = cand_t[0], cand_c[0]
    n_channels, L_mask = ts.shape[chan_axis], ts.shape[time_axis]
    print(f"  => TIME axis    = {time_axis} (length {L_mask}, "
          f"matches seq_len_channel={seq_expected})")
    print(f"  => CHANNEL axis = {chan_axis} (count {n_channels}, "
          f"matches channel_description_emb)")
    print(f"  Binding joint shuffle permutes axis {time_axis}, one shared "
          f"permutation across all {n_channels} channels.")
    report.update({"time_axis": time_axis, "channel_axis": chan_axis,
                   "n_channels": int(n_channels), "seq_len": int(L_mask),
                   "input_mask_kind": mask_kind,
                   "timeseries_dtype": str(ts.dtype)})

    # =====================================================================
    # D2 — row order
    # =====================================================================
    print("\n--- D2: row order " + "-" * 53)
    B = ts.shape[0]
    d_batch = first.description_emb.cpu().float()
    d_cache = raw_rows[:B]
    if d_batch.shape != d_cache.shape:
        fail("D2-shape", f"batch description_emb {tuple(d_batch.shape)} vs "
                         f"cache slice {tuple(d_cache.shape)}")
    d_diff = float((d_batch - d_cache).abs().max())
    print(f"  first {B} rows: batch description_emb vs cached rows, "
          f"max abs diff = {d_diff:.3e}   (E3 expects 0.0)")
    if d_diff > 1e-5:
        fail("D2-order", "dataloader does NOT yield parquet row order — the "
                         "diagonal ground truth and ts_emb[row] indexing are "
                         "both invalid. Prediction E3 falsified.")
    print("  D2 PASSED: dataloader order == parquet order")
    report["D2_row_order_max_diff"] = d_diff

    # loader order stability across iterations
    again = next(iter(data_loader))
    st_diff = float((again.timeseries - ts).abs().max())
    print(f"  loader re-iteration stability: max abs diff = {st_diff:.3e}")
    if st_diff != 0.0:
        fail("D2-stability", "the dataloader does not yield the same first "
                             "batch twice — order is not reproducible")
    report["D2_loader_stability_diff"] = st_diff

    # =====================================================================
    # D3 + D4 — padding, constants, natural zeros (no model needed)
    # =====================================================================
    print("\n--- D3/D4: padding, constant series, natural zeros " + "-" * 20)
    valid_per_channel = Counter()      # valid timesteps per (row, channel)
    valid_per_row = Counter()          # union of valid timesteps per row
    rows_uniform_mask = 0              # all 7 channels share one mask pattern
    rows_all_full = 0
    const_channels = 0                 # flat over >=2 valid points
    singleton_channels = 0             # exactly 1 valid point (trivially flat)
    empty_channels = 0                 # 0 valid points
    const_rows = 0
    total_channels = 0
    zeros_valid = 0
    total_valid_points = 0
    nan_rows = 0
    mask_is_binary = True
    n_seen_rows = 0

    for b in tqdm(data_loader, total=len(data_loader), desc="scanning"):
        x = b.timeseries.float()
        m = b.input_mask.float()
        if mask_kind == "time_only":                 # broadcast to per-channel
            m = m.unsqueeze(chan_axis).expand_as(x).contiguous()
        if not set(torch.unique(m).tolist()) <= {0.0, 1.0}:
            mask_is_binary = False
        # normalise to [B, C, T] so the rest of the loop is layout-free
        if chan_axis == 2:
            x = x.transpose(1, 2).contiguous()
            m = m.transpose(1, 2).contiguous()
        Bn, C, T = x.shape
        n_seen_rows += Bn
        valid = m > 0.5
        for i in range(Bn):
            xi, vi = x[i], valid[i]                  # [C, T] each
            vc = vi.sum(dim=1)                       # valid count per channel
            for c in range(C):
                valid_per_channel[int(vc[c].item())] += 1
            total_channels += C
            if bool((vi == vi[0:1]).all()):
                rows_uniform_mask += 1
            union = int(vi.any(dim=0).sum().item())
            valid_per_row[union] += 1
            if union == T and bool(vi.all()):
                rows_all_full += 1
            if torch.isnan(xi[vi]).any():
                nan_rows += 1
            # masked per-channel range, vectorised
            big = torch.where(vi, xi, torch.full_like(xi, float("inf")))
            small = torch.where(vi, xi, torch.full_like(xi, float("-inf")))
            rng_ = small.max(dim=1).values - big.min(dim=1).values
            flat = (rng_ == 0.0) & (vc >= 2)
            const_channels += int(flat.sum().item())
            singleton_channels += int((vc == 1).sum().item())
            empty_channels += int((vc == 0).sum().item())
            if bool(((rng_ == 0.0) | (vc <= 1)).all()):
                const_rows += 1
            zeros_valid += int(((xi == 0.0) & vi).sum().item())
            total_valid_points += int(vi.sum().item())

    if n_seen_rows != n_rows:
        fail("D3-rows", f"scanned {n_seen_rows} rows != {n_rows}")
    print(f"  input_mask binary (0/1 only): {mask_is_binary}")
    print(f"  rows fully valid (all {n_channels} channels x {L_mask} steps): "
          f"{rows_all_full}/{n_rows}")
    print(f"  rows where all channels share ONE mask pattern: "
          f"{rows_uniform_mask}/{n_rows}")
    print(f"    E6 expected: {n_rows}/{n_rows} fully valid")
    all_full = (rows_all_full == n_rows)
    uniform = (rows_uniform_mask == n_rows)
    print(f"  valid timesteps per channel (top 5): "
          f"{valid_per_channel.most_common(5)}")
    print(f"  valid timesteps per row, channel-union (top 5): "
          f"{valid_per_row.most_common(5)}")
    print(f"  channels: empty {empty_channels} | single-point "
          f"{singleton_channels} | total {total_channels}")
    print(f"  rows containing NaN in valid positions: {nan_rows}")
    print(f"  G8 targets — constant channels (>=2 valid pts, flat): "
          f"{const_channels}/{total_channels} | fully-constant rows: "
          f"{const_rows}/{n_rows}")
    print(f"    E7 expected: constant channels > 0, fully-constant rows = 0")
    print(f"  G2 natural exact-zeros on valid points: {zeros_valid}/"
          f"{total_valid_points} ({zeros_valid/max(1,total_valid_points):.4%})")

    print("\n  CONSEQUENCE FOR THE SHUFFLE DESIGN:")
    if all_full:
        print("    All rows fully valid -> one permutation over all "
              f"{L_mask} steps, shared across channels. Binding joint-shuffle "
              "decision applies unchanged; call #3 is a no-op precaution.")
    elif uniform:
        print("    Padding present, but every channel in a row shares the "
              "SAME valid set -> one permutation over that row's valid "
              "positions, still shared across channels. Binding decision "
              "holds; call #3 becomes load-bearing.")
    else:
        print("    *** Padding present AND channels differ within a row. ***")
        print("    'One shared permutation' and 'permute only valid "
              "positions' cannot both hold as written. This needs an "
              "explicit decision before the runner is built — STOP here.")
    report.update({
        "D3_valid_per_channel_counts": {str(k): v for k, v
                                        in sorted(valid_per_channel.items())},
        "D3_valid_per_row_counts": {str(k): v for k, v
                                    in sorted(valid_per_row.items())},
        "D3_rows_all_full": rows_all_full,
        "D3_rows_uniform_mask": rows_uniform_mask,
        "D3_all_rows_full_length": bool(all_full),
        "D3_channels_uniform_within_row": bool(uniform),
        "D3_rows_with_nan": nan_rows,
        "D4_constant_channels": const_channels,
        "D4_singleton_channels": singleton_channels,
        "D4_empty_channels": empty_channels,
        "D4_total_channels": total_channels,
        "D4_fully_constant_rows": const_rows,
        "G2_natural_zero_rate": zeros_valid / max(1, total_valid_points),
    })

    # =====================================================================
    # D5 — G6 reproduction + determinism + input-independence of the RNG
    # =====================================================================
    print("\n--- D5: G6 reproduction and mask determinism " + "-" * 26)
    frozen_file = json.load(open(args.frozen, encoding="utf-8"))
    frozen_p1 = {int(k): v for k, v in
                 frozen_file["unperturbed_P@1_text2ts_by_seed"].items()}
    for s, v in FROZEN_P1.items():
        if s in frozen_p1 and frozen_p1[s] != v:
            fail("D5-frozen", f"seed {s}: summary file says {frozen_p1[s]!r} "
                              f"but the registered literal is {v!r} — the "
                              f"frozen record changed underneath us")
    print(f"  frozen record cross-checked against registered literals: OK")

    @torch.no_grad()
    def full_pass(seed, perturb=False, limit=None):
        """One full (or limited) encoding pass at a fixed seed.
        Returns (embeddings, rng_state_after)."""
        torch.manual_seed(seed)
        chunks = []
        for bi, b in enumerate(data_loader):
            if limit is not None and bi >= limit:
                break
            x = b.timeseries.float()
            if perturb:
                # orientation-only joint shuffle on the time axis; identical
                # permutation across channels, per the binding decision
                g = torch.Generator().manual_seed(20260815 + bi)
                perm = torch.randperm(x.shape[time_axis], generator=g)
                x = x.index_select(time_axis, perm)
            o = model(
                x_enc=x.to(device),
                input_mask=b.input_mask.long().to(device),
                channel_description_emb=b.channel_description_emb.to(device),
                description_emb=b.description_emb.to(device),
                event_emb=b.event_emb.to(device),
            )
            chunks.append(o.embeddings.detach().cpu())
        return torch.cat(chunks).float(), torch.get_rng_state()

    def legacy_p1(ts_emb):
        """The narrative runner's P@1 definition, VERBATIM: a tie for top
        counts as correct (max <= truth), and the mean is taken in float32.
        Used ONLY for the G6 comparison, never for measurement.

        The float32 mean matters. The frozen record 0.4406779706478119 is a
        float32 mean of 2006 booleans; the float64 division 884/2006 is
        0.4406779661016949 — they differ at ~4.4e-9. Reproducing the frozen
        digits requires the same arithmetic, so it is done the same way here.
        The COUNT is the primary gate; the float comparison is a cross-check."""
        e = F.normalize(ts_emb, dim=-1)
        scores = text_rows @ e.T
        truth = scores.diag()
        hits = (scores.max(dim=1).values <= truth)
        return float(hits.float().mean()), int(hits.sum())

    g6 = {}
    emb_seed_first = None
    for seed in mask_seeds:
        t_s = time.time()
        emb, _ = full_pass(seed)
        if emb.shape[0] != n_rows:
            fail("D5-count", f"seed {seed}: {emb.shape[0]} embeddings != {n_rows}")
        p1, n_ok = legacy_p1(emb)
        exp_v = FROZEN_P1.get(seed)
        exp_n = FROZEN_CORRECT.get(seed)
        dev = abs(p1 - exp_v) if exp_v is not None else float("nan")
        print(f"  seed {seed}: legacy P@1 = {p1:.16f}  ({n_ok}/{n_rows})"
              f"  frozen {exp_v!r} ({exp_n})  |dev| = {dev:.3e}"
              f"   [{time.time()-t_s:.0f}s]")
        g6[seed] = {"p1": p1, "n_correct": n_ok, "frozen": exp_v,
                    "frozen_n": exp_n, "dev": dev}
        if exp_n is not None and n_ok != exp_n:
            fail("D5-G6", f"seed {seed}: {n_ok} correct != frozen {exp_n}. "
                          f"The unperturbed pass does not reproduce the frozen "
                          f"record under re-seeding. Prediction E2 falsified — "
                          f"the runner cannot proceed on this design.")
        if seed == mask_seeds[0]:
            emb_seed_first = emb
    print("  D5(a) PASSED: all seeds reproduce the frozen record exactly")
    report["D5_G6"] = {str(k): v for k, v in g6.items()}

    # (b) determinism: same seed twice, first 2 batches, bitwise
    s0 = mask_seeds[0]
    e1, rng1 = full_pass(s0, perturb=False, limit=2)
    e2, rng2 = full_pass(s0, perturb=False, limit=2)
    det = float((e1 - e2).abs().max())
    print(f"\n  D5(b) determinism, seed {s0}, 2 batches: max abs diff = "
          f"{det:.3e}   (E4 expects exactly 0.0)")
    if det != 0.0:
        fail("D5-determinism", "two identically-seeded passes differ — the "
                               "forward pass is not reproducible, so no paired "
                               "comparison is possible. E4 falsified.")
    print("  D5(b) PASSED: bitwise identical")
    report["D5_determinism_max_diff"] = det

    # (c) input-independence of RNG consumption
    e3, rng3 = full_pass(s0, perturb=True, limit=2)
    same_state = bool(torch.equal(rng1, rng3))
    cos = float(F.cosine_similarity(F.normalize(e1, dim=-1),
                                    F.normalize(e3, dim=-1), dim=-1).mean())
    print(f"\n  D5(c) RNG state after unperturbed vs shuffled input: "
          f"identical = {same_state}   (E5 expects True)")
    print(f"        mean cosine(unperturbed, shuffled) over {e1.shape[0]} rows"
          f" = {cos:.4f}   [ORIENTATION ONLY — not a probe measurement]")
    if not same_state:
        print("\n  *** E5 MISSED ***")
        print("  RNG consumption depends on the input, so a re-seeded "
              "perturbed pass does NOT see the same protocol mask as the "
              "unperturbed pass. The paired design in call #2 does not hold "
              "as written. STOP and paste this back before anything else.")
    else:
        print("  D5(c) PASSED: the protocol mask is shape-driven, so "
              "re-seeding gives every condition the same mask — the "
              "comparison is properly paired.")
    report["D5_rng_input_independent"] = same_state
    report["D5_orientation_cosine_shuffled"] = cos

    # ---- write ----------------------------------------------------------
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    report["runtime_seconds"] = round(time.time() - t0, 1)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n[done] diagnostic -> {out}  ({report['runtime_seconds']} s)")
    print("\nPREDICTION LEDGER (fill from the output above):")
    print(f"  E1 rows/shapes ......... {n_rows} rows, text {tuple(text_rows.shape)}")
    _cnt_ok = all(v["n_correct"] == v["frozen_n"] for v in g6.values())
    _flt_ok = all(v["dev"] == 0.0 for v in g6.values())
    print(f"  E2 G6 counts ........... {'HIT' if _cnt_ok else 'MISS'}"
          f"   | float32 digits exact: {'yes' if _flt_ok else 'no'}")
    print(f"  E3 row order ........... {'HIT' if d_diff == 0.0 else f'diff {d_diff:.2e}'}")
    print(f"  E4 determinism ......... {'HIT' if det == 0.0 else 'MISS'}")
    print(f"  E5 RNG input-indep ..... {'HIT' if same_state else 'MISS'}")
    print(f"  E6 all rows full valid . {'HIT' if all_full else 'MISS'}"
          f"   | channels uniform within row: {'yes' if uniform else 'NO'}")
    print(f"  E7 const ch>0, rows=0 .. channels {const_channels}, rows {const_rows}")
    print("\nPaste the ENTIRE output back, including any gate messages.")


if __name__ == "__main__":
    main()

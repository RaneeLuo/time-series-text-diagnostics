#!/usr/bin/env python3
"""
run_probe2.py — TRACE Probe-2 runner (order-invariance shuffle + masking).

Design record: handoff §4.6 (Q1-Q5) and §4.7 item 3; PROJECT_CONTEXT binding
decisions (joint-channel shuffle, mask 0.2 fill 0, M1-M3, D1, D2). Mechanics
T1-T13 accepted 2026-08-15 after the two setup diagnostics
(results/analysis/probe2_trace_shape_diagnostic.json and
 results/analysis/probe2_trace_data_addendum.json).

WHAT THIS MEASURES
TRACE's caption-group DiD is UNPOSABLE (census: 2,005 dependent / 1
ambiguous / 0 invariant of 2,006 — recorded finding). Per the accepted
reframe, TRACE's conclusion-carrying comparison is the DEGRADATION PROFILE
across perturbation types: the shuffle family (destroys order, preserves
values) versus masking (destroys values, largely preserves order). No
prediction is registered on which hurts more — that ranking is the open
question the profile answers. P2-9 (registered 2026-08-10) predicts only
that sf_all degradation exceeds the margin in all three seeds.

FACTS THIS RESTS ON — measured, not assumed (diagnostics of 2026-08-15):
  T1  Every row is padded and RIGHT-ALIGNED: the valid block is the LAST V
      of 186 slots, [186-V, 186), contiguous in all 2,006 rows, and all 7
      channels of a row share one mask. V ranges 102-180, mean 170.4.
      Perturbations touch the valid block ONLY; padding is never moved.
      (Assuming a leading valid block would have shuffled padding and left
      the signal largely intact — a clean run with wrong numbers.)
  T7  The input is instance-normalised: every (row, channel) has mean 0 and
      sd 1 over its valid points. Filling 0 therefore means "replace with
      this channel's own mean", exactly as M1 specifies. The parent's
      mechanic transfers unchanged.
  T8  The 0.3 protocol mask is drawn inside the forward pass from torch's
      global RNG, and RNG consumption is INPUT-INDEPENDENT (verified: RNG
      state after an unperturbed pass equals the state after a shuffled
      pass). Re-seeding before every condition therefore gives the baseline
      and all four perturbations the SAME protocol mask, which is what makes
      this a paired comparison. Batch size 32 and the dataloader are FROZEN
      — changing either shifts the draws and breaks G6.
  --  Masking is reported as 0.3 protocol + 0.2 input, never as bare 0.2.
      The two are drawn independently (ours on the tensor, theirs inside the
      model), so they overlap by chance; the overlap is not observable from
      outside the model and is not claimed here.

PERTURBATIONS (per row, on the float32 tensor the model receives; the valid
block B = x[:, 186-V:186] with one permutation shared across all 7 channels):
  sf_all   permute all V valid positions
  sf_half  permute the first V//2 of the valid block; the rest untouched
  ex_half  swap B[:V//2] with B[V//2:] (uneven for odd V, mirroring CLaSP)
  masking  k = int(0.2*V) valid positions set to 0, SAME positions in every
           channel; k is 0.2 of the REAL length, not of 186
M3 seed scheme: per-row seed = int(sha256("trace_{row}|{pert}|{mask_seed}")
[:12hex], 16); recorded per row in the signal-meta output. Texts are NEVER
perturbed and are embedded once (mask-independent).

GATES (fatal policy per §4.6 item 5, adapted where TRACE differs):
  G3  census buckets 2,005 dep / 1 amb / 0 inv, row_idx contiguous 0..2005,
      ambiguous row is 1191 — HARD STOP
  G0  row order: first-batch description_emb == first 32 cached rows — HARD
      STOP (the diagonal ground truth depends on it)
  R   reconciliation vs the diagnostics: 2,393,055 valid points, 77,278
      natural zeros, 498 dead channels — HARD STOP
  G6  frozen-baseline reproduction: legacy P@1 (max<=truth, float32 mean)
      must equal 884/859/863 for seeds 13/14/15 EXACTLY — HARD STOP.
      Legacy P@1 is used for G6 only; all measurement uses D2 average rank.
  G7  permutation validity: true permutation of the valid block, one shared
      index array asserted identical across all 7 channels, mask position
      count exact — HARD STOP
  G1  applied-check per row: shuffles preserve each channel's value multiset
      and change order (identity survivors FLAGGED no-op, never failed);
      sf_half's tail byte-identical; masking writes only zeros at exactly k
      positions; PADDING BYTE-IDENTICAL in every case — HARD STOP on
      violations
  G8-T identity control (replaces CLaSP's G8; T11): the 498 dead channels
      must be elementwise UNCHANGED under the shuffle family. TRACE has no
      constant rows, so an embedding-level control is impossible — but this
      catches the single most damaging available bug: permuting the CHANNEL
      axis instead of the time axis would move dead channels. Masking is
      excepted (it legitimately zeroes the 57 float-dust dead channels).
  G4  direction sanity: the dependent group must not IMPROVE under sf_all in
      any seed, text->ts — HARD STOP
  G9  pairing: identical query set and order across all five conditions
      before any rank is compared — HARD STOP
  G2  natural-zero rate printed before any masked run (report only)

D2: all measurements use deterministic AVERAGE RANK (ground truth receives
the mean of its exact-tie positions). Direction text->ts is PRIMARY (matches
the frozen anchor and the authors' direction); ts->text is reported as
secondary descriptive from the same score matrix.

USAGE (from the thesis repo root; TRACE repo cloned as a sibling, demo
reproduction already run once so the text caches exist):

  python models/trace/run_probe2.py \
      --trace-repo ../TRACE-Multimodal-TSEncoder \
      --census results/analysis/probe2_trace_order_census.json

Writes:
  results/experiments/probe2_trace_per_query_seed{S}.jsonl
  results/experiments/probe2_trace_signal_meta_seed{S}.json
  results/experiments/probe2_trace_summary.json

Runtime: 15 encoding passes (5 conditions x 3 seeds) at ~110 s each on CPU,
roughly 30-40 minutes total. Paste the ENTIRE console output back.
"""

from __future__ import annotations
import argparse
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

PERTS = ["sf_all", "sf_half", "ex_half", "masking"]
MASK_RATIO = 0.2
BATCH_SIZE = 32          # FROZEN — changing it shifts the protocol-mask draws
SEQ_LEN = 186
N_CHANNELS = 7
EXPECTED_ROWS = 2006
FROZEN_CORRECT = {13: 884, 14: 859, 15: 863}
R_VALID_POINTS = 2393055
R_NATURAL_ZEROS = 77278
R_DEAD_CHANNELS = 498
AMBIGUOUS_ROW = 1191


def fail(gate, msg):
    print(f"\n[GATE FAILED] {gate}: {msg}")
    print("Hard stop. Investigate before proceeding — do not work around.")
    sys.exit(1)


def row_seed(row: int, pert: str, mask_seed: int) -> int:
    h = hashlib.sha256(f"trace_{row}|{pert}|{mask_seed}".encode()).hexdigest()
    return int(h[:12], 16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-repo", required=True)
    ap.add_argument("--census",
                    default="results/analysis/probe2_trace_order_census.json")
    ap.add_argument("--split", default="test")
    ap.add_argument("--mask-seeds", default="13,14,15")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out-dir", default="results/experiments")
    args = ap.parse_args()
    t0 = time.time()
    mask_seeds = [int(s) for s in args.mask_seeds.split(",")]
    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 74)
    print("TRACE PROBE-2 RUNNER — shuffle family + masking, three mask seeds")
    print("=" * 74)

    # ---- census (G3) ----------------------------------------------------
    cen = json.load(open(args.census, encoding="utf-8"))
    per_row = cen["per_row"]
    if [r["row_idx"] for r in per_row] != list(range(EXPECTED_ROWS)):
        fail("G3-order", "census row_idx is not contiguous 0..2005 — it cannot "
                         "be aligned with the dataloader order")
    bucket = [r["bucket"] for r in per_row]
    bc = Counter(bucket)
    print(f"[G3] census buckets: {dict(bc)}")
    if dict(bc) != {"dependent": 2005, "ambiguous": 1}:
        fail("G3-counts", f"census buckets {dict(bc)} != 2005 dep / 1 amb / 0 inv")
    amb_rows = [i for i, b in enumerate(bucket) if b != "dependent"]
    if amb_rows != [AMBIGUOUS_ROW]:
        fail("G3-amb", f"ambiguous rows {amb_rows} != [{AMBIGUOUS_ROW}]")
    print(f"[G3] PASSED: 2,005 dependent / 1 ambiguous (row {AMBIGUOUS_ROW}) / "
          f"0 invariant. No DiD is posable — the profile is the diagnostic.")

    # ---- repo / checkpoint / data ---------------------------------------
    repo = Path(args.trace_repo).resolve()
    for rel in ("src/models/mm_encoder.py", "src/data/dataloader.py"):
        if not (repo / rel).is_file():
            fail("G1-repo", f"{repo/rel} not found")
    import os
    os.environ["TTRAG_DATA_DIR"] = str(repo / "dataset") + "/"
    os.environ["TTRAG_CHECKPOINTS_DIR"] = str(repo / "results/model_checkpoints") + "/"
    os.environ["TTRAG_RESULTS_DIR"] = str(repo / "results/model_results") + "/"
    sys.path.insert(0, str(repo))

    import numpy as np
    import pyarrow.parquet as pq
    import torch
    import torch.nn.functional as F
    from tqdm import tqdm

    device = torch.device(args.device if args.device
                          else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    ckpt_path = repo / "results/model_checkpoints/context_align/retriever_demo.pt"
    parquet = repo / "dataset" / "retrieval" / args.split / f"{args.split}.parquet"
    df = pq.read_table(parquet).to_pandas()
    n_rows = len(df)
    print(f"[env] torch {torch.__version__} | device {device} | "
          f"batch {BATCH_SIZE} (FROZEN) | seeds {mask_seeds}")
    if n_rows != EXPECTED_ROWS:
        fail("G3-rows", f"{n_rows} rows != {EXPECTED_ROWS}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    margs = ckpt["args"]
    enc_name = getattr(margs, "text_encoder_name", None)
    protocol_ratio = getattr(margs, "mask_ratio", None)
    print(f"[model] encoder {enc_name!r} | protocol mask_ratio {protocol_ratio!r}")
    if enc_name != "nomic-ai/nomic-embed-text-v1.5":
        fail("G-enc", f"encoder {enc_name!r} is not Nomic v1.5")

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
        fail("G4-statedict", f"missing {rep.missing_keys} / "
                             f"unexpected {rep.unexpected_keys}")
    model.to(device).eval()
    if model.training:
        fail("G4-eval", "model not in eval mode")
    print("[G4] state dict strict-clean; model in eval mode")
    del ckpt

    short = enc_name.split("/")[-1]
    cache_path = repo / "dataset" / "retrieval" / args.split / f"description_emb_{short}.pt"
    if not cache_path.is_file():
        fail("G3-cache", f"{cache_path} missing")
    raw_rows = torch.load(cache_path, map_location="cpu").float()

    @torch.no_grad()
    def text_path(raw):
        e = model.text_adapter(raw.to(device))
        e = model.norm(e)
        e = model.dropout(e)
        return F.normalize(e, dim=-1).cpu()

    text_rows = text_path(raw_rows)
    print(f"[text] text embeddings {tuple(text_rows.shape)} (mask-independent, "
          f"computed once; captions are never perturbed)")

    margs.task_name = "retrieval"
    margs.data_split = args.split
    margs.batch_size = BATCH_SIZE
    margs.device = device
    margs.distributed = False
    if args.split != "test":
        fail("G-order", "only the test split preserves row order")
    loader = get_dataloader(margs)

    # ---- G0 row order ----------------------------------------------------
    first = next(iter(loader))
    d0 = float((first.description_emb.cpu().float()
                - raw_rows[:first.description_emb.shape[0]]).abs().max())
    print(f"[G0] row order check: max abs diff = {d0:.3e}")
    if d0 > 1e-5:
        fail("G0-order", "dataloader order != parquet order; the diagonal "
                         "ground truth is invalid")

    # =====================================================================
    # Pre-scan: geometry per row + reconciliation gates
    # =====================================================================
    print("\n--- pre-scan: valid blocks, dead channels, natural zeros " + "-" * 15)
    starts = np.zeros(n_rows, dtype=np.int64)
    vlens = np.zeros(n_rows, dtype=np.int64)
    dead = {}                      # row -> np.array of dead channel indices
    n_dead = zeros_tot = valid_tot = 0
    r = 0
    for b in tqdm(loader, total=len(loader), desc="pre-scan"):
        x = b.timeseries.float().numpy()
        m = b.input_mask.numpy() > 0.5
        for i in range(x.shape[0]):
            mi, xi = m[i], x[i]
            if not (mi == mi[0]).all():
                fail("T1-uniform", f"row {r}: channels do not share one mask")
            idx = np.flatnonzero(mi[0])
            V = len(idx)
            if V == 0 or idx.max() - idx.min() + 1 != V:
                fail("T1-contig", f"row {r}: valid positions not contiguous")
            if idx.max() != SEQ_LEN - 1:
                fail("T1-align", f"row {r}: valid block ends at {idx.max()}, "
                                 f"not {SEQ_LEN-1} — not right-aligned. The "
                                 f"perturbation slice would be wrong.")
            starts[r], vlens[r] = SEQ_LEN - V, V
            blk = xi[:, SEQ_LEN - V:]
            rng_ = blk.max(axis=1) - blk.min(axis=1)
            dd = np.flatnonzero(rng_ == 0.0)
            if len(dd):
                dead[r] = dd
            n_dead += len(dd)
            zeros_tot += int((blk == 0.0).sum())
            valid_tot += int(blk.size)
            r += 1
    if r != n_rows:
        fail("R-rows", f"pre-scan saw {r} rows != {n_rows}")
    print(f"  valid length V: min {vlens.min()} max {vlens.max()} "
          f"mean {vlens.mean():.1f} | distinct {len(set(vlens.tolist()))}")
    print(f"  valid block = [186-V, 186) in all rows (right-aligned, verified)")
    print(f"[G2] natural exact-zeros: {zeros_tot:,}/{valid_tot:,} "
          f"({zeros_tot/valid_tot:.4%}) — report only, never fatal")
    print(f"  dead (constant) channels: {n_dead} in {len(dead)} rows")
    for nm, got, exp in [("valid points", valid_tot, R_VALID_POINTS),
                         ("natural zeros", zeros_tot, R_NATURAL_ZEROS),
                         ("dead channels", n_dead, R_DEAD_CHANNELS)]:
        mark = "OK" if got == exp else "MISMATCH"
        print(f"  [R] {nm:15s} {got:>10,}  expected {exp:>10,}   {mark}")
        if got != exp:
            fail("R-reconcile", f"{nm} disagrees with the diagnostics")
    ks = {int(V): int(V * MASK_RATIO) for V in sorted(set(vlens.tolist()))}
    print(f"  masking k = int(0.2*V): e.g. V=168 -> k={ks.get(168)}, "
          f"V=180 -> k={ks.get(180)}; {len(ks)} distinct (V,k) pairs")
    print(f"  effective input mask ratio: "
          f"{np.mean([ks[int(v)]/v for v in vlens]):.4f} "
          f"(reported as 0.3 protocol + 0.2 input, never bare 0.2)")

    # =====================================================================
    # Perturbation
    # =====================================================================
    diag = defaultdict(lambda: {"noop": 0, "already_zero": 0, "masked": 0,
                                "g8t_checked": 0})

    def perturb_batch(x, base_row, pert, seeds_rec):
        """x: float32 [B,7,186], modified IN PLACE on a clone. All gates fire
        here so a violation stops before any embedding is computed."""
        for i in range(x.shape[0]):
            row = base_row + i
            V, st = int(vlens[row]), int(starts[row])
            rng = np.random.default_rng(row_seed(row, pert, cur_seed))
            seeds_rec[row] = int(row_seed(row, pert, cur_seed))
            before = x[i].copy()
            blk = x[i][:, st:]                       # [7, V] view
            if pert == "sf_all":
                perm = rng.permutation(V)
                if sorted(perm.tolist()) != list(range(V)):
                    fail("G7", f"row {row}: sf_all draw is not a permutation")
                blk[:] = blk[:, perm]
            elif pert == "sf_half":
                mid = V // 2
                perm = rng.permutation(mid)
                if sorted(perm.tolist()) != list(range(mid)):
                    fail("G7", f"row {row}: sf_half draw is not a permutation")
                blk[:, :mid] = blk[:, perm]
            elif pert == "ex_half":
                mid = V // 2
                blk[:] = np.concatenate([blk[:, mid:], blk[:, :mid]], axis=1)
            elif pert == "masking":
                k = int(V * MASK_RATIO)
                idx = rng.permutation(V)[:k]
                if len(set(idx.tolist())) != k:
                    fail("G7", f"row {row}: mask positions not distinct")
                diag[pert]["already_zero"] += int((blk[:, idx] == 0.0).sum())
                diag[pert]["masked"] += k * N_CHANNELS
                blk[:, idx] = 0.0
            else:
                fail("pert", f"unknown perturbation {pert}")

            after = x[i]
            # G1: padding must be byte-identical
            if not np.array_equal(before[:, :st], after[:, :st]):
                fail("G1-pad", f"row {row}: padding changed under {pert}")
            if pert in ("sf_all", "sf_half", "ex_half"):
                # G1: per-channel value multiset preserved
                if not np.array_equal(np.sort(before[:, st:], axis=1),
                                      np.sort(after[:, st:], axis=1)):
                    fail("G1-multiset",
                         f"row {row}: {pert} did not preserve values")
                if pert == "sf_half":
                    mid = V // 2
                    if not np.array_equal(before[:, st + mid:],
                                          after[:, st + mid:]):
                        fail("G1-tail", f"row {row}: sf_half touched the tail")
                # G8-T: dead channels must be elementwise unchanged
                if row in dead:
                    dd = dead[row]
                    if not np.array_equal(before[dd], after[dd]):
                        fail("G8-T", f"row {row}: dead channel(s) {dd.tolist()} "
                                     f"moved under {pert} — this is the "
                                     f"signature of permuting the CHANNEL axis "
                                     f"instead of the time axis")
                    diag[pert]["g8t_checked"] += len(dd)
                if np.array_equal(before[:, st:], after[:, st:]):
                    diag[pert]["noop"] += 1
            else:
                changed = before[:, st:] != after[:, st:]
                if not np.all(after[:, st:][changed] == 0.0):
                    fail("G1-mask", f"row {row}: masking wrote a non-zero value")
                if np.array_equal(before[:, st:], after[:, st:]):
                    diag[pert]["noop"] += 1
        return x

    @torch.no_grad()
    def encode(seed, pert):
        """One full encoding pass. torch.manual_seed is re-set FIRST so every
        condition sees the identical protocol mask (T8)."""
        torch.manual_seed(seed)
        chunks, base = [], 0
        seeds_rec = {}
        for b in tqdm(loader, total=len(loader),
                      desc=f"seed {seed} / {pert or 'unperturbed'}", leave=False):
            x = b.timeseries.float().clone()
            if pert is not None:
                xn = x.numpy()
                perturb_batch(xn, base, pert, seeds_rec)
            o = model(
                x_enc=x.to(device),
                input_mask=b.input_mask.long().to(device),
                channel_description_emb=b.channel_description_emb.to(device),
                description_emb=b.description_emb.to(device),
                event_emb=b.event_emb.to(device),
            )
            chunks.append(o.embeddings.detach().cpu())
            base += x.shape[0]
        if base != n_rows:
            fail("G9-count", f"encoded {base} rows != {n_rows}")
        return F.normalize(torch.cat(chunks).float(), dim=-1), seeds_rec

    def legacy_p1(ts_emb):
        """Narrative-runner definition, verbatim (tie-for-top counts correct,
        float32 mean). G6 comparison ONLY — never a measurement."""
        s = text_rows @ ts_emb.T
        hits = (s.max(dim=1).values <= s.diag())
        return float(hits.float().mean()), int(hits.sum())

    def avg_ranks(ts_emb):
        """D2 deterministic average rank, both directions, from one matrix.
        text->ts ranks the 2,006 series for each caption; ts->text ranks the
        2,006 captions for each series."""
        S = (text_rows @ ts_emb.T).numpy().astype(np.float64)
        gt = np.diag(S)
        t2s_above = (S > gt[:, None]).sum(axis=1)
        t2s_eq = (S == gt[:, None]).sum(axis=1)
        s2t_above = (S > gt[None, :]).sum(axis=0)
        s2t_eq = (S == gt[None, :]).sum(axis=0)
        return ({"text2ts": t2s_above + (t2s_eq + 1) / 2.0,
                 "ts2text": s2t_above + (s2t_eq + 1) / 2.0},
                {"text2ts": t2s_eq - 1, "ts2text": s2t_eq - 1})

    def grp(ranks, sel):
        r_ = ranks[sel]
        return {"n": int(len(r_)), "mrr": float((1.0 / r_).mean()),
                "recall@1": float((r_ <= 1).mean()),
                "recall@10": float((r_ <= 10).mean()),
                "median_rank": float(np.median(r_))}

    is_dep = np.array([b == "dependent" for b in bucket])
    summary = {"script": "run_probe2.py", "perts": PERTS,
               "mask_ratio_input": MASK_RATIO,
               "protocol_mask_ratio": protocol_ratio,
               "seed_scheme": "int(sha256('trace_{row}|{pert}|{mask_seed}')[:12hex],16)",
               "census_buckets": dict(bc), "seeds": {}}

    for cur_seed in mask_seeds:
        print(f"\n===== mask seed {cur_seed} " + "=" * 45)
        diag.clear()
        t_s = time.time()

        emb_u, _ = encode(cur_seed, None)
        p1, n_ok = legacy_p1(emb_u)
        exp_n = FROZEN_CORRECT.get(cur_seed)
        print(f"[G6] legacy P@1 = {p1:.16f} ({n_ok}/{n_rows}); frozen {exp_n}")
        if exp_n is not None and n_ok != exp_n:
            fail("G6", f"{n_ok} correct != frozen {exp_n} — the unperturbed "
                       f"pass no longer reproduces the committed baseline")
        print("[G6] PASSED (digit-exact vs the frozen record)")

        ranks_u, ntied_u = avg_ranks(emb_u)
        print(f"[D2] unperturbed exact-tie queries: text2ts "
              f"{int((ntied_u['text2ts']>0).sum())}, ts2text "
              f"{int((ntied_u['ts2text']>0).sum())}")

        ranks_p, meta = {}, {}
        for pert in PERTS:
            e, seeds_rec = encode(cur_seed, pert)
            d = diag[pert]
            msg = (f"[{pert}] G1/G7/G8-T PASSED | no-ops {d['noop']}"
                   f" | dead-channel checks {d['g8t_checked']}")
            if pert == "masking":
                msg += (f" | masked cells {d['masked']:,}"
                        f" (already 0: {d['already_zero']:,} = "
                        f"{d['already_zero']/max(1,d['masked']):.1%})")
            print(msg)
            ranks_p[pert], _nt = avg_ranks(e)
            meta[pert] = seeds_rec
            del e

        # G4 direction sanity (primary direction, dependent group)
        mu = grp(ranks_u["text2ts"], is_dep)["mrr"]
        mp = grp(ranks_p["sf_all"]["text2ts"], is_dep)["mrr"]
        print(f"[G4] dependent MRR text2ts: unpert {mu:.4f} -> sf_all {mp:.4f} "
              f"(degradation {mu-mp:+.4f}, relative {(mu-mp)/mu:+.1%})")
        if mp > mu:
            fail("G4", "the order-dependent group IMPROVED under sf_all")
        print("[G4] PASSED")

        tables = {}
        for direction in ("text2ts", "ts2text"):
            tag = "PRIMARY" if direction == "text2ts" else "secondary"
            print(f"\n  {direction} ({tag}) — MRR by group  [n]")
            print(f"    {'group':<12}{'unpert':>9}" +
                  "".join(f"{p:>10}" for p in PERTS))
            for gname, sel in [("dependent", is_dep), ("ambiguous", ~is_dep)]:
                row = {"unperturbed": grp(ranks_u[direction], sel)}
                for p in PERTS:
                    row[p] = grp(ranks_p[p][direction], sel)
                tables[f"{direction}/{gname}"] = row
                print(f"    {gname:<12}{row['unperturbed']['mrr']:>9.4f}" +
                      "".join(f"{row[p]['mrr']:>10.4f}" for p in PERTS) +
                      f"   [{row['unperturbed']['n']}]")
            dep = tables[f"{direction}/dependent"]
            base = dep["unperturbed"]["mrr"]
            print(f"    relative degradation vs unperturbed (dependent): " +
                  ", ".join(f"{p} {(base-dep[p]['mrr'])/base:+.1%}"
                            for p in PERTS))

        # per-query records
        rec = outdir / f"probe2_trace_per_query_seed{cur_seed}.jsonl"
        with open(rec, "w", encoding="utf-8") as f:
            for direction in ("text2ts", "ts2text"):
                for i in range(n_rows):
                    f.write(json.dumps({
                        "mask_seed": cur_seed, "direction": direction,
                        "row_idx": i, "id": str(df.iloc[i]["id"]),
                        "group": bucket[i],
                        "valid_len": int(vlens[i]),
                        "rank_unperturbed": float(ranks_u[direction][i]),
                        "ntied_unperturbed": int(ntied_u[direction][i]),
                        **{f"rank_{p}": float(ranks_p[p][direction][i])
                           for p in PERTS}}) + "\n")
        mp_ = outdir / f"probe2_trace_signal_meta_seed{cur_seed}.json"
        json.dump({p: {str(k): v for k, v in meta[p].items()} for p in PERTS},
                  open(mp_, "w", encoding="utf-8"), indent=2)
        print(f"\n  wrote {rec.name} and {mp_.name}  "
              f"[{time.time()-t_s:.0f} s for this seed]")
        summary["seeds"][str(cur_seed)] = {
            "legacy_p1": p1, "legacy_correct": n_ok, "frozen_correct": exp_n,
            "tables": tables,
            "diagnostics": {p: dict(diag[p]) for p in PERTS}}

    sp = outdir / "probe2_trace_summary.json"
    summary["runtime_seconds"] = round(time.time() - t0, 1)
    json.dump(summary, open(sp, "w", encoding="utf-8"), indent=2)
    print(f"\nwrote {sp}  ({summary['runtime_seconds']} s total)")
    print("ALL GATES PASSED — per-query records ready for the stats script.")
    print("Reminder: no prediction is registered on whether shuffling or "
          "masking hurts more. That ranking is the open question.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
run_probe3.py — TRACE Probe-3 runner (summary-statistics sufficiency ladder).

Design record: Probe-3 Q1-Q5 (handoff §4.8) + the TRACE mechanics frame
accepted 2026-08-15/16 (this session): renorm-always construction,
joint-channel resample, per-channel matched gaussian, dead ptp==0
pass-through, row-level joint redraw (max 8). Diagnostic record:
results/analysis/probe3_trace_setup_diagnostic.json — its measured redraw
census {13:41, 14:25, 15:42} and exact collapse rows are HARD-GATED here.

CONDITIONS (the ladder's two new rungs; rung 2 = sf_all is READ from the
committed Probe-2 records at stats time, never rerun):
  resample  one JOINT index draw per row (V draws with replacement over the
            valid block, the same indices for all 7 channels — preserves
            cross-channel co-occurrence, per the joint-shuffle precedent),
            then per-channel renormalisation (ddof=0, stats float64, tensor
            float32). Preserves value-distribution shape, destroys order
            and the exact multiset.
  gaussian  per-channel Normal(mu_c, sd_c) over the valid block, then the
            same renormalisation. After renorm this is provably identical
            to standard-normal draws (diagnostic B11) — the matching is
            vacuous post-renorm, stated in the writeup. The length-and-
            moments-only floor; also destroys cross-channel structure
            (accepted anchor limitation, stated).
  Dead channels (ptp==0, 498 of them: 441 all-zero + 57 float-dust) pass
  through UNCHANGED under both conditions — identity controls (G8-T).
  Renorm-always rationale (measured, diagnostic B8): resample and matched-
  gaussian draws commute with per-channel affine maps once renormalised,
  so this exactly reconstructs raw-level construction regardless of where
  the pipeline's StandardScaler lives (locus: loader-side, ddof=0 —
  diagnostic B4/B5).

REDRAW RULE (pinned): if the joint draw leaves any NON-DEAD channel
constant (near-dead channels — e.g. dry-week precipitation — collapse at
~e^-2 per draw), redraw the ENTIRE row's index set with seed suffix
"|redraw{k}", max 8, HARD STOP if exhausted. Deterministic, so the census
must reproduce the diagnostic EXACTLY (gate R-census). The sparsest-channel
nonzero count is reported per collapse row (mechanism check, predict 1-4).

GATES (fatal policy per the standing rule):
  G3    census 2,005 dep / 1 amb (row 1191) — HARD
  G0    dataloader order == parquet order — HARD
  R     reconciliation: 2,393,055 valid points, 77,278 natural zeros,
        498 dead channels, valid_len == committed records — HARD
  G6    frozen legacy P@1 == 884/859/863 digit-exact — HARD
  R-JG  unperturbed D2 ranks == committed probe2 per-query
        rank_unperturbed, both directions, every row, <=1e-9 — HARD
  R-census  redraw totals AND exact (row, k) pairs == diagnostic — HARD
  G7    seeds recorded; draw indices in range — HARD
  G1    resample drawn bag subset of the channel's own values; post-renorm
        |mean|,|sd-1| <= 1e-5 per non-dead channel; padding byte-identical;
        full-row no-ops FLAGGED never failed — HARD on real violations
  G8-T  all 498 dead channels elementwise unchanged under BOTH conditions —
        HARD
  G4    buffered direction sanity: gaussian dep text2ts MRR must not exceed
        resample by more than 0.02 — HARD
  G9    row count and order identical across conditions — HARD
  G12   draw quality (REPORT): frac positions never drawn (~0.367), mean
        frac unique values missing per non-dead channel
  G2    natural-zero rate on surrogates (REPORT)

D2 average rank is the measurement rank throughout; text->ts PRIMARY,
ts->text secondary, as in Probe 2. Texts are never perturbed. torch is
re-seeded before EVERY encoding pass so all conditions share the identical
0.3 protocol mask (T8, measured input-independent).

USAGE (repo root; PowerShell continuation):
  python models/trace/run_probe3.py `
      --trace-repo ..\\TRACE-Multimodal-TSEncoder `
      --census results/analysis/probe2_trace_order_census.json

Writes:
  results/experiments/probe3_trace_per_query_seed{S}.jsonl
  results/experiments/probe3_trace_signal_meta_seed{S}.json
  results/experiments/probe3_trace_summary.json

Runtime: 9 encoding passes (3 conditions x 3 seeds) at ~110 s each on CPU,
~20-25 minutes total. Paste the ENTIRE console output back.
"""

from __future__ import annotations
import argparse
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

CONDITIONS = ["resample", "gaussian"]
BATCH_SIZE = 32          # FROZEN — changing it shifts the protocol-mask draws
SEQ_LEN = 186
N_CHANNELS = 7
EXPECTED_ROWS = 2006
FROZEN_CORRECT = {13: 884, 14: 859, 15: 863}
R_VALID_POINTS = 2393055
R_NATURAL_ZEROS = 77278
R_DEAD_CHANNELS = 498
AMBIGUOUS_ROW = 1191
EXPECTED_REDRAWS = {13: 41, 14: 25, 15: 42}   # diagnostic B10, hard-gated
MAX_REDRAWS = 8
G4_BUFFER = 0.02
MOMENT_TOL = 1e-5
JG_TOL = 1e-9


def fail(gate, msg):
    print(f"\n[GATE FAILED] {gate}: {msg}")
    print("Hard stop. Investigate before proceeding — do not work around.")
    sys.exit(1)


def row_seed(row: int, cond: str, mask_seed: int, redraw: int = 0) -> int:
    tag = f"trace_{row}|{cond}|{mask_seed}"
    if redraw:
        tag += f"|redraw{redraw}"
    h = hashlib.sha256(tag.encode()).hexdigest()
    return int(h[:12], 16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-repo", required=True)
    ap.add_argument("--census",
                    default="results/analysis/probe2_trace_order_census.json")
    ap.add_argument("--diagnostic",
                    default="results/analysis/probe3_trace_setup_diagnostic.json")
    ap.add_argument("--probe2-dir", default="results/experiments")
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
    print("TRACE PROBE-3 RUNNER — resample + gaussian, renorm-always, "
          "three mask seeds")
    print("=" * 74)

    # ---- diagnostic record (for R-census) --------------------------------
    diag_rec = json.load(open(args.diagnostic, encoding="utf-8"))
    exp_collapse = {s: set() for s in mask_seeds}
    for row, seed, k in diag_rec["part_b"]["calibration"]["collapse_rows"]:
        if seed in exp_collapse:
            exp_collapse[seed].add((int(row), int(k)))
    print(f"[R-census] diagnostic redraw census loaded: "
          + ", ".join(f"seed {s}: {len(exp_collapse[s])} rows "
                      f"({sum(k for _, k in exp_collapse[s])} redraws)"
                      for s in mask_seeds))

    # ---- census (G3) ------------------------------------------------------
    cen = json.load(open(args.census, encoding="utf-8"))
    per_row = cen["per_row"]
    if [r["row_idx"] for r in per_row] != list(range(EXPECTED_ROWS)):
        fail("G3-order", "census row_idx is not contiguous 0..2005")
    bucket = [r["bucket"] for r in per_row]
    bc = Counter(bucket)
    if dict(bc) != {"dependent": 2005, "ambiguous": 1}:
        fail("G3-counts", f"census buckets {dict(bc)} != 2005 dep / 1 amb")
    if [i for i, b in enumerate(bucket) if b != "dependent"] != [AMBIGUOUS_ROW]:
        fail("G3-amb", f"ambiguous row != {AMBIGUOUS_ROW}")
    print(f"[G3] PASSED: 2,005 dependent / 1 ambiguous (row {AMBIGUOUS_ROW})")

    # ---- committed Probe-2 unperturbed ranks (for R-JG) -------------------
    p2_unpert = {}   # seed -> direction -> np.array of ranks by row
    import numpy as np
    for s in mask_seeds:
        p = Path(args.probe2_dir) / f"probe2_trace_per_query_seed{s}.jsonl"
        rows = [json.loads(l) for l in open(p, encoding="utf-8")]
        if len(rows) != 2 * EXPECTED_ROWS:
            fail("R-JG-load", f"{p.name}: {len(rows)} lines != 4,012")
        p2_unpert[s] = {}
        for d in ("text2ts", "ts2text"):
            sel = [r for r in rows if r["direction"] == d]
            if [r["row_idx"] for r in sel] != list(range(EXPECTED_ROWS)):
                fail("R-JG-load", f"{p.name}/{d}: row order broken")
            p2_unpert[s][d] = np.array([r["rank_unperturbed"] for r in sel],
                                       dtype=np.float64)
    print(f"[R-JG] committed Probe-2 unperturbed ranks loaded for seeds "
          f"{mask_seeds}")

    # ---- repo / checkpoint / data (verbatim from run_probe2.py) -----------
    repo = Path(args.trace_repo).resolve()
    for rel in ("src/models/mm_encoder.py", "src/data/dataloader.py"):
        if not (repo / rel).is_file():
            fail("G1-repo", f"{repo/rel} not found")
    import os
    os.environ["TTRAG_DATA_DIR"] = str(repo / "dataset") + "/"
    os.environ["TTRAG_CHECKPOINTS_DIR"] = str(repo / "results/model_checkpoints") + "/"
    os.environ["TTRAG_RESULTS_DIR"] = str(repo / "results/model_results") + "/"
    sys.path.insert(0, str(repo))

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
    print(f"[text] text embeddings {tuple(text_rows.shape)} (computed once; "
          f"captions are never perturbed)")

    margs.task_name = "retrieval"
    margs.data_split = args.split
    margs.batch_size = BATCH_SIZE
    margs.device = device
    margs.distributed = False
    if args.split != "test":
        fail("G-order", "only the test split preserves row order")
    loader = get_dataloader(margs)

    # ---- G0 row order ------------------------------------------------------
    first = next(iter(loader))
    d0 = float((first.description_emb.cpu().float()
                - raw_rows[:first.description_emb.shape[0]]).abs().max())
    print(f"[G0] row order check: max abs diff = {d0:.3e}")
    if d0 > 1e-5:
        fail("G0-order", "dataloader order != parquet order")

    # =====================================================================
    # Pre-scan (verbatim mechanics from run_probe2.py) + reconciliation
    # =====================================================================
    print("\n--- pre-scan: valid blocks, dead channels, natural zeros " + "-" * 15)
    vlens = np.zeros(n_rows, dtype=np.int64)
    dead = {}
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
                fail("T1-align", f"row {r}: valid block not right-aligned")
            vlens[r] = V
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
    for nm, got, exp in [("valid points", valid_tot, R_VALID_POINTS),
                         ("natural zeros", zeros_tot, R_NATURAL_ZEROS),
                         ("dead channels", n_dead, R_DEAD_CHANNELS)]:
        mark = "OK" if got == exp else "MISMATCH"
        print(f"  [R] {nm:15s} {got:>10,}  expected {exp:>10,}   {mark}")
        if got != exp:
            fail("R-reconcile", f"{nm} disagrees with the diagnostics")
    vlen_committed = np.array(
        [json.loads(l)["valid_len"] for l in
         open(Path(args.probe2_dir) / f"probe2_trace_per_query_seed"
              f"{mask_seeds[0]}.jsonl", encoding="utf-8")
         if json.loads(l)["direction"] == "text2ts"], dtype=np.int64)
    if not np.array_equal(vlens, vlen_committed):
        fail("R-vlen", "loader valid lengths != committed records")
    print("  [R] valid_len identical to the committed Probe-2 records   OK")

    # =====================================================================
    # Surrogate construction (mechanics identical to the diagnostic)
    # =====================================================================
    def renorm_block(blk):
        """Per-channel renorm of [C, V] float32; stats float64 ddof=0;
        ptp==0 channels pass through bit-identically."""
        out = blk.astype(np.float64, copy=True)
        for c in range(out.shape[0]):
            if np.ptp(out[c]) == 0.0:
                out[c] = blk[c]
                continue
            mu = out[c].mean()
            sdv = out[c].std()
            out[c] = (out[c] - mu) / sdv
        return out.astype(np.float32)

    def build_resample(blk, dmask, row, seed):
        C, V = blk.shape
        for k in range(MAX_REDRAWS + 1):
            rng = np.random.default_rng(row_seed(row, "resample", seed, k))
            idx = rng.integers(0, V, size=V)
            if idx.min() < 0 or idx.max() >= V:
                fail("G7", f"row {row}: resample indices out of range")
            drawn = blk[:, idx]
            collapsed = any((not dmask[c]) and np.ptp(drawn[c]) == 0.0
                            for c in range(C))
            if not collapsed:
                sur = drawn.copy()
                for c in range(C):
                    if dmask[c]:
                        sur[c] = blk[c]
                return renorm_block(sur), k, idx
        fail("B10-redraw", f"row {row} seed {seed}: collapsed after "
                           f"{MAX_REDRAWS} redraws")

    def build_gaussian(blk, dmask, row, seed):
        C, V = blk.shape
        rng = np.random.default_rng(row_seed(row, "gaussian", seed))
        sur = np.empty_like(blk)
        for c in range(C):
            if dmask[c]:
                sur[c] = blk[c]
            else:
                mu = float(blk[c].astype(np.float64).mean())
                sdv = float(blk[c].astype(np.float64).std())
                sur[c] = rng.normal(mu, sdv, size=V).astype(np.float32)
        return renorm_block(sur)

    diag = defaultdict(lambda: {"noop": 0, "dead_passthrough": 0,
                                "redraws": 0, "collapse": [],
                                "frac_never_drawn": [], "frac_uniq_missing": [],
                                "surrogate_zeros": 0, "surrogate_points": 0})

    def perturb_batch(x, base_row, cond, seeds_rec, meta_rec):
        for i in range(x.shape[0]):
            row = base_row + i
            V, st = int(vlens[row]), int(starts_of(row))
            dmask = np.zeros(N_CHANNELS, bool)
            if row in dead:
                dmask[dead[row]] = True
            before = x[i].copy()
            blk = x[i][:, st:]
            d = diag[cond]
            if cond == "resample":
                sur, k, idxd = build_resample(blk, dmask, row, cur_seed)
                seeds_rec[row] = int(row_seed(row, "resample", cur_seed, k))
                if k:
                    d["redraws"] += k
                    min_nz = min(int((blk[c] != 0).sum())
                                 for c in range(N_CHANNELS) if not dmask[c])
                    min_uq = min(len(np.unique(blk[c]))
                                 for c in range(N_CHANNELS) if not dmask[c])
                    d["collapse"].append((row, k, min_nz, min_uq))
                d["frac_never_drawn"].append(
                    1.0 - len(np.unique(idxd)) / V)
                fq = [1.0 - len(np.unique(blk[c][idxd]))
                      / len(np.unique(blk[c]))
                      for c in range(N_CHANNELS) if not dmask[c]]
                if fq:
                    d["frac_uniq_missing"].append(float(np.mean(fq)))
                # G1: pre-renorm drawn bag subset of the channel's values
                for c in range(N_CHANNELS):
                    if not dmask[c] and not np.isin(blk[c][idxd], blk[c]).all():
                        fail("G1-subset", f"row {row} ch {c}: drawn value "
                                          f"not in the channel")
                meta_rec[row] = {"seed": seeds_rec[row], "redraws": k,
                                 "frac_never_drawn":
                                     d["frac_never_drawn"][-1]}
            else:
                sur = build_gaussian(blk, dmask, row, cur_seed)
                seeds_rec[row] = int(row_seed(row, "gaussian", cur_seed))
                meta_rec[row] = {"seed": seeds_rec[row]}
            # write back + gates
            x[i][:, st:] = sur
            after = x[i]
            if not np.array_equal(before[:, :st], after[:, :st]):
                fail("G1-pad", f"row {row}: padding changed under {cond}")
            for c in range(N_CHANNELS):
                ch = after[c, st:]
                if dmask[c]:
                    if not np.array_equal(before[c, st:], ch):
                        fail("G8-T", f"row {row} ch {c}: dead channel moved "
                                     f"under {cond}")
                    d["dead_passthrough"] += 1
                else:
                    ch64 = ch.astype(np.float64)
                    if abs(ch64.mean()) > MOMENT_TOL or \
                            abs(ch64.std() - 1.0) > MOMENT_TOL:
                        fail("G1-moments", f"row {row} ch {c}: post-renorm "
                                           f"moments off under {cond}")
            if not np.all(np.isfinite(after[:, st:])):
                fail("G1-finite", f"row {row}: non-finite value under {cond}")
            if np.array_equal(before[:, st:], after[:, st:]):
                d["noop"] += 1
            d["surrogate_zeros"] += int((after[:, st:] == 0.0).sum())
            d["surrogate_points"] += int(after[:, st:].size)

    starts = SEQ_LEN - vlens

    def starts_of(row):
        return starts[row]

    @torch.no_grad()
    def encode(seed, cond):
        """One full pass. torch re-seeded FIRST: every condition sees the
        identical protocol mask (T8)."""
        torch.manual_seed(seed)
        chunks, base = [], 0
        seeds_rec, meta_rec = {}, {}
        for b in tqdm(loader, total=len(loader),
                      desc=f"seed {seed} / {cond or 'unperturbed'}",
                      leave=False):
            x = b.timeseries.float().clone()
            if cond is not None:
                xn = x.numpy()
                perturb_batch(xn, base, cond, seeds_rec, meta_rec)
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
        return F.normalize(torch.cat(chunks).float(), dim=-1), meta_rec

    def legacy_p1(ts_emb):
        s = text_rows @ ts_emb.T
        hits = (s.max(dim=1).values <= s.diag())
        return float(hits.float().mean()), int(hits.sum())

    def avg_ranks(ts_emb):
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

    is_dep = np.array([b_ == "dependent" for b_ in bucket])
    summary = {"script": "run_probe3.py (TRACE)",
               "conditions": CONDITIONS,
               "construction": "renorm-always ddof=0; joint-channel "
                               "resample; per-channel matched gaussian; "
                               "dead ptp==0 pass-through; redraw max 8",
               "protocol_mask_ratio": protocol_ratio,
               "seed_scheme": "int(sha256('trace_{row}|{cond}|{mask_seed}"
                              "[|redraw{k}]')[:12hex],16)",
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
            fail("G6", f"{n_ok} correct != frozen {exp_n}")
        print("[G6] PASSED (digit-exact vs the frozen record)")

        ranks_u, ntied_u = avg_ranks(emb_u)
        for d_ in ("text2ts", "ts2text"):
            jg = float(np.abs(ranks_u[d_] - p2_unpert[cur_seed][d_]).max())
            print(f"[R-JG] {d_}: max |rank - committed Probe-2 "
                  f"rank_unperturbed| = {jg:.3e}")
            if jg > JG_TOL:
                fail("R-JG", f"{d_}: unperturbed ranks do not reproduce the "
                             f"committed Probe-2 records")
        print("[R-JG] PASSED — this run is paired with the committed "
              "Probe-2 arm")
        print(f"[D2] unperturbed exact-tie queries: text2ts "
              f"{int((ntied_u['text2ts']>0).sum())}, ts2text "
              f"{int((ntied_u['ts2text']>0).sum())}")

        ranks_c, meta = {}, {}
        for cond in CONDITIONS:
            e, meta_rec = encode(cur_seed, cond)
            d = diag[cond]
            msg = (f"[{cond}] G1/G7/G8-T PASSED | full-row no-ops "
                   f"{d['noop']} | dead pass-throughs "
                   f"{d['dead_passthrough']} (expect {R_DEAD_CHANNELS})")
            if cond == "resample":
                msg += (f" | redraws {d['redraws']} "
                        f"(diagnostic {EXPECTED_REDRAWS.get(cur_seed)})")
            print(msg)
            if d["dead_passthrough"] != R_DEAD_CHANNELS:
                fail("G8-T-count", f"{cond}: dead pass-throughs "
                                   f"{d['dead_passthrough']} != "
                                   f"{R_DEAD_CHANNELS}")
            if cond == "resample":
                got = {(row, k) for row, k, _, _ in d["collapse"]}
                exp = exp_collapse.get(cur_seed, set())
                if d["redraws"] != EXPECTED_REDRAWS.get(cur_seed) or got != exp:
                    fail("R-census", f"seed {cur_seed}: redraw census does "
                                     f"not reproduce the diagnostic "
                                     f"(got {sorted(got)})")
                print(f"[R-census] PASSED — {len(got)} collapse rows, "
                      f"{d['redraws']} redraws, exact match")
                if d["collapse"]:
                    nz = [c[2] for c in d["collapse"]]
                    uq = [c[3] for c in d["collapse"]]
                    mark = ("HIT" if max(nz) <= 4 else "MISS (recorded)")
                    print(f"[R11] collapse-row sparsest channel: nonzero "
                          f"counts min {min(nz)} max {max(nz)} "
                          f"(predicted 1-4)  {mark}; unique-value counts "
                          f"min {min(uq)} max {max(uq)}")
                fnd = float(np.mean(d["frac_never_drawn"]))
                fum = float(np.mean(d["frac_uniq_missing"]))
                mark = "HIT" if 0.36 <= fnd <= 0.375 else "MISS (recorded)"
                print(f"[G12] frac positions never drawn: mean {fnd:.4f} "
                      f"(predicted ~0.367, band [0.36, 0.375])  {mark} | "
                      f"mean frac unique values missing {fum:.4f}")
            print(f"[G2] {cond} surrogate exact-zeros: "
                  f"{d['surrogate_zeros']:,}/{d['surrogate_points']:,} "
                  f"({d['surrogate_zeros']/max(1,d['surrogate_points']):.4%})"
                  f" — report only")
            ranks_c[cond], _nt = avg_ranks(e)
            nt_t = int((_nt["text2ts"] > 0).sum())
            nt_s = int((_nt["ts2text"] > 0).sum())
            print(f"[D2] {cond} exact-tie queries: text2ts {nt_t}, "
                  f"ts2text {nt_s} (predicted 0)")
            meta[cond] = meta_rec
            del e

        # G4 buffered direction sanity
        m_res = grp(ranks_c["resample"]["text2ts"], is_dep)["mrr"]
        m_gau = grp(ranks_c["gaussian"]["text2ts"], is_dep)["mrr"]
        m_u = grp(ranks_u["text2ts"], is_dep)["mrr"]
        print(f"[G4] dep text2ts MRR: unpert {m_u:.4f} | resample "
              f"{m_res:.4f} | gaussian {m_gau:.4f} | global chance 0.0041 "
              f"| ceilings V168 0.0072 / V180 0.0126")
        if m_gau > m_res + G4_BUFFER:
            fail("G4", f"gaussian MRR exceeds resample by more than "
                       f"{G4_BUFFER}")
        print("[G4] PASSED (buffered)")

        tables = {}
        for direction in ("text2ts", "ts2text"):
            tag = "PRIMARY" if direction == "text2ts" else "secondary"
            print(f"\n  {direction} ({tag}) — MRR by group  [n]")
            print(f"    {'group':<12}{'unpert':>9}" +
                  "".join(f"{c:>10}" for c in CONDITIONS))
            for gname, sel in [("dependent", is_dep), ("ambiguous", ~is_dep)]:
                row = {"unperturbed": grp(ranks_u[direction], sel)}
                for c in CONDITIONS:
                    row[c] = grp(ranks_c[c][direction], sel)
                tables[f"{direction}/{gname}"] = row
                print(f"    {gname:<12}{row['unperturbed']['mrr']:>9.4f}" +
                      "".join(f"{row[c]['mrr']:>10.4f}" for c in CONDITIONS) +
                      f"   [{row['unperturbed']['n']}]")
            dep_t = tables[f"{direction}/dependent"]
            base = dep_t["unperturbed"]["mrr"]
            print(f"    relative degradation vs unperturbed (dependent): " +
                  ", ".join(f"{c} {(base-dep_t[c]['mrr'])/base:+.1%}"
                            for c in CONDITIONS))
            # per-stratum readout (strata are the quotable unit)
            print(f"    by stratum (dependent):")
            for vlab, vsel in [("V=168", vlens == 168),
                               ("V=180", vlens == 180),
                               ("other",
                                (vlens != 168) & (vlens != 180))]:
                s_ = is_dep & vsel
                line = (f"      {vlab:<7}"
                        f"{grp(ranks_u[direction], s_)['mrr']:>9.4f}")
                for c in CONDITIONS:
                    line += f"{grp(ranks_c[c][direction], s_)['mrr']:>10.4f}"
                print(line + f"   [{int(s_.sum())}]")

        rec = outdir / f"probe3_trace_per_query_seed{cur_seed}.jsonl"
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
                        **{f"rank_{c}": float(ranks_c[c][direction][i])
                           for c in CONDITIONS}}) + "\n")
        mp_ = outdir / f"probe3_trace_signal_meta_seed{cur_seed}.json"
        json.dump({c: {str(k): v for k, v in meta[c].items()}
                   for c in CONDITIONS},
                  open(mp_, "w", encoding="utf-8"), indent=2)
        print(f"\n  wrote {rec.name} and {mp_.name}  "
              f"[{time.time()-t_s:.0f} s for this seed]")
        summary["seeds"][str(cur_seed)] = {
            "legacy_p1": p1, "legacy_correct": n_ok, "frozen_correct": exp_n,
            "tables": tables,
            "diagnostics": {c: {k: v for k, v in diag[c].items()
                                if k != "collapse"} for c in CONDITIONS},
            "collapse_rows": diag["resample"]["collapse"]}

    sp = outdir / "probe3_trace_summary.json"
    summary["runtime_seconds"] = round(time.time() - t0, 1)
    json.dump(summary, open(sp, "w", encoding="utf-8"), indent=2)
    print(f"\nwrote {sp}  ({summary['runtime_seconds']} s total)")
    print("ALL GATES PASSED — per-query records ready for the stats script.")
    print("Scientific scoring (P3-2a / P3-3 / P3-6) happens in the stats "
          "script against the pre-registered references — not by eye here.")


if __name__ == "__main__":
    main()

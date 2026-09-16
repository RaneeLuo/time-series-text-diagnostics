#!/usr/bin/env python3
"""
TRACE task 2 of 2 — run the authors' own pipeline against the released
checkpoint and measure whether it reproduces retrieval at all.

This is demo.ipynb converted to a script (project convention: no
notebooks for runs), with one necessary repair and one extension:

  REPAIR 1 (verified 2026-08-07): the published demo cannot run as-is.
  Cell 1 reads `batch_x.sample_id`, but the TimeseriesData dataclass in
  src/data/base.py has NO sample_id field and RetrievalDataset never
  sets one — the notebook would raise AttributeError on the published
  code. We instead rely on dataloader order: for non-train splits the
  loader is shuffle=False (src/data/dataloader.py line ~142), and NaN
  handling interpolates in place rather than dropping rows, so batch
  order == parquet row order. A gate checks the counts reconcile.

  REPAIR 2 (v2, added after the first run failed 2026-08-07): the
  MultiModalEncoder constructor cannot build from the released
  checkpoint on the published code, for three stacked reasons, all
  read from source:
    (a) `_load_model` tries to open the Stage-1 pretraining checkpoint
        `model_checkpoints/swift-glitter-75/CATSEncoder.pth`, which was
        never released -> FileNotFoundError (the crash observed).
    (b) Even if it existed, the next line raises NotImplementedError:
        the stored args say model_name='CATSEncoder' and the public
        `_load_model` only implements 'TraceEncoder'.
    (c) TS_Encoder builds a structurally different model per
        model_name: `channel_special_tokens = (model_name ==
        "TraceEncoder")`. Under 'CATSEncoder' the channel-identity
        tokens — the architecture's signature — would not be built.
  Working hypothesis: 'CATSEncoder' is the pre-publication name of the
  same architecture now called 'TraceEncoder' (the run predates the
  repo's rename). Repair: (i) set model_name='TraceEncoder' before
  construction; (ii) replace `_load_model` with a stub that builds
  TS_Encoder WITHOUT loading Stage-1 weights — legitimate because the
  Stage-2 checkpoint contains the FULL model (11.55M params = whole
  model incl. ts_encoder), which then overwrites every parameter.
  THE RENAME HYPOTHESIS IS TESTED, NOT ASSUMED: gate G4 requires the
  state dict to load with zero missing and zero unexpected keys. If
  CATSEncoder were genuinely a different architecture, shapes/keys
  would not match and G4 hard-stops.

  REPAIR 3 (v3, after the second run 2026-08-07): the data loader reads
  retrieval/<split>/<split>.parquet (load_data.py line 99), but the
  README documents — and the dataset zip ships — flat files at
  retrieval/<split>.parquet. The code's nested layout wins (embedding
  caches are also written there). Gate G3 now requires the nested
  layout and prints move instructions when it finds the flat one.

  EXTENSION: the demo only prints top-5 neighbours for one query. To
  answer "does the checkpoint reproduce anything," we compute full
  retrieval metrics (P@1, P@5, P@10, MRR, median rank) over the whole
  split, in BOTH directions (text->ts and ts->text), ground-truth pair
  included in the pool. The paper's P@1 = 44.10% is printed alongside
  as an orientation line, NOT as a pass/fail gate: this script does not
  know which split, direction, or pool size produced the published
  number, and says so.

Prerequisites (run in this order):
  1. read_checkpoint_args.py has been run and its output reviewed.
  2. TRACE repo cloned: github.com/Graph-and-Geometric-Learning/TRACE-Multimodal-TSEncoder
  3. Dataset zip from the Google Drive link in the TRACE README, unzipped
     so that <trace-repo>/dataset/retrieval/test.parquet exists.
  4. Python deps: torch, pyarrow, pandas, sentence-transformers,
     transformers, einops, python-dotenv, tqdm.
     (conda env from the authors' environment.yml also works.)

First run will download the sentence encoder named in the checkpoint's
args and embed all descriptions (cached to .pt beside the parquet, so
subsequent runs are fast).

Usage (from anywhere):
  python run_authors_demo_eval.py --trace-repo /path/to/TRACE-Multimodal-TSEncoder \
      --split test --out trace_demo_repro_test.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path


def fail(gate: str, msg: str):
    print(f"\n[GATE FAILED] {gate}: {msg}")
    print("Hard stop. Investigate before proceeding — do not work around.")
    sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-repo", required=True)
    ap.add_argument("--split", default="test", choices=["test", "train"],
                    help="test (default): shuffle=False, descriptions available. "
                         "train: only for a deliberate second look; shuffle is "
                         "forced off here to keep order == parquet order.")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default=None, help="cpu | cuda (default: auto)")
    ap.add_argument("--out", default=None, help="JSON output path "
                    "(default: trace_demo_repro_<split>.json in cwd)")
    args = ap.parse_args()

    t0 = time.time()

    # ---- Gate 1: repo layout -------------------------------------------
    repo = Path(args.trace_repo).resolve()
    for rel in ("src/models/mm_encoder.py", "src/data/dataloader.py",
                "context_align.py"):
        if not (repo / rel).is_file():
            fail("G1-repo", f"{repo/rel} not found — is --trace-repo the "
                            "clone root of TRACE-Multimodal-TSEncoder?")
    print(f"[G1] TRACE repo: {repo}")

    # Environment variables BEFORE importing src.* (src/common.py reads
    # them at import; the repo ships an 'env' file that load_dotenv does
    # not read — it looks for '.env' — so we set values explicitly and
    # do not depend on that file at all).
    os.environ["TTRAG_DATA_DIR"] = str(repo / "dataset") + "/"
    os.environ["TTRAG_CHECKPOINTS_DIR"] = str(repo / "results/model_checkpoints") + "/"
    os.environ["TTRAG_RESULTS_DIR"] = str(repo / "results/model_results") + "/"
    sys.path.insert(0, str(repo))

    # ---- Gate 2: checkpoint --------------------------------------------
    ckpt_path = repo / "results/model_checkpoints/context_align/retriever_demo.pt"
    if not ckpt_path.is_file():
        fail("G2-ckpt", f"{ckpt_path} missing")
    print(f"[G2] checkpoint: {ckpt_path} ({ckpt_path.stat().st_size:,} bytes)")

    # ---- Gate 3: dataset -----------------------------------------------
    # DEFECT 5 (read from load_data.py line 99, 2026-08-07): the code
    # reads retrieval/<split>/<split>.parquet, but the README documents
    # (and the zip ships) a flat retrieval/<split>.parquet. The code's
    # layout wins, since the embedding caches are also written into the
    # per-split subfolder.
    retr_dir = repo / "dataset" / "retrieval"
    parquet = retr_dir / args.split / f"{args.split}.parquet"
    if not parquet.is_file():
        flat = retr_dir / f"{args.split}.parquet"
        hint = ""
        if flat.is_file():
            hint = (f"\nThe file exists at the flat location {flat}.\n"
                    f"Create the subfolder and move it so the path reads:\n"
                    f"  {parquet}\n"
                    f"(The zip follows the README's layout; the code expects "
                    f"the per-split subfolder. The code wins.)")
        fail("G3-data", f"{parquet} missing.{hint}")
    # Evidence line: if the zip ships pre-computed embedding caches, their
    # filenames name the encoder they were made with — cross-check this
    # against the checkpoint args output from script 1.
    print(f"[G3] contents of {retr_dir} (cached *_emb_<encoder>.pt files are "
          "evidence of the intended text encoder):")
    for p in sorted(retr_dir.rglob("*")):
        if p.is_file():
            print(f"      {p.relative_to(retr_dir)}  ({p.stat().st_size:,} B)")

    import numpy as np
    import pyarrow.parquet as pq
    import torch
    import torch.nn.functional as F
    from tqdm import tqdm

    device = torch.device(args.device if args.device
                          else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    print(f"[env] torch {torch.__version__}, device {device}")

    df = pq.read_table(parquet).to_pandas()
    n_rows = len(df)
    print(f"[G3] {parquet.name}: {n_rows} rows, columns {list(df.columns)}")
    if n_rows > 20000:
        print(f"[G3][warn] {n_rows} rows — embedding may take a while on CPU.")

    # ---- Load model exactly as the demo does ---------------------------
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    margs = ckpt["args"]
    print(f"[model] stored text_encoder_name = {getattr(margs, 'text_encoder_name', '<absent>')!r}")
    print(f"[model] stored hard_negative_mining = {getattr(margs, 'hard_negative_mining', '<absent>')!r}")

    from src.models.mm_encoder import MultiModalEncoder
    from src.data.dataloader import get_dataloader
    from src.models.timeseries_encoders.ts_encoder import TS_Encoder
    from copy import deepcopy

    # ---- REPAIR 2 (see header): bypass the Stage-1 load ----------------
    stored_model_name = getattr(margs, "model_name", "<absent>")
    print(f"[repair2] stored model_name = {stored_model_name!r}; overriding "
          f"to 'TraceEncoder' (rename hypothesis — tested by G4 below)")
    margs.model_name = "TraceEncoder"

    def _patched_load_model(self, pretraining_task_name):
        # Original tries to load the unreleased Stage-1 checkpoint
        # ('swift-glitter-75'). The Stage-2 state dict holds the full
        # model, so we only need the module *constructed*; every weight
        # is overwritten by load_state_dict right after.
        pretraining_args = deepcopy(self.args)
        pretraining_args.task_name = pretraining_task_name
        self.ts_encoder = TS_Encoder(configs=pretraining_args)
        print("[repair2] ts_encoder constructed at random init; Stage-1 "
              "load skipped (checkpoint never released); all weights come "
              "from the Stage-2 state dict next.")

    MultiModalEncoder._load_model = _patched_load_model

    # mm_encoder derives its device from args.rank; force a value that
    # resolves to our chosen device.
    margs.rank = 0
    model = MultiModalEncoder(margs)
    state_dict = {k.replace("module.", ""): v
                  for k, v in ckpt["model_state_dict"].items()}
    load_report = model.load_state_dict(state_dict, strict=False)

    # ---- Gate 4: the state dict must fit the constructed model ---------
    if load_report.missing_keys or load_report.unexpected_keys:
        print(f"[G4] missing keys: {load_report.missing_keys}")
        print(f"[G4] unexpected keys: {load_report.unexpected_keys}")
        fail("G4-statedict", "state dict does not match the constructed model. "
                             "This FALSIFIES the CATSEncoder==TraceEncoder "
                             "rename hypothesis (or another config mismatch) — "
                             "whatever ran after this would not be the "
                             "released model. Paste the key lists back.")
    print("[G4] state dict loaded strict-clean (no missing/unexpected keys) "
          "— rename hypothesis CONFIRMED: the checkpoint's weights fit the "
          "TraceEncoder-built architecture exactly.")
    model.to(device).eval()

    # Demo's arg surgery, verbatim:
    margs.task_name = "retrieval"
    margs.data_split = args.split
    margs.batch_size = args.batch_size
    margs.device = device
    margs.distributed = False
    # Repair for --split train only: the non-distributed loader shuffles
    # the train split, which would break row correspondence. Not needed
    # for test, asserted for train:
    if args.split == "train":
        fail("G-order", "train split shuffles in the non-distributed loader, "
             "breaking index<->parquet correspondence. Use --split test; a "
             "train-split evaluation needs a dedicated order-preserving path "
             "— ask for it if wanted.")

    data_loader = get_dataloader(margs)

    # ---- Embed everything (demo cell 1, minus raw-ts hoarding) ---------
    ts_chunks, text_chunks, n_seen = [], [], 0
    with torch.no_grad():
        for batch_x in tqdm(data_loader, total=len(data_loader)):
            timeseries = batch_x.timeseries.float().to(device)
            input_mask = batch_x.input_mask.long().to(device)
            outputs = model(
                x_enc=timeseries,
                input_mask=input_mask,
                channel_description_emb=batch_x.channel_description_emb.to(device),
                description_emb=batch_x.description_emb.to(device),
                event_emb=batch_x.event_emb.to(device),
            )
            ts_chunks.append(outputs.embeddings.detach().cpu())
            text_chunks.append(outputs.description_emb.detach().cpu())
            n_seen += timeseries.shape[0]

    ts_emb = F.normalize(torch.cat(ts_chunks, dim=0).float(), dim=-1)
    text_emb = F.normalize(torch.cat(text_chunks, dim=0).float(), dim=-1)

    # ---- Gate 5: counts reconcile --------------------------------------
    print(f"[G5] parquet rows {n_rows} | batches yielded {n_seen} | "
          f"ts_emb {tuple(ts_emb.shape)} | text_emb {tuple(text_emb.shape)}")
    if not (n_rows == n_seen == ts_emb.shape[0] == text_emb.shape[0]):
        fail("G5-counts", "row counts do not reconcile — index<->row pairing "
                          "cannot be trusted, so no metric below would mean "
                          "anything. (NaN handling should interpolate, not "
                          "drop; if a row vanished, find where.)")

    # ---- Gate 6: normalisation and dims --------------------------------
    for name, e in (("ts", ts_emb), ("text", text_emb)):
        norms = e.norm(dim=-1)
        print(f"[G6] {name} norm min/max: {norms.min():.6f}/{norms.max():.6f}")
        if (norms - 1).abs().max() > 1e-3:
            fail("G6-norm", f"{name} embeddings not unit-norm after F.normalize")
    if ts_emb.shape[1] != text_emb.shape[1]:
        fail("G6-dim", f"dim mismatch ts {ts_emb.shape[1]} vs text "
                       f"{text_emb.shape[1]} — cosine comparison impossible")
    print(f"[G6] shared embedding dim: {ts_emb.shape[1]}")

    # ---- Full retrieval metrics, both directions -----------------------
    def rank_of_truth(queries, candidates, chunk=256):
        """rank (1-based) of candidate i for query i, full pool."""
        n = queries.shape[0]
        ranks = torch.empty(n, dtype=torch.long)
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            scores = queries[s:e] @ candidates.T            # [c, N]
            true_scores = scores[torch.arange(e - s), torch.arange(s, e)]
            # rank = 1 + number of candidates scoring strictly higher
            ranks[s:e] = 1 + (scores > true_scores.unsqueeze(1)).sum(dim=1)
        return ranks

    results = {}
    for direction, q, c in (("text->ts", text_emb, ts_emb),
                            ("ts->text", ts_emb, text_emb)):
        r = rank_of_truth(q, c).float()
        m = {
            "n_queries": int(r.numel()),
            "pool_size": int(c.shape[0]),
            "P@1": float((r <= 1).float().mean()),
            "P@5": float((r <= 5).float().mean()),
            "P@10": float((r <= 10).float().mean()),
            "MRR": float((1.0 / r).mean()),
            "median_rank": float(r.median()),
        }
        results[direction] = m
        print(f"\n[metrics] {direction}  (pool = all {m['pool_size']} "
              f"candidates, ground truth included)")
        for k in ("P@1", "P@5", "P@10", "MRR", "median_rank"):
            print(f"    {k:<12} {m[k]:.4f}")

    chance_p1 = 1.0 / n_rows
    print(f"\n[orientation] chance P@1 at this pool size = {chance_p1:.4f}")
    print("[orientation] paper's published P@1 = 0.4410 — but the split, "
          "direction and pool behind that number are NOT established by this "
          "script; treat proximity as orientation, not reproduction, until "
          "the paper's protocol is pinned down.")

    # ---- Authors' demo cell, replicated for the record ------------------
    # (their cell 5-6: query 0's top-5 among *other* series, self excluded)
    print("\n[demo-replica] authors' demo output for query_idx=0 "
          "(top-5 ts neighbours of text query, self EXCLUDED as in the "
          "notebook — note the notebook itself crashes on sample_id; row "
          "order stands in for it, see header):")
    has_text_cols = "description" in df.columns
    q0 = text_emb[0:1]
    mask = torch.arange(n_rows) != 0
    scores0 = (q0 @ ts_emb[mask].T).squeeze(0)
    top5_s, top5_p = torch.topk(scores0, k=min(5, n_rows - 1))
    top5_idx = torch.where(mask)[0][top5_p]
    if has_text_cols:
        print(f"  [query row 0] description: {str(df.iloc[0]['description'])[:200]}")
    for rank, (i, s) in enumerate(zip(top5_idx.tolist(), top5_s.tolist()), 1):
        line = f"  top-{rank} | row={i} | score={s:.4f}"
        if has_text_cols:
            line += f" | {str(df.iloc[i]['description'])[:120]}"
        print(line)

    # Three worked examples with ranks, per project norms (print examples,
    # not just summary statistics):
    r_t2s = rank_of_truth(text_emb, ts_emb)
    print("\n[worked examples] text->ts rank of the true series for rows 0, "
          f"{n_rows//2}, {n_rows-1}:")
    for i in (0, n_rows // 2, n_rows - 1):
        line = f"  row {i}: true-pair rank {int(r_t2s[i])} of {n_rows}"
        if has_text_cols:
            line += f" | {str(df.iloc[i]['description'])[:100]}"
        print(line)

    # ---- Write JSON ------------------------------------------------------
    out_path = Path(args.out or f"trace_demo_repro_{args.split}.json").resolve()
    payload = {
        "script": "run_authors_demo_eval.py",
        "trace_repo": str(repo),
        "checkpoint": str(ckpt_path),
        "checkpoint_bytes": ckpt_path.stat().st_size,
        "split": args.split,
        "n_rows": n_rows,
        "embedding_dim": int(ts_emb.shape[1]),
        "text_encoder_name_in_ckpt": str(getattr(margs, "text_encoder_name", None)),
        "hard_negative_mining_in_ckpt": getattr(margs, "hard_negative_mining", None),
        "chance_P@1": chance_p1,
        "paper_P@1_for_orientation_only": 0.4410,
        "metrics": results,
        "runtime_seconds": round(time.time() - t0, 1),
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\n[done] wrote {out_path}  ({payload['runtime_seconds']} s). "
          "Paste the full console output back, and attach/commit the JSON.")


if __name__ == "__main__":
    main()

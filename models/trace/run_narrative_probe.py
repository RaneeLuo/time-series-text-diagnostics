#!/usr/bin/env python3
"""
TRACE narrative probe runner (Probe 1, narrative variant, option (b)).

Scores every item in data/processed/narrative_probe_items_certified.jsonl
(3,178 items; N1/N2/N5 400+400, N3 389+389) against the released TRACE
checkpoint, and writes per-item records in the schema
analyze_probe1_stats.py --per-item consumes (field names lifted from
models/openai_embed/run_probe1.py, which that script already consumes;
precedent for a non-numeric seed field: it wrote seed="api").

DESIGN FACTS THIS SCRIPT RESTS ON — all read from source 2026-08-08,
this session (file:line refer to the authors' repo):

  D1. The description side is INDEPENDENT of the signal side at
      inference. src/models/mm_encoder.py lines 144-152: raw Nomic
      embedding (768) -> text_adapter (Linear 768->384, l.145)
      -> LayerNorm (l.150) -> Dropout (l.151, identity in eval)
      -> F.normalize (l.152); returned untouched at l.181.
      Cross-attention (l.167-169) modifies only the channel-level
      outputs, which the retrieval score never uses.
      => swap texts embed once; signal embeddings computed separately.
      Gate G7 VERIFIES this equivalence rather than trusting the read.

  D2. The authors evaluate with a RANDOM pretrain mask. The forward
      pass, given pretrain_mask=None, draws a random mask at the stored
      mask_ratio (ts_encoder.py l.49; masking.py l.72 uses torch.rand;
      the authors' own eval, context_align_task.py l.189, passes no
      mask). model.eval() does not stop torch.rand. Policy (agreed
      2026-08-08): follow the authors' protocol, but SEED it, and
      replicate over three mask seeds -- the TRACE analogue of CLaSP's
      three checkpoints. Text embeddings are mask-independent and
      computed once.

  D3. The authors' Nomic encode call takes BARE strings (no task
      prefix), batch_size=64, unnormalised output
      (src/data/load_data.py l.133-141). Swap texts go through the
      identical call. Cached row embeddings
      (description_emb_nomic-embed-text-v1.5.pt beside the parquet,
      written by the demo run) are reused for correct captions and
      random distractors -- gated on exact string match to
      generate_dsp(row) (G6), so cache reuse is proven, not assumed.

  D4. Model construction reuses REPAIR 2 from run_authors_demo_eval.py
      verbatim (model_name override to 'TraceEncoder'; _load_model
      patched to skip the never-released Stage-1 checkpoint; strict
      state-dict gate G4; then .eval()).

REGISTERED PREDICTIONS (2026-08-08, before any run):
  P1. G7 passes: mini text path == full-forward description_emb,
      max abs diff < 1e-5.
  P2. Unperturbed full-pool P@1 (text->ts) across the three mask seeds
      stays within +-0.02 of each other, in the neighbourhood of 0.42.
      A miss here matters: mask noise would then be a real variance
      term for every probe number.
  P3. Every caption_correct and every random caption_distractor exactly
      matches generate_dsp of its source row (G6a/G6b) -- i.e. the item
      generator and this runner see the same serialisation.
      [2026-08-08: first run fired G6 on 3,162/3,178 items. Diagnosis:
      NOT data drift -- v1 of this script read the items file without
      encoding="utf-8", so a cp936 Windows locale decoded the degree
      sign into mojibake at read time. File byte-verified clean. P3
      stands; gate G5b (mojibake canary) added; this is error #10 in
      the project ledger, ours, caught by a gate + printed repr diffs.]
  P4. Random-condition accuracy > 0.90 for all four components (full-
      pool P@1 0.42 implies binary discrimination against a different
      row's caption is easy). Needed for any shortcut verdict: the
      VOID rule requires high random-condition accuracy.
  P5. N5 (location, the designed negative control) swap accuracy lands
      in [0.45, 0.60].
  Misses get recorded as misses.

USAGE (from the thesis repo root; TRACE repo cloned as a sibling and
the demo reproduction already run once, so the text caches exist):

  python models/trace/run_narrative_probe.py \
      --trace-repo ../TRACE-Multimodal-TSEncoder \
      --items data/processed/narrative_probe_items_certified.jsonl

Outputs:
  results/experiments/trace_narrative_per_item.jsonl   (all 3 seeds)
  results/experiments/trace_narrative_summary.json
  .cache/trace_swap_text_emb.pt                        (one-time, ~17 min)

Then:
  python scripts/analyze_probe1_stats.py --per-item results/experiments/trace_narrative_per_item.jsonl --out results/experiments/trace_narrative_statistics.json
  python scripts/audit_item_balance.py --results results/experiments/trace_narrative_per_item.jsonl

Paste the full console output back. Expected runtime with warm text
caches: ~17 min one-time swap-text embedding + ~4 min per mask seed.
"""

import argparse
import hashlib
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path


def fail(gate: str, msg: str):
    print(f"\n[GATE FAILED] {gate}: {msg}")
    print("Hard stop. Investigate before proceeding — do not work around.")
    sys.exit(1)


def sha(t: str) -> str:
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


EXPECTED_COUNTS = {
    ("N1", "swap"): 400, ("N1", "random"): 400,
    ("N2", "swap"): 400, ("N2", "random"): 400,
    ("N3", "swap"): 389, ("N3", "random"): 389,
    ("N5", "swap"): 400, ("N5", "random"): 400,
}
EXPECTED_TOTAL = 3178
EXPECTED_SWAP_TEXTS = 1589
HEADER_COMPONENTS = {"N1", "N2", "N5"}   # header-field swaps (overlap diagnostic)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-repo", required=True)
    ap.add_argument("--items", default="data/processed/narrative_probe_items_certified.jsonl")
    ap.add_argument("--split", default="test")
    ap.add_argument("--mask-seeds", default="13,14,15",
                    help="comma list; 13 is the checkpoint's own training seed")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="results/experiments/trace_narrative_per_item.jsonl")
    ap.add_argument("--summary", default="results/experiments/trace_narrative_summary.json")
    ap.add_argument("--swap-cache", default=".cache/trace_swap_text_emb.pt")
    args = ap.parse_args()
    t0 = time.time()
    mask_seeds = [int(s) for s in args.mask_seeds.split(",")]

    # ---- G1: repo layout -------------------------------------------------
    repo = Path(args.trace_repo).resolve()
    for rel in ("src/models/mm_encoder.py", "src/data/dataloader.py",
                "src/data/load_data.py"):
        if not (repo / rel).is_file():
            fail("G1-repo", f"{repo/rel} not found — is --trace-repo the clone root?")
    print(f"[G1] TRACE repo: {repo}")

    import os
    os.environ["TTRAG_DATA_DIR"] = str(repo / "dataset") + "/"
    os.environ["TTRAG_CHECKPOINTS_DIR"] = str(repo / "results/model_checkpoints") + "/"
    os.environ["TTRAG_RESULTS_DIR"] = str(repo / "results/model_results") + "/"
    sys.path.insert(0, str(repo))

    # ---- G2: checkpoint --------------------------------------------------
    ckpt_path = repo / "results/model_checkpoints/context_align/retriever_demo.pt"
    if not ckpt_path.is_file():
        fail("G2-ckpt", f"{ckpt_path} missing")
    print(f"[G2] checkpoint: {ckpt_path} ({ckpt_path.stat().st_size:,} bytes)")

    # ---- G3: data + text caches -----------------------------------------
    retr_dir = repo / "dataset" / "retrieval"
    parquet = retr_dir / args.split / f"{args.split}.parquet"
    if not parquet.is_file():
        fail("G3-data", f"{parquet} missing (nested layout — see demo script REPAIR 3)")

    import numpy as np
    import pyarrow.parquet as pq
    import torch
    import torch.nn.functional as F
    from tqdm import tqdm

    device = torch.device(args.device if args.device
                          else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    print(f"[env] torch {torch.__version__}, device {device}, "
          f"mask seeds {mask_seeds}, batch {args.batch_size}")

    df = pq.read_table(parquet).to_pandas()
    n_rows = len(df)
    print(f"[G3] {parquet.name}: {n_rows} rows")

    # ---- Load model: REPAIR 2, verbatim from run_authors_demo_eval.py ---
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    margs = ckpt["args"]
    enc_name = getattr(margs, "text_encoder_name", None)
    mask_ratio = getattr(margs, "mask_ratio", None)
    print(f"[model] stored text_encoder_name = {enc_name!r}")
    print(f"[model] stored mask_ratio        = {mask_ratio!r}  "
          f"(applied RANDOMLY at every forward — hence the seeded replication)")
    print(f"[model] stored hard_negative_mining = "
          f"{getattr(margs, 'hard_negative_mining', '<absent>')!r}")
    if enc_name != "nomic-ai/nomic-embed-text-v1.5":
        fail("G-enc", f"checkpoint names encoder {enc_name!r}, not Nomic v1.5 — "
                      "the cached embeddings and this script's assumptions break")

    from src.models.mm_encoder import MultiModalEncoder
    from src.data.dataloader import get_dataloader
    from src.data.load_data import generate_dsp
    from src.models.timeseries_encoders.ts_encoder import TS_Encoder
    from copy import deepcopy

    stored_model_name = getattr(margs, "model_name", "<absent>")
    print(f"[repair2] stored model_name = {stored_model_name!r}; overriding "
          f"to 'TraceEncoder' (rename hypothesis — tested by G4)")
    margs.model_name = "TraceEncoder"

    def _patched_load_model(self, pretraining_task_name):
        pretraining_args = deepcopy(self.args)
        pretraining_args.task_name = pretraining_task_name
        self.ts_encoder = TS_Encoder(configs=pretraining_args)
        print("[repair2] ts_encoder constructed at random init; Stage-1 load "
              "skipped (never released); all weights come from Stage-2 next.")

    MultiModalEncoder._load_model = _patched_load_model
    margs.rank = 0
    model = MultiModalEncoder(margs)
    state_dict = {k.replace("module.", ""): v
                  for k, v in ckpt["model_state_dict"].items()}
    load_report = model.load_state_dict(state_dict, strict=False)

    # ---- G4: strict state-dict fit --------------------------------------
    if load_report.missing_keys or load_report.unexpected_keys:
        print(f"[G4] missing: {load_report.missing_keys}")
        print(f"[G4] unexpected: {load_report.unexpected_keys}")
        fail("G4-statedict", "state dict does not match the constructed model")
    model.to(device).eval()
    if model.training:
        fail("G4-eval", "model not in eval mode — dropout would be live")
    print("[G4] state dict strict-clean; model in eval mode")

    # ---- G5: item-file integrity ----------------------------------------
    items_path = Path(args.items)
    if not items_path.is_file():
        fail("G5-items", f"{items_path} missing")
    # FIX 2026-08-08 (error #10, ours): read as UTF-8 EXPLICITLY. The
    # original line used open() with no encoding; on a cp936-locale
    # Windows machine that decoded the UTF-8 degree sign (C2 B0) as
    # the character U+63B3, corrupting every temperature clause at READ
    # time and firing G6 for 3,162/3,178 items. The file itself was
    # byte-verified clean (14,421 proper degree signs, zero mojibake).
    items = [json.loads(l) for l in items_path.open(encoding="utf-8")]

    # G5b mojibake canary: the certified file's only non-ASCII char is
    # the degree sign. If locale decoding ever slips back in, these
    # characters appear; stop before they can reach any comparison.
    canary = {"\u63b3", "\ufffd"}  # mojibake degree sign, replacement char
    n_deg = sum(it["caption_correct"].count("\u00b0") for it in items)
    hit = [it["item_id"] for it in items
           if canary & set(it["caption_correct"] + it["caption_distractor"])]
    print(f"[G5b] degree signs seen after decode: {n_deg} | "
          f"mojibake canary hits: {len(hit)}")
    if hit:
        fail("G5b-mojibake", f"corrupted decode detected, e.g. {hit[:3]} — "
                             "the items file was not read as UTF-8")
    if n_deg == 0:
        fail("G5b-degrees", "zero degree signs after decode — the certified "
                            "file contains 14,421; the read path is wrong")
    counts = Counter((it["component"], it["condition"]) for it in items)
    ids = {it["item_id"] for it in items}
    keysets = {tuple(sorted(it.keys())) for it in items}
    print(f"[G5] items: {len(items)} | unique item_id: {len(ids)} | "
          f"key-set variants: {len(keysets)}")
    for k in sorted(EXPECTED_COUNTS):
        print(f"      {k}: {counts.get(k, 0)}")
    if len(items) != EXPECTED_TOTAL or len(ids) != EXPECTED_TOTAL:
        fail("G5-count", f"expected {EXPECTED_TOTAL} items with unique ids")
    if dict(counts) != EXPECTED_COUNTS:
        fail("G5-composition", f"component x condition counts differ from the "
                               f"certified record: {dict(counts)}")
    if len(keysets) != 1:
        fail("G5-schema", "items do not share one key set")

    # ---- G6: row linkage — serialisation must match char-for-char -------
    # G6a: caption_correct == generate_dsp(source row), for ALL items.
    # G6b: random caption_distractor == generate_dsp(noted distractor row).
    dsp_cache = {}
    def dsp(row: int) -> str:
        if row not in dsp_cache:
            dsp_cache[row] = generate_dsp(df.iloc[row]["description"])
        return dsp_cache[row]

    bad_a, bad_b = [], []
    rand_src_row = {}
    for it in items:
        row = int(it["sample_id"].rsplit("_", 1)[1])
        it["_row"] = row
        if not (0 <= row < n_rows):
            fail("G6-range", f"{it['item_id']}: row {row} outside 0..{n_rows-1}")
        if it["caption_correct"] != dsp(row):
            bad_a.append(it["item_id"])
        if it["condition"] == "random":
            m = re.search(r"random distractor row (\d+)", it.get("swap_note", ""))
            if not m:
                fail("G6-note", f"{it['item_id']}: swap_note lacks distractor row")
            drow = int(m.group(1))
            rand_src_row[it["item_id"]] = drow
            if it["caption_distractor"] != dsp(drow):
                bad_b.append(it["item_id"])
    print(f"[G6] caption_correct matches serialisation: "
          f"{len(items)-len(bad_a)}/{len(items)}")
    print(f"[G6] random distractor matches source row:  "
          f"{sum(1 for it in items if it['condition']=='random')-len(bad_b)}"
          f"/{sum(1 for it in items if it['condition']=='random')}")
    if bad_a or bad_b:
        for x in (bad_a + bad_b)[:3]:
            print(f"      first mismatches: {x}")
        fail("G6-drift", "item text != serializer output — generator and runner "
                         "do not see the same data (prediction P3 falsified; "
                         "paste this output back)")

    # ---- Cached raw row embeddings (written by the demo run) ------------
    short = enc_name.split("/")[-1]
    cache_path = retr_dir / args.split / f"description_emb_{short}.pt"
    if not cache_path.is_file():
        fail("G3-cache", f"{cache_path} missing — run run_authors_demo_eval.py "
                         "once first (it writes the text caches; ~60 min CPU)")
    raw_rows = torch.load(cache_path, map_location="cpu").float()
    print(f"[G3] cached row embeddings: {tuple(raw_rows.shape)}")
    if raw_rows.shape[0] != n_rows or raw_rows.shape[1] != model.text_embedding_dim:
        fail("G3-shape", f"cache shape {tuple(raw_rows.shape)} != "
                         f"({n_rows}, {model.text_embedding_dim})")

    # ---- Swap texts: embed once via the authors' exact call (D3) --------
    swap_items = [it for it in items if it["condition"] == "swap"]
    swap_texts = [it["caption_distractor"] for it in swap_items]
    uniq = list(dict.fromkeys(swap_texts))
    print(f"[swap] swap items: {len(swap_items)} | unique texts: {len(uniq)}")
    if len(swap_items) != EXPECTED_SWAP_TEXTS:
        fail("G-swapcount", f"expected {EXPECTED_SWAP_TEXTS} swap items")

    cache_file = Path(args.swap_cache)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    hashes = [sha(t) for t in uniq]
    raw_swap = None
    if cache_file.is_file():
        blob = torch.load(cache_file, map_location="cpu")
        if blob.get("encoder") == enc_name and blob.get("hashes") == hashes:
            raw_swap = blob["emb"].float()
            print(f"[swap] cache hit: {cache_file} {tuple(raw_swap.shape)}")
        else:
            print("[swap] cache present but stale (encoder/hash mismatch) — re-embedding")
    if raw_swap is None:
        from sentence_transformers import SentenceTransformer
        print(f"[swap] embedding {len(uniq)} texts with {enc_name} "
              f"(authors' call: bare strings, batch 64) — one-time cost")
        st = SentenceTransformer(enc_name, trust_remote_code=True)
        t_emb = time.time()
        raw_swap = st.encode(uniq, batch_size=64, show_progress_bar=True,
                             convert_to_tensor=True).cpu().float()
        print(f"[swap] embedded in {time.time()-t_emb:.0f} s")
        torch.save({"encoder": enc_name, "hashes": hashes, "emb": raw_swap},
                   cache_file)
        print(f"[swap] cached to {cache_file}")
    swap_idx = {h: i for i, h in enumerate(hashes)}

    # ---- Mini text path (D1) --------------------------------------------
    @torch.no_grad()
    def text_path(raw: torch.Tensor) -> torch.Tensor:
        e = model.text_adapter(raw.to(device))
        e = model.norm(e)
        e = model.dropout(e)          # identity in eval; kept for fidelity
        return F.normalize(e, dim=-1).cpu()

    text_rows = text_path(raw_rows)          # [n_rows, 384], mask-independent
    text_swap = text_path(raw_swap)          # [1589, 384]

    # ---- Dataloader (authors' path; test split is shuffle=False) --------
    margs.task_name = "retrieval"
    margs.data_split = args.split
    margs.batch_size = args.batch_size
    margs.device = device
    margs.distributed = False
    if args.split != "test":
        fail("G-order", "only the test split preserves row order here")
    data_loader = get_dataloader(margs)

    # ---- G7: independence/equivalence check (prediction P1) -------------
    first = next(iter(data_loader))
    with torch.no_grad():
        out = model(
            x_enc=first.timeseries.float().to(device),
            input_mask=first.input_mask.long().to(device),
            channel_description_emb=first.channel_description_emb.to(device),
            description_emb=first.description_emb.to(device),
            event_emb=first.event_emb.to(device),
        )
    full = F.normalize(out.description_emb.detach().cpu().float(), dim=-1)
    mini = text_rows[: full.shape[0]]
    diff = (full - mini).abs().max().item()
    print(f"[G7] full-forward vs mini text path, first {full.shape[0]} rows: "
          f"max abs diff = {diff:.2e}")
    if diff > 1e-4:
        fail("G7-equivalence", "description path is NOT reproduced by "
             "adapter+norm+normalize — the independence read (D1) is wrong "
             "somewhere; prediction P1 falsified; paste this back")

    # ---- Signal embeddings per mask seed + orientation (P2) -------------
    per_seed_ts = {}
    p1_by_seed = {}
    for seed in mask_seeds:
        torch.manual_seed(seed)
        chunks, n_seen = [], 0
        t_s = time.time()
        with torch.no_grad():
            for batch_x in tqdm(data_loader, total=len(data_loader),
                                desc=f"signal emb (mask seed {seed})"):
                o = model(
                    x_enc=batch_x.timeseries.float().to(device),
                    input_mask=batch_x.input_mask.long().to(device),
                    channel_description_emb=batch_x.channel_description_emb.to(device),
                    description_emb=batch_x.description_emb.to(device),
                    event_emb=batch_x.event_emb.to(device),
                )
                chunks.append(o.embeddings.detach().cpu())
                n_seen += batch_x.timeseries.shape[0]
        ts_emb = F.normalize(torch.cat(chunks).float(), dim=-1)
        if not (n_seen == n_rows == ts_emb.shape[0]):
            fail("G8-counts", f"seed {seed}: rows {n_rows} vs seen {n_seen} "
                              f"vs emb {ts_emb.shape[0]}")
        per_seed_ts[seed] = ts_emb

        # full-pool orientation, authors' direction text->ts
        scores = text_rows @ ts_emb.T
        truth = scores.diag()
        p1 = float((scores.max(dim=1).values <= truth).float().mean())
        p1_by_seed[seed] = p1
        print(f"[G9] seed {seed}: unperturbed full-pool P@1 (text->ts) = "
              f"{p1:.4f}  ({time.time()-t_s:.0f} s)")
        if p1 < 0.30:
            fail("G9-orientation", f"P@1 {p1:.3f} < 0.30 — wiring error more "
                                   "likely than model behaviour; stop and check")
    spread = max(p1_by_seed.values()) - min(p1_by_seed.values())
    print(f"[G9] P@1 across mask seeds: "
          f"{ {k: round(v,4) for k,v in p1_by_seed.items()} } | spread {spread:.4f}"
          f"  (prediction P2: spread <= 0.04 total, i.e. +-0.02)")

    # ---- Score all items, all seeds -------------------------------------
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for seed in mask_seeds:
        ts_emb = per_seed_ts[seed]
        for it in items:
            v = ts_emb[it["_row"]]
            c = text_rows[it["_row"]]
            if it["condition"] == "swap":
                d = text_swap[swap_idx[sha(it["caption_distractor"])]]
            else:
                d = text_rows[rand_src_row[it["item_id"]]]
            sc = float(v @ c)
            sd = float(v @ d)
            records.append({
                "seed": f"mask{seed}",
                "item_id": it["item_id"],
                "pair_key": it["item_id"].rsplit("|", 1)[0],
                "component": it["component"],
                "condition": it["condition"],
                "swap_from": it["swap_from"], "swap_to": it["swap_to"],
                "sample_id": it["sample_id"],
                "duration_class": it["duration_class"],
                "header_or_prose": ("header" if it["component"] in
                                    HEADER_COMPONENTS else "prose"),
                "sim_correct": sc, "sim_distractor": sd,
                "margin": sc - sd, "correct": bool(sc > sd),
            })
    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"\n[out] wrote {len(records)} records "
          f"({len(items)} items x {len(mask_seeds)} seeds) -> {out_path}")

    # ---- Summary table ---------------------------------------------------
    print(f"\n{'seed':<8}{'component':<11}{'random':>8}{'swap':>8}{'gap':>8}"
          f"{'margin(swap)':>14}")
    summary = defaultdict(dict)
    for seed in mask_seeds:
        for comp in sorted({it["component"] for it in items}):
            sel = [r for r in records
                   if r["seed"] == f"mask{seed}" and r["component"] == comp]
            rnd = [r["correct"] for r in sel if r["condition"] == "random"]
            swp = [r["correct"] for r in sel if r["condition"] == "swap"]
            mg = [r["margin"] for r in sel if r["condition"] == "swap"]
            a_r, a_s = float(np.mean(rnd)), float(np.mean(swp))
            summary[f"mask{seed}"][comp] = {
                "acc_random": a_r, "acc_swap": a_s, "gap": a_r - a_s,
                "mean_margin_swap": float(np.mean(mg)),
                "n_random": len(rnd), "n_swap": len(swp),
            }
            print(f"{'mask'+str(seed):<8}{comp:<11}{a_r:>8.3f}{a_s:>8.3f}"
                  f"{a_r-a_s:>8.3f}{float(np.mean(mg)):>14.4f}")

    payload = {
        "script": "run_narrative_probe.py",
        "items_file": str(items_path), "n_items": len(items),
        "trace_repo": str(repo), "checkpoint": str(ckpt_path),
        "text_encoder": enc_name, "mask_ratio": mask_ratio,
        "mask_seeds": mask_seeds,
        "unperturbed_P@1_text2ts_by_seed": p1_by_seed,
        "G7_max_abs_diff": diff,
        "per_seed_component_summary": {k: dict(v) for k, v in summary.items()},
        "predictions": {
            "P1_G7_diff_lt_1e-5": diff < 1e-5,
            "P2_P@1_spread": spread,
            "P3_serialisation_match": True,   # gates would have stopped otherwise
        },
        "runtime_seconds": round(time.time() - t0, 1),
    }
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n[done] summary -> {args.summary}  "
          f"({payload['runtime_seconds']} s total)")
    print("[next] run analyze_probe1_stats.py --per-item and "
          "audit_item_balance.py on the per-item file; paste ALL console "
          "output back, including this run's.")


if __name__ == "__main__":
    main()

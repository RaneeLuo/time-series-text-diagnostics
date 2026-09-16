#!/usr/bin/env python3
"""
TRACE task 1 of 2 — read the released checkpoint's stored `args`.

Purpose: settle, from the checkpoint itself, (a) which text encoder the
released model was aligned against and (b) whether hard-negative mining
was on. Nothing downstream (embedding SUSHI captions, the substrate
baseline) may proceed until these two are read from source.

Why this cannot be inferred from the repo alone (verified 2026-08-07 by
reading context_align.py lines 24-64):
  - configs/align.yaml asks for nomic-ai/nomic-embed-text-v1.5,
    num_negatives=64, hard_negative_mining=True.
  - BUT context_align.py unconditionally overwrites those three values
    with the CLI values, whose *defaults* are bert-base-uncased,
    num_negatives=10, hard_negative_mining=False.
  - The README's example Stage-2 command passes only --cross_attend.
    If that literal command produced the released checkpoint, then the
    yaml was silently overridden and the checkpoint was trained with
    BERT text embeddings and NO hard negatives.
The stored args are the only decisive record. This script prints them.

Usage (from anywhere; needs only torch + the checkpoint file):
    python read_checkpoint_args.py \
        --checkpoint /path/to/TRACE-Multimodal-TSEncoder/results/model_checkpoints/context_align/retriever_demo.pt

Security note: torch.load must run with weights_only=False because the
checkpoint stores an argparse.Namespace/config object. Only run this on
the checkpoint downloaded from the authors' official repository.
"""

import argparse
import os
import sys
from pathlib import Path


def fail(gate: str, msg: str):
    print(f"\n[GATE FAILED] {gate}: {msg}")
    print("Hard stop. Investigate before proceeding — do not work around.")
    sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True,
                    help="Path to retriever_demo.pt inside the TRACE clone")
    args = ap.parse_args()

    import torch
    print(f"[env] torch {torch.__version__}, python {sys.version.split()[0]}")

    ckpt_path = Path(args.checkpoint).resolve()

    # ---- Gate 1: file exists and has a plausible size -------------------
    if not ckpt_path.is_file():
        fail("G1-exists", f"no file at {ckpt_path}")
    size = ckpt_path.stat().st_size
    print(f"[G1] checkpoint file: {ckpt_path}")
    print(f"[G1] size on disk: {size:,} bytes ({size/1e6:.1f} MB)")
    # Handoff §4.0 recorded 46.3 MB. Allow slack in case the authors
    # re-uploaded, but a wildly different size means a different artifact.
    if not (30e6 < size < 70e6):
        fail("G1-size", f"{size/1e6:.1f} MB is far from the documented 46.3 MB "
                        "— this is not the artifact §4.0 verified. Stop and compare.")
    if size != 46_271_904:
        print(f"[G1][note] size differs from the 46,271,904 bytes I saw in a fresh "
              f"clone on 2026-08-07 — not fatal, but record the difference.")

    # ---- Make the TRACE repo importable, in case the pickled args -------
    # object references a class defined in src/ (walk up from the
    # checkpoint until a directory containing src/ is found).
    repo_root = None
    for parent in ckpt_path.parents:
        if (parent / "src").is_dir() and (parent / "context_align.py").is_file():
            repo_root = parent
            break
    if repo_root is not None:
        sys.path.insert(0, str(repo_root))
        # src.common reads env vars at import time; give it harmless values
        # so a missing .env cannot crash the unpickle.
        os.environ.setdefault("TTRAG_DATA_DIR", str(repo_root / "dataset") + "/")
        os.environ.setdefault("TTRAG_CHECKPOINTS_DIR", str(repo_root / "results/model_checkpoints") + "/")
        os.environ.setdefault("TTRAG_RESULTS_DIR", str(repo_root / "results/model_results") + "/")
        print(f"[setup] TRACE repo root detected and added to sys.path: {repo_root}")
    else:
        print("[setup] no TRACE repo root found above the checkpoint; "
              "proceeding — fine if the pickle only contains stdlib types.")

    # ---- Load -----------------------------------------------------------
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except Exception as e:
        fail("G2-load", f"torch.load failed: {type(e).__name__}: {e}")

    # ---- Gate 2: expected top-level structure ---------------------------
    if not isinstance(ckpt, dict):
        fail("G2-structure", f"checkpoint is {type(ckpt)}, expected dict")
    print(f"\n[G2] top-level keys: {sorted(ckpt.keys())}")
    for required in ("args", "model_state_dict"):
        if required not in ckpt:
            fail("G2-keys", f"missing key '{required}' — the demo.ipynb loading "
                            "recipe would not work on this file")

    # Print any bookkeeping the authors stored (epoch, loss, metrics...).
    for k, v in ckpt.items():
        if k in ("args", "model_state_dict", "optimizer_state_dict"):
            continue
        print(f"[G2] extra key '{k}': {v!r}")

    # ---- The stored args, in full ---------------------------------------
    raw = ckpt["args"]
    if hasattr(raw, "__dict__"):
        d = dict(vars(raw))
    elif isinstance(raw, dict):
        d = dict(raw)
    else:
        fail("G3-args-type", f"args is {type(raw)}; cannot render")

    print(f"\n[args] type: {type(raw).__name__}, {len(d)} entries")
    print("[args] full dump (sorted):")
    for k in sorted(d):
        print(f"    {k} = {d[k]!r}")

    # ---- Gate 3: the two decisive fields must exist ---------------------
    decisive = ["text_encoder_name", "hard_negative_mining", "num_negatives"]
    missing = [k for k in decisive if k not in d]
    if missing:
        fail("G3-decisive", f"stored args lack {missing} — the training-time "
                            "values cannot be read from this checkpoint; "
                            "that itself is the finding to record.")

    print("\n" + "=" * 62)
    print("DECISIVE VALUES (read from the checkpoint, not inferred)")
    print("=" * 62)
    print(f"  text_encoder_name    = {d['text_encoder_name']!r}")
    print(f"  num_negatives        = {d['num_negatives']!r}")
    print(f"  hard_negative_mining = {d['hard_negative_mining']!r}")
    for extra in ("cross_attend", "random_seed", "seq_len_channel", "d_model",
                  "train_batch_size", "init_lr", "max_epoch", "task_name",
                  "model_name", "scale"):
        if extra in d:
            print(f"  {extra:<20} = {d[extra]!r}")
    print("=" * 62)
    if not d["hard_negative_mining"]:
        print("⚠ hard_negative_mining is FALSE in the released checkpoint.")
        print("  This changes what the TRACE arm can test. Do not proceed to")
        print("  adapter design before discussing; the model's scientific")
        print("  interest rested on hard-negative training.")
    if "bert" in str(d["text_encoder_name"]).lower():
        print("⚠ Text encoder is BERT-family, not Nomic — the yaml was")
        print("  overridden by CLI defaults. Captions must be embedded with")
        print("  exactly this encoder or the comparison is meaningless.")

    # ---- Gate 4: parameter count vs file size and paper claim -----------
    sd = ckpt["model_state_dict"]
    n_params = sum(t.numel() for t in sd.values())
    fp32_bytes = n_params * 4
    print(f"\n[G4] state-dict tensors: {len(sd)}")
    print(f"[G4] total parameters: {n_params:,} ({n_params/1e6:.2f} M)")
    print(f"[G4] fp32 footprint: {fp32_bytes/1e6:.1f} MB vs file {size/1e6:.1f} MB")
    if not (0.6 < fp32_bytes / size < 1.3):
        fail("G4-size-reconcile", "parameter bytes and file size do not "
             "reconcile — the file may contain more than one model or "
             "non-fp32 tensors; inspect before trusting anything above.")
    # §4.0 recorded ≈11.5M params consistent with the paper's model size.
    if not (8e6 < n_params < 16e6):
        print(f"[G4][warn] {n_params/1e6:.2f}M params is outside the 8-16M "
              f"band around the documented ≈11.5M — record and investigate.")

    # Per-component breakdown: shows whether a text tower is inside the
    # checkpoint (there should NOT be one — text arrives pre-embedded).
    by_prefix = {}
    for k, t in sd.items():
        prefix = k.split(".")[0]
        by_prefix[prefix] = by_prefix.get(prefix, 0) + t.numel()
    print("[G4] parameters by top-level module:")
    for p, n in sorted(by_prefix.items(), key=lambda kv: -kv[1]):
        print(f"    {p:<30} {n:>12,}")

    # Text-projection input width is indirect evidence of the text
    # encoder's embedding dim (BERT-base: 768; nomic-v1.5: 768 too, so
    # this alone cannot separate them — printed for the record only).
    for k, t in sd.items():
        if "text" in k.lower() or "description" in k.lower():
            print(f"[G4] text-related tensor: {k} shape {tuple(t.shape)}")

    print("\n[done] All gates passed. Paste this full output back.")


if __name__ == "__main__":
    main()

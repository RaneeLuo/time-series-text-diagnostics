"""
ChatTS task zero — CPU-only verification of the pinned paper-era checkpoint.
V2 (2026-08-15): download as plain files (local_dir), no symlinks —
v1 crashed on Windows with WinError 1314 because the default cache
layout of huggingface_hub creates symlinks, which Windows forbids
without Developer Mode. No gate had run yet; no fact was affected.

Pins revision 1e661101dcfff86dc66f3397336b85f2f1cc5e89 of
bytedance-research/ChatTS-14B (the last paper-era commit, 2025-07-24,
before the 0801 in-place weight replacement).

Downloads ONLY small files (no *.bin weights). Loads tokenizer and
processor on CPU with trust_remote_code and gates every fact the design
depends on. Gates GZ1-GZ4 are HARD STOPS; GZ5-GZ6 are reports.

No GPU, no weights, no model forward pass happens here.

Place at: models/chatts/verify_task_zero.py
Run from repo root: python models/chatts/verify_task_zero.py
"""
import argparse
import sys
import json
import math
from pathlib import Path

import numpy as np

EXPECTED_REVISION = "1e661101dcfff86dc66f3397336b85f2f1cc5e89"
REPO_ID = "bytedance-research/ChatTS-14B"
EXPECTED_TS_CONFIG = {
    "hidden_size": 5120,
    "num_features": 2,
    "num_layers": 5,
    "patch_size": 16,
    "max_length": 2048,
}
TS_START_ID = 151665
TS_END_ID = 151666

FAILURES = []


def gate(name: str, ok: bool, detail: str, hard: bool = True):
    status = "PASS" if ok else ("FAIL (HARD)" if hard else "MISS (report)")
    print(f"[{name}] {status} — {detail}")
    if not ok and hard:
        FAILURES.append(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local-dir", default="data/chatts_pinned_meta",
                    help="plain folder where the small files land (weights are NOT downloaded)")
    ap.add_argument("--revision", default=EXPECTED_REVISION)
    args = ap.parse_args()

    print("=== ChatTS task-zero verifier (v2) ===")
    print(f"repo      : {REPO_ID}")
    print(f"revision  : {args.revision}")
    print(f"local dir : {args.local_dir}")

    import transformers
    import torch
    print(f"[env] transformers {transformers.__version__} | torch {torch.__version__} "
          f"| numpy {np.__version__}  (pinned config was saved under transformers 4.46.2)")

    # ---- download small files only, as PLAIN files (Windows-safe, no symlinks) ----
    from huggingface_hub import snapshot_download
    local = snapshot_download(
        repo_id=REPO_ID,
        revision=args.revision,
        local_dir=args.local_dir,
        allow_patterns=["*.json", "*.py", "*.txt", "*.jsonl",
                        "merges.txt", "LICENSE", "NOTICE"],
    )
    local = Path(local)
    files = sorted(p.name for p in local.iterdir() if p.is_file())
    total_bytes = sum(p.stat().st_size for p in local.iterdir() if p.is_file())
    bins = [f for f in files if f.endswith(".bin") or f.endswith(".safetensors")]
    print(f"[download] {len(files)} files, {total_bytes/2**20:.2f} MiB total")
    print(f"[download] files: {files}")
    gate("GZ0-noweights", len(bins) == 0,
         f"weight files fetched: {bins if bins else 'none'} (expected none)")

    # ---- GZ1: config facts ----
    cfg = json.loads((local / "config.json").read_text(encoding="utf-8"))
    ts_cfg = cfg.get("ts", {})
    diffs = {k: (ts_cfg.get(k), v) for k, v in EXPECTED_TS_CONFIG.items()
             if ts_cfg.get(k) != v}
    gate("GZ1-tsconfig", not diffs,
         f"ts block observed={ts_cfg} | mismatches vs expected: {diffs if diffs else 'none'}")
    gate("GZ1-tokens",
         cfg.get("ts_token_start_index") == TS_START_ID
         and cfg.get("ts_token_end_index") == TS_END_ID,
         f"ts token ids observed=({cfg.get('ts_token_start_index')},"
         f"{cfg.get('ts_token_end_index')}) expected=({TS_START_ID},{TS_END_ID})")
    gate("GZ1-dtype", cfg.get("torch_dtype") == "float16",
         f"torch_dtype={cfg.get('torch_dtype')} (expected float16)")

    # ---- GZ2: revision canary on the processing source ----
    proc_src = (local / "processing_qwen2_ts.py").read_text(encoding="utf-8")
    has_paper_prefix = "Value Offset" in proc_src
    has_0801_prefix = "left=" in proc_src
    gate("GZ2-revision-canary", has_paper_prefix and not has_0801_prefix,
         f"'Value Offset' present={has_paper_prefix}, 'left=' present={has_0801_prefix} "
         f"(paper-era expects True/False)")

    # ---- load tokenizer + processor on CPU ----
    from transformers import AutoTokenizer, AutoProcessor
    tok = AutoTokenizer.from_pretrained(str(local), trust_remote_code=True)
    proc = AutoProcessor.from_pretrained(str(local), trust_remote_code=True, tokenizer=tok)

    # ---- GZ3: special tokens are single tokens ----
    ts_open = tok.encode("<ts>", add_special_tokens=False)
    ts_close = tok.encode("<ts/>", add_special_tokens=False)
    gate("GZ3-specials", ts_open == [TS_START_ID] and ts_close == [TS_END_ID],
         f"'<ts>'->{ts_open} '<ts/>'->{ts_close} expected [{TS_START_ID}]/[{TS_END_ID}]")

    # ---- GZ4: processor round-trip on a fixed series ----
    test_series = np.arange(8, dtype=np.float64)  # mean 3.5, max|x-mean|=3.5 -> scale 3.5/3
    expected_prefix = "[Value Offset: -3.5000|Value Scaling: 1.1667]"
    prompt = "<|im_start|>user\nSeries: <ts><ts/>. Reply with A or B.<|im_end|>\n<|im_start|>assistant\n"
    out = proc(text=[prompt], timeseries=[test_series], padding=True, return_tensors="pt")
    ts_tensor = out["timeseries"]
    decoded = tok.decode(out["input_ids"][0])
    prefix_ok = expected_prefix in decoded
    shape_ok = tuple(ts_tensor.shape) == (1, 16, 1)
    dtype_ok = str(ts_tensor.dtype) == "torch.float16"
    gate("GZ4-prefix", prefix_ok,
         f"expected prefix {expected_prefix!r} in decoded prompt: {prefix_ok}")
    gate("GZ4-tensor", shape_ok and dtype_ok,
         f"timeseries tensor shape={tuple(ts_tensor.shape)} dtype={ts_tensor.dtype} "
         f"(expected (1,16,1) float16)")
    if not prefix_ok:
        print("  decoded prompt for diagnosis:")
        print("  " + repr(decoded))

    # ---- GZ5 (report): patch arithmetic for our substrates ----
    ps = ts_cfg.get("patch_size", 16)
    for L, name in [(12, "TRUCE"), (2048, "SUSHI")]:
        print(f"[GZ5-report] {name} L={L}: ceil(L/{ps}) = {math.ceil(L/ps)} patches "
              f"(arithmetic only; model-side count is a GPU-session gate)")

    # ---- GZ6 (report): MCQ skeleton token budget ----
    mcq = ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
           "<|im_start|>user\nHere is a time series: <ts><ts/>.\n"
           "Which description matches it?\n"
           "(A) PLACEHOLDER CAPTION ONE TEXT GOES HERE.\n"
           "(B) PLACEHOLDER CAPTION TWO TEXT GOES HERE.\n"
           "Answer with exactly one letter, A or B.<|im_end|>\n"
           "<|im_start|>assistant\n")
    out2 = proc(text=[mcq], timeseries=[np.arange(2048, dtype=np.float64)],
                padding=True, return_tensors="pt")
    n_text_tokens = int(out2["input_ids"].shape[1])
    print(f"[GZ6-report] MCQ skeleton text tokens (1 series, L=2048): {n_text_tokens} "
          f"(expected < 400; series enters as patch embeddings, not text tokens)")

    # ---- verdict ----
    print("=" * 50)
    if FAILURES:
        print(f"HARD STOP — failed gates: {FAILURES}")
        sys.exit(1)
    print("ALL HARD GATES GREEN. Task-zero CPU facts verified for the pinned revision.")


if __name__ == "__main__":
    main()

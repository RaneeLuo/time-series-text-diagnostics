"""
ChatTS GPU runner — all three probes, logit readout, smoke-gated.

Runs on the rented pod ONLY (see docs/chatts_gpu_runbook.md). Everything
here was prepared and gated on CPU beforehand; this script's job is
pure execution plus the gates that only the live model can check.

Readout (accepted design): one forward pass per question; compare the
model's next-token logits for the A-candidates vs B-candidates (max
over variants, rule pinned pre-run). Greedy generation runs only on the
agreement samples (200/probe; pre-named fallback: if agreement < 0.95
in a probe, generation becomes primary for that probe).

Smoke stage (--smoke) is a HARD prerequisite: 16 rows per block through
the full pipeline, including the splice-arithmetic gate that only the
GPU can verify (logits length - input length == patches - 2 per series:
SUSHI +126, TRUCE -1) and a GPU-side re-check of manual-path
equivalence on 50 rows (the environment changed; re-prove, don't trust).

Resume: responses append per (mcq_id, condition); already-written keys
are skipped, so an interrupted run continues where it stopped.

Usage on the pod (repo root):
  python -m models.chatts.run_probes_gpu --smoke   [gates only, ~minutes]
  python -m models.chatts.run_probes_gpu --probe 1
  python -m models.chatts.run_probes_gpu --probe 2
  python -m models.chatts.run_probes_gpu --probe 3
Required flags: --checkpoint ckpt_paper --pairs ... --p1-manifest ...
                --p2-manifest ... --p3-manifest ...
"""
import argparse
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from models.chatts import perturbations as P
from models.chatts.manual_encoding import (load_checkpoint_sp_encoding,
                                           manual_encode, prefix_text_of)

EXPECTED_WEIGHT_BYTES = 29_750_198_302  # FILE bytes (#17; the pod ran with this value)                                           
# EXPECTED_WEIGHT_BYTES = 29_749_997_568  
MCQ_SHA = "4029f94e2f6d"
FN_SHA = "4a4b3475f9e7"
BASE_SEED = 42
SMOKE_N = 16
AGREEMENT_N = 200
AGREEMENT_THRESHOLD = 0.95

FAILURES = []


def gate(name, ok, detail, hard=True):
    status = "PASS" if ok else ("FAIL (HARD)" if hard else "MISS (report)")
    print(f"[{name}] {status} — {detail}", flush=True)
    if not ok and hard:
        FAILURES.append(name)


def hard_stop_if_failed(stage):
    if FAILURES:
        print(f"HARD STOP at {stage} — failed gates: {FAILURES}", flush=True)
        sys.exit(1)


# ------------------------------------------------------------- loading
def load_model(ckpt_dir):
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
    bins = sorted(Path(ckpt_dir).glob("pytorch_model-*.bin"))
    total = sum(p.stat().st_size for p in bins)
    gate("GR0-weights", total == EXPECTED_WEIGHT_BYTES,
         f"weight bytes = {total:,} (expected {EXPECTED_WEIGHT_BYTES:,}; "
         f"{len(bins)} shards)")
    hard_stop_if_failed("weight check")
    tok = AutoTokenizer.from_pretrained(ckpt_dir, trust_remote_code=True)
    proc = AutoProcessor.from_pretrained(ckpt_dir, trust_remote_code=True, tokenizer=tok)
    model = AutoModelForCausalLM.from_pretrained(
        ckpt_dir, trust_remote_code=True, torch_dtype=torch.float16, device_map=0)
    model.eval()
    sp = load_checkpoint_sp_encoding(ckpt_dir)
    return tok, proc, model, sp


def letter_token_ids(tok):
    """Pinned rule: candidates are 'A'/' A' and 'B'/' B'; each must encode
    to a single token; readout compares max logit over each side's set."""
    sets = {}
    for letter in ("A", "B"):
        ids = []
        for variant in (letter, " " + letter):
            enc = tok.encode(variant, add_special_tokens=False)
            if len(enc) == 1:
                ids.append(enc[0])
        sets[letter] = sorted(set(ids))
    gate("GR3-letters", all(len(v) >= 1 for v in sets.values()),
         f"letter candidate token ids: {sets} (each side needs >=1 single-token id)")
    return sets


# ------------------------------------------------------------ encoding
def encode_row(row, series_of, tok, proc, sp, condition):
    """Route a manifest row + condition to the right encoding path.
    Returns (inputs dict on CPU, extras dict for the record)."""
    extras = {}
    block = row.get("block", "p1_or_p2")
    x = series_of.get(row["sample_id"])

    if block == "five_number":
        out = proc(text=[row["prompt"]], timeseries=[], padding=True,
                   return_tensors="pt")
        return out, extras

    if block == "pj_control":
        out = manual_encode(row["prompt"],
                            [{"tensor_source": x,
                              "prefix_text": row["prefix_jittered"]}], tok, sp)
        return out, extras

    if condition in ("unperturbed", None):
        pert = x
    elif condition == "sf_all":
        pert = P.sf_all(x, row["sample_id"], BASE_SEED)
    elif condition == "sf_half":
        pert = P.sf_half(x, row["sample_id"], BASE_SEED)
    elif condition == "ex_half":
        pert = P.ex_half(x)
    elif condition == "masking":
        pert, info = P.masking_m1c(x, row["sample_id"], BASE_SEED)
        m0, s0 = P.sp_prefix_numbers(x)
        m1, s1 = P.sp_prefix_numbers(pert)
        extras["prefix_offset_drifted"] = (f"{-m0:.4f}" != f"{-m1:.4f}")
        extras["prefix_scale_drifted"] = (f"{s0:.4f}" != f"{s1:.4f}")
    elif condition == "sf_within_patch":
        pert = P.sf_within_patch(x, row["sample_id"], BASE_SEED)
    elif condition == "sf_across_patch":
        pert = P.sf_across_patch(x, row["sample_id"], BASE_SEED)
    elif condition == "resample":
        pert = P.resample(x, row["sample_id"], BASE_SEED)
    elif condition == "gaussian":
        pert, _ = P.gaussian_matched(x, row["sample_id"], BASE_SEED)
    elif condition == "cond_A":
        g, _ = P.gaussian_matched(x, row["sample_id"], BASE_SEED)
        out = manual_encode(row["prompt"],
                            [{"tensor_source": g,
                              "prefix_text": prefix_text_of(x, sp)}], tok, sp)
        return out, extras
    elif condition == "cond_B":
        donor = series_of[row["donor_sample_id"]]
        out = manual_encode(row["prompt"],
                            [{"tensor_source": x,
                              "prefix_text": prefix_text_of(donor, sp)}], tok, sp)
        return out, extras
    elif condition == "cond_C":
        g, _ = P.gaussian_matched(x, row["sample_id"], BASE_SEED)
        out = manual_encode(row["prompt"], [{"tensor_source": g}], tok, sp)
        return out, extras
    else:
        raise ValueError(f"unknown condition {condition}")

    out = proc(text=[row["prompt"]], timeseries=[pert], padding=True,
               return_tensors="pt")
    return out, extras


@torch.no_grad()
def ask(model, inputs, letter_ids):
    dev = {k: (v.to(0) if torch.is_tensor(v) else v) for k, v in inputs.items()
           if k in ("input_ids", "attention_mask", "timeseries")}
    if dev.get("timeseries") is None:
        dev.pop("timeseries", None)
    out = model(**dev)
    logits = out.logits[0, -1, :].float()
    la = max(float(logits[i]) for i in letter_ids["A"])
    lb = max(float(logits[i]) for i in letter_ids["B"])
    return ("A" if la >= lb else "B"), la, lb, out.logits.shape[1], \
        dev["input_ids"].shape[1]


@torch.no_grad()
def ask_generate(model, tok, inputs):
    dev = {k: (v.to(0) if torch.is_tensor(v) else v) for k, v in inputs.items()
           if k in ("input_ids", "attention_mask", "timeseries")}
    if dev.get("timeseries") is None:
        dev.pop("timeseries", None)
    out = model.generate(**dev, max_new_tokens=4, do_sample=False)
    text = tok.decode(out[0, dev["input_ids"].shape[1]:], skip_special_tokens=True)
    for ch in text:
        if ch in ("A", "B"):
            return ch, text
    return None, text


# ------------------------------------------------------------- driving
def load_manifest(path, expected_n, sha_by_block):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    gate(f"GR2-{Path(path).stem}", len(rows) == expected_n,
         f"rows = {len(rows)} (expected {expected_n})")
    bad_sha = [r["mcq_id"] for r in rows
               if r["template_sha"] != sha_by_block.get(r.get("block", "mcq"), MCQ_SHA)]
    gate(f"GR1-{Path(path).stem}", not bad_sha,
         f"template-sha mismatches: {len(bad_sha)} (expected 0)")
    return rows


def jobs_for(probe, rows):
    for r in rows:
        for cond in (r.get("conditions") or [None]):
            yield r, cond


def run_probe(probe, rows, series_of, tok, proc, model, sp, letter_ids,
              out_path, limit=None):
    done = set()
    if out_path.exists():
        for l in open(out_path, encoding="utf-8"):
            rec = json.loads(l)
            done.add((rec["mcq_id"], rec["condition"]))
        print(f"[resume] {len(done)} responses already on disk; skipping those")
    f = open(out_path, "a", encoding="utf-8")
    t0, n = time.time(), 0
    for r, cond in jobs_for(probe, rows):
        key = (r["mcq_id"], str(cond))
        if key in done:
            continue
        inputs, extras = encode_row(r, series_of, tok, proc, sp, cond)
        choice, la, lb, out_len, in_len = ask(model, inputs, letter_ids)
        rec = {"mcq_id": r["mcq_id"], "condition": str(cond),
               "choice": choice, "correct": choice == r["correct_letter"],
               "logit_A": la, "logit_B": lb,
               "correct_letter": r["correct_letter"],
               "splice_delta": out_len - in_len, **extras}
        f.write(json.dumps(rec) + "\n")
        n += 1
        if n % 500 == 0:
            rate = n / (time.time() - t0)
            print(f"  [{n}] {rate:.1f} q/s", flush=True)
            f.flush()
        if limit and n >= limit:
            break
    f.close()
    dt = time.time() - t0
    print(f"[probe {probe}] wrote {n} new responses in {dt/60:.1f} min "
          f"-> {out_path}")


def smoke(rows_by_probe, series_of, tok, proc, model, sp, letter_ids):
    print("=== SMOKE STAGE ===")
    rng = np.random.Generator(np.random.PCG64(BASE_SEED))
    timings = []
    for probe, rows in rows_by_probe.items():
        blocks = defaultdict(list)
        for r in rows:
            blocks[r.get("block", "main")].append(r)
        for bname, brows in blocks.items():
            picks = [brows[i] for i in rng.choice(len(brows),
                     size=min(SMOKE_N, len(brows)), replace=False)]
            deltas = Counter()
            answered = 0
            for r in picks:
                cond = (r.get("conditions") or [None])[0]
                t0 = time.time()
                inputs, _ = encode_row(r, series_of, tok, proc, sp, cond)
                choice, la, lb, out_len, in_len = ask(model, inputs, letter_ids)
                timings.append(time.time() - t0)
                ds = r.get("dataset", "sushi" if "sushi" in r["sample_id"] else "truce")
                deltas[(ds, out_len - in_len)] += 1
                answered += 1
                # determinism: same question twice, pre-named fallback rule
                choice2, la2, lb2, _, _ = ask(model, inputs, letter_ids)
                if choice2 != choice:
                    gate("GR5-determinism", False,
                         f"choice flipped on repeat: {r['mcq_id']}")
            print(f"[smoke] probe {probe} block {bname}: {answered} answered; "
                  f"splice deltas {dict(deltas)}")
            if bname not in ("five_number",):
                ok = all((d == 126 if ds == "sushi" else d == -1)
                         for (ds, d) in deltas)
                gate(f"GR4-splice-{probe}-{bname}", ok,
                     f"splice arithmetic (SUSHI +126, TRUCE -1): {dict(deltas)}")
    med = float(np.median(timings))
    total = 31356
    proj_h = med * total / 3600
    print(f"[smoke] median forward time {med*1000:.0f} ms; projected full arm "
          f"~{proj_h:.1f} h for {total} questions (registered band: 1-3 h)")
    gate("GR5-determinism", "GR5-determinism" not in FAILURES,
         "repeat-question choices all stable (letter-level; logit bitwise "
         "not required — GPU float reality, rule pre-named)")


def manual_path_recheck(rows, series_of, tok, proc, sp):
    rng = np.random.Generator(np.random.PCG64(BASE_SEED + 1))
    picks = [rows[i] for i in rng.choice(len(rows), size=50, replace=False)]
    bad = 0
    for r in picks:
        x = series_of[r["sample_id"]]
        stock = proc(text=[r["prompt"]], timeseries=[x], padding=True,
                     return_tensors="pt")
        man = manual_encode(r["prompt"], [{"tensor_source": x}], tok, sp)
        if (stock["input_ids"][0].tolist() != man["input_ids"][0].tolist()
                or not torch.equal(stock["timeseries"], man["timeseries"])):
            bad += 1
    gate("GR6-manualpath", bad == 0,
         f"GPU-env manual-path equivalence: {bad}/50 mismatches (expected 0)")


def agreement_check(probe, rows, series_of, tok, proc, model, sp, letter_ids):
    rng = np.random.Generator(np.random.PCG64(BASE_SEED + probe))
    picks = [rows[i] for i in rng.choice(len(rows),
             size=min(AGREEMENT_N, len(rows)), replace=False)]
    agree, unparse = 0, 0
    for r in picks:
        cond = (r.get("conditions") or [None])[0]
        inputs, _ = encode_row(r, series_of, tok, proc, sp, cond)
        c_logit, *_ = ask(model, inputs, letter_ids)
        c_gen, raw = ask_generate(model, tok, inputs)
        if c_gen is None:
            unparse += 1
        elif c_gen == c_logit:
            agree += 1
    frac = agree / max(1, len(picks) - unparse)
    gate(f"GR8-agreement-p{probe}", frac >= AGREEMENT_THRESHOLD,
         f"logit-vs-greedy agreement {agree}/{len(picks)-unparse} = {frac:.3f} "
         f"(threshold {AGREEMENT_THRESHOLD}; unparseable {unparse}; pre-named "
         f"fallback: generation becomes primary for this probe)", hard=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--p1-manifest", required=True)
    ap.add_argument("--p2-manifest", required=True)
    ap.add_argument("--p3-manifest", required=True)
    ap.add_argument("--out-dir", default="results/experiments")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--probe", choices=["1", "2", "3"])
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    print("=== ChatTS GPU runner ===")
    print(f"torch {torch.__version__}; cuda available: {torch.cuda.is_available()}")
    gate("GR-cuda", torch.cuda.is_available(), "CUDA device present")
    hard_stop_if_failed("environment")

    tok, proc, model, sp = load_model(args.checkpoint)
    letter_ids = letter_token_ids(tok)
    hard_stop_if_failed("startup")

    series_of = {}
    with open(args.pairs, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["sample_id"] not in series_of:
                series_of[r["sample_id"]] = np.asarray(r["series"], dtype=np.float64)

    p1 = load_manifest(args.p1_manifest, 11080, {"mcq": MCQ_SHA})
    p2 = load_manifest(args.p2_manifest, 1756, {"mcq": MCQ_SHA})
    p3 = load_manifest(args.p3_manifest, 3912,
                       {"base": MCQ_SHA, "five_number": FN_SHA,
                        "pj_control": MCQ_SHA})
    hard_stop_if_failed("manifests")

    if args.smoke:
        manual_path_recheck(p2, series_of, tok, proc, sp)
        smoke({1: p1, 2: p2, 3: p3}, series_of, tok, proc, model, sp, letter_ids)
        hard_stop_if_failed("smoke")
        print("SMOKE GREEN — full runs may start.")
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.probe == "1":
        run_probe(1, p1, series_of, tok, proc, model, sp, letter_ids,
                  out_dir / "chatts_probe1_responses.jsonl", args.limit)
        agreement_check(1, p1, series_of, tok, proc, model, sp, letter_ids)
    elif args.probe == "2":
        run_probe(2, p2, series_of, tok, proc, model, sp, letter_ids,
                  out_dir / "chatts_probe2_responses.jsonl", args.limit)
        agreement_check(2, p2, series_of, tok, proc, model, sp, letter_ids)
    elif args.probe == "3":
        run_probe(3, p3, series_of, tok, proc, model, sp, letter_ids,
                  out_dir / "chatts_probe3_responses.jsonl", args.limit)
        agreement_check(3, p3, series_of, tok, proc, model, sp, letter_ids)
    else:
        print("Specify --smoke or --probe {1,2,3}.")
        sys.exit(2)
    print("Done. Verify gate lines above before leaving the pod.")


if __name__ == "__main__":
    main()

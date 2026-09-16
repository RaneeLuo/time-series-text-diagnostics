"""
run_probe1.py (text-embedding-3-large) — the floor baseline for Probe 1.

This model has never seen a time series. It is included precisely for that
reason: it establishes what a general-purpose text embedder achieves when the
series is handed to it as text, which is the floor against which CLaSP's
contrastive training must be judged.

THE SERIALISATION IS THE EXPERIMENT'S WEAKEST POINT, SO IT IS EXPLICIT.
If the way we write the series as text destroys spike information, then a
failure on the fluctuation component would tell us about our encoding rather
than about the model. Two rules follow:

  * every one of the 2,048 points is kept by default. No decimation. Spikes,
    steps and noise survive into the text.
  * the series is z-normalised first -- identical to what CLaSP receives -- so
    nothing is available here that was hidden from the other model.

Values are quantised to integers on a fixed scale (z x 10, clipped to +/-99)
purely to fit the 8,191-token input limit; that quantisation preserves relative
shape and large excursions, which are what the probe tests. Run --dry-run first:
it prints the realised token counts and an example serialisation so the encoding
can be inspected and reported in the thesis before any money is spent.

Output format is identical to the CLaSP runner, so analyze_probe1_stats.py
applies unchanged. Note that this model has no random seed: there is a single
condition, so cross-seed replication does not apply and the statistics will
report one row per component.

Setup:
    python -m pip install openai tiktoken
    $env:OPENAI_API_KEY = "sk-..."        # PowerShell

Usage:
    python -m models.openai_embed.run_probe1 --dry-run
    python -m models.openai_embed.run_probe1 --yes

Writes: results/experiments/probe1_openai_per_item.jsonl
        results/experiments/probe1_openai_summary.json
        .cache/openai_embeddings.jsonl        (never re-pay for the same text)
"""

from __future__ import annotations
import argparse
import hashlib
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

PAIRS = Path("data/processed/pairs.jsonl")
ITEMS = Path("data/processed/probe1_items.jsonl")
OUT_ITEMS = Path("results/experiments/probe1_openai_per_item.jsonl")
OUT_SUMMARY = Path("results/experiments/probe1_openai_summary.json")
CACHE = Path(".cache/openai_embeddings.jsonl")

MODEL = "text-embedding-3-large"
MAX_TOKENS = 8191
PRICE_PER_1M = 0.13          # USD, verify against current pricing before relying on it


# ---------------------------------------------------------------- serialisation

def znorm(x):
    sd = x.std()
    return np.zeros_like(x) if sd < 1e-8 else (x - x.mean()) / sd


def serialize(series, scale=10, clip=99):
    """z-normalise, quantise to integers, comma-join. All points retained."""
    x = znorm(np.asarray(series, dtype=np.float64))
    q = np.clip(np.rint(x * scale), -clip, clip).astype(int)
    return ",".join(str(v) for v in q)


def count_tokens(texts):
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return [len(enc.encode(t)) for t in texts]
    except Exception:
        return [max(1, len(t) // 4) for t in texts]      # crude fallback


# ---------------------------------------------------------------- cache

def load_cache():
    if not CACHE.exists():
        return {}
    out = {}
    with open(CACHE, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out[r["key"]] = np.asarray(r["vec"], dtype=np.float32)
    return out


def append_cache(key, vec):
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE, "a", encoding="utf-8") as f:
        f.write(json.dumps({"key": key, "vec": [float(v) for v in vec]}) + "\n")


def key_of(text):
    return hashlib.sha256((MODEL + "||" + text).encode("utf-8")).hexdigest()[:32]


# ---------------------------------------------------------------- embedding

def embed_all(texts, cache, batch_size, label):
    """Embed texts not already cached; return {text: vector}."""
    from openai import OpenAI
    client = OpenAI()

    todo = [t for t in texts if key_of(t) not in cache]
    print(f"  {label}: {len(texts)} texts, {len(texts) - len(todo)} cached, "
          f"{len(todo)} to embed")

    for i in range(0, len(todo), batch_size):
        chunk = todo[i:i + batch_size]
        for attempt in range(5):
            try:
                resp = client.embeddings.create(model=MODEL, input=chunk)
                break
            except Exception as e:
                wait = 2 ** attempt
                print(f"    retry {attempt + 1}/5 in {wait}s ({type(e).__name__})")
                time.sleep(wait)
        else:
            raise SystemExit("embedding failed after 5 attempts")
        for t, d in zip(chunk, resp.data):
            v = np.asarray(d.embedding, dtype=np.float32)
            v /= (np.linalg.norm(v) + 1e-12)
            cache[key_of(t)] = v
            append_cache(key_of(t), v)
        print(f"    {min(i + batch_size, len(todo))}/{len(todo)}", end="\r")
    print()
    return {t: cache[key_of(t)] for t in texts}


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="serialise and report token counts and cost; no API calls")
    ap.add_argument("--yes", action="store_true", help="proceed with API calls")
    ap.add_argument("--scale", type=int, default=10)
    ap.add_argument("--batch-signals", type=int, default=8)
    ap.add_argument("--batch-captions", type=int, default=128)
    args = ap.parse_args()

    items = [json.loads(l) for l in open(ITEMS, encoding="utf-8")]
    sample_ids = {it["sample_id"] for it in items}
    captions = sorted({it["caption_correct"] for it in items}
                      | {it["caption_distractor"] for it in items})

    series = {}
    with open(PAIRS, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["sample_id"] in sample_ids and r["sample_id"] not in series:
                series[r["sample_id"]] = r["series"]
    missing = sample_ids - set(series)
    if missing:
        raise SystemExit(f"{len(missing)} sample_ids not found")

    sig_ids = sorted(series)
    sig_text = {s: serialize(series[s], args.scale) for s in sig_ids}

    # ---------------- token report
    st = count_tokens([sig_text[s] for s in sig_ids])
    ct = count_tokens(captions)
    total = sum(st) + sum(ct)
    over = [s for s, n in zip(sig_ids, st) if n > MAX_TOKENS]

    print(f"items: {len(items)}   signals: {len(sig_ids)}   captions: {len(captions)}")
    print(f"signal tokens : min {min(st)}, median {sorted(st)[len(st)//2]}, max {max(st)}"
          f"   (limit {MAX_TOKENS})")
    print(f"caption tokens: median {sorted(ct)[len(ct)//2]}, max {max(ct)}")
    print(f"total tokens  : {total:,}   estimated cost ${total / 1e6 * PRICE_PER_1M:.2f}")
    if over:
        print(f"\n!! {len(over)} signals exceed the input limit. Raise --scale to "
              f"shrink numbers, or the serialisation must be revised. Do not "
              f"silently truncate: that would remove exactly the local detail "
              f"the fluctuation component tests.")

    ex = sig_ids[0]
    print(f"\nexample serialisation [{ex}], first 220 chars:")
    print("  " + sig_text[ex][:220] + " ...")

    if args.dry_run or not args.yes:
        print("\ndry run — no API calls made. Re-run with --yes to embed.")
        return
    if over:
        raise SystemExit("refusing to run while signals exceed the token limit")

    # ---------------- embed
    cache = load_cache()
    print(f"\ncache: {len(cache)} vectors")
    sig_emb = embed_all([sig_text[s] for s in sig_ids], cache,
                        args.batch_signals, "signals")
    cap_emb = embed_all(captions, cache, args.batch_captions, "captions")

    # ---------------- score
    records = []
    for it in items:
        v = sig_emb[sig_text[it["sample_id"]]]
        sc = float(v @ cap_emb[it["caption_correct"]])
        sd = float(v @ cap_emb[it["caption_distractor"]])
        records.append({
            "seed": "api", "item_id": it["item_id"],
            "pair_key": it["item_id"].rsplit("|", 1)[0],
            "component": it["component"], "condition": it["condition"],
            "swap_from": it["swap_from"], "swap_to": it["swap_to"],
            "sample_id": it["sample_id"],
            "sim_correct": sc, "sim_distractor": sd,
            "margin": sc - sd, "correct": bool(sc > sd),
        })

    # ---------------- summary
    agg = defaultdict(list)
    for r in records:
        agg[(r["component"], r["condition"])].append(r)
    comps = sorted({r["component"] for r in records})

    print("\n" + "=" * 74)
    print(f"PROBE 1 — {MODEL} forced-choice accuracy (chance = 0.500)")
    print("=" * 74)
    print(f"{'component':<26}{'random':>10}{'swap':>10}{'gap':>9}{'margin(swap)':>14}")
    print("-" * 74)
    summary = {}
    for c in comps:
        rnd, swp = agg[(c, "random")], agg[(c, "swap")]
        a_r = float(np.mean([x["correct"] for x in rnd]))
        a_s = float(np.mean([x["correct"] for x in swp]))
        m_s = float(np.mean([x["margin"] for x in swp]))
        print(f"{c:<26}{a_r:>10.3f}{a_s:>10.3f}{a_r - a_s:>9.3f}{m_s:>14.4f}")
        summary[c] = {"n_random": len(rnd), "n_swap": len(swp),
                      "acc_random": a_r, "acc_swap": a_s, "gap": a_r - a_s,
                      "mean_margin_swap": m_s}

    OUT_ITEMS.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_ITEMS, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        json.dump({"model": MODEL, "serialisation":
                   {"z_normalised": True, "scale": args.scale, "clip": 99,
                    "points_retained": "all (no decimation)",
                    "median_signal_tokens": sorted(st)[len(st) // 2]},
                   "n_items": len(items), "per_component": summary}, f, indent=2)
    print(f"\nsaved -> {OUT_ITEMS}\nsaved -> {OUT_SUMMARY}")
    print("\nNote: no random seeds for an API model. Cross-seed replication does")
    print("not apply; the statistics script will report a single row per component.")


if __name__ == "__main__":
    main()

"""
run_probe1_clasp.py — score the Probe-1 forced-choice items with CLaSP.

For every item the model sees one signal and two captions. It "chooses" whichever
caption its embedding is closer to. Chance is 50% by construction, for every
component, which is why the items are binary.

Two numbers per component matter, and only together:

    accuracy(random)  — distractor is a lexically unrelated caption
    accuracy(swap)    — distractor is the SAME caption with one clause replaced

    gap = accuracy(random) - accuracy(swap)

A model that reads compositionally scores high on both and the gap is small.
A model matching on surface statistics scores high on random (the words differ)
and falls toward 50% on swap (the words barely differ). A LARGE GAP IS THE
SHORTCUT SIGNATURE. Neither number is interpretable alone.

The runner also records, per item:
    margin = cos(signal, correct) - cos(signal, distractor)
a continuous measure that is more sensitive than the binary outcome and is what
the bootstrap and Wilcoxon tests will use.

Every item is scored under all three baseline checkpoints (binding project
decision): significance comes from paired tests within a seed, replication from
agreement across seeds.

Run from repo root:
    python -m models.clasp.run_probe1 \
        --checkpoints results/checkpoints/best_baseline_seed42.pt \
                      results/checkpoints/best_baseline_seed43.pt \
                      results/checkpoints/best_baseline_seed44.pt

Writes: results/experiments/probe1_clasp_per_item.jsonl   (input to statistics)
        results/experiments/probe1_clasp_summary.json
"""

from __future__ import annotations
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from dataset import znorm
from models.clasp.model import ClaspModel, ClaspConfig

PAIRS = Path("data/processed/pairs.jsonl")
ITEMS = Path("data/processed/probe1_items.jsonl")
OUT_ITEMS = Path("results/experiments/probe1_clasp_per_item.jsonl")
OUT_SUMMARY = Path("results/experiments/probe1_clasp_summary.json")


def load_items():
    items = []
    with open(ITEMS, encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
    return items


def load_series(sample_ids: set[str]) -> dict[str, np.ndarray]:
    out = {}
    with open(PAIRS, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            sid = r["sample_id"]
            if sid in sample_ids and sid not in out:
                out[sid] = np.asarray(r["series"], dtype=np.float32)
    missing = sample_ids - set(out)
    if missing:
        raise SystemExit(f"{len(missing)} sample_ids not found in {PAIRS}")
    return out


@torch.no_grad()
def embed_signals(model, series: dict[str, np.ndarray], device, max_tokens=16384):
    """Length-grouped batching (same guard that fixed the earlier OOM)."""
    ids = sorted(series)
    by_len = defaultdict(list)
    for sid in ids:
        by_len[len(series[sid])].append(sid)
    out = {}
    for L, group in sorted(by_len.items()):
        B = max(1, min(64, max_tokens // L))
        for i in range(0, len(group), B):
            chunk = group[i:i + B]
            x = torch.zeros(len(chunk), L)
            m = torch.ones(len(chunk), L, dtype=torch.bool)
            for j, sid in enumerate(chunk):
                x[j] = torch.from_numpy(znorm(series[sid]).copy())
            z = model.encode_series(x.to(device), m.to(device)).cpu()
            for j, sid in enumerate(chunk):
                out[sid] = z[j]
    return out


@torch.no_grad()
def embed_captions(model, captions: list[str], device, batch=64):
    out = {}
    for i in range(0, len(captions), batch):
        chunk = captions[i:i + batch]
        z = model.encode_text(chunk, device=device).cpu()
        for c, v in zip(chunk, z):
            out[c] = v
    return out


def seed_of(path: str) -> str:
    m = re.search(r"seed(\d+)", Path(path).stem)
    return m.group(1) if m else Path(path).stem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--out-items", default=str(OUT_ITEMS))
    ap.add_argument("--out-summary", default=str(OUT_SUMMARY))
    args = ap.parse_args()

    items = load_items()
    print(f"items: {len(items)}")
    sample_ids = {it["sample_id"] for it in items}
    captions = sorted({it["caption_correct"] for it in items}
                      | {it["caption_distractor"] for it in items})
    print(f"unique signals: {len(sample_ids)}, unique captions: {len(captions)}")

    series = load_series(sample_ids)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    records = []
    for ckpt in args.checkpoints:
        seed = seed_of(ckpt)
        print(f"\n--- checkpoint seed {seed} ---")
        model = ClaspModel(ClaspConfig())
        state = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(state["model"] if "model" in state else state)
        model.to(device).eval()

        sig_emb = embed_signals(model, series, device)
        txt_emb = embed_captions(model, captions, device)

        for it in items:
            s = sig_emb[it["sample_id"]]
            sc = float(s @ txt_emb[it["caption_correct"]])
            sd = float(s @ txt_emb[it["caption_distractor"]])
            records.append({
                "seed": seed,
                "item_id": it["item_id"],
                "pair_key": it["item_id"].rsplit("|", 1)[0],
                "component": it["component"],
                "condition": it["condition"],
                "swap_from": it["swap_from"],
                "swap_to": it["swap_to"],
                "sample_id": it["sample_id"],
                "sim_correct": sc,
                "sim_distractor": sd,
                "margin": sc - sd,
                "correct": bool(sc > sd),
            })

    # ---------------------------------------------------------------- summary
    agg = defaultdict(list)
    for r in records:
        agg[(r["seed"], r["component"], r["condition"])].append(r)

    seeds = sorted({r["seed"] for r in records})
    comps = sorted({r["component"] for r in records})

    print("\n" + "=" * 78)
    print("PROBE 1 — CLaSP forced-choice accuracy (chance = 0.500)")
    print("=" * 78)
    print(f"{'component':<26}{'seed':>6}{'random':>10}{'swap':>10}{'gap':>9}{'margin(swap)':>14}")
    print("-" * 78)

    summary = {}
    for comp in comps:
        for seed in seeds:
            rnd = agg[(seed, comp, "random")]
            swp = agg[(seed, comp, "swap")]
            if not rnd or not swp:
                continue
            a_r = float(np.mean([x["correct"] for x in rnd]))
            a_s = float(np.mean([x["correct"] for x in swp]))
            m_s = float(np.mean([x["margin"] for x in swp]))
            print(f"{comp:<26}{seed:>6}{a_r:>10.3f}{a_s:>10.3f}"
                  f"{a_r - a_s:>9.3f}{m_s:>14.4f}")
            summary[f"{comp}|seed{seed}"] = {
                "n_random": len(rnd), "n_swap": len(swp),
                "acc_random": a_r, "acc_swap": a_s, "gap": a_r - a_s,
                "mean_margin_random": float(np.mean([x["margin"] for x in rnd])),
                "mean_margin_swap": m_s,
            }
        print("-" * 78)

    # across-seed means
    print("\nACROSS SEEDS (mean +/- sd)")
    print(f"{'component':<26}{'random':>16}{'swap':>16}{'gap':>16}")
    print("-" * 74)
    across = {}
    for comp in comps:
        ar = [summary[f"{comp}|seed{s}"]["acc_random"] for s in seeds
              if f"{comp}|seed{s}" in summary]
        as_ = [summary[f"{comp}|seed{s}"]["acc_swap"] for s in seeds
               if f"{comp}|seed{s}" in summary]
        gp = [summary[f"{comp}|seed{s}"]["gap"] for s in seeds
              if f"{comp}|seed{s}" in summary]
        f = lambda v: f"{np.mean(v):.3f} +/- {np.std(v, ddof=1) if len(v) > 1 else 0:.3f}"
        print(f"{comp:<26}{f(ar):>16}{f(as_):>16}{f(gp):>16}")
        across[comp] = {"acc_random_mean": float(np.mean(ar)),
                        "acc_swap_mean": float(np.mean(as_)),
                        "gap_mean": float(np.mean(gp)),
                        "gap_sd": float(np.std(gp, ddof=1)) if len(gp) > 1 else 0.0}

    Path(args.out_items).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_items, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    with open(args.out_summary, "w", encoding="utf-8") as f:
        json.dump({"checkpoints": args.checkpoints, "n_items": len(items),
                   "per_seed": summary, "across_seeds": across}, f, indent=2)
    print(f"\nsaved -> {args.out_items}")
    print(f"saved -> {args.out_summary}")

    print("\n" + "=" * 78)
    print("HOW TO READ THIS")
    print("=" * 78)
    print("high random + high swap + small gap -> component is genuinely encoded")
    print("high random + swap near 0.500        -> shortcut: the model is matching")
    print("                                        surface text, not this component")
    print("both near 0.500                      -> model cannot do the task at all;")
    print("                                        that component's result is void")
    print("\nNothing is significant yet — these are point estimates. The per-item")
    print("file feeds the paired tests (bootstrap over SIGNALS, not items).")


if __name__ == "__main__":
    main()

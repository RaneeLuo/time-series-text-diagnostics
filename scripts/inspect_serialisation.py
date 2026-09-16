"""
inspect_serialisation.py — does the text encoding preserve what Probe 1 tests?

The floor-baseline model reads the series as text. If that encoding flattens
spikes, noise or steps, then a failure on the fluctuation component would be an
artefact of our encoding rather than a property of the model -- an uninterpretable
result. This script checks the encoding before any money is spent, by showing one
signal per fluctuation class and asking three questions of each:

  * what range do the quantised values span?
  * how many values hit the clip boundary (information destroyed)?
  * is the largest excursion actually VISIBLE as a large number in the text?

Run from repo root:
    python scripts/inspect_serialisation.py
"""

from __future__ import annotations
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

PAIRS = Path("data/processed/pairs.jsonl")
SCALE = 10
CLIP = 99


def znorm(x):
    sd = x.std()
    return np.zeros_like(x) if sd < 1e-8 else (x - x.mean()) / sd


def quantise(series, scale=SCALE, clip=CLIP):
    x = znorm(np.asarray(series, dtype=np.float64))
    raw = np.rint(x * scale)
    return np.clip(raw, -clip, clip).astype(int), raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=int, default=SCALE)
    ap.add_argument("--clip", type=int, default=CLIP)
    args = ap.parse_args()
    scale, clip = args.scale, args.clip

    by_fluct = defaultdict(list)
    with open(PAIRS, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["dataset"] != "sushi":
                continue
            fl = r["class_label"].split(";")[0].strip()
            by_fluct[fl].append(r)

    print(f"scale={scale}, clip=+/-{clip}, all 2048 points retained\n")
    print(f"{'fluctuation':<28}{'q range':>14}{'clipped':>10}{'|q|>20':>9}{'verdict':>12}")
    print("-" * 73)

    examples = {}
    for fl in sorted(by_fluct):
        # pick the signal with the largest excursion, i.e. the best case for
        # seeing whether extremes survive
        best, best_amp, best_q, best_raw = None, -1, None, None
        for r in by_fluct[fl][:40]:
            q, raw = quantise(r["series"], scale, clip)
            amp = float(np.abs(raw).max())
            if amp > best_amp:
                best, best_amp, best_q, best_raw = r, amp, q, raw
        q, raw = best_q, best_raw
        n_clip = int((np.abs(raw) > clip).sum())
        n_big = int((np.abs(q) > 20).sum())
        verdict = ("FLAT" if np.abs(q).max() == 0
                   else "clipped!" if n_clip > 0
                   else "visible" if n_big > 0
                   else "subtle")
        print(f"{fl:<28}{f'[{q.min()},{q.max()}]':>14}{n_clip:>10}{n_big:>9}{verdict:>12}")
        examples[fl] = (best, q, raw)

    print("\n" + "=" * 73)
    print("TEXT AROUND THE LARGEST EXCURSION (is it visible to a reader?)")
    print("=" * 73)
    for fl, (rec, q, raw) in examples.items():
        i = int(np.argmax(np.abs(q)))
        lo, hi = max(0, i - 8), min(len(q), i + 9)
        window = ",".join(str(v) for v in q[lo:hi])
        print(f"\n[{fl}]  class '{rec['class_label']}'  peak at index {i}")
        print(f"   ...{window}...")

    print("\n" + "=" * 73)
    print("HOW TO READ THIS")
    print("=" * 73)
    print("visible   large numbers appear where the signal spikes -> encoding is fine")
    print("clipped!  values hit the clip boundary -> magnitude information destroyed;")
    print("          lower --scale (e.g. 5) so extremes fit inside the range")
    print("subtle    no value exceeds 20 -> the class differs from 'clean' only in")
    print("          fine detail; check the class is genuinely distinguishable")
    print("FLAT      constant signal; expected only for 'clean' with a constant shape")


if __name__ == "__main__":
    main()

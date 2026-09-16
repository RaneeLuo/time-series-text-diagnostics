#!/usr/bin/env python3
"""
verify_n5_investigation_v2.py — same analysis as verify_n5_investigation.py,
plus a committed JSON record.

Why v2 exists (2026-09-07, writing session 6): the thesis section 4.2.2.2
quotes the place-name-only slice (0.900 in every seed), but no committed
results/ file carries it — v1 prints the numbers and the local terminal was
the only record. This script reproduces v1's analyses A/B/C *unchanged*
(the frame() and content_words() rules are IMPORTED from v1, not
re-implemented — error #18 in SESSION_HANDOFF §2b) and writes the result to
results/analysis/n5_investigation.json so every number in 4.2.2.2 has a
file + key.

Registered expectations (from v1's constants; a mismatch is REPORTED, never
silently accepted):
  items       : 400 N5 swap items                        [HARD]
  joined      : 1200 result records (400 x 3 seeds)      [HARD]
  A  fidelity : 0/400 caption diff != recorded replacement
  B  vocab    : 336/400 swaps change non-place vocabulary
  B  frame    : 40/400 swaps are frame-identical (place-name-only)
  C  identical: acc 0.900 / 0.900 / 0.900 (seeds 13/14/15), pooled 0.900,
                mean margin +0.2004
  C  changed  : acc 0.928 / 0.939 / 0.919, pooled 0.929, mean margin +0.2506
  closure     : 40*acc_identical + 360*acc_changed == 400*acc_overall per
                seed, where acc_overall is read from the committed
                trace_narrative_summary.json (0.925 / 0.935 / 0.9175)

Run from the thesis repo root (PowerShell):
  python models\\trace\\verify_n5_investigation_v2.py
Optional flags: --items, --results, --summary, --out (defaults = v1 paths).
"""

import argparse
import importlib.util
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
V1 = HERE / "verify_n5_investigation.py"

EXPECTED = {
    "n_items": 400,
    "n_joined": 1200,
    "A_mismatch": 0,
    "B_vocab_changed": 336,
    "B_frame_identical": 40,
    "C": {
        "identical": {"by_seed": {"mask13": 0.900, "mask14": 0.900, "mask15": 0.900},
                      "pooled": 0.900, "mean_margin": 0.2004},
        "changed":   {"by_seed": {"mask13": 0.928, "mask14": 0.939, "mask15": 0.919},
                      "pooled": 0.929, "mean_margin": 0.2506},
    },
}


def load_v1_rules():
    """Import frame() and content_words() from v1 by path (no re-implementation)."""
    if not V1.exists():
        sys.exit(f"HARD STOP: v1 script not found at {V1}")
    spec = importlib.util.spec_from_file_location("verify_n5_v1", V1)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.frame, mod.content_words, mod.ITEMS, mod.RESULTS


def read_jsonl(p: Path):
    with p.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def flag(ok: bool) -> str:
    return "OK  " if ok else "MISS"


def main():
    frame, content_words, V1_ITEMS, V1_RESULTS = load_v1_rules()

    ap = argparse.ArgumentParser()
    ap.add_argument("--items", type=Path, default=V1_ITEMS)
    ap.add_argument("--results", type=Path, default=V1_RESULTS)
    ap.add_argument("--summary", type=Path,
                    default=Path("results/experiments/trace_narrative_summary.json"))
    ap.add_argument("--out", type=Path,
                    default=Path("results/analysis/n5_investigation.json"))
    args = ap.parse_args()

    print("verify_n5_investigation_v2")
    print(f"  items   : {args.items}")
    print(f"  results : {args.results}")
    print(f"  summary : {args.summary}")
    print(f"  out     : {args.out}")
    print(f"  rules   : frame()/content_words() imported from {V1.name}")
    for p in (args.items, args.results, args.summary):
        if not p.exists():
            sys.exit(f"HARD STOP: missing input {p}")

    items = read_jsonl(args.items)
    n5 = {it["item_id"]: it for it in items
          if it["component"] == "N5" and it["condition"] == "swap"}
    print(f"\nN5 swap items: {len(n5)}   (expected {EXPECTED['n_items']})")
    if len(n5) != EXPECTED["n_items"]:
        sys.exit("HARD STOP: N5 swap item count != 400")

    # ---- A. replacement-record fidelity (verbatim v1 logic) ------------
    mismatch = 0
    for it in n5.values():
        a, b = it["caption_correct"], it["caption_distractor"]
        fr, to = it["clause_replaced_from"], it["clause_replaced_to"]
        if not (fr in a and to in b and a.replace(fr, to, 1) == b):
            mismatch += 1
    print(f"[A] caption diff != recorded replacement: {mismatch}/400   "
          f"(expected {EXPECTED['A_mismatch']}/400)  {flag(mismatch == EXPECTED['A_mismatch'])}")

    # ---- B. what changes (verbatim v1 logic) ----------------------------
    vocab_changed = sum(
        1 for it in n5.values()
        if content_words(it["clause_replaced_from"])
        != content_words(it["clause_replaced_to"]))
    same_frame_n = 0
    for it in n5.values():
        it["_same_frame"] = (frame(it["clause_replaced_from"])
                             == frame(it["clause_replaced_to"]))
        same_frame_n += it["_same_frame"]
    print(f"[B] swaps changing non-place vocabulary: {vocab_changed}/400   "
          f"(expected {EXPECTED['B_vocab_changed']}/400)  {flag(vocab_changed == EXPECTED['B_vocab_changed'])}")
    print(f"[B] frame-identical (place-name-only) swaps: {same_frame_n}/400   "
          f"(expected {EXPECTED['B_frame_identical']}/400)  {flag(same_frame_n == EXPECTED['B_frame_identical'])}")

    # ---- C. decisive slice (verbatim v1 logic + record) -----------------
    recs = [r for r in read_jsonl(args.results) if r["item_id"] in n5]
    print(f"[C] joined N5 swap records: {len(recs)}   (expected {EXPECTED['n_joined']})")
    if len(recs) != EXPECTED["n_joined"]:
        sys.exit("HARD STOP: joined record count != 1200")

    record = {"script": Path(__file__).name,
              "items_file": str(args.items), "results_file": str(args.results),
              "rules_imported_from": V1.name,
              "n_items": len(n5), "n_joined": len(recs),
              "A_mismatch": mismatch,
              "B_vocab_changed": vocab_changed,
              "B_frame_identical": same_frame_n,
              "slices": {}, "expected": EXPECTED}
    for keep, label in ((True, "identical"), (False, "changed")):
        sel = [r for r in recs if n5[r["item_id"]]["_same_frame"] == keep]
        by_seed = defaultdict(list)
        for r in sel:
            by_seed[r["seed"]].append(r["correct"])
        accs = {s: sum(v) / len(v) for s, v in sorted(by_seed.items())}
        pooled = sum(r["correct"] for r in sel) / len(sel)
        marg = statistics.mean(r["margin"] for r in sel)
        exp = EXPECTED["C"][label]
        # duration composition of the slice (report only; no expectation registered)
        dur = defaultdict(int)
        for i in {r["item_id"] for r in sel}:
            dur[n5[i]["duration_class"]] += 1
        print(f"\n[C] frame-{label.upper()}: {len(sel)//3} items; duration classes {dict(dur)}")
        for s in sorted(accs):
            print(f"    {s}: acc {accs[s]:.3f} (expected {exp['by_seed'][s]:.3f})  "
                  f"{flag(round(accs[s], 3) == exp['by_seed'][s])}")
        print(f"    pooled acc {pooled:.3f} (expected {exp['pooled']:.3f})  "
              f"{flag(round(pooled, 3) == exp['pooled'])}")
        print(f"    mean margin {marg:+.4f} (expected {exp['mean_margin']:+.4f})  "
              f"{flag(round(marg, 4) == exp['mean_margin'])}")
        record["slices"][label] = {
            "n_items": len(sel) // 3, "n_records": len(sel),
            "acc_by_seed": accs, "acc_pooled": pooled, "mean_margin": marg,
            "n_correct_by_seed": {s: int(sum(v)) for s, v in sorted(by_seed.items())},
            "duration_classes": dict(dur),
        }

    # ---- closure against the committed summary ---------------------------
    with args.summary.open(encoding="utf-8") as fh:
        summ = json.load(fh)
    print("\n[closure] 40*acc_id + 360*acc_ch vs 400*acc_overall (committed summary):")
    record["closure"] = {}
    for s in ("mask13", "mask14", "mask15"):
        n_id = record["slices"]["identical"]["n_correct_by_seed"][s]
        n_ch = record["slices"]["changed"]["n_correct_by_seed"][s]
        overall = summ["per_seed_component_summary"][s]["N5"]["acc_swap"]
        lhs = n_id + n_ch
        rhs = round(400 * overall)
        print(f"    {s}: {n_id} + {n_ch} = {lhs} correct; summary 400 x {overall:.4f} = {rhs}  {flag(lhs == rhs)}")
        record["closure"][s] = {"correct_identical": n_id, "correct_changed": n_ch,
                                "summary_acc_swap": overall, "closes": lhs == rhs}

    ids = sorted(i for i, it in n5.items() if it["_same_frame"])
    record["frame_identical_item_ids"] = ids
    print(f"\n[record] {len(ids)} frame-identical item_ids written to JSON")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2, ensure_ascii=False)
    print(f"\nWROTE {args.out}")
    print("Every line marked OK matches v1's registered value; any MISS must be "
          "reported, not explained away.")


if __name__ == "__main__":
    main()

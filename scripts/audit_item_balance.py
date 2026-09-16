"""
audit_item_balance.py — measure surface confounds in the shared Probe-1 item set.

Written after a gap was found in the generator's own reporting: it measured
caption-length balance for SWAP items and reported it per component, but never
for RANDOM items. The floor-baseline run then showed that a text embedder's
choices correlate with caption length (r ~ +0.13 to +0.17) and that it falls
BELOW chance exactly where correct captions are shorter -- which means the
random condition's length balance is a property of the shared item set that
affects every model's baseline, not a quirk of one model.

This audit reports, for both conditions and every component:
  * mean signed length difference (correct - distractor), in words and characters
  * mean absolute difference
  * how often the correct caption is the longer of the two
  * a "surface-guessable" rate: the accuracy a model would score by always
    choosing the longer caption

That last number is the useful one. If always-pick-longer scores 0.50 for a
component, length carries no information there. If it scores 0.70, a model could
reach 0.70 without reading anything, and any reported accuracy must be judged
against that, not against 0.50.

Run from repo root:
    python scripts/audit_item_balance.py

Writes: results/analysis/probe1_item_balance.json
"""

from __future__ import annotations
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ITEMS = Path("data/processed/probe1_items.jsonl")
OUT = Path("results/analysis/probe1_item_balance.json")


def correlate_length_with_margin(items, results_path):
    """Does a model's preference track caption LENGTH rather than content?

    Reproduces the diagnostic used on the floor baseline: for every scored item,
    correlate (words in correct caption - words in distractor) against the
    model's margin. A positive coefficient means the model systematically favours
    the longer caption, which is a content-blind surface heuristic.
    """
    by_id = {it["item_id"]: it for it in items}
    recs = [json.loads(l) for l in open(results_path, encoding="utf-8")]
    rows = []
    for r in recs:
        it = by_id.get(r["item_id"])
        if it is None:
            continue
        dl = len(it["caption_correct"].split()) - len(it["caption_distractor"].split())
        rows.append((r["component"], r["condition"], dl, r["margin"]))
    if not rows:
        print("  no matching items — check that the results file uses the same item set")
        return {}

    out = {"n_matched": len(rows), "overall": {}, "per_component": {}}
    print(f"\nscored items matched: {len(rows)} of {len(recs)}")
    print("\ncorrelation between (words_correct - words_distractor) and margin:")
    for cond in ("random", "swap"):
        d = [x for x in rows if x[1] == cond]
        if len(d) < 3:
            continue
        r_ = float(np.corrcoef([x[2] for x in d], [x[3] for x in d])[0, 1])
        out["overall"][cond] = r_
        print(f"  {cond:<8} r = {r_:+.3f}   (n={len(d)})")

    print("\nper component:")
    for comp in sorted({x[0] for x in rows}):
        out["per_component"][comp] = {}
        line = f"  {comp:<26}"
        for cond in ("random", "swap"):
            d = [x for x in rows if x[0] == comp and x[1] == cond]
            if len(d) < 3:
                continue
            r_ = float(np.corrcoef([x[2] for x in d], [x[3] for x in d])[0, 1])
            out["per_component"][comp][cond] = r_
            line += f"{cond} {r_:+.3f}   "
        print(line)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=None,
                    help="optional per-item results JSONL from a model runner; "
                         "adds the length-vs-margin correlation diagnostic")
    ap.add_argument("--out", default=None,
                    help="output path; defaults to probe1_item_balance.json, or "
                         "probe1_item_balance_<model>.json when --results is given")
    # ADDED 2026-08-08: the narrative substrate is the first second item set;
    # the hardcoded ITEMS constant silently audited the SUSHI items against
    # TRACE narrative results (0 matched). Default preserves every existing
    # CLaSP/floor command unchanged.
    ap.add_argument("--items", default=str(ITEMS),
                    help="item-set JSONL to audit; MUST be the set the "
                         "--results file was scored on")
    args = ap.parse_args()

    # The item-set audit is model-independent, but the correlation is not, so
    # a per-model default prevents one model's run from silently overwriting
    # another's.
    if args.out:
        out_path = Path(args.out)
    elif args.results:
        tag = Path(args.results).stem
        for pre in ("probe1_", "_per_item"):
            tag = tag.replace(pre, "")
        out_path = OUT.with_name(f"probe1_item_balance_{tag}.json")
    else:
        out_path = OUT

    items = [json.loads(l) for l in open(args.items, encoding="utf-8")]
    print(f"items file: {args.items}")
    print(f"items: {len(items)}\n")

    if args.results:
        res_ids = {json.loads(l)["item_id"]
                   for l in open(args.results, encoding="utf-8")}
        item_ids = {it["item_id"] for it in items}
        n_hit = len(res_ids & item_ids)
        print(f"[check] results item_ids matched to this item set: "
              f"{n_hit}/{len(res_ids)}")
        if n_hit == 0:
            print("[check][WARNING] ZERO overlap — you are auditing a "
                  "DIFFERENT item set than the results were scored on. "
                  "Pass the right --items; every number below would be "
                  "about the wrong items.")

    groups = defaultdict(list)
    for it in items:
        groups[(it["component"], it["condition"])].append(it)

    print("=" * 96)
    print("CAPTION LENGTH BALANCE")
    print("=" * 96)
    print(f"{'component':<26}{'cond':<8}{'d_words':>9}{'|d_words|':>11}"
          f"{'d_chars':>9}{'correct longer':>16}{'pick-longer acc':>17}")
    print("-" * 96)

    report = {}
    for comp in sorted({c for c, _ in groups}):
        for cond in ("random", "swap"):
            g = groups.get((comp, cond), [])
            if not g:
                continue
            dw, dc, longer, guess = [], [], [], []
            for it in g:
                a, b = it["caption_correct"], it["caption_distractor"]
                wa, wb = len(a.split()), len(b.split())
                dw.append(wa - wb)
                dc.append(len(a) - len(b))
                longer.append(wa > wb)
                # a model that always picks the longer caption:
                if wa != wb:
                    guess.append(wa > wb)
                else:
                    guess.append(0.5)          # tie -> coin flip
            m_dw, m_adw = float(np.mean(dw)), float(np.mean(np.abs(dw)))
            m_dc = float(np.mean(dc))
            p_long = float(np.mean(longer))
            p_guess = float(np.mean(guess))
            flag = "  <-- exploitable" if abs(p_guess - 0.5) > 0.10 else ""
            print(f"{comp:<26}{cond:<8}{m_dw:>+9.2f}{m_adw:>11.2f}"
                  f"{m_dc:>+9.1f}{p_long:>16.3f}{p_guess:>17.3f}{flag}")
            report[f"{comp}|{cond}"] = {
                "n": len(g), "mean_signed_word_diff": m_dw,
                "mean_abs_word_diff": m_adw, "mean_signed_char_diff": m_dc,
                "prop_correct_longer": p_long,
                "always_pick_longer_accuracy": p_guess,
            }
        print("-" * 96)

    payload = {"n_items": len(items), "per_component_condition": report}

    if args.results:
        print("\n" + "=" * 96)
        print("LENGTH-VS-MARGIN CORRELATION")
        print("=" * 96)
        print(f"results file: {args.results}")
        payload["length_margin_correlation"] = correlate_length_with_margin(
            items, args.results)
        payload["length_margin_correlation"]["results_file"] = args.results

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nsaved -> {out_path}")

    print("\n" + "=" * 96)
    print("HOW TO USE THIS")
    print("=" * 96)
    print("'pick-longer acc' is what a model scores by ignoring content entirely")
    print("and choosing whichever caption is longer. Read every reported accuracy")
    print("against this number, not against 0.500.")
    print()
    print("A component where pick-longer is far from 0.500 has an exploitable")
    print("surface cue. Note the DIRECTION of the resulting bias: above 0.500 it")
    print("inflates apparent competence; below 0.500 it can push a content-blind")
    print("model beneath chance, which is what produced the floor baseline's")
    print("0.221 on the periodic component.")


if __name__ == "__main__":
    main()

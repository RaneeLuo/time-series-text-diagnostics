"""
generate_probe1_items.py — build the Probe-1 component-swap test set.

Each item is a BINARY FORCED CHOICE over one fixed signal:

    signal  +  { correct caption , distractor caption }

and comes in two conditions that are generated as a matched pair:

    condition="swap"    distractor = the correct caption with ONE clause replaced,
                        so the two captions differ in exactly one semantic
                        component and are otherwise near-identical text.
    condition="random"  distractor = a full caption from a class differing in
                        BOTH slots — lexically very different, semantically
                        unrelated.

Reading the two together is the whole point. A compositionally sensitive model
beats both. A model matching on surface statistics beats "random" (the words
look different) but falls toward chance on "swap" (the words look the same).
The gap between the two conditions is the diagnostic, not either number alone.

Binary choice is deliberate: component pools differ in size (C1 has one
opposite, C4 has five alternatives), so a shared k-way pool would give each
component a different chance level and make cross-component comparison
meaningless. With one distractor, chance is 50% everywhere.

Design constraints enforced here (all traceable to measured facts):
  * only source records whose SHAPE clause is a single sentence are used
    (1,313 of 1,400), and replacement clauses are drawn only from standalone
    phrasings — so every swapped caption has the same sentence count as its
    original. The 87 excluded records are multi-sentence cubic phrasings;
    cubic classes remain represented via their single-sentence phrasings.
  * 'clean' is never a source or target for fluctuation swaps (it has no
    fluctuation clause). The clean<->fluctuation contrast is component C6,
    generated only with --include-c6 and reported separately, because it
    changes caption length.
  * the replacement clause is chosen from the k length-closest candidates in
    the target pool, to keep word-count differences small; the realised
    difference is reported per component so the confound is visible.

Run from repo root:
    python scripts/generate_probe1_items.py
    python scripts/generate_probe1_items.py --per-component 1500 --seed 43

Writes: data/processed/probe1_items.jsonl
        results/analysis/probe1_generation_report.json
"""

from __future__ import annotations
import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

PAIRS = Path("data/processed/pairs.jsonl")
TABLE = Path("results/analysis/component_table.json")
OUT_ITEMS = Path("data/processed/probe1_items.jsonl")
OUT_REPORT = Path("results/analysis/probe1_generation_report.json")
NO_FLUCT = "clean"
LENGTH_CANDIDATES = 3          # choose among the k length-closest phrasings


def sentences(t):
    return [p.strip() for p in re.split(r"(?<=[.!?])\s+", t.strip()) if p.strip()]


def split_clauses(fluct, sents):
    """Same corrected rule as build_component_table.py rev 2."""
    if fluct == NO_FLUCT:
        return sents, []
    if len(sents) < 2:
        return sents, []
    return sents[:-1], [sents[-1]]


def load_records():
    recs = []
    with open(PAIRS, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["dataset"] != "sushi":
                continue
            fl, sh = [p.strip() for p in r["class_label"].split(";")]
            sents = sentences(r["caption"])
            sh_s, fl_s = split_clauses(fl, sents)
            recs.append({
                "sample_id": r["sample_id"], "split": r["split"],
                "fluct": fl, "shape": sh, "class": r["class_label"],
                "caption": r["caption"], "sents": sents,
                "shape_sents": sh_s, "fluct_sents": fl_s,
            })
    return recs


def build_pools(recs):
    """Standalone pools only: phrasings that occur as the sole clause of their
    component, so substituting one never changes the sentence count."""
    shape_pool, fluct_pool = defaultdict(set), defaultdict(set)
    for r in recs:
        if len(r["shape_sents"]) == 1:
            shape_pool[r["shape"]].add(r["shape_sents"][0])
        if len(r["fluct_sents"]) == 1:
            fluct_pool[r["fluct"]].add(r["fluct_sents"][0])
    return ({k: sorted(v) for k, v in shape_pool.items()},
            {k: sorted(v) for k, v in fluct_pool.items()})


def pick_replacement(original: str, pool: list[str], rng: random.Random) -> str:
    """Length-closest-k, then random among them — variety without a length cue."""
    n = len(original.split())
    ranked = sorted(pool, key=lambda s: (abs(len(s.split()) - n), s))
    return rng.choice(ranked[:max(1, min(LENGTH_CANDIDATES, len(ranked)))])


def swap_caption(rec, slot, target_value, pools, rng):
    """Return (new_caption, replaced_from, replaced_to) or None if impossible."""
    shape_pool, fluct_pool = pools
    if slot == "shape":
        if len(rec["shape_sents"]) != 1:
            return None
        pool = shape_pool.get(target_value, [])
        if not pool:
            return None
        old = rec["shape_sents"][0]
        new = pick_replacement(old, pool, rng)
        sents = [new] + rec["fluct_sents"]
    else:
        if len(rec["fluct_sents"]) != 1:
            return None
        pool = fluct_pool.get(target_value, [])
        if not pool:
            return None
        old = rec["fluct_sents"][0]
        new = pick_replacement(old, pool, rng)
        sents = rec["shape_sents"] + [new]
    if new == old:
        return None
    return " ".join(sents), old, new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-component", type=int, default=1000,
                    help="target swap items per component (random controls double this)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--splits", nargs="+", default=["test"],
                    help="which data splits to draw source signals from")
    ap.add_argument("--include-c6", action="store_true",
                    help="also generate the length-changing presence/absence contrast")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    recs = [r for r in load_records() if r["split"] in args.splits]
    print(f"source records ({'+'.join(args.splits)}): {len(recs)}")

    pools = build_pools(load_records())      # pools from ALL splits: text resource
    shape_pool, fluct_pool = pools
    print(f"standalone shape pools: {len(shape_pool)} values, "
          f"sizes {min(map(len, shape_pool.values()))}-{max(map(len, shape_pool.values()))}")
    print(f"standalone fluct pools: {len(fluct_pool)} values, "
          f"sizes {min(map(len, fluct_pool.values()))}-{max(map(len, fluct_pool.values()))}")

    with open(TABLE, encoding="utf-8") as f:
        table = json.load(f)
    components = table["components"]

    by_shape, by_fluct, by_class = defaultdict(list), defaultdict(list), defaultdict(list)
    for r in recs:
        by_shape[r["shape"]].append(r)
        by_fluct[r["fluct"]].append(r)
        by_class[r["class"]].append(r)
    all_classes = sorted(by_class)

    items, skipped = [], Counter()

    for comp, spec in components.items():
        if comp.startswith("C6") and not args.include_c6:
            continue
        slot = "fluct" if comp.startswith(("C4", "C6")) else "shape"
        # both directions of every pair
        swap_types = []
        for a, b in spec["pairs"]:
            swap_types.append((a, b))
            swap_types.append((b, a))
        if comp.startswith("C6"):
            swap_types = [(a, b) for a, b in swap_types if a != NO_FLUCT]
        per_type = max(1, args.per_component // max(1, len(swap_types)))

        for src_val, tgt_val in swap_types:
            source_recs = (by_shape if slot == "shape" else by_fluct).get(src_val, [])
            source_recs = [r for r in source_recs
                           if (len(r["shape_sents"]) == 1 if slot == "shape"
                               else len(r["fluct_sents"]) == 1)]
            if not source_recs:
                skipped[f"{comp}:no_source:{src_val}"] += 1
                continue
            chosen = source_recs if len(source_recs) <= per_type else \
                rng.sample(source_recs, per_type)

            for rec in chosen:
                res = swap_caption(rec, slot, tgt_val, pools, rng)
                if res is None:
                    skipped[f"{comp}:no_replacement:{tgt_val}"] += 1
                    continue
                new_caption, old_clause, new_clause = res
                if new_caption == rec["caption"]:
                    skipped[f"{comp}:identical"] += 1
                    continue

                swapped_class = (f"{rec['fluct']}; {tgt_val}" if slot == "shape"
                                 else f"{tgt_val}; {rec['shape']}")
                base = {
                    "component": comp,
                    "slot": slot,
                    "swap_from": src_val,
                    "swap_to": tgt_val,
                    "sample_id": rec["sample_id"],
                    "split": rec["split"],
                    "source_class": rec["class"],
                    "caption_correct": rec["caption"],
                }
                items.append({**base,
                              "item_id": f"{comp}|{rec['sample_id']}|{tgt_val}|swap",
                              "condition": "swap",
                              "caption_distractor": new_caption,
                              "distractor_class": swapped_class,
                              "clause_replaced_from": old_clause,
                              "clause_replaced_to": new_clause})

                # matched random control: differs in BOTH slots
                cands = [c for c in all_classes
                         if c.split(";")[0].strip() != rec["fluct"]
                         and c.split(";")[1].strip() != rec["shape"]]
                rc = rng.choice(cands)
                rrec = rng.choice(by_class[rc])
                if rrec["caption"] != rec["caption"]:
                    items.append({**base,
                                  "item_id": f"{comp}|{rec['sample_id']}|{tgt_val}|random",
                                  "condition": "random",
                                  "caption_distractor": rrec["caption"],
                                  "distractor_class": rc,
                                  "clause_replaced_from": None,
                                  "clause_replaced_to": None})

    # ------------------------------------------------------------ validation
    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)
    problems = Counter()
    for it in items:
        if it["caption_correct"] == it["caption_distractor"]:
            problems["identical captions"] += 1
        if it["condition"] == "swap":
            a, b = sentences(it["caption_correct"]), sentences(it["caption_distractor"])
            if len(a) != len(b):
                problems["sentence count changed"] += 1
            if sum(x != y for x, y in zip(a, b)) != 1:
                problems["not exactly one clause changed"] += 1
    print("problems:", dict(problems) if problems else "none")
    if skipped:
        print("skipped (top 5):", dict(skipped.most_common(5)))

    # ------------------------------------------------------------ report
    print("\n" + "=" * 70)
    print("ITEMS PER COMPONENT")
    print("=" * 70)
    per_comp = Counter((it["component"], it["condition"]) for it in items)
    print(f"{'component':<28}{'swap':>8}{'random':>8}{'|Δwords| swap':>16}")
    print("-" * 62)
    report_rows = {}
    for comp in sorted({c for c, _ in per_comp}):
        deltas = [abs(len(it["caption_correct"].split())
                      - len(it["caption_distractor"].split()))
                  for it in items
                  if it["component"] == comp and it["condition"] == "swap"]
        mean_d = sum(deltas) / len(deltas) if deltas else 0.0
        print(f"{comp:<28}{per_comp[(comp,'swap')]:>8}"
              f"{per_comp[(comp,'random')]:>8}{mean_d:>16.2f}")
        report_rows[comp] = {"swap": per_comp[(comp, "swap")],
                             "random": per_comp[(comp, "random")],
                             "mean_abs_word_delta": mean_d}
    print(f"\ntotal items: {len(items)}  "
          f"(unique signals used: {len({it['sample_id'] for it in items})})")

    OUT_ITEMS.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_ITEMS, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        json.dump({"seed": args.seed, "splits": args.splits,
                   "per_component_target": args.per_component,
                   "length_candidates": LENGTH_CANDIDATES,
                   "n_items": len(items), "per_component": report_rows,
                   "validation_problems": dict(problems),
                   "skipped": dict(skipped)}, f, indent=2)
    print(f"\nsaved -> {OUT_ITEMS}\nsaved -> {OUT_REPORT}")

    # ------------------------------------------------------------ examples
    print("\n" + "=" * 70)
    print("EXAMPLE ITEMS (read these before trusting the set)")
    print("=" * 70)
    for comp in sorted({it["component"] for it in items}):
        ex = next((it for it in items
                   if it["component"] == comp and it["condition"] == "swap"), None)
        if not ex:
            continue
        print(f"\n[{comp}]  {ex['swap_from']} -> {ex['swap_to']}")
        print(f"  correct   : {ex['caption_correct']}")
        print(f"  distractor: {ex['caption_distractor']}")


if __name__ == "__main__":
    main()

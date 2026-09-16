"""
build_component_table.py — construct and VALIDATE the Probe-1 component table.
REVISION 2 (2026-07-28). Two corrections, both driven by revision 1's own
failure output:

  1. CLAUSE RULE. Rev 1 assumed "sentence 0 = shape, sentences 1+ = fluctuation".
     That failed: cubic shapes describe themselves in TWO sentences ("...starting
     with a rise, followed by a fall..." then "This produces an inverted S-shape,
     rotated by 90 degrees."), so the S-shape restatement was misfiled as a
     fluctuation clause, and 'clean' acquired a spurious fluctuation pool.
     Corrected rule: when fluctuation != clean, the LAST sentence is the
     fluctuation clause and everything before it is the shape clause; when
     fluctuation == clean, ALL sentences are shape and there is no fluctuation
     clause. Rev 1 section B supports this independently (0/70 violations among
     third sentences).

  2. DIRECTION VALIDATION + C1. Rev 1 counted rise/fall vocabulary per shape in
     isolation, which flagged 'inverted exponential decay' because "decay" sits
     in the name of the thing being inverted, though its captions read
     "increasing inverse decay ... towards saturation". Corrected: compare the
     two members of an explicit opposite PAIR against each other (relative, so
     shared vocabulary cancels), and for non-monotone pairs test which direction
     word appears FIRST rather than which is more frequent. C1 is now an explicit
     pair list rather than any up-value crossed with any down-value, which had
     admitted swaps changing two things at once (exponential growth -> inverted
     exponential growth).

Run from repo root:
    python scripts/build_component_table.py
Writes: results/analysis/component_table.json  (consumed by the swap generator)
"""

from __future__ import annotations
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

PAIRS = Path("data/processed/pairs.jsonl")
OUT = Path("results/analysis/component_table.json")
NO_FLUCT = "clean"

SHAPE = {
    "constant":                     ("trend", "constant",    "none"),
    "linear increase":              ("trend", "linear",      "up"),
    "linear decrease":              ("trend", "linear",      "down"),
    "exponential growth":           ("trend", "exponential", "up"),
    "exponential decay":            ("trend", "exponential", "down"),
    "inverted exponential growth":  ("trend", "exponential", "down"),
    "inverted exponential decay":   ("trend", "exponential", "up"),
    "sigmoid":                      ("trend", "sigmoid",     "up"),
    "inverted sigmoid":             ("trend", "sigmoid",     "down"),
    "gaussian":                     ("trend", "gaussian",    "up_down"),
    "inverted gaussian":            ("trend", "gaussian",    "down_up"),
    "concave":                      ("trend", "quadratic",   "up_down"),
    "convex":                       ("trend", "quadratic",   "down_up"),
    "cubic function":               ("trend", "cubic",       "up_down"),
    "negative cubic function":      ("trend", "cubic",       "down_up"),
    "sinusoidal wave":              ("periodic", "sine",     "phase"),
    "square wave":                  ("periodic", "square",   "phase"),
    "triangle wave":                ("periodic", "triangle", "phase"),
    "sawtooth wave":                ("periodic", "sawtooth", "rising"),
    "reverse sawtooth wave":        ("periodic", "sawtooth", "falling"),
}

# C1: explicit opposite pairs — each flips exactly one semantic dimension.
OPPOSITE_PAIRS = [
    ("linear increase",             "linear decrease",             "monotone"),
    ("exponential growth",          "exponential decay",           "monotone"),
    ("inverted exponential decay",  "inverted exponential growth", "monotone"),
    ("sigmoid",                     "inverted sigmoid",            "monotone"),
    ("gaussian",                    "inverted gaussian",           "order"),
    ("concave",                     "convex",                      "order"),
    ("cubic function",              "negative cubic function",     "order"),
    ("sawtooth wave",               "reverse sawtooth wave",       "untestable"),
]

UP = ["ris", "increas", "ascend", "escalat", "growth", "grow", "upturn",
      "upward", "climb", "surge", "incline", "going up"]
DOWN = ["fall", "decreas", "descend", "declin", "decay", "plunge",
        "downward", "drop", "diminish", "subsid", "lower"]


def sentences(t):
    return [p.strip() for p in re.split(r"(?<=[.!?])\s+", t.strip()) if p.strip()]


def load():
    rows = []
    with open(PAIRS, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["dataset"] != "sushi":
                continue
            fl, sh = [p.strip() for p in r["class_label"].split(";")]
            rows.append({"fluct": fl, "shape": sh, "caption": r["caption"],
                         "sents": sentences(r["caption"]),
                         "sample_id": r["sample_id"]})
    return rows


def split_clauses(r):
    """CORRECTED RULE -> (shape_sentences, fluctuation_sentences)"""
    s = r["sents"]
    if r["fluct"] == NO_FLUCT:
        return s, []
    if len(s) < 2:
        return s, []          # anomaly: non-clean caption with a single sentence
    return s[:-1], [s[-1]]


def first_direction(text):
    t = text.lower()
    iu = min([t.find(w) for w in UP if t.find(w) >= 0], default=10**6)
    idn = min([t.find(w) for w in DOWN if t.find(w) >= 0], default=10**6)
    if iu == idn == 10**6:
        return "none"
    return "up" if iu < idn else "down"


def main():
    rows = load()
    print(f"SUSHI records: {len(rows)}\n")

    # ------------------------------------------------------------------ A
    print("=" * 70)
    print("A. CLAUSE ATTRIBUTION (corrected rule)")
    print("=" * 70)

    shape_sents, fluct_sents = defaultdict(set), defaultdict(set)
    anomalies = []
    for r in rows:
        sh_s, fl_s = split_clauses(r)
        if r["fluct"] != NO_FLUCT and not fl_s:
            anomalies.append(r)
        for s in sh_s:
            shape_sents[s].add(r["shape"])
        for s in fl_s:
            fluct_sents[s].add(r["fluct"])

    bad_shape = {s: v for s, v in shape_sents.items() if len(v) > 1}
    bad_fluct = {s: v for s, v in fluct_sents.items() if len(v) > 1}
    print(f"shape clauses      : {len(shape_sents)} distinct, "
          f"{len(bad_shape)} tied to >1 shape")
    print(f"fluctuation clauses: {len(fluct_sents)} distinct, "
          f"{len(bad_fluct)} tied to >1 fluctuation")
    for s, v in list(bad_shape.items())[:3]:
        print(f'   VIOLATION shape "{s[:58]}..." -> {sorted(v)}')
    for s, v in list(bad_fluct.items())[:3]:
        print(f'   VIOLATION fluct "{s[:58]}..." -> {sorted(v)}')
    print(f"non-clean records lacking a fluctuation clause: {len(anomalies)}")

    clean_in_fluct = NO_FLUCT in {r["fluct"] for r in rows if split_clauses(r)[1]}
    print(f"'clean' records assigned a fluctuation clause: {clean_in_fluct}"
          "   (must be False)")

    attribution_ok = (not bad_shape and not bad_fluct
                      and not anomalies and not clean_in_fluct)
    print("-> clause attribution " + ("HOLDS" if attribution_ok else "STILL VIOLATED"))

    lens = Counter((r["fluct"] == NO_FLUCT, len(r["sents"])) for r in rows)
    print(f"\nsentence counts (is_clean, n): {dict(sorted(lens.items()))}")

    # ------------------------------------------------------------------ B
    print("\n" + "=" * 70)
    print("B. DIRECTION VALIDATION (pair-relative)")
    print("=" * 70)

    caps = defaultdict(list)
    for r in rows:
        for s in split_clauses(r)[0]:
            caps[r["shape"]].append(s)

    def score(shape):
        t = " ".join(caps[shape]).lower()
        return sum(t.count(w) for w in UP) - sum(t.count(w) for w in DOWN)

    def up_first_ratio(shape):
        firsts = [first_direction(s) for s in caps[shape]]
        n = sum(1 for f in firsts if f != "none")
        return (sum(1 for f in firsts if f == "up") / n) if n else None

    mismatches = []
    print(f"{'pair':<50}{'test':<12}{'result'}")
    print("-" * 76)
    for a, b, kind in OPPOSITE_PAIRS:
        label = f"{a} <-> {b}"
        if kind == "monotone":
            sa, sb = score(a), score(b)
            ok = sa > sb
            detail = f"net up-words {sa:+d} vs {sb:+d}"
        elif kind == "order":
            ra, rb = up_first_ratio(a), up_first_ratio(b)
            ok = ra is not None and rb is not None and ra > rb
            detail = (f"up-first {ra:.0%} vs {rb:.0%}"
                      if ra is not None and rb is not None else "no direction words")
        else:
            ok, detail = None, "no direction vocabulary (by design)"
        if ok is False:
            mismatches.append(label)
        mark = "OK" if ok else ("n/a" if ok is None else "MISMATCH")
        print(f"{label:<50}{kind:<12}{mark}  ({detail})")
    print(f"\ndirection mismatches: {len(mismatches)}"
          + (f" -> {mismatches}" if mismatches else "  (declared table consistent)"))

    # ------------------------------------------------------------------ C
    print("\n" + "=" * 70)
    print("C. COMPONENT TABLE")
    print("=" * 70)

    trends = sorted(s for s in SHAPE if SHAPE[s][0] == "trend")
    periodics = sorted(s for s in SHAPE if SHAPE[s][0] == "periodic")
    flucts = sorted({r["fluct"] for r in rows} - {NO_FLUCT})

    c1 = [(a, b) for a, b, _ in OPPOSITE_PAIRS]
    c2 = []
    for i, a in enumerate(trends):
        for b in trends[i + 1:]:
            _, fa, da = SHAPE[a]
            _, fb, db = SHAPE[b]
            if da == db and fa != fb and da != "none":
                c2.append((a, b))
    c3 = [(a, b) for i, a in enumerate(periodics) for b in periodics[i + 1:]]
    c4 = [(a, b) for i, a in enumerate(flucts) for b in flucts[i + 1:]]
    c5 = [(a, b) for a in trends for b in periodics]
    c6 = [(NO_FLUCT, f) for f in flucts]

    table = {
        "C1_trend_direction": {
            "desc": "explicit opposite pair; family and curve character held fixed",
            "pairs": c1, "primary": True},
        "C2_trend_family": {
            "desc": "same direction, different shape family",
            "pairs": c2, "primary": True},
        "C3_periodic_waveform": {
            "desc": "waveform swap within the periodic regime",
            "pairs": c3, "primary": True},
        "C4_fluctuation_type": {
            "desc": "fluctuation swap; 'clean' excluded (no clause to replace)",
            "pairs": c4, "primary": True},
        "C5_signal_regime": {
            "desc": "trend <-> periodic; most severe, expect largest degradation",
            "pairs": c5, "primary": True},
        "C6_fluctuation_presence": {
            "desc": "clean <-> fluctuation; CHANGES SENTENCE COUNT",
            "pairs": c6, "primary": False,
            "caveat": "insertion/deletion alters caption length, a surface cue; "
                      "not comparable with C1-C5 and must be reported separately"},
    }
    for k, v in table.items():
        flag = "" if v.get("primary") else "   [SECONDARY]"
        print(f"{k:<26}{len(v['pairs']):>4} pairs{flag}")
        print(f"      {v['desc']}")
        for a, b in v["pairs"][:2]:
            print(f"      e.g. {a}  ->  {b}")

    # ------------------------------------------------------------------ D
    print("\n" + "=" * 70)
    print("D. SENTENCE POOLS")
    print("=" * 70)
    shape_pool, fluct_pool = defaultdict(set), defaultdict(set)
    for r in rows:
        sh_s, fl_s = split_clauses(r)
        for s in sh_s:
            shape_pool[r["shape"]].add(s)
        for s in fl_s:
            fluct_pool[r["fluct"]].add(s)

    sp = sorted(len(v) for v in shape_pool.values())
    fp = sorted(len(v) for v in fluct_pool.values())
    print(f"shape pools      : {len(shape_pool)} values, sizes {sp[0]}-{sp[-1]} "
          f"(median {sp[len(sp)//2]})")
    print(f"fluctuation pools: {len(fluct_pool)} values, sizes {fp[0]}-{fp[-1]} "
          f"(median {fp[len(fp)//2]})")
    print(f"'{NO_FLUCT}' present in fluctuation pools: {NO_FLUCT in fluct_pool}"
          "   (must be False)")
    thin = [k for k, v in {**shape_pool, **fluct_pool}.items() if len(v) < 8]
    print("thin pools (<8): " + (", ".join(thin) if thin else "none"))

    # ------------------------------------------------------------------ save
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({
            "revision": 2,
            "clause_rule": ("fluct != clean: last sentence = fluctuation, rest = shape; "
                            "fluct == clean: all sentences = shape, no fluctuation clause"),
            "clause_attribution_holds": attribution_ok,
            "shape_decomposition": {k: {"kind": v[0], "family": v[1], "direction": v[2]}
                                    for k, v in SHAPE.items()},
            "opposite_pairs": [{"a": a, "b": b, "test": t} for a, b, t in OPPOSITE_PAIRS],
            "direction_mismatches": mismatches,
            "components": table,
            "shape_sentence_pool": {k: sorted(v) for k, v in shape_pool.items()},
            "fluctuation_sentence_pool": {k: sorted(v) for k, v in fluct_pool.items()},
        }, f, indent=2, ensure_ascii=False)
    print(f"\nsaved -> {OUT}")

    print("\n" + "=" * 70)
    print("GATE FOR THE SWAP GENERATOR")
    print("=" * 70)
    pools_ok = not thin and NO_FLUCT not in fluct_pool
    for name, val in [("clause attribution holds", attribution_ok),
                      ("direction table consistent", not mismatches),
                      ("pools usable", pools_ok)]:
        print(f"  {name:<32}{val}")
    print("\n-> " + ("ALL PASS: safe to generate swaps by clause substitution"
                     if attribution_ok and not mismatches and pools_ok
                     else "NOT ALL PASS: fix before generating"))


if __name__ == "__main__":
    main()

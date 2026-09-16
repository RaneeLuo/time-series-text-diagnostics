#!/usr/bin/env python3
"""
classify_truce_order_groups.py — Probe-2 TRUCE caption classifier, RULES v2
(3-way, truth-conditional per the binding Q3 rule).

v2 changes over v1 (2026-08-13, mechanisms found by reading the v1 sheets;
v1 provisional P2-8 rate 0.0328 remains in the record):
  A. NEGATED bumpiness/change ("no volatility", "negligible amplitude",
     "without any real change") -> invariant, per the SUSHI certified
     precedent (negated order-word survives shuffling) — UNLESS the rest
     of the caption still carries an order claim (one claim decides).
  B. Missed dependent vocabulary added: bare up/down, raise, descent,
     soar, crest, dive, finish, advance, curve, shape words (hill,
     valley, arch, bowl, convex/concave, u-shaped, cyclic, parabolic),
     trajectories (high-to-low, to-the-top, digit fractions), center.
  C. straight/linear/horizontal WITHOUT another invariant anchor ->
     ambiguous (reason straight_alone): this corpus uses "straight" for
     steep rises too ("Goes straight up").
  D. Invariant anchors gain stabil*/maintain*/consist* (typo family).

Priority chain, one order-sensitive claim decides:
  0  junk/empty                                   -> ambiguous (junk)
  1  negated stillness: mask negation+idiom spans;
     dependent hit on the remainder -> dependent; else -> invariant
  2  bumpiness idiom (unnegated): dependent is checked on idiom-masked
     text; idiom without dependent -> ambiguous (idiom)
  3  any dependent pattern (on idiom-masked text) -> dependent
  4  straight-family without other invariant anchor -> ambiguous
  5  any invariant anchor                          -> invariant
  6  nothing                                       -> ambiguous (no_anchor)
Unique texts classified once and propagated (no split verdicts possible).

Run from the repository root:
    python scripts/classify_truce_order_groups.py

Reads:  data/processed/pairs.jsonl
Writes: results/analysis/probe2_truce_groups.json
        results/analysis/probe2_truce_invariant_judgment_sheet.csv  (CENSUS)
        results/analysis/probe2_truce_dependent_sample.csv          (seeded 30)
        results/analysis/probe2_truce_ambiguous_sheet.csv           (all if <=200)

Gates: T1 rows 7380 / unique 5087 / test 738 (hard); T2 buckets
reconcile (hard); T3 extended panel exact (hard); T4 invariant anchors
nonempty (hard); T5 degenerate series printed (warn >10); T6 junk count.
"""

import csv
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

PAIRS = Path("data/processed/pairs.jsonl")
OUTDIR = Path("results/analysis")
SEED = 42

# ---------------------------------------------------------------------------
NEG_STILL_RE = re.compile(
    r"\b(?:no|non|not|little|low|negligible|minimal|slight|small|hardly|"
    r"barely|rarely|without)\b[\s\w-]{0,20}?"
    r"\b(?:volatil\w*|fluctuat\w*|oscillat\w*|variat\w*|varying|vary|"
    r"movement|moving|move[sd]?|change[sd]?|changing|"
    r"ups?\s+(?:and|&)\s+downs?|amplitude|parabol\w*|wave[sd]?|"
    r"radical\s+change|directional\s+bias)\b", re.I)

IDIOM_PATTERNS = [  # unnegated bumpiness -> ambiguous-excluded
    r"\bups?\s+(?:and|&)\s+downs?\b",
    r"\bpeaks\s+(?:and|&)\s+troughs\b",
    r"\bhighs\s+(?:and|&)\s+lows\b",
    r"\bhills\s+(?:and|&)\s+valleys\b",
    r"\bvolatil\w*", r"\boscillat\w*", r"\bfluctuat\w*",
    r"\bchoppy\b", r"\bjagged\b", r"\bwobbl\w*", r"\berratic\w*",
    r"\bzig\s*zag\w*", r"\bwav(?:e|es|y|ing)\b",
]

DEPENDENT_PATTERNS = [
    # direction / motion
    r"\bris(?:e|es|ing|en)\b", r"\brose\b", r"\brais\w*",
    r"\bfall(?:s|ing|en)?\b", r"\bfell\b",
    r"\bincreas\w*", r"\bdecreas\w*", r"\bclimb\w*",
    r"\bdrop\w*", r"\bdip\w*", r"\bplunge\w*", r"\bplummet\w*",
    r"\bjump\w*", r"\bspik\w*", r"\bsurge\w*", r"\bsoar\w*",
    r"\bgrow\w*", r"\bgrew\b", r"\bdeclin\w*", r"\binclin\w*",
    r"\breduc\w*", r"\bdescen\w*", r"\bascen\w*",
    r"\brecover\w*", r"\brebound\w*", r"\bbounc\w*\s+back\b",
    r"\bflatten\w*", r"\bplateau\w*",
    r"\blevel(?:s|ed|ing)?\s+(?:off|out)\b",
    r"\breach\w*", r"\brevers\w*", r"\bturn\w*", r"\badvanc\w*",
    r"\bcrest\w*", r"\bdive\w*", r"\bdove\b", r"\bfinish\w*",
    r"\bpeak\w*", r"\btrough\w*",
    r"\bbottom(?:s|ed|ing)?\s+out\b", r"\btop(?:s|ped|ping)?\s+out\b",
    r"\bto\s+the\s+(?:top|bottom)\b",
    r"\btrend\w*", r"\bslop\w*", r"\bsteep\w*", r"\bgradual\w*",
    r"\bsharply\b",
    r"\bup\b", r"\bdown\b",
    r"\bupward\w*", r"\bdownward\w*", r"\buphill\b", r"\bdownhill\b",
    r"\bhigher\b", r"\blower\b", r"\blowest\b", r"\bhighest\b",
    r"\bmaximum\b", r"\bminimum\b", r"\bmax\b", r"\bmin\b",
    r"\bhigh\s+to\s+low\b", r"\blow\s+to\s+high\b",
    # shape / curvature (arrangement claims)
    r"\bcurv\w*", r"\bhill\w*", r"\bvalley\w*", r"\barch\w*",
    r"\bbowl\b", r"\bconvex\b", r"\bconcave\b", r"\bu[\s-]?shap\w*",
    r"\bcycl\w*", r"\bperiodic\w*", r"\bparabol\w*",
    # sequence connectives / temporal anchors
    r"\bthen\b", r"\bafter\w*", r"\bbefore\b", r"\buntil\b",
    r"\bfollow\w*", r"\beventual\w*",
    r"\bstart\w*", r"\bbeg+in\w*", r"\bbegan\b", r"\bbegun\b",
    r"\bend\w*", r"\bfinal\w*", r"\binitial\w*",
    r"\bearly\b", r"\blate[rst]*\b",
    # positional
    r"\bfirst\b", r"\bsecond\b", r"\bthird\w*", r"\bfourth\b", r"\bfifth\b",
    r"\blast\b", r"\bmid\w*", r"\bcent(?:er|re)s?\b",
    r"\bhalf?way\b", r"\bhalf\b", r"\bhalves\b",
    r"\b\d+(?:st|nd|rd|th)\b",
    r"\b\d+\s*/\s*\d+(?:ths?|rds?|nds?)?\b",
    r"\b(?:one|two|three)[\s-]*(?:third|quarter|fourth|half)s?\b",
    r"\bpoint\s+\d+\b", r"\bvalue\s+\d+\b",
    r"\bfrom\b[^.]*\bto\b",
    r"\bover\s+time\b",
]

STRAIGHT_RE = re.compile(r"\bstraight\b|\blinear\b|\bhorizontal\b", re.I)

INVARIANT_PATTERNS = [
    r"\bflat\w*", r"\blevel\b", r"\bstead\w*", r"\bstable\b", r"\bstabil\w*",
    r"\bconstant\w*", r"\bstagnant\b", r"\bstagnent\b", r"\bunchang\w*",
    r"\bunwaver\w*", r"\bconsist\w*", r"\beven\b", r"\bevenly\b",
    r"\bmaintain\w*", r"\bunevent\w*",
    r"\bstays?\b", r"\bstayed\b", r"\bstaye[sd]\b", r"\bremain\w*",
    r"\bkeeps?\b",
    r"\b(?:no|little|barely\s+any|hardly\s+any)\s+(?:change|movement)\b",
]

IDIOM_RE = [(re.compile(p, re.I), p) for p in IDIOM_PATTERNS]
DEP_RE = [(re.compile(p, re.I), p) for p in DEPENDENT_PATTERNS]
INV_RE = [(re.compile(p, re.I), p) for p in INVARIANT_PATTERNS]

# ---------------------------------------------------------------------------
# Panel (gate T3): v1 cases carried forward + v2 mechanism cases, all read
# verbatim from the 2026-08-13 inspection output and v1 sheets.
# ONE DELIBERATE CHANGE from the v1 panel: 'oscillates with a negligible
# amplitude' was ambiguous in v1 and is invariant in v2 (negation rule A).
# ---------------------------------------------------------------------------
PANEL = [
    # v1 carried
    ("reaches its maximum height two-thirds in.", "dependent"),
    ("reverses course at midpoint", "dependent"),
    ("Steady slow incline throughout", "dependent"),
    ("this plot comes from lowest & to highest", "dependent"),
    ("ends at lower value than it began at", "dependent"),
    ("very level and flat at the begging", "dependent"),
    ("spike in the middle", "dependent"),
    ("the first value is high", "dependent"),
    ("it increases steadlily in the beginning.", "dependent"),
    ("mostly flat", "invariant"),
    ("remains relatively flat throughout", "invariant"),
    ("the graph is stable", "invariant"),
    ("stays fairly low throughout.", "invariant"),
    ("remains unwavering the entire way.", "invariant"),
    ("{}", "ambiguous"),
    ("plot definition, a secret plan or scheme purpose", "ambiguous"),
    ("the up and down f the lines.", "ambiguous"),
    # v2: invariant-sheet leaks now dependent
    ("Goes straight up", "dependent"),
    ("straight down", "dependent"),
    ("steady advance to top", "dependent"),
    ("line is steady with slight raise at thend", "dependent"),
    # v2: missed dependent vocabulary from the ambiguous sheet
    ("A slow continuous descent.", "dependent"),
    ("Soars continuously the entire way.", "dependent"),
    ("crests 3/4ths of the way in", "dependent"),
    ("goes high to low", "dependent"),
    ("it is one large hill", "dependent"),
    ("gently curves down", "dependent"),
    ("evinces a cyclic pattern", "dependent"),
    ("mostly headed down", "dependent"),
    ("makes run to the top", "dependent"),
    # v2: negated stillness -> invariant (rule A)
    ("low volatility throughout", "invariant"),
    ("there is little to no volatility", "invariant"),
    ("bounces up and down without any real change", "invariant"),
    ("oscillates with a negligible amplitude", "invariant"),
    ("does not move very positively or negatively throughout", "invariant"),
    ("stays flat with slight volatility.", "invariant"),
    ("has a negligible parabolic tendency", "invariant"),
    ("includes no radical change", "invariant"),
    # v2: anchor additions (rule D)
    ("maintains stability throughout", "invariant"),
    ("consistant throughout", "invariant"),
    ("remains relatively centered thoughout.", "invariant"),
    # v2: straight-alone (rule C) and unnegated idioms stay excluded
    ("Plot is straight line", "ambiguous"),
    ("linear throughout", "ambiguous"),
    ("many ups and downs", "ambiguous"),
    ("It is very volatile.", "ambiguous"),
    ("fluctuates throughout", "ambiguous"),
    # v1 invariants that must survive v2's bare up/down and straight rules
    ("nearly flat straight line throughout", "invariant"),
    ("very steady and straight", "invariant"),
    ("stays mostly flat with a slight wave", "invariant"),
]


def is_junk(text):
    letters = re.sub(r"[^a-zA-Z]", "", text)
    return len(letters) < 3


def _mask(text, regexes):
    for rx, _ in regexes:
        text = rx.sub(" ", text)
    return text


def classify(text):
    """Return (label, reason, matched_patterns)."""
    if is_junk(text):
        return "ambiguous", "junk", []
    negs = NEG_STILL_RE.findall(text)
    if negs:
        remainder = _mask(NEG_STILL_RE.sub(" ", text), IDIOM_RE)
        dep = [p for rx, p in DEP_RE if rx.search(remainder)]
        if dep:
            return "dependent", "order_claim", dep
        return "invariant", "negated_motion", ["NEG:" + n if isinstance(n, str)
                                               else "NEG" for n in negs] or ["NEG"]
    idiom = [p for rx, p in IDIOM_RE if rx.search(text)]
    dep = [p for rx, p in DEP_RE if rx.search(_mask(text, IDIOM_RE))]
    if idiom and not dep:
        return "ambiguous", "idiom", idiom
    if dep:
        return "dependent", "order_claim", dep + (["(+idiom)"] if idiom else [])
    inv = [p for rx, p in INV_RE if rx.search(text)]
    if STRAIGHT_RE.search(text) and not inv:
        return "ambiguous", "straight_alone", [STRAIGHT_RE.pattern]
    if inv:
        return "invariant", "anchored", inv
    return "ambiguous", "no_anchor", []


def fail(msg):
    print(f"\nGATE FAILED: {msg}", file=sys.stderr)
    sys.exit(1)


def main():
    if not PAIRS.exists():
        fail(f"{PAIRS} not found — run from the repository root")

    truce = []
    with open(PAIRS, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["dataset"].startswith("truce"):
                truce.append(r)

    n_test = sum(1 for r in truce if r["split"] == "test")
    uniq_texts = sorted({r["caption"] for r in truce})
    print(f"RULES v2 | TRUCE rows: {len(truce)}   unique texts: "
          f"{len(uniq_texts)}   test rows: {n_test}")
    if len(truce) != 7380:
        fail(f"T1: rows {len(truce)} != 7380")
    if len(uniq_texts) != 5087:
        fail(f"T1: unique texts {len(uniq_texts)} != 5087")
    if n_test != 738:
        fail(f"T1: test rows {n_test} != 738")
    print("T1 PASSED")

    bad = []
    for text, want in PANEL:
        got, reason, pats = classify(text)
        if got != want:
            bad.append((text, want, got, reason, pats[:4]))
    if bad:
        for b in bad:
            print(f"  PANEL MISS: {b}", file=sys.stderr)
        fail(f"T3: {len(bad)}/{len(PANEL)} panel cases misclassified")
    print(f"T3 PASSED: panel {len(PANEL)}/{len(PANEL)}")

    label_of = {t: classify(t) for t in uniq_texts}

    rows_by_label = Counter()
    uniq_by_label = Counter()
    reason_uniq = Counter()
    test_rows_by_label = Counter()
    n_rows_of = Counter(r["caption"] for r in truce)
    splits_of = defaultdict(set)
    for t in uniq_texts:
        uniq_by_label[label_of[t][0]] += 1
        reason_uniq[(label_of[t][0], label_of[t][1])] += 1
    for r in truce:
        lab = label_of[r["caption"]][0]
        rows_by_label[lab] += 1
        splits_of[r["caption"]].add(r["split"])
        if r["split"] == "test":
            test_rows_by_label[lab] += 1

    if sum(rows_by_label.values()) != 7380 or sum(uniq_by_label.values()) != 5087:
        fail(f"T2: buckets do not reconcile — rows {dict(rows_by_label)}, "
             f"unique {dict(uniq_by_label)}")
    print("T2 PASSED: buckets reconcile")

    empty_anchor = [t for t in uniq_texts
                    if label_of[t][0] == "invariant" and not label_of[t][2]]
    if empty_anchor:
        fail(f"T4: {len(empty_anchor)} invariant texts with no anchors, "
             f"e.g. {empty_anchor[:3]}")
    print("T4 PASSED: every invariant text carries an anchor")

    const_series = sorted({r["sample_id"] for r in truce
                           if len(set(r["series"])) == 1})
    print(f"T5: degenerate (constant) series: {len(const_series)}"
          f"{'   ' + str(const_series[:5]) if const_series else ''}")
    if len(const_series) > 10:
        print("  WARN T5: more than 10 constant series — report back.")

    junk_texts = [t for t in uniq_texts if label_of[t][1] == "junk"]
    junk_rows = sum(n_rows_of[t] for t in junk_texts)
    print(f"T6: junk/empty captions: {len(junk_texts)} unique texts, "
          f"{junk_rows} rows   examples: {[repr(t) for t in junk_texts[:5]]}")

    print("\nBUCKETS (rows | unique texts | test rows):")
    for lab in ("dependent", "invariant", "ambiguous"):
        print(f"  {lab:10s} {rows_by_label[lab]:5d} | {uniq_by_label[lab]:5d} | "
              f"{test_rows_by_label[lab]:4d}")
    print(f"invariant reasons (unique): "
          f"{ {k[1]: v for k, v in reason_uniq.items() if k[0]=='invariant'} }")
    print(f"ambiguous reasons (unique): "
          f"{ {k[1]: v for k, v in reason_uniq.items() if k[0]=='ambiguous'} }")

    dep_r, inv_r = rows_by_label["dependent"], rows_by_label["invariant"]
    dep_u, inv_u = uniq_by_label["dependent"], uniq_by_label["invariant"]
    dep_t, inv_t = test_rows_by_label["dependent"], test_rows_by_label["invariant"]
    p28 = inv_r / (dep_r + inv_r)
    print(f"\nP2-8 PROVISIONAL v2 (pre-certification; v1 rate 0.0328 stays in "
          f"the record): invariant/(dep+inv)")
    print(f"  row-level population (PINNED PRIMARY): {p28:.4f}  "
          f"({inv_r}/{dep_r + inv_r})   registered threshold 0.15")
    print(f"  unique-level population: {inv_u/(dep_u+inv_u):.4f}   "
          f"test-split row-level: {inv_t/(dep_t+inv_t):.4f}")
    print(f"  provisional verdict: "
          f"{'>= 0.15' if p28 >= 0.15 else 'BELOW 0.15 — P2-8 heading for a MISS'}")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    def write_sheet(path, texts, extra_note=""):
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["caption", "n_rows", "splits", "label", "reason",
                        "matched_patterns", "verdict_y_n", "notes"])
            for t in texts:
                lab, reason, pats = label_of[t]
                w.writerow([t, n_rows_of[t],
                            "/".join(sorted(splits_of[t])), lab, reason,
                            "; ".join(str(p) for p in pats[:6]), "", ""])
        print(f"wrote {path}  ({len(texts)} texts){extra_note}")

    inv_texts = [t for t in uniq_texts if label_of[t][0] == "invariant"]
    write_sheet(OUTDIR / "probe2_truce_invariant_judgment_sheet.csv",
                inv_texts, "   <- CENSUS: judge every row")
    dep_texts = [t for t in uniq_texts if label_of[t][0] == "dependent"]
    write_sheet(OUTDIR / "probe2_truce_dependent_sample.csv",
                rng.sample(dep_texts, min(30, len(dep_texts))),
                "   <- seeded sample (fresh draw over v2 dependents)")
    amb_texts = [t for t in uniq_texts if label_of[t][0] == "ambiguous"]
    if len(amb_texts) <= 200:
        write_sheet(OUTDIR / "probe2_truce_ambiguous_sheet.csv", amb_texts,
                    "   <- full (<=200)")
    else:
        write_sheet(OUTDIR / "probe2_truce_ambiguous_sheet.csv",
                    rng.sample(amb_texts, 60),
                    f"   <- seeded 60 of {len(amb_texts)}")

    with open(OUTDIR / "probe2_truce_groups.json", "w", encoding="utf-8") as f:
        json.dump({
            "rules_version": 2, "seed": SEED,
            "v1_p2_8_row_population": 0.0328,
            "rows": dict(rows_by_label), "unique": dict(uniq_by_label),
            "test_rows": dict(test_rows_by_label),
            "invariant_reasons_unique":
                {k[1]: v for k, v in reason_uniq.items() if k[0] == "invariant"},
            "ambiguous_reasons_unique":
                {k[1]: v for k, v in reason_uniq.items() if k[0] == "ambiguous"},
            "junk": {"unique": len(junk_texts), "rows": junk_rows},
            "degenerate_series": const_series,
            "p2_8_provisional": {
                "primary_row_population": p28,
                "unique_population": inv_u / (dep_u + inv_u),
                "test_row": inv_t / (dep_t + inv_t),
                "threshold": 0.15,
            },
            "per_text": {t: {"label": label_of[t][0],
                             "reason": label_of[t][1],
                             "patterns": [str(p) for p in label_of[t][2]],
                             "n_rows": n_rows_of[t]}
                         for t in uniq_texts},
        }, f, indent=2, ensure_ascii=False)
    print(f"wrote {OUTDIR / 'probe2_truce_groups.json'}")
    print("\nALL GATES PASSED (warn-level notes above if any)")


if __name__ == "__main__":
    main()

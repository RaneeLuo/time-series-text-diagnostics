#!/usr/bin/env python3
"""
inspect_truce_order_language.py — Probe-2 TRUCE classifier build, STEP 1 (design stage).

Purpose: show the TRUCE caption corpus to the rule designer BEFORE any
classification rule exists. The order-word list below is a CANDIDATE scan
for design purposes only — it is NOT the classifier and its counts are
not quotable. The classifier (3-way, truth-conditional) is built next,
from what this script prints.

Run from the repository root:
    python scripts/inspect_truce_order_language.py

Reads:  data/processed/pairs.jsonl   (canonical corpus; utf-8 enforced)
Writes: results/analysis/truce_order_inspection.json

Gates (fail loudly, in order):
  I1  total pairs == 8,780                                   HARD
  I2  truce_synth == 1,680 and truce_stock == 5,700
      (560 x 3 and 1,900 x 3, from dataset.py + verified
      series counts) and sushi == 1,400                      HARD
  I3  every TRUCE sample_id carries exactly 3 captions       HARD
  I4  mojibake canary: no U+FFFD, no 'Â'/'æ'/'ã' bytes-
      misdecoded markers in any TRUCE caption                HARD
  I5  TRUCE test rows == 738 (246 series x 3; INFERRED from
      pool-386 arithmetic, not source-verified)              WARN only
All counts are DESCRIPTION-LEVEL (captions containing a pattern),
never occurrence counts — labelled as such. (TRACE census lesson.)
"""

import json
import re
import sys
import random
from collections import Counter, defaultdict
from pathlib import Path

PAIRS = Path("data/processed/pairs.jsonl")
OUT = Path("results/analysis/truce_order_inspection.json")
SEED = 42

# ---------------------------------------------------------------------------
# Candidate order vocabulary — DESIGN-STAGE ONLY. Word-boundary regexes.
# Two sub-lists, counted separately because Q4(ii) requires the eventual
# classifier to catch positional captions specifically.
# ---------------------------------------------------------------------------
ARRANGEMENT_PATTERNS = [
    r"\bris(?:e|es|ing|en)\b", r"\brose\b",
    r"\bfall(?:s|ing|en)?\b", r"\bfell\b",
    r"\bincreas\w*", r"\bdecreas\w*",
    r"\bclimb\w*", r"\bdrop\w*", r"\bdip\w*", r"\bplunge\w*",
    r"\bplummet\w*", r"\bjump\w*", r"\bspik\w*", r"\bsurge\w*",
    r"\bgrow\w*", r"\bgrew\b", r"\bdeclin\w*", r"\bdescend\w*",
    r"\bascend\w*", r"\brecover\w*", r"\brebound\w*",
    r"\bflatten\w*", r"\bplateau\w*", r"\blevel(?:s|ed|ing)?\s+(?:off|out)\b",
    r"\bpeak\w*", r"\btrough\w*", r"\bbottom\w*", r"\btop(?:s|ped|ping)?\s+out\b",
    r"\btrend\w*", r"\bslop\w*", r"\bsteep\w*", r"\bgradual\w*",
    r"\bsteadily\b", r"\bsharply\b",
    r"\bthen\b", r"\bafter\w*", r"\bbefore\b", r"\buntil\b",
    r"\bstart\w*", r"\bbegin\w*", r"\bbegan\b", r"\bbegun\b",
    r"\bend\w*", r"\bfinal\w*", r"\binitial\w*", r"\bearly\b", r"\blate[rst]*\b",
    r"\bgo(?:es|ing)?\s+(?:up|down)\b", r"\bwent\s+(?:up|down)\b",
    r"\bupward\w*", r"\bdownward\w*", r"\buphill\b", r"\bdownhill\b",
    r"\bhigher\b", r"\blower\b",   # comparative motion words; noisy, design-stage
]
POSITIONAL_PATTERNS = [
    r"\bfirst\b", r"\bsecond\b", r"\bthird\b", r"\bfourth\b", r"\bfifth\b",
    r"\blast\b", r"\bmiddle\b", r"\bbeginning\b",
    r"\b\d+(?:st|nd|rd|th)\b",
    r"\bpoint\s+\d+\b", r"\bvalue\s+\d+\b",
    r"\bhalf\b", r"\bhalfway\b",
]
ARR_RE = [re.compile(p, re.IGNORECASE) for p in ARRANGEMENT_PATTERNS]
POS_RE = [re.compile(p, re.IGNORECASE) for p in POSITIONAL_PATTERNS]

MOJIBAKE_MARKERS = ["\ufffd", "Â", "Ã", "æ\x80", "掳"]

STOPWORDS = set("""a an the is are was were be been being it its this that these
those and or but of in on at to for with as by from very quite really just
there their they i we you he she""".split())


def fail(msg):
    print(f"\nGATE FAILED: {msg}", file=sys.stderr)
    sys.exit(1)


def words(text):
    return re.findall(r"[a-z']+", text.lower())


def main():
    if not PAIRS.exists():
        fail(f"{PAIRS} not found — run from the repository root")

    rows = []
    with open(PAIRS, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    # ---- I1 ----
    print(f"total pairs: {len(rows)}   (expected 8780)")
    if len(rows) != 8780:
        fail(f"I1: total pairs {len(rows)} != 8780")

    by_ds = Counter(r["dataset"] for r in rows)
    print(f"by dataset: {dict(sorted(by_ds.items()))}")
    # ---- I2 ----
    exp = {"truce_synth": 1680, "truce_stock": 5700, "sushi": 1400}
    for ds, n in exp.items():
        if by_ds.get(ds) != n:
            fail(f"I2: {ds} count {by_ds.get(ds)} != {n}")
    print("I1, I2 PASSED")

    truce = [r for r in rows if r["dataset"].startswith("truce")]

    # ---- I3 ----
    per_sid = Counter(r["sample_id"] for r in truce)
    bad = {s: c for s, c in per_sid.items() if c != 3}
    if bad:
        fail(f"I3: {len(bad)} TRUCE sample_ids without exactly 3 captions, e.g. "
             f"{list(bad.items())[:3]}")
    print(f"I3 PASSED: {len(per_sid)} TRUCE series x 3 captions")

    # ---- I4 ----
    hit = [(r["caption_id"], m) for r in truce
           for m in MOJIBAKE_MARKERS if m in r["caption"]]
    if hit:
        fail(f"I4 mojibake canary: {len(hit)} hits, first: {hit[:3]}")
    print("I4 PASSED: no mojibake markers in any TRUCE caption")

    # ---- I5 (WARN) ----
    split_counts = Counter((r["dataset"], r["split"]) for r in truce)
    print("\nTRUCE rows per (dataset, split):")
    for k in sorted(split_counts):
        print(f"  {k[0]:12s} {k[1]:6s} {split_counts[k]}")
    n_test = sum(v for (ds, sp), v in split_counts.items() if sp == "test")
    print(f"TRUCE test rows total: {n_test}   (expected 738 — INFERRED, warn-level)")
    if n_test != 738:
        print("  WARN I5: differs from the pool-386 inference; "
              "informative miss, not fatal — report back.")

    # ---- corpus shape ----
    caps = [r["caption"] for r in truce]
    uniq = len(set(caps))
    lens = sorted(len(words(c)) for c in caps)
    print(f"\ncaptions: {len(caps)}  unique: {uniq}  "
          f"duplicates: {len(caps) - uniq}")
    print(f"caption length in words: min {lens[0]}  "
          f"median {lens[len(lens)//2]}  max {lens[-1]}")

    # ---- candidate scan (description-level: captions containing >=1 hit) ----
    def hits(regexes, text):
        return [rx.pattern for rx in regexes if rx.search(text)]

    n_arr = n_pos = n_both = 0
    zero_hit = []
    pat_caption_counts = Counter()   # captions containing each pattern
    for r in truce:
        a = hits(ARR_RE, r["caption"])
        p = hits(POS_RE, r["caption"])
        for pat in set(a) | set(p):
            pat_caption_counts[pat] += 1
        if a:
            n_arr += 1
        if p:
            n_pos += 1
        if a and p:
            n_both += 1
        if not a and not p:
            zero_hit.append(r)

    n = len(truce)
    print(f"\nCANDIDATE SCAN (design-stage, not the classifier).")
    print(f"All counts = captions CONTAINING the pattern, of {n} TRUCE captions:")
    print(f"  >=1 arrangement word : {n_arr}  ({n_arr/n:.1%})")
    print(f"  >=1 positional word  : {n_pos}  ({n_pos/n:.1%})")
    print(f"  both                 : {n_both}")
    print(f"  ZERO candidate hits  : {len(zero_hit)}  ({len(zero_hit)/n:.1%})"
          f"   <- invariant candidates, exploratory")
    zh_split = Counter((r["dataset"], r["split"]) for r in zero_hit)
    print(f"  zero-hit by (dataset,split): {dict(sorted(zh_split.items()))}")

    print("\nTop 25 patterns by captions containing them:")
    for pat, c in pat_caption_counts.most_common(25):
        print(f"  {c:5d}  {pat}")

    # ---- vocabulary of the zero-hit set: what did the candidate list miss? ----
    freq = Counter(w for r in zero_hit for w in set(words(r["caption"]))
                   if w not in STOPWORDS)
    print("\nTop 40 words in ZERO-HIT captions (captions containing the word):")
    for w, c in freq.most_common(40):
        print(f"  {c:5d}  {w}")

    # ---- verbatim samples ----
    rng = random.Random(SEED)
    zh_sample = rng.sample(zero_hit, min(25, len(zero_hit)))
    print(f"\n25 ZERO-HIT captions, seeded sample (seed {SEED}), verbatim:")
    for r in zh_sample:
        print(f"  [{r['caption_id']}] {r['caption']!r}")

    for ds in ("truce_synth", "truce_stock"):
        pool = [r for r in truce if r["dataset"] == ds]
        smp = rng.sample(pool, 10)
        print(f"\n10 {ds} captions, seeded sample, verbatim:")
        for r in smp:
            print(f"  [{r['caption_id']}] {r['caption']!r}")

    # ---- record ----
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({
            "seed": SEED,
            "total_pairs": len(rows),
            "by_dataset": dict(by_ds),
            "truce_split_counts": {f"{k[0]}/{k[1]}": v
                                   for k, v in sorted(split_counts.items())},
            "unique_captions": uniq,
            "n_arrangement": n_arr, "n_positional": n_pos,
            "n_both": n_both, "n_zero_hit": len(zero_hit),
            "pattern_caption_counts": dict(pat_caption_counts),
            "zero_hit_top_words": freq.most_common(40),
            "zero_hit_caption_ids": [r["caption_id"] for r in zero_hit],
            "note": "design-stage candidate scan; NOT the classifier; "
                    "counts are description-level (captions containing).",
        }, f, indent=2, ensure_ascii=False)
    print(f"\nwrote {OUT}")
    print("ALL GATES PASSED (I5 warn-level noted above if fired)")


if __name__ == "__main__":
    main()

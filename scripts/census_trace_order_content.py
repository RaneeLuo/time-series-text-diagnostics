"""
census_trace_order_content.py — Probe-2 Q3 census: how many TRACE test
descriptions contain NO order-dependent content?

Rule (BINDING, accepted 2026-08-09): truth-conditional grouping at the
whole-description level, 3-way:
  DEPENDENT  — at least one sentence makes a claim about the ARRANGEMENT of
               values along the sequence axis (trend direction, dated/located
               extremes, phase language like "then" / "toward the end").
  INVARIANT  — no dependent sentence AND no ambiguous sentence anywhere.
  AMBIGUOUS  — no dependent sentence, but at least one sentence uses
               vocabulary whose order-sensitivity is genuinely unclear
               (fluctuation/stability/variability wording). Excluded from
               both groups and COUNTED, per the committed 3-way design.

Date-range sentences (temporal extent, N2-style "from March to June") are
INVARIANT here: shuffling the VALUES does not change what period the series
covers — the model input carries no timestamps (metadata claim, not an
arrangement claim). Dated EXTREMES ("maximum on June 12") are DEPENDENT:
via the date range they map to "max at position k", an arrangement claim.

The vocabulary below is v1 FOR THE COUNT ONLY. Sample-to-screen: this
script emits a stratified 40-row sample sheet for human judgment. No group
is certified from this script alone; if the sample fails, the vocabulary is
revised and the census re-run (same pattern as the item-validation arc).

REGISTERED EXPECTATIONS (logged before first run):
  rows (test split): 2006
  P2-7: INVARIANT < 5% of 2006 (i.e. < ~100). If < 50, within-TRACE DiD is
        declared too thin and the recorded-finding path is invoked.
  No numeric expectation registered for AMBIGUOUS (first measurement).

Run from repo root (same convention as run_narrative_probe.py):
    python scripts/census_trace_order_content.py --trace-repo /path/to/TRACE
Writes: results/analysis/probe2_trace_order_census.json
        results/analysis/probe2_trace_order_sample.csv   (40 rows, human check)
First line of output should read:  [G1] test parquet rows: 2006
"""

from __future__ import annotations
import argparse
import csv
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

OUT = Path("results/analysis/probe2_trace_order_census.json")
SAMPLE = Path("results/analysis/probe2_trace_order_sample.csv")

# ------------------------------------------------------------- vocabulary v1
# DEPENDENT: arrangement-of-values claims.
DEPENDENT_PATTERNS = [
    # trend direction (superset of the N3 vocab in generate_narrative_items.py)
    r"\brising\b", r"\brose\b", r"\bincreas\w*", r"\bupward\b", r"\bclimb\w*",
    r"\bdeclin\w*", r"\bdecreas\w*", r"\bdownward\b", r"\bfalling\b", r"\bfell\b",
    r"\bdropp?\w*", r"\btrend\w*", r"\bwarming\b", r"\bcooling\b",
    # phase / sequence language
    r"\bthen\b", r"\bbefore\b", r"\bafter\w*\b", r"\bfollowed\b", r"\bsubsequent\w*",
    r"\bbeginning\b", r"\bstart\w*\b", r"\bend\w*\b", r"\bearly\b", r"\blate[rs]?\b",
    r"\bmid[- ]", r"\btoward\w*\b", r"\bgradual\w*", r"\bsteadily\b", r"\bshift\w*",
    # located extremes: peak/dip/spike tied to a date or position
    r"\bpeak\w*", r"\bdipp?\w*", r"\bspik\w*", r"\bsurg\w*", r"\bjump\w*",
    r"\breach\w*\b", r"\brecord\w* (?:high|low)", r"\bmaximum .{0,20}\bon\b",
    r"\bminimum .{0,20}\bon\b", r"\bhighest .{0,20}\bon\b", r"\blowest .{0,20}\bon\b",
]
# AMBIGUOUS: order-sensitivity genuinely unclear; 3-way excluded bucket.
AMBIGUOUS_PATTERNS = [
    r"\bfluctuat\w*", r"\bvariab\w*", r"\bvary\w*|\bvaried\b", r"\bstable\b",
    r"\bstability\b", r"\bsteady\b", r"\bvolatil\w*", r"\bconsistent\w*",
    r"\bconstant\w*", r"\boscillat\w*",
]

DEP_RE = [re.compile(p, re.IGNORECASE) for p in DEPENDENT_PATTERNS]
AMB_RE = [re.compile(p, re.IGNORECASE) for p in AMBIGUOUS_PATTERNS]


def fail(gate: str, msg: str):
    print(f"\n[GATE {gate}] FAILED: {msg}", file=sys.stderr)
    sys.exit(1)


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def hits(sent: str, regexes) -> list[str]:
    out = []
    for rx in regexes:
        m = rx.search(sent)
        if m:
            out.append(m.group(0).lower())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-repo", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--sample-n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    repo = Path(args.trace_repo).resolve()
    sys.path.insert(0, str(repo))
    try:
        from src.data.load_data import generate_dsp  # authors' own rebuild — zero drift
    except Exception as e:
        fail("G0-import", f"cannot import generate_dsp from {repo}: {e}")

    import pyarrow.parquet as pq
    parquet = repo / "dataset" / "retrieval" / args.split / f"{args.split}.parquet"
    if not parquet.is_file():
        fail("G1", f"{parquet} missing")
    df = pq.read_table(parquet).to_pandas()
    print(f"[G1] {args.split} parquet rows: {len(df)}   (expected 2006 for test)")
    if args.split == "test" and len(df) != 2006:
        fail("G1-count", f"{len(df)} rows, expected 2006")

    print(f"[vocab] dependent patterns: {len(DEP_RE)}   ambiguous patterns: {len(AMB_RE)}")

    buckets = Counter()
    dep_word_freq = Counter()
    amb_word_freq = Counter()
    per_row = []
    empty_sent = 0
    for i in range(len(df)):
        d = df.iloc[i]["description"]
        try:
            text = generate_dsp(d)
        except Exception as e:
            fail("G2-rebuild", f"generate_dsp failed on row {i}: {e}")
        if not text or not text.strip():
            fail("G2-empty", f"row {i} rebuilt to empty text")
        if "\ufffd" in text or "掳" in text:
            fail("G2-encoding", f"row {i} contains mojibake: {text[:80]!r}")
        sents = split_sentences(text)
        if not sents:
            empty_sent += 1
        dep_hits, amb_hits = [], []
        for s in sents:
            dh = hits(s, DEP_RE)
            ah = hits(s, AMB_RE)
            dep_hits += dh
            amb_hits += ah
        for w in dep_hits:
            dep_word_freq[w] += 1
        for w in amb_hits:
            amb_word_freq[w] += 1
        if dep_hits:
            b = "dependent"
        elif amb_hits:
            b = "ambiguous"
        else:
            b = "invariant"
        buckets[b] += 1
        per_row.append({"row_idx": i, "bucket": b,
                        "n_sentences": len(sents),
                        "dep_hits": sorted(set(dep_hits)),
                        "amb_hits": sorted(set(amb_hits))})

    if empty_sent:
        fail("G3-sentences", f"{empty_sent} descriptions produced zero sentences")
    total = sum(buckets.values())
    if total != len(df):
        fail("G4-closure", f"buckets sum to {total}, rows are {len(df)}")

    print("\ndescription-level buckets:")
    for k in ("dependent", "ambiguous", "invariant"):
        n = buckets.get(k, 0)
        print(f"  {k:10s} {n:5d}   ({n / total:6.1%})")
    inv = buckets.get("invariant", 0)
    print(f"\nP2-7 assessment: invariant {inv} / {total} = {inv / total:.1%} -> "
          + ("CONFIRMED (< 5%)" if inv / total < 0.05 else "MISSED (>= 5%)"))
    if inv < 50:
        print("  -> below the 50-description floor registered in advance: "
              "within-TRACE DiD declared TOO THIN pending human check of the "
              "sample; recorded-finding path applies.")

    print("\ntop dependent-vocabulary hits (word: descriptions containing it):")
    for w, n in dep_word_freq.most_common(12):
        print(f"  {w:20s} {n:5d}")
    print("top ambiguous-vocabulary hits:")
    for w, n in amb_word_freq.most_common(8):
        print(f"  {w:20s} {n:5d}")

    # ------------------------------------------------ examples + sample sheet
    rng = random.Random(args.seed)
    print("\nexamples (2 per bucket, flagged words listed):")
    for b in ("dependent", "ambiguous", "invariant"):
        pool = [r for r in per_row if r["bucket"] == b]
        for r in rng.sample(pool, min(2, len(pool))):
            d = df.iloc[r["row_idx"]]["description"]
            text = generate_dsp(d)
            print(f"\n[{b.upper()} | row {r['row_idx']} | dep={r['dep_hits']} "
                  f"amb={r['amb_hits']}]")
            print("   " + text[:400].replace("\n", " "))

    strata = {"dependent": 16, "ambiguous": 8, "invariant": 16}
    SAMPLE.parent.mkdir(parents=True, exist_ok=True)
    with open(SAMPLE, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["row_idx", "bucket", "dep_hits", "amb_hits",
                    "description", "human_verdict (dep/inv/amb)", "note"])
        for b, k in strata.items():
            pool = [r for r in per_row if r["bucket"] == b]
            take = rng.sample(pool, min(k, len(pool)))
            if len(pool) < k:
                print(f"[sample] NOTE: bucket {b} has only {len(pool)} rows; "
                      f"sample takes all of them")
            for r in take:
                text = generate_dsp(df.iloc[r["row_idx"]]["description"])
                w.writerow([r["row_idx"], r["bucket"],
                            ";".join(r["dep_hits"]), ";".join(r["amb_hits"]),
                            text, "", ""])
    print(f"\nhuman-check sample ({sum(min(strata[b], buckets.get(b, 0)) for b in strata)} rows, "
          f"seed {args.seed}) -> {SAMPLE}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({
            "rule": "truth-conditional, 3-way (binding 2026-08-09)",
            "split": args.split, "rows": len(df),
            "buckets": dict(buckets),
            "dependent_patterns": DEPENDENT_PATTERNS,
            "ambiguous_patterns": AMBIGUOUS_PATTERNS,
            "dep_word_freq": dict(dep_word_freq),
            "amb_word_freq": dict(amb_word_freq),
            "per_row": per_row,
        }, f, indent=2, ensure_ascii=False)
    print(f"full detail -> {OUT}")
    print("\nREMINDER: vocabulary is v1, count-only. No group is certified until "
          "the sample sheet has been human-judged (sample-to-screen, census-to-certify).")


if __name__ == "__main__":
    main()

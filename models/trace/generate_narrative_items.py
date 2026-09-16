#!/usr/bin/env python3
"""
generate_narrative_items.py — build the TRACE narrative-level Probe-1 item
set (option (b), grammar N1-N5 ratified 2026-08-08).

v2 (2026-08-08, after validation round 1: 50 items judged, 10 internal-
consistency defects — N1 1/10, N3 4/10, N4 5/10, N2 and N5 clean 10/10):
  * N1: contradiction check extended (incl. Humid+Dry and cross-pair
    combinations) and applied to the POST-SWAP set as well (flag I4).
  * N3: new exclusions for checkable references — month names, seasons,
    day-parts, and trend-verbs pinned to values (flags I25/I27/I29/I30).
  * N4: evidence-clause blocks for both carriers — pinned values, min/max
    ranges, drop/dip/surge mentions, 'trend' (flags I34-I42) — plus a
    PRE-COMMITTED drop rule: a direction with <50 clean rows is dropped;
    a total pool <100 drops N4 entirely, reported as unbuildable-clean.
  * Per-component RNGs: rule changes in one component no longer shift
    another component's samples. N2/N5 rules are unchanged and carry
    their round-1 validation at the rule level; the fresh validation
    sheet covers the changed components (N1/N3/N4) by default.

Each item is a BINARY FORCED CHOICE over one fixed NOAA test signal:

    signal + { correct description , distractor description }

    condition="swap"    distractor = the correct description with ONE
                        component perturbed, otherwise character-identical.
    condition="random"  distractor = the full description of an unrelated
                        row (different US state AND different duration
                        class where satisfiable).

The description text is the EXACT string TRACE retrieves by, rebuilt with
the authors' own template (generate_dsp, read from src/data/load_data.py
2026-08-08):

    "Weather time series location: {location} Time range: {DATE} The
     weather is {labels}. {temperature} \n {precipitation} \n
     {relative_humidity} \n {visibility} \n {wind_u} \n {wind_v} \n
     {sky_code}"

COMPONENTS
  N1 labels     antonym swap inside 'The weather is {labels}.'
                Pairs: Hot<->Cold, Warm<->Cool, Clear<->Cloudy, Rainy<->Dry.
                Rows containing both members of any pair (internally
                spanning sets, e.g. six-month Hot+Cold) are excluded.
  N2 date       temporal extent swap, week <-> six months, as a WHOLE-SLOT
                date-consistent rewrite (end date kept, start recomputed,
                canonical phrasing) so the swapped text is internally
                coherent and wrong only w.r.t. the signal. 28-day rows
                excluded (kept binary), counted.
  N3 trend      direction swap in the temperature field only, by
                simultaneous antonym replacement (incl. the ratified
                comparatives family). Rows with mixed polarity in the
                field are excluded. NUMERIC-LEAK EXCLUSION (design note
                below): rows whose numbers reveal direction ("starting
                from X ... peaking at Y") are excluded, since the swapped
                text would be detectably inconsistent from text alone.
  N4 flux       fluctuation<->stability swap on a DESIGNATED CARRIER:
                temperature (flux->stable) and visibility (stable->flux),
                per the coverage scan. Rows whose carrier also contains
                spike/outlier words are excluded (those words have no
                slot-preserving antonym; polarity would flip only partly).
  N5 location   whole-field donor swap to a different US state. Negative
                control: location is not inferable from the signal, so an
                aligned model should show NO degradation here.

DESIGN NOTES (decisions surfaced during construction, disclosed not buried)
  * Word-surgery over donor-field swaps for N3/N4: a donor field would be
    grammatical by construction but changes numbers and phrasing wholesale
    — and since the original channel prose rides on the SIGNAL side, a
    donor swap would maximally confound the text-overlap shortcut. Minimal
    edits isolate the component; the grammar risk of word surgery is
    handled by exclusion rules plus the mandatory validation sample.
  * The numeric-leak exclusion (N3) and spike-word exclusion (N4) follow
    Probe-1 discipline: drop what cannot be swapped cleanly and report the
    count; never loosen a pattern to inflate coverage.
  * Swapped label sets can be rare in the corpus (Cool: 3 occurrences).
    Distributional oddity of swapped tokens is annotated per item, not
    hidden.

Run from the thesis repo root:
    python models/trace/generate_narrative_items.py --trace-repo ../TRACE-Multimodal-TSEncoder
    optional: --per-component 400 --seed 42

Writes: data/processed/narrative_probe_items.jsonl
        results/analysis/narrative_items_generation_report.json
        results/analysis/narrative_items_validation_sample.csv
"""

from __future__ import annotations
import argparse
import csv
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

OUT_ITEMS = Path("data/processed/narrative_probe_items.jsonl")
OUT_REPORT = Path("results/analysis/narrative_items_generation_report.json")
OUT_VALID = Path("results/analysis/narrative_items_validation_sample.csv")

CHANNEL_ORDER = ["temperature", "precipitation", "relative_humidity",
                 "visibility", "wind_u", "wind_v", "sky_code"]


def fail(gate: str, msg: str):
    print(f"\n[GATE FAILED] {gate}: {msg}")
    print("Hard stop. Investigate before proceeding — do not work around.")
    sys.exit(1)


# ------------------------------------------------------------ template
def generate_dsp(d: dict) -> str:
    """Verbatim reimplementation of the authors' serializer
    (src/data/load_data.py::generate_dsp, read 2026-08-08)."""
    return (f"Weather time series location: {d['location']} "
            f"Time range: {d['DATE']} "
            f"The weather is {d['labels']}. "
            f"{d['temperature']} \n {d['precipitation']} \n "
            f"{d['relative_humidity']} \n {d['visibility']} \n "
            f"{d['wind_u']} \n {d['wind_v']} \n {d['sky_code']}")


# ------------------------------------------------- simultaneous word swap
def simultaneous_swap(text: str, mapping: dict) -> tuple[str, list]:
    """Replace every whole-word occurrence of each mapping key by its value,
    simultaneously (no chaining), case-preserving on first letter.
    Returns (new_text, [(from, to), ...] actually applied)."""
    applied = []
    keys = sorted(mapping, key=len, reverse=True)
    pattern = re.compile(r"\b(" + "|".join(re.escape(k) for k in keys) + r")\b",
                         re.IGNORECASE)

    def repl(m):
        src = m.group(0)
        tgt = mapping[src.lower()]
        if src[0].isupper():
            tgt = tgt[0].upper() + tgt[1:]
        applied.append((src, tgt))
        return tgt

    return pattern.sub(repl, text), applied


def stem_swap(text: str, stem_pairs: list) -> tuple[str, list]:
    """Suffix-preserving stem swap, e.g. increas<->decreas covers
    increased/increasing/increases. Simultaneous via markers; case of the
    first letter is carried through the marker as U/L."""
    applied = []
    out = text
    marks = []
    for i, (a, b) in enumerate(stem_pairs):
        for j, (src, tgt) in enumerate(((a, b), (b, a))):
            mark = f"\x00S{i}_{j}_"
            def mk(m, tgt=tgt, mark=mark):
                head, suffix = m.group(1), m.group(2)
                t = tgt[0].upper() + tgt[1:] if head[0].isupper() else tgt
                applied.append((head + suffix, t + suffix))
                return mark + ("U" if head[0].isupper() else "L") + suffix
            out = re.sub(r"\b(" + src + r")(\w*)", mk, out, flags=re.IGNORECASE)
            marks.append((mark, tgt))
    for mark, tgt in marks:
        out = out.replace(mark + "U", tgt[0].upper() + tgt[1:])
        out = out.replace(mark + "L", tgt)
    return out, applied


# ------------------------------------------------------------ N-builders
ANTONYM_LABEL_PAIRS = [("Hot", "Cold"), ("Warm", "Cool"),
                       ("Clear", "Cloudy"), ("Rainy", "Dry")]
# v2 (2026-08-08, after validation round 1 found Dry+Humid in a swapped set):
# contradictions are checked on BOTH the original and the post-swap set,
# against this fuller list — not just the four swap pairs.
CONTRADICTION_SETS = [{"hot", "cold"}, {"warm", "cool"}, {"hot", "cool"},
                      {"warm", "cold"}, {"clear", "cloudy"},
                      {"rainy", "dry"}, {"humid", "dry"}]


def has_contradiction(tokens: set) -> bool:
    return any(pair <= tokens for pair in CONTRADICTION_SETS)

N3_FORM_PAIRS = {  # exact surface forms, both directions installed below
    "rising": "declining", "rise": "decline", "rises": "declines",
    "rose": "fell", "risen": "declined",
    "upward": "downward", "warming": "cooling",
    "warmer": "colder", "hotter": "cooler",
    "falling": "rising", "fell": "rose",
    "climb": "drop", "climbs": "drops", "climbed": "dropped",
    "climbing": "dropping",
}
N3_UP = [r"\brising\b", r"\brose\b", r"\bincreas\w*", r"\bupward\b",
         r"\bwarming\b", r"\bclimb\w*", r"\bwarmer\b", r"\bhotter\b"]
N3_DOWN = [r"\bdeclin\w*", r"\bdecreas\w*", r"\bdownward\b", r"\bfalling\b",
           r"\bfell\b", r"\bdropp?\w*", r"\bcooling\b", r"\bcolder\b",
           r"\bcooler\b"]
N3_NUMERIC_LEAK = re.compile(
    r"\bstart\w*\s+(from|at|around)\b|\bbegan\b|\bbeginning\s+(at|around)\b|"
    r"\binitial\w*\b", re.IGNORECASE)
# v2 exclusions (validation round 1, flags I25/I27/I29/I30): trend claims
# anchored to checkable references become detectably wrong from text alone
# once flipped. Excluded: month names, season words, day-part words, and
# trend-verbs pinned directly to a value ("climbing to -0.15°C").
N3_CHECKABLE_REF = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September|"
    r"October|November|December|summer|winter|spring|autumn|fall|seasonal|"
    r"night|nights|nighttime|morning|mornings|afternoon|afternoons|"
    r"evening|evenings|diurnal|daytime)\b", re.IGNORECASE)
N3_TREND_TO_VALUE = re.compile(
    r"\b(ris\w*|rose|climb\w*|dropp?\w*|declin\w*|increas\w*|decreas\w*|"
    r"fell|falling)\s+(to|towards|toward)\s+(a\s+)?(low|high|lows|highs|"
    r"around|approximately)?\s*(of\s+)?-?\d", re.IGNORECASE)

N4_TEMP_MAP = {  # flux -> stable, slot-preserving
    "fluctuated": "remained steady", "fluctuating": "steady",
    "fluctuations": "steady conditions", "fluctuation": "steadiness",
    "variability": "steadiness", "variable": "steady",
    "varied": "remained steady", "volatility": "steadiness",
    "volatile": "steady",
}
N4_TEMP_BLOCK = re.compile(r"\bspike\w*|\boutlier\w*|\bgust\w*|\bstable\b|"
                           r"\bsteady\b|\bconsistent\w*|\bconstant\w*",
                           re.IGNORECASE)
# v2 (validation round 1, flags I34-I42): evidence clauses betray a flipped
# adjective. Blocked for BOTH carriers: pinned single values ("at around
# 16.09 km"), explicit min/max ranges, and drop/dip/surge mentions. Values
# cited as evidence make any polarity flip detectable from text alone.
N4_EVIDENCE_BLOCK = re.compile(
    r"\b(at|around|approximately)\s+-?\d|"
    r"\branged?\s+from\b|\ba\s+low\s+of\b|\ba\s+high\s+of\b|"
    r"\bdropp?\w*|\bdip\w*|\bsurge\w*|\bplung\w*|\bsoar\w*|"
    r"\btrend\w*\b", re.IGNORECASE)
# Pre-committed BEFORE regeneration (2026-08-08): if a carrier-direction
# yields < N4_MIN_PER_DIRECTION clean rows it is dropped; if the total N4
# pool is < N4_MIN_TOTAL, N4 is dropped entirely and reported as
# unbuildable-clean on this corpus. Do not lower these after seeing counts.
N4_MIN_PER_DIRECTION = 50
N4_MIN_TOTAL = 100
N4_VIS_MAP = {  # stable -> flux
    "stable": "volatile", "consistently": "erratically",
    "consistent": "erratic", "steady": "erratic", "steadily": "erratically",
    "constant": "erratic", "unchanged": "erratic",
}
N4_VIS_BLOCK = re.compile(r"\bfluctuat\w*|\bvolatil\w*|\bvariab\w*|"
                          r"\bvaried\b|\bvarying\b|\bspike\w*|\boutlier\w*",
                          re.IGNORECASE)

US_STATES = ["Alabama", "Alaska", "Arizona", "Arkansas", "California",
             "Colorado", "Connecticut", "Delaware", "Florida", "Georgia",
             "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", "Kansas",
             "Kentucky", "Louisiana", "Maine", "Maryland", "Massachusetts",
             "Michigan", "Minnesota", "Mississippi", "Missouri", "Montana",
             "Nebraska", "Nevada", "New Hampshire", "New Jersey",
             "New Mexico", "New York", "North Carolina", "North Dakota",
             "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island",
             "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah",
             "Vermont", "Virginia", "Washington", "West Virginia",
             "Wisconsin", "Wyoming"]
STATE_RE = re.compile("|".join(sorted((s for s in US_STATES), key=len,
                                      reverse=True)), re.IGNORECASE)

MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]
MONTH_IDX = {m.lower(): i + 1 for i, m in enumerate(MONTHS)}

WEEK_RE = re.compile(r"past (week|seven days)", re.IGNORECASE)
SIXM_RE = re.compile(r"past six months", re.IGNORECASE)
D28_RE = re.compile(r"past 28 days|past month", re.IGNORECASE)
FROMTO_RE = re.compile(r"from\s+(.+?)\s+to\s+(.+?)\.?\s*$", re.IGNORECASE)
DAY_DATE_RE = re.compile(r"^([A-Za-z]+)\s+(\d{1,2})(?:,\s*(\d{4}))?,?$")
MY_DATE_RE = re.compile(r"^([A-Za-z]+)\s+(\d{4})$")


def parse_end_date(s: str, other: str):
    """Return (month, day_or_None, year). Year may come from `other` side."""
    s = s.strip().rstrip(".,")
    m = DAY_DATE_RE.match(s)
    if m and m.group(1).lower() in MONTH_IDX:
        year = m.group(3)
        if year is None:
            y = re.search(r"(\d{4})", other)
            year = y.group(1) if y else None
        return (MONTH_IDX[m.group(1).lower()], int(m.group(2)),
                int(year) if year else None)
    m = MY_DATE_RE.match(s)
    if m and m.group(1).lower() in MONTH_IDX:
        return (MONTH_IDX[m.group(1).lower()], None, int(m.group(2)))
    return None


def month_shift(month: int, year: int, delta_months: int):
    idx = (month - 1) + delta_months
    # Python floor division is correct for negative idx: -5 // 12 == -1
    return (idx % 12) + 1, year + (idx // 12)


def build_n2_slot(end, to_six: bool):
    """end = (month, day, year). Return canonical rewritten DATE slot."""
    month, day, year = end
    if year is None:
        return None
    if to_six:
        sm, sy = month_shift(month, year, -6)
        if day is None:
            start = f"{MONTHS[sm - 1]} {sy}"
            endtxt = f"{MONTHS[month - 1]} {year}"
        else:
            start = f"{MONTHS[sm - 1]} {day}, {sy}"
            endtxt = f"{MONTHS[month - 1]} {day}, {year}"
        return f"The past six months from {start} to {endtxt}."
    else:
        d = day if day is not None else 28
        sd = d - 7
        if sd >= 1:
            start = f"{MONTHS[month - 1]} {sd}"
            endtxt = f"{MONTHS[month - 1]} {d}, {year}"
        else:
            pm, py = month_shift(month, year, -1)
            days_prev = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][pm - 1]
            start = f"{MONTHS[pm - 1]} {days_prev + sd}"
            endtxt = f"{MONTHS[month - 1]} {d}, {year}"
        return f"The past week from {start} to {endtxt}."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-repo", required=True)
    ap.add_argument("--per-component", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-validate", type=int, default=10,
                    help="validation sample size per component")
    ap.add_argument("--validate-components", default="N1,N3,N4",
                    help="comma list; v2 default covers only the components "
                         "whose rules changed after round 1 (N2/N5 carry "
                         "their round-1 validation)")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    import pyarrow.parquet as pq
    repo = Path(args.trace_repo).resolve()
    parquet = repo / "dataset" / "retrieval" / "test" / "test.parquet"
    if not parquet.is_file():
        fail("G1", f"{parquet} missing")
    df = pq.read_table(parquet).to_pandas()
    if len(df) != 2006:
        fail("G1-count", f"expected 2006 rows, got {len(df)}")

    rows = []
    for i in range(len(df)):
        d = df.iloc[i]["description"]
        if not (isinstance(d, dict)
                and all(isinstance(d.get(k), str) and d[k].strip()
                        for k in CHANNEL_ORDER + ["DATE", "labels", "location"])):
            fail("G1-fields", f"row {i} lacks a complete description dict")
        rows.append({"row_idx": i, "pq_id": int(df.iloc[i]["id"]),
                     "fields": dict(d)})
    print(f"[G1] {len(rows)} rows, all 10 fields present")

    # ---- G2: template fidelity ------------------------------------------
    for r in rows[:50] + rows[-50:]:
        full = generate_dsp(r["fields"])
        for k in CHANNEL_ORDER + ["DATE", "labels", "location"]:
            if r["fields"][k] not in full:
                fail("G2-template", f"field {k} not a substring of the "
                                    f"rebuilt description (row {r['row_idx']})")
    print("[G2] template reconstruction contains every field verbatim "
          "(checked 100 rows)")
    print("[G2] example rebuilt description (row 0, first 300 chars):")
    print("     " + generate_dsp(rows[0]["fields"])[:300].replace("\n", "\\n"))

    # ---- per-row metadata for random-condition matching ------------------
    for r in rows:
        date = r["fields"]["DATE"]
        r["dur"] = ("week" if WEEK_RE.search(date) else
                    "six_months" if SIXM_RE.search(date) else
                    "28_days" if D28_RE.search(date) else "other")
        sm = STATE_RE.search(r["fields"]["location"])
        r["state"] = sm.group(0).title() if sm else None

    dur_counts = Counter(r["dur"] for r in rows)
    state_none = sum(1 for r in rows if r["state"] is None)
    print(f"[meta] duration classes: {dict(dur_counts)}; "
          f"rows without recognised state: {state_none}")

    # ---- component builders ---------------------------------------------
    skipped = Counter()
    candidates = defaultdict(list)   # component -> list of item dicts (swap only)

    # ---------------- N1 labels
    n1_map = {}
    for a, b in ANTONYM_LABEL_PAIRS:
        n1_map[a.lower()] = b
        n1_map[b.lower()] = a
    for r in rows:
        raw = r["fields"]["labels"]
        toks = {t.strip().strip("{}").lower() for t in raw.split(",")}
        if has_contradiction(toks):
            skipped["N1:internal_contradiction"] += 1
            continue
        swappable = [t for t in toks if t in n1_map]
        if not swappable:
            skipped["N1:no_swappable_token"] += 1
            continue
        new_labels, applied = simultaneous_swap(raw, n1_map)
        if new_labels == raw:
            skipped["N1:identical"] += 1
            continue
        new_toks = {t.strip().strip("{}").lower() for t in new_labels.split(",")}
        if has_contradiction(new_toks):
            skipped["N1:post_swap_contradiction"] += 1   # v2 gate (round-1 flag I4)
            continue
        candidates["N1"].append(dict(
            r=r, slot="labels", old_field=raw, new_field=new_labels,
            swap_from=";".join(a for a, _ in applied),
            swap_to=";".join(b for _, b in applied),
            note=f"{len(applied)} token(s) swapped"))

    # ---------------- N2 date
    for r in rows:
        if r["dur"] == "28_days":
            skipped["N2:28day_excluded"] += 1
            continue
        if r["dur"] == "other":
            skipped["N2:unrecognised_duration"] += 1
            continue
        raw = r["fields"]["DATE"]
        m = FROMTO_RE.search(raw)
        if not m:
            skipped["N2:no_from_to"] += 1
            continue
        end = parse_end_date(m.group(2), m.group(1))
        if end is None or end[2] is None:
            skipped["N2:end_date_unparsed"] += 1
            continue
        to_six = (r["dur"] == "week")
        new_slot = build_n2_slot(end, to_six=to_six)
        if new_slot is None:
            skipped["N2:rewrite_failed"] += 1
            continue
        candidates["N2"].append(dict(
            r=r, slot="DATE", old_field=raw, new_field=new_slot,
            swap_from=r["dur"],
            swap_to="six_months" if to_six else "week",
            note="whole-slot rewrite, end date kept"))

    # ---------------- N3 trend (temperature field)
    up_c = [re.compile(p, re.IGNORECASE) for p in N3_UP]
    down_c = [re.compile(p, re.IGNORECASE) for p in N3_DOWN]
    n3_map = dict(N3_FORM_PAIRS)
    n3_map.update({v.lower(): k for k, v in N3_FORM_PAIRS.items()
                   if v.lower() not in n3_map})
    for r in rows:
        t = r["fields"]["temperature"]
        u = any(c.search(t) for c in up_c)
        v = any(c.search(t) for c in down_c)
        if u and v:
            skipped["N3:mixed_polarity"] += 1
            continue
        if not (u or v):
            skipped["N3:no_trend_phrase"] += 1
            continue
        if N3_NUMERIC_LEAK.search(t):
            skipped["N3:numeric_leak"] += 1
            continue
        if N3_CHECKABLE_REF.search(t):
            skipped["N3:checkable_reference"] += 1   # v2 gate (round-1 flags)
            continue
        if N3_TREND_TO_VALUE.search(t):
            skipped["N3:trend_pinned_to_value"] += 1  # v2 gate (flag I30)
            continue
        new_t, applied_forms = simultaneous_swap(t, n3_map)
        new_t2, applied_stems = stem_swap(new_t, [("increas", "decreas")])
        applied = applied_forms + applied_stems
        if new_t2 == t or not applied:
            skipped["N3:no_replacement_applied"] += 1
            continue
        candidates["N3"].append(dict(
            r=r, slot="temperature", old_field=t, new_field=new_t2,
            swap_from="up" if u else "down",
            swap_to="down" if u else "up",
            note="; ".join(f"{a}->{b}" for a, b in applied[:6])))

    # ---------------- N4 flux
    for r in rows:
        t = r["fields"]["temperature"]
        if any(re.search(r"\b" + k + r"\b", t, re.IGNORECASE) for k in N4_TEMP_MAP):
            if N4_TEMP_BLOCK.search(t) or N4_EVIDENCE_BLOCK.search(t):
                skipped["N4:temp_blocked_words"] += 1
            else:
                new_t, applied = simultaneous_swap(t, N4_TEMP_MAP)
                if applied and new_t != t:
                    candidates["N4"].append(dict(
                        r=r, slot="temperature", old_field=t, new_field=new_t,
                        swap_from="fluctuating", swap_to="stable",
                        note="; ".join(f"{a}->{b}" for a, b in applied[:6])))
                else:
                    skipped["N4:temp_no_replacement"] += 1
        else:
            skipped["N4:temp_no_flux_word"] += 1
        vfield = r["fields"]["visibility"]
        if any(re.search(r"\b" + k + r"\b", vfield, re.IGNORECASE) for k in N4_VIS_MAP):
            if N4_VIS_BLOCK.search(vfield) or N4_EVIDENCE_BLOCK.search(vfield):
                skipped["N4:vis_blocked_words"] += 1
            else:
                new_v, applied = simultaneous_swap(vfield, N4_VIS_MAP)
                if applied and new_v != vfield:
                    candidates["N4"].append(dict(
                        r=r, slot="visibility", old_field=vfield, new_field=new_v,
                        swap_from="stable", swap_to="fluctuating",
                        note="; ".join(f"{a}->{b}" for a, b in applied[:6])))
                else:
                    skipped["N4:vis_no_replacement"] += 1
        else:
            skipped["N4:vis_no_stable_word"] += 1

    # ---------------- N5 location (donor swap, negative control)
    staterows = [r for r in rows if r["state"]]
    n5rng = random.Random(f"{args.seed}:N5donor")   # v2: decoupled
    for r in rows:
        if not r["state"]:
            skipped["N5:no_state_recognised"] += 1
            continue
        donors = [d for d in staterows if d["state"] != r["state"]]
        if not donors:
            skipped["N5:no_donor"] += 1
            continue
        donor = n5rng.choice(donors)
        candidates["N5"].append(dict(
            r=r, slot="location", old_field=r["fields"]["location"],
            new_field=donor["fields"]["location"],
            swap_from=r["state"], swap_to=donor["state"],
            note=f"donor row {donor['row_idx']}"))

    # ---- v2: pre-committed N4 drop rule (do not lower after seeing counts)
    n4_by_dir = Counter(c["swap_from"] + "->" + c["swap_to"]
                        for c in candidates["N4"])
    print(f"\n[N4] clean pool by direction: {dict(n4_by_dir)}")
    for direction, n in list(n4_by_dir.items()):
        if n < N4_MIN_PER_DIRECTION:
            candidates["N4"] = [c for c in candidates["N4"]
                                if c["swap_from"] + "->" + c["swap_to"] != direction]
            skipped[f"N4:direction_dropped:{direction}"] += n
            print(f"[N4] direction {direction} dropped: {n} < "
                  f"{N4_MIN_PER_DIRECTION} (pre-committed rule)")
    if len(candidates["N4"]) < N4_MIN_TOTAL:
        skipped["N4:component_dropped_total"] += len(candidates["N4"])
        print(f"[N4] COMPONENT DROPPED: total clean pool "
              f"{len(candidates['N4'])} < {N4_MIN_TOTAL} (pre-committed rule). "
              f"N4 is reported as unbuildable-clean on this corpus — a "
              f"finding, not a failure.")
        candidates["N4"] = []

    # ---- sample to cap, build items --------------------------------------
    # v2: per-component rngs decouple components — a rule change in one
    # component can no longer shift another component's samples.
    items = []
    length_delta = defaultdict(list)
    originals = {r["row_idx"]: generate_dsp(r["fields"]) for r in rows}
    all_original_texts = set(originals.values())

    for comp in ("N1", "N2", "N3", "N4", "N5"):
        crng = random.Random(f"{args.seed}:{comp}")
        pool = candidates[comp]
        chosen = pool if len(pool) <= args.per_component else \
            crng.sample(pool, args.per_component)
        for c in chosen:
            r = c["r"]
            fields2 = dict(r["fields"])
            fields2[c["slot"]] = c["new_field"]
            swapped = generate_dsp(fields2)
            original = originals[r["row_idx"]]

            # G3: single-slot change — remainder identical after removing slot
            if original.replace(c["old_field"], "\x01", 1) != \
               swapped.replace(c["new_field"], "\x01", 1):
                fail("G3-single-slot", f"{comp} row {r['row_idx']}: swapped "
                     "text differs outside the intended slot")
            if swapped == original:
                skipped[f"{comp}:identical_after_build"] += 1
                continue
            # G4: uniqueness vs any original description
            if swapped in all_original_texts:
                skipped[f"{comp}:collides_with_original"] += 1
                continue

            base = {
                "component": comp,
                "slot": c["slot"],
                "swap_from": c["swap_from"],
                "swap_to": c["swap_to"],
                "sample_id": f"noaa_test_{r['row_idx']}",
                "pq_id": r["pq_id"],
                "split": "noaa_test",
                "duration_class": r["dur"],
                "state": r["state"],
                "caption_correct": original,
            }
            items.append({**base,
                          "item_id": f"{comp}|{r['row_idx']}|swap",
                          "condition": "swap",
                          "caption_distractor": swapped,
                          "clause_replaced_from": c["old_field"],
                          "clause_replaced_to": c["new_field"],
                          "swap_note": c["note"]})
            length_delta[comp].append(
                len(swapped.split()) - len(original.split()))

            # matched random control
            cands = [x for x in rows
                     if x["row_idx"] != r["row_idx"]
                     and x["dur"] != r["dur"]
                     and x["state"] and r["state"] and x["state"] != r["state"]]
            if not cands:
                cands = [x for x in rows if x["row_idx"] != r["row_idx"]]
                skipped[f"{comp}:random_constraint_relaxed"] += 1
            rr = crng.choice(cands)
            items.append({**base,
                          "item_id": f"{comp}|{r['row_idx']}|random",
                          "condition": "random",
                          "caption_distractor": originals[rr["row_idx"]],
                          "clause_replaced_from": None,
                          "clause_replaced_to": None,
                          "swap_note": f"random distractor row {rr['row_idx']}"})

    # ---- G5: arithmetic closes -------------------------------------------
    print("\n" + "=" * 74)
    print("GENERATION REPORT")
    print("=" * 74)
    n_swap = Counter(i["component"] for i in items if i["condition"] == "swap")
    n_rand = Counter(i["component"] for i in items if i["condition"] == "random")
    print(f"{'component':<6}{'eligible':>10}{'swap items':>12}{'random':>9}"
          f"{'mean len delta':>16}")
    for comp in ("N1", "N2", "N3", "N4", "N5"):
        deltas = length_delta[comp]
        md = sum(deltas) / len(deltas) if deltas else float("nan")
        print(f"{comp:<6}{len(candidates[comp]):>10}{n_swap[comp]:>12}"
              f"{n_rand[comp]:>9}{md:>16.2f}")
    print(f"\ntotal items: {len(items)} "
          f"({sum(n_swap.values())} swap + {sum(n_rand.values())} random)")
    print("\nskip reasons:")
    for k, v in sorted(skipped.items()):
        print(f"    {k:<40} {v}")
    uniq_rows = len({i["sample_id"] for i in items})
    print(f"\nunique signals used: {uniq_rows} "
          f"(items per signal: {len(items) / uniq_rows:.1f})")

    # ---- write ------------------------------------------------------------
    OUT_ITEMS.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_ITEMS, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        json.dump({"seed": args.seed, "per_component_cap": args.per_component,
                   "eligible": {c: len(candidates[c]) for c in candidates},
                   "n_items": len(items),
                   "swap_items": dict(n_swap), "random_items": dict(n_rand),
                   "mean_length_delta": {c: (sum(v) / len(v) if v else None)
                                         for c, v in length_delta.items()},
                   "skips": dict(skipped),
                   "duration_classes": dict(dur_counts)}, f, indent=2)

    # validation sample: n per component, swap condition only
    swap_items = [i for i in items if i["condition"] == "swap"]
    with open(OUT_VALID, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "component", "slot", "replaced_from",
                    "replaced_to", "grammatical? (y/n)",
                    "internally_consistent? (y/n)",
                    "polarity_flipped? (y/n)", "notes"])
        val_comps = [c.strip() for c in args.validate_components.split(",") if c.strip()]
        for comp in val_comps:
            pool = [i for i in swap_items if i["component"] == comp]
            if not pool:
                continue
            for it in random.Random(f"{args.seed}:val:{comp}").sample(pool, min(args.n_validate, len(pool))):
                w.writerow([it["item_id"], comp, it["slot"],
                            it["clause_replaced_from"],
                            it["clause_replaced_to"], "", "", "", ""])
    print(f"\n[done] items -> {OUT_ITEMS}")
    print(f"[done] report -> {OUT_REPORT}")
    print(f"[done] validation sheet -> {OUT_VALID} "
          f"({args.n_validate}/component for {args.validate_components}, "
          f"judge before any run is trusted)")
    print("\nPaste the full console output back. Do not run any model on "
          "these items until the validation sheet has been judged.")


if __name__ == "__main__":
    main()

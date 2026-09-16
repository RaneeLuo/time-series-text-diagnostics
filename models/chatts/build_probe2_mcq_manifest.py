"""
Build the ChatTS Probe-2 MCQ manifest (CPU only, deterministic).
V2 (2026-08-15): fix error #16 — pairs.jsonl datasets are named
'truce_synth' and 'truce_stock', never 'truce'; v1's exact-match filter
silently dropped all 738 TRUCE test rows and the population gate
hard-stopped (nothing was written; zero cost). V2 maps any dataset
starting with 'truce' to the TRUCE substrate and carries the raw
sub-source into every row as 'source' (useful later: the duplicate
-signal clusters live in truce_synth).

One MCQ per test row per answer order, with a FROZEN distractor: the
same distractor caption is used for the unperturbed condition and every
perturbed condition of that row. Conditions are NOT multiplied into the
manifest — the prompt text is identical across conditions; only the
series fed at GPU time differs. The GPU runner loops conditions over
these rows, recomputing perturbed series from seeds (M1-C for masking).

Group labels come from the two CERTIFIED grouping artifacts and are
never recomputed here (binding: groupings are census-certified).

The only randomness is the per-row distractor draw, seeded by
sha256(caption_id|p2mcq|42) — the M3 pattern; reruns are identical.

Gates GP1-GP5, GP7 are HARD STOPS; GP6, GP8 are reports.

Place at: models/chatts/build_probe2_mcq_manifest.py
Run from repo root:
  python models/chatts/build_probe2_mcq_manifest.py --pairs data/processed/pairs.jsonl --truce-groups results/analysis/probe2_truce_groups_certified.json --sushi-groups results/analysis/probe2_sushi_groups.json
"""
import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

SYSTEM_MSG = "You are a helpful assistant."
USER_TEMPLATE = (
    "Here is a time series: <ts><ts/>.\n"
    "Which description matches this time series?\n"
    "(A) {option_a}\n"
    "(B) {option_b}\n"
    "Answer with exactly one letter, A or B."
)
FULL_TEMPLATE = (
    "<|im_start|>system\n" + SYSTEM_MSG + "<|im_end|>\n"
    "<|im_start|>user\n" + USER_TEMPLATE + "<|im_end|>\n"
    "<|im_start|>assistant\n"
)
TEMPLATE_SHA = hashlib.sha256(FULL_TEMPLATE.encode("utf-8")).hexdigest()[:12]
EXPECTED_TEMPLATE_SHA = "4029f94e2f6d"  # must equal the Probe-1 manifest's template

CONDITIONS = {
    "truce": ["unperturbed", "sf_all", "sf_half", "ex_half", "masking"],
    "sushi": ["unperturbed", "sf_all", "sf_half", "ex_half", "masking",
              "sf_within_patch", "sf_across_patch"],  # two-level extra: SUSHI-only
}
EXPECTED = {
    "truce_rows": 738, "truce_groups": {"dependent": 715, "invariant": 18, "ambiguous": 5},
    "truce_sources": {"truce_stock": 570, "truce_synth": 168},  # dataset.py committed counts
    "sushi_rows": 140, "sushi_groups": {"dependent": 135, "invariant": 4, "degenerate": 1},
}
MIN_POOL_TEXT_LEN = 3  # junk filter for the DISTRACTOR POOL only (e.g. '{}')

FAILURES = []


def substrate_of(dataset_name: str):
    """Map raw dataset names to probe substrates (error #16 fix)."""
    if dataset_name.startswith("truce"):
        return "truce"
    if dataset_name == "sushi":
        return "sushi"
    return None


def gate(name, ok, detail, hard=True):
    status = "PASS" if ok else ("FAIL (HARD)" if hard else "MISS (report)")
    print(f"[{name}] {status} — {detail}")
    if not ok and hard:
        FAILURES.append(name)


def draw_distractor(caption_id: str, own_text: str, pool_sorted: list) -> str:
    seed = int(hashlib.sha256(f"{caption_id}|p2mcq|42".encode("utf-8")).hexdigest()[:8], 16)
    rng = random.Random(seed)
    for _ in range(64):
        cand = pool_sorted[rng.randrange(len(pool_sorted))]
        if cand != own_text:
            return cand
    return None  # gated below


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--truce-groups", required=True)
    ap.add_argument("--sushi-groups", required=True)
    ap.add_argument("--out", default="data/processed/chatts_probe2_mcq.jsonl")
    ap.add_argument("--report", default="results/analysis/chatts_probe2_mcq_report.json")
    args = ap.parse_args()

    print("=== ChatTS Probe-2 MCQ manifest builder (v2) ===")
    print(f"template sha256[:12] = {TEMPLATE_SHA} (must equal Probe-1: {EXPECTED_TEMPLATE_SHA})")
    print("determinism: only the seeded distractor draw uses RNG; seed = sha256(caption_id|p2mcq|42)")

    # ---- load test rows (series column ignored on purpose — never copied) ----
    rows_in = {"truce": [], "sushi": []}
    seen_datasets = Counter()
    with open(args.pairs, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["split"] != "test":
                continue
            seen_datasets[r["dataset"]] += 1
            sub = substrate_of(r["dataset"])
            if sub is not None:
                rows_in[sub].append(
                    {"source": r["dataset"],
                     **{k: r[k] for k in ("sample_id", "caption_id",
                                          "caption", "class_label")}})
    print(f"[input] test rows by raw dataset name: {dict(seen_datasets)}")

    src_c = Counter(r["source"] for r in rows_in["truce"])
    gate("GP1-counts",
         len(rows_in["truce"]) == EXPECTED["truce_rows"]
         and len(rows_in["sushi"]) == EXPECTED["sushi_rows"],
         f"test rows: truce={len(rows_in['truce'])} (exp {EXPECTED['truce_rows']}), "
         f"sushi={len(rows_in['sushi'])} (exp {EXPECTED['sushi_rows']})")
    gate("GP1-sources", dict(src_c) == EXPECTED["truce_sources"],
         f"truce sub-sources observed={dict(src_c)} expected={EXPECTED['truce_sources']}")

    # ---- group labels from the certified artifacts (hard join, GBK lesson) ----
    tg = json.load(open(args.truce_groups, encoding="utf-8"))["per_text"]
    sg = json.load(open(args.sushi_groups, encoding="utf-8"))["class_verdicts"]

    unmatched = []
    for r in rows_in["truce"]:
        rec = tg.get(r["caption"])
        if rec is None:
            unmatched.append(r["caption_id"])
            r["group"] = None
        else:
            r["group"] = rec["label"]
    for r in rows_in["sushi"]:
        v = sg.get(r["class_label"])
        if v is None:
            unmatched.append(r["caption_id"])
        r["group"] = v
    gate("GP2-join", not unmatched,
         f"rows with no certified group label: {len(unmatched)} (expected 0)"
         + (f"; first: {unmatched[:3]}" if unmatched else ""))

    tc = Counter(r["group"] for r in rows_in["truce"])
    sc = Counter(r["group"] for r in rows_in["sushi"])
    gate("GP2-truce-groups", dict(tc) == EXPECTED["truce_groups"],
         f"truce groups observed={dict(tc)} expected={EXPECTED['truce_groups']}")
    gate("GP3-sushi-groups", dict(sc) == EXPECTED["sushi_groups"],
         f"sushi groups observed={dict(sc)} expected={EXPECTED['sushi_groups']}")

    # ---- distractor pools (per substrate; junk filtered from POOL only) ----
    report_pools = {}
    pools = {}
    for ds in ("truce", "sushi"):
        texts = sorted({r["caption"] for r in rows_in[ds]})
        junk = [t for t in texts if len(t.strip()) < MIN_POOL_TEXT_LEN]
        pools[ds] = [t for t in texts if len(t.strip()) >= MIN_POOL_TEXT_LEN]
        report_pools[ds] = {"unique_test_texts": len(texts),
                            "junk_excluded": junk,
                            "pool_size": len(pools[ds])}
        print(f"[GP8-report] {ds}: unique test caption texts={len(texts)}, "
              f"junk excluded from pool={junk if junk else 'none'}, "
              f"pool size={len(pools[ds])}")

    # ---- build rows: 2 orders per test row, frozen distractor ----
    out_rows = []
    dist_group = Counter()
    for ds in ("truce", "sushi"):
        for r in rows_in[ds]:
            distractor = draw_distractor(r["caption_id"], r["caption"], pools[ds])
            if distractor is None:
                gate("GP4-draw", False,
                     f"could not draw a differing distractor for {r['caption_id']}")
                continue
            if ds == "truce":
                dist_group[tg[distractor]["label"] if distractor in tg else "unknown"] += 1
            for order in ("corrA", "corrB"):
                a, b, letter = ((r["caption"], distractor, "A") if order == "corrA"
                                else (distractor, r["caption"], "B"))
                out_rows.append({
                    "mcq_id": f"p2|{r['caption_id']}|{order}",
                    "caption_id": r["caption_id"],
                    "sample_id": r["sample_id"],
                    "dataset": ds,
                    "source": r["source"] if ds == "truce" else "sushi",
                    "group": r["group"],
                    "order": order,
                    "option_a": a,
                    "option_b": b,
                    "correct_letter": letter,
                    "distractor_frozen": distractor,
                    "conditions": CONDITIONS[ds],
                    "template_sha": TEMPLATE_SHA,
                    "prompt": FULL_TEMPLATE.format(option_a=a, option_b=b),
                })

    # ---- GP4/GP5: structure ----
    bad_same = [r["mcq_id"] for r in out_rows if r["option_a"] == r["option_b"]]
    gate("GP4-distinct", not bad_same,
         f"rows where distractor == correct: {len(bad_same)} (expected 0)")
    n_rows = len(out_rows)
    expect_rows = 2 * (EXPECTED["truce_rows"] + EXPECTED["sushi_rows"])
    order_c = Counter(r["order"] for r in out_rows)
    letter_c = Counter(r["correct_letter"] for r in out_rows)
    dup = [k for k, v in Counter(r["mcq_id"] for r in out_rows).items() if v > 1]
    gate("GP5-rows", n_rows == expect_rows
         and order_c["corrA"] == expect_rows // 2 and order_c["corrB"] == expect_rows // 2,
         f"rows={n_rows} (exp {expect_rows}); corrA/corrB="
         f"{order_c['corrA']}/{order_c['corrB']}")
    gate("GP5-letters", letter_c["A"] == expect_rows // 2 and letter_c["B"] == expect_rows // 2,
         f"letters A/B = {letter_c['A']}/{letter_c['B']} (expected exact 50/50)")
    gate("GP5-unique", not dup, f"duplicate mcq_ids: {len(dup)} (expected 0)")

    print(f"[GP6-report] TRUCE distractor group composition: {dict(dist_group)} (descriptive)")
    planned = (EXPECTED["truce_rows"] * 2 * len(CONDITIONS["truce"])
               + EXPECTED["sushi_rows"] * 2 * len(CONDITIONS["sushi"]))
    print(f"[plan] GPU questions = 738*2*{len(CONDITIONS['truce'])} + "
          f"140*2*{len(CONDITIONS['sushi'])} = {planned}")

    gate("GP7-template", TEMPLATE_SHA == EXPECTED_TEMPLATE_SHA,
         f"template sha {TEMPLATE_SHA} == Probe-1's {EXPECTED_TEMPLATE_SHA} "
         f"(one question format across probes)")

    # ---- verdict + write ----
    print("=" * 50)
    if FAILURES:
        print(f"HARD STOP — failed gates: {FAILURES}. Nothing written.")
        sys.exit(1)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    rep_path = Path(args.report)
    rep_path.parent.mkdir(parents=True, exist_ok=True)
    with open(rep_path, "w", encoding="utf-8") as f:
        json.dump({
            "rows_out": n_rows,
            "planned_gpu_questions": planned,
            "conditions": CONDITIONS,
            "groups": {"truce": dict(tc), "sushi": dict(sc)},
            "truce_sources": dict(src_c),
            "pools": report_pools,
            "truce_distractor_groups": dict(dist_group),
            "template_sha": TEMPLATE_SHA,
            "seed_formula": "sha256(caption_id|p2mcq|42)[:8] as int",
            "masking_rule": "M1-C: fill masked positions with survivors' mean "
                            "(exact fill-0 at model input level); prefix-drift "
                            "and scaling-touched counts reported by the GPU runner",
        }, f, indent=2, ensure_ascii=False)
    print(f"ALL HARD GATES GREEN. Wrote {n_rows} rows -> {out_path}")
    print(f"Report -> {rep_path}")


if __name__ == "__main__":
    main()

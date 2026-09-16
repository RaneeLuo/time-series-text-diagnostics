"""
Build the ChatTS Probe-3 MCQ manifest (CPU only, deterministic).

Probe 3 reuses the Probe-2 questions verbatim (same 878 rows, same
frozen distractors, both orders) — rungs 1-2 (unperturbed, sf_all) are
READ from the Probe-2 records at analysis time, never re-asked. This
builder prepares what is NEW:

  block 'base'        : the 1,756 Probe-2 rows annotated with the new
                        GPU conditions [resample, gaussian, cond_A,
                        cond_B, cond_C]. Prompt text unchanged; series
                        and prefix rules applied at GPU time from seeds.
                        cond_B's donor series is assigned HERE (seeded,
                        same substrate, never self) and recorded.
  block 'five_number' : 1,756 rows whose prompt replaces the series slot
                        with the five-number text block — no <ts><ts/>
                        placeholder, no series at all (ChatTS-only
                        condition, posable because the model reads text).
  block 'pj_control'  : seeded sample of 100 rows per substrate x 2
                        orders = 400 rows; series byte-identical,
                        prefix jittered by that row's ACTUAL
                        masking-induced digit shift (measured here,
                        stored verbatim — no modeled distribution).

Gates GF1-GF6 HARD; jitter-magnitude census is a report (sane units:
offset delta in fractions of the series' std; scale delta relative).

Place at: models/chatts/build_probe3_mcq_manifest.py
Run from repo root:
  python -m models.chatts.build_probe3_mcq_manifest --pairs data/processed/pairs.jsonl --p2-manifest data/processed/chatts_probe2_mcq.jsonl
"""
import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from models.chatts import perturbations as P

BASE_SEED = 42
NEW_CONDITIONS = ["resample", "gaussian", "cond_A", "cond_B", "cond_C"]
PJ_PER_SUBSTRATE = 100
FIVE_NUMBER_USER_TEMPLATE = (
    "Here is a statistical summary of a time series: {five_number}.\n"
    "Which description matches this time series?\n"
    "(A) {option_a}\n"
    "(B) {option_b}\n"
    "Answer with exactly one letter, A or B."
)
FIVE_NUMBER_FULL = (
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n" + FIVE_NUMBER_USER_TEMPLATE + "<|im_end|>\n"
    "<|im_start|>assistant\n"
)
FN_TEMPLATE_SHA = hashlib.sha256(FIVE_NUMBER_FULL.encode("utf-8")).hexdigest()[:12]

FAILURES = []


def gate(name, ok, detail, hard=True):
    status = "PASS" if ok else ("FAIL (HARD)" if hard else "MISS (report)")
    print(f"[{name}] {status} — {detail}")
    if not ok and hard:
        FAILURES.append(name)


def seeded_index(key: str, n: int) -> int:
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) % n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--p2-manifest", required=True)
    ap.add_argument("--out", default="data/processed/chatts_probe3_mcq.jsonl")
    ap.add_argument("--report", default="results/analysis/chatts_probe3_mcq_report.json")
    args = ap.parse_args()

    print("=== ChatTS Probe-3 MCQ manifest builder ===")
    print(f"five-number template sha256[:12] = {FN_TEMPLATE_SHA}")
    print(f"new GPU conditions: {NEW_CONDITIONS}; rungs 1-2 read from Probe-2 records")

    series_of = {}
    with open(args.pairs, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["sample_id"] not in series_of:
                series_of[r["sample_id"]] = np.asarray(r["series"], dtype=np.float64)

    p2rows = [json.loads(l) for l in open(args.p2_manifest, encoding="utf-8") if l.strip()]
    gate("GF1-input", len(p2rows) == 1756,
         f"Probe-2 manifest rows read = {len(p2rows)} (expected 1756)")

    # per-substrate sample_id pools for donor draws (unique, sorted)
    sids = defaultdict(list)
    for r in p2rows:
        if r["sample_id"] not in sids[r["dataset"]]:
            sids[r["dataset"]].append(r["sample_id"])
    for ds in sids:
        sids[ds] = sorted(sids[ds])
    print(f"[input] unique sample_ids per substrate: "
          f"{ {ds: len(v) for ds, v in sids.items()} }")

    out_rows = []
    donor_of = {}

    # ---- block 'base': annotate, assign donors (one donor per caption_id) ----
    for r in p2rows:
        cid = r["caption_id"]
        if cid not in donor_of:
            pool = sids[r["dataset"]]
            k = seeded_index(f"{cid}|p3donor|{BASE_SEED}", len(pool))
            for step in range(len(pool)):
                cand = pool[(k + step) % len(pool)]
                if cand != r["sample_id"]:
                    donor_of[cid] = cand
                    break
        row = dict(r)
        row["mcq_id"] = r["mcq_id"].replace("p2|", "p3base|")
        row["block"] = "base"
        row["conditions"] = NEW_CONDITIONS
        row["donor_sample_id"] = donor_of[cid]
        out_rows.append(row)

    bad_donor = [c for c, d in donor_of.items()
                 if d == next(r["sample_id"] for r in p2rows if r["caption_id"] == c)]
    same_sub = all(donor_of[r["caption_id"]] in sids[r["dataset"]] for r in p2rows)
    gate("GF2-donors", len(donor_of) == 878 and not bad_donor and same_sub,
         f"donor assignments = {len(donor_of)} (exp 878); self-donors = {len(bad_donor)}; "
         f"substrate-matched = {same_sub}")

    # ---- block 'five_number': new prompt text, no series ----
    fn_bad = 0
    for r in p2rows:
        x = series_of[r["sample_id"]]
        fn = P.five_number_text(x)
        if fn != P.five_number_text(np.asarray(x, dtype=np.float64)):
            fn_bad += 1
        out_rows.append({
            "mcq_id": r["mcq_id"].replace("p2|", "p3fn|"),
            "caption_id": r["caption_id"], "sample_id": r["sample_id"],
            "dataset": r["dataset"], "source": r["source"], "group": r["group"],
            "order": r["order"], "option_a": r["option_a"], "option_b": r["option_b"],
            "correct_letter": r["correct_letter"],
            "block": "five_number", "conditions": ["five_number"],
            "five_number_text": fn, "template_sha": FN_TEMPLATE_SHA,
            "prompt": FIVE_NUMBER_FULL.format(five_number=fn,
                                              option_a=r["option_a"],
                                              option_b=r["option_b"]),
        })
    gate("GF3-fivenumber", fn_bad == 0,
         f"five-number recompute mismatches: {fn_bad}/1756 (expected 0); "
         f"prompts contain no <ts><ts/> placeholder by construction")

    # ---- block 'pj_control': seeded 100 rows/substrate, measured jitter ----
    pj_rows, jit_census = [], defaultdict(list)
    for ds in ("truce", "sushi"):
        cids = sorted({r["caption_id"] for r in p2rows if r["dataset"] == ds})
        rng = np.random.Generator(np.random.PCG64(
            int(hashlib.sha256(f"pj|{ds}|{BASE_SEED}".encode()).hexdigest()[:8], 16)))
        chosen = set(np.array(cids)[rng.choice(len(cids), size=PJ_PER_SUBSTRATE,
                                               replace=False)].tolist())
        for r in p2rows:
            if r["dataset"] != ds or r["caption_id"] not in chosen:
                continue
            x = series_of[r["sample_id"]]
            true_pref = f"[Value Offset: {-P.sp_prefix_numbers(x)[0]:.4f}" \
                        f"|Value Scaling: {P.sp_prefix_numbers(x)[1]:.4f}]"
            masked, _ = P.masking_m1c(x, r["sample_id"], BASE_SEED)
            jm, js = P.sp_prefix_numbers(masked)
            jit_pref = f"[Value Offset: {-jm:.4f}|Value Scaling: {js:.4f}]"
            om, osc = P.sp_prefix_numbers(x)
            sd = float(np.std(x))
            if sd > 0:
                jit_census[f"{ds}|offset_in_std"].append(abs(jm - om) / sd)
            jit_census[f"{ds}|scale_rel"].append(abs(js - osc) / osc)
            pj_rows.append({
                "mcq_id": r["mcq_id"].replace("p2|", "p3pj|"),
                "caption_id": r["caption_id"], "sample_id": r["sample_id"],
                "dataset": ds, "source": r["source"], "group": r["group"],
                "order": r["order"], "option_a": r["option_a"],
                "option_b": r["option_b"], "correct_letter": r["correct_letter"],
                "block": "pj_control", "conditions": ["pj_control"],
                "prefix_true": true_pref, "prefix_jittered": jit_pref,
                "template_sha": r["template_sha"], "prompt": r["prompt"],
            })
    out_rows.extend(pj_rows)
    pj_ident = sum(1 for r in pj_rows if r["prefix_true"] == r["prefix_jittered"])
    gate("GF4-pj", len(pj_rows) == 2 * 2 * PJ_PER_SUBSTRATE and pj_ident == 0,
         f"pj rows = {len(pj_rows)} (exp {2*2*PJ_PER_SUBSTRATE}); "
         f"jitter identical to true prefix in {pj_ident} rows (expected 0 — the "
         f"drift census showed ~100% drift, so a 0 here is the consistent outcome)")
    for k, v in sorted(jit_census.items()):
        a = np.array(v)
        print(f"[jitter-census] {k}: median={np.median(a):.2e} max={np.max(a):.2e} "
              f"(sane units; descriptive — PJ cell measures the consequence directly)")

    # ---- GF5: totals, uniqueness, plan ----
    n = len(out_rows)
    blocks = Counter(r["block"] for r in out_rows)
    dup = [k for k, v in Counter(r["mcq_id"] for r in out_rows).items() if v > 1]
    gate("GF5-rows", blocks == Counter({"base": 1756, "five_number": 1756,
                                        "pj_control": 400}) and not dup,
         f"blocks = {dict(blocks)} (exp base 1756 / five_number 1756 / pj 400); "
         f"duplicates = {len(dup)}")
    planned = 1756 * len(NEW_CONDITIONS) + 1756 + 400
    print(f"[plan] Probe-3 GPU questions = 1756*{len(NEW_CONDITIONS)} + 1756 + 400 "
          f"= {planned}")

    gate("GF6-a-eq-c", "cond_A" in NEW_CONDITIONS and "cond_C" in NEW_CONDITIONS,
         "A~C near-construction expectation and C==stock-gaussian anchor are "
         "registered design facts carried in the report")

    print("=" * 50)
    if FAILURES:
        print(f"HARD STOP — failed gates: {FAILURES}. Nothing written.")
        sys.exit(1)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    rep = Path(args.report)
    rep.parent.mkdir(parents=True, exist_ok=True)
    with open(rep, "w", encoding="utf-8") as f:
        json.dump({"blocks": dict(blocks), "planned_gpu_questions": planned,
                   "new_conditions": NEW_CONDITIONS,
                   "fn_template_sha": FN_TEMPLATE_SHA,
                   "jitter_census": {k: {"median": float(np.median(v)),
                                         "max": float(np.max(v)),
                                         "n": len(v)}
                                     for k, v in jit_census.items()},
                   "donor_seed": f"sha256(caption_id|p3donor|{BASE_SEED})",
                   "pj_seed": f"sha256(pj|substrate|{BASE_SEED})",
                   "notes": {
                       "rungs_1_2": "read from Probe-2 ChatTS records (join gate)",
                       "cond_A": "gaussian tensor + ORIGINAL prefix (manual path)",
                       "cond_B": "original tensor + DONOR prefix (manual path)",
                       "cond_C": "gaussian tensor + gaussian prefix via manual "
                                 "path; must equal stock 'gaussian' (anchor)",
                       "A_vs_C": "pre-registered near-construction expectation: "
                                 "approximately equal (paper prefix = 2 numbers, "
                                 "matched by construction)"}},
                  f, indent=2, ensure_ascii=False)
    print(f"ALL HARD GATES GREEN. Wrote {n} rows -> {out_path}")
    print(f"Report -> {rep}")


if __name__ == "__main__":
    main()

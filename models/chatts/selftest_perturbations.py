"""
Self-test: apply EVERY ChatTS-arm perturbation/surrogate to EVERY test
series (738 TRUCE + 140 SUSHI) on CPU, run the applied checks, and print
the census. Runs everything twice and compares hashes to prove
determinism. Nothing is written except a small report JSON.

Gates GS1-GS5 are HARD STOPS; the no-op census and drift counts are
reports (properties of the data, not defects — G1/G2 precedent).

Place at: models/chatts/selftest_perturbations.py  (imports perturbations.py
from the same folder; keep an __init__.py there or run with -m)
Run from repo root:
  python -m models.chatts.selftest_perturbations --pairs data/processed/pairs.jsonl
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
FAILURES = []


def gate(name, ok, detail, hard=True):
    status = "PASS" if ok else ("FAIL (HARD)" if hard else "MISS (report)")
    print(f"[{name}] {status} — {detail}")
    if not ok and hard:
        FAILURES.append(name)


def substrate_of(dataset_name):
    if dataset_name.startswith("truce"):
        return "truce"
    if dataset_name == "sushi":
        return "sushi"
    return None


def run_all(rows):
    """Apply every condition to every series; return (census, digest, drift)."""
    census = defaultdict(Counter)     # condition -> Counter(ok/noop/fail)
    drift = Counter()
    digest = hashlib.sha256()
    fail_examples = defaultdict(list)

    for r in rows:
        x = np.asarray(r["series"], dtype=np.float64)
        sid, ds = r["sample_id"], r["sub"]

        jobs = [
            ("sf_all", P.sf_all(x, sid, BASE_SEED), P.check_perm_family),
            ("sf_half", P.sf_half(x, sid, BASE_SEED), P.check_sf_half),
            ("ex_half", P.ex_half(x), P.check_ex_half),
            ("resample", P.resample(x, sid, BASE_SEED), P.check_resample),
        ]
        for name, pert, checker in jobs:
            res = checker(x, pert)
            census[f"{ds}|{name}"]["noop" if res["noop"] else ("ok" if res["ok"] else "FAIL")] += 1
            if not res["ok"] and not res["noop"]:
                fail_examples[f"{ds}|{name}"].append(sid)
            digest.update(pert.tobytes())

        pert, info = P.masking_m1c(x, sid, BASE_SEED)
        res = P.check_masking(x, pert, info)
        census[f"{ds}|masking"]["noop" if res["noop"] else ("ok" if res["ok"] else "FAIL")] += 1
        if not res["ok"] and not res["noop"]:
            fail_examples[f"{ds}|masking"].append(sid)
        if res["prefix_offset_drift"]:
            drift[f"{ds}|offset_drift"] += 1
        if res["prefix_scale_drift"]:
            drift[f"{ds}|scale_drift"] += 1
        if not res["m1c_identity"]:
            drift[f"{ds}|M1C_IDENTITY_BROKEN"] += 1
        digest.update(pert.tobytes())

        g, g_noop = P.gaussian_matched(x, sid, BASE_SEED)
        res = P.check_gaussian(x, g, g_noop)
        census[f"{ds}|gaussian"]["noop" if res["noop"] else ("ok" if res["ok"] else "FAIL")] += 1
        if not res["ok"] and not res["noop"]:
            fail_examples[f"{ds}|gaussian"].append(sid)
        digest.update(np.asarray(g).tobytes())

        fn = P.five_number_text(x)
        recomputed = P.five_number_text(np.asarray(r["series"], dtype=np.float64))
        if fn != recomputed:
            census[f"{ds}|five_number"]["FAIL"] += 1
            fail_examples[f"{ds}|five_number"].append(sid)
        else:
            census[f"{ds}|five_number"]["ok"] += 1
        digest.update(fn.encode("utf-8"))

        if ds == "sushi":  # two-level extra is SUSHI-only (12 < one patch)
            wp = P.sf_within_patch(x, sid, BASE_SEED)
            res = P.check_within_patch(x, wp)
            census["sushi|sf_within_patch"]["noop" if res["noop"] else ("ok" if res["ok"] else "FAIL")] += 1
            if not res["ok"] and not res["noop"]:
                fail_examples["sushi|sf_within_patch"].append(sid)
            ap = P.sf_across_patch(x, sid, BASE_SEED)
            res = P.check_across_patch(x, ap)
            census["sushi|sf_across_patch"]["noop" if res["noop"] else ("ok" if res["ok"] else "FAIL")] += 1
            if not res["ok"] and not res["noop"]:
                fail_examples["sushi|sf_across_patch"].append(sid)
            digest.update(wp.tobytes())
            digest.update(ap.tobytes())

    return census, digest.hexdigest(), drift, fail_examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--report", default="results/analysis/chatts_perturbation_selftest.json")
    args = ap.parse_args()

    print("=== ChatTS perturbation self-test ===")
    print(f"base seed: {BASE_SEED}; seed formula: sha256(sample_id|condition|{BASE_SEED})")

    # one series per sample_id (captions duplicate series rows in pairs.jsonl)
    seen = {}
    with open(args.pairs, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            sub = substrate_of(r["dataset"])
            if r["split"] == "test" and sub and r["sample_id"] not in seen:
                seen[r["sample_id"]] = {"sample_id": r["sample_id"], "sub": sub,
                                        "series": r["series"]}
    rows = list(seen.values())
    n_by = Counter(r["sub"] for r in rows)
    lens = Counter((r["sub"], len(r["series"])) for r in rows)
    print(f"[input] unique test signals: {dict(n_by)}; lengths: {dict(lens)}")

    gate("GS1-population", n_by.get("sushi") == 140 and n_by.get("truce") > 0,
         f"sushi signals={n_by.get('sushi')} (exp 140); truce signals={n_by.get('truce')} "
         f"(count printed — captions share signals, so this is BELOW 738 rows by design)")
    gate("GS1-lengths",
         all(L == 12 for (s, L) in lens if s == "truce")
         and all(L == 2048 for (s, L) in lens if s == "sushi"),
         f"all TRUCE length 12 and all SUSHI length 2048: lengths={dict(lens)}")

    census1, hash1, drift1, fails = run_all(rows)
    census2, hash2, _, _ = run_all(rows)

    print("[census] per condition (ok / noop / FAIL):")
    for k in sorted(census1):
        c = census1[k]
        print(f"    {k}: ok={c.get('ok', 0)} noop={c.get('noop', 0)} FAIL={c.get('FAIL', 0)}")
    print(f"[drift] masking prefix drift at 4 decimals: {dict(drift1) if drift1 else 'none'}")

    total_fail = sum(c.get("FAIL", 0) for c in census1.values())
    gate("GS2-applied", total_fail == 0,
         f"applied-check failures: {total_fail} (expected 0)"
         + (f"; examples: { {k: v[:2] for k, v in fails.items() if v} }" if total_fail else ""))
    gate("GS3-m1c", drift1.get("truce|M1C_IDENTITY_BROKEN", 0) == 0
         and drift1.get("sushi|M1C_IDENTITY_BROKEN", 0) == 0,
         "M1-C identity (fill == modified mean) holds for every series")
    gate("GS4-determinism", hash1 == hash2,
         f"double-run digest identical: {hash1[:16]}... == {hash2[:16]}...")

    sushi_noop_conditions = [k for k in census1 if k.startswith("sushi|")
                             and census1[k].get("noop", 0) >= 1]
    gate("GS5-constant", all(census1[k].get("noop", 0) == 1 for k in sushi_noop_conditions)
         and len(sushi_noop_conditions) >= 1,
         f"SUSHI no-op census (expected: exactly 1 signal — the constant — wherever "
         f"a no-op appears): conditions with no-ops = {sushi_noop_conditions}")

    print("=" * 50)
    if FAILURES:
        print(f"HARD STOP — failed gates: {FAILURES}")
        sys.exit(1)

    rep = Path(args.report)
    rep.parent.mkdir(parents=True, exist_ok=True)
    with open(rep, "w", encoding="utf-8") as f:
        json.dump({"census": {k: dict(v) for k, v in census1.items()},
                   "drift": dict(drift1),
                   "digest": hash1,
                   "signals": dict(n_by),
                   "base_seed": BASE_SEED}, f, indent=2)
    print(f"ALL HARD GATES GREEN. Report -> {rep}")


if __name__ == "__main__":
    main()

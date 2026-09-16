#!/usr/bin/env python
"""
Probe-3 ladder rung-pair contrasts (follow-up to analyze_chatts_probes.py).
Computes the four paired contrasts the ladder verdict needs:
  rung2_sf_all vs resample, resample vs gaussian, rung2_sf_all vs gaussian,
  five_number vs rung2_sf_all — per substrate, paired per caption,
  cluster bootstrap on signals, TOST +/-0.05 (same flagged new application).
Structural gates only; NO outcome expectations are registered for this run
(the point estimates have been seen — registering outcomes now would be
retrofitting). CPU-only.

Run from the repo root:
  python scripts/analyze_chatts_probe3_contrasts.py
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

B_BOOT, SEED, MARGIN = 2000, 42, 0.05


def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def paired(diffs, clus, rng):
    v = np.asarray(diffs, float); labs = np.asarray(clus)
    uniq = np.unique(labs)
    sums = np.array([v[labs == c].sum() for c in uniq])
    cnts = np.array([(labs == c).sum() for c in uniq], float)
    k = len(uniq)
    out = {}
    for alpha, tag in ((0.05, "ci95"), (0.10, "ci90")):
        boots = np.empty(B_BOOT)
        for b in range(B_BOOT):
            idx = rng.integers(0, k, k)
            boots[b] = sums[idx].sum() / cnts[idx].sum()
        out[tag] = [round(float(x), 4) for x in
                    np.percentile(boots, [100*alpha/2, 100*(1-alpha/2)])]
    out["mean_diff"] = round(float(v.mean()), 4)
    lo, hi = out["ci90"]
    out["tost_pm005"] = "PASS(flat)" if (lo > -MARGIN and hi < MARGIN) else "FAIL(not flat)"
    out["n_units"], out["n_signals"] = int(len(v)), int(k)
    return out


def main():
    rng = np.random.default_rng(SEED)
    man3 = load_jsonl("data/processed/chatts_probe3_mcq.jsonl")
    p2 = {(r["mcq_id"], r["condition"]): r
          for r in load_jsonl("results/experiments/chatts_probe2_responses.jsonl")}
    p3 = {(r["mcq_id"], r["condition"]): r
          for r in load_jsonl("results/experiments/chatts_probe3_responses.jsonl")}

    lad = defaultdict(lambda: defaultdict(dict)); sig = {}
    n_base = 0
    for r in man3:
        if r["block"] == "pj_control":
            continue
        u, ds = r["caption_id"], r["dataset"]
        sig[u] = r["sample_id"]
        if r["block"] == "base":
            n_base += 1
            p2id = r["mcq_id"].replace("p3base|", "p2|")
            lad[ds]["rung2_sf_all"].setdefault(u, []).append(
                int(p2[(p2id, "sf_all")]["correct"]))
            for c in ("resample", "gaussian"):
                lad[ds][c].setdefault(u, []).append(int(p3[(r["mcq_id"], c)]["correct"]))
        else:
            lad[ds]["five_number"].setdefault(u, []).append(
                int(p3[(r["mcq_id"], "five_number")]["correct"]))
    ok = n_base == 1756
    print(f"[G-join] {'PASS' if ok else 'FAIL'} — base rows joined = {n_base} (exp 1756)")
    if not ok:
        raise SystemExit(1)

    pairs = [("rung2_sf_all", "resample"), ("resample", "gaussian"),
             ("rung2_sf_all", "gaussian"), ("five_number", "rung2_sf_all")]
    results = {}
    for ds in ("sushi", "truce"):
        uts = sorted(lad[ds]["rung2_sf_all"])
        exp_n = 140 if ds == "sushi" else 738
        print(f"[G-units-{ds}] {'PASS' if len(uts) == exp_n else 'FAIL'} — "
              f"units = {len(uts)} (exp {exp_n})")
        cl = [sig[u] for u in uts]
        for a, b in pairs:
            d = [np.mean(lad[ds][a][u]) - np.mean(lad[ds][b][u]) for u in uts]
            res = paired(d, cl, rng)
            results[f"{ds}|{a}_minus_{b}"] = res
            print(f"  {ds}: {a} - {b} = {res['mean_diff']:+.4f} "
                  f"ci95 {res['ci95']} ci90 {res['ci90']} -> {res['tost_pm005']}")
    out = "results/experiments/chatts_probe3_contrasts.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(out, "w", encoding="utf-8"), indent=1)
    print(f"Wrote {out} — paste the full output back.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
run_probe2.py — CLaSP Probe-2 runner (order-invariance shuffle + masking).

Design record: handoff §4.6 (Q1–Q5) and §4.7; mechanics M1–M3 accepted
2026-08-13 after source verification of the parent's ablUtils.py
(sha256 4e0552…a6f0; masking = int(L*0.2) positions set to 0 on the
normalized model input; StandardScaler applied in the loader before
batching — data_loader.py:111–133).

Perturbations (per unique pool signal, applied to the z-normalized
series the model sees; every pool signal is perturbed):
  sf_all   permute all L points
  sf_half  permute the first L//2 points, second half untouched
  ex_half  deterministic swap of the two halves (parent: post + pre)
  masking  int(0.2*L) positions set to 0  (M1: post-znorm; M2: int();
           SUSHI 409/2048, TRUCE 2/12 = 16.7% effective, reported)
M3 seed scheme: per-signal seed = first 12 hex digits of
sha256("{sample_id}|{perturbation}|{ckpt_seed}") as an integer; recorded
per signal in the signal-meta output. Queries (captions) are NEVER
perturbed; caption embeddings are computed once per checkpoint.

Gates (§4.6 item 5; fatal policy fixed):
  G6 frozen-baseline reproduction: unperturbed strict table must match
     results/experiments/eval_baseline_seed{S}.json to <=1e-9 — HARD STOP
  G3 grouping counts: SUSHI test queries 135/4/1 (dep/inv/degen),
     TRUCE test rows 715/18/5 (dep/inv/amb), total 878 — HARD STOP
  G7 permutation validity + seed recorded per signal — HARD STOP
  G1 applied-check per signal: shuffles preserve the value multiset and
     change order (identity survivors FLAGGED no-op, never failed);
     masking touches exactly k positions, all set to 0
  G2 natural-zero rate per substrate printed before masking (report only)
  G8 identity control: the constant signal's embedding unchanged under
     the shuffle family (<=1e-6). CORRECTED 2026-08-13: rank-invariance
     was dropped from G8 — every pool signal is perturbed, so the
     constant signal's rank can legitimately move; only its EMBEDDING
     is pinned. Correction recorded in the log.
  G9 pairing: the query tuple list is asserted identical across all
     conditions before any rank is compared — HARD STOP
  G4 direction sanity: SUSHI dependent-group MRR must not IMPROVE under
     sf_all in any seed — HARD STOP

Decisions D1/D2 (accepted 2026-08-13 after the tie investigation;
probe2_pool_duplicates.json + probe2_pool_neighbours.json hold the
evidence: TRUCE-synth clusters {165,326,360,(87 at 1 ulp)}, {58,86},
{249,362 at 1 ulp}; 15/738 queries affected; SUSHI min NN 0.124):
  D1  G6 splits: G6a = tie-immune metrics (all SUSHI; TRUCE and all
      R@5/R@10) reproduce the frozen table to <=1e-9 — HARD STOP;
      G6b = TRUCE/all R@1 and MRR within 6e-3 of frozen (bound derived
      from the 15 tie queries; beyond it cannot be tie-explained) —
      HARD STOP. G6a/G6b are computed with the LEGACY argsort rank,
      like-for-like against the frozen table.
  D2  All probe measurements use deterministic AVERAGE RANK: the ground
      truth receives the mean of its exact-tie positions (tie for 1st-
      2nd -> 1.5). Fractional, torch-version-independent. Near-ties at
      one float32 ulp remain fragile to float-level changes: documented
      per query via the tie count, footnoted, not silenced.
Self-check (once per run): znorm(raw[perm]) vs znorm(raw)[perm] max
abs diff printed — expected float noise (~1e-7), confirming that
norm-then-shuffle (what we do) equals shuffle-then-norm conceptually.

Usage (all path flags REQUIRED — no hardcoded checkpoint paths):
    python -m models.clasp.run_probe2 \
        --checkpoints CK42 CK43 CK44 --seeds 42 43 44 \
        --sushi-groups results/analysis/probe2_sushi_groups.json \
        --truce-groups results/analysis/probe2_truce_groups_certified.json

Writes:
  results/experiments/probe2_clasp_per_query_seed{S}.jsonl
  results/experiments/probe2_clasp_signal_meta_seed{S}.json
  results/experiments/probe2_clasp_summary.json
Formal statistics (bootstrap CIs, Wilcoxon, TOST, DiD, P2-1/2/3/5
scoring) come from the separate stats script run on these records.
"""

from __future__ import annotations
import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from dataset import load_pairs, znorm
from models.clasp.evaluate import encode_pool, strict_metrics
from models.clasp.model import ClaspModel, ClaspConfig

PERTS = ["sf_all", "sf_half", "ex_half", "masking"]
MASK_RATIO = 0.2
MAX_TOKENS = 16384  # same length-grouped batching budget as evaluate.py


def fail(msg):
    print(f"\nGATE FAILED: {msg}", file=sys.stderr)
    sys.exit(1)


def signal_seed(sample_id: str, pert: str, ckpt_seed: int) -> int:
    h = hashlib.sha256(f"{sample_id}|{pert}|{ckpt_seed}".encode()).hexdigest()
    return int(h[:12], 16)


def perturb(z: np.ndarray, pert: str, rng: np.random.Generator):
    """Perturb a z-normalized series. Returns (z_pert, meta)."""
    L = len(z)
    meta = {}
    if pert == "sf_all":
        perm = rng.permutation(L)
        if sorted(perm.tolist()) != list(range(L)):
            fail("G7: sf_all draw is not a permutation")
        return z[perm], meta
    if pert == "sf_half":
        mid = L // 2
        perm = rng.permutation(mid)
        if sorted(perm.tolist()) != list(range(mid)):
            fail("G7: sf_half draw is not a permutation")
        return np.concatenate([z[:mid][perm], z[mid:]]), meta
    if pert == "ex_half":
        mid = L // 2
        return np.concatenate([z[mid:], z[:mid]]), meta
    if pert == "masking":
        k = int(L * MASK_RATIO)
        idx = rng.permutation(L)[:k]
        zp = z.copy()
        zp[idx] = 0.0
        meta["n_masked"] = k
        meta["n_already_zero"] = int((z[idx] == 0.0).sum())
        return zp, meta
    fail(f"unknown perturbation {pert}")


def applied_check(z, zp, pert, meta):
    """G1. Returns 'ok' | 'noop'; hard-fails on real violations."""
    if pert in ("sf_all", "sf_half", "ex_half"):
        if not np.array_equal(np.sort(z), np.sort(zp)):
            fail(f"G1: {pert} did not preserve the value multiset")
        if pert == "sf_half" and not np.array_equal(z[len(z)//2:],
                                                    zp[len(z)//2:]):
            fail("G1: sf_half touched the second half")
        return "noop" if np.array_equal(z, zp) else "ok"
    # masking
    diff = ~np.isclose(zp, z, rtol=0, atol=0)
    if not np.all(zp[diff] == 0.0):
        fail("G1: masking wrote a non-zero value")
    if int(diff.sum()) + meta["n_already_zero"] != meta["n_masked"]:
        fail(f"G1: masking touched {int(diff.sum())} positions + "
             f"{meta['n_already_zero']} already-zero != k={meta['n_masked']}")
    return "noop" if int(diff.sum()) == 0 else "ok"


@torch.no_grad()
def encode_znormed(model, device, series_list):
    """Encode ALREADY-normalized series with evaluate.py's batching scheme."""
    by_len = defaultdict(list)
    for j, s in enumerate(series_list):
        by_len[len(s)].append(j)
    buf = [None] * len(series_list)
    for L0, idxs in sorted(by_len.items()):
        B = max(1, min(64, MAX_TOKENS // L0))
        for i in range(0, len(idxs), B):
            idx = idxs[i:i + B]
            x = torch.zeros(len(idx), L0)
            m = torch.ones(len(idx), L0, dtype=torch.bool)
            for k, j in enumerate(idx):
                x[k] = torch.from_numpy(series_list[j])
            zb = model.encode_series(x.to(device), m.to(device)).cpu()
            for k, j in enumerate(idx):
                buf[j] = zb[k]
    return torch.stack(buf)


def avg_ranks(sig_emb, queries, q_emb, id2idx):
    """Deterministic average rank (D2) + exact-tie count per query."""
    sims = (q_emb @ sig_emb.T).numpy()
    ranks = np.empty(len(queries), dtype=np.float64)
    ntied = np.empty(len(queries), dtype=np.int64)
    for qi, (_, gt, _) in enumerate(queries):
        s = sims[qi]
        sg = s[id2idx[gt]]
        n_above = int((s > sg).sum())
        n_eq = int((s == sg).sum())          # includes gt itself
        ranks[qi] = n_above + (n_eq + 1) / 2.0
        ntied[qi] = n_eq - 1
    return ranks, ntied


def group_metrics(ranks, sel):
    r = ranks[sel]
    return {"n": int(len(r)), "mrr": float((1.0 / r).mean()),
            "recall@10": float((r <= 10).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs=3, required=True)
    ap.add_argument("--seeds", nargs=3, type=int, required=True)
    ap.add_argument("--sushi-groups", required=True)
    ap.add_argument("--truce-groups", required=True)
    ap.add_argument("--baselines-dir", default="results/experiments")
    ap.add_argument("--out-dir", default="results/experiments")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- groups (G3 part 1: artifacts) ----
    sg = json.load(open(args.sushi_groups, encoding="utf-8"))
    class_verdict = sg["class_verdicts"]
    if {k: v for k, v in sg["counts"].items()} != \
            {"dependent": 135, "invariant": 4, "degenerate": 1}:
        fail(f"G3: SUSHI artifact counts {sg['counts']} != 135/4/1")
    tg = json.load(open(args.truce_groups, encoding="utf-8"))
    if not tg.get("certified"):
        fail("G3: TRUCE groups file is not the CERTIFIED artifact")
    truce_label = {t: rec["label"] for t, rec in tg["per_text"].items()}

    # ---- pairs / query-side groups ----
    pairs = load_pairs(splits=("test",))
    cls_of = {p.sample_id: p.class_label for p in pairs}
    q_group, q_substrate = [], []
    for p in pairs:
        if p.dataset == "sushi":
            q_group.append(class_verdict[p.class_label])
            q_substrate.append("sushi")
        else:
            q_group.append(truce_label[p.caption])
            q_substrate.append("truce")
    q_group = np.array(q_group)
    q_substrate = np.array(q_substrate)

    from collections import Counter
    cs = Counter(zip(q_substrate.tolist(), q_group.tolist()))
    print("query groups:", dict(sorted(cs.items())))
    exp = {("sushi", "dependent"): 135, ("sushi", "invariant"): 4,
           ("sushi", "degenerate"): 1, ("truce", "dependent"): 715,
           ("truce", "invariant"): 18, ("truce", "ambiguous"): 5}
    if dict(cs) != exp or len(pairs) != 878:
        fail(f"G3: query-group counts differ from certified: {dict(cs)}")
    print(f"G3 PASSED: 878 queries; SUSHI 135/4/1, TRUCE 715/18/5")

    # ---- self-check: norm-then-shuffle vs shuffle-then-norm (float noise) ----
    rng0 = np.random.default_rng(0)
    diffs = []
    for p in (pairs[0], next(p for p in pairs if p.dataset == "sushi")):
        perm = rng0.permutation(len(p.series))
        diffs.append(float(np.max(np.abs(
            znorm(p.series[perm]) - znorm(p.series)[perm]))))
    print(f"self-check znorm-commute max diffs: {['%.2e' % d for d in diffs]} "
          f"(expected float noise)")

    summary = {"mask_ratio": MASK_RATIO, "perts": PERTS, "seeds": {},
               "seed_scheme": "int(sha256('{sample_id}|{pert}|{ckpt_seed}')"
                              "[:12hex],16)"}

    for ck_path, ck_seed in zip(args.checkpoints, args.seeds):
        print(f"\n===== checkpoint seed {ck_seed}: {ck_path} =====")
        model = ClaspModel(ClaspConfig())
        state = torch.load(ck_path, map_location="cpu")
        model.load_state_dict(state["model"] if "model" in state else state)
        model.eval().to(device)

        sig_ids, sig_emb, queries, q_emb, _ = encode_pool(model, device)

        # G9 pre-check: query list order must equal our pairs read
        if [(q[0], q[1], q[2]) for q in queries] != \
                [(p.caption, p.sample_id, p.dataset) for p in pairs]:
            fail("G9: encode_pool query order differs from load_pairs order")

        # G6: reproduce the frozen per-seed table
        base_file = Path(args.baselines_dir) / f"eval_baseline_seed{ck_seed}.json"
        frozen = json.load(open(base_file))["strict"]
        rep = {"all": None}
        strict_all, _legacy_ranks = strict_metrics(sig_ids, sig_emb,
                                                    queries, q_emb)
        rep["all"] = strict_all
        for ds_group, name in [(("truce_stock", "truce_synth"), "truce"),
                               (("sushi",), "sushi")]:
            qsel = [i for i, q in enumerate(queries) if q[2] in ds_group]
            m, _ = strict_metrics(sig_ids, sig_emb,
                                  [queries[i] for i in qsel], q_emb[qsel])
            rep[name] = m
        # Partition corrected 2026-08-13 after the seed-43 stop: only SUSHI
        # is STRUCTURALLY tie-immune (min NN distance 0.124). TRUCE ties can
        # straddle ANY rank boundary depending on the checkpoint, so every
        # TRUCE-containing metric is tie-exposed.
        TIE_IMMUNE = [("sushi", m) for m in
                      ("recall@1", "recall@5", "recall@10", "mrr")]
        TIE_EXPOSED = [(k, m) for k in ("truce", "all")
                       for m in ("recall@1", "recall@5", "recall@10", "mrr")]
        for k, m in TIE_IMMUNE + TIE_EXPOSED:
            d = abs(rep[k][m] - frozen[k][m])
            if d > 1e-9:
                print(f"  dev {k}/{m}: ours {rep[k][m]:.12f}  "
                      f"frozen {frozen[k][m]:.12f}  |d|={d:.2e}"
                      f"  (~{d*rep[k]['n_queries']:.2f} query-steps)")
        dev_a = max(abs(rep[k][m] - frozen[k][m]) for k, m in TIE_IMMUNE)
        dev_b = max(abs(rep[k][m] - frozen[k][m]) for k, m in TIE_EXPOSED)
        print(f"G6a (SUSHI, structurally tie-free, digit-exact): "
              f"max |dev| = {dev_a:.2e}")
        print(f"G6b (TRUCE-containing, tie bound 6e-3): "
              f"max |dev| = {dev_b:.2e}")
        if dev_a > 1e-9:
            fail(f"G6a: SUSHI metrics not reproduced (dev {dev_a:.2e}) — "
                 f"NOT tie-explainable, real drift")
        if dev_b > 6e-3:
            fail(f"G6b: deviation {dev_b:.2e} exceeds the tie-explainable "
                 f"bound 6e-3")
        print("G6a/G6b PASSED (legacy-rank comparison vs frozen table)")

        # measurement ranks (D2 average rank) for the unperturbed condition
        id2idx_q = {s: i for i, s in enumerate(sig_ids)}
        ranks_u, ntied_u = avg_ranks(sig_emb, queries, q_emb, id2idx_q)
        n_tied_q = {"truce": int(sum(1 for qi, p in enumerate(pairs)
                                     if p.dataset != "sushi" and ntied_u[qi] > 0)),
                    "sushi": int(sum(1 for qi, p in enumerate(pairs)
                                     if p.dataset == "sushi" and ntied_u[qi] > 0))}
        print(f"D2: unperturbed queries with exact ties: {n_tied_q} "
              f"(audit: 15 exact-cluster + up to 9 near-dup collisions "
              f"=> expected truce 15-24, sushi 0)")
        if not (15 <= n_tied_q["truce"] <= 24 and n_tied_q["sushi"] == 0):
            print("  NOTE: outside the audit-derived range — "
                  "investigate before trusting", file=sys.stderr)

        # base znormed series per pool signal (order = sig_ids)
        raw_of = {}
        for p in pairs:
            raw_of.setdefault(p.sample_id, p.series)
        z_base = [znorm(raw_of[s]) for s in sig_ids]
        id2idx = {s: i for i, s in enumerate(sig_ids)}

        # G2: natural zeros on the znormed base, per substrate
        for sub, sel_ids in [("truce", [s for s in sig_ids
                                        if not s.startswith("sushi")]),
                             ("sushi", [s for s in sig_ids
                                        if s.startswith("sushi")])]:
            if not sel_ids:  # substrate prefix convention check
                continue
            tot = sum(len(z_base[id2idx[s]]) for s in sel_ids)
            zz = sum(int((z_base[id2idx[s]] == 0.0).sum()) for s in sel_ids)
            print(f"G2: natural exact-zeros [{sub}]: {zz}/{tot} "
                  f"({zz/tot:.4%}) on the znormed input")

        # constant signal(s) for G8
        const_idx = [i for i, z in enumerate(z_base) if np.ptp(z) == 0.0]
        print(f"G8 targets (constant znormed signals): "
              f"{[sig_ids[i] for i in const_idx]}")

        ranks_by_pert = {}
        sig_meta = {}
        for pert in PERTS:
            z_pert, flags, seeds_rec = [], [], {}
            for i, s in enumerate(sig_ids):
                sd = signal_seed(s, pert, ck_seed)
                seeds_rec[s] = sd
                zp, meta = perturb(z_base[i], pert,
                                   np.random.default_rng(sd))
                flags.append(applied_check(z_base[i], zp, pert, meta))
                z_pert.append(zp)
            n_noop = flags.count("noop")
            print(f"[{pert}] G1/G7 PASSED over {len(sig_ids)} signals; "
                  f"no-ops flagged: {n_noop} "
                  f"({[sig_ids[i] for i, f in enumerate(flags) if f == 'noop'][:4]})")
            emb_p = encode_znormed(model, device, z_pert)

            # G8: embedding identity for constants under the shuffle family
            if pert != "masking":
                for i in const_idx:
                    d = float((emb_p[i] - sig_emb[i]).abs().max())
                    if d > 1e-6:
                        fail(f"G8: constant signal {sig_ids[i]} embedding "
                             f"moved by {d:.2e} under {pert}")
                if const_idx:
                    print(f"[{pert}] G8 PASSED (max emb diff <= 1e-6)")

            ranks_p, ntied_p = avg_ranks(emb_p, queries, q_emb, id2idx_q)
            n_still_tied = int((ntied_p > 0).sum())
            print(f"[{pert}] queries with exact ties after perturbation: "
                  f"{n_still_tied}")
            ranks_by_pert[pert] = ranks_p
            sig_meta[pert] = {"signal_seeds": seeds_rec,
                              "noop_ids": [sig_ids[i] for i, f
                                           in enumerate(flags) if f == "noop"]}

        # G4: SUSHI dependent stratum must not improve under sf_all
        sel = (q_substrate == "sushi") & (q_group == "dependent")
        mu = group_metrics(ranks_u, sel)["mrr"]
        mp = group_metrics(ranks_by_pert["sf_all"], sel)["mrr"]
        print(f"G4: SUSHI dependent MRR unpert {mu:.4f} -> sf_all {mp:.4f} "
              f"(degradation {mu - mp:+.4f})")
        if mp > mu:
            fail("G4: known-order-dependent stratum IMPROVED under sf_all")
        print("G4 PASSED")

        # per-group descriptive tables (binding: baselines side by side)
        tables = {}
        print("\nper-group MRR (unperturbed | sf_all | sf_half | ex_half | "
              "masking)  [n]")
        for sub in ("sushi", "truce"):
            for grp in sorted(set(q_group[q_substrate == sub])):
                sel = (q_substrate == sub) & (q_group == grp)
                row = {"unperturbed": group_metrics(ranks_u, sel)}
                for pert in PERTS:
                    row[pert] = group_metrics(ranks_by_pert[pert], sel)
                tables[f"{sub}/{grp}"] = row
                print(f"  {sub}/{grp:10s} "
                      f"{row['unperturbed']['mrr']:.4f} | "
                      + " | ".join(f"{row[p]['mrr']:.4f}" for p in PERTS)
                      + f"   [{row['unperturbed']['n']}]")

        # per-query records
        rec_path = outdir / f"probe2_clasp_per_query_seed{ck_seed}.jsonl"
        with open(rec_path, "w", encoding="utf-8") as f:
            for qi, p in enumerate(pairs):
                f.write(json.dumps({
                    "caption_id": p.caption_id, "dataset": p.dataset,
                    "substrate": q_substrate[qi], "group": q_group[qi],
                    "gt": p.sample_id,
                    "rank_unperturbed": float(ranks_u[qi]),
                    "ntied_unperturbed": int(ntied_u[qi]),
                    **{f"rank_{pt}": float(ranks_by_pert[pt][qi])
                       for pt in PERTS}}) + "\n")
        meta_path = outdir / f"probe2_clasp_signal_meta_seed{ck_seed}.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(sig_meta, f, indent=2)
        print(f"wrote {rec_path} and {meta_path.name}")
        summary["seeds"][str(ck_seed)] = {"g6a_dev": dev_a,
                                          "g6b_dev": dev_b,
                                          "tied_queries_unperturbed": n_tied_q,
                                          "tables": tables}

    with open(outdir / "probe2_clasp_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {outdir / 'probe2_clasp_summary.json'}")
    print("ALL GATES PASSED — per-query records ready for the stats script")


if __name__ == "__main__":
    main()

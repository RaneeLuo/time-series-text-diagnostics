"""
run_probe2.py (text-embedding-3-large) — Probe-2 floor negative control (P2-4).

The floor model is PRE-DECLARED VOID for Probe 2 (binding, PROJECT_CONTEXT):
baseline MRR 0.027 vs chance 0.017 — there is no capability for a shuffle to
degrade. This run exists as the pipeline's negative control: meaningful
"degradation" here would mean a broken pipeline, not a finding (G4's sibling,
inverted — see gate GF below).

Perturbations and mechanics are IDENTICAL to models/clasp/run_probe2.py
(M1 fill-0 on the z-normed series; M2 int(0.2*L); M3 per-signal seed =
sha256(sample_id|pert|arm_seed)[:12hex]; sf_all / sf_half / ex_half / masking).
The floor has no checkpoints, so the three "seeds" are arm labels that vary
only the permutation draws through the M3 hash — mirroring the CLaSP matrix
by decision of 2026-08-14.

Serialisation: unperturbed strings come from run_probe1.serialize (IMPORTED,
not copied — cache-key identity with the committed baseline). Perturbed
strings are quantised by a local quantize_z() on the already-z-normed,
already-perturbed series; gate SC-1 asserts quantize_z(znorm(x)) ==
serialize(x) byte-identical for all 386 pool signals, so the two paths
cannot drift. Perturbing AFTER z-norm (not raw) is required by M1 for
masking; for shuffles the two orders agree up to float noise (self-check
printed, as in the CLaSP runner).

Determinism note: OpenAI embeddings are NOT guaranteed deterministic across
API calls. Determinism here is supplied by the CACHE — every unique string is
embedded exactly once, ever, and reused. In particular ex_half strings are
identical across the three arms (no RNG), so ex_half ranks must be
byte-identical across arms (gate SC-2, free consistency check).

Gates (fatal policy mirrors the CLaSP runner; PROJECT_CONTEXT G1–G9):
  G3   grouping counts vs certified artifacts AND query-side counts:
       SUSHI 135/4/1 (dep/inv/degen), TRUCE 715/18/5 (dep/inv/amb),
       total 878, pool 386 — HARD STOP
  SC-1 serialisation-path identity over all 386 pool signals — HARD STOP
  G1/G7 applied-check per signal per perturbation (multiset preserved,
       order changed, permutation valid, seed recorded; no-ops FLAGGED,
       never failed) + serialized-token multiset equality for shuffles
  G2   natural exact-zeros on the z-normed base, per substrate (report only),
       plus quantised-zero token rate (a masked 0 and a small value are
       indistinguishable AFTER quantisation — reported, never fatal)
  G6-pre  every unperturbed signal string and every caption string must
       already be in the cache (the baseline run put them there) — the
       frozen-baseline reproduction must cost $0 — HARD STOP
  G6   digit-exact reproduction of results/experiments/baseline_openai_embed
       .json with the SAME legacy argsort rank as run_baseline.py:
       G6a sushi metrics <= 1e-9 (structurally tie-free here iff no two
           SUSHI serialisations collide — collision count printed) — HARD STOP
       G6b truce/all metrics: any deviation is printed in query-steps and
           must be tie-explainable (<= n_tied_queries steps for recalls;
           <= 0.5*n_tied/n for MRR) — HARD STOP beyond
       Registered expectation: ZERO deviation (same machine, cached vectors).
  G8   structural identity control: targets = pool signals with ptp(z)==0.
       EXPECTED EMPTY — the 2026-08-13 log finding: the constant SUSHI
       signal z-norms to ±1, NOT zeros, so it is not a fixed point and is
       excluded here exactly as in the CLaSP run. If a target exists, its
       shuffle-family serialisation must be byte-identical (=> same cache
       key) — HARD STOP on violation. No-op identity: every G1 no-op must
       serialize byte-identically — HARD STOP.
  G9   pairing: query tuple list asserted identical across conditions; the
       signal-row order of every similarity matrix is the one sig_ids
       order — HARD STOP
  GF   (G4's sibling, INVERTED for a floor): on the two large certified
       cells (sushi/dependent n=135, truce/dependent n=715), perturbed MRR
       must not EXCEED unperturbed MRR by more than 0.05 in any arm.
       A floor gaining capability-scale MRR from a perturbation = broken
       pipeline — HARD STOP. (Downward moves cannot reach 0.05: the
       baselines are 0.004 / 0.032.) Small cells are printed loudly but
       not gated — their verdicts belong to the stats script.

Rank metric: D2 deterministic AVERAGE RANK for all measurements, with the
D2-F amendment (accepted 2026-08-14, floor arm only): ties are detected at
construction level (identical serialisation string => identical cached
vector => definitionally equal similarity) by equalising each identity
group's similarity COLUMNS before ranking. Diagnosed motivation: BLAS
computes different output columns through different float paths and splits
bitwise-identical pool vectors at ~1 ulp — float-equality found only 22 of
the 24 forced tied queries (scripts/diagnose_floor_ties.py, 2026-08-14).
Both counts (identity-level and float-level) are printed side by side in
every condition. The legacy argsort rank is used only inside the G6
comparison, like-for-like against the frozen baseline JSON.

Registered expectations (2026-08-14, BEFORE first run):
    pool 386 (sushi 140 / truce 246); queries 878; groups exact (G3)
    SC-1: 386/386 byte-identical
    duplicate-serialisation clusters (CONFIRMED at dry-run 2026-08-14):
        {58,86,249,362} and {87,165,326,360} — two groups of four
        (cluster-structure prediction missed: quantisation merges more
        than the one-ulp pairs; membership union as predicted) => tied
        truce queries = 24 exactly; sushi collisions = 0
    G6: zero deviation on every metric (all MRR 0.027, R@10 0.052 digit-exact)
    fresh cost ceiling ~= $0.76 (10 condition-instances: ex_half costs 1x);
        dry run prints the exact token count — approved budget ~$1
    ex_half strings identical across arms (SC-2)
    tie dissolution (identity level, D2-F): sf_all / sf_half / masking
        -> 0 identity groups, 0 tied queries; ex_half preserves the
        duplicate groups -> exactly 24 tied queries in every arm
    masking no-ops: 0 or near-0 (report; TRUCE natural zeros make >0 possible)
    G8 targets: EMPTY (corrected expectation, ±1 z-norm finding)
    GF: no gated cell moves upward by > 0.05 MRR; expected jiggle |Δ| <~ 0.02
    P2-4 verdict (scored later by the stats script): VOID everywhere,
        no group difference beyond noise

Usage:
    python -m models.openai_embed.run_probe2 --dry-run \
        --sushi-groups results/analysis/probe2_sushi_groups.json \
        --truce-groups results/analysis/probe2_truce_groups_certified.json \
        --baseline results/experiments/baseline_openai_embed.json \
        --seeds 42 43 44
    ... then the same command with --yes instead of --dry-run.

Writes:
    results/experiments/probe2_openai_per_query_seed{S}.jsonl
    results/experiments/probe2_openai_signal_meta_seed{S}.json
    results/experiments/probe2_openai_summary.json
"""

from __future__ import annotations
import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from dataset import load_pairs
from models.openai_embed.run_probe1 import (
    serialize, znorm, count_tokens, load_cache, embed_all, key_of,
    MODEL, MAX_TOKENS, PRICE_PER_1M,
)

PERTS = ["sf_all", "sf_half", "ex_half", "masking"]
MASK_RATIO = 0.2
SCALE, CLIP = 10, 99            # must match run_probe1.serialize defaults

EXPECTED_POOL = 386
EXPECTED_QUERIES = 878
GROUPS_EXPECTED = {("sushi", "dependent"): 135, ("sushi", "invariant"): 4,
                   ("sushi", "degenerate"): 1, ("truce", "dependent"): 715,
                   ("truce", "invariant"): 18, ("truce", "ambiguous"): 5}
TIE_RANGE = (15, 24)            # registered; point prediction 24
GF_CELLS = [("sushi", "dependent"), ("truce", "dependent")]   # n>=100 only
GF_MARGIN = 0.05


def fail(msg):
    print(f"\nGATE FAILED: {msg}", file=sys.stderr)
    sys.exit(1)


def gate(ok: bool, name: str, msg: str):
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name}: {msg}")
    if not ok:
        fail(f"{name} — {msg}")


# ------------------------------------------------------------- serialisation
def quantize_z(z: np.ndarray) -> str:
    """Quantise an ALREADY-z-normed series exactly as serialize() does
    internally. SC-1 asserts the two paths are byte-identical."""
    q = np.clip(np.rint(np.asarray(z, dtype=np.float64) * SCALE),
                -CLIP, CLIP).astype(int)
    return ",".join(str(v) for v in q)


# ------------------------------------------------------------- perturbations
def signal_seed(sample_id: str, pert: str, arm_seed: int) -> int:
    h = hashlib.sha256(f"{sample_id}|{pert}|{arm_seed}".encode()).hexdigest()
    return int(h[:12], 16)


def perturb(z: np.ndarray, pert: str, rng: np.random.Generator):
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
    diff = ~np.isclose(zp, z, rtol=0, atol=0)
    if not np.all(zp[diff] == 0.0):
        fail("G1: masking wrote a non-zero value")
    if int(diff.sum()) + meta["n_already_zero"] != meta["n_masked"]:
        fail(f"G1: masking touched {int(diff.sum())} + already-zero "
             f"{meta['n_already_zero']} != k={meta['n_masked']}")
    return "noop" if int(diff.sum()) == 0 else "ok"


# ------------------------------------------------------------- rank metrics
def identity_groups(strings):
    """Pool indices sharing an identical serialisation string (D2-F)."""
    by = defaultdict(list)
    for i, t in enumerate(strings):
        by[t].append(i)
    return [v for v in by.values() if len(v) > 1]


def avg_ranks(S: np.ndarray, C: np.ndarray, gt_idx: np.ndarray,
              ident=None):
    """D2 deterministic average rank + exact-tie count per query.
    S: pool x d, C: queries x d, gt_idx: pool row of each query's GT.

    D2-F amendment (accepted 2026-08-14; floor arm only): ties are
    detected at CONSTRUCTION level. Identical serialisation strings mean
    identical cached vectors, i.e. definitionally identical similarity;
    BLAS matmul splits such columns at ~1 ulp depending on column path
    (diagnosed 2026-08-14: float-equality found 22 of the 24 forced tied
    queries, per-query loop-dot found all 24). So the similarity COLUMNS
    of each identity group are equalised to the group's first column
    before ranking. Copying the vectors would NOT fix it — bitwise-
    identical rows are exactly what BLAS splits. `ident` is the list of
    identity groups (pool index lists) for THIS condition's strings.
    n_float_tied is also returned (pre-canonicalisation float-equality
    count) so both numbers are always visible side by side."""
    sims = C @ S.T
    n_float_tied = None
    if ident is not None:
        eq = np.empty(len(gt_idx), dtype=bool)
        for qi in range(len(gt_idx)):
            s = sims[qi]
            eq[qi] = int((s == s[gt_idx[qi]]).sum()) > 1
        n_float_tied = eq
        for g in ident:
            sims[:, g[1:]] = sims[:, g[0]][:, None]
    ranks = np.empty(len(gt_idx), dtype=np.float64)
    ntied = np.empty(len(gt_idx), dtype=np.int64)
    for qi in range(len(gt_idx)):
        s = sims[qi]
        sg = s[gt_idx[qi]]
        n_above = int((s > sg).sum())
        n_eq = int((s == sg).sum())      # includes gt itself
        ranks[qi] = n_above + (n_eq + 1) / 2.0
        ntied[qi] = n_eq - 1
    return ranks, ntied, n_float_tied


def legacy_strict(S, C, gt_idx, sel):
    """EXACT copy of run_baseline.py's metric block (argsort rank),
    restricted to query indices `sel` — used ONLY for the G6 comparison."""
    ranks = []
    for qi in sel:
        sims = C[qi] @ S.T
        rank = int((np.argsort(-sims) == gt_idx[qi]).nonzero()[0][0]) + 1
        ranks.append(rank)
    r = np.array(ranks)
    return {"recall@1": float((r <= 1).mean()),
            "recall@5": float((r <= 5).mean()),
            "recall@10": float((r <= 10).mean()),
            "mrr": float((1.0 / r).mean()),
            "median_rank": float(np.median(r)),
            "n_queries": len(r), "pool_size": S.shape[0]}


def group_metrics(ranks, sel):
    r = ranks[sel]
    return {"n": int(sel.sum()), "mrr": float((1.0 / r).mean()),
            "recall@10": float((r <= 10).mean())}


# ------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="all local gates + counts + cost; no API calls")
    ap.add_argument("--yes", action="store_true", help="proceed to embed")
    ap.add_argument("--sushi-groups", required=True)
    ap.add_argument("--truce-groups", required=True)
    ap.add_argument("--baseline", required=True,
                    help="results/experiments/baseline_openai_embed.json")
    ap.add_argument("--seeds", nargs=3, type=int, required=True,
                    help="arm labels for the M3 hash, e.g. 42 43 44")
    ap.add_argument("--out-dir", default="results/experiments")
    ap.add_argument("--batch-signals", type=int, default=8)
    args = ap.parse_args()
    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- groups (G3 part 1: artifacts) ----
    sg = json.load(open(args.sushi_groups, encoding="utf-8"))
    if dict(sg["counts"]) != {"dependent": 135, "invariant": 4,
                              "degenerate": 1}:
        fail(f"G3: SUSHI artifact counts {sg['counts']} != 135/4/1")
    class_verdict = sg["class_verdicts"]
    tg = json.load(open(args.truce_groups, encoding="utf-8"))
    if not tg.get("certified"):
        fail("G3: TRUCE groups file is not the CERTIFIED artifact")
    truce_label = {t: rec["label"] for t, rec in tg["per_text"].items()}

    # ---- pool + queries ----
    pairs = load_pairs(splits=("test",))
    seen, sig_ids, sig_series = set(), [], []
    for p in pairs:
        if p.sample_id not in seen:
            seen.add(p.sample_id)
            sig_ids.append(p.sample_id)
            sig_series.append(p.series)
    id2idx = {s: i for i, s in enumerate(sig_ids)}
    gt_idx = np.array([id2idx[p.sample_id] for p in pairs])
    substrate = np.array(["sushi" if p.dataset == "sushi" else "truce"
                          for p in pairs])
    q_group = np.array([class_verdict[p.class_label] if p.dataset == "sushi"
                        else truce_label[p.caption] for p in pairs])

    print("=" * 74)
    print("SELF-REPORT — counts")
    print("=" * 74)
    cs = Counter(zip(substrate.tolist(), q_group.tolist()))
    print(f"  queries {len(pairs)}  pool {len(sig_ids)}  "
          f"groups {dict(sorted(cs.items()))}")
    gate(len(sig_ids) == EXPECTED_POOL, "G3-pool",
         f"pool {len(sig_ids)} == {EXPECTED_POOL}")
    gate(len(pairs) == EXPECTED_QUERIES and dict(cs) == GROUPS_EXPECTED,
         "G3-groups", "query-group counts match the certified grouping")

    # ---- SC-1: serialisation-path identity ----
    z_base = [znorm(np.asarray(x, dtype=np.float64)) for x in sig_series]
    ser_u = [serialize(x) for x in sig_series]
    mism = sum(1 for z, s in zip(z_base, ser_u) if quantize_z(z) != s)
    gate(mism == 0, "SC-1",
         f"quantize_z(znorm(x)) == serialize(x) for {len(sig_ids)-mism}"
         f"/{len(sig_ids)} pool signals")

    # ---- znorm-commute self-check (float noise; as in the CLaSP runner) ----
    rng0 = np.random.default_rng(0)
    picks = [0, next(i for i, s in enumerate(sig_ids)
                     if s.startswith("sushi"))]
    diffs = []
    for i in picks:
        raw = np.asarray(sig_series[i], dtype=np.float64)
        perm = rng0.permutation(len(raw))
        diffs.append(float(np.max(np.abs(znorm(raw[perm])
                                         - znorm(raw)[perm]))))
    print(f"  self-check znorm-commute max diffs: "
          f"{['%.2e' % d for d in diffs]} (expected float noise)")

    # ---- duplicate serialisations => guaranteed exact ties ($0, pre-spend) --
    by_str = defaultdict(list)
    for s, t in zip(sig_ids, ser_u):
        by_str[t].append(s)
    dup_groups = sorted([v for v in by_str.values() if len(v) > 1],
                        key=lambda g: g[0])
    dup_sushi = [g for g in dup_groups
                 if any(s.startswith("sushi") for s in g)]
    dup_members = {s for g in dup_groups for s in g}
    n_tied_pred = int(sum(1 for p in pairs if p.sample_id in dup_members))
    print(f"\nDUPLICATE SERIALISATIONS (exact ties by construction)")
    for g in dup_groups:
        print(f"    {g}")
    print(f"  tie-affected queries: {n_tied_pred}  "
          f"[registered: point 24, range {TIE_RANGE[0]}–{TIE_RANGE[1]}]")
    gate(len(dup_sushi) == 0, "SC-dup-sushi",
         f"{len(dup_sushi)} SUSHI serialisation collisions (registered: 0; "
         f"a collision would also break G6a's tie-immunity premise)")
    if not (TIE_RANGE[0] <= n_tied_pred <= TIE_RANGE[1]):
        print("  NOTE: outside the registered range — investigate before "
              "trusting results", file=sys.stderr)

    # ---- G2: natural zeros, z-level and token-level ----
    for sub, pref in [("truce", ("truce",)), ("sushi", ("sushi",))]:
        idxs = [i for i, s in enumerate(sig_ids) if s.startswith(pref)]
        tot = sum(len(z_base[i]) for i in idxs)
        zz = sum(int((z_base[i] == 0.0).sum()) for i in idxs)
        t0 = sum(ser_u[i].split(",").count("0") for i in idxs)
        print(f"  G2 [{sub}]: exact z-zeros {zz}/{tot} ({zz/tot:.4%}); "
              f"quantised-'0' tokens {t0}/{tot} ({t0/tot:.4%}) — report only")

    # ---- G8 targets (registered: EMPTY — the ±1 z-norm finding) ----
    const_idx = [i for i, z in enumerate(z_base) if np.ptp(z) == 0.0]
    print(f"  G8 targets (ptp==0 z-series): "
          f"{[sig_ids[i] for i in const_idx]}   [registered: empty]")

    # ---- build all perturbed serialisations (local, $0) ----
    # cond[(arm, pert)] = list of strings in sig_ids order
    cond, meta_all = {}, {}
    for arm in args.seeds:
        for pert in PERTS:
            strs, flags, seeds_rec, mask_noop_zero = [], [], {}, 0
            for i, s in enumerate(sig_ids):
                sd = signal_seed(s, pert, arm)
                seeds_rec[s] = sd
                zp, meta = perturb(z_base[i], pert,
                                   np.random.default_rng(sd))
                flags.append(applied_check(z_base[i], zp, pert, meta))
                t = quantize_z(zp)
                if pert in ("sf_all", "sf_half", "ex_half"):
                    if sorted(t.split(",")) != sorted(ser_u[i].split(",")):
                        fail(f"G1-ser: {pert} changed the serialized token "
                             f"multiset for {s}")
                strs.append(t)
            noops = [sig_ids[i] for i, f in enumerate(flags) if f == "noop"]
            # no-op identity: byte-identical serialisation => same cache key
            for s in noops:
                if strs[id2idx[s]] != ser_u[id2idx[s]]:
                    fail(f"G8-noop: {s} flagged no-op under {pert} but its "
                         f"serialisation changed")
            # G8 structural targets under the shuffle family
            if pert != "masking":
                for i in const_idx:
                    if strs[i] != ser_u[i]:
                        fail(f"G8: constant signal {sig_ids[i]} serialisation "
                             f"changed under {pert}")
            cond[(arm, pert)] = strs
            meta_all[(arm, pert)] = {"signal_seeds": seeds_rec,
                                     "noop_ids": noops}
            print(f"  [arm {arm}][{pert}] G1/G7 PASSED over {len(sig_ids)} "
                  f"signals; no-ops: {len(noops)} {noops[:4]}")

    # ---- SC-2: ex_half deterministic across arms ----
    a0 = args.seeds[0]
    for arm in args.seeds[1:]:
        gate(cond[(arm, "ex_half")] == cond[(a0, "ex_half")], "SC-2",
             f"ex_half serialisations arm {arm} == arm {a0} (deterministic)")

    # ---- token / cost report ----
    cache = load_cache()
    print(f"\ncache: {len(cache)} vectors on disk")
    uniq_new = sorted({t for strs in cond.values() for t in strs
                       if key_of(t) not in cache})
    toks = count_tokens(uniq_new) if uniq_new else []
    over = sum(1 for n in toks if n > MAX_TOKENS)
    total = sum(toks)
    print("TOKENS / COST (perturbed strings only; unperturbed are cached)")
    print(f"  unique NEW strings to embed: {len(uniq_new)} "
          f"(of {sum(len(v) for v in cond.values())} condition-signal slots; "
          f"dedup across arms/conditions/duplicates)")
    print(f"  total NEW tokens: {total:,}   cost ceiling "
          f"${total / 1e6 * PRICE_PER_1M:.2f}   [registered ceiling ~$0.76; "
          f"approved budget ~$1]")
    gate(over == 0, "G-token-limit", f"{over} strings over {MAX_TOKENS} tokens")
    if uniq_new:
        print(f"  example perturbed serialisation, first 120 chars:\n"
              f"    {uniq_new[0][:120]} ...")

    # ---- G6-pre: baseline reproduction must be fully cached ----
    missing_sig = [s for s, t in zip(sig_ids, ser_u) if key_of(t) not in cache]
    captions = sorted({p.caption for p in pairs})
    missing_cap = [c for c in captions if key_of(c) not in cache]
    gate(not missing_sig and not missing_cap, "G6-pre",
         f"unperturbed strings cached: signals missing {len(missing_sig)}, "
         f"captions missing {len(missing_cap)} (baseline must have run; "
         f"reproduction costs $0)")

    # ---- G6: reproduce the frozen baseline (legacy argsort, like-for-like) --
    cap_vec = {c: cache[key_of(c)] for c in captions}
    C = np.stack([cap_vec[p.caption] for p in pairs])
    S_u = np.stack([cache[key_of(t)] for t in ser_u])
    frozen = json.load(open(args.baseline, encoding="utf-8"))["strict"]
    sels = {"all": np.arange(len(pairs)),
            "truce": np.where(substrate == "truce")[0],
            "sushi": np.where(substrate == "sushi")[0]}
    rep = {k: legacy_strict(S_u, C, gt_idx, sel) for k, sel in sels.items()}
    METRICS = ("recall@1", "recall@5", "recall@10", "mrr")
    for k in ("all", "truce", "sushi"):
        for m in METRICS:
            d = abs(rep[k][m] - frozen[k][m])
            if d > 1e-9:
                print(f"  dev {k}/{m}: ours {rep[k][m]:.12f} frozen "
                      f"{frozen[k][m]:.12f} |d|={d:.2e} "
                      f"(~{d*rep[k]['n_queries']:.2f} query-steps)")
    dev_a = max(abs(rep["sushi"][m] - frozen["sushi"][m]) for m in METRICS)
    if dev_a > 1e-9:
        fail(f"G6a: SUSHI metrics not reproduced (dev {dev_a:.2e}) — "
             f"no SUSHI serialisation collisions exist, so this is NOT "
             f"tie-explainable: real drift")
    print(f"  G6a (sushi, digit-exact): max |dev| = {dev_a:.2e}  PASS")
    for k in ("truce", "all"):
        n = rep[k]["n_queries"]
        for m in METRICS:
            d = abs(rep[k][m] - frozen[k][m])
            bound = (0.5 * n_tied_pred / n) if m == "mrr" \
                else (n_tied_pred / n)
            if d > bound + 1e-12:
                fail(f"G6b: {k}/{m} deviation {d:.2e} exceeds the "
                     f"tie-explainable bound {bound:.2e}")
    print(f"  G6b (truce/all, tie bound from {n_tied_pred} tied queries): "
          f"PASS  [registered: zero deviation]")

    # ---- D2 unperturbed ranks + tie count (D2-F identity groups) ----
    ident_u = identity_groups(ser_u)
    ranks_u, ntied_u, ft_u = avg_ranks(S_u, C, gt_idx, ident=ident_u)
    tied = {"truce": int(((substrate == "truce") & (ntied_u > 0)).sum()),
            "sushi": int(((substrate == "sushi") & (ntied_u > 0)).sum())}
    print(f"  D2-F unperturbed tied queries (identity level): {tied}  "
          f"[registered: truce exactly {n_tied_pred}, sushi 0]")
    print(f"  float-equality ties before canonicalisation: "
          f"{int(ft_u.sum())}  [diagnosed 2026-08-14: 22]")
    if tied["truce"] != n_tied_pred or tied["sushi"] != 0:
        fail(f"D2-F: identity-level tie count {tied} differs from the "
             f"serialisation-derived count ({n_tied_pred}/0) — these are "
             f"two computations of the same fact and MUST agree")

    if args.dry_run or not args.yes:
        print("\ndry run — no API calls made. Re-run with --yes to embed.")
        return

    # ---- embed perturbed strings (cache appended by embed_all) ----
    vec = embed_all(uniq_new, cache, args.batch_signals, "perturbed signals")
    # embed_all returns {text: vector} for its inputs; cached ones we fetch:
    def vec_of(t):
        return vec[t] if t in vec else cache[key_of(t)]

    # ---- rank every condition; gates GF; write records ----
    summary = {"model": MODEL, "mask_ratio": MASK_RATIO, "perts": PERTS,
               "seed_scheme": "int(sha256('{sample_id}|{pert}|{arm_seed}')"
                              "[:12hex],16)",
               "n_new_strings": len(uniq_new), "new_tokens": int(total),
               "tied_queries_unperturbed": tied,
               "duplicate_serialisation_groups": dup_groups,
               "arms": {}}
    for arm in args.seeds:
        ranks_by_pert, tables = {}, {}
        for pert in PERTS:
            S_p = np.stack([vec_of(t) for t in cond[(arm, pert)]])
            norms_ok = (np.abs(np.linalg.norm(S_p, axis=1) - 1.0) < 1e-3).all()
            if not norms_ok:
                fail(f"unit-norm violated in arm {arm}/{pert}")
            ident_p = identity_groups(cond[(arm, pert)])
            r, nt, ft = avg_ranks(S_p, C, gt_idx, ident=ident_p)
            ranks_by_pert[pert] = r
            print(f"[arm {arm}][{pert}] identity groups: {len(ident_p)}; "
                  f"tied queries after perturbation: {int((nt > 0).sum())} "
                  f"(float-level before canonicalisation: {int(ft.sum())})")
        # GF gate + tables
        print(f"\narm {arm}: per-group MRR (unpert | " +
              " | ".join(PERTS) + ")  [n]")
        for sub in ("sushi", "truce"):
            for grp in sorted(set(q_group[substrate == sub])):
                sel = (substrate == sub) & (q_group == grp)
                row = {"unperturbed": group_metrics(ranks_u, sel)}
                for pert in PERTS:
                    row[pert] = group_metrics(ranks_by_pert[pert], sel)
                tables[f"{sub}/{grp}"] = row
                print(f"  {sub}/{grp:10s} {row['unperturbed']['mrr']:.4f} | "
                      + " | ".join(f"{row[p]['mrr']:.4f}" for p in PERTS)
                      + f"   [{row['unperturbed']['n']}]")
        for sub, grp in GF_CELLS:
            sel = (substrate == sub) & (q_group == grp)
            mu = group_metrics(ranks_u, sel)["mrr"]
            for pert in PERTS:
                mp = group_metrics(ranks_by_pert[pert], sel)["mrr"]
                if mp - mu > GF_MARGIN:
                    fail(f"GF: floor {sub}/{grp} MRR rose {mu:.4f} -> "
                         f"{mp:.4f} under {pert} (arm {arm}) — capability-"
                         f"scale movement in a floor = broken pipeline")
        print(f"  GF PASSED (no gated cell rose by > {GF_MARGIN} MRR)")

        rec_path = outdir / f"probe2_openai_per_query_seed{arm}.jsonl"
        with open(rec_path, "w", encoding="utf-8") as f:
            for qi, p in enumerate(pairs):
                f.write(json.dumps({
                    "caption_id": p.caption_id, "dataset": p.dataset,
                    "substrate": substrate[qi], "group": q_group[qi],
                    "gt": p.sample_id,
                    "rank_unperturbed": float(ranks_u[qi]),
                    "ntied_unperturbed": int(ntied_u[qi]),
                    **{f"rank_{pt}": float(ranks_by_pert[pt][qi])
                       for pt in PERTS}}) + "\n")
        meta_path = outdir / f"probe2_openai_signal_meta_seed{arm}.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({pt: meta_all[(arm, pt)] for pt in PERTS}, f, indent=2)
        print(f"  wrote {rec_path.name} and {meta_path.name}")
        summary["arms"][str(arm)] = {"tables": tables}

    with open(outdir / "probe2_openai_summary.json", "w",
              encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {outdir / 'probe2_openai_summary.json'}")
    print("ALL GATES PASSED — per-query records ready for the stats step")
    print("Reading aid: a floor cannot 'degrade'. The tables should show "
          "small jiggle around tiny baselines; the P2-4 VOID verdict is "
          "scored by the stats script, not read from these tables.")


if __name__ == "__main__":
    main()

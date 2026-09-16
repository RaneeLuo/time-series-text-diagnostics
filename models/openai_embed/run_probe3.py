"""
run_probe3.py (text-embedding-3-large) — Probe-3 floor negative control (P3-7).

The floor model is PRE-DECLARED VOID for Probe 3 (design record, handoff
§4.8, accepted 2026-08-15): baseline MRR 0.027 vs chance 0.017 — there is
no capability for a surrogate to degrade. This run exists as the pipeline's
negative control; the registered expectation (P3-7) is the P2-4 pattern:
TOSTs pass at ±0.05 ABSOLUTE MRR on the inference cells, no movement
anywhere beyond floor jiggle.

CONDITIONS (the ladder's two new rungs; rung 2 = sf_all is READ from the
committed floor Probe-2 records by the stats script via the JG join gate,
never rerun here):
  resample  i.i.d. draw WITH replacement from the series' own RAW values
            (same length).
  gaussian  i.i.d. Normal(raw mean, raw std), same length; ptp==0
            degenerate pass-through (the matched gaussian of a truly
            constant series IS the constant).

CONSTRUCTION is RAW-level, then the model's own preprocessing (Q1/Q2):
surrogate built on the raw series, THEN serialize() — which is the floor's
entire native pipeline (float64 cast + znorm + x10 quantise + clip ±99 +
comma-join). There is no two-path serialisation here (Probe 2 needed SC-1
because M1 masking forced z-level perturbation; Probe 3 perturbs raw and
feeds serialize directly), so SC-1 is retired for this runner.

DTYPE NOTE (deliberate, not a repeat of error #14): raw series are cast to
float64 at load. The CLaSP v2 rule "keep native float32" exists because
CLaSP's pipeline is float32 end-to-end and dataset.znorm branches
differently per dtype. The floor's native path is float64 — serialize()
performs the cast itself — and its local znorm (sd<1e-8 guard) takes the
ZEROS branch on the constant signal. That zeros-string is what the
committed floor baseline has cached (Probe-2 floor arc, 2026-08-14: G8
targets NOT empty on this pipeline). Following the model's own path is
the Q1 rule.

Seed scheme (M3 pattern, documented reuse): per-signal seed =
int(sha256("{sample_id}|{condition}|{arm_seed}")[:12 hex], 16). The floor
has no checkpoints; arms 42/43/44 are labels that vary only the draws,
mirroring the CLaSP matrix (decision of 2026-08-14).

Determinism is supplied by the CACHE — every unique string embedded once,
ever, and reused. Unlike Probe 2 there is no ex_half analogue: BOTH
conditions are RNG-dependent and the arm label enters the seed hash, so
NO cross-arm string sharing is expected (dedup count printed; any hit is
a bonus, not an expectation).

Gates (fatal policy mirrors the parents; Q5 record §4.8):
  G3    grouping counts vs certified artifacts AND query-side counts:
        SUSHI 135/4/1 (dep/inv/degen), TRUCE 715/18/5 (dep/inv/amb),
        queries 878, pool 386 — HARD STOP
  G10   degenerate census on the RAW pool: raw-constant signals must
        equal the registered list ['sushi:clean\\00\\0000009'] exactly;
        TRUCE degenerates 0; any non-finite anywhere — HARD STOP
  G1/G7 applied-check per signal per condition (resample values subset of
        raw, length preserved, indices in range; gaussian moments within
        6-sigma sampling bands; seed recorded; no-ops FLAGGED never
        failed) — HARD STOP on real violations
  G8    identity control, EXACT for the floor: the constant signal's
        surrogate serialisation must be BYTE-IDENTICAL to its cached
        unperturbed string under BOTH conditions (same cache key), and
        its POOLED VECTOR bitwise-equal to the unperturbed vector.
        Likewise every flagged no-op — HARD STOP. Its RANK is NOT
        gated: whole-pool replacement moves the other 385 signals, so
        the constant's rank legitimately shifts (v2 correction — the
        Probe-2 record "G8 is embedding-identity only" applies here
        too). Constant-GT rank movement is printed as a REPORT.
  G6-pre unperturbed signal strings + caption strings all in cache
        (baseline reproduction costs $0) — HARD STOP
  G6    digit-exact reproduction of baseline_openai_embed.json with the
        SAME legacy argsort rank as run_baseline.py: G6a sushi <=1e-9
        (tie-free premise: zero SUSHI serialisation collisions, count
        printed); G6b truce/all within the tie-explainable bound from
        the 24 construction-forced tied queries — HARD STOP beyond.
        Registered expectation: ZERO deviation (same machine, cache).
  G-cost the --yes path REFUSES to spend outside the pre-registered
        $0.30–0.50 band — HARD STOP (dry run prints the exact number;
        an out-of-band estimate is investigated, not overridden)
  G4    direction sanity (buffered): overall gaussian MRR must not
        exceed overall resample MRR by more than 0.02. At floor level
        this cannot detect a condition swap (both sit at chance); it is
        kept as GF's sibling — capability-scale separation between two
        chance conditions = broken pipeline — HARD STOP
  GF    (inverted direction gate, from the Probe-2 floor): on the two
        large certified cells (sushi/dependent n=135, truce/dependent
        n=715), perturbed MRR must not EXCEED unperturbed by more than
        0.05 in any arm/condition — HARD STOP
  G9    pairing: query tuple list asserted identical across conditions;
        similarity-matrix row order is the one sig_ids order — HARD STOP
  G12   draw-quality REPORT (P3-2c evidence base): per substrate, mean
        frac of raw positions never drawn and mean frac of unique raw
        values missing from the resample (expected ~0.35–0.37); plus the
        Probe-3 coarseness line — mean unique QUANTISED tokens per TRUCE
        serialisation (CLaSP-arm mechanism-corrected expectation ~9.6/12)
  G2    quantised-'0' token rate per substrate on the surrogate strings
        (REPORT ONLY)

Rank metric: D2 deterministic AVERAGE RANK with the D2-F amendment
(accepted 2026-08-14, floor arm): ties detected at CONSTRUCTION level —
identical serialisation strings => identical cached vectors => their
similarity COLUMNS are equalised before ranking (BLAS column paths split
bitwise-identical vectors at ~1 ulp; float-equality found 22 of 24 forced
ties, scripts/diagnose_floor_ties.py). Float-level counts printed
alongside in every condition. Legacy argsort rank is used only inside G6.

Registered expectations (2026-08-15, BEFORE first run — this session):
    G3: queries 878, pool 386, groups exact
    G10: degenerates exactly ['sushi:clean\\00\\0000009']; TRUCE 0
    no-ops per condition-set: exactly 1 (the constant) under BOTH
        conditions (resample of a constant is itself; gaussian ptp==0
        pass-through); TRUCE resample no-ops 0
    G8: the constant's surrogate string byte-identical to its cached
        unperturbed (zeros-branch) string in all 6 condition-sets, and
        its pooled vector bitwise-equal; rank movement REPORTED only
    RERUN (post-spend, v2): 0 new strings, $0 (all 2,310 cached);
        G-cost passes on the zero-spend branch; arm 42 resample
        constant-GT query 739 movement 298.0 -> 273.0 reproduced
        exactly (deterministic: cached vectors, fixed seeds)
    D2-F unperturbed ties: truce 24 / sushi 0; duplicate groups
        {58,86,249,362} and {87,165,326,360}; float-level count printed
        (diagnosed 22 on this machine, report not gate)
    per-condition identity groups: 0, tied queries 0 in every arm x
        condition (per-signal seeds dissolve the duplicate groups);
        any chance collision is printed and handled by D2-F correctly
    unique NEW strings: point 2,310 (= 6 sets x 385 non-constant pool
        signals), range 2,290–2,310 (12-token TRUCE quantised collisions
        possible, rare)
    NEW tokens ~3.3–3.7M; cost point ~$0.45, MUST land in $0.30–0.50
    G6: zero deviation on every metric; G6-pre: 0 missing
    G12: frac ~0.35–0.37 both substrates; TRUCE unique tokens ~9.6/12
    GF/G4: no gated cell rises > 0.05; expected jiggle |dMRR| <~ 0.02
        (Probe-2 floor max inference-cell |d| was 0.0105)
    P3-7 verdict (scored later by the stats script, pinned): P2-4
        pattern — TOST ±0.05 abs passes on all inference cells, both
        conditions, all arms; VOID everywhere

V2 (2026-08-15), after the first --yes run's G8-rank stop at arm 42:
  ERROR (Claude's, delivered script, ledger #15): v1 gated the
  constant-GT query's RANK as unchanged. False inference — the vector
  is unchanged but the rank is measured against a fully replaced pool,
  so ~25 surrogate signals moving below the constant moved its rank
  298 -> 273. The Probe-2 CLaSP arc had already recorded this exact
  correction ("G8 is embedding-identity only", 2026-08-13); v1 failed
  to carry it. The spend is unaffected: all 2,310 vectors were cached
  before the stop; no result files were written.
  FIXES: (i) G8 post-rank check replaced by bitwise vector identity +
  rank-movement REPORT; (ii) G-cost passes explicitly when 0 new
  strings are needed (a $0 rerun is below the band by design, not a
  violation).

Usage (PowerShell):
    python -m models.openai_embed.run_probe3 --dry-run `
        --sushi-groups results/analysis/probe2_sushi_groups.json `
        --truce-groups results/analysis/probe2_truce_groups_certified.json `
        --baseline results/experiments/baseline_openai_embed.json `
        --seeds 42 43 44
    ... then the same command with --yes instead of --dry-run.

Writes:
    results/experiments/probe3_openai_per_query_seed{S}.jsonl
    results/experiments/probe3_openai_signal_meta_seed{S}.json
    results/experiments/probe3_openai_summary.json
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
    serialize, count_tokens, load_cache, embed_all, key_of,
    MODEL, MAX_TOKENS, PRICE_PER_1M,
)

CONDITIONS = ["resample", "gaussian"]
REGISTERED_DEGENERATES = ["sushi:clean\\00\\0000009"]
G4_BUFFER = 0.02
GF_CELLS = [("sushi", "dependent"), ("truce", "dependent")]
GF_MARGIN = 0.05
COST_BAND = (0.30, 0.50)          # pre-registered; --yes refuses outside
TIE_PRED = 24                     # unperturbed D2-F ties (Probe-2 floor)

EXPECTED_POOL = 386
EXPECTED_QUERIES = 878
GROUPS_EXPECTED = {("sushi", "dependent"): 135, ("sushi", "invariant"): 4,
                   ("sushi", "degenerate"): 1, ("truce", "dependent"): 715,
                   ("truce", "invariant"): 18, ("truce", "ambiguous"): 5}


def fail(msg):
    print(f"\nGATE FAILED: {msg}", file=sys.stderr)
    sys.exit(1)


def gate(ok: bool, name: str, msg: str):
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name}: {msg}")
    if not ok:
        fail(f"{name} — {msg}")


# ------------------------------------------------------------- construction
def signal_seed(sample_id: str, cond: str, arm_seed: int) -> int:
    h = hashlib.sha256(f"{sample_id}|{cond}|{arm_seed}".encode()).hexdigest()
    return int(h[:12], 16)


def build_surrogate(raw: np.ndarray, cond: str, rng: np.random.Generator):
    """Raw-level surrogate; identical mechanics to the CLaSP v2 runner."""
    L = len(raw)
    meta = {}
    if cond == "resample":
        idx = rng.integers(0, L, size=L)
        if idx.min() < 0 or idx.max() >= L:
            fail("G7: resample indices out of range")
        meta["frac_positions_never_drawn"] = float(
            1.0 - len(np.unique(idx)) / L)
        u_raw = np.unique(raw)
        u_out = np.unique(raw[idx])
        meta["frac_unique_values_missing"] = float(
            1.0 - len(u_out) / len(u_raw))
        return raw[idx], meta
    if cond == "gaussian":
        mu, sd = float(raw.mean()), float(raw.std())
        meta["target_mu"], meta["target_sd"] = mu, sd
        if float(np.ptp(raw)) == 0.0:
            meta["degenerate_passthrough"] = True
            return raw.copy(), meta
        return rng.normal(mu, sd, size=L), meta
    fail(f"unknown condition {cond}")


def applied_check(raw, sur, cond, meta):
    """G1. Returns 'ok' | 'noop'; hard-fails on real violations."""
    if len(sur) != len(raw):
        fail(f"G1: {cond} changed the length {len(raw)} -> {len(sur)}")
    if not np.all(np.isfinite(sur)):
        fail(f"G1/G10: {cond} produced a non-finite value")
    if cond == "resample":
        if not np.isin(sur, raw).all():
            fail("G1: resample emitted a value not present in the raw series")
        return "noop" if np.array_equal(sur, raw) else "ok"
    # gaussian
    if meta.get("degenerate_passthrough"):
        return "noop"
    L = len(raw)
    mu, sd = meta["target_mu"], meta["target_sd"]
    band_mu = 6.0 * sd / np.sqrt(L)
    band_sd = 6.0 * sd / np.sqrt(2.0 * L)
    if abs(float(sur.mean()) - mu) > band_mu:
        fail(f"G1: gaussian sample mean {sur.mean():.4g} outside 6-sigma "
             f"band of target {mu:.4g} (band {band_mu:.4g})")
    if abs(float(sur.std()) - sd) > band_sd:
        fail(f"G1: gaussian sample std {sur.std():.4g} outside 6-sigma "
             f"band of target {sd:.4g} (band {band_sd:.4g})")
    return "ok"


# ------------------------------------------------------------- rank metrics
def identity_groups(strings):
    """Pool indices sharing an identical serialisation string (D2-F)."""
    by = defaultdict(list)
    for i, t in enumerate(strings):
        by[t].append(i)
    return [v for v in by.values() if len(v) > 1]


def avg_ranks(S: np.ndarray, C: np.ndarray, gt_idx: np.ndarray, ident=None):
    """D2 average rank + tie counts; D2-F column canonicalisation.
    Verbatim mechanics from run_probe2.py (floor v2)."""
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
    """EXACT copy of run_baseline.py's metric block — G6 comparison only."""
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
    seen, sig_ids, sig_raw = set(), [], []
    for p in pairs:
        if p.sample_id not in seen:
            seen.add(p.sample_id)
            sig_ids.append(p.sample_id)
            # float64 by DESIGN for the floor (see dtype note in docstring):
            # serialize() casts to float64 itself; this is the model-native
            # path, unlike CLaSP v2's native-float32 rule.
            sig_raw.append(np.asarray(p.series, dtype=np.float64))
    id2idx = {s: i for i, s in enumerate(sig_ids)}
    gt_idx = np.array([id2idx[p.sample_id] for p in pairs])
    substrate = np.array(["sushi" if p.dataset == "sushi" else "truce"
                          for p in pairs])
    q_group = np.array([class_verdict[p.class_label] if p.dataset == "sushi"
                        else truce_label[p.caption] for p in pairs])
    flucts = {qi: p.class_label.split(";")[0].strip()
              for qi, p in enumerate(pairs) if p.dataset == "sushi"}

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

    # ---- G10: degenerate census on the RAW pool ----
    degen = sorted(s for s, r in zip(sig_ids, sig_raw)
                   if float(np.ptp(r)) == 0.0)
    print(f"  G10 raw-constant pool signals: {degen}")
    gate(degen == sorted(REGISTERED_DEGENERATES), "G10",
         f"degenerate list == registered {REGISTERED_DEGENERATES}")
    gate(sum(1 for s in degen if not s.startswith("sushi")) == 0,
         "G10-truce", "0 TRUCE degenerate series (certified 0)")
    const_idx = [id2idx[s] for s in degen]

    # ---- unperturbed serialisations (the committed baseline strings) ----
    ser_u = [serialize(r) for r in sig_raw]

    # ---- duplicate serialisations => forced ties ($0, pre-spend) ----
    by_str = defaultdict(list)
    for s, t in zip(sig_ids, ser_u):
        by_str[t].append(s)
    dup_groups = sorted([v for v in by_str.values() if len(v) > 1],
                        key=lambda g: g[0])
    dup_sushi = [g for g in dup_groups
                 if any(s.startswith("sushi") for s in g)]
    dup_members = {s for g in dup_groups for s in g}
    n_tied_pred = int(sum(1 for p in pairs if p.sample_id in dup_members))
    print(f"\nDUPLICATE SERIALISATIONS (unperturbed; ties by construction)")
    for g in dup_groups:
        print(f"    {g}")
    print(f"  tie-affected queries: {n_tied_pred}  [registered: {TIE_PRED}]")
    gate(len(dup_sushi) == 0, "SC-dup-sushi",
         f"{len(dup_sushi)} SUSHI serialisation collisions (registered 0; "
         f"G6a's tie-immunity premise)")
    if n_tied_pred != TIE_PRED:
        print("  NOTE: differs from the registered 24 — investigate before "
              "trusting results", file=sys.stderr)

    # ---- self-check: resample does NOT commute with the serialize path ----
    demo_i = next(i for i, s in enumerate(sig_ids)
                  if s.startswith("sushi") and i not in const_idx)
    rr = sig_raw[demo_i]
    ridx = np.random.default_rng(0).integers(0, len(rr), size=len(rr))
    print(f"  self-check non-commutation on {sig_ids[demo_i]}: resample "
          f"mean shift {rr[ridx].mean() - rr.mean():+.4g}, std ratio "
          f"{rr[ridx].std() / rr.std():.4f} (EXPECTED nonzero — why "
          f"construction is raw-level, then serialize())")

    # ---- build all surrogate serialisations (local, $0) ----
    cond, meta_all = {}, {}
    g12 = {("resample", "sushi"): {"pos": [], "val": []},
           ("resample", "truce"): {"pos": [], "val": []}}
    for arm in args.seeds:
        for cnd in CONDITIONS:
            strs, flags, seeds_rec = [], [], {}
            for i, s in enumerate(sig_ids):
                sd = signal_seed(s, cnd, arm)
                seeds_rec[s] = sd
                sur, meta = build_surrogate(sig_raw[i], cnd,
                                            np.random.default_rng(sd))
                flags.append(applied_check(sig_raw[i], sur, cnd, meta))
                strs.append(serialize(sur))
                if cnd == "resample":
                    sub = "sushi" if s.startswith("sushi") else "truce"
                    g12[("resample", sub)]["pos"].append(
                        meta["frac_positions_never_drawn"])
                    g12[("resample", sub)]["val"].append(
                        meta["frac_unique_values_missing"])
            noops = [sig_ids[i] for i, f in enumerate(flags) if f == "noop"]
            # G8: no-op => byte-identical serialisation => same cache key
            for s in noops:
                if strs[id2idx[s]] != ser_u[id2idx[s]]:
                    fail(f"G8-noop: {s} flagged no-op under {cnd} but its "
                         f"serialisation changed")
            # G8: the constant must be a no-op under BOTH conditions
            for i in const_idx:
                if strs[i] != ser_u[i]:
                    fail(f"G8: constant {sig_ids[i]} serialisation changed "
                         f"under {cnd} (arm {arm})")
                if sig_ids[i] not in noops:
                    fail(f"G8: constant {sig_ids[i]} not flagged no-op "
                         f"under {cnd} (arm {arm})")
            cond[(arm, cnd)] = strs
            meta_all[(arm, cnd)] = {"signal_seeds": seeds_rec,
                                    "noop_ids": noops}
            print(f"  [arm {arm}][{cnd}] G1/G7/G8/G10 PASSED over "
                  f"{len(sig_ids)} signals; no-ops: {len(noops)} {noops[:4]} "
                  f"[registered: exactly 1, the constant]")

    # ---- G12 report (resample draw quality + TRUCE coarseness) ----
    print("\nG12 (report only)")
    for (cnd, sub), d in g12.items():
        print(f"  [{cnd}][{sub}] mean frac positions never drawn "
              f"{np.mean(d['pos']):.4f}; mean frac unique raw values "
              f"missing {np.mean(d['val']):.4f}  [expected ~0.35-0.37]")
    tr_idx = [i for i, s in enumerate(sig_ids) if not s.startswith("sushi")]
    uq = [len(set(cond[(args.seeds[0], 'resample')][i].split(",")))
          for i in tr_idx]
    print(f"  [resample][truce] mean unique quantised tokens per "
          f"serialisation: {np.mean(uq):.2f}/12  [CLaSP-arm corrected "
          f"expectation ~9.6]")

    # ---- G2 report: quantised-'0' token rate on surrogate strings ----
    for cnd in CONDITIONS:
        for sub, pref in [("truce", False), ("sushi", True)]:
            idxs = [i for i, s in enumerate(sig_ids)
                    if s.startswith("sushi") == pref]
            tot = sum(len(cond[(args.seeds[0], cnd)][i].split(","))
                      for i in idxs)
            t0 = sum(cond[(args.seeds[0], cnd)][i].split(",").count("0")
                     for i in idxs)
            print(f"  G2 [{cnd}][{sub}] quantised-'0' tokens (arm "
                  f"{args.seeds[0]}): {t0}/{tot} ({t0/tot:.4%}) — report")

    # ---- token / cost report + G-cost band ----
    cache = load_cache()
    print(f"\ncache: {len(cache)} vectors on disk")
    all_slots = sum(len(v) for v in cond.values())
    uniq_new = sorted({t for strs in cond.values() for t in strs
                       if key_of(t) not in cache})
    toks = count_tokens(uniq_new) if uniq_new else []
    over = sum(1 for n in toks if n > MAX_TOKENS)
    total = sum(toks)
    cost = total / 1e6 * PRICE_PER_1M
    print("TOKENS / COST (surrogate strings only; unperturbed are cached)")
    print(f"  unique NEW strings to embed: {len(uniq_new)} of {all_slots} "
          f"condition-signal slots  [registered: point 2,310, "
          f"range 2,290-2,310]")
    print(f"  total NEW tokens: {total:,}   cost ${cost:.2f}   "
          f"[registered band ${COST_BAND[0]:.2f}-{COST_BAND[1]:.2f}, "
          f"point ~$0.45]")
    gate(over == 0, "G-token-limit", f"{over} strings over {MAX_TOKENS}")
    in_band = COST_BAND[0] <= cost <= COST_BAND[1]
    if not in_band and len(uniq_new) > 0:
        print(f"  NOTE: cost ${cost:.2f} outside the registered band — "
              f"--yes will refuse; investigate first", file=sys.stderr)
    if uniq_new:
        print(f"  example surrogate serialisation, first 120 chars:\n"
              f"    {uniq_new[0][:120]} ...")

    # ---- G6-pre: baseline reproduction must be fully cached ----
    missing_sig = [s for s, t in zip(sig_ids, ser_u) if key_of(t) not in cache]
    captions = sorted({p.caption for p in pairs})
    missing_cap = [c for c in captions if key_of(c) not in cache]
    gate(not missing_sig and not missing_cap, "G6-pre",
         f"unperturbed strings cached: signals missing {len(missing_sig)}, "
         f"captions missing {len(missing_cap)} (reproduction costs $0)")

    # ---- G6: reproduce the frozen baseline (legacy argsort) ----
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
                      f"{frozen[k][m]:.12f} |d|={d:.2e}")
    dev_a = max(abs(rep["sushi"][m] - frozen["sushi"][m]) for m in METRICS)
    if dev_a > 1e-9:
        fail(f"G6a: SUSHI metrics not reproduced (dev {dev_a:.2e}) — no "
             f"SUSHI collisions exist, so this is NOT tie-explainable")
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

    # ---- D2 unperturbed ranks (D2-F identity groups) ----
    ident_u = identity_groups(ser_u)
    ranks_u, ntied_u, ft_u = avg_ranks(S_u, C, gt_idx, ident=ident_u)
    tied = {"truce": int(((substrate == "truce") & (ntied_u > 0)).sum()),
            "sushi": int(((substrate == "sushi") & (ntied_u > 0)).sum())}
    print(f"  D2-F unperturbed tied queries (identity level): {tied}  "
          f"[registered: truce {TIE_PRED}, sushi 0]")
    print(f"  float-equality ties before canonicalisation: "
          f"{int(ft_u.sum())}  [diagnosed 2026-08-14: 22]")
    if tied["truce"] != n_tied_pred or tied["sushi"] != 0:
        fail(f"D2-F: identity-level tie count {tied} differs from the "
             f"serialisation-derived count ({n_tied_pred}/0)")

    if args.dry_run or not args.yes:
        print("\ndry run — no API calls made. Re-run with --yes to embed.")
        return

    if len(uniq_new) == 0:
        print("  [PASS] G-cost: 0 new strings — rerun from cache, $0 spend")
    else:
        gate(in_band, "G-cost",
             f"cost ${cost:.2f} inside the registered band "
             f"${COST_BAND[0]:.2f}-{COST_BAND[1]:.2f}")

    # ---- embed surrogate strings (cache appended by embed_all) ----
    vec = embed_all(uniq_new, cache, args.batch_signals, "surrogate signals")

    def vec_of(t):
        return vec[t] if t in vec else cache[key_of(t)]

    # ---- rank every condition; gates G4/GF/G8-rank; write records ----
    summary = {"model": MODEL, "conditions": CONDITIONS,
               "seed_scheme": "int(sha256('{sample_id}|{cond}|{arm_seed}')"
                              "[:12hex],16)",
               "registered_degenerates": REGISTERED_DEGENERATES,
               "g4_buffer": G4_BUFFER, "gf_margin": GF_MARGIN,
               "n_new_strings": len(uniq_new), "new_tokens": int(total),
               "cost_usd": round(cost, 4),
               "tied_queries_unperturbed": tied,
               "duplicate_serialisation_groups": dup_groups,
               "arms": {}}
    for arm in args.seeds:
        ranks_by_cond, tables = {}, {}
        for cnd in CONDITIONS:
            S_p = np.stack([vec_of(t) for t in cond[(arm, cnd)]])
            if not (np.abs(np.linalg.norm(S_p, axis=1) - 1.0) < 1e-3).all():
                fail(f"unit-norm violated in arm {arm}/{cnd}")
            ident_p = identity_groups(cond[(arm, cnd)])
            r, nt, ft = avg_ranks(S_p, C, gt_idx, ident=ident_p)
            ranks_by_cond[cnd] = r
            print(f"[arm {arm}][{cnd}] identity groups: {len(ident_p)}; "
                  f"tied queries: {int((nt > 0).sum())} (float-level: "
                  f"{int(ft.sum())})  [registered: 0 identity groups]")
            # G8 (v2): constant's POOLED VECTOR bitwise-identical (same
            # cache key). Its rank is NOT gated — the surrounding pool is
            # replaced, so rank movement is legitimate; report it.
            for i in const_idx:
                if not np.array_equal(S_p[i], S_u[i]):
                    fail(f"G8-vec: constant {sig_ids[i]} pooled vector "
                         f"differs under {cnd} (arm {arm}) despite "
                         f"byte-identical serialisation")
                qsel = np.where(gt_idx == i)[0]
                for qi in qsel:
                    print(f"  [arm {arm}][{cnd}] G8 report: constant-GT "
                          f"query {qi} rank {ranks_u[qi]} -> {r[qi]} "
                          f"(pool-side movement; vector bitwise-identical)")
        # G4 buffered
        allsel = np.ones(len(pairs), bool)
        m_res = group_metrics(ranks_by_cond["resample"], allsel)["mrr"]
        m_gau = group_metrics(ranks_by_cond["gaussian"], allsel)["mrr"]
        m_u = group_metrics(ranks_u, allsel)["mrr"]
        print(f"\narm {arm} G4: overall MRR unpert {m_u:.4f} | resample "
              f"{m_res:.4f} | gaussian {m_gau:.4f} | chance ref 0.0170")
        if m_gau > m_res + G4_BUFFER:
            fail(f"G4: gaussian {m_gau:.4f} exceeds resample {m_res:.4f} "
                 f"+ {G4_BUFFER} (arm {arm})")
        print("  G4 PASSED")
        # tables + GF
        print(f"  per-group MRR (unpert | " + " | ".join(CONDITIONS)
              + ")  [n]")
        for sub in ("sushi", "truce"):
            for grp in sorted(set(q_group[substrate == sub])):
                sel = (substrate == sub) & (q_group == grp)
                row = {"unperturbed": group_metrics(ranks_u, sel)}
                for cnd in CONDITIONS:
                    row[cnd] = group_metrics(ranks_by_cond[cnd], sel)
                tables[f"{sub}/{grp}"] = row
                print(f"  {sub}/{grp:10s} {row['unperturbed']['mrr']:.4f} | "
                      + " | ".join(f"{row[c]['mrr']:.4f}"
                                   for c in CONDITIONS)
                      + f"   [{row['unperturbed']['n']}]")
        for sub, grp in GF_CELLS:
            sel = (substrate == sub) & (q_group == grp)
            mu = group_metrics(ranks_u, sel)["mrr"]
            for cnd in CONDITIONS:
                mp = group_metrics(ranks_by_cond[cnd], sel)["mrr"]
                if mp - mu > GF_MARGIN:
                    fail(f"GF: floor {sub}/{grp} MRR rose {mu:.4f} -> "
                         f"{mp:.4f} under {cnd} (arm {arm})")
        print(f"  GF PASSED (no gated cell rose by > {GF_MARGIN} MRR)")

        rec_path = outdir / f"probe3_openai_per_query_seed{arm}.jsonl"
        with open(rec_path, "w", encoding="utf-8") as f:
            for qi, p in enumerate(pairs):
                rec = {"caption_id": p.caption_id, "dataset": p.dataset,
                       "substrate": substrate[qi], "group": q_group[qi],
                       "gt": p.sample_id,
                       "rank_unperturbed": float(ranks_u[qi]),
                       "ntied_unperturbed": int(ntied_u[qi]),
                       **{f"rank_{c}": float(ranks_by_cond[c][qi])
                          for c in CONDITIONS}}
                if p.dataset == "sushi":
                    rec["class_label"] = p.class_label
                    rec["fluct"] = flucts[qi]
                f.write(json.dumps(rec) + "\n")
        meta_path = outdir / f"probe3_openai_signal_meta_seed{arm}.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({c: meta_all[(arm, c)] for c in CONDITIONS}, f,
                      indent=2)
        print(f"  wrote {rec_path.name} and {meta_path.name}")
        summary["arms"][str(arm)] = {
            "overall_mrr": {"unperturbed": m_u, "resample": m_res,
                            "gaussian": m_gau},
            "tables": tables}

    with open(outdir / "probe3_openai_summary.json", "w",
              encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {outdir / 'probe3_openai_summary.json'}")
    print("ALL GATES PASSED — per-query records ready for the stats step")
    print("Reading aid: a floor cannot 'degrade'. These tables should show "
          "small jiggle around tiny baselines; the P3-7 VOID verdict is "
          "scored by the stats script (P2-4 pattern, TOST ±0.05 abs), "
          "not read from these tables.")


if __name__ == "__main__":
    main()

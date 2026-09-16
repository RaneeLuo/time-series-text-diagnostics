#!/usr/bin/env python3
"""
run_probe3.py — CLaSP Probe-3 runner (summary-statistics sufficiency ladder).

Design record: Probe-3 Q1-Q5 accepted 2026-08-15 (this chat's arc; doc
blocks to be committed with the arm). Conditions (the ladder's two NEW
rungs; rung 2 = sf_all is READ from the committed Probe-2 records, never
rerun):
  resample  i.i.d. draw WITH replacement from the series' own raw
            values (same length). Preserves the value distribution,
            destroys the exact multiset and all order.
  gaussian  i.i.d. Normal(raw mean, raw std), same length. The chance
            anchor: CLaSP z-normalises at input, so nothing
            signal-specific survives by construction (P3-1).

Construction is RAW-level (Q1/Q2): surrogate built on the raw series,
THEN znorm, then encode. Reason: resampling does NOT commute with
z-normalisation (the resampled bag has its own mean/std), unlike the
Probe-2 shuffles. A self-check prints the non-commutation on a sample.

Seed scheme (M3 pattern, documented reuse): per-signal seed =
int(sha256("{sample_id}|{condition}|{ckpt_seed}")[:12 hex], 16).
Queries (captions) are NEVER touched; caption embeddings computed once
per checkpoint. Whole pool replaced per condition (Q2 call 1).

Gates (fatal policy fixed at Q5):
  G3   query-group counts vs certified (878; SUSHI 135/4/1;
       TRUCE 715/18/5) — HARD STOP
  G6   frozen-baseline reproduction, D1 split intact: G6a SUSHI
       digit-exact <=1e-9; G6b TRUCE-containing <=6e-3 tie bound —
       HARD STOP (legacy argsort rank, like-for-like vs frozen)
  G7   surrogate validity: seed recorded per signal; resample indices
       in range; all outputs finite — HARD STOP
  G1   applied-check per signal: resample values subset of raw values,
       length preserved (identity survivors FLAGGED no-op, never
       failed); gaussian moments within 6-sigma sampling bands of the
       targets (catches swapped-parameter bugs) — HARD STOP on real
       violations
  G10  degenerate census: raw-constant pool signals must equal the
       registered list exactly (['sushi:clean\\00\\0000009']); TRUCE
       degenerates must be 0; any non-finite value anywhere —
       HARD STOP
  G4   direction sanity (buffered): overall gaussian MRR must not
       exceed overall resample MRR by more than 0.02 (a condition-swap
       bug would violate this by far more; the buffer absorbs
       chance-level noise) — HARD STOP
  G8   identity control: the constant signal's embedding unchanged
       under BOTH conditions (<=1e-6) — its resample is itself and its
       zero-spread gaussian is itself — HARD STOP
  G9   pairing: query tuple list asserted identical to load_pairs
       order before any rank is compared — HARD STOP
  G12  draw-quality report (REPORT ONLY): per substrate, mean fraction
       of raw positions never drawn and mean fraction of unique raw
       values absent from the resample. Expected ~(1-1/L)^L ~ 35-37%.
       This is the P3-2c evidence base, a property of sampling, not a
       defect.
  G2   natural-zeros rate on the znormed surrogates, per substrate
       (REPORT ONLY).

D2 average rank is the measurement rank throughout, as in Probe 2.
Per-query records carry the SUSHI class label and fluctuation type so
the stats script can score P3-5 (pin: 3 spike classes vs {clean,
smooth}; noisy/step descriptive).

Usage (all path flags REQUIRED — no hardcoded checkpoint paths):
    python -m models.clasp.run_probe3 \
        --checkpoints CK42 CK43 CK44 --seeds 42 43 44 \
        --sushi-groups results/analysis/probe2_sushi_groups.json \
        --truce-groups results/analysis/probe2_truce_groups_certified.json

Writes:
  results/experiments/probe3_clasp_per_query_seed{S}.jsonl
  results/experiments/probe3_clasp_signal_meta_seed{S}.json
  results/experiments/probe3_clasp_summary.json
Formal statistics (P3-1/2/4/5 scoring, CIs, TOST, ratio-to-chance)
come from the separate stats script run on these records.

V2 (2026-08-15), after the first run's G8 stop and diagnostics v1/v2:
  MECHANISM: dataset.znorm's constant guard (sd<1e-8) is float64-
  calibrated; on the native float32 pipeline the constant series' std is
  2.384e-7, the guard misses, and znorm returns a constant +-1 row — the
  representation inside EVERY committed baseline (docstring inaccurate;
  znorm itself must NOT be changed — frozen baselines depend on it).
  V1's float64 cast of raw_of flipped that branch, feeding zeros where
  the baseline feeds +-1: a real delivered-script error, caught by G8.
  FIXES: (i) raw_of kept in NATIVE float32; (ii) gaussian degenerate
  guard is ptp==0 (exact), not sd<1e-8. Expectation changes: SUSHI
  natural zeros ~0 (the constant no longer contributes 2048 zeros);
  everything else as registered.
"""

from __future__ import annotations
import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from dataset import load_pairs, znorm
from models.clasp.evaluate import encode_pool, strict_metrics
from models.clasp.model import ClaspModel, ClaspConfig

CONDITIONS = ["resample", "gaussian"]
MAX_TOKENS = 16384  # same length-grouped batching budget as evaluate.py
REGISTERED_DEGENERATES = ["sushi:clean\\00\\0000009"]  # G10 registered list
G4_BUFFER = 0.02


def fail(msg):
    print(f"\nGATE FAILED: {msg}", file=sys.stderr)
    sys.exit(1)


def signal_seed(sample_id: str, cond: str, ckpt_seed: int) -> int:
    h = hashlib.sha256(f"{sample_id}|{cond}|{ckpt_seed}".encode()).hexdigest()
    return int(h[:12], 16)


def build_surrogate(raw: np.ndarray, cond: str, rng: np.random.Generator):
    """Build a raw-level surrogate. Returns (surrogate, meta)."""
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
            # v2 FIX: constancy by ptp==0 (exact), NOT sd<1e-8 — a float32
            # constant has sd ~2.4e-7, so the old guard would draw
            # near-constant jitter and G8 would fire. The matched gaussian
            # of a truly constant series IS the constant.
            meta["degenerate_passthrough"] = True
            return raw.copy(), meta
        return rng.normal(mu, sd, size=L).astype(raw.dtype), meta
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
                x[k] = torch.from_numpy(np.ascontiguousarray(
                    series_list[j], dtype=np.float32))
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

    cs = Counter(zip(q_substrate.tolist(), q_group.tolist()))
    print("query groups:", dict(sorted(cs.items())))
    exp = {("sushi", "dependent"): 135, ("sushi", "invariant"): 4,
           ("sushi", "degenerate"): 1, ("truce", "dependent"): 715,
           ("truce", "invariant"): 18, ("truce", "ambiguous"): 5}
    if dict(cs) != exp or len(pairs) != 878:
        fail(f"G3: query-group counts differ from certified: {dict(cs)}")
    print(f"G3 PASSED: 878 queries; SUSHI 135/4/1, TRUCE 715/18/5")

    # ---- raw series per pool signal ----
    raw_of = {}
    for p in pairs:
        raw_of.setdefault(p.sample_id, np.asarray(p.series))  # NATIVE float32 —
        # v2 FIX: the float64 cast flipped znorm's constant guard
        # (float32 std of the constant is 2.384e-7 > 1e-8; float64 ~1e-13
        # < 1e-8), feeding zeros where every committed baseline feeds the
        # +-1 row. Caught by G8 + diagnostics v1/v2, 2026-08-15.

    # ---- G10: degenerate census on the RAW pool ----
    degen = sorted(s for s, r in raw_of.items() if float(np.ptp(r)) == 0.0)
    print(f"G10: raw-constant pool signals: {degen}")
    if degen != sorted(REGISTERED_DEGENERATES):
        fail(f"G10: degenerate list {degen} != registered "
             f"{REGISTERED_DEGENERATES}")
    n_truce_degen = sum(1 for s in degen if not s.startswith("sushi"))
    if n_truce_degen != 0:
        fail(f"G10: {n_truce_degen} TRUCE degenerate series (certified 0)")
    print("G10 PASSED: degenerate census matches the registered list; "
          "TRUCE degenerates 0")

    # ---- self-check: resample does NOT commute with znorm (stated fact) ----
    demo = next(p for p in pairs if p.dataset == "sushi"
                and p.sample_id not in degen)
    rr = np.asarray(demo.series, dtype=np.float64)
    ridx = np.random.default_rng(0).integers(0, len(rr), size=len(rr))
    d_commute = float(np.max(np.abs(znorm(rr[ridx]) - znorm(rr)[ridx])))
    print(f"self-check non-commutation: |znorm(resample(raw)) - "
          f"znorm(raw)[idx]| max = {d_commute:.3e} on {demo.sample_id} "
          f"(EXPECTED nonzero — this is why construction is raw-level; "
          f"resample mean shift {rr[ridx].mean()-rr.mean():+.4g}, "
          f"std ratio {rr[ridx].std()/rr.std():.4f})")

    summary = {"conditions": CONDITIONS, "g4_buffer": G4_BUFFER,
               "seed_scheme": "int(sha256('{sample_id}|{cond}|{ckpt_seed}')"
                              "[:12hex],16)",
               "registered_degenerates": REGISTERED_DEGENERATES,
               "seeds": {}}

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

        # G6: reproduce the frozen per-seed table (verbatim from Probe 2)
        base_file = Path(args.baselines_dir) / f"eval_baseline_seed{ck_seed}.json"
        frozen = json.load(open(base_file))["strict"]
        rep = {}
        strict_all, _legacy = strict_metrics(sig_ids, sig_emb, queries, q_emb)
        rep["all"] = strict_all
        for ds_group, name in [(("truce_stock", "truce_synth"), "truce"),
                               (("sushi",), "sushi")]:
            qsel = [i for i, q in enumerate(queries) if q[2] in ds_group]
            m, _ = strict_metrics(sig_ids, sig_emb,
                                  [queries[i] for i in qsel], q_emb[qsel])
            rep[name] = m
        TIE_IMMUNE = [("sushi", m) for m in
                      ("recall@1", "recall@5", "recall@10", "mrr")]
        TIE_EXPOSED = [(k, m) for k in ("truce", "all")
                       for m in ("recall@1", "recall@5", "recall@10", "mrr")]
        for k, m in TIE_IMMUNE + TIE_EXPOSED:
            d = abs(rep[k][m] - frozen[k][m])
            if d > 1e-9:
                print(f"  dev {k}/{m}: ours {rep[k][m]:.12f}  "
                      f"frozen {frozen[k][m]:.12f}  |d|={d:.2e}")
        dev_a = max(abs(rep[k][m] - frozen[k][m]) for k, m in TIE_IMMUNE)
        dev_b = max(abs(rep[k][m] - frozen[k][m]) for k, m in TIE_EXPOSED)
        print(f"G6a (SUSHI, digit-exact): max |dev| = {dev_a:.2e}")
        print(f"G6b (TRUCE-containing, tie bound 6e-3): max |dev| = {dev_b:.2e}")
        if dev_a > 1e-9:
            fail(f"G6a: SUSHI metrics not reproduced (dev {dev_a:.2e})")
        if dev_b > 6e-3:
            fail(f"G6b: deviation {dev_b:.2e} exceeds the tie bound 6e-3")
        print("G6a/G6b PASSED (legacy-rank comparison vs frozen table)")

        # unperturbed measurement ranks (D2)
        id2idx_q = {s: i for i, s in enumerate(sig_ids)}
        ranks_u, ntied_u = avg_ranks(sig_emb, queries, q_emb, id2idx_q)
        n_tied_q = {"truce": int(sum(1 for qi, p in enumerate(pairs)
                                     if p.dataset != "sushi" and ntied_u[qi] > 0)),
                    "sushi": int(sum(1 for qi, p in enumerate(pairs)
                                     if p.dataset == "sushi" and ntied_u[qi] > 0))}
        print(f"D2: unperturbed queries with exact ties: {n_tied_q} "
              f"(committed Probe-2 values: truce 18, sushi 0)")
        if not (n_tied_q["truce"] == 18 and n_tied_q["sushi"] == 0):
            print("  NOTE: differs from the committed Probe-2 tie counts — "
                  "investigate before trusting", file=sys.stderr)

        id2idx = {s: i for i, s in enumerate(sig_ids)}
        const_idx = [id2idx[s] for s in REGISTERED_DEGENERATES]

        ranks_by_cond = {}
        sig_meta = {}
        for cond in CONDITIONS:
            z_sur, flags, seeds_rec = [], [], {}
            frac_pos, frac_val = {"sushi": [], "truce": []}, \
                                 {"sushi": [], "truce": []}
            for s in sig_ids:
                sd_ = signal_seed(s, cond, ck_seed)
                seeds_rec[s] = sd_
                raw = raw_of[s]
                sur, meta = build_surrogate(raw, cond,
                                            np.random.default_rng(sd_))
                flags.append(applied_check(raw, sur, cond, meta))
                zs = znorm(sur)
                if not np.all(np.isfinite(zs)):
                    fail(f"G10: non-finite znormed surrogate for {s} [{cond}]")
                z_sur.append(zs)
                if cond == "resample":
                    sub = "sushi" if s.startswith("sushi") else "truce"
                    frac_pos[sub].append(meta["frac_positions_never_drawn"])
                    frac_val[sub].append(meta["frac_unique_values_missing"])
            n_noop = flags.count("noop")
            noop_ids = [sig_ids[i] for i, f in enumerate(flags) if f == "noop"]
            print(f"[{cond}] G1/G7/G10 PASSED over {len(sig_ids)} signals; "
                  f"no-ops flagged: {n_noop} ({noop_ids[:4]})")
            if cond == "resample":
                for sub in ("sushi", "truce"):
                    if frac_pos[sub]:
                        print(f"[{cond}] G12 [{sub}]: mean frac positions "
                              f"never drawn {np.mean(frac_pos[sub]):.4f}; "
                              f"mean frac unique values missing "
                              f"{np.mean(frac_val[sub]):.4f} "
                              f"(expected ~0.35-0.37)")

            # G2: natural zeros on the znormed surrogates
            for sub, pred in [("truce", lambda s: not s.startswith("sushi")),
                              ("sushi", lambda s: s.startswith("sushi"))]:
                sel = [i for i, s in enumerate(sig_ids) if pred(s)]
                tot = sum(len(z_sur[i]) for i in sel)
                zz = sum(int((z_sur[i] == 0.0).sum()) for i in sel)
                print(f"[{cond}] G2 natural exact-zeros [{sub}]: {zz}/{tot} "
                      f"({zz/tot:.4%}) on the znormed surrogate")

            emb_c = encode_znormed(model, device, z_sur)

            # G8: constant signal's embedding unchanged under BOTH conditions
            for i in const_idx:
                d = float((emb_c[i] - sig_emb[i]).abs().max())
                if d > 1e-6:
                    fail(f"G8: constant signal {sig_ids[i]} embedding moved "
                         f"by {d:.2e} under {cond}")
            print(f"[{cond}] G8 PASSED (constant embedding diff <= 1e-6)")

            ranks_c, ntied_c = avg_ranks(emb_c, queries, q_emb, id2idx_q)
            print(f"[{cond}] queries with exact ties: "
                  f"{int((ntied_c > 0).sum())}")
            ranks_by_cond[cond] = ranks_c
            sig_meta[cond] = {"signal_seeds": seeds_rec, "noop_ids": noop_ids}

        # G4 (buffered direction sanity): gaussian must not beat resample
        m_res = group_metrics(ranks_by_cond["resample"],
                              np.ones(len(pairs), bool))["mrr"]
        m_gau = group_metrics(ranks_by_cond["gaussian"],
                              np.ones(len(pairs), bool))["mrr"]
        m_all_u = group_metrics(ranks_u, np.ones(len(pairs), bool))["mrr"]
        # chance MRR for pool 386 = mean(1/r) = H(386)/386 ~ 0.0170
        # (the committed reference value from the floor baseline record)
        print(f"G4: overall MRR unpert {m_all_u:.4f} | resample {m_res:.4f} "
              f"| gaussian {m_gau:.4f} | chance ref 0.0170")
        if m_gau > m_res + G4_BUFFER:
            fail(f"G4: gaussian MRR {m_gau:.4f} exceeds resample "
                 f"{m_res:.4f} + {G4_BUFFER} — conditions swapped?")
        print("G4 PASSED")

        # per-group descriptive tables (baselines side by side, Q3)
        tables = {}
        print("\nper-group MRR (unperturbed | resample | gaussian)  [n]")
        for sub in ("sushi", "truce"):
            for grp in sorted(set(q_group[q_substrate == sub])):
                sel = (q_substrate == sub) & (q_group == grp)
                row = {"unperturbed": group_metrics(ranks_u, sel)}
                for cond in CONDITIONS:
                    row[cond] = group_metrics(ranks_by_cond[cond], sel)
                tables[f"{sub}/{grp}"] = row
                print(f"  {sub}/{grp:10s} "
                      f"{row['unperturbed']['mrr']:.4f} | "
                      + " | ".join(f"{row[c]['mrr']:.4f}" for c in CONDITIONS)
                      + f"   [{row['unperturbed']['n']}]")

        # descriptive SUSHI fluctuation-class table (P3-5 evidence; formal
        # scoring in the stats script under the accepted pin)
        print("\nSUSHI per-fluctuation MRR (unperturbed | resample | "
              "gaussian)  [n]")
        fluct_tab = {}
        flucts = {}
        for qi, p in enumerate(pairs):
            if p.dataset == "sushi":
                flucts[qi] = p.class_label.split(";")[0].strip()
        for fl in sorted(set(flucts.values())):
            sel = np.array([flucts.get(qi) == fl for qi in range(len(pairs))])
            row = {"unperturbed": group_metrics(ranks_u, sel)}
            for cond in CONDITIONS:
                row[cond] = group_metrics(ranks_by_cond[cond], sel)
            fluct_tab[fl] = row
            print(f"  {fl:28s} {row['unperturbed']['mrr']:.4f} | "
                  + " | ".join(f"{row[c]['mrr']:.4f}" for c in CONDITIONS)
                  + f"   [{row['unperturbed']['n']}]")

        # per-query records
        rec_path = outdir / f"probe3_clasp_per_query_seed{ck_seed}.jsonl"
        with open(rec_path, "w", encoding="utf-8") as f:
            for qi, p in enumerate(pairs):
                rec = {"caption_id": p.caption_id, "dataset": p.dataset,
                       "substrate": q_substrate[qi], "group": q_group[qi],
                       "gt": p.sample_id,
                       "rank_unperturbed": float(ranks_u[qi]),
                       "ntied_unperturbed": int(ntied_u[qi]),
                       **{f"rank_{c}": float(ranks_by_cond[c][qi])
                          for c in CONDITIONS}}
                if p.dataset == "sushi":
                    rec["class_label"] = p.class_label
                    rec["fluct"] = flucts[qi]
                f.write(json.dumps(rec) + "\n")
        meta_path = outdir / f"probe3_clasp_signal_meta_seed{ck_seed}.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(sig_meta, f, indent=2)
        print(f"wrote {rec_path} and {meta_path.name}")
        summary["seeds"][str(ck_seed)] = {
            "g6a_dev": dev_a, "g6b_dev": dev_b,
            "tied_queries_unperturbed": n_tied_q,
            "overall_mrr": {"unperturbed": m_all_u, "resample": m_res,
                            "gaussian": m_gau},
            "tables": tables, "sushi_fluct_tables": fluct_tab}

    with open(outdir / "probe3_clasp_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {outdir / 'probe3_clasp_summary.json'}")
    print("ALL GATES PASSED — per-query records ready for the stats script")


if __name__ == "__main__":
    main()

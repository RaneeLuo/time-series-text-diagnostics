#!/usr/bin/env python3
"""
trace_downsample_survival.py — §4.1 gate (i): does the information Probe 1
measures survive the mandatory 2,048 -> 186 downsample for TRACE?

TRACE's released checkpoint accepts at most seq_len_channel = 186 time steps
(read from the checkpoint's stored args, 2026-08-07). SUSHI signals are 2,048
points. Feeding SUSHI to TRACE therefore requires an ~11x downsample, which
can erase exactly what C4 measures: narrow spikes and fine fluctuation
texture. This gate must PASS before any SUSHI caption is embedded and before
any adapter code is written (state document, next steps item 1).

METHOD — identical logic to the committed difficulty control
(scripts/information_availability_control.py): compute the same 16
hand-written features and run the same logistic-regression CV on the same
component pairs, but on three versions of each signal:

  native      : 2,048 points, original feature function VERBATIM. Gated on
                exact reproduction of the committed pair accuracies in
                results/analysis/information_availability.json — if this run
                cannot reproduce the committed numbers, nothing downstream
                is trustworthy.
  interp186   : linear interpolation to 186 points — the method the adapter
                would use (matches the checkpoint's stored
                downsampling_type='interpolate').
  decimate186 : strided decimation to 186 points — the harshest plausible
                method, included as sensitivity. Spikes between strides
                vanish entirely rather than attenuate.

WINDOW SCALING — a documented design choice, not a silent one. Two features
have fixed windows (moving average w=25; level-shift window w=101). On 186
points, 101 is more than half the series, so running them unchanged would
confound "information lost" with "window no longer local". The 186-point
conditions therefore use proportionally scaled windows (25 -> 3, 101 -> 9)
as PRIMARY, with the fixed-window variant reported as sensitivity. The
native condition keeps the original windows verbatim (required for the
exact-reproduction gate).

PRE-REGISTERED DECISION RULE (locked before running; the standing rule that
no verdict rests on a threshold alone still applies — per-pair detail and
the visual plot are part of the evidence, and a borderline result is
discussed, not silently passed):
  PASS  = under interp186 with scaled windows, C4 mean binary accuracy
          >= 0.85 AND no C4 pair below 0.70.
  FAIL  = C4 mean < 0.85 or any pair < 0.70 -> option (a) is dead for C4
          regardless of the retrieval baseline; the substrate decision
          shifts toward option (b). Report, do not work around.
Context rows (C1/C2/C3/C5) are reported but carry no pass/fail role here.

Run from the thesis repo root:
    python models/trace/downsample_survival_gate.py
Requires: numpy, scikit-learn, matplotlib.
Writes: results/analysis/trace_downsample_survival.json
        results/analysis/trace_downsample_survival.png
"""

from __future__ import annotations
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PAIRS = Path("data/processed/pairs.jsonl")
TABLE = Path("results/analysis/component_table.json")
COMMITTED = Path("results/analysis/information_availability.json")
OUT_JSON = Path("results/analysis/trace_downsample_survival.json")
OUT_PNG = Path("results/analysis/trace_downsample_survival.png")

TARGET_LEN = 186          # TRACE seq_len_channel, read from checkpoint args
EXPECT_SIGNALS = 1400     # committed SUSHI population of the control
EXPECT_LEN = 2048

PASS_MEAN = 0.85          # pre-registered, see header
PASS_MIN_PAIR = 0.70


def fail(gate: str, msg: str):
    print(f"\n[GATE FAILED] {gate}: {msg}")
    print("Hard stop. Investigate before proceeding — do not work around.")
    sys.exit(1)


# ---------------------------------------------------------------- features
# Verbatim from scripts/information_availability_control.py, with the two
# window sizes lifted into parameters (defaults = the committed values, so
# calling with defaults IS the committed feature function).

def znorm(x):
    sd = x.std()
    return np.zeros_like(x) if sd < 1e-8 else (x - x.mean()) / sd


def moving_average(x, w):
    k = np.ones(w) / w
    return np.convolve(x, k, mode="same")


def safe_corr(a, b):
    sa, sb = a.std(), b.std()
    if sa < 1e-12 or sb < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def features(raw, w_smooth=25, w_shift=101):
    x = znorm(np.asarray(raw, dtype=np.float64))
    d = np.diff(x)
    sd = d.std() + 1e-9
    resid = x - moving_average(x, w_smooth)
    n = len(x)

    def kurt(v):
        s = v.std() + 1e-9
        return float(((v - v.mean()) ** 4).mean() / s ** 4)

    def skew(v):
        s = v.std() + 1e-9
        return float(((v - v.mean()) ** 3).mean() / s ** 3)

    t = np.linspace(0.0, 1.0, n)
    thirds = (x[:n // 3].mean(), x[n // 3:2 * n // 3].mean(), x[2 * n // 3:].mean())

    return [
        float(np.polyfit(t, x, 1)[0]),
        float(np.polyfit(t, x, 2)[0]),
        float(np.polyfit(t, x, 3)[0]),
        float(thirds[2] - thirds[0]),
        float(thirds[1] - 0.5 * (thirds[0] + thirds[2])),
        skew(resid),
        float(d.std()),
        kurt(d),
        skew(d),
        float(np.abs(d).max() / sd),
        float((np.abs(d) > 4 * sd).mean()),
        safe_corr(x[:-1], x[1:]),
        float(resid.std()),
        kurt(resid),
        float(np.abs(np.diff(moving_average(x, w_shift))).max()),
        float((np.sign(d[:-1]) != np.sign(d[1:])).mean()),
    ]


# ------------------------------------------------------------ downsampling

def interp_to(x, m=TARGET_LEN):
    x = np.asarray(x, dtype=np.float64)
    src = np.linspace(0.0, 1.0, len(x))
    dst = np.linspace(0.0, 1.0, m)
    return np.interp(dst, src, x)


def decimate_to(x, m=TARGET_LEN):
    x = np.asarray(x, dtype=np.float64)
    idx = np.linspace(0, len(x) - 1, m).round().astype(int)
    return x[idx]


# scaled windows for 186 points: round(w * 186/2048), forced odd and >= 3
def scaled(w):
    s = max(3, int(round(w * TARGET_LEN / EXPECT_LEN)))
    return s if s % 2 == 1 else s + 1


CONDITIONS = {
    "native":              dict(transform=None,        w_smooth=25,          w_shift=101),
    "interp186":           dict(transform=interp_to,   w_smooth=scaled(25),  w_shift=scaled(101)),
    "decimate186":         dict(transform=decimate_to, w_smooth=scaled(25),  w_shift=scaled(101)),
    "interp186_fixedwin":  dict(transform=interp_to,   w_smooth=25,          w_shift=101),
}


def main():
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import cross_val_score, StratifiedKFold
    except ImportError:
        raise SystemExit("needs scikit-learn:  python -m pip install scikit-learn")

    print(f"[setup] target length {TARGET_LEN} (TRACE seq_len_channel, from "
          f"checkpoint args); scaled windows: smooth 25->{scaled(25)}, "
          f"shift 101->{scaled(101)}")

    # ---- Gate 1: load and count -----------------------------------------
    if not PAIRS.is_file():
        fail("G1-pairs", f"{PAIRS} missing — run `python dataset.py build` first")
    rows = []
    with open(PAIRS, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["dataset"] != "sushi":
                continue
            fl, sh = [p.strip() for p in r["class_label"].split(";")]
            rows.append({"fluct": fl, "shape": sh, "series": r["series"]})
    print(f"[G1] SUSHI signals: {len(rows)}")
    if len(rows) != EXPECT_SIGNALS:
        fail("G1-count", f"expected {EXPECT_SIGNALS} SUSHI signals (committed "
                         f"control population), got {len(rows)}")

    # ---- Gate 2: lengths -------------------------------------------------
    lens = {len(r["series"]) for r in rows}
    print(f"[G2] distinct signal lengths: {sorted(lens)}")
    if lens != {EXPECT_LEN}:
        fail("G2-length", f"expected all signals length {EXPECT_LEN}")

    # ---- feature matrices per condition ---------------------------------
    y_fluct = np.array([r["fluct"] for r in rows])
    y_shape = np.array([r["shape"] for r in rows])
    X = {}
    for cond, spec in CONDITIONS.items():
        print(f"[features] extracting: {cond} ...")
        mats = []
        for r in rows:
            s = r["series"] if spec["transform"] is None else spec["transform"](r["series"])
            mats.append(features(s, w_smooth=spec["w_smooth"], w_shift=spec["w_shift"]))
        M = np.array(mats)
        n_bad = int((~np.isfinite(M)).any(axis=1).sum())
        if n_bad:
            print(f"[features] {cond}: non-finite features in {n_bad} signals "
                  f"-> set to 0 (same handling as the committed control)")
            M = np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)
        X[cond] = M
    print(f"[features] matrices: " +
          ", ".join(f"{c} {X[c].shape}" for c in CONDITIONS))

    def clf():
        return make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=2000, C=1.0))

    cv = StratifiedKFold(5, shuffle=True, random_state=42)

    # ---- component pairs -------------------------------------------------
    if not TABLE.is_file():
        fail("G-table", f"{TABLE} missing")
    with open(TABLE, encoding="utf-8") as f:
        components = json.load(f)["components"]

    def run_pairs(M):
        per_component, pair_detail = {}, defaultdict(list)
        for comp, spec in components.items():
            if not spec.get("primary", True):
                continue
            slot_y = y_fluct if comp.startswith("C4") else y_shape
            accs = []
            for a, b in spec["pairs"]:
                mask = (slot_y == a) | (slot_y == b)
                if mask.sum() < 20:
                    continue
                acc = cross_val_score(clf(), M[mask], slot_y[mask], cv=cv).mean()
                accs.append(acc)
                pair_detail[comp].append({"a": a, "b": b, "acc": float(acc),
                                          "n": int(mask.sum())})
            if accs:
                per_component[comp] = {"mean_binary_acc": float(np.mean(accs)),
                                       "min": float(np.min(accs)),
                                       "max": float(np.max(accs)),
                                       "n_pairs": len(accs)}
        return per_component, dict(pair_detail)

    results = {}
    for cond in CONDITIONS:
        print(f"[cv] scoring pairs: {cond} ...")
        results[cond] = dict(zip(("per_component", "pair_detail"), run_pairs(X[cond])))

    # ---- Gate 3: native run reproduces the committed control -------------
    if COMMITTED.is_file():
        with open(COMMITTED, encoding="utf-8") as f:
            committed = json.load(f)
        mismatches = []
        committed_detail = committed.get("pair_detail", {})
        for comp, plist in results["native"]["pair_detail"].items():
            cl = {(p["a"], p["b"]): p["acc"] for p in committed_detail.get(comp, [])}
            for p in plist:
                key = (p["a"], p["b"])
                if key in cl and abs(cl[key] - p["acc"]) > 1e-9:
                    mismatches.append((comp, key, cl[key], p["acc"]))
        if mismatches:
            for m in mismatches[:10]:
                print(f"[G3] mismatch {m[0]} {m[1]}: committed {m[2]:.6f} "
                      f"vs now {m[3]:.6f}")
            fail("G3-reproduction", f"{len(mismatches)} native pair accuracies "
                 "differ from the committed control. Environment or data has "
                 "drifted; resolve before trusting the downsampled runs.")
        print("[G3] native pair accuracies reproduce the committed control "
              "exactly — same data, same environment, same protocol.")
    else:
        print(f"[G3][warn] {COMMITTED} not found — exact-reproduction gate "
              f"SKIPPED. The native numbers below are self-consistent but "
              f"unanchored to the committed record. Investigate why the file "
              f"is missing before treating this run as final.")

    # ---- the table that decides ------------------------------------------
    print("\n" + "=" * 74)
    print("FEATURE SEPARABILITY BY RESOLUTION (mean binary accuracy per component)")
    print("=" * 74)
    comps = sorted(results["native"]["per_component"])
    hdr = f"{'component':<26}" + "".join(f"{c:>13}" for c in ("native", "interp186", "decim186", "fixedwin"))
    print(hdr); print("-" * len(hdr))
    order = ("native", "interp186", "decimate186", "interp186_fixedwin")
    for comp in comps:
        cells = []
        for cond in order:
            v = results[cond]["per_component"].get(comp, {}).get("mean_binary_acc")
            cells.append(f"{v:>13.3f}" if v is not None else f"{'n/a':>13}")
        print(f"{comp:<26}" + "".join(cells))

    # C4 per-pair, the rows the decision actually reads
    print("\nC4 per-pair detail (native -> interp186, scaled windows):")
    c4_comp = next((c for c in comps if c.startswith("C4")), None)
    if c4_comp is None:
        fail("G-c4", "no C4 component found in the component table")
    nat = {(p["a"], p["b"]): p["acc"] for p in results["native"]["pair_detail"][c4_comp]}
    for p in results["interp186"]["pair_detail"][c4_comp]:
        key = (p["a"], p["b"])
        print(f"    {p['a']:<22} vs {p['b']:<22} "
              f"{nat.get(key, float('nan')):.3f} -> {p['acc']:.3f}  "
              f"(delta {p['acc'] - nat.get(key, float('nan')):+.3f}, n={p['n']})")

    # ---- verdict ---------------------------------------------------------
    c4 = results["interp186"]["per_component"][c4_comp]
    verdict = "PASS" if (c4["mean_binary_acc"] >= PASS_MEAN
                         and c4["min"] >= PASS_MIN_PAIR) else "FAIL"
    print("\n" + "=" * 74)
    print(f"PRE-REGISTERED VERDICT (interp186, scaled windows): {verdict}")
    print(f"  C4 mean {c4['mean_binary_acc']:.3f} (rule: >= {PASS_MEAN}) ; "
          f"min pair {c4['min']:.3f} (rule: >= {PASS_MIN_PAIR})")
    if verdict == "PASS":
        print("  Information survives interpolation to 186 points. Gate (i) "
              "clears; gate (ii) — the unperturbed retrieval baseline on the "
              "279 probe signals — is next. The visual plot is still part of "
              "the evidence: look at it before proceeding.")
    else:
        print("  C4-distinguishing information does NOT survive the downsample. "
              "Option (a) is dead for C4 regardless of retrieval performance; "
              "the substrate decision shifts toward option (b). Report this — "
              "do not shop for a friendlier downsampling method after the fact.")
    print("=" * 74)

    # ---- plot: one exemplar per fluctuation class ------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        classes = sorted(set(y_fluct))
        fig, axes = plt.subplots(len(classes), 2, figsize=(12, 2.2 * len(classes)))
        for i, cls in enumerate(classes):
            idx = int(np.where(y_fluct == cls)[0][0])   # first exemplar, deterministic
            raw = np.asarray(rows[idx]["series"], dtype=np.float64)
            axes[i, 0].plot(raw, lw=0.6)
            axes[i, 0].set_ylabel(cls, rotation=0, ha="right", fontsize=8)
            axes[i, 1].plot(interp_to(raw), lw=0.8, label="interp186")
            axes[i, 1].plot(decimate_to(raw), lw=0.5, alpha=0.6, label="decimate186")
            if i == 0:
                axes[i, 0].set_title(f"native ({EXPECT_LEN})")
                axes[i, 1].set_title(f"downsampled ({TARGET_LEN})")
                axes[i, 1].legend(fontsize=7)
            for ax in axes[i]:
                ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle("Downsampling survival — first exemplar per fluctuation class",
                     fontsize=10)
        fig.tight_layout()
        OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUT_PNG, dpi=150)
        print(f"[plot] saved -> {OUT_PNG}  (open it — the visual check is part "
              "of the gate, especially the spike classes)")
    except ImportError:
        print("[plot][warn] matplotlib not installed — numeric verdict stands "
              "but the visual half of the evidence is missing. Install and "
              "re-run before treating the gate as fully passed.")

    # ---- write -----------------------------------------------------------
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({
            "purpose": "§4.1 gate (i): C4 information survival under 2048->186 downsampling",
            "target_len": TARGET_LEN,
            "windows": {"native": [25, 101],
                        "scaled": [scaled(25), scaled(101)]},
            "decision_rule": {"pass_mean": PASS_MEAN, "pass_min_pair": PASS_MIN_PAIR,
                              "condition": "interp186 (scaled windows)"},
            "verdict": verdict,
            "n_signals": len(rows),
            "results": {c: results[c] for c in CONDITIONS},
        }, f, indent=2)
    print(f"[done] saved -> {OUT_JSON}. Paste the full console output back and "
          f"attach the PNG.")


if __name__ == "__main__":
    main()

"""
Shared perturbation and surrogate constructions for the ChatTS arm.
Pure functions + applied-check helpers. No I/O here; the self-test and
the GPU runner both import from this file so there is exactly ONE
implementation of every transformation.

Probe-2 (committed mechanics, adapted where flagged):
  sf_all, sf_half, ex_half           — Tan et al., point-level, raw series
  masking (M1-C)                     — fill = survivors' mean => exact 0 at
                                       ChatTS's model-level input (mean is
                                       subtracted by sp encoding)
  sf_within_patch, sf_across_patch   — two-level extra, SUSHI-only, patch=16

Probe-3 (committed constructions):
  resample                           — i.i.d. with replacement from the
                                       series' own raw values
  gaussian_matched                   — normal(mean, std ddof=0), matched;
                                       ptp==0 pass-through (counted no-op)
  five_number_text                   — the ChatTS-only text condition

Seeds: M3 pattern — sha256(f"{sample_id}|{condition}|{base_seed}").
All randomness goes through numpy Generator(PCG64(seed)) — platform-stable.

Prefix arithmetic (for M1-C drift reporting) replicates the checkpoint's
sp_encoding exactly as read from source at revision 1e661101:
  mean = np.mean(x); scale = max|x-mean|/3 if any |x-mean|>=3 else 1.0
  prefix text uses :.4f formatting.
"""
import hashlib
import numpy as np

PATCH_SIZE = 16  # paper-era checkpoint, config-verified (GZ1)


# ---------------------------------------------------------------- seeds
def seed_for(sample_id: str, condition: str, base_seed: int) -> int:
    h = hashlib.sha256(f"{sample_id}|{condition}|{base_seed}".encode("utf-8"))
    return int(h.hexdigest()[:8], 16)


def _rng(sample_id, condition, base_seed):
    return np.random.Generator(np.random.PCG64(seed_for(sample_id, condition, base_seed)))


# ------------------------------------------------- sp-prefix arithmetic
def sp_prefix_numbers(x: np.ndarray):
    """The two numbers ChatTS's paper-era prefix carries, exact formulas."""
    x = np.asarray(x, dtype=np.float64)
    mean = float(np.mean(x))
    centered = x - mean
    scale = 1.0
    if np.any(np.abs(centered) >= 3.0):
        scale = float(np.max(np.abs(centered)) / 3.0)
    return mean, scale


def sp_prefix_text(x: np.ndarray) -> str:
    mean, scale = sp_prefix_numbers(x)
    return f"[Value Offset: {-mean:.4f}|Value Scaling: {scale:.4f}]"


# ---------------------------------------------------------- Probe-2 ops
def sf_all(x, sample_id, base_seed):
    x = np.asarray(x, dtype=np.float64)
    perm = _rng(sample_id, "sf_all", base_seed).permutation(len(x))
    return x[perm]


def sf_half(x, sample_id, base_seed):
    """Permute the first floor(L/2) points; second half untouched (parent)."""
    x = np.asarray(x, dtype=np.float64).copy()
    h = len(x) // 2
    perm = _rng(sample_id, "sf_half", base_seed).permutation(h)
    x[:h] = x[:h][perm]
    return x


def ex_half(x, sample_id=None, base_seed=None):
    """Deterministic half swap — no randomness (parent)."""
    x = np.asarray(x, dtype=np.float64)
    h = len(x) // 2
    return np.concatenate([x[h:], x[:h]])


def masking_m1c(x, sample_id, base_seed, ratio=0.2):
    """M1-C: k=int(ratio*L) seeded non-contiguous positions filled with the
    SURVIVORS' mean => modified-series mean == fill exactly, so masked
    positions are exactly 0 after ChatTS subtracts the mean."""
    x = np.asarray(x, dtype=np.float64).copy()
    L = len(x)
    k = int(ratio * L)
    pos = _rng(sample_id, "masking", base_seed).choice(L, size=k, replace=False)
    surv_mask = np.ones(L, dtype=bool)
    surv_mask[pos] = False
    fill = float(np.mean(x[surv_mask]))
    x[pos] = fill
    return x, {"masked_positions": np.sort(pos), "fill": fill, "k": k}


def sf_within_patch(x, sample_id, base_seed, patch=PATCH_SIZE):
    """Permute inside each patch independently; patch order intact. SUSHI-only."""
    x = np.asarray(x, dtype=np.float64).copy()
    L = len(x)
    assert L % patch == 0, f"within-patch needs L divisible by {patch}, got {L}"
    rng = _rng(sample_id, "sf_within_patch", base_seed)
    for b in range(L // patch):
        seg = x[b * patch:(b + 1) * patch]
        x[b * patch:(b + 1) * patch] = seg[rng.permutation(patch)]
    return x


def sf_across_patch(x, sample_id, base_seed, patch=PATCH_SIZE):
    """Permute whole patches; inside of each patch intact. SUSHI-only."""
    x = np.asarray(x, dtype=np.float64)
    L = len(x)
    assert L % patch == 0, f"across-patch needs L divisible by {patch}, got {L}"
    nb = L // patch
    perm = _rng(sample_id, "sf_across_patch", base_seed).permutation(nb)
    blocks = x.reshape(nb, patch)
    return blocks[perm].reshape(L)


# ---------------------------------------------------------- Probe-3 ops
def resample(x, sample_id, base_seed):
    """i.i.d. with replacement from the series' own raw values."""
    x = np.asarray(x, dtype=np.float64)
    idx = _rng(sample_id, "resample", base_seed).integers(0, len(x), size=len(x))
    return x[idx]


def gaussian_matched(x, sample_id, base_seed):
    """normal(mean, std ddof=0), same length; ptp==0 => pass-through no-op."""
    x = np.asarray(x, dtype=np.float64)
    if float(np.ptp(x)) == 0.0:
        return x.copy(), True  # degenerate pass-through (counted, never failed)
    g = _rng(sample_id, "gaussian", base_seed).normal(
        loc=float(np.mean(x)), scale=float(np.std(x)), size=len(x))
    return g, False


def five_number_text(x) -> str:
    """The ChatTS-only Probe-3 text condition. std is ddof=0, stated."""
    x = np.asarray(x, dtype=np.float64)
    return (f"[mean={np.mean(x):.4f}|std={np.std(x):.4f}"
            f"|min={np.min(x):.4f}|max={np.max(x):.4f}|length={len(x)}]")


# ------------------------------------------------------- applied checks
def check_perm_family(orig, pert):
    """Multiset preserved; order changed unless no-op (no-op flagged, not failed)."""
    orig = np.asarray(orig, dtype=np.float64)
    pert = np.asarray(pert, dtype=np.float64)
    multiset_ok = np.array_equal(np.sort(orig), np.sort(pert))
    noop = np.array_equal(orig, pert)
    return {"ok": multiset_ok, "noop": noop}


def check_sf_half(orig, pert):
    r = check_perm_family(orig, pert)
    h = len(orig) // 2
    r["second_half_identical"] = np.array_equal(
        np.asarray(orig, dtype=np.float64)[h:], np.asarray(pert, dtype=np.float64)[h:])
    r["ok"] = r["ok"] and r["second_half_identical"]
    return r


def check_ex_half(orig, pert):
    orig = np.asarray(orig, dtype=np.float64)
    pert = np.asarray(pert, dtype=np.float64)
    h = len(orig) // 2
    exact = (np.array_equal(pert[:len(orig) - h], orig[h:])
             and np.array_equal(pert[len(orig) - h:], orig[:h]))
    return {"ok": exact, "noop": np.array_equal(orig, pert)}


def check_masking(orig, pert, info):
    orig = np.asarray(orig, dtype=np.float64)
    pert = np.asarray(pert, dtype=np.float64)
    pos = info["masked_positions"]
    surv = np.ones(len(orig), dtype=bool)
    surv[pos] = False
    survivors_untouched = np.array_equal(orig[surv], pert[surv])
    # the M1-C identity: fill == modified-series mean (exactly, up to float)
    new_mean = float(np.mean(pert))
    ident = abs(info["fill"] - new_mean) <= 1e-9 * max(1.0, abs(new_mean))
    # prefix drift at printed precision
    drift_offset = (f"{-np.mean(orig):.4f}" != f"{-new_mean:.4f}")
    _, scale_o = sp_prefix_numbers(orig)
    _, scale_p = sp_prefix_numbers(pert)
    drift_scale = (f"{scale_o:.4f}" != f"{scale_p:.4f}")
    noop = np.array_equal(orig, pert)
    return {"ok": survivors_untouched and ident, "noop": noop,
            "m1c_identity": ident, "survivors_untouched": survivors_untouched,
            "prefix_offset_drift": drift_offset, "prefix_scale_drift": drift_scale}


def check_within_patch(orig, pert, patch=PATCH_SIZE):
    orig = np.asarray(orig, dtype=np.float64)
    pert = np.asarray(pert, dtype=np.float64)
    nb = len(orig) // patch
    ob = orig.reshape(nb, patch)
    pb = pert.reshape(nb, patch)
    per_block_multiset = all(np.array_equal(np.sort(ob[i]), np.sort(pb[i]))
                             for i in range(nb))
    return {"ok": per_block_multiset, "noop": np.array_equal(orig, pert)}


def check_across_patch(orig, pert, patch=PATCH_SIZE):
    orig = np.asarray(orig, dtype=np.float64)
    pert = np.asarray(pert, dtype=np.float64)
    nb = len(orig) // patch
    ob = {ob_i.tobytes() for ob_i in orig.reshape(nb, patch)}
    pb = {pb_i.tobytes() for pb_i in pert.reshape(nb, patch)}
    return {"ok": ob == pb, "noop": np.array_equal(orig, pert)}


def check_resample(orig, pert):
    orig_vals = set(np.asarray(orig, dtype=np.float64).tolist())
    pert_arr = np.asarray(pert, dtype=np.float64)
    subset = all(v in orig_vals for v in pert_arr.tolist())
    return {"ok": subset and len(pert_arr) == len(np.asarray(orig)),
            "noop": np.array_equal(np.asarray(orig, dtype=np.float64), pert_arr)}


def check_gaussian(orig, pert, is_noop):
    orig = np.asarray(orig, dtype=np.float64)
    pert = np.asarray(pert, dtype=np.float64)
    if is_noop:
        return {"ok": np.array_equal(orig, pert), "noop": True}
    L = len(orig)
    m, s = float(np.mean(orig)), float(np.std(orig))
    band_mean = 6.0 * s / np.sqrt(L)                     # 6-sigma of the sample mean
    band_std = 6.0 * s / np.sqrt(2.0 * L)                # ~6-sigma of the sample std
    ok = (abs(float(np.mean(pert)) - m) <= band_mean
          and abs(float(np.std(pert)) - s) <= band_std
          and len(pert) == L)
    return {"ok": ok, "noop": False}

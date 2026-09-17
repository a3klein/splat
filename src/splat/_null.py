"""Nulls — what a spatially homogeneous cell's half-vs-half cosine looks like.

The cosine from :mod:`splat._core` is not comparable across cells: a cell with 30
transcripts and a cell with 3,000 have very different expected cosines under the null of
"no spatial structure at all", and different spreads. Both are removed here, giving the
standardized score ``ax_coh`` in units of null standard deviations, negative meaning less
coherent than a homogeneous cell of the same depth.

Two nulls, both permuting **each cell's own** transcript labels — so both are invariant to
the segmentation and depend only on the cell's transcript count and profile:

``fitted`` (default)
    Permute an N-stratified subsample of cells, fit the residual bias of the analytic mean
    and the null SD as smooth functions of N, and interpolate them onto every cell::

        ax_coh_a = (cos_a - (enull - bias(N))) / sd(N)

``permutation``
    Permute every cell, and standardize each cell by its own permutation mean and SD. No
    functional form, no pooling; the reference against which ``fitted`` is calibrated.
    Costs one permutation batch per cell, which is why it takes ``n_jobs``.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from ._core import ALL_AXES, DEFAULT_AXES, CellStore, analytic_mean

__all__ = [
    "perm_cosines", "perm_null", "binned_fit", "fit_null", "apply_null",
    "permutation_null", "flag_rate", "SD_FLOOR",
]

#: Null SDs below this are clipped, so a degenerate cell cannot manufacture a huge score.
SD_FLOOR = 1e-3


# --------------------------------------------------------------- permutation ---
def perm_cosines(local: np.ndarray, n_unique: int, npos: int, perms: int, rng) -> np.ndarray:
    """``perms`` half-vs-half cosines from random label splits of one cell.

    Vectorized over permutations. The identity ``b = total - a`` means only one half has to
    be counted, and the three quantities the cosine needs follow from dot products against
    the cell's total count vector::

        a . b = a.t - a.a        |a|^2 = a.a        |b|^2 = t.t - 2 a.t + a.a
    """
    n = int(local.size)
    if n < 2 or npos < 1 or npos >= n:
        return np.full(perms, np.nan)
    t = np.bincount(local, minlength=n_unique).astype(float)
    tt = float(t @ t)
    # A random subset of size npos per permutation: partition random keys, don't sort them.
    pick = np.argpartition(rng.random((perms, n)), npos - 1, axis=1)[:, :npos]
    flat = local[pick] + (np.arange(perms, dtype=np.int64) * n_unique)[:, None]
    a = np.bincount(flat.ravel(), minlength=perms * n_unique).reshape(perms, n_unique).astype(float)
    # einsum, not `a @ t`: this runs inside forked workers, and keeping BLAS (hence
    # OpenMP) out of the child is what makes forking a threaded parent safe here.
    s = np.einsum("ij,j->i", a, t)
    q = np.einsum("ij,ij->i", a, a)
    denom = np.sqrt(q * (tt - 2 * s + q))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denom > 0, (s - q) / denom, np.nan)


def perm_null(local: np.ndarray, n_unique: int, npos: int, perms: int, rng) -> tuple[float, float]:
    """Mean and SD of one cell's null cosine distribution."""
    cs = perm_cosines(local, n_unique, npos, perms, rng)
    return float(np.nanmean(cs)), float(np.nanstd(cs))


def _n_workers(n_jobs) -> int:
    if n_jobs in (None, 0, 1):
        return 1
    try:
        avail = len(os.sched_getaffinity(0))
    except AttributeError:  # pragma: no cover - not Linux
        avail = os.cpu_count() or 1
    return avail if n_jobs < 0 else min(int(n_jobs), avail)


def _payload(store: CellStore, idx: np.ndarray):
    """Pack just the cells in ``idx`` for shipping to a worker."""
    locals_ = [store.cell(int(i)) for i in idx]
    flat = np.concatenate(locals_) if locals_ else np.empty(0, np.int32)
    return flat, store.lengths[idx], store.n_unique[idx], idx


def _chunk_stats(payload, perms: int, seed: int):
    """Null mean/SD for one chunk of cells. Each cell is seeded from its own index, so the
    result does not depend on how the cells were chunked or on ``n_jobs``."""
    flat, lengths, n_unique, idx = payload
    means = np.full(len(idx), np.nan)
    sds = np.full(len(idx), np.nan)
    off = 0
    for k in range(len(idx)):
        n = int(lengths[k])
        local = flat[off : off + n]
        off += n
        rng = np.random.default_rng([int(seed), int(idx[k])])
        means[k], sds[k] = perm_null(local, int(n_unique[k]), n - n // 2, perms, rng)
    return idx, means, sds


def _perm_stats(store: CellStore, idx: np.ndarray, *, perms: int, seed: int, n_jobs) -> tuple:
    """Null mean/SD for ``store`` cells ``idx``, optionally across processes."""
    idx = np.asarray(idx, dtype=np.int64)
    means = np.full(len(idx), np.nan)
    sds = np.full(len(idx), np.nan)
    if len(idx) == 0:
        return means, sds

    workers = _n_workers(n_jobs)
    if workers == 1:
        _, means, sds = _chunk_stats(_payload(store, idx), perms, seed)
        return means, sds

    pos = {int(i): k for k, i in enumerate(idx)}
    chunks = np.array_split(idx, min(len(idx), workers * 4))
    # fork, deliberately: forkserver and spawn re-import __main__ in the child, which
    # makes n_jobs>1 fail from a notebook or an unguarded script. The usual hazard of
    # forking a threaded parent is the child touching an inherited BLAS/OpenMP lock, and
    # the worker is pure numpy ufuncs for exactly that reason (see perm_cosines).
    methods = mp.get_all_start_methods()
    ctx = mp.get_context("fork" if "fork" in methods else methods[0])
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*fork.*")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
            futures = [pool.submit(_chunk_stats, _payload(store, c), perms, seed)
                       for c in chunks if len(c)]
            for fut in futures:
                cidx, cm, cs = fut.result()
                for k, i in enumerate(cidx):
                    j = pos[int(i)]
                    means[j], sds[j] = cm[k], cs[k]
    return means, sds


# ----------------------------------------------------------------- fitted null ---
def binned_fit(n: np.ndarray, y: np.ndarray, nbins: int = 12) -> tuple[np.ndarray, np.ndarray]:
    """Median of ``y`` in ``nbins`` quantile bins of ``n``, as interpolation knots."""
    edges = np.unique(np.quantile(n, np.linspace(0, 1, nbins)))
    if len(edges) < 2:  # degenerate (near-constant N): a single knot
        return np.array([float(np.mean(n))]), np.array([float(np.nanmedian(y))])
    idx = np.clip(np.digitize(n, edges) - 1, 0, len(edges) - 2)
    bn, by = [], []
    for b in range(len(edges) - 1):
        m = idx == b
        if m.any():
            bn.append(n[m].mean())
            by.append(np.nanmedian(y[m]))
    return np.asarray(bn), np.asarray(by)


def fit_null(
    store: CellStore,
    *,
    calib_n: int = 1500,
    perms: int = 200,
    seed: int = 0,
    n_jobs=1,
) -> dict:
    """Fit ``bias(N)`` and ``sd(N)`` by permuting an N-stratified subsample of cells.

    The returned dict is reusable: it depends only on the panel's gene frequencies and the
    depth distribution, so one null can be fitted once and passed to :func:`splat.score`
    for every candidate segmentation of the same tissue.
    """
    if len(store) == 0:
        return {
            "bias_n": np.array([1.0]), "bias": np.array([0.0]),
            "sd_n": np.array([1.0]), "sd": np.array([1.0]),
            "corr": np.nan, "n_calib": 0, "perms": perms, "seed": seed,
        }

    counts = store.lengths.astype(float)
    order = np.argsort(counts)
    pick = order[np.linspace(0, len(order) - 1, min(calib_n, len(order))).astype(int)]

    emean, esd = _perm_stats(store, pick, perms=perms, seed=seed, n_jobs=n_jobs)
    n_pick = counts[pick]
    analytic = np.array([
        analytic_mean(store.p2[i], n, n - n // 2, n // 2)
        for i, n in ((i, int(store.lengths[i])) for i in pick)
    ])

    bias_n, bias = binned_fit(n_pick, analytic - emean)
    sd_n, sd = binned_fit(n_pick, esd)
    ok = np.isfinite(analytic) & np.isfinite(emean)
    corr = float(np.corrcoef(analytic[ok], emean[ok])[0, 1]) if ok.sum() > 1 else np.nan
    return {
        "bias_n": bias_n, "bias": bias, "sd_n": sd_n, "sd": sd,
        "corr": corr, "n_calib": int(len(pick)), "perms": perms, "seed": seed,
    }


def apply_null(cells: pd.DataFrame, null: dict, *, axes=DEFAULT_AXES, z_thresh: float = -3.0) -> pd.DataFrame:
    """Standardize every axis against a fitted null; add ``ax_coh_min`` and ``is_flagged``."""
    cells = cells.copy()
    n = cells["n_transcripts"].to_numpy(dtype=float)
    bias = np.interp(n, null["bias_n"], null["bias"])
    sd = np.clip(np.interp(n, null["sd_n"], null["sd"]), SD_FLOOR, None)
    centre = cells["enull"].to_numpy(dtype=float) - bias
    for axis in ALL_AXES:
        cells[f"ax_coh_{axis}"] = (cells[f"cos_{axis}"].to_numpy(dtype=float) - centre) / sd
    return _finish(cells, axes, z_thresh)


def permutation_null(
    cells: pd.DataFrame,
    store: CellStore,
    *,
    axes=DEFAULT_AXES,
    z_thresh: float = -3.0,
    perms: int = 200,
    seed: int = 0,
    n_jobs=1,
) -> pd.DataFrame:
    """Standardize every axis against each cell's **own** permutation null.

    Self-calibrating and slow — one permutation batch per cell. This is the reference the
    ``fitted`` null is checked against; the two per-cell null means and SDs are returned as
    ``null_mean`` / ``null_sd`` so that calibration can be inspected directly.
    """
    cells = cells.copy()
    idx = np.arange(len(store))
    mean, sd = _perm_stats(store, idx, perms=perms, seed=seed, n_jobs=n_jobs)

    cell_ids = cells["cell_id"].to_numpy()
    if not (len(cell_ids) == len(store) and np.array_equal(cell_ids, store.cell_ids)):
        # score_cells builds both in one pass, so this is belt-and-braces for a caller
        # that filtered `cells` after scoring.
        mean = pd.Series(mean, index=store.cell_ids).reindex(cell_ids).to_numpy(dtype=float)
        sd = pd.Series(sd, index=store.cell_ids).reindex(cell_ids).to_numpy(dtype=float)
    sd = np.clip(sd, SD_FLOOR, None)

    cells["null_mean"] = mean
    cells["null_sd"] = sd
    for axis in ALL_AXES:
        cells[f"ax_coh_{axis}"] = (cells[f"cos_{axis}"].to_numpy(dtype=float) - mean) / sd
    return _finish(cells, axes, z_thresh)


def _finish(cells: pd.DataFrame, axes, z_thresh: float) -> pd.DataFrame:
    axes = tuple(axes)
    unknown = set(axes) - set(ALL_AXES)
    if unknown:
        raise ValueError(f"unknown axes {sorted(unknown)}; choose from {list(ALL_AXES)}.")
    cols = [f"ax_coh_{a}" for a in axes]
    cells["ax_coh_min"] = cells[cols].min(axis=1) if cols else np.nan
    cells["is_flagged"] = cells["ax_coh_min"] < z_thresh
    return cells


def flag_rate(cells: pd.DataFrame, column: str = "is_flagged") -> float:
    """Impurity rate: the flagged fraction of scored cells.

    Read this as impurity — merged cells, spillover from neighbours and genuine subcellular
    structure all lower axial coherence. It is an upper bound on the two-nucleus doublet
    rate, not an estimate of it.
    """
    if column not in cells or len(cells) == 0:
        return float("nan")
    return float(cells[column].to_numpy(dtype=float).mean())

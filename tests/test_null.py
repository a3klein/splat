"""The nulls: the vectorized permutation, its calibration, and reproducibility."""

from __future__ import annotations

import numpy as np
import pytest

import splat
from splat import _core, _null

from _synth import doublets, homogeneous


def _one_cell(seed=0, n=300, n_genes=40):
    rng = np.random.default_rng(seed)
    p = rng.dirichlet(np.full(n_genes, 0.4))
    local = rng.choice(n_genes, size=n, p=p)
    return np.unique(local, return_inverse=True)[1].astype(np.int32)


def test_vectorized_cosine_matches_a_direct_cosine_on_a_fixed_split():
    """The b = total - a identity the fast path rests on, checked exactly."""
    local = _one_cell(seed=2)
    u = int(local.max()) + 1
    rng = np.random.default_rng(0)
    total = np.bincount(local, minlength=u).astype(float)
    for _ in range(20):
        pick = rng.permutation(local.size)[: local.size // 2]
        a = np.bincount(local[pick], minlength=u).astype(float)
        s, q, tt = float(a @ total), float(a @ a), float(total @ total)
        fast = (s - q) / np.sqrt(q * (tt - 2 * s + q))
        assert fast == pytest.approx(_core.cosine(a, total - a), rel=1e-12)


def test_permutation_mean_matches_a_naive_loop_within_monte_carlo_error():
    local = _one_cell(seed=3)
    u = int(local.max()) + 1
    n = local.size
    npos = n - n // 2

    perms = 4000
    fast = _null.perm_cosines(local, u, npos, perms, np.random.default_rng(11))

    rng = np.random.RandomState(7)  # the loop stqc used
    total = np.bincount(local, minlength=u).astype(float)
    naive = np.empty(perms)
    for k in range(perms):
        p = rng.permutation(n)
        a = np.bincount(local[p[:npos]], minlength=u).astype(float)
        naive[k] = _core.cosine(a, total - a)

    tol = 4 * naive.std() / np.sqrt(perms)
    assert abs(fast.mean() - naive.mean()) < tol
    assert fast.std() == pytest.approx(naive.std(), rel=0.05)


def test_analytic_mean_tracks_the_permutation_mean():
    """The analytic form is the null's mean up to a finite-N bias; it should be close and
    strongly correlated, which is what makes fitting the residual worthwhile."""
    tx = homogeneous(n_cells=30, depth=(60, 600), seed=13)
    cells, store = _core.score_cells(tx)
    idx = np.arange(len(store))
    emean, _ = _null._perm_stats(store, idx, perms=400, seed=0, n_jobs=1)
    analytic = cells["enull"].to_numpy()
    assert np.corrcoef(analytic, emean)[0, 1] > 0.99
    assert np.abs(analytic - emean).max() < 0.05


def test_results_do_not_depend_on_n_jobs():
    tx = homogeneous(n_cells=16, seed=17)
    serial = splat.score(tx, null="permutation", perms=100, seed=0, n_jobs=1)
    parallel = splat.score(tx, null="permutation", perms=100, seed=0, n_jobs=3)
    for a in splat.ALL_AXES:
        np.testing.assert_allclose(serial[f"ax_coh_{a}"], parallel[f"ax_coh_{a}"])
    np.testing.assert_allclose(serial["null_mean"], parallel["null_mean"])


def test_same_seed_reproduces_and_a_different_seed_perturbs_only_slightly():
    tx = homogeneous(n_cells=12, seed=19)
    a = splat.score(tx, perms=150, seed=0)
    b = splat.score(tx, perms=150, seed=0)
    c = splat.score(tx, perms=150, seed=1)
    np.testing.assert_allclose(a["ax_coh_min"], b["ax_coh_min"])
    assert np.abs(a["ax_coh_min"] - c["ax_coh_min"]).max() < 0.5


def test_fitted_null_matches_the_per_cell_null_in_the_centre():
    """The fitted null's centre is the analytic mean, which uses the cell's own profile
    concentration, so it should land within a fraction of an SD of permuting that cell."""
    tx = homogeneous(n_cells=60, depth=(50, 800), seed=23)
    cells, store = _core.score_cells(tx)
    null = splat.fit_null(store, perms=500, seed=0)

    n = store.lengths.astype(float)
    centre = cells["enull"].to_numpy() - np.interp(n, null["bias_n"], null["bias"])
    mean_p, sd_p = _null._perm_stats(store, np.arange(len(store)), perms=500, seed=0, n_jobs=1)
    assert np.abs((centre - mean_p) / sd_p).max() < 1.0
    assert np.abs(np.median(centre - mean_p)) < 0.005


def test_fitted_null_smooths_the_spread_over_profile_concentration():
    """Documented limitation, not a bug. The null SD tracks profile concentration (p2/N)
    as much as depth, and the fitted curve is a function of depth alone, so per-cell SDs
    are right on median and wrong in the tails. This is the reason the permutation null is
    kept available."""
    tx = homogeneous(n_cells=120, depth=(50, 800), seed=23)
    cells, store = _core.score_cells(tx)
    null = splat.fit_null(store, perms=500, seed=0)
    n = store.lengths.astype(float)
    sd_f = np.interp(n, null["sd_n"], null["sd"])
    _, sd_p = _null._perm_stats(store, np.arange(len(store)), perms=500, seed=0, n_jobs=1)

    ratio = sd_f / sd_p
    assert np.median(ratio) == pytest.approx(1.0, abs=0.05)  # unbiased on median
    assert np.quantile(ratio, 0.95) > 1.15                   # but spread in the tails
    # and the spread is explained by concentration, which the curve cannot see
    conc = store.p2 / n
    assert np.corrcoef(np.log(sd_p), np.log(conc))[0, 1] < -0.9


def test_fitted_and_permutation_nulls_give_the_same_flag_rate():
    tx = homogeneous(n_cells=120, depth=(50, 800), seed=23)
    fitted = splat.score(tx, null="fitted", perms=400, seed=0)
    perm = splat.score(tx, null="permutation", perms=400, seed=0)
    assert splat.impurity_rate(fitted) == pytest.approx(splat.impurity_rate(perm), abs=0.02)
    diff = (fitted["ax_coh_min"] - perm["ax_coh_min"]).abs()
    assert diff.median() < 0.2
    assert np.quantile(diff, 0.95) < 0.7


def test_permutation_null_also_flags_the_doublets():
    cells = splat.score(doublets(n_cells=10, seed=29), null="permutation", perms=150, seed=0)
    assert cells.attrs["null_mode"] == "permutation"
    assert splat.impurity_rate(cells) > 0.9


def test_binned_fit_survives_near_constant_n():
    """Regression: quantile edges collapse when N is near-constant, which used to leave an
    empty bin and a single degenerate knot."""
    n = np.full(50, 100.0)
    bn, by = splat.binned_fit(n, np.linspace(0, 1, 50))
    assert len(bn) == len(by) == 1
    assert np.isfinite(bn).all() and np.isfinite(by).all()

    n2 = np.concatenate([np.full(40, 100.0), np.full(10, 101.0)])
    bn2, by2 = splat.binned_fit(n2, np.linspace(0, 1, 50))
    assert len(bn2) == len(by2) >= 1
    assert np.isfinite(bn2).all() and np.isfinite(by2).all()


def test_fit_null_on_an_empty_store_is_usable():
    cells, store = _core.score_cells(homogeneous(n_cells=2, seed=31).iloc[:0])
    null = splat.fit_null(store)
    assert null["n_calib"] == 0
    out = splat.apply_null(cells, null)
    assert len(out) == 0


def test_sd_floor_bounds_the_score_of_a_degenerate_cell():
    """A single-gene cell has an identically zero null spread; the floor must keep its
    score finite rather than infinite."""
    import pandas as pd

    n = 40
    tx = pd.DataFrame({
        "gene": ["G000"] * n,
        "x": np.arange(n, dtype=float),
        "y": np.zeros(n),
        "z": np.zeros(n),
        "cell_id": ["c0"] * n,
    })
    cells = splat.score(tx, perms=50, seed=0, null="permutation")
    assert len(cells) == 1
    assert np.isfinite(cells["ax_coh_min"]).all() or cells["ax_coh_min"].isna().all()

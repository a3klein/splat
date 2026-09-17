"""Agreement with the implementation SPLAT was extracted from (stqc's ``metrics/_vsi``).

Skipped unless that module can be found — set ``SPLAT_STQC_VSI`` to its path, or keep an
stqc checkout beside this repo.

What is and is not expected to match:

* The half-vs-half cosines and the analytic null mean are deterministic and must agree
  **exactly** (SPLAT counts over each cell's own genes instead of the whole panel, which
  only removes structural zeros).
* The fitted null is a Monte Carlo estimate, and SPLAT draws its permutations differently
  (vectorized, seeded per cell so the result is independent of ``n_jobs``). The
  standardized scores therefore agree to within permutation error, not bitwise.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

import splat

from _synth import doublets, homogeneous

import pandas as pd

CANDIDATES = [
    os.environ.get("SPLAT_STQC_VSI"),
    Path(__file__).resolve().parents[2] / "stqc" / "src" / "stqc" / "metrics" / "_vsi.py",
]


def _load_vsi():
    for cand in CANDIDATES:
        if cand and Path(cand).exists():
            spec = importlib.util.spec_from_file_location("_stqc_vsi", str(cand))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    return None


vsi = _load_vsi()
pytestmark = pytest.mark.skipif(vsi is None, reason="stqc metrics/_vsi.py not found")

AXES4 = ("x", "y", "z", "pc1")


@pytest.fixture(scope="module")
def table():
    return pd.concat([homogeneous(n_cells=30, depth=(40, 500), seed=61),
                      doublets(n_cells=8, seed=62)], ignore_index=True)


@pytest.fixture(scope="module")
def legacy(table):
    vocab, g = vsi.build_vocab(table["gene"])
    cells, cache = vsi.score_cells(table, vocab, g, 20)
    null = vsi.fit_null(cache, g, calib_n=1500, perms=400, seed=0)
    return vsi.apply_null(cells, null, axes=AXES4).sort_values("cell_id").reset_index(drop=True)


@pytest.fixture(scope="module")
def new(table):
    return splat.score(table, axes=AXES4, perms=400, seed=0, return_cosines=True
                       ).sort_values("cell_id").reset_index(drop=True)


def test_same_cells_scored(legacy, new):
    assert list(new["cell_id"]) == list(legacy["cell_id"])
    np.testing.assert_array_equal(new["n_transcripts"], legacy["N"])


def test_cosines_are_bit_identical(legacy, new):
    for a in splat.ALL_AXES:
        np.testing.assert_allclose(new[f"cos_{a}"], legacy[f"VSI_{a}"], rtol=0, atol=0)


def test_analytic_null_mean_is_bit_identical(legacy, new):
    # stqc stored one Enull column per axis; they are all the same number, because the
    # split sizes do not depend on the axis. SPLAT stores it once.
    for a in splat.ALL_AXES:
        np.testing.assert_allclose(new["enull"], legacy[f"Enull_{a}"], rtol=0, atol=0)


def test_standardized_scores_agree_within_permutation_error(legacy, new):
    """Both standardize by a permutation-fitted SD, so the scores differ by a per-cell
    scale factor of a few percent. On a strongly flagged cell (min_z near -20) that is an
    absolute difference of order 1, which is why the tolerance is relative."""
    a, b = new["ax_coh_min"].to_numpy(), legacy["min_z"].to_numpy()
    rel = np.abs(a - b) / np.maximum(np.abs(b), 1.0)
    assert np.median(rel) < 0.05
    assert rel.max() < 0.25
    near = np.abs(b) < 5  # the region the -3 threshold actually cuts in
    assert np.abs(a[near] - b[near]).max() < 0.5


def test_the_same_cells_are_flagged(legacy, new):
    assert (new["is_flagged"].to_numpy() == (legacy["min_z"] < -3.0).to_numpy()).all()


def test_flag_rate_agrees(legacy, new):
    assert splat.impurity_rate(new) == pytest.approx(vsi.doublet_rate(legacy, -3.0), abs=0.02)

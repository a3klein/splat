"""Input coercion: pandas, polars, dask, pyarrow, and renamed columns."""

from __future__ import annotations

import numpy as np
import pytest

import splat

from _synth import homogeneous

PERMS = 80


@pytest.fixture(scope="module")
def reference():
    tx = homogeneous(n_cells=8, seed=53)
    return tx, splat.score(tx, perms=PERMS, seed=0)


def _same(a, b):
    np.testing.assert_allclose(a["ax_coh_min"], b["ax_coh_min"])
    assert list(a["cell_id"]) == list(b["cell_id"])


def test_renamed_columns(reference):
    tx, want = reference
    renamed = tx.rename(columns={"gene": "target", "x": "global_x", "y": "global_y",
                                 "z": "global_z", "cell_id": "EntityID"})
    got = splat.score(renamed, gene="target", x="global_x", y="global_y", z="global_z",
                      cell_id="EntityID", perms=PERMS, seed=0)
    _same(got, want)


def test_extra_columns_are_ignored(reference):
    tx, want = reference
    got = splat.score(tx.assign(fov=1, junk="x"), perms=PERMS, seed=0)
    _same(got, want)


def test_polars(reference):
    pl = pytest.importorskip("polars")
    tx, want = reference
    _same(splat.score(pl.from_pandas(tx), perms=PERMS, seed=0), want)


def test_pyarrow(reference):
    pa = pytest.importorskip("pyarrow")
    tx, want = reference
    _same(splat.score(pa.Table.from_pandas(tx), perms=PERMS, seed=0), want)


def test_dask(reference):
    dd = pytest.importorskip("dask.dataframe")
    tx, want = reference
    _same(splat.score(dd.from_pandas(tx, npartitions=4), perms=PERMS, seed=0), want)

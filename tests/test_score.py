"""End-to-end behaviour of splat.score on tables with a known answer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import splat

from _synth import doublets, homogeneous, mixed

PERMS = 120  # enough for the null's mean/SD at test speed


def test_homogeneous_cells_are_not_flagged():
    cells = splat.score(homogeneous(seed=3), perms=PERMS, seed=0)
    assert len(cells) == 40
    # Null-centred: the median worst-axis score sits a little below 0 because it is a
    # minimum over six axes, but nothing should reach the -3 flag.
    assert cells["ax_coh_x"].median() == pytest.approx(0.0, abs=0.8)
    assert splat.impurity_rate(cells) < 0.05


def test_synthetic_doublets_are_flagged():
    cells = splat.score(doublets(seed=5), perms=PERMS, seed=0)
    assert splat.impurity_rate(cells) > 0.9
    assert cells["ax_coh_x"].median() < -5


def test_separation_between_homogeneous_and_doublets():
    tx = pd.concat([homogeneous(seed=7), doublets(seed=8)], ignore_index=True)
    cells = splat.score(tx, perms=PERMS, seed=0).set_index("cell_id")
    hom = cells.loc[cells.index.str.startswith("hom"), "ax_coh_min"]
    dbl = cells.loc[cells.index.str.startswith("dbl"), "ax_coh_min"]
    assert dbl.max() < hom.min()


def test_displacement_axis_is_the_flagging_axis():
    cells = splat.score(doublets(seed=9, axis="z"), perms=PERMS, seed=0)
    # The cut along z sees the separation; x and y see two well-mixed halves.
    assert cells["ax_coh_z"].median() < cells["ax_coh_x"].median()
    assert cells["ax_coh_z"].median() < cells["ax_coh_y"].median()


def test_output_contract():
    cells = splat.score(mixed(seed=11), perms=PERMS, seed=0, return_cosines=True)
    expected = (
        ["cell_id", "n_transcripts", "x", "y", "z"]
        + [f"cos_{a}" for a in splat.ALL_AXES]
        + ["enull"]
        + [f"ax_coh_{a}" for a in splat.ALL_AXES]
        + ["ax_coh_min", "is_flagged"]
    )
    assert list(cells.columns) == expected
    assert cells["is_flagged"].dtype == bool
    assert cells["cell_id"].is_unique
    assert (cells["n_transcripts"] >= 20).all()
    assert cells.attrs["null_mode"] == "fitted"
    assert cells.attrs["impurity_rate"] == splat.impurity_rate(cells)


def test_unassigned_and_blank_rows_are_dropped():
    cells = splat.score(mixed(seed=13), perms=PERMS, seed=0)
    assert "-1" not in set(cells["cell_id"])
    # Blank probes were duplicated onto real cells; dropping them must not drop the cells.
    assert len(cells) == 60


def test_min_transcripts_filters_shallow_cells():
    tx = homogeneous(n_cells=12, depth=(25, 60), seed=17)
    loose = splat.score(tx, min_transcripts=20, perms=PERMS)
    strict = splat.score(tx, min_transcripts=50, perms=PERMS)
    assert len(strict) < len(loose)
    assert (strict["n_transcripts"] >= 50).all()


def test_centroids_match_a_direct_groupby():
    tx = homogeneous(seed=19)
    cells = splat.score(tx, perms=PERMS).set_index("cell_id")
    want = tx.groupby("cell_id")[["x", "y", "z"]].mean()
    pd.testing.assert_frame_equal(cells[["x", "y", "z"]].sort_index(), want.sort_index())


def test_axes_argument_changes_only_the_minimum():
    tx = mixed(seed=23)
    all_six = splat.score(tx, perms=PERMS, seed=0)
    cardinal = splat.score(tx, perms=PERMS, seed=0, axes=("x", "y", "z", "pc1"))
    for a in splat.ALL_AXES:  # per-axis scores are untouched
        np.testing.assert_allclose(all_six[f"ax_coh_{a}"], cardinal[f"ax_coh_{a}"])
    # A minimum over more axes can only go down, so the rate can only go up.
    assert (all_six["ax_coh_min"] <= cardinal["ax_coh_min"] + 1e-12).all()
    assert splat.impurity_rate(all_six) >= splat.impurity_rate(cardinal)


def test_panel_restricts_genes():
    tx = homogeneous(n_genes=60, seed=29)
    panel = [f"G{i:03d}" for i in range(20)]
    on_panel = splat.score(tx, panel=panel, perms=PERMS).set_index("cell_id")["n_transcripts"]
    full = splat.score(tx, perms=PERMS).set_index("cell_id")["n_transcripts"]
    shared = on_panel.index.intersection(full.index)
    assert len(shared) > 0
    assert (on_panel[shared] < full[shared]).all()


def test_null_can_be_reused_across_tables():
    tx = homogeneous(seed=31)
    first = splat.score(tx, perms=PERMS, seed=0)
    second = splat.score(tx, null=first.attrs["null"])
    assert second.attrs["null_mode"] == "reused"
    np.testing.assert_allclose(first["ax_coh_min"], second["ax_coh_min"])


def test_z_scale_rescales_and_changes_principal_axes():
    tx = homogeneous(seed=37)
    plain = splat.score(tx, perms=PERMS, seed=0)
    scaled = splat.score(tx, z_scale=10.0, perms=PERMS, seed=0)
    np.testing.assert_allclose(scaled["z"], plain["z"] * 10.0)
    np.testing.assert_allclose(scaled["ax_coh_x"], plain["ax_coh_x"])  # cardinal cuts unchanged
    assert not np.allclose(scaled["ax_coh_pc1"], plain["ax_coh_pc1"])


def test_unknown_axis_raises():
    with pytest.raises(ValueError, match="unknown axes"):
        splat.score(homogeneous(n_cells=4, seed=41), axes=("w",), perms=20)


def test_missing_column_raises_with_a_useful_message():
    tx = homogeneous(n_cells=3, seed=43).drop(columns=["z"])
    with pytest.raises(ValueError, match="missing column"):
        splat.score(tx)


def test_empty_input_returns_an_empty_frame():
    tx = homogeneous(n_cells=2, seed=47).iloc[:0]
    cells = splat.score(tx)
    assert len(cells) == 0
    assert np.isnan(splat.impurity_rate(cells))

"""The one function most callers need."""

from __future__ import annotations

import pandas as pd

from ._core import ALL_AXES, DEFAULT_AXES, score_cells
from ._frames import to_transcript_frame
from ._null import apply_null, fit_null, flag_rate, permutation_null

__all__ = ["score", "impurity_rate"]


def score(
    transcripts,
    *,
    gene: str = "gene",
    x: str = "x",
    y: str = "y",
    z: str = "z",
    cell_id: str = "cell_id",
    z_scale: float = 1.0,
    panel=None,
    drop_blank: bool = True,
    unassigned=("-1",),
    min_transcripts: int = 20,
    axes=DEFAULT_AXES,
    z_thresh: float = -3.0,
    null="fitted",
    calib_n: int = 1500,
    perms: int = 200,
    seed: int = 0,
    n_jobs=1,
    return_cosines: bool = False,
) -> pd.DataFrame:
    """Score every cell's axial coherence from a table of assigned transcripts.

    Parameters
    ----------
    transcripts
        One row per detected transcript, with a gene, an x/y/z location and the id of the
        cell it was assigned to. pandas, polars, dask and pyarrow are all accepted;
        column names are given by the ``gene``/``x``/``y``/``z``/``cell_id`` arguments.
        Assignment is the caller's job — SPLAT never touches masks or polygons.
    z_scale
        Multiplier applied to ``z``, for tables whose z is a slice index rather than µm.
        The principal axes are computed after scaling, so this changes ``pc*``.
    panel
        Restrict to these genes. Transcripts of any other gene are dropped, which also
        makes the score comparable across tables detected with different gene sets.
    drop_blank
        Drop probes whose name starts with "blank" (Vizgen's negative controls).
    unassigned
        ``cell_id`` values meaning "not in a cell". Compared as strings.
    min_transcripts
        Cells with fewer usable transcripts are not scored at all: a cosine between two
        halves of a handful of transcripts carries no information.
    axes
        Which axes' coherence enters ``ax_coh_min``. Default all six; all six are always
        computed and reported regardless.
    z_thresh
        ``is_flagged`` is ``ax_coh_min < z_thresh``. In null SDs, so −3 means "the worst
        axis is 3 SDs less coherent than a homogeneous cell of the same depth".
    null
        ``"fitted"`` (default) fits ``bias(N)``/``sd(N)`` from an N-stratified subsample;
        ``"permutation"`` permutes every cell against itself, which is slower and takes
        ``n_jobs``; or pass a null dict from a previous call's ``df.attrs["null"]`` to
        reuse one null across several candidate segmentations of the same tissue.
    calib_n, perms, seed
        Calibration subsample size, permutations per cell, and the RNG seed. Each cell is
        seeded from its own index, so results do not depend on ``n_jobs``.
    n_jobs
        Processes for the permutation work. ``-1`` uses every core this process may use.
    return_cosines
        Also return the raw, unstandardized ``cos_<axis>`` half-vs-half cosines and the
        analytic null mean ``enull`` they are centred on.

    Returns
    -------
    pandas.DataFrame
        One row per scored cell: ``cell_id``, ``n_transcripts``, the transcript centroid
        ``x``/``y``/``z``, ``ax_coh_<axis>`` for the six axes, ``ax_coh_min`` (the cell's
        worst axis over ``axes``) and ``is_flagged``. Negative ``ax_coh`` means less
        coherent than the null. ``df.attrs`` carries the null and the parameters used.

    Notes
    -----
    The flagged fraction is an **impurity** rate, not a doublet rate. Low axial coherence
    means a cell's transcript composition changes across a spatial cut, which a merged pair
    of cells produces — but so do spillover from a neighbour and real subcellular
    structure. On cerebellum MERFISH the flag rate ran ~6x the rate of cells that a 3D
    resegmentation actually splits in two.
    """
    tx = to_transcript_frame(transcripts, gene=gene, x=x, y=y, z=z, cell_id=cell_id)
    if z_scale != 1.0:
        tx = tx.assign(z=tx["z"].astype(float) * z_scale)

    cells, store = score_cells(
        tx, panel=panel, drop_blank=drop_blank, unassigned=unassigned,
        min_transcripts=min_transcripts,
    )

    if isinstance(null, dict):
        fitted, mode = null, "reused"
    elif null == "permutation":
        fitted, mode = None, "permutation"
    elif null == "fitted":
        fitted = fit_null(store, calib_n=calib_n, perms=perms, seed=seed, n_jobs=n_jobs)
        mode = "fitted"
    else:
        raise ValueError(f"null must be 'fitted', 'permutation' or a null dict, got {null!r}.")

    if mode == "permutation":
        out = permutation_null(cells, store, axes=axes, z_thresh=z_thresh,
                               perms=perms, seed=seed, n_jobs=n_jobs)
    else:
        out = apply_null(cells, fitted, axes=axes, z_thresh=z_thresh)

    keep = ["cell_id", "n_transcripts", "x", "y", "z"]
    if return_cosines:
        keep += [f"cos_{a}" for a in ALL_AXES] + ["enull"]
    keep += [f"ax_coh_{a}" for a in ALL_AXES] + ["ax_coh_min", "is_flagged"]
    if mode == "permutation":
        keep += ["null_mean", "null_sd"]
    out = out[keep].reset_index(drop=True)

    out.attrs["null"] = fitted
    out.attrs["null_mode"] = mode
    out.attrs["params"] = {
        "axes": tuple(axes), "z_thresh": z_thresh, "min_transcripts": min_transcripts,
        "perms": perms, "calib_n": calib_n, "seed": seed, "z_scale": z_scale,
        "drop_blank": drop_blank, "panel": None if panel is None else len(list(panel)),
    }
    out.attrs["n_cells_scored"] = int(len(out))
    out.attrs["impurity_rate"] = flag_rate(out)
    return out


def impurity_rate(cells: pd.DataFrame) -> float:
    """Flagged fraction of the cells in a :func:`score` result.

    An upper bound on the two-nucleus doublet rate — see :func:`score`'s notes.
    """
    return flag_rate(cells)

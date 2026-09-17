"""Per-cell axial coherence — the deterministic half of SPLAT.

For one cell, along one axis: sort its transcripts by position on that axis, cut at the
median into two balanced halves, and take the cosine similarity of the two halves' gene
count vectors. A spatially homogeneous cell gives two halves that look like two random
draws from the same profile; a cell whose composition changes across the cut gives a low
cosine. The cosine is compared against a null in :mod:`splat._null` — nothing here is
standardized yet.

Six axes are cut: the three cardinal axes ``x``, ``y``, ``z`` and the three principal axes
``pc1``, ``pc2``, ``pc3`` of the cell's own transcript cloud.

Two notes on the implementation, both numerically inert:

* Counts are accumulated over the cell's **own** unique genes, not the whole panel. The
  zero entries of a panel-length vector contribute nothing to a dot product or a norm.
  The gene panel therefore enters only as a *filter* on which transcripts count.
* The split is always ``N // 2`` against ``N - N // 2`` transcripts with a stable sort, so
  the analytic null mean depends on the cell but not on the axis, and is stored once.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = [
    "CARDINAL_AXES", "PC_AXES", "ALL_AXES", "DEFAULT_AXES",
    "CellStore", "cosine", "analytic_mean", "pc_projections", "score_cells",
]

CARDINAL_AXES = ("x", "y", "z")
PC_AXES = ("pc1", "pc2", "pc3")
ALL_AXES = CARDINAL_AXES + PC_AXES
#: Axes whose coherence enters ``ax_coh_min`` (the cell's worst axis) by default.
#: All six. stqc's benchmarking default was the narrower ``("x", "y", "z", "pc1")``, on the
#: grounds that the minor principal axes add flags from within-cell gradients; pass
#: ``axes=`` to reproduce that.
DEFAULT_AXES = ALL_AXES


@dataclass
class CellStore:
    """Per-cell transcript identities, flattened, for null calibration.

    ``gene_local[starts[i] : starts[i] + lengths[i]]`` are cell ``i``'s transcripts as
    indices into that cell's own unique-gene list (``0 <= g < n_unique[i]``). That is all
    the null needs: it permutes labels within a cell and never looks across cells.
    """

    cell_ids: np.ndarray
    gene_local: np.ndarray
    starts: np.ndarray
    lengths: np.ndarray
    n_unique: np.ndarray
    p2: np.ndarray

    def __len__(self) -> int:
        return len(self.cell_ids)

    def cell(self, i: int) -> np.ndarray:
        s = self.starts[i]
        return self.gene_local[s : s + self.lengths[i]]


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity, NaN if either vector is all zero."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return np.nan if na == 0 or nb == 0 else float(a @ b / (na * nb))


def analytic_mean(p2: float, n: int, npos: int, nneg: int) -> float:
    """Expected half-vs-half cosine for a spatially homogeneous cell.

    Closed form under the null that the two halves are a random label split of the cell's
    own multiset of transcripts. Depends only on the transcript count ``n``, the split
    sizes, and ``p2`` = sum of squared per-gene counts (the cell's profile concentration).
    Carries a finite-``n`` bias that :func:`splat._null.fit_null` corrects by permutation.
    """
    if n < 2 or npos < 1 or nneg < 1:
        return np.nan
    f = npos / n
    s2 = p2 - n
    edot = f * (1 - f) * s2
    epp = f * (1 - f) * n + f * f * p2
    emm = f * (1 - f) * n + (1 - f) ** 2 * p2
    return np.nan if epp <= 0 or emm <= 0 else float(edot / np.sqrt(epp * emm))


def pc_projections(cx: np.ndarray, cy: np.ndarray, cz: np.ndarray) -> dict:
    """Project a transcript cloud onto its own principal axes, pc1 the longest."""
    p = np.stack([cx, cy, cz], axis=1)
    pc = p - p.mean(0)
    try:
        _, evecs = np.linalg.eigh(pc.T @ pc)
        return {"pc1": pc @ evecs[:, 2], "pc2": pc @ evecs[:, 1], "pc3": pc @ evecs[:, 0]}
    except np.linalg.LinAlgError:  # degenerate cloud: fall back to the cardinal axes
        return {"pc1": cx.copy(), "pc2": cy.copy(), "pc3": cz.copy()}


def _filter(
    tx: pd.DataFrame,
    *,
    panel=None,
    drop_blank: bool = True,
    unassigned=("-1",),
) -> pd.DataFrame:
    t = tx.dropna(subset=["gene", "x", "y", "z", "cell_id"])
    genes = t["gene"].astype(str)
    if panel is not None:
        t = t[genes.isin(set(map(str, panel)))]
        genes = genes.loc[t.index]
    if drop_blank:
        t = t[~genes.str.lower().str.startswith("blank")]
    if unassigned:
        drop = set(map(str, unassigned))
        t = t[~t["cell_id"].astype(str).isin(drop)]
    return t


def score_cells(
    tx: pd.DataFrame,
    *,
    panel=None,
    drop_blank: bool = True,
    unassigned=("-1",),
    min_transcripts: int = 20,
) -> tuple[pd.DataFrame, CellStore]:
    """Per-cell half-vs-half cosine on all six axes.

    Returns ``(cells, store)``. ``cells`` carries ``cell_id``, ``n_transcripts``, the cell
    centroid ``x/y/z``, ``cos_<axis>`` for the six axes and ``enull`` (the analytic null
    mean). ``store`` feeds null calibration. Cells with fewer than ``min_transcripts``
    usable transcripts are dropped — the cosine is meaningless when the halves are tiny.
    """
    t = _filter(tx, panel=panel, drop_blank=drop_blank, unassigned=unassigned)
    if len(t) == 0:
        return _empty_cells(), _empty_store()

    genes = t["gene"].astype(str).to_numpy()
    gene_code = pd.factorize(genes, sort=False)[0]
    try:
        cell_code, cell_vals = pd.factorize(t["cell_id"], sort=True)
    except TypeError:  # mixed-type cell ids: fall back to order of appearance
        cell_code, cell_vals = pd.factorize(t["cell_id"], sort=False)

    order = np.argsort(cell_code, kind="stable")
    cell_code = cell_code[order]
    gene_code = gene_code[order]
    coords = {a: t[a].to_numpy(dtype=float)[order] for a in CARDINAL_AXES}

    bounds = np.flatnonzero(np.diff(cell_code)) + 1
    starts_all = np.concatenate(([0], bounds))
    ends_all = np.concatenate((bounds, [len(cell_code)]))

    rows = []
    s_ids, s_local, s_len, s_u, s_p2 = [], [], [], [], []
    for s, e in zip(starts_all, ends_all):
        n = int(e - s)
        if n < min_transcripts:
            continue
        local = np.unique(gene_code[s:e], return_inverse=True)[1].astype(np.int32)
        u = int(local.max()) + 1
        counts = np.bincount(local, minlength=u).astype(float)
        p2 = float((counts * counts).sum())
        cc = {a: coords[a][s:e] for a in CARDINAL_AXES}
        splits = {**cc, **pc_projections(cc["x"], cc["y"], cc["z"])}

        h = n // 2  # low side; the high side gets n - h
        rec = {
            "cell_id": cell_vals[cell_code[s]],
            "n_transcripts": n,
            "x": float(cc["x"].mean()),
            "y": float(cc["y"].mean()),
            "z": float(cc["z"].mean()),
        }
        for axis, proj in splits.items():
            hi = np.argsort(proj, kind="stable")[h:]
            a = np.bincount(local[hi], minlength=u).astype(float)
            rec[f"cos_{axis}"] = cosine(a, counts - a)
        rec["enull"] = analytic_mean(p2, n, n - h, h)
        rows.append(rec)

        s_ids.append(cell_vals[cell_code[s]])
        s_local.append(local)
        s_len.append(n)
        s_u.append(u)
        s_p2.append(p2)

    if not rows:
        return _empty_cells(), _empty_store()

    lengths = np.asarray(s_len, dtype=np.int64)
    store = CellStore(
        cell_ids=np.asarray(s_ids),
        gene_local=np.concatenate(s_local) if s_local else np.empty(0, np.int32),
        starts=np.concatenate(([0], np.cumsum(lengths)[:-1])).astype(np.int64),
        lengths=lengths,
        n_unique=np.asarray(s_u, dtype=np.int64),
        p2=np.asarray(s_p2, dtype=float),
    )
    return pd.DataFrame(rows), store


def _empty_cells() -> pd.DataFrame:
    cols = ["cell_id", "n_transcripts", "x", "y", "z", *(f"cos_{a}" for a in ALL_AXES), "enull"]
    return pd.DataFrame({c: pd.Series(dtype=float) for c in cols})


def _empty_store() -> CellStore:
    z64, z32, zf = np.empty(0, np.int64), np.empty(0, np.int32), np.empty(0, float)
    return CellStore(np.empty(0), z32, z64, z64, z64, zf)

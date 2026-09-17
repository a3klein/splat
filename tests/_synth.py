"""Synthetic transcript tables with a known answer."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _profile(rng, n_genes, concentration=0.4):
    return rng.dirichlet(np.full(n_genes, concentration))


def homogeneous(n_cells=40, n_genes=60, depth=(40, 400), seed=0, radius=5.0):
    """Cells with no spatial structure: gene identity independent of position."""
    rng = np.random.default_rng(seed)
    genes = np.array([f"G{i:03d}" for i in range(n_genes)])
    rows = []
    for c in range(n_cells):
        n = int(rng.integers(*depth))
        p = _profile(rng, n_genes)
        g = rng.choice(genes, size=n, p=p)
        xyz = rng.normal(scale=radius, size=(n, 3)) + rng.normal(scale=200, size=3)
        rows.append(pd.DataFrame({"gene": g, "x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2],
                                  "cell_id": f"hom{c:04d}"}))
    return pd.concat(rows, ignore_index=True)


def doublets(n_cells=20, n_genes=60, depth=(80, 400), seed=1, axis="x", sep=12.0, radius=3.0):
    """Cells that are two disjoint-profile halves displaced along ``axis``."""
    rng = np.random.default_rng(seed)
    genes = np.array([f"G{i:03d}" for i in range(n_genes)])
    half = n_genes // 2
    ai = np.arange(0, half)
    bi = np.arange(half, n_genes)
    k = {"x": 0, "y": 1, "z": 2}[axis]
    rows = []
    for c in range(n_cells):
        n = int(rng.integers(*depth))
        na = n // 2
        centre = rng.normal(scale=200, size=3)
        parts = []
        for idx, count, shift in ((ai, na, -sep / 2), (bi, n - na, sep / 2)):
            p = _profile(rng, len(idx))
            g = rng.choice(genes[idx], size=count, p=p)
            xyz = rng.normal(scale=radius, size=(count, 3)) + centre
            xyz[:, k] += shift
            parts.append(pd.DataFrame({"gene": g, "x": xyz[:, 0], "y": xyz[:, 1],
                                       "z": xyz[:, 2], "cell_id": f"dbl{c:04d}"}))
        rows.extend(parts)
    return pd.concat(rows, ignore_index=True)


def mixed(seed=0, **kw):
    """Homogeneous cells plus doublets plus some unassigned and blank rows."""
    hom = homogeneous(seed=seed, **kw)
    dbl = doublets(seed=seed + 1)
    junk = hom.head(50).assign(cell_id="-1")
    blank = hom.head(30).assign(gene="Blank-17")
    return pd.concat([hom, dbl, junk, blank], ignore_index=True)

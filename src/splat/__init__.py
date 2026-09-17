"""SPLAT — spatial purification leveraging axial transcriptomics.

Per-cell spatial impurity from a table of assigned transcripts, and nothing else. The
metric is **axial coherence** (``ax_coh``): cut a cell in half along an axis, compare the
two halves' gene profiles, and standardize against what two random halves of a
homogeneous cell of the same transcript count would give.

    import splat
    cells = splat.score(transcripts)           # gene, x, y, z, cell_id
    splat.impurity_rate(cells)                 # flagged fraction

``ax_coh`` is negative when a cell is less coherent than the null. ``ax_coh_min`` is the
cell's worst axis and ``is_flagged`` marks ``ax_coh_min < -3``.

The flagged fraction is an impurity rate, not a doublet rate: merges, spillover from
neighbours and genuine subcellular structure all lower coherence, and only the first is a
doublet. Treat it as an upper bound on the doublet rate, and read it beside cell yield and
cell size — a segmentation that over-fragments cells into same-type pieces scores well
here, because a cut through two pieces of the same cell type is coherent.
"""

from ._core import (
    ALL_AXES,
    CARDINAL_AXES,
    DEFAULT_AXES,
    PC_AXES,
    CellStore,
    analytic_mean,
    cosine,
    score_cells,
)
from ._api import impurity_rate, score
from ._frames import to_transcript_frame
from ._null import apply_null, binned_fit, fit_null, perm_null, permutation_null

__version__ = "0.1.0"

__all__ = [
    "score",
    "impurity_rate",
    # axis sets
    "ALL_AXES",
    "CARDINAL_AXES",
    "PC_AXES",
    "DEFAULT_AXES",
    # the pieces, for diagnostics and shared-null workflows
    "score_cells",
    "fit_null",
    "apply_null",
    "permutation_null",
    "perm_null",
    "binned_fit",
    "analytic_mean",
    "cosine",
    "to_transcript_frame",
    "CellStore",
]

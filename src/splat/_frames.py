"""Input coercion — any dataframe flavour in, one pandas frame of five columns out.

SPLAT's math is per-cell over the whole table, so the transcripts have to be in memory
one way or another; a dask frame is narrowed to the five needed columns *before* it is
computed, which is the only place laziness buys anything here.
"""

from __future__ import annotations

import pandas as pd

__all__ = ["to_transcript_frame"]

REQUIRED = ("gene", "x", "y", "z", "cell_id")


def _select(obj, cols):
    """Narrow to ``cols`` while the object is still in its native flavour."""
    cols = list(cols)
    for attempt in (lambda: obj[cols], lambda: obj.select(cols)):  # frames, then pyarrow
        try:
            return attempt()
        except Exception:
            continue
    return obj


def _columns(obj):
    # pyarrow's `.columns` holds the arrays, not the names, so ask for names first.
    cols = getattr(obj, "column_names", None)
    if cols is None:
        cols = getattr(obj, "columns", None)
    return [] if cols is None else list(cols)


def to_transcript_frame(
    transcripts,
    *,
    gene: str = "gene",
    x: str = "x",
    y: str = "y",
    z: str = "z",
    cell_id: str = "cell_id",
) -> pd.DataFrame:
    """Coerce ``transcripts`` to a pandas frame with columns ``(gene, x, y, z, cell_id)``.

    Accepts a pandas, polars, dask or pyarrow object — anything exposing ``to_pandas``
    or ``compute`` — and renames the five source columns to SPLAT's canonical names.
    """
    src = {gene: "gene", x: "x", y: "y", z: "z", cell_id: "cell_id"}
    have = _columns(transcripts)
    if have:
        missing = [c for c in src if c not in have]
        if missing:
            raise ValueError(
                f"transcript table is missing column(s) {missing}; it has {have}. "
                "Pass the source names via gene=/x=/y=/z=/cell_id=."
            )
        transcripts = _select(transcripts, src)

    if isinstance(transcripts, pd.DataFrame):
        df = transcripts
    elif hasattr(transcripts, "to_pandas"):  # polars, pyarrow
        df = transcripts.to_pandas()
    elif hasattr(transcripts, "compute"):  # dask
        df = transcripts.compute()
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)
    else:
        df = pd.DataFrame(transcripts)

    df = df.rename(columns=src)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"transcript table is missing column(s) {missing} after renaming.")
    return df[list(REQUIRED)]

# SPLAT

**S**patial **p**urification **l**everaging **a**xial **t**ranscriptomics — a per-cell
spatial impurity score for segmented spatial transcriptomics, computed from nothing but a
table of assigned transcripts.

The metric is **axial coherence** (`ax_coh`). Cut a cell in half along an axis, compare the
two halves' gene profiles, and ask how that compares to cutting a spatially homogeneous
cell of the same transcript count. A cell whose composition changes across the cut scores
negative; a cell with no spatial structure scores around zero.

```python
import splat

cells = splat.score(transcripts)      # columns: gene, x, y, z, cell_id
splat.impurity_rate(cells)            # 0.087
```

## What it is for

Comparing segmentations, and flagging individual cells that are probably not one clean
cell. It needs no cell-type labels, no reference, no embedding, no masks and no
normalization — only which transcript went into which cell. That makes it usable as a
segmentation QC metric before any downstream analysis exists, and it makes it insensitive
to the choices in that downstream analysis.

## What it is not

**The flagged fraction is an impurity rate, not a doublet rate.** Low axial coherence says
a cell's transcript composition changes across a spatial cut. A merged pair of cells does
that — but so does spillover from a neighbouring cell, and so does genuine subcellular
structure. On human cerebellum MERFISH the flag rate ran about **6x** the fraction of
flagged cells that a 3D resegmentation actually splits into two, and flagged cells were no
more likely to be genuinely splittable than unflagged ones. Treat the number as an upper
bound on the doublet rate, and treat a *difference* between segmentations as a difference
in impurity.

**It is blind to homotypic merges.** Two merged cells of the same type have the same
profile on both sides of the cut, so they are coherent. A segmentation that over-fragments
cells into same-type pieces therefore scores well. Always read the impurity rate beside
cell yield, median transcript count and median cell volume.

## Install

```bash
pip install git+https://github.com/a3klein/splat
```

numpy and pandas are the only dependencies. polars, dask and pyarrow tables are accepted
as input if you have them; they are not required.

## Input

One row per detected transcript:

| column | meaning |
|---|---|
| `gene` | gene / probe name |
| `x`, `y`, `z` | transcript location, ideally in the same physical unit on all three axes |
| `cell_id` | the cell the transcript was assigned to |

Pass the source column names if they differ, e.g.
`splat.score(tx, gene="target", cell_id="EntityID")`. Assignment is the caller's job:
SPLAT never looks at masks or polygons. If `z` is a slice index rather than a physical
distance, give `z_scale=<µm per slice>` so the principal axes are geometric.

## Output

One row per scored cell.

| column | meaning |
|---|---|
| `cell_id` | as given |
| `n_transcripts` | transcripts used (after filtering) |
| `x`, `y`, `z` | centroid of those transcripts |
| `ax_coh_x` … `ax_coh_pc3` | standardized coherence on each of the six axes |
| `ax_coh_min` | the cell's worst axis — its score |
| `is_flagged` | `ax_coh_min < z_thresh` (default −3) |

`ax_coh` is in null standard deviations: −3 means "3 SDs less coherent than a homogeneous
cell of the same depth". The six axes are the three cardinal axes and the three principal
axes of the cell's own transcript cloud, `pc1` being the longest.

`cells.attrs` carries the fitted null, the parameters used, and the impurity rate.

## How it works

For each cell and each axis: rank its transcripts along the axis, split them into two
balanced halves, and take the cosine similarity of the two halves' gene count vectors.

That raw cosine is not comparable between cells — it depends strongly on how many
transcripts the cell has and on how concentrated its profile is — so it is standardized
against the null that the two halves are a random label split of the cell's own
transcripts. The null's mean has a closed form; its residual finite-N bias and its spread
are obtained by permutation. Both nulls permute each cell against itself, which is what
makes the score independent of the segmentation being scored and of any other cell.

```python
splat.score(tx)                       # fitted null: permute a stratified subsample,
                                      #   interpolate bias(N) and sd(N) onto every cell
splat.score(tx, null="permutation",   # per-cell null: permute every cell against itself
            n_jobs=-1)
```

The fitted null is the default and is ~3x faster than permuting every cell. Its centre is
accurate per-cell (it uses each cell's own profile concentration), but its spread is a
function of transcript count alone, and the true null spread also depends on profile
concentration. On real data the two agree on the flag rate to well under a percentage
point and on 97% of individual flags, with per-cell score differences of median 0.14 SD.
Use `null="permutation"` when a per-cell score matters more than throughput, or to check
the calibration.

A null can also be reused across candidate segmentations of the same tissue, which is the
right thing to do when comparing them:

```python
first = splat.score(tx_a)
second = splat.score(tx_b, null=first.attrs["null"])
```

## Two things to know before trusting a number

**The z axis is sensitive to how tied coordinates are split.** If z takes only a handful
of distinct values — 7 planes in a typical MERFISH stack — the balanced split falls *inside*
a plane, and which of the tied transcripts land on which side is decided by the input row
order. On pooled cerebellum data, reordering the input rows moved individual cells'
`cos_z` by a median of 0.065 and moved the overall impurity rate by ±0.2 percentage points
(8.4–8.9% across five orderings). The cardinal x/y and principal axes are unaffected. Do
not compare z-axis numbers computed from differently ordered inputs, and be careful reading
a z-specific result off a coarsely sampled z.

**A rate depends on the axis set.** `ax_coh_min` is a minimum, so adding axes can only
lower it and raise the flag rate. On the same cerebellum data: 7.5% over `(x, y, z)`, 8.5%
adding `pc1`, 11.1% over all six. The minor principal axes pick up within-cell gradients as
well as merges. Set `axes=` explicitly and report which set you used.

## Provenance

The metric began as a per-cell multi-axis generalization of ovrlpy's vertical signal
integrity, developed on human thalamus MERFISH and then used to benchmark a 192-config
cerebellum segmentation grid. It was extracted here from `stqc`, where it lived as
`metrics/_vsi.py`; "VSI" was retired as a name because five of the six axes are not
vertical, and "doublet rate" was retired because the score measures how mixed a cell is,
not how many nuclei it has.

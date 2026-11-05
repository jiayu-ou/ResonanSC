# Source provenance and adaptation

This package was assembled from the ResonanSC research workspace. Paths below
identify the original study components; they are historical provenance, not
runtime requirements. `SOURCE_SHA256SUMS.txt` records hashes of selected
source scripts used during the extraction.

- `src/resonansc/core` and the configuration/data adapters come from
  `sec3_all_donor_hvg/resonansc_integrated`. The optimizer includes the
  zero-weight-decay revision used in that study run.
- `hierarchical.py`, `all_donor.py`, and `rescue_alignment.py` were adapted
  from `sec3_all_donor_hvg/run_hierarchical.py`, `all_donor_align.py`, and
  `rescue_alignment.py`. Workspace paths became explicit arguments; the
  condition hierarchy and thresholds are captured in the YAML template.
- `post_alignment.py` implements the pairwise mode of
  `sec3_all_donor_hvg/compare_pairwise_union.py`, excluding ground-truth
  plotting and union mode. In the original study, its labels matched saved
  pairwise results for 166,845 cells.
- `pairwise_prepare.py` was adapted from
  `sec3_pairwise_PWM_optimization/prepare.py`, with explicit paths and without
  a dataset-specific cluster count. `alignment_prepare.py` bridges the
  alignment output to this preparation step.
- `embedding/balanced_train.py` was adapted from
  `sec3_WM20_umap/train_balanced_mmd.py`. It retains the MMD200 setup, the
  16-dimensional latent representation, 256 hidden units, all-cell
  pretraining, and balanced sampling while removing study-specific paths and
  fixed cell/LSI sizes. `embedding/lsi.py` corresponds to `prepare_lsi.py`.

The upstream licensing and author permissions have not been confirmed. This
provenance record is not a software license.

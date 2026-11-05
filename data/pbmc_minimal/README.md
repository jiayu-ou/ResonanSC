# PBMC 10k minimal example

These files are a small derived subset of the paired 10x Multiome PBMC 10k
RNA and ATAC data. The prepared full inputs were `expression.h5ad` and
`peak_counts.h5ad` from the local `resonansc_integrated` PBMC directory;
those were built from `10x-Multiome-Pbmc10k-RNA.h5ad` and
`10x-Multiome-Pbmc10k-ATAC.h5ad`. The full matrices and genome annotation are
not included here.

`make_subset.py` selects 400 shared raw barcodes with NumPy seed 2026 after
sorting them. It retains the 500 genes with highest sampled RNA variance and
the 2,000 peaks detected in the most sampled ATAC cells. The small expression
matrix contains 400 RNA and 400 ATAC rows in the same 500-gene space. The
peak matrix contains the 400 ATAC rows and raw integer counts. Only `batch`
and `raw_barcode` cell metadata are included; no cell-type labels are used.

Run `python examples/run_pbmc_minimal.py` from the repository root to use
these files. The script computes RNA PCA and ATAC peak LSI from this subset,
runs batch-mode alignment on CPU, and checks the 800 output cell barcodes.

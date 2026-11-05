"""Recreate the small paired PBMC example from prepared 10x Multiome inputs."""
import argparse
import gzip
from pathlib import Path
import re

import anndata as ad
import numpy as np
from scipy import sparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="directory containing expression.h5ad and peak_counts.h5ad")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--gtf", type=Path, help="matching GRCh38 GTF, optionally gzipped")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    expression = ad.read_h5ad(args.source / "expression.h5ad", backed="r")
    peaks = ad.read_h5ad(args.source / "peak_counts.h5ad", backed="r")
    try:
        # Both modalities use the same raw 10x barcodes. Sort before sampling so
        # this selection is independent of the input file's row order.
        rna = expression.obs[expression.obs.batch.astype(str) == "rna_1"]
        atac = expression.obs[expression.obs.batch.astype(str) == "atac_1"]
        rna_by_raw = dict(zip(rna.raw_barcode.astype(str), rna.index.astype(str)))
        atac_by_raw = dict(zip(atac.raw_barcode.astype(str), atac.index.astype(str)))
        shared = sorted(set(rna_by_raw) & set(atac_by_raw))
        chosen = np.random.default_rng(2026).choice(shared, size=400, replace=False)
        rna_cells = [rna_by_raw[x] for x in chosen]
        atac_cells = [atac_by_raw[x] for x in chosen]
        rna_part = expression[rna_cells].to_memory()
        atac_part = expression[atac_cells].to_memory()
        peak_part = peaks[atac_cells].to_memory()

        # Feature choice is based only on the sampled input matrices; it does
        # not use cell-type annotations or any alignment result.
        x = sparse.csr_matrix(rna_part.X)
        gene_score = np.asarray(x.power(2).mean(axis=0)).ravel() - np.asarray(x.mean(axis=0)).ravel() ** 2
        gene_idx = np.sort(np.argsort(-gene_score, kind="stable")[:500])
        p = sparse.csr_matrix(peak_part.X)
        peak_score = np.asarray((p > 0).sum(axis=0)).ravel()
        peak_idx = np.sort(np.argsort(-peak_score, kind="stable")[:2000])
        combined = ad.concat([rna_part[:, gene_idx], atac_part[:, gene_idx]], join="inner")
        combined.obs = combined.obs[["batch", "raw_barcode"]].copy()
        combined.write_h5ad(args.output / "expression.h5ad", compression="gzip")
        peak_small = peak_part[:, peak_idx].copy()
        peak_small.obs = peak_small.obs[["batch", "raw_barcode"]].copy()
        peak_small.write_h5ad(args.output / "peak_counts.h5ad", compression="gzip")
        if args.gtf:
            wanted = set(combined.var_names.astype(str))
            names = set()
            opener = gzip.open if str(args.gtf).endswith(".gz") else open
            pattern = re.compile(r'gene_name "([^"]+)"')
            with opener(args.gtf, "rt") as source, gzip.open(args.output / "genes.gtf.gz", "wt") as target:
                for line in source:
                    fields = line.split("\t", 3)
                    if len(fields) < 4 or fields[2] != "gene":
                        continue
                    match = pattern.search(line)
                    if match and match.group(1) in wanted:
                        target.write(line)
                        names.add(match.group(1))
            if len(names) < 0.90 * len(wanted):
                raise ValueError(f"GTF matched only {len(names)}/{len(wanted)} genes")
            print(f"Wrote GRCh38 GTF records for {len(names)} genes")
        print(f"Wrote {combined.shape} expression and {peak_small.shape} peak matrices")
    finally:
        expression.file.close()
        peaks.file.close()


if __name__ == "__main__":
    main()

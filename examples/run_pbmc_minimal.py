"""Run ResonanSC alignment on the bundled paired PBMC 10k subset (CPU)."""
import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import yaml
from scipy import sparse
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer

import resonansc as rs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("pbmc_minimal_run"))
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    data_dir = Path(__file__).resolve().parents[1] / "data" / "pbmc_minimal"
    expression = ad.read_h5ad(data_dir / "expression.h5ad")
    peaks = ad.read_h5ad(data_dir / "peak_counts.h5ad")
    rna = expression[expression.obs.batch.astype(str) == "rna_1"]
    atac = expression[expression.obs.batch.astype(str) == "atac_1"]
    rna_x = rna.X.toarray() if sparse.issparse(rna.X) else np.asarray(rna.X)
    rna_pca = PCA(n_components=30, random_state=42).fit_transform(rna_x)
    peak_x = sparse.csr_matrix(peaks.X).astype(np.float32)
    atac_lsi = TruncatedSVD(n_components=30, random_state=42).fit_transform(
        TfidfTransformer(norm="l2").fit_transform(peak_x)
    )
    config = {
        "input": {
            "expression_h5ad": str(data_dir / "expression.h5ad"),
            "peak_h5ad": str(data_dir / "peak_counts.h5ad"),
            "batch_key": "batch",
            "batches": ["rna_1", "atac_1"],
        },
        "output_dir": "alignment",
        "runtime": {"gpu": "", "neighbors_backend": "cpu", "seed": 42},
        "alignment": {
            "n_top_genes": expression.n_vars,
            "n_pcs": 30,
            "n_lsi": 30,
            "n_neighbors": 15,
            "min_peak_cells": 5,
            "leiden_resolution": 0.6,
        },
    }
    config_path = output / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    run_dir = rs.align(
        config_path,
        rna_pca=rna_pca,
        rna_cell_names=rna.obs_names.tolist(),
        atac_lsi=atac_lsi,
        atac_cell_names=atac.obs_names.tolist(),
    )
    labels = __import__("pandas").read_csv(run_dir / "cell_labels.csv")
    assert len(labels) == expression.n_obs and labels.cell_barcode.is_unique
    print(f"PASS: {len(labels)} PBMC cells aligned; results in {run_dir}")


if __name__ == "__main__":
    main()

"""Standalone embedding worker; callable using a separate RAPIDS Python."""
import argparse
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-pcs", type=int, default=0)
    parser.add_argument("--n-neighbors", type=int, default=15)
    args = parser.parse_args()
    try:
        import cupy as cp
        from cuml.decomposition import PCA
        from cuml.neighbors import NearestNeighbors
    except ImportError as exc:
        raise RuntimeError("GPU KNN requires CuPy/cuML in runtime.gpu_python; alternatively explicitly choose neighbors_backend: cpu") from exc
    values = cp.asarray(np.load(args.input), dtype=cp.float32)
    pca = None
    if args.n_pcs:
        pca = PCA(n_components=args.n_pcs, output_type="cupy")
        values = pca.fit_transform(values)
    if args.n_neighbors >= len(values):
        raise ValueError("n_neighbors must be smaller than the number of cells")
    model = NearestNeighbors(n_neighbors=args.n_neighbors, output_type="cupy")
    model.fit(values)
    distances, indices = model.kneighbors(values)
    output = {"pcs": cp.asnumpy(values).astype(np.float32),
              "indices": cp.asnumpy(indices).astype(np.int32),
              "distances": cp.asnumpy(distances).astype(np.float32)}
    if pca is not None:
        output["explained_variance_ratio"] = cp.asnumpy(pca.explained_variance_ratio_)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **output)


if __name__ == "__main__":
    main()

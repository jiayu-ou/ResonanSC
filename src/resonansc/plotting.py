"""Post-run alignment and training diagnostics.

UMAP uses RNA expression, ATAC peak counts, and their combined expression
matrix as three separate reference projections; only the colours change
across stages. Training correlations use the learned
pseudobulks saved in each checkpoint, including the learned ATAC mapping.
"""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
import scanpy as sc
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import StandardScaler, normalize
import torch
from umap import UMAP

from .config import load_config
from .data import rows


def _numpy(value):
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def _obs_labels(source, key, sampled):
    labels = source.obs[key].astype("string").str.strip()
    labels = labels.mask(labels.str.lower().isin(("", "nan", "none", "<na>")), pd.NA)
    return labels.fillna("Unknown").to_numpy()[sampled]


def _reference_embedding(source, peak_path, gene_names, batch_names, eligible_cells,
                         batch_key, max_cells, seed, min_peak_cells,
                         rna_cell_type_key=None, atac_cell_type_key=None,
                         condition_key=None):
    if max_cells < 3:
        raise ValueError("max_cells must be at least 3")
    gene_names = list(map(str, gene_names))
    if source.var_names.get_indexer(gene_names).min() < 0:
        raise ValueError("Input expression is missing checkpoint genes")
    requested_keys = {"rna": rna_cell_type_key, "atac": atac_cell_type_key}
    for modality, key in requested_keys.items():
        if key is not None and key not in source.obs:
            raise ValueError(f"{modality.upper()} cell type column {key!r} is missing from expression obs")
    if condition_key is not None and condition_key not in source.obs:
        raise ValueError(f"Condition column {condition_key!r} is missing from expression obs")
    condition_key = condition_key or ("condition" if "condition" in source.obs else None)
    batches = source.obs[batch_key].astype(str).to_numpy()
    eligible = source.obs_names.astype(str).isin(eligible_cells)
    rng = np.random.default_rng(seed)
    output = {}
    sampled_by_modality = {}
    values_by_modality = {}
    for modality in ("rna", "atac"):
        candidates = np.flatnonzero(eligible & np.isin(
            batches, [b for b in batch_names if b.startswith(modality + "_")]))
        if not len(candidates):
            continue
        # Allocate cells approximately in proportion to batch size while
        # retaining at least one cell from each batch in the reference plot.
        sampled = []
        groups = [(b, candidates[batches[candidates] == b]) for b in batch_names if b.startswith(modality + "_")]
        target = min(max_cells, len(candidates))
        if target < len(groups):
            raise ValueError(f"max_cells={max_cells} is smaller than the number of {modality} batches")
        allocation = {b: min(len(indices), max(1, int(target * len(indices) / len(candidates))))
                      for b, indices in groups}
        while sum(allocation.values()) > target:
            b = max((b for b in allocation if allocation[b] > 1), key=lambda b: allocation[b], default=None)
            if b is None:
                break
            allocation[b] -= 1
        while sum(allocation.values()) < target:
            b = max((b for b, indices in groups if allocation[b] < len(indices)),
                    key=lambda b: len(dict(groups)[b]) - allocation[b], default=None)
            if b is None:
                break
            allocation[b] += 1
        for b, indices in groups:
            sampled.extend(rng.choice(indices, size=allocation[b], replace=False).tolist())
        sampled = np.asarray(sorted(sampled), dtype=np.int64)
        sampled_by_modality[modality] = sampled
        values = source[sampled, gene_names].to_memory().X
        if not sparse.issparse(values):
            values = sparse.csr_matrix(np.asarray(values, dtype=np.float32))
        else:
            values = values.tocsr().astype(np.float32)
        values_by_modality[modality] = values
        if modality == "atac":
            peak_source = ad.read_h5ad(peak_path, backed="r")
            try:
                peak_indices = peak_source.obs_names.get_indexer(source.obs_names[sampled])
                if (peak_indices < 0).any():
                    raise ValueError(f"{(peak_indices < 0).sum()} sampled ATAC cells missing from peak matrix")
                counts = rows(peak_source.X, peak_indices)
            finally:
                peak_source.file.close()
            prevalence = np.asarray((counts > 0).sum(axis=0)).ravel()
            threshold = min(min_peak_cells, max(2, len(sampled) // 100))
            keep = (prevalence >= threshold) & (prevalence <= .95 * len(sampled))
            counts = counts[:, keep]
            if counts.shape[1] < 4:
                raise ValueError("Too few ATAC peaks remain for reference LSI")
            tfidf = TfidfTransformer(norm="l2", use_idf=True, smooth_idf=True).fit_transform(counts)
            n_lsi = min(31, min(tfidf.shape) - 1)
            if n_lsi < 3:
                raise ValueError("Too few ATAC cells or peaks for reference LSI")
            lsi = TruncatedSVD(n_components=n_lsi, n_iter=7, random_state=seed).fit_transform(tfidf)
            reduced = normalize(StandardScaler().fit_transform(lsi[:, 1:])).astype(np.float32)
            del counts, tfidf, lsi
        else:
            n_components = min(30, min(values.shape) - 1)
            if n_components < 2:
                raise ValueError("RNA: too few sampled cells or genes for PCA")
            reduced = sc.pp.pca(values, n_comps=n_components, zero_center=True,
                                svd_solver="arpack", random_state=seed)
        coords = UMAP(n_neighbors=min(15, len(sampled) - 1), min_dist=0.3,
                      random_state=seed, n_jobs=1).fit_transform(reduced)
        frame = pd.DataFrame({
            "cell_barcode": source.obs_names[sampled].astype(str),
            "batch": batches[sampled],
            "UMAP1": coords[:, 0], "UMAP2": coords[:, 1],
        })
        frame["condition"] = (_obs_labels(source, condition_key, sampled)
                              if condition_key else "Unavailable")
        cell_type_key = requested_keys[modality]
        if cell_type_key is None:
            cell_type_key = next((key for key in ("cell_type", "celltype") if key in source.obs), None)
        frame["cell_type"] = (_obs_labels(source, cell_type_key, sampled)
                              if cell_type_key else "Unavailable")
        frame["cell_type_source_key"] = cell_type_key or "Unavailable"
        output[modality] = frame
    if not all(modality in sampled_by_modality for modality in ("rna", "atac")):
        raise ValueError("Combined expression UMAP requires sampled RNA and ATAC cells")
    combined_indices = np.concatenate([sampled_by_modality["rna"], sampled_by_modality["atac"]])
    combined_values = sparse.vstack([values_by_modality["rna"], values_by_modality["atac"]], format="csr")
    n_components = min(30, min(combined_values.shape) - 1)
    reduced = sc.pp.pca(combined_values, n_comps=n_components, zero_center=True,
                        svd_solver="arpack", random_state=seed)
    coords = UMAP(n_neighbors=min(15, len(combined_indices) - 1), min_dist=0.3,
                  random_state=seed, n_jobs=1).fit_transform(reduced)
    combined = pd.concat([output["rna"], output["atac"]], ignore_index=True)
    combined[["UMAP1", "UMAP2"]] = coords
    combined["source_modality"] = np.repeat(("RNA", "ATAC"),
                                            [len(sampled_by_modality["rna"]), len(sampled_by_modality["atac"])])
    output["expression"] = combined
    return output


def _scatter_panel(ax, frame, color, title):
    labels = frame[color].astype(str)
    levels = sorted(labels.unique())
    palette = plt.get_cmap("tab20" if len(levels) <= 20 else "hsv")
    for i, level in enumerate(levels):
        mask = labels.eq(level).to_numpy()
        ax.scatter(frame.loc[mask, "UMAP1"], frame.loc[mask, "UMAP2"],
                   s=2.5, alpha=0.72, linewidths=0, color=palette(i / max(len(levels), 1)),
                   label=level, rasterized=True)
    ax.set_title(title)
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_box_aspect(1)
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if len(levels) <= 30:
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=6,
                  markerscale=3, frameon=False, ncol=1)


def _plot_umaps(reference, labels, output_dir, title):
    label_map = labels.set_index("cell_barcode")["cluster"]
    if not label_map.index.is_unique:
        raise ValueError("Duplicate cell_barcode in stage labels")
    paths = []
    for modality, base in reference.items():
        frame = base.copy()
        frame["cluster"] = label_map.reindex(frame["cell_barcode"]).to_numpy()
        if frame["cluster"].isna().any():
            raise ValueError(f"{modality}: stage labels do not cover sampled input cells")
        frame.to_csv(output_dir / f"umap_{modality}_coordinates.csv", index=False)
        cols = ["batch", "condition", "cell_type", "cluster"]
        fig, axes = plt.subplots(2, 2, figsize=(16, 13), constrained_layout=True)
        for ax, col in zip(axes.flat, cols):
            _scatter_panel(ax, frame, col, f"{modality.upper()} | {col.replace('_', ' ')}")
        fig.suptitle(f"{title}: reference input UMAP")
        path = output_dir / f"umap_{modality}.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        paths.append(str(path))
    return paths


def _alignment_bulks(source, labels, gene_names, batch_names, batch_key):
    labels = labels.set_index("cell_barcode")
    if not labels.index.is_unique:
        raise ValueError("Duplicate alignment cell barcode")
    genes = list(map(str, gene_names))
    K = int(labels["cluster"].max()) + 1
    batches = source.obs[batch_key].astype(str).to_numpy()
    profiles, counts = [], []
    for batch in batch_names:
        indices = np.flatnonzero(batches == batch)
        names = source.obs_names[indices].astype(str)
        stage = labels.reindex(names)
        if stage["cluster"].isna().any() or not stage["batch"].astype(str).eq(batch).all():
            raise ValueError(f"{batch}: alignment labels differ from input")
        clusters = stage["cluster"].astype(int).to_numpy()
        count = np.bincount(clusters, minlength=K)
        X = source[indices, genes].to_memory().X
        if sparse.issparse(X):
            X = X.tocsr().astype(np.float32)
            X.data = np.expm1(X.data)
        else:
            X = np.expm1(np.asarray(X, dtype=np.float32))
        assignments = sparse.csr_matrix((np.ones(len(indices), dtype=np.float32),
                                         (np.arange(len(indices)), clusters)), shape=(len(indices), K))
        sums = assignments.T @ X
        mean = np.asarray(sums.toarray() if sparse.issparse(sums) else sums, dtype=np.float32)
        mean /= np.maximum(count[:, None], 1)
        profiles.append(np.log1p(mean).T)
        counts.append(count)
    return profiles, counts


def _train_bulks(checkpoint):
    bulks = [_numpy(x).astype(np.float32) for x in checkpoint["bulks_list"]]
    counts = [np.bincount(_numpy(p).argmax(axis=1), minlength=p.shape[1])
              for p in checkpoint["probs_soft"]]
    return bulks, counts


def _pearson_rows(matrix):
    matrix = matrix - matrix.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norm, 1e-8)


def _correlation(profiles, counts, batch_names, top_genes, min_cells):
    rna = [i for i, b in enumerate(batch_names) if b.startswith("rna_")]
    atac = [i for i, b in enumerate(batch_names) if b.startswith("atac_")]
    if not rna or not atac:
        raise ValueError("Correlation requires RNA and ATAC batches")
    K = profiles[0].shape[1]
    G = profiles[0].shape[0]
    if any(p.shape != (G, K) for p in profiles):
        raise ValueError("Pseudobulk gene/cluster dimensions differ across batches")
    stacked = np.concatenate([p.T for p in profiles], axis=0)
    variance = np.var(stacked, axis=0)
    n = min(top_genes, G)
    if n < 2:
        raise ValueError("top_genes must select at least two genes")
    selected = np.argpartition(variance, -n)[-n:]
    vectors = [_pearson_rows(p[selected, :].T.copy()) for p in profiles]
    total = np.zeros((K, K), dtype=np.float64)
    support = np.zeros((K, K), dtype=np.int32)
    batch_diag = np.full((len(atac), len(rna)), np.nan, dtype=np.float32)
    all_batch_diag = np.full((len(batch_names), len(batch_names)), np.nan, dtype=np.float32)
    for i in range(len(batch_names)):
        for j in range(i, len(batch_names)):
            corr = np.clip(vectors[i] @ vectors[j].T, -1, 1)
            valid = (counts[i][:, None] >= min_cells) & (counts[j][None, :] >= min_cells)
            diagonal = np.diag(corr)[np.diag(valid)]
            if len(diagonal):
                all_batch_diag[i, j] = all_batch_diag[j, i] = float(diagonal.mean())
    for ai, a in enumerate(atac):
        for ri, r in enumerate(rna):
            corr = np.clip(vectors[a] @ vectors[r].T, -1, 1)
            valid = (counts[a][:, None] >= min_cells) & (counts[r][None, :] >= min_cells)
            total += np.where(valid, corr, 0)
            support += valid
            diagonal = np.diag(corr)[np.diag(valid)]
            if len(diagonal):
                batch_diag[ai, ri] = float(diagonal.mean())
    avg = np.divide(total, support, out=np.full_like(total, np.nan), where=support > 0)
    return (avg, support, batch_diag, all_batch_diag,
            [batch_names[i] for i in atac], [batch_names[i] for i in rna], n)


def _heatmap(matrix, rows, cols, title, path, annotate=False):
    fig, ax = plt.subplots(figsize=(max(7, 0.38 * len(cols) + 3), max(6, 0.36 * len(rows) + 2)),
                           constrained_layout=True)
    masked = np.ma.masked_invalid(matrix)
    image = ax.imshow(masked, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(cols)), cols, rotation=90, fontsize=7)
    ax.set_yticks(np.arange(len(rows)), rows, fontsize=7)
    ax.set_title(title)
    if annotate and len(rows) * len(cols) <= 400:
        for i, j in np.ndindex(matrix.shape):
            if np.isfinite(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=5)
    fig.colorbar(image, ax=ax, label="Pearson r", shrink=0.75)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_correlation(profiles, counts, batch_names, output_dir, top_genes, min_cells):
    avg, support, batch_diag, all_batch_diag, atac, rna, n = _correlation(
        profiles, counts, batch_names, top_genes, min_cells)
    clusters = [str(i) for i in range(avg.shape[0])]
    pd.DataFrame(avg, index=clusters, columns=clusters).to_csv(output_dir / "cross_modal_cluster_correlation.csv")
    pd.DataFrame(support, index=clusters, columns=clusters).to_csv(output_dir / "cross_modal_pair_support.csv")
    pd.DataFrame(batch_diag, index=atac, columns=rna).to_csv(output_dir / "matched_cluster_batch_correlation.csv")
    pd.DataFrame(all_batch_diag, index=batch_names, columns=batch_names).to_csv(
        output_dir / "all_batch_matched_correlation.csv")
    path1 = output_dir / "cross_modal_cluster_correlation.png"
    path2 = output_dir / "matched_cluster_batch_correlation.png"
    path3 = output_dir / "all_batch_matched_correlation.png"
    _heatmap(avg, clusters, clusters, f"ATAC × RNA pseudobulk correlation ({n} variable genes)",
             path1, annotate=len(clusters) <= 20)
    _heatmap(batch_diag, atac, rna, "Mean same-cluster correlation by batch pair", path2)
    _heatmap(all_batch_diag, batch_names, batch_names,
             "Mean same-cluster correlation across all batch pairs", path3)
    return [str(path1), str(path2), str(path3)]


def plot_results(config, stage="all", train_stage="all", max_cells=12000,
                 top_genes=1000, min_cells=10, rna_cell_type_key=None,
                 atac_cell_type_key=None, condition_key=None):
    cfg = load_config(config)
    plot_cfg = cfg["plotting"]
    rna_cell_type_key = rna_cell_type_key if rna_cell_type_key is not None else plot_cfg["rna_cell_type_key"]
    atac_cell_type_key = atac_cell_type_key if atac_cell_type_key is not None else plot_cfg["atac_cell_type_key"]
    condition_key = condition_key if condition_key is not None else plot_cfg["condition_key"]
    root = Path(cfg["output_dir"])
    plots = root / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    if min_cells < 1 or top_genes < 2:
        raise ValueError("min_cells must be positive and top_genes at least 2")
    source = ad.read_h5ad(cfg["input"]["expression_h5ad"], backed="r")
    result = {}
    try:
        plans = []
        if stage in ("align", "all"):
            alignment = root / "alignment"
            labels_path = alignment / "cell_labels.csv"
            checkpoint_path = alignment / "alignment_checkpoint.pt"
            if not labels_path.is_file() or not checkpoint_path.is_file():
                raise FileNotFoundError("Alignment output is incomplete; run align first")
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            labels = pd.read_csv(labels_path, dtype={"cell_barcode": str, "batch": str})
            labels = labels.rename(columns={"final_alignment": "cluster"})[["cell_barcode", "batch", "cluster"]]
            batch_names = list(cfg["input"].get("batches") or labels["batch"].drop_duplicates())
            plans.append(("align", labels, checkpoint["gene_names"], batch_names, checkpoint))
        if stage in ("train", "all"):
            summary_path = root / "optimization" / "training_summary.json"
            if not summary_path.is_file():
                raise FileNotFoundError("Training summary is missing; run train first")
            summary = json.loads(summary_path.read_text())
            selected = list(summary["stage_checkpoints"]) if train_stage == "all" else [train_stage]
            for name in selected:
                if name not in summary["stage_checkpoints"]:
                    raise ValueError(f"Training stage {name!r} was not run")
                checkpoint_path = Path(summary["stage_checkpoints"][name])
                if not checkpoint_path.is_absolute():
                    checkpoint_path = summary_path.parent / checkpoint_path
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                labels_path = checkpoint_path.with_suffix(".labels.csv")
                labels = pd.read_csv(labels_path, dtype={"cell_barcode": str, "batch": str})
                labels = labels[["cell_barcode", "batch", "cluster"]]
                plans.append((f"train/{name}", labels, checkpoint["gene_names"],
                              list(checkpoint["batch_names"]), checkpoint))
        if not plans:
            raise ValueError("stage must be align, train, or all")
        first_genes = list(map(str, plans[0][2]))
        if any(list(map(str, item[2])) != first_genes for item in plans):
            raise ValueError("Stage checkpoints use different gene sets; plot them separately")
        batch_names = list(dict.fromkeys(b for item in plans for b in item[3]))
        eligible_cells = set.intersection(*[
            set(item[1]["cell_barcode"].astype(str)) for item in plans
        ])
        if not eligible_cells:
            raise ValueError("No input cells are shared by the selected result stages")
        batch_key = cfg["input"]["batch_key"]
        reference = _reference_embedding(source, cfg["input"]["peak_h5ad"], first_genes,
                                         batch_names, eligible_cells, batch_key, max_cells,
                                         int(cfg["runtime"]["seed"]),
                                         int(cfg["alignment"]["min_peak_cells"]),
                                         rna_cell_type_key=rna_cell_type_key,
                                         atac_cell_type_key=atac_cell_type_key,
                                         condition_key=condition_key)
        for name, labels, genes, names, checkpoint in plans:
            output = plots / name
            output.mkdir(parents=True, exist_ok=True)
            if name == "align":
                profiles, counts = _alignment_bulks(source, labels, genes, names, batch_key)
            else:
                profiles, counts = _train_bulks(checkpoint)
            files = _plot_correlation(profiles, counts, names, output, top_genes, min_cells)
            files += _plot_umaps(reference, labels, output, name)
            result[name] = {"directory": str(output), "figures": files,
                            "correlation_genes": min(top_genes, len(genes)),
                            "umap_reference": "RNA expression PCA; ATAC peak TF-IDF/LSI; combined expression PCA"}
            result[name]["umap_label_keys"] = {
                "condition": condition_key or ("condition" if "condition" in source.obs else "Unavailable"),
                "rna_cell_type": reference["rna"]["cell_type_source_key"].iloc[0],
                "atac_cell_type": reference["atac"]["cell_type_source_key"].iloc[0],
            }
    finally:
        source.file.close()
    summary_path = plots / "plot_summary.json"
    previous = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
    previous.update(result)
    summary_path.write_text(json.dumps(previous, indent=2) + "\n")
    return result

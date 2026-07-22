#!/usr/bin/env python3
"""Marker-weighted cell-type correlation for ResonanSC gene-space results.

This version is intentionally reference-oriented:
1. Collapse RNA/ATAC cell-type names to a shared ontology.
2. Derive up-regulated markers for each collapsed cell type.
3. Compute RNA and ATAC cell-type pseudobulks per batch in gene space.
4. Z-score each gene across cell types within each batch.
5. For each ATAC x RNA type pair, correlate only the reference type's marker
   genes with marker weights.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba-cache")

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import seaborn as sns
import torch

EPS = 1e-8


LABEL_MAP = {
    "CD14 Mono": "Monocyte",
    "CD16 Mono": "Monocyte",
    "Monocyte": "Monocyte",
    "B naive": "Naive B",
    "Naive B": "Naive B",
    "B memory": "Memory B",
    "Memory B": "Memory B",
    "Plasmablast": "Plasma cell",
    "Plasma cell": "Plasma cell",
    "CD4 Naive": "Naive CD4 T",
    "Naive CD4 T": "Naive CD4 T",
    "CD4 TCM": "Memory CD4 T",
    "CD4 TEM": "Memory CD4 T",
    "Memory CD4 T": "Memory CD4 T",
    "Treg": "Treg",
    "CD8 Naive": "Naive CD8 T",
    "Naive CD8 T": "Naive CD8 T",
    "CD8 TEM": "Effector memory CD8 T",
    "Effector memory CD8 T": "Effector memory CD8 T",
    "CD8 TCM": "Central memory CD8 T",
    "Central memory CD8 T": "Central memory CD8 T",
    "NK": "NK",
    "NK_CD56bright": "NK",
    "MAIT": "MAIT",
    "cDC": "cDC",
    "cDC2": "cDC",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary",
        default="outputs/result2/1(old)/training_round_4_summary.json",
    )
    parser.add_argument("--checkpoint-stage", default="learn_M", choices=["learn_P", "learn_mapping", "learn_M"])
    parser.add_argument("--group-key", default="cell_type")
    parser.add_argument("--de-top-n", type=int, default=75)
    parser.add_argument("--de-pval-th", type=float, default=0.05)
    parser.add_argument("--min-marker-genes", type=int, default=8)
    parser.add_argument("--min-cells", type=int, default=10)
    parser.add_argument("--marker-source", default="rna", choices=["rna", "both"])
    parser.add_argument("--flip-atac-sign", action="store_true")
    parser.add_argument("--unique-markers", action="store_true")
    parser.add_argument("--output", default=None)
    parser.add_argument("--annot", action="store_true")
    return parser.parse_args()


def log(msg: str) -> None:
    print(msg, flush=True)


def read_batch(summary: dict, batch_idx: int, batch_name: str) -> ad.AnnData:
    path = Path(summary["training_data_dir"]) / f"{batch_idx:02d}_{batch_name}.h5ad"
    if not path.exists():
        raise FileNotFoundError(path)
    return ad.read_h5ad(path)


def collapsed_groups(adata: ad.AnnData, group_key: str) -> pd.Series:
    raw = adata.obs[group_key].astype(str)
    mapped = raw.map(LABEL_MAP)
    return mapped.astype("string")


def valid_common_labels(summary: dict, batch_names: list[str], rna_idx: list[int], atac_idx: list[int], group_key: str) -> list[str]:
    rna = set()
    atac = set()
    for i in rna_idx:
        groups = collapsed_groups(read_batch(summary, i, batch_names[i]), group_key).dropna()
        rna.update(groups.unique().tolist())
    for i in atac_idx:
        groups = collapsed_groups(read_batch(summary, i, batch_names[i]), group_key).dropna()
        atac.update(groups.unique().tolist())
    return sorted(rna & atac)


def rna_gene_space_adata(adata: ad.AnnData, gene_names: list[str]) -> ad.AnnData:
    if list(adata.var_names.astype(str)) == list(gene_names):
        return adata
    return adata[:, gene_names].copy()


def rna_bulk(adata: ad.AnnData, labels: list[str], group_key: str, min_cells: int) -> tuple[np.ndarray, np.ndarray]:
    groups = collapsed_groups(adata, group_key)
    rows = []
    counts = []
    for label in labels:
        mask = groups.eq(label).fillna(False).to_numpy()
        n = int(mask.sum())
        counts.append(n)
        if n < min_cells:
            rows.append(np.full(adata.n_vars, np.nan, dtype=np.float32))
            continue
        x = adata.X[mask]
        if sp.issparse(x):
            x = x.copy()
            x.data = np.expm1(x.data)
            mean = np.asarray(x.mean(axis=0)).ravel()
        else:
            mean = np.expm1(np.asarray(x)).mean(axis=0)
        rows.append(np.log1p(np.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0)).astype(np.float32))
    return np.vstack(rows), np.asarray(counts, dtype=int)


def peak_gene_mapping(adata: ad.AnnData, mapping_info: dict, mapping_weights: torch.Tensor, n_genes: int) -> sp.csr_matrix:
    peak_idx = np.asarray(mapping_info["peak_idx"], dtype=np.int64)
    gene_idx = np.asarray(mapping_info["gene_idx"], dtype=np.int64)
    weights = mapping_weights.detach().cpu().numpy().astype(np.float32)
    return sp.coo_matrix((weights, (peak_idx, gene_idx)), shape=(adata.n_vars, n_genes), dtype=np.float32).tocsr()


def atac_cell_gene_adata(
    adata: ad.AnnData, mapping_info: dict, mapping_weights: torch.Tensor, gene_names: list[str]
) -> ad.AnnData:
    mapping = peak_gene_mapping(adata, mapping_info, mapping_weights, len(gene_names))
    x = adata.X
    if not sp.issparse(x):
        x = sp.csr_matrix(np.asarray(x, dtype=np.float32))
    else:
        x = x.tocsr().astype(np.float32, copy=False)
    row_sum = np.asarray(x.sum(axis=1)).ravel()
    tf = x.multiply(1.0 / (row_sum[:, None] + EPS)).tocsr()
    feature_sum = np.asarray(tf.sum(axis=0)).ravel()
    idf = np.log1p(x.shape[0] / (feature_sum + EPS)).astype(np.float32)
    tfidf = tf.multiply(idf).tocsr()
    gene_activity = (tfidf @ mapping).tocsr()
    gene_activity.data = np.log1p(gene_activity.data)
    gene_activity.eliminate_zeros()
    return ad.AnnData(X=gene_activity, obs=adata.obs.copy(), var=pd.DataFrame(index=pd.Index(gene_names)))


def gene_space_bulk_from_cell_matrix(adata: ad.AnnData, labels: list[str], group_key: str, min_cells: int) -> tuple[np.ndarray, np.ndarray]:
    groups = collapsed_groups(adata, group_key)
    rows = []
    counts = []
    for label in labels:
        mask = groups.eq(label).fillna(False).to_numpy()
        n = int(mask.sum())
        counts.append(n)
        if n < min_cells:
            rows.append(np.full(adata.n_vars, np.nan, dtype=np.float32))
            continue
        if sp.issparse(adata.X):
            rows.append(np.asarray(adata.X[mask].mean(axis=0)).ravel().astype(np.float32))
        else:
            rows.append(np.asarray(adata.X[mask]).mean(axis=0).astype(np.float32))
    return np.vstack(rows), np.asarray(counts, dtype=int)


def prepare_rna_for_de(adata: ad.AnnData, group_key: str, gene_names: list[str]) -> ad.AnnData:
    adata = rna_gene_space_adata(adata, gene_names).copy()
    groups = collapsed_groups(adata, group_key)
    keep = groups.notna().to_numpy()
    adata = adata[keep].copy()
    groups = groups[keep].astype(str)
    adata.obs["collapsed_cell_type"] = pd.Categorical(groups.to_numpy())
    return adata


def prepare_gene_adata_for_de(adata: ad.AnnData, group_key: str) -> ad.AnnData:
    adata = adata.copy()
    groups = collapsed_groups(adata, group_key)
    keep = groups.notna().to_numpy()
    adata = adata[keep].copy()
    groups = groups[keep].astype(str)
    adata.obs["collapsed_cell_type"] = pd.Categorical(groups.to_numpy())
    return adata


def _pick_markers(df: pd.DataFrame, top_n: int, pval_th: float) -> dict[str, float]:
    if df is None or df.empty:
        return {}
    use = df
    if "pvals_adj" in use.columns:
        sig = use[use["pvals_adj"] < pval_th]
        if len(sig) >= min(top_n, len(use)):
            use = sig
    if "logfoldchanges" in use.columns:
        use = use[use["logfoldchanges"] > 0].sort_values("logfoldchanges", ascending=False)
    out = {}
    for row in use.head(top_n).itertuples(index=False):
        gene = str(getattr(row, "names"))
        lfc = float(getattr(row, "logfoldchanges", 1.0))
        padj = float(getattr(row, "pvals_adj", 1.0))
        out[gene] = max(0.0, lfc) * max(1.0, -np.log10(max(padj, 1e-300)))
    return out


def marker_weights_by_label(
    batches: list[ad.AnnData],
    labels: list[str],
    top_n: int,
    pval_th: float,
    min_cells: int,
    log_prefix: str,
) -> dict[str, dict[str, float]]:
    accum: dict[str, dict[str, list[float]]] = {label: defaultdict(list) for label in labels}
    for bi, adata in enumerate(batches):
        counts = adata.obs["collapsed_cell_type"].astype(str).value_counts()
        valid = [label for label in labels if counts.get(label, 0) >= min_cells and adata.n_obs - counts.get(label, 0) >= min_cells]
        if not valid:
            continue
        log(f"{log_prefix} marker DEG batch {bi + 1}: {len(valid)} groups")
        sc.tl.rank_genes_groups(
            adata,
            groupby="collapsed_cell_type",
            groups=valid,
            reference="rest",
            method="wilcoxon",
            use_raw=False,
            n_genes=min(top_n * 3, adata.n_vars),
        )
        for label in valid:
            markers = _pick_markers(sc.get.rank_genes_groups_df(adata, group=label), top_n, pval_th)
            for gene, weight in markers.items():
                accum[label][gene].append(weight)
    return {
        label: {gene: float(np.mean(weights)) for gene, weights in genes.items()}
        for label, genes in accum.items()
    }


def combine_marker_weights(
    rna_weights: dict[str, dict[str, float]],
    atac_weights: dict[str, dict[str, float]],
    labels: list[str],
) -> tuple[dict[str, dict[str, float]], pd.DataFrame]:
    combined: dict[str, dict[str, float]] = {}
    rows = []
    for label in labels:
        genes = sorted(set(rna_weights.get(label, {})) | set(atac_weights.get(label, {})))
        combined[label] = {}
        for gene in genes:
            vals = []
            if gene in rna_weights.get(label, {}):
                vals.append(float(rna_weights[label][gene]))
            if gene in atac_weights.get(label, {}):
                vals.append(float(atac_weights[label][gene]))
            weight = float(np.mean(vals)) if vals else 0.0
            combined[label][gene] = weight
            rows.append(
                {
                    "cell_type": label,
                    "gene": gene,
                    "weight": weight,
                    "rna_weight": rna_weights.get(label, {}).get(gene, np.nan),
                    "atac_weight": atac_weights.get(label, {}).get(gene, np.nan),
                    "source": "both"
                    if gene in rna_weights.get(label, {}) and gene in atac_weights.get(label, {})
                    else ("rna" if gene in rna_weights.get(label, {}) else "atac"),
                }
            )
    return combined, pd.DataFrame(rows)


def keep_unique_marker_owner(marker_weights: dict[str, dict[str, float]], labels: list[str]) -> dict[str, dict[str, float]]:
    owners: dict[str, tuple[str, float]] = {}
    for label in labels:
        for gene, weight in marker_weights.get(label, {}).items():
            weight = float(weight)
            if gene not in owners or weight > owners[gene][1]:
                owners[gene] = (label, weight)
    unique = {label: {} for label in labels}
    for gene, (label, weight) in owners.items():
        unique[label][gene] = weight
    return unique


def zscore_genes_across_labels(bulk: np.ndarray) -> np.ndarray:
    out = bulk.copy().astype(np.float64)
    mean = np.nanmean(out, axis=0, keepdims=True)
    std = np.nanstd(out, axis=0, keepdims=True)
    out = (out - mean) / np.maximum(std, EPS)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def weighted_corr(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0).clip(min=0)
    if w.sum() <= EPS:
        return np.nan
    w = w / (w.sum() + EPS)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    xm = x - np.sum(w * x)
    ym = y - np.sum(w * y)
    denom = np.sqrt(np.sum(w * xm * xm) * np.sum(w * ym * ym))
    return float(np.clip(np.sum(w * xm * ym) / max(denom, EPS), -1.0, 1.0))


def corr_from_reference_markers(
    atac_bulk: np.ndarray,
    rna_bulk: np.ndarray,
    labels: list[str],
    marker_weights: dict[str, dict[str, float]],
    gene_to_idx: dict[str, int],
    min_marker_genes: int,
    flip_atac_sign: bool,
) -> tuple[np.ndarray, np.ndarray]:
    atac_z = zscore_genes_across_labels(atac_bulk)
    if flip_atac_sign:
        atac_z = -atac_z
    rna_z = zscore_genes_across_labels(rna_bulk)
    corr = np.full((len(labels), len(labels)), np.nan, dtype=np.float64)
    marker_counts = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for i in range(len(labels)):
        for j, ref_label in enumerate(labels):
            markers = marker_weights.get(ref_label, {})
            genes = [gene for gene in markers if gene in gene_to_idx]
            if len(genes) < min_marker_genes:
                continue
            idx = np.asarray([gene_to_idx[g] for g in genes], dtype=int)
            weights = np.asarray([markers[g] for g in genes], dtype=np.float64)
            marker_counts[i, j] = len(idx)
            corr[i, j] = weighted_corr(atac_z[i, idx], rna_z[j, idx], weights)
    return corr, marker_counts


def plot_heatmap(corr_df: pd.DataFrame, out: Path, annot: bool, prefix: str, title_source: str) -> None:
    fig, ax = plt.subplots(figsize=(max(9, 0.5 * corr_df.shape[1] + 3), max(8, 0.5 * corr_df.shape[0] + 2)), constrained_layout=True)
    sns.heatmap(
        corr_df,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        center=0,
        annot=annot,
        fmt=".2f",
        linewidths=0.4,
        cbar_kws={"label": "Weighted Pearson r", "shrink": 0.7, "aspect": 24},
        ax=ax,
    )
    ax.set_title(f"{title_source} marker-weighted cell-type correlation\nbatch-pair averaged, gene z-scored")
    ax.set_xlabel("RNA reference cell type")
    ax.set_ylabel("ATAC cell type")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    fig.savefig(out / f"{prefix}_marker_weighted_corr.png", dpi=300)
    fig.savefig(out / f"{prefix}_marker_weighted_corr.pdf")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    summary_path = Path(args.summary)
    summary = json.loads(summary_path.read_text())
    checkpoint = Path(summary["stage_checkpoints"][args.checkpoint_stage])
    output = (
        Path(args.output)
        if args.output
        else summary_path.parent / f"round4_{args.checkpoint_stage}_celltype_{args.marker_source}_marker_weighted_corr"
    )
    output.mkdir(parents=True, exist_ok=True)

    log(f"Reading checkpoint: {checkpoint}")
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    batch_names = list(ckpt.get("batch_names", summary["batch_names"]))
    gene_names = list(ckpt["gene_names"])
    gene_to_idx = {gene: idx for idx, gene in enumerate(gene_names)}
    rna_idx = [i for i, name in enumerate(batch_names) if str(name).startswith("rna")]
    atac_idx = [i for i, name in enumerate(batch_names) if str(name).startswith("atac")]
    labels = valid_common_labels(summary, batch_names, rna_idx, atac_idx, args.group_key)
    log(f"Common collapsed cell types: {len(labels)} ({', '.join(labels)})")

    rna_adatas = [prepare_rna_for_de(read_batch(summary, i, batch_names[i]), args.group_key, gene_names) for i in rna_idx]
    rna_marker_weights = marker_weights_by_label(
        rna_adatas, labels, args.de_top_n, args.de_pval_th, args.min_cells, "RNA"
    )

    rna_bulks, rna_counts = {}, {}
    for local_i, batch_i in enumerate(rna_idx):
        bulk, counts = rna_bulk(rna_adatas[local_i], labels, "collapsed_cell_type", args.min_cells)
        rna_bulks[batch_i] = bulk
        rna_counts[batch_i] = counts

    atac_bulks, atac_counts = {}, {}
    atac_gene_adatas = []
    for batch_i in atac_idx:
        raw = read_batch(summary, batch_i, batch_names[batch_i])
        gene_adata = atac_cell_gene_adata(raw, ckpt["mapping_init"][batch_i], ckpt["mapping_weights"][batch_i], gene_names)
        gene_adata = prepare_gene_adata_for_de(gene_adata, args.group_key)
        atac_gene_adatas.append(gene_adata)
        bulk, counts = gene_space_bulk_from_cell_matrix(gene_adata, labels, args.group_key, args.min_cells)
        atac_bulks[batch_i] = bulk
        atac_counts[batch_i] = counts

    if args.marker_source == "both":
        atac_marker_weights = marker_weights_by_label(
            atac_gene_adatas, labels, args.de_top_n, args.de_pval_th, args.min_cells, "ATAC gene-space"
        )
        marker_weights, marker_weight_df = combine_marker_weights(rna_marker_weights, atac_marker_weights, labels)
    else:
        atac_marker_weights = {label: {} for label in labels}
        marker_weights = rna_marker_weights
        marker_weight_df = pd.DataFrame(
            [
                {
                    "cell_type": label,
                    "gene": gene,
                    "weight": weight,
                    "rna_weight": weight,
                    "atac_weight": np.nan,
                    "source": "rna",
                }
                for label, genes in marker_weights.items()
                for gene, weight in genes.items()
            ]
        )
    if args.unique_markers:
        marker_weights = keep_unique_marker_owner(marker_weights, labels)
        keep = {
            (label, gene)
            for label, genes in marker_weights.items()
            for gene in genes
        }
        marker_weight_df = marker_weight_df[
            marker_weight_df.apply(lambda row: (row["cell_type"], row["gene"]) in keep, axis=1)
        ].copy()
    marker_weight_df.to_csv(output / f"{args.marker_source}_marker_weights.csv", index=False)

    corr_sum = np.zeros((len(labels), len(labels)), dtype=np.float64)
    corr_count = np.zeros((len(labels), len(labels)), dtype=np.int64)
    marker_sum = np.zeros((len(labels), len(labels)), dtype=np.float64)
    for ai in atac_idx:
        for ri in rna_idx:
            corr, marker_counts = corr_from_reference_markers(
                atac_bulks[ai],
                rna_bulks[ri],
                labels,
                marker_weights,
                gene_to_idx,
                args.min_marker_genes,
                args.flip_atac_sign,
            )
            valid = (atac_counts[ai][:, None] >= args.min_cells) & (rna_counts[ri][None, :] >= args.min_cells)
            corr = np.where(valid, corr, np.nan)
            corr_sum += np.nan_to_num(corr, nan=0.0)
            corr_count += np.isfinite(corr)
            marker_sum += np.where(np.isfinite(corr), marker_counts, 0)

    corr_avg = np.divide(corr_sum, np.maximum(corr_count, 1), out=np.full_like(corr_sum, np.nan), where=corr_count > 0)
    marker_avg = np.divide(marker_sum, np.maximum(corr_count, 1), out=np.zeros_like(marker_sum), where=corr_count > 0)
    corr_df = pd.DataFrame(corr_avg, index=labels, columns=labels)
    count_df = pd.DataFrame(corr_count, index=labels, columns=labels)
    marker_df = pd.DataFrame(marker_avg, index=labels, columns=labels)
    corr_df.to_csv(output / f"{args.marker_source}_marker_weighted_corr.csv")
    count_df.to_csv(output / f"{args.marker_source}_marker_weighted_corr_pair_counts.csv")
    marker_df.to_csv(output / f"{args.marker_source}_marker_weighted_corr_marker_counts.csv")
    plot_heatmap(
        corr_df,
        output,
        args.annot,
        args.marker_source,
        "RNA+ATAC DEG" if args.marker_source == "both" else "RNA DEG",
    )

    config = {
        "summary": str(summary_path),
        "checkpoint": str(checkpoint),
        "checkpoint_stage": args.checkpoint_stage,
        "group_key": args.group_key,
        "de_top_n": args.de_top_n,
        "de_pval_th": args.de_pval_th,
        "min_marker_genes": args.min_marker_genes,
        "min_cells": args.min_cells,
        "marker_source": args.marker_source,
        "flip_atac_sign": args.flip_atac_sign,
        "unique_markers": args.unique_markers,
        "labels": labels,
        "label_map": LABEL_MAP,
        "rna_batches": [batch_names[i] for i in rna_idx],
        "atac_batches": [batch_names[i] for i in atac_idx],
        "n_batch_pairs": len(rna_idx) * len(atac_idx),
        "marker_counts_by_label": {
            label: {
                "rna": len(rna_marker_weights.get(label, {})),
                "atac": len(atac_marker_weights.get(label, {})),
                "combined": len(marker_weights.get(label, {})),
            }
            for label in labels
        },
    }
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    log(f"Saved to {output}")


if __name__ == "__main__":
    main()

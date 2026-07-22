#!/usr/bin/env python3
"""Batch-pair averaged ATAC-vs-RNA correlation using obs cell-type bulks."""

from __future__ import annotations

import argparse
import json
import os
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary",
        default="outputs/result2/1(old)/training_round_4_summary.json",
    )
    parser.add_argument("--checkpoint-stage", default="learn_M", choices=["learn_P", "learn_mapping", "learn_M"])
    parser.add_argument("--group-key", default="cell_type")
    parser.add_argument("--corr-mode", choices=["variable", "deg"], default="variable")
    parser.add_argument("--top-variable-genes", type=int, default=1000)
    parser.add_argument("--de-top-n", type=int, default=50)
    parser.add_argument("--de-pval-th", type=float, default=0.05)
    parser.add_argument("--min-de-genes", type=int, default=5)
    parser.add_argument("--min-cells", type=int, default=10)
    parser.add_argument("--output", default=None)
    parser.add_argument("--annot", action="store_true")
    return parser.parse_args()


def log(msg: str) -> None:
    print(msg, flush=True)


def read_batch(summary: dict, batch_idx: int, batch_name: str) -> ad.AnnData:
    data_dir = Path(summary["training_data_dir"])
    path = data_dir / f"{batch_idx:02d}_{batch_name}.h5ad"
    if not path.exists():
        raise FileNotFoundError(path)
    return ad.read_h5ad(path)


def group_labels(adata: ad.AnnData, group_key: str) -> pd.Series:
    if group_key not in adata.obs:
        raise KeyError(f"{adata}: missing obs[{group_key!r}]")
    return adata.obs[group_key].astype(str)


def rna_celltype_bulk(adata: ad.AnnData, labels: list[str], group_key: str, min_cells: int) -> tuple[np.ndarray, np.ndarray]:
    groups = group_labels(adata, group_key)
    rows = []
    counts = []
    for label in labels:
        mask = groups.eq(label).to_numpy()
        n = int(mask.sum())
        counts.append(n)
        if n < min_cells:
            rows.append(np.full(adata.n_vars, np.nan, dtype=np.float32))
            continue
        x = adata.X[mask]
        # RNA training input is log1p-normalized. Aggregate in linear space,
        # then return to log1p space, matching ResonanSC.model.rna_bulk.
        if sp.issparse(x):
            x = x.copy()
            x.data = np.expm1(x.data)
            mean = np.asarray(x.mean(axis=0)).ravel()
        else:
            mean = np.expm1(np.asarray(x)).mean(axis=0)
        rows.append(np.log1p(np.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0)).astype(np.float32))
    return np.vstack(rows), np.asarray(counts, dtype=int)


def atac_celltype_gene_bulk(
    adata: ad.AnnData,
    labels: list[str],
    group_key: str,
    min_cells: int,
    mapping_info: dict,
    mapping_weights: torch.Tensor,
    n_genes: int,
) -> tuple[np.ndarray, np.ndarray]:
    groups = group_labels(adata, group_key)
    peak_idx = np.asarray(mapping_info["peak_idx"], dtype=np.int64)
    gene_idx = np.asarray(mapping_info["gene_idx"], dtype=np.int64)
    weights = mapping_weights.detach().cpu().numpy().astype(np.float32)
    mapping = sp.coo_matrix(
        (weights, (peak_idx, gene_idx)),
        shape=(adata.n_vars, n_genes),
        dtype=np.float32,
    ).tocsr()

    rows = []
    counts = []
    for label in labels:
        mask = groups.eq(label).to_numpy()
        n = int(mask.sum())
        counts.append(n)
        if n < min_cells:
            rows.append(np.full(n_genes, np.nan, dtype=np.float32))
            continue
        x = adata.X[mask]
        if not sp.issparse(x):
            x = sp.csr_matrix(np.asarray(x, dtype=np.float32))
        else:
            x = x.tocsr().astype(np.float32, copy=False)

        counts_peak = np.asarray(x.mean(axis=0)).ravel()[None, :]
        tf = counts_peak / (counts_peak.sum(axis=1, keepdims=True) + EPS)
        idf = np.log1p(tf.shape[0] / (tf.sum(axis=0, keepdims=True) + EPS))
        tfidf = tf * idf
        gene_score = np.log1p(tfidf @ mapping)
        rows.append(np.asarray(gene_score).ravel().astype(np.float32))
    return np.vstack(rows), np.asarray(counts, dtype=int)


def build_peak_gene_mapping(adata: ad.AnnData, mapping_info: dict, mapping_weights: torch.Tensor, n_genes: int) -> sp.csr_matrix:
    peak_idx = np.asarray(mapping_info["peak_idx"], dtype=np.int64)
    gene_idx = np.asarray(mapping_info["gene_idx"], dtype=np.int64)
    weights = mapping_weights.detach().cpu().numpy().astype(np.float32)
    return sp.coo_matrix(
        (weights, (peak_idx, gene_idx)),
        shape=(adata.n_vars, n_genes),
        dtype=np.float32,
    ).tocsr()


def atac_cell_gene_matrix(
    adata: ad.AnnData,
    mapping_info: dict,
    mapping_weights: torch.Tensor,
    gene_names: list[str],
) -> ad.AnnData:
    mapping = build_peak_gene_mapping(adata, mapping_info, mapping_weights, len(gene_names))
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


def rna_gene_space_adata(adata: ad.AnnData, gene_names: list[str]) -> ad.AnnData:
    if list(adata.var_names.astype(str)) == list(gene_names):
        return adata
    common = pd.Index(gene_names).intersection(pd.Index(adata.var_names.astype(str)))
    if len(common) != len(gene_names):
        raise ValueError(f"RNA batch has {len(common)}/{len(gene_names)} checkpoint genes")
    return adata[:, gene_names].copy()


def _pick_ranked_genes(df: pd.DataFrame, top_n: int, pval_th: float) -> set[str]:
    if df is None or df.empty:
        return set()
    use = df
    if "pvals_adj" in df.columns:
        sig = df[df["pvals_adj"] < pval_th]
        if len(sig) >= min(top_n, len(df)):
            use = sig
    if "logfoldchanges" in use.columns:
        use = use.sort_values("logfoldchanges", ascending=False)
    return set(use.head(top_n)["names"].astype(str))


def de_genes_by_group(
    adata: ad.AnnData,
    labels: list[str],
    group_key: str,
    min_cells: int,
    top_n: int,
    pval_th: float,
) -> dict[str, set[str]]:
    groups = group_labels(adata, group_key)
    counts = groups.value_counts()
    valid = [label for label in labels if int(counts.get(label, 0)) >= min_cells and (adata.n_obs - int(counts.get(label, 0))) >= min_cells]
    de = {label: set() for label in labels}
    if not valid:
        return de
    work = adata.copy()
    work.obs[group_key] = pd.Categorical(groups.to_numpy())
    sc.tl.rank_genes_groups(
        work,
        groupby=group_key,
        groups=valid,
        reference="rest",
        method="wilcoxon",
        use_raw=False,
        n_genes=min(top_n * 3, work.n_vars),
    )
    for label in valid:
        try:
            de[label] = _pick_ranked_genes(sc.get.rank_genes_groups_df(work, group=label), top_n, pval_th)
        except Exception:
            de[label] = set()
    return de


def pearson_corr_matrix(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = x - np.nanmean(x, axis=1, keepdims=True)
    y = y - np.nanmean(y, axis=1, keepdims=True)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    denom = np.sqrt((x * x).sum(axis=1, keepdims=True)) @ np.sqrt((y * y).sum(axis=1, keepdims=True)).T
    corr = (x @ y.T) / np.maximum(denom, EPS)
    return np.clip(np.nan_to_num(corr, nan=0.0, posinf=1.0, neginf=-1.0), -1.0, 1.0)


def select_variable_genes(rna_bulk: np.ndarray, atac_bulk: np.ndarray, n_top: int) -> np.ndarray:
    var = np.nanvar(rna_bulk, axis=0) + np.nanvar(atac_bulk, axis=0)
    keep_n = min(n_top, var.size)
    idx = np.argpartition(-np.nan_to_num(var, nan=0.0), keep_n - 1)[:keep_n]
    return np.sort(idx)


def deg_corr_matrix(
    atac_bulk: np.ndarray,
    rna_bulk: np.ndarray,
    atac_labels: list[str],
    rna_labels: list[str],
    atac_de: dict[str, set[str]],
    rna_de: dict[str, set[str]],
    gene_to_idx: dict[str, int],
    min_de_genes: int,
) -> tuple[np.ndarray, np.ndarray]:
    corr = np.full((len(atac_labels), len(rna_labels)), np.nan, dtype=np.float64)
    n_genes = np.zeros((len(atac_labels), len(rna_labels)), dtype=np.int64)
    for i, a_label in enumerate(atac_labels):
        for j, r_label in enumerate(rna_labels):
            genes = (atac_de.get(a_label, set()) | rna_de.get(r_label, set()))
            idx = [gene_to_idx[g] for g in genes if g in gene_to_idx]
            if len(idx) < min_de_genes:
                continue
            n_genes[i, j] = len(idx)
            corr[i, j] = pearson_corr_matrix(atac_bulk[i : i + 1, idx], rna_bulk[j : j + 1, idx])[0, 0]
    return corr, n_genes


def plot_heatmap(corr_df: pd.DataFrame, out: Path, annot: bool) -> None:
    width = max(10, 0.35 * corr_df.shape[1] + 4)
    height = max(8, 0.35 * corr_df.shape[0] + 3)
    fig, ax = plt.subplots(figsize=(width, height), constrained_layout=True)
    sns.heatmap(
        corr_df,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        center=0,
        annot=annot,
        fmt=".2f",
        linewidths=0.4,
        cbar_kws={"label": "Pearson r", "shrink": 0.7, "aspect": 24},
        ax=ax,
    )
    ax.set_title("ResonanSC cell-type pseudobulk correlation\nbatch-pair averaged ATAC gene score vs RNA expression")
    ax.set_xlabel("RNA cell_type")
    ax.set_ylabel("ATAC cell_type")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    fig.savefig(out / "celltype_avg_batch_corr.png", dpi=300)
    fig.savefig(out / "celltype_avg_batch_corr.pdf")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    summary_path = Path(args.summary)
    summary = json.loads(summary_path.read_text())
    checkpoint = Path(summary["stage_checkpoints"][args.checkpoint_stage])
    output = Path(args.output) if args.output else summary_path.parent / f"round4_{args.checkpoint_stage}_celltype_batch_avg_corr"
    if args.output is None and args.corr_mode == "deg":
        output = summary_path.parent / f"round4_{args.checkpoint_stage}_celltype_deg_batch_avg_corr"
    output.mkdir(parents=True, exist_ok=True)

    log(f"Reading checkpoint: {checkpoint}")
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    batch_names = list(ckpt.get("batch_names", summary["batch_names"]))
    gene_names = list(ckpt["gene_names"])
    n_genes = len(gene_names)
    rna_idx = [i for i, name in enumerate(batch_names) if str(name).startswith("rna")]
    atac_idx = [i for i, name in enumerate(batch_names) if str(name).startswith("atac")]

    log("Collecting cell_type labels...")
    rna_label_set = set()
    atac_label_set = set()
    for i in rna_idx:
        adata = read_batch(summary, i, batch_names[i])
        rna_label_set.update(group_labels(adata, args.group_key).unique())
    for i in atac_idx:
        adata = read_batch(summary, i, batch_names[i])
        atac_label_set.update(group_labels(adata, args.group_key).unique())
    rna_labels = sorted(rna_label_set)
    atac_labels = sorted(atac_label_set)
    log(f"RNA cell_types: {len(rna_labels)}; ATAC cell_types: {len(atac_labels)}")

    rna_bulks = {}
    rna_counts = {}
    rna_de = {}
    for i in rna_idx:
        adata = read_batch(summary, i, batch_names[i])
        adata = rna_gene_space_adata(adata, gene_names)
        bulk, counts = rna_celltype_bulk(adata, rna_labels, args.group_key, args.min_cells)
        rna_bulks[i] = bulk
        rna_counts[i] = counts
        if args.corr_mode == "deg":
            log(f"RNA DEG: {batch_names[i]}")
            rna_de[i] = de_genes_by_group(
                adata, rna_labels, args.group_key, args.min_cells, args.de_top_n, args.de_pval_th
            )

    atac_bulks = {}
    atac_counts = {}
    atac_de = {}
    for i in atac_idx:
        adata = read_batch(summary, i, batch_names[i])
        info = ckpt["mapping_init"][i]
        weights = ckpt["mapping_weights"][i]
        bulk, counts = atac_celltype_gene_bulk(
            adata, atac_labels, args.group_key, args.min_cells, info, weights, n_genes
        )
        atac_bulks[i] = bulk
        atac_counts[i] = counts
        if args.corr_mode == "deg":
            log(f"ATAC gene-space DEG: {batch_names[i]}")
            gene_adata = atac_cell_gene_matrix(adata, info, weights, gene_names)
            atac_de[i] = de_genes_by_group(
                gene_adata, atac_labels, args.group_key, args.min_cells, args.de_top_n, args.de_pval_th
            )

    corr_sum = np.zeros((len(atac_labels), len(rna_labels)), dtype=np.float64)
    corr_count = np.zeros((len(atac_labels), len(rna_labels)), dtype=np.int64)
    deg_gene_sum = np.zeros((len(atac_labels), len(rna_labels)), dtype=np.int64)
    gene_to_idx = {gene: idx for idx, gene in enumerate(gene_names)}
    for ai in atac_idx:
        for ri in rna_idx:
            if args.corr_mode == "deg":
                corr, n_deg = deg_corr_matrix(
                    atac_bulks[ai],
                    rna_bulks[ri],
                    atac_labels,
                    rna_labels,
                    atac_de[ai],
                    rna_de[ri],
                    gene_to_idx,
                    args.min_de_genes,
                )
                deg_gene_sum += n_deg
            else:
                keep = select_variable_genes(rna_bulks[ri], atac_bulks[ai], args.top_variable_genes)
                corr = pearson_corr_matrix(atac_bulks[ai][:, keep], rna_bulks[ri][:, keep])
            valid = (atac_counts[ai][:, None] >= args.min_cells) & (rna_counts[ri][None, :] >= args.min_cells)
            corr = np.where(valid, corr, np.nan)
            corr_sum += np.nan_to_num(corr, nan=0.0)
            corr_count += np.isfinite(corr)

    corr_avg = np.divide(
        corr_sum,
        np.maximum(corr_count, 1),
        out=np.full_like(corr_sum, np.nan, dtype=np.float64),
        where=corr_count > 0,
    )
    corr_df = pd.DataFrame(corr_avg, index=atac_labels, columns=rna_labels)
    count_df = pd.DataFrame(corr_count, index=atac_labels, columns=rna_labels)
    prefix = "celltype_deg" if args.corr_mode == "deg" else "celltype"
    corr_df.to_csv(output / f"{prefix}_avg_batch_corr.csv")
    count_df.to_csv(output / f"{prefix}_avg_batch_corr_pair_counts.csv")
    if args.corr_mode == "deg":
        avg_deg = np.divide(
            deg_gene_sum,
            np.maximum(corr_count, 1),
            out=np.zeros_like(deg_gene_sum, dtype=np.float64),
            where=corr_count > 0,
        )
        pd.DataFrame(avg_deg, index=atac_labels, columns=rna_labels).to_csv(
            output / "celltype_deg_avg_gene_counts.csv"
        )
    plot_heatmap(corr_df, output, args.annot)

    config = {
        "summary": str(summary_path),
        "checkpoint": str(checkpoint),
        "checkpoint_stage": args.checkpoint_stage,
        "group_key": args.group_key,
        "corr_mode": args.corr_mode,
        "top_variable_genes": args.top_variable_genes,
        "de_top_n": args.de_top_n,
        "de_pval_th": args.de_pval_th,
        "min_de_genes": args.min_de_genes,
        "min_cells": args.min_cells,
        "rna_batches": [batch_names[i] for i in rna_idx],
        "atac_batches": [batch_names[i] for i in atac_idx],
        "n_batch_pairs": len(rna_idx) * len(atac_idx),
        "rna_cell_types": rna_labels,
        "atac_cell_types": atac_labels,
    }
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    log(f"Saved to {output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Merge ResonanSC cell-type correlation blocks without recomputing DEG."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba-cache")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch


HELPER_PATH = Path(__file__).with_name("plot_round_summary_celltype_marker_weighted_corr.py")
spec = importlib.util.spec_from_file_location("celltype_marker_corr", HELPER_PATH)
helper = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(helper)

MERGE_MAP = {
    "Central memory CD8 T": "CD8 T",
    "Effector memory CD8 T": "CD8 T",
    "Naive CD8 T": "CD8 T",
    "Memory CD4 T": "CD4 T",
    "Naive CD4 T": "CD4 T",
    "Memory B": "B",
    "Naive B": "B",
}
LABEL_ORDER = ["CD8 T", "MAIT", "CD4 T", "Treg", "B", "Plasma cell", "Monocyte", "cDC", "NK"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary",
        default="outputs/result2/1(old)/training_round_4_summary.json",
    )
    parser.add_argument("--checkpoint-stage", default="learn_M", choices=["learn_P", "learn_mapping", "learn_M"])
    parser.add_argument(
        "--marker-weights",
        default="outputs/result2/1(old)/round4_learn_M_celltype_best_corr/both_marker_weights.csv",
    )
    parser.add_argument("--group-key", default="cell_type")
    parser.add_argument("--min-marker-genes", type=int, default=8)
    parser.add_argument("--min-cells", type=int, default=10)
    parser.add_argument("--output", default="outputs/result2/1(old)/round4_learn_M_merged_block_best_corr")
    return parser.parse_args()


def broad(label: str) -> str:
    return MERGE_MAP.get(label, label)


def load_broad_markers(path: Path, broad_labels: list[str]) -> dict[str, dict[str, float]]:
    df = pd.read_csv(path)
    accum: dict[str, dict[str, list[float]]] = {label: defaultdict(list) for label in broad_labels}
    for row in df.itertuples(index=False):
        label = broad(str(getattr(row, "cell_type")))
        if label not in accum:
            continue
        accum[label][str(getattr(row, "gene"))].append(float(getattr(row, "weight")))
    return {label: {gene: float(np.max(vals)) for gene, vals in genes.items()} for label, genes in accum.items()}


def corr_with_broad_reference_markers(
    atac_bulk: np.ndarray,
    rna_bulk: np.ndarray,
    labels: list[str],
    marker_weights: dict[str, dict[str, float]],
    gene_to_idx: dict[str, int],
    min_marker_genes: int,
) -> tuple[np.ndarray, np.ndarray]:
    atac_z = -helper.zscore_genes_across_labels(atac_bulk)
    rna_z = helper.zscore_genes_across_labels(rna_bulk)
    corr = np.full((len(labels), len(labels)), np.nan, dtype=np.float64)
    marker_counts = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for i in range(len(labels)):
        for j, ref_label in enumerate(labels):
            markers = marker_weights.get(broad(ref_label), {})
            genes = [gene for gene in markers if gene in gene_to_idx]
            if len(genes) < min_marker_genes:
                continue
            idx = np.asarray([gene_to_idx[g] for g in genes], dtype=int)
            weights = np.asarray([markers[g] for g in genes], dtype=np.float64)
            marker_counts[i, j] = len(idx)
            corr[i, j] = helper.weighted_corr(atac_z[i, idx], rna_z[j, idx], weights)
    return corr, marker_counts


def merge_matrix(df: pd.DataFrame, broad_labels: list[str]) -> pd.DataFrame:
    out = np.full((len(broad_labels), len(broad_labels)), np.nan, dtype=np.float64)
    row_broad = pd.Series([broad(x) for x in df.index], index=df.index)
    col_broad = pd.Series([broad(x) for x in df.columns], index=df.columns)
    for i, rb in enumerate(broad_labels):
        for j, cb in enumerate(broad_labels):
            block = df.loc[row_broad[row_broad == rb].index, col_broad[col_broad == cb].index].to_numpy(float)
            out[i, j] = np.nanmean(block)
    return pd.DataFrame(out, index=broad_labels, columns=broad_labels)


def plot_heatmap(corr_df: pd.DataFrame, out: Path) -> None:
    numeric = [str(i) for i in range(corr_df.shape[0])]
    plot_df = corr_df.copy()
    plot_df.index = numeric
    plot_df.columns = numeric
    fig, ax = plt.subplots(figsize=(7.2, 6.4), constrained_layout=True)
    sns.heatmap(
        plot_df,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        center=0,
        annot=False,
        linewidths=0.4,
        cbar_kws={"label": "Weighted Pearson r", "shrink": 0.75, "aspect": 24},
        ax=ax,
    )
    ax.set_title("Merged cell-type correlation")
    ax.set_xlabel("RNA cell type")
    ax.set_ylabel("ATAC cell type")
    plt.setp(ax.get_xticklabels(), rotation=0)
    plt.setp(ax.get_yticklabels(), rotation=0)
    fig.savefig(out / "merged_avg_corr.png", dpi=300)
    fig.savefig(out / "merged_avg_corr.pdf")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    summary_path = Path(args.summary)
    summary = json.loads(summary_path.read_text())
    checkpoint = Path(summary["stage_checkpoints"][args.checkpoint_stage])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    batch_names = list(ckpt.get("batch_names", summary["batch_names"]))
    gene_names = list(ckpt["gene_names"])
    gene_to_idx = {gene: idx for idx, gene in enumerate(gene_names)}
    rna_idx = [i for i, name in enumerate(batch_names) if str(name).startswith("rna")]
    atac_idx = [i for i, name in enumerate(batch_names) if str(name).startswith("atac")]
    labels = helper.valid_common_labels(summary, batch_names, rna_idx, atac_idx, args.group_key)
    broad_labels = [x for x in LABEL_ORDER if x in {broad(label) for label in labels}]
    marker_weights = load_broad_markers(Path(args.marker_weights), broad_labels)

    rna_bulks, rna_counts = {}, {}
    for batch_i in rna_idx:
        raw = helper.read_batch(summary, batch_i, batch_names[batch_i])
        raw = helper.rna_gene_space_adata(raw, gene_names)
        bulk, counts = helper.rna_bulk(raw, labels, args.group_key, args.min_cells)
        rna_bulks[batch_i] = bulk
        rna_counts[batch_i] = counts

    atac_bulks, atac_counts = {}, {}
    for batch_i in atac_idx:
        raw = helper.read_batch(summary, batch_i, batch_names[batch_i])
        gene_adata = helper.atac_cell_gene_adata(raw, ckpt["mapping_init"][batch_i], ckpt["mapping_weights"][batch_i], gene_names)
        bulk, counts = helper.gene_space_bulk_from_cell_matrix(gene_adata, labels, args.group_key, args.min_cells)
        atac_bulks[batch_i] = bulk
        atac_counts[batch_i] = counts

    corr_sum = np.zeros((len(labels), len(labels)), dtype=np.float64)
    corr_count = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for ai in atac_idx:
        for ri in rna_idx:
            corr, _ = corr_with_broad_reference_markers(
                atac_bulks[ai],
                rna_bulks[ri],
                labels,
                marker_weights,
                gene_to_idx,
                args.min_marker_genes,
            )
            valid = (atac_counts[ai][:, None] >= args.min_cells) & (rna_counts[ri][None, :] >= args.min_cells)
            corr = np.where(valid, corr, np.nan)
            corr_sum += np.nan_to_num(corr, nan=0.0)
            corr_count += np.isfinite(corr)

    full_corr = np.divide(corr_sum, np.maximum(corr_count, 1), out=np.full_like(corr_sum, np.nan), where=corr_count > 0)
    full_df = pd.DataFrame(full_corr, index=labels, columns=labels)
    merged_df = merge_matrix(full_df, broad_labels)
    full_df.to_csv(output / "full_subtype_corr_with_merged_markers.csv")
    merged_df.to_csv(output / "merged_avg_corr.csv")
    pd.DataFrame({"id": list(range(len(broad_labels))), "cell_type": broad_labels}).to_csv(output / "label_key.csv", index=False)
    pd.DataFrame(
        [{"cell_type": label, "gene": gene, "weight": weight} for label, genes in marker_weights.items() for gene, weight in genes.items()]
    ).to_csv(output / "merged_marker_weights.csv", index=False)
    plot_heatmap(merged_df, output)
    (output / "config.json").write_text(
        json.dumps(
            {
                "summary": str(summary_path),
                "checkpoint": str(checkpoint),
                "marker_weights": str(Path(args.marker_weights)),
                "labels": broad_labels,
                "merge_map": MERGE_MAP,
                "mode": "subtype_corr_then_block_average",
                "n_batch_pairs": len(rna_idx) * len(atac_idx),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Saved to {output}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Replot ResonanSC marker-weighted correlation after merging related labels.

This script reuses a saved marker-weight table and does not recompute DEG.
"""

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
import scipy.sparse as sp
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
    parser.add_argument("--flip-atac-sign", action="store_true", default=True)
    parser.add_argument(
        "--output",
        default="outputs/result2/1(old)/round4_learn_M_merged_celltype_best_corr",
    )
    return parser.parse_args()


def merged_groups(adata, group_key: str) -> pd.Series:
    groups = helper.collapsed_groups(adata, group_key)
    return groups.map(lambda x: MERGE_MAP.get(str(x), str(x)) if pd.notna(x) else pd.NA).astype("string")


def common_merged_labels(summary: dict, batch_names: list[str], rna_idx: list[int], atac_idx: list[int], group_key: str) -> list[str]:
    rna = set()
    atac = set()
    for i in rna_idx:
        groups = merged_groups(helper.read_batch(summary, i, batch_names[i]), group_key).dropna()
        rna.update(groups.unique().tolist())
    for i in atac_idx:
        groups = merged_groups(helper.read_batch(summary, i, batch_names[i]), group_key).dropna()
        atac.update(groups.unique().tolist())
    common = rna & atac
    ordered = [label for label in LABEL_ORDER if label in common]
    ordered.extend(sorted(common - set(ordered)))
    return ordered


def load_merged_marker_weights(path: Path, labels: list[str]) -> dict[str, dict[str, float]]:
    df = pd.read_csv(path)
    accum: dict[str, dict[str, list[float]]] = {label: defaultdict(list) for label in labels}
    for row in df.itertuples(index=False):
        old_label = str(getattr(row, "cell_type"))
        label = MERGE_MAP.get(old_label, old_label)
        if label not in accum:
            continue
        gene = str(getattr(row, "gene"))
        weight = float(getattr(row, "weight"))
        accum[label][gene].append(weight)
    return {
        label: {gene: float(np.max(weights)) for gene, weights in genes.items()}
        for label, genes in accum.items()
    }


def gene_space_rna_bulk(adata, labels: list[str], group_key: str, gene_names: list[str], min_cells: int) -> tuple[np.ndarray, np.ndarray]:
    adata = helper.rna_gene_space_adata(adata, gene_names)
    groups = merged_groups(adata, group_key)
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


def gene_space_atac_bulk(gene_adata, labels: list[str], group_key: str, min_cells: int) -> tuple[np.ndarray, np.ndarray]:
    groups = merged_groups(gene_adata, group_key)
    rows = []
    counts = []
    for label in labels:
        mask = groups.eq(label).fillna(False).to_numpy()
        n = int(mask.sum())
        counts.append(n)
        if n < min_cells:
            rows.append(np.full(gene_adata.n_vars, np.nan, dtype=np.float32))
            continue
        if sp.issparse(gene_adata.X):
            rows.append(np.asarray(gene_adata.X[mask].mean(axis=0)).ravel().astype(np.float32))
        else:
            rows.append(np.asarray(gene_adata.X[mask]).mean(axis=0).astype(np.float32))
    return np.vstack(rows), np.asarray(counts, dtype=int)


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
    labels = common_merged_labels(summary, batch_names, rna_idx, atac_idx, args.group_key)
    marker_weights = load_merged_marker_weights(Path(args.marker_weights), labels)

    pd.DataFrame({"id": list(range(len(labels))), "cell_type": labels}).to_csv(output / "label_key.csv", index=False)
    pd.DataFrame(
        [
            {"cell_type": label, "gene": gene, "weight": weight}
            for label, genes in marker_weights.items()
            for gene, weight in genes.items()
        ]
    ).to_csv(output / "merged_marker_weights.csv", index=False)

    rna_bulks, rna_counts = {}, {}
    for batch_i in rna_idx:
        raw = helper.read_batch(summary, batch_i, batch_names[batch_i])
        bulk, counts = gene_space_rna_bulk(raw, labels, args.group_key, gene_names, args.min_cells)
        rna_bulks[batch_i] = bulk
        rna_counts[batch_i] = counts

    atac_bulks, atac_counts = {}, {}
    for batch_i in atac_idx:
        raw = helper.read_batch(summary, batch_i, batch_names[batch_i])
        gene_adata = helper.atac_cell_gene_adata(raw, ckpt["mapping_init"][batch_i], ckpt["mapping_weights"][batch_i], gene_names)
        bulk, counts = gene_space_atac_bulk(gene_adata, labels, args.group_key, args.min_cells)
        atac_bulks[batch_i] = bulk
        atac_counts[batch_i] = counts

    corr_sum = np.zeros((len(labels), len(labels)), dtype=np.float64)
    corr_count = np.zeros((len(labels), len(labels)), dtype=np.int64)
    marker_sum = np.zeros((len(labels), len(labels)), dtype=np.float64)
    for ai in atac_idx:
        for ri in rna_idx:
            corr, marker_counts = helper.corr_from_reference_markers(
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
    corr_df.to_csv(output / "merged_avg_corr.csv")
    pd.DataFrame(corr_count, index=labels, columns=labels).to_csv(output / "merged_avg_corr_pair_counts.csv")
    pd.DataFrame(marker_avg, index=labels, columns=labels).to_csv(output / "merged_avg_corr_marker_counts.csv")
    plot_heatmap(corr_df, output)

    config = {
        "summary": str(summary_path),
        "checkpoint": str(checkpoint),
        "checkpoint_stage": args.checkpoint_stage,
        "marker_weights": str(Path(args.marker_weights)),
        "group_key": args.group_key,
        "min_marker_genes": args.min_marker_genes,
        "min_cells": args.min_cells,
        "flip_atac_sign": args.flip_atac_sign,
        "labels": labels,
        "merge_map": MERGE_MAP,
        "n_batch_pairs": len(rna_idx) * len(atac_idx),
    }
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"Saved to {output}", flush=True)


if __name__ == "__main__":
    main()

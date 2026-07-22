#!/usr/bin/env python3
"""Boxplot of cross-modal pseudobulk correlations by merged cell type."""

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
EPS = 1e-8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary",
        default="outputs/result2/1(old)/training_round_4_summary.json",
    )
    parser.add_argument("--checkpoint-stage", default="learn_M", choices=["learn_P", "learn_mapping", "learn_M"])
    parser.add_argument("--group-key", default="cell_type")
    parser.add_argument(
        "--marker-weights",
        default="outputs/result2/1(old)/round4_marker_grid/both_top50_flipyes/both_marker_weights.csv",
    )
    parser.add_argument("--corr-mode", default="marker", choices=["marker", "all_genes"])
    parser.add_argument("--min-marker-genes", type=int, default=8)
    parser.add_argument("--min-cells", type=int, default=10)
    parser.add_argument("--flip-atac-sign", action="store_true", default=True)
    parser.add_argument("--require-complete-boxes", action="store_true")
    parser.add_argument(
        "--output",
        default="outputs/result2/1(old)/round4_learn_M_merged_bulk_corr_boxplot",
    )
    return parser.parse_args()


def broad(label: str) -> str:
    return MERGE_MAP.get(label, label)


def merged_groups(adata, group_key: str) -> pd.Series:
    groups = helper.collapsed_groups(adata, group_key)
    return groups.map(lambda x: broad(str(x)) if pd.notna(x) else pd.NA).astype("string")


def common_merged_labels(summary: dict, batch_names: list[str], rna_idx: list[int], atac_idx: list[int], group_key: str) -> list[str]:
    rna = set()
    atac = set()
    for i in rna_idx:
        rna.update(merged_groups(helper.read_batch(summary, i, batch_names[i]), group_key).dropna().unique().tolist())
    for i in atac_idx:
        atac.update(merged_groups(helper.read_batch(summary, i, batch_names[i]), group_key).dropna().unique().tolist())
    common = rna & atac
    labels = [label for label in LABEL_ORDER if label in common]
    labels.extend(sorted(common - set(labels)))
    return labels


def load_marker_weights(path: Path, labels: list[str]) -> dict[str, dict[str, float]]:
    df = pd.read_csv(path)
    accum: dict[str, dict[str, list[float]]] = {label: defaultdict(list) for label in labels}
    for row in df.itertuples(index=False):
        label = broad(str(getattr(row, "cell_type")))
        if label not in accum:
            continue
        accum[label][str(getattr(row, "gene"))].append(float(getattr(row, "weight")))
    return {label: {gene: float(np.max(vals)) for gene, vals in genes.items()} for label, genes in accum.items()}


def rna_bulk(adata, labels: list[str], group_key: str, gene_names: list[str], min_cells: int) -> tuple[np.ndarray, np.ndarray]:
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


def atac_bulk(gene_adata, labels: list[str], group_key: str, min_cells: int) -> tuple[np.ndarray, np.ndarray]:
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


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt(np.sum(x * x) * np.sum(y * y))
    if denom <= EPS:
        return np.nan
    return float(np.clip(np.sum(x * y) / denom, -1.0, 1.0))


def corr_matrix(
    atac_mat: np.ndarray,
    rna_mat: np.ndarray,
    labels: list[str],
    corr_mode: str,
    marker_weights: dict[str, dict[str, float]],
    gene_to_idx: dict[str, int],
    min_marker_genes: int,
    flip_atac_sign: bool,
) -> np.ndarray:
    atac_z = helper.zscore_genes_across_labels(atac_mat)
    if flip_atac_sign:
        atac_z = -atac_z
    rna_z = helper.zscore_genes_across_labels(rna_mat)
    corr = np.full((len(labels), len(labels)), np.nan, dtype=np.float64)
    for i in range(len(labels)):
        for j, ref_label in enumerate(labels):
            if corr_mode == "all_genes":
                corr[i, j] = pearson(atac_z[i], rna_z[j])
                continue
            markers = marker_weights.get(ref_label, {})
            genes = [gene for gene in markers if gene in gene_to_idx]
            if len(genes) < min_marker_genes:
                continue
            idx = np.asarray([gene_to_idx[g] for g in genes], dtype=int)
            weights = np.asarray([markers[g] for g in genes], dtype=np.float64)
            corr[i, j] = helper.weighted_corr(atac_z[i, idx], rna_z[j, idx], weights)
    return corr


def plot_boxplot(df: pd.DataFrame, labels: list[str], output: Path, corr_mode: str) -> None:
    id_map = {label: str(i) for i, label in enumerate(labels)}
    plot_df = df.copy()
    plot_df["cell_type_id"] = plot_df["cell_type"].map(id_map)
    fig, ax = plt.subplots(figsize=(max(8.5, 0.85 * len(labels) + 2.5), 5.2), constrained_layout=True)
    sns.boxplot(
        data=plot_df,
        x="cell_type_id",
        y="correlation",
        hue="comparison",
        order=[str(i) for i in range(len(labels))],
        hue_order=["same", "different"],
        palette={"same": "#d94e45", "different": "#4878b7"},
        width=0.72,
        showfliers=False,
        ax=ax,
    )
    sns.stripplot(
        data=plot_df,
        x="cell_type_id",
        y="correlation",
        hue="comparison",
        order=[str(i) for i in range(len(labels))],
        hue_order=["same", "different"],
        dodge=True,
        palette={"same": "#8f1f1b", "different": "#1f4f8f"},
        size=2.2,
        alpha=0.55,
        linewidth=0,
        ax=ax,
    )
    handles, labels_ = ax.get_legend_handles_labels()
    ax.legend(handles[:2], labels_[:2], title="", frameon=False, loc="best")
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.55)
    ax.set_xlabel("Cell type")
    ax.set_ylabel("Cross-modal bulk correlation")
    ax.set_title(f"Same vs different cell-type bulk correlation ({corr_mode})")
    fig.savefig(output / f"bulk_corr_boxplot_{corr_mode}.png", dpi=300)
    fig.savefig(output / f"bulk_corr_boxplot_{corr_mode}.pdf")
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
    marker_weights = load_marker_weights(Path(args.marker_weights), labels) if args.corr_mode == "marker" else {}

    rna_bulks, rna_counts = {}, {}
    for batch_i in rna_idx:
        raw = helper.read_batch(summary, batch_i, batch_names[batch_i])
        bulk, counts = rna_bulk(raw, labels, args.group_key, gene_names, args.min_cells)
        rna_bulks[batch_i] = bulk
        rna_counts[batch_i] = counts

    atac_bulks, atac_counts = {}, {}
    for batch_i in atac_idx:
        raw = helper.read_batch(summary, batch_i, batch_names[batch_i])
        gene_adata = helper.atac_cell_gene_adata(raw, ckpt["mapping_init"][batch_i], ckpt["mapping_weights"][batch_i], gene_names)
        bulk, counts = atac_bulk(gene_adata, labels, args.group_key, args.min_cells)
        atac_bulks[batch_i] = bulk
        atac_counts[batch_i] = counts

    rows = []
    for ai in atac_idx:
        for ri in rna_idx:
            corr = corr_matrix(
                atac_bulks[ai],
                rna_bulks[ri],
                labels,
                args.corr_mode,
                marker_weights,
                gene_to_idx,
                args.min_marker_genes,
                args.flip_atac_sign,
            )
            valid = (atac_counts[ai][:, None] >= args.min_cells) & (rna_counts[ri][None, :] >= args.min_cells)
            corr = np.where(valid, corr, np.nan)
            for i, cell_type in enumerate(labels):
                same = corr[i, i]
                different = np.nanmean(np.delete(corr[i, :], i))
                rows.append(
                    {
                        "cell_type": cell_type,
                        "comparison": "same",
                        "correlation": same,
                        "atac_batch": batch_names[ai],
                        "rna_batch": batch_names[ri],
                    }
                )
                rows.append(
                    {
                        "cell_type": cell_type,
                        "comparison": "different",
                        "correlation": different,
                        "atac_batch": batch_names[ai],
                        "rna_batch": batch_names[ri],
                    }
                )

    expected_points = len(rna_idx) * len(atac_idx)
    df = pd.DataFrame(rows).dropna(subset=["correlation"])
    if args.require_complete_boxes:
        counts = df.groupby(["cell_type", "comparison"]).size().unstack(fill_value=0)
        keep_labels = [
            label
            for label in labels
            if label in counts.index
            and counts.loc[label].get("same", 0) == expected_points
            and counts.loc[label].get("different", 0) == expected_points
        ]
        df = df[df["cell_type"].isin(keep_labels)].copy()
        labels = keep_labels
    df.to_csv(output / f"bulk_corr_boxplot_points_{args.corr_mode}.csv", index=False)
    pd.DataFrame({"id": list(range(len(labels))), "cell_type": labels}).to_csv(output / "label_key.csv", index=False)
    plot_boxplot(df, labels, output, args.corr_mode)
    (output / f"config_{args.corr_mode}.json").write_text(
        json.dumps(
            {
                "summary": str(summary_path),
                "checkpoint": str(checkpoint),
                "corr_mode": args.corr_mode,
                "marker_weights": str(Path(args.marker_weights)) if args.corr_mode == "marker" else None,
                "group_key": args.group_key,
                "min_marker_genes": args.min_marker_genes,
                "min_cells": args.min_cells,
                "flip_atac_sign": args.flip_atac_sign,
                "labels": labels,
                "merge_map": MERGE_MAP,
                "points_per_box_expected": expected_points,
                "require_complete_boxes": args.require_complete_boxes,
                "different_definition": "mean correlation to all non-matching RNA cell types within the same batch pair",
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Saved to {output}", flush=True)


if __name__ == "__main__":
    main()

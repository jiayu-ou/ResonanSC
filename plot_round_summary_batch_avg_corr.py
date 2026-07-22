#!/usr/bin/env python3
"""Batch-pair averaged ATAC-vs-RNA bulk correlation from a ResonanSC round summary."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba-cache")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary",
        default="outputs/result2/1(old)/training_round_4_summary.json",
    )
    parser.add_argument("--checkpoint-stage", default="learn_M", choices=["learn_P", "learn_mapping", "learn_M"])
    parser.add_argument("--top-variable-genes", type=int, default=1000)
    parser.add_argument("--min-cells", type=int, default=10)
    parser.add_argument("--output", default=None)
    parser.add_argument("--annot", action="store_true")
    return parser.parse_args()


def log(msg: str) -> None:
    print(msg, flush=True)


def pearson_corr_matrix(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    # Inputs: K_atac x G and K_rna x G, returns K_atac x K_rna.
    x = x - x.mean(axis=1, keepdims=True)
    y = y - y.mean(axis=1, keepdims=True)
    denom = np.sqrt((x * x).sum(axis=1, keepdims=True)) @ np.sqrt((y * y).sum(axis=1, keepdims=True)).T
    corr = (x @ y.T) / np.maximum(denom, 1e-8)
    return np.clip(np.nan_to_num(corr, nan=0.0, posinf=1.0, neginf=-1.0), -1.0, 1.0)


def select_variable_genes(rna_bulk_gk: np.ndarray, atac_bulk_gk: np.ndarray, n_top: int) -> np.ndarray:
    var = np.var(rna_bulk_gk, axis=1) + np.var(atac_bulk_gk, axis=1)
    keep_n = min(n_top, var.size)
    idx = np.argpartition(-var, keep_n - 1)[:keep_n]
    return np.sort(idx)


def hard_counts(p: torch.Tensor) -> np.ndarray:
    labels = torch.as_tensor(p).detach().cpu().argmax(dim=1).numpy()
    return np.bincount(labels, minlength=int(p.shape[1]))


def plot_heatmap(corr_df: pd.DataFrame, out: Path, annot: bool) -> None:
    width = max(10, 0.34 * corr_df.shape[1] + 3)
    height = max(8, 0.32 * corr_df.shape[0] + 2)
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
    ax.set_title("ResonanSC batch-averaged ATAC vs RNA bulk correlation")
    ax.set_xlabel("RNA clusters")
    ax.set_ylabel("ATAC clusters")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    fig.savefig(out / "avg_batch_corr.png", dpi=300)
    fig.savefig(out / "avg_batch_corr.pdf")
    plt.close(fig)


def plot_diagonal(corr_df: pd.DataFrame, out: Path) -> None:
    common = [x for x in corr_df.index if x in corr_df.columns]
    if not common:
        return
    diag = pd.DataFrame({"cluster": common, "pearson": [corr_df.loc[x, x] for x in common]})
    diag = diag.dropna().sort_values("pearson", ascending=False)
    fig, ax = plt.subplots(figsize=(max(8, 0.35 * len(diag)), 4.5), constrained_layout=True)
    sns.barplot(data=diag, x="cluster", y="pearson", color="#4C78A8", ax=ax)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylim(-1, 1)
    ax.set_xlabel("ResonanSC cluster")
    ax.set_ylabel("ATAC vs RNA Pearson r")
    ax.set_title("Same-cluster batch-averaged correlation")
    plt.setp(ax.get_xticklabels(), rotation=90)
    fig.savefig(out / "same_cluster_corr.png", dpi=300)
    fig.savefig(out / "same_cluster_corr.pdf")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    summary_path = Path(args.summary)
    summary = json.loads(summary_path.read_text())
    checkpoint = Path(summary["stage_checkpoints"][args.checkpoint_stage])
    output = Path(args.output) if args.output else summary_path.parent / f"round4_{args.checkpoint_stage}_batch_avg_corr"
    output.mkdir(parents=True, exist_ok=True)

    log(f"Reading checkpoint: {checkpoint}")
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    batch_names = list(ckpt.get("batch_names", summary["batch_names"]))
    bulks = [torch.as_tensor(x).detach().cpu().float().numpy() for x in ckpt["bulks_list"]]
    probs = ckpt.get("P_align", ckpt.get("probs_soft"))
    if probs is None:
        raise KeyError("Checkpoint must contain P_align or probs_soft for batch-specific cluster counts.")

    rna_idx = [i for i, name in enumerate(batch_names) if str(name).startswith("rna")]
    atac_idx = [i for i, name in enumerate(batch_names) if str(name).startswith("atac")]
    if not rna_idx or not atac_idx:
        raise RuntimeError("Could not infer RNA/ATAC batches from batch_names.")

    cluster_count = int(ckpt.get("bulknum", bulks[0].shape[1]))
    labels = [str(i) for i in range(cluster_count)]
    corr_sum = np.zeros((cluster_count, cluster_count), dtype=np.float64)
    corr_count = np.zeros((cluster_count, cluster_count), dtype=np.int64)

    counts_by_batch = [hard_counts(p) for p in probs]
    log(f"RNA batches: {len(rna_idx)}; ATAC batches: {len(atac_idx)}; clusters: {cluster_count}")

    for ai in atac_idx:
        atac_bulk = np.nan_to_num(bulks[ai], nan=0.0, posinf=0.0, neginf=0.0)  # G x K
        atac_counts = counts_by_batch[ai]
        for ri in rna_idx:
            rna_bulk = np.nan_to_num(bulks[ri], nan=0.0, posinf=0.0, neginf=0.0)  # G x K
            rna_counts = counts_by_batch[ri]
            keep = select_variable_genes(rna_bulk, atac_bulk, args.top_variable_genes)
            corr = pearson_corr_matrix(atac_bulk[keep, :].T, rna_bulk[keep, :].T)
            valid = (atac_counts[:, None] >= args.min_cells) & (rna_counts[None, :] >= args.min_cells)
            corr = np.where(valid, corr, np.nan)
            corr_sum += np.nan_to_num(corr, nan=0.0)
            corr_count += np.isfinite(corr)

    corr_avg = np.divide(
        corr_sum,
        np.maximum(corr_count, 1),
        out=np.full_like(corr_sum, np.nan, dtype=np.float64),
        where=corr_count > 0,
    )
    corr_df = pd.DataFrame(corr_avg, index=labels, columns=labels)
    count_df = pd.DataFrame(corr_count, index=labels, columns=labels)

    corr_df.to_csv(output / "avg_batch_corr.csv")
    count_df.to_csv(output / "avg_batch_corr_pair_counts.csv")
    plot_heatmap(corr_df, output, args.annot)
    plot_diagonal(corr_df, output)

    config = {
        "summary": str(summary_path),
        "checkpoint": str(checkpoint),
        "checkpoint_stage": args.checkpoint_stage,
        "top_variable_genes": args.top_variable_genes,
        "min_cells": args.min_cells,
        "batch_names": batch_names,
        "rna_batches": [batch_names[i] for i in rna_idx],
        "atac_batches": [batch_names[i] for i in atac_idx],
        "n_batch_pairs": len(rna_idx) * len(atac_idx),
        "cluster_count": cluster_count,
    }
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    log(f"Saved to {output}")


if __name__ == "__main__":
    main()

from itertools import combinations
from math import ceil
from pathlib import Path
from typing import Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.sparse as sp
import torch
from scipy.stats import pearsonr, spearmanr


EPS = 1e-8


def _numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def mapping_edge_table(train_out) -> pd.DataFrame:
    """Return one row per trainable peak-gene edge."""
    rows = []
    mapping_init = train_out["mapping_init"]
    mapping_weights = train_out["mapping_weights"]

    for batch_id in sorted(mapping_weights):
        info = mapping_init[batch_id]
        peak_idx = _numpy(info["peak_idx"]).astype(np.int64)
        gene_idx = _numpy(info["gene_idx"]).astype(np.int64)
        distance = _numpy(info["distance"]).astype(np.float64)
        weight = _numpy(mapping_weights[batch_id]).astype(np.float64)
        init_weight = _numpy(info["init_weight"]).astype(np.float64)
        peak_names = np.asarray(info["peak_names"], dtype=object)
        gene_names = np.asarray(info["gene_names"], dtype=object)

        rows.append(
            pd.DataFrame(
                {
                    "batch_id": int(batch_id),
                    "batch": str(info.get("batch_name", batch_id)),
                    "peak_idx": peak_idx,
                    "gene_idx": gene_idx,
                    "peak": peak_names[peak_idx],
                    "gene": gene_names[gene_idx],
                    "distance": distance,
                    "init_weight": init_weight,
                    "weight": weight,
                }
            )
        )

    if not rows:
        raise ValueError("train_out does not contain any ATAC mapping weights.")
    return pd.concat(rows, ignore_index=True)


def save_train_mapping(
    train_out,
    output_dir,
    prefix: str = "mapping",
):
    """Save learned peak-gene mappings as tables and a lossless torch file.

    The compressed edge table is intended for inspection with pandas. The
    torch file preserves the original sparse mapping tensors for later reuse.
    """
    required = {"mapping_init", "mapping_weights"}
    missing = sorted(required.difference(train_out))
    if missing:
        raise KeyError(f"train_out is missing required keys: {missing}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    table_input = dict(train_out)
    table_input["mapping_init"] = train_out.get(
        "mapping_init_before_training",
        train_out["mapping_init"],
    )
    edges = mapping_edge_table(table_input)
    edges["weight_delta"] = edges["weight"] - edges["init_weight"]
    edges["weight_fold"] = edges["weight"] / (edges["init_weight"] + EPS)
    edges["gene_rank_for_peak"] = (
        edges.groupby(["batch_id", "peak_idx"])["weight"]
        .rank(method="first", ascending=False)
        .astype(np.int64)
    )
    edges = edges.sort_values(
        ["batch_id", "peak_idx", "gene_rank_for_peak"],
        ignore_index=True,
    )

    gene_totals = mapping_gene_totals(train_out, normalize=False).reset_index()
    batch_summary = (
        edges.groupby(["batch_id", "batch"], as_index=False)
        .agg(
            n_edges=("weight", "size"),
            n_peaks=("peak_idx", "nunique"),
            n_genes=("gene_idx", "nunique"),
            init_weight_mean=("init_weight", "mean"),
            learned_weight_mean=("weight", "mean"),
            learned_weight_median=("weight", "median"),
            learned_weight_sum=("weight", "sum"),
        )
    )

    paths = {
        "edges": output_dir / f"{prefix}_edges.csv.gz",
        "gene_totals": output_dir / f"{prefix}_gene_totals.csv",
        "batch_summary": output_dir / f"{prefix}_batch_summary.csv",
        "torch": output_dir / f"{prefix}.pt",
    }
    edges.to_csv(paths["edges"], index=False, compression="gzip")
    gene_totals.to_csv(paths["gene_totals"], index=False)
    batch_summary.to_csv(paths["batch_summary"], index=False)

    torch.save(
        {
            "mapping_init": train_out["mapping_init"],
            "mapping_init_before_training": train_out.get(
                "mapping_init_before_training"
            ),
            "mapping_weights": train_out["mapping_weights"],
            "gene_names": train_out.get("gene_names"),
            "batch_names": train_out.get("batch_names"),
            "source_train_checkpoint": train_out.get("source_train_checkpoint"),
        },
        paths["torch"],
    )

    print(f"[mapping] saved {len(edges):,} edges to {paths['edges']}")
    return {
        "paths": paths,
        "edges": edges,
        "gene_totals": gene_totals,
        "batch_summary": batch_summary,
    }


def mapping_gene_totals(train_out, normalize: bool = True) -> pd.DataFrame:
    """Aggregate mapping weights to globally normalized gene totals per batch."""
    gene_names = list(map(str, train_out["gene_names"]))
    result = pd.DataFrame(index=pd.Index(gene_names, name="gene"))

    for batch_id in sorted(train_out["mapping_weights"]):
        info = train_out["mapping_init"][batch_id]
        gene_idx = _numpy(info["gene_idx"]).astype(np.int64)
        weights = _numpy(train_out["mapping_weights"][batch_id]).astype(np.float64)
        totals = np.bincount(gene_idx, weights=weights, minlength=len(gene_names))
        if normalize:
            totals = totals / (totals.sum() + EPS)
        result[str(info.get("batch_name", batch_id))] = totals

    return result


def plot_mapping_distance(
    train_out,
    max_points_per_batch: int = 30_000,
    random_state: int = 0,
    figsize=(7, 5),
):
    """Plot learned mapping weight against peak-gene distance."""
    edges = mapping_edge_table(train_out)
    sampled = pd.concat(
        [
            group.sample(
                min(len(group), max_points_per_batch),
                random_state=random_state,
            )
            for _, group in edges.groupby("batch", sort=False)
        ],
        ignore_index=True,
    )

    fig, ax = plt.subplots(figsize=figsize)
    sns.scatterplot(
        data=sampled,
        x="distance",
        y="weight",
        hue="batch",
        alpha=0.25,
        s=10,
        linewidth=0,
        ax=ax,
    )
    ax.set_xlabel("Peak-gene distance (bp)")
    ax.set_ylabel("Learned mapping weight")
    ax.set_title("Peak-gene distance versus mapping weight")
    fig.tight_layout()
    return fig, ax, edges


def plot_mapping_batch_scatter(train_out, figsize_per_panel=(4.5, 4.5)):
    """Compare normalized per-gene incoming mapping weight between batches."""
    totals = mapping_gene_totals(train_out, normalize=True)
    pairs = list(combinations(totals.columns, 2))
    if not pairs:
        raise ValueError("At least two ATAC batches are required for comparison.")

    ncols = min(3, len(pairs))
    nrows = ceil(len(pairs) / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
        squeeze=False,
    )

    stats = []
    for ax, (left, right) in zip(axes.flat, pairs):
        x = totals[left].to_numpy()
        y = totals[right].to_numpy()
        pearson = pearsonr(x, y).statistic
        spearman = spearmanr(x, y).statistic
        ax.scatter(x, y, s=8, alpha=0.35)
        limit = max(float(x.max()), float(y.max())) * 1.03
        ax.plot([0, limit], [0, limit], color="black", linewidth=1, linestyle="--")
        ax.set(xlabel=left, ylabel=right)
        ax.set_title(f"Pearson={pearson:.3f}, Spearman={spearman:.3f}")
        stats.append(
            {
                "batch_1": left,
                "batch_2": right,
                "pearson": pearson,
                "spearman": spearman,
            }
        )

    for ax in axes.flat[len(pairs):]:
        ax.axis("off")
    fig.suptitle("Normalized incoming mapping weight per gene", y=1.02)
    fig.tight_layout()
    return fig, axes, totals, pd.DataFrame(stats)


def plot_mapping_gene_heatmap(
    train_out,
    genes: Optional[Sequence[str]] = None,
    top_n: int = 40,
    selection: str = "variable",
    row_zscore: bool = True,
    figsize=(7, 10),
):
    """Plot gene-by-ATAC-batch total mapping weights."""
    totals = mapping_gene_totals(train_out, normalize=True)
    if totals.shape[1] == 1:
        return 0
    
    if genes is not None:
        selected = [str(g) for g in genes if str(g) in totals.index]
        if not selected:
            raise ValueError("None of the requested genes occur in gene_names.")
        values = totals.loc[selected]
    else:
        if selection == "variable" and totals.shape[1] > 1:
            score = totals.var(axis=1)
        elif selection == "high":
            score = totals.mean(axis=1)
        else:
            raise ValueError("selection must be 'variable' or 'high'.")
        values = totals.loc[score.nlargest(min(top_n, len(score))).index]

    plotted = values.copy()
    if row_zscore:
        plotted = plotted.sub(plotted.mean(axis=1), axis=0)
        plotted = plotted.div(plotted.std(axis=1).replace(0, 1), axis=0)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        plotted,
        cmap="vlag" if row_zscore else "viridis",
        center=0 if row_zscore else None,
        ax=ax,
    )
    ax.set_title("Gene-level incoming mapping weight")
    ax.set_xlabel("ATAC batch")
    ax.set_ylabel("Gene")
    fig.tight_layout()
    return fig, ax, values


def _atac_tfidf(adata, probs):
    probs = _numpy(probs).astype(np.float64)
    probs = probs / (probs.sum(axis=0, keepdims=True) + EPS)
    if sp.issparse(adata.X):
        peak_bulk = adata.X.T @ probs
    else:
        peak_bulk = np.asarray(adata.X).T @ probs
    counts = np.asarray(peak_bulk).T
    tf = counts / (counts.sum(axis=1, keepdims=True) + EPS)
    idf = np.log1p(counts.shape[0] / (tf.sum(axis=0, keepdims=True) + EPS))
    return tf * idf


def effective_gene_activity(
    train_out,
    multidata: Mapping,
    batch_key: str,
):
    """Calculate cell-type-specific TF-IDF x mapping gene activity."""
    gene_names = list(map(str, train_out["gene_names"]))
    activities = {}
    tfidf_by_batch = {}

    for batch_id in sorted(train_out["mapping_weights"]):
        adata = multidata[batch_id]
        batch_name = str(adata.obs[batch_key].iloc[0])
        tfidf = _atac_tfidf(adata, train_out["probs_soft"][batch_id])
        info = train_out["mapping_init"][batch_id]
        peak_idx = _numpy(info["peak_idx"]).astype(np.int64)
        gene_idx = _numpy(info["gene_idx"]).astype(np.int64)
        weights = _numpy(train_out["mapping_weights"][batch_id]).astype(np.float64)
        mapping = sp.coo_matrix(
            (weights, (peak_idx, gene_idx)),
            shape=(adata.n_vars, len(gene_names)),
        ).tocsr()
        activities[batch_name] = np.log1p(tfidf @ mapping)
        tfidf_by_batch[int(batch_id)] = tfidf

    return activities, tfidf_by_batch


def plot_effective_activity_heatmap(
    train_out,
    multidata: Mapping,
    batch_key: str,
    genes: Optional[Sequence[str]] = None,
    top_n: int = 40,
    figsize=(13, 9),
):
    """Plot TF-IDF x W activity for each ATAC batch and cell type."""
    activities, tfidf_by_batch = effective_gene_activity(
        train_out, multidata, batch_key
    )
    gene_names = np.asarray(train_out["gene_names"], dtype=object)
    combined = np.concatenate(list(activities.values()), axis=0)

    if genes is None:
        gene_idx = np.argsort(combined.var(axis=0))[-min(top_n, combined.shape[1]):]
    else:
        lookup = {str(g): i for i, g in enumerate(gene_names)}
        gene_idx = np.asarray([lookup[str(g)] for g in genes if str(g) in lookup])
        if gene_idx.size == 0:
            raise ValueError("None of the requested genes occur in gene_names.")

    columns = []
    matrices = []
    for batch_name, activity in activities.items():
        matrices.append(activity[:, gene_idx].T)
        columns.extend(
            [f"{batch_name} | type {k}" for k in range(activity.shape[0])]
        )
    values = pd.DataFrame(
        np.concatenate(matrices, axis=1),
        index=gene_names[gene_idx],
        columns=columns,
    )

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(values, cmap="mako", ax=ax)
    ax.set_title("Cell-type-specific effective gene activity (TF-IDF x W)")
    ax.set_xlabel("ATAC batch and cell type")
    ax.set_ylabel("Gene")
    fig.tight_layout()
    return fig, ax, values, activities, tfidf_by_batch


def plot_gene_mapping_tracks(
    train_out,
    multidata: Mapping,
    batch_key: str,
    gene: str,
    top_n: int = 30,
    figsize_per_batch=(10, 3.5),
):
    """Plot strongest peak edges and mean effective contribution for one gene."""
    edges = mapping_edge_table(train_out)
    gene_edges = edges[edges["gene"] == str(gene)].copy()
    if gene_edges.empty:
        raise ValueError(f"No candidate mapping edges found for gene {gene!r}.")

    _, tfidf_by_batch = effective_gene_activity(train_out, multidata, batch_key)
    batches = sorted(gene_edges["batch_id"].unique())
    fig, axes = plt.subplots(
        len(batches),
        1,
        figsize=(figsize_per_batch[0], figsize_per_batch[1] * len(batches)),
        squeeze=False,
    )

    selected_rows = []
    for ax, batch_id in zip(axes.flat, batches):
        current = gene_edges[gene_edges["batch_id"] == batch_id].copy()
        current["mean_tfidf"] = tfidf_by_batch[batch_id][:, current["peak_idx"]].mean(
            axis=0
        )
        current["effective_contribution"] = (
            current["mean_tfidf"] * current["weight"]
        )
        current = current.nlargest(min(top_n, len(current)), "effective_contribution")
        current = current.sort_values("effective_contribution")
        selected_rows.append(current)

        labels = current["peak"].astype(str)
        y = np.arange(len(current))
        ax.hlines(y, 0, current["effective_contribution"], color="0.75")
        ax.scatter(
            current["effective_contribution"],
            y,
            c=current["distance"],
            cmap="viridis",
            s=35,
        )
        ax.set_yticks(y, labels=labels)
        ax.set_xlabel("Mean TF-IDF x mapping weight")
        ax.set_title(f"{current['batch'].iloc[0]}: peaks linked to {gene}")

    fig.tight_layout()
    return fig, axes, pd.concat(selected_rows, ignore_index=True)


def plot_mapping_diagnostics(
    train_out,
    multidata: Mapping,
    batch_key: str,
    top_n_genes: int = 40,
):
    """Run the four general mapping diagnostics used after training."""
    distance = plot_mapping_distance(train_out)
    # heatmap = plot_mapping_gene_heatmap(train_out, top_n=top_n_genes)
    
    heatmap = plot_mapping_gene_heatmap(
        train_out=train_out,
        genes=None,
        top_n=top_n_genes,
        selection="variable"
    )
    effective = plot_effective_activity_heatmap(
        train_out,
        multidata,
        batch_key,
        top_n=top_n_genes,
    )

    batch_scatter = None
    if len(train_out["mapping_weights"]) > 1:
        batch_scatter = plot_mapping_batch_scatter(train_out)

    return {
        "distance": distance,
        "batch_scatter": batch_scatter,
        "gene_heatmap": heatmap,
        "effective_activity": effective,
    }

"""Bridge pooled modality alignment to arbitrary-batch ResonanSC peak training.

The saved alignment has two pooled modality tensors rather than per-training
batches, so this reconstructs per-batch P from its saved cell labels. ATAC
training uses the original peak-count matrix, never gene activity as input.
"""

import argparse
import gzip
import json
from pathlib import Path
import re
import sys



import anndata as ad
import numpy as np
np.NaN = np.nan  # episcanpy in the torch_gpu environment expects this legacy alias.
import pandas as pd
from scipy import sparse
import torch

from .core.corr import marker_init_from_1vall
from .core.mapping import build_peak_gene_mapping_init


def subset_backed_rows(matrix, indices):
    from .data import rows
    return rows(matrix, indices)


def filter_gtf(gtf_gz, gene_names, output):
    wanted = set(gene_names)
    seen = set()
    pattern = re.compile(r'gene_name "([^"]+)"')
    with (gzip.open(gtf_gz, "rt") if str(gtf_gz).endswith(".gz") else open(gtf_gz)) as source, output.open("w") as target:
        for line in source:
            if line.startswith("#"):
                continue
            fields = line.split("\t", 3)
            if len(fields) < 4 or fields[2] != "gene":
                continue
            hit = pattern.search(line)
            if hit and hit.group(1) in wanted:
                target.write(line)
                seen.add(hit.group(1))
    print(f"GTF: matched {len(seen)}/{len(wanted)} genes", flush=True)
    if len(seen) < 0.90 * len(wanted):
        raise ValueError("Too many selected genes are absent from the provided annotation")
    return seen


def prepare(cfg, alignment_dir=None):
    from types import SimpleNamespace
    from .data import validate, require_counts
    validate(cfg, require_gtf=True)
    root = Path(cfg['output_dir'])
    alignment_dir = Path(alignment_dir) if alignment_dir else root / 'alignment'
    args = SimpleNamespace(source=Path(cfg['input']['expression_h5ad']),
        labels=alignment_dir / 'cell_labels.csv', alignment=alignment_dir / 'alignment_checkpoint.pt',
        peak_matrix=Path(cfg['input']['peak_h5ad']), gtf_gz=Path(cfg['input']['gtf']),
        output_dir=root / 'prepared', min_peak_cells=cfg['preparation']['min_peak_cells'],
        max_distance=cfg['preparation']['max_distance'])
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    training_dir = args.output_dir / "training_multidata"
    training_dir.mkdir(exist_ok=True)

    saved = torch.load(args.alignment, map_location="cpu", weights_only=False)
    labels = pd.read_csv(args.labels, dtype=str)
    if labels["cell_barcode"].tolist() != saved["cell_names"]:
        raise ValueError("Saved alignment/checkpoint cell order differs")
    gene_names = list(map(str, saved["gene_names"]))
    batch_names = labels["batch"].drop_duplicates().tolist()
    expected_order = [b for b in batch_names if b.startswith('rna_')] + [b for b in batch_names if b.startswith('atac_')]
    if batch_names != expected_order:
        raise ValueError('Alignment batches must be ordered RNA then ATAC')
    expected_cells = sum([labels.loc[labels['batch'].eq(b), 'cell_barcode'].tolist() for b in batch_names], [])
    if expected_cells != labels['cell_barcode'].tolist() or labels['cell_barcode'].duplicated().any():
        raise ValueError('Cells must be unique and grouped by batch')
    final = torch.cat(saved['p_final']).argmax(1).numpy()
    if not np.array_equal(final, labels['final_alignment'].astype(int).to_numpy()):
        raise ValueError('Checkpoint and exported cluster assignments differ')
    cluster = labels["final_alignment"].astype(int).to_numpy()
    K = int(cluster.max()) + 1
    if sorted(np.unique(cluster).tolist()) != list(range(K)):
        raise ValueError("Final cluster IDs must be contiguous zero-based integers")

    source = ad.read_h5ad(args.source, backed="r")
    source_index = source.obs_names.get_indexer(labels["cell_barcode"])
    if (source_index < 0).any():
        raise ValueError("Source and saved labels do not contain the same cells")
    if not np.array_equal(source.obs[cfg["input"]["batch_key"]].astype(str).to_numpy()[source_index],
                          labels["batch"].to_numpy()):
        raise ValueError("Batch labels differ from source")
    order = np.argsort(source_index)
    gene_activity = source[source_index[order], gene_names].to_memory()[np.argsort(order)].copy()
    source.file.close()
    metadata = {"batch": labels["batch"].to_numpy(),
                "final_alignment": labels["final_alignment"].to_numpy()}
    condition_keys = [key for key in ("condition_group", "condition") if key in labels]
    for key in condition_keys:
        metadata[key] = labels[key].to_numpy()
    gene_activity.obs = pd.DataFrame(metadata, index=gene_activity.obs_names)
    gene_activity.write_h5ad(args.output_dir / "aligned_gene_activity.h5ad", compression="gzip")

    P_onehot = []
    for i, batch_name in enumerate(batch_names):
        selected = labels["batch"].eq(batch_name).to_numpy()
        p = torch.nn.functional.one_hot(torch.from_numpy(cluster[selected]), K).float()
        P_onehot.append(p)
    # BulkProjecting applies softmax to P_init unless one_hot_start is enabled.
    # Feed logits, not 0/1 probabilities, so the saved alignment is preserved.
    # With many global slots, -3 leaves enough softmax mass in every empty
    # column to exceed the model's per-batch presence threshold.  This makes
    # all K slots active and inflates cross-batch correlation memory.
    P_logits = [torch.where(p.bool(), 3.0, -5.0) for p in P_onehot]
    align_masks = {
        "batch_keys": batch_names, "prototype_count": K, "bulknum": K,
        "slot2proto": torch.arange(K, dtype=torch.long),
        "counts_per_batch_proto": torch.stack([(p.sum(0) > 0).long() for p in P_onehot]),
        "maxc_per_proto": torch.ones(K, dtype=torch.long),
    }
    print(f"Initializing M from aligned gene-activity clusters: {gene_activity.shape}, K={K}", flush=True)
    M_init = marker_init_from_1vall(gene_activity, torch.cat(P_onehot),
                                   n_top=cfg["preparation"]["marker_top_n"], min_cells_1vAll=cfg["preparation"]["marker_min_cells"])
    if M_init.shape != (len(gene_names), K):
        raise ValueError("M_init shape differs from alignment")
    print("M_init selected genes per cluster:", (M_init > .05).sum(0).tolist(), flush=True)

    mini_gtf = args.output_dir / "selected_genes.gtf"
    matched_genes = filter_gtf(args.gtf_gz, gene_names, mini_gtf)
    peak_source = ad.read_h5ad(args.peak_matrix, backed="r")
    if not peak_source.obs_names.is_unique or not peak_source.var_names.is_unique:
        raise ValueError("Peak source must have unique cells and peaks")
    peak_names = peak_source.var_names.astype(str).to_numpy()
    training = {}
    for i, batch_name in enumerate(batch_names):
        use = labels["batch"].eq(batch_name).to_numpy()
        names = labels.loc[use, "cell_barcode"].to_numpy()
        if batch_name.startswith("rna_"):
            batch = gene_activity[use].copy()
            batch.obs = batch.obs[["batch"] + condition_keys].copy()
        else:
            idx = peak_source.obs_names.get_indexer(names)
            if (idx < 0).any():
                raise ValueError(f"{batch_name}: {(idx < 0).sum()} cells missing from peak counts")
            counts = subset_backed_rows(peak_source.X, idx)
            require_counts(counts)
            if not sparse.issparse(counts) or counts.nnz == 0:
                raise ValueError(f"{batch_name}: empty peak count matrix")
            prevalence = np.asarray((counts > 0).sum(axis=0)).ravel()
            keep = (prevalence >= args.min_peak_cells) & (prevalence <= .95 * len(names))
            counts = counts[:, keep].tocsr().astype(np.float32)
            atac_obs = {"batch": batch_name}
            for key in condition_keys:
                atac_obs[key] = labels.loc[use, key].to_numpy()
            batch = ad.AnnData(X=counts,
                               obs=pd.DataFrame(atac_obs, index=names),
                               var=pd.DataFrame(index=peak_names[keep]))
            print(f"{batch_name}: {len(names)} cells, {batch.n_vars} prevalent peaks", flush=True)
        training[i] = batch
    peak_source.file.close()

    mapping_init = build_peak_gene_mapping_init(
        training, gene_names, mini_gtf, batch_key="batch",
        max_distance=args.max_distance, distance_scale=cfg["preparation"]["distance_scale"])
    for i, batch_name in enumerate(batch_names):
        if not batch_name.startswith("atac_"):
            continue
        info = mapping_init[i]
        used = np.unique(info["peak_idx"].numpy())
        if len(used) == 0:
            raise ValueError(f"{batch_names[i]}: no peaks linked to selected genes")
        old = info["peak_idx"].numpy()
        info["peak_idx"] = torch.from_numpy(np.searchsorted(used, old).astype(np.int64))
        training[i] = training[i][:, used].copy()
        info["peak_names"] = training[i].var_names.astype(str).tolist()
        print(f"{batch_names[i]}: {training[i].n_vars} mapped peaks, "
              f"{len(info['peak_idx'])} candidate edges", flush=True)

    for i, batch in training.items():
        path = training_dir / f"{i}_{batch_names[i]}.h5ad"
        batch.write_h5ad(path, compression="gzip")
        if batch.obs_names.astype(str).tolist() != labels.loc[
                labels["batch"].eq(batch_names[i]), "cell_barcode"].tolist():
            raise ValueError(f"Training cell order differs for {batch_names[i]}")
        print(f"Saved {path} {batch.shape}", flush=True)

    checkpoint_path = args.output_dir / "init_for_training.pt"
    torch.save({
        "P_align": P_logits, "P_align_onehot": P_onehot,
        "M_align": M_init, "M_init": M_init,
        "align_masks": align_masks, "mapping_init": mapping_init,
        "gene_names": gene_names, "batch_key": "batch", "batch_names": batch_names,
        "training_data_dir": str(training_dir.resolve()),
        "dap_deg_window": args.max_distance,
        "source_alignment_checkpoint": str(args.alignment.resolve()),
        "source_alignment_labels": str(args.labels.resolve()),
        "condition_groups": saved.get("condition_groups"),
        "cluster_offsets": saved.get("cluster_offsets"),
        "condition_cluster_map": saved.get("condition_cluster_map"),
    }, checkpoint_path)
    (args.output_dir / "preparation_manifest.json").write_text(json.dumps({
        "alignment": str(args.alignment.resolve()), "source": str(args.source.resolve()),
        "peak_matrix": str(args.peak_matrix.resolve()),
        "gtf": str(args.gtf_gz.resolve()), "gtf_release": args.gtf_gz.name, "genome_build": cfg["input"]["peak_genome_build"],
        "matched_genes": len(matched_genes), "n_genes": len(gene_names),
        "n_cells": len(labels), "n_clusters": K,
        "batch_names": batch_names,
        "condition_groups": saved.get("condition_groups"),
        "cluster_offsets": saved.get("cluster_offsets"),
        "condition_cluster_map": saved.get("condition_cluster_map"),
        "peak_filter": f"within-batch detected in >= {args.min_peak_cells} cells; linked within {args.max_distance} bp to selected genes",
        "M_init": "one-vs-rest DE on aligned gene-activity cells; no GT usage",
        "P_init_parameterization": "3/-5 logits; assigned probability = 1/(1+(K-1)*exp(-8)); trainable gradients",
        "gt_used_in_training": False,
    }, indent=2) + "\n")
    print(checkpoint_path, flush=True)

    return checkpoint_path

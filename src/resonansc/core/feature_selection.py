from typing import Iterable, Mapping, Optional, Sequence

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch

from .mapping import build_peak_gene_mapping_init, prepare_multimodal_training_data


def _as_numpy(x, dtype=None):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=dtype)


def _iter_items(data):
    if isinstance(data, Mapping):
        for key in sorted(data):
            yield key, data[key]
    else:
        for key, value in enumerate(data):
            yield key, value


def rank_rna_de_genes(
    rna_data,
    groupby: str,
    n_top: int = 2000,
    pval_adj_max: float = 0.05,
    logfc_min: Optional[float] = None,
    method: str = "wilcoxon",
    min_cells_per_group: int = 5,
) -> list[str]:
    """Return the union of RNA DE genes across RNA batches.

    ``rna_data`` may be a single AnnData or a dict/list of RNA AnnData objects.
    Genes are returned in the order they appear in the first RNA object.
    """
    items = list(_iter_items(rna_data)) if isinstance(rna_data, (Mapping, list, tuple)) else [(0, rna_data)]
    ordered_genes = [str(x) for x in items[0][1].var_names]
    selected = set()

    for _, adata in items:
        if groupby not in adata.obs:
            raise KeyError(f"{groupby!r} is not in RNA obs.")
        current = adata.copy()
        counts = current.obs[groupby].astype(str).value_counts()
        groups = counts[counts >= min_cells_per_group].index.tolist()
        if not groups:
            raise ValueError(f"No groups have >= {min_cells_per_group} cells for RNA DEG.")

        sc.tl.rank_genes_groups(
            current,
            groupby=groupby,
            groups=groups,
            reference="rest",
            method=method,
            use_raw=False,
        )
        for group in groups:
            df = sc.get.rank_genes_groups_df(current, group=group)
            if "pvals_adj" in df.columns:
                df = df[df["pvals_adj"] <= pval_adj_max]
            if logfc_min is not None and "logfoldchanges" in df.columns:
                df = df[df["logfoldchanges"] >= logfc_min]
            selected.update(df.head(n_top)["names"].astype(str).tolist())

    return [gene for gene in ordered_genes if gene in selected]


def _subset_mapping_info(info, keep_peak_mask, final_gene_names):
    old_gene_names = [str(x) for x in info["gene_names"]]
    final_gene_names = [str(x) for x in final_gene_names]
    gene_lookup = {gene: i for i, gene in enumerate(final_gene_names)}

    peak_idx = _as_numpy(info["peak_idx"], dtype=np.int64)
    gene_idx = _as_numpy(info["gene_idx"], dtype=np.int64)
    distance = _as_numpy(info["distance"], dtype=np.float32)
    init_weight = _as_numpy(info["init_weight"], dtype=np.float32)

    kept_old_peaks = np.where(keep_peak_mask)[0]
    peak_lookup = {int(old): new for new, old in enumerate(kept_old_peaks)}

    keep_edges = [
        edge_i
        for edge_i, (pidx, gidx) in enumerate(zip(peak_idx, gene_idx))
        if int(pidx) in peak_lookup and old_gene_names[int(gidx)] in gene_lookup
    ]
    keep_edges = np.asarray(keep_edges, dtype=np.int64)

    new_peak_idx = np.asarray(
        [peak_lookup[int(peak_idx[i])] for i in keep_edges], dtype=np.int64
    )
    new_gene_idx = np.asarray(
        [gene_lookup[old_gene_names[int(gene_idx[i])]] for i in keep_edges],
        dtype=np.int64,
    )

    out = dict(info)
    out["peak_names"] = [str(info["peak_names"][i]) for i in kept_old_peaks]
    out["gene_names"] = final_gene_names
    out["peak_idx"] = torch.as_tensor(new_peak_idx, dtype=torch.long)
    out["gene_idx"] = torch.as_tensor(new_gene_idx, dtype=torch.long)
    out["distance"] = torch.as_tensor(distance[keep_edges], dtype=torch.float32)
    out["init_weight"] = torch.as_tensor(init_weight[keep_edges], dtype=torch.float32)
    return out


def build_deg_dap_training_inputs(
    initialized_data,
    atacdata_da,
    deg_genes: Sequence[str],
    gtf_file,
    batch_key: str,
    window: int = 250_000,
    distance_scale: float = 5_000.0,
    promoter_upstream: int = 2_000,
):
    """Filter training features to DAP peaks linked to DEG genes.

    Returns
    -------
    training_data
        Dict with RNA batches as cell x filtered_gene and ATAC batches as
        cell x filtered_batch_specific_DAP_peak.
    mapping_init
        Sparse peak-gene mapping over the filtered features.
    gene_names
        Common filtered RNA gene space.
    filtered_atacdata_da
        DA peak dict after removing peaks that do not link to any selected gene.
    """
    deg_genes = [str(g) for g in deg_genes]
    if not deg_genes:
        raise ValueError("deg_genes is empty.")

    rough_training = prepare_multimodal_training_data(
        initialized_data=initialized_data,
        atacdata_da=atacdata_da,
        batch_key=batch_key,
    )
    rough_mapping = build_peak_gene_mapping_init(
        atacdata=rough_training,
        gene_names=deg_genes,
        gtf_file=gtf_file,
        batch_key=batch_key,
        max_distance=window,
        distance_scale=distance_scale,
        promoter_upstream=promoter_upstream,
    )

    linked_genes = set()
    keep_peak_masks = {}
    for batch_idx, info in rough_mapping.items():
        gene_idx = _as_numpy(info["gene_idx"], dtype=np.int64)
        peak_idx = _as_numpy(info["peak_idx"], dtype=np.int64)
        genes = [info["gene_names"][i] for i in np.unique(gene_idx)]
        linked_genes.update(map(str, genes))
        mask = np.zeros(len(info["peak_names"]), dtype=bool)
        mask[np.unique(peak_idx)] = True
        keep_peak_masks[int(batch_idx)] = mask

    final_gene_names = [gene for gene in deg_genes if gene in linked_genes]
    if not final_gene_names:
        raise ValueError("No DEG genes are linked to any DA peak within window.")

    training_data = {}
    filtered_atacdata_da = {}
    mapping_init = {}
    for batch_idx, adata_i in _iter_items(rough_training):
        batch_idx = int(batch_idx)
        batch_name = str(adata_i.obs[batch_key].iloc[0]).lower()
        if "atac" in batch_name:
            keep_mask = keep_peak_masks.get(batch_idx)
            if keep_mask is None or not keep_mask.any():
                raise ValueError(f"ATAC batch {batch_idx} has no kept DAP peaks.")
            training_data[batch_idx] = adata_i[:, keep_mask].copy()
            filtered_atacdata_da[batch_idx] = training_data[batch_idx].copy()
            mapping_init[batch_idx] = _subset_mapping_info(
                rough_mapping[batch_idx],
                keep_mask,
                final_gene_names,
            )
        else:
            missing = [g for g in final_gene_names if g not in adata_i.var_names]
            if missing:
                raise ValueError(f"RNA batch {batch_idx} misses {len(missing)} selected genes.")
            training_data[batch_idx] = adata_i[:, final_gene_names].copy()

    return training_data, mapping_init, final_gene_names, filtered_atacdata_da


def make_gene_activity_reference(training_data, mapping_init, batch_key: str):
    """Create RNA gene + ATAC mapped-gene reference data for init corr/merge."""
    gene_names = None
    for info in mapping_init.values():
        gene_names = [str(x) for x in info["gene_names"]]
        break
    if gene_names is None:
        raise ValueError("mapping_init is empty.")

    out = {}
    for batch_idx, adata_i in _iter_items(training_data):
        batch_idx = int(batch_idx)
        batch_name = str(adata_i.obs[batch_key].iloc[0]).lower()
        if "atac" not in batch_name:
            out[batch_idx] = adata_i[:, gene_names].copy()
            continue

        info = mapping_init[batch_idx]
        peak_idx = _as_numpy(info["peak_idx"], dtype=np.int64)
        gene_idx = _as_numpy(info["gene_idx"], dtype=np.int64)
        weights = _as_numpy(info["init_weight"], dtype=np.float32)
        mapping = sp.coo_matrix(
            (weights, (peak_idx, gene_idx)),
            shape=(adata_i.n_vars, len(gene_names)),
        ).tocsr()
        X = adata_i.X @ mapping
        if sp.issparse(X):
            X = X.tocsr()
            X.data = np.log1p(X.data)
        else:
            X = np.log1p(np.asarray(X, dtype=np.float32))
        ref = ad.AnnData(X=X, obs=adata_i.obs.copy(), var=pd.DataFrame(index=gene_names))
        out[batch_idx] = ref
    return out

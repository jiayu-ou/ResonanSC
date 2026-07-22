import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

EPS = 1e-8

def to_torch_X(ad, device):
    X = ad.X
    if sp.issparse(X):
        return X.tocsr().astype(np.float32, copy=False)
    return torch.as_tensor(X, dtype=torch.float32, device="cpu")

def iter_batches(rnadata):
    if isinstance(rnadata, dict):
        keys = sorted(list(rnadata.keys()))
        for k in keys:
            yield k, rnadata[k]
    else:
        for i, ad in enumerate(rnadata):
            yield i, ad

def get_batch_names(rnadata, col="batch"):
    names = []
    for _, ad in iter_batches(rnadata):
        u = ad.obs[col].unique().tolist() if col in ad.obs.columns else []
        names.append(str(u[0]) if len(u) == 1 else str(len(names)))
    return names

def get_bulk(rnadata, p_list, device="cpu", chunk_size=256):
    """Compute feature-by-cluster bulk profiles without densifying all of X.

    The calculation intentionally runs on CPU even when ``device="cuda"`` is
    supplied. Only small feature chunks are materialized at a time.
    """
    bulks = []
    for i, (bn, ad) in enumerate(iter_batches(rnadata)):
        print(
            "iter batch:", i, bn,
            "ncell:", ad.n_obs,
            "nfeature:", ad.n_vars,
            "use p shape:", tuple(p_list[i].shape),
            "compute device: cpu",
        )
        X = ad.X
        p = p_list[i].detach().cpu().to(dtype=torch.float32)
        p = p / (p.sum(dim=0, keepdim=True) + 1e-8)
        p_numpy = p.numpy()
        chunks = []

        for start in range(0, ad.n_vars, chunk_size):
            stop = min(start + chunk_size, ad.n_vars)
            X_chunk = X[:, start:stop]

            if sp.issparse(X_chunk):
                X_chunk = X_chunk.tocsr().astype(np.float32, copy=True)
                X_chunk.data = np.expm1(X_chunk.data)
                weighted_sum = X_chunk.T @ p_numpy
            else:
                X_chunk = np.asarray(X_chunk, dtype=np.float32)
                print("X_chunk shape:", X_chunk.shape)
                print("p_numpy shape:", p_numpy.shape)
                print("diff:", X_chunk.shape[0] - len(p_numpy))
                
                weighted_sum = np.expm1(X_chunk).T @ p_numpy

            chunks.append(
                torch.from_numpy(np.asarray(weighted_sum, dtype=np.float32))
            )

        bulk = torch.log1p(torch.cat(chunks, dim=0))
        bulks.append(bulk)
        print("bulk shape:", tuple(bulks[-1].shape))
    return bulks

def _as_numpy_p(p):
    if hasattr(p, "detach"):
        p = p.detach().cpu().numpy()
    p = np.asarray(p, dtype=np.float32)
    return p / (p.sum(axis=0, keepdims=True) + EPS)

def _as_numpy(x, dtype=None):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=dtype)

def _batch_modality(ad, batch_key):
    if batch_key in ad.obs.columns:
        batch_name = str(ad.obs[batch_key].iloc[0]).lower()
        if "atac" in batch_name:
            return "atac"
        if "rna" in batch_name:
            return "rna"
    return "rna"

def _rna_bulk_for_training(ad, p_norm, rna_layer="norm"):
    X = ad.layers[rna_layer] if rna_layer in ad.layers else ad.X
    if sp.issparse(X):
        bulk = X.T @ p_norm
    else:
        bulk = np.asarray(X, dtype=np.float32).T @ p_norm
    return torch.from_numpy(np.log1p(np.asarray(bulk, dtype=np.float32)))

def _atac_bulk_with_mapping(ad, p_norm, mapping_info, mapping_weights):
    X = ad.X
    if sp.issparse(X):
        peak_bulk = X.T @ p_norm
    else:
        peak_bulk = np.asarray(X, dtype=np.float32).T @ p_norm

    counts = np.asarray(peak_bulk, dtype=np.float32).T
    tf = counts / (counts.sum(axis=1, keepdims=True) + EPS)
    idf = np.log1p(counts.shape[0] / (tf.sum(axis=0, keepdims=True) + EPS))
    tfidf = tf * idf

    peak_idx = _as_numpy(mapping_info["peak_idx"], dtype=np.int64)
    gene_idx = _as_numpy(mapping_info["gene_idx"], dtype=np.int64)
    weights = _as_numpy(mapping_weights, dtype=np.float32)
    n_genes = len(mapping_info["gene_names"])
    mapping = sp.coo_matrix(
        (weights, (peak_idx, gene_idx)),
        shape=(ad.n_vars, n_genes),
    ).tocsr()
    gene_bulk = np.log1p(tfidf @ mapping)
    return torch.from_numpy(np.asarray(gene_bulk.T, dtype=np.float32))


def _resolve_mapping_batch_id(train_out, adata=None, batch_id=None, batch_key="batch"):
    mapping_init = train_out.get("mapping_init", {})
    mapping_weights = train_out.get("mapping_weights", {})
    available = sorted(int(k) for k in mapping_weights)

    if batch_id is not None:
        batch_id = int(batch_id)
        if batch_id not in mapping_init or batch_id not in mapping_weights:
            raise ValueError(
                f"batch_id={batch_id} is not available in mapping_init/mapping_weights. "
                f"Available ATAC mapping batches: {available}."
            )
        return batch_id

    if len(available) == 1:
        return available[0]

    if adata is not None and batch_key in adata.obs.columns:
        batch_name = str(adata.obs[batch_key].iloc[0])
        for candidate in available:
            info_name = str(mapping_init[candidate].get("batch_name", candidate))
            if info_name == batch_name:
                return candidate
        raise ValueError(
            f"Could not match AnnData {batch_key}={batch_name!r} to learned "
            f"ATAC mappings. Pass batch_id explicitly. Available batches: "
            f"{[(i, mapping_init[i].get('batch_name', i)) for i in available]}."
        )

    raise ValueError(
        "Multiple ATAC mappings are available; pass batch_id explicitly or "
        f"provide an AnnData with obs[{batch_key!r}]. Available batches: "
        f"{[(i, mapping_init[i].get('batch_name', i)) for i in available]}."
    )


def project_atac_to_gene_space(
    adata,
    train_out,
    batch_id=None,
    batch_key="batch",
    layer=None,
    log1p=True,
    return_adata=True,
    obsm_key=None,
    output_h5ad=None,
):
    """Project cell-level ATAC profiles into the learned gene space.

    The returned matrix has shape ``n_cells x n_genes``. It uses the learned
    sparse peak-gene mapping in ``train_out["mapping_weights"]`` without
    materializing a dense peak-by-gene matrix.
    """
    required = {"mapping_init", "mapping_weights", "gene_names"}
    missing = sorted(required.difference(train_out))
    if missing:
        raise KeyError(f"train_out is missing required keys: {missing}")

    batch_id = _resolve_mapping_batch_id(
        train_out, adata=adata, batch_id=batch_id, batch_key=batch_key
    )
    info = train_out["mapping_init"][batch_id]
    gene_names = list(map(str, train_out["gene_names"]))
    info_gene_names = list(map(str, info.get("gene_names", gene_names)))
    if info_gene_names != gene_names:
        raise ValueError(
            "mapping_init gene_names do not match train_out['gene_names']; "
            "use the checkpoint generated by the same training run."
        )

    expected_peaks = list(map(str, info["peak_names"]))
    adata_peaks = list(map(str, adata.var_names))
    if adata_peaks != expected_peaks:
        raise ValueError(
            "ATAC peak order does not match mapping_init. Use the same ATAC "
            "AnnData used for training, or reorder it to mapping_init['peak_names']."
        )

    X = adata.layers[layer] if layer is not None else adata.X
    if sp.issparse(X):
        X = X.tocsr().astype(np.float32, copy=False)
    else:
        X = sp.csr_matrix(np.asarray(X, dtype=np.float32))

    row_sum = np.asarray(X.sum(axis=1)).ravel()
    tf = X.multiply(1.0 / (row_sum[:, None] + EPS)).tocsr()
    feature_sum = np.asarray(tf.sum(axis=0)).ravel()
    idf = np.log1p(X.shape[0] / (feature_sum + EPS)).astype(np.float32)
    tfidf = tf.multiply(idf).tocsr()

    peak_idx = _as_numpy(info["peak_idx"], dtype=np.int64)
    gene_idx = _as_numpy(info["gene_idx"], dtype=np.int64)
    weights = _as_numpy(train_out["mapping_weights"][batch_id], dtype=np.float32)
    mapping = sp.coo_matrix(
        (weights, (peak_idx, gene_idx)),
        shape=(adata.n_vars, len(gene_names)),
        dtype=np.float32,
    ).tocsr()

    projected = (tfidf @ mapping).tocsr()
    if log1p:
        projected.data = np.log1p(projected.data)
        projected.eliminate_zeros()

    if obsm_key is not None:
        adata.obsm[obsm_key] = projected

    if not return_adata:
        return projected

    try:
        import anndata as ad
    except ImportError as exc:
        raise ImportError(
            "return_adata=True requires anndata. Use return_adata=False to "
            "return only the sparse matrix."
        ) from exc

    projected_adata = ad.AnnData(
        X=projected,
        obs=adata.obs.copy(),
        var=pd.DataFrame(index=pd.Index(gene_names, name=adata.var.index.name)),
    )
    projected_adata.uns["atac_gene_projection"] = {
        "source_batch_id": int(batch_id),
        "source_batch_name": str(info.get("batch_name", batch_id)),
        "layer": layer,
        "log1p": bool(log1p),
        "n_edges": int(weights.size),
    }

    if output_h5ad is not None:
        projected_adata.write_h5ad(output_h5ad)
        print(f"[mapping] saved cell-level ATAC gene projection to {output_h5ad}")

    return projected_adata

def get_multimodal_bulk(
    rnadata,
    p_list,
    train_out,
    batch_key="batch",
    rna_layer="norm",
):
    """Recompute gene-space bulks for mixed RNA/ATAC batches.

    RNA batches are aggregated in the common gene space. ATAC batches are first
    aggregated in peak space, transformed with dynamic bulk-level TF-IDF, then
    projected to the same gene space using the learned peak-gene mapping in
    ``train_out``.
    """
    bulks = []
    mapping_init = train_out.get("mapping_init", {})
    mapping_weights = train_out.get("mapping_weights", {})

    for i, (_, ad) in enumerate(iter_batches(rnadata)):
        p_norm = _as_numpy_p(p_list[i])
        modality = _batch_modality(ad, batch_key)

        if modality == "rna":
            bulk = _rna_bulk_for_training(ad, p_norm, rna_layer=rna_layer)
        else:
            if i not in mapping_init or i not in mapping_weights:
                raise ValueError(f"Missing learned mapping for ATAC batch {i}.")
            bulk = _atac_bulk_with_mapping(
                ad,
                p_norm,
                mapping_init[i],
                mapping_weights[i],
            )
        bulks.append(bulk)
        print("bulk shape:", i, modality, tuple(bulk.shape))
    return bulks

def get_subtype_onehot(rnadata, subtype_key='subtype'):
    rnadata_list = list(rnadata.values())
    all_subtypes = sorted(set(
        subtype 
        for adata in rnadata_list 
        for subtype in adata.obs[subtype_key].unique()
    ))
    
    onehot_list = []
    for adata in rnadata_list:
        labels = adata.obs[subtype_key]
        
        onehot_df = pd.get_dummies(labels, dtype=np.float32)
        
        for subtype in all_subtypes:
            if subtype not in onehot_df.columns:
                onehot_df[subtype] = 0.0
        
        onehot_df = onehot_df[all_subtypes]  # 按统一顺序排列
        onehot_list.append(onehot_df.values)  # shape: (n_cells, n_subtypes)
    
    return onehot_list, all_subtypes

def prune_empty_cols(p_list, eps=0.0):
    out = []
    keep_idx = []
    for p in p_list:
        keep = torch.where(p.sum(0) > eps)[0]
        out.append(p[:, keep])
        keep_idx.append(keep)
    return out, keep_idx

import os
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F


EPS = 1e-8


def _inverse_softplus(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable inverse of softplus for positive tensors."""
    return x + torch.log(-torch.expm1(-x))


def _get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _normalize_devices(
    devices: Optional[Union[Sequence[Union[int, str]], str]] = None,
    multi_gpu: bool = False,
) -> List[str]:
    if not torch.cuda.is_available():
        return ["cpu"]

    visible_count = torch.cuda.device_count()
    if devices is None:
        indices = list(range(visible_count)) if multi_gpu else [torch.cuda.current_device()]
    else:
        if isinstance(devices, str):
            parts = [x.strip() for x in devices.split(",") if x.strip()]
        else:
            parts = list(devices)
        indices = []
        for item in parts:
            text = str(item)
            if text.startswith("cuda:"):
                text = text.split(":", 1)[1]
            indices.append(int(text))

    if not indices:
        indices = [0]

    bad = [idx for idx in indices if idx < 0 or idx >= visible_count]
    if bad:
        raise ValueError(
            f"Requested CUDA device ids {bad}, but only {visible_count} "
            "visible device(s) are available. Device ids are logical ids after "
            "CUDA_VISIBLE_DEVICES has been applied."
        )

    if not multi_gpu and len(indices) > 1:
        indices = indices[:1]
    return [f"cuda:{idx}" for idx in indices]


def _same_device(left: str, right: str) -> bool:
    if left == right:
        return True
    if left == "cuda" and right in {"cuda", "cuda:0"}:
        return True
    if right == "cuda" and left in {"cuda", "cuda:0"}:
        return True
    return False


def _to_dense_float_tensor(x, device: str) -> torch.Tensor:
    """
    Convert numpy / sparse matrix to float tensor on target device.
    """
    if not isinstance(x, np.ndarray):
        x = x.toarray()
    return torch.from_numpy(x).float().to(device)


def _slice_to_float_tensor(x, start: int, stop: int, device: str) -> torch.Tensor:
    x_chunk = x[start:stop]
    if sp.issparse(x_chunk):
        x_chunk = x_chunk.toarray()
    return torch.as_tensor(x_chunk, dtype=torch.float32, device=device)


def _feature_weighted_sum(
    x,
    weights: torch.Tensor,
    device: str,
    chunk_size: int,
) -> torch.Tensor:
    if x.shape[0] != weights.shape[0]:
        raise ValueError(
            "Feature matrix / weights row mismatch in _feature_weighted_sum: "
            f"x.shape={tuple(x.shape)}, weights.shape={tuple(weights.shape)}. "
            "This usually means the training AnnData for a batch does not match "
            "the P/probs rows loaded from the init checkpoint."
        )
    if isinstance(x, torch.Tensor):
        return x.T @ weights

    n_vars = x.shape[1]
    out = torch.zeros(
        (n_vars, weights.shape[1]),
        dtype=weights.dtype,
        device=device,
    )
    for st in range(0, x.shape[0], chunk_size):
        ed = min(st + chunk_size, x.shape[0])
        x_chunk = _slice_to_float_tensor(x, st, ed, device)
        out = out + x_chunk.T @ weights[st:ed]
    return out


def _feature_weighted_sum_from_log1p(
    x,
    weights: torch.Tensor,
    device: str,
    chunk_size: int,
) -> torch.Tensor:
    """Aggregate log1p cell profiles in linear space."""
    if x.shape[0] != weights.shape[0]:
        raise ValueError(
            "Feature matrix / weights row mismatch in "
            "_feature_weighted_sum_from_log1p: "
            f"x.shape={tuple(x.shape)}, weights.shape={tuple(weights.shape)}."
        )
    if isinstance(x, torch.Tensor):
        return torch.expm1(x).T @ weights

    n_vars = x.shape[1]
    out = torch.zeros(
        (n_vars, weights.shape[1]),
        dtype=weights.dtype,
        device=device,
    )
    for st in range(0, x.shape[0], chunk_size):
        ed = min(st + chunk_size, x.shape[0])
        x_chunk = _slice_to_float_tensor(x, st, ed, device)
        out = out + torch.expm1(x_chunk).T @ weights[st:ed]
    return out


def _feature_weighted_sums_for_square_distance(
    x,
    weights: torch.Tensor,
    device: str,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if x.shape[0] != weights.shape[0]:
        raise ValueError(
            "Feature matrix / weights row mismatch in "
            "_feature_weighted_sums_for_square_distance: "
            f"x.shape={tuple(x.shape)}, weights.shape={tuple(weights.shape)}. "
            "This usually means the training AnnData for a batch does not match "
            "the P/probs rows loaded from the init checkpoint."
        )
    if isinstance(x, torch.Tensor):
        sum_x = x.T @ weights
        sum_x2 = (x * x).T @ weights
    else:
        n_vars = x.shape[1]
        sum_x = torch.zeros(
            (n_vars, weights.shape[1]),
            dtype=weights.dtype,
            device=device,
        )
        sum_x2 = torch.zeros_like(sum_x)
        for st in range(0, x.shape[0], chunk_size):
            ed = min(st + chunk_size, x.shape[0])
            x_chunk = _slice_to_float_tensor(x, st, ed, device)
            w_chunk = weights[st:ed]
            sum_x = sum_x + x_chunk.T @ w_chunk
            sum_x2 = sum_x2 + (x_chunk * x_chunk).T @ w_chunk

    sum_w = weights.sum(dim=0)
    return sum_x, sum_x2, sum_w


def _to_float_tensor(x, device: str) -> torch.Tensor:
    """Convert dense or scipy sparse input without densifying sparse ATAC."""
    if sp.issparse(x):
        x = x.tocoo()
        indices = torch.tensor(
            np.vstack([x.row, x.col]), dtype=torch.long, device=device
        )
        values = torch.tensor(x.data, dtype=torch.float32, device=device)
        return torch.sparse_coo_tensor(
            indices, values, size=x.shape, device=device
        ).coalesce()
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def _batch_modality(ad, batch_key: str) -> str:
    batch_name = str(ad.obs[batch_key].iloc[0]).lower()
    if "atac" in batch_name:
        return "atac"
    if "rna" in batch_name:
        return "rna"
    # Backward compatibility for the original RNA-only workflow.
    return "rna"


def filter_mapping_init_by_max_distance(
    mapping_init: Optional[Dict[int, Dict[str, Any]]],
    max_distance: Optional[float],
) -> Optional[Dict[int, Dict[str, Any]]]:
    """Restrict peak-gene candidate edges to a maximum genomic distance."""
    if mapping_init is None or max_distance is None:
        return mapping_init

    filtered = {}
    for batch_idx, info in mapping_init.items():
        distance = info["distance"].float()
        keep = distance <= float(max_distance)
        new_info = dict(info)
        for key in ("peak_idx", "gene_idx", "distance", "init_weight"):
            if key in info:
                new_info[key] = info[key][keep].clone()
        new_info["max_distance"] = float(max_distance)
        filtered[int(batch_idx)] = new_info
    return filtered


def prepare_input_cache(
    rnadata,
    device: str,
    batch_key: str,
    rna_layer: str = "norm",
    cache_rna_on_device: bool = False,
    batch_devices: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """Materialize per-batch tensors once and reuse them across epochs."""
    cache = []
    for i in range(len(rnadata)):
        target_device = (
            batch_devices[i % len(batch_devices)]
            if batch_devices
            else device
        )
        modality = _batch_modality(rnadata[i], batch_key)
        if modality == "rna":
            X_source = (
                rnadata[i].layers[rna_layer]
                if rna_layer in rnadata[i].layers
                else rnadata[i].X
            )
            if cache_rna_on_device:
                X = _to_dense_float_tensor(X_source, device=target_device)
                X_for_cell_loss = X
            else:
                X = X_source.tocsr() if sp.issparse(X_source) else np.asarray(X_source)
                X_for_cell_loss = X
            X_peak_source = None
        else:
            X_peak_source = (
                rnadata[i].X.tocsr().astype(np.float32, copy=False)
                if sp.issparse(rnadata[i].X)
                else sp.csr_matrix(np.asarray(rnadata[i].X, dtype=np.float32))
            )
            X = _to_float_tensor(X_peak_source, device=target_device)
            X_for_cell_loss = None

        cache.append(
            {
                "modality": modality,
                "device": target_device,
                "X": X,
                "X_for_cell_loss": X_for_cell_loss,
                "X_peak_source": X_peak_source,
            }
        )
    return cache


def build_active_mask_from_probs(
    colsum_list: Sequence[torch.Tensor],
    stage: str,
    batch_names: Sequence[str],
    bulknum: int,
    slot2proto: torch.Tensor,
    cellnum_list: Sequence[int],
    present_eps: float,
) -> Dict[str, torch.Tensor]:
    """
    Return active_mask_dict {bname: (bulknum,) bool}

    learn_P:
        active = present
    learn_M:
        active = present, but mask out the '1' side in one-to-many prototypes
    """
    dev = colsum_list[0].device
    s2p = slot2proto.to(dev)
    P = int(s2p.max().item()) + 1
    B = len(colsum_list)

    present = []
    counts = torch.zeros((B, P), dtype=torch.long, device=dev)

    for bi in range(B):
        cellnum = float(cellnum_list[bi])
        eps_bi = present_eps * (cellnum / bulknum)
        pm = colsum_list[bi] > eps_bi
        present.append(pm)

        for p in range(P):
            idx = s2p == p
            counts[bi, p] = pm[idx].sum().long()

    if stage == "learn_P":
        return {bn: present[bi] for bi, bn in enumerate(batch_names)}

    maxc = counts.max(dim=0).values
    active = {}

    for bi, bn in enumerate(batch_names):
        pm = present[bi]
        am = pm.clone()

        for p in range(P):
            if int(maxc[p].item()) > 1 and int(counts[bi, p].item()) == 1:
                idx = ((s2p == p) & pm).nonzero(as_tuple=False).flatten()
                if idx.numel() == 1:
                    am[idx[0].item()] = False

        active[bn] = am

    return active


class BulkProjecting(nn.Module):
    def __init__(
        self,
        cellnum_list: Sequence[int],
        bulknum: int,
        n_genes: int,
        p_init: Optional[Sequence[torch.Tensor]] = None,
        m_init: Optional[torch.Tensor] = None,
        mapping_init: Optional[Dict[int, Dict[str, Any]]] = None,
        one_hot_start: bool = False,
        marker_init_mode: str = "continuous",
        marker_init_eps: float = 0.05,
    ):
        super().__init__()
        self.bulknum = bulknum
        self.one_hot_start = one_hot_start

        if not 0.0 < marker_init_eps < 0.5:
            raise ValueError("marker_init_eps must be between 0 and 0.5.")
        if marker_init_mode not in {"binary", "continuous"}:
            raise ValueError(
                "marker_init_mode must be 'binary' or 'continuous'."
            )

        if p_init is not None:
            self.probs = nn.ParameterList([
                nn.Parameter(p_init[i].clone())
                for i in range(len(cellnum_list))
            ])
        else:
            self.probs = nn.ParameterList([
                nn.Parameter(torch.randn(cellnum_list[i], bulknum))
                for i in range(len(cellnum_list))
            ])

        if m_init is None:
            self.markers = nn.Parameter(torch.full((n_genes, bulknum), 2.0))
        else:
            if tuple(m_init.shape) != (n_genes, bulknum):
                raise ValueError(
                    f"m_init must have shape ({n_genes}, {bulknum}), got {tuple(m_init.shape)}"
                )
            marker_prob = m_init.float()
            if marker_init_mode == "binary":
                marker_prob = (marker_prob >= 0.5).to(marker_prob.dtype)
            marker_prob = marker_prob.clamp(
                marker_init_eps,
                1.0 - marker_init_eps,
            )
            self.markers = nn.Parameter(torch.logit(marker_prob))

        self.mapping_logits = nn.ParameterDict()
        self.mapping_info = {}
        for batch_idx, info in (mapping_init or {}).items():
            key = str(batch_idx)
            init_weight = info["init_weight"].float().clamp_min(1e-6)
            if init_weight.numel() == 0:
                raise ValueError(
                    f"ATAC batch {batch_idx} has no candidate peak-gene edges."
                )
            self.mapping_logits[key] = nn.Parameter(_inverse_softplus(init_weight))
            self.mapping_info[key] = {
                "peak_idx": info["peak_idx"].long(),
                "gene_idx": info["gene_idx"].long(),
                "distance": info["distance"].float(),
                "max_distance": float(info.get("max_distance", 100_000)),
            }

    def get_probs(self, batch_id: int):
        if self.one_hot_start:
            probs_soft = self.probs[batch_id]
        else:
            probs_soft = torch.softmax(self.probs[batch_id], dim=1)

        probs_norm = probs_soft / (probs_soft.sum(dim=0, keepdim=True) + EPS)
        unprobs = 1.0 - probs_soft
        unprobs_norm = unprobs / (unprobs.sum(dim=0, keepdim=True) + EPS)

        colsum = probs_soft.sum(dim=0)
        return probs_soft, probs_norm, unprobs_norm, colsum

    def get_markers(self):
        markers_prob = torch.sigmoid(self.markers)
        markers_hard = (markers_prob >= 0.5).to(markers_prob.dtype)
        empty = markers_hard.sum(dim=0) == 0
        if empty.any():
            fallback = F.one_hot(
                markers_prob.argmax(dim=0),
                num_classes=markers_prob.shape[0],
            ).T.to(markers_prob.dtype)
            markers_hard = torch.where(
                empty.unsqueeze(0), fallback, markers_hard
            )
        markers_binary = (
            markers_hard.detach() - markers_prob.detach() + markers_prob
        )
        markers_norm = markers_binary / (
            markers_binary.sum(dim=0, keepdim=True) + EPS
        )
        return markers_binary, markers_norm, markers_prob

    def rna_bulk(self, X: torch.Tensor, probs_norm: torch.Tensor):
        # RNA input is normalize_total + log1p. Aggregate in linear space,
        # then return to log1p space so cells and bulks share one scale.
        return torch.log1p(torch.expm1(X).T @ probs_norm)

    def rna_bulk_from_source(
        self,
        X,
        probs_norm: torch.Tensor,
        device: str,
        chunk_size: int = 512,
    ):
        weighted_sum = _feature_weighted_sum_from_log1p(
            X, probs_norm, device=device, chunk_size=chunk_size
        )
        return torch.log1p(weighted_sum)

    def mapping_weights(self, batch_id: int):
        return F.softplus(self.mapping_logits[str(batch_id)])

    def atac_tfidf(self, X: torch.Tensor, probs_norm: torch.Tensor):
        if not X.is_sparse:
            peak_bulk = X.T @ probs_norm
        else:
            peak_bulk = torch.sparse.mm(X.transpose(0, 1), probs_norm)

        counts = peak_bulk.T
        tf = counts / (counts.sum(dim=1, keepdim=True) + EPS)
        idf = torch.log1p(
            counts.shape[0] / (tf.sum(dim=0, keepdim=True) + EPS)
        )
        return tf * idf

    def atac_cell_gene_from_source(
        self,
        X,
        batch_id: int,
    ) -> sp.csr_matrix:
        """Map single-cell ATAC profiles into the learned gene space.

        Mapping is fixed during ``learn_P``, so this cell-level representation
        can be cached as a CPU sparse matrix while gradients still flow through
        the assignment weights and the P-dependent ATAC bulks.
        """
        X = X.tocsr().astype(np.float32, copy=False)
        row_sum = np.asarray(X.sum(axis=1)).ravel()
        tf = X.multiply(1.0 / (row_sum[:, None] + EPS)).tocsr()
        feature_sum = np.asarray(tf.sum(axis=0)).ravel()
        idf = np.log1p(X.shape[0] / (feature_sum + EPS)).astype(np.float32)
        tfidf = tf.multiply(idf).tocsr()

        info = self.mapping_info[str(batch_id)]
        peak_idx = info["peak_idx"].detach().cpu().numpy()
        gene_idx = info["gene_idx"].detach().cpu().numpy()
        weights = self.mapping_weights(batch_id).detach().cpu().numpy()
        mapping = sp.coo_matrix(
            (weights, (peak_idx, gene_idx)),
            shape=(X.shape[1], self.markers.shape[0]),
            dtype=np.float32,
        ).tocsr()

        gene_activity = (tfidf @ mapping).tocsr()
        gene_activity.data = np.log1p(gene_activity.data)
        gene_activity.eliminate_zeros()
        return gene_activity

    def gene_bulk_from_tfidf(
        self,
        tfidf: torch.Tensor,
        batch_id: int,
        edge_chunk_size: int = 200_000,
    ):
        info = self.mapping_info[str(batch_id)]
        peak_idx = info["peak_idx"].to(tfidf.device)
        gene_idx = info["gene_idx"].to(tfidf.device)
        weights = self.mapping_weights(batch_id).to(tfidf.device)
        gene_bulk = torch.zeros(
            (tfidf.shape[0], self.markers.shape[0]),
            dtype=tfidf.dtype,
            device=tfidf.device,
        )

        for start in range(0, weights.numel(), edge_chunk_size):
            stop = min(start + edge_chunk_size, weights.numel())
            contribution = (
                tfidf.index_select(1, peak_idx[start:stop])
                * weights[start:stop].unsqueeze(0)
            )
            gene_bulk.index_add_(
                1, gene_idx[start:stop], contribution
            )

        return torch.log1p(gene_bulk).T

    def atac_bulk(
        self,
        X: torch.Tensor,
        probs_norm: torch.Tensor,
        batch_id: int,
        edge_chunk_size: int = 200_000,
    ):
        tfidf = self.atac_tfidf(X, probs_norm)
        return self.gene_bulk_from_tfidf(
            tfidf, batch_id, edge_chunk_size=edge_chunk_size
        )

    def forward(self, X: torch.Tensor, batch_id: int, modality: str):
        probs_soft, probs_norm, unprobs_norm, colsum = self.get_probs(batch_id)
        if modality == "rna":
            bulk = self.rna_bulk(X, probs_norm)
        elif modality == "atac":
            bulk = self.atac_bulk(X, probs_norm, batch_id)
        else:
            raise ValueError(f"Unsupported modality: {modality}")

        markers_binary, markers_norm, markers_prob = self.get_markers()
        return (
            bulk,
            probs_soft,
            probs_norm,
            unprobs_norm,
            markers_binary,
            markers_norm,
            markers_prob,
            colsum,
        )

    def mapping_regularization(self):
        if len(self.mapping_logits) == 0:
            zero = self.markers.sum() * 0.0
            return zero, zero

        distance_terms = []
        gene_totals = []
        for key, logits in self.mapping_logits.items():
            weights = F.softplus(logits)
            info = self.mapping_info[key]
            distance = info["distance"].to(weights.device)
            max_distance = info["max_distance"]
            distance_terms.append(
                (weights * (distance / max_distance)).sum()
                / (weights.sum() + EPS)
            )

            totals = torch.zeros(
                self.markers.shape[0], dtype=weights.dtype, device=weights.device
            )
            totals.index_add_(0, info["gene_idx"].to(weights.device), weights)
            gene_totals.append(totals / (totals.sum() + EPS))

        distance_loss = torch.stack(distance_terms).mean()
        if len(gene_totals) == 1:
            batch_loss = distance_loss * 0.0
        else:
            totals = torch.stack(gene_totals)
            batch_loss = ((totals - totals.mean(dim=0)) ** 2).mean()
        return distance_loss, batch_loss

class LowDimClustering(nn.Module):
    def __init__(
        self,
        stage: str,
        margin: Sequence[float],
        lambdas: Sequence[float],
        batch_names: Sequence[str],
        cellbulk_chunk_size: int = 512,
    ):
        super().__init__()
        self.stage = stage
        self.margin = margin
        self.lambdas = lambdas
        self.batch_names = batch_names
        self.cellbulk_chunk_size = cellbulk_chunk_size
        
    def correlation(self, x, y, w):
        target_device = x.device
        if y.device != target_device:
            y = y.to(target_device)
        if w.device != target_device:
            w = w.to(target_device)
        w = torch.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        w = w / (w.sum() + EPS)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

        mean_x = (w * x).sum()
        mean_y = (w * y).sum()
        xm = x - mean_x
        ym = y - mean_y
        cov = (w * xm * ym).sum()
        var_x = (w * xm ** 2).sum()
        var_y = (w * ym ** 2).sum()
        denom = torch.sqrt(torch.clamp(var_x * var_y, min=EPS))
        corr = cov / denom
        return torch.nan_to_num(corr, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)

    def distance(self, diff, prob=None, gene_w=None):
        """
        diff: (cell, gene)
        prob: (cell,)
        gene_w: (gene,) 
        """
        if gene_w is None:
            dist = (diff**2).mean(dim=1)  # (cell,)
        else:
            # 加权 MSE：sum_g w_g * diff^2
            dist = (diff**2 * gene_w.unsqueeze(0)).sum(dim=1)  # (cell,)

        if prob is not None:
            return (dist * prob).sum()
        return dist.mean()
    
    def _bname(self, i):
        return self.batch_names[i]

    def forward(self, bulks_list, probs_list, X_list, unprobs_list, M_global, active_mask_dict):
        device = M_global.device
        m_all = M_global.mean(dim=1)
        
        # 1) in-batch M[a]+M[b]
        inbatch_terms = []
        for i in range(len(bulks_list)):
            bulksi = bulks_list[i]  # (G, bulk)
            local_device = bulksi.device
            bname = self._bname(i)
            K_i = bulksi.shape[1]

            mask_i = active_mask_dict[bname].to(local_device)  # (bulknum,)

            pair_corrs = []
            for a in range(K_i):
                if not mask_i[a].item():
                    continue
                for b in range(a + 1, K_i):
                    if not mask_i[b].item():
                        continue
                    m_ab = (M_global[:, a] + M_global[:, b]) / 2.0
                    corr = self.correlation(bulksi[:, a], bulksi[:, b], m_ab)

                    if self.stage in ("learn_M", "learn_mapping"):
                        viol = torch.relu(corr + self.margin[0]) 
                            
                    elif self.stage == 'learn_P':
                        viol = torch.relu(corr**2 - self.margin[0])

                    pair_corrs.append(viol.to(device))

            if pair_corrs:
                inbatch_terms.append(torch.stack(pair_corrs).mean())
        inbatch_loss = (
            torch.stack(inbatch_terms).mean()
            if inbatch_terms
            else M_global.sum() * 0.0
        )

        # 2) cross-batch same-vs-other(mean) loss
        contrast_terms = []
        for i in range(len(bulks_list)):
            for j in range(i + 1, len(bulks_list)):   # different batches
                    bi = self._bname(i)
                    bj = self._bname(j)
                    bulksi = bulks_list[i]  # (G, K_i)
                    bulksj = bulks_list[j]  # (G, K_j)
                    local_device = bulksi.device
                    K_i = bulksi.shape[1]
                    # K_j = bulksj.shape[1]

                    mask_i = active_mask_dict[bi].to(local_device)
                    mask_j = active_mask_dict[bj].to(local_device)

                    # Kmin = min(K_i, K_j)
                    for a in range(K_i):
                        other_corrs = []       

                        if (not mask_i[a].item()) and (not mask_j[a].item()):
                            continue

                        if mask_i[a].item():
                            for b in range(K_i): #a+1, 
                                if not mask_j[b].item():
                                    continue
                                if b == a:
                                    continue
                                m_ab = (M_global[:, a] + M_global[:, b]) / 2.0
                                # if self.stage == 'learn_M':
                                other_corrs.append(self.correlation(bulksi[:, a], bulksj[:, b], m_ab))

                        elif mask_j[a].item():
                            for b in range(K_i):
                                if not mask_i[b].item():
                                    continue
                                if b == a:
                                    continue
                                m_ab = (M_global[:, a] + M_global[:, b]) / 2.0
                                # if self.stage == 'learn_M':
                                other_corrs.append(self.correlation(bulksi[:, b], bulksj[:, a], m_ab))

                        if len(other_corrs)==0:
                            corr_other_mean = torch.tensor(1.0, device=local_device)
                        else:
                            corr_other_mean = (torch.exp(torch.stack(other_corrs))).mean()

                        corr_other = torch.log(corr_other_mean)
                        if corr_other.device != local_device:
                            corr_other = corr_other.to(local_device)

                        if mask_i[a].item() and mask_j[a].item():
                            corr_same = self.correlation(bulksi[:, a], bulksj[:, a], m_all)
                        else:
                            corr_same = torch.tensor(0.0, device=local_device)
                        
                        viol = torch.relu(self.margin[1] - corr_same + corr_other)

                        contrast_terms.append(viol.to(device))

        # contrast_loss = torch.stack(contrast_terms).mean()
        if len(contrast_terms) == 0:
            contrast_loss = torch.tensor(0.0, device=M_global.device)
        else:
            contrast_loss = torch.stack(contrast_terms).mean()

        
        # 3) cell-bulk distance M[all]
        # This term only contributes to learn_P in the current objective.
        # Skipping it in learn_M/learn_mapping avoids a large RNA cell x gene
        # pass without changing the optimized loss.
        cellbulk_dist = torch.tensor(0.0, device=device)
        if self.stage == "learn_P":
            cellbulk_cnt = 0
            cellbulk_sum = torch.tensor(0.0, device=device)

            for i in range(len(X_list)):
                # RNA expression or mapped ATAC gene activity: (cell, gene).
                X_i = X_list[i]
                if X_i is None:
                    continue
                unprobsi = unprobs_list[i]   # (cell, K)
                bulksi = bulks_list[i]       # (gene, K)
                local_device = bulksi.device

                K_i = bulksi.shape[1]
                M_all = M_global.mean(dim=1).to(local_device).view(-1, 1)  # (gene, 1)
                if unprobsi.device != local_device:
                    unprobsi = unprobsi.to(local_device)
                if isinstance(X_i, torch.Tensor) and X_i.device != local_device:
                    X_i = X_i.to(local_device)
                sum_x, sum_x2, sum_w = _feature_weighted_sums_for_square_distance(
                    X_i,
                    unprobsi,
                    device=local_device,
                    chunk_size=self.cellbulk_chunk_size,
                )
                # Equivalent to sum_c unprob[c,k] * 0.5 *
                # sum_g M[g] * (X[c,g] - bulk[g,k])^2, but avoids materializing
                # a large cell x gene diff tensor.
                out_acc = 0.5 * (
                    M_all
                    * (
                        sum_x2
                        - 2.0 * bulksi * sum_x
                        + bulksi.pow(2) * sum_w.view(1, -1)
                    )
                ).sum(dim=0)

                viol = torch.relu(-out_acc + self.margin[2])
                cellbulk_sum = cellbulk_sum + viol.sum().to(device)
                cellbulk_cnt += K_i

            cellbulk_dist = cellbulk_sum / max(cellbulk_cnt, 1)

    
        # total loss

        if self.stage in ("learn_M", "learn_mapping"):
            loss = self.lambdas[0]*inbatch_loss + self.lambdas[1]*contrast_loss #+ 5*cellbulk_dist#+ cross_loss# #intype_loss   # + intype_loss
        elif self.stage == 'learn_P':
            loss = self.lambdas[0]*inbatch_loss + self.lambdas[1]*contrast_loss + self.lambdas[2]*cellbulk_dist

        return loss, cellbulk_dist, inbatch_loss, contrast_loss
    

def build_lists(
    rnadata,
    model: BulkProjecting,
    device: str,
    batch_key: str,
    rna_layer: str = "norm",
    input_cache: Optional[List[Dict[str, Any]]] = None,
    fixed_rna_bulk_cache: Optional[Dict[int, torch.Tensor]] = None,
    fixed_tfidf_cache: Optional[Dict[int, torch.Tensor]] = None,
    fixed_pack: Optional[Dict[str, Any]] = None,
    rna_chunk_size: int = 512,
    batch_devices: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    markers_binary, markers_norm, markers_prob = model.get_markers()
    if fixed_pack is not None:
        pack = {
            "bulks_list": fixed_pack["bulks_list"],
            "probs_soft_list": fixed_pack["probs_soft_list"],
            "probs_norm_list": fixed_pack["probs_norm_list"],
            "X_list": fixed_pack["X_list"],
            "unprobs_norm_list": fixed_pack["unprobs_norm_list"],
            "colsum_list": fixed_pack["colsum_list"],
            "modalities": fixed_pack["modalities"],
            "markers_binary": markers_binary,
            "markers_norm": markers_norm,
            "markers_prob": markers_prob,
        }
        return pack

    bulks_list = []
    probs_soft_list = []
    probs_norm_list = []
    unprobs_norm_list = []
    X_list = []
    colsum_list = []
    modalities = []

    for i in range(len(rnadata)):
        target_device = (
            batch_devices[i % len(batch_devices)]
            if batch_devices
            else device
        )
        if input_cache is None:
            modality = _batch_modality(rnadata[i], batch_key)
            if modality == "rna":
                X_source = (
                    rnadata[i].layers[rna_layer]
                    if rna_layer in rnadata[i].layers
                    else rnadata[i].X
                )
                X = _to_dense_float_tensor(X_source, device=target_device)
                X_for_cell_loss = X
            else:
                X = _to_float_tensor(rnadata[i].X, device=target_device)
                X_for_cell_loss = None
        else:
            cached = input_cache[i]
            modality = cached["modality"]
            target_device = cached.get("device", target_device)
            X = cached["X"]
            X_for_cell_loss = cached["X_for_cell_loss"]

        probs_soft, probs_norm, unprobs_norm, colsum = model.get_probs(i)
        probs_norm_for_bulk = (
            probs_norm
            if _same_device(str(probs_norm.device), target_device)
            else probs_norm.to(target_device)
        )
        if modality == "rna":
            if fixed_rna_bulk_cache is not None and i in fixed_rna_bulk_cache:
                bulk = fixed_rna_bulk_cache[i]
            elif isinstance(X, torch.Tensor):
                bulk = model.rna_bulk(X, probs_norm_for_bulk)
            else:
                bulk = model.rna_bulk_from_source(
                    X,
                    probs_norm_for_bulk,
                    device=target_device,
                    chunk_size=rna_chunk_size,
                )
        elif modality == "atac":
            if fixed_tfidf_cache is not None and i in fixed_tfidf_cache:
                bulk = model.gene_bulk_from_tfidf(fixed_tfidf_cache[i], i)
            else:
                bulk = model.atac_bulk(X, probs_norm_for_bulk, i)
        else:
            raise ValueError(f"Unsupported modality: {modality}")
        if batch_devices is None and not _same_device(str(bulk.device), device):
            bulk = bulk.to(device)

        bulks_list.append(bulk)
        probs_soft_list.append(probs_soft)
        probs_norm_list.append(probs_norm)
        unprobs_norm_list.append(unprobs_norm)
        if (
            batch_devices is None
            and
            isinstance(X_for_cell_loss, torch.Tensor)
            and not _same_device(str(X_for_cell_loss.device), device)
        ):
            X_for_cell_loss = X_for_cell_loss.to(device)
        X_list.append(X_for_cell_loss)
        colsum_list.append(colsum.detach())
        modalities.append(modality)

    return {
        "bulks_list": bulks_list,
        "probs_soft_list": probs_soft_list,
        "probs_norm_list": probs_norm_list,
        "X_list": X_list,
        "unprobs_norm_list": unprobs_norm_list,
        "markers_binary": markers_binary,
        "markers_norm": markers_norm,
        "markers_prob": markers_prob,
        "colsum_list": colsum_list,
        "modalities": modalities,
    }


def build_fixed_bulk_caches(
    model: BulkProjecting,
    input_cache: List[Dict[str, Any]],
    device: str,
    rna_chunk_size: int = 512,
    batch_devices: Optional[Sequence[str]] = None,
) -> Dict[str, Dict[int, torch.Tensor]]:
    """Cache quantities that are fixed when P is fixed."""
    fixed_rna_bulk_cache = {}
    fixed_tfidf_cache = {}

    with torch.no_grad():
        for i, cached in enumerate(input_cache):
            target_device = cached.get(
                "device",
                batch_devices[i % len(batch_devices)]
                if batch_devices
                else device,
            )
            _, probs_norm, _, _ = model.get_probs(i)
            probs_norm_for_bulk = (
                probs_norm
                if _same_device(str(probs_norm.device), target_device)
                else probs_norm.to(target_device)
            )
            if cached["modality"] == "rna":
                if isinstance(cached["X"], torch.Tensor):
                    bulk = model.rna_bulk(cached["X"], probs_norm_for_bulk)
                else:
                    bulk = model.rna_bulk_from_source(
                        cached["X"],
                        probs_norm_for_bulk,
                        device=target_device,
                        chunk_size=rna_chunk_size,
                    )
                fixed_rna_bulk_cache[i] = bulk.detach()
            elif cached["modality"] == "atac":
                fixed_tfidf_cache[i] = model.atac_tfidf(
                    cached["X"], probs_norm_for_bulk
                ).detach()

    return {
        "fixed_rna_bulk_cache": fixed_rna_bulk_cache,
        "fixed_tfidf_cache": fixed_tfidf_cache,
    }


def LearnPseudoMaker(
    rnadata,
    epochs: int = 200,
    lr: float = 1e-2,
    reg_1: float = 1e-5,
    reg_2: float = 1e-5,
    M_init: Optional[torch.Tensor] = None,
    P_init: Optional[Sequence[torch.Tensor]] = None,
    align_masks: Optional[Dict[str, Any]] = None,
    batch_keys: Optional[Sequence[str]] = None,
    margin: Sequence[float] = (0.5, 1.5, 3.0),
    lambdas: Sequence[float] = (1.0, 2.0, 5.0),
    stage: str = "learn_M",
    one_hot_start: bool = False,
    resume_path: Optional[str] = None,
    present_eps: float = 0.1,
    de_gene_set=None,
    mapping_init: Optional[Dict[int, Dict[str, Any]]] = None,
    batch_key: str = "batch",
    gene_names: Optional[Sequence[str]] = None,
    rna_layer: str = "norm",
    mapping_lambdas: Sequence[float] = (1.0, 1.0),
    marker_sparsity: float = 1.0,
    marker_init_mode: str = "continuous",
    marker_init_eps: float = 0.05,
    verbose_every: int = 1,
    rna_chunk_size: int = 512,
    cellbulk_chunk_size: int = 512,
    cache_rna_on_device: bool = False,
    mapping_max_distance: Optional[float] = None,
    multi_gpu: bool = False,
    devices: Optional[Union[Sequence[Union[int, str]], str]] = None,
) -> Dict[str, Any]:
    device_list = _normalize_devices(devices=devices, multi_gpu=multi_gpu)
    device = device_list[0]
    batch_devices = device_list if len(device_list) > 1 else None
    if len(device_list) > 1:
        print(f"[multi-gpu] primary={device}; batch_devices={device_list}")
    else:
        print(f"[device] {device}")
    if gene_names is None:
        rna_batches = [
            rnadata[i]
            for i in range(len(rnadata))
            if _batch_modality(rnadata[i], batch_key) == "rna"
        ]
        if not rna_batches:
            raise ValueError("At least one RNA batch is required.")
        gene_names = rna_batches[0].var_names.astype(str).tolist()
    gene_names = list(map(str, gene_names))
    G = len(gene_names)

    for i in range(len(rnadata)):
        modality = _batch_modality(rnadata[i], batch_key)
        if modality == "rna":
            current = rnadata[i].var_names.astype(str).tolist()
            if current != gene_names:
                raise ValueError(
                    f"RNA batch {i} var_names do not match gene_names order."
                )
        elif mapping_init is None or i not in mapping_init:
            raise ValueError(f"Missing mapping_init for ATAC batch {i}.")
        else:
            info = mapping_init[i]
            if "gene_names" in info and list(map(str, info["gene_names"])) != gene_names:
                raise ValueError(
                    f"ATAC batch {i} mapping gene order does not match gene_names."
                )
            if "peak_names" in info:
                current_peaks = rnadata[i].var_names.astype(str).tolist()
                if list(map(str, info["peak_names"])) != current_peaks:
                    raise ValueError(
                        f"ATAC batch {i} peak order does not match mapping_init."
                    )

    mapping_init = filter_mapping_init_by_max_distance(
        mapping_init, mapping_max_distance
    )
    if mapping_max_distance is not None and mapping_init is not None:
        for batch_idx, info in sorted(mapping_init.items()):
            print(
                f"[mapping] batch {batch_idx}: keep "
                f"{info['peak_idx'].numel()} edges within "
                f"{float(mapping_max_distance):g} bp"
            )

    if resume_path is not None:
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        bulknum = ckpt["bulknum"]
    else:
        if P_init is None:
            raise ValueError("P_init must be provided when resume_path is None.")
        bulknum = int(P_init[0].shape[1])

    print(f"bulknum(K_global): {bulknum}")

    if align_masks is None:
        raise ValueError("align_masks must be provided.")

    if batch_keys is None:
        batch_keys = align_masks.get("batch_keys", None)
    if batch_keys is None:
        raise ValueError("batch_keys is None and align_masks does not contain 'batch_keys'.")

    batch_names = [str(x) for x in batch_keys]

    slot2proto = align_masks["slot2proto"]
    if isinstance(slot2proto, np.ndarray):
        slot2proto = torch.from_numpy(slot2proto)
    slot2proto = slot2proto.long()

    protoP = int(slot2proto.max().item()) + 1
    print(f"batch_names: {batch_names}")
    print(f"prototype_count={protoP}, bulknum={bulknum}")

    cellnum_list = [rnadata[i].shape[0] for i in range(len(rnadata))]
    if P_init is not None:
        bad_p_init = []
        for i, p in enumerate(P_init):
            shape = tuple(p.shape)
            if len(shape) != 2 or shape[0] != cellnum_list[i] or shape[1] != bulknum:
                bad_p_init.append((i, batch_names[i], shape))
        if bad_p_init:
            raise ValueError(
                "P_init must have shape (n_cells, bulknum) for every batch; "
                f"bulknum={bulknum}, mismatches={bad_p_init}. "
                "If this came from a merge checkpoint, use P_align/P_init rather "
                "than P_merge for training."
            )
    if int(slot2proto.numel()) != int(bulknum):
        raise ValueError(
            "align_masks['slot2proto'] length must match bulknum; "
            f"got {int(slot2proto.numel())} and {bulknum}."
        )

    model = BulkProjecting(
        cellnum_list=cellnum_list,
        bulknum=bulknum,
        n_genes=G,
        p_init=P_init,
        m_init=M_init,
        mapping_init=mapping_init,
        one_hot_start=one_hot_start,
        marker_init_mode=marker_init_mode,
        marker_init_eps=marker_init_eps,
    ).to(device)

    if resume_path is None:
        print(
            f"[M INIT] mode={marker_init_mode}; "
            f"source={'M_init' if M_init is not None else 'default logits'}; "
            f"eps={marker_init_eps:g}"
        )

    if resume_path is not None:
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        print(f"[M INIT] resume model_state_dict from {resume_path}")

    if stage == "learn_M":
        for p in model.probs:
            p.requires_grad_(False)
        model.markers.requires_grad_(True)
        for p in model.mapping_logits.parameters():
            p.requires_grad_(False)
        opt = torch.optim.Adam([model.markers], lr=lr, weight_decay=1e-5)

    elif stage == "learn_P":
        model.markers.requires_grad_(False)
        for p in model.mapping_logits.parameters():
            p.requires_grad_(False)
        for p in model.probs:
            p.requires_grad_(True)
        opt = torch.optim.Adam(model.probs.parameters(), lr=lr, weight_decay=1e-5)

    elif stage == "learn_mapping":
        model.markers.requires_grad_(False)
        for p in model.probs:
            p.requires_grad_(False)
        for p in model.mapping_logits.parameters():
            p.requires_grad_(True)
        if len(model.mapping_logits) == 0:
            raise ValueError("No ATAC mapping parameters were initialized.")
        opt = torch.optim.Adam(
            model.mapping_logits.parameters(), lr=lr, weight_decay=1e-5
        )

    else:
        raise ValueError(
            "stage must be 'learn_P', 'learn_mapping', or 'learn_M'"
        )

    if resume_path is not None:
        resumed_stage = ckpt.get("training_stage", ckpt.get("stage"))
        if resumed_stage == stage and "optimizer_state_dict" in ckpt:
            opt.load_state_dict(ckpt["optimizer_state_dict"])
            print(f"[resume] restored optimizer state for {stage}")

    loss_fn = LowDimClustering(
        stage=stage,
        margin=margin,
        lambdas=lambdas,
        batch_names=batch_names,
        cellbulk_chunk_size=cellbulk_chunk_size,
    )

    input_cache = prepare_input_cache(
        rnadata,
        device=device,
        batch_key=batch_key,
        rna_layer=rna_layer,
        cache_rna_on_device=cache_rna_on_device,
        batch_devices=batch_devices,
    )
    if stage == "learn_P":
        with torch.no_grad():
            for i, cached in enumerate(input_cache):
                if cached["modality"] != "atac":
                    continue
                X_gene = model.atac_cell_gene_from_source(
                    cached["X_peak_source"], batch_id=i
                )
                cached["X_for_cell_loss"] = X_gene
                print(
                    f"[cellbulk] ATAC batch {i}: mapped cells to gene space "
                    f"shape={X_gene.shape}, nnz={X_gene.nnz}"
                )
    fixed_rna_bulk_cache = None
    fixed_tfidf_cache = None
    fixed_pack = None
    if stage in ("learn_M", "learn_mapping"):
        fixed = build_fixed_bulk_caches(
            model,
            input_cache,
            device=device,
            rna_chunk_size=rna_chunk_size,
            batch_devices=batch_devices,
        )
        fixed_rna_bulk_cache = fixed["fixed_rna_bulk_cache"]
        fixed_tfidf_cache = fixed["fixed_tfidf_cache"]
    if stage == "learn_M":
        with torch.no_grad():
            fixed_pack = build_lists(
                rnadata,
                model,
                device,
                batch_key=batch_key,
                rna_layer=rna_layer,
                input_cache=input_cache,
                fixed_rna_bulk_cache=fixed_rna_bulk_cache,
                fixed_tfidf_cache=fixed_tfidf_cache,
                rna_chunk_size=rna_chunk_size,
                batch_devices=batch_devices,
            )

    def train_one_step(verbose: bool = True) -> Dict[str, float]:
        model.train()
        pack = build_lists(
            rnadata,
            model,
            device,
            batch_key=batch_key,
            rna_layer=rna_layer,
            input_cache=input_cache,
            fixed_rna_bulk_cache=fixed_rna_bulk_cache,
            fixed_tfidf_cache=fixed_tfidf_cache,
            fixed_pack=fixed_pack,
            rna_chunk_size=rna_chunk_size,
            batch_devices=batch_devices,
        )

        active_mask_dict = build_active_mask_from_probs(
            colsum_list=pack["colsum_list"],
            stage=stage,
            batch_names=batch_names,
            bulknum=bulknum,
            slot2proto=slot2proto,
            cellnum_list=cellnum_list,
            present_eps=present_eps,
        )
        if verbose:
            print("present slots:", {bn: int(active_mask_dict[bn].sum().item()) for bn in batch_names})

        ent_loss = torch.tensor(0.0, device=device)
        if stage == "learn_P":
            for i in range(len(rnadata)):
                probs_soft = pack["probs_soft_list"][i]
                Ki = probs_soft.shape[1]

                loss1_1 = -(probs_soft * torch.log2(probs_soft + EPS)).sum(dim=1).mean() / np.log2(Ki)
                probs_mean = probs_soft.mean(dim=0)
                loss1_2 = 1.0 + (probs_mean * torch.log2(probs_mean + EPS)).sum() / np.log2(Ki)

                ent_loss = ent_loss + loss1_1 + loss1_2
                if verbose:
                    print(f"entropy: {loss1_1.item():.6f}, {loss1_2.item():.6f}")

        markers_binary = pack["markers_binary"]
        markers_norm = pack["markers_norm"]
        markers_prob = pack["markers_prob"]
        if stage == "learn_M":
            # Only marker optimization uses continuous marker weights.  Other
            # stages keep the original hard-binary marker objective.
            markers_norm = markers_prob / (
                markers_prob.sum(dim=0, keepdim=True) + EPS
            )

        if stage == "learn_M" or verbose:
            l1_markers = markers_binary.abs().sum()
            l2_markers = (markers_norm.pow(2)).sum()
            ent_markers = 1.0 + (
                (markers_norm * torch.log2(markers_norm + EPS)).sum(dim=0).mean()
            ) / np.log2(G)
        else:
            l1_markers = markers_norm.sum() * 0.0
            l2_markers = markers_norm.sum() * 0.0
            ent_markers = markers_norm.sum() * 0.0

        if verbose:
            print(
                f"l1_markers:{l1_markers.item():.6f} "
                f"l2_markers:{l2_markers.item():.6f} "
                f"ent_markers:{ent_markers.item():.6f}"
            )

        total_loss, cellbulk_dist, inbatch_m, contrast_m = loss_fn(
            pack["bulks_list"],
            pack["probs_norm_list"],
            pack["X_list"],
            pack["unprobs_norm_list"],
            markers_norm,
            active_mask_dict,
        )

        if stage == "learn_mapping":
            distance_loss, mapping_batch_loss = model.mapping_regularization()
        else:
            zero = markers_norm.sum() * 0.0
            distance_loss, mapping_batch_loss = zero, zero

        if stage == "learn_M":
            marker_sparse_loss = markers_prob.mean()
        else:
            marker_sparse_loss = markers_prob.sum() * 0.0

        loss = total_loss
        if stage == "learn_P":
            loss = loss + ent_loss
        elif stage == "learn_mapping":
            loss = (
                loss
                + mapping_lambdas[0] * distance_loss
                + mapping_lambdas[1] * mapping_batch_loss
            )
        elif stage == "learn_M":
            loss = loss + ent_markers + marker_sparsity * marker_sparse_loss

        opt.zero_grad()
        loss.backward()

        if model.markers.requires_grad and model.markers.grad is not None:
            if not torch.isfinite(model.markers.grad).all():
                raise RuntimeError("markers.grad contains NaN/Inf")

        if stage == "learn_P":
            for bi, p in enumerate(model.probs):
                if p.grad is not None and (not torch.isfinite(p.grad).all()):
                    raise RuntimeError(f"probs[{bi}].grad contains NaN/Inf")
        if stage == "learn_mapping":
            for key, p in model.mapping_logits.items():
                if p.grad is not None and (not torch.isfinite(p.grad).all()):
                    raise RuntimeError(
                        f"mapping_logits[{key}].grad contains NaN/Inf"
                    )

        opt.step()

        if verbose:
            print(
                f"loss={loss.item():.6f}  "
                f"cellbulk={cellbulk_dist.item():.6f}  "
                f"inbatch={inbatch_m.item():.6f}  "
                f"contrast={contrast_m.item():.6f}"
                f"  map_distance={distance_loss.item():.6f}"
                f"  map_batch={mapping_batch_loss.item():.6f}"
                f"  marker_sparse={marker_sparse_loss.item():.6f}"
            )

        return {
            "loss": float(loss.item()),
            "cellbulk": float(cellbulk_dist.item()),
            "inbatch": float(inbatch_m.item()),
            "contrast": float(contrast_m.item()),
            "mapping_distance": float(distance_loss.item()),
            "mapping_batch": float(mapping_batch_loss.item()),
            "marker_sparsity": float(marker_sparse_loss.item()),
        }

    print(f"stage={stage}")
    if P_init is not None:
        print(f"P_init batches: {len(P_init)}")
    if M_init is not None:
        print(f"M_init shape: {tuple(M_init.shape)}")

    history = []
    loss0 = 1e-6
    for ep in range(epochs):
        verbose = (
            verbose_every is None
            or verbose_every <= 1
            or ep == 0
            or ep == epochs - 1
            or (ep + 1) % verbose_every == 0
        )
        if verbose:
            print(f"\nAlt-Epoch {ep + 1}/{epochs}\n-------------------------------")
        metrics = train_one_step(verbose=verbose)
        history.append(metrics)
        # if abs(metrics['loss'] - loss0) < 1e-6:
        #     break
        loss0 = metrics['loss']

    model.eval()
    pack = build_lists(
        rnadata,
        model,
        device,
        batch_key=batch_key,
        rna_layer=rna_layer,
        input_cache=input_cache,
        fixed_rna_bulk_cache=fixed_rna_bulk_cache,
        fixed_tfidf_cache=fixed_tfidf_cache,
        fixed_pack=fixed_pack,
        rna_chunk_size=rna_chunk_size,
        batch_devices=batch_devices,
    )

    if one_hot_start:
        probs_soft = [model.probs[i].detach().cpu() for i in range(len(model.probs))]
    else:
        probs_soft = [torch.softmax(model.probs[i], dim=1).detach().cpu() for i in range(len(model.probs))]

    out = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": opt.state_dict(),
        "training_stage": stage,
        "bulknum": bulknum,
        "G": G,
        "batch_names": batch_names,
        "probs_soft": probs_soft,
        "M": pack["markers_binary"].detach().cpu(),
        "M_prob": pack["markers_prob"].detach().cpu(),
        "M_norm": (
            pack["markers_binary"]
            / (pack["markers_binary"].sum(dim=0, keepdim=True) + EPS)
        ).detach().cpu(),
        "mapping_weights": {
            int(key): F.softplus(value).detach().cpu()
            for key, value in model.mapping_logits.items()
        },
        "mapping_init": mapping_init,
        "gene_names": gene_names,
        "batch_key": batch_key,
        "bulks_list": [x.detach().cpu() for x in pack["bulks_list"]],
        "history": history,
    }
    return out

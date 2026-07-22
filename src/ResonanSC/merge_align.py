from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment

from ResonanSC.mapping import mapping_init_from_weights

def inbatch_align(matrix, m=0.90):
    """
    Merge columns within one batch based on row-pattern consistency of thresholded corr matrix.
    """
    K = matrix.shape[0]
    mask = (matrix > m)
    parent = np.arange(K)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    edges = np.argwhere(np.triu(mask, 1))
    for i, j in edges:
        if np.array_equal(mask[i], mask[j]):
            union(i, j)

    groups_dict = {}
    for i in range(K):
        r = find(i)
        groups_dict.setdefault(r, []).append(i)

    groups = list(groups_dict.values())
    rep = {i: min(g) for g in groups for i in g}
    new_labels = np.array([rep[i] for i in range(K)], dtype=int)
    return new_labels, groups, mask


def inbatch_merge(p_list, groups, M=None, device="cpu"):
    """
    Merge subtype columns within batch.

    Parameters
    ----------
    p_list : list[Tensor]
        Each element has shape [n_cells_i, K_i]
    groups : list[list[list[int]]]
        groups[i] is the merge grouping for batch i
    M : Tensor or None
        Shared marker matrix [G, K] or None

    Returns
    -------
    p_merge : list[Tensor]
    M_merge : list[Tensor] or None
    """
    p_merge = []
    M_merge = [] if M is not None else None

    for i in range(len(p_list)):
        p_old = p_list[i].to(device)
        Knew = len(groups[i])

        p_new = torch.zeros(p_old.shape[0], Knew, device=device, dtype=p_old.dtype)
        for j, group in enumerate(groups[i]):
            p_new[:, j] = p_old[:, group].sum(dim=1)
        p_merge.append(p_new)

        if M is not None:
            M_i = M[i] if isinstance(M, (list, tuple)) else M
            M_i = M_i.to(device)
            m_new = torch.zeros(M_i.shape[0], Knew, device=device, dtype=M_i.dtype)
            for j, group in enumerate(groups[i]):
                m_new[:, j] = M_i[:, group].mean(dim=1)
            M_merge.append(m_new)

    return p_merge, M_merge


def write_p_labels_by_obs_names(
    data_dict, p_list, adata_all, key, batch_key=None
):
    """Write hard labels from per-batch P matrices back by observation names.

    This avoids assuming that concatenating labels in ``data_dict`` order has
    the same cell order as ``adata_all``.  When barcodes are reused across
    batches, ``batch_key`` and the observation name form a composite key.

    Parameters
    ----------
    batch_key : str or None
        Observation column identifying batches.  If omitted, ``"batch"`` is
        selected automatically when it exists in every input and in
        ``adata_all``.  A batch key is required only when observation names
        are not globally unique.
    """
    label_pieces = []

    items = (
        sorted(data_dict.items())
        if isinstance(data_dict, dict)
        else list(enumerate(data_dict))
    )
    for i, adata in items:
        hard = torch.argmax(p_list[i], dim=1).detach().cpu().numpy()
        if len(hard) != adata.n_obs:
            raise ValueError(
                f"Batch {i}: label length {len(hard)} != n_obs {adata.n_obs}."
            )

        labels = hard.astype(str)
        adata.obs[key] = labels
        piece = pd.DataFrame(
            {"_obs_name": np.asarray(adata.obs_names), key: labels}
        )
        if batch_key is not None:
            if batch_key not in adata.obs:
                raise KeyError(f"Batch {i} has no obs column {batch_key!r}.")
            piece["_batch"] = np.asarray(adata.obs[batch_key])
        label_pieces.append(piece)

    labels_frame = pd.concat(label_pieces, ignore_index=True)

    if batch_key is None and "batch" in adata_all.obs and all(
        "batch" in adata.obs for _, adata in items
    ):
        batch_key = "batch"
        labels_frame["_batch"] = np.concatenate(
            [np.asarray(adata.obs[batch_key]) for _, adata in items]
        )

    if batch_key is None:
        labels_all = labels_frame.set_index("_obs_name")[key]
        if labels_all.index.has_duplicates:
            dup = (
                labels_all.index[labels_all.index.duplicated()]
                .unique()[:10]
                .tolist()
            )
            raise ValueError(
                f"Duplicated obs_names when writing {key}: {dup}. "
                "Pass batch_key so labels can be matched by "
                "(batch, obs_name)."
            )
        target_index = adata_all.obs_names
    else:
        if batch_key not in adata_all.obs:
            raise KeyError(f"adata_all has no obs column {batch_key!r}.")
        labels_all = labels_frame.set_index(["_batch", "_obs_name"])[key]
        if labels_all.index.has_duplicates:
            dup = labels_all.index[labels_all.index.duplicated()].unique()[:10].tolist()
            raise ValueError(
                f"Duplicated ({batch_key}, obs_name) keys when writing "
                f"{key}: {dup}"
            )
        target_index = pd.MultiIndex.from_arrays(
            [np.asarray(adata_all.obs[batch_key]), np.asarray(adata_all.obs_names)],
            names=["_batch", "_obs_name"],
        )

    aligned = labels_all.reindex(target_index)

    # ``anndata.concat(..., index_unique="-")`` appends a dataset suffix to
    # duplicated observation names (for example ``barcode-1`` becomes
    # ``barcode-1-1``).  Recover the original names within each batch, but
    # only when one uniform number of trailing ``-...`` components gives a
    # complete, one-to-one match.  This avoids guessing for partial matches.
    if aligned.isna().any() and batch_key is not None:
        target_batches = np.asarray(adata_all.obs[batch_key])
        target_names = np.asarray(adata_all.obs_names).astype(str)

        for batch in pd.unique(target_batches):
            positions = np.flatnonzero(target_batches == batch)
            if not aligned.iloc[positions].isna().any():
                continue

            source = labels_all.xs(batch, level="_batch")
            normalized = target_names[positions].tolist()
            for _ in range(3):
                normalized_index = pd.Index(normalized)
                if (
                    normalized_index.is_unique
                    and normalized_index.isin(source.index).all()
                ):
                    aligned.iloc[positions] = source.loc[normalized_index].values
                    break
                normalized = [
                    name.rsplit("-", 1)[0] if "-" in name else name
                    for name in normalized
                ]

    if aligned.isna().any():
        missing = target_index[aligned.isna()]
        examples = missing[:10].tolist()
        raise ValueError(
            f"{key}: {len(missing)} cells in adata_all are missing labels; "
            f"examples: {examples}"
        )

    adata_all.obs[key] = aligned.astype(str).values
    return labels_all


def align_clusters_hungarian(S, cap=3, null_score=-0.5):
    S = np.nan_to_num(np.array(S, float), nan=-1.0)
    Krow, Kcol = S.shape

    row_slots = np.repeat(np.arange(Krow), cap)
    col_slots = np.repeat(np.arange(Kcol), cap)
    R, C = len(row_slots), len(col_slots)

    S_exp = S[row_slots][:, col_slots]
    N = R + C
    S_sq = null_score * np.ones((N, N))
    S_sq[:R, :C] = S_exp

    rr, cc = linear_sum_assignment(-S_sq)

    row_to_cols = {i: [] for i in range(Krow)}
    col_to_rows = {j: [] for j in range(Kcol)}
    for r, c in zip(rr, cc):
        if r < R and c < C:
            i = int(row_slots[r])
            j = int(col_slots[c])
            row_to_cols[i].append(j)
            col_to_rows[j].append(i)

    for i in row_to_cols:
        row_to_cols[i] = sorted(set(row_to_cols[i]))
    for j in col_to_rows:
        col_to_rows[j] = sorted(set(col_to_rows[j]))

    return row_to_cols, col_to_rows


class UnionFind:
    def __init__(self, n):
        self.parent = np.arange(n)

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra

    def components(self):
        groups = {}
        for i in range(len(self.parent)):
            r = self.find(i)
            groups.setdefault(r, []).append(i)
        return list(groups.values())


def build_connected_components_label_map(uf, offsets, K_list):
    total_nodes = sum(K_list)
    comps_raw = uf.components()

    comps = []
    for comp in comps_raw:
        comp = sorted([int(x) for x in comp if 0 <= int(x) < total_nodes])
        if len(comp) > 0:
            comps.append(comp)
    comps = sorted(comps, key=lambda xs: min(xs))

    label_map = {}
    for new_label, comp in enumerate(comps):
        for node in comp:
            label_map[int(node)] = int(new_label)

    return comps, label_map


def build_p_align(p_merge, batch_names, label_map, comps, offsets, device="cpu"):
    K_new = max(label_map.values()) + 1

    p_align = []
    proto_of_cluster = {}
    counts_per_batch_proto = []

    for bi, bname in enumerate(batch_names):
        K_i = p_merge[bi].shape[1]
        p_old = p_merge[bi].to(device)
        Ncells, K_old = p_old.shape
        off = offsets[bi]

        p_new = torch.zeros(Ncells, K_new, device=device, dtype=p_old.dtype)

        proto = []
        for k in range(K_old):
            node = off + k
            if node not in label_map:
                raise ValueError(f"Missing label_map for batch {bname}, cluster {k}")
            proto.append(label_map[node])
            new_k = label_map[node]
            p_new[:, new_k] += p_old[:, k]

        p_align.append(p_new)

        proto = torch.tensor(proto, dtype=torch.long, device=device)
        proto_of_cluster[bname] = proto.detach().cpu()
        counts_per_batch_proto.append(torch.bincount(proto, minlength=K_new))

    counts_per_batch_proto = torch.stack(counts_per_batch_proto, dim=0)
    slot2proto = torch.arange(K_new, dtype=torch.long)

    align_masks = {
        "batch_keys": list(batch_names),
        "prototype_count": int(K_new),
        "bulknum": int(K_new),
        "slot2proto": slot2proto.detach().cpu(),
        "proto_of_cluster": proto_of_cluster,
        "counts_per_batch_proto": counts_per_batch_proto.detach().cpu(),
        "maxc_per_proto": torch.ones(K_new, dtype=torch.long),
        "label_map": {int(k): int(v) for k, v in label_map.items()},
        "groups_cc": [[int(x) for x in comp] for comp in comps],
    }
    return p_align, align_masks


def build_M_align(
    M_list,
    label_map,
    offsets,
    device="cpu",
    eps=1e-8,
    normalize=True,
):
    if len(label_map) == 0:
        raise ValueError("label_map is empty.")

    M_list = [M.to(device) for M in M_list]
    G = M_list[0].shape[0]
    dtype = M_list[0].dtype

    for i, M in enumerate(M_list):
        if M.shape[0] != G:
            raise ValueError(f"M_list[{i}] has inconsistent gene dimension.")

    K_new = max(label_map.values()) + 1
    M_align = torch.zeros((G, K_new), device=device, dtype=dtype)
    counts = torch.zeros(K_new, device=device, dtype=dtype)

    for bi, M in enumerate(M_list):
        K_i = M.shape[1]
        off = offsets[bi]

        for local_k in range(K_i):
            node = off + local_k
            if node not in label_map:
                continue
            new_k = label_map[node]
            M_align[:, new_k] += M[:, local_k]
            counts[new_k] += 1.0

    M_align = M_align / (counts.unsqueeze(0) + eps)

    if normalize:
        M_align = M_align / (M_align.sum(dim=0, keepdim=True) + eps)

    return M_align

def run_inbatch_merge(inbatch_corrs, p_list, thresholds, M=None, device="cpu"):
    new_labels = []
    groups = []
    masks = []

    for i, matrix in enumerate(inbatch_corrs):
        l, g, mask = inbatch_align(matrix, m=thresholds[i])
        new_labels.append(l)
        groups.append(g)
        masks.append(mask)

    p_merge, M_merge = inbatch_merge(p_list, groups, M=M, device=device)

    return {
        "new_labels": new_labels,
        "groups": groups,
        "masks": masks,
        "p_merge": p_merge,
        "M_merge": M_merge,
    }


def build_merge_only_init(
    p_merge, M_merge, batch_names, device="cpu"
):
    """Treat current merged label indices as the final aligned labels.

    Local merged label ``k`` in every batch maps to global label ``k``.  When
    batches have different merged label counts, shorter P matrices are padded
    to ``max(K_i)``.  M columns with the same current label are averaged across
    the batches in which that label exists.

    P remains a probability matrix, matching the standard initialization
    checkpoint schema and the output of ``run_cross_batch_align``.
    """
    batch_names = list(map(str, batch_names))
    if len(p_merge) != len(batch_names):
        raise ValueError("p_merge and batch_names must have the same length.")
    if M_merge is None or len(M_merge) != len(p_merge):
        raise ValueError("M_merge must contain one matrix per P_merge batch.")
    if not p_merge:
        raise ValueError("p_merge must not be empty.")

    K_list = [int(p.shape[1]) for p in p_merge]
    offsets = []
    offset = 0
    for K_i in K_list:
        offsets.append(offset)
        offset += K_i
    K_global = max(K_list)

    p_init = []
    proto_of_cluster = {}
    counts_per_batch_proto = torch.zeros(
        (len(p_merge), K_global), dtype=torch.long
    )
    for bi, (batch_name, p_old) in enumerate(zip(batch_names, p_merge)):
        p_old = p_old.detach().to(device)
        p_new = torch.zeros(
            (p_old.shape[0], K_global), dtype=p_old.dtype, device=device
        )
        start = 0
        stop = K_list[bi]
        p_new[:, start:stop] = p_old
        p_init.append(p_new.detach().cpu())

        local_slots = torch.arange(K_list[bi], dtype=torch.long)
        proto_of_cluster[batch_name] = local_slots
        counts_per_batch_proto[bi, :K_list[bi]] = 1

    gene_count = int(M_merge[0].shape[0])
    marker_dtype = M_merge[0].dtype
    M_init = torch.zeros((gene_count, K_global), dtype=marker_dtype)
    M_counts = torch.zeros(K_global, dtype=marker_dtype)
    for bi, m_old in enumerate(M_merge):
        m_old = m_old.detach().cpu()
        expected = (gene_count, K_list[bi])
        if tuple(m_old.shape) != expected:
            raise ValueError(
                f"M_merge[{bi}] has shape {tuple(m_old.shape)}; expected {expected}."
            )
        M_init[:, :K_list[bi]] += m_old
        M_counts[:K_list[bi]] += 1
    M_init = M_init / M_counts.clamp_min(1).unsqueeze(0)

    slot2proto = torch.arange(K_global, dtype=torch.long)
    label_map = {
        offsets[bi] + local_k: local_k
        for bi, K_i in enumerate(K_list)
        for local_k in range(K_i)
    }
    groups_cc = [
        [
            offsets[bi] + global_k
            for bi, K_i in enumerate(K_list)
            if global_k < K_i
        ]
        for global_k in range(K_global)
    ]
    align_masks = {
        "batch_keys": batch_names,
        "prototype_count": K_global,
        "bulknum": K_global,
        "slot2proto": slot2proto,
        "proto_of_cluster": proto_of_cluster,
        "counts_per_batch_proto": counts_per_batch_proto,
        "maxc_per_proto": torch.ones(K_global, dtype=torch.long),
        "label_map": label_map,
        "groups_cc": groups_cc,
        "merge_only": True,
        "alignment_rule": "same_merged_label_index",
        "batch_offsets": offsets,
        "batch_cluster_counts": K_list,
    }
    return {
        "P_align": p_init,
        "M_align": M_init,
        "align_masks": align_masks,
        "offsets": offsets,
        "K_list": K_list,
        "P_init_is_logits": False,
    }


def save_merge_only_checkpoint(
    output_path,
    p_merge,
    M_merge,
    train_out,
    batch_names,
    source_checkpoint=None,
    batch_key=None,
    training_data_dir=None,
    mapping_max_distance=None,
):
    """Save merge output as a complete init checkpoint without alignment.

    ``source_checkpoint`` must be the init checkpoint loaded at the start of
    the current round (or its path). Its complete context is carried forward,
    while learned state is replaced from ``train_out`` and the merge result.
    """
    def _load_checkpoint(value):
        if value is None:
            return {}
        if isinstance(value, (str, Path)):
            return torch.load(value, map_location="cpu", weights_only=False)
        return dict(value)

    source_checkpoint = _load_checkpoint(source_checkpoint)
    if not source_checkpoint:
        raise ValueError(
            "source_checkpoint is required and must be the init checkpoint "
            "used as input to the current training round."
        )
    merge_init = build_merge_only_init(
        p_merge=p_merge,
        M_merge=M_merge,
        batch_names=batch_names,
        device="cpu",
    )

    required_train_keys = {"mapping_init", "mapping_weights", "gene_names"}
    missing_train = sorted(required_train_keys.difference(train_out))
    if missing_train:
        raise KeyError(f"train_out is missing required keys: {missing_train}")

    mapping_init = mapping_init_from_weights(
        train_out["mapping_init"], train_out["mapping_weights"]
    )
    if int(merge_init["M_align"].shape[0]) != len(train_out["gene_names"]):
        raise ValueError(
            "M_align gene dimension does not match train_out['gene_names']: "
            f"{merge_init['M_align'].shape[0]} != {len(train_out['gene_names'])}."
        )
    batch_key = batch_key or train_out.get(
        "batch_key", source_checkpoint.get("batch_key")
    )
    training_data_dir = training_data_dir or source_checkpoint.get(
        "training_data_dir"
    )
    if mapping_max_distance is None:
        mapping_max_distance = source_checkpoint.get(
            "mapping_max_distance",
            source_checkpoint.get("dap_deg_window", 250_000),
        )

    # Start from this round's input checkpoint, then replace state that changed
    # during training/merge. This preserves the full round-to-round lineage
    # without reverting metadata to the original initialization.
    checkpoint = {
        **source_checkpoint,
        "P_init": merge_init["P_align"],
        "P_merge": [p.detach().cpu() for p in p_merge],
        "M_merge": [m.detach().cpu() for m in M_merge],
        **merge_init,
        "M_align": (merge_init["M_align"] >= 0.5).to(merge_init["M_align"].dtype),
        "M_prob_align": merge_init["M_align"],
        "mapping_init": mapping_init,
        "mapping_init_before_training": train_out["mapping_init"],
        "mapping_weights": {
            int(k): v.detach().cpu()
            for k, v in train_out["mapping_weights"].items()
        },
        "gene_names": train_out["gene_names"],
        "batch_key": batch_key,
        "batch_names": list(map(str, batch_names)),
        "training_batch_order": list(map(str, batch_names)),
        "training_data_dir": (
            str(training_data_dir) if training_data_dir is not None else None
        ),
        "mapping_max_distance": mapping_max_distance,
        "dap_deg_window": mapping_max_distance,
        "cross_batch_aligned": False,
        "checkpoint_type": "merge_only_init",
    }

    required = {
        "P_align", "M_align", "align_masks", "mapping_init", "gene_names",
        "P_init", "P_merge", "batch_key", "batch_names",
        "training_batch_order", "training_data_dir",
        "mapping_max_distance", "dap_deg_window",
    }
    missing = sorted(k for k in required if checkpoint.get(k) is None)
    if missing:
        raise ValueError(
            "Refusing to save an incomplete next-round init checkpoint; "
            f"missing operational fields: {missing}."
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)
    return checkpoint


def manual_inbatch_merge(
    p_list,
    batch_names,
    merge_maps,
    M=None,
    device="cpu",
):
    """Merge P/M columns from explicit per-batch label maps.

    ``merge_maps[batch_name]`` maps every old integer P-column index to a
    batch-local merged label. Labels may be strings or integers.
    """
    if len(p_list) != len(batch_names):
        raise ValueError("p_list and batch_names must have the same length.")

    groups = []
    merged_labels = {}
    new_labels = []

    for bi, batch_name in enumerate(map(str, batch_names)):
        if batch_name not in merge_maps:
            raise KeyError(f"merge_maps is missing batch {batch_name!r}.")

        K_old = p_list[bi].shape[1]
        merge_map = {int(k): str(v) for k, v in merge_maps[batch_name].items()}
        expected = set(range(K_old))
        observed = set(merge_map)
        if observed != expected:
            raise ValueError(
                f"{batch_name}: merge map must cover columns 0..{K_old - 1}; "
                f"missing={sorted(expected - observed)}, "
                f"extra={sorted(observed - expected)}"
            )

        labels_i = list(dict.fromkeys(merge_map[k] for k in range(K_old)))
        groups_i = [
            [k for k in range(K_old) if merge_map[k] == label]
            for label in labels_i
        ]
        local_index = {label: k for k, label in enumerate(labels_i)}

        groups.append(groups_i)
        merged_labels[batch_name] = labels_i
        new_labels.append(
            np.asarray([local_index[merge_map[k]] for k in range(K_old)], dtype=int)
        )

    p_merge, M_merge = inbatch_merge(
        p_list=p_list,
        groups=groups,
        M=M,
        device=device,
    )
    return {
        "new_labels": new_labels,
        "groups": groups,
        "masks": [None] * len(groups),
        "merged_labels": merged_labels,
        "p_merge": p_merge,
        "M_merge": M_merge,
    }


def manual_cross_batch_align(
    p_merge,
    batch_names,
    merged_labels,
    align_maps,
    M_merge=None,
    global_label_order=None,
    device="cpu",
):
    """Align merged P/M columns from explicit global-label assignments."""
    batch_names = list(map(str, batch_names))
    if len(p_merge) != len(batch_names):
        raise ValueError("p_merge and batch_names must have the same length.")

    normalized_maps = {}
    all_global_labels = []
    for bi, batch_name in enumerate(batch_names):
        if batch_name not in merged_labels:
            raise KeyError(f"merged_labels is missing batch {batch_name!r}.")
        if batch_name not in align_maps:
            raise KeyError(f"align_maps is missing batch {batch_name!r}.")

        local_labels = list(map(str, merged_labels[batch_name]))
        if len(local_labels) != p_merge[bi].shape[1]:
            raise ValueError(
                f"{batch_name}: {len(local_labels)} merged labels for "
                f"{p_merge[bi].shape[1]} P columns."
            )

        align_map = {str(k): str(v) for k, v in align_maps[batch_name].items()}
        if set(align_map) != set(local_labels):
            raise ValueError(
                f"{batch_name}: align map mismatch; "
                f"missing={sorted(set(local_labels) - set(align_map))}, "
                f"extra={sorted(set(align_map) - set(local_labels))}"
            )
        normalized_maps[batch_name] = align_map
        all_global_labels.extend(align_map[label] for label in local_labels)

    observed_global = set(all_global_labels)
    if global_label_order is None:
        global_labels = list(dict.fromkeys(all_global_labels))
    else:
        global_labels = list(map(str, global_label_order))
        if len(global_labels) != len(set(global_labels)):
            raise ValueError("global_label_order contains duplicate labels.")
        if set(global_labels) != observed_global:
            raise ValueError(
                "global_label_order mismatch: "
                f"missing={sorted(observed_global - set(global_labels))}, "
                f"extra={sorted(set(global_labels) - observed_global)}"
            )

    global_to_idx = {label: i for i, label in enumerate(global_labels)}
    K_list = [p.shape[1] for p in p_merge]
    offsets = []
    offset = 0
    for K_i in K_list:
        offsets.append(offset)
        offset += K_i

    label_map = {}
    for bi, batch_name in enumerate(batch_names):
        local_labels = list(map(str, merged_labels[batch_name]))
        align_map = normalized_maps[batch_name]
        for local_k, local_label in enumerate(local_labels):
            label_map[offsets[bi] + local_k] = global_to_idx[align_map[local_label]]

    components = [
        sorted(node for node, global_k in label_map.items() if global_k == k)
        for k in range(len(global_labels))
    ]
    p_align, align_masks = build_p_align(
        p_merge=p_merge,
        batch_names=batch_names,
        label_map=label_map,
        comps=components,
        offsets=offsets,
        device=device,
    )
    align_masks["global_labels"] = global_labels

    M_align = None
    if M_merge is not None:
        M_align = build_M_align(
            M_list=M_merge,
            label_map=label_map,
            offsets=offsets,
            device=device,
            normalize=False,
        )

    return {
        "p_align": p_align,
        "M_align": M_align,
        "align_masks": align_masks,
        "offsets": offsets,
        "components": components,
        "label_map": label_map,
        "global_labels": global_labels,
        "edges_list": [],
    }

def run_cross_batch_align(
    p_merge,
    batch_names,
    cross_corrs,
    cap=3,
    null_score=-1.0,
    device="cpu",
):
    """Align adjacent batches from cross-correlation matrices.

    ``cross_corrs[t]`` must have shape ``(K_t, K_t1)``: rows are clusters in
    ``batch_names[t]`` and columns are clusters in ``batch_names[t + 1]``.
    """

    K_list = [p.shape[1] for p in p_merge]
    offsets = [0]

    print("K_list:", K_list)
    print("sum(K_list):", sum(K_list))

    for k in K_list[:-1]:
        offsets.append(offsets[-1] + k)
    print("offsets:", offsets)

    for t in range(len(cross_corrs)):
        expected = (K_list[t], K_list[t + 1])
        observed = tuple(np.asarray(cross_corrs[t]).shape)
        if observed != expected:
            reverse = (expected[1], expected[0])
            if observed == reverse and expected[0] != expected[1]:
                raise ValueError(
                    f"cross_corrs[{t}] has reversed shape {observed}; "
                    f"expected rows={batch_names[t]} ({expected[0]}), "
                    f"cols={batch_names[t + 1]} ({expected[1]}). "
                    "Transpose this matrix before run_cross_batch_align."
                )
            raise ValueError(
                f"cross_corrs[{t}] shape {observed} does not match expected "
                f"{expected} for {batch_names[t]} vs {batch_names[t + 1]}."
            )
        print(f"\n[t={t}] {batch_names[t]} vs {batch_names[t+1]}")
        print("cross_corrs shape:", cross_corrs[t].shape)
        print(f"K_t={K_list[t]}, K_t1={K_list[t+1]}")


        
    uf = UnionFind(sum(K_list))

    edges_list = []
    for t in range(len(p_merge) - 1):

        row_to_cols, col_to_rows = align_clusters_hungarian(
            cross_corrs[t], cap=cap, null_score=null_score
        )
        print(f'{batch_names[t]} vs {batch_names[t+1]} row to col: {row_to_cols} col to rows:{col_to_rows}')
        edges_list.append((row_to_cols, col_to_rows))

        offA = offsets[t]
        offB = offsets[t + 1]

        # r indexes batch A rows, c indexes batch B columns.
        for r, cols in row_to_cols.items():
            for c in cols:
                uf.union(offA + int(r), offB + int(c))

        print(f"[cross] {batch_names[t]} <- {batch_names[t+1]}: S shape={cross_corrs[t].shape} edges={sum(len(v) for v in row_to_cols.values())}")


    comps, label_map = build_connected_components_label_map(
        uf=uf,
        offsets=offsets,
        K_list=K_list,
    )

    print("[global] num aligned groups:", len(comps))

    p_align, align_masks = build_p_align(
        p_merge=p_merge,
        batch_names=batch_names,
        label_map=label_map,
        comps=comps,
        offsets=offsets,
        device=device,
    )

    print("[global] prototype_count:", align_masks["prototype_count"], "bulknum(K_global):", align_masks["bulknum"])
    print("[global] p_align shapes:", [tuple(p.shape) for p in p_align])

    return {
        "p_align": p_align,
        "align_masks": align_masks,
        "offsets": offsets,
        "components": comps,
        "label_map": label_map,
        "edges_list": edges_list,
    }

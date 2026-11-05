"""Algorithms extracted from the tuned alignment; no plotting or GT dependencies."""
import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import StandardScaler, normalize
import torch
from .core.bulk import get_bulk
from .core.corr import compute_crossbatch_corr_de, compute_inbatch_corr_de, de_1vAll_kbulk
from .core.merge_align import run_cross_batch_align, run_inbatch_merge

def scores_for_pair(data, labels, method):
    if method == "peak_lsi":
        values = []
        for x, p in zip(data, labels):
            weights = p.detach().cpu().numpy().astype(np.float32)
            weights /= np.maximum(weights.sum(axis=0, keepdims=True), 1e-8)
            values.append(weights.T @ x)
        within = [pearson_matrix(x, x) for x in values]
        cross = pearson_matrix(*values)
        if not np.isfinite(cross).all() or not all(np.isfinite(x).all() for x in within):
            raise ValueError("Non-finite peak LSI cluster correlation")
        return cross, within
    if method == "peak_tfidf":
        values = []
        for x, p in zip(data, labels):
            weights = p.detach().cpu().numpy().astype(np.float32)
            weights /= np.maximum(weights.sum(axis=0, keepdims=True), 1e-8)
            values.append(np.asarray((x.T @ weights).T, dtype=np.float32))
        within = [pearson_matrix(x, x) for x in values]
        cross = pearson_matrix(*values)
        if not np.isfinite(cross).all() or not all(np.isfinite(x).all() for x in within):
            raise ValueError("Non-finite peak TF-IDF cluster correlation")
        return cross, within
    bulk = get_bulk(data, labels)
    if method == "DE":
        de = [de_1vAll_kbulk(x, p) for x, p in zip(data, labels)]
        within = [compute_inbatch_corr_de(x, b, p, use_pairwise=False, de_1v=d)
                  for x, b, p, d in zip(data, bulk, labels, de)]
        cross = compute_crossbatch_corr_de(
            data[0], data[1], bulk[0], bulk[1], labels[0], labels[1],
            deA_1v=de[0], deB_1v=de[1],
        ).T
    else:
        values = [x.numpy().T for x in bulk]
        within = [pearson_matrix(x, x) for x in values]
        cross = pearson_matrix(*values)
    if not np.isfinite(cross).all() or not all(np.isfinite(x).all() for x in within):
        count = int((~np.isfinite(cross)).sum()) + sum(int((~np.isfinite(x)).sum()) for x in within)
        print('WARNING: undefined DE cluster correlations excluded from merging/alignment:', count, flush=True)
        cross = np.nan_to_num(cross, nan=-1., posinf=-1., neginf=-1.)
        within = [np.nan_to_num(x, nan=-1., posinf=-1., neginf=-1.) for x in within]
    return cross, within

def ambiguous_pairs(cross, within, side, min_within, min_cross, max_top_gap, max_profile_rmse):
    profiles = cross if side == 0 else cross.T
    rows = []
    for i in range(len(profiles)):
        for j in range(i + 1, len(profiles)):
            a, b = profiles[i], profiles[j]
            top_i, top_j = int(np.argmax(a)), int(np.argmax(b))
            same_target = top_i == top_j
            top_gap = float(abs(a[top_i] - b[top_j]))
            rmse = float(np.sqrt(np.mean((a - b) ** 2)))
            self_corr = float(within[side][i, j])
            min_best = float(min(a[top_i], b[top_j]))
            eligible = bool(same_target and self_corr >= min_within
                            and min_best >= min_cross and top_gap <= max_top_gap
                            and rmse <= max_profile_rmse)
            rows.append({"side": side, "cluster_i": i, "cluster_j": j,
                         "best_target_i": top_i, "best_target_j": top_j,
                         "within_corr": self_corr, "min_cross_corr": min_best,
                         "top_score_gap": top_gap, "profile_rmse": rmse,
                         "eligible": eligible, "selected": False})
    return rows

def choose_disjoint(rows):
    chosen, used = [], set()
    eligible = [row for row in rows if row["eligible"]]
    eligible.sort(key=lambda row: (-row["within_corr"], row["profile_rmse"], row["top_score_gap"]))
    for row in eligible:
        i, j = row["cluster_i"], row["cluster_j"]
        if i not in used and j not in used:
            row["selected"] = True
            used.update((i, j))
            chosen.append((i, j))
    return chosen

def merge_pairs(p, origins, pairs):
    if not pairs:
        return p, origins
    partner = {i: j for i, j in pairs}
    consumed = {j for _, j in pairs}
    groups = [[i, partner[i]] if i in partner else [i]
              for i in range(p.shape[1]) if i not in consumed]
    new_p = torch.stack([p[:, indices].sum(axis=1) for indices in groups], dim=1)
    new_origins = [sum((origins[i] for i in indices), []) for indices in groups]
    return new_p, new_origins

def refine_pair(data, initial, stage, method, args, atac_lsi=None):
    current = list(initial)
    origins = [[[i] for i in range(p.shape[1])] for p in current]
    audit = []
    if method in ("peak_tfidf", "peak_lsi"):
        thresholds = (args.atac_min_within, args.atac_min_cross,
                      args.atac_max_top_gap, args.atac_max_profile_rmse)
    elif method == "DE":
        thresholds = (args.min_within, args.min_cross, args.max_top_gap,
                      args.max_profile_rmse)
    else:
        thresholds = (args.cross_min_within, args.cross_min_cross,
                      args.cross_max_top_gap, args.cross_max_profile_rmse)
    for iteration in range(args.max_merge_rounds):
        cross, within = scores_for_pair(data, current, method)
        if atac_lsi is not None:
            if stage != 'cross_modality' or len(atac_lsi) != current[1].shape[0]:
                raise ValueError('ATAC LSI rows must match cross-modality labels')
            weights = current[1].detach().cpu().numpy().astype(np.float64)
            counts = weights.sum(axis=0)
            if (counts <= 0).any(): raise ValueError('Empty ATAC cluster in cross-modality stage')
            profiles = (weights.T @ atac_lsi) / counts[:, None]
            centered = profiles - profiles.mean(axis=1, keepdims=True)
            valid = np.linalg.norm(centered, axis=1) > 1e-12
            within[1] = pearson_matrix(profiles, profiles)
            within[1][~valid, :] = -1.0
            within[1][:, ~valid] = -1.0
        candidates = [ambiguous_pairs(cross, within, side, *thresholds)
                      for side in range(2)]
        selected = [choose_disjoint(rows) for rows in candidates]
        for side in range(2):
            for row in candidates[side]:
                row.update(stage=stage, method=method, within_metric="ATAC_peak_LSI_Pearson" if atac_lsi is not None and side == 1 else "gene_activity_Pearson" if atac_lsi is not None else method, iteration=iteration,
                           source_original_clusters_i=";".join(map(str, origins[side][row["cluster_i"]])),
                           source_original_clusters_j=";".join(map(str, origins[side][row["cluster_j"]])))
            audit.extend(candidates[side])
        print(stage, "round", iteration, "clusters", [p.shape[1] for p in current],
              "merges", selected, flush=True)
        if not any(selected):
            break
        current, origins = zip(*(merge_pairs(current[side], origins[side], selected[side])
                                 for side in range(2)))
        current, origins = list(current), list(origins)
    else:
        cross, _ = scores_for_pair(data, current, method)
    return current, cross, origins, audit

def load_pooled_peak_tfidf(path, batches, min_cells=20, space="peak_tfidf", n_lsi=30,
                           lsi_transform="standardized_drop_first"):
    """Build a common peak TF-IDF feature space for current ATAC batches."""
    peak_source = ad.read_h5ad(path, backed="r")
    if not peak_source.obs_names.is_unique or not peak_source.var_names.is_unique:
        raise ValueError("Peak matrix requires unique cell and peak names")
    matrices = []
    for batch in batches:
        indices = peak_source.obs_names.get_indexer(batch.obs_names)
        if (indices < 0).any():
            raise ValueError(f"{(indices < 0).sum()} cells missing from peak matrix")
        order = np.argsort(indices)
        matrix = peak_source.X[indices[order], :].tocsr()[np.argsort(order), :]
        matrices.append(matrix)
    peak_source.file.close()
    counts = sparse.vstack(matrices, format="csr")
    prevalence = np.asarray((counts > 0).sum(axis=0)).ravel()
    keep = (prevalence >= min_cells) & (prevalence <= int(.95 * counts.shape[0]))
    counts = counts[:, keep].tocsr()
    if counts.shape[1] < 32:
        raise ValueError("Too few peaks remain after prevalence filtering")
    tfidf = TfidfTransformer(norm="l2", use_idf=True, smooth_idf=True).fit_transform(counts)
    splits = np.cumsum([matrix.shape[0] for matrix in matrices[:-1]])
    print("Pooled ATAC peak TF-IDF:", tfidf.shape, "nnz", tfidf.nnz, flush=True)
    if space == "peak_lsi":
        n_components = n_lsi + 1 if lsi_transform == "standardized_drop_first" else n_lsi
        svd = TruncatedSVD(n_components=n_components, n_iter=7, random_state=42)
        lsi = svd.fit_transform(tfidf)
        if lsi_transform == "standardized_drop_first":
            lsi = StandardScaler().fit_transform(lsi[:, 1:])
        lsi = normalize(lsi).astype(np.float32)
        print("Pooled ATAC peak LSI:", lsi.shape, flush=True)
        return list(np.split(lsi, splits)), int(tfidf.shape[1])
    boundaries = [0] + splits.tolist() + [tfidf.shape[0]]
    return [tfidf[start:end].tocsr() for start, end in zip(boundaries[:-1], boundaries[1:])], int(tfidf.shape[1])

def adjacent_refine_and_align(data, initial, names, method, stage, args):
    current = list(initial)
    audit = []
    for i in range(len(current) - 1):
        refined, _, _, rows = refine_pair(data[i:i + 2], current[i:i + 2],
                                           f"{stage}_{names[i]}_{names[i + 1]}", method, args)
        current[i:i + 2] = refined
        audit.extend(rows)
    cross_corrs = [scores_for_pair(data[i:i + 2], current[i:i + 2], method)[0]
                   for i in range(len(current) - 1)]
    prefix = 'rna' if method == 'DE' else 'atac'
    aligned = run_cross_batch_align(
        current, names, cross_corrs,
        cap=getattr(args, f'{prefix}_align_cap', 3),
        null_score=getattr(args, f'{prefix}_align_min_score', -.5))
    return current, aligned["p_align"], cross_corrs, audit

def deferred_refine_and_align(data, initial, names, method, stage, args):
    """Delay within-batch merges until every same-modality batch has been checked.

    An ambiguous match to one batch proposes a merge. A confident, distinct
    match for the two source clusters in any other batch vetoes it. Missing or
    weak matches do neither. The downstream adjacent Hungarian aligner is
    deliberately unchanged.
    """
    current = list(initial)
    origins = [[[i] for i in range(p.shape[1])] for p in current]
    if method == "peak_lsi":
        thresholds = (args.atac_min_within, args.atac_min_cross,
                      args.atac_max_top_gap, args.atac_max_profile_rmse)
    else:
        thresholds = (args.min_within, args.min_cross,
                      args.max_top_gap, args.max_profile_rmse)
    audit = []
    for iteration in range(args.max_merge_rounds):
        evidence = [dict() for _ in current]
        for left in range(len(current)):
            for right in range(left + 1, len(current)):
                cross, within = scores_for_pair(
                    [data[left], data[right]],
                    [current[left], current[right]], method)
                for side, source, target in ((0, left, right), (1, right, left)):
                    profiles = cross if side == 0 else cross.T
                    candidates = ambiguous_pairs(cross, within, side, *thresholds)
                    for row in candidates:
                        i, j = row["cluster_i"], row["cluster_j"]
                        key = (i, j)
                        item = evidence[source].setdefault(key, {
                            "within_corr": float(row["within_corr"]),
                            "support": [], "distinguish": [],
                            "top_score_gap": float("inf"),
                            "profile_rmse": float("inf")})
                        if row["eligible"]:
                            item["support"].append(names[target])
                            item["top_score_gap"] = min(item["top_score_gap"], row["top_score_gap"])
                            item["profile_rmse"] = min(item["profile_rmse"], row["profile_rmse"])
                        a, b = profiles[i], profiles[j]
                        ai, bj = int(np.argmax(a)), int(np.argmax(b))
                        if ai != bj and min(a[ai], b[bj]) >= thresholds[1]:
                            margin_a = a[ai] - np.partition(a, -2)[-2] if len(a) > 1 else np.inf
                            margin_b = b[bj] - np.partition(b, -2)[-2] if len(b) > 1 else np.inf
                            if min(margin_a, margin_b) >= args.deferred_min_match_margin:
                                item["distinguish"].append(names[target])
        selected = []
        for batch_index, batch_name in enumerate(names):
            rows = []
            for (i, j), item in evidence[batch_index].items():
                rows.append({
                    "stage": f"{stage}_{batch_name}_deferred", "method": method,
                    "iteration": iteration, "side": batch_name,
                    "cluster_i": i, "cluster_j": j,
                    "source_original_clusters_i": ";".join(map(str, origins[batch_index][i])),
                    "source_original_clusters_j": ";".join(map(str, origins[batch_index][j])),
                    "within_corr": item["within_corr"],
                    "supporting_batches": ";".join(item["support"]),
                    "distinguishing_batches": ";".join(item["distinguish"]),
                    "top_score_gap": item["top_score_gap"],
                    "profile_rmse": item["profile_rmse"],
                    "eligible": bool(item["support"] and not item["distinguish"]),
                    "selected": False})
            chosen = choose_disjoint(rows)
            selected.append(chosen)
            audit.extend(rows)
        print(stage, "deferred round", iteration, "clusters",
              [p.shape[1] for p in current], "merges", selected, flush=True)
        if not any(selected):
            break
        updated = [merge_pairs(current[i], origins[i], selected[i])
                   for i in range(len(current))]
        current = [item[0] for item in updated]
        origins = [item[1] for item in updated]
    cross_corrs = [scores_for_pair(data[i:i + 2], current[i:i + 2], method)[0]
                   for i in range(len(current) - 1)]
    prefix = 'rna' if method == 'DE' else 'atac'
    aligned = run_cross_batch_align(
        current, names, cross_corrs,
        cap=getattr(args, f'{prefix}_align_cap', 3), null_score=-.5)
    return current, aligned["p_align"], cross_corrs, audit

def pearson_matrix(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    a /= np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    b /= np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    return a @ b.T

def local_one_hot(labels):
    labels = pd.Series(labels).astype(str)
    levels = sorted(labels.unique(), key=lambda x: int(x) if x.isdigit() else x)
    index = {label: i for i, label in enumerate(levels)}
    cols = torch.tensor([index[x] for x in labels], dtype=torch.long)
    return torch.nn.functional.one_hot(cols, len(levels)).float(), levels

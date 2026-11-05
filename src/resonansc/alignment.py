"""Fresh initialization and the tuned three-merge, two-level alignment."""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from copy import deepcopy

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import torch
from umap.umap_ import fuzzy_simplicial_set

from .data import rows, validate, require_counts, resolve_condition_groups
from .alignment_algorithms import (local_one_hot, pearson_matrix, scores_for_pair,
    adjacent_refine_and_align, deferred_refine_and_align, refine_pair, load_pooled_peak_tfidf)
from .core.bulk import get_bulk
from .core.corr import de_1vAll_kbulk, compute_inbatch_corr_de
from .core.merge_align import run_inbatch_merge, run_cross_batch_align


def embedding_knn(x, name, out, cfg, pca=False):
    a, runtime = cfg['alignment'], cfg['runtime']
    x = np.asarray(x, dtype=np.float32)
    k = a['n_neighbors']
    if k >= len(x):
        raise ValueError(f'{name}: n_neighbors must be smaller than cell count')
    if runtime['neighbors_backend'] == 'gpu':
        inp, dest = out / f'{name}.npy', out / f'{name}_knn.npz'
        np.save(inp, x)
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(runtime['gpu'])
        subprocess.run([runtime.get('gpu_python') or sys.executable,
                        str(Path(__file__).with_name('gpu_embedding.py')),
                        '--input', str(inp), '--output', str(dest), '--n-pcs',
                        str(a['n_pcs'] if pca else 0), '--n-neighbors', str(k)],
                       check=True, env=env)
        with np.load(dest) as result:
            return result['pcs'], result['indices'], result['distances']
    if pca:
        x = PCA(n_components=a['n_pcs'], random_state=runtime['seed']).fit_transform(x)
    dist, idx = NearestNeighbors(n_neighbors=k).fit(x).kneighbors(x)
    return x, idx, dist


def run_alignment(cfg):
    if cfg['alignment'].get('condition_groups'):
        return run_condition_alignment(cfg)
    return _run_single_alignment(cfg)


def _shared_hvg_genes(cfg, info):
    """Use one expression feature space for every condition and for training."""
    # Prepared input already contains the original run's 2,000 shared HVGs.
    source = ad.read_h5ad(cfg['input']['expression_h5ad'], backed='r')
    try:
        genes = source.var_names.astype(str).tolist()
    finally:
        source.file.close()
    if len(genes) != cfg['alignment']['n_top_genes']:
        raise ValueError('Prepared data must contain the original shared HVGs')
    return genes


def _split_pooled(pooled, names, labels):
    pieces = {}
    start = 0
    for name in names:
        n = int(labels['batch'].eq(name).sum())
        pieces[name] = pooled[start:start+n]
        start += n
    if start != pooled.shape[0]:
        raise ValueError('Pooled P rows do not match condition batch labels')
    return pieces


def _pad_slots(p, offset, total):
    return torch.nn.functional.pad(p, (offset, total - offset - p.shape[1]))


def _condition_rna_profile(source, record, genes, batch_key):
    """Log1p mean RNA pseudobulk for each condition-local final cluster."""
    state, labels = record['state'], record['labels']
    rna_batches = record['info']['batches'][:record['info']['n_rna']]
    k = state['p_final'][0].shape[1]
    rna_labels = labels.loc[labels['batch'].isin(rna_batches), 'final_alignment'].to_numpy(dtype=int)
    if len(rna_labels) != state['p_final'][0].shape[0] or not np.array_equal(
            rna_labels, state['p_final'][0].argmax(1).numpy()):
        raise ValueError('Condition RNA final labels differ from the saved alignment')
    lookup = labels.set_index('cell_barcode')
    if not lookup.index.is_unique:
        raise ValueError('Condition alignment cell barcodes must be unique')
    observed_batches = source.obs[batch_key].astype(str).to_numpy()
    sums = np.zeros((k, len(genes)), dtype=np.float64)
    counts = np.zeros(k, dtype=np.int64)
    for batch_name in rna_batches:
        indices = np.flatnonzero(observed_batches == batch_name)
        names = source.obs_names[indices].astype(str)
        selected = lookup.reindex(names)
        if selected['final_alignment'].isna().any() or not selected['batch'].eq(batch_name).all():
            raise ValueError(f'{batch_name}: saved condition labels differ from expression input')
        cluster = selected['final_alignment'].to_numpy(dtype=int)
        if (cluster < 0).any() or (cluster >= k).any():
            raise ValueError(f'{batch_name}: final cluster index outside checkpoint')
        try:
            x = source[indices, genes].to_memory().X
        except TypeError as exc:
            # Dense HDF5 datasets allow only one fancy index at a time.
            if 'Only one indexing vector' not in str(exc):
                raise
            x = source[indices].to_memory()[:, genes].X
        if sparse.issparse(x):
            x = x.tocsr().astype(np.float64)
            x.data = np.expm1(x.data)
        else:
            x = np.expm1(np.asarray(x, dtype=np.float64))
        assignment = sparse.csr_matrix(
            (np.ones(len(indices), dtype=np.float64),
             (np.arange(len(indices)), cluster)),
            shape=(len(indices), k))
        batch_sum = assignment.T @ x
        sums += batch_sum.toarray() if sparse.issparse(batch_sum) else batch_sum
        counts += np.bincount(cluster, minlength=k)
    profile = np.log1p(sums / np.maximum(counts[:, None], 1))
    return profile, counts


def _condition_similarity(left, right, left_counts, right_counts, mutual_best,
                          reciprocal_top_k=1):
    """Pearson similarity, retaining reciprocal top-k candidates when requested."""
    score = pearson_matrix(left, right)
    supported = (left_counts[:, None] > 0) & (right_counts[None, :] > 0)
    score = np.where(supported, score, -1.0)
    if not mutual_best:
        return score, score.copy()
    row_rank = np.argsort(np.argsort(-score, axis=1), axis=1)
    col_rank = np.argsort(np.argsort(-score, axis=0), axis=0)
    eligible = supported & (row_rank < reciprocal_top_k) & (col_rank < reciprocal_top_k)
    return score, np.where(eligible, score, -1.0)


def _remap_condition_slots(p, local_to_global, total):
    """Apply one condition's final-cluster map to RNA or ATAC cells."""
    mapped = torch.zeros((p.shape[0], total), dtype=p.dtype)
    for local, global_slot in enumerate(local_to_global):
        mapped[:, int(global_slot)] += p[:, local].detach().cpu()
    return mapped


def run_condition_alignment(cfg):
    info = validate(cfg)
    groups, _ = resolve_condition_groups(cfg, info)
    genes = _shared_hvg_genes(cfg, info)
    out = Path(cfg['output_dir']) / 'alignment'
    out.mkdir(parents=True, exist_ok=False)
    conditions_dir = out / 'conditions'
    conditions_dir.mkdir()
    pd.Series(genes, name='gene').to_csv(out / 'shared_hvg_genes.csv', index=False)

    for group_name, group_batches in groups.items():
        group_cfg = deepcopy(cfg)
        group_cfg['input']['batches'] = group_batches
        _run_single_alignment(group_cfg, out=conditions_dir / group_name,
                              gene_names=genes)
    return combine_condition_alignments(cfg, out=out, genes=genes)


def combine_condition_alignments(cfg, out=None, condition_dirs=None, genes=None,
                                 overwrite=False):
    """Combine independently saved condition alignments for prepare/train."""
    info = validate(cfg)
    groups, per_batch_condition = resolve_condition_groups(cfg, info)
    out = Path(out) if out is not None else Path(cfg['output_dir']) / 'alignment'
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / 'alignment_checkpoint.pt'
    labels_path = out / 'cell_labels.csv'
    if not overwrite and (checkpoint.exists() or labels_path.exists()):
        raise FileExistsError(f'Combined alignment already exists in {out}')
    if condition_dirs is not None and set(condition_dirs) != set(groups):
        raise ValueError('condition_dirs must specify exactly the configured condition groups')
    shared_path = out / 'shared_hvg_genes.csv'
    if genes is None and shared_path.exists():
        genes = pd.read_csv(shared_path)['gene'].astype(str).tolist()

    records = {}
    for group_name, group_batches in groups.items():
        group_cfg = deepcopy(cfg)
        group_cfg['input']['batches'] = group_batches
        group_info = validate(group_cfg)
        group_out = (Path(condition_dirs[group_name]) if condition_dirs is not None
                     else out / 'conditions' / group_name)
        if not (group_out / 'alignment_checkpoint.pt').is_file() or not (group_out / 'cell_labels.csv').is_file():
            raise FileNotFoundError(f'{group_name}: saved alignment is incomplete in {group_out}')
        state = torch.load(group_out / 'alignment_checkpoint.pt', map_location='cpu', weights_only=False)
        labels = pd.read_csv(group_out / 'cell_labels.csv', dtype={'cell_barcode': str, 'batch': str})
        if genes is None:
            genes = list(map(str, state['gene_names']))
        if state['gene_names'] != genes or labels['cell_barcode'].tolist() != state['cell_names']:
            raise ValueError(f'{group_name}: alignment checkpoint differs from labels/shared genes')
        if labels['batch'].drop_duplicates().tolist() != group_batches or len(labels) != group_info['n_cells']:
            raise ValueError(f'{group_name}: saved labels do not match the configured condition batches')
        records[group_name] = dict(info=group_info, state=state, labels=labels, out=group_out)

    if len(genes) != cfg['alignment']['n_top_genes']:
        raise ValueError('Saved condition alignments use the wrong number of shared HVGs')
    if shared_path.exists():
        if pd.read_csv(shared_path)['gene'].astype(str).tolist() != genes:
            raise ValueError('Saved shared_hvg_genes.csv differs from condition checkpoints')
    else:
        pd.Series(genes, name='gene').to_csv(shared_path, index=False)

    # Align condition-local final clusters in the shared HVG RNA expression
    # space. A local slot without RNA cells remains condition-specific.
    condition_names = list(records)
    source = ad.read_h5ad(cfg['input']['expression_h5ad'], backed='r')
    try:
        profiles_and_counts = {
            name: _condition_rna_profile(source, records[name], genes, cfg['input']['batch_key'])
            for name in condition_names
        }
    finally:
        source.file.close()
    min_score = cfg['alignment'].get('condition_align_min_score', .9)
    mutual_best = cfg['alignment'].get('condition_align_mutual_best', True)
    reciprocal_top_k = cfg['alignment'].get('condition_align_reciprocal_top_k', 1)
    similarity = []
    similarity_files = []
    for left, right in zip(condition_names[:-1], condition_names[1:]):
        left_profile, left_counts = profiles_and_counts[left]
        right_profile, right_counts = profiles_and_counts[right]
        raw, eligible = _condition_similarity(
            left_profile, right_profile, left_counts, right_counts,
            mutual_best, reciprocal_top_k)
        path = out / f'condition_similarity_{left}_vs_{right}.csv'
        pd.DataFrame(
            raw, index=[f'{left}_{i}' for i in range(raw.shape[0])],
            columns=[f'{right}_{j}' for j in range(raw.shape[1])]).to_csv(path)
        similarity_files.append(path.name)
        similarity.append(eligible)
    aligned = run_cross_batch_align(
        [records[name]['state']['p_final'][0] for name in condition_names],
        condition_names, similarity, cap=1, null_score=-.5, min_score=min_score)
    final_offsets = dict(zip(condition_names, aligned['offsets']))
    final_total = len(aligned['components'])
    local_maps = {
        name: np.array([
            aligned['label_map'][final_offsets[name] + i]
            for i in range(records[name]['state']['p_final'][0].shape[1])
        ], dtype=int)
        for name in condition_names
    }
    mapping_rows = []
    for name in condition_names:
        _, rna_counts = profiles_and_counts[name]
        for local, global_cluster in enumerate(local_maps[name]):
            mapping_rows.append({
                'condition_group': name, 'local_cluster': local,
                'global_cluster': int(global_cluster), 'rna_cells': int(rna_counts[local]),
                'matched_across_conditions': len(aligned['components'][global_cluster]) > 1,
            })
    pd.DataFrame(mapping_rows).to_csv(out / 'condition_cluster_mapping.csv', index=False)
    pair_rows = []
    for i, (row_to_cols, _) in enumerate(aligned['edges_list']):
        for left_cluster, right_clusters in row_to_cols.items():
            for right_cluster in right_clusters:
                pair_rows.append({
                    'left_group': condition_names[i], 'left_cluster': left_cluster,
                    'right_group': condition_names[i + 1], 'right_cluster': right_cluster,
                    'pearson_r': float(similarity[i][left_cluster, right_cluster]),
                })
    pd.DataFrame(pair_rows, columns=[
        'left_group', 'left_cluster', 'right_group', 'right_cluster', 'pearson_r',
    ]).to_csv(out / 'condition_alignment_pairs.csv', index=False)

    cross_offsets = {}
    cross_total = 0
    for name, record in records.items():
        cross_offsets[name] = cross_total
        cross_total += max(p.shape[1] for p in record['state']['p_refined_cross'])

    stage_keys = ('p_init', 'p_first', 'p_small', 'p_refined_within', 'p_within')
    stage_by_batch = {key: {} for key in stage_keys}
    final_by_batch, cross_by_batch, labels_by_batch = {}, {}, {}
    groups_by_batch, first_corr_by_batch = {}, {}
    audits = []
    for name, record in records.items():
        group_batches = record['info']['batches']
        nr = record['info']['n_rna']
        state, labels = record['state'], record['labels']
        for key in stage_keys:
            stage_by_batch[key].update(zip(group_batches, state[key]))
        groups_by_batch.update(zip(group_batches, state['first_merge_groups']))
        for modality, names in enumerate((group_batches[:nr], group_batches[nr:])):
            mapped_final = _remap_condition_slots(
                state['p_final'][modality], local_maps[name], final_total)
            for batch, matrix in _split_pooled(mapped_final, names, labels).items():
                final_by_batch[batch] = matrix
            for batch, matrix in _split_pooled(state['p_refined_cross'][modality], names, labels).items():
                cross_by_batch[batch] = _pad_slots(matrix, cross_offsets[name], cross_total)
        for batch in group_batches:
            frame = labels.loc[labels['batch'].eq(batch)].copy()
            frame['condition_group'] = name
            frame['condition'] = per_batch_condition[batch]
            frame['after_cross_modal_merge'] += cross_offsets[name]
            frame['final_alignment'] = local_maps[name][frame['final_alignment'].to_numpy(dtype=int)]
            labels_by_batch[batch] = frame
        with np.load(record['out'] / 'first_inbatch_correlations.npz') as archive:
            for i, batch in enumerate(group_batches):
                first_corr_by_batch[batch] = archive[f'matrix_{i}']
        audit = pd.read_csv(record['out'] / 'merge_candidate_audit.csv')
        audit['condition_group'] = name
        audits.append(audit)

    batch_names = info['batches']
    combined = pd.concat([labels_by_batch[name] for name in batch_names], ignore_index=True)
    if combined['cell_barcode'].duplicated().any() or len(combined) != info['n_cells']:
        raise ValueError('Combined condition labels do not cover every cell exactly once')
    p_final = [torch.cat([final_by_batch[name] for name in batch_names[:info['n_rna']]]),
               torch.cat([final_by_batch[name] for name in batch_names[info['n_rna']:]])]
    p_cross = [torch.cat([cross_by_batch[name] for name in batch_names[:info['n_rna']]]),
               torch.cat([cross_by_batch[name] for name in batch_names[info['n_rna']:]])]
    if not np.array_equal(torch.cat(p_final).argmax(1).numpy(),
                          combined['final_alignment'].to_numpy()):
        raise ValueError('Combined checkpoint and cluster labels differ')
    state = dict(cell_names=combined['cell_barcode'].tolist(), gene_names=genes,
        **{key: [stage_by_batch[key][name] for name in batch_names] for key in stage_keys},
        p_refined_cross=p_cross, p_final=p_final,
        first_merge_groups=[groups_by_batch[name] for name in batch_names],
        condition_groups=groups, cluster_offsets=final_offsets,
        condition_cluster_map={name: local_maps[name].tolist() for name in condition_names})
    torch.save(state, out / 'alignment_checkpoint.pt')
    combined.to_csv(out / 'cell_labels.csv', index=False)
    np.savez_compressed(out / 'first_inbatch_correlations.npz',
        **{f'matrix_{i}': first_corr_by_batch[name] for i, name in enumerate(batch_names)})
    pd.concat(audits, ignore_index=True).to_csv(out / 'merge_candidate_audit.csv', index=False)
    (out / 'manifest.json').write_text(json.dumps({**info, 'config': cfg,
        'condition_key': cfg['alignment']['condition_key'], 'condition_groups': groups,
        'cluster_offsets': final_offsets, 'n_clusters': final_total,
        'condition_alignment': {
            'method': 'RNA shared-HVG pseudobulk Pearson + adjacent Hungarian',
            'cap': 1, 'min_score': min_score, 'mutual_best': mutual_best,
            'reciprocal_top_k': reciprocal_top_k,
            'matched_pairs': len(pair_rows),
            'similarity_files': similarity_files,
            'mapping_file': 'condition_cluster_mapping.csv',
            'pairs_file': 'condition_alignment_pairs.csv',
        },
        'n_pooled_peaks_by_condition': {
            name: json.loads((record['out'] / 'manifest.json').read_text())['n_pooled_peaks']
            for name, record in records.items()},
        'gt_used': False, 'umap_computed': False, 'leiden_backend': 'CPU igraph'}, indent=2))
    return out


def merge_small_batch_clusters(batch, peak_x, p, modality, threshold, batch_name):
    """Iteratively merge smallest eligible cluster and recompute correlations."""
    origins=[[i] for i in range(p.shape[1])];audit=[];blocked=set()
    while p.shape[1]>1:
        counts=p.sum(0).detach().cpu().numpy(); candidates=[i for i,n in enumerate(counts) if n<100 and tuple(origins[i]) not in blocked]
        if not candidates: break
        i=min(candidates,key=lambda j:(counts[j],j)); n=int(counts[i]); score_method='peak_LSI_Pearson' if modality=='ATAC' else 'DE_gene_Pearson'
        if modality=='ATAC':
            weights=p.detach().cpu().numpy().astype(np.float32);weights/=np.maximum(weights.sum(0,keepdims=True),1e-8);profiles=weights.T@peak_x;C=pearson_matrix(profiles,profiles)
        else:
            bulk=get_bulk([batch],[p])[0];de=de_1vAll_kbulk(batch,p);C=compute_inbatch_corr_de(batch,bulk,p,use_pairwise=False,de_1v=de)
            # For tiny groups with no available DE genes, compare all original
            # 2,000 shared genes; record this fallback explicitly.
            if not np.isfinite(np.delete(C[i],i)).any():
                profiles=bulk.detach().cpu().numpy().T;C=pearson_matrix(profiles,profiles);score_method='all_HVG_Pearson_fallback'
        scores=np.asarray(C[i],dtype=float).copy();scores[i]=-np.inf;scores[~np.isfinite(scores)]=-np.inf;j=int(np.argmax(scores));score=float(scores[j]);accept=np.isfinite(score) and (n<50 or score>=threshold)
        audit.append(dict(batch=batch_name,modality=modality,source_initial_first_merge_ids='|'.join(map(str,origins[i])),source_n=n,target_initial_first_merge_ids='|'.join(map(str,origins[j])) if np.isfinite(score) else '',target_n=int(counts[j]) if np.isfinite(score) else 0,corr=score if np.isfinite(score) else np.nan,metric=score_method,threshold=threshold,rule='under50_best' if n<50 else '50to99_threshold',merged=bool(accept)))
        if not accept:
            blocked.add(tuple(origins[i]));continue
        keep=[k for k in range(p.shape[1]) if k!=i];cols=[];neworig=[]
        for k in keep:
            cols.append(p[:,k]+p[:,i] if k==j else p[:,k]);neworig.append(origins[k]+origins[i] if k==j else origins[k])
        p=torch.stack(cols,dim=1);origins=neworig;blocked=set()
        print('SMALL_MERGE',batch_name,n,'to',int(counts[j]),'corr',round(score,4),score_method,flush=True)
    return p,audit

def _run_single_alignment(cfg, out=None, gene_names=None, init_features=None):
    info = validate(cfg)
    out = Path(out) if out is not None else Path(cfg['output_dir']) / 'alignment'
    out.mkdir(parents=True, exist_ok=False)
    work = out / 'initialization'
    work.mkdir()
    a, inp = cfg['alignment'], cfg['input']
    seed = cfg['runtime']['seed']
    np.random.seed(seed)
    torch.manual_seed(seed)
    source = ad.read_h5ad(inp['expression_h5ad'], backed='r')
    try:
        data = source[source.obs[inp['batch_key']].astype(str).isin(info['batches']).to_numpy()].to_memory()
    finally:
        source.file.close()
    data.obs = pd.DataFrame({'batch': data.obs[inp['batch_key']].astype(str).to_numpy()}, index=data.obs_names)
    if gene_names is None:
        sc.pp.highly_variable_genes(data, flavor='seurat', n_top_genes=a['n_top_genes'], batch_key='batch')
        data = data[:, data.var['highly_variable']].copy()
    else:
        if data.var_names.get_indexer(gene_names).min() < 0:
            raise ValueError('Condition input is missing shared HVGs')
        data = data[:, gene_names].copy()
    if data.n_vars != a['n_top_genes']:
        raise ValueError(f"Expected {a['n_top_genes']} HVGs, got {data.n_vars}")
    batches = [data[data.obs['batch'] == n].copy() for n in info['batches']]
    nr = info['n_rna']
    peak = ad.read_h5ad(inp['peak_h5ad'], backed='r')
    if a.get('initialization_unit', 'condition') == 'batch':
        if init_features is None or not {'RNA', 'ATAC'}.issubset(init_features):
            raise ValueError('Explicit RNA PCA and ATAC LSI plus barcode order are required')
        peak_batches, n_peaks = load_pooled_peak_tfidf(inp['peak_h5ad'], batches[nr:],
            min_cells=a['min_peak_cells'], space='peak_lsi', n_lsi=a['n_lsi'])
        p_original=[]; p_first=[]; p_small=[]; original_labels={}; pre_rows=[]; audits=[]; first_groups=[]
        for i, (name, batch) in enumerate(zip(info['batches'], batches)):
            mod = 'RNA' if i < nr else 'ATAC'
            global_names, features = init_features[mod]
            global_names = pd.Index(np.asarray(global_names, dtype=str)); features = np.asarray(features)
            if not global_names.is_unique or features.ndim != 2 or len(global_names) != len(features) or not np.isfinite(features).all():
                raise ValueError(f'Invalid {mod} features and cell names')
            ix = global_names.get_indexer(batch.obs_names.astype(str))
            if (ix < 0).any(): raise ValueError(f'{name}: cells absent from {mod} features')
            pooled=ad.AnnData(X=np.zeros((len(ix),1),dtype=np.float32),obs=pd.DataFrame(index=batch.obs_names))
            pooled.obsm['X_init']=np.asarray(features[ix,:30],dtype=np.float32)
            sc.pp.neighbors(pooled,n_neighbors=a['n_neighbors'],use_rep='X_init',random_state=seed)
            sc.tl.leiden(pooled,resolution=a['rna_leiden_resolution'] if mod=='RNA' else a['atac_leiden_resolution'],key_added='cluster',random_state=seed,flavor='igraph',n_iterations=2,directed=False)
            p, levels=local_one_hot(pooled.obs['cluster'].astype(str)); p_original.append(p)
            if mod=='RNA':
                bulk=get_bulk([batch],[p])[0]; de=de_1vAll_kbulk(batch,p); C=compute_inbatch_corr_de(batch,bulk,p,use_pairwise=False,de_1v=de)
                feat=None
            else:
                feat=peak_batches[i-nr]; w=p.numpy().astype(np.float32); w/=np.maximum(w.sum(0,keepdims=True),1e-8); C=pearson_matrix(w.T@feat,w.T@feat)
            threshold=a['rna_first_threshold'] if mod=='RNA' else a['atac_first_threshold']
            merged=run_inbatch_merge([C],[p],[threshold],M=None); first=merged['p_merge'][0]; first_groups.append(merged['groups'][0]); p_first.append(first)
            small, audit=merge_small_batch_clusters(batch,feat,first,mod,threshold,name);p_small.append(small);audits.extend(audit)
            for cell, initial, label in zip(batch.obs_names.astype(str), pooled.obs['cluster'].astype(str),first.argmax(1).numpy()):
                original_labels[cell]=initial;pre_rows.append({'cell_barcode':cell,'condition_premerge_cluster':int(label)})
            print('BATCH_INITIAL_MERGE',name,p.shape[1],first.shape[1],small.shape[1],flush=True)
        p_init=p_first
        pd.DataFrame({'cell_barcode':list(original_labels),'condition_init_cluster':list(original_labels.values())}).to_csv(out/'condition_initial_labels.csv',index=False)
        pd.DataFrame(pre_rows).to_csv(out/'condition_premerge_labels.csv',index=False)
        pd.DataFrame(audits).to_csv(out/'small_cluster_merge_audit.csv',index=False)
        first={'groups':first_groups}
    else:
        # One graph per modality within the current condition; cached PCA/LSI was
        # computed on all cells without labels. Neighbor graph and Leiden use only
        # the current condition/modality. Split labels back to original batches.
        if init_features is None or not {'RNA','ATAC'}.issubset(init_features):
            raise ValueError('Explicit RNA PCA and ATAC LSI plus barcode order are required')
        original_labels = {}
        p_init = []
        for mod, batch_group in [('RNA', batches[:nr]), ('ATAC', batches[nr:])]:
            global_names,features=init_features[mod]
            global_names=np.asarray(global_names,dtype=str);features=np.asarray(features)
            if features.ndim!=2 or not np.isfinite(features).all() or not pd.Index(global_names).is_unique:raise ValueError(f'Invalid {mod} initial features')
            if len(global_names) != len(features): raise ValueError('Cached features and barcodes differ')
            cells = sum([b.obs_names.astype(str).tolist() for b in batch_group], [])
            indices = pd.Index(global_names).get_indexer(cells)
            if (indices < 0).any() or len(set(cells)) != len(cells): raise ValueError('Unmatched condition cells')
            emb = np.asarray(features[indices, :30], dtype=np.float32)
            pooled = ad.AnnData(X=np.zeros((len(cells), 1), dtype=np.float32), obs=pd.DataFrame(index=cells))
            pooled.obsm['X_init'] = emb
            print('Pooled condition initialization', out.name, mod, len(cells), flush=True)
            sc.pp.neighbors(pooled, n_neighbors=a['n_neighbors'], use_rep='X_init', random_state=seed)
            sc.tl.leiden(pooled, resolution=a['rna_leiden_resolution'] if mod == 'RNA' else a['atac_leiden_resolution'], key_added='condition_cluster', random_state=seed, flavor='igraph', n_iterations=2, directed=False)
            initial = pooled.obs['condition_cluster'].astype(str)
            for batch in batch_group:
                local=initial.loc[batch.obs_names].astype(str)
                p, _ = local_one_hot(local)
                p_init.append(p)
                original_labels.update(zip(batch.obs_names.astype(str), local))
                print('Condition cluster split', mod, batch.obs_names.size, 'local classes', p.shape[1], flush=True)
        pd.DataFrame({'cell_barcode':list(original_labels), 'condition_init_cluster':list(original_labels.values())}).to_csv(out / 'condition_initial_labels.csv', index=False)
        peak_batches, n_peaks = load_pooled_peak_tfidf(inp['peak_h5ad'], batches[nr:],
            min_cells=a['min_peak_cells'], space='peak_lsi', n_lsi=a['n_lsi'])
        # Merge condition-pooled initial clusters separately per modality, then
        # split back by donor. Save both condition IDs and local donor IDs.
        from scipy.sparse.csgraph import connected_components
        from .core.merge_align import inbatch_merge
        p_original=list(p_init);p_pre=[];pre_rows=[];pre_audits=[]
        for mod,group,features,threshold in [('RNA',batches[:nr],None,a['rna_first_threshold']),('ATAC',batches[nr:],np.concatenate(peak_batches,axis=0),a['atac_first_threshold'])]:
            names=sum([b.obs_names.astype(str).tolist() for b in group],[])
            condition_labels=pd.Series([original_labels[n] for n in names],index=names)
            P,levels=local_one_hot(condition_labels)
            if mod=='RNA':
                pooled=ad.concat(group,index_unique=None);bulk=get_bulk([pooled],[P])[0];de=de_1vAll_kbulk(pooled,P);C=compute_inbatch_corr_de(pooled,bulk,P,use_pairwise=False,de_1v=de)
                mapped=run_inbatch_merge([C],[P],[threshold],M=None);groups=mapped['groups'][0];Pnew=mapped['p_merge'][0]
            else:
                weights=P.numpy().astype(np.float32);weights/=np.maximum(weights.sum(0,keepdims=True),1e-8);profiles=weights.T@features;C=pearson_matrix(profiles,profiles)
                mapped=run_inbatch_merge([C],[P],[threshold],M=None);groups=mapped['groups'][0];Pnew=mapped['p_merge'][0]
            orig=P.argmax(1).numpy();new=Pnew.argmax(1).numpy()
            for cell,i,j in zip(names,orig,new):pre_rows.append({'cell_barcode':cell,'modality':mod,'condition_init_cluster':levels[i],'condition_premerge_cluster':int(j)})
            for gid,grp in enumerate(groups):
                vals=[float(C[i,j]) for n,i in enumerate(grp) for j in grp[n+1:] if np.isfinite(C[i,j])]
                pre_audits.append({'modality':mod,'new_cluster':gid,'initial_clusters':'|'.join(levels[i] for i in grp),'n_initial_clusters':len(grp),'minimum_pair_corr':min(vals) if vals else np.nan,'n_cells':int(Pnew[:,gid].sum()),'threshold':threshold})
            start=0
            for b in group:
                n=b.n_obs;local=Pnew[start:start+n];active=local.sum(0)>0;assert bool(active.any());p_pre.append(local[:,active]);start+=n
            print('CONDITION_PREMERGE',out.name,mod,'before',P.shape[1],'after',Pnew.shape[1],flush=True)
        pd.DataFrame(pre_rows).to_csv(out/'condition_premerge_labels.csv',index=False)
        pd.DataFrame(pre_audits).to_csv(out/'condition_premerge_groups.csv',index=False)
        p_init=p_pre
        # Skip both donor-local correlation merging and donor-local small-cluster merging.
        # Condition-pooled premerge remains intact; cross-donor alignment follows.
        p_first = list(p_init)
        p_small = list(p_init)
        first = {'groups': [[[i] for i in range(p.shape[1])] for p in p_init]}
        pd.DataFrame(columns=['batch','initial_i','initial_j','correlation','merged_component']).to_csv(out / 'ATAC_first_merge_edges.csv',index=False)
        pd.DataFrame(columns=['batch','reason']).to_csv(out / 'small_cluster_merge_audit.csv',index=False)
        print('DONOR_LOCAL_MERGE_SKIPPED (including small-cluster merge)', flush=True)
    args = SimpleNamespace(**a)
    method = deferred_refine_and_align if a['defer_within_merge_across_batches'] else adjacent_refine_and_align
    def within(xs, ps, names, metric, stage):
        return method(xs, ps, names, metric, stage, args) if len(xs)>1 else (ps, ps, [], [])
    if a.get('all_donor_hungarian', False):
        from .all_donor import align_all_donors
        rprofiles=[x.detach().cpu().numpy().T for x in get_bulk(batches[:nr], p_small[:nr])]
        aprofiles=[]
        for features, prob in zip(peak_batches,p_small[nr:]):
            w=prob.detach().cpu().numpy().astype(np.float32)
            w=w/np.maximum(w.sum(0,keepdims=True),1e-8)
            aprofiles.append(w.T@features)
        rw,ra_frame=align_all_donors(rprofiles,p_small[:nr],info['batches'][:nr])
        aw,aa_frame=align_all_donors(aprofiles,p_small[nr:],info['batches'][nr:])
        ra=ra_frame.assign(modality='RNA').to_dict('records')
        aa=aa_frame.assign(modality='ATAC').to_dict('records')
        rr=list(p_small[:nr]);ar=list(p_small[nr:])
    else:
        rr, rw, _, ra = within(batches[:nr], p_small[:nr], info['batches'][:nr], 'DE', 'within_RNA')
        ar, aw, _, aa = within(peak_batches, p_small[nr:], info['batches'][nr:], 'peak_lsi', 'within_ATAC')
    names = sum([x.obs_names.astype(str).tolist() for x in batches], [])
    state = dict(cell_names=names, gene_names=data.var_names.astype(str).tolist(),
        batches=info['batches'], n_rna=nr, p_init=p_original, p_condition_premerge=p_init, p_first=p_first,
        p_small=p_small, p_refined_within=rr+ar, p_within=rw+aw,
        first_merge_groups=first['groups'])
    torch.save(state, out / 'within_condition_checkpoint.pt')
    labels = pd.DataFrame({'cell_barcode':names,'batch':np.concatenate([np.repeat(n,x.n_obs) for n,x in zip(info['batches'],batches)])})
    for key,ps in [('leiden_init',p_original),('after_condition_premerge',p_init),('after_first_inbatch_merge',p_first),('after_small_cluster_merge',p_small),('within_condition',rw+aw)]:
        labels[key]=np.concatenate([p.argmax(1).numpy() for p in ps])
    labels.to_csv(out/'within_condition_labels.csv', index=False)
    pd.DataFrame(ra+aa).to_csv(out/'within_condition_audit.csv',index=False)
    print('WITHIN_CONDITION_COMPLETE',out,flush=True)
    return out

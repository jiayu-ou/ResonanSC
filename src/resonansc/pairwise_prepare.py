"""Build optimization initialization from pairwise-resolved labels."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import anndata as ad
import torch
from .core.corr import marker_init_from_1vall


def prepare_pairwise_optimization(*, base_checkpoint, aligned_h5ad, training_data_dir, labels, output_dir, label_key='pairwise_label', batch_key='batch', marker_top_n=100, marker_min_cells=10):
    """Create fixed 3/-5 P and one-vs-rest M with original W initialization.

    `labels` is a DataFrame indexed by cell barcode; `base_checkpoint` must be
    trusted. The destination must not already exist. Training data are read in
    the exact saved batch and barcode order; no cell labels are changed.
    """
    dest=Path(output_dir);dest.mkdir(parents=True,exist_ok=False)
    state=torch.load(base_checkpoint,map_location='cpu',weights_only=False);a=ad.read_h5ad(aligned_h5ad);lab=labels.copy()
    if not lab.index.is_unique or not a.obs_names.is_unique or set(a.obs_names)!=set(lab.index):raise ValueError('cell names must match uniquely')
    lab=lab.loc[a.obs_names]
    if not np.array_equal(lab[batch_key].astype(str),a.obs[batch_key].astype(str)):raise ValueError('batch labels differ')
    if list(a.var_names.astype(str))!=list(map(str,state['gene_names'])):raise ValueError('gene order differs from checkpoint')
    try:
        numeric=pd.to_numeric(lab[label_key],errors='raise').to_numpy(dtype=float)
        if not np.isfinite(numeric).all() or not np.equal(numeric,np.floor(numeric)).all():raise ValueError('cluster labels must be integers')
        ids=numeric.astype(int)
    except (ValueError,TypeError) as exc:raise ValueError('cluster labels must be integers') from exc
    k=int(ids.max())+1
    if sorted(np.unique(ids).tolist())!=list(range(k)):raise ValueError('labels must be contiguous integers starting at zero')
    parts=[];ps=[];ordered=[]
    for i,batch in enumerate(state['batch_names']):
        b=ad.read_h5ad(Path(training_data_dir)/f'{i}_{batch}.h5ad',backed='r');names=b.obs_names.astype(str).tolist();b.file.close();q=lab.loc[names]
        if not q[batch_key].astype(str).eq(str(batch)).all():raise ValueError(f'batch mismatch: {batch}')
        ps.append(torch.nn.functional.one_hot(torch.tensor(pd.to_numeric(q[label_key]).to_numpy(dtype=int)),k).float());parts.append(a[a.obs_names.get_indexer(names)].copy());ordered.extend(names)
    if len(ordered)!=a.n_obs or len(set(ordered))!=len(ordered):raise ValueError('training batch cells differ from input')
    a=ad.concat(parts,axis=0,join='inner',index_unique=None)
    marker=marker_init_from_1vall(a,torch.cat(ps),n_top=marker_top_n,min_cells_1vAll=marker_min_cells)
    if marker.shape!=(a.n_vars,k):raise RuntimeError('unexpected marker matrix shape')
    state['P_align_onehot']=ps;state['P_align']=[torch.where(p.bool(),3.,-5.) for p in ps];state['M_align']=marker;state['M_init']=marker
    state['align_masks']={'batch_keys':state['batch_names'],'prototype_count':k,'bulknum':k,'slot2proto':torch.arange(k,dtype=torch.long),'counts_per_batch_proto':torch.stack([(p.sum(0)>0).long() for p in ps]),'maxc_per_proto':torch.ones(k,dtype=torch.long)}
    state['training_data_dir']=str(Path(training_data_dir).resolve());state['source_alignment_checkpoint']='pairwise resolved cell labels';state['cluster_offsets']=None;state['condition_cluster_map']=None
    torch.save(state,dest/'init_for_training.pt');a.obs['final_alignment']=lab.loc[ordered,label_key].astype(str).to_numpy();a.write_h5ad(dest/'aligned_gene_activity.h5ad',compression='gzip')
    (dest/'manifest.json').write_text(json.dumps({'n_cells':len(ordered),'n_clusters':k,'n_genes':a.n_vars,'P':'fixed pairwise labels, logits 3/-5','M':'one-vs-rest marker initialization','W':'preserved from base checkpoint','training_data_dir':str(Path(training_data_dir).resolve())},indent=2))
    return dest

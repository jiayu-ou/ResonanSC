"""Extract the model's saved P-weighted pseudo-bulks, without re-estimation."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch


def export_model_pseudobulk(checkpoint, labels, *, modality_prefix='rna_', batch_key='batch', cluster_key='final_alignment', output_dir=None):
    """Return genes × batch/cluster profiles and sample metadata.

    Loads a *trusted* PyTorch checkpoint using pickle. `labels` must be a DataFrame
    indexed by unique cell barcode, containing original batch and cluster IDs.
    The saved probabilities' argmax must agree with those labels in every batch.
    """
    labels=labels.copy()
    if not labels.index.is_unique or labels[[batch_key,cluster_key]].isna().any().any():raise ValueError('unique cell index and non-null labels required')
    state=torch.load(Path(checkpoint),map_location='cpu',weights_only=False)
    keys=['gene_names','batch_names','cell_names','probs_soft','bulks_list']
    if any(k not in state for k in keys):raise ValueError('checkpoint lacks saved pseudo-bulk metadata')
    genes=list(map(str,state['gene_names']));
    if len(genes)!=len(set(genes)):raise ValueError('duplicate gene names')
    if len({len(state[k]) for k in ['batch_names','cell_names','probs_soft','bulks_list']})!=1:raise ValueError('checkpoint batch dimensions differ')
    out=[];samples=[]
    for i,batch in enumerate(state['batch_names']):
        if not str(batch).startswith(modality_prefix):continue
        names=list(map(str,state['cell_names'][i])); missing=pd.Index(names).difference(labels.index)
        if len(missing):raise ValueError(f'{batch}: {len(missing)} cells missing in labels')
        q=labels.loc[names];p=torch.as_tensor(state['probs_soft'][i]).detach().cpu().numpy();bulk=torch.as_tensor(state['bulks_list'][i]).detach().cpu().numpy()
        if p.ndim!=2 or bulk.shape!=(len(genes),p.shape[1]) or len(q)!=len(p):raise ValueError(f'{batch}: shape mismatch')
        if not (q[batch_key].astype(str)==str(batch)).all() or not np.array_equal(p.argmax(1),pd.to_numeric(q[cluster_key]).to_numpy()):raise ValueError(f'{batch}: checkpoint and labels disagree')
        for k in range(p.shape[1]):
            sample_id=f'{batch}_cluster{k}';samples.append(dict(id=sample_id,batch=str(batch),cluster=k,n_cells=int((p.argmax(1)==k).sum()),prob_weight_sum=float(p[:,k].sum())));out.append(bulk[:,k])
    if not out:raise ValueError(f'no batches with prefix {modality_prefix!r}')
    sample=pd.DataFrame(samples);matrix=pd.DataFrame(np.stack(out,axis=1),index=pd.Index(genes,name='gene'),columns=sample.id)
    if output_dir is not None:
        root=Path(output_dir);root.mkdir(parents=True,exist_ok=False);matrix.to_csv(root/'model_pseudobulk.tsv.gz',sep='\t',compression='gzip');sample.to_csv(root/'samples.tsv',sep='\t',index=False)
    return matrix,sample

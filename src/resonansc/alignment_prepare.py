"""Bridge hierarchical or pairwise-resolved labels to ResonanSC initialization."""
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import pandas as pd
import torch


def prepare_from_alignment(config, *, alignment_dir, labels=None, label_key='pairwise_label'):
    """Initialize P/M and peak-to-gene W from a new hierarchical run.

    If provided, `labels` must be indexed by cell barcode and include label_key;
    this permits the pairwise resolver's entire-cluster ATAC relabeling. If not,
    the original hierarchical labels are used. Checkpoints must be trusted.
    """
    from .preparation import prepare
    if isinstance(config,(str,Path)):
        from .config import load_config
        config=load_config(config)
    root=Path(alignment_dir);path=root/'hierarchical_alignment_checkpoint.pt';state=torch.load(path,map_location='cpu',weights_only=False);names=list(map(str,state['cell_names']));saved=pd.read_csv(root/'cell_labels.csv',dtype={'cell_barcode':str}).set_index('cell_barcode')
    if not saved.index.is_unique or set(names)!=set(saved.index):raise ValueError('alignment labels and checkpoint cells differ')
    saved=saved.loc[names].copy()
    if labels is not None:
        if not labels.index.is_unique or set(labels.index)!=set(names):raise ValueError('resolved labels must match all alignment cells')
        saved['final_alignment']=labels.loc[names,label_key].to_numpy()
    numeric=pd.to_numeric(saved['final_alignment'],errors='raise').to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or not np.equal(numeric,np.floor(numeric)).all():raise ValueError('labels must be integers')
    ids=numeric.astype(int);K=int(ids.max())+1
    if sorted(np.unique(ids).tolist())!=list(range(K)):raise ValueError('cluster IDs must be contiguous starting at zero')
    batches=saved.batch.astype(str).tolist();unique=list(dict.fromkeys(batches))
    if sum(([b]*batches.count(b) for b in unique),[])!=batches:raise ValueError('checkpoint cells are not grouped by batch')
    if unique!=[b for b in unique if b.startswith('rna_')]+[b for b in unique if b.startswith('atac_')]:raise ValueError('checkpoint batches must be RNA then ATAC')
    state['p_final']=[torch.nn.functional.one_hot(torch.tensor(ids),K).float()]
    with TemporaryDirectory(prefix='resonansc_prepare_') as tmp:
        tmp=Path(tmp);torch.save(state,tmp/'alignment_checkpoint.pt');saved.index.name='cell_barcode';saved.reset_index().to_csv(tmp/'cell_labels.csv',index=False)
        return prepare(config,tmp)

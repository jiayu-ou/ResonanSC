"""Bridge ResonanSC sparse mapping output to dual-SCANVI input.

Only trusted torch checkpoints should be opened. Peak TF-IDF is fit separately
in each ATAC batch, on the exact filtered peaks used by the mapping model.
"""
import argparse
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
import torch


def as_numpy(x):
    return x.detach().cpu().numpy() if hasattr(x,'detach') else np.asarray(x)


def prepare(rna_path,labels_path,checkpoint_path,training_dir,output,allow_init_weights=False):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    ck=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
    for key in ('gene_names','batch_names','mapping_init'):
        if key not in ck:raise ValueError(f'Checkpoint missing {key}')
    genes=list(map(str,ck['gene_names']))
    if len(set(genes))!=len(genes):raise ValueError('Duplicate checkpoint gene names')
    labels=pd.read_csv(labels_path,dtype={'cell_barcode':str,'batch':str})
    required={'cell_barcode','batch','final_alignment'}
    if not required.issubset(labels.columns) or labels[list(required)].isna().any().any():raise ValueError('Alignment labels must contain non-null cell_barcode, batch, final_alignment')
    if labels.cell_barcode.duplicated().any():raise ValueError('Cell barcodes are duplicated')
    if set(labels.batch)!=set(map(str,ck['batch_names'])):raise ValueError('Alignment and checkpoint batch sets differ')
    rna=ad.read_h5ad(rna_path,backed='r')
    try:
        if not rna.obs_names.is_unique or not rna.var_names.is_unique:raise ValueError('RNA cell and gene names must be unique')
        keep=np.array([i for i,g in enumerate(genes) if g in rna.var_names],dtype=int)
        if not len(keep):raise ValueError('No checkpoint genes measured in RNA')
        measured=[genes[i] for i in keep]
        labels=labels.set_index('cell_barcode',drop=False);arrays=[];meta=[]
        for bid,batch in enumerate(ck['batch_names']):
            batch=str(batch);path=Path(training_dir)/f'{bid}_{batch}.h5ad'
            a=ad.read_h5ad(path)
            if not a.obs_names.is_unique or not a.var_names.is_unique:raise ValueError(f'{batch}: duplicate cell or feature names')
            names=a.obs_names.astype(str);expected=labels[labels.batch.eq(batch)]
            if set(names)!=set(expected.index):raise ValueError(f'{batch}: training and alignment cell sets differ')
            ob=expected.loc[names].copy();ob.index=pd.Index(names,name='cell_id')
            if batch.startswith('rna_'):
                idx=rna.obs_names.get_indexer(names)
                if (idx<0).any():raise ValueError(f'{batch}: RNA cell barcode absent (resolve barcode aliases explicitly before preparation)')
                # Backed h5ad fancy indexing requires increasing rows. Restore requested order afterward.
                order=np.argsort(idx);sorted_idx=idx[order];inverse=np.argsort(order)
                block=rna[sorted_idx,measured].to_memory().X;X=sparse.csr_matrix(block)[inverse].astype(np.float32)
                if not np.isfinite(X.data).all() or (X.data<0).any() or not np.allclose(X.data,np.round(X.data),atol=1e-5,rtol=0):raise ValueError('RNA input must be finite nonnegative integer counts')
                ob['modality']='RNA'
            elif batch.startswith('atac_'):
                mapping=ck['mapping_init'].get(bid,ck['mapping_init'].get(str(bid)))
                if mapping is None:raise ValueError(f'{batch}: missing mapping')
                if list(a.var_names.astype(str))!=list(map(str,mapping['peak_names'])) or list(map(str,mapping['gene_names']))!=genes:raise ValueError(f'{batch}: checkpoint peak/gene order mismatch')
                counts=sparse.csr_matrix(a.X,dtype=np.float32)
                if not np.isfinite(counts.data).all() or (counts.data<0).any() or not np.allclose(counts.data,np.round(counts.data),atol=1e-5,rtol=0):raise ValueError(f'{batch}: peak matrix must be counts')
                tf=counts.multiply(1/(np.asarray(counts.sum(1)).ravel()[:,None]+1e-8)).tocsr()
                idf=np.log1p(counts.shape[0]/(np.asarray(tf.sum(0)).ravel()+1e-8)).astype(np.float32)
                weights=ck.get('mapping_weights',{});w=weights.get(bid,weights.get(str(bid)))
                if w is None:
                    if not allow_init_weights:raise ValueError('No mapping_weights in checkpoint; --allow-init-weights explicitly allows mapping_init weights')
                    w=mapping['init_weight']
                w=as_numpy(w).ravel();pi=as_numpy(mapping['peak_idx']).astype(int);gi=as_numpy(mapping['gene_idx']).astype(int)
                if len(w)!=len(pi) or len(pi)!=len(gi) or not np.isfinite(w).all() or (w<0).any():raise ValueError(f'{batch}: invalid mapping weights')
                W=sparse.coo_matrix((w,(pi,gi)),shape=(a.n_vars,len(genes))).tocsr()
                X=(tf.multiply(idf).tocsr()@W).tocsr()[:,keep].astype(np.float32);X.eliminate_zeros();ob['modality']='ATAC'
            else:raise ValueError(f'Batch {batch} must begin rna_ or atac_')
            if np.any(np.asarray(X.sum(1)).ravel()<=0):raise ValueError(f'{batch}: cells have zero activity after selecting common genes')
            arrays.append(X);meta.append(ob)
        allobs=pd.concat(meta);result=ad.AnnData(X=sparse.vstack(arrays,format='csr'),obs=allobs,var=pd.DataFrame(index=pd.Index(measured,name='gene')))
        result.uns['input_note']='RNA raw counts; ATAC batch-specific TF-IDF @ mapping, without log1p. ATAC continuous, NB likelihood is approximate.'
        output.parent.mkdir(parents=True,exist_ok=True);result.write_h5ad(output,compression='gzip')
    finally:rna.file.close()
    print(f'Saved {output}: {result.n_obs} cells, {result.n_vars}/{len(genes)} measured genes')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--rna-counts',required=True);p.add_argument('--alignment-labels',required=True);p.add_argument('--mapping-checkpoint',required=True);p.add_argument('--training-data-dir',required=True);p.add_argument('--output',required=True);p.add_argument('--allow-init-weights',action='store_true');a=p.parse_args()
    prepare(a.rna_counts,a.alignment_labels,a.mapping_checkpoint,a.training_data_dir,a.output,a.allow_init_weights)

if __name__=='__main__':main()

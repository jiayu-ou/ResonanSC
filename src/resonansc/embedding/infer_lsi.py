"""Infer a WM20-style dual embedding with RNA genes and sign-split ATAC LSI."""
from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad
import torch


def infer_embedding(input_path, *, rna_model, atac_model, atac_lsi_values, atac_lsi_names, output, device='cuda'):
    """Apply trained models to the exact genes, cell IDs and known batch categories."""
    import scvi
    torch.set_float32_matmul_precision('high')
    out=Path(output)
    if out.exists():raise FileExistsError(out)
    data=ad.read_h5ad(input_path)
    if not data.obs_names.is_unique or not data.var_names.is_unique:raise ValueError('duplicate cells or genes')
    if data.obs[['batch','modality']].isna().any().any():raise ValueError('batch and modality must be provided')
    ck={k:torch.load(v,map_location='cpu',weights_only=False) for k,v in [('RNA',rna_model),('ATAC',atac_model)]}
    labels=ck['RNA']['labels']
    if ck['ATAC']['labels']!=labels:raise ValueError('model label mappings differ')
    values=np.load(atac_lsi_values,allow_pickle=False);names=np.load(atac_lsi_names,allow_pickle=False).astype(str)
    if values.ndim!=2 or values.shape[0]!=len(names) or values.shape[1]%2 or not np.isfinite(values).all() or (values<0).any() or not pd.Index(names).is_unique:raise ValueError('invalid ATAC LSI values or names')
    z=None
    for mod in ['RNA','ATAC']:
        sel=data.obs.modality.astype(str).eq(mod).to_numpy();sub=data[sel].copy();state=ck[mod]
        if not sub.n_obs:raise ValueError(f'no {mod} cells')
        if mod=='ATAC':
            if not np.array_equal(names,sub.obs_names.astype(str).to_numpy()):raise ValueError('ATAC LSI cell order differs')
            width=values.shape[1]//2;var=pd.DataFrame(index=[f'LSI_positive_{i+1}' for i in range(width)]+[f'LSI_negative_{i+1}' for i in range(width)])
            sub=ad.AnnData(X=values,obs=sub.obs.copy(),var=var)
        if sub.var_names.astype(str).tolist()!=list(map(str,state['genes'])):raise ValueError(f'{mod} model feature names differ')
        batches=state['batch_categories'];present=set(sub.obs.batch.astype(str))
        if present!=set(batches):raise ValueError(f'{mod}: batches must exactly match training')
        sub.obs['batch']=pd.Categorical(sub.obs.batch.astype(str),categories=batches);sub.obs['train_label']=pd.Categorical(['Unknown']*sub.n_obs,categories=labels+['Unknown'])
        scvi.model.SCANVI.setup_anndata(sub,batch_key='batch',labels_key='train_label',unlabeled_category='Unknown');model=scvi.model.SCANVI(sub,n_latent=state.get('n_latent',16),n_hidden=state.get('n_hidden',256),n_layers=2,gene_likelihood='nb',dispersion='gene-batch',use_batch_norm='none',use_layer_norm='both')
        if list(model.adata_manager.get_state_registry('batch').categorical_mapping)!=batches or list(model.adata_manager.get_state_registry('labels').categorical_mapping)!=labels+['Unknown']:raise ValueError('training category order differs')
        model.module.load_state_dict(state['model_state_dict']);model.to_device(device);model.is_trained_=True;model.module.eval();latent=model.get_latent_representation(batch_size=1024)
        if z is None:z=np.empty((data.n_obs,latent.shape[1]),dtype=np.float32)
        if latent.shape[1]!=z.shape[1]:raise ValueError('model latent dimensions differ')
        z[sel]=latent
    result=ad.AnnData(obs=data.obs.copy());result.obsm['X_embedding']=z;out.parent.mkdir(parents=True,exist_ok=True);result.write_h5ad(out,compression='gzip');return out

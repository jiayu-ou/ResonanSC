"""Pooled LSI of per-batch TF-IDF × W activity, without a second TF-IDF."""
import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler,normalize


def atac_w_lsi(adata, *, modality_key='modality', n_components=30, seed=42, split_scale=10.0):
    """Return (nonnegative sign-split features, ATAC barcodes, fitted transforms)."""
    if modality_key not in adata.obs:raise ValueError('modality column required')
    if not adata.obs_names.is_unique or not adata.var_names.is_unique:raise ValueError('cell and gene names must be unique')
    mask=adata.obs[modality_key].astype(str).eq('ATAC').to_numpy();names=adata.obs_names[mask].astype(str).to_numpy(dtype=str);X=sparse.csr_matrix(adata.X[mask],dtype=np.float32)
    if n_components<1 or min(X.shape)<=n_components+1:raise ValueError('ATAC matrix is too small for requested LSI')
    if not np.isfinite(X.data).all() or (X.data<0).any():raise ValueError('W-mapped ATAC activity must be finite and nonnegative')
    if not np.isfinite(split_scale) or split_scale<=0:raise ValueError('split_scale must be positive')
    svd=TruncatedSVD(n_components=n_components+1,n_iter=7,random_state=seed);z=svd.fit_transform(X);scaler=StandardScaler();lsi=normalize(scaler.fit_transform(z[:,1:])).astype(np.float32)
    feat=np.concatenate([np.maximum(lsi,0),np.maximum(-lsi,0)],axis=1).astype(np.float32)*split_scale
    return feat,names,{'svd':svd,'scaler':scaler,'gene_names':adata.var_names.astype(str).tolist(),'split_scale':float(split_scale)}

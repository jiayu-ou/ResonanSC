"""Resolve ATAC-only labels with RNA pairwise rank-sum features."""
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import rankdata


def _pearson(a,b):
    a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float);a=a-a.mean();b=b-b.mean();den=np.linalg.norm(a)*np.linalg.norm(b)
    return float(a@b/den) if den>0 else np.nan


def _rank_features(x, left, right, n_genes):
    n1=int(left.sum());n2=int(right.sum());n=n1+n2
    if not n1 or not n2:raise ValueError('both RNA groups require cells')
    block=sparse.vstack([x[left],x[right]],format='csc');z=np.empty(x.shape[1]);den=np.sqrt(n1*n2*(n+1)/12.0)
    for j in range(x.shape[1]):
        start,end=block.indptr[j:j+2];vals=block.data[start:end];rows=block.indices[start:end];zero=n-len(vals);first=rows<n1
        sum_first=(n1-int(first.sum()))*((zero+1)/2.0)
        if len(vals):sum_first+=np.sum(rankdata(vals,method='average')[first]+zero)
        z[j]=(sum_first-n1*(n+1)/2.0)/den
    return np.argsort(-np.abs(z),kind='stable')[:min(n_genes,x.shape[1])],z


def resolve_atac_only(adata, *, cluster_key='final_alignment', modality_key='modality', margin=.02, n_genes=100, allow_rna_only=False):
    """Return (cell labels, decisions, DE-feature audit); never change RNA labels.

    RNA and ATAC values must share genes and be finite nonnegative, before
    per-cell normalization. Tied nearest clusters are reranked using the
    top pairwise RNA rank-sum genes. Matching is exploratory, not a DE test.
    """
    if margin<0 or n_genes<1:raise ValueError('margin must be nonnegative and n_genes positive')
    if not adata.obs_names.is_unique or not adata.var_names.is_unique:raise ValueError('cell and feature names must be unique')
    if adata.obs[[cluster_key,modality_key]].isna().any().any():raise ValueError('labels must not be missing')
    mods=adata.obs[modality_key].astype(str).to_numpy();labels=adata.obs[cluster_key].astype(str).to_numpy()
    if set(mods)!={'RNA','ATAC'}:raise ValueError('both RNA and ATAC are required')
    x=sparse.csr_matrix(adata.X,dtype=np.float64);x.sum_duplicates();x.eliminate_zeros()
    if not np.isfinite(x.data).all() or (x.data<0).any():raise ValueError('X must be finite and nonnegative')
    totals=np.asarray(x.sum(1)).ravel()
    if (totals<=0).any():raise ValueError('each cell requires positive total signal')
    x=x.multiply((10000/totals)[:,None]).tocsr();x.data=np.log1p(x.data)
    def order_key(value):
        return (0,int(value)) if value.lstrip('-').isdigit() else (1,value)
    rna=set(labels[mods=='RNA']);atac=set(labels[mods=='ATAC']);shared=sorted(rna&atac,key=order_key);only=sorted(atac-rna,key=order_key)
    if rna-atac and not allow_rna_only:raise ValueError('RNA-only clusters are not handled by this resolver')
    candidates=sorted(rna,key=order_key) if allow_rna_only else shared
    if len(candidates)<2:raise ValueError('at least two RNA candidate labels are required')
    profiles={k:np.asarray(x[(mods=='RNA')&(labels==k)].mean(0)).ravel() for k in candidates};out=labels.copy();decisions=[];features=[];cache={};genes=np.asarray(adata.var_names.astype(str))
    for source in only:
        mask=(mods=='ATAC')&(labels==source);v=np.asarray(x[mask].mean(0)).ravel();ranked=sorted([(k,_pearson(v,profiles[k])) for k in candidates],key=lambda z:(-z[1],order_key(z[0])))
        if not np.isfinite([score for _,score in ranked]).all():raise ValueError(f'undefined correlation for {source}')
        (c1,s1),(c2,s2)=ranked[:2];gap=s1-s2;chosen=c1;d1=d2=np.nan
        if gap<=margin:
            key=tuple(sorted((c1,c2)))
            if key not in cache:
                i,z=_rank_features(x,(mods=='RNA')&(labels==key[0]),(mods=='RNA')&(labels==key[1]),n_genes);cache[key]=i
                features.extend(dict(rna_cluster_1=key[0],rna_cluster_2=key[1],rank=k+1,gene=genes[j],wilcoxon_z=float(z[j])) for k,j in enumerate(i))
            i=cache[key];d1=_pearson(v[i],profiles[c1][i]);d2=_pearson(v[i],profiles[c2][i])
            if not np.isfinite([d1,d2]).all():raise ValueError(f'undefined DE-feature correlation for {source}')
            chosen=c1 if d1>=d2 else c2
        out[mask]=chosen;decisions.append(dict(source_cluster=source,n_cells=int(mask.sum()),first=c1,second=c2,first_corr=s1,second_corr=s2,gap=gap,ambiguous=gap<=margin,pairwise_corr_first=d1,pairwise_corr_second=d2,chosen=chosen))
    result=adata.obs.copy();result['original_alignment_cluster']=labels;result['pairwise_label']=out
    return result,pd.DataFrame(decisions),pd.DataFrame(features)

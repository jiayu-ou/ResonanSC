"""Condition-level, all-donor, cross-condition and cross-modality alignment.

The ATAC LSI and barcode order are explicit inputs; no workspace paths are used.
"""
import json
from pathlib import Path
from copy import deepcopy
from types import SimpleNamespace
import numpy as np,pandas as pd,anndata as ad,torch,scanpy as sc
from .data import validate,resolve_condition_groups
from .alignment import _run_single_alignment
from .alignment_algorithms import scores_for_pair,refine_pair
from .core.merge_align import run_cross_batch_align
from .rescue_alignment import hungarian_then_rescue


def run_hierarchical_alignment(config, *, rna_pca, rna_cell_names, atac_lsi, atac_cell_names, mode="condition"):
    """Run the sec3 hierarchy in a new output directory; return its path.

    config is a resolved config dict; exactly two condition groups required.
    `atac_lsi` is the global ATAC-only peak-LSI matrix in `atac_cell_names`
    order, as used for within-ATAC cross-condition correlation.
    """
    if mode not in ('batch', 'condition'):
        raise ValueError('mode must be batch or condition')
    cfg=deepcopy(config)
    cfg['alignment']['initialization_unit']=mode
    cfg['alignment']['all_donor_hungarian']=True
    out=Path(cfg['output_dir']);out.mkdir(parents=True,exist_ok=False)
    info=validate(cfg);groups,_=resolve_condition_groups(cfg,info) if mode == "condition" else ({"all_batches":info["batches"]}, None)
    src=ad.read_h5ad(cfg['input']['expression_h5ad'],backed='r');genes=src.var_names.astype(str).tolist();src.file.close()
    records={}
    for condition,batches in groups.items():
     c=deepcopy(cfg);c['input']['batches']=batches;path=out/'conditions'/condition
     if not (path/'within_condition_checkpoint.pt').exists():_run_single_alignment(c,out=path,gene_names=genes,init_features={'RNA':(rna_cell_names,rna_pca),'ATAC':(atac_cell_names,atac_lsi)})
     state=torch.load(path/'within_condition_checkpoint.pt',map_location='cpu',weights_only=False);labels=pd.read_csv(path/'within_condition_labels.csv');init=pd.read_csv(path/'condition_initial_labels.csv',dtype={'condition_init_cluster':str});labels=labels.merge(init,on='cell_barcode',validate='one_to_one');pre=pd.read_csv(path/'condition_premerge_labels.csv')[['cell_barcode','condition_premerge_cluster']];labels=labels.merge(pre,on='cell_barcode',validate='one_to_one');assert labels.cell_barcode.tolist()==state['cell_names'];records[condition]=(state,labels)
    print('Within-modality batch alignment complete',flush=True)
    # Merge aligned clusters within each condition, using the same metrics, thresholds,
    # and neighbor-pattern requirement as the original condition premerge.
    from resonansc.core.bulk import get_bulk
    from resonansc.core.corr import de_1vAll_kbulk, compute_inbatch_corr_de
    from resonansc.core.merge_align import run_inbatch_merge
    from resonansc.alignment_algorithms import load_pooled_peak_tfidf, pearson_matrix
    src2=ad.read_h5ad(cfg['input']['expression_h5ad'],backed='r')
    audit=[]
    for condition,(st,lab) in (records.items() if mode == "condition" else []):
     nr=st['n_rna']; outputs=[]; offset=0
     for modality,ps,threshold in [('RNA',st['p_within'][:nr],cfg['alignment']['rna_first_threshold']),('ATAC',st['p_within'][nr:],cfg['alignment']['atac_first_threshold'])]:
      P=torch.cat(ps); sub=lab[lab.batch.str.startswith('rna_' if modality=='RNA' else 'atac_')]; names_local=sub.cell_barcode.to_numpy(); assert len(P)==len(names_local)
      ii=src2.obs_names.get_indexer(names_local); assert (ii>=0).all();order=np.argsort(ii); pooled=src2[ii[order]].to_memory()[np.argsort(order)].copy()
      if modality=='RNA':
       bulk=get_bulk([pooled],[P])[0];de=de_1vAll_kbulk(pooled,P); C=compute_inbatch_corr_de(pooled,bulk,P,use_pairwise=False,de_1v=de)
      else:
       batches_local=[pooled[pooled.obs[cfg['input']['batch_key']].astype(str)==batch].copy() for batch in st['batches'][nr:]]
       assert sum([b.obs_names.astype(str).tolist() for b in batches_local],[])==names_local.tolist()
       feats,_=load_pooled_peak_tfidf(cfg['input']['peak_h5ad'],batches_local,min_cells=cfg['alignment']['min_peak_cells'],space='peak_lsi',n_lsi=cfg['alignment']['n_lsi'])
       X=np.concatenate(feats,axis=0); weights=P.numpy().astype(np.float32);weights/=np.maximum(weights.sum(0,keepdims=True),1e-8);C=pearson_matrix(weights.T@X,weights.T@X)
      result=run_inbatch_merge([C],[P],[threshold],M=None);new=result['p_merge'][0];pos=0
      for old in ps:
       outputs.append(new[pos:pos+len(old)]);pos+=len(old)
      for group_id,group in enumerate(result['groups'][0]):
       vals=[float(C[i,j]) for k,i in enumerate(group) for j in group[k+1:] if np.isfinite(C[i,j])]
       audit.append({'condition':condition,'modality':modality,'new_cluster':group_id,'previous_clusters':'|'.join(map(str,group)),'n_previous_clusters':len(group),'min_pair_corr':min(vals) if vals else np.nan,'n_cells':int(new[:,group_id].sum()),'threshold':threshold})
      print('POST_DONOR_CONDITION_MERGE',condition,modality,P.shape[1],'->',new.shape[1],flush=True)
      del pooled
     st['p_within']=outputs
     lab['post_donor_condition_merge']=np.concatenate([p.argmax(1).numpy() for p in outputs])
    src2.file.close()
    pd.DataFrame(audit).to_csv(out/'post_donor_condition_merge_groups.csv',index=False)

    # Represent each modality within each condition in a common space; ATAC uses global peak LSI.
    atac_z=np.asarray(atac_lsi);atac_index=pd.Index(list(map(str,atac_cell_names)));
    if atac_z.ndim!=2 or len(atac_index)!=len(atac_z) or not atac_index.is_unique or not np.isfinite(atac_z).all():raise ValueError('ATAC LSI must match unique cell names and contain finite values')
    source=ad.read_h5ad(cfg['input']['expression_h5ad'],backed='r');cond=list(groups);assert len(cond)==(2 if mode == "condition" else 1)
    pdata={};names={};matrices={};condition_inputs={}
    for modality in ['RNA','ATAC']:
     ps=[]; ds=[]; ixnames=[]
     for condition in cond:
      st,lab=records[condition];nr=st['n_rna'];mask=lab.batch.str.startswith('rna_' if modality=='RNA' else 'atac_'); local=lab[mask];ns=local.cell_barcode.to_numpy();pieces=st['p_within'][:nr] if modality=='RNA' else st['p_within'][nr:];p=torch.cat(pieces);assert len(ns)==len(p)
      if modality=='RNA':
       ii=source.obs_names.get_indexer(ns);assert (ii>=0).all();order=np.argsort(ii);data=source[ii[order]].to_memory()[np.argsort(order)].copy()
      else:
       ii=atac_index.get_indexer(ns);assert (ii>=0).all();data=np.asarray(atac_z[ii,:30],dtype=np.float32)
      ps.append(p);ds.append(data);ixnames.append(ns)
     if mode == 'batch':
      pdata[modality]=ps[0];names[modality]=ixnames[0];condition_inputs[modality]=ds
     else:
      method='all_2000' if modality=='RNA' else 'peak_lsi';cross,within=scores_for_pair(ds,ps,method);minscore=cfg['alignment']['condition_align_min_score'] if modality=='RNA' else cfg['alignment']['cross_condition_atac_min_corr'];args=cfg['alignment'];cm=args['min_cross'] if modality=='RNA' else args['atac_min_cross'];gap=0.05 if modality=='RNA' else 0.15;rmse=0.12;p_aligned,initial_pairs,rescue=hungarian_then_rescue(ps,cross,within,[condition+'_'+modality for condition in cond],minscore,.7,cm,gap,rmse,out/f'{modality}_cross_condition');pdata[modality]=torch.cat(p_aligned);names[modality]=np.concatenate(ixnames);condition_inputs[modality]=ds;pd.DataFrame(cross).to_csv(out/f'{modality}_cross_condition_similarity.csv',index=True);print('CROSS_CONDITION_DONE',modality,'scoreshape',cross.shape,'labels',p_aligned[0].shape[1], 'min',minscore,flush=True)
    source.file.close()
    # Shared gene-activity profiles retain the original cross-modality scoring; ATAC-side within scores use global LSI.
    src=ad.read_h5ad(cfg['input']['expression_h5ad'],backed='r');pooled=[]
    for modality in ['RNA','ATAC']:
     ii=src.obs_names.get_indexer(names[modality]);assert (ii>=0).all();order=np.argsort(ii);q=src[ii[order]].to_memory()[np.argsort(order)].copy();pooled.append(q)
    src.file.close()
    # RNA marker discovery uses the RNA clusters after cross-condition alignment.
    # Use all significant one-vs-rest genes, not a fixed top-N list.
    rna=pooled[0].copy();rna.obs['marker_cluster']=pd.Categorical(pdata['RNA'].argmax(1).numpy().astype(str));counts=rna.obs['marker_cluster'].value_counts();valid=sorted(counts[counts>=2].index.tolist(),key=int)
    print('RNA 1-vs-all DE: start',len(valid),'clusters',flush=True)
    sc.tl.rank_genes_groups(rna,groupby='marker_cluster',groups=valid,reference='rest',method='wilcoxon',use_raw=False,n_genes=rna.n_vars)
    records_de=[]
    for group in valid:
     t=sc.get.rank_genes_groups_df(rna,group=group);t=t[(t.pvals_adj<0.05)&t.pvals_adj.notna()];records_de.extend([{'RNA_cluster':group,'gene':str(v)} for v in t.names])
    de=pd.DataFrame(records_de,columns=['RNA_cluster','gene']);de.to_csv(out/'RNA_cluster_1vsall_DE_gene_membership.csv',index=False);gene_set=set(de.gene);gene_union=[g for g in genes if g in gene_set];pd.DataFrame({'gene':gene_union}).to_csv(out/'RNA_cluster_1vsall_DE_gene_union.csv',index=False)
    if len(gene_union)<5:raise ValueError('Too few significant RNA DE genes for cross-modality correlation')
    print('RNA 1-vs-all DE: union',len(gene_union),'of',len(genes),'HVGs; membership',len(de),flush=True)
    pooled=[q[:,gene_union].copy() for q in pooled]
    atac_idx=atac_index.get_indexer(names['ATAC']);global_atac=np.asarray(atac_z[atac_idx,:30],dtype=np.float32);cross,within=scores_for_pair(pooled,[pdata['RNA'],pdata['ATAC']],'all_2000');w=pdata['ATAC'].numpy().astype(float);n=w.sum(axis=0);assert (n>0).all();profiles=(w.T@global_atac)/n[:,None];from resonansc.alignment_algorithms import pearson_matrix
    v=profiles-profiles.mean(axis=1,keepdims=True);valid=np.linalg.norm(v,axis=1)>1e-12;within[1]=pearson_matrix(profiles,profiles);within[1][~valid,:]=-1;within[1][:,~valid]=-1
    final,initial_pairs,rescue=hungarian_then_rescue([pdata['RNA'],pdata['ATAC']],cross,within,['RNA','ATAC'],cfg['alignment']['cross_min_cross'],.7,cfg['alignment']['cross_min_cross'],cfg['alignment']['cross_max_top_gap'],cfg['alignment']['cross_max_profile_rmse'],out/'cross_modality');cr=[pdata['RNA'],pdata['ATAC']];pd.DataFrame(cross).to_csv(out/'cross_modality_similarity.csv')
    result=[]
    for mi,mod in enumerate(['RNA','ATAC']):
     frame=pd.DataFrame({'cell_barcode':names[mod], 'modality':mod,'cross_condition_label':pdata[mod].argmax(1).numpy(),'after_cross_modality_merge':cr[mi].argmax(1).numpy(),'final_alignment':final[mi].argmax(1).numpy()});result.append(frame)
    lab=pd.concat([x[1] for x in records.values()],ignore_index=True);finaldf=lab.merge(pd.concat(result),on='cell_barcode',validate='one_to_one');finaldf['condition_group']=finaldf.batch.map({batch:condition for condition,bs in groups.items() for batch in bs});assert len(finaldf)==info['n_cells'] and finaldf.cell_barcode.is_unique;finaldf.to_csv(out/'cell_labels.csv',index=False);torch.save({'cell_names':np.concatenate([names['RNA'],names['ATAC']]).tolist(),'gene_names':genes,'RNA_names':names['RNA'].tolist(),'ATAC_names':names['ATAC'].tolist(),'p_cross_condition':pdata,'p_refined_cross':cr,'p_final':final,'condition_groups':groups,'workflow':('batch initialization -> within-batch merge -> all-pairs batch alignment -> cross modality' if mode == 'batch' else 'condition pooled premerge -> all-pairs donor HVG/peak LSI alignment -> condition postmerge -> within modality across condition -> cross modality')},out/'hierarchical_alignment_checkpoint.pt');(out/'run_parameters.json').write_text(json.dumps({'mode':mode,'config':cfg,'condition_cross_RNA':('Hungarian and rescue; see config' if mode == 'condition' else None),'condition_cross_ATAC':('Hungarian and rescue; see config' if mode == 'condition' else None),'cross_modality_within_ATAC':'global peak LSI Pearson; threshold 0.7; Hungarian first; unmatched rescue to matched anchors','cross_modality_match_cap':1,'cross_modality_hungarian_min':0.12,'cross_modality_genes':'union of all significant RNA one-versus-rest DE genes from 2000 original HVGs; per-cluster Wilcoxon BH adjusted p < 0.05','cross_modality_n_genes':len(gene_union),'pretrained_or_optimized':False},indent=2));print('ALL_STAGES_COMPLETE',len(finaldf),'final_clusters',finaldf.final_alignment.nunique(),flush=True)
    return out

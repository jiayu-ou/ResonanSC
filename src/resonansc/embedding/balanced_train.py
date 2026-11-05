"""Original SCANVAE modules with external CITEpool-style partial-label objective."""
from pathlib import Path
import argparse,json,time
import numpy as np
import pandas as pd
import anndata as ad
import torch
import scvi
from scipy import sparse
from scvi.data.fields import ObsmField

LABELS=[]
RNA_REF={}
N_STEPS=0

def gpu(t):return {k:v.cuda() if torch.is_tensor(v) else v for k,v in t.items()}
def loader(model,ix,shuffle):
    if shuffle:
        obs=model.adata.obs
        if obs.modality.astype(str).iloc[0]=='RNA': labels=obs.train_label.astype(str).to_numpy()
        else: labels=obs.parent_candidates.astype(str).to_numpy()
        groups=[np.flatnonzero(labels==name) for name in LABELS]
        if any(len(g)==0 for g in groups):raise ValueError('Balanced batches require each label in both modalities')
        n_batches=N_STEPS;rng=np.random.default_rng(np.random.randint(0,2**32-1));chunks=[]
        for _ in range(n_batches):
            sizes=np.full(len(groups),512//len(groups));sizes[:512%len(groups)]+=1
            batch=np.concatenate([rng.choice(g,n,replace=True) for g,n in zip(groups,sizes)]);rng.shuffle(batch);chunks.append(batch)
        ix=np.concatenate(chunks)
    return model._make_data_loader(adata=model.adata,indices=ix,batch_size=512,shuffle=False,num_workers=0)
def labelled(t):
    mask=t['labels'].reshape(-1)<len(LABELS)
    if not mask.any():return None
    return {k:v[mask] if torch.is_tensor(v) and v.ndim and v.shape[0]==len(mask) else v for k,v in t.items()}

def partial_loss(module,z,parent_mask):
    mask=parent_mask.bool();valid=mask.any(1)&(~mask.all(1))
    if not valid.any():return z.sum()*0
    logits=module.classify_helper(z[valid]);logp=torch.nn.functional.log_softmax(logits,dim=1)
    return -torch.logsumexp(logp.masked_fill(~mask[valid],-torch.inf),dim=1).mean()

def parent_mmd(r,a,yr,parent_mask):
    weights=(2**torch.arange(len(LABELS),device=a.device));code=(parent_mask.long()*weights).sum(1);vals=[]
    for value in torch.unique(code):
        if int(value)==0:continue
        am=code==value;leaf=(torch.bitwise_and(value,weights)>0);valid=yr<len(LABELS)
        rm=torch.zeros_like(valid);rm[valid]=leaf[yr[valid]]
        x=r[rm][:128];y=a[am][:128]
        if min(len(x),len(y))<5:continue
        z=torch.cat([x,y]);dist=torch.cdist(z,z).square()/z.shape[1]
        scale=dist.detach()[dist.detach()>0].median().clamp(min=1e-8)
        k=sum(torch.exp(-dist/(2*scale*s)) for s in [.25,.5,1.,2.,4.])/5;n=len(x)
        vals.append(k[:n,:n].mean()+k[n:,n:].mean()-2*k[:n,n:].mean())
    return torch.stack(vals).mean() if vals else (r.sum()+a.sum())*0

def make_data(input_path,parent_path):
    data=ad.read_h5ad(input_path);obs=data.obs.copy()
    needed={'modality','batch','final_alignment'}
    if not needed.issubset(obs.columns):raise ValueError(f'Input obs is missing {needed-set(obs.columns)}')
    if not data.obs_names.is_unique or not data.var_names.is_unique:raise ValueError('Input cell and gene names must be unique')
    if set(obs.modality.astype(str))!={'RNA','ATAC'}:raise ValueError('modality must contain RNA and ATAC')
    if obs[['batch','final_alignment','modality']].isna().any().any():raise ValueError('batch/modality/final_alignment may not be missing')
    try:
        numeric=pd.to_numeric(obs['final_alignment'],errors='raise')
        if not np.isfinite(numeric).all() or not np.equal(numeric,np.floor(numeric)).all():raise ValueError('cluster IDs must be integers')
        obs['final_alignment']=numeric.astype(int)
    except (ValueError,TypeError) as exc:raise ValueError('final_alignment must contain integer cluster IDs') from exc
    x=data.X.tocsr() if sparse.issparse(data.X) else np.asarray(data.X)
    vals=x.data if sparse.issparse(x) else x
    if not np.isfinite(vals).all() or (vals<0).any():raise ValueError('Input must be finite and non-negative')
    xr=x[obs.modality.astype(str).eq('RNA').to_numpy()];rv=xr.data if sparse.issparse(xr) else xr
    if not np.allclose(rv,np.round(rv),atol=1e-5,rtol=0):raise ValueError('RNA input must be raw integer counts')
    if 'cell_barcode' not in obs:obs['cell_barcode']=obs.index.astype(str)
    if obs.cell_barcode.duplicated().any() or not np.array_equal(obs.cell_barcode.astype(str),obs.index.astype(str)):raise ValueError('cell_barcode must uniquely match obs_names in order')
    rna=obs.final_alignment.map(RNA_REF).fillna('Unknown');obs['train_label']=np.where(obs.modality.eq('RNA'),rna,'Unknown')
    assignments=pd.read_csv(parent_path,dtype={'atac_batch':str,'atac_cluster':str,'parent_candidates':str}).fillna({'parent_candidates':''})
    if not {'atac_batch','atac_cluster','parent_candidates'}.issubset(assignments.columns):raise ValueError('Parent CSV needs atac_batch, atac_cluster, parent_candidates')
    if assignments.duplicated(['atac_batch','atac_cluster']).any():raise ValueError('Duplicate ATAC batch/cluster parent assignment')
    invalid={name for s in assignments.parent_candidates for name in s.split('|') if name and name not in LABELS}
    if invalid:raise ValueError(f'Unknown parents in mapping: {invalid}')
    mapping={(r.atac_batch,int(r.atac_cluster)):r.parent_candidates for r in assignments.itertuples(index=False)}
    source_map={(r.atac_batch,int(r.atac_cluster)):str(getattr(r,'source','provided')) for r in assignments.itertuples(index=False)}
    parent=['']*len(obs);sources=['none']*len(obs)
    for i,(m,b,c) in enumerate(zip(obs.modality.astype(str),obs.batch.astype(str),obs.final_alignment)):
        if m=='ATAC':
            parent[i]=mapping.get((b,int(c)),'')
            if parent[i]:sources[i]=source_map[(b,int(c))]
    obs['parent_candidates']=parent
    obs['parent_source']=sources
    obs['parent_status']=np.where(obs.modality.eq('RNA'),np.where(obs.train_label.eq('Unknown'),'Unknown','RNA exact'),np.where(obs.parent_candidates.eq(''),'Unknown',np.where(obs.parent_candidates.str.contains(r'\|'),'partial set','singleton')))
    obs['train_label']=pd.Categorical(obs.train_label,categories=LABELS+['Unknown'])
    data.obs=obs;mask=np.zeros((len(obs),len(LABELS)),dtype=np.float32)
    for i,s in enumerate(parent):
        if s:
            for name in s.split('|'):mask[i,LABELS.index(name)]=1
    data.obsm['parent_mask']=mask
    print('Parent labels: ',data.obs.groupby(['modality','parent_status'],observed=True).size().to_dict(),flush=True)
    return data

def main():
    p=argparse.ArgumentParser(description='Train post-ResonanSC parent-supervised dual SCANVI embedding')
    p.add_argument('--input',required=True,help='AnnData: raw RNA counts and weighted ATAC activity on common genes')
    p.add_argument('--rna-map',required=True,help='JSON mapping RNA cluster IDs to ordered parent labels')
    p.add_argument('--parents',required=True,help='CSV mapping ATAC batch and cluster to parent candidate labels')
    p.add_argument('--output',required=True,help='New output directory')
    p.add_argument('--atac-lsi-values',required=True,help='Nonnegative signed-split ATAC LSI .npy')
    p.add_argument('--atac-lsi-names',required=True,help='ATAC barcode .npy in the exact same order')
    p.add_argument('--weight',type=float,default=200);p.add_argument('--latent',type=int,default=16);p.add_argument('--hidden',type=int,default=256)
    p.add_argument('--pretrain-epochs',type=int,default=100);p.add_argument('--epochs',type=int,default=50);p.add_argument('--device',default='cuda')
    args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    global LABELS,RNA_REF,N_STEPS
    raw=json.loads(Path(args.rna_map).read_text());RNA_REF={int(k):str(v) for k,v in raw.items()};LABELS=list(dict.fromkeys(RNA_REF.values()))
    if len(LABELS)<2 or len(LABELS)>60 or 'Unknown' in LABELS or any(not x or '|' in x for x in LABELS):raise ValueError('RNA map needs 2–60 nonempty parent names, excluding Unknown and |')
    if args.weight<0 or min(args.latent,args.hidden,args.pretrain_epochs,args.epochs)<1:raise ValueError('Invalid training hyperparameters')
    if args.device!='cuda' or not torch.cuda.is_available():raise ValueError('This training module currently requires CUDA; select a GPU with CUDA_VISIBLE_DEVICES')
    torch.set_num_threads(8);torch.set_float32_matmul_precision('high');scvi.settings.seed=42
    data=make_data(args.input,args.parents);models={};splits={};N_STEPS=int(np.ceil((data.obs.modality.astype(str)=='RNA').sum()/512))
    missing=[label for label in LABELS if int((data.obs.train_label.astype(str)==label).sum())<2]
    if missing:raise ValueError(f'RNA parent labels need at least 2 cells each: {missing}')
    if args.weight and not data.obs.parent_candidates.ne('').any():raise ValueError('MMD requested but no ATAC cells have parent labels')
    data.obs[['cell_barcode','batch','modality','final_alignment','train_label','parent_candidates','parent_status','parent_source']].to_csv(out/'cell_parent_labels.csv',index=False)
    for mod in ['RNA','ATAC']:
        sub=data[data.obs.modality.eq(mod)].copy();sub.obs['train_label']=pd.Categorical(sub.obs.train_label.astype(str),categories=LABELS+['Unknown']);sub.obs['batch']=sub.obs.batch.astype(str).astype('category')
        if mod=='ATAC':
            names=np.load(args.atac_lsi_names,allow_pickle=False);values=np.load(args.atac_lsi_values,allow_pickle=False)
            if values.ndim!=2 or values.shape[0]!=sub.n_obs or values.shape[1]%2 or not np.isfinite(values).all() or (values<0).any():raise ValueError('ATAC LSI values must be finite nonnegative cells x 2k')
            if not np.array_equal(names,sub.obs_names.astype(str).to_numpy()):raise ValueError('ATAC LSI cell order mismatch')
            sub=ad.AnnData(X=values,obs=sub.obs.copy(),var=pd.DataFrame(index=[f'LSI_positive_{i+1}' for i in range(values.shape[1]//2)]+[f'LSI_negative_{i+1}' for i in range(values.shape[1]//2)]),obsm={'parent_mask':sub.obsm['parent_mask'].copy()})
        splits[mod]=(np.arange(sub.n_obs,dtype=int),np.array([],dtype=int))
        print(mod,'training ALL cells:',sub.n_obs,flush=True)
        predata=sub[splits[mod][0]].copy()
        scvi.model.SCVI.setup_anndata(predata,batch_key='batch')
        scvi.settings.seed=42
        pre=scvi.model.SCVI(predata,n_latent=args.latent,n_hidden=args.hidden,n_layers=2,gene_likelihood='nb',dispersion='gene-batch',use_batch_norm='none',use_layer_norm='both')
        pretrained=out/f'pretrained_{mod}.pt'
        if pretrained.exists():
            pre.module.load_state_dict(torch.load(pretrained,map_location='cpu',weights_only=True));print('Reusing all-cell pretrained',mod,flush=True)
        else:
            pre.train(max_epochs=args.pretrain_epochs,train_size=1.0,accelerator='gpu',devices=1,batch_size=1024,early_stopping=False,enable_progress_bar=False,datasplitter_kwargs={'num_workers':0})
            torch.save(pre.module.state_dict(),pretrained)
        scvi.model.SCANVI.setup_anndata(sub,batch_key='batch',labels_key='train_label',unlabeled_category='Unknown')
        model=scvi.model.SCANVI(sub,n_latent=args.latent,n_hidden=args.hidden,n_layers=2,gene_likelihood='nb',dispersion='gene-batch',use_batch_norm='none',use_layer_norm='both')
        assert list(model.adata_manager.get_state_registry('labels').categorical_mapping)==LABELS+['Unknown'] and model.module.n_labels==len(LABELS)
        missing,unexpected=model.module.load_state_dict(pre.module.state_dict(),strict=False)
        assert set(unexpected)<={'pyro_param_store'} and all(x.startswith(('classifier.','encoder_z2_z1.','decoder_z1_z2.','y_prior')) for x in missing),(missing,unexpected)
        del pre,predata
        model.adata_manager.register_new_fields([ObsmField('parent_mask','parent_mask')])
        model.to_device('cuda');models[mod]=model
    models['ATAC'].module.classifier=models['RNA'].module.classifier
    init=f'independent SCVI pretraining on all cells, {args.pretrain_epochs} epochs'
    assert models['ATAC'].module.classifier is models['RNA'].module.classifier
    scvi.settings.seed=43;params=list({id(p):p for model in models.values() for p in model.module.parameters() if p.requires_grad}.values());opt=torch.optim.Adam(params,lr=.001,eps=.01,weight_decay=1e-6)
    config=dict(model='two original SCANVAE modules, separate encoders/decoders and shared classifier',ATAC_supervision='-log(sum(probabilities of allowed parents)); none for Unknown',loss='0.5*(RNA SCANVI + ATAC SCANVI + 50*ATAC parent loss) + MMD_weight*parent_MMD',parent_MMD='per parent set, unpaired RBF MMD',MMD_weight=args.weight,epochs=args.epochs,seed=43,initialization=init,input=str(Path(args.input).resolve()),parents=str(Path(args.parents).resolve()),rna_map=str(Path(args.rna_map).resolve()),caveat='ATAC weighted activity is continuous; NB is an approximation',labels=LABELS,n_cells=data.n_obs,genes=data.var_names.astype(str).tolist())
    config['sampling']='label-balanced per modality, 512 cells per batch, sampling with replacement; each label in every batch';config['mmd_bandwidth']='median per-dimension squared distance with floor 1e-8';config['input_description']='RNA raw counts; ATAC per-batch TF-IDF times WM20 optimized W, pooled 31-component SVD, first removed, column standardization, row L2, signed split x10';config['split']='none: all cells used in pretraining and joint training';config['n_latent']=args.latent;config['n_hidden']=args.hidden;config['pretraining']=init
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    hist=[];start=time.time();checkpoint=out/'joint_resume.pt';first_epoch=0
    if checkpoint.exists():
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        for mod in ['RNA','ATAC']:models[mod].module.load_state_dict(state['models'][mod])
        opt.load_state_dict(state['optimizer']);hist=state['history'];first_epoch=state['epoch'];print('Resuming joint training at epoch',first_epoch,flush=True)
    for epoch in range(first_epoch,args.epochs):
        for model in models.values():model.module.train()
        dl={m:loader(model,splits[m][0],True) for m,model in models.items()};its={m:iter(d) for m,d in dl.items()};stats=[]
        for step in range(max(map(len,dl.values()))):
            opt.zero_grad();loss={};z={};ys={};pm=None;pl=None
            for mod,model in models.items():
                try:t=next(its[mod])
                except StopIteration:its[mod]=iter(dl[mod]);t=next(its[mod])
                t=gpu(t);known=labelled(t) if mod=='RNA' else None
                inf,_,l=model.module(t,loss_kwargs={'labelled_tensors':known,'classification_ratio':50.,'kl_weight':(epoch+1)/args.epochs})
                loss[mod]=l.loss;z[mod]=inf['qz'].loc;ys[mod]=t['labels'].reshape(-1)
                if mod=='ATAC':pm=t['parent_mask'];pl=partial_loss(model.module,z[mod],pm)
            align=parent_mmd(z['RNA'],z['ATAC'],ys['RNA'],pm) if args.weight else z['RNA'].sum()*0
            total=.5*(loss['RNA']+loss['ATAC']+50*pl)+config['MMD_weight']*align
            assert torch.isfinite(total);total.backward();norm=torch.nn.utils.clip_grad_norm_(params,200.);assert torch.isfinite(norm);opt.step()
            stats.append([float(total.detach()),float(loss['RNA'].detach()),float(loss['ATAC'].detach()),float(pl.detach()),float(align.detach())])
        rec=dict(epoch=epoch+1,seconds=time.time()-start,**dict(zip(['loss','RNA_loss','ATAC_loss','ATAC_parent_set_loss','MMD'],np.mean(stats,axis=0))))
        for mod,model in []:  # No held-out cells: user requested all cells for training.
            model.module.eval();correct=0;n=0;parentmass=0.;recon_sum=0.;recon_n=0
            with torch.no_grad():
                for t in loader(model,splits[mod][1],False):
                    t=gpu(t);inf=model.module.inference(**model.module._get_inference_input(t));logits=model.module.classify_helper(inf['qz'].loc);p=torch.softmax(logits,dim=1);y=t['labels'].reshape(-1)
                    if epoch==args.epochs-1:
                        _,_,val_loss=model.module(t,loss_kwargs={'kl_weight':1.0})
                        reconstruction=val_loss.reconstruction_loss
                        recon_sum+=float(sum(v.sum() for v in reconstruction.values()) if isinstance(reconstruction,dict) else reconstruction.sum());recon_n+=len(y)
                    if mod=='RNA':
                        v=y<len(LABELS);correct+=int((logits.argmax(1)[v]==y[v]).sum());n+=int(v.sum())
                    else:
                        masks=t['parent_mask'].bool();v=masks.any(1);correct+=int(masks[v].gather(1,logits.argmax(1)[v,None]).sum());n+=int(v.sum());parentmass+=float((p[v]*masks[v]).sum())
            rec[f'{mod}_heldout_allowed_accuracy']=correct/n if n else None
            if recon_n:rec[f'{mod}_heldout_reconstruction_NLL']=recon_sum/recon_n
            if mod=='ATAC':rec['ATAC_heldout_allowed_probability']=parentmass/n if n else None
        hist.append(rec);pd.DataFrame(hist).to_csv(out/'history.csv',index=False)
        temp=out/'joint_resume.tmp';torch.save({'epoch':epoch+1,'models':{m:model.module.state_dict() for m,model in models.items()},'optimizer':opt.state_dict(),'history':hist},temp);temp.replace(checkpoint)
        if epoch==0 or (epoch+1)%10==0:print(rec,flush=True)
    z=np.empty((data.n_obs,args.latent),dtype=np.float32);meta=data.obs.copy();meta['split']=''
    for mod,model in models.items():
        model.module.eval();model.is_trained_=True
        torch.save({'model_state_dict':model.module.state_dict(),'labels':LABELS,'batch_categories':list(model.adata.obs.batch.cat.categories),'genes':model.adata.var_names.astype(str).tolist(),'n_input':model.adata.n_vars,'n_batch':model.summary_stats.n_batch,'n_labels':len(LABELS),'n_latent':args.latent,'n_hidden':args.hidden},out/f'model_{mod}.pt')
        ix=data.obs_names.get_indexer(model.adata.obs_names);assert (ix>=0).all()
        z[ix]=model.get_latent_representation(batch_size=1024);meta.loc[model.adata.obs_names,'prediction']=np.asarray(model.predict(batch_size=1024))
        for name,idx in zip(['train','unused'],splits[mod]):meta.loc[model.adata.obs_names[idx],'split']=name
    assert np.isfinite(z).all();np.save(out/'X_latent.npy',z);meta.to_csv(out/'cell_metadata.csv')
    result=ad.AnnData(obs=meta);result.obsm['X_embedding']=z;result.uns['note']='Original pre-final alignment labels unchanged; RNA counts and ATAC W-LSI sign-split; X omitted; see config.json.';result.write_h5ad(out/'embedding_results.h5ad',compression='gzip')
    print('Completed embedding training',flush=True)

if __name__=='__main__':main()

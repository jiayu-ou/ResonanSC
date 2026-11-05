from pathlib import Path
import numpy as np,pandas as pd,anndata as ad,yaml
import tempfile,torch
from scipy import sparse
import resonansc
out=Path(tempfile.mkdtemp(prefix='resonansc_hierarchy_smoke_'));rng=np.random.default_rng(6);genes=[f'G{i}' for i in range(120)];full=[];peak=[];rp=[];rn=[];ap=[];an=[]
for modality in ['rna','atac']:
 for condition in ['acute','recovery']:
  for donor in [1,2]:
   batch=f'{modality}_{condition}_{donor}';cluster=np.repeat(np.arange(3),30);rate=np.full((90,120),.2)
   for k in range(3):rate[cluster==k,k*30:(k+1)*30]=2
   counts=rng.poisson(rate).astype('float32');features=np.eye(3)[cluster]+rng.normal(0,.08,(90,3));features=np.column_stack([features,rng.normal(0,.05,(90,3))]).astype('float32');obs=pd.DataFrame({'batch':batch,'condition':condition},index=[f'{batch}_cell{i}' for i in range(90)]);full.append(ad.AnnData(sparse.csr_matrix(np.log1p(counts)),obs=obs,var=pd.DataFrame(index=genes)))
   if modality=='rna':rp.append(features);rn+=obs.index.tolist()
   else:peak.append(ad.AnnData(sparse.csr_matrix(counts),obs=obs.copy(),var=pd.DataFrame(index=[f'chr1:{i*500+1}-{i*500+100}' for i in range(120)])));ap.append(features);an+=obs.index.tolist()
ad.concat(full).write_h5ad(out/'expression.h5ad');ad.concat(peak).write_h5ad(out/'peaks.h5ad');cfg={'input':{'expression_h5ad':'expression.h5ad','peak_h5ad':'peaks.h5ad'},'output_dir':'out','runtime':{'gpu':'','neighbors_backend':'cpu'},'alignment':{'n_top_genes':120,'n_lsi':5,'n_pcs':5,'n_neighbors':10,'min_peak_cells':2,'rna_leiden_resolution':.9,'atac_leiden_resolution':1.0,'rna_first_threshold':.8,'atac_first_threshold':.6,'cross_condition_atac_min_corr':.5,'condition_align_min_score':.95,'cross_min_within':.7,'cross_min_cross':.12,'condition_key':'condition','condition_groups':{'acute':{'rna':['acute'],'atac':['acute']},'recovery':{'rna':['recovery'],'atac':['recovery']}}}}
(out/'config.yaml').write_text(yaml.safe_dump(cfg));print(resonansc.align(out/'config.yaml',rna_pca=np.concatenate(rp),rna_cell_names=rn,atac_lsi=np.concatenate(ap),atac_cell_names=an,mode="condition"))

# Bridge to training with a synthetic, matching genome annotation.
gtf=''.join(f'chr1\ttest\tgene\t{i*500+1}\t{i*500+200}\t.\t+\t.\tgene_id "g{i}"; gene_name "G{i}";\n' for i in range(120));(out/'genes.gtf').write_text(gtf)
cfg['input'].update(gtf='genes.gtf',peak_genome_build='synthetic',gtf_genome_build='synthetic');cfg['preparation']={'min_peak_cells':2,'marker_top_n':10,'marker_min_cells':5,'max_distance':1000};(out/'config.yaml').write_text(yaml.safe_dump(cfg));resonansc.prepare_from_alignment(out/'config.yaml',alignment_dir=out/'out')
config=yaml.safe_load((Path(__file__).resolve().parents[1]/'examples/optimization_WM20.yaml').read_text());config['input']={'init_checkpoint':str(out/'out/prepared/init_for_training.pt'),'training_data_dir':str(out/'out/prepared/training_multidata'),'init_adata':str(out/'out/prepared/aligned_gene_activity.h5ad')};config['output']['output_dir']=str(out/'training');config['runtime']['cuda_visible_devices']='';config['training']['learn_mapping']['epochs']=1;config['training']['learn_M']['epochs']=1;(out/'training.yaml').write_text(yaml.safe_dump(config));resonansc.optimize(out/'training.yaml');state=torch.load(out/'training/train_M_1.pt',map_location='cpu',weights_only=False);assert len(state['cell_names'])==8;print('HIERARCHY_PREPARE_WM_SMOKE_PASS',out,flush=True)

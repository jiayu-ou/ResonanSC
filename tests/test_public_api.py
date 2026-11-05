import tempfile,unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from scipy.stats import ranksums
import anndata as ad
import resonansc
from resonansc.all_donor import align_all_donors
from resonansc.post_alignment import _rank_features

class PublicAPITest(unittest.TestCase):
 def test_all_donor_alignment_and_mass(self):
  profiles=[np.array([[1,2,5,8],[9,4,3,1.]]),np.array([[9,4,3,1.],[1,2,5,8]])]
  p=[torch.eye(2),torch.eye(2)];q,a=align_all_donors(profiles,p,['A','B'])
  self.assertEqual((q[0].shape[1],len(a)),(2,2));self.assertTrue(torch.equal(q[0].sum(1),p[0].sum(1)))
  self.assertEqual(q[0].argmax(1).tolist(),q[1].argmax(1).flip(0).tolist())
 def test_sparse_rank_matches_scipy(self):
  x=np.array([[0,1,4,1],[1,2,1,0],[0,0,3,2],[3,1,0,2],[4,0,0,3],[3,2,1,2]],float);l=np.array([1,1,1,0,0,0],bool);_,z=_rank_features(sparse.csr_matrix(x),l,~l,4)
  np.testing.assert_allclose(z,ranksums(x[l],x[~l],axis=0).statistic)
 def test_pairwise_labels_do_not_mutate(self):
  x=np.array([[9,1,1,4],[8,1,1,5],[1,9,1,5],[1,8,1,4],[8,2,1,4],[1,8,2,4],[7,2,1,4]],float);obs=pd.DataFrame({'modality':['RNA']*4+['ATAC']*3,'final_alignment':[0,0,1,1,0,1,2]},index=[f'c{i}' for i in range(7)]);a=ad.AnnData(sparse.csr_matrix(x),obs=obs);before=a.obs.copy();labels,dec,_=resonansc.resolve_labels(a,margin=0,n_genes=3);pd.testing.assert_frame_equal(before,a.obs);self.assertEqual(labels.iloc[-1].pairwise_label,'0');self.assertEqual(dec.iloc[0].chosen,'0')
 def test_pairwise_optional_rna_only_candidates(self):
  x=np.array([[9,1,1],[1,9,1],[1,1,9],[8,1,1],[1,9,1],[1,1,8]],float);obs=pd.DataFrame({'modality':['RNA']*3+['ATAC']*3,'final_alignment':[0,1,2,0,1,3]},index=[f'c{i}' for i in range(6)]);a=ad.AnnData(sparse.csr_matrix(x),obs=obs)
  with self.assertRaises(ValueError):resonansc.resolve_labels(a)
  labels,dec,_=resonansc.resolve_labels(a,allow_rna_only=True,margin=0)
  self.assertEqual(labels.iloc[-1].pairwise_label,'2');self.assertEqual(labels.iloc[2].pairwise_label,'2');self.assertEqual(dec.iloc[0].chosen,'2')
 def test_pseudobulk_load_and_reject_mismatch(self):
  with tempfile.TemporaryDirectory() as tmp:
   ck={'gene_names':['A','B'],'batch_names':['rna_1'],'cell_names':[['c1','c2']],'probs_soft':[torch.tensor([[.9,.1],[.2,.8]])],'bulks_list':[torch.tensor([[1.,2.],[3.,4.]])]};path=Path(tmp)/'c.pt';torch.save(ck,path);obs=pd.DataFrame({'batch':['rna_1','rna_1'],'final_alignment':[0,1]},index=['c1','c2']);m,s=resonansc.model_pseudobulk(path,obs);self.assertEqual(m.shape,(2,2));self.assertEqual(s.n_cells.tolist(),[1,1]);obs.loc['c1','final_alignment']=1
   with self.assertRaises(ValueError):resonansc.model_pseudobulk(path,obs)
 def test_lsi_shapes_and_reconstruction(self):
  rng=np.random.default_rng(3);a=ad.AnnData(sparse.csr_matrix(rng.uniform(.1,2,(15,8))),obs=pd.DataFrame({'modality':['ATAC']*12+['RNA']*3},index=[str(i) for i in range(15)]));x,names,t=resonansc.atac_w_lsi(a,n_components=3);self.assertEqual(x.shape,(12,6));self.assertEqual(len(names),12);self.assertTrue((x>=0).all());self.assertEqual(len(t['gene_names']),8)
 def test_prepare_optimization_preserves_labels_and_batch_order(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);obs=pd.DataFrame({'batch':['rna_1','rna_1','atac_1','atac_1']},index=['a','b','c','d']);a=ad.AnnData(sparse.csr_matrix(np.ones((4,3))),obs=obs,var=pd.DataFrame(index=['g1','g2','g3']));a.write_h5ad(root/'aligned.h5ad');td=root/'train';td.mkdir();a[:2].write_h5ad(td/'0_rna_1.h5ad');a[2:].write_h5ad(td/'1_atac_1.h5ad');torch.save({'gene_names':['g1','g2','g3'],'batch_names':['rna_1','atac_1']},root/'init.pt');labels=obs.assign(pairwise_label=[0,1,0,1])
   with patch('resonansc.pairwise_prepare.marker_init_from_1vall',return_value=torch.full((3,2),.05)):
    resonansc.prepare_optimization(base_checkpoint=root/'init.pt',aligned_h5ad=root/'aligned.h5ad',training_data_dir=td,labels=labels,output_dir=root/'out')
   state=torch.load(root/'out/init_for_training.pt',weights_only=False);self.assertEqual(state['P_align'][0].argmax(1).tolist(),[0,1]);self.assertEqual(state['M_init'].shape,(3,2))
if __name__=='__main__':unittest.main()


def test_align_defaults_to_batch_and_condition_is_opt_in(monkeypatch):
    import resonansc.api as api
    import resonansc.hierarchical as hierarchy
    calls = []
    monkeypatch.setattr(hierarchy, 'run_hierarchical_alignment', lambda config, **kwargs: calls.append(kwargs) or 'ok')
    kwargs = dict(rna_pca=[], rna_cell_names=[], atac_lsi=[], atac_cell_names=[])
    assert api.align({}, **kwargs) == 'ok'
    assert api.align({}, mode='condition', **kwargs) == 'ok'
    assert [call['mode'] for call in calls] == ['batch', 'condition']

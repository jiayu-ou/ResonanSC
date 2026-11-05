import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
import torch

from resonansc.embedding.prepare import prepare


class EmbeddingTest(unittest.TestCase):
    def test_prepare_and_optional_gpu_train(self):
        with tempfile.TemporaryDirectory(prefix='resonansc-embedding-test-') as td:
            root=Path(td);train=root/'training';train.mkdir();rng=np.random.default_rng(42)
            genes=[f'g{i}' for i in range(20)];peaks=[f'chr1:{i*10}-{i*10+10}' for i in range(30)]
            nr=100;na=100;rn=[f'r{i}' for i in range(nr)];an=[f'a{i}' for i in range(na)]
            r=sparse.csr_matrix(rng.poisson(2,(nr,20)).astype(np.float32));p=sparse.csr_matrix(rng.poisson(1,(na,30)).astype(np.float32))
            # Reversed source order exercises barcode alignment and indexed backed reads.
            ad.AnnData(X=r[::-1],obs=pd.DataFrame(index=rn[::-1]),var=pd.DataFrame(index=genes)).write_h5ad(root/'rna.h5ad')
            ad.AnnData(X=r,obs=pd.DataFrame(index=rn),var=pd.DataFrame(index=genes)).write_h5ad(train/'0_rna_1.h5ad')
            ad.AnnData(X=p,obs=pd.DataFrame(index=an),var=pd.DataFrame(index=peaks)).write_h5ad(train/'1_atac_1.h5ad')
            labels=pd.DataFrame({'cell_barcode':rn+an,'batch':['rna_1']*nr+['atac_1']*na,'final_alignment':([0]*50+[1]*50)*2})
            labels.to_csv(root/'labels.csv',index=False)
            mapping={1:{'peak_names':peaks,'gene_names':genes,'peak_idx':torch.arange(30),'gene_idx':torch.arange(30)%20,'init_weight':torch.ones(30)}}
            torch.save({'gene_names':genes,'batch_names':['rna_1','atac_1'],'mapping_init':mapping,'mapping_weights':{1:torch.ones(30)}},root/'mapping.pt')
            output=root/'input.h5ad';prepare(root/'rna.h5ad',root/'labels.csv',root/'mapping.pt',train,output)
            a=ad.read_h5ad(output);self.assertEqual(a.shape,(200,20));np.testing.assert_array_equal(a[:nr].X.toarray(),r.toarray())
            self.assertEqual(a.obs_names.tolist(),rn+an);self.assertTrue((a[nr:].X.data>=0).all())
            with self.assertRaises(FileExistsError):prepare(root/'rna.h5ad',root/'labels.csv',root/'mapping.pt',train,output)
            if os.environ.get('RESONANSC_TEST_GPU')=='1':
                (root/'rna_map.json').write_text(json.dumps({'0':'typeA','1':'typeB'}))
                pd.DataFrame({'atac_batch':['atac_1','atac_1'],'atac_cluster':[0,1],'parent_candidates':['typeA','typeB']}).to_csv(root/'parents.csv',index=False)
                from resonansc import atac_w_lsi, train_embedding, infer_embedding
                lsi,names,_=atac_w_lsi(a,n_components=5)
                np.save(root/'lsi.npy',lsi);np.save(root/'names.npy',names)
                train_embedding(input_path=output,rna_map=root/'rna_map.json',parents=root/'parents.csv',atac_lsi_values=root/'lsi.npy',atac_lsi_names=root/'names.npy',output=root/'out',pretrain_epochs=1,epochs=1)
                result=ad.read_h5ad(root/'out/embedding_results.h5ad');self.assertEqual(result.obsm['X_embedding'].shape,(200,16));self.assertEqual(result.obs.final_alignment.astype(int).tolist(),labels.final_alignment.tolist())
                for f in ('model_RNA.pt','model_ATAC.pt','history.csv','cell_parent_labels.csv'):self.assertTrue((root/'out'/f).exists())
                infer_embedding(output,rna_model=root/'out/model_RNA.pt',atac_model=root/'out/model_ATAC.pt',atac_lsi_values=root/'lsi.npy',atac_lsi_names=root/'names.npy',output=root/'inferred.h5ad')
                inferred=ad.read_h5ad(root/'inferred.h5ad');np.testing.assert_allclose(inferred.obsm['X_embedding'],result.obsm['X_embedding'],rtol=0,atol=1e-5)

if __name__=='__main__':unittest.main()

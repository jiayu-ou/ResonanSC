import tempfile
from pathlib import Path
import unittest
import numpy as np
import torch
from scipy import sparse
from resonansc.config import load_config
from resonansc.data import rows, require_counts
from resonansc.alignment_algorithms import merge_pairs


class Contracts(unittest.TestCase):
    def test_row_order(self):
        x=sparse.csr_matrix(np.arange(12).reshape(4,3))
        np.testing.assert_array_equal(rows(x,[3,0,2]).toarray(), x.toarray()[[3,0,2]])

    def test_reject_activity(self):
        with self.assertRaises(ValueError): require_counts(sparse.csr_matrix([[.1,1.]]))
        require_counts(sparse.csr_matrix([[0.,2.]]))

    def test_pair_merge_conserves_cells(self):
        p=torch.eye(4)
        q,origins=merge_pairs(p,[[0],[1],[2],[3]],[(0,2)])
        self.assertEqual(q.shape,(4,3))
        self.assertTrue(torch.equal(q.sum(1),torch.ones(4)))
        self.assertEqual(origins,[[0,2],[1],[3]])

    def test_relative_config(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'c.yaml'
            p.write_text('output_dir: result\ninput: {expression_h5ad: expr.h5ad}\n')
            cfg=load_config(p)
            self.assertEqual(cfg['input']['expression_h5ad'],str(Path(d)/'expr.h5ad'))
            self.assertEqual(cfg['alignment']['leiden_resolution'],1.5)

    def test_venv_executable_symlink_is_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            executable=Path(d)/'venv-python'
            executable.symlink_to('/usr/bin/python3')
            p=Path(d)/'c.yaml'
            p.write_text('runtime: {gpu_python: venv-python}\n')
            self.assertEqual(load_config(p)['runtime']['gpu_python'],str(executable))

    def test_selected_stages_resume_from_previous_selected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'c.yaml'
            p.write_text('training: {stages: [learn_mapping, learn_M]}\n')
            cfg=load_config(p)
            self.assertIsNone(cfg['training']['resume']['learn_mapping'])
            self.assertEqual(cfg['training']['resume']['learn_M'],'learn_mapping')

    def test_reject_stage_order(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'c.yaml'
            p.write_text('training: {stages: [learn_M, learn_P]}\n')
            with self.assertRaises(ValueError): load_config(p)


if __name__=='__main__': unittest.main()

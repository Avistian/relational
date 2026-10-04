"""Invariant tests: no trained-model or empirical robustness claims."""
import unittest
import numpy as np
import torch
from relkit.structural_b21 import fixture, changed_cells, validate_assignment, direction_score, adjacency, legal_states, forward, experiment, validate_graph

class IntegrityTests(unittest.TestCase):
    def test_count(self):
        self.assertEqual(changed_cells([0,0,1],[2,2,1]),2)
        self.assertEqual(changed_cells([0,0],[0,0]),0)
        with self.assertRaises(ValueError):changed_cells([0],[0,1])
    def test_validity(self):
        d=fixture();a=d['original']
        self.assertTrue(validate_assignment(d,a,0))
        self.assertTrue(validate_assignment(d,[2,2,1,2,1,2],2))
        for bad,budget in [([2,0,1,2,1,2],2),([2,2,1,2,1,2],1),([3,3,1,2,1,2],2),([4,4,1,2,1,2],2),([-1,-1,1,2,1,2],2),([0,0,1,2,1],2),([0.,0.,1.5,2.,1.,2.],2),(a,-1),(a,1.5)]:
            with self.assertRaises(ValueError):validate_assignment(d,bad,budget)
    def test_direction(self):
        # Replace child0 parent0 by parent1: remove2+7; add5+11 =>7.
        gf=np.array([[2.,5.]]);gr=np.array([[7.],[11.]])
        self.assertEqual(direction_score(gf,gr,np.array([[1.,0.]]),np.array([[0.,1.]])),7.)
        self.assertEqual(direction_score(gf,gr,np.array([[1.,0.]]),np.array([[1.,0.]])),0.)
        with self.assertRaises(ValueError):direction_score(gf,gr,np.zeros((2,2)),np.zeros((2,2)))

class ExperimentTests(unittest.TestCase):
    def test_states(self):
        d=fixture()
        self.assertEqual([len(legal_states(d,b)) for b in [0,1,2]],[1,9,35])
        self.assertEqual(len(set(map(tuple,legal_states(d,2)))),35)
        for a in legal_states(d,2):self.assertTrue(validate_assignment(d,a,2))
    def test_graph_identity(self):
        d=fixture();owners=[1,1,1,2,1,2];a=adjacency(owners)
        self.assertTrue(validate_graph(d,owners,2,a,a.T))
        with self.assertRaises(ValueError):validate_graph(d,owners,2,a,adjacency(d['original']).T)
        with self.assertRaises(ValueError):validate_graph(d,owners,2,adjacency(d['original']),a.T)
    def test_forward_roundtrip(self):
        d=fixture();a=adjacency(d['original'])
        w={'embed_parent':np.eye(2),'embed_child':np.eye(2),'head':np.array([1.,0.]),'head_bias':0.}
        for layer in range(2):
            for t in ['parent','child']:
                w[f'{layer}_{t}_own']=np.zeros((2,2));w[f'{layer}_{t}_neighbor']=np.eye(2);w[f'{layer}_{t}_bias']=np.zeros(2)
        y=forward(d,w,torch.tensor(a),torch.tensor(a.T)).detach().numpy()
        np.testing.assert_allclose(y,[1,0,1],atol=1e-12)
    def test_complete_grid(self):
        r=experiment()
        self.assertEqual(len(r['conditions']),27)
        self.assertEqual(len(r['states']),105)
        for seed in range(3):
            for b in [0,1,2]:
                rows=[x for x in r['conditions'] if x['seed']==seed and x['budget']==b]
                opt=next(x['loss'] for x in rows if x['method']=='exhaustive')
                for row in rows:self.assertLessEqual(row['loss'],opt+1e-12)

if __name__=='__main__':unittest.main()

"""Independent small-graph and hand-counted analysis contracts."""
import json
from pathlib import Path
import numpy as np
from relkit import error_l114 as m

def check_neighborhood(fn):
    # Duplicates, reciprocal entries, self loops and an isolate must not inflate degree.
    e=np.array([[0,0,1,1,2,2,3],[1,1,0,2,1,2,3]])
    y=np.array([0,0,1,1]);r=fn(e,y,np.array([0,2]))
    np.testing.assert_array_equal(r['degree'],[1,2,1,0])
    np.testing.assert_allclose(r['homophily'],[1,.5,0,np.nan],equal_nan=True)
    np.testing.assert_allclose(r['train_neighbor_fraction'],[0,1,0,np.nan],equal_nan=True)
    altered=fn(e,np.array([1,1,0,0]),np.array([0,2]))
    np.testing.assert_allclose(altered['homophily'],r['homophily'],equal_nan=True)
    isolated=fn(np.empty((2,0),dtype=int),y,np.array([0]))
    assert np.all(isolated['degree']==0) and np.isnan(isolated['homophily']).all()

def check_metrics(fn):
    y=np.array([0,0,1,1]);g=np.array([[0,1,1,0],[0,0,0,1]]);b=np.array([[1,0,1,0],[0,1,0,1]])
    r=fn(y,g,b,np.array([3,0,2]))
    assert r['n']==3
    np.testing.assert_allclose(r['gcn'],[2/3,2/3]);np.testing.assert_allclose(r['mlp'],[1/3,2/3])
    np.testing.assert_allclose(r['delta_pp'],[100/3,0])
    np.testing.assert_array_equal(r['discordance'],[[1,1,0,1],[2,0,0,1]])
    assert fn(y,g,b,np.array([],dtype=int))['gcn'] is None

def check_choice(fn):
    rows=[{'family':'class','slice':'rare','n':3,'mean_delta_pp':-90},
          {'family':'degree','slice':'low','n':250,'mean_delta_pp':-3},
          {'family':'homophily','slice':'mixed','n':300,'mean_delta_pp':-2}]
    assert fn(rows)['slice']=='low'
    assert fn(rows,1000) is None
    assert fn(rows[:1],1)['slice']=='rare'

def main():
    check_neighborhood(m.neighborhood_properties);check_metrics(m.slice_metrics);check_choice(m.choose_failure)
    return {'status':'PASS','graph_cases':'duplicates, reciprocals, loops, isolates, label relabeling','metrics':'hand-counted paired disagreements','selection':'minimum-support validation-only rows'}
if __name__=='__main__':
    r=main();Path(__file__).with_name('_check_l114_results.json').write_text(json.dumps(r,indent=2));print(r)

"""Behavioral constructor tests and independently defined task checks."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from relkit.reg_l122 import key_index, relation_edges, construct_reg

def check_keys(fn):
    assert fn([90,10,300])=={90:0,10:1,300:2}
    assert fn(['z','a'])=={'z':0,'a':1}
    assert fn([])=={}
    for keys in [[1,1],[None,1],[np.nan],[pd.NA]]:
        try:fn(keys)
        except ValueError:pass
        else:raise AssertionError('Reject null or duplicate primary keys')

def check_edges(fn):
    np.testing.assert_array_equal(fn([90,10,300],[10,None,90,10,np.nan]),[[0,2,3],[1,0,1]])
    assert fn([1],[None,pd.NA]).shape==(2,0)
    assert fn([],[]).shape==(2,0)
    for pk,fk in [([1],[2]),([1,1],[1]),([1],[-1])]:
        try:fn(pk,fk)
        except ValueError:pass
        else:raise AssertionError('Reject ambiguous or dangling links')

def check_graph(fn):
    tables={'person':pd.DataFrame({'id':[90,10,300],'age':[40.,20.,60.]}),
            'transfer':pd.DataFrame({'id':[7,8,9], 'sender':[10,90,None], 'receiver':[90,10,90], 'amount':[5.,8.,2.]}),
            'tag':pd.DataFrame({'id':[2,4]})}
    schema={'person':{'pk':'id','fks':{}},'transfer':{'pk':'id','fks':{'sender':'person','receiver':'person'}},'tag':{'pk':'id','fks':{}}}
    data,features=fn(tables,schema)
    assert data['person'].num_nodes==3 and data['tag'].num_nodes==2
    assert features=={'person':['age'],'transfer':['amount'],'tag':[]}
    assert len(data.edge_types)==4
    expected={('transfer','f2p_sender','person'):[[0,1],[1,0]],('transfer','f2p_receiver','person'):[[0,1,2],[0,1,0]]}
    for kind,edges in expected.items():
        torch.testing.assert_close(data[kind].edge_index,torch.tensor(edges))
        rev=(kind[2],'rev_'+kind[1],kind[0]);torch.testing.assert_close(data[rev].edge_index,data[kind].edge_index.flip(0))
    assert data.validate(raise_on_error=True)
    # Permutation changes local coordinates, never the underlying key pairs.
    tables['person']=tables['person'].iloc[[2,0,1]].reset_index(drop=True)
    other,_=fn(tables,schema)
    for kind in expected:
        actual={(int(tables[kind[0]].iloc[s]['id']),int(tables[kind[2]].iloc[d]['id'])) for s,d in other[kind].edge_index.T.tolist()}
        assert actual==({(7,10),(8,90)} if kind[1]=='f2p_sender' else {(7,90),(8,10),(9,90)})
    assert list(tables['transfer'].columns)==['id','sender','receiver','amount'],'Do not mutate input'
    # Junction rows remain nodes; repeated endpoint pairs are distinct observations.
    tables['transfer']=pd.concat([tables['transfer'],pd.DataFrame({'id':[99],'sender':[10],'receiver':[90],'amount':[12.]})],ignore_index=True)
    extended,_=fn(tables,schema);assert extended['transfer'].num_nodes==4
    assert extended['transfer','f2p_sender','person'].num_edges==3
    for value in [999,-1]:
        broken={k:v.copy() for k,v in tables.items()};broken['transfer'].loc[0,'sender']=value
        try:fn(broken,schema)
        except ValueError:pass
        else:raise AssertionError('Dangling keys must not become negative/wrapped indices')

if __name__=='__main__':
    check_keys(key_index);check_edges(relation_edges);check_graph(construct_reg)
    out={'status':'PASS','checks':['raw nonconsecutive keys','null primary keys','duplicate primary keys','null foreign keys','dangling foreign keys','isolated rows','two FK roles','reverse edges','row permutation','junction row multiplicity','input immutability']}
    Path(__file__).with_name('_check_l122_results.json').write_text(json.dumps(out,indent=2));print(out)

"""Behavioral checks for schema composition, overlapping taxonomy and evidence gates."""
import json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent/'relkit'))
from hin_l094 import compose_path,classify_method,compare_protocols,table_arithmetic,count_graph

def main():
    ap=np.array([[1,1,0],[0,1,1],[0,0,0]])
    m={('A','writes','P'):ap,('P','by','A'):ap.T}
    actual=compose_path(m,[('A','writes','P'),('P','by','A')])
    expected=np.array([[2,1,0],[1,2,0],[0,0,0]])
    np.testing.assert_array_equal(actual,expected)
    for path in [[],[('A','writes','P'),('A','writes','P')]]:
        try:compose_path(m,path)
        except ValueError:pass
        else:raise AssertionError('Reject empty or type-invalid path')
    assert classify_method({'encoder':'message_passing','route':'explicit_path'})==['GNN','explicit meta-path']
    assert classify_method({'encoder':'lookup','route':'explicit_path'})==['lookup embedding','explicit meta-path']
    a={k:'same' for k in ['dataset_hash','task','split_hash','features','target_mask','selection','budget','metric','aggregation']}
    assert compare_protocols(a,a)=={'status':'COMPARABLE','differences':[]}
    b=dict(a,metric='different');assert compare_protocols(a,b)['differences']==['metric']
    assert compare_protocols({}, {})['status']=='NOT_ESTABLISHED'
    assert compare_protocols(dict(a,metric=None),dict(a,metric=None))['status']=='NOT_ESTABLISHED'
    assert table_arithmetic({'nodes':3,'edges':4,'paper':1,'author':2,'field':0,'venue':0,'institute':0,'PA':1,'PF':1,'PV':1,'AI':0,'PP':0})=={'node_sum':3,'node_delta':0,'edge_sum':3,'edge_delta':-1}
    class Graph:pass
    g=Graph();g.node_feature={'paper':[0],'author':[0,1]};g.edge_list={'paper':{'author':{'AP':{0:{0:None,1:None}}}},'author':{'paper':{'rev_AP':{0:{0:None},1:{0:None}}}}}
    c=count_graph(g);assert c['all_stored_edges']==4 and c['forward_edges']==2 and c['table_columns']['PA']==2
    assert c['reverse_content_matches']
    del g.edge_list['author']['paper']['rev_AP'][1]
    assert not count_graph(g)['reverse_content_matches']
    result={'status':'PASS','checks':['typed path multiplicity and zero-degree nodes','invalid path rejection','HAN belongs to overlapping categories','missing protocol fails closed','changed metric rejected','table arithmetic','independent reverse-edge content audit']}
    (Path(__file__).parent/'_verify_l094_results.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
if __name__=='__main__':main()

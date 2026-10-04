"""Run pinned upstream primitives unchanged; save demonstrated source discrepancies.

Load the package namespace without eager __init__ exports so unused optional GNN
families cannot mask a selector audit. Actual selection/encoding/graph files are
loaded unmodified from the authenticated release.
"""
import hashlib,importlib,json,sys,types
from pathlib import Path
sys.dont_write_bytecode=True
import pandas as pd
from relkit.partitions_b16 import fixture,subsets,score_subset,select_columns,incidence_partition,color_refinement
P=Path(__file__).resolve().parent;source=P/'sources/b16';manifest=json.loads((source/'manifest.json').read_text())
for path,digest in manifest['files'].items():assert hashlib.sha256((source/path).read_bytes()).hexdigest()==digest,path
package=types.ModuleType('b16_upstream');package.__path__=[str(source/'autograble/src/autograble')];sys.modules['b16_upstream']=package
selection=importlib.import_module('b16_upstream.selection');utils=importlib.import_module('b16_upstream.utils');graph=importlib.import_module('b16_upstream.graph')
comparisons=[]
for kind in ('signal','xor','null'):
 tr,va=fixture(kind),fixture(kind,'va');t,v=pd.DataFrame(tr),pd.DataFrame(va)
 for penalty in (0,.5,1):
  for cols in subsets(['A','B','K']):
   for loss in ('0-1','logloss'):
    ours=score_subset(tr,va,cols,penalty,loss)
    theirs=selection.evaluate_subset(t,t.y,v,v.y,cols,lambda_=penalty,loss_name=loss,omega_on='train')
    delta=max(abs(ours['risk']-theirs['val_loss']),abs(ours['omega']-theirs['omega']),abs(ours['J']-theirs['J']))
    assert delta<1e-12
    comparisons.append(dict(world=kind,penalty=penalty,cols=cols,loss=loss,max_delta=delta))
tr,va=fixture('null'),fixture('null','va');t,v=pd.DataFrame(tr),pd.DataFrame(va)
released=selection.greedy_selection(t,t.y,v,v.y,['A','B'],direction='backward',lambda_=.5,loss_name='0-1',omega_on='train')
assert len(released[0])==1
one=selection.evaluate_subset(t,t.y,v,v.y,released[0],lambda_=.5,loss_name='0-1',omega_on='train')
empty=selection.evaluate_subset(t,t.y,v,v.y,[],lambda_=.5,loss_name='0-1',omega_on='train')
assert empty['J']<one['J']
train=pd.DataFrame({'C':['a','a','b','b']});valid=pd.DataFrame({'C':['a','a','b','c']})
encoded,maps=utils.cardinality_encode(train,['C']);encoded_val=utils.apply_cardinality_encode_transductive(valid,['C'],maps)
assert encoded.C.tolist()==[2,2,2,2] and encoded_val.C.tolist()==[4,4,3,1]
union,_=utils.cardinality_encode(pd.concat([train,valid],ignore_index=True),['C'])
assert union.C.iloc[:4].tolist()==[4,4,3,3]
# Build the actual released graph and convert its actual node x and edge arrays
# to an untrained colour-refinement fixture. Vocabulary is metadata, not x.
rows=fixture();g,vocab=graph.build_hetero_graph(pd.DataFrame(rows),['A'],other_columns=None)
assert g['A'].x.tolist()==[[0.0],[0.0]]
colors=[('row',)]*8+[('value','A',tuple(x)) for x in g['A'].x.tolist()];neighbors=[[] for _ in colors]
for i,j in g['row','has','A'].edge_index.t().tolist():
 neighbors[i].append(('A',8+j));neighbors[8+j].append(('A',i))
state=color_refinement(colors,neighbors)
assert len(set(state[:8]))==1
result=dict(status='PASS',commit=manifest['commit'],primitives=comparisons,
 backward=dict(released_selected=released[0],released_J=one['J'],empty_J=empty['J'],course_selected=select_columns(tr,va,['A','B'],.5,'backward')['selected']),
 frequency=dict(train_raw=train.C.tolist(),valid_raw=valid.C.tolist(),released_train=encoded.C.tolist(),released_valid=encoded_val.C.tolist(),paper_union_train=union.C.iloc[:4].tolist()),
 graph=dict(value_features=g['A'].x.tolist(),vocabulary=vocab,actual_released_row_blocks=1,paper_typed_row_blocks=len(incidence_partition(rows,['A']))),
 boundaries='144 primitive comparisons on finite binary fixtures; not all-source or historical Table 1 parity. 0-1 ties agree only because these fixtures encounter class 0 first.')
(P/'evidence/b16/source-audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='primitives'},indent=2))
print('Primitive comparisons:',len(comparisons))

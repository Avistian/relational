"""Independent raw-data/split audit and release-versus-corrected counterexamples."""
import ast,hashlib,json,random
from pathlib import Path
import numpy as np,pandas as pd
from _fetch_l103 import fetch
from relkit.tgat_l103 import load_wikipedia,NeighborFinder,event_batches
P=Path(__file__).resolve().parent;root,_=fetch();nodes,edges,data,a=load_wikipedia(P/'data/l102')
# Parse raw bytes independently with pandas; compare every feature and event ID.
raw=pd.read_csv(P/'data/l102/wikipedia.csv',skiprows=1,header=None)
u=raw[0].to_numpy(dtype=np.int64)+1;v=raw[1].to_numpy(dtype=np.int64)+int(raw[0].max())+2;t=raw[2].to_numpy()
np.testing.assert_array_equal(data['full']['u'],u);np.testing.assert_array_equal(data['full']['v'],v);np.testing.assert_array_equal(data['full']['t'],t)
np.testing.assert_array_equal(edges[1:],raw.iloc[:,4:].to_numpy(dtype=np.float32));assert not nodes.any() and not edges[0].any()
# Execute only original reindex function; process.py's import-time run is excluded.
module=ast.parse((root/'process.py').read_text());fn=next(x for x in module.body if isinstance(x,ast.FunctionDef) and x.name=='reindex')
ns={'np':np,'pd':pd};exec(compile(ast.Module(body=[fn],type_ignores=[]),'<original-reindex>','exec'),ns)
df=ns['reindex'](pd.DataFrame({'u':raw[0],'i':raw[1],'idx':np.arange(len(raw))}))
np.testing.assert_array_equal(df.u,u);np.testing.assert_array_equal(df.i,v);np.testing.assert_array_equal(df.idx,data['full']['e'])
# Independent direct translation of the released split.
cut1,cut2=np.quantile(t,[.7,.85]);held=set(random.Random(2020).sample(tuple(set(u[t>cut1])|set(v[t>cut1])),int(.1*len(set(u)|set(v)))))
train=(t<=cut1)&np.array([x not in held and y not in held for x,y in zip(u,v)])
unseen=(set(u)|set(v))-(set(u[train])|set(v[train]));new=np.array([x in unseen or y in unseen for x,y in zip(u,v)])
masks={'train':train,'val':(t>cut1)&(t<=cut2),'test':t>cut2,'new_val':new&(t>cut1)&(t<=cut2),'new_test':new&(t>cut2)}
for name,mask in masks.items():np.testing.assert_array_equal(data[name]['e'],np.flatnonzero(mask)+1)
fixture={'u':np.ones(3,dtype=int),'v':np.array([2,3,4]),'t':np.array([1.,3.,5.]),'e':np.arange(1,4)}
release=NeighborFinder(fixture,5,True,False);correct=NeighborFinder(fixture,5,False,False)
assert release.find_before(1,4)[:,2].tolist()==[1.]
assert correct.find_before(1,4)[:,2].tolist()==[1.,3.]
assert correct.find_before(1,3)[:,2].tolist()==[1.]
# The two release bugs have measurable coverage effects; do not silently repair them.
coverage={k:sum(len(b['e']) for b in event_batches(data[k],200 if k=='train' else 30)) for k in masks}
assert all(coverage[k]==len(data[k]['e'])-1 for k in masks)
report={'status':'PASS','data':a,'all_raw_features_and_events':'EXACT','timestamp_float32_changes':int(np.count_nonzero(t!=t.astype(np.float32))),'released_reindex':'EXACT','released_split':'EXACT on current Python tuple order','release_scored_counts':coverage,'counterexample':{'history_times':[1,3,5],'cutoff':4,'release_returns':[1],'correct_returns':[1,3]},'historical_set_order':'NOT_ESTABLISHED','last_event_by_split':{k:int(v['e'][-1]) for k,v in data.items()},'processed_cache_sha256':hashlib.file_digest((P/'data/l102/processed.npz').open('rb'),'sha256').hexdigest()}
(P/'_audit_l103_results.json').write_text(json.dumps(report,indent=2)+'\n');print({k:v for k,v in report.items() if k!='data'})

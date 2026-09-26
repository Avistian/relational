"""Authenticate the cited preprocessing and compare its typed-ID mapping."""
import ast,hashlib,json,urllib.request
from pathlib import Path
import numpy as np
import pandas as pd
P=Path(__file__).resolve().parent
commit='e38cdf85998c6ca077167610dc4e769a688efa95'
url=f'https://raw.githubusercontent.com/twitter-research/tgn/{commit}/utils/preprocess_data.py'
expected='b4f55659708cce71f6028a8d83899e6dccdbc076729be7d932ad870743e15931'
raw=urllib.request.urlopen(url,timeout=30).read();assert hashlib.sha256(raw).hexdigest()==expected
folder=P/'sources/l105';folder.mkdir(parents=True,exist_ok=True);(folder/'tgn_preprocess_data.py.txt').write_bytes(raw)
license_url=f'https://raw.githubusercontent.com/twitter-research/tgn/{commit}/LICENSE'
license_bytes=urllib.request.urlopen(license_url,timeout=30).read()
assert hashlib.sha256(license_bytes).hexdigest()=='cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30'
(folder/'LICENSE-TGN').write_bytes(license_bytes)
tree=ast.parse(raw);fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='reindex')
space={};exec(compile(ast.Module(body=[fn],type_ignores=[]),url,'exec'),space)
with np.load(P/'evidence/l105/events.npz') as d:
 frame=pd.DataFrame({'u':d['u'],'i':d['v'],'ts':d['t'],'idx':d['e']})
 transformed=space['reindex'](frame,bipartite=True)
 np.testing.assert_array_equal(transformed.u,frame.u+1)
 np.testing.assert_array_equal(transformed.i,frame.i+8228)
 np.testing.assert_array_equal(transformed.ts,frame.ts)
 assert set(transformed.u).isdisjoint(set(transformed.i))
manifest={'status':'PASS','primary_sources':[{'title':'TGN v3, section 2 Dynamic Graphs','url':'https://arxiv.org/html/2006.10637v3#S2','use':'event and snapshot terminology; not an empirical score target'},{'title':'JODIE authors: data and citation','url':'https://snap.stanford.edu/jodie/','use':'raw Wikipedia release'},{'title':'JODIE authors: data schema','url':'https://github.com/claws-lab/jodie#dataset-format','use':'one record per interaction; endpoint/time/label/feature schema'}],'data':json.loads((P/'_analysis_l105_results.json').read_text())['dataset'],'preprocessing_reference':{'repository':'https://github.com/twitter-research/tgn','commit':commit,'url':url,'sha256':expected,'license':'Apache-2.0; source retains copyright notice in upstream repository','archive':'sources/l105/tgn_preprocess_data.py.txt','executed_function':'reindex only, all 157474 projected events','verdict':'EXACT typed identity mapping: raw user u -> u+1; raw item v -> v+8228'},'deviations':['No model or paper-table experiment in L105; this is a course representation audit.','Retain separate raw user/item ID columns instead of remapping to one ID space; bijection checked.','Exclude state labels and 172-dimensional features from the studied projection.','Window interaction snapshots differ from cumulative graph states.','Arrival times are unobserved; use a=t only for the measured audit.','First/last timestamps in artifacts are audit metadata, not the studied snapshot features.'],'historical_data_identity':'Pinned currently obtained release bytes, not independently established historical identity','full_paper_reproduction':'NOT_APPLICABLE to this course audit'}
(P/'_sources_l105.json').write_text(json.dumps(manifest,indent=2)+'\n');print('PASS: pinned original reindex on every projected row')

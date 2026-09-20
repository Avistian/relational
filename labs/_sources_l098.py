"""Archive primary API/implementation sources; preserve installed native identities."""
import hashlib,importlib.metadata as md,json,urllib.request
from pathlib import Path
import torch_geometric.loader.neighbor_loader as loader
import torch_geometric.sampler.neighbor_sampler as sampler
import pyg_lib
P=Path(__file__).resolve().parent;D=P/'sources/l098';D.mkdir(exist_ok=True)
commit='584a03d518b2b655580ea8e1cfbbb26bec0a2841'
specs=[('neighbor-loader-2.6.1.py','https://raw.githubusercontent.com/pyg-team/pytorch_geometric/2.6.1/torch_geometric/loader/neighbor_loader.py','Typed fanouts, seed prefix, n_id/e_id and temporal disjoint API'),('relbench-gnn-entity.py',f'https://raw.githubusercontent.com/stanford-star/relbench/{commit}/examples/gnn_entity.py','Actual RelBench NeighborLoader integration; no benchmark replay'),('pyg-lib-sampler.cpp','https://raw.githubusercontent.com/pyg-team/pyg-lib/edc9e2a88d1c5d0953b5f69c98b8365597c6b699/pyg_lib/csrc/sampler/cpu/neighbor_kernel.cpp','Pinned native CPU sampling implementation')]
rows=[]
for name,url,scope in specs:
 path=D/name
 if not path.exists():urllib.request.urlretrieve(url,path)
 rows.append({'url':url,'file':str(path.relative_to(P)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'scope':scope})
for name,module in [('installed-neighbor-loader.py',loader),('installed-neighbor-sampler.py',sampler)]:
 path=D/name;path.write_bytes(Path(module.__file__).read_bytes());rows.append({'file':str(path.relative_to(P)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'scope':'Exact locally executed Python wrapper'})
direct=md.distribution('pyg-lib').read_text('direct_url.json')
report={'sources':rows,'retrieved':'2026-09-20','relbench_commit':commit,'native_source':json.loads(direct),'environment':{k:md.version(k) for k in ['torch','torch-geometric','pyg-lib']},'published_score_target':None,'reason':'Curriculum specifies a synthetic training unit','full_course_protocol':'96 correctness configurations plus 9 fits and temporal intervention','paper_parity':'NOT_ESTABLISHED','relbench_benchmark':'NOT_RUN'}
(P/'_sources_l098.json').write_text(json.dumps(report,indent=2)+'\n');print('Archived',len(rows),'sources at RelBench',commit)

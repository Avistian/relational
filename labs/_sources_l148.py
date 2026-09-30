"""Current byte audit against the fixed upstream release."""
import hashlib,json,urllib.request
from pathlib import Path
P=Path(__file__).resolve().parent;commit='9aa346267c2e1c560bd92da07d6f4ad1ca2f0639';mapping={'gnn_node.py':'examples/gnn_node.py','model.py':'examples/model.py','text_embedder.py':'examples/text_embedder.py','nn.py':'relbench/modeling/nn.py','graph.py':'relbench/modeling/graph.py','utils.py':'relbench/modeling/utils.py','LICENSE':'LICENSE'}
sources={}
for name,path in mapping.items():
 url=f'https://raw.githubusercontent.com/snap-stanford/relbench/{commit}/{path}';raw=urllib.request.urlopen(url,timeout=30).read();assert raw==(P/'sources/l117'/name).read_bytes()
 sources[name]=dict(url=url,sha256=hashlib.sha256(raw).hexdigest())
r=dict(status='PASS',commit=commit,paper='https://arxiv.org/html/2407.20060v1',experiment='Table7 RDL rel-f1/driver-position',sources=sources,canonical_model_sha256=hashlib.sha256((P/'relkit/rdl_l117.py').read_bytes()).hexdigest(),course_extensions=['encoder','messages','history','combined'])
(P/'evidence/l148/sources.json').write_text(json.dumps(r,indent=2));print('PASS:7upstream files byte-identical')

"""Derive a portable subset of existing L074 bytes; authenticate the full parent first."""
import ast,hashlib,json,shutil
from pathlib import Path
import numpy as np
import torch
P=Path(__file__).resolve().parent;E=P/'evidence/b07';D=P/'data/b07';D.mkdir(exist_ok=True)
cfg=json.loads((E/'course-protocol.json').read_text());old=P/'data/l074'
for name,digest in cfg['inputs'].items():
 assert hashlib.sha256((old/name).read_bytes()).hexdigest()==digest,name
for name in ['manifest.json','vectors.npz','wine_pl.parquet','wine_dot_com_prices.parquet','wine_vivino_price.parquet','README.md']:
 shutil.copyfile(old/name,D/name)
source=(P/'relkit/carte_l074.py').read_text();tree=ast.parse(source)
names=['make_graph','grouped_attention','Attention','Readout','Encoder','load_encoder','batch_graphs','embed','normalize_numeric']
code='"""Exact selected L074 CARTE definitions; source parity audited separately."""\nimport math\nimport numpy as np\nimport pandas as pd\nimport torch\nfrom torch import nn\nfrom sklearn.preprocessing import PowerTransformer,StandardScaler\n\n'
code+='\n\n'.join(ast.get_source_segment(source,n) for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names)+'\n'
(P/'relkit/carte_b07.py').write_text(code)
from relkit.carte_b07 import Encoder
state=torch.load(old/'kg_pretrained.pt',map_location='cpu',weights_only=True)
selected={'ft_base.'+k:state['ft_base.'+k] for k in Encoder().state_dict()}
torch.save(selected,D/'selected_checkpoint.pt')
shutil.copyfile(P/'sources/carte-l074/LICENSE.txt',D/'CARTE-LICENSE.txt')
cfg['portable_inputs']={str(p.relative_to(P)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(D.iterdir()) if p.is_file()}
cfg['checkpoint_derivation']=dict(parent_sha256=cfg['inputs']['kg_pretrained.pt'],rule='Preserve every ft_base.+Encoder.state_dict key, tensor exactly equal',selected_keys=list(selected),unused_tensors='omitted from portable package')
cfg['implementation_hashes']={str(p.relative_to(P)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [P/'relkit/carte_b07.py',P/'relkit/semantic_b07.py']}
(E/'course-protocol.json').write_text(json.dumps(cfg,indent=2)+'\n')
print('Prepared portable inputs',sum(p.stat().st_size for p in D.iterdir()),'bytes; encoder tensors',len(selected))

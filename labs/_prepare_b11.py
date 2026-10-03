"""Freeze existing primary-source bytes and complete saved prediction populations."""
import ast,hashlib,json,shutil
from pathlib import Path
P=Path(__file__).resolve().parent;S=P/'sources/b11';E=P/'evidence/b11'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def extract(path,names):
    s=path.read_text();return '\n\n'.join(ast.get_source_segment(s,n) for n in ast.parse(s).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names)
def prepare():
    S.mkdir(parents=True,exist_ok=True);E.mkdir(parents=True,exist_ok=True)
    inherited=json.loads((P/'evidence/l143/training.json').read_text())['source_artifact_hashes']
    for n,h in inherited.items():assert sha(P/'evidence/l143'/n)==h,n
    for n,record in json.loads((P/'_sources_l143.json').read_text())['files'].items():assert sha(P/'sources/l141'/n)==record['sha256'],n
    for n,h in json.loads((P/'evidence/l145/source.json').read_text())['sha256'].items():assert sha(P/'sources/l145'/n)==h,n
    paths=[]
    for old in ['l141','l145']:
        for f in (P/'sources'/old).iterdir():
            if f.is_file():paths.append(f)
    for name in ['l143-reproduction.md','l145-reproduction.md','l146-reproduction.md','_full_l143.py','_full_l145.py','relkit/relgnn_l143.py','relkit/relgnn_l141.py','relkit/relgt_l145.py','relkit/relgt_contracts_l145.py']:
        paths.append(P/name)
    for seed in range(5):
        for name in ['result.json','predictions.npz']:paths.append(P/f'evidence/l143/seed-{seed}/{name}')
    for seed in range(3):
        for arm in ['gnn','relgt']:
            for name in ['result.json','predictions.npz']:paths.append(P/f'evidence/l146/fit-{arm}-{seed}/{name}')
    for split in ['val','test']:paths.append(P/f'evidence/l146/prepared/{split}.npz')
    for n in ['l143/training.json','l146/summary.json','l146/sources.json','l145/prepared/audit.json','l145/cost-decision.json','l145/source.json']:
        if (P/'evidence'/n).exists():paths.append(P/'evidence'/n)
    ledger={}
    for f in paths:
        rel=f.relative_to(P);dest=S/'archive'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(f,dest);ledger[str(rel)]=sha(f)
    # Preserve whole source files above; these executable block extracts have no repo imports.
    imports='import math\nimport numpy as np\nimport torch\nfrom torch import nn\nimport torch.nn.functional as F\nfrom einops import rearrange\nfrom torch_geometric.nn.dense.linear import Linear\n'
    gt=imports+'\n'+extract(P/'sources/l145/codebook.py',['VectorQuantizerEMA'])+'\n'+extract(P/'sources/l145/local_module.py',['LocalModule','FeedForwardNetwork','EncoderLayer'])+'\n'+extract(P/'sources/l145/model.py',['RelGTLayer'])+'\n'
    (S/'relgt_original.py').write_text(gt)
    (S/'relgt_visible.py').write_text('from relkit.baselines_b11 import explicit_attention\n'+gt.replace('F.scaled_dot_product_attention(', 'explicit_attention('))
    gnn='import math\nimport torch\nfrom torch_geometric.nn.dense.linear import Linear\nfrom torch_geometric.nn.conv import TransformerConv, SAGEConv\n'
    (S/'relgnn_visible.py').write_text(gnn+extract(P/'relkit/relgnn_l141.py',['destination_softmax','RelGNNConv'])+'\n')
    shutil.copyfile(P/'sources/l141/examples__relgnn_conv.py',S/'relgnn_original.py')
    result=dict(revisions=dict(RelGNN='cffdb8b54627e92c7dd112c1243dde739c90d35b',RelGT='19e423ca3e7cac761130aba790857f2dc3a46ef7'),archive=ledger,extracts={f.name:sha(f) for f in S.glob('*.py')},authentication='Both upstream source sets and L143 prediction/result bytes match inherited ledgers. L146 predictions sealed at B11 preparation and checked against saved aggregate metrics; new sealing alone is not historical provenance.')
    (S/'source-ledger.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Prepared',len(ledger),'archived files and four executable block extracts')
if __name__=='__main__':prepare()

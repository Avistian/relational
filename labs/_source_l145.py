"""Generate the visible source-adapted model; vendor files remain unmodified."""
from pathlib import Path
import ast,hashlib,json
P=Path(__file__).resolve().parent;S=P/'sources/l145'
chunks=['# Source-adapted RelGT, MIT license in sources/l145/LICENSE.\n# Pinned revision 19e423ca3e7cac761130aba790857f2dc3a46ef7.\nfrom relkit.relgt_contracts_l145 import mix_five\n']
for name in ['codebook.py','encoders.py','local_module.py','model.py']:
    s=(S/name).read_text()
    # Flatten imports of the four source modules; architecture remains visible.
    s='\n'.join(line for line in s.splitlines() if not line.startswith(('from codebook import','from local_module import','from encoders import')))
    if name=='model.py':
        s=s.replace('x_set = torch.cat(cat_list, dim=-1)        \n        x_set = self.in_mixture(x_set)', 'x_set = mix_five(cat_list, self.in_mixture) if self.ablate_idx is None else self.in_mixture(torch.cat(cat_list, dim=-1))')
    chunks.append('\n# ---- '+name+' ----\n'+s+'\n')
(P/'relkit/relgt_l145.py').write_text('\n'.join(chunks))
ledger={'revision':'19e423ca3e7cac761130aba790857f2dc3a46ef7','sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in S.iterdir() if p.is_file()},'adaptations':['Four model files flattened; five-component concatenation routes through learner mix_five. No model behavior fixes.','Serial token precomputation preserves released last-entity overwrite and Python hash seed is fixed to 0.','Single-device runner omits DDP/SyncBatchNorm and tracking services; worker count differs. Historical environment and cache identity unavailable.']}
(P/'evidence/l145/source.json').write_text(json.dumps(ledger,indent=2))

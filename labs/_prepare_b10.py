"""Authenticate inherited primary-source bytes, retain provenance, freeze B10 inputs."""
import hashlib,json,shutil
from pathlib import Path
P=Path(__file__).resolve().parent;S=P/'sources/b10';E=P/'evidence/b10';old=P/'sources/l175'
ledger=json.loads((old/'source-ledger.json').read_text())
for row in ledger['sources']:
    path=old/row['file'];assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256'],path
shutil.copytree(old/'upstream',S/'upstream',dirs_exist_ok=True)
for name in ['model-card.md','example_pretrain.py','example_finetune.py','table_info.json','column_index.json','relbench-f1-task.py']:
    shutil.copyfile(old/name,S/name)
shutil.copyfile(old/'source-ledger.json',S/'inherited-source-ledger.json')
files={str(p.relative_to(S)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(S.rglob('*')) if p.is_file() and '__pycache__' not in str(p)}
inputs=[P/f'evidence/l175/audit-3/contexts-{s}.npz' for s in range(3)]+[P/'evidence/l175/audit-3/context-audit.json',P/'evidence/l175/label-oracle.npz']
packet={'source_revision':ledger['code_revision'],'checkpoint_revision':ledger['checkpoint_revision'],'data_revision':ledger['data_revision'],'files':files,'inherited_contexts':{str(p.relative_to(P)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},'checkpoint_bytes':'NOT_DOWNLOADED_NOT_AUTHENTICATED','scope':'Original RT-v1, not current RT-J; inherited source and saved native contexts'}
(S/'source-ledger.json').write_text(json.dumps(packet,indent=2)+'\n')
print('Authenticated',len(files),'source files and',len(inputs),'saved audit inputs')

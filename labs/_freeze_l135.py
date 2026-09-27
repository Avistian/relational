"""Freeze validation-only choice and exact complete search bytes before test runs."""
import datetime,hashlib,json
from pathlib import Path
from relkit.tuning_l135 import select_configuration
P=Path(__file__).parent;R=P.parent;E=P/'evidence/l135'
protocol=json.loads((P/'_protocol_l135.json').read_text());rows=[];hashes={}
assert not (E/'frozen.json').exists(),'Freeze is immutable; use a new experiment for a new decision'
for c in protocol['configurations']:
 for seed in protocol['search_seeds']:
  root=E/'search'/c['id']/f'seed-{seed}'
  result=json.loads((root/'result.json').read_text());done=json.loads((root/'completed.json').read_text())
  assert done['status']=='COMPLETE' and result['epochs']==10 and len(result['trace'])==10
  assert not result['evaluate_test'] and set(result['scores'])=={'val'} and set(result['temporal_audit'])=={'train','val'}
  assert result['selection_mae']==min(t['val_mae'] for t in result['trace'])
  rows.append(dict(config=c['id'],seed=seed,selection_mae=result['selection_mae']))
  for path in root.iterdir():hashes[str(path.relative_to(R))]=hashlib.sha256(path.read_bytes()).hexdigest()
frozen=select_configuration(rows,[c['id'] for c in protocol['configurations']],protocol['search_seeds'])
frozen.update(protocol_sha256=hashlib.sha256((P/'_protocol_l135.json').read_bytes()).hexdigest(),search_hashes=hashes,frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),records=rows,test_access='FORBIDDEN_IN_SEARCH',scope='Winner chosen before any final phase runs')
(E/'frozen.json').write_text(json.dumps(frozen,indent=2));print(json.dumps({k:frozen[k] for k in ['winner','ranking','frozen_utc']},indent=2))

"""Authenticate archived upstream code and the exact scientific implementation bytes."""
import hashlib,json,importlib.metadata,platform
from pathlib import Path
P=Path(__file__).resolve().parent;r=json.loads((P/'_sources_l110.json').read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for name,digest in r['archived_source_sha256'].items():assert sha(P/'sources/l102'/name)==digest,name
assert sha(P/'relkit/checkpoint_l110.py')==r['implementation_sha256']
pilot=P/'evidence/l110/pilot/identity.json'
if pilot.exists():assert json.loads(pilot.read_text())['source_sha256']==r['implementation_sha256']
out={'status':'PASS','archived_source_files':len(r['archived_source_sha256']),'source_commit':r['source_commit'],'implementation_sha256':r['implementation_sha256'],'local_environment':{'python':platform.python_version(),**{k:importlib.metadata.version(k) for k in ['torch','numpy','pandas','scikit-learn','nbformat','nbclient','nbconvert']}},'gpu_environment':'Per-run identity.json and GPU pilot source_parity.json','raw_and_processed_arrays':'Independently checked in _audit_l110_results.json','data_historical_identity':'NOT_ESTABLISHED'}
(P/'_provenance_l110_results.json').write_text(json.dumps(out,indent=2));print(out)

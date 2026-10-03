"""Freeze the approved original source, environment recipe and four-case protocol."""
import hashlib,json,shutil,subprocess
from pathlib import Path
P=Path(__file__).resolve().parent;E=P/'evidence/l196';S=E/'packet/source';up=P/'sources/l193'
ledger=json.loads((up/'source-ledger.json').read_text());files={}
for name,digest in ledger['files'].items():
 src=up/name
 assert hashlib.sha256(src.read_bytes()).hexdigest()==digest,name
 if name.startswith(('rdblearn/','fastdfs/')):
  dest=S/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(src.read_bytes());files['source/'+name]=digest
freeze=subprocess.check_output(['/tmp/l192-repro-env/bin/python','-m','pip','freeze'],text=True)
(E/'requirements-diagnostic.txt').write_text(freeze)
protocol=dict(name='L196 complete original-preprocessor four-case diagnostic',commit=ledger['commit'],cases=['a','z','0','e'],train_categories=['b','c','d']*4,numeric_train=list(range(12)),known_categories=['b','c','d'],known_numeric=[1,2,3],scope='Synthetic full preprocessing only; no model or task score',cloud_usd=0,cap_local_seconds=1800)
(E/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
(E/'packet/source-manifest.json').write_text(json.dumps(dict(commit=ledger['commit'],files=files,provenance='Copied byte-for-byte from authenticated L193 source ledger; fresh import hash checks'),indent=2)+'\n')
print('Frozen',len(files),'source files')

from pathlib import Path
import json,modal
P=Path(__file__).resolve().parent;v=modal.Volume.from_name('l153-recommendation-evidence');root=P/'evidence/l153/notebook-retry1';root.mkdir(parents=True,exist_ok=True)
for name in ['execution.json','cost.json','l153-report.json','l153-portfolio-entry.json']:
 (root/name).write_bytes(b''.join(v.read_file('notebook-retry1/'+name)))
r=json.loads((root/'execution.json').read_text());r['cost']=json.loads((root/'cost.json').read_text());(P/'_notebook_l153_results.json').write_text(json.dumps(r,indent=2));print(r)

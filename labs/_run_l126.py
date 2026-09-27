"""Execute the complete available-data API tour, never call it beta reproduction."""
import hashlib,importlib.metadata,json,shutil,tempfile
from pathlib import Path
from relkit.beta_l126 import f1_tour
P=Path(__file__).resolve().parent;E=P/'evidence/l126';E.mkdir(parents=True,exist_ok=True)
for name in ['f1-db.zip','f1-task.zip']:
    target=E/name
    if not target.exists():shutil.copyfile(P/'evidence/l125'/name,target)
with tempfile.TemporaryDirectory(prefix='l126-f1-') as tmp:
    report,preds=f1_tour((E/'f1-db.zip').read_bytes(),(E/'f1-task.zip').read_bytes(),tmp)
report['source_archives']={name:hashlib.sha256((E/name).read_bytes()).hexdigest() for name in ['f1-db.zip','f1-task.zip']}
(E/'summary.json').write_text(json.dumps(report,indent=2)+'\n');preds.to_csv(E/'f1-predictions.csv',index=False)
env={x:importlib.metadata.version(x) for x in ['relbench','numpy','pandas','pyarrow','scikit-learn','nbformat','nbclient','playwright']}
(P/'_environment_l126.json').write_text(json.dumps(env,indent=2)+'\n');print({k:v for k,v in report.items() if k!='schema'})

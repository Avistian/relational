"""Freeze validation-only candidate selection; refuse any later reselection."""
import datetime,hashlib,json
from pathlib import Path
from relkit.portfolio_l151 import select_candidate
P=Path(__file__).resolve().parent;E=P/'evidence/l151';out=E/'frozen.json';assert not out.exists()
rows=[];files={}
for lr in [.00005,.0001,.0002]:
 path=E/f'search-{int(lr*1000000):03d}/result.json';r=json.loads(path.read_text())
 assert r['test_access']=='FORBIDDEN' and set(r['scores'])=={'val'}
 assert r['selection_auc']==max(h['val']['roc_auc'] for h in r['history'])
 rows.append(dict(lr=r['lr'],seed=r['seed'],epochs=r['epochs'],complete=r['status']=='COMPLETE',split='val',selection_auc=r['selection_auc']))
 files[str(path.relative_to(P.parent))]=hashlib.sha256(path.read_bytes()).hexdigest()
selected=select_candidate(rows)
r=dict(lr=selected,candidates=rows,files=files,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),selection='first maximum-validation checkpoint score; lower learning rate breaks ties; seed100 only',selected_seeds=list(range(10,15)),test_access=False,prior_test_exposure='Prior course lessons used this test population; no pristine holdout claim')
out.write_text(json.dumps(r,indent=2));print(r)

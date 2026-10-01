"""One-time input freeze. Refuses to silently bless changed evidence."""
import hashlib,json
from pathlib import Path
from bs4 import BeautifulSoup
P=Path(__file__).resolve().parent;E=P/'evidence/l154';E.mkdir(exist_ok=True)
if (E/'input-manifest.json').exists():raise SystemExit('Already frozen; do not overwrite')
source='https://arxiv.org/html/2407.20060v1'
context=dict(source=source,paper_version='v1',tasks={
 'rel-trial/study-outcome':dict(metric='AUROC',unit='fraction',model='RDL',model_mean=.686,model_sd=.0101,baseline='Raw-table LightGBM',baseline_mean=.7009,baseline_sd=.0141,tolerance=.01,source=source+'#A2.T6'),
 'rel-f1/driver-position':dict(metric='MAE',unit='position',model='RDL',model_mean=4.022,model_sd=.119,baseline='Raw-table LightGBM',baseline_mean=4.170,baseline_sd=.137,tolerance=.20,source=source+'#A2.T7'),
 'rel-trial/site-sponsor-run':dict(metric='MAP@10',unit='fraction',model='GraphSAGE',model_mean=.107,model_sd=.011,baseline='Past Visit (ranking heuristic)',baseline_mean=.1731,baseline_sd=None,tolerance=.02,source=source+'#A2.T8')},
 note='Published test values only. Not fresh FE/tree measurements. Recommendation target is GraphSAGE, not ID-GNN. Percentage metrics stored as fractions.')
# Exact table excerpt retained from the already source-pinned paper. Manually read target rows.
soup=BeautifulSoup((P/'sources/l151/paper.html').read_text(),'html.parser')
excerpt='<!doctype html><html lang="en"><meta charset="utf-8"><title>RelBench v1 tables 6–8</title><body><p>Source: <a href="'+source+'">RelBench v1</a></p>'+''.join(str(soup.find(id=i)) for i in ['A2.T6','A2.T7','A2.T8'])+'</body></html>'
(E/'published-tables.html').write_text(excerpt)
(E/'published-context.json').write_text(json.dumps(context,indent=2)+'\n')
files=['sources/l151/protocol.json','sources/l151/manifest.json','sources/l151/paper.html',
       'sources/l152/protocol.json','sources/l153/protocol.json','evidence/l151/frozen.json',
       'evidence/l151/prepared/queries.npz','evidence/l151/prepared/task_audit.json',
       'evidence/l152/label-audit.json','evidence/l153/prepared/val-truth.json.gz',
       'evidence/l153/cost-decision.json','evidence/l154/published-context.json','evidence/l154/published-tables.html']
for phase in [f'ref-{s}' for s in range(5)]+[f'selected-{s}' for s in range(10,15)]:
 files += [f'evidence/l151/{phase}/{name}' for name in ['result.json','predictions.npz']]
for s in range(5):
 files += [f'evidence/l152/paper/seed-{s}/{name}' for name in ['result.json','predictions.npz','completed.json','temporal-audit.json']]
files += ['evidence/l153/pilot/'+name for name in ['result.json','predictions.npz']]
manifest=dict(experiment='L154 RelBench portfolio evidence replay',freeze_date='2026-10-01',
              boundary='Hash integrity of captured inputs; no historical-identity claim.',
              files={name:hashlib.sha256((P/name).read_bytes()).hexdigest() for name in sorted(files)})
(E/'input-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Pinned',len(files),'files')

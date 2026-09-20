"""Execute exact standalone solution code in a second environment, with fresh data fetch."""
import hashlib,importlib.metadata as metadata,json,tempfile,os
from pathlib import Path
P=Path(__file__).resolve().parent
nb=json.loads((P/'solutions/0097-negative-sampling.ipynb').read_text());ns={'__name__':'__main__'}
with tempfile.TemporaryDirectory(prefix='l097-portable-') as folder:
 os.chdir(folder)
 count=0
 for cell in nb['cells']:
  if cell['cell_type']=='code':
   exec(compile(''.join(cell['source']),'<l097-inline>','exec'),ns);count+=1
 fresh=json.loads(Path('l097-fresh.json').read_text())
author=json.loads((P/'_experiment_l097_results.json').read_text())
gap=max(abs(a[k]-b[k]) for a,b in zip(author['runs'],fresh['runs']) for k in ['full_recall','full_ndcg','sampled_recall','sampled_ndcg'])
assert len(fresh['runs'])==45 and gap<.01
report={'status':'PASS','fresh_fits':45,'executed_code_cells':count,'max_absolute_per_run_metric_gap':gap,'portability_tolerance':.01,'same_run_records':fresh['runs']==author['runs'],'environment':{x:metadata.version(x) for x in ['numpy','torch']},'archive_sha256':fresh['archive_sha256'],'inline_source_sha256':hashlib.sha256(''.join(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code').encode()).hexdigest(),'live_colab':'NOT_CHECKED','scope':'local second Python environment, fresh verified network download, all runs recomputed'}
(P/'_portable_l097_results.json').write_text(json.dumps(report,indent=2)+'\n');print(report)

"""Audit complete fetched original-suite tables; expose missing IID counterpart."""
import hashlib,json,re,urllib.request
from pathlib import Path
import pandas as pd
P=Path(__file__).resolve().parent;S=P/'sources/b19';E=P/'evidence/b19'
receipts=json.loads((S/'result-fetch.json').read_text());records=[];coverage=[]
for receipt in receipts:
 if receipt['part']!='results/model_results.parquet':continue
 path=S/receipt['file'];assert hashlib.sha256(path.read_bytes()).hexdigest()==receipt['sha256']
 df=pd.read_parquet(path);sel=df[df.dataset.str.startswith(('musk','sat11_hand_algo_runtime'))].copy()
 assert not sel.duplicated(['dataset','fold','method']).any()
 assert set(sel.ta_suite)=={'beyond_iid_benchmark_2026'}
 assert sel.metric_error.notna().all()
 for dataset,group in sel.groupby('dataset'):
  n=60 if dataset.startswith('musk') else 30
  configs=1 if receipt['method'] in ['TA-TabPFN-2.6','TA-TabICLv2'] else 26
  assert len(group)==n*configs and set(group.fold)==set(range(n)) and group.method.nunique()==configs
  coverage.append(dict(model=receipt['method'],dataset=dataset,folds=n,configs=configs,rows=len(group),source=receipt['file']))
 records.extend(sel.to_dict('records'))
 assert not df.dataset.str.startswith(('musk_iid','sat11_hand_algo_runtime_iid')).any(), 'IID variants found: audit target before gating'
(E/'released-grouped-scores.json').write_text(json.dumps(records,indent=2,allow_nan=False)+'\n')
defaults=[]
for c in coverage:
 rows=[r for r in records if r['dataset']==c['dataset'] and r['ta_name']==c['model'] and r['method']==c['model']+'_c1_BAG_L1']
 assert len(rows)==c['folds']
 defaults.append(dict(dataset=c['dataset'],model=c['model'],folds=len(rows),metric=rows[0]['metric'],mean_error=sum(r['metric_error'] for r in rows)/len(rows)))
# Complete root listing (not the initial recursive response's 1000-entry prefix).
u='https://huggingface.co/api/datasets/TabArena/BeyondArena/tree/main?limit=1000';req=urllib.request.Request(u,headers={'User-Agent':'B19-source-audit'})
with urllib.request.urlopen(req,timeout=30) as response:
 root=json.load(response);link=response.headers.get('Link','')
(S/'hf-root.json').write_text(json.dumps(dict(url=u,next_link=link,entries=root),indent=2)+'\n')
assert 'rel="next"' not in link,'Paginate before concluding root absence'
names=[x['path'] for x in root];missing=[x for x in ['musk_iid','sat11_hand_algo_runtime_iid'] if x not in names]
report=dict(status='INCOMPLETE_SOURCE_PROTOCOL_GATE',target='B19-BEYONDARENA-FIG-F2',released_grouped_table='COMPLETE_SELECTED_SCORE_TABLE_AUDIT',rows=len(records),coverage=coverage,default_summary=defaults,iid_score_rows_found=0,root_names_absent=missing,source_identity='Downloaded original-suite namespace at pinned current implementation; historical paper identity NOT_ESTABLISHED',gaps=['No IID counterpart in any of the six fetched original-suite score tables','Original Figure F.2 full result artifact and exact plotting/variant selection not authenticated','Raw per-query predictions and checkpoint identities not authenticated; stored metric recomputation is not prediction rescoring'],fresh_runs='NOT_RUN',whole_paper='NOT_RUN',search_boundary='Pinned repository tree, declared original-suite metadata/model/HPO objects, curation notebooks, complete public dataset root listing; this is not a proof artifacts do not exist elsewhere')
scoped=[]
for dataset,uuid in [('musk','019dd49e-4009-7822-a3a5-bb615a182e0f'),('sat11_hand_algo_runtime','019dd4a1-d364-7e7e-9ebd-af85a6b33ecb')]:
    path=S/(dataset+'-hf-tree.json')
    if not path.exists():
        uri=f'https://huggingface.co/api/datasets/TabArena/BeyondArena/tree/main/{dataset}?recursive=true&limit=1000'
        with urllib.request.urlopen(uri,timeout=30) as response:
            entries=json.load(response);next_link=response.headers.get('Link','')
        path.write_text(json.dumps(dict(url=uri,next_link=next_link,entries=entries),indent=2)+'\n')
    tree=json.loads(path.read_text())
    assert 'rel="next"' not in tree['next_link']
    scoped.append(dict(dataset=dataset,notebook_iid_uuid=uuid,present=any(uuid in x['path'] for x in tree['entries']),checked_entries=len(tree['entries'])))
report['scoped_iid_uuid_search']=scoped
(E/'source-audit.json').write_text(json.dumps(report,indent=2)+'\n')
# Seal every archived input, including discovery responses. Do not silently reseal at execution.
old=json.loads((S/'manifest.json').read_text());known={x['file']:x.get('url') for x in old['files']}
known.update({r['file']:r['url'] for r in receipts if r['file']})
known['downloader.py']='https://raw.githubusercontent.com/autogluon/tabarena/1ce4cae6c12972227dea6247bac83234a2915748/packages/tabarena/src/tabarena/models/_artifacts/downloader.py'
known['hf-root.json']=u
old['files']=[dict(file=p.name,url=known.get(p.name,'local discovery receipt'),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(S.iterdir()) if p.is_file() and p.name!='manifest.json']
(S/'manifest.json').write_text(json.dumps(old,indent=2)+'\n')
print({k:v for k,v in report.items() if k not in ['coverage','default_summary']})

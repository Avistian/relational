"""Authenticated historical score-table reconstruction; never dispatches model fits."""
import argparse,hashlib,json,math,sys,importlib.metadata
from pathlib import Path
P=Path(__file__).resolve().parent;S=P/'sources/b19a';E=P/'evidence/b19a'

def replay(records):
    """Portable mean-rank/median reconstruction from ALL released selected scores.

Rows: [dataset, model, fold, crps, r2, crls]. None denotes missing, never zero.
Same release policy: model coverage>=90%, then complete dataset intersection.
"""
    from collections import defaultdict
    import statistics
    keys=set();groups=defaultdict(list);datasets=set();models=set()
    for row in records:
        if len(row)!=6:raise ValueError('Invalid row width')
        d,m,f,*scores=row
        if not isinstance(d,str) or not isinstance(m,str) or f not in range(5) or (d,m,f) in keys:raise ValueError('Duplicate or invalid score key')
        if any(x is not None and not math.isfinite(x) for x in scores):raise ValueError('Nonfinite score')
        keys.add((d,m,f));datasets.add(d);models.add(m);groups[d,m].append(row)
    for key,rows in groups.items():
        if {r[2] for r in rows}!=set(range(5)):raise ValueError('Incomplete folds: '+str(key))
    output={}
    for j,metric in enumerate(['crps','r2','crls'],3):
        means={key:statistics.mean(r[j] for r in rows if r[j] is not None) for key,rows in groups.items() if any(r[j] is not None for r in rows)}
        included=sorted(m for m in models if sum((d,m) in means for d in datasets)>=.9*len(datasets))
        support=sorted(d for d in datasets if all((d,m) in means for m in included))
        if not included or not support:raise ValueError('Empty comparable support')
        stats=[]
        for m in included:
            ranks=[]
            for d in support:
                v=means[d,m];others=[means[d,x] for x in included]
                better=sum(x>v if metric=='r2' else x<v for x in others);ties=sum(x==v for x in others)
                ranks.append(1+better+(ties-1)/2)
            stats.append(dict(model=m,meanrank=statistics.mean(ranks),median=statistics.median(means[d,m] for d in support)))
        output[metric]=dict(datasets=support,dropped_datasets=sorted(datasets-set(support)),models=included,rows=sorted(stats,key=lambda x:(x['meanrank'],x['model'])))
    return output

def run():
    import numpy as np,pandas as pd
    from bs4 import BeautifulSoup
    manifest=json.loads((S/'manifest.json').read_text())
    for entry in manifest['files']:
        if hashlib.sha256((S/entry['file']).read_bytes()).hexdigest()!=entry['sha256']:raise ValueError('SOURCE_HASH_MISMATCH '+entry['file'])
    records=[]
    for file in sorted((S/'output').glob('*.parquet')):
        frame=pd.read_parquet(file)
        assert set(frame.model)=={file.stem}
        for row in frame.to_dict('records'):
            records.append([row['dataset'],file.stem,int(row['fold']),*[float(row[k]) if pd.notna(row[k]) else None for k in ['crps','r2','crls']]])
    records.sort(key=lambda r:(r[0],r[1],r[2]));summary=replay(records)
    (E/'released-scores.json').write_text(json.dumps(records,separators=(',',':'),allow_nan=False)+'\n')
    (E/'portable-replay.json').write_text(json.dumps(summary,indent=2)+'\n')
    # Explicit local version; original environment was not locked by authors.
    sys.path.insert(0,'/tmp/b19a-python-deps')
    from autorank import autorank
    source=BeautifulSoup((S/'paper.html').read_text(),'html.parser');comparisons={};raw= pd.DataFrame(records,columns=['dataset','model','fold','crps','r2','crls'])
    for metric,table in [('crps','S4.T1'),('r2','A4.T4'),('crls','A4.T5')]:
        matrix=raw.groupby(['dataset','model'])[metric].mean().unstack().loc[summary[metric]['datasets'],summary[metric]['models']]
        result=autorank(matrix,alpha=.05,order='descending' if metric=='r2' else 'ascending')
        released=json.loads((S/f'output/figures/leaderboard/cd_data_{metric}.json').read_text())['autorank']['models'];released={r['name']:r for r in released}
        targets={}
        for tr in source.find(id=table).select('tr'):
            cells=[x.get_text(' ',strip=True) for x in tr.find_all(['td','th'])]
            if len(cells)!=7 or cells[0] not in released:continue
            lo,hi=[float(v.strip()) for v in cells[4].strip('[]').split(',')]
            targets[cells[0]]=dict(meanrank=float(cells[1]),median=float(cells[2]),mad=float(cells[3]),ci_lower=lo,ci_upper=hi,effect_size=float(cells[5]),magnitude=cells[6])
        assert set(targets)==set(matrix.columns) and len(targets)==38
        errors=[];release_errors=[];rows=[]
        for model,row in result.rankdf.iterrows():
            actual={k:float(row[k]) for k in ['meanrank','median','mad','ci_lower','ci_upper','effect_size']};actual['magnitude']=row['magnitude']
            for field,value in actual.items():
                expected=targets[model][field];rel=released[model][field]
                if isinstance(value,str):
                    if value!=expected:errors.append([model,field,value,expected])
                    if value!=rel:release_errors.append([model,field,value,rel])
                else:
                    if abs(value-expected)>.000500001:errors.append([model,field,value,expected])
                    if not math.isclose(value,rel,abs_tol=1e-10,rel_tol=1e-10):release_errors.append([model,field,value,rel])
            portable=next(x for x in summary[metric]['rows'] if x['model']==model)
            assert abs(actual['meanrank']-portable['meanrank'])<1e-12 and abs(actual['median']-portable['median'])<1e-10
            rows.append(dict(model=model,actual=actual,published=targets[model]))
        comparisons[metric]=dict(table=table,models=38,datasets=len(matrix),published_cells=38*7,paper_mismatches=errors,release_mismatches=release_errors,rows=rows,omnibus=result.omnibus,posthoc=result.posthoc,pvalue=float(result.pvalue),cd=float(result.cd))
    status='COMPLETE_SELECTED_PUBLISHED_TABLE_RECONSTRUCTION' if all(not c['paper_mismatches'] for c in comparisons.values()) else 'INCOMPLETE_TABLE_PARITY'
    report=dict(status=status,score_records=len(records),source_models=len(set(r[1] for r in records)),source_datasets=len(set(r[0] for r in records)),code_commit=manifest['code_commit'],output_commit=manifest['output_commit'],comparisons=comparisons,identity='Historical release on v3 submission day; table equality supports selected score identity, not original environment/checkpoint/raw-prediction identity',fresh_fits='NOT_RUN',raw_prediction_rescoring='NOT_RUN',whole_paper='NOT_RUN',versions={k:importlib.metadata.version(k) for k in ['autorank','pandas','numpy','scipy','statsmodels']})
    (E/'reproduction.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(status,len(records),[(k,c['datasets'],len(c['paper_mismatches']),len(c['release_mismatches'])) for k,c in comparisons.items()])
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--fresh',action='store_true');args=ap.parse_args()
    if args.fresh:raise SystemExit('NOT_RUN: fresh fits require authenticated data/split/checkpoint/environment protocol and separate approved compute scope')
    run()

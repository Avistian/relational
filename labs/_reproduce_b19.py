"""Authenticate archived sources and replay released score aggregates. No trainer."""
import argparse,hashlib,json,math
from pathlib import Path
P=Path(__file__).resolve().parent

def replay_grouped(records):
    """Recompute every default mean; authenticate full 26-config/1-config coverage."""
    groups={};keys=set()
    for r in records:
        key=(r['dataset'],r['fold'],r['method'])
        if key in keys or not math.isfinite(r['metric_error']):raise ValueError('Duplicate or invalid published score')
        keys.add(key)
        if r['ta_suite']!='beyond_iid_benchmark_2026':raise ValueError('Wrong source suite')
        groups.setdefault((r['dataset'],r['ta_name']),[]).append(r)
    expected_models={'LinearModel','ExtraTrees','LightGBM','TA-RealMLP','TA-TabPFN-2.6','TA-TabICLv2'}
    expected_datasets={'musk-16dca0dbddf5','sat11_hand_algo_runtime-cda3af888024'}
    if set(groups)!={(d,m) for d in expected_datasets for m in expected_models}:raise ValueError('Incomplete selected matrix')
    result=[]
    for (d,m),rows in sorted(groups.items()):
        n=60 if d.startswith('musk') else 30;configs=1 if m in ['TA-TabPFN-2.6','TA-TabICLv2'] else 26
        # Actual naming is audited below by count and complete per-config folds; do not infer search values from names.
        names={r['method'] for r in rows}
        if len(rows)!=n*configs or len(names)!=configs:raise ValueError('Wrong config count')
        for name in names:
            if {r['fold'] for r in rows if r['method']==name}!=set(range(n)):raise ValueError('Missing original fold')
        metric='roc_auc' if d.startswith('musk') else 'rmse'
        if {r['metric'] for r in rows}!={metric}:raise ValueError('Wrong metric')
        default=[r['metric_error'] for r in rows if r['method']==m+'_c1_BAG_L1']
        if len(default)!=n:raise ValueError('Missing default config')
        result.append(dict(dataset=d,model=m,folds=n,metric=metric,mean_error=sum(default)/n))
    return result

def audit():
    s=P/'sources/b19';manifest=json.loads((s/'manifest.json').read_text())
    for r in manifest['files']:
        if hashlib.sha256((s/r['file']).read_bytes()).hexdigest()!=r['sha256']:raise ValueError('SOURCE_HASH_MISMATCH: '+r['file'])
    records=json.loads((P/'evidence/b19/released-grouped-scores.json').read_text());result=replay_grouped(records)
    # Full-table extraction equality, independently performed here from the immutable originals.
    import pandas as pd
    extracted=[]
    for file in sorted(s.glob('*--results-model_results.parquet')):
        df=pd.read_parquet(file);extracted.extend(df[df.dataset.isin(['musk-16dca0dbddf5','sat11_hand_algo_runtime-cda3af888024'])].to_dict('records'))
    key=lambda r:(r['dataset'],r['fold'],r['method'])
    if sorted(records,key=key)!=sorted(extracted,key=key):raise ValueError('EXTRACTION_MISMATCH')
    report=json.loads((P/'evidence/b19/source-audit.json').read_text())
    expected=sorted(report['default_summary'],key=lambda r:(r['dataset'],r['model']))
    for a,b in zip(result,expected):
        if any(a[k]!=b[k] for k in ['dataset','model','metric','folds']) or abs(a['mean_error']-b['mean_error'])>1e-12:raise ValueError('AGGREGATE_MISMATCH')
    return dict(status='PASS',source_files=len(manifest['files']),released_score_rows=len(records),default_aggregates=len(result),paper_figure='INCOMPLETE_SOURCE_PROTOCOL_GATE',prediction_rescoring='NOT_RUN')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['audit','paper'],default='audit');args=parser.parse_args()
    result=audit();print(json.dumps(result,indent=2))
    if args.phase=='paper':raise SystemExit('NOT_RUN: missing authenticated IID ablation scores, original figure selection and checkpoint/prediction identities. This source gate is not a trainer.')
    (P/'_reproduce_b19_results.json').write_text(json.dumps(result,indent=2)+'\n')

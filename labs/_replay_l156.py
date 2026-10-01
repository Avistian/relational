"""Portable independent audit replay; all three learner functions are injected."""
import hashlib,json,statistics
from pathlib import Path
import numpy as np
from relkit.temporal_audit_l156 import audit_observations,audit_label_windows,audit_verdict

def verify_inputs(root,manifest):
    root=Path(root)
    for name,digest in manifest['files'].items():
        p=(root/name).resolve()
        if not p.is_relative_to(root.resolve()):raise ValueError('Unsafe path')
        if hashlib.sha256(p.read_bytes()).hexdigest()!=digest:raise ValueError('Changed evidence: '+name)

def replay(root,manifest,observations=audit_observations,labels=audit_label_windows,verdict=audit_verdict):
    root=Path(root);verify_inputs(root,manifest)
    pre=json.loads((root/'preflight.json').read_text());sql=json.loads((root/'fe/sql-audit.json').read_text())
    assert pre['status']==sql['status']=='PASS'
    a=np.load(root/'label-events.npz');events=[dict(entity=int(e),time=int(t),value=float(y)) for e,t,y in zip(a['entity'],a['time'],a['value'])]
    label_counts={};deps={};truth={}
    for split,n in [('train',7453),('val',499),('test',760)]:
        a=np.load(root/f'{split}-labels.npz')
        qs=[dict(entity=int(e),time=int(t),target=float(y)) for e,t,y in zip(a['entity'],a['time'],a['target'])]
        r=labels(qs,events,60*86400);assert r['status']=='PASS' and r['queries']==n;label_counts[split]=n
        truth[split]={(q['entity'],q['time']*10**9):q['target'] for q in qs}
        x=np.load(root/f'{split}-dependencies.npz')
        records=[dict(owner=int(i),cutoff=int(t),event=int(e) if k!='static' else None,available=None,rule='strict' if k=='event' else 'inclusive',kind=str(k)) for i,t,e,k in zip(x['owner'],x['cutoff'],x['event'],x['kind'])]
        r=observations(records);assert r['status']=='NOT_ESTABLISHED' and r['counts']['FAIL']==0;deps[split]=r['counts']
    lanes={};all_predictions=0
    for lane in ['paper','fit_horizon']:
        records=[];coverage={k:0 for k in ['queries','batches','dated_node_occurrences','edge_occurrences','equality_nodes','owner_table_checks']};gradient={};maxerr=0
        for seed in range(5):
            p=root/lane/f'seed-{seed}';r=json.loads((p/'result.json').read_text());done=json.loads((p/'completed.json').read_text());t=json.loads((p/'temporal-audit.json').read_text());new=json.loads((p/'audit-l156.json').read_text())
            assert done['status']=='COMPLETE' and r['epochs']==10 and len(r['trace'])==10
            assert [x['epoch'] for x in r['trace']]==list(range(1,11)) and all(x['train_queries']==7453 for x in r['trace'])
            assert r['selected_epoch']==min(r['trace'],key=lambda x:x['val_mae'])['epoch']
            assert new['lane']==('released' if lane=='paper' else 'fit_horizon')
            assert t['status']==new['status']=='PASS' and len(new['mutants_rejected'])==6
            assert done['seed']==r['seed']==seed
            for k in ['queries','batches','dated_node_occurrences','edge_occurrences']:coverage[k]+=sum(s[k] for s in t['splits'].values())
            for k in ['equality_nodes','owner_table_checks']:coverage[k]+=sum(s[k] for s in new['splits'].values())
            if lane=='fit_horizon':
                for item in new['preprocessing'].values():
                    if item['status']=='PASS':assert item['latest_fit_event']<='2005-01-01 00:00:00'
            a=np.load(p/'predictions.npz');scores={}
            for split in ['val','test']:
                keys=list(zip(a[split+'_entity'].tolist(),a[split+'_time'].tolist()))
                assert len(set(keys))==len(keys) and set(keys)==set(truth[split])
                expected=np.array([truth[split][k] for k in keys]);np.testing.assert_allclose(expected,a[split+'_target'],rtol=0,atol=1e-12)
                pred=a[split+'_pred'];assert np.isfinite(pred).all();scores[split]=float(np.abs(pred-expected).mean())
                assert abs(scores[split]-r['scores'][split])<1e-12;all_predictions+=len(keys)
                maxerr=max(maxerr,r['replay'][split]['max_original_logit_error'])
            gradient[str(seed)]=sum(json.loads((p/'diagnostics.json').read_text())['first_backward_nonfinite'].values())
            records.append(dict(seed=seed,selected_epoch=r['selected_epoch'],**scores))
        metrics={s:dict(mean=statistics.mean(r[s] for r in records),sd=statistics.stdev(r[s] for r in records)) for s in ['val','test']}
        checks=[dict(id=k,status='PASS',evidence='hash-bound complete audit inputs') for k in ['query_identity','label_windows','sampling','fe_history','selection','prediction_keys']]
        checks.extend([dict(id='fit_scope',status='FAIL' if lane=='paper' else 'PASS',evidence='Declared2005-01-01dated-table fit horizon; released2010-01-01materialization' if lane=='paper' else 'Every dated processor fitted by2005-01-01'),dict(id='availability',status='NOT_ESTABLISHED',evidence='No ingestion,static attribute version or schedule publication histories')])
        required=['query_identity','label_windows','sampling','fe_history','fit_scope','availability','selection','prediction_keys']
        signoff=verdict(checks,required)
        assert signoff['status']==('FAIL' if lane=='paper' else 'NOT_ESTABLISHED')
        lanes[lane]=dict(records=records,metrics=metrics,coverage=coverage,checks=checks,strict_policy_verdict=signoff,first_backward_nonfinite=gradient,original_model_max_error=maxerr)
    delta={s:lanes['fit_horizon']['metrics'][s]['mean']-lanes['paper']['metrics'][s]['mean'] for s in ['val','test']}
    return dict(status='PASS',experiment='L156 full F1 temporal audit',label_counts=label_counts,dependencies=deps,sql_values=sql['total_values'],lanes=lanes,corrected_minus_released_mae=delta,predictions=all_predictions,reference_paper_band={s:'CLOSE' if abs(lanes['paper']['metrics'][s]['mean']-target)<=.2 else 'OUTSIDE_TOLERANCE' for s,target in [('val',3.193),('test',4.022)]},availability='NOT_ESTABLISHED',whole_paper='NOT_RUN',learner='PENDING_WRITTEN_DEFENSE')

def render_report(r):
    lines=['# Lesson156 · Temporal audit report','', 'Full archived F1 audit and five fresh complete fits per lane. Author evidence; learner PENDING_WRITTEN_DEFENSE.','', '| Lane | Validation MAE mean ± seed SD | Test MAE mean ± seed SD | Strict policy verdict |','|---|---:|---:|---|']
    for name,lane in r['lanes'].items():
        v=lane['metrics']['val'];t=lane['metrics']['test'];lines.append(f"| {name} | {v['mean']:.6f} ± {v['sd']:.6f} | {t['mean']:.6f} ± {t['sd']:.6f} | {lane['strict_policy_verdict']['status']} |")
    lines+=['',f"Labels: {sum(r['label_counts'].values()):,}; independently reconstructed SQL values: {r['sql_values']:,}; scored held-out predictions: {r['predictions']:,}.",f"Corrected minus released test MAE: {r['corrected_minus_released_mae']['test']:+.6f}; positive is worse. Descriptive, one database, no superiority inference.",'','The released lane matches the released preprocessing protocol, but fails the declared2005-01-01fit-horizon policy. The correction fits dated-table types and processors by that horizon; it is a course intervention, not Table7 parity. Static histories and schedule publication remain unobserved. No leak-free sign-off.','',f"Reference paper comparison: {r['reference_paper_band']}. Whole paper NOT_RUN; historical identity NOT_ESTABLISHED. First-backward nonfinite gradient counts remain in the JSON."]
    return '\n'.join(lines)+'\n'

if __name__=='__main__':
    E=Path(__file__).resolve().parent/'evidence/l156';r=replay(E,json.loads((E/'input-manifest.json').read_text()))
    (E/'report.json').write_text(json.dumps(r,indent=2)+'\n');(E/'report.md').write_text(render_report(r));print(render_report(r))

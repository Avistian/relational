"""Read-only replay of pinned author predictions. No network or training path."""
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
from relkit.portfolio_l154 import summarize_runs, compare_entries, portfolio_verdict


def aligned_score(truth_keys, targets, prediction_keys, predictions, metric, catalog=None):
    truth_keys=[tuple(map(int,k)) for k in truth_keys]
    prediction_keys=[tuple(map(int,k)) for k in prediction_keys]
    if (not truth_keys or len(set(truth_keys))!=len(truth_keys) or
        len(set(prediction_keys))!=len(prediction_keys) or set(truth_keys)!=set(prediction_keys) or
        len(targets)!=len(truth_keys) or len(predictions)!=len(prediction_keys)):
        raise ValueError('Missing, duplicate or mismatched query keys')
    index={k:i for i,k in enumerate(prediction_keys)}
    pred=np.asarray(predictions)[[index[k] for k in truth_keys]]
    if not np.isfinite(pred).all():
        raise ValueError('Nonfinite predictions')
    if metric=='MAP@10':
        if pred.ndim!=2 or pred.shape[1]<1 or not catalog:
            raise ValueError('Missing candidate protocol')
        if not np.issubdtype(pred.dtype,np.integer) or pred.min()<0 or pred.max()>=catalog:
            raise ValueError('Invalid candidate identity')
        if any(len(set(row))!=len(row) for row in pred.tolist()):
            raise ValueError('Duplicate recommendations')
        values=[]
        for positives,ranking in zip(targets,pred.tolist()):
            relevant=set(positives)
            if not relevant or min(relevant)<0 or max(relevant)>=catalog:
                raise ValueError('Invalid relevance set')
            hits=0;total=0.
            for rank,item in enumerate(ranking,1):
                if item in relevant:
                    hits+=1;total+=hits/rank
            values.append(total/min(len(relevant),len(ranking)))
        return float(np.mean(values))
    y=np.asarray(targets,dtype=float)
    if pred.ndim!=1 or y.ndim!=1 or not np.isfinite(y).all():
        raise ValueError('Invalid scalar labels or predictions')
    if metric=='MAE':
        return float(np.mean(np.abs(pred.astype(float)-y)))
    if metric=='AUROC':
        if set(y.tolist())!={0.,1.}:
            raise ValueError('AUC requires both binary classes')
        order=np.argsort(pred,kind='stable');ranks=np.empty(len(pred),dtype=float)
        start=0
        while start<len(order):
            stop=start+1
            while stop<len(order) and pred[order[stop]]==pred[order[start]]:
                stop+=1
            ranks[order[start:stop]]=(start+1+stop)/2
            start=stop
        npos=int(y.sum());nneg=len(y)-npos
        return float((ranks[y==1].sum()-npos*(npos+1)/2)/(npos*nneg))
    raise ValueError('Unsupported metric')


def verify_inputs(root, manifest):
    root=Path(root).resolve()
    if not manifest.get('files'):
        raise ValueError('Empty input manifest')
    for relative,digest in manifest['files'].items():
        path=(root/relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError('Missing or unsafe input: '+relative)
        if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            raise ValueError('Changed frozen input: '+relative)
    return len(manifest['files'])


def replay(root, manifest, summarize=summarize_runs, compare=compare_entries, verdict=portfolio_verdict):
    """Recompute every saved split in each selected evidence lane, without refitting."""
    root=Path(root);verified=verify_inputs(root,manifest)
    def read(name):
        return json.loads((root/name).read_text())
    def keys(z,split,entity):
        return list(zip(map(int,z[split+'_'+entity]),map(int,z[split+'_time'])))
    def population(ks):
        return hashlib.sha256(json.dumps(sorted(ks),separators=(',',':')).encode()).hexdigest()
    def close(a,b):
        if abs(a-b)>1e-12:
            raise ValueError('Saved score disagrees with replay')
    counts={};summaries={};runs={};selection_checks=0
    base='evidence/l151';truth=np.load(root/base/'prepared/queries.npz',allow_pickle=False)
    frozen=read(base+'/frozen.json')
    if frozen['test_access'] or frozen['lr']!=max(frozen['candidates'],key=lambda x:(x['selection_auc'],-x['lr']))['lr']:
        raise ValueError('Invalid validation selection')
    ph=manifest['files']['sources/l151/protocol.json']
    for lane,prefix,seeds in [('reference','ref-',list(range(5))),('selected','selected-',list(range(10,15)))]:
        bucket={s:[] for s in ['val','test']};counts['l151_'+lane]=0
        for seed in seeds:
            directory=base+'/'+prefix+str(seed);r=read(directory+'/result.json')
            if (r['status']!='COMPLETE' or r['track']!=lane or r['seed']!=seed or
                r['epochs']!=20 or len(r['history'])!=20 or r['lr']!=frozen['lr']):
                raise ValueError('L151 incomplete or mixed lane')
            if r['best_epoch']!=1+int(np.argmax([h['val']['roc_auc'] for h in r['history']])):
                raise ValueError('L151 checkpoint selection')
            if r['temporal_audit']['future_violations']!=0:
                raise ValueError('Observed temporal violation')
            selection_checks+=1
            z=np.load(root/directory/'predictions.npz',allow_pickle=False)
            for split in bucket:
                tk=keys(truth,split,'study');pk=keys(z,split,'study')
                target_by_key=dict(zip(tk,truth[split+'_target'].tolist()))
                if any(target_by_key.get(k)!=float(y) for k,y in zip(pk,z[split+'_target'])):
                    raise ValueError('Classification target mismatch')
                score=aligned_score(tk,truth[split+'_target'],pk[::-1],z[split+'_pred'][::-1],'AUROC')
                close(score,r['scores'][split]['roc_auc']);counts['l151_'+lane]+=len(tk)
                contract=dict(task='rel-trial/study-outcome',metric='AUROC',unit='fraction',direction='higher',
                              split=split,population=population(tk),protocol=ph,lane=lane,epochs=20,
                              n_queries=len(tk),origin='LOCAL',evaluation='full-split-keyed-scalar')
                bucket[split].append(dict(contract,seed=seed,status='COMPLETE',score=score))
        runs['classification_'+lane]=bucket
        summaries['classification_'+lane]={s:summarize(rows,seeds,{k:v for k,v in rows[0].items() if k not in ('seed','status','score')}) for s,rows in bucket.items()}
    base='evidence/l152';truth=np.load(root/base/'paper/seed-0/predictions.npz',allow_pickle=False)
    labels=read(base+'/label-audit.json')
    if labels['status']!='PASS' or labels['independently_rebuilt_labels']!=8712:
        raise ValueError('Missing upstream label audit')
    bucket={s:[] for s in ['val','test']};counts['l152_reference']=0
    protocol=read('sources/l152/protocol.json');ph=manifest['files']['sources/l152/protocol.json']
    for seed in range(5):
        directory=base+'/paper/seed-'+str(seed);r=read(directory+'/result.json');done=read(directory+'/completed.json')
        if (done['status']!='COMPLETE' or done['seed']!=seed or r['seed']!=seed or
            r['epochs']!=10 or len(r['trace'])!=10 or r['protocol']!={
                'channels':protocol['channels'],'num_layers':protocol['layers'],
                'batch_size':protocol['batch_size'],'lr':protocol['lr'],'epochs':protocol['epochs'],
                'fanout':protocol['fanouts'],'aggr':protocol['aggr'],'temporal_strategy':'uniform'}):
            raise ValueError('L152 incomplete or changed recipe')
        if r['selected_epoch']!=min(r['trace'],key=lambda row:row['val_mae'])['epoch']:
            raise ValueError('L152 selection not first validation minimum')
        if read(directory+'/temporal-audit.json')['status']!='PASS':
            raise ValueError('Missing temporal audit')
        selection_checks+=1;z=np.load(root/directory/'predictions.npz',allow_pickle=False)
        for split in bucket:
            tk=keys(truth,split,'entity');pk=keys(z,split,'entity')
            target_by_key=dict(zip(tk,truth[split+'_target'].tolist()))
            if any(target_by_key.get(k)!=float(y) for k,y in zip(pk,z[split+'_target'])):
                raise ValueError('Regression target mismatch')
            if (hashlib.sha256(np.asarray(tk,dtype='<i8').tobytes()).hexdigest()!=labels['query_key_hashes'][split] or
                len(tk)!=labels['counts'][split]):
                raise ValueError('Regression population mismatch')
            score=aligned_score(tk,truth[split+'_target'],pk[::-1],z[split+'_pred'][::-1],'MAE')
            close(score,r['scores'][split]);counts['l152_reference']+=len(tk)
            contract=dict(task='rel-f1/driver-position',metric='MAE',unit='position',direction='lower',
                          split=split,population=population(tk),protocol=ph,lane='reference',epochs=10,
                          n_queries=len(tk),origin='LOCAL',evaluation='full-split-keyed-scalar')
            bucket[split].append(dict(contract,seed=seed,status='COMPLETE',score=score))
    runs['regression_reference']=bucket
    summaries['regression_reference']={s:summarize(rows,list(range(5)),{k:v for k,v in rows[0].items() if k not in ('seed','status','score')}) for s,rows in bucket.items()}
    base='evidence/l153';r=read(base+'/pilot/result.json');z=np.load(root/base/'pilot/predictions.npz',allow_pickle=False)
    truth=json.loads(gzip.decompress((root/base/'prepared/val-truth.json.gz').read_bytes()))
    if r['status']!='PARTIAL_TIMING_PILOT' or set(r['scores'])!={'val'} or any(k.startswith('test') for k in z.files):
        raise ValueError('Recommendation packet must remain a validation pilot')
    if r['protocol_hash']!=manifest['files']['sources/l153/protocol.json'] or z['val_pred'].shape!=(37003,10):
        raise ValueError('Changed ranking protocol')
    score=aligned_score([(t['entity'],t['time']) for t in truth],[t['positives'] for t in truth],
                        keys(z,'val','entity')[::-1],z['val_pred'][::-1],'MAP@10',53241)
    close(score,r['scores']['val']);counts['l153_pilot']=len(truth)
    incomplete=dict(task='rel-trial/site-sponsor-run',metric='MAP@10',unit='fraction',direction='higher',
                    split='test',population='NOT_EVALUATED',protocol=r['protocol_hash'],lane='reference',
                    epochs=20,n_queries=27428,origin='LOCAL',evaluation='full-catalog:53241@10',status='INCOMPLETE',n_seeds=0,
                    mean=None,sample_sd=None)
    pilot=dict(task=incomplete['task'],split='val',status='PARTIAL_TIMING_PILOT',mean=score,
               n_queries=len(truth),catalog=53241,training_batches=32,seed=0)
    entries=[summaries['classification_reference']['test'],summaries['regression_reference']['test'],incomplete]
    context=read('evidence/l154/published-context.json');comparisons=[];paper_replay=[]
    for entry in entries:
        if entry['status']!='COMPLETE':
            continue
        row=context['tasks'][entry['task']]
        published=dict(entry,origin='PUBLISHED',population='HISTORICAL_NOT_ESTABLISHED',mean=row['baseline_mean'])
        c=compare(entry,published);c.update(baseline=row['baseline'],source=row['source']);comparisons.append(c)
        target=dict(published,mean=row['model_mean'])
        comparison=compare(entry,target)
        paper_replay.append(dict(task=entry['task'],gap=comparison['oriented_gap'],
                                 status='CLOSE' if abs(entry['mean']-row['model_mean'])<=row['tolerance'] else 'OUTSIDE_TOLERANCE',
                                 tolerance=row['tolerance'],meaning='Descriptive frozen band; not equivalence'))
    result=verdict(entries,comparisons,[e['task'] for e in entries])
    return dict(experiment='L154 RelBench portfolio evidence replay',status='PASS',input_files_verified=verified,
                input_manifest_sha256=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest(),
                prediction_rows_rescored=counts,total_prediction_rows=sum(counts.values()),
                validation_selection_checks=selection_checks,entries=entries,summaries=summaries,runs=runs,
                pilot=pilot,published_context=context,comparisons=comparisons,paper_replay=paper_replay,
                verdict=result,recommendation_cost_decision=read(base+'/cost-decision.json'),
                fresh_manual_fe='NOT_RUN',fresh_tree_baselines='NOT_RUN',full_paper='NOT_RUN',
                historical_identity='NOT_ESTABLISHED',feature_arrival_legality='NOT_ESTABLISHED',
                new_training='NOT_RUN',additional_cloud_spend_usd=0,
                target_label_boundary='L151 pinned task table; L152 targets from seed0 packet previously checked against archive SQL; L153 pinned relevance sets. No new source-SQL execution.',
                selection_boundary='Reference classification selected by design, not test score; course-selected lane remains separate. No retuning.',
                gradient_boundary='Source nonfinite-gradient observations in L151-L153 persist; rescoring cannot certify optimization health.',
                live_colab='NOT_CHECKED',deployment='NOT_CHECKED')


def render_report(report):
    """Same data feeds the lesson table and portable report."""
    lines=['# RelBench portfolio synthesis','',
           'Named experiment: **'+report['experiment']+'**. This is saved-prediction replay; no new fits.',
           '', '| Task / metric | Local test mean ± sample SD | Completed seeds | Published comparator (context only) | Fresh FE / tree |',
           '|---|---:|---:|---|---|']
    for entry in report['entries']:
        row=report['published_context']['tasks'][entry['task']];scale=100 if entry['unit']=='fraction' else 1
        suffix='%' if scale==100 else ' positions'
        local=(f"{entry['mean']*scale:.4f} ± {entry['sample_sd']*scale:.4f}{suffix}" if entry['status']=='COMPLETE' else 'NOT_RUN — INCOMPLETE')
        lines.append(f"| {entry['task']} / {entry['metric']} | {local} | {entry['n_seeds']}/5 | {row['baseline']}: {row['baseline_mean']*scale:.3f}{suffix} | NOT_RUN / NOT_RUN |")
    v=report['verdict'];lines+=['',f"**Coverage: {v['complete_tasks']}/{v['required_tasks']} tasks; {v['matched_baseline_tasks']} fresh matched comparator tasks.** Local superiority: **{v['local_superiority']}**.",
        '', 'The recommendation pilot evaluated validation only: '+f"{report['pilot']['mean']*100:.6f}% MAP@10 on {report['pilot']['n_queries']:,} queries. This is not a third completed test result.",
        '', '## Within-task descriptive context']
    for c in report['comparisons']:
        scale=100 if c['unit']=='fraction' else 1;unit='percentage points' if scale==100 else 'positions'
        lines.append(f"- {c['task']}: oriented benefit relative to the published baseline {c['oriented_gap']*scale:+.4f} {unit}. {c['scope']}; local win NOT_ESTABLISHED.")
    selected=report['summaries']['classification_selected']['test']
    lines+=['',f"Separate course-selected classification lane: {100*selected['mean']:.4f} ± {100*selected['sample_sd']:.4f}% AUROC, seeds10–14. Never pooled with reference seeds0–4.",
        '', '## Evidence and limits',
        f"Independently rescored {report['total_prediction_rows']:,} prediction rows; checked {report['validation_selection_checks']} validation-selected checkpoints and {report['input_files_verified']} frozen input files.",
        '', 'No arithmetic mean across AUROC, MAE and MAP; no win rate with missing matched baselines. Sample SD describes training-seed variability on the same fixed test population; it is not uncertainty across databases.',
        '', report['target_label_boundary'], '', report['selection_boundary'], '', report['gradient_boundary'],
        '', 'Full selected recommendation reproduction INCOMPLETE. Whole paper / fresh FE / fresh trees NOT_RUN. Historical identity and feature-arrival legality NOT_ESTABLISHED. Learner PENDING_WRITTEN_DEFENSE. Live Colab / deployment NOT_CHECKED.',
        '', 'Additional cloud spend: USD0. Replay does not count historical training costs again.',
        '', '## Sources',
        '[RelBench v1 Tables 6–8](https://arxiv.org/html/2407.20060v1#A2.SS1); upstream reproduction ledgers: [L151](../../l151-reproduction.md), [L152](../../l152-reproduction.md), [L153](../../l153-reproduction.md).',
        '', '## Learner defense',
        'PENDING_WRITTEN_DEFENSE — explain coverage, baseline identity, a within-task difference, seed uncertainty and the missing experiment before claiming completion.']
    return '\n'.join(lines)+'\n'


if __name__=='__main__':
    root=Path(__file__).resolve().parent
    manifest=json.loads((root/'evidence/l154/input-manifest.json').read_text())
    report=replay(root,manifest)
    (root/'evidence/l154/report.json').write_text(json.dumps(report,indent=2)+'\n')
    (root/'evidence/l154/report.md').write_text(render_report(report))
    print({k:report[k] for k in ['status','input_files_verified','total_prediction_rows','verdict','additional_cloud_spend_usd']})

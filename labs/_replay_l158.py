"""Full selected evidence replay: no fitting, downloads or cloud dispatch."""
import hashlib,json,statistics
from pathlib import Path
import numpy as np
from _replay_l154 import replay as replay_portfolio, aligned_score, verify_inputs
from _report_l155 import make_report
from relkit.synthesis_l158 import evidence_coverage,claim_verdict,validate_falsifier

def replay(root,manifest,coverage=evidence_coverage,verdict=claim_verdict,falsifier=validate_falsifier):
    root=Path(root);verify_inputs(root,manifest)
    read=lambda n:json.loads((root/n).read_text())
    p=replay_portfolio(root,read('evidence/l154/input-manifest.json'))
    f=make_report(root/'evidence/l155',read('evidence/l155/input-manifest.json'))
    # Reports must agree with the replay; they are never substituted for saved predictions.
    if p!=read('evidence/l154/report.json') or f!=read('evidence/l155/report.json'):
        raise ValueError('Inherited report differs from recomputation')
    temporal={};checks=0;count=0
    for lesson in [156,157]:
        base=f'evidence/l{lesson}/';prior=read(base+'report.json');old=read(base+'input-manifest.json')['files'];lanes={}
        for lane in ['paper','fit_horizon']:
            scores={s:[] for s in ['val','test']}
            for seed in range(5):
                directory=f'{base}{lane}/seed-{seed}/'
                for name in ['result.json','predictions.npz','completed.json']:
                    relative=f'{lane}/seed-{seed}/{name}'
                    if hashlib.sha256((root/directory/name).read_bytes()).hexdigest()!=old[relative]:raise ValueError('Upstream frozen bytes changed')
                result=read(directory+'result.json');done=read(directory+'completed.json')
                if done['status']!='COMPLETE' or done['seed']!=seed or result['seed']!=seed or result['epochs']!=10 or len(result['trace'])!=10:
                    raise ValueError('Incomplete temporal lane')
                if result['selected_epoch']!=min(result['trace'],key=lambda r:r['val_mae'])['epoch']:raise ValueError('Test-led checkpoint selection')
                checks+=1
                z=np.load(root/directory/'predictions.npz',allow_pickle=False)
                for split in scores:
                    truth=np.load(root/base/(split+'-labels.npz'),allow_pickle=False)
                    # Label packets use Unix seconds; prediction packets use nanoseconds.
                    tk=list(zip(map(int,truth['entity']),[int(t)*1_000_000_000 for t in truth['time']]))
                    pk=list(zip(map(int,z[split+'_entity']),map(int,z[split+'_time'])))
                    labels=dict(zip(tk,truth['target']))
                    if any(labels.get(k)!=y for k,y in zip(pk,z[split+'_target'])):raise ValueError('Changed target')
                    score=aligned_score(tk,truth['target'],pk[::-1],z[split+'_pred'][::-1],'MAE')
                    if abs(score-result['scores'][split])>1e-10:raise ValueError('Saved MAE differs')
                    scores[split].append(score);count+=len(tk)
            lanes[lane]={s:dict(mean=statistics.mean(v),sample_sd=statistics.stdev(v),seeds=list(range(5)),scores=v) for s,v in scores.items()}
            lanes[lane]['policy_verdict']=prior['lanes'][lane]['strict_policy_verdict']['status']
        for split in ['val','test']:
            delta=lanes['fit_horizon'][split]['mean']-lanes['paper'][split]['mean']
            if abs(delta-prior['corrected_minus_released_mae'][split])>1e-10:raise ValueError('Temporal difference mismatch')
        temporal[str(lesson)]=lanes
    lineage=[]
    for name,digest in manifest['files'].items():
        if not name.endswith('/predictions.npz'):continue
        if '/l153/' in name:continue # validation pilot is not a test packet
        database='trial' if '/l151/' in name else 'f1';task='study-outcome' if database=='trial' else 'driver-position'
        lineage.append(dict(id=name,task='rel-'+database+'/'+task,database='rel-'+database,prediction_hash=digest,status='COMPLETE',split='test'))
    # A report is another view of the same predictions, not another experiment.
    alias=next(r for r in lineage if '/l151/ref-0/' in r['id'])
    lineage.append(dict(alias,id='L154 report view of L151 reference seed0'))
    lineage.append(dict(id='L153 missing test',task='rel-trial/site-sponsor-run',database='rel-trial',prediction_hash=None,status='INCOMPLETE',split='test'))
    e=dict(matched_tasks=1,required_tasks=3,benefit=f['metrics']['test']['benefit_mae']['mean'],
           interval=f['test_benefit_driver_bootstrap_95'],human_effort=f['human_effort_ratio']['status'],historical_availability='NOT_ESTABLISHED')
    future=dict(task='new-held-out-database/predeclared-task',metric='MAE',direction='lower',minimum_benefit=.1,
                selection_split='val',decision_split='test',baseline='SQL features + LightGBM',information_policy='same owner cutoff and observed availability',
                disconfirm_if='upper bound of predeclared benefit interval is below 0.1 positions')
    return dict(experiment=manifest['experiment'],status='PASS',frozen_inputs=len(manifest['files']),
                input_manifest_sha256=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest(),
                portfolio=p,matched=f,temporal=temporal,lineage=lineage,coverage=coverage(lineage),
                claims={c:verdict(c,e) for c in ['local_quality','portfolio_superiority','human_effort','leak_free']},
                future_test=falsifier(future),prediction_rows_rescored=p['total_prediction_rows']+f['scored_prediction_rows']+count,
                selection_checks=p['validation_selection_checks']+10+checks,
                boundaries=dict(replay='Saved-prediction CPU replay; no fresh training',
                                temporal='Policy verdicts and gradient/SQL audits inherited, hash-pinned; no new raw SQL or sampling audit',
                                uncertainty='L155 driver-cluster interval conditional on fitted models; not database-level or race/time uncertainty',
                                independence='Same task/database across F1 repeats; packet identity does not establish statistical independence',
                                graph_construction_157b='NOT_RUN; lesson absent',human_study='NOT_RUN',whole_paper='NOT_RUN',
                                historical_identity='NOT_ESTABLISHED',public_contribution='PENDING_PUBLICATION',
                                learner='PENDING_WRITTEN_DEFENSE',live_colab='NOT_CHECKED',deployment='NOT_CHECKED'),
                additional_cloud_spend_usd=0)

def render_report(r):
    p=r['portfolio'];f=r['matched'];m=f['metrics']['test'];lo,hi=f['test_benefit_driver_bootstrap_95']
    lines=['# Year 4 thesis evidence replay','',r['experiment']+' — '+r['status'],
           '',f"{r['prediction_rows_rescored']:,} prediction rows rescored; {r['selection_checks']} selection checks; {r['frozen_inputs']} frozen inputs.",
           '', '| Evidence | Measured result | Permitted interpretation |','|---|---|---|',
           f"| L151 classification | {100*p['entries'][0]['mean']:.4f}% AUROC | Execution evidence; fresh FE missing |",
           f"| L152 regression | {p['entries'][1]['mean']:.6f} MAE | Execution evidence; use L155 for matched FE |",
           '| L153 recommendation | Test NOT_RUN | INCOMPLETE; validation pilot cannot fill test |',
           f"| L155 matched F1 | FE {m['fe_mae']['mean']:.6f}; RDL {m['rdl_mae']['mean']:.6f} MAE | Point estimate favors FE |",
           f"| L155 benefit FE − RDL | {m['benefit_mae']['mean']:+.6f}; conditional 95% [{lo:+.6f}, {hi:+.6f}] | Neither superiority nor equivalence established |",
           '| Human effort | NOT_OBSERVED | No local effort-saving ratio |']
    for lesson,lanes in r['temporal'].items():
        a=lanes['paper']['test']['mean'];b=lanes['fit_horizon']['test']['mean']
        lines.append(f'| L{lesson} released → fixed horizon | {a:.6f} → {b:.6f} MAE | Policy verdict FAIL → NOT_ESTABLISHED; no leak-free sign-off |')
    lines+=['','## Bounded interim verdict',
            'The selected pipelines are executable and their saved predictions are auditable. The one matched FE comparison does not establish an RDL advantage, and local human-effort savings remain unmeasured. The broad undervaluation thesis remains a research hypothesis.',
            '',f"Coverage: {r['coverage']['completed_tasks']} completed tasks on {r['coverage']['completed_databases']} databases; one matched FE task. Repeated F1 seeds and reports add no new database.",
            '', '## Limits']+[f'- {k}: {v}' for k,v in r['boundaries'].items()]
    lines+=['','Additional cloud spend: USD0. Full replay is not fresh training or whole-paper reproduction.']
    return '\n'.join(lines)+'\n'

if __name__=='__main__':
    P=Path(__file__).resolve().parent;E=P/'evidence/l158'
    r=replay(P,json.loads((E/'input-manifest.json').read_text()))
    (E/'report.json').write_text(json.dumps(r,indent=2)+'\n');(E/'report.md').write_text(render_report(r))
    print(render_report(r))

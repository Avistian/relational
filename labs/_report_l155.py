"""Pure portable report from frozen records; no training or hidden task functions."""
import hashlib,json,statistics
from pathlib import Path
import numpy as np
from relkit.effort_l155 import paired_losses,summarize_effort,effort_ratio

def make_report(root,manifest,pair=paired_losses,effort=summarize_effort,ratio=effort_ratio):
    root=Path(root)
    for name,digest in manifest['files'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
    rows=[];query_count=0;all_benefits=[];ids=None
    for seed in range(5):
        froot=root/f'fe/paper/seed-{seed}';groot=root/f'paper/seed-{seed}'
        f=np.load(froot/'predictions.npz');g=np.load(groot/'predictions.npz')
        fr=json.loads((froot/'result.json').read_text());gr=json.loads((groot/'result.json').read_text())
        assert fr['seed']==gr['seed']==seed and fr['trials']==10 and fr['rounds_cap']==2000
        assert len(fr['trace'])==10 and fr['selected_trial']==min(fr['trace'],key=lambda x:(x['val_mae'],x['number']))['number']
        assert gr['epochs']==10 and len(gr['trace'])==10 and all(x['train_queries']==7453 for x in gr['trace'])
        assert gr['selected_epoch']==min(gr['trace'],key=lambda x:x['val_mae'])['epoch']
        for split,n in [('val',499),('test',760)]:
            # Independent labels must agree by key before any prediction comparison.
            truth={(int(e),int(t)):float(y) for e,t,y in zip(f[split+'_entity'],f[split+'_time'],f[split+'_target'])}
            gtruth={(int(e),int(t)):float(y) for e,t,y in zip(g[split+'_entity'],g[split+'_time'],g[split+'_target'])}
            assert len(truth)==len(f[split+'_target'])==len(gtruth)==len(g[split+'_target'])==n and truth==gtruth
            q=[dict(entity=k[0],time=k[1],target=y) for k,y in truth.items()]
            preds=lambda a:[dict(entity=int(e),time=int(t),prediction=float(p)) for e,t,p in zip(a[split+'_entity'],a[split+'_time'],a[split+'_pred'])]
            paired=pair(q,list(reversed(preds(f))),preds(g))
            assert paired['n']==n
            assert abs(paired['fe_mae']-fr['scores'][split])<1e-10
            assert abs(paired['rdl_mae']-gr['scores'][split])<1e-10
            rows.append(dict(seed=seed,split=split,fe_mae=paired['fe_mae'],rdl_mae=paired['rdl_mae'],benefit_mae=paired['benefit_mae']))
            query_count+=n
            if split=='test':
                keys=[(r['entity'],r['time']) for r in paired['rows']]
                if ids is None:ids=keys
                assert ids==keys
                all_benefits.append([r['benefit'] for r in paired['rows']])
    metrics={s:{arm:dict(mean=statistics.mean(r[arm] for r in rows if r['split']==s),sample_sd=statistics.stdev(r[arm] for r in rows if r['split']==s)) for arm in ['fe_mae','rdl_mae','benefit_mae']} for s in ['val','test']}
    # Resample drivers, retaining all their cutoff queries. Conditional on fitted models.
    entity=np.array([k[0] for k in ids]);unique=np.unique(entity);d=np.mean(all_benefits,axis=0)
    sums=np.array([d[entity==i].sum() for i in unique]);counts=np.array([(entity==i).sum() for i in unique]);rng=np.random.default_rng(155)
    draw=rng.integers(0,len(unique),size=(2000,len(unique)))
    means=sums[draw].sum(axis=1)/counts[draw].sum(axis=1)
    interval=np.quantile(means,[.025,.975]).tolist()
    log=json.loads((root/'effort-log.json').read_text())
    observed=effort(log['sessions'],'rel-f1/driver-position','learner','declared_consistently',{'FE':'not_observed','RDL':'not_observed'})
    return dict(status='COMPLETE_SELECTED_COMPUTATIONAL_REPLAY',task='rel-f1/driver-position',metric='MAE',unit='finishing positions',rows=rows,metrics=metrics,
                scored_prediction_rows=query_count*2,unique_test_queries=len(ids),test_drivers=len(unique),
                test_benefit_driver_bootstrap_95=interval,bootstrap_draws=2000,
                bootstrap_scope='Conditional on fitted models/split; driver clusters only, not common race/time or training uncertainty',
                effort=observed,human_effort_ratio=ratio(observed),
                portfolio=dict(matched_tasks=1,classification_fe='NOT_RUN',recommendation='INCOMPLETE; test NOT_RUN'),
                paper_gnn_test_target=4.022,paper_gnn_tolerance=.2,
                paper_gnn_test_verdict='CLOSE' if abs(metrics['test']['rdl_mae']['mean']-4.022)<=.2 else 'OUTSIDE_TOLERANCE',
                figure3_reproduction='NOT_RUN: boosted regression comparator differs',human_study='NOT_RUN',historical_identity='NOT_ESTABLISHED',learner='PENDING_WRITTEN_DEFENSE')

def render_report(r):
    lines=['# L155 · Matched quality and observed effort','', '| Split | FE MAE mean ± seed SD | Basic RDL MAE mean ± seed SD | FE − RDL MAE |','|---|---:|---:|---:|']
    for split,m in r['metrics'].items():
        fmt=lambda arm:f"{m[arm]['mean']:.6f} ± {m[arm]['sample_sd']:.6f}"
        lines.append(f"| {split} | {fmt('fe_mae')} | {fmt('rdl_mae')} | {m['benefit_mae']['mean']:+.6f} |")
    lo,hi=r['test_benefit_driver_bootstrap_95']
    lines+=['',f"Positive benefit favors RDL. Driver-cluster conditional 95% interval: [{lo:+.6f}, {hi:+.6f}].",'',f"{r['scored_prediction_rows']:,} held-out predictions rescored. Five complete runs per method. Seed labels do not create paired random initializations.",'',f"Human effort: **{r['human_effort_ratio']['status']}**. Original human study **NOT_RUN**. One matched task; classification FE and recommendation test remain missing.",'', 'Basic GNN targets Table7; it is not Figure3 boosted regression. Historical identity and feature-arrival legality NOT_ESTABLISHED. Learner PENDING_WRITTEN_DEFENSE.']
    return '\n'.join(lines)+'\n'

if __name__=='__main__':
    E=Path(__file__).resolve().parent/'evidence/l155';m=json.loads((E/'input-manifest.json').read_text());r=make_report(E,m)
    (E/'report.json').write_text(json.dumps(r,indent=2)+'\n');(E/'report.md').write_text(render_report(r));print(render_report(r))

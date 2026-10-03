"""Independent scalar scorer and coverage audit; does not import the learner."""
import hashlib,json,math,statistics
from pathlib import Path

def audit_course(root):
    root=Path(root);cpath=root/'course-protocol.json';c=json.loads(cpath.read_text());runs=root/'runs'
    tasks=json.loads((runs/'tasks.json').read_text());taskmap={t['id']:t for t in tasks}
    expected={(t['id'],j) for t in tasks for j in range(c['support'],c['support']+c['query'])}
    assert len(taskmap)==len(tasks)==60 and len(expected)==480
    training_seeds={c['training_seed_base']+c['training_seed_stride']*seed+i for seed in c['seeds'] for i in range(c['steps']*c['batch_tasks'])}
    assert not training_seeds.intersection(t['seed'] for t in tasks)
    summary=[];initial={};final={};identities={};total=0
    actual={(p.stem.split('-')[1],int(p.stem.split('-')[2])) for p in runs.glob('run-*.json')}
    assert actual=={(a,s) for a in c['arms'] for s in c['seeds']},'Incomplete or extra runs'
    for seed in c['seeds']:
        for arm,p in c['arms'].items():
            r=json.loads((runs/f'run-{arm}-{seed}.json').read_text());assert (r['arm'],r['seed'])==(arm,seed)
            ident=r['identity'];assert ident['protocol_sha256']==hashlib.sha256(cpath.read_bytes()).hexdigest()
            assert ident['task_sha256']==hashlib.sha256((runs/'tasks.json').read_bytes()).hexdigest()
            identities[(arm,seed)]=ident;initial[(arm,seed)]=r['training']['initial_sha256'];final[(arm,seed)]=r['training']['final_sha256']
            assert hashlib.sha256((runs/f'weights-{arm}-{seed}.pt').read_bytes()).hexdigest()==r['weights_sha256']
            assert len(r['training']['loss'])==c['steps'] and all(math.isfinite(x) for x in r['training']['loss'])
            counts=r['training']['prior_counts'];n=c['steps']*c['batch_tasks'];assert counts=={'scm':int(n*p),'tree':n-int(n*p)}
            schedule=r['training']['schedule'];assert len(schedule)==n and {k:schedule.count(k) for k in counts}==counts
            keys=[(x['task_id'],x['query_id']) for x in r['records']]
            assert len(keys)==len(set(keys)) and set(keys)==expected,'Prediction identity coverage'
            byfamily={f:[] for f in c['eval_families']}
            for x in r['records']:
                t=taskmap[x['task_id']];assert x['family']==t['family'] and x['y']==t['y'][x['query_id']]
                prob=x['p'];assert len(prob)==2 and all(math.isfinite(v) and 0<v<1 for v in prob) and abs(sum(prob)-1)<1e-12
                byfamily[x['family']].append((int((prob[1]>prob[0])==x['y']),-math.log(prob[x['y']])))
            for family,values in byfamily.items():
                assert len(values)==160
                summary.append(dict(arm=arm,seed=seed,family=family,accuracy=statistics.mean(x[0] for x in values),cross_entropy=statistics.mean(x[1] for x in values)))
            total+=len(keys)
    assert len({json.dumps(x,sort_keys=True) for x in identities.values()})==1
    for seed in c['seeds']:
        assert len({initial[(a,seed)] for a in c['arms']})==1,'Unpaired initialization'
        assert len({final[(a,seed)] for a in c['arms']})==3,'Arms did not produce separate fits'
    aggregate=[];contrasts=[]
    for f in c['eval_families']:
        for arm in c['arms']:
            v=[x for x in summary if x['family']==f and x['arm']==arm]
            aggregate.append(dict(family=f,arm=arm,accuracy_mean=statistics.mean(x['accuracy'] for x in v),accuracy_sd=statistics.stdev(x['accuracy'] for x in v),ce_mean=statistics.mean(x['cross_entropy'] for x in v),ce_sd=statistics.stdev(x['cross_entropy'] for x in v)))
        for baseline in ['scm','tree']:
            deltas=[]
            for seed in c['seeds']:
                values={x['arm']:x['cross_entropy'] for x in summary if x['family']==f and x['seed']==seed}
                deltas.append(values['mixed']-values[baseline])
            contrasts.append(dict(family=f,comparison='mixed-minus-'+baseline,ce_deltas=deltas,mean=statistics.mean(deltas),sd=statistics.stdev(deltas)))
    return dict(status='COMPLETE_COURSE_EXPERIMENT',fits=9,updates=9*c['steps'],training_tasks=9*c['steps']*c['batch_tasks'],evaluation_tasks=60,predictions=total,paired_initialization='PASS',disjoint_task_seeds='PASS',rows=summary,aggregate=aggregate,paired_effects=contrasts,scope='Synthetic generators and small learner only; no real-data or Mitra paper parity',learner='PENDING_WRITTEN_DEFENSE')

if __name__=='__main__':
    p=Path(__file__).parent/'evidence/b06';r=audit_course(p);(p/'course-audit.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))

"""Collect immutable complete runs and independently score aligned query errors."""
import json,hashlib,sys
from pathlib import Path
import numpy as np,modal
from sklearn.metrics import mean_absolute_error
from relkit.pathology_l142 import paired_loss_gap
P=Path(__file__).resolve().parent;E=P/'evidence/l142';v=modal.Volume.from_name('l142-pathology-evidence')
def fetch(name):
    raw=b''.join(v.read_file(name));dest=E/name;dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists():assert dest.read_bytes()==raw,('Changed artifact',name)
    else:dest.write_bytes(raw)
    return dest

def collect(phase):
    if phase=='prepare':
        for name in ['prepared/prepared.json','prepare-started.json','prepare-cost.json','operator-parity.json']:fetch(name)
        return
    phases=[f'{a}-{s}' for a in ['composite','ordinary'] for s in range(5)] if phase=='full' else [phase]
    records={};hashes={};count=0;uuids=[]
    for name in phases:
        for file in ['result.json','predictions.npz','sampled_trace.npz']:
            p=fetch(name+'/'+file);hashes[name+'/'+file]=hashlib.sha256(p.read_bytes()).hexdigest()
        for suffix in ['-started.json','-cost.json']:fetch(name+suffix)
        uuids.append(json.loads((E/(name+'-started.json')).read_text())['uuid'])
        r=json.loads((E/name/'result.json').read_text());z=np.load(E/name/'predictions.npz');assert r['status']=='COMPLETE'
        assert r['count']=={'train':7453,'val':499,'test':760}
        for split in ['val','test']:
            assert len(set(zip(z[split+'_entity'],z[split+'_time'])))==r['count'][split]
            score=mean_absolute_error(z[split+'_target'],z[split+'_pred'])
            assert abs(score-r['scores'][split]['mae'])<1e-10;count+=r['count'][split]
        assert r['epochs']==10 and len(r['history'])==10
        assert r['best_epoch']==1+int(np.argmin([h['val_mae'] for h in r['history']]))
        assert all(h['queries']==7453 and h['steps']==15 for h in r['history'])
        assert r['temporal_audit']['future_violations']==0
        trace=np.load(E/name/'sampled_trace.npz')
        for k in trace.files:
            if k.endswith('_times'):assert (trace[k]<=trace['cutoffs'][trace[k[:-6]+'_owners']]).all()
        dest=P/'results/l142'/f'{name}.pt';dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists():dest.write_bytes(b''.join(v.read_file(name+'/selected.pt')))
        assert hashlib.sha256(dest.read_bytes()).hexdigest()==r['checkpoint_sha256']
        records[name]=r
    assert len(set(uuids))==len(uuids)
    if phase=='full':
        stats={arm:{split:dict(mean=float(np.mean([records[f'{arm}-{s}']['scores'][split]['mae'] for s in range(5)])),sample_sd=float(np.std([records[f'{arm}-{s}']['scores'][split]['mae'] for s in range(5)],ddof=1))) for split in ['val','test']} for arm in ['composite','ordinary']}
        paired={}
        for split in ['val','test']:
            gaps=[]
            for seed in range(5):
                a=np.load(E/f'composite-{seed}/predictions.npz');b=np.load(E/f'ordinary-{seed}/predictions.npz')
                key=lambda z:list(zip(z[split+'_entity'],z[split+'_time']))
                assert key(a)==key(b);np.testing.assert_array_equal(a[split+'_target'],b[split+'_target'])
                order=np.arange(len(key(b)))[::-1]
                values=paired_loss_gap(key(a),a[split+'_target'],key(a),a[split+'_pred'],[key(b)[i] for i in order],b[split+'_pred'][order])
                gaps.append(float(values.mean()))
            paired[split]=dict(seed_gaps=gaps,mean=float(np.mean(gaps)),sample_sd=float(np.std(gaps,ddof=1)),interpretation='positive favors composite; seed variation, not independent-query uncertainty')
        summary=dict(status='COMPLETE',arms=stats,paired=paired,verified_predictions=count,unique_runs=len(uuids),paper_target=3.798,tolerance=.2,closeness='CLOSE' if abs(stats['composite']['test']['mean']-3.798)<=.2 else 'OUTSIDE_TOLERANCE',parameter_counts={a:records[f'{a}-0']['parameter_count'] for a in stats},source_artifact_hashes=hashes,temporal_query_occurrences=sum(r['temporal_audit']['query_occurrences'] for r in records.values()),first_batch_nonfinite_gradients={a:records[f'{a}-0']['first_batch_nonfinite_gradients'] for a in stats},historical_training='NOT_ESTABLISHED',whole_paper='NOT_RUN',learner='PENDING_WRITTEN_DEFENSE')
        (E/'training.json').write_text(json.dumps(summary,indent=2));print(summary)
    else:print({n:r['scores'] for n,r in records.items()})
if __name__=='__main__':collect(sys.argv[1])

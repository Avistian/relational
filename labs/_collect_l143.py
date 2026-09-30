"""Immutable artifact collection and independent complete-population scoring."""
import argparse,json,hashlib
from pathlib import Path
import modal,numpy as np
from sklearn.metrics import mean_absolute_error
from relkit.relgnn_l143 import keyed_mae
P=Path(__file__).resolve().parent;E=P/'evidence/l143';v=modal.Volume.from_name('l143-relgnn-evidence')
def fetch(name):
    raw=b''.join(v.read_file(name));dest=E/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw);return dest

def collect(phase):
    if phase=='prepare':
        for name in ['prepared/prepared.json','prepare-started.json','prepare-cost.json','operator-parity.json']:fetch(name)
        return
    phases=['replay-compatible']+[f'seed-{s}' for s in range(5)] if phase=='full' else [phase]
    results=[];hashes={};count=0
    for name in phases:
        for file in ['result.json','predictions.npz','sampled_trace.npz']:
            p=fetch(name+'/'+file);hashes[name+'/'+file]=hashlib.sha256(p.read_bytes()).hexdigest()
        for file in ['-started.json','-cost.json']:fetch(name+file)
        r=json.loads((E/name/'result.json').read_text());z=np.load(E/name/'predictions.npz');assert r['status']=='COMPLETE'
        assert r['count']=={'train':7453,'val':499,'test':760}
        for split in ['val','test']:
            keys=list(zip(z[split+'_entity'],z[split+'_time']));order=np.arange(len(keys))[::-1]
            score=keyed_mae(keys,z[split+'_target'],[keys[i] for i in order],z[split+'_pred'][order])
            assert abs(score-r['scores'][split]['mae'])<1e-10
            assert abs(score-mean_absolute_error(z[split+'_target'],z[split+'_pred']))<1e-10;count+=len(keys)
        trace=np.load(E/name/'sampled_trace.npz')
        for k in trace.files:
            if k.endswith('_times'):assert (trace[k]<=trace['cutoffs'][trace[k[:-6]+'_owners']]).all()
        assert r['temporal_audit']['future_violations']==0
        if name!='replay-compatible':
            assert r['epochs']==10 and len(r['history'])==10
            assert r['best_epoch']==1+int(np.argmin([x['val_mae'] for x in r['history']]))
            assert all(x['queries']==7453 and x['steps']==15 for x in r['history'])
            path=P/f'results/l143/{name}.pt';path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('wb') as f:
                for block in v.read_file(name+'/selected.pt'):f.write(block)
            assert hashlib.sha256(path.read_bytes()).hexdigest()==r['checkpoint_sha256']
        results.append(r)
    if phase=='full':
        fetch('checkpoint-incompatibility.txt');fetch('compatible/prepared.json')
        fitted=[r for r in results if r['kind']=='RECONSTRUCTED_TRAINING'];assert sorted(r['seed'] for r in fitted)==list(range(5))
        stats={split:dict(mean=float(np.mean([r['scores'][split]['mae'] for r in fitted])),sample_sd=float(np.std([r['scores'][split]['mae'] for r in fitted],ddof=1))) for split in ['val','test']}
        stats['test']['descriptive_closeness']='CLOSE' if abs(stats['test']['mean']-3.798)<=.20+1e-12 else 'OUTSIDE_TOLERANCE'
        summary=dict(status='COMPLETE',replay=results[0]['scores'],reconstruction=stats,verified_predictions=count,paper_test_mae=3.798,tolerance=.20,source_artifact_hashes=hashes,historical_training='NOT_ESTABLISHED',whole_paper='NOT_RUN',learner='PENDING_WRITTEN_DEFENSE')
        from relkit.reproduction_l143 import evidence_verdict
        summary['verdict']=evidence_verdict([dict(seed=r['seed'],kind=r['kind'],epochs=r['epochs'],complete=r['status']=='COMPLETE',test_mae=r['scores']['test']['mae']) for r in fitted])
        (E/'training.json').write_text(json.dumps(summary,indent=2));print(summary)
    else:print([(r['kind'],r['scores'],r['seconds']) for r in results])
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase');collect(p.parse_args().phase)

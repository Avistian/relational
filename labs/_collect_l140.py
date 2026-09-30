"""Collect each immutable artifact. Never average an incomplete seed set."""
import argparse,json,hashlib
from pathlib import Path
import modal,numpy as np
from relkit.amazon_l138 import rank_auc
from relkit.checkpoint_l140 import select_checkpoint,align_predictions,reproduction_verdict
P=Path(__file__).resolve().parent;E=P/'evidence/l140';v=modal.Volume.from_name('l140-checkpoint-evidence')
def fetch(name):
    raw=b''.join(v.read_file(name));dest=E/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw);return dest

def collect(phase):
    if phase=='preflight':
        for name in ['preflight.json','preflight-cost.json']:fetch(name)
        return
    seeds=[0] if phase=='pilots' else list(range(5));summaries={};all_hashes={};uuids=[]
    for task,n,entity,subdir,epochs,targets in [('amazon',138,'customer','final',10,[.7045,.7042]),('trial',139,'study','full',20,[.6818,.686])]:
        base=np.load(P/(f'evidence/l{n}/recency_predictions.npz' if task=='amazon' else f'evidence/l{n}/queries.npz'))
        results=[];checkpoint_info={};count=0
        for seed in seeds:
            folder=f'{task}/seed-{seed}'
            for name in ['result.json','predictions.npz','data_identity.json','progress.json','sampled_trace.npz','source_parity.json','run_identity.json']:
                p=fetch(folder+'/'+name);all_hashes[str(p.relative_to(E))]=hashlib.sha256(p.read_bytes()).hexdigest()
            fetch(f'{task}-seed-{seed}-cost.json')
            r=json.loads((E/folder/'result.json').read_text());identity=json.loads((E/folder/'run_identity.json').read_text());uuids.append(identity['run_uuid'])
            assert identity['source_sha256']==hashlib.sha256((P/'_full_l140.py').read_bytes()).hexdigest()
            assert r['seed']==seed and r['epochs']==epochs and r['status']=='COMPLETE' and len(r['history'])==epochs
            assert r['best_epoch']==select_checkpoint(r['history']) and r['selection_auc']==max(x['val']['roc_auc'] for x in r['history'])
            assert r['original_model_parity']['status']=='NUMERIC_CLOSE'
            assert r['temporal_audit']['future_violations']==0
            assert all(x['steps']==(2001 if task=='amazon' else 24) and x['queries']==(2001*512 if task=='amazon' else 11994) for x in r['history'])
            d=json.loads((E/folder/'data_identity.json').read_text());assert all(x['match'] and x['sha256']==x['historical'] for x in d.values())
            z=np.load(E/folder/'predictions.npz')
            for split in ['val','test']:
                assert np.array_equal(z[split+'_target'],base[split+'_target'])
                q=list(zip(base[split+'_'+entity],base[split+'_time']));pk=list(zip(z[split+'_entity'],z[split+'_time']))
                pred=align_predictions(q,pk,z[split+'_pred']);auc=rank_auc(base[split+'_target'],pred)
                assert abs(auc-r['scores'][split]['roc_auc'])<1e-12
                count+=len(pred)
            path=P/f'results/l140/checkpoints/{task}-seed-{seed}.pt';path.parent.mkdir(parents=True,exist_ok=True);h=hashlib.sha256();size=0
            with path.open('wb') as f:
                for block in v.read_file(folder+'/selected.pt'):f.write(block);h.update(block);size+=len(block)
            checkpoint_info[str(seed)]=dict(sha256=h.hexdigest(),bytes=size,volume='l140-checkpoint-evidence',path=folder+'/selected.pt',published=False)
            trace=np.load(E/folder/'sampled_trace.npz')
            for key in trace.files:
                if key.endswith('_times'):assert (trace[key]<=trace['cutoffs'][trace[key[:-6]+'_owners']]).all()
            results.append(r)
        metrics={}
        for split,target in zip(['val','test'],targets):
            values={r['seed']:r['scores'][split]['roc_auc'] for r in results}
            metrics[split]=dict(reproduction_verdict(values,list(range(5)),target,.01,task!='amazon'),values=values,paper_target=target,tolerance=.01)
        summaries[task]=dict(status='COMPLETE' if len(results)==5 else 'INCOMPLETE',metrics=metrics,verified_predictions=count,checkpoints=checkpoint_info,audited_query_occurrences=sum(r['temporal_audit']['query_occurrences'] for r in results))
    assert len(uuids)==len(set(uuids))
    if phase=='full':
        summary=dict(status='COMPLETE',tasks=summaries,source_artifact_hashes=all_hashes,run_uuids=uuids,historical_identity='NOT_ESTABLISHED',whole_paper='NOT_ESTABLISHED',learner='PENDING_WRITTEN_DEFENSE')
        (E/'training.json').write_text(json.dumps(summary,indent=2));print({k:v['metrics'] for k,v in summaries.items()})
    else:print('Collected both complete seed0 pilots')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['preflight','pilots','full']);collect(p.parse_args().phase)

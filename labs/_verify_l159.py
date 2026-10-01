"""Independent metric/forward oracle plus scientific and learner-mutation checks."""
import hashlib,json,math
from pathlib import Path
import numpy as np
import torch
from _check_l159 import check
from relkit.foundation_preview_l159 import (mask_serializations,masked_cross_entropy,freeze_codec,make_examples,batch,RowCodec,TableGraph)
P=Path(__file__).resolve().parent

def verify(report):
    assert report['status']=='COMPLETE' and report['paper_reproduction']=='NOT_RUN'
    assert report['config']['seeds']==[0,1,2] and [run['seed'] for run in report['runs']]==[0,1,2]
    splits=report['splits'];assert [len(splits[s]) for s in ['train','val','test']]==[84,24,12]
    assert len(set(sum(splits.values(),[])))==120
    examples=make_examples();x,y,m,rows=batch(examples,'test');tokens=x.numpy()
    expected={(r['key'],arm) for r in rows for arm in ['row','graph']}
    checked=0;largest=0.;scores=[]
    for run in report['runs']:
        assert run['seed'] in [0,1,2]
        assert run['frozen_before']==run['frozen_after']
        for stage in ['row','graph']:
            history=run['histories'][stage]
            assert len(history)==80 and [h['epoch'] for h in history]==list(range(80))
            assert all(math.isfinite(h['train_loss']) for h in history)
            assert run['selected_epoch'][stage]==max(range(80),key=lambda i:history[i]['val_accuracy'])
        assert {(p['key'],p['arm']) for p in run['predictions']}==expected
        assert len(run['predictions'])==len(expected)==72
        c={k:np.array(v,dtype=np.float64) for k,v in run['codec'].items()}
        g={k:np.array(v,dtype=np.float64) for k,v in run['graph'].items()}
        assert hashlib.sha256(json.dumps(run['codec'],sort_keys=True).encode()).hexdigest()==run['frozen_after']
        # Independent numpy mean embedding, linear layers and tanh; no torch model used.
        h=np.tanh(c['embedding.weight'][tokens].mean(axis=2) @ c['encoder.weight'].T + c['encoder.bias'])
        graph_h=np.tanh(np.broadcast_to(h.mean(axis=1,keepdims=True),h.shape) @ g['weight']+g['bias'])
        for arm,z in [('row',h),('graph',graph_h)]:
            logits=(z[:,0]@c['decoder.weight'].T+c['decoder.bias']).reshape(-1,3,2)
            lookup={r['key']:(i,r) for i,r in enumerate(rows)}
            for p in [p for p in run['predictions'] if p['arm']==arm]:
                i,r=lookup[p['key']];actual=logits[i,p['task']]
                assert p['truth']==int(r['labels'][p['task']]) and p['table']==r['table']
                error=float(np.max(np.abs(actual-p['logits'])));largest=max(largest,error)
                assert error<2e-5 and int(actual.argmax())==p['prediction'];checked+=1
            for task in range(3):
                ps=[p for p in run['predictions'] if p['arm']==arm and p['task']==task]
                scores.append(dict(seed=run['seed'],arm=arm,task=task,correct=sum(p['prediction']==p['truth'] for p in ps),n=len(ps)))
        val_x,val_y,val_m,_=batch(examples,'val')
        vh=np.tanh(c['embedding.weight'][val_x.numpy()].mean(axis=2) @ c['encoder.weight'].T+c['encoder.bias'])
        vg=np.tanh(np.broadcast_to(vh.mean(axis=1,keepdims=True),vh.shape) @ g['weight']+g['bias'])
        for stage,z in [('row',vh),('graph',vg)]:
            vl=(z[:,0]@c['decoder.weight'].T+c['decoder.bias']).reshape(-1,3,2)
            observed=float((vl.argmax(-1)[val_m.numpy()]==val_y.numpy()[val_m.numpy()]).mean())
            saved=run['histories'][stage][run['selected_epoch'][stage]]['val_accuracy']
            assert abs(observed-saved)<1e-6, 'Saved checkpoint does not match selected validation score'
    # Counterfactual target changes must disappear before encoding; no global value masking.
    clean=torch.tensor([[2,4,6],[2,4,7],[2,4,6],[2,4,7]])
    ids=torch.tensor([[10,20,30],[10,20,31],[10,20,32],[10,20,33]])
    for target in [10,20,30]:
        changed=clean.clone();changed[ids==target]+=1
        assert torch.equal(mask_serializations(clean,ids,target),mask_serializations(changed,ids,target))
    check(mask_serializations,masked_cross_entropy,freeze_codec)
    # Each deliberately broken learner implementation must fail the common contracts.
    def first_copy(tokens,identities,target):
        out=tokens.clone();index=(identities==target).nonzero()[0];out[tuple(index)]=0;return out
    def all_losses(logits,labels,selected):
        return torch.nn.functional.cross_entropy(logits.reshape(-1,2),labels.flatten())
    def keep_trainable(codec): return codec
    rejected=[]
    for name,functions in [('one-copy-only',(first_copy,masked_cross_entropy,freeze_codec)),('unmasked-loss',(mask_serializations,all_losses,freeze_codec)),('unfrozen-codec',(mask_serializations,masked_cross_entropy,keep_trainable))]:
        try: check(*functions)
        except (AssertionError,ValueError,RuntimeError): rejected.append(name)
        else: raise AssertionError('Surviving learner mutant: '+name)
    manifest=json.loads((P/'sources/l159/manifest.json').read_text())
    for source in manifest['sources']:
        if 'file' in source:
            assert hashlib.sha256((P/'sources/l159'/source['file']).read_bytes()).hexdigest()==source['sha256']
    return dict(status='PASS',independent_prediction_checks=checked,numpy_logit_max_error=largest,
                scores=scores,semantic_target_counterfactuals=3,learner_mutants_rejected=rejected,
                source_hashes='PASS',paper_reproduction='NOT_RUN',historical_fidelity='NOT_ESTABLISHED',
                learner='PENDING_WRITTEN_DEFENSE',cloud_spend_usd=0)

if __name__=='__main__':
    result=verify(json.loads((P/'evidence/l159/mechanism.json').read_text()))
    (P/'_verify_l159_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print({k:v for k,v in result.items() if k!='scores'})

"""One fresh process per course setting. Query targets never enter predict()."""
import time
START=time.perf_counter()
import hashlib,json,resource,sys
from pathlib import Path
import numpy as np
from relkit.scaling_b04a import linear_readout,fit_scale,class_values
P=Path(__file__).resolve().parent;E=P/'evidence/b04a'

def predict(support,labels,query,classes,projection,kernel):
    mean,scale=fit_scale(support)
    k=((support-mean)/scale)@projection;q=((query-mean)/scale)@projection
    values=class_values(labels,classes,10)
    if kernel=='linear':
        prob=linear_readout(q,k,values)
        prob=prob/prob.sum(axis=1,keepdims=True)
    elif kernel=='softmax':
        scores=q@k.T/np.sqrt(k.shape[1]);scores-=scores.max(axis=1,keepdims=True)
        weights=np.exp(scores);weights/=weights.sum(axis=1,keepdims=True);prob=weights@values
    else:raise ValueError('Unknown kernel')
    return prob,mean,scale

if __name__=='__main__':
    protocol=json.loads((E/'protocol.json').read_text());name=sys.argv[1];config=next(x for x in protocol['configs'] if x['name']==name)
    out=E/'runs'/(name+'.json');out.parent.mkdir(exist_ok=True)
    if out.exists():raise SystemExit('Immutable result exists')
    raw_bytes=(E/'inputs.json').read_bytes()
    if hashlib.sha256(raw_bytes).hexdigest()!=protocol['inputs_sha256']:raise ValueError('Input identity mismatch')
    raw=json.loads(raw_bytes)['tables'][str(config['seed'])];n=config['support'];f=config['features']
    start=time.perf_counter();latent=np.array(raw['latent']);extra=np.array(raw['noise'])
    table=np.concatenate([latent,extra],axis=1)[:,:f] if config['widening']=='noise' else latent[:,np.arange(f)%8]
    support_ids=raw['support_pool'][:n];query_ids=raw['query_ids'];support=table[support_ids];query=table[query_ids];labels=np.array(raw['y'])[support_ids]
    projection=np.random.default_rng(4000+f).normal(size=(f,16))/np.sqrt(f)
    materialize=time.perf_counter()-start;begin=time.perf_counter()
    prob,mean,scale=predict(support,labels,query,protocol['classes'],projection,config['kernel'])
    cold=time.perf_counter()-begin;warm=[];delta=0
    for _ in range(protocol['warm_repeats']):
        begin=time.perf_counter();repeat,_,_=predict(support,labels,query,protocol['classes'],projection,config['kernel']);warm.append(time.perf_counter()-begin);delta=max(delta,float(np.max(np.abs(repeat-prob))))
    result={'config':config,'inputs_sha256':protocol['inputs_sha256'],'support_ids':support_ids,'query_ids':query_ids,'classes':protocol['classes'],'mean':mean.tolist(),'scale':scale.tolist(),'probabilities':prob.tolist(),'cold_pipeline_seconds':cold,'warm_pipeline_seconds':warm,'materialization_seconds':materialize,'process_seconds':time.perf_counter()-START,'peak_process_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'warm_max_delta':delta,'status':'COMPLETE'}
    out.write_text(json.dumps(result,separators=(',',':'))+'\n')

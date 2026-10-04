"""Independent full forward replay using NumPy and scalar math, no model imports.

Reconstructs the approved inputs and initialization-independent operations, then
compares all encoder coordinates and output probabilities with saved arrays.
"""
import json,math,hashlib
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent

def verify(r):
    def softmax(x):
        a=np.exp(x-np.max(x,axis=-1,keepdims=True));return a/a.sum(-1,keepdims=True)
    def linear(x,s,key):return x@np.array(s[key+'.weight']).T+np.array(s[key+'.bias'])
    def norm(x,s,key):
        return (x-x.mean(-1,keepdims=True))/np.sqrt(x.var(-1,keepdims=True)+1e-5)*s[key+'.weight']+s[key+'.bias']
    def att(q,kv,s,key):
        query=linear(q,s,key+'.q');keys=linear(kv,s,key+'.k');values=linear(kv,s,key+'.v')
        return linear(softmax(query@keys.swapaxes(-2,-1)/4)@values,s,key+'.out')
    def ff(x,s,key):
        h=linear(x,s,key+'.0');h=h*.5*(1+np.vectorize(math.erf)(h/math.sqrt(2)))
        return linear(h,s,key+'.2')
    def encode(x,s):
        t=linear(x[...,None],s,'numeric')+s['column']
        t=np.concatenate([t,np.broadcast_to(s['row'],(6,1,16))],axis=1);layers=[]
        for j in range(2):
            b=f'blocks.{j}';a=norm(t,s,b+'.norms.0');t=t+att(a,a[:,:-1],s,b+'.columns')
            a=norm(t,s,b+'.norms.1').swapaxes(0,1);t=t+att(a,a[:,:4],s,b+'.rows').swapaxes(0,1)
            t=t+ff(norm(t,s,b+'.norms.2'),s,b+'.ff');layers.append(linear(t[:,-1],s,f'projections.{j}'))
        return linear(layers[0]+layers[1],s,'final')
    def decode(z,y,s):
        t=np.array(s['label.weight'])[list(y)+[2,2]]
        for j in range(2):
            b=f'blocks.{j}';t=t+att(norm(t,s,b+'.norms.0')[:,None],z[:,None],s,b+'.features')[:,0]
            a=norm(t,s,b+'.norms.1');t=t+att(a,a[:4],s,b+'.targets');t=t+ff(norm(t,s,b+'.norms.2'),s,b+'.ff')
        return softmax(linear(t[4:],s,'head'))[:,1]
    expected=np.array([[-1,0],[0,1],[1,0],[0,-1],[.5,.5],[-.5,-.5]],dtype=float)
    ids=['s0','s1','s2','s3','q0','q1'];tasks={'A':[0,0,1,1],'B':[0,1,0,1]}
    assert r['inputs']==expected.tolist() and r['ids']==ids and r['tasks']==tasks and r['support_size']==4
    assert [s['seed'] for s in r['seeds']]==[0,1,2]
    modes=['baseline','support-labels','support-feature','other-query','support-order','query-order'];maxerr=0;count=0
    for seed in r['seeds']:
        assert len(seed['records'])==12
        assert {(x['task'],x['mode']) for x in seed['records']}=={(t,m) for t in tasks for m in modes}
        for item in seed['records']:
            x=expected.copy();labels=np.array(tasks[item['task']]);order=list(range(6));mode=item['mode']
            if mode=='support-labels':labels=1-labels
            if mode=='support-feature':x[0,0]+=2
            if mode=='other-query':x[5]=[7,-3]
            if mode=='support-order':order=[2,0,3,1,4,5];labels=labels[order[:4]]
            if mode=='query-order':order=[0,1,2,3,5,4]
            x=x[order]
            assert item['inputs']==x.tolist() and item['ids']==[ids[i] for i in order] and item['support_labels']==labels.tolist()
            z=encode(x,seed['state']['encoder']);p=decode(z,labels,seed['state']['decoder'])
            err=max(np.max(np.abs(z-item['embedding'])),np.max(np.abs(p-item['probability'])))
            assert np.isfinite(err) and err<1e-10;maxerr=max(maxerr,float(err));count+=1
            base=next(v for v in seed['records'] if v['task']==item['task'] and v['mode']=='baseline')
            dz=np.max(np.abs(z[np.argsort(order)]-base['embedding']))
            aligned=p[np.argsort([i-4 for i in order[4:]])];dp=np.max(np.abs(aligned-base['probability']))
            assert abs(dz-item['embedding_delta'])<1e-10 and abs(dp-item['prediction_delta'])<1e-10
            assert abs(aligned[0]-item['q0_probability'])<1e-10
            assert item['cache_hit']==(mode in ['baseline','support-labels']) and item['cache_fresh_delta']==0
            h=hashlib.sha256()
            h.update(json.dumps([item['ids'],4,'numeric-v1','b17-encoder-v1'],separators=(',',':')).encode())
            for name,value in [('features',x),*sorted(seed['state']['encoder'].items())]:
                tensor=np.asarray(value,dtype=np.float64)
                h.update(json.dumps([name,list(tensor.shape),'torch.float64']).encode())
                h.update(tensor.tobytes())
            assert h.hexdigest()==item['cache_key']
            assert item['cache_hit']==(item['cache_key']==base['cache_key'])
            if mode in ['support-order','query-order']:assert dp<1e-10 and dz<1e-10
            if mode=='support-labels':assert dz<1e-10 and dp>1e-9
            if mode=='other-query':assert abs(aligned[0]-base['probability'][0])<1e-10
        assert len(seed['checks'])==8
        assert {(c['task'],tuple(c['external_query_labels'])) for c in seed['checks']}=={(t,(a,b)) for t in tasks for a in [0,1] for b in [0,1]}
        for c in seed['checks']:
            base=next(v for v in seed['records'] if v['task']==c['task'] and v['mode']=='baseline')
            assert c['prediction']==base['probability'] and c['unchanged']
        assert seed['cache_invalidation']==dict(weights=True,preprocessing=True)
    return dict(status='PASS',independent_forward_records=count,embedding_coordinates=count*6*16,
                keyed_query_probabilities=count*2,independent_cache_fingerprints=count,max_absolute_error=maxerr)

if __name__=='__main__':
    report=json.loads((P/'evidence/b17/diagnostic.json').read_text());out=verify(report)
    (P/'_verify_b17_results.json').write_text(json.dumps(out,indent=2)+'\n');print(out)

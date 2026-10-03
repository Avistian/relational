"""Independent complete prediction and metric reconstruction; no worker imports."""
import hashlib,json,math
from pathlib import Path
import numpy as np

def audit_records(inputs,protocol,records):
    expected={c['name']:c for c in protocol['configs']}
    if len(records)!=len(expected) or len({r['config']['name'] for r in records})!=len(expected):raise ValueError('Missing or duplicate settings')
    rows=[]
    for record in records:
        c=record['config'];name=c['name']
        if c!=expected.get(name) or record['status']!='COMPLETE':raise ValueError('Wrong configuration')
        t=inputs['tables'][str(c['seed'])];ids=t['support_pool'][:c['support']];queries=t['query_ids']
        if record['inputs_sha256']!=protocol['inputs_sha256'] or record['support_ids']!=ids or record['query_ids']!=queries or set(ids)&set(queries):raise ValueError('Input/row identity mismatch')
        if record['classes']!=protocol['classes']:raise ValueError('Class columns changed')
        x=np.array(t['latent']);f=c['features']
        if c['widening']=='noise':x=np.column_stack([x,np.array(t['noise'])])[:,:f]
        else:x=x[:,[i%8 for i in range(f)]]
        a=x[ids];b=x[queries];mu=np.sum(a,axis=0)/len(a);sd=np.sqrt(np.sum((a-mu)**2,axis=0)/len(a));sd[sd==0]=1
        if not np.allclose(mu,record['mean'],rtol=1e-12,atol=1e-12) or not np.allclose(sd,record['scale'],rtol=1e-12,atol=1e-12):raise ValueError('Preprocessing mismatch')
        w=np.random.default_rng(4000+f).normal(size=(f,16))/math.sqrt(f);k=((a-mu)/sd).dot(w);q=((b-mu)/sd).dot(w)
        # Explicit query×support kernel, independently from the associative worker.
        if c['kernel']=='linear':
            pq=np.where(q<0,np.exp(np.minimum(q,0)),q+1);pk=np.where(k<0,np.exp(np.minimum(k,0)),k+1);weights=pq.dot(pk.T)
        else:
            logits=q.dot(k.T)/4;weights=np.exp(logits-np.max(logits,axis=1)[:,None])
        labels=np.array(t['y'])[ids]
        reconstructed=np.column_stack([weights[:,labels==cl].sum(axis=1) for cl in protocol['classes']]);reconstructed/=reconstructed.sum(axis=1)[:,None]
        p=np.asarray(record['probabilities']);truth=np.array(t['y'])[queries]
        if p.shape!=(len(queries),2) or not np.isfinite(p).all() or (p<0).any() or not np.allclose(p.sum(1),1,atol=1e-12,rtol=0):raise ValueError('Invalid probabilities')
        if not np.allclose(p,reconstructed,rtol=1e-11,atol=1e-12):raise ValueError('Prediction reconstruction mismatch')
        correct=sum(int(protocol['classes'][max(range(2),key=lambda j:p[i,j])]==int(y)) for i,y in enumerate(truth))
        loss=sum(-math.log(max(float(p[i,protocol['classes'].index(int(y))]),1e-15)) for i,y in enumerate(truth))/len(truth)
        timings=[record['cold_pipeline_seconds'],record['materialization_seconds'],record['process_seconds'],*record['warm_pipeline_seconds']]
        if len(record['warm_pipeline_seconds'])!=3 or any(not math.isfinite(x) or x<0 for x in timings) or not math.isfinite(record['peak_process_rss_mib']) or record['peak_process_rss_mib']<=0 or record['warm_max_delta']!=0:raise ValueError('Malformed resource receipt')
        rows.append(dict(**c,correct=correct,total=len(truth),accuracy=correct/len(truth),log_loss=loss,cold_seconds=record['cold_pipeline_seconds'],warm_median_seconds=sorted(record['warm_pipeline_seconds'])[1],peak_rss_mib=record['peak_process_rss_mib']))
    rows.sort(key=lambda x:x['name'])
    # Exact negative control: both widening methods are identical at F=8.
    by={r['config']['name']:r for r in records}
    for c in protocol['configs']:
        if c['features']==8 and c['widening']=='noise':
            other=c['name'].replace('-noise-','-copies-')
            if by[c['name']]['probabilities']!=by[other]['probabilities']:raise ValueError('Identical-input negative control failed')
    return {'status':'COMPLETE','configurations':len(rows),'predictions':sum(r['total'] for r in rows),'rows':rows,'prediction_reconstruction':'Independent explicit kernel, atol1e-12 rtol1e-11','eight_feature_negative_control':'EXACT','scope':'Course fixed random projection/kernel classifier; no pretrained TabFlex weights'}

if __name__=='__main__':
    p=Path(__file__).resolve().parent;e=p/'evidence/b04a';protocol=json.loads((e/'protocol.json').read_text());b=(e/'inputs.json').read_bytes()
    if hashlib.sha256(b).hexdigest()!=protocol['inputs_sha256']:raise ValueError('Input hash mismatch')
    records=[json.loads((e/'runs'/(c['name']+'.json')).read_text()) for c in protocol['configs']]
    report=audit_records(json.loads(b),protocol,records);(e/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(report['status'],report['configurations'],report['predictions'])

"""Independent scalar-loop oracle; no imports from the course implementation."""
import copy,json,math
from pathlib import Path
import numpy as np


def oracle(weights, data, mode):
    w={k:np.asarray(v) for k,v in weights.items()};n=len(data['support_y'])
    h=np.vstack([data['support_x'],data['query_x']])@w['embed']
    for i,label in enumerate(data['support_y']):h[i]+=w['labels'][label]
    def normalize(rows):
        result=[]
        for row in rows:
            mean=sum(row)/len(row);var=sum((v-mean)**2 for v in row)/len(row)
            result.append([(v-mean)/math.sqrt(var+1e-5) for v in row])
        return np.asarray(result)
    states=[h.copy()];inputs=[];outputs=[]
    for layer in range(3):
        old=h.copy();z=normalize(h);q=z@w[f'{layer}_q'];k=z[:n]@w[f'{layer}_k'];v=z[:n]@w[f'{layer}_v'];reads=[]
        for query in q:
            scores=[sum(float(a)*float(b) for a,b in zip(query,key))/math.sqrt(8) for key in k]
            e=[math.exp(x-max(scores)) for x in scores];den=sum(e)
            reads.append([sum(e[j]*float(v[j,c]) for j in range(n))/den for c in range(8)])
        g=z@w[f'{layer}_gate'];silu=np.asarray([[x/(1+math.exp(-x)) for x in row] for row in g])
        h=old+(normalize(reads)*silu)@w[f'{layer}_out'];inputs.append(old);outputs.append(h.copy())
        if layer==1:
            if mode=='skip':h[:n]=old[:n]
            elif mode=='permute':
                fresh=h.copy()
                for i in range(n):h[i]=old[i]+fresh[(i-1)%n]-old[(i-1)%n]
        states.append(h.copy())
    return h[n:]@w['head'],states,inputs,outputs


def audit(report):
    assert report['experiment']=='B22-SUPPORT-WRITE' and report['training']=='NONE_FIXED_RANDOM_WEIGHTS'
    assert report['seeds']==[0,1,2] and report['modes']==['normal','identity','skip','permute']
    assert len(report['data'])==24 and len(report['conditions'])==288 and len(report['paired'])==216
    data={d['id']:d for d in report['data']};assert set(data)==set(range(24))
    for i,d in data.items():
        assert d['generator_seed']==22000+i and d['family']==['linear','xor','radial'][i//8] and d['replicate']==i%8
        for prefix,counts in [('support',[6,6]),('query',[8,8])]:
            x=np.array(d[prefix+'_x']);y=np.array(d[prefix+'_y']);assert x.shape==(sum(counts),2)
            assert np.all(abs(x)<=1) and np.bincount(y).tolist()==counts
            expected=x[:,0]>0 if d['family']=='linear' else x[:,0]*x[:,1]>0 if d['family']=='xor' else (x*x).sum(axis=1)>.5
            np.testing.assert_array_equal(y,expected.astype(int))
        assert len(set(tuple(x) for x in d['support_x']+d['query_x']))==28
    rows={(x['seed'],x['episode'],x['mode']):x for x in report['conditions']}
    assert set(rows)=={(s,i,m) for s in [0,1,2] for i in range(24) for m in report['modes']}
    effects={(x['seed'],x['episode'],x['mode']):x for x in report['paired']}
    assert set(effects)=={(s,i,m) for s in [0,1,2] for i in range(24) for m in ['identity','skip','permute']}
    maximum=0.;summary=[]
    for (s,i,m),row in rows.items():
        pred,states,inputs,outputs=oracle(report['weights'][str(s)],data[i],m)
        for key,expected in [('logits',pred),('states',states),('block_inputs',inputs),('block_outputs',outputs)]:
            np.testing.assert_allclose(row[key],expected,rtol=0,atol=2e-12)
        maximum=max(maximum,float(np.max(np.abs(np.array(row['logits'])-pred))))
        normal=rows[(s,i,'normal')]
        np.testing.assert_array_equal(np.array(row['states'][2])[12:],np.array(normal['states'][2])[12:])
        if m=='identity':assert row['logits']==normal['logits'] and row['states']==normal['states']
        if m=='skip':np.testing.assert_array_equal(np.array(row['states'][2])[:12],np.array(row['block_inputs'][1])[:12])
        if m=='normal':continue
        y=data[i]['query_y'];ces=[];acc=[]
        for logits in [normal['logits'],row['logits']]:
            total=0.;correct=0
            for logits_row,label in zip(logits,y):
                peak=max(logits_row);total+=math.log(sum(math.exp(x-peak) for x in logits_row))+peak-logits_row[label]
                correct+=int(max(range(2),key=lambda j:logits_row[j])==label)
            ces.append(total/16);acc.append(correct/16)
        expected=dict(baseline_ce=ces[0],changed_ce=ces[1],delta_ce=ces[1]-ces[0],baseline_accuracy=acc[0],changed_accuracy=acc[1],delta_accuracy_pp=100*(acc[1]-acc[0]))
        for key,value in expected.items():assert abs(effects[(s,i,m)][key]-value)<2e-12,(s,i,m,key)
    for m in ['identity','skip','permute']:
        for s in [0,1,2]:
            values=[x for x in report['paired'] if x['mode']==m and x['seed']==s]
            d=[x['delta_ce'] for x in values]
            summary.append(dict(mode=m,seed=s,mean_delta_ce=float(np.mean(d)),min_delta_ce=min(d),max_delta_ce=max(d),positive=sum(v>0 for v in d),episodes=24,mean_delta_accuracy_pp=float(np.mean([x['delta_accuracy_pp'] for x in values]))))
    return dict(status='PASS',trajectories=288,query_predictions=4608,paired_effects=216,oracle_max_logit_error=maximum,preserved_intervention_query_rows=288*16,summary=summary)

if __name__=='__main__':
 P=Path(__file__).resolve().parent;r=json.loads((P/'evidence/b22/diagnostic.json').read_text());out=audit(r)
 mutations=[]
 for kind in ['missing_trajectory','wrong_query_state','wrong_metric','changed_label','wrong_permutation']:
    bad=copy.deepcopy(r)
    if kind=='missing_trajectory':bad['conditions'].pop()
    elif kind=='wrong_query_state':bad['conditions'][2]['states'][2][12][0]+=.1
    elif kind=='wrong_metric':bad['paired'][1]['delta_ce']+=.1
    elif kind=='changed_label':bad['data'][0]['query_y'][0]=1-bad['data'][0]['query_y'][0]
    elif kind=='wrong_permutation':bad['conditions'][3]['states'][2][0][0]+=.1
    try:audit(bad)
    except (AssertionError,ValueError):mutations.append(kind)
    else:raise AssertionError('Accepted corrupt evidence: '+kind)
 out['corruptions_rejected']=mutations;(P/'_verify_b22_results.json').write_text(json.dumps(out,indent=2)+'\n');print(out)

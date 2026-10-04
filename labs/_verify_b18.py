"""Independent saved-sample oracle. Does not import the experiment implementation."""
import copy,gzip,itertools,json,math
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent

def verify(r):
    assert r['experiment']=='B18-CONTEXT-SUFFICIENCY' and len(r['fixtures'])==24 and len(r['conditions'])==96
    keys={(s,n,h) for s in range(3) for n in [8,64,512,4096] for h in ['uniform','concentrated']}
    fixtures={(f['seed'],f['degree'],f['shape']):f for f in r['fixtures']}
    assert set(fixtures)==keys
    assert {(c['seed'],c['degree'],c['shape'],c['budget']) for c in r['conditions']}=={(*k,b) for k in keys for b in [8,32,128,512]}
    count=0;maxerr=0.
    for key,f in fixtures.items():
        seed,n,shape=key;rows=f['rows']
        assert len(rows)==n+2 and len({x['id'] for x in rows})==n+2
        assert f['future_invariant'] is True
        values=[x['amount'] for x in rows if x['kind']=='credit' and x['day']<=30]
        assert len(values)==n and len(f['draws'])==100
        expected=[1200/n]*n if shape=='uniform' else [960.]+[240/(n-1)]*(n-1)
        assert sorted(values)==sorted(expected)
        assert [x['id'] for x in rows[:n]]==[f'e{i}' for i in range(n)]
        assert all(x['day']==i%28+1 and x['month']==1 for i,x in enumerate(rows[:n]))
        assert rows[-2]==dict(id='debit',day=15,month=1,kind='debit',amount=700.)
        assert rows[-1]==dict(id='future',day=31,month=2,kind='credit',amount=1e9)
        # Validate the frozen PRNG schedule independently, including the initial amount shuffle.
        rng=np.random.default_rng(np.random.SeedSequence([seed,n,shape=='concentrated',18]))
        assert values==np.array(expected)[rng.permutation(n)].tolist()
        assert f['draws']==[rng.permutation(n)[:min(512,n)].tolist() for _ in range(100)]
        truth=math.fsum(values);assert abs(f['truth']-truth)<1e-9
        for c in [c for c in r['conditions'] if (c['seed'],c['degree'],c['shape'])==key]:
            k=min(c['budget'],n);assert c['k']==k
            assert set(c['estimates'])==set(c['metrics'])=={'exact','sampled','corrected','aggregate'}
            for arm in c['estimates']:
                assert len(c['estimates'][arm])==100
                calculated=[]
                for draw in f['draws']:
                    assert len(draw)==min(512,n) and len(set(draw))==len(draw) and all(0<=i<n for i in draw)
                    subtotal=math.fsum(values[i] for i in draw[:k])
                    calculated.append(subtotal if arm=='sampled' else subtotal*n/k if arm=='corrected' else truth)
                err=max(abs(a-b) for a,b in zip(calculated,c['estimates'][arm]));maxerr=max(maxerr,err);assert err<1e-7
                errors=[x-truth for x in calculated]
                metric=dict(n=100,bias=math.fsum(errors)/100,rmse=math.sqrt(math.fsum(x*x for x in errors)/100))
                assert c['metrics'][arm]['n']==100
                assert max(abs(metric[x]-c['metrics'][arm][x]) for x in ['bias','rmse'])<1e-7
                count+=100
    # An exhaustive population oracle establishes unbiasedness without Monte Carlo tolerance.
    estimates=[sum(s)*4/2 for s in itertools.combinations([1,2,7,10],2)]
    assert sum(estimates)/len(estimates)==20 and len(set(estimates))>1
    return dict(arm_estimates=count,max_absolute_oracle_error=maxerr,exhaustive_unbiasedness=True)

if __name__=='__main__':
    r=json.loads(gzip.decompress((P/'evidence/b18/diagnostic.json.gz').read_bytes()))
    result=verify(r)
    summary=copy.deepcopy(r);summary.pop('fixtures')
    for c in summary['conditions']:c.pop('estimates')
    assert summary==json.loads((P/'evidence/b18/summary.json').read_text()),'Displayed summary differs from verified raw evidence'
    rejected=[]
    for label in ['prediction','metric','draw','time','shape','grid','future']:
        bad=copy.deepcopy(r)
        if label=='prediction':bad['conditions'][0]['estimates']['corrected'][0]+=1
        elif label=='metric':bad['conditions'][0]['metrics']['sampled']['bias']+=1
        elif label=='draw':bad['fixtures'][0]['draws'][0][0]=bad['fixtures'][0]['draws'][0][1]
        elif label=='time':bad['fixtures'][0]['rows'][0]['day']=31
        elif label=='shape':bad['fixtures'][0]['rows'][0]['amount']+=1
        elif label=='grid':bad['conditions'].pop()
        else:bad['fixtures'][0]['rows'][-1]['day']=30
        try:verify(bad)
        except AssertionError:rejected.append(label)
        else:raise AssertionError('Corruption accepted: '+label)
    result.update(status='PASS',corruptions_rejected=rejected)
    (P/'_verify_b18_results.json').write_text(json.dumps(result,indent=2)+'\n');print(result)

"""Independent scalar reconstruction of saved B19 evidence; no experiment imports."""
import copy,itertools,json,math,statistics
from pathlib import Path
P=Path(__file__).resolve().parent

def close(a,b):
    assert math.isfinite(a) and math.isfinite(b) and abs(a-b)<1e-12,(a,b)

def verify(r):
    assert r['protocol']=='B19-EVIDENCE-BOUNDARIES-v1'
    assert len(r['fixtures'])==3 and len(r['arms'])==12
    fixtures={f['seed']:f for f in r['fixtures']};assert set(fixtures)=={0,1,2}
    seen=set();predictions=0
    for a in r['arms']:
        key=(a['seed'],a['regime'],a['model']);assert key not in seen;seen.add(key)
        f=fixtures[a['seed']];rows={x['id']:x for x in f['rows']}
        assert len(rows)==192 and set(rows)==set(range(192))
        assert {x['group'] for x in rows.values()}==set(range(24))
        assert sum(next(x['y'] for x in rows.values() if x['group']==g) for g in range(24))==12
        for i,x in rows.items():assert x['group']==i//8 and x['y'] in (0,1) and x['signal'] in (0,1)
        for g in range(24):assert len({x['y'] for x in rows.values() if x['group']==g})==1
        train=a['train_ids'];test=f['tests'][a['regime']]
        assert len(set(train))==len(train) and set(train)==set(rows)-set(test)
        assert len(test)==(48 if a['regime']=='random' else 64)
        counts=[sum(rows[i]['group']==g for i in test) for g in range(24)]
        assert counts==[2]*24 if a['regime']=='random' else sorted(counts)==[0]*16+[8]*8
        assert [p['id'] for p in a['predictions']]==test
        tg={rows[i]['group'] for i in train};qg={rows[i]['group'] for i in test};overlap=sorted(tg&qg)
        assert a['audit']==dict(train_n=len(train),test_n=len(test),overlap_groups=overlap,group_disjoint=not overlap)
        assert len(overlap)==(24 if a['regime']=='random' else 0)
        prevalence=sum(rows[i]['y'] for i in train)/len(train)
        losses=[]
        for p in a['predictions']:
            x=rows[p['id']];assert p['y']==x['y'] and p['group']==x['group']
            same=[rows[i]['y'] for i in train if rows[i]['group']==x['group']]
            expected=(sum(same)/len(same) if same else prevalence) if a['model']=='group_memory' else .2+.6*x['signal']
            close(p['p'],expected);losses.append((expected-x['y'])**2);predictions+=1
        close(a['brier'],sum(losses)/len(losses))
    assert seen==set(itertools.product(range(3),['random','grouped'],['group_memory','signal']))
    source={(x['dataset'],x['seed'],x['model']):x for x in r['score_rows']};assert len(source)==12
    frozen={'A':[.1,.1,.1,None],'B':[.2]*4,'RF':[.25,.25,.25,.9]}
    for (d,s,m),v in source.items():assert s==0 and v['loss']==frozen[m][int(d[1:])] and v['origin']==('missing' if v['loss'] is None else 'measured')
    for policy,n in [('measured',3),('rf',4)]:
        p=r['policies'][policy];assert p['keys']==[[f'd{i}',0] for i in range(n)] and len(p['cells'])==n*3
        expected_keys=set(itertools.product([f'd{i}' for i in range(n)],[0],['A','B','RF']))
        assert {(c['dataset'],c['seed'],c['model']) for c in p['cells']}==expected_keys
        for c in p['cells']:
            raw=source[(c['dataset'],c['seed'],c['model'])]
            if raw['loss'] is None:assert c['origin']=='imputed:RF' and c['loss']==.9
            else:assert c==raw
        for m in frozen:close(p['means'][m],sum(c['loss'] for c in p['cells'] if c['model']==m)/n)
        assert p['order']==sorted(frozen,key=lambda m:(p['means'][m],m))
    vals={'a':[.10,.11,.09],'b':[-.04,-.05,-.03],'c':[.02,.03,.01]}
    assert r['deltas']==[dict(dataset=d,seed=s,delta=v) for d,vs in vals.items() for s,v in enumerate(vs)]
    means=[sum(v)/3 for v in vals.values()];s=r['uncertainty'];assert s['n_datasets']==3 and s['n_seed_rows']==9
    close(s['mean'],sum(means)/3);close(s['se_dataset'],statistics.stdev(means)/math.sqrt(3));close(s['se_naive_rows'],statistics.stdev(sum(vals.values(),[]))/3)
    for d,v in zip(vals,means):close(s['dataset_means'][d],v)
    for key,pop in [('dataset_bootstrap',means),('three_row_resampling',sum(vals.values(),[]))]:
        expected=sorted(sum(x)/3 for x in itertools.product(pop,repeat=3));assert len(expected)==len(r[key])
        for a,b in zip(expected,r[key]):close(a,b)
    assert r['duplicated_seeds']['n_datasets']==3 and r['duplicated_seeds']['n_seed_rows']==18
    close(r['duplicated_seeds']['se_dataset'],s['se_dataset'])
    assert r['duplicated_seeds']['se_naive_rows']<s['se_naive_rows']
    return dict(status='PASS',arms=12,predictions=predictions,score_cells=21,dataset_resamples=27,three_row_resamples=729)

if __name__=='__main__':
    r=json.loads((P/'evidence/b19/diagnostic.json').read_text());out=verify(r)
    mutations=[lambda x:x['arms'][0]['predictions'][0].update(p=.123),lambda x:x['arms'][1]['predictions'][0].update(y=9),lambda x:x['arms'][0]['train_ids'].append(0),lambda x:x['arms'].pop(),lambda x:x['policies']['rf']['cells'][-3].update(origin='measured'),lambda x:x['uncertainty'].update(n_datasets=9),lambda x:x['dataset_bootstrap'].pop(),lambda x:x['score_rows'][0].update(loss=.99)]
    for mutate in mutations:
        bad=copy.deepcopy(r);mutate(bad)
        try:verify(bad)
        except (AssertionError,KeyError):pass
        else:raise AssertionError('Corruption accepted')
    out['corruptions_rejected']=len(mutations)
    (P/'_verify_b19_results.json').write_text(json.dumps(out,indent=2)+'\n');print(out)

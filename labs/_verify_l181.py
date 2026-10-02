"""Independent SQL labels, scalar baseline predictions and sklearn metrics."""
def independent181(packet,report):
    import json,statistics
    from pathlib import Path
    import duckdb,numpy as np,pandas as pd
    from sklearn.metrics import mean_absolute_error,r2_score
    packet=Path(packet);cfg=json.loads((packet/'config.json').read_text());count=0;max_error=0.;labels=0
    for task,spec in cfg['tasks'].items():
        path=str(packet/'db'/(spec['table']+'.parquet'));raw=duckdb.sql('SELECT * FROM read_parquet(?)',params=[path]).df()
        tables={}
        for split in ['train','val','test']:
            lo,hi=spec['split_ranges'][split]
            oracle=duckdb.sql(f'SELECT "{spec["key"]}" AS entity, epoch_ns(date) AS time, CAST(position AS DOUBLE) AS y FROM raw WHERE date > ? AND date <= ? AND position IS NOT NULL ORDER BY time, entity',params=[lo,hi]).df()
            saved=pd.read_parquet(packet/task/(split+'.parquet'));assert oracle.equals(saved),(task,split);labels+=len(saved);tables[split]=oracle
        for split in ['val','test']:
            fit=tables['train'] if split=='val' else pd.concat([tables['train'],tables['val']],ignore_index=True)
            q=tables[split];ys=fit.y.tolist();groups={}
            for row in fit.itertuples():groups.setdefault(row.entity,[]).append(row.y)
            for kind,m in report['tasks'][task]['scores'][split].items():
                if kind=='global_zero':pred=[0.]*len(q)
                elif kind=='global_mean':pred=[statistics.mean(ys)]*len(q)
                elif kind=='global_median':pred=[statistics.median(ys)]*len(q)
                else:
                    op=statistics.mean if kind=='entity_mean' else statistics.median
                    pred=[op(groups[e]) if e in groups else 0. for e in q.entity]
                saved=pd.read_parquet(packet/task/(split+'-'+kind+'.parquet'))
                assert np.array_equal(saved[['entity','time']],q[['entity','time']]);assert np.allclose(saved.pred,pred,rtol=0,atol=1e-12)
                vals={'r2':r2_score(q.y,pred),'mae':mean_absolute_error(q.y,pred)}
                for name,value in vals.items():
                    error=abs(value-m[name]);max_error=max(max_error,error);assert error<1e-10
                count+=len(q)
    return dict(labels=labels,predictions=count,max_metric_error=max_error)

if __name__=='__main__':
    import copy,hashlib,json,tempfile,shutil,ast,re
    from pathlib import Path
    import numpy as np,pandas as pd,torch,torch_frame
    from bs4 import BeautifulSoup
    from _audit_l181 import audit181
    from _check_l181 import checks
    from relkit.autocomplete_l181 import visible_columns,baseline_predictions,keyed_scores
    P=Path(__file__).resolve().parent;E=P/'evidence/l181';packet=E/'packet';r=json.loads((E/'report.json').read_text());pins=json.loads((E/'input-manifest.json').read_text())
    assert checks(visible_columns,baseline_predictions,keyed_scores)=='PASS';out=independent181(packet,r)
    # Extract numbers from the archived primary paper, retaining means and source rows.
    soup=BeautifulSoup((P/'sources/l181/relbench-v2.html').read_text(),'html.parser');targets={};comparisons=[]
    kinds=['global_zero','global_mean','global_median','entity_mean','entity_median','LightGBM','GNN']
    for number,metric in [(5,'r2'),(15,'mae')]:
        figure=next(f for f in soup.find_all('figure') if (f.get('id') or '').endswith('T'+str(number)))
        task=None;split=None
        for row in figure.find_all('tr'):
            text=row.get_text(' ',strip=True)
            if 'qualifying-position' in text:task='qualifying-position'
            elif 'results-position' in text:task='results-position'
            elif any(x in text for x in ['transactions-price','users-birthyear','review-rating']):task=None
            if task is None:continue
            split='val' if 'Val' in text else 'test' if 'Test' in text else None
            if split is None:continue
            maths=row.find_all('math');numbers=[]
            for m in maths:
                alt=m.get('alttext','');matches=re.findall(r'[-+]?\d+(?:\.\d+)?',alt)
                if matches:numbers.append(float(matches[0]))
            assert len(numbers)==7,(task,split,numbers)
            for kind,value in zip(kinds,numbers):
                targets.setdefault(task,{}).setdefault(split,{}).setdefault(kind,{})[metric]=value
                if kind in kinds[:5]:
                    actual=r['tasks'][task]['scores'][split][kind][metric];delta=abs(actual-value)
                    comparisons.append(dict(task=task,split=split,kind=kind,metric=metric,paper=value,actual=actual,absolute_difference=delta,status='CLOSE_ROUNDED' if delta<=.001 else 'OUTSIDE_TOLERANCE'))
    assert len(comparisons)==40;assert all(c['status']=='CLOSE_ROUNDED' for c in comparisons),comparisons
    (E/'paper-comparison.json').write_text(json.dumps(dict(targets=targets,comparisons=comparisons),indent=2)+'\n')
    # A dense per-second index and its two extrema imply identical source predicates.
    rng=np.random.default_rng(181)
    for i in range(100):
        a=int(rng.integers(0,10000));length=int(rng.integers(1,200));end=pd.Timestamp('2000-01-01')+pd.Timedelta(seconds=a)
        start=end+pd.Timedelta(seconds=length);dense=pd.date_range(start,end,freq='-1s')
        assert dense.min()==end and dense.max()==start
        x=end+pd.to_timedelta(rng.integers(-3,length+4,50),unit='s');assert np.array_equal((x>dense.min())&(x<=dense.max()),(x>end)&(x<=start))
    # Permute all keys and vary targets: no positional scoring can survive these checks.
    for i in range(100):
        n=20;y=rng.normal(size=n);p=rng.normal(size=n)
        truth=pd.DataFrame({'entity':np.arange(n)//2,'time':np.arange(n)%2,'y':y});pred=truth[['entity','time']].copy();pred['pred']=p;pred=pred.sample(frac=1,random_state=i)
        from sklearn.metrics import r2_score,mean_absolute_error
        got=keyed_scores(truth,pred);assert abs(got['r2']-r2_score(y,p))<1e-12 and abs(got['mae']-mean_absolute_error(y,p))<1e-12
    rejected=0
    for funcs in [(lambda c,*a:c,baseline_predictions,keyed_scores),(visible_columns,lambda fit,q,k:np.zeros(len(q)),keyed_scores),(visible_columns,baseline_predictions,lambda y,p:dict(n=len(y),mae=0,r2=1))]:
        try:checks(*funcs)
        except (AssertionError,ValueError):rejected+=1
        else:raise AssertionError('Incorrect learner function passed')
    for name in ['config.json','db/results.parquet','gradient-preflight.json','results-position/val-global_mean.parquet']:
        bad=copy.deepcopy(pins);bad['files'][name]='0'*64
        try:audit181(packet,bad,visible_columns,baseline_predictions,keyed_scores)
        except ValueError:pass
        else:raise AssertionError('Corrupted input admitted')
    # Independent actual LinearEncoder experiment and mathematical NaN witness.
    torch.set_num_threads(1);torch.manual_seed(0)
    frame=pd.read_parquet(packet/'results-position-circuits-gradient-input.parquet')
    ds=torch_frame.data.Dataset(frame,{c:torch_frame.numerical for c in frame}).materialize()
    from torch_frame.data.stats import StatType
    enc=torch_frame.nn.LinearEncoder(out_channels=128,stats_list=[ds.col_stats[c] for c in frame],stype=torch_frame.numerical);enc.reset_parameters()
    feat=ds.tensor_frame.feat_dict[torch_frame.numerical];z=enc(feat);assert torch.isfinite(z).all();z.square().mean().backward()
    nonfinite=int((~torch.isfinite(enc.weight.grad)).sum());expected=int(frame.isna().any().sum())*128;assert nonfinite==expected==128
    # d(x*w)/dw = x. Post-product masking can leave 0*NaN in this derivative.
    x=torch.tensor([float('nan'),2.]);w=torch.tensor(1.,requires_grad=True);torch.nan_to_num(x*w).sum().backward();assert torch.isnan(w.grad)
    out.update(status='PASS',paper_metric_comparisons=len(comparisons),paper_metric_status='ALL_CLOSE_ROUNDED',split_equivalence_cases=100,random_keyed_cases=100,wrong_learner_functions_rejected=rejected,corrupt_packets_rejected=4,independent_linear_encoder_nonfinite=nonfinite,gradient_environment=dict(torch=torch.__version__,torch_frame=torch_frame.__version__,device='cpu'),cloud_usd=0)
    (P/'_verify_l181_results.json').write_text(json.dumps(out,indent=2)+'\n');print(out)

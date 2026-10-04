"""Independent scalar oracle and tamper checks. Does not import course functions."""
import copy,json,math,statistics,hashlib
from pathlib import Path
P=Path(__file__).resolve().parent;E=P/'evidence/b19a'
def oracle(report):
    assert report['protocol']=='B19a-SAME-MEAN'
    assert report['truth']==[[-2,.25],[0,.5],[2,.25]]
    assert report['forecasts']=={'narrow':{'values':[-1,1],'probabilities':[.5,.5]},'calibrated':{'values':[-2,0,2],'probabilities':[.25,.5,.25]},'wide':{'values':[-4,4],'probabilities':[.5,.5]}}
    expected={'narrow':(1,4,.5,2),'calibrated':(.75,4,.75,2),'wide':(2,8,1,8)}
    assert len(report['rows'])==9 and len(report['summary'])==3
    seen=set()
    for r in report['rows']:
        key=(r['forecast'],r['y']);assert key not in seen;seen.add(key)
        v=report['forecasts'][r['forecast']]['values'];p=report['forecasts'][r['forecast']]['probabilities'];y=r['y']
        # Integrate (F(z)-1{y<=z})^2 over all constant CDF segments.
        knots=sorted(set(v+[y]));score=0
        for a,b in zip(knots,knots[1:]):
            mid=(a+b)/2;cdf=sum(w for x,w in zip(v,p) if x<=mid)
            score+=(b-a)*(cdf-int(y<=mid))**2
        assert abs(score-r['crps'])<1e-12
        assert r['mean']==0 and r['squared_error']==y*y
        assert r['weight']==({-2:.25,0:.5,2:.25}[y])
        assert (r['lower'],r['upper'])==({'narrow':(-1,1),'calibrated':(-2,0),'wide':(-4,4)}[r['forecast']])
        assert r['covered']==(r['lower']<=y<=r['upper']) and r['width']==r['upper']-r['lower']
        width=r['upper']-r['lower'];penalty=4*(r['lower']-y if y<r['lower'] else y-r['upper'] if y>r['upper'] else 0)
        assert r['interval_score']==width+penalty
    for s in report['summary']:
        assert math.isclose(s['rmse'],math.sqrt(2))
        assert tuple(s[k] for k in ['crps','interval_score','covered','width'])==expected[s['forecast']]
    assert len(report['propriety_grid'])==15
    assert len({tuple(r['probabilities']) for r in report['propriety_grid']})==15
    for r in report['propriety_grid']:
        a,b,c=r['probabilities'];assert a+b+c==1
        # Integral expected squared CDF error + irreducible truth score.
        assert abs(r['expected_crps']-(.75+2*(a-.25)**2+2*(a+b-.75)**2))<1e-12
    winners=[r for r in report['propriety_grid'] if r['expected_crps']==.75];assert len(winners)==1 and winners[0]['probabilities']==[.25,.5,.25]
    assert len(report['unit_interventions'])==18
    for r in report['unit_interventions']:assert r['crps_scaled']==r['expected'] and r['interval_scaled']==r['interval_expected']
    c=report['calibration'];assert c['fitted']['radius']==7 and c['fitted']['k']==8 and c['radius_unchanged']
    assert {r['id'] for r in c['test']}.isdisjoint(c['fitted']['ids']);assert c['coverage']==.5 and c['shifted_coverage']==0

def main():
    r=json.loads((E/'diagnostic.json').read_text());oracle(r)
    mutations=[lambda x:x['rows'][0].update(crps=9),lambda x:x['rows'].append(x['rows'][0]),lambda x:x['summary'][0].update(rmse=0),lambda x:x['propriety_grid'][0].update(expected_crps=0),lambda x:x['calibration']['fitted'].update(radius=20),lambda x:x['unit_interventions'][0].update(crps_scaled=4),lambda x:x['rows'][0].update(weight=.5),lambda x:x['calibration']['fitted']['ids'].append('t0'),lambda x:x['rows'][0].update(covered=True)]
    for mutate in mutations:
        bad=copy.deepcopy(r);mutate(bad)
        try:oracle(bad)
        except AssertionError:pass
        else:raise AssertionError('Corruption accepted')
    # Independently recompute paired ranks from raw parquet with pandas; compare ALL compact bytes.
    import pandas as pd,numpy as np
    records=json.loads((E/'released-scores.json').read_text());tables=[]
    for p in sorted((P/'sources/b19a/output').glob('*.parquet')):
        frame=pd.read_parquet(p)[['dataset','model','fold','crps','r2','crls']];tables.append(frame)
    frame=pd.concat(tables,ignore_index=True).sort_values(['dataset','model','fold'])
    authentic=[[str(row.dataset),str(row.model),int(row.fold),*[None if pd.isna(getattr(row,k)) else float(getattr(row,k)) for k in ['crps','r2','crls']]] for row in frame.itertuples()]
    assert records==authentic
    report=json.loads((E/'reproduction.json').read_text());portable=json.loads((E/'portable-replay.json').read_text())
    for metric in ['crps','r2','crls']:
        pivot=frame.groupby(['dataset','model'])[metric].mean().unstack();pivot=pivot.loc[:,pivot.notna().mean()>=.9].dropna()
        ranks=pivot.rank(axis=1,method='average',ascending=metric!='r2').mean()
        assert list(sorted(pivot.index))==portable[metric]['datasets']
        for row in portable[metric]['rows']:assert abs(row['meanrank']-ranks[row['model']])<1e-12
        assert not report['comparisons'][metric]['paper_mismatches']
    from _reproduce_b19a import replay
    for mutation in ['duplicate','missing_fold','nonfinite']:
        bad=copy.deepcopy(records)
        if mutation=='duplicate':bad.append(bad[0])
        elif mutation=='missing_fold':bad.pop(0)
        else:bad[0][3]=float('inf')
        try:replay(bad)
        except ValueError:pass
        else:raise AssertionError('Invalid score evidence accepted')
    result=dict(status='PASS',independent_cdf_integral_rows=9,finite_propriety_candidates=15,unit_interventions=18,diagnostic_corruptions_rejected=9,score_corruptions_rejected=3,authenticated_score_rows=len(records),published_tables_checked=3)
    (P/'_verify_b19a_results.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
if __name__=='__main__':main()

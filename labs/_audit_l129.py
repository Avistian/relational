"""Independent Python reconstruction of ALL released SQL feature values and labels.
No calls to make_features or the Jinja SQL in the reference calculation.
"""
import io,json,math,zipfile
from pathlib import Path
import numpy as np,pandas as pd
from relkit.fe_experiment_l129 import load_archives
P=Path(__file__).resolve().parent;S=P/'sources/l129';O=P/'evidence/l129'
tables,labels,_=load_archives(S)
def index(df,cols):
    assert not df.duplicated(cols).any()
    return {tuple(row[c] for c in cols):row for row in df.to_dict('records')}
drivers=index(tables['drivers'],['driverId']);constructors=index(tables['constructors'],['constructorId'])
races=index(tables['races'],['raceId']);circuits=index(tables['circuits'],['circuitId'])
results=tables['results'].drop_duplicates(['raceId','driverId'],keep='first')
res=index(results,['raceId','driverId']);cr=index(tables['constructor_results'],['raceId','constructorId'])
def enrich(df):
    out=df.sort_values(['raceId','position'],kind='stable').copy()
    out['points_lag']=out.points-out.groupby('raceId').points.shift(1)
    out['points_lead']=out.points-out.groupby('raceId').points.shift(-1)
    return out
stands=enrich(tables['standings']);cs=index(enrich(tables['constructor_standings']),['raceId','constructorId'])
bydriver={i:g.sort_values('date').to_dict('records') for i,g in stands.groupby('driverId')}
with zipfile.ZipFile(S/'db.zip') as z:all_results=pd.read_parquet(io.BytesIO(z.read('db/results.parquet')))
future={i:g for i,g in all_results.groupby('driverId')}
report={};total=0;past_links=0;schedule_links=0;example=None
for split,label in labels.items():
    observed=pd.read_parquet(O/f'{split}-features.parquet').set_index(['driverId','date'])
    refs=[];lap_pairs=[];past_max=[];schedule_dates=[]
    for t,queries in label.groupby('date',sort=False):
        before=[r for r in races.values() if r['date']<t]
        last=max(before,key=lambda r:r['date']) if before else None
        lower=t-pd.DateOffset(months=2);upper=t+pd.DateOffset(months=1)
        for q in queries.to_dict('records'):
            entity=q['driverId'];d=drivers[(entity,)];row={c:None for c in observed.columns}
            if split!='test':row['position']=q['position']
            row.update(week_of_year=t.isocalendar().week,driver_ref=d['driverRef'],driver_age=t.year-d['dob'].year,driver_nationality=d['nationality'])
            eligible=[r for r in bydriver.get(entity,[]) if r['date']<t]
            st=max(eligible,key=lambda r:r['date']) if eligible else None
            if st:
                for col in ['position','points','wins','points_lag','points_lead']:row['driver_'+col]=st[col]
                row['days_since_last_race']=(t.normalize()-st['date'].normalize()).days
                rr=res.get((st['raceId'],entity));c=cs.get((st['raceId'],rr['constructorId'])) if rr else None
                if c:
                    for col in ['position','points','wins','points_lag','points_lead']:row['constructor_'+col]=c[col]
                    cc=constructors[(c['constructorId'],)];row['constructor_ref']=cc['constructorRef'];row['constructor_nationality']=cc['nationality']
                    row['position_diff']=st['position']-c['position']
                    row['points_ratio']=st['points']/c['points'] if c['points'] else None
                    row['wins_ratio']=st['wins']/c['wins'] if c['wins'] else None
            for i in [1,2,3]:
                past=res.get((last['raceId']-i+1,entity)) if last else None
                if past and past['date']>=lower:
                    assert past['date']<t,('future result selected by raceId arithmetic',past,t)
                    past_links+=1;past_max.append((t-past['date']).total_seconds())
                    for col in ['position','points','grid','rank']:row[f'past_{i}_driver_{col}']=past[col]
                    row[f'past_{i}_position_gain']=past['position']-past['grid']
                    row[f'past_{i}_dnf']=int(past['statusId']!=1)
                    cp=cr.get((past['raceId'],past['constructorId']))
                    row[f'past_{i}_constructor_points']=cp['points'] if cp else None
                    lap_pairs.append((len(refs),i,past['raceId'],past['laps']))
                up=races.get((last['raceId']+i,)) if last else None
                if up and up['date']<=upper:
                    row[f'upcoming_{i}_round']=up['round'];row[f'upcoming_{i}_circuit_id']=up['circuitId']
                    schedule_dates.append((up['date']-t).total_seconds());schedule_links+=1
            refs.append(((entity,t),row))
            fr=future[entity];fr=fr[(fr.date>t)&(fr.date<=t+pd.Timedelta(days=60))]
            assert abs(fr.positionOrder.mean()-q['position'])<1e-12
    maxima={}
    for _,_,race,laps in lap_pairs:maxima[race]=max(maxima.get(race,0),laps)
    for idx,i,race,laps in lap_pairs:refs[idx][1][f'past_{i}_pct_laps_completed']=laps/maxima[race] if maxima[race] else None
    count=0;maxerr=0
    for key,ref in refs:
        actual=observed.loc[key]
        for col,expected in ref.items():
            got=actual[col]
            if pd.isna(expected):assert pd.isna(got),(split,key,col,got,expected)
            elif isinstance(expected,str):assert got==expected,(split,key,col,got,expected)
            else:
                err=abs(float(got)-float(expected));assert err<1e-10,(split,key,col,got,expected);maxerr=max(err,maxerr)
            count+=1
        if split=='val' and example is None and ref['past_1_driver_position'] is not None:
            example=dict(entity=int(key[0]),cutoff=str(key[1]),features={k:None if pd.isna(v) else int(v) if isinstance(v,(int,np.integer)) else float(v) if isinstance(v,(float,np.floating)) else str(v) for k,v in ref.items()})
    report[split]=dict(rows=len(refs),checked_values=count,max_absolute_error=maxerr,minimum_past_lag_seconds=min(past_max) if past_max else None,schedule_links=len(schedule_dates),future_schedule_links=sum(d>0 for d in schedule_dates))
    total+=count
out=dict(status='PASS',method='Independent pandas/Python reconstruction of all50 engineered fields and raw-event future targets',splits=report,total_values=total,past_result_links=past_links,schedule_links=schedule_links,example=example,availability='Race event dates verified; actual schedule publication/ingestion and static attribute histories unavailable',release_quirks=['raceId arithmetic chooses last/next slots, not driver-specific last three appearances','pct_laps_completed denominator uses eligible joined rows within each split','driverId retained by trainer','date ignored by LightGBM input adapter','database capped at2010-01-01 even for later test queries'])
(O/'sql-audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))

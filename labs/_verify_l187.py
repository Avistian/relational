"""Independent SQL, explicit neighbor deletion, scalar scoring and analytic checks."""
import json,math
from pathlib import Path
import numpy as np
import pandas as pd
import duckdb


def independent187(db, report, counts, releases, bounded_histogram):
    schema={'circuits':('circuitId',{}),'constructors':('constructorId',{}),'drivers':('driverId',{}),
        'races':('raceId',{'circuitId':'circuits'}),
        'results':('resultId',{'driverId':'drivers','raceId':'races','constructorId':'constructors'}),
        'qualifying':('qualifyId',{'driverId':'drivers','raceId':'races','constructorId':'constructors'}),
        'standings':('driverStandingsId',{'driverId':'drivers','raceId':'races'}),
        'constructor_results':('constructorResultsId',{'raceId':'races','constructorId':'constructors'}),
        'constructor_standings':('constructorStandingsId',{'raceId':'races','constructorId':'constructors'})}
    assert set(db)==set(schema)
    edges=0;fk_columns=0
    for name,(pk,fks) in schema.items():
        frame=db[name];assert frame[pk].notna().all() and frame[pk].is_unique
        for fk,parent in fks.items():
            assert frame[fk].notna().all() and frame[fk].isin(db[parent][schema[parent][0]]).all(),(name,fk)
            edges+=len(frame);fk_columns+=1
    con=duckdb.connect()
    for name,frame in db.items():con.register(name,frame)
    sql=con.execute('''SELECT d.driverId,
        (SELECT count(*) FROM results r WHERE r.driverId=d.driverId) results,
        (SELECT count(*) FROM qualifying q WHERE q.driverId=d.driverId) qualifying,
        (SELECT count(*) FROM standings s WHERE s.driverId=d.driverId) standings
        FROM drivers d ORDER BY d.driverId''').df()
    for col in ['driverId','results','qualifying','standings']:
        assert np.array_equal(sql[col],counts[col]),col
    children=sql.results+sql.qualifying+sql.standings
    owned_edges=3*sql.results+3*sql.qualifying+2*sql.standings
    assert np.array_equal(counts.owned_rows,children+1)
    assert np.array_equal(counts.driver_incident_edges,children)
    assert np.array_equal(counts.owned_incident_edges,owned_edges)
    assert report['tables']=={k:len(v) for k,v in db.items()}
    assert report['drivers']==len(sql) and report['owned_rows']==int((children+1).sum())
    assert report['driver_edges']==int(children.sum()) and report['owned_edges']==int(owned_edges.sum())
    maximal=counts.sort_values(['owned_rows','driverId'],ascending=[False,True]).iloc[0]
    assert report['maximum_owner']=={k:int(v) for k,v in maximal.to_dict().items()}
    domain=report['domain'];events=db['results'];queries=0;residual_pairs=0
    all_rows=[(int(r),int(d),int(c)) for r,d,c in events[['resultId','driverId','constructorId']].itertuples(index=False,name=None)]
    by_owner={int(d):[] for d in db['drivers'].driverId}
    for r,d,c in sorted(all_rows):by_owner[d].append(c)
    raw=[sum(c==cat for _,_,c in all_rows) for cat in domain]
    assert raw==report['raw_histogram']
    for cap in [1,5,20]:
        reference=[sum(c==cat for values in by_owner.values() for c in values[:cap]) for cat in domain]
        assert reference==report['clipped_histograms'][str(cap)]
        for owner,values in by_owner.items():
            # Recompute on each actual neighboring full event table, no sampled owners.
            neighbor=bounded_histogram(events[events.driverId!=owner],domain,cap)
            delta=np.array(reference)-neighbor
            expected=[values[:cap].count(cat) for cat in domain]
            assert np.array_equal(delta,expected) and int(abs(delta).sum())<=cap
            queries+=1
    for d in db['drivers'].driverId:
        expected=int(counts.loc[counts.driverId==d,'owned_rows'].iloc[0]);removed=1
        for table in ['results','qualifying','standings']:
            removed+=len(db[table])-len(db[table][db[table].driverId!=d])
        assert expected==removed
    # Shared constructor aggregates remain connected by public race/constructor keys.
    residual_pairs=con.execute('''SELECT count(*) FROM
      (SELECT DISTINCT r.driverId, c.constructorResultsId FROM results r
       JOIN constructor_results c USING(raceId,constructorId))''').fetchone()[0]
    assert len(releases)==270 and len({(r['cap'],r['epsilon'],r['seed']) for r in releases})==270
    max_error=0.;expected_keys={(c,e,s) for c in [1,5,20] for e in [.5,1.,2.] for s in range(30)}
    assert {(r['cap'],r['epsilon'],r['seed']) for r in releases}==expected_keys
    for r in releases:
        h=report['clipped_histograms'][str(r['cap'])];assert len(r['values'])==len(domain)
        assert all(math.isfinite(x) for x in r['values'])
        a=math.fsum(abs(x-y) for x,y in zip(r['values'],h))/len(domain)
        b=math.fsum(abs(x-y) for x,y in zip(r['values'],raw))/len(domain)
        max_error=max(max_error,abs(a-r['mae_clipped']),abs(b-r['mae_raw']))
        rng=np.random.default_rng(np.random.SeedSequence([187,r['cap'],int(r['epsilon']*10),r['seed']]))
        assert np.array_equal(np.array(h)+rng.laplace(0,r['cap']/r['epsilon'],len(domain)),r['values'])
    for row in report['summaries']:
        rr=[r for r in releases if (r['cap'],r['epsilon'])==(row['cap'],row['epsilon'])]
        assert len(rr)==30
        for field,target in [('mae_clipped','noise_mae'),('mae_raw','raw_mae')]:
            mean=math.fsum(r[field] for r in rr)/30
            sd=math.sqrt(math.fsum((r[field]-mean)**2 for r in rr)/29)
            assert abs(mean-row[target+'_mean'])<1e-10 and abs(sd-row[target+'_sd'])<1e-10
        assert row['scale']==row['cap']/row['epsilon']
        h=report['clipped_histograms'][str(row['cap'])]
        assert row['clipping_l1']==sum(abs(a-b) for a,b in zip(raw,h))
    assert max_error<1e-10
    assert report['hypothetical_basic_composition_epsilon']==315.
    # Tight scalar density-ratio witness: counts C versus 0 at output C.
    # log(p_C(C)/p_0(C)) = C / (C/epsilon) = epsilon.
    for cap in [1,5,20]:
        for eps in [.5,1.,2.]:assert abs(cap/(cap/eps)-eps)<1e-12
    con.close()
    return {'status':'PASS','all_tables':len(schema),'foreign_key_columns':fk_columns,'graph_edges':edges,
        'drivers_deleted':len(by_owner),'histogram_neighbors':queries,'simulation_vectors':len(releases),
        'coordinates':len(releases)*len(domain),'max_scalar_metric_error':max_error,
        'retained_driver_constructor_aggregate_links':residual_pairs,
        'privacy_proof':'Analytical under declared domain; empirical neighbors are implementation checks only',
        'production_dp':'NOT_ESTABLISHED'}


if __name__=='__main__':
    from _run_l187 import load187
    from relkit.privacy_l187 import bounded_histogram
    P=Path(__file__).resolve().parent;E=P/'evidence/l187'
    db=load187(E/'packet',json.loads((E/'input-manifest.json').read_text()))
    result=independent187(db,json.loads((E/'report.json').read_text()),pd.read_csv(E/'driver-audit.csv'),json.loads((E/'releases.json').read_text()),bounded_histogram)
    (P/'_verify_l187_results.json').write_text(json.dumps(result,indent=2)+'\n');print(result)

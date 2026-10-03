"""Independent full-grid arithmetic, availability and receipt checks."""
import bisect,hashlib,itertools,json,math,random,sqlite3,tempfile,shutil
from pathlib import Path
from relkit import serving_l186 as model
from _check_l186 import checks
from _audit_l186 import audit_receipts
P=Path(__file__).resolve().parent;E=P/'evidence/l186'

def independent(trace,policy,c):
    # Indexed prefix maxima replace the simulator's forward state machine.
    streams={}
    for arrival,entity,table,event in trace['events']:
        key=(entity,table)
        times,latest=streams.setdefault(key,([],[]))
        times.append(arrival);latest.append(max(event,latest[-1]) if latest else event)
    service=c['policies'][policy]['service_ms'];refresh=c['policies'][policy]['refresh_ms']
    previous_arrival=0;wait=0;previous_alert=False;rows=[];history=[]
    for i,(a,entity) in enumerate(zip(trace['arrivals'],trace['customers'])):
        wait=0 if i==0 else max(0,wait+service-(a-previous_arrival))
        start=a+wait;finish=start+service;previous_arrival=a
        if not refresh:read=start;material=start
        elif start<20:read=-4900;material=-4880
        else:read=((start-20)//refresh)*refresh;material=read+20
        deps=[]
        for table in range(2):
            times,latest=streams[(entity,table)];j=bisect.bisect_right(times,read)-1
            assert j>=0;deps.append(latest[j]);assert deps[-1]<=read
        age=finish-min(deps);stale=age>c['freshness_ms'];history.append(int(stale))
        # Integer arithmetic independently checks strict 10% and complete windows.
        alert=i+1>=100 and sum(history[-100:])>10
        rows.append(dict(request_id=i,customer=entity,arrival_ms=a,start_ms=start,finish_ms=finish,read_ms=read,
                         dependency_events=deps,source_age_ms=age,material_age_ms=finish-material,
                         latency_ms=finish-a,stale=stale,alert=alert))
    return rows

def main():
    r=json.loads((E/'report.json').read_text());c=json.loads((E/'config.json').read_text())
    assert c==model.CONFIG and len(r['results'])==81 and r['requests']==810000
    lookup={(x['rate'],x['condition'],x['seed'],x['policy']):x for x in r['results']};assert len(lookup)==81
    compared=0
    for rate,condition,seed in itertools.product(c['rates'],c['conditions'],c['seeds']):
        trace=model.make_trace(rate,condition,seed,c)
        assert max(trace['arrivals'])+c['drain_margin_ms']>max(trace['arrivals'][0]+150000,trace['arrivals'][-1]+15)
        for policy in c['policies']:
            actual=model.simulate(trace,policy,c);expected=independent(trace,policy,c)
            assert actual==expected,(rate,condition,seed,policy)
            saved=lookup[rate,condition,seed,policy]
            assert saved['response_sha256']==model.digest(expected)
            # Metrics reconstructed without production summarizer.
            latency=sorted(x['finish_ms']-x['arrival_ms'] for x in expected)
            for q,key in [(.5,'p50_ms'),(.95,'p95_ms'),(.99,'p99_ms')]:assert saved[key]==latency[math.ceil(q*len(latency))-1]
            assert saved['deadline_misses']==sum(x>100 for x in latency)
            assert saved['stale_responses']==sum(x['source_age_ms']>2000 for x in expected)
            assert saved['source_age_p95_ms']==sorted(x['source_age_ms'] for x in expected)[9499]
            assert saved['material_age_p95_ms']==sorted(x['material_age_ms'] for x in expected)[9499]
            assert saved['max_queue_wait_ms']==max(x['start_ms']-x['arrival_ms'] for x in expected)
            alerts=[x['finish_ms'] for x in expected if x['alert']]
            assert saved['first_alert_ms']==(min(alerts) if alerts else None)
            edges=[x['finish_ms'] for i,x in enumerate(expected) if x['alert'] and (i==0 or not expected[i-1]['alert'])]
            assert saved['alert_episodes']==len(edges)
            onset=trace['onset_ms']
            after=[] if onset is None else [t for t in alerts if t>=onset]
            assert saved['post_onset_alert_delay_ms']==(min(after)-onset if after else None)
            assert saved['labels_available_at_last_response']==bisect.bisect_right(trace['arrivals'],expected[-1]['finish_ms']-30500)
            compared+=len(expected)
    # Random FIFO conservation and exact freshness examples.
    rng=random.Random(186)
    for _ in range(100):
        arrivals=sorted(rng.randrange(1000) for _ in range(100));service=rng.randrange(1,30)
        times=model.schedule(arrivals,service)
        for i,(start,end) in enumerate(times):
            assert end-start==service and start>=arrivals[i]
            if i:assert start>=times[i-1][1]
            assert start==max(arrivals[i],times[i-1][1] if i else 0)
    original=[model.schedule,model.source_age,model.freshness_alert]
    broken=[lambda a,s:[(t,t+s) for t in a],lambda t,d:None if None in d else t-max(d),lambda x,w=100,t=.1:sum(x)/max(1,len(x))>t]
    for i,wrong in enumerate(broken):
        funcs=original[:];funcs[i]=wrong
        try:checks(*funcs)
        except AssertionError:pass
        else:raise AssertionError('Wrong learner implementation accepted')
    # SQL reconstructs the full batch grid and load-event grouping independently.
    con=sqlite3.connect(':memory:');con.execute('create table r(phase,task,arm,k,seed,seconds,load,rows,primary key(task,arm,k,seed))')
    for phase in ['pilot-1','remaining-1']:
        for x in json.loads((E/f'packet/evidence/l176/{phase}/receipt.json').read_text())['records']:
            con.execute('insert into r values(?,?,?,?,?,?,?,?)',(phase,x['database'],x['arm'],x['context'],x['seed'],x['seconds'],x['load_seconds'],x['rows']))
    replay=r['receipt_replay'];assert con.execute('select count(*),sum(rows) from r').fetchone()==(300,229050)
    assert math.isclose(con.execute('select sum(seconds) from r').fetchone()[0],replay['evaluation_seconds'],abs_tol=1e-9)
    assert math.isclose(con.execute('select sum(load) from (select max(load) load from r group by phase,task,arm)').fetchone()[0],replay['distinct_load_seconds'],abs_tol=1e-9)
    pins=json.loads((E/'input-manifest.json').read_text())
    assert audit_receipts(E,pins)==replay
    rejected=0
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);shutil.copytree(E/'packet',root/'packet')
        path=root/'packet/evidence/l176/pilot-1/receipt.json';original_bytes=path.read_bytes()
        for corrupt in ['bytes','duplicate','missing','negative']:
            local=json.loads(json.dumps(pins));receipt=json.loads(original_bytes)
            if corrupt=='bytes':path.write_bytes(original_bytes+b' ')
            else:
                if corrupt=='duplicate':receipt['records'].append(receipt['records'][0])
                if corrupt=='missing':receipt['records'].pop()
                if corrupt=='negative':receipt['records'][0]['seconds']=-1
                path.write_text(json.dumps(receipt))
                # Even a resealed local input must fail inherited authenticity.
                local['files'][path.relative_to(root).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
            try:audit_receipts(root,local)
            except ValueError:rejected+=1
            else:raise AssertionError('Corrupt receipt accepted')
            path.write_bytes(original_bytes)
    result=dict(status='PASS',independent_response_rows=compared,cells=81,random_fifo_cases=100,
                rejected_learner_functions=3,rejected_corrupt_packets=rejected,sql_receipts=300,
                verification='Independent prefix-max availability, Lindley queue recurrence, integer rolling monitor, direct quantiles, SQL timing aggregation')
    (P/'_verify_l186_results.json').write_text(json.dumps(result,indent=2)+'\n');print(result)

if __name__=='__main__':main()

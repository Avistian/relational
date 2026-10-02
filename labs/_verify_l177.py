"""Independent decimal and SQL accounting, full coverage and corrupt-packet rejection."""
import hashlib,itertools,json,math,random,shutil,sqlite3,tempfile
from decimal import Decimal as D
from pathlib import Path
from relkit.compute_l177 import reservation_total,forecast_seconds,assess_plan
from _audit_l177 import audit177
from _check_l177 import checks
P=Path(__file__).resolve().parent;E=P/'evidence/l177'
pins=json.loads((E/'input-manifest.json').read_text());r=json.loads((E/'report.json').read_text())
read=lambda n:json.loads((P/n).read_text())
def close(a,b):assert math.isclose(float(a),float(b),abs_tol=1e-9,rel_tol=1e-12),(a,b)
db=sqlite3.connect(':memory:');db.execute('create table timing(phase text, task text, arm text, k int, seed int, seconds real, load real, peak real, primary key(task,arm,k,seed))')
for phase in ['pilot-1','remaining-1']:
    for row in read(f'evidence/l176/{phase}/receipt.json')['records']:
        db.execute('insert into timing values(?,?,?,?,?,?,?,?)',(phase,row['database'],row['arm'],row['context'],row['seed'],row['seconds'],row['load_seconds'],row['peak_gpu_bytes']))
assert db.execute('select count(*) from timing').fetchone()[0]==300
assert db.execute('select min(n),max(n) from (select count(*) n from timing group by task,arm,k)').fetchone()==(10,10)
close(db.execute('select sum(seconds) from timing').fetchone()[0],r['l176']['evaluation_seconds'])
close(db.execute('select sum(load) from (select max(load) load from timing group by phase,task,arm)').fetchone()[0],r['l176']['distinct_load_seconds'])
close(db.execute('select max(peak)/1073741824.0 from timing').fetchone()[0],r['l176']['max_allocator_gib'])
rate=D('0.000222')+D(2)*D('0.0000131')+D(16)*D('0.00000222')
close(rate*(D(600+30)+D(7200+30))+D(3),r['l176']['reserved_usd'])
close(sum(D(str(read(f'evidence/l176/{p}/cost.json')['worker_body_seconds'])) for p in ['pilot-1','remaining-1'])*rate,r['l176']['worker_estimate_usd'])
close(D(3)+D(3*930)*(D(2)*D('0.0000131')+D(16)*D('0.00000222')),r['l175']['reserved_usd'])
worst=db.execute("select max(seconds) from timing where phase='pilot-1'").fetchone()[0]
close(D(str(worst))*294*3+D(str(read('evidence/l176/pilot-1/cost.json')['worker_body_seconds'])),r['l176']['forecast_seconds'])
for lesson,base,arms in [(173,'evidence/l173',['cell','task']),(174,'evidence/l174/runs',['freeze','full','adapter','scratch'])]:
    identities={(read(f'{base}/{a}-{s}/result.json')['arm'],read(f'{base}/{a}-{s}/result.json')['seed']) for a,s in itertools.product(arms,range(3))}
    assert len(identities)==r['fit_groups'][str(lesson)]['runs']
    b=read(f'evidence/l{lesson}/local-budget.json')
    close(sum(D(str(a['seconds'])) for a in b['attempts']),r['local_commands'][str(lesson)]['seconds'])
    if lesson==174:close(sum(D(str(read(f'{base}/{a}-{s}/result.json')['seconds'])) for a,s in itertools.product(arms,range(3))),r['fit_groups']['174']['fit_seconds'])
rng=random.Random(177)
for i in range(200):
    rows=[dict(id=str(j),seconds=rng.randrange(10000),lifecycle_seconds=rng.randrange(60),rate=rng.randrange(1,1000)/1e6) for j in range(rng.randrange(1,10))];overhead=rng.randrange(10)
    expected=D(overhead)+sum((D(a['seconds'])+D(a['lifecycle_seconds']))*D(str(a['rate'])) for a in rows)
    close(reservation_total(rows,overhead),expected)
    times=[rng.randrange(1,100)/10 for _ in range(6)];n=rng.randrange(300);m=rng.randrange(1,5);fixed=rng.randrange(100)
    close(forecast_seconds(times,n,m,fixed),D(str(sorted(times)[-1]))*n*m+fixed)
wrong=0
for funcs in [(lambda a,o:o,forecast_seconds,assess_plan),(reservation_total,lambda a,n,m,f:sum(a)/len(a)*n+f,assess_plan),(reservation_total,forecast_seconds,lambda *a,**k:dict(status='FEASIBLE_SCENARIO'))]:
    try:checks(*funcs)
    except (AssertionError,ValueError):wrong+=1
assert wrong==3
bad=0
for case in ['changed_bytes','duplicate_run','missing_epoch','negative_timer']:
    with tempfile.TemporaryDirectory(prefix='l177-corrupt-') as tmp:
        root=Path(tmp);p=json.loads(json.dumps(pins))
        for name in pins['files']:
            dest=root/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(P/name,dest)
        def change(name,fn,rehash=True):
            path=root/name;value=json.loads(path.read_text());fn(value);path.write_text(json.dumps(value))
            if rehash:p['files'][name]=hashlib.sha256(path.read_bytes()).hexdigest()
        if case=='changed_bytes':change('evidence/l176/cost.json',lambda v:v.update(total_reserved_usd=0),False)
        elif case=='duplicate_run':change('evidence/l176/remaining-1/receipt.json',lambda v:v['records'].__setitem__(1,v['records'][0]))
        elif case=='missing_epoch':change('evidence/l174/runs/freeze-0/result.json',lambda v:v['epochs'].pop())
        else:
            name='evidence/l176/pilot-1/rel-f1-RDBPFN-1024-0.json'
            change(name,lambda v:v.update(seconds=-1))
            change('evidence/l176/pilot-1/receipt.json',lambda v:v['records'][0].update(seconds=-1))
        try:audit177(root,p,reservation_total,forecast_seconds,assess_plan)
        except (AssertionError,ValueError):bad+=1
assert bad==4
assert audit177(P,pins,reservation_total,forecast_seconds,assess_plan)==r
result=dict(status='PASS',full_inference_records=300,full_fit_records=18,sql_accounting='PASS',decimal_accounting='PASS',randomized_reservations=200,randomized_forecasts=200,wrong_learner_functions_rejected=wrong,corrupt_packets_rejected=bad,report_parity='EXACT')
(P/'_verify_l177_results.json').write_text(json.dumps(result,indent=2)+'\n');print(result)

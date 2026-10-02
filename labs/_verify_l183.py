"""Independent SQL scoring, randomized contracts and deliberately wrong implementations."""
def sql_mae(reference,predictions):
    import math,sqlite3
    if not reference or not predictions: raise ValueError('Empty population')
    if any(not math.isfinite(float(r[2])) for r in list(reference)+list(predictions)): raise ValueError('Nonfinite')
    with sqlite3.connect(':memory:') as db:
        db.execute('create table truth(entity integer, cutoff integer, value real, primary key(entity,cutoff))')
        db.execute('create table pred(entity integer, cutoff integer, value real, primary key(entity,cutoff))')
        try:
            db.executemany('insert into truth values(?,?,?)',reference);db.executemany('insert into pred values(?,?,?)',predictions)
        except sqlite3.IntegrityError as e: raise ValueError('Duplicate key') from e
        missing=db.execute('select count(*) from (select entity,cutoff from truth except select entity,cutoff from pred)').fetchone()[0]
        extra=db.execute('select count(*) from (select entity,cutoff from pred except select entity,cutoff from truth)').fetchone()[0]
        if missing or extra: raise ValueError('Key mismatch')
        return db.execute('select avg(abs(t.value-p.value)) from truth t join pred p using(entity,cutoff)').fetchone()[0]

if __name__=='__main__':
    import copy,json,random,math
    from pathlib import Path
    from relkit.pretraining_l183 import temporal_mask,keyed_mae,factorial_effect,transfer_gate
    from _check_l183 import checks
    from _audit_l183 import audit183
    P=Path(__file__).resolve().parent;E=P/'evidence/l183';manifest=json.loads((E/'input-manifest.json').read_text())
    report=json.loads((E/'report.json').read_text());assert checks(temporal_mask,keyed_mae,factorial_effect,transfer_gate)=='PASS'
    independent=audit183(E/'packet',manifest,temporal_mask,sql_mae,factorial_effect,transfer_gate)
    for arm in ['gnn','relgt']:
        for split in ['val','test']:
            for a,b in zip(report['summary'][arm][split]['values'],independent['summary'][arm][split]['values']): assert abs(a-b)<1e-12
    rng=random.Random(183)
    for _ in range(100):
        ref=[(i//4,i%4,rng.uniform(-5,5)) for i in range(40)];pred=[(a,b,rng.uniform(-5,5)) for a,b,_ in ref];rng.shuffle(pred)
        assert abs(keyed_mae(ref,pred)-sql_mae(ref,pred))<1e-12
        cells=[];expected=[]
        for _ in range(30):
            t=rng.choice([None,5,10,11]);a=rng.choice([None,5,10,11]);label=rng.choice([True,False]);query=rng.choice([True,False]);end=15 if t is None else t+2
            cells.append(dict(event_time=t,available_at=a,is_label=label,label_end=end,query_target=query))
            expected.append(False if query or t is None or a is None or t>10 or a>10 else (not label or end<=10))
        assert temporal_mask(cells,10)==expected
    originals=[temporal_mask,keyed_mae,factorial_effect,transfer_gate];rejected=0
    mutants=[lambda cells,cutoff:[True]*len(cells),lambda ref,pred:sum(abs(a[2]-b[2]) for a,b in zip(ref,pred))/len(ref),lambda arms,higher_is_better=False:dict(interaction_mean=0),lambda e:dict(status='READY_FOR_SEPARATELY_AUTHORIZED_PROBE',blockers=[],transfer_gain='ESTABLISHED')]
    for i,mutant in enumerate(mutants):
        fns=originals.copy();fns[i]=mutant
        try:checks(*fns)
        except (AssertionError,KeyError):rejected+=1
        else:raise AssertionError('Learner mutant accepted')
    corrupted=copy.deepcopy(manifest);corrupted['files'][next(iter(corrupted['files']))]='0'*64
    try:audit183(E/'packet',corrupted,*originals)
    except ValueError:pass
    else:raise AssertionError('Bad digest accepted')
    result=dict(status='PASS',predictions_independently_scored=7554,oracle='SQLite composite-primary-key join and AVG absolute error',random_keyed_cases=100,random_cell_cases=3000,rejected_learner_mutants=rejected,tampered_input_rejected=True,source_authentication='Git HEAD plus original model/trainer hashes',fresh_training='NOT_RUN',learner='PENDING_WRITTEN_DEFENSE')
    (P/'_verify_l183_results.json').write_text(json.dumps(result,indent=2)+'\n');print(result)

"""Reconstruct every approved accounting receipt with live learner functions."""
def audit177(root, pins, reservation_total, forecast_seconds, assess_plan):
    import hashlib,itertools,json,math
    from pathlib import Path
    root=Path(root)
    for name,digest in pins['files'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,'Changed input: '+name
    def read(name):return json.loads((root/name).read_text())
    def close(a,b):assert math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-10),(a,b)
    def positive(v):assert isinstance(v,(float,int)) and not isinstance(v,bool) and math.isfinite(v) and v>=0
    local={}
    for lesson in [173,174,175,176]:
        b=read(f'evidence/l{lesson}/local-budget.json');assert not b.get('active')
        for a in b['attempts']:positive(a['seconds'])
        local[str(lesson)]=dict(attempts=len(b['attempts']),seconds=math.fsum(a['seconds'] for a in b['attempts']),
           non_pass_attempts=sum(a['status']!='PASS' for a in b['attempts']),
           scope='Enclosing local commands including failures and validations; do not add nested fit timers')
    fit_groups={}
    for lesson,arms,epochs,directory,reportname in [(173,['cell','task'],3,'evidence/l173','evidence/l173/report.json'),
                  (174,['freeze','full','adapter','scratch'],10,'evidence/l174/runs','evidence/l174/runs/report.json')]:
        report=read(reportname);runs=[];seen=set()
        for arm,seed in itertools.product(arms,range(3)):
            row=read(f'{directory}/{arm}-{seed}/result.json')
            assert (row['arm'],row['seed'])==(arm,seed)
            assert [x['epoch'] for x in row['epochs']]==list(range(1,epochs+1))
            assert 1<=row['selected_epoch']<=epochs
            assert row==next(r for r in report['runs'] if (r['arm'],r['seed'])==(arm,seed))
            if 'seconds' in row:positive(row['seconds'])
            seen.add((arm,seed));runs.append(row)
        assert len(report['runs'])==len(seen)==len(arms)*3
        command=next(a for a in read(f'evidence/l{lesson}/local-budget.json')['attempts'] if any(f'_run_l{lesson}.py' in str(c) for c in a['command']) and a['status']=='PASS')
        seconds=report['seconds'] if lesson==173 else math.fsum(r['seconds'] for r in runs)
        assert seconds<=command['seconds']
        fit_groups[str(lesson)]=dict(runs=len(runs),epochs_per_run=epochs,fit_seconds=seconds,command_seconds=command['seconds'],
            per_fit_seconds=[dict(arm=r['arm'],seed=r['seed'],seconds=r.get('seconds')) for r in runs],
            peak_memory_gib=None,active_researcher_seconds=None,cloud_spend_usd=0,
            scope='Six-fit group timer' if lesson==173 else 'Sum of twelve fit timers including evaluation and serialization')
    rates=read('sources/l177/rates.json');rate=rates['L4']+2*rates['physical_cpu']+16*rates['memory_gib']
    records=[];loads={};phases={}
    for phase,n in [('pilot-1',6),('remaining-1',294)]:
        receipt=read(f'evidence/l176/{phase}/receipt.json');assert receipt['device']=='cuda'
        assert len(receipt['records'])==n
        assert receipt['input_manifest_sha256']==hashlib.sha256((root/'evidence/l176/input-manifest.json').read_bytes()).hexdigest()
        c=read(f'evidence/l176/{phase}/cost.json');positive(c['worker_body_seconds']);close(c['rate'],rate)
        subtotal=0
        for r in receipt['records']:
            assert r==read(f"evidence/l176/{phase}/{Path(r['filename']).stem}.json")
            assert r['rows']==({'rel-f1':702,'rel-trial':825}[r['database']])
            assert 0<=r['auc']<=1
            for k in ['seconds','load_seconds','peak_gpu_bytes']:positive(r[k])
            assert (r['context']==1024 and r['seed']==0)==(phase=='pilot-1')
            group=(phase,r['database'],r['arm'])
            if group in loads:assert loads[group]==r['load_seconds']
            loads[group]=r['load_seconds'];subtotal+=r['seconds'];records.append(r)
        phase_load=math.fsum(v for k,v in loads.items() if k[0]==phase)
        assert subtotal+phase_load<=c['worker_body_seconds']
        phases[phase]=dict(runs=n,worker_seconds=c['worker_body_seconds'],evaluation_seconds=subtotal,distinct_load_seconds=phase_load,
                          residual_seconds=c['worker_body_seconds']-subtotal-phase_load)
    keys=[(r['database'],r['arm'],r['context'],r['seed']) for r in records]
    expected=set(itertools.product(['rel-f1','rel-trial'],['RDBPFN','RDBPFN_single','TabICLv1.1'],[64,128,256,512,1024],range(10)))
    assert len(keys)==len(set(keys))==300 and set(keys)==expected
    b=read('evidence/l176/budget.json')
    assert [(a['attempt'],a['seconds'],a['lifecycle_seconds']) for a in b['reservations']]==[('pilot-1',600,30),('remaining-1',7200,30)]
    attempts=[dict(id=a['attempt'],seconds=a['seconds'],lifecycle_seconds=a['lifecycle_seconds'],rate=rate) for a in b['reservations']]
    for a,original in zip(attempts,b['reservations']):close(reservation_total([a],0),original['upper_usd'])
    reserved=reservation_total(attempts,b['overhead_usd']);close(reserved,b['reserved_usd'])
    costs=read('evidence/l176/cost.json');worker=math.fsum(p['worker_seconds'] for p in phases.values())
    close(worker,costs['worker_body_seconds']);close(worker*rate,costs['worker_body_estimate_usd']);close(reserved,costs['total_reserved_usd'])
    pilot=[r['seconds'] for r in read('evidence/l176/pilot-1/receipt.json')['records']]
    projection=forecast_seconds(pilot,294,3,phases['pilot-1']['worker_seconds'])
    decision=read('evidence/l176/cost-decision.json');close(projection,decision['remaining_projection_seconds_with_3x_margin'])
    close(max(pilot),decision['worst_per_run_seconds']);assert projection<=7200 and reserved<=8 and decision['decision']=='PROCEED'
    old=read('evidence/l175/cloud-budget.json');cpu_rate=2*rates['physical_cpu']+16*rates['memory_gib']
    assert [a['phase'] for a in old['reservations']]==['audit-1','audit-2','audit-3']
    old_attempts=[dict(id=a['phase'],seconds=a['timeout'],lifecycle_seconds=30,rate=a['rate']) for a in old['reservations']]
    for a,original in zip(old_attempts,old['reservations']):
        close(a['rate'],cpu_rate);close(reservation_total([a],0),original['upper_usd'])
    old_total=reservation_total(old_attempts,old['overhead_reserve_usd']+old['build_reservation_usd'])
    oldcost=read('evidence/l175/cost-summary.json');close(old_total,oldcost['reserved_upper_usd'])
    assert oldcost['gpu_inference']=='NOT_RUN' and oldcost['model_reproduction']=='INCOMPLETE_TEMPORAL_GATE'
    for phase,key in [('audit-2','failed_sampler_worker_seconds'),('audit-3','successful_sampler_worker_seconds')]:
        close(read(f'evidence/l175/{phase}/cost.json')['worker_seconds'],oldcost[key])
    curves=[]
    for db,arm,k in itertools.product(['rel-f1','rel-trial'],['RDBPFN','RDBPFN_single','TabICLv1.1'],[64,128,256,512,1024]):
        subset=[r for r in records if (r['database'],r['arm'],r['context'])==(db,arm,k)]
        curves.append(dict(database=db,arm=arm,context=k,mean_seconds=math.fsum(r['seconds'] for r in subset)/10,
                           max_allocator_gib=max(r['peak_gpu_bytes'] for r in subset)/2**30))
    rt={}
    for label,hours in [('pretraining',rates['rt_pretrain_hours']),('fine_tuning',rates['rt_finetune_hours'])]:
        rt[label]=dict(reported_elapsed_hours=hours,gpu_hours=hours*8,
                      gpu_only_usd_40gb=hours*8*3600*rates['A100_40'],gpu_only_usd_80gb=hours*8*3600*rates['A100_80'],
                      status='OVER_BUDGET',basis='Hypothetical posted A100 rental scenario; reported approximate time, not our measurement')
    scenarios={}
    for minutes in [60,180]:
        scenarios[str(minutes)]=dict(
          observed_gaps=assess_plan(reserved,projection+120,None,None,24,minutes*60),
          explicit_assumptions=assess_plan(reserved,projection+120,1200,16,24,minutes*60),
          stopped_rt=assess_plan(old_total,930,1200,8,24,minutes*60,False))
    return dict(experiment=pins['experiment'],status='COMPLETE_SELECTED_ACCOUNTING_REPLAY',authenticated_inputs=len(pins['files']),
      local_commands=local,fit_groups=fit_groups,
      l176=dict(runs=300,query_predictions=sum(r['rows'] for r in records),phases=phases,distinct_load_events=len(loads),
        evaluation_seconds=math.fsum(r['seconds'] for r in records),distinct_load_seconds=math.fsum(loads.values()),
        naive_repeated_load_seconds=math.fsum(r['load_seconds'] for r in records),worker_seconds=worker,worker_gpu_hours=worker/3600,
        rate=rate,worker_estimate_usd=worker*rate,reserved_usd=reserved,invoice=costs['invoice'],
        pilot_worst_seconds=max(pilot),forecast_seconds=projection,forecast_method='294 * slowest six-run pilot * 3 + pilot worker duration; heuristic, not confidence bound',
        max_allocator_gib=max(r['peak_gpu_bytes'] for r in records)/2**30,total_device_memory_required_gib=None,active_researcher_seconds=None),
      l175=dict(attempts=3,reserved_usd=old_total,observed_worker_seconds=oldcost['failed_sampler_worker_seconds']+oldcost['successful_sampler_worker_seconds'],
        missing_worker_attempts=['audit-1'],invoice=oldcost['invoice'],model_reproduction=oldcost['model_reproduction']),
      timing_by_task_model_context=curves,rt_price_scenarios=rt,session_scenarios=scenarios,
      boundaries=dict(new_cloud_usd=0,fresh_training='NOT_RUN',fresh_inference='NOT_RUN',whole_paper='NOT_RUN',learner='PENDING_WRITTEN_DEFENSE',
        human_time='NOT_MEASURED',hardware_speed_comparison='NOT_ESTABLISHED',historical_lineage='NOT_ESTABLISHED',invoice_reconciliation='NOT_CHECKED'))

if __name__=='__main__':
    import json
    from pathlib import Path
    from relkit.compute_l177 import reservation_total,forecast_seconds,assess_plan
    P=Path(__file__).resolve().parent;E=P/'evidence/l177'
    r=audit177(P,json.loads((E/'input-manifest.json').read_text()),reservation_total,forecast_seconds,assess_plan)
    (E/'report.json').write_text(json.dumps(r,indent=2)+'\n')
    print(r['status'],r['l176'])

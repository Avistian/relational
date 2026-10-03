"""All 300 original batch receipts; never treat them as request latency samples."""
def audit_receipts(root, pins):
    import hashlib,itertools,json,math
    def read(name):return json.loads((root/'packet'/name).read_text())
    for name,h in pins['files'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=h:
            raise ValueError('Input hash mismatch: '+name)
    original=read('evidence/l176/artifact-manifest.json')['files']
    for name,h in pins['files'].items():
        original_name='labs/'+name.removeprefix('packet/')
        if original_name in original and original[original_name]!=h:
            raise ValueError('Inherited seal mismatch')
    keys=set();phases={};batch_rows=0;all_records=[]
    for phase in ['pilot-1','remaining-1']:
        receipt=read(f'evidence/l176/{phase}/receipt.json');loads={};seconds=[]
        for row in receipt['records']:
            key=(row['database'],row['arm'],row['context'],row['seed'])
            if key in keys:raise ValueError('Duplicate experiment cell')
            keys.add(key)
            if row['rows'] != {'rel-f1':702,'rel-trial':825}[row['database']]:raise ValueError('Batch size mismatch')
            for field in ['seconds','load_seconds','peak_gpu_bytes']:
                if not math.isfinite(row[field]) or row[field]<0:raise ValueError('Bad measurement')
            single=read(f'evidence/l176/{phase}/'+row['filename'].replace('.npz','.json'))
            if single!=row:raise ValueError('Phase/per-run receipt mismatch')
            load_key=(row['database'],row['arm'])
            if load_key in loads and loads[load_key]!=row['load_seconds']:raise ValueError('Load event mismatch')
            loads[load_key]=row['load_seconds'];seconds.append(row['seconds']);batch_rows+=row['rows']
            all_records.append(dict(phase=phase,**row))
        worker=read(f'evidence/l176/{phase}/cost.json')['worker_body_seconds']
        inner=math.fsum(seconds);load=math.fsum(loads.values())
        if inner+load>worker:raise ValueError('Nested timer violation')
        phases[phase]=dict(runs=len(seconds),worker_seconds=worker,evaluation_seconds=inner,distinct_load_seconds=load,load_events=len(loads),residual_seconds=worker-inner-load)
    expected=set(itertools.product(['rel-f1','rel-trial'],['RDBPFN','RDBPFN_single','TabICLv1.1'],[64,128,256,512,1024],range(10)))
    if keys!=expected:raise ValueError('Incomplete grid')
    return dict(status='COMPLETE_BATCH_RECEIPT_REPLAY',runs=len(keys),batch_prediction_rows=batch_rows,
                phases=phases,worker_seconds=math.fsum(x['worker_seconds'] for x in phases.values()),
                evaluation_seconds=math.fsum(x['evaluation_seconds'] for x in phases.values()),
                distinct_load_seconds=math.fsum(x['distinct_load_seconds'] for x in phases.values()),
                load_events=sum(x['load_events'] for x in phases.values()),
                request_latency='NOT_MEASURED',prediction_values_replayed=False,
                provenance='L176 original sealed JSON receipts; hashes authenticate bytes, not physical clock accuracy')

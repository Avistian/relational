"""Apply the frozen validation nomination; independently score every query and slice."""
import datetime,hashlib,json,math,sqlite3,statistics
from pathlib import Path
import numpy as np
from relkit.error_reg_l149 import paired_errors,cluster_interval
P=Path(__file__).resolve().parent;E=P/'evidence/l149';frozen=json.loads((E/'frozen.json').read_text())
for name,sha in frozen['files'].items():assert hashlib.sha256((P.parent/name).read_bytes()).hexdigest()==sha
assert frozen['analysis_source_sha256']==hashlib.sha256((P/'relkit/error_reg_l149.py').read_bytes()).hexdigest()
assert frozen['diagnostics_sha256']==hashlib.sha256((E/'diagnostics.npz').read_bytes()).hexdigest()
a=np.load(E/'diagnostics.npz');output={};portable={};independent_rows=0
for split in ['val','test']:
    keys=list(zip(a[split+'_entity'],a[split+'_time']));deltas=[];gp=[];fp=[];scores=[]
    for seed in range(5):
        g=np.load(E/f'final/lr005-full/seed-{seed}/predictions.npz');f=np.load(E/f'fe/paper/seed-{seed}/predictions.npz');y=f[split+'_target']
        np.testing.assert_allclose(g[split+'_target'],y,rtol=0,atol=1e-6)
        for obj in [g,f]:
            assert list(zip(obj[split+'_entity'],obj[split+'_time']))==keys
        d=paired_errors(keys,y,keys,g[split+'_pred'],keys,f[split+'_pred'])
        # Independent SQL losses use archive float64 targets for both pipelines.
        con=sqlite3.connect(':memory:');con.execute('CREATE TABLE q (y REAL,g REAL,f REAL)');con.executemany('INSERT INTO q VALUES (?,?,?)',list(zip(y.tolist(),g[split+'_pred'].tolist(),f[split+'_pred'].tolist())))
        sql=con.execute('SELECT avg(abs(y-g)),avg(abs(y-f)),avg(abs(y-g)-abs(y-f)) FROM q').fetchone();con.close()
        assert abs(d.mean()-sql[2])<1e-12;independent_rows+=len(y)
        deltas.append(d);gp.append(g[split+'_pred']);fp.append(f[split+'_pred']);scores.append(dict(seed=seed,gnn=sql[0],fe=sql[1],difference=sql[2]))
    mean_delta=np.mean(deltas,axis=0);masks={'all':np.ones(len(y),dtype=bool),**{k[len(split+'_mask_'):]:a[k] for k in a.files if k.startswith(split+'_mask_')}};slices={}
    for name,mask in masks.items():
        count=int(mask.sum());drivers=len(np.unique(a[split+'_entity'][mask]));supported=count>=30 and drivers>=10
        r=dict(rows=count,drivers=drivers,supported=supported,mean=float(mean_delta[mask].mean()) if count else None,interval=None)
        if supported:r['interval']=cluster_interval(mean_delta[mask],a[split+'_entity'][mask])
        slices[name]=r
    selected=frozen['selected'];selected_row=slices[selected] if selected else None
    output[split]=dict(seeds=scores,slices=slices,gnn_mean=statistics.mean(r['gnn'] for r in scores),gnn_sd=statistics.stdev(r['gnn'] for r in scores),fe_mean=statistics.mean(r['fe'] for r in scores),fe_sd=statistics.stdev(r['fe'] for r in scores),selected_slice=selected,selected=selected_row)
    portable[split]=dict(keys=[[int(i),int(t)] for i,t in keys],target=y.tolist(),gnn=np.array(gp).tolist(),fe=np.array(fp).tolist(),masks={k:v.tolist() for k,v in masks.items()},history=a[split+'_history'].tolist(),recency=[None if not np.isfinite(v) else float(v) for v in a[split+'_recency']],missing=a[split+'_missing'].tolist())
summary=dict(status='COMPLETE_PAIRED_PIPELINE_ANALYSIS',splits=output,paired_query_seed_rows=independent_rows,individual_predictions=2*independent_rows,frozen_sha256=hashlib.sha256((E/'frozen.json').read_bytes()).hexdigest(),utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),claim='Original course error-analysis extension; not a paper slice result or architecture-only causal effect',uncertainty='2000 driver-cluster percentile bootstrap draws of per-query mean losses across five fitted runs; fixed splits/models; not seed or cross-database uncertainty',historical_test_reuse=True)
(E/'errors.json').write_text(json.dumps(summary,indent=2));(E/'portable.json').write_text(json.dumps(portable,separators=(',',':')))
print(json.dumps(summary,indent=2))

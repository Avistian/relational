"""Collect every fresh search and independently replay every tree and MAE."""
import hashlib,json,math,sqlite3,statistics,time
from pathlib import Path
import numpy as np,pandas as pd,lightgbm
P=Path(__file__).resolve().parent;O=P/'evidence/l129';a=np.load(O/'matrices.npz')

def tree_predict(node,x,rows):
    if 'leaf_value' in node:return np.full(len(rows),node['leaf_value'])
    v=x[rows,node['split_feature']];missing=np.isnan(v)
    if node['decision_type']=='==':
        missing|=v<0
        left=np.isin(v,[int(t) for t in node['threshold'].split('||')])
    else:
        if node['missing_type']=='Zero':missing|=v==0
        left=v<=float(node['threshold'])
    left[missing]=node['default_left'];out=np.empty(len(rows))
    if left.any():out[left]=tree_predict(node['left_child'],x,rows[left])
    if (~left).any():out[~left]=tree_predict(node['right_child'],x,rows[~left])
    return out

rows=[];prediction_count=0;max_tree_error=0;max_label_error=0;seconds=0;trace_example=None
feature_names=json.loads((O/'preparation.json').read_text())['feature_names'];names=feature_names['categorical']+feature_names['numerical']
example=json.loads((O/'sql-audit.json').read_text())['example']
for seed in range(5):
    folder=O/f'paper/seed-{seed}';r=json.loads((folder/'result.json').read_text())
    assert r['seed']==seed and r['trials']==10 and r['rounds_cap']==2000
    assert [t['number'] for t in r['trace']]==list(range(10))
    assert min(r['trace'],key=lambda t:(t['val_mae'],t['number']))['number']==r['selected_trial']
    assert r['matrix_sha256']==hashlib.sha256((O/'matrices.npz').read_bytes()).hexdigest()
    for path,h in r['source_sha256'].items():assert hashlib.sha256((P/path).read_bytes()).hexdigest()==h
    for path,h in r['files'].items():assert hashlib.sha256((folder/path).read_bytes()).hexdigest()==h
    pred=np.load(folder/'predictions.npz');rdl=np.load(P/f'evidence/l127/paper/seed-{seed}/predictions.npz')
    model=lightgbm.Booster(model_file=str(folder/'model.txt'));dump=model.dump_model()
    for split in ['val','test']:
        x=a[split+'_x'];raw=sum(tree_predict(t['tree_structure'],x,np.arange(len(x))) for t in dump['tree_info'])
        error=float(np.max(np.abs(raw-model.predict(x))));assert error<1e-10;max_tree_error=max(max_tree_error,error)
        lookup={(int(i),int(t)):float(p) for i,t,p in zip(a[split+'_feature_id'],a[split+'_feature_time'],raw)}
        independent=np.array([lookup[(int(i),int(t))] for i,t in zip(pred[split+'_entity'],pred[split+'_time'])])
        np.testing.assert_allclose(independent,pred[split+'_pred'],rtol=0,atol=1e-10)
        con=sqlite3.connect(':memory:');con.execute('create table predictions(y real,p real)');con.executemany('insert into predictions values (?,?)',list(zip(pred[split+'_target'].tolist(),independent.tolist())))
        score=con.execute('select avg(abs(y-p)) from predictions').fetchone()[0];con.close();assert abs(score-r['scores'][split])<1e-12
        np.testing.assert_array_equal(pred[split+'_entity'],rdl[split+'_entity']);np.testing.assert_array_equal(pred[split+'_time'],rdl[split+'_time'])
        le=float(np.max(abs(pred[split+'_target']-rdl[split+'_target'])));assert le<1e-6;max_label_error=max(max_label_error,le)
        prediction_count+=len(independent)
        if seed==0 and split=='val':
            idx=np.flatnonzero((a['val_feature_id']==example['entity'])&(a['val_feature_time']==pd.Timestamp(example['cutoff']).value))[0]
            path=[];node=dump['tree_info'][0]['tree_structure']
            while 'leaf_value' not in node:
                value=x[idx,node['split_feature']];missing=math.isnan(value)
                if node['decision_type']=='==':missing=missing or value<0;left=value in [int(v) for v in node['threshold'].split('||')]
                else:
                    missing=missing or (node['missing_type']=='Zero' and value==0);left=value<=float(node['threshold'])
                if missing:left=node['default_left']
                path.append(dict(feature=names[node['split_feature']],value=None if math.isnan(value) else float(value),decision=node['decision_type'],threshold=node['threshold'],branch='left' if left else 'right',missing=missing));node=node['left_child' if left else 'right_child']
            contributions=[float(tree_predict(t['tree_structure'],x,np.array([idx]))[0]) for t in dump['tree_info']]
            trace_example=dict(entity=example['entity'],cutoff=example['cutoff'],first_tree_path=path,first_tree_leaf=node['leaf_value'],first_ten_sum=sum(contributions[:10]),remaining_sum=sum(contributions[10:]),tree_count=len(contributions),prediction=sum(contributions),target=example['features']['position'])
    rows.append(dict(seed=seed,selected_trial=r['selected_trial'],trees=r['best_iteration'],**r['scores']));seconds+=r['seconds']
metrics={s:dict(mean=statistics.mean(r[s] for r in rows),sample_sd=statistics.stdev(r[s] for r in rows)) for s in ['val','test']}
rdl=json.loads((P/'evidence/l127/summary.json').read_text())
summary=dict(status='COMPLETE_RELEASED_PIPELINE_REPLAY',experiment='RelBench manual FE user study, F1 driver-position, original SQL and ten-trial LightGBM search',seeds=rows,metrics=metrics,training_seconds=seconds,cloud_usd=0,predictions=prediction_count,independent_tree_error=max_tree_error,independent_scoring='SQLite AVG(ABS(y-p)), all predictions',comparison={s:dict(rdl_mean=rdl['metrics'][s]['mean'],rdl_sample_sd=rdl['metrics'][s]['sample_sd'],fe_minus_rdl=metrics[s]['mean']-rdl['metrics'][s]['mean']) for s in metrics},comparison_keys='EXACT',maximum_target_float32_difference=max_label_error,historical_identity='NOT_ESTABLISHED',paper_score_parity='NOT_ESTABLISHED: Figure3 reports normalized bars; exact task scalar and historical models/seeds not released here',human_effort_reproduction='NOT_RUN: automated replay does not repeat human study',learner_status='PENDING_WRITTEN_DEFENSE',protocol_differences=['SQL row order frozen to archived query order; source unspecified','Explicit TPE seeds0..4; original default unseeded','Four CPU threads','Historical staging unavailable; checksum-pinned v1 archives used','L127 is basic RDL; Figure3 regression may use boosted RDL head; do not equate comparators','FE train-fitted preprocessing versus L127 test-cap-fitted encoders','FE schedule availability unverified'])
(O/'summary.json').write_text(json.dumps(summary,indent=2));(O/'prediction-trace.json').write_text(json.dumps(trace_example,indent=2));print(json.dumps(summary,indent=2))

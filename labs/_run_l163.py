"""Fit all heads on training/validation, freeze a receipt, then evaluate test rows."""
def fit_heads(packet, arrays, typed, choose):
    import numpy as np
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    rows=packet['rows'];y=np.asarray([r['target'] for r in rows]);frozen=[]
    for split in packet['splits']:
        tr=split['train'];va=split['validation']
        typed_x,meta=typed([rows[i] for i in tr],rows)
        for method,x in [('typed',typed_x),('bart',arrays['baseline'])]:
            x=np.asarray(x,dtype=np.float64)  # stable head replay; encoder cache remains float32
            scaler=StandardScaler().fit(x[tr]);train=scaler.transform(x[tr]);valid=scaler.transform(x[va])
            candidates=[Ridge(alpha=a,solver='svd').fit(train,y[tr]) for a in packet['alphas']]
            errors=[float(np.mean(np.abs(y[va]-model.predict(valid)))) for model in candidates]
            alpha=choose(packet['alphas'],errors);model=candidates[packet['alphas'].index(alpha)]
            frozen.append(dict(seed=split['seed'],method=method,alpha=alpha,validation_mae=errors,
              feature_mean=scaler.mean_.tolist(),feature_scale=scaler.scale_.tolist(),coef=model.coef_.tolist(),intercept=float(model.intercept_),
              typed_metadata=meta if method=='typed' else None,train_mean_target=float(y[tr].mean())))
    return frozen


def evaluate_heads(packet, arrays, heads, typed):
    import numpy as np
    rows=packet['rows'];y=np.asarray([r['target'] for r in rows]);results=[];predictions=[]
    for head in heads:
        split=next(s for s in packet['splits'] if s['seed']==head['seed']);test=split['test'];train=split['train']
        typed_x,_=typed([rows[i] for i in train],rows);baseline_pred=None
        for variant in packet['variants']:
            # Canonical field identities are retained for the typed representation.
            x=typed_x if head['method']=='typed' else arrays[variant]
            pred=((x[test]-np.asarray(head['feature_mean']))/np.asarray(head['feature_scale']))@np.asarray(head['coef'])+head['intercept']
            if variant=='baseline':baseline_pred=pred.copy()
            mae=float(np.abs(pred-y[test]).mean())
            results.append(dict(seed=head['seed'],method=head['method'],variant=variant,alpha=head['alpha'],test_mae=mae,
                train_mean_baseline_mae=float(np.abs(y[test]-head['train_mean_target']).mean()),prediction_change=float(np.abs(pred-baseline_pred).mean())))
            for i,estimate in zip(test,pred):predictions.append(dict(seed=head['seed'],method=head['method'],variant=variant,id=rows[i]['id'],target=float(y[i]),prediction=float(estimate),absolute_error=float(abs(estimate-y[i]))))
    summary=[]
    for method in ['typed','bart']:
        for variant in packet['variants']:
            r=[v for v in results if v['method']==method and v['variant']==variant]
            summary.append(dict(method=method,variant=variant,mean_mae=float(np.mean([v['test_mae'] for v in r])),sample_sd=float(np.std([v['test_mae'] for v in r],ddof=1)),mean_prediction_change=float(np.mean([v['prediction_change'] for v in r]))))
    return dict(experiment=packet['experiment'],status='COMPLETE',scope=packet['scope'],results=results,summary=summary,predictions=predictions,
      row_count=len(rows),split_seeds=[s['seed'] for s in packet['splits']],historical_reproduction='NOT_RUN',historical_fidelity='NOT_ESTABLISHED',general_encoder_superiority='NOT_ESTABLISHED',learner='PENDING_WRITTEN_DEFENSE',cloud_spend_usd=0)


def render_report(report):
    lines=['# L163 Frozen Row Encoder Comparison','', 'Executed synthetic regression task. MAE in synthetic target units; lower is better.','', '| Encoder | Input variant | Mean MAE | Split SD | Mean prediction change |','|---|---|---:|---:|---:|']
    for r in report['summary']:lines.append(f"| {r['method']} | {r['variant']} | {r['mean_mae']:.6f} | {r['sample_sd']:.6f} | {r['mean_prediction_change']:.6f} |")
    lines+=['','| Split | Encoder | Input | Alpha | Test MAE | Train-mean baseline MAE |','|---:|---|---|---:|---:|---:|']
    for r in report['results']:lines.append(f"| {r['seed']} | {r['method']} | {r['variant']} | {r['alpha']} | {r['test_mae']:.6f} | {r['train_mean_baseline_mae']:.6f} |")
    lines+=['','Three overlapping splits are not independent datasets; SD is descriptive, not a confidence interval. The target is explicitly numeric/additive, favoring the typed linear path. No LM fine-tuning, natural text semantics, missing-data comparison, GNN or database transfer is tested.','', 'All 864 keyed predictions are in report.json. Six heads were selected on validation before test evaluation. Reordered and renamed inputs use unchanged baseline heads. Historical reproduction NOT_RUN; fidelity and general superiority NOT_ESTABLISHED. Learner PENDING_WRITTEN_DEFENSE.']
    return '\n'.join(lines)+'\n'


if __name__=='__main__':
    import os
    os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    import hashlib,json,time
    from pathlib import Path
    import numpy as np
    from relkit.rows_l163 import typed_features,choose_alpha
    p=Path(__file__).resolve().parent;e=p/'evidence/l163';start=time.perf_counter()
    raw=(e/'fixtures.json').read_bytes();packet=json.loads(raw);receipt=json.loads((e/'encoding-receipt.json').read_text())
    assert hashlib.sha256(raw).hexdigest()==receipt['fixtures_sha256']
    assert hashlib.sha256((e/'embeddings.npz').read_bytes()).hexdigest()==receipt['embeddings_sha256']
    for name,expected in receipt['source_sha256'].items():assert hashlib.sha256((p/name).read_bytes()).hexdigest()==expected,'Changed encoding helper; regenerate embeddings explicitly'
    arrays=dict(np.load(e/'embeddings.npz'))
    heads=fit_heads(packet,arrays,typed_features,choose_alpha)
    frozen=dict(status='VALIDATION_SELECTED_BEFORE_TEST',heads=heads,fixtures_sha256=receipt['fixtures_sha256'],embeddings_sha256=receipt['embeddings_sha256'])
    (e/'selection.json').write_text(json.dumps(frozen,indent=2)+'\n')
    selection_hash=hashlib.sha256((e/'selection.json').read_bytes()).hexdigest()
    report=evaluate_heads(packet,arrays,heads,typed_features)
    (e/'report.json').write_text(json.dumps(report,indent=2)+'\n');(e/'report.md').write_text(render_report(report))
    traces=json.loads((e/'token-traces.json').read_text());sensitivity=[]
    for variant,x in arrays.items():
        b=arrays['baseline'];cos=(x*b).sum(1)/(np.linalg.norm(x,axis=1)*np.linalg.norm(b,axis=1));lengths=[r['length'] for r in traces[variant]]
        sensitivity.append(dict(variant=variant,min_tokens=min(lengths),mean_tokens=float(np.mean(lengths)),max_tokens=max(lengths),mean_cosine_distance=float(np.mean(1-np.clip(cos,-1,1))),max_cosine_distance=float(np.max(1-np.clip(cos,-1,1)))))
    (e/'sensitivity.json').write_text(json.dumps(sensitivity,indent=2)+'\n')
    (e/'head-receipt.json').write_text(json.dumps(dict(status='PASS',seconds=time.perf_counter()-start,selection_sha256=selection_hash,report_sha256=hashlib.sha256((e/'report.json').read_bytes()).hexdigest(),head_count=len(heads),prediction_count=len(report['predictions'])),indent=2)+'\n')
    print(render_report(report))

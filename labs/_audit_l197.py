"""Complete audit, parameterized so visible notebook implementations do the work."""
def audit197(root, tables_replay, predictions_replay, pair, verdict, scope, auc, interval, coverage, admit):
    import hashlib,json
    from pathlib import Path
    e=Path(root);q=e/'packet';manifest=json.loads((e/'input-manifest.json').read_text())
    actual={str(p.relative_to(q)) for p in q.rglob('*') if p.is_file() and '__pycache__' not in str(p)}
    if actual!=set(manifest['files']):raise ValueError('Incomplete or extra frozen inputs')
    for name,digest in manifest['files'].items():
        path=(q/name).resolve()
        if not path.is_relative_to(q.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            raise ValueError('Frozen input differs: '+name)
        if manifest['origins'][name]['sha256']!=digest:raise ValueError('Origin chain differs')
    read=lambda name:json.loads((q/name).read_text())
    t=tables_replay(q/'evidence/l191/packet',read('evidence/l191/input-manifest.json'))
    p=predictions_replay(q/'evidence/l195/packet',read('evidence/l195/input-manifest.json'),pair,verdict,scope,auc,interval)
    if t!=read('evidence/l191/report.json') or p!=read('evidence/l195/report.json'):
        raise ValueError('Full recomputed report differs from frozen predecessor')
    inventory=p['rdblearn']['tasks']
    cov=coverage([r['task'] for r in inventory],[dict(task=r['task'],status=r['status'],score=r['fresh_mean']) for r in inventory])
    if cov['declared']!=21 or cov['measured']!=0:raise ValueError('Fresh result scope changed')
    return dict(experiment='L197-LANDSCAPE-EVIDENCE-AUDIT',status='COMPLETE_SELECTED_EVIDENCE_AUDIT',
        authenticated_files=len(actual),tables=t,predictions=p,coverage=cov,
        admissions={lane:{claim:admit(lane,claim,True,True) for claim in ['published_comparison','scoped_pipeline_comparison','implementation_observation','fresh_model_reproduction','architecture_cause','economic_undervaluation']} for lane in ['published_table','saved_predictions','source_diagnostic']},
        notes=['Admission grid is a teaching policy; source_diagnostic is a hypothetical lane, not a newly executed L196 diagnostic.',
               'No pooling AUROC, MAE, or normalized MAE across suites.',
               'Complete replay reuses prior evidence and is not an independent replication.'],
        full_model_reproduction='INCOMPLETE_SOURCE_PREPROCESSING_GATE',fresh_training='NOT_RUN',
        historical_identity='NOT_ESTABLISHED',economic_undervaluation='NOT_ESTABLISHED',learner='PENDING_WRITTEN_DEFENSE',cloud_usd=0)

if __name__=='__main__':
    import json,sys
    from pathlib import Path
    e=Path(__file__).resolve().parent/'evidence/l197'
    # Import only the frozen implementations; prevent writes of __pycache__ in packet.
    sys.dont_write_bytecode=True
    import relkit
    relkit.__path__.insert(0,str(e/'packet/relkit'))
    sys.path.insert(0,str(e/'packet'))
    from _replay_l191 import replay191
    from _replay_l195 import replay195
    from relkit.stress_l195 import paired_mae,interval_verdict,claim_scope,keyed_auc,cluster_interval
    from relkit.landscape_l197 import coverage_report,admit_claim
    r=audit197(e,replay191,replay195,paired_mae,interval_verdict,claim_scope,keyed_auc,cluster_interval,coverage_report,admit_claim)
    (e/'report.json').write_text(json.dumps(r,indent=2)+'\n')
    print(r['status'],r['tables']['task_cells'],'table cells;',r['predictions']['prediction_rows'],'saved predictions;',r['coverage']['measured'],'fresh tasks')

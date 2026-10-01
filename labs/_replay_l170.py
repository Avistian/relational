"""Full original audit plus live learner contracts; never substitutes stored scores."""
def replay170(root,manifest,audit_fn,auc_fn,sample_fn,curve_fn,scaling_fn,pair_fn,adaptation_fn,gate_fn):
    import hashlib,json
    from pathlib import Path
    root=Path(root)
    for name,digest in manifest['files'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:raise ValueError('Pinned input differs: '+name)
    original_pins=json.loads((root/'evidence/l169/audit-manifest.json').read_text())
    if any(manifest['files'].get(n)!=d for n,d in original_pins['files'].items()):raise ValueError('Original evidence pins differ')
    audit=audit_fn(root,original_pins,auc_fn,sample_fn,curve_fn,scaling_fn)
    if audit!=json.loads((root/'evidence/l169/report.json').read_text()):raise ValueError('Replay differs from original report')
    if audit['fresh_runs']+audit['reused_runs']!=300 or audit['all_predictions_checked']!=229050:raise ValueError('Incomplete original experiment')
    records=[]
    for name,curve in audit['curves'].items():
        db,arm=name.split('/')
        for level in curve['levels']:
            for seed,value in enumerate(level['per_seed']):records.append(dict(database=db,arm=arm,context=level['context'],seed=seed,auc=value))
    paired=pair_fn(records)
    facts=dict(artifacts=True,metrics=True,dfs=False,temporal=False,exposure=False,fresh_training=False,matched_pipeline=False,heldout_databases=False,repeatability=all(x['exact_probability_match'] for x in audit['sentinel_checks']))
    claims={claim:gate_fn(claim,facts) for claim in ['saved_replay','full_pipeline','fresh_pretraining','general_advantage','exact_repeatability']}
    griffin=json.loads((root/'evidence/l164/report.json').read_text())
    if griffin['full_reproduction']!='INCOMPLETE_BUDGET_GATE':raise ValueError('Reassess changed Griffin evidence')
    return dict(experiment=manifest['experiment'],status='COMPLETE_SAVED_EVIDENCE_REPLAY',runs=300,predictions=229050,
        l170_fresh_runs=0,l170_reused_runs=300,original_l169_fresh_runs=240,original_l169_reused_runs=60,
        selected_tasks=2,contexts=[64,128,256,512,1024],paired=paired,curves=audit['curves'],paper_comparisons=audit['comparisons'],
        adaptation={str(k):adaptation_fn(k,0) for k in [64,128,256,512,1024]},evidence=facts,claims=claims,
        exact_repeatability=audit['exact_repeatability'],sentinel_checks=audit['sentinel_checks'],
        griffin=griffin['full_reproduction'],rdblearn_pipeline='NOT_RUN',fresh_pretraining='NOT_RUN',whole_paper='NOT_RUN',
        historical_identity='NOT_ESTABLISHED',historical_availability='NOT_ESTABLISHED',exact_target_schema_exclusion='NOT_ESTABLISHED',
        original_dfs_regeneration='NOT_RUN',learner='PENDING_WRITTEN_DEFENSE',additional_cloud_spend_usd=0,hashed_files=len(manifest['files']))

if __name__=='__main__':
    import json
    from pathlib import Path
    from _audit_l169 import audit169
    from relkit.transfer_l167 import keyed_auc
    from relkit.scaling_l169 import sample_context,scaling_curve,scaling_claim
    from relkit.design_l170 import paired_design,adaptation_mode,claim_gate
    P=Path(__file__).resolve().parent;E=P/'evidence/l170'
    report=replay170(P,json.loads((E/'input-manifest.json').read_text()),audit169,keyed_auc,sample_context,scaling_curve,scaling_claim,paired_design,adaptation_mode,claim_gate)
    (E/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(report['status'],report['runs'],report['predictions'])

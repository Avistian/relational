"""Full proposal audit; injected learner functions control matrices and interpretation."""
def authenticate198(root):
    import hashlib,json
    from pathlib import Path
    e=Path(root);q=e/'packet';m=json.loads((e/'input-manifest.json').read_text())
    actual={str(p.relative_to(q)) for p in q.rglob('*') if p.is_file() and '__pycache__' not in str(p)}
    if actual!=set(m['files']):raise ValueError('Incomplete or extra frozen inputs')
    for name,digest in m['files'].items():
        path=(q/name).resolve()
        if not path.is_relative_to(q.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            raise ValueError('Changed frozen input: '+name)
        if m['origins'][name]['sha256']!=digest:raise ValueError('Origin digest mismatch')
    return len(actual)


def audit198(root, ranking_audit, admission, priority, sensitivity, landscape_audit, tables_replay, predictions_replay, pair, verdict, scope, auc, interval, coverage, admit, contrast, decision, matrix):
    import json
    from pathlib import Path
    e=Path(root);q=e/'packet';count=authenticate198(e);read=lambda name:json.loads((q/name).read_text())
    ranking=ranking_audit(q/'evidence/l189/packet',read('evidence/l189/input-manifest.json'),admission,priority,sensitivity)
    landscape=landscape_audit(q/'evidence/l197',tables_replay,predictions_replay,pair,verdict,scope,auc,interval,coverage,admit)
    if ranking!=read('evidence/l189/report.json') or landscape!=read('evidence/l197/report.json'):
        raise ValueError('Complete predecessor replay differs')
    proposals=json.loads((e/'proposals.json').read_text());original=read('evidence/l189/packet/cases.json')
    if [c['id'] for c in proposals]!=['temporal','composite','transfer']:raise ValueError('Complete ordered proposal set required')
    outputs=[]
    for c,old in zip(proposals,original):
        for key in ['impact','feasibility','related_work','evidence','budget']:
            if c[key]!=old[key]:raise ValueError('Original ranking/evidence assumptions changed')
        for field in ['delta','after_l197','hypothesis_refined','falsifier_refined','count_note','next_artifact']:
            if not isinstance(c.get(field),str) or not c[field].strip():raise ValueError('Missing proposal argument: '+field)
        if not c.get('unresolved_execution_fields') or c['protocol_state']!='PROPOSAL_NOT_EXECUTION_READY':raise ValueError('Hidden execution gaps')
        if c['novelty']!='NOT_ESTABLISHED' or c['full_experiment']!='NOT_RUN':raise ValueError('Unsupported evidence promotion')
        batches={name:matrix(axes) for name,axes in c['matrices'].items()}
        example=c['illustration'];score=contrast(example['scores'],example['mode'])
        contrasts={mode:contrast(example['scores'],mode) for mode in (['gain'] if len(example['scores'])==2 else ['conditional','interaction'])}
        outputs.append(dict(id=c['id'],priority=priority(c['impact'],c['feasibility'],[1,1,1]),counts={name:len(rows) for name,rows in batches.items()},matrices=batches,budget=admission(c['budget']),illustrative_contrast=score,illustrative_contrasts=contrasts,illustrative_interval=example['interval'],illustrative_verdict=decision(*example['interval'],c['margin_auroc'],example['decision_mode']),illustration_scope='HYPOTHETICAL_NOT_MEASURED',unresolved_execution_fields=c['unresolved_execution_fields'],future_experiment='NOT_RUN',novelty='NOT_ESTABLISHED'))
    changed=[dict(c,impact=3) if c['id']=='temporal' else c for c in proposals]
    changed_scores={c['id']:priority(c['impact'],c['feasibility'],[1,1,1]) for c in changed}
    return dict(experiment='L198-RESEARCH-PROPOSAL-AUDIT',status='COMPLETE_SELECTED_PROPOSAL_AUDIT',authenticated_files=count,ranking=ranking,landscape=landscape,proposals=outputs,changed_assumption=dict(temporal_impact=3,weights=[1,1,1],scores=changed_scores,leaders=sorted(k for k,v in changed_scores.items() if v==max(changed_scores.values()))),full_model_reproduction=landscape['full_model_reproduction'],novelty='NOT_ESTABLISHED',new_model_experiments='NOT_RUN',independent_replication='NOT_ESTABLISHED',learner='PENDING_WRITTEN_DEFENSE',cloud_usd=0)


def run198(root):
    import sys
    from pathlib import Path
    # Authenticate before importing any frozen executable source.
    authenticate198(root);q=Path(root)/'packet';inner=q/'evidence/l197/packet'
    sys.dont_write_bytecode=True
    import relkit
    relkit.__path__[:0]=[str(q/'relkit'),str(inner/'relkit')]
    sys.path[:0]=[str(q),str(inner)]
    from _audit_l189 import audit
    from relkit.gaps_l189 import admission,priority,sensitivity
    from _audit_l197 import audit197
    from _replay_l191 import replay191
    from _replay_l195 import replay195
    from relkit.stress_l195 import paired_mae,interval_verdict,claim_scope,keyed_auc,cluster_interval
    from relkit.landscape_l197 import coverage_report,admit_claim
    from relkit.proposals_l198 import paired_contrast,interval_decision,expand_matrix
    return audit198(root,audit,admission,priority,sensitivity,audit197,replay191,replay195,paired_mae,interval_verdict,claim_scope,keyed_auc,cluster_interval,coverage_report,admit_claim,paired_contrast,interval_decision,expand_matrix)


if __name__=='__main__':
    import json
    from pathlib import Path
    e=Path(__file__).resolve().parent/'evidence/l198';r=run198(e)
    (e/'report.json').write_text(json.dumps(r,indent=2)+'\n')
    print(r['status'],r['ranking']['weight_scenarios'],'weight settings;',r['landscape']['predictions']['prediction_rows'],'predictions;',[(x['id'],x['counts']) for x in r['proposals']])

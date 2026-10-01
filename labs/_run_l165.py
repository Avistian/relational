"""Named complete deterministic audit; no model or service inference."""
def audit165(packet, context_fn, graph_fn, auc_fn):
    if packet['scope']!='SYNTHETIC_CONTRACT_AUDIT':
        raise ValueError('Scope mismatch')
    context=[]
    for case in packet['context_cases']:
        chosen=context_fn(case['context'],case['query'])
        context.append(dict(id=case['id'],selected=[[r['entity'],r['cutoff']] for r in chosen]))
    graphs=[]
    for case in packet['graph_cases']:
        args={k:v for k,v in case.items() if k!='id'}
        graphs.append(dict(id=case['id'],visible=graph_fn(**args)))
    scores=[dict(id=c['id'],auc=auc_fn(c['truth'],c['predictions'])) for c in packet['score_cases']]
    # Only entity/cutoff enter the query contract. Held-out labels remain in truth.
    query=packet['mask_case']['query']
    masked_inputs=[]
    for hidden_label in [0,1]:
        heldout=dict(query,label=hidden_label)
        model_query={k:heldout[k] for k in ['entity','cutoff']}
        masked_inputs.append(dict(query=model_query,context=context_fn(packet['mask_case']['context'],model_query)))
    assert masked_inputs[0]==masked_inputs[1]
    return dict(experiment='L165 KumoRFM-v1 Context and Reproduction Audit',scope=packet['scope'],
                context=context,graphs=graphs,scores=scores,query_label_intervention='INPUTS_IDENTICAL',
                counts=dict(context=len(context),graph=len(graphs),scoring=len(scores),query_label_interventions=2),
                cloud_spend_usd=0,model_inference='NOT_RUN',historical_reproduction='NOT_RUN',
                historical_fidelity='NOT_ESTABLISHED',learner='PENDING_WRITTEN_DEFENSE')

def render165(report):
    lines=['# L165 KumoRFM-v1 Context and Reproduction Audit','',
           'Complete synthetic contract audit. No KumoRFM inference, fitted weights or benchmark predictions.','',
           'Counts: '+str(report['counts'])+'.',
           'Changing the isolated held-out label leaves query/context inputs identical. This checks this course packet, not proprietary internal masking.','',
           '| Scoring case | AUROC |','|---|---:|']
    lines += [f"| {c['id']} | {c['auc']:.6f} |" for c in report['scores']]
    lines += ['', 'The identical scoring values come from permutations of four invented records; they are not independent seeds or measured performance.', '',
              'All context and graph interventions are recorded in report.json. Historical v1 driver-dnf target: 0.8241 AUROC (82.41 on the report scale), NOT_RUN; fidelity NOT_ESTABLISHED. Learner PENDING_WRITTEN_DEFENSE.']
    return '\n'.join(lines)+'\n'

if __name__=='__main__':
    import hashlib,json
    from pathlib import Path
    from relkit.context_l165 import eligible_context,visible_graph,keyed_auc
    p=Path(__file__).resolve().parent/'evidence/l165';raw=(p/'fixtures.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest()==json.loads((p/'input-manifest.json').read_text())['fixtures_sha256']
    result=audit165(json.loads(raw),eligible_context,visible_graph,keyed_auc)
    (p/'report.json').write_text(json.dumps(result,indent=2)+'\n');(p/'report.md').write_text(render165(result));print(result['counts'])

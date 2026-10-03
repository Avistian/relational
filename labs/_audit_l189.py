"""Full frozen report/source audit, not underlying prediction/model replay."""
import hashlib,json
from pathlib import Path


def audit(packet,manifest,admission,priority,sensitivity):
    packet=Path(packet);files=manifest['files']
    actual={str(p.relative_to(packet)) for p in packet.rglob('*') if p.is_file()}
    if set(files)!=actual:raise ValueError('Input inventory mismatch')
    for name,expected in files.items():
        if Path(name).is_absolute() or '..' in Path(name).parts:raise ValueError('Unsafe manifest path')
        if hashlib.sha256((packet/name).read_bytes()).hexdigest()!=expected:raise ValueError('Changed evidence: '+name)
    config=json.loads((packet/'config.json').read_text());cases=json.loads((packet/'cases.json').read_text())
    sources=json.loads((packet/'sources.json').read_text());inherited=json.loads((packet/'inherited.json').read_text())
    if [c['id'] for c in cases]!=config['case_ids'] or len(cases)!=3:raise ValueError('Incomplete case set')
    source_ids={r['id'] for r in sources};receipts={Path(r['file']).name for r in inherited}
    if len(source_ids)!=6 or len(receipts)!=6:raise ValueError('Incomplete source/receipt set')
    for r in sources+inherited:
        if r.get('status',200)!=200 or files.get(r['file'])!=r['sha256']:raise ValueError('Unverified source/receipt')
    for c in cases:
        for field in ['question','hypothesis','known','gap','approach','baseline','minimum_design','estimand','failure','next_step']:
            if not isinstance(c.get(field),str) or not c[field].strip():raise ValueError('Missing research contract: '+field)
        if not set(c['related_work'])<=source_ids or not c['related_work']:raise ValueError('Unverified related work')
        if not set(c['evidence'])<=receipts or not c['evidence']:raise ValueError('Missing inherited evidence')
        if c['novelty']!='NOT_ESTABLISHED' or c['full_experiment']!='NOT_RUN':raise ValueError('Unsupported claim upgrade')
        if len(c['feasibility_reasons'])!=3:raise ValueError('Unexplained scores')
    reports={n:json.loads((packet/'inherited'/f'l{n}-report.json').read_text()) for n in [169,177,182,183,184]}
    if reports[182]['hybrid_training']!='NOT_RUN' or reports[183]['hybrid_pretraining']!='NOT_RUN':raise ValueError('Revisit case interpretation')
    collision=reports[184]['cache_collisions']
    for split,row in collision.items():
        if row['queries']-row['unique_entities']!=row['overwritten_query_entries']:raise ValueError('Cache arithmetic inconsistent')
        if row['queries']!=reports[184]['counts'][split]:raise ValueError('Query population mismatch')
    attempts=json.loads((packet/'inherited/l188-collection.json').read_text())['attempts']
    # A request receipt alone does not establish drained pagination and screened coverage.
    l188={'observed_attempts':len(attempts),'failed_attempts':sum(not a.get('accepted',False) for a in attempts),'coverage':'NOT_ESTABLISHED_FROM_RECEIPT'}
    ranking=[]
    for c in cases:
        ranking.append(dict(id=c['id'],score=priority(c['impact'],c['feasibility'],config['weights']),cost_gate=admission(c['budget'])))
    ranking.sort(key=lambda r:(-r['score'],r['id']))
    for r in ranking:r['rank']=1+sum(x['score']>r['score'] for x in ranking)
    grid=sensitivity(cases)
    if len(grid)!=27:raise ValueError('Incomplete weight sweep')
    stability={c['id']:dict(score_min=min(g['scores'][c['id']] for g in grid),score_max=max(g['scores'][c['id']] for g in grid),leader_scenarios=sum(c['id'] in g['leaders'] for g in grid)) for c in cases}
    observations={
      'l169':{'status':reports[169]['status'],'claimed_predictions':reports[169]['all_predictions_checked'],'pretraining_scaling_law':reports[169]['pretraining_scaling_law']},
      'l177':{'status':reports[177]['status'],'claimed_authenticated_inputs':reports[177]['authenticated_inputs']},
      'l182':{'status':reports[182]['status'],'claimed_predictions':reports[182]['predictions'],'hybrid_training':reports[182]['hybrid_training']},
      'l183':{'status':reports[183]['status'],'claimed_predictions':reports[183]['verified_predictions'],'hybrid_pretraining':reports[183]['hybrid_pretraining']},
      'l184':{'status':reports[184]['selected_experiment'],'query_count':sum(x['queries'] for x in collision.values()),'overwritten_query_entries':sum(x['overwritten_query_entries'] for x in collision.values()),'fresh_training':reports[184]['fresh_training']},'l188':l188}
    return dict(experiment=config['experiment'],status='COMPLETE_SELECTED_AUDIT',case_count=3,authenticated_files=len(files),primary_sources=len(sources),inherited_receipts=len(inherited),ranking=ranking,weight_scenarios=len(grid),stability=stability,weight_grid=grid,observations=observations,source_review='SELECTED_PRIMARY_TEXTS',literature_coverage='NOT_ESTABLISHED',novelty='NOT_ESTABLISHED',prediction_replay='NOT_RUN',fresh_training='NOT_RUN',whole_paper='NOT_RUN',cloud_usd=0,learner='PENDING_WRITTEN_DEFENSE')


if __name__=='__main__':
    from relkit.gaps_l189 import admission,priority,sensitivity
    E=Path(__file__).resolve().parent/'evidence/l189'
    report=audit(E/'packet',json.loads((E/'input-manifest.json').read_text()),admission,priority,sensitivity)
    (E/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print({k:v for k,v in report.items() if k not in ['weight_grid','observations']})

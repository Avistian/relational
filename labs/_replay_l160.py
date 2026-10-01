"""Read-only full saved-evidence replay plus Year 4 readiness assessment."""
import json,statistics
from pathlib import Path
import numpy as np
from _replay_l158 import replay as replay_synthesis
from _replay_l154 import verify_inputs
from relkit.exam_l160 import aligned_losses,observed_effort,exit_gates

def replay160(root,manifest,align=aligned_losses,effort=observed_effort,gates=exit_gates):
    root=Path(root);verify_inputs(root,manifest)
    read=lambda n:json.loads((root/n).read_text())
    inherited=replay_synthesis(root,read('evidence/l158/input-manifest.json'))
    if inherited!=read('evidence/l158/report.json'):raise ValueError('Inherited replay differs')
    paired=[]
    for seed in range(5):
        with np.load(root/f'evidence/l155/fe/paper/seed-{seed}/predictions.npz') as fe,np.load(root/f'evidence/l155/paper/seed-{seed}/predictions.npz') as rdl:
            for split in ['val','test']:
                keys=lambda p:list(zip(map(int,p[split+'_entity']),map(int,p[split+'_time'])))
                fk=keys(fe);rk=keys(rdl)
                if dict(zip(fk,fe[split+'_target']))!=dict(zip(rk,rdl[split+'_target'])):raise ValueError('Targets differ')
                # Deliberately reverse one packet: the learner must join, not zip.
                a=align(fk,fe[split+'_target'],fk[::-1],fe[split+'_pred'][::-1],rk,rdl[split+'_pred'])
                expected=next(r for r in inherited['matched']['rows'] if r['seed']==seed and r['split']==split)
                if abs(a['mean_benefit']-expected['benefit_mae'])>1e-10:raise ValueError('Learner benefit differs from frozen replay')
                paired.append(dict(seed=seed,split=split,queries=len(fk),benefit=a['mean_benefit']))
    original_effort=read('evidence/l155/effort-log.json')
    # The inherited archive has no observations. Do not invent event durations.
    if original_effort['sessions']:raise ValueError('New effort records need an explicit, reviewed schema adapter')
    human=effort([])
    entries=[]
    for x in inherited['portfolio']['entries']:
        is_f1=x['task']=='rel-f1/driver-position'
        entries.append(dict(task=x['task'],status=x['status'],split='test',
                            matched_fe='COMPLETE' if is_f1 else 'NOT_RUN',effort=human['status'],
                            temporal='NOT_ESTABLISHED',basis='L155 matched predictions' if is_f1 else 'L151–L153 frozen portfolio',
                            audit_scope='Historical feature availability not established; L156/L157 fixed-horizon corrections do not prove it'))
    failures=[dict(evidence='evidence/l155/report.json',limitation='FE point estimate wins; conditional driver interval crosses zero'),
              dict(evidence='evidence/l153/cost-decision.json',limitation='Recommendation five-seed reproduction exceeds budget; test NOT_RUN'),
              dict(evidence='evidence/l156/report.json',limitation='Released strict policy FAIL; fixed-horizon policy NOT_ESTABLISHED')]
    assessment=gates(entries,failures,None)
    return dict(experiment=manifest['experiment'],replay_status='PASS',frozen_inputs=len(manifest['files']),
                prediction_rows_rescored=inherited['prediction_rows_rescored'],selection_checks=inherited['selection_checks'],
                portfolio=entries,assessment=assessment,paired_loss_checks=paired,human_effort=human,failure_cases=failures,
                matched_test=inherited['matched']['metrics']['test'],conditional_driver_interval=inherited['matched']['test_benefit_driver_bootstrap_95'],
                paper_targets=inherited['portfolio']['published_context'],recommendation_cost_decision=inherited['portfolio']['recommendation_cost_decision'],
                boundaries=inherited['boundaries'],additional_cloud_spend_usd=0,
                count_note='98,918 counts repeated seed/split predictions including validation pilot; not independent observations. Extra paired-loss checks reuse 12,590 rows.')

def render160(r):
    a=r['assessment'];c=a['counts'];m=r['matched_test'];lo,hi=r['conditional_driver_interval']
    lines=['# Year 4 exit evidence replay','',r['experiment'],'',f"Replay **{r['replay_status']}**; Year 4 exit **{a['exit']}**; learner **{a['written_defense']}**.",
           '',f"{r['prediction_rows_rescored']:,} prediction rows rescored; {r['selection_checks']} validation-selection checks; {r['frozen_inputs']} frozen inputs.",
           '', '| Task | Test experiment | Matched FE | Human effort | Temporal sign-off |','|---|---|---|---|---|']
    for x in r['portfolio']:lines.append('| '+' | '.join(x[k] for k in ['task','status','matched_fe','effort','temporal'])+' |')
    lines+=['',f"Coverage: {c['completed_tasks']}/3 completed tasks; {c['matched_fe_tasks']}/3 matched FE tasks; {c['observed_effort_tasks']}/3 observed effort ratios; {c['temporal_pass_tasks']}/3 temporal sign-offs.",
            '',f"F1 test MAE: FE {m['fe_mae']['mean']:.6f}, RDL {m['rdl_mae']['mean']:.6f}; FE − RDL {m['benefit_mae']['mean']:+.6f} positions. Conditional driver-bootstrap 95% interval [{lo:+.6f}, {hi:+.6f}]. No superiority or equivalence established.",
            '', '## Gate verdicts']+[f"- {k}: {'PASS' if v else 'INCOMPLETE'}" for k,v in a['gates'].items()]
    lines+=['','## Evidence boundaries']+[f'- {k}: {v}' for k,v in r['boundaries'].items()]
    lines+=['',r['count_note'],'','USD0 additional cloud spending; no fresh fits. Original human study and whole-paper reproduction NOT_RUN.']
    return '\n'.join(lines)+'\n'

if __name__=='__main__':
    P=Path(__file__).resolve().parent;E=P/'evidence/l160'
    result=replay160(P,json.loads((E/'input-manifest.json').read_text()))
    (E/'report.json').write_text(json.dumps(result,indent=2)+'\n');(E/'report.md').write_text(render160(result));print(render160(result))

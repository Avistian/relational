"""Execute every declared synthetic design case, with caller-supplied learner code."""
import hashlib,json
from pathlib import Path
from relkit.scope_l161 import adaptation_route,database_boundary,scope_verdict

def audit161(packet, route, boundary, verdict):
    rows=[]
    for case in packet['cases']:
        answer=verdict(case['record'],route=route,boundary=boundary)
        rows.append(dict(name=case['name'],**answer))
    return dict(experiment='L161 Relational Foundation-Model Scope Audit',
                data_kind=packet['kind'],cases=len(rows),rows=rows,
                training='NOT_RUN',paper_performance_reproduction='NOT_APPLICABLE_TO_SELECTED_CONCEPT_SCOPE',
                transfer_performance='NOT_ESTABLISHED',cloud_spend_usd=0,
                learner='PENDING_WRITTEN_DEFENSE')

def render161(report):
    lines=['# L161 declared-protocol audit','',
           'Synthetic metadata only. READY_FOR_REVIEW is conditional on supplied facts, not verified performance.',
           '', '| Case | Adaptation | Pretraining boundary | Design status |',
           '|---|---|---|---|']
    lines += [f"| {r['name']} | {r['mode']} | {r['pretraining_boundary']} | {r['status']} |" for r in report['rows']]
    return '\n'.join(lines)+'\n\nNo training, transfer measurement or learner mastery is established. Cloud spend USD0.\n'

if __name__=='__main__':
    p=Path(__file__).resolve().parent/'evidence/l161'
    raw=(p/'fixtures.json').read_bytes();manifest=json.loads((p/'input-manifest.json').read_text())
    assert hashlib.sha256(raw).hexdigest()==manifest['fixtures_sha256'],'Frozen fixture bytes changed'
    r=audit161(json.loads(raw),adaptation_route,database_boundary,scope_verdict)
    (p/'report.json').write_text(json.dumps(r,indent=2)+'\n');(p/'report.md').write_text(render161(r))
    print(render161(r))

"""Authenticate source packet and summarize a fresh diagnostic."""
import hashlib,json
from pathlib import Path
from relkit.community_l196 import summarize_cases,route_question,feedback_state

def make_report(root):
    root=Path(root);ledger=json.loads((root/'packet/source-manifest.json').read_text())
    for name,digest in ledger['files'].items():
        if hashlib.sha256((root/'packet'/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('Source bytes changed: '+name)
    diagnostic=json.loads((root/'diagnostic.json').read_text())
    if diagnostic['commit']!=ledger['commit'] or diagnostic['model_evaluations']!=0:
        raise ValueError('Scope or source revision changed')
    return dict(experiment='L196 complete original-preprocessor four-case diagnostic',
        status=diagnostic['status'],source_commit=ledger['commit'],
        summary=summarize_cases(diagnostic['observations']),
        question_destination=route_question('rdblearn'),participation=feedback_state('', '', False),
        learner='PENDING_WRITTEN_DEFENSE',full_model_reproduction=diagnostic['full_model_reproduction'])
if __name__=='__main__':
    E=Path(__file__).resolve().parent/'evidence/l196';r=make_report(E)
    (E/'report.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))

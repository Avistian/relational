"""Complete deterministic course experiment; no historical training dispatch."""
def audit162(packet, reach, budget, verdict):
    if packet['scope']!='SYNTHETIC_COURSE_AUDIT':raise ValueError('Unexpected experiment scope')
    graphs=[]
    for case in packet['graph_cases']:
        args={k:v for k,v in case.items() if k!='id'}
        rows=reach(**args)
        graphs.append(dict(id=case['id'],reachable=rows,tables=sorted({case['nodes'][n] for n in rows}),count=len(rows)))
    tokens=[budget(**case) for case in packet['token_cases']]
    claims=[verdict(case) for case in packet['evidence_cases']]
    return dict(experiment='L162 Relational Context and Scale Audit',scope=packet['scope'],graphs=graphs,tokens=tokens,claims=claims,
                counts=dict(graph=len(graphs),token=len(tokens),evidence=len(claims)),cloud_spend_usd=0,
                historical_reproduction='NOT_RUN',historical_fidelity='NOT_ESTABLISHED',transfer='NOT_ESTABLISHED',learner='PENDING_WRITTEN_DEFENSE')

def render162(report):
    lines=['# L162 Relational Context and Scale Audit','', 'Complete synthetic computation; no model training or measured transfer.','', '| Graph intervention | Reachable rows | Tables |','|---|---|---|']
    lines += ['| '+r['id']+' | '+', '.join(r['reachable'])+' | '+', '.join(r['tables'])+' |' for r in report['graphs']]
    lines += ['', '| Token case | Total | Dropped | Whole-table pairs | Row pairs | Truncated row pairs |','|---|---:|---:|---:|---:|---:|']
    lines += [f"| {i} | {r['total_tokens']} | {r['dropped_tokens']} | {r['whole_table_pairs']} | {r['row_pairs']} | {r['retained_row_pairs']} |" for i,r in enumerate(report['tokens'])]
    from collections import Counter
    lines += ['', 'Evidence grid: '+str(dict(Counter(r['supported'] for r in report['claims'])))+'.','', 'All 64 evidence declarations still return transfer NOT_ESTABLISHED. Review eligibility never authenticates evidence.', '', 'Historical reproduction NOT_RUN; fidelity NOT_ESTABLISHED; learner PENDING_WRITTEN_DEFENSE.']
    return '\n'.join(lines)+'\n'

if __name__=='__main__':
    import json,hashlib,signal
    from pathlib import Path
    from relkit.vision_l162 import reachable_rows,token_budget,evidence_verdict
    signal.alarm(600)
    p=Path(__file__).resolve().parent/'evidence/l162'
    raw=(p/'fixtures.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest()==json.loads((p/'input-manifest.json').read_text())['fixtures_sha256']
    result=audit162(json.loads(raw),reachable_rows,token_budget,evidence_verdict)
    (p/'report.json').write_text(json.dumps(result,indent=2)+'\n');(p/'report.md').write_text(render162(result))
    print(result['counts'])

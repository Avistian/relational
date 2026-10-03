"""Review full PluRel target and fail closed before unapproved dispatch."""
import argparse,json
from pathlib import Path
P=Path(__file__).resolve().parent
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--audit',action='store_true');p.add_argument('--run-full',action='store_true');a=p.parse_args()
    table=json.loads((P/'sources/b13/plurel-table1.json').read_text())
    assert len(table['rows'])==18
    summary=dict(target='PluRel v1 Table1',tasks=18,arms=2,seeds=3,real_data_training_runs=36,test_evaluations=108,synthetic_base='1024 DB / 4B tokens; additional pretraining',paper_code='AVAILABLE_PINNED',all_historical_seed_artifacts='NOT_ESTABLISHED',fresh_training='NOT_RUN',status='INCOMPLETE_SOURCE_PROTOCOL_AND_BUDGET_GATE')
    slots=[]
    for index,row in enumerate(table['rows']):
        for arm,column in [('real-only',2),('synthetic+real',3)]:
            for seed_slot in range(1,4):
                slots.append(dict(database=row[0],paper_task_alias=row[1],metric='AUROC_percent' if index<10 else 'R2_percent',arm=arm,paper_mean_percent=float(row[column]),seed_slot=seed_slot,historical_seed_id='UNRESOLVED',status='NOT_RUN'))
    assert len(slots)==108
    if a.audit:
        out=P/'evidence/b13/plurel-full-target.json';out.parent.mkdir(parents=True,exist_ok=True)
        out.write_text(json.dumps(dict(**summary,slots=slots),indent=2)+'\n')
    print(json.dumps(summary,indent=2))
    if a.run_full:raise SystemExit('BLOCKED: USD0 paid scope; unresolved three-seed historical mapping, data identity and cost admission. Read b13-reproduction.md.')

"""Record final measured scope; never infer learner mastery from author execution."""
import json,re
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parent;summary=json.loads((P/'evidence/l113/summary.json').read_text());budget=json.loads((P/'_budget_l113.json').read_text())
replays=[]
for seed in range(10):
 path=P/'evidence/l113/paper'/f'seed-{seed}'/'replay.json'
 if path.exists():replays.append(json.loads(path.read_text()))
delivery=json.loads((P/'_delivery_l113_results.json').read_text())
r={'status':'COMPLETE_SELECTED_EXPERIMENT' if summary['status']=='COMPLETE' else summary['status'],'experiment':'OGB v6 Table4 ClusterGCN on ogbn-products, SAGE aggregation','fresh_runs':len(summary['seeds']),'epochs':50*len(summary['seeds']),'summary':summary.get('summary',{}),'independent_node_predictions':2449029*len(summary['seeds']),'original_source_checkpoint_replay':{'completed':len(replays),'class_disagreements':sum(x['class_disagreements'] for x in replays)},'historical_identity':'NOT_ESTABLISHED','full_paper_reproduction':'NOT_ESTABLISHED','learner_status':'PENDING_WRITTEN_DEFENSE','live_colab':'NOT_CHECKED','deployment':'NOT_CHECKED','budget':budget,'protocol':'labs/l113-reproduction.md','evidence':'labs/evidence/l113/summary.json','delivery':delivery['status']}
p=P/'reproductions/execution_evidence.json';ledger=json.loads(p.read_text());ledger['l113']=r;p.write_text(json.dumps(ledger,indent=2)+'\n')
text='## Lesson 113 prepared · 2026-09-26\n\n'
text+='- Approved scope: full-arxiv GCN mini-batch bridge plus OGB products ClusterGCN named reproduction. GraphSAGE aggregation is explicitly separate from the sampler.\n'
text+='- '+summary['interpretation']+'\n'
text+=f"- {len(summary['seeds'])}/10 complete 50-epoch fits; {len(replays)} independent original-model checkpoint replays. Official-reader feature/edge-multiset equality, induced batches, sparse/dense gradients, duplicate edges, live student tasks and selection checks pass.\n"
text+='- HTML, reference, three-task student notebook, executed solution, visible canonical model/trainer, five portable figures, pinned source and exact commands ship together. Desktop/mobile/keyboard/no-JS/print and copied Pages passed.\n'
text+=f"- Budget USD10 aggregate; conservative authorized-resource ceiling USD{budget['all_authorized_calls_maximum_resource_usd']:.6f}. Actual recorded resource estimates are not an invoice; overhead unitemized. See labs/l113-reproduction.md.\n"
text+='- Learner PENDING_WRITTEN_DEFENSE; historical partition/seeds and whole-paper identity NOT_ESTABLISHED. Live Colab/deployment NOT_CHECKED.\n'
p=R/'NOTES.md';s=p.read_text();s=re.sub(r'\n## Lesson 113 prepared · 2026-09-26\n.*?(?=\n## |\Z)','',s,flags=re.S);p.write_text(s.rstrip()+'\n\n'+text)
p=R/'plan/year-3.md';s=p.read_text();anchor='### 113 · Scaling the OGB run — *Hu 2020 OGB + Cluster-GCN/SAGE*';s=re.sub(r'(?<=\n)- \*\*L113 delivery 2026-09-26:\*\*[^\n]*\n','',s);s=s.replace(anchor,anchor+'\n- **L113 delivery 2026-09-26:** '+summary['interpretation']+' Visible GCN bridge, full-data cluster runner and compute ledger; learner PENDING_WRITTEN_DEFENSE.');p.write_text(s)
p=R/'thesis-dossier.md';s=p.read_text();s=re.sub(r'\n- \*\*BAR · L113 \(2026-09-26\):\*\*[^\n]*\n?','\n',s);s+='\n- **BAR · L113 (2026-09-26):** '+summary['interpretation']+' Scaling evidence must specify both sampler and aggregation, preserve split/selection rules, and account for training AND inference. This one transductive products benchmark does not establish database superiority or deployment-time validity. [Contract](labs/l113-reproduction.md).\n';p.write_text(s)
print(r['status'],r['fresh_runs'],'fits;',len(replays),'replays')

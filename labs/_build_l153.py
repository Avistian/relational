"""Deterministic lesson/reference plus standalone notebooks with visible full pipeline."""
import ast,base64,hashlib,json,re,zlib
from pathlib import Path
import nbformat as nb
from nbconvert.filters.markdown import markdown2html_mistune as render
P=Path(__file__).resolve().parent;R=P.parent;E=P/'evidence/l153';S='0153-recommendation-portfolio';TITLE='Recommendation portfolio: rank candidates without temporal leakage'
summary=json.loads((E/'summary.json').read_text());decision=json.loads((E/'cost-decision.json').read_text());budget=json.loads((P/'_budget_l153.json').read_text());pilot=summary['pilot'];ceiling=sum(x['upper_usd'] for x in budget['reservations'])+3
if summary['portfolio']:
 results='| Seed | Selected epoch | Validation MAP@10 | Test MAP@10 |\n|---|---:|---:|---:|\n'+''.join(f"| {r['seed']} | {r['selected_epoch']} | {100*r['scores']['val']:.3f}% | {100*r['scores']['test']:.3f}% |\n" for r in summary['records'])
 results+=f"\nFive-run test mean {summary['portfolio']['test']['mean']*100:.3f}%: **{summary['portfolio']['paper_comparison']}** under the frozen descriptive tolerance.\n"
else:
 results=f"""**Full selected reproduction: INCOMPLETE — budget gate STOP.** The full code is provided, but no five-seed test comparison is claimed.

The fresh pilot trained **32 batches / 16,384 query occurrences**, then evaluated all **{pilot['counts']['val']:,} validation queries** against the full **53,241-candidate catalog**. Its validation MAP@10 was **{pilot['scores']['val']*100:.3f}%**, Hit@10 **{pilot['val_metrics']['hit']*100:.3f}%**, Recall@10 **{pilot['val_metrics']['recall']*100:.3f}%**. These are partial-training diagnostics, not final baseline results. Test was not evaluated.

Training the 32 pilot batches took **{decision['pilot_training_seconds']:.1f}s**; a full validation pass took **{decision['full_validation_seconds']:.1f}s**. The released timestamp sampler yields **{pilot['loader_batches']:,} full batches** per epoch, covering **{pilot['training_queries_per_full_epoch']:,} query occurrences**. The remaining timestamp-group tails are dropped by source design.

A linear scenario for five 20-epoch runs projects **USD{decision['projected_five_run_compute_usd']:.2f} compute before overhead**. The projection includes the pilot's audit/cold-start cost; it is not a mathematical lower bound. The safety-adjusted forecast exceeds the remaining allowance, so no full fits were dispatched. We preserved the recipe rather than reducing epochs, candidates or seeds. [Recorded cost decision](../labs/evidence/l153/cost-decision.json).
"""
a=pilot['audit'];nonfinite=sum(pilot['first_batch_nonfinite_gradients'].values());labels=sum(summary['labels']['independently_rebuilt_queries'].values())
audit=f"""**What passed:** all **{labels:,} query labels** were independently reconstructed through interval/join logic in addition to the source-SQL check. All **{summary['independently_rescored_rankings']:,} saved rankings** were independently aligned and rescored. The pilot audited **{a['queries']:,} root occurrences**, **{a['dated_node_occurrences']:,} dated-node occurrences** and **{a['edge_occurrences']:,} sampled-edge occurrences**, with zero observed owner-cutoff violations.

The first32training batches contained **{a['positive_collisions']:,} positive collisions out of {a['negative_comparisons']:,} shared comparisons**. This diagnostic preserves source sampling; it does not establish a false-negative rate for unobserved relevance. Candidate/feature arrival history remains unobserved.

The first real backward pass contained **{nonfinite:,} nonfinite gradient entries**. Source behavior was retained and recorded. Finite prediction parity with the released model (maximum absolute difference **{pilot['original_model_max_abs']:.3g}**) and the healthy synthetic mechanism check do not certify healthy optimization on this real dataset.

Current immutable resource reservations plus the USD3 overhead allowance total **USD{ceiling:.6f}** under the USD10 cap. Recorded main-worker body estimate: **USD{summary['worker_body_usd']:.6f}**; this excludes unitemized overhead and is not an invoice. [Budget](../labs/_budget_l153.json) · [Evidence](../labs/evidence/l153/summary.json).
"""
if (P/'_notebook_l153_results.json').exists():audit+='\nThe archived default notebook passed in an isolated pinned runtime. After correcting only the inclusive score-tolerance boundary and its check, all revised default cells passed locally on CPU. The original module and notebook hashes are retained for the pinned execution; the source model and trainer are unchanged. The full training gate remains NOT_RUN; live Colab remains NOT_CHECKED.\n'
captions={'architecture':'Shared temporal row/graph encoder, three training root groups, BPR comparisons and full-catalog inference. Shapes describe actual computation.','negatives':'A two-query shared pool has two positive collisions. Observed availability can reveal a different pair of violations; missing arrival history stays unknown.','ranking':'AP counts precision at each relevant rank and divides by min(k,positive count). Hit and Recall use different denominators.','evidence':'Measured execution scope and the cost gate. A pilot validation score cannot replace five completed test runs.'}
def prose(portable=False):
 s=(R/'lessons/content'/f'{S}.md').read_text().replace('[[RESULTS]]',results).replace('[[AUDIT]]',audit)
 for name,caption in captions.items():
  src='data:image/png;base64,'+base64.b64encode((P/f'figures/l153/{name}.png').read_bytes()).decode() if portable else f'../labs/figures/l153/{name}.svg'
  s=s.replace('[[FIG:'+name+']]',f'<figure class="route-figure" tabindex="0"><img src="{src}" alt="{caption}"><figcaption>{caption} Scroll horizontally on narrow screens.</figcaption></figure>')
 s=s.replace('[[WARMUP]]','Retrieve: why is the cutoff part of query identity? Why must validation choose the checkpoint?' if portable else '<div id="warmup"></div>')
 s=s.replace('[[PREDICT]]','Predict: remove B from ranking A,B,C,D with truth A,C. What changes in AP@2?' if portable else '<div id="predict"></div><noscript>Removing B raises AP@2 from .50 to1.00 without changing the model.</noscript>')
 s=s.replace('[[RANKING_WIDGET]]','Worked intervention: full ranking A,B,C,D at k2 has AP.50; remove B and AP becomes1.00. Candidate protocol changed.' if portable else '<div class="route-widget" id="l153-ranking"></div><noscript>Full A,B,C,D at k2: AP.50. Remove B: AP1.00. This changes the candidate protocol.</noscript>')
 s=s.replace('[[TEACHBACK]]','Write your defense before reviewing the model answer; send it to the teaching agent.' if portable else '<div id="teachback"></div>')
 if portable:s=s.replace('](../','](https://avistian.github.io/relational/');s=re.sub(r'\]\((\d{4}-[^)]+\.html)\)',r'](https://avistian.github.io/relational/lessons/\1)',s) if portable else s
 return s

def doc(title,body,interactive=False):
 body=re.sub(r'<table([^>]*)>',r'<div class="route-scroll" tabindex="0"><table\1>',render(body)).replace('</table>','</table></div>')
 scripts=['retrieval-pool','retrieval-bank','predict','teachback','recommendation-ranking-viz','l153-lesson'] if interactive else []
 return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Lesson 153 — '+title+'</title><link rel="stylesheet" href="../assets/lesson.css"><link rel="stylesheet" href="../assets/atomic-route.css"><link rel="stylesheet" href="../assets/checkpoint.css"></head><body class="checkpoint"><article><nav><a href="../index.html">Course</a> · <a href="../lessons/0152-regression-portfolio.html">Lesson 152</a></nav><header><p class="route-kicker">Year 4 · Quarter 4 · Lesson 153</p><h1>'+title+'</h1></header>'+body+'</article>'+''.join(f'<script src="../assets/{x}.js"></script>' for x in scripts)+'</body></html>'
(R/'lessons'/f'{S}.html').write_text(doc(TITLE,prose(),True))
reference='''## Ranking entry contract
Query population + full candidate catalog → source/data/runtime pins → cutoff audit + negative-sampling rule → validation-only selection → complete rankings + keyed metrics → seed coverage → bounded verdict.

| Quantity | Definition | Distinction |
|---|---|---|
| AP@k | sum(hit at r × prefix precision at r) / min(k,positive count) | Sensitive to ordering |
| MAP@k | Mean query AP | Each query has equal weight |
| Hit@k | Any relevant item in topk | Does not reward all positives |
| Recall@k | Retrieved positives / all positives | Can be low while AP=1 |
| Positive collision | Sampled training candidate belongs to positive set | Not automatically temporal leakage |
| Temporal violation | Observed information later than owner cutoff | Unknown availability is not a pass |

Truth{A,C},ranking[A,B,C]: AP@3=5/6,Hit=1,Recall=1. RemoveB: AP@2 changes from.5to1 without improving the model. Align repeated entities by full(entity,cutoff)keys. Reject duplicate recommendations.

## Released GraphSAGE recipe
Two128-channel sum layers;128/64fanouts; shallow sponsor embeddings; BPR across timestamp-shared512×512negative pairs;Adam.001;20epochs;latest tied validation maximum;full53,241-candidate catalog. Source drops incomplete timestamp batches and caps at2001batches. Five seeds0–4; source/paper differences and feature-arrival limits remain explicit.

## Measured evidence
'''+results+'\n'+audit+'''
## Read / execute / defend
[Lesson](../lessons/0153-recommendation-portfolio.html) · [Protocol](../labs/l153-reproduction.md) · [Entry template](../labs/l153-entry-template.md) · [RelBench Table8](https://arxiv.org/html/2407.20060v1#A2.T8).

Do not equate a checked notebook, pilot or source parity with five completed benchmark fits. Author evidence does not establish learner mastery.
'''
(R/'reference/recommendation-portfolio.html').write_text(doc('Recommendation portfolio reference',reference))

def defs(path):
 s=path.read_text();return {n.name:ast.get_source_segment(s,n) for n in ast.parse(s).body if isinstance(n,ast.FunctionDef)}
functions=defs(P/'relkit/recommendation_l153.py');checks=defs(P/'_check_l153.py')
files={str(p.relative_to(P)):p.read_text() for p in sorted((P/'sources/l153').glob('*')) if p.is_file() and 'before_boundary' not in p.name}
for name in ['_prepare_l153.py','_run_l153.py','_labels_l153.py','_mechanism_l153.py','relkit/recommendation_model_l153.py','relkit/recommendation_l153.py','relkit/batch_audit_l123.py','requirements-l117-runtime.txt']:files[name]=(P/name).read_text()
payload={}
for directory in [E/'pilot']+[E/f'seed-{s}' for s in range(5)]:
 if (directory/'predictions.npz').exists():
  raw=(directory/'predictions.npz').read_bytes();payload[directory.name]=dict(data=base64.b64encode(raw).decode(),sha256=hashlib.sha256(raw).hexdigest())
truth_packet={s:base64.b64encode((E/f'prepared/{s}-truth.json.gz').read_bytes()).decode() for s in ['val','test']}
bootstrap='''# PROVIDED: default CPU audit; full reproduction needs the exact CUDA runtime.
import os,sys
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
from pathlib import Path
import math,statistics,json,io,base64,hashlib,zlib,gzip,copy,time
import numpy as np,pandas as pd,torch
from IPython.display import display
RUN_FULL_REPRODUCTION=False
'''
packet='''# PROVIDED: compressed source/evidence transport; executable pipeline is visible below.
SOURCE_FILES=json.loads(zlib.decompress(base64.b64decode('''+repr(base64.b64encode(zlib.compress(json.dumps(files).encode())).decode())+''')))
P=Path.cwd()
for name,text in SOURCE_FILES.items():
 dest=P/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(text)
sys.path.insert(0,str(P))
AUTHOR_SUMMARY='''+repr(summary)+'''
COST_DECISION='''+repr(decision)+'''
PAYLOAD='''+repr(payload)+'''
TRUTH_PACKET='''+repr(truth_packet)+'''
'''
scoring='''# CHECK: all saved rankings are rescored using your live metric function.
truth={s:json.loads(gzip.decompress(base64.b64decode(v))) for s,v in TRUTH_PACKET.items()}
count=0;measured={};fresh_records=[]
for name,item in PAYLOAD.items():
 raw=base64.b64decode(item['data']);assert hashlib.sha256(raw).hexdigest()==item['sha256'];z=np.load(io.BytesIO(raw));measured[name]={}
 for split in ['val','test']:
  if split+'_pred' not in z:continue
  ranked=[dict(entity=int(e),time=int(t),ranking=list(map(int,row))) for e,t,row in zip(z[split+'_entity'],z[split+'_time'],z[split+'_pred'])]
  scores=ranking_metrics(truth[split],list(reversed(ranked)),10,53241);measured[name][split]=scores;count+=scores['n']
  expected=AUTHOR_SUMMARY['independent_metrics'][name][split]
  for key in ['map','hit','recall']:assert abs(scores[key]-expected[key])<1e-12
 if name.startswith('seed-'):
  record=next(r for r in AUTHOR_SUMMARY['records'] if r['seed']==int(name.split('-')[1]));fresh_records.append(dict(record,scores={s:v['map'] for s,v in measured[name].items()}))
protocol_hash=hashlib.sha256(SOURCE_FILES['sources/l153/protocol.json'].encode()).hexdigest()
if len(fresh_records)==5:
 verdict=portfolio_entry(fresh_records,protocol_hash,{'val':37003,'test':27428})
else:
 try:portfolio_entry(fresh_records,protocol_hash,{'val':37003,'test':27428})
 except ValueError:pass
 else:raise AssertionError('Incomplete evidence accepted')
 verdict=dict(status='INCOMPLETE',paper_comparison='NOT_RUN',reason='Measured cost gate blocks five full fits')
entry=dict(evidence_owner='AUTHOR_PACKET_INDEPENDENTLY_RESCORED',task='rel-trial/site-sponsor-run',metrics=measured,verdict=verdict,protocol_hash=protocol_hash,cost_decision=COST_DECISION,written_defense=None,learner='PENDING_WRITTEN_DEFENSE')
Path('l153-portfolio-entry.json').write_text(json.dumps(entry,indent=2));display(pd.DataFrame({k:v['val'] for k,v in measured.items()}).T)
print('AUTHOR_PACKET_INDEPENDENTLY_RESCORED',count,'rankings;',verdict)
'''
# Full readable model classes are the actual class passed into the runner.
core=(P/'relkit/recommendation_model_l153.py').read_text()
prep=(P/'_prepare_l153.py').read_text().replace(' from _labels_l153 import independent_labels\n','')
labels=(P/'_labels_l153.py').read_text();batch=(P/'relkit/batch_audit_l123.py').read_text()
runner=(P/'_run_l153.py').read_text()
for line in ['from relkit.recommendation_model_l153 import Model\n','from relkit.recommendation_l153 import ranking_metrics,negative_audit\n','from relkit.batch_audit_l123 import audit_batch\n','from _prepare_l153 import sha\n']:runner=runner.replace(line,'')
runner=runner.replace('P=Path(__file__).resolve().parent','P=Path.cwd()')
mechanism=(P/'_mechanism_l153.py').read_text().split("if __name__=='__main__':")[0].replace('from relkit.recommendation_model_l153 import Model\n','')
gate='''# PROVIDED: fresh five-seed lane; time and money are separate from notebook checks.
if RUN_FULL_REPRODUCTION:
 import importlib.metadata as md
 for name,version in {'torch':'2.5.1','torch-geometric':'2.6.1','pytorch-frame':'0.2.3','relbench':'1.1.0','numpy':'1.26.4','pandas':'2.2.3','sentence-transformers':'3.3.1'}.items():
  assert md.version(name).split('+')[0]==version,(name,'Pinned runtime required')
 for name,info in json.loads(SOURCE_FILES['sources/l153/manifest.json'])['sources'].items():
  assert hashlib.sha256((P/'sources/l153'/name).read_bytes()).hexdigest()==info['sha256'],('Changed pinned source',name)
 assert torch.cuda.is_available(),'T4 CUDA runtime with pyg-lib required; allow32GiB host RAM'
 assert COST_DECISION['decision']=='PROCEED' or globals().get('ACKNOWLEDGE_LARGER_MANUAL_RUN',False),'Budget gate STOP: manual full execution requires acknowledging its larger projected resource use. Managed Modal dispatch remains capped at USD10.'
 root=Path('l153-own-runs');root.mkdir(exist_ok=False)
 prepare(root/'prepared',P/'sources/l153');fresh=[]
 for seed in range(5):fresh.append(run_fit(root/'prepared',root/f'seed-{seed}',seed,pilot=False,model_class=Model,metric_fn=ranking_metrics,negative_fn=negative_audit))
 own=dict(evidence_owner='OWN_FRESH_RUN',records=fresh,verdict=portfolio_entry(fresh,protocol_hash,{'val':37003,'test':27428}),written_defense=None)
 Path('l153-own-run-entry.json').write_text(json.dumps(own,indent=2));print(own['verdict'])
else:print('Your fresh five-seed run NOT_RUN. Author evidence is labeled separately.')
'''
for solution in [False,True]:
 cells=[nb.v4.new_markdown_cell('# Lesson 153 · '+TITLE+'\n\n'+('Teacher solution' if solution else 'Student lab')+' · PROVIDED / TODO / CHECK / EXIT. Default: CPU audit of real author evidence. Post-EXIT: full visible released pipeline and separate pinned GPU gate. TierB benchmark; synthetic fixtures only isolate mechanisms. Install torch, torch-geometric, pytorch-frame, numpy and pandas for the CPU lane; use the embedded pinned requirements and Modal recipe for reproduction. No source checkout or pre-existing data folder is needed.'),nb.v4.new_code_cell(bootstrap),nb.v4.new_code_cell(packet,metadata={'tags':['data-payload']})]
 for section in re.split(r'(?=^## )',prose(True),flags=re.M):
  if not section.strip():continue
  cells.append(nb.v4.new_markdown_cell(section))
  for number,name,check in [('4','negative_audit','check_negatives'),('5','ranking_metrics','check_ranking'),('6','portfolio_entry','check_portfolio')]:
   if section.startswith('## '+number):
    code=functions[name] if solution else functions[name].split('\n',1)[0]+'\n    raise NotImplementedError("TODO: '+name+'")'
    cells.extend([nb.v4.new_code_cell('# TODO: implement the live function used below.\n'+code),nb.v4.new_code_cell('# CHECK: unchanged behavioral checks.\n'+checks['rejects']+'\n'+checks[check]+'\n'+check+'('+name+')\nprint("PASS '+name+'")')])
  if section.startswith('## 6'):cells.append(nb.v4.new_code_cell(scoring))
 cells.append(nb.v4.new_markdown_cell('## Post-EXIT · Complete implementation and fresh-run appendix\n\nRead the shared encoders, GraphSAGE and output head below. These are the exact classes passed to the full runner and exercised by the synthetic output/gradient check. The loader and graph-building source are supplied visibly as released primitives; installed versions must match their pinned bytes before training. The visible trainer string is written to the source file that the instrumented runner actually executes. Its mathematical training/selection behavior is preserved.\n\n`RUN_FULL_REPRODUCTION=False` by default. Full manual execution is longer than a teaching notebook and has no monetary meter. The managed Modal operator reserves each allocation and blocks dispatch when the full forecast exceedsUSD10. Historical reconstruction gaps remain even after a full fresh run. Install using `labs/requirements-l117-runtime.txt` plus torch2.5.1 CUDA12.4 and pyg-lib0.4.0+pt25cu124; the complete reproducible image recipe is in `modal/l153_repro.py`.'))
 for name,code in [('Released encoder and model classes',core),('Independent source-label reconstruction',labels),('Fresh graph and task preparation',prep),('Every sampled occurrence audit',batch)]:cells.append(nb.v4.new_code_cell('# PROVIDED: '+name+'\n'+code))
 for name in ['graph.py','loader.py','gnn_link.py']:
  variable={'graph.py':'RELEASED_GRAPH','loader.py':'RELEASED_LOADER','gnn_link.py':'RELEASED_TRAINER'}[name]
  cells.append(nb.v4.new_code_cell('# PROVIDED: full visible released source; this text is supplied to the runner.\n'+variable+' = '+"r'''"+files['sources/l153/'+name]+"'''"+'\n_ = (P/"sources/l153/'+name+'").write_text('+variable+')'))
 cells.extend([nb.v4.new_code_cell('# PROVIDED: source-preserving full runner; live model/metrics/negative audit\n'+runner),nb.v4.new_code_cell('# CHECK: full synthetic neural update and original-source parity\n'+mechanism+'\nmechanism=mechanism_check(P/"sources/l153",model_class=Model)\nprint(mechanism)'),nb.v4.new_code_cell(gate),nb.v4.new_code_cell("report=dict(status='PASS',rankings=count,mechanism=mechanism,full_training='FIVE_FRESH_FITS' if RUN_FULL_REPRODUCTION else 'NOT_RUN',learner='PENDING_WRITTEN_DEFENSE')\nPath('l153-report.json').write_text(json.dumps(report,indent=2))\nprint(report)")])
 for i,c in enumerate(cells):c.id=f'l153-{i:03d}'
 notebook=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}});path=P/('solutions' if solution else '')/f'{S}.ipynb'
 if solution and path.exists():
  old=nb.read(path,4);notebook.metadata=old.metadata
  previous=[c for c in old.cells if c.cell_type=='code'];current=[c for c in notebook.cells if c.cell_type=='code']
  if [c.source for c in previous]==[c.source for c in current]:
   for prev,new in zip(previous,current):
    new.outputs=prev.outputs;new.execution_count=prev.execution_count;new.metadata=prev.metadata
 nb.write(notebook,path)
from nbconvert import HTMLExporter
from nbconvert.preprocessors import TagRemovePreprocessor
executed=nb.read(P/'solutions'/f'{S}.ipynb',4)
if all(c.execution_count is not None for c in executed.cells if c.cell_type=='code'):
 exporter=HTMLExporter(template_name='lab');exporter.register_preprocessor(TagRemovePreprocessor(remove_input_tags={'data-payload'}),enabled=True);html,_=exporter.from_notebook_node(executed);(P/'html'/f'{S}.html').write_text(html)
print('Built153 lesson, reference and portable student/solution notebooks')

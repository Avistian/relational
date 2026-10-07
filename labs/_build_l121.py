"""Deterministic lesson, standalone student/reference notebooks, quick reference."""
import ast,base64,json,re
from pathlib import Path
import nbformat as nbf
from nbconvert import HTMLExporter
from nbconvert.filters.markdown import markdown2html_mistune as render
P=Path(__file__).resolve().parent;R=P.parent;S='0121-history-relational-ml';TITLE='History of relational ML: rules, features, and graphs'
FIGS={'lineage':('l121','Historical anchors and representation choices; reading order does not establish replacement.'),'grouping':('l121','Same line values [2,8], different order ownership. Flat [count,sum,max] agrees; nested sums of squares yield 68 and 100.'),'architecture':('l118','Selected Home Credit GCN: table-specific encoders, one shared GCN, graph-local gate/value pooling, and two logits. Reused from Lesson 118.')}
TASKS={'exists_late':('check_rule','Evaluate existence of a matching late order; preserve query order.'),'aggregate_at':('check_aggregate','Filter with both clocks, then construct count, sum, and maximum.'),'path_signal':('check_path','Accumulate lines into orders, square order totals, then accumulate into customers.')}
checks=(P/'_check_l121.py').read_text();CHECKS={n.name:ast.get_source_segment(checks,n) for n in ast.parse(checks).body if isinstance(n,ast.FunctionDef)}
def results():
 pilot=json.loads((P/'_pilot_l121_results.json').read_text());audit=json.loads((P/'_data_l121_results.json').read_text());neo=json.loads((P/'_neo4j_l121_results.json').read_text()) if (P/'_neo4j_l121_results.json').exists() else {'status':'NOT_CHECKED'}
 return f'''**Full five-fold Home Credit experiment: NOT_RUN.** No fresh held-out AUROC is claimed.

| Evidence | Measured result | Boundary |
|---|---|---|
| Original course grouping computation | 68 versus 100 | Fixed operator, not trained GCN accuracy |
| Raw previous-application key universe | {audit['full_previous_key_count']:,} keys scanned | {audit['sample_graphs']} selected applicants |
| Missing previous references in that sample | {audit['globally_missing_previous_references']} globally absent; {audit['cross_owner_references']} cross-owner | Missing relation is not a dropped applicant edge |
| Original Neo4j extraction comparison | {neo['status']} after empty→null correction | {neo.get('graphs',0)} applicants; full-population equivalence NOT_CHECKED |
| Raw categorical cells corrected | {neo.get('empty_to_null_cells_corrected',0)} | Encoded tensors unchanged; original raw mismatch preserved in audit |
| Cached tensor/output/gradient check | EXACT on eight real graphs | Controlled computation, current runtime |
| Fresh cached timing pilot | 1,024 graphs; 130,886 nodes | Fixed reconstructed batch, no score evaluation |
| Cached batch collation median | {__import__('statistics').median(pilot['cached_collation_seconds']):.4f} seconds | Original preparation {pilot['original_preparation_seconds']:.3f} seconds |
| Five-fold maximum-schedule projection | USD {pilot['projected_five_fold_300_epoch_usd']:.2f} | Excludes full cache construction, disk IO and overhead |
| Aggregate cap / overhead reserve | USD 10 / USD 2 | Pilot worker reservation USD 0.406296 |

The improved projection still exceeds the cap. **Decision: do not launch five folds.** Pilot function-body resource estimate: USD {pilot['resource_estimate_usd']:.6f}; unitemized startup/build costs are separate. Historical training identity and whole-paper parity remain NOT_ESTABLISHED. Live Colab and deployment remain NOT_CHECKED.

[Extraction audit](../labs/_neo4j_l121_results.json) · [Raw-key audit](../labs/_data_l121_results.json) · [Cache check](../labs/_cache_check_l121_results.json) · [Timing evidence](../labs/_pilot_l121_results.json) · [Budget](../labs/_budget_l121.json).
'''
def prose(portable=False):
 text=(R/'lessons/content'/f'{S}.md').read_text().replace('[[RESULTS]]',results())
 for name,(folder,caption) in FIGS.items():
  source='data:image/png;base64,'+base64.b64encode((P/f'figures/{folder}/{name}.png').read_bytes()).decode() if portable else f'../labs/figures/{folder}/{name}.svg'
  image_style=' style="width:100%;min-width:620px;max-width:none"' if portable else ''
  text=text.replace('[[FIG:'+name+']]',f'<figure class="stream-figure" tabindex="0" style="max-width:100%;overflow:auto"><img src="{source}" alt="{caption}"{image_style}><figcaption>{caption} Scroll horizontally on narrow screens.</figcaption></figure>')
 for token,id,fallback in [('WARMUP','warmup','Without notes: what does a foreign key preserve? Why can a flat feature vector lose a distinction? Why is author execution not learner mastery?'),('GROUP_WIDGET','group','Static trace: A has two order totals [2,8], B has one [10]. Flat vectors agree at [2,10,8]. Nested sum of squares gives 68 versus 100. Splitting B into two orders changes its nested result to 68.'),('TEACHBACK','teachback','Explain why a nested engineered feature can repair this flat collision, and why relational learning predates this GNN. Compare your explanation with the historical map.')]:
  text=text.replace('[['+token+']]',fallback if portable else f'<div id="l121-{id}"></div><noscript><p>{fallback}</p></noscript>')
 if portable:
  text=text.replace('](../','](https://avistian.github.io/relational/');text=re.sub(r'\]\((\d{4}-[^)]+\.html)\)',r'](https://avistian.github.io/relational/lessons/\1)',text)
 return text

def document(title,body,interactive=False):
 html=re.sub(r'<table([^>]*)>',r'<div class="stream-scroll" tabindex="0"><table\1>',render(body)).replace('</table>','</table></div>')
 styles=''.join(f'<link rel="stylesheet" href="../assets/{n}.css">' for n in ['lesson','event-snapshot','reproduction'])
 scripts=''.join(f'<script src="../assets/{n}.js"></script>' for n in ['retrieval-pool','retrieval-bank','teachback','l121-lesson']) if interactive else ''
 return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title>'+styles+'</head><body><article><nav><a href="../index.html">Course</a> · <a href="../lessons/0120-year-3-exit-exam.html">Lesson 120</a></nav><header><p class="stream-kicker">Year 4 · Quarter 1 · Lesson 121</p><h1>'+title+'</h1></header>'+html+'</article>'+scripts+'</body></html>'
(R/'lessons'/f'{S}.html').write_text(document(TITLE,prose(),True))
bootstrap='''# @colab-bootstrap: install only missing default-lab packages and report versions.
import importlib.util,importlib.metadata,subprocess,sys
required={'numpy':'numpy','torch':'torch','sklearn':'scikit-learn'}
missing=[package for module,package in required.items() if importlib.util.find_spec(module) is None]
if missing:subprocess.check_call([sys.executable,'-m','pip','install',*missing])
print('Actual runtime:',sys.version)
print({n:importlib.metadata.version(n) for n in ['numpy','torch','scikit-learn']})
import json,sqlite3
from pathlib import Path
'''
for solution in [False,True]:
 cells=[nbf.v4.new_markdown_cell('# Lesson 121 · '+TITLE+'\n\n'+('Teacher solution' if solution else 'Student lab')+' · Three live TODO/CHECK tasks, prior-art map, and a separate complete visible Home Credit model/trainer. Default execution is COURSE_ONLY; full five-fold reproduction NOT_RUN.'),nbf.v4.new_code_cell(bootstrap)]
 for section in re.split(r'(?=^## )',prose(True),flags=re.M):
  if section.strip():cells.append(nbf.v4.new_markdown_cell(section))
 cells.append(nbf.v4.new_markdown_cell('## Live tasks · original course mechanisms\n\nImplement each function before its CHECK. These are rule evaluation, a feature recipe, and a fixed differentiable ownership path. They are not full ILP/DFS search or the paper GCN. The RUN calls your definitions directly.'))
 for chunk in re.split(r'^# %% ',(P/'relkit/history_l121.py').read_text(),flags=re.M)[1:]:
  heading,body=chunk.split('\n',1);task=next((k for k in TASKS if 'def '+k+'(' in body),None)
  cells.append(nbf.v4.new_markdown_cell('### '+heading+('\n\n**TODO:** '+TASKS[task][1] if task else ' · PROVIDED')))
  if task and not solution:
   node=next(n for n in ast.parse(body).body if isinstance(n,ast.FunctionDef) and n.name==task);body=body.splitlines()[node.lineno-1]+'\n    raise NotImplementedError("TODO: '+task+'")\n'
  cells.append(nbf.v4.new_code_cell(body.strip()))
  if task:
   check=TASKS[task][0];cells.append(nbf.v4.new_code_cell(CHECKS[check]+'\n'+check+'('+task+')\nprint("PASS: '+task+'")'))
 cells.extend([nbf.v4.new_markdown_cell('## RUN · compare the actual live computations\n\nPredict the day-5/day-7 vectors and 68/100 outputs before running. All values are authored course examples; no predictive model comparison is implied.'),nbf.v4.new_code_cell((P/'_teaching_l121.py').read_text().split('# NOTEBOOK_RUN\n')[1]),nbf.v4.new_markdown_cell('## PROVIDED · full selected Home Credit implementation\n\nThese annotated definitions are the visible L118 modern port used by the source checks and real-data pilot. The original source is MIT licensed and pinned at 57195ccab62d23dcbcac1a317f8a9811a9fd6cb5. The DGL primitive references are Apache-2.0. Read the protocol and original licenses in the repository. The full trainer requires trusted locally prepared data and is not called by default.')])
 for chunk in re.split(r'^# %% ',(P/'relkit/cvitkovic_l118.py').read_text(),flags=re.M)[1:]:
  heading,body=chunk.split('\n',1);cells.extend([nbf.v4.new_markdown_cell('### PROVIDED · '+heading),nbf.v4.new_code_cell(body.strip())])
 cache=(P/'relkit/cache_l121.py').read_text().replace('from relkit.cvitkovic_l118 import encode_features,computation_edges\n','')
 cells.extend([nbf.v4.new_markdown_cell('### PROVIDED · deterministic encoding cache\n\nThe fresh pilot uses this cache. It preserves raw-to-tensor conversion and graph order. Learned encoders, messages and dropout are still executed by the model.'),nbf.v4.new_code_cell(cache),nbf.v4.new_markdown_cell('## Optional reading aid · execute the actual model on four synthetic graphs\n\nThis small smoke experiment exercises the provided GCN and training path; it is separate from your three historical-mechanism tasks. Falling training loss is not held-out performance or a Home Credit result.'),nbf.v4.new_code_cell((P/'_teaching_l118.py').read_text().split('# NOTEBOOK_RUN\n')[1].replace("Path('l118-task-report.json').write_text(json.dumps(report,indent=2));print(report)","print('Separate synthetic GCN smoke check:', report)")),nbf.v4.new_markdown_cell('## EXIT and full-data NEXT STEP\n\nSubmit the three live functions, `l121-task-report.json`, and your sourced prior-art map. Explain both a fair flat-feature repair and the full reproduction blockers. Learner status remains PENDING_WRITTEN_DEFENSE.\n\nThe full runner is a plan by default and refuses missing graphs or insufficient aggregate budget. It does not launch from this notebook.\n\n```bash\n.venv/bin/python labs/_run_l121.py --prepared "$HOME/RDB_data/homecreditdefaultrisk/preprocessed_datapoints"\n```\n\nSee the [complete preparation/execution protocol](https://avistian.github.io/relational/labs/l121-reproduction.md). All five folds NOT_RUN. Live Colab NOT_CHECKED.')])
 for i,c in enumerate(cells):c.id=f'l121-{i:03d}'
 nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'},'language_info':{'name':'python'}});path=P/('solutions' if solution else '')/f'{S}.ipynb'
 if solution and path.exists():
  old=nbf.read(path,as_version=4);aa=[c for c in nb.cells if c.cell_type=='code'];bb=[c for c in old.cells if c.cell_type=='code']
  if [c.source for c in aa]==[c.source for c in bb]:
   nb.metadata=old.metadata
   for a,b in zip(aa,bb):a.outputs=b.outputs;a.execution_count=b.execution_count;a.metadata=b.metadata
 nbf.write(nb,path)
 if solution and all(c.execution_count is not None for c in nb.cells if c.cell_type=='code'):
  html,_=HTMLExporter(template_name='lab').from_notebook_node(nb);(P/'html'/f'{S}.html').write_text(html)
ref='''## Ask where representation work happens

ILP learns relational rules from examples/background knowledge. Executing a supplied rule is not learning it. Propositionalization constructs a fixed attribute vector. DFS composes feature primitives along relationships; “deep” refers to composition, not necessarily neural networks. Neural relational learning predates Cvitkovic: include Lam 2018. Cvitkovic's workshop2019 and expanded2020 experiments must be distinguished. Fey/RelBench2024 adds a temporal typed-graph blueprint and shared evaluation.

## One numeric memory hook

Both customers have line amounts[2,8], hence leaf features[count,sum,max]=[2,10,8]. Separate order totals[2,8] give sum of squares68. One combined order[10] gives100. A nested engineered feature repairs the collision. This is a course calculation, not a GNN superiority result.

## Read the actual model

Home Credit: table encoders → one shared256-wide GCN → graph-local gate/value pooling → two logits. Compare this with the fixed square exercise before making a model claim. Inspect the full implementation in the notebook.

## Evidence boundary

Selected target: expanded2020 Table4 Home Credit GCN .780±.004 AUROC across five folds. Fresh cached timing projection15.70USD exceeds10USD cap before overhead. Full experiment NOT_RUN. Source checks, sample extraction and cached computation parity do not establish predictive performance. Historical/whole-paper parity NOT_ESTABLISHED. Learner PENDING_WRITTEN_DEFENSE.

[Lesson](../lessons/0121-history-relational-ml.html) · [Prior-art template](../labs/l121-prior-art-map.md) · [Protocol](../labs/l121-reproduction.md) · [Cvitkovic paper](https://arxiv.org/abs/2002.02046v1) · [DFS paper](https://www.jmaxkanter.com/papers/DSAA_DSM_2015.pdf).
'''
(R/'reference/history-relational-ml.html').write_text(document('History of relational ML · quick reference',ref))
print('Built L121 lesson, standalone notebooks, and reference')

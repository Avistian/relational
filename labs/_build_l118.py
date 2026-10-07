"""One source for the lesson, standalone notebooks and reference."""
import ast,base64,json,re
from pathlib import Path
import nbformat as nbf
from nbconvert import HTMLExporter
from nbconvert.filters.markdown import markdown2html_mistune as render
P=Path(__file__).resolve().parent;R=P.parent;S='0118-cvitkovic-relational-gnn';TITLE='Cvitkovic: relational graphs before modern RDL'
CANONICAL=(P/'relkit/cvitkovic_l118.py').read_text()
CAP={'extraction':'One fixed graph, three algorithm states. Incoming closure selects orders and lines; outgoing closure adds Country but does not restart incoming expansion.','architecture':'The selected Home Credit GCN: seven table-specific initializers, one shared message layer, gate/value pooling branches and a two-logit prediction. Source metadata is globally prepared.','normalization':'Synthetic one-coordinate example, before SELU: self-loop degrees [3,2,2] yield root value 2.783; this is not the neighbor mean.'}
TASKS={'rdb_to_graph':('check_extract','Select incoming closure, then outgoing closure, preserving all induced edges.'),'normalized_sum':('check_normalize','Accumulate degree-normalized source messages for every receiver.'),'attention_pool':('check_pool','Normalize gates within each graph and combine its value vectors.')}
checktext=(P/'_check_l118.py').read_text();CHECKS={n.name:ast.get_source_segment(checktext,n) for n in ast.parse(checktext).body if isinstance(n,ast.FunctionDef)}
def results():
 p=json.loads((P/'_pilot_l118_results.json').read_text());a=json.loads((P/'_source_check_l118_results.json').read_text())
 return f'''### Author evidence: source checks and real-data pilot

**Full selected Home Credit experiment: NOT_RUN.** The real-data timing pilot exceeded the approved aggregate cost projection; no paper AUROC has been measured here.

| Check or measurement | Result | Interpretation |
|---|---:|---|
| Original Python-module forward discrepancy | {a['forward_max_abs_error']:.3g} | Controlled inputs, modern runtime and dense adapter |
| Original Python-module gradient discrepancy | {a['gradient_max_abs_error']:.3g} | Numerical operator agreement, not full training |
| Algorithm 1 random target cases | {a['random_extraction_cases']} | Independent closure oracle |
| Real pilot graphs / nodes | {p['graphs']:,} / {p['nodes']:,} | One fixed batch from real Home Credit rows |
| Warm training-step median | {__import__('statistics').median(p['training_step_seconds'][1:]):.3f} seconds | GPU compute after preparation |
| Batch preparation | {p['batch_preparation_seconds']:.3f} seconds | Dominant cost in the current port |
| Five-fold 300-epoch projection | USD {p['projected_five_fold_300_epoch_usd']:.2f} | Excludes full preparation, final tests and overhead |
| Approved aggregate ceiling | USD 10 | Includes all attempts and overhead |

The pilot has reconstructed CSV graphs, not verified Neo4j graph identity. It is a cost probe. Repeated updates to that batch establish neither validation quality nor a paper result. Caching could reduce the estimate, but no optimized timing has been measured. [Pilot evidence](../labs/_pilot_l118_results.json) · [Data audit](../labs/_data_l118_results.json) · [Full preflight](../labs/_preflight_l118_results.json).
'''
def prose(portable=False):
 text=(R/'lessons/content'/f'{S}.md').read_text().replace('[[RESULTS]]',results())
 for name,caption in CAP.items():
  source='data:image/png;base64,'+base64.b64encode((P/f'figures/l118/{name}.png').read_bytes()).decode() if portable else f'../labs/figures/l118/{name}.svg'
  extra=' style="max-width:100%;overflow:auto"' if portable else ''
  image_style=' style="min-width:620px;width:100%;max-width:none"' if portable else ''
  text=text.replace('[[FIG:'+name+']]',f'<figure class="stream-figure" tabindex="0"{extra}><img src="{source}" alt="{caption}"{image_style}><figcaption>{caption} Scroll horizontally on narrow screens.</figcaption></figure>')
 for token,id,fallback in [('WARMUP','warmup','Without notes: distinguish schema nodes from REG nodes. Why add reverse edges? What does a prediction timestamp constrain?'),('EXTRACTION_WIDGET','extraction','Static trace: Algorithm 1 selects Customer A, Order A, Country, Line A. Two undirected hops additionally select Customer B. A course cutoff at 7 removes Line A (available at 9), leaving three versus four nodes.'),('TEACHBACK','teachback','Explain why graph-level pooling lets a selected distant node affect a one-layer GCN prediction even without sending its information all the way to the target.')]:
  text=text.replace('[['+token+']]',fallback if portable else f'<div id="l118-{id}"></div><noscript><p>{fallback}</p></noscript>')
 if portable:
  text=text.replace('](../','](https://avistian.github.io/relational/');text=re.sub(r'\]\((\d{4}-[^)]+\.html)\)',r'](https://avistian.github.io/relational/lessons/\1)',text)
 return text

def document(title,body,interactive=False):
 html=re.sub(r'<table([^>]*)>',r'<div class="stream-scroll" tabindex="0"><table\1>',render(body)).replace('</table>','</table></div>')
 styles=''.join(f'<link rel="stylesheet" href="../assets/{name}.css">' for name in ['lesson','event-snapshot','reproduction','rdb-extraction'])
 scripts=''.join(f'<script src="../assets/{name}.js"></script>' for name in ['retrieval-pool','retrieval-bank','teachback','rdb-extraction-viz','l118-lesson']) if interactive else ''
 return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title>'+styles+'</head><body><article><nav><a href="../index.html">Course</a> · <a href="../lessons/0117-rdl-bridge.html">Lesson 117</a></nav><header><p class="stream-kicker">Year 3 · Quarter 4 · Lesson 118</p><h1>'+title+'</h1></header>'+html+'</article>'+scripts+'</body></html>'
(R/'lessons'/f'{S}.html').write_text(document(TITLE,prose(),True))
bootstrap='''# @colab-bootstrap: installs missing packages only; reports the actual runtime.
import importlib.util,importlib.metadata,subprocess,sys
required={'numpy':'numpy','torch':'torch','sklearn':'scikit-learn'}
missing=[package for module,package in required.items() if importlib.util.find_spec(module) is None]
if missing:subprocess.check_call([sys.executable,'-m','pip','install',*missing])
print('Actual runtime:',sys.version)
print({name:importlib.metadata.version(name) for name in ['numpy','torch','scikit-learn']})
'''
run=(P/'_teaching_l118.py').read_text().split('# NOTEBOOK_RUN\n',1)[1]
for solution in [False,True]:
 cells=[nbf.v4.new_markdown_cell('# Lesson 118 · '+TITLE+'\n\n'+('Teacher solution' if solution else 'Student lab')+' · Three live TODO/CHECK tasks. The default run is a synthetic computation exercise. Author source checks and the real Home Credit timing pilot are separate evidence. Five-fold paper experiment NOT_RUN.'),nbf.v4.new_code_cell(bootstrap),nbf.v4.new_markdown_cell('## Execution contract\n\nThe default notebook needs only PyTorch, NumPy and scikit-learn. It downloads no competition data and starts no paid job. The full modern-port runtime is pinned separately in `requirements-l118-runtime.txt`; the historical author runtime is not reconstructed. Portable figures are embedded, but live Colab remains NOT_CHECKED.')]
 for section in re.split(r'(?=^## )',prose(True),flags=re.M):
  if section.strip():cells.append(nbf.v4.new_markdown_cell(section))
 cells.append(nbf.v4.new_markdown_cell('## Visible implementation and live tasks\n\nRead the encoding and model alongside Sections 3 and 5. The full prepared-data loader, model, split construction and trainer follow. These are definitions; the full trainer does not run automatically. The final course experiment calls your three functions. Original source and licenses are linked in the protocol.'))
 for chunk in re.split(r'^# %% ',CANONICAL,flags=re.M)[1:]:
  heading,body=chunk.split('\n',1);task=next((k for k in TASKS if 'def '+k+'(' in body),None)
  cells.append(nbf.v4.new_markdown_cell('### '+heading+('\n\n**TODO:** '+TASKS[task][1] if task else ' · PROVIDED')))
  if task and not solution:
   node=next(n for n in ast.parse(body).body if isinstance(n,ast.FunctionDef) and n.name==task)
   body=body.splitlines()[node.lineno-1]+'\n    raise NotImplementedError("TODO: '+task+'")\n'
  cells.append(nbf.v4.new_code_cell(body.strip()))
  if task:
   check=TASKS[task][0];cells.append(nbf.v4.new_code_cell(CHECKS[check]+'\n'+check+'('+task+')\nprint("PASS: '+task+'")'))
 cells.extend([nbf.v4.new_markdown_cell('## RUN · synthetic computation experiment\n\nPredict which rows extraction retains and whether the related-node signal can lower loss. Four tiny graphs isolate wiring and gradient flow. **This is COURSE_ONLY**, not a Home Credit subset or paper benchmark. The graph constructor and forward pass use all three live functions.'),nbf.v4.new_code_cell(run),nbf.v4.new_markdown_cell('## EXIT\n\nSubmit your three functions, `l118-task-report.json`, and the six-part written defense from Section 8. Explain why the source check and pilot do not establish a reproduced paper AUROC. **PENDING_WRITTEN_DEFENSE**.'),nbf.v4.new_markdown_cell('## NEXT STEP · full named experiment\n\nThe complete model and trainer are visible above. The repository runner adds full-population preflight, data fingerprints, aggregate budget gating and prediction/identity audits. Prepare the original Neo4j data using the protocol, then run the command below from the repository root. It defaults to a plan; `--execute` is explicit and still refuses missing data or an insufficient budget.\n\n```bash\n.venv/bin/python labs/_run_l118.py --prepared "$HOME/RDB_data/homecreditdefaultrisk/preprocessed_datapoints"\n```\n\n[Exact preparation, execution, and deviation ledger](https://avistian.github.io/relational/labs/l118-reproduction.md). Five-fold Home Credit GCN target: 0.780 +/- 0.004 AUROC. Current full-data status: **NOT_RUN**. No code cell launches cloud training.')])
 for i,c in enumerate(cells):c.id=f'l118-{i:03d}'
 nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'},'language_info':{'name':'python'}});path=P/('solutions' if solution else '')/f'{S}.ipynb'
 if solution and path.exists():
  old=nbf.read(path,as_version=4);aa=[c for c in nb.cells if c.cell_type=='code'];bb=[c for c in old.cells if c.cell_type=='code']
  if [c.source for c in aa]==[c.source for c in bb]:
   nb.metadata=old.metadata
   for a,b in zip(aa,bb):a.outputs=b.outputs;a.execution_count=b.execution_count;a.metadata=b.metadata
 nbf.write(nb,path)
 if solution and all(c.execution_count is not None for c in nb.cells if c.cell_type=='code'):
  html,_=HTMLExporter(template_name='lab').from_notebook_node(nb);(P/'html'/f'{S}.html').write_text(html)
ref='''## The essential distinction

Both Cvitkovic and Fey map rows to typed nodes and foreign-key references to graph edges. Compare context selection, time handling, message functions, readout and evaluation before claiming a difference.

## RDBToGraph

Foreign-key direction: referencing row → referenced row. Start with target. Finish incoming closure. Then finish outgoing closure. Keep all induced edges. Do not alternate the phases; do not add reverse computation edges before selecting nodes. Source Home Credit uses a separate undirected two-hop query.

## Selected model

Table-specific categorical embeddings and robust scalar encodings → d→4d→256 SELU initializer. Add reverse edges and self loops. One shared GCN: symmetric degree-normalized sum, 256×256 weight, bias, SELU and dropout .5. Two readout branches: scalar gate and 256-wide value. Softmax gates within each graph; weighted sum of values; linear two-logit head. Pooling reads every selected node, not just the target.

## Protocol to remember

307,511 labeled applications. Five shuffled folds, split seed14; 15% of trainval held for validation. Model seed1234 each fold. Batch1024, AdamW1e-4, zero weight decay, max300epochs, patience50. Validate before training; first-best AUROC. Global released feature metadata is not fold-local preprocessing. Table4 target: AUROC .780±.004 across folds.

## Evidence

Full-data reproduction **NOT_RUN**. Controlled original Python-module agreement and complete fold-identity checks PASS. Real-data pilot is resource calibration, not predictive evidence; current projection exceeds USD10. Historical DGL runtime and live Colab NOT_CHECKED. Learner PENDING_WRITTEN_DEFENSE.

[Lesson](../lessons/0118-cvitkovic-relational-gnn.html) · [Protocol](../labs/l118-reproduction.md) · [Expanded paper](https://arxiv.org/abs/2002.02046v1) · [Source identities](../labs/_sources_l118.json).
'''
(R/'reference/cvitkovic-relational-gnn.html').write_text(document('Cvitkovic relational GNN · quick reference',ref))
print('Built L118 lesson, portable notebooks and reference')

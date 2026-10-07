"""One-source lesson, portable notebooks, reference and measured evidence."""
import ast,base64,json,re
from pathlib import Path
import nbformat as nbf
from nbconvert import HTMLExporter
from nbconvert.filters.markdown import markdown2html_mistune as render
P=Path(__file__).resolve().parent;R=P.parent;S='0117-rdl-bridge';TITLE='RDL bridge: from database rows to predictions'
CANONICAL=(P/'relkit/rdl_l117.py').read_text()
CAP={'graphs':'Schema nodes denote tables; REG nodes denote individual rows; a query extracts eligible context using one fixed prediction time.','queries':'input_id selects a query and its target; n_id identifies a database row; batch identifies which query owns the sampled copy.','architecture':'The released regression path: query-specific sampling, per-table encoders and relative ages, two typed sum-aggregation layers, seed readout, loss and validation selection.'}
TASKS={'foreign_key_edges':('check_edges','Map arbitrary keys to row positions; preserve repeated references and reject invalid identities.'),'temporal_nodes':('check_cutoff','Expand permitted neighbors while holding the original prediction time fixed.'),'query_targets':('check_targets','Attach future targets in sampled query order, preserving repeated entities.')}
checktext=(P/'_check_l117.py').read_text();CHECKS={n.name:ast.get_source_segment(checktext,n) for n in ast.parse(checktext).body if isinstance(n,ast.FunctionDef)}
def results():
 path=P/'evidence/l117/summary.json'
 if not path.exists():return '**Author experiment is running. No completed result is claimed yet.**'
 s=json.loads(path.read_text());txt='**Five complete full-data runs. Author-reference evidence, separate from student execution.**\n\n| Split | Mean MAE | Sample seed SD | Paper mean | Descriptive verdict |\n|---|---:|---:|---:|---|\n'
 for pop in ['val','test']:
  x=s['metrics'][pop];txt+=f"| {pop} | {x['mean']:.5f} | {x['sample_sd']:.5f} | {x['target']:.3f} | {x['verdict']} |\n"
 txt+=f"\nAll **{s['evaluated_queries']:,} final query predictions** were independently rescored and compared with the original released model on the same sampled inputs. Maximum raw-output discrepancy: **{s['maximum_original_output_error']:.3g}**, within the predeclared numerical tolerance. The complete graph has **74,063 rows** at the release test cutoff.\n\n"
 txt+=f"Recorded pilot plus five-fit worker resource estimate: **USD{s['worker_resource_usd']:.4f}**. Build/startup/storage overhead is not itemized; the plan reserves USD3.499264 for overhead beneath the USD10 aggregate cap. See [all seeds and protocol status](../labs/evidence/l117/summary.json).\n\n"
 txt+='**The selected released-protocol experiment is COMPLETE. Historical identity and whole-paper parity remain NOT_ESTABLISHED.** Score verdicts compare means only; they are not statistical equivalence tests. Live Colab remains NOT_CHECKED.'
 return txt

def prose(portable=False):
 text=(R/'lessons/content'/f'{S}.md').read_text().replace('[[RESULTS]]',results())
 for name,caption in CAP.items():
  source='data:image/png;base64,'+base64.b64encode((P/f'figures/l117/{name}.png').read_bytes()).decode() if portable else f'../labs/figures/l117/{name}.svg'
  text=text.replace('[[FIG:'+name+']]',f'<figure class="stream-figure" tabindex="0"><img src="{source}" alt="{caption}"><figcaption>{caption}</figcaption></figure>')
 for token,id,fallback in [('WARMUP','warmup','Without notes: what does a row encoder produce? What is the difference between a global node ID and a sampled local position? Which timestamp bounds a two-hop temporal neighborhood?'),('CUTOFF_WIDGET','cutoff','Static trace: query at 7, two hops. Include the timeless seed and dimension plus event A at time 5. Exclude event B at 11. Move the query to 11: include both results. Bypassing filtering at 7 improperly includes one future row.'),('TEACHBACK','teachback','Explain why two predictions for the same driver at different dates require distinct query labels and potentially distinct computation graphs. Include the role of input_id, n_id and batch.')]:
  text=text.replace('[['+token+']]',fallback if portable else f'<div id="l117-{id}"></div><noscript><p>{fallback}</p></noscript>')
 if portable:
  text=text.replace('](../','](https://avistian.github.io/relational/');text=re.sub(r'\]\((\d{4}-[^)]+\.html)\)',r'](https://avistian.github.io/relational/lessons/\1)',text)
 return text

def document(title,body,interactive=False):
 html=re.sub(r'<table([^>]*)>',r'<div class="stream-scroll" tabindex="0"><table\1>',render(body)).replace('</table>','</table></div>')
 styles=''.join(f'<link rel="stylesheet" href="../assets/{name}.css">' for name in ['lesson','event-snapshot','reproduction'])
 scripts=''.join(f'<script src="../assets/{name}.js"></script>' for name in ['retrieval-pool','retrieval-bank','teachback','reg-query-viz','l117-lesson']) if interactive else ''
 return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title>'+styles+'</head><body><article><nav><a href="../index.html">Course</a> · <a href="../lessons/0116-debug-gnn-training.html">Lesson 116</a></nav><header><p class="stream-kicker">Year 3 · Quarter 4 · Lesson 117</p><h1>'+title+'</h1></header>'+html+'</article>'+scripts+'</body></html>'
(R/'lessons'/f'{S}.html').write_text(document(TITLE,prose(),True))
bootstrap='''# Install missing packages only. Full benchmark runtime is pinned separately.
import importlib.util,importlib.metadata,subprocess,sys
required={'numpy':'numpy','pandas':'pandas','pyarrow':'pyarrow','torch':'torch','torch_frame':'pytorch-frame==0.2.3','torch_geometric':'torch-geometric==2.6.1','relbench':'relbench==1.1.0','pooch':'pooch'}
missing=[package for module,package in required.items() if importlib.util.find_spec(module) is None]
if missing:subprocess.check_call([sys.executable,'-m','pip','install',*missing])
print('Actual notebook runtime:',sys.version)
print({k:importlib.metadata.version(k) for k in ['numpy','pandas','torch','pytorch-frame','torch-geometric','relbench']})
'''
run=(P/'_teaching_l117.py').read_text().split('# NOTEBOOK_RUN\n',1)[1]
text_spec=json.loads((P/'sources/l117/text_model.json').read_text())
# The full gate needs no repository checkout; original model source is embedded for replay.
gate='''# Optional five-seed full-data experiment. Native pyg-lib and a compatible GPU runtime required.
RUN_FULL_REPRODUCTION = False
if RUN_FULL_REPRODUCTION:
    from relbench.datasets import get_dataset
    from relbench.tasks import get_task
    from relbench.modeling.utils import get_stype_proposal
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(1)
    seed_everything(42)
    dataset=get_dataset('rel-f1',download=True);db=dataset.get_db()
    task=get_task('rel-f1','driver-position',download=True)
    for name,expected in [('db.zip','ec31a4e1bc2b2f9c36c05fcd3dfe2a40a506f335dc51ce79c3ec8bb40feb1482'),('tasks/driver-position.zip','775b28a51604169539bbe712a2f0d15158c112bc6abf316cdd0995087a7ae03e')]:
        assert hashlib.sha256((Path(dataset.cache_dir)/name).read_bytes()).hexdigest()==expected
    types=get_stype_proposal(db)
    text_model=SentenceTransformer(TEXT_SPEC['model'],revision=TEXT_SPEC['revision'],device='cpu')
    def embed(strings):return torch.from_numpy(text_model.encode(strings,show_progress_bar=False))
    data,stats=make_pkey_fkey_graph(db,types,TextEmbedderConfig(text_embedder=embed,batch_size=256))
    reference_namespace={}
    exec(ORIGINAL_MODEL_SOURCE,reference_namespace)
    records=[fit_rdl(data,stats,task,seed,Path('l117-full')/f'seed-{seed}',10,'cuda' if torch.cuda.is_available() else 'cpu',reference_namespace['Model']) for seed in range(5)]
    print({s:float(np.mean([r['scores'][s] for r in records])) for s in ['val','test']})
else:
    print('Full experiment NOT_RUN in this kernel. Author five-seed evidence is separate.')
'''
gate='TEXT_SPEC = '+repr(text_spec)+'\nORIGINAL_MODEL_SOURCE = '+repr((P/'sources/l117/model.py').read_text())+'\n'+gate
for solution in [False,True]:
 cells=[nbf.v4.new_markdown_cell('# Lesson 117 · '+TITLE+'\n\n'+('Teacher solution' if solution else 'Student lab')+' · Three live TODO/CHECK tasks. The default run applies your functions to genuine F1 rows. The full five-seed benchmark is a separate optional gate.'),nbf.v4.new_code_cell(bootstrap),nbf.v4.new_markdown_cell('## Execution contract\n\nThe default lab downloads less than 1 MB of hash-checked data and uses NumPy/pandas for the learner tasks. Full model definitions are visible and executed as definitions; they do not train until you enable the final gate. The author GPU environment is Python 3.11, torch 2.5.1+cu124, PyG 2.6.1, Frame 0.2.3, RelBench 1.1.0 and pyg-lib 0.4.0. Existing notebook packages are reported, not silently claimed identical. Install sentence-transformers and compatible native pyg-lib before opting into the full gate; use the pinned environment and bounded Modal runner in the protocol for exact author commands. Live Colab NOT_CHECKED. No cloud job is launched automatically.')]
 for section in re.split(r'(?=^## )',prose(True),flags=re.M):
  if section.strip():cells.append(nbf.v4.new_markdown_cell(section))
 cells.append(nbf.v4.new_markdown_cell('## Visible implementation and live tasks\n\nThe full relational model, graph constructor, sampler configuration, loss, checkpoint selection and trainer are below. RelBench-derived classes are MIT licensed. Frame and PyG operators remain named library dependencies; their pinned source is distributed alongside the lesson. Each task CHECK uses your live function, and the real-data RUN cell depends on all three.'))
 for chunk in re.split(r'^# %% ',CANONICAL,flags=re.M)[1:]:
  heading,body=chunk.split('\n',1);task=next((k for k in TASKS if 'def '+k+'(' in body),None)
  cells.append(nbf.v4.new_markdown_cell('### '+heading+('\n\n**Goal:** '+TASKS[task][1]+'\n\n**Why:** the graph computation must preserve key, time and query identity.' if task else ' · PROVIDED')))
  if task and not solution:
   node=next(n for n in ast.parse(body).body if isinstance(n,ast.FunctionDef) and n.name==task)
   body=body.splitlines()[node.lineno-1]+'\n    raise NotImplementedError("TODO: '+task+'")\n'
  cells.append(nbf.v4.new_code_cell(body.strip()))
  if task:
   check=TASKS[task][0];cells.append(nbf.v4.new_code_cell(CHECKS[check]+'\n'+check+'('+task+')\nprint("PASS: '+task+'")'))
 cells.extend([nbf.v4.new_markdown_cell('## RUN · your functions on real F1 rows\n\nThis is a three-table teaching view of drivers, results and constructors from the complete released archive, including rows after the query date so your cutoff has work to do. The author benchmark uses all nine tables with the release censoring policy. Neither this graph walk nor its success is a predictive benchmark.'),nbf.v4.new_code_cell(run)])
 for name in ['resnet.py','sage_conv.py']:
  path=P/'sources/l117/primitives'/name
  if path.exists():cells.append(nbf.v4.new_markdown_cell('## Read the pinned library operator · '+name+'\n\nThe model calls this library implementation. It is reproduced here for inspection; execute the installed pinned package, not an improvised partial port.\n\n```python\n'+path.read_text()+'\n```'))
 cells.extend([nbf.v4.new_markdown_cell('## Full named reproduction · default OFF\n\nFive complete 10-epoch runs. This may require GPU setup and incurs compute costs on paid hosts. The notebook gate has no spending cap; the supplied author Modal runner limits reservations under the USD10 aggregate plan. A run in a different environment is new evidence, not automatically paper-identical.'),nbf.v4.new_code_cell(gate),nbf.v4.new_markdown_cell('## EXIT TICKET\n\nSubmit your three functions, `l117-task-report.json`, and the five-part defense from Section 8. Explain one rejected counterexample for each function. **PENDING_WRITTEN_DEFENSE**.')])
 for i,c in enumerate(cells):c.id=f'l117-{i:03d}'
 nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'},'language_info':{'name':'python'}});path=P/('solutions' if solution else '')/f'{S}.ipynb'
 if solution and path.exists():
  old=nbf.read(path,as_version=4);aa=[c for c in nb.cells if c.cell_type=='code'];bb=[c for c in old.cells if c.cell_type=='code']
  if [c.source for c in aa]==[c.source for c in bb]:
   nb.metadata=old.metadata
   for a,b in zip(aa,bb):a.outputs=b.outputs;a.execution_count=b.execution_count;a.metadata=b.metadata
 nbf.write(nb,path)
 if solution and all(c.execution_count is not None for c in nb.cells if c.cell_type=='code'):
  html,_=HTMLExporter(template_name='lab').from_notebook_node(nb);(P/'html'/f'{S}.html').write_text(html)
ref='''## The three graph objects

Schema: tables and allowed relationship types. REG: individual rows with typed PK/FK connections and attributes. Computation graph: query-specific, temporally eligible sampled copies used by the neural network.

## Identity and time contracts

- Map keys to row positions; reject duplicate keys and define a policy for dangling references.
- Add separately typed reverse edges so each side can receive messages.
- Keep the original query cutoff at every hop, including after timeless rows.
- `input_id`: query-row index for labels. `n_id`: global entity-row index. `batch`: owning query for a sampled copy.
- Future outcomes define targets; they are not query-time features.
- Event dates, ingestion history, and preprocessing-fit scope require separate audits.

## Released regression computation

Per-table column encodings → four-block ResNet → 128 channels. Add learned sinusoidal relative-age encoding. Two typed sum-GraphSAGE layers, each followed by node-wise LayerNorm/ReLU. Read first B driver vectors → scalar linear head. Mean L1, Adam .005, ten epochs, first minimum validation state. Inference clips to train 2nd/98th target percentiles.

## Evidence boundaries

Selected target: RelBench v1 Table 7, F1 driver-position RDL. Five runs, complete queries, source fanout [128,64], uniform temporal sampling. Paper means val3.193/test4.022; declared CLOSE tolerance .2 MAE. Historical random state, train-only preprocessing and true ingestion histories are not established by the release replay. Whole-paper parity NOT_ESTABLISHED.

[Full lesson](../lessons/0117-rdl-bridge.html) · [Protocol](../labs/l117-reproduction.md) · [Results](../labs/evidence/l117/summary.json) · [Fey ICML paper](https://proceedings.mlr.press/v235/fey24a.html) · [RelBench companion](https://arxiv.org/html/2407.20060v1).
'''
(R/'reference/rdl-bridge.html').write_text(document('RDL bridge · quick reference',ref))
print('Built lesson, reference and aligned portable notebooks')

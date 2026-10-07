"""Build synchronized lesson, reference and portable full-protocol notebooks."""
import ast,base64,copy,hashlib,json,re,textwrap,zlib
from pathlib import Path
import nbformat as nb
from nbconvert.filters.markdown import markdown2html_mistune as render
from _report_l155 import make_report,render_report
P=Path(__file__).resolve().parent;R=P.parent;E=P/'evidence/l155';F=P/'figures/l155';S='0155-compare-manual-fe';TITLE='Manual features vs RDL: compare quality and human effort'
m=json.loads((E/'input-manifest.json').read_text());r=make_report(E,m)
(E/'report.json').write_text(json.dumps(r,indent=2)+'\n');(E/'report.md').write_text(render_report(r))
def definitions(path):
 s=path.read_text();return {n.name:ast.get_source_segment(s,n) for n in ast.parse(s).body if isinstance(n,ast.FunctionDef)}
fns=definitions(P/'relkit/effort_l155.py');checks=definitions(P/'_check_l155.py');reporters=definitions(P/'_report_l155.py')
report_text=render_report(r);table='| Split'+report_text.split('| Split',1)[1].split('\n\n',1)[0]
lo,hi=r['test_benefit_driver_bootstrap_95'];diff=r['metrics']['test']['benefit_mae']['mean']
result=f"**Measured result:** FE's test MAE is **{r['metrics']['test']['fe_mae']['mean']:.4f}**, compared with **{r['metrics']['test']['rdl_mae']['mean']:.4f}** for basic RDL. The signed RDL benefit is **{diff:+.4f} finishing positions**. The driver-cluster conditional 95% interval is **[{lo:+.4f}, {hi:+.4f}]**. It crosses zero; the point estimate favors FE, but this does not establish superiority or equivalence. The GNN mean is within the frozen ±0.20 descriptive band around Table 7's 4.022."
summary=json.loads((E/'summary.json').read_text());fe=json.loads((E/'fe/summary.json').read_text());budget=json.loads((P/'_budget_l155.json').read_text())
audit=f"**Author evidence:** all 8,712 labels and 50 engineered fields were independently reconstructed. All 12,590 held-out predictions were rescored; FE tree replay maximum error {fe['independent_tree_error']:.3g}, GNN original-output maximum difference {summary['maximum_original_output_error']:.3g}. Every GNN seed retained 640 nonfinite entries in its first backward gradients. That is an unresolved source behavior, not a clean optimization-health result. The full pipeline results remain reportable with this limitation."
audit+=f"\n\n**Budget:** primary GNN worker-body estimate ${summary['worker_body_usd']:.6f}; local FE cloud spend $0. Current reserved worker resources plus the $3 overhead allowance total ${sum(x['upper_usd'] for x in budget['reservations'])+3:.6f}, under the $10 aggregate cap. These are resource estimates/reservations, not an itemized invoice. [Budget](../labs/_budget_l155.json)."
for name,label in [('_notebook_gnn_l155_results.json','GPU'),('_notebook_fe_l155_results.json','FE')]:
 if (P/name).exists():
  a=json.loads((P/name).read_text());audit+=f"\n\nThe separate full {label} notebook gate passed: five additional complete fits, {a['predictions']:,} independently checked predictions. Excluded from primary means."
captions={'pipelines':'Actual validation query for driver 7 at cutoff 2009-08-08: trace FE tree sums and the basic GNN prediction. The complete held-out populations, not this example, determine the comparison.', 'scores':'Actual fresh primary fits: points are runs; diamonds and bars show mean and sample seed SD. Separate vertical scales preserve within-split readability.', 'clocks':'Synthetic accounting example only. Marginal human work, allocated shared setup and machine runtime answer different questions. No human hours were measured for the actual F1 runs.'}
def prose(portable=False):
 s=(R/'lessons/content'/f'{S}.md').read_text().replace('[[TABLE]]',table).replace('[[RESULT]]',result).replace('[[AUDIT]]',audit)
 for name,caption in captions.items():
  src='data:image/png;base64,'+base64.b64encode((F/(name+'.png')).read_bytes()).decode() if portable else '../labs/figures/l155/'+name+'.svg'
  s=s.replace('[[FIG:'+name+']]',f'<figure class="route-figure" tabindex="0"><img src="{src}" alt="{caption}"><figcaption>{caption}</figcaption></figure>')
 for marker,web,plain in [('WARMUP','<div id="warmup"></div>','Recall without notes: why is cutoff part of identity? Which split selects the checkpoint? What does missing evidence mean?'),('PREDICT','<div id="predict"></div>','Predict before reading: does closeness to a paper score establish a win over FE?'),('WIDGET','<div id="l155-effort" class="effort-accounting"></div><noscript>Synthetic: marginal 2/.5=4×; including1.5h setup for one task gives2/2=1×. Actual human effort NOT_OBSERVED.</noscript>','Try the synthetic fixture in the CHECK cell: FE2h, RDL.5h marginal and1.5h shared; marginal ratio4×, first-task total1×.'),('TEACHBACK','<div id="teachback"></div>','Explain why a full training reproduction can coexist with unmeasured human effort before consulting the teacher.')]:s=s.replace('[['+marker+']]',plain if portable else web)
 if portable:
  s=s.replace('](../','](https://avistian.github.io/relational/')
  s=re.sub(r'\]\((\d{4}-[^)]+\.html)\)',r'](https://avistian.github.io/relational/lessons/\1)',s)
 return s

def doc(title,body,interactive=False):
 body=re.sub(r'<table([^>]*)>',r'<div class="route-scroll" tabindex="0"><table\1>',render(body)).replace('</table>','</table></div>')
 scripts=['retrieval-pool','retrieval-bank','predict','teachback','effort-accounting','l155-lesson'] if interactive else []
 return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Lesson 155 — '+title+'</title>'+''.join('<link rel="stylesheet" href="../assets/'+x+'.css">' for x in ['lesson','atomic-route','checkpoint','effort-accounting'])+'</head><body class="checkpoint"><article><nav><a href="../index.html">Course</a> · <a href="../lessons/0154-portfolio-synthesis.html">Lesson 154</a></nav><header><p class="route-kicker">Year 4 · Quarter 4 · Lesson 155</p><h1>'+title+'</h1></header>'+body+'</article>'+''.join('<script src="../assets/'+x+'.js"></script>' for x in scripts)+'</body></html>'
(R/'lessons'/f'{S}.html').write_text(doc(TITLE,prose(),True))
reference='''## Two separate measurements

| Quantity | Definition | Boundary |
|---|---|---|
| MAE | Mean absolute target−prediction error | Same complete(entity,cutoff)queries |
| RDL benefit | FE MAE−RDL MAE | Positive favors RDL; units are finishing positions |
| Marginal effort | Active human work specific to the task | Excludes reusable setup and unattended compute |
| Human-effort ratio | FE hours/RDL hours | Both complete, observed, comparable logs; positive denominator |
| Reduction | 100×(1−RDL hours/FE hours) | A4×ratio means75%less time |
| First-task effort | Marginal + shared setup | Different estimand from paper's marginal effort |
| Amortized effort | Marginal + shared setup/task count | Declare allocation; cannot reconstruct missing hours |

## Log contract
Participant,task,method,assistance,phase,kind(human/machine),scope(marginal/shared),timezone-aware start/end and coverage. Reject duplicate sessions and overlapping human work. Missing is not zero. Complete coverage is a self-attestation. Copying released SQL measures adaptation, not feature ideation; practice order and expertise confound personal ratios.

## Actual computation
'''+table+'\n\n'+result+'''

Human effort NOT_OBSERVED; original human trial NOT_RUN. One matched portfolio task. Classification FE and recommendation test remain missing. Table7 basicGNN differs from Figure3 boosted regression. Source nonfinite gradients and unobserved feature-arrival histories remain limitations.

[Lesson](../lessons/0155-compare-manual-fe.html) · [Notebook](../labs/0155-compare-manual-fe.ipynb) · [Report](../labs/evidence/l155/report.md) · [Protocol](../labs/l155-reproduction.md) · [Primary reading: Section6](https://arxiv.org/html/2407.20060v1#S6).

Ask the teaching agent to review your report and250–400word defense. LearnerPENDING_WRITTEN_DEFENSE.
'''
(R/'reference/compare-manual-fe.html').write_text(doc('Manual-FE comparison reference',reference))
packet={name:base64.b64encode((E/name).read_bytes()).decode() for name in m['files']};packed=base64.b64encode(zlib.compress(json.dumps(packet,sort_keys=True).encode(),9)).decode()
bootstrap='''# @colab-bootstrap: offline default audit; numpy and IPython required.
import os,sys,io,copy,gzip,base64,hashlib,json,math,statistics,tempfile,zlib
from datetime import datetime
from pathlib import Path
import numpy as np
from IPython.display import display,Markdown
os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
RUN_FULL_FE_REPRODUCTION=False
RUN_FULL_GNN_REPRODUCTION=False
'''
transport='''# PROVIDED: integrity-pinned primary author evidence, not newly trained models.
MANIFEST='''+repr(m)+'''
PACKET='''+repr(packed)+'''
assert hashlib.sha256(PACKET.encode()).hexdigest()=='''+repr(hashlib.sha256(packed.encode()).hexdigest())+'''
packet=json.loads(zlib.decompress(base64.b64decode(PACKET)))
workspace=tempfile.TemporaryDirectory(prefix='l155-author-')
REPLAY_ROOT=Path(workspace.name)
assert set(packet)==set(MANIFEST['files'])
for name,digest in MANIFEST['files'].items():
    dest=(REPLAY_ROOT/name).resolve();assert dest.is_relative_to(REPLAY_ROOT.resolve())
    raw=base64.b64decode(packet[name]);assert hashlib.sha256(raw).hexdigest()==digest
    dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw)
'''
# Reuse FE computation from the tested L149 notebook, preserving actual source.
old=nb.read(P/'solutions/0149-weakest-relbench-tasks.ipynb',4)
fe_code=old.cells[26].source.replace('l149','l155')
fe_first,fe_second=fe_code.split('\nif RUN_FULL_FE_REPRODUCTION:',1)
fe_nodes=ast.parse(fe_first).body[0].body
fe_cells=[]
for node in fe_nodes:
 seg=textwrap.dedent(ast.get_source_segment(fe_first,node))
 # AST source segments omit indentation on first line only; use line slices instead.
 seg=textwrap.dedent('\n'.join(fe_first.splitlines()[node.lineno-1:node.end_lineno]))
 fe_cells.append(nb.v4.new_code_cell('if RUN_FULL_FE_REPRODUCTION:\n'+textwrap.indent(seg,'    ')))
# Keep transport separate so implementation and training remain visible in exported HTML.
fe_second='if RUN_FULL_FE_REPRODUCTION:'+fe_second
prefix,training=fe_second.split(' tables,queries,_=load_archives(source)',1)
fe_cells.append(nb.v4.new_code_cell(prefix,metadata={'tags':['data-payload']}))
fe_cells.append(nb.v4.new_code_cell('if RUN_FULL_FE_REPRODUCTION:\n tables,queries,_=load_archives(source)'+training))
# Full GNN code is identical to the audited L152 inline implementation, gated off.
old_gnn=nb.read(P/'solutions/0152-regression-portfolio.ipynb',4)
source_files={str(p.relative_to(P)):p.read_text() for p in (P/'sources/l117').glob('*') if p.is_file()}
for name in ['_run_l152.py','relkit/rdl_l117.py','relkit/batch_audit_l123.py','relkit/regression_l152.py','requirements-l117-runtime.txt']:source_files[name]=(P/name).read_text()
gnn_transport='''if RUN_FULL_GNN_REPRODUCTION:
    SOURCE_FILES='''+repr(source_files)+'''
    source_workspace=tempfile.TemporaryDirectory(prefix='l155-gnn-source-')
    GNN_SOURCE_ROOT=Path(source_workspace.name)
    for name,text in SOURCE_FILES.items():
        dest=GNN_SOURCE_ROOT/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(text)
'''
gnn_code=[]
for i in list(range(5,14))+[26,27,28]:
 code=old_gnn.cells[i].source.replace('P=Path.cwd()','P=GNN_SOURCE_ROOT')
 gnn_code.append(nb.v4.new_code_cell('if RUN_FULL_GNN_REPRODUCTION:\n'+textwrap.indent(code,'    ')))
# Required audit functions are visible PROVIDED functions, not this lesson's TODOs.
gnn_helpers=(P/'relkit/regression_l152.py').read_text()
gate='''if RUN_FULL_GNN_REPRODUCTION:
    import importlib.metadata as md
    for package,version in {'torch':'2.5.1','pytorch-frame':'0.2.3','relbench':'1.1.0','torch-geometric':'2.6.1','numpy':'1.26.4','pandas':'2.2.3','sentence-transformers':'3.3.1'}.items():
        assert md.version(package).split('+')[0]==version,(package,'Use pinned GPU runtime')
    assert torch.cuda.is_available(),'Pinned GPU runtime required'
    own_root=Path('l155-gnn-full');own_root.mkdir(exist_ok=False);fresh=[]
    for seed in range(5):
        result=run(seed,10,own_root/f'seed-{seed}')
        fresh.append(dict(seed=seed,scores=result['scores']))
    (own_root/'packet.json').write_text(json.dumps(dict(status='PASS',fits=5,records=fresh),indent=2))
    print('Five fresh GNN fits complete; excluded from the author primary report')
'''
for solution in [False,True]:
 cells=[nb.v4.new_markdown_cell('# Lesson155 · '+TITLE+'\n\n'+('Teacher solution' if solution else 'Student lab')+' · TierB full real F1 data. PROVIDED / TODO / CHECK / EXIT. Default CPU execution replays real author predictions offline. Three live functions construct the report and validate your effort log. Full FE and GNN gates require separate pinned runtimes and are off by default.'),nb.v4.new_code_cell(bootstrap),nb.v4.new_code_cell(transport,metadata={'tags':['data-payload']})]
 for section in re.split(r'(?=^## )',prose(True),flags=re.M):
  if not section.strip():continue
  cells.append(nb.v4.new_markdown_cell(section))
  for number,name,check in [('3','paired_losses','check_pairs'),('4','summarize_effort','check_effort'),('4','effort_ratio','check_ratio')]:
   if section.startswith('## '+number):
    src=fns[name] if solution else fns[name].split('\n',1)[0]+'\n    raise NotImplementedError("TODO: '+name+'")'
    cells.append(nb.v4.new_code_cell('# TODO: live report/effort contract.\n'+src))
    cells.append(nb.v4.new_code_cell('# CHECK: synthetic boundary cases, not observed human data.\n'+checks['rejects']+'\n'+checks['fixture']+'\n'+checks[check]+'\n'+check+'('+name+')\nprint("PASS '+name+'")'))
  if section.startswith('## 5'):
   cells.append(nb.v4.new_code_cell('# PROVIDED: visible full primary report and driver bootstrap.\n'+reporters['make_report']+'\n\n'+reporters['render_report']))
   cells.append(nb.v4.new_code_cell('''# CHECK: actual evidence flows through your three functions.
report=make_report(REPLAY_ROOT,MANIFEST,pair=paired_losses,effort=summarize_effort,ratio=effort_ratio)
assert report['scored_prediction_rows']==12590
assert report['human_effort_ratio']['status']=='NOT_OBSERVED'
display(Markdown(render_report(report)))
'''))
 cells.append(nb.v4.new_markdown_cell('### PROVIDED · Your prospective effort log\n\nEdit the JSON file after a real session, then rerun this cell. Each session has `id`, `task`, `participant`, `assistance`, `method`(FE/RDL), `phase`, `kind`(human/machine), `scope`(marginal/shared), `start` and `end`(ISO8601 with timezone). Set coverage to partial while work remains. Attest complete only after the declared finish criterion is met. Use the same assistance policy in both arms. Do not copy synthetic CHECK sessions into your real log.'))
 cells.append(nb.v4.new_code_cell('''log_path=Path('l155-my-effort.json')
if not log_path.exists():
    log_path.write_text(json.dumps(dict(task='rel-f1/driver-position',participant='YOUR_NAME',assistance='DECLARE_POLICY',coverage={'FE':'not_observed','RDL':'not_observed'},sessions=[]),indent=2))
my_log=json.loads(log_path.read_text())
my_effort=summarize_effort(my_log['sessions'],my_log['task'],my_log['participant'],my_log['assistance'],my_log['coverage'])
my_ratio=effort_ratio(my_effort)
print('Your logged evidence:',my_ratio)
'''))
 cells.append(nb.v4.new_code_cell('''# EXIT: human reasoning is reviewed separately from code execution.
WRITTEN_DEFENSE=""  # 250–400 words; submit to the teaching agent.
export=dict(report,personal_effort=my_effort,personal_ratio=my_ratio,written_defense=WRITTEN_DEFENSE)
Path('l155-comparison.json').write_text(json.dumps(export,indent=2))
Path('l155-comparison.md').write_text(render_report(report)+'\\n'+WRITTEN_DEFENSE)
print('Exported comparison; learner PENDING_WRITTEN_DEFENSE')
'''))
 cells.append(nb.v4.new_markdown_cell('## Post-EXIT · Full released computational reproduction\n\nFE and GNN use different historical dependency stacks. Run each gate in its own empty working directory with its pinned requirements. Use `RUN_FULL_FE_REPRODUCTION` or `RUN_FULL_GNN_REPRODUCTION` in the bootstrap cell. The Modal operator contains the GPU image recipe and author budget guard. Notebook gates do not enforce spending limits. FE needs Python3.11, Torch2.2.2, Frame0.2.2 and LightGBM4.3.0; GNN needs Torch2.5.1/CUDA12.4, Frame0.2.3, PyG2.6.1 and native pyg-lib. Full requirements and all deviations are in the reproduction ledger. Source definitions below are visible; payload cells transport original source/data for offline FE execution.'))
 cells.append(nb.v4.new_markdown_cell(old.cells[25].source.replace('L149','L155').replace('l149','l155')))
 cells.extend(copy.deepcopy(fe_cells))
 cells.append(nb.v4.new_markdown_cell('### PROVIDED · Basic RDL computation\n\nRead in order: key identity → typed row encoders and graph convolutions → scalar head → graph construction → validation-selected full trainer. These are the same definitions used in the primary GNN lane.'))
 cells.append(nb.v4.new_code_cell(gnn_transport,metadata={'tags':['data-payload']}))
 cells.append(nb.v4.new_code_cell('if RUN_FULL_GNN_REPRODUCTION:\n'+textwrap.indent(gnn_helpers,'    ')))
 cells.extend(copy.deepcopy(gnn_code));cells.append(nb.v4.new_code_cell(gate))
 cells.append(nb.v4.new_code_cell("Path('l155-report.json').write_text(json.dumps(dict(status='PASS',predictions=12590,full_fe=RUN_FULL_FE_REPRODUCTION,full_gnn=RUN_FULL_GNN_REPRODUCTION,learner='PENDING_WRITTEN_DEFENSE'),indent=2))\nworkspace.cleanup()"))
 for i,c in enumerate(cells):c.id=f'l155-{i:03d}'
 notebook=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python3','language':'python','name':'python3'}});path=P/('solutions' if solution else '')/(S+'.ipynb')
 if solution and path.exists():
  previous=nb.read(path,4);notebook.metadata=previous.metadata
  before=[c for c in previous.cells if c.cell_type=='code'];after=[c for c in notebook.cells if c.cell_type=='code']
  if [c.source for c in before]==[c.source for c in after]:
   for oldcell,new in zip(before,after):
    new.outputs=oldcell.outputs;new.execution_count=oldcell.execution_count;new.metadata=oldcell.metadata
 nb.write(notebook,path)
print('Built lesson, reference and portable notebooks')

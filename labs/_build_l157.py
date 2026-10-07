"""Deterministic lesson, reference and standalone contribution notebooks."""
import ast,base64,hashlib,json,re
from pathlib import Path
import nbformat as nb
from nbconvert.filters.markdown import markdown2html_mistune as render
P=Path(__file__).resolve().parent;R=P.parent;D=P/'releases/l157-f1-audit';E=P/'evidence/l157';F=P/'figures/l157';S='0157-open-source-contribution';TITLE='Open-source contribution: make your evidence reviewable'
r=json.loads((E/'report.json').read_text());budget=json.loads((D/'budget.json').read_text());reserved=sum(x['upper_usd'] for x in budget['reservations'])+budget['overhead_reserve_usd']
def functions(path):
 s=path.read_text();return {n.name:ast.get_source_segment(s,n) for n in ast.parse(s).body if isinstance(n,ast.FunctionDef)}
fns=functions(P/'relkit/contribution_l157.py');checks=functions(P/'_check_l157.py')
table='| Fresh L157 lane | Validation MAE | Test MAE | Availability |\n|---|---:|---:|---|\n'
for name,lane in r['lanes'].items():
 v=lane['metrics']['val'];t=lane['metrics']['test'];table+=f"| {'Released baseline' if name=='paper' else 'Fixed-2005 intervention'} | {v['mean']:.6f} ± {v['sd']:.6f} | {t['mean']:.6f} ± {t['sd']:.6f} | NOT_ESTABLISHED |\n"
result=f"**Measured policy difference:** corrected minus released test MAE = **{r['corrected_minus_released_mae']['test']:+.6f}**. Positive means the correction has higher error. The released mean is **{r['reference_paper_band']['test']}** to the paper test target under the frozen tolerance."
coverage=f"**Coverage:** ten complete fresh fits; all **8,712 labels** reconstructed; **{r['sql_values']:,} SQL values** independently checked during isolated preparation; **{r['predictions']:,} held-out predictions** independently rescored. Both lanes check every yielded batch, original node/edge identities and each root's query cutoff. First-backward nonfinite entries per released seed: {list(r['lanes']['paper']['first_backward_nonfinite'].values())}."
budget_text=f"**Cost boundary:** **${reserved:.2f} reserved resources plus overhead** against the approved **$10 total cap**. This is a conservative ledger, not an itemized invoice. Preparation, failures and validation count toward the cap; full runs are stopped rather than silently shortened. [Budget ledger](../labs/releases/l157-f1-audit/budget.json)."
delivery='**Delivery status:** the standalone 27-code-cell solution passed in an empty directory. Its full pinned GPU gate completed ten additional fits and independently replayed another 12,590 predictions, excluded from the primary means. Desktop/mobile, 80 interactive states, keyboard/reset, print/no-JavaScript, deterministic generation and a clean Git-index Pages build passed. The final delivery checks are recorded in [verification](../labs/_verify_l157_results.json). Public contribution **PENDING_PUBLICATION**; live Colab and deployment **NOT_CHECKED**; learner **PENDING_WRITTEN_DEFENSE**. No upstream issue or PR has been sent.'
captions={'flow':'Actual evidence path: freeze computation, run complete experiments, package and independently replay, then restrict the claim.','results':'New L157 five-seed results per policy. Mean and sample seed SD; separate validation/test scales.'}
def prose(portable=False):
 s=(R/'lessons/content'/(S+'.md')).read_text()
 for k,v in [('TABLE',table),('RESULT',result),('COVERAGE',coverage),('BUDGET',budget_text),('DELIVERY',delivery)]:s=s.replace('[['+k+']]',v)
 for name,caption in captions.items():
  src='data:image/png;base64,'+base64.b64encode((F/(name+'.png')).read_bytes()).decode() if portable else '../labs/figures/l157/'+name+'.svg'
  s=s.replace('[[FIG:'+name+']]',f'<figure class="route-figure" tabindex="0"><img src="{src}" alt="{caption}"><figcaption>{caption}</figcaption></figure>')
 for name,code in fns.items():s=s.replace('[[CODE:'+name+']]', '' if portable else '\n```python\n'+code+'\n```\n')
 for name,web,plain in [('TEACHBACK','<div id="teachback"></div>','Explain in your own words why evidence replay, fresh training, public access and historical availability need separate evidence. Write before consulting the author solution.'),('WARMUP','<div id="warmup"></div>','Retrieve: what does a temporal sampler fail to tell you about feature-processor fitting?'),('PREDICT','<div id="predict"></div>','Predict: do ten completed runs establish missing feature-arrival histories? Explain before reading further.'),('WIDGET','<div id="l157-review" class="contribution-review"></div><noscript>Intact ten-fit evidence supports selected reproduction. Missing seed: INCOMPLETE. Availability: NOT_ESTABLISHED. No public URL: PENDING_PUBLICATION.</noscript>','Try the three CHECK fixtures: changed evidence, missing seed, unsupported historical claim. Predict each verdict before executing.')]:s=s.replace('[['+name+']]',plain if portable else web)
 if portable:
  s=s.replace('](../','](https://avistian.github.io/relational/');s=re.sub(r'\]\((\d{4}-[^)]+\.html)\)',r'](https://avistian.github.io/relational/lessons/\1)',s)
 return s

def doc(title,body,interactive=False):
 body=re.sub(r'<table([^>]*)>',r'<div class="route-scroll" tabindex="0"><table\1>',render(body)).replace('</table>','</table></div>')
 scripts=['retrieval-pool','retrieval-bank','predict','teachback','contribution-review','l157-lesson'] if interactive else []
 return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Lesson 157 — '+title+'</title>'+''.join('<link rel="stylesheet" href="../assets/'+x+'.css">' for x in ['lesson','atomic-route','checkpoint','contribution-review'])+'</head><body class="checkpoint"><article><nav><a href="../index.html">Course</a> · <a href="../lessons/0156-temporal-leakage-audit.html">Lesson156</a></nav><header><p class="route-kicker">Year4 · Quarter4 · Lesson157</p><h1>'+title+'</h1></header>'+body+'</article>'+''.join('<script src="../assets/'+x+'.js"></script>' for x in scripts)+'</body></html>'
(R/'lessons'/(S+'.html')).write_text(doc(TITLE,prose(),True))
ref='''## A reviewable contribution

| Artifact | Purpose | Does not establish |
|---|---|---|
| Protocol + experiment manifest | Freeze what will run | Future success |
| Complete seed evidence | Cover planned experiment | All-paper reproduction |
| Independent replay | Check integrity, keys and metrics | Fresh training |
| Pinned runtime + executable code | Let readers repeat the computation | Perpetual upstream availability |
| Notices + download provenance | Preserve attribution and source scope | Universal data redistribution rights |
| Release manifest | Identify shipped bytes | Author authentication |
| Verified public URL | Establish public access | Maintainer acceptance |

## Claim gate
Changed evidence → REJECTED. Missing planned run → INCOMPLETE. Ten audited fits → selected reproduction supported. Missing arrival histories → NOT_ESTABLISHED. No public URL → PENDING_PUBLICATION. Whole benchmark → NOT_RUN.

## Contribution paragraph
State question, exact source/data version, expected and observed policy, minimal reproduction command, measured result and remaining uncertainty. Match your headline to this paragraph. Before an upstream PR, reproduce on current upstream and provide a small regression test.

## Current experiment
'''+table+'\n'+result+'''

[Lesson](../lessons/0157-open-source-contribution.html) · [Package](../labs/releases/l157-f1-audit.zip) · [README](../labs/releases/l157-f1-audit/README.md) · [Contribution guide](https://github.com/stanford-star/relbench/blob/main/CONTRIBUTING.md).

Ask the teaching agent to review your contribution draft. Learner PENDING_WRITTEN_DEFENSE; publication PENDING_PUBLICATION.
'''
(R/'reference/reproducibility-contribution.html').write_text(doc('Reproducibility contribution reference',ref))
archive=(P/'releases/l157-f1-audit.zip').read_bytes();payload=base64.b64encode(archive).decode();digest=hashlib.sha256(archive).hexdigest()
bootstrap='''# Default: offline evidence replay. Full GPU training is explicitly opt-in.
import base64,hashlib,io,json,math,os,statistics,subprocess,sys,tempfile,zipfile
from pathlib import Path
import numpy as np
from IPython.display import display,Markdown
RUN_FULL_REPRODUCTION=False
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
'''
transport='''# PROVIDED: compact release transport; it contains no raw databases or weights.
PACKAGE_B64='''+repr(payload)+'''
raw=base64.b64decode(PACKAGE_B64)
assert hashlib.sha256(raw).hexdigest()=='''+repr(digest)+'''
workspace=tempfile.TemporaryDirectory(prefix='l157-notebook-');PACKAGE_ROOT=Path(workspace.name)
with zipfile.ZipFile(io.BytesIO(raw)) as archive:
 for item in archive.infolist():
  rel=Path(item.filename)
  assert not rel.is_absolute() and '..' not in rel.parts
  assert (item.external_attr>>16)&0o170000 != 0o120000
  dest=PACKAGE_ROOT/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(archive.read(item))
EVIDENCE_ROOT=PACKAGE_ROOT/'labs/evidence/l157'
'''
full='''# PROVIDED: fresh complete experiment, using the visible files above.
# Run in the pinned CUDA runtime from run_modal.py. Manual gate has no dollar meter.
subprocess.run([sys.executable,str(PACKAGE_ROOT/'cli.py'),'check'],check=True,cwd=PACKAGE_ROOT)
if RUN_FULL_REPRODUCTION:
 import shutil
 shutil.move(str(EVIDENCE_ROOT),str(PACKAGE_ROOT/'labs/evidence/author-l157'))
 subprocess.run([sys.executable,str(PACKAGE_ROOT/'cli.py'),'prepare'],check=True,cwd=PACKAGE_ROOT,timeout=3600)
 for lane in ['paper','fit_horizon']:
  for seed in range(5):
   subprocess.run([sys.executable,str(PACKAGE_ROOT/'cli.py'),'train','--lane',lane,'--seed',str(seed)],check=True,cwd=PACKAGE_ROOT,timeout=1800)
 subprocess.run([sys.executable,str(PACKAGE_ROOT/'cli.py'),'freeze-evidence'],check=True,cwd=PACKAGE_ROOT,timeout=180)
 fresh=json.loads((EVIDENCE_ROOT/'report.json').read_text())
 assert fresh['contribution']['fits']==10 and fresh['predictions']==12590
 Path('l157-fresh-report.json').write_text(json.dumps(fresh,indent=2))
 print('PASS: ten fresh complete notebook fits; separate from primary means')
'''
for solution in [False,True]:
 cells=[nb.v4.new_markdown_cell('# Lesson157 · '+TITLE+'\n\n'+('Author solution' if solution else 'Student lab')+' · PROVIDED / TODO / CHECK / EXIT. Default requires numpy and IPython. Three live functions verify the actual package evidence and write your claim report. Full training requires the separately pinned GPU environment.'),nb.v4.new_code_cell(bootstrap),nb.v4.new_code_cell(transport,metadata={'tags':['data-payload']})]
 for section in re.split(r'(?=^## )',prose(True),flags=re.M):
  if not section.strip():continue
  cells.append(nb.v4.new_markdown_cell(section))
  for number,name,check in [('3','verify_manifest','check_integrity'),('5','summarize_runs','check_runs'),('6','review_claim','check_claims')]:
   if section.startswith('## '+number):
    code=fns[name] if solution else fns[name].split('\n',1)[0]+'\n    raise NotImplementedError("TODO: '+name+'")'
    cells.append(nb.v4.new_code_cell('# TODO: used on the actual released evidence below.\n'+code))
    cells.append(nb.v4.new_code_cell('# CHECK: wrong evidence must change the verdict.\n'+checks['rejects']+'\n'+checks['fixtures']+'\n'+checks[check]+'\n'+check+'('+name+')\nprint("PASS '+name+'")'))
 cells.append(nb.v4.new_code_cell('''# Live learner functions operate on the real package, not only toy fixtures.
release_manifest=json.loads((PACKAGE_ROOT/'release-manifest.json').read_text())
file_count=verify_manifest(PACKAGE_ROOT,release_manifest)
verify_manifest(PACKAGE_ROOT,json.loads((PACKAGE_ROOT/'experiment-manifest.json').read_text()))
verify_manifest(EVIDENCE_ROOT,json.loads((EVIDENCE_ROOT/'input-manifest.json').read_text())['files'])
# Independent metric/epoch/query auditor remains visible in the package.
subprocess.run([sys.executable,str(PACKAGE_ROOT/'cli.py'),'replay'],check=True,cwd=PACKAGE_ROOT,timeout=180)
rows=json.loads((EVIDENCE_ROOT/'run-rows.json').read_text());summary=summarize_runs(rows)
evidence=dict(integrity='PASS',reproduction=summary['status'],availability='NOT_ESTABLISHED',public_url=None)
claims={k:review_claim(k,evidence) for k in ['selected_reproduction','historically_leak_free','public_contribution','whole_paper','upstream_bug']}
assert summary['fits']==10 and claims['selected_reproduction']=='SUPPORTED'
display(Markdown((EVIDENCE_ROOT/'report.md').read_text()))
WRITTEN_DEFENSE=''
Path('l157-report.json').write_text(json.dumps(dict(status='PASS',files=file_count,summary=summary,claims=claims,written_defense=WRITTEN_DEFENSE,learner='PENDING_WRITTEN_DEFENSE'),indent=2))
print(claims)
'''))
 cells.append(nb.v4.new_markdown_cell('## Optional full reproduction · visible implementation\n\nThese are the actual package sources: graph and model, original identity checks, temporal correction, runner and independent evaluator. Each visible cell writes the same bytes into the isolated package. The experiment manifest rejects changes before fitting. Read the model and fitting loop before enabling the gate.'))
 paths=['labs/relkit/rdl_l117.py','labs/relkit/batch_audit_l123.py','labs/relkit/temporal_audit_l156.py','labs/relkit/regression_l152.py','labs/_run_l117.py','labs/_run_l152.py','labs/_run_l156.py','labs/_replay_l156.py']
 for path in paths:
  source=(D/path).read_text();parts=re.split(r'(?=^# %%)',source,flags=re.M) if path.endswith('rdl_l117.py') else [source]
  for i,part in enumerate(parts):
   cells.append(nb.v4.new_markdown_cell('### PROVIDED · '+path+' · part'+str(i+1)))
   cells.append(nb.v4.new_code_cell('%%writefile '+('-a ' if i else '')+'{PACKAGE_ROOT}/'+path+'\n'+part))
 cells.append(nb.v4.new_code_cell(full))
 cells.append(nb.v4.new_code_cell("workspace.cleanup()\nprint('Contribution lab complete; publication PENDING_PUBLICATION; learner PENDING_WRITTEN_DEFENSE')"))
 for i,c in enumerate(cells):c.id=f'l157-{i:03d}'
 notebook=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python3','language':'python','name':'python3'}});path=P/('solutions' if solution else '')/(S+'.ipynb')
 if solution and path.exists():
  old=nb.read(path,4);notebook.metadata=old.metadata
  previous=[c for c in old.cells if c.cell_type=='code'];current=[c for c in notebook.cells if c.cell_type=='code']
  if [c.source for c in previous]==[c.source for c in current]:
   for prev,new in zip(previous,current):
    new.outputs=prev.outputs;new.execution_count=prev.execution_count;new.metadata=prev.metadata
 nb.write(notebook,path)
print('Built Lesson157, reference and portable notebooks')

# Preserve verified outputs when only prose changes, and refresh the rendered solution.
from nbconvert import HTMLExporter
from nbconvert.preprocessors import TagRemovePreprocessor
executed=nb.read(P/'solutions'/(S+'.ipynb'),4)
if all(c.execution_count is not None for c in executed.cells if c.cell_type=='code'):
 exporter=HTMLExporter(template_name='lab');exporter.register_preprocessor(TagRemovePreprocessor(remove_input_tags={'data-payload'}),enabled=True)
 html,_=exporter.from_notebook_node(executed);(P/'html'/(S+'.html')).write_text(html)

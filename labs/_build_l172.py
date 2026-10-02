"""Generate lesson, reference and portable notebooks from canonical code/evidence."""
import ast,base64,hashlib,io,json,re,zipfile
from pathlib import Path
import nbformat as nb
from nbconvert.filters.markdown import markdown2html_mistune as render
P=Path(__file__).resolve().parent;R=P.parent;S='0172-schema-tokenization';E=P/'evidence/l172'
r=json.loads((E/'report.json').read_text());manifest=json.loads((E/'input-manifest.json').read_text())
def definitions(path):
 text=path.read_text();return [(x.name,ast.get_source_segment(text,x)) for x in ast.parse(text).body if isinstance(x,ast.FunctionDef)]
def result_table():
 lines=['| Table | Rows transformed | Columns | Rows admitted to fit |','|---|---:|---:|---:|']
 for n,t in r['tables'].items():lines.append(f"| {n} | {t['rows']:,} | {t['columns']} | {t['admitted_rows']:,} |")
 return '\n'.join(lines)+f"\n\n**Complete: {r['rows']:,} rows, {r['columns']} columns, {r['cells']:,} cells.** All-table column reordering, all-cell mask erasure and every numerical/category held-out fit intervention pass."
captions={'roles':'Illustrative equal integers route to different handlers. Keys preserve identity, quantities preserve magnitude, and category codes are local labels.',
 'trace':'Measured F1 results.points cell. The same raw value, fitted mean and scale produce the displayed payload; masking erases that payload. Rounded display, full precision report.',
 'fit-boundary':'Measured full-snapshot row counts. Teal rows are admitted to fitting; all rows are transformed. Untimed tables admit zero rows. This is not a query availability certificate.'}
def prose(portable=False):
 s=(R/'lessons/content'/(S+'.md')).read_text().replace('[[RESULTS]]',result_table())
 w=r['worked_trace'];s=s.replace('[[TRACE]]',f"`resultId={w['resultId']}` has {w['raw']:g} points on {w['date'][:10]}. With mean {w['mean']:.6f} and scale {w['scale']:.6f}, its payload is **{w['payload']:.6f}**. This row is transformed but excluded from fitting.")
 for n,c in captions.items():
  src='data:image/png;base64,'+base64.b64encode((P/'figures/l172'/(n+'.png')).read_bytes()).decode() if portable else '../labs/figures/l172/'+n+'.svg'
  s=s.replace('[[FIG:'+n+']]',f'<figure class="route-figure" tabindex="0" style="overflow-x:auto"><img style="min-width:760px;width:100%" src="{src}" alt="{c}"><figcaption>{c} Scroll horizontally on narrow screens.</figcaption></figure>')
 replacements={
 'WARMUP':('Recall: why do clean foreign keys not certify availability?','<div id="warmup"></div>'),
 'PREDICT':('Predict: should unseen and masked categories share a state? Explain before checking.','<div id="predict"></div>'),
 'FIT_EXPLORER':('Notebook intervention below: change the held-out value and compare admitted-only with all-row fit. Correct fit stays mean 20, SD 8.164966; leaky fit changes.','<div id="token-fit"></div><noscript>Admitted values 10, 20, 30 give mean 20 and SD8.164966. Including held-out 999 changes both. The notebook computes this intervention.</noscript>'),
 'STATE_EXPLORER':('Frozen vocabulary a→1,b→2. Known b is VALUE 2; unseen future is UNKNOWN 0; null is MISSING 0; hidden b is MASKED 0.','<div id="token-states"></div><noscript>Vocabulary a→1,b→2: b is VALUE 2, unseen future UNKNOWN 0, null MISSING 0, hidden b MASKED 0.</noscript>'),
 'TEACHBACK':('Write your defense before opening the teacher outline; submit it for review.','<div id="teachback"></div>')}
 for k,(plain,html) in replacements.items():s=s.replace('[['+k+']]',plain if portable else html)
 if portable:
  s=s.replace('](../','](https://avistian.github.io/relational/')
  s=re.sub(r'\]\((\d{4}-[^)]+\.html)\)',r'](https://avistian.github.io/relational/lessons/\1)',s)
 return s

def doc(title,body,interactive=False):
 html=render(body).replace('<table>','<div class="route-scroll" tabindex="0"><table>').replace('</table>','</table></div>')
 css=['lesson','atomic-route','checkpoint','lab-access','foundation-scope','schema-tokenization']
 scripts=['retrieval-pool','retrieval-bank','predict','teachback','schema-tokenization','l172-lesson'] if interactive else []
 return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title>'+''.join('<link rel="stylesheet" href="../assets/'+x+'.css">' for x in css)+'</head><body class="checkpoint"><article><nav><a href="../index.html">Course</a> · <a href="../lessons/0171-corpus-of-databases.html">Lesson 171</a></nav><header><p class="route-kicker">Year 5 · Quarter 2 · Lesson 172</p><h1>'+title+'</h1></header>'+html+'</article>'+''.join('<script src="../assets/'+x+'.js"></script>' for x in scripts)+'</body></html>'
(R/'lessons'/(S+'.html')).write_text(doc('Schema tokenization',prose(),True))
ref='''**Input contract:** storage dtype ≠ semantic kind ≠ database role. A column token stream carries a named schema descriptor, fitted state, N states and N payloads. It is not yet a learned embedding.

| Kind | Fit on admitted nonnull rows | Transform observed values |
|---|---|---|
| Number | Mean, population SD; empty0/1; constant SD1 | (value−mean)/SD |
| Category | Sorted string vocabulary; codes start1 | Local code or UNKNOWN 0 |
| Text | No learned state | Preserve Unicode string |
| Key | No learned state | Exact string identity; retain target table |
| Timestamp | No learned state in this course | UTC epoch days |

**State precedence:** MASKED overrides source null; otherwise MISSING for null; UNKNOWN for out-of-vocabulary category; otherwise VALUE. Non-value payload is0 for number/category/time, empty string for key/text. Preserve the state. Never infer unknown versus missing from payload alone.

**Fit policy:** date<2005-01-01 in time-bearing tables. No admitted rows in untimed tables. Their numerical defaults and empty category vocabulary are recorded, not learned. Transforming a row does not make it available to a historical query.

**Three invariants:** (1) permuting columns preserves named outputs; (2) changing heldout values preserves fitted statistics; (3) changing masked values preserves masked output. Fail on undeclared columns/types. Do not cast keys through float. Do not treat local category codes as cross-database meaning.

'''+result_table()+'''

**Boundary:** full selected pipeline audit COMPLETE. Learned embeddings, fresh pretraining and whole-paper reproduction NOT_RUN; historical availability NOT_ESTABLISHED; learner PENDING_WRITTEN_DEFENSE.

[Lesson](../lessons/0172-schema-tokenization.html) · [Notebook](../labs/0172-schema-tokenization.ipynb) · [Protocol](../labs/l172-reproduction.md) · [Input schema](../labs/evidence/l172/schema.json) · [Primary reading: RTv1 §3.1](https://arxiv.org/html/2510.06377v1#S3.SS1).
'''
(R/'reference/schema-tokenization.html').write_text(doc('Schema tokenization — quick reference',ref))
(E/'report.md').write_text('# L172 Full F1 Schema-Tokenization Audit\n\n'+result_table()+'\n\nComplete selected course pipeline audit. Whole-paper reproduction/fresh pretraining NOT_RUN; historical availability NOT_ESTABLISHED. See report.json for per-column state/fit counts and output hashes.\n')
buf=io.BytesIO()
with zipfile.ZipFile(buf,'w',compression=zipfile.ZIP_DEFLATED) as z:
 for name in sorted(manifest['files']):
  info=zipfile.ZipInfo(name);info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,(P/name).read_bytes())
raw=buf.getvalue();payload=base64.b64encode(raw).decode();digest=hashlib.sha256(raw).hexdigest()
contracts={
 'fit_column':('Fit only admitted rows','Accept values, kind and one strict boolean admission per row. Kinds: number/category/text/key/timestamp. Return kind,fit_rows,fit_nonnull. Numbers add mean and scale using admitted nonnull values and population SD. Empty fit uses0/1; constant uses SD1. Reject nonfinite admitted numbers. Categories add sorted distinct str(value) vocabulary from admitted nonnull rows. Unsupported kind or invalid admission raises ValueError. Do not mutate inputs.',
 '''f=fit_column(pd.Series([10.,20.,30.,999.]),"number",[True,True,True,False])
assert f["mean"]==20 and abs(f["scale"]-8.16496580927726)<1e-12
print("CHECK: held-out 999 did not enter the fit")'''),
 'encode_column':('Preserve states and erase masked payloads','Accept values, fitted state, optional strict boolean mask (default all False). Return state and payload lists of the same length. MASKED has priority over source null; otherwise MISSING, then UNKNOWN for unseen category, otherwise VALUE. Non-value payload0 for number/category/timestamp, empty string for text/key. Normalize observed finite numbers with frozen mean/scale; reject invalid/nonfinite numbers. Category codes start1 in vocabulary order. Text/key use str without a float cast. Timestamp uses UTC epoch days. Unsupported kind or wrong mask raises ValueError. Never encode a masked raw value.',
 '''f=fit_column(pd.Series(["b","a"]),"category",[True,True])
e=encode_column(pd.Series(["b","future",None,"a"]),f,[False,False,False,True])
assert e==dict(state=["VALUE","UNKNOWN","MISSING","MASKED"],payload=[2,0,0,0])
print("CHECK: identical neutral payloads retain different meanings")'''),
 'tokenize_table':('Assemble by schema identity','Accept DataFrame, schema mapping column to descriptor, admitted flags and optional masks mapping column to boolean list. Reject duplicate/missing/extra columns, absent mask names, invalid roles, and primary/foreign keys whose kind is not key. Roles: feature,primary_key,foreign_key,event_time. Iterate names sorted; call your fit_column and encode_column. Return rows and columns. Each column contains copied schema, fitted, state,payload. Preserve descriptor fields, including foreign-key target. Input column order must not affect output.',
 '''frame=pd.DataFrame({"n":[10.,20.],"c":["a","b"]})
spec={"n":dict(kind="number",role="feature"),"c":dict(kind="category",role="feature")}
assert tokenize_table(frame,spec,[True,False])==tokenize_table(frame[["c","n"]],spec,[True,False])
print("CHECK: schema identity survives reordering")''')}
for solution in [False,True]:
 cells=[nb.v4.new_markdown_cell('# Lesson 172 · Schema tokenization\n\n**One skill:** implement and defend a typed input contract. Real complete F1 snapshot; separate synthetic failure cases. PROVIDED supplies evidence; TODO functions drive the actual audit; CHECK gives feedback; EXIT requires your reasoning. No training or cloud access. Author-reference results below are not your kernel results.'),nb.v4.new_markdown_cell(prose(True)),
 nb.v4.new_code_cell('# @colab-bootstrap\nimport os\nos.environ.update(OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1")\nimport base64,hashlib,io,json,zipfile\nfrom pathlib import Path\ntry:\n import pandas as pd\n import numpy as np\n import pyarrow.parquet as pq\nexcept ImportError as exc:\n raise RuntimeError("Install pandas==3.0.3, pyarrow==24.0.0 and numpy, then rerun") from exc\nP=Path("l172-portable");P.mkdir(exist_ok=True)')]
 cells.append(nb.v4.new_markdown_cell('## PROVIDED · Authenticate complete evidence\nThe packet contains all nine tables and the original archive, schema policies and source provenance. Hashes detect changed bytes, not historical truth. No network is required after installing dependencies.'))
 c=nb.v4.new_code_cell('encoded='+repr(payload)+'\nraw=base64.b64decode(encoded)\nassert hashlib.sha256(raw).hexdigest()=='+repr(digest)+'\nwith zipfile.ZipFile(io.BytesIO(raw)) as z:\n for name in z.namelist():\n  assert not Path(name).is_absolute() and ".." not in Path(name).parts\n z.extractall(P)\nmanifest='+repr(manifest)+'\nprint("Complete evidence packet authenticated")');c.metadata['tags']=['data-payload'];cells.append(c)
 for name,code in definitions(P/'relkit/tokenization_l172.py'):
  title,contract,check=contracts[name]
  cells += [nb.v4.new_markdown_cell('## TODO · '+title+'\n\n'+contract+'\n\nPredict one failing case before you implement this operation.'),nb.v4.new_code_cell(code if solution else code.split('\n',1)[0]+'\n    raise NotImplementedError("TODO: '+name+'")'),nb.v4.new_code_cell('# CHECK\n'+check)]
 cells.append(nb.v4.new_markdown_cell('## CHECK · Attack the boundary\nThis suite calls your live functions. It includes a large integer key, constant and empty numeric fits, unknown/missing/masked categories, wrong shapes, nonfinite values and held-out-value interventions.'))
 for name,code in definitions(P/'_check_l172.py'):cells.append(nb.v4.new_code_cell(code))
 cells.append(nb.v4.new_code_cell('assert check172(fit_column,encode_column,tokenize_table)=="PASS"\nprint("Live tokenizer contracts PASS")'))
 for name,code in definitions(P/'_audit_l172.py'):
  cells += [nb.v4.new_markdown_cell('## PROVIDED · '+('Authenticate and load every table' if name=='load172' else 'Run the full declared audit')+'\nRead the code: the audit authenticates inputs and calls your live operations. It does not import a hidden model.'),nb.v4.new_code_cell(code)]
 cells.append(nb.v4.new_code_cell('report=replay172(P,manifest,fit_column,encode_column,tokenize_table)\nPath("l172-report.json").write_text(json.dumps(report,indent=2)+"\\n")\nassert (report["rows"],report["columns"],report["cells"])==(97606,67,866746)\nprint(report["status"])\ndisplay(pd.DataFrame([{ "table":n,"rows":t["rows"],"columns":t["columns"],"fit_rows":t["admitted_rows"]} for n,t in report["tables"].items()]))\nprint("Real points trace:",report["worked_trace"])\nprint("Availability:",report["historical_availability"],"| whole paper:",report["whole_paper_reproduction"])'))
 cells += [nb.v4.new_markdown_cell('## Intervention · Can the future change the past encoding?\nBefore executing, predict which fit will change when 999 becomes 300. This synthetic example isolates the mistake; it is not an intervention on the real archive.'),nb.v4.new_code_cell('for future in [999.,300.]:\n values=pd.Series([10.,20.,30.,future])\n correct=fit_column(values,"number",[True,True,True,False])\n leaky=fit_column(values,"number",[True]*4)\n print("heldout",future,"correct probe",encode_column(pd.Series([20.]),correct),"leaky probe",encode_column(pd.Series([20.]),leaky))'),nb.v4.new_markdown_cell('## EXIT · Defend the contract\nWrite 200–400 words: choose an unseen table and assign types/roles; give a dtype-only counterexample; identify allowed fit rows; distinguish three neutral states; explain why local category codes do not establish semantic transfer; name one missing temporal guarantee. Submit code, report and defense to the teacher. Author execution alone does not establish mastery.'),nb.v4.new_code_cell('defense={k:"" for k in ["new_schema","dtype_counterexample","fit_admission","neutral_states","category_transfer","availability_gap"]}\nsubmission=dict(defense=defense,report_sha256=hashlib.sha256(Path("l172-report.json").read_bytes()).hexdigest(),teacher_review=None,learner="PENDING_WRITTEN_DEFENSE")\nPath("l172-submission.json").write_text(json.dumps(submission,indent=2)+"\\n")\nprint("Learner PENDING_WRITTEN_DEFENSE")'),nb.v4.new_markdown_cell('## Reproduction boundary\nThis notebook executes the complete approved course tokenization audit. The independent scalar oracle is `_verify_l172.py`; exact commands and budget controls are in the protocol. No model objective or trainer is claimed. RT learned embeddings, whole-paper reproduction and fresh pretraining remain NOT_RUN. Live Colab and deployment NOT_CHECKED.')]
 book=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'}})
 for i,c in enumerate(cells):c.id=f'l172-{i:03d}'
 path=P/('solutions' if solution else '')/(S+'.ipynb')
 if solution and path.exists():
  old=nb.read(path,4)
  if [(c.cell_type,c.source) for c in old.cells]==[(c.cell_type,c.source) for c in cells]:
   book.metadata=old.metadata
   for c,prior in zip(cells,old.cells):
    if c.cell_type=='code':c.outputs=prior.outputs;c.execution_count=prior.execution_count;c.metadata=prior.metadata
 nb.write(book,path)
print('Built L172 lesson, reference and portable notebooks')
